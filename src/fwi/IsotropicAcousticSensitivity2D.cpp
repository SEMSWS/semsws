/**
 * @file IsotropicAcousticSensitivity2D.cpp
 * @brief Sensitivity kernel implementation for 2D isotropic acoustic media
 *
 * Vp kernel: K_Vp += -2/(ρ·Vp³) · p̈_fwd · p_adj · dt
 * ρ  kernel: K_ρ  += -(1/ρ²) · ∇p_fwd · ∇p_adj · dt
 *
 * Pseudo-Hessian accumulated as raw forward illumination; chain-rule coefficient²
 * applied at SaveHessian time.
 *   H_Vp_raw += p̈_fwd² · dt           saved as 4/(ρ²·Vp⁶) · H_Vp_raw
 *   H_ρ_raw  += |∇p_fwd|² · dt        saved as (1/ρ²)     · H_ρ_raw
 */

#include "fwi/IsotropicAcousticSensitivity2D.hpp"
#include "util/FESOrder.hpp"
#include "integ/core/SEMKernelDispatch.hpp"
#include "common/GpuMacros.hpp"
#include <mfem.hpp>
#include "io/ADIOS2IO.hpp"
#include <sstream>
#include <iomanip>

namespace SEM {

// =============================================================================
// Constructor
// =============================================================================

IsotropicAcousticSensitivity2D::IsotropicAcousticSensitivity2D(
    ParFiniteElementSpace& fes,
    const MaterialField& kappa,
    const MaterialField& inv_rho,
    const MaterialField* unrelaxed_correction)
    : fes_(&fes)
{
    int order = SafeFESOrder(fes);
    ngll_ = order + 1;
    ne_ = fes.GetParMesh()->GetNE();

    int total_gll = ngll_ * ngll_ * ne_;

    // Compute geometry and DOF ordering
    geom_.Compute(fes);
    dofs_.ComputeScalar(fes, order);

    // Enable device memory
    geom_.EnableDevice();
    geom_.SyncToDevice();
    dofs_.EnableDevice();
    dofs_.SyncToDevice();

    // Pre-compute Vp gradient coefficient and copy 1/ρ
    vp_coeff_.SetSize(total_gll);
    vp_coeff_.UseDevice(true);
    inv_rho_.SetSize(total_gll);
    inv_rho_.UseDevice(true);
    InitCoefficients(kappa, inv_rho, unrelaxed_correction);

    // Initialize kernels to zero
    vp_kernel_.SetSize(total_gll);
    vp_kernel_.UseDevice(true);
    vp_kernel_ = 0.0;

    rho_kernel_.SetSize(total_gll);
    rho_kernel_.UseDevice(true);
    rho_kernel_ = 0.0;

    // Initialize pseudo-Hessian to zero
    vp_hessian_.SetSize(total_gll);
    vp_hessian_.UseDevice(true);
    vp_hessian_ = 0.0;

    rho_hessian_.SetSize(total_gll);
    rho_hessian_.UseDevice(true);
    rho_hessian_ = 0.0;
}

// =============================================================================
// InitCoefficients — separated from ctor so CUDA can take its address
// =============================================================================

void IsotropicAcousticSensitivity2D::InitCoefficients(
    const MaterialField& kappa, const MaterialField& inv_rho,
    const MaterialField* unrelaxed_correction)
{
    const int total_gll = ngll_ * ngll_ * ne_;

    // Pre-compute Vp gradient coefficient w.r.t. the USER-facing Vp:
    //   K_Vp = -∫ c_vp · p_tt · p_adj · dt,
    //   c_vp = 2/(c·ρ·Vp_user³),
    //   Vp_user² = κ_user/ρ, κ_user = κ_u / c,
    // where c = unrelaxed_correction (1 for pure acoustic). For c ≡ 1 this
    // reduces to the original 2/(ρ·Vp³).
    {
        const real_t* d_kappa = kappa.Read();
        const real_t* d_inv_rho = inv_rho.Read();
        const real_t* d_corr = unrelaxed_correction ? unrelaxed_correction->Read()
                                                     : nullptr;
        real_t* d_vp_coeff = vp_coeff_.Write();
        mfem::forall(total_gll, [=] MFEM_HOST_DEVICE (int i) {
            const real_t k_u = d_kappa[i];
            const real_t ir = d_inv_rho[i];
            const real_t rho = 1.0 / ir;
            const real_t c_corr = d_corr ? d_corr[i] : 1.0;
            const real_t k_user = k_u / c_corr;       // user-facing κ
            const real_t vp2 = k_user * ir;           // Vp_user²
            const real_t vp = sqrt(vp2);
            const real_t rho_vp3 = rho * vp2 * vp;    // ρ·Vp_user³
            const real_t c = 2.0 / (c_corr * rho_vp3);  // 2/(c·ρ·Vp_user³)
            d_vp_coeff[i] = c;
        });
    }

    // Copy 1/ρ (for rho kernel, unchanged)
    {
        const real_t* d_src = inv_rho.Read();
        real_t* d_dst = inv_rho_.Write();
        mfem::forall(total_gll, [=] MFEM_HOST_DEVICE (int i) {
            d_dst[i] = d_src[i];
        });
    }
}

// =============================================================================
// Reset
// =============================================================================

void IsotropicAcousticSensitivity2D::Reset() {
    vp_kernel_ = 0.0;
    rho_kernel_ = 0.0;
}

void IsotropicAcousticSensitivity2D::ResetHessian() {
    vp_hessian_ = 0.0;
    rho_hessian_ = 0.0;
}

// =============================================================================
// Accumulate (dispatch to GPU/CPU kernels)
// =============================================================================

void IsotropicAcousticSensitivity2D::Accumulate(
    const Vector& fwd_p, const Vector& fwd_a,
    const Vector& adj_p, real_t dt)
{
    SEM_DISPATCH_NGLL(ngll_, AccumulateVpKernel, fwd_a, adj_p, dt);
    SEM_DISPATCH_NGLL(ngll_, AccumulateRhoKernel, fwd_p, adj_p, dt);
}

// =============================================================================
// Vp Kernel: K_Vp += -2/(ρ·Vp³) · p̈_fwd · p_adj · dt
// =============================================================================
// This is a simple GLL point-wise product — no gradient computation needed.

template<int NGLL>
void IsotropicAcousticSensitivity2D::AccumulateVpKernel(
    const Vector& fwd_a, const Vector& adj_p, real_t dt)
{
    const int ne = ne_;

    const real_t* d_fwd_a = fwd_a.Read();
    const real_t* d_adj_p = adj_p.Read();
    real_t* d_kernel = vp_kernel_.ReadWrite();
    real_t* d_hessian = vp_hessian_.ReadWrite();

    auto gather_map = dofs_.ViewGatherMap();
    auto vp_coeff = Reshape(vp_coeff_.Read(), ngll_, ngll_, ne_);

    mfem::forall_2D(ne, NGLL, NGLL, [=] MFEM_HOST_DEVICE (int ei)
    {
        MFEM_FOREACH_THREAD(iy, y, NGLL)
        {
            MFEM_FOREACH_THREAD(ix, x, NGLL)
            {
                const int gll_idx = gather_map(ix, iy, ei);
                const int local_idx = ix + iy * NGLL + ei * NGLL * NGLL;

                const real_t c_vp = vp_coeff(ix, iy, ei);
                const real_t a_fwd = d_fwd_a[gll_idx];
                const real_t p_adj = d_adj_p[gll_idx];

                d_kernel[local_idx] -= c_vp * a_fwd * p_adj * dt;
                d_hessian[local_idx] += a_fwd * a_fwd * dt;
            }
        }
    });
}

// =============================================================================
// ρ Kernel: K_ρ += -(1/ρ²) · ∇p_fwd · ∇p_adj · dt  [d(1/ρ)/dρ = -1/ρ²]
// =============================================================================
// Requires gradient computation — follows the same pattern as the stiffness kernel.

template<int NGLL>
void IsotropicAcousticSensitivity2D::AccumulateRhoKernel(
    const Vector& fwd_p, const Vector& adj_p, real_t dt)
{
    const int ne = ne_;

    const real_t* d_fwd_p = fwd_p.Read();
    const real_t* d_adj_p = adj_p.Read();
    real_t* d_kernel = rho_kernel_.ReadWrite();
    real_t* d_rho_hessian = rho_hessian_.ReadWrite();

    auto gather_map = dofs_.ViewGatherMap();
    auto inv_rho = Reshape(inv_rho_.Read(), ngll_, ngll_, ne_);
    auto invJPacked = geom_.ViewInvJPacked();
    auto dshape = geom_.ViewDShape();

    mfem::forall_2D(ne, NGLL, NGLL, [=] MFEM_HOST_DEVICE (int ei)
    {
        // Shared memory for element wavefields
        MFEM_SHARED real_t s_fwd[NGLL][NGLL];
        MFEM_SHARED real_t s_adj[NGLL][NGLL];
        MFEM_SHARED real_t s_hprime[NGLL][NGLL];

        // Phase 1: Load basis functions
        if (MFEM_THREAD_ID(y) == 0)
        {
            MFEM_FOREACH_THREAD(ix, x, NGLL)
            {
                for (int k = 0; k < NGLL; k++)
                {
                    s_hprime[ix][k] = dshape(k, ix);
                }
            }
        }

        // Phase 2: Gather wavefields to shared memory
        MFEM_FOREACH_THREAD(iy, y, NGLL)
        {
            MFEM_FOREACH_THREAD(ix, x, NGLL)
            {
                const int gll_idx = gather_map(ix, iy, ei);
                s_fwd[iy][ix] = d_fwd_p[gll_idx];
                s_adj[iy][ix] = d_adj_p[gll_idx];
            }
        }
        MFEM_SYNC_THREAD;

        // Phase 3: Compute gradients and accumulate kernel + Hessian
        MFEM_FOREACH_THREAD(iy, y, NGLL)
        {
            MFEM_FOREACH_THREAD(ix, x, NGLL)
            {
                // Reference gradients for forward wavefield
                real_t dfwd_xi = 0, dfwd_eta = 0;
                // Reference gradients for adjoint wavefield
                real_t dadj_xi = 0, dadj_eta = 0;

                SEM_NOUNROLL
                for (int k = 0; k < NGLL; k++)
                {
                    const real_t hp_x = s_hprime[ix][k];
                    dfwd_xi += s_fwd[iy][k] * hp_x;
                    dadj_xi += s_adj[iy][k] * hp_x;

                    const real_t hp_y = s_hprime[iy][k];
                    dfwd_eta += s_fwd[k][ix] * hp_y;
                    dadj_eta += s_adj[k][ix] * hp_y;
                }

                // Inverse Jacobian: reference → physical
                const real_t j0 = invJPacked(0, ix, iy, ei);
                const real_t j1 = invJPacked(1, ix, iy, ei);
                const real_t j2 = invJPacked(2, ix, iy, ei);
                const real_t j3 = invJPacked(3, ix, iy, ei);

                // Physical gradients: ∇p_fwd
                const real_t dfwd_dx = dfwd_xi * j0 + dfwd_eta * j3;
                const real_t dfwd_dy = dfwd_xi * j2 + dfwd_eta * j1;

                // Physical gradients: ∇p_adj
                const real_t dadj_dx = dadj_xi * j0 + dadj_eta * j3;
                const real_t dadj_dy = dadj_xi * j2 + dadj_eta * j1;

                // Dot product: ∇p_fwd · ∇p_adj
                const real_t grad_dot = dfwd_dx * dadj_dx + dfwd_dy * dadj_dy;

                const int local_idx = ix + iy * NGLL + ei * NGLL * NGLL;
                const real_t ir = inv_rho(ix, iy, ei);
                d_kernel[local_idx] -= ir * ir * grad_dot * dt;

                const real_t fwd_grad_sq = dfwd_dx * dfwd_dx + dfwd_dy * dfwd_dy;
                d_rho_hessian[local_idx] += fwd_grad_sq * dt;
            }
        }
    });
}

// =============================================================================
// Save kernels to files
// =============================================================================

void IsotropicAcousticSensitivity2D::Save(
    const std::string& dir, ParMesh& mesh, int source_id)
{
    int total = ngll_ * ngll_ * ne_;

    MaterialField rho_field(ne_, ngll_, ngll_);
    MaterialField vp_field(ne_, ngll_, ngll_);

    // K_Vp: directly computed in Vp space — copy as-is
    {
        const real_t* h_src = vp_kernel_.HostRead();
        real_t* h_dst = vp_field.HostWrite();
        for (int i = 0; i < total; i++) {
            h_dst[i] = h_src[i];
        }
    }
    // K_ρ: raw kernel (dJ/dρ) — copy directly
    {
        const real_t* h_src = rho_kernel_.HostRead();
        real_t* h_dst = rho_field.HostWrite();
        for (int i = 0; i < total; i++) {
            h_dst[i] = h_src[i];
        }
    }

    // Build source suffix for filenames
    std::ostringstream oss;
    oss << "_src" << std::setfill('0') << std::setw(3) << source_id;
    std::string suffix = oss.str();

    std::string mesh_path = dir + "/mesh.mesh";
    std::string rho_path   = dir + "/kernel_rho" + suffix + ".bp";
    std::string vp_path    = dir + "/kernel_vp" + suffix + ".bp";

    MPI_Comm comm = fes_->GetComm();

    SaveFieldBP(rho_path, "data", rho_field, mesh_path, comm);
    SaveFieldBP(vp_path, "data", vp_field, mesh_path, comm);
}

// =============================================================================
// Save pseudo-Hessian to files
// =============================================================================

void IsotropicAcousticSensitivity2D::SaveHessian(
    const std::string& dir, ParMesh& mesh, int source_id)
{
    int total = ngll_ * ngll_ * ne_;

    MaterialField vp_hfield(ne_, ngll_, ngll_);
    MaterialField rho_hfield(ne_, ngll_, ngll_);

    const real_t* h_vp_src  = vp_hessian_.HostRead();
    const real_t* h_rho_src = rho_hessian_.HostRead();
    const real_t* h_c_vp    = vp_coeff_.HostRead();
    const real_t* h_ir      = inv_rho_.HostRead();
    real_t* h_vp_dst  = vp_hfield.HostWrite();
    real_t* h_rho_dst = rho_hfield.HostWrite();
    for (int i = 0; i < total; i++) {
        const real_t c  = h_c_vp[i];
        const real_t ir = h_ir[i];
        h_vp_dst[i]  = c * c * h_vp_src[i];
        h_rho_dst[i] = ir * ir * h_rho_src[i];
    }

    // Build source suffix for filenames
    std::ostringstream oss;
    oss << "_src" << std::setfill('0') << std::setw(3) << source_id;
    std::string suffix = oss.str();

    std::string mesh_path = dir + "/mesh.mesh";
    std::string vp_path    = dir + "/hessian_vp" + suffix + ".bp";
    std::string rho_path   = dir + "/hessian_rho" + suffix + ".bp";

    MPI_Comm comm = fes_->GetComm();

    SaveFieldBP(vp_path, "data", vp_hfield, mesh_path, comm);
    SaveFieldBP(rho_path, "data", rho_hfield, mesh_path, comm);
}

// =============================================================================
// Explicit Template Instantiations
// =============================================================================

#define INSTANTIATE_VP_2D(NGLL) \
    template void IsotropicAcousticSensitivity2D::AccumulateVpKernel<NGLL>( \
        const Vector&, const Vector&, real_t);

#define INSTANTIATE_RHO_2D(NGLL) \
    template void IsotropicAcousticSensitivity2D::AccumulateRhoKernel<NGLL>( \
        const Vector&, const Vector&, real_t);

SEM_INSTANTIATE_NGLL_2D(INSTANTIATE_VP_2D)
SEM_INSTANTIATE_NGLL_2D(INSTANTIATE_RHO_2D)

#undef INSTANTIATE_VP_2D
#undef INSTANTIATE_RHO_2D

}  // namespace SEM
