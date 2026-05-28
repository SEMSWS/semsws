/**
 * @file IsotropicAcousticSensitivityAD2D.cpp
 * @brief AD-based implementation of acoustic 2D sensitivity kernels.
 *
 * Forward-mode AD via mfem::future::dual<real_t, real_t> replaces the hand
 * chain-rule formulas in IsotropicAcousticSensitivity2D. See the header
 * for design rationale and the chain-rule conventions.
 *
 * Per-GLL-point structure mirrors the hand version exactly except that
 * the material-dependent product is evaluated with a dual-typed material
 * coefficient, and the final accumulation goes into "raw" (1/ρ, 1/κ)
 * space. Save() converts to (V_p, ρ) using the standard (V_p, ρ)-independent
 * chain rule.
 */

#include "fwi/IsotropicAcousticSensitivityAD2D.hpp"
#include "integ/kernels/AcousticFluxKernel2D.hpp"
#include "util/FESOrder.hpp"
#include "integ/core/SEMKernelDispatch.hpp"
#include "common/GpuMacros.hpp"
#include <mfem.hpp>
#include "io/ADIOS2IO.hpp"
#include <sstream>
#include <iomanip>

namespace SEM {

using Dual = mfem::future::dual<real_t, real_t>;

// =============================================================================
// Constructor
// =============================================================================

IsotropicAcousticSensitivityAD2D::IsotropicAcousticSensitivityAD2D(
    ParFiniteElementSpace& fes,
    const MaterialField& kappa,
    const MaterialField& inv_rho,
    const MaterialField* unrelaxed_correction)
    : fes_(&fes)
{
    int order = SafeFESOrder(fes);
    ngll_ = order + 1;
    ne_ = fes.GetParMesh()->GetNE();

    const int total_gll = ngll_ * ngll_ * ne_;

    geom_.Compute(fes);
    dofs_.ComputeScalar(fes, order);
    geom_.EnableDevice();
    geom_.SyncToDevice();
    dofs_.EnableDevice();
    dofs_.SyncToDevice();

    inv_rho_.SetSize(total_gll);   inv_rho_.UseDevice(true);
    inv_kappa_.SetSize(total_gll); inv_kappa_.UseDevice(true);
    kappa_.SetSize(total_gll);     kappa_.UseDevice(true);
    unrelaxed_correction_.SetSize(total_gll); unrelaxed_correction_.UseDevice(true);
    InitMaterialFields(kappa, inv_rho, unrelaxed_correction);

    k_invrho_.SetSize(total_gll);   k_invrho_.UseDevice(true);   k_invrho_   = 0.0;
    k_invkappa_.SetSize(total_gll); k_invkappa_.UseDevice(true); k_invkappa_ = 0.0;

    vp_hessian_.SetSize(total_gll);  vp_hessian_.UseDevice(true);  vp_hessian_  = 0.0;
    rho_hessian_.SetSize(total_gll); rho_hessian_.UseDevice(true); rho_hessian_ = 0.0;

    // Public-facing (Vp, ρ) kernels — populated at Save() time.
    vp_kernel_.SetSize(total_gll);  vp_kernel_.UseDevice(true);
    rho_kernel_.SetSize(total_gll); rho_kernel_.UseDevice(true);
}

// =============================================================================
// InitMaterialFields
// =============================================================================

void IsotropicAcousticSensitivityAD2D::InitMaterialFields(
    const MaterialField& kappa, const MaterialField& inv_rho,
    const MaterialField* unrelaxed_correction)
{
    const int total_gll = ngll_ * ngll_ * ne_;

    // Copy inv_rho as-is.
    {
        const real_t* d_src = inv_rho.Read();
        real_t* d_dst = inv_rho_.Write();
        mfem::forall(total_gll, [=] MFEM_HOST_DEVICE (int i) {
            d_dst[i] = d_src[i];
        });
    }

    // Copy κ as-is (for bitwise-close Save conversion, no double reciprocal).
    {
        const real_t* d_src = kappa.Read();
        real_t* d_dst = kappa_.Write();
        mfem::forall(total_gll, [=] MFEM_HOST_DEVICE (int i) {
            d_dst[i] = d_src[i];
        });
    }

    // Derive inv_kappa = 1/κ for the mass-path dual seeding.
    {
        const real_t* d_k = kappa.Read();
        real_t* d_ik = inv_kappa_.Write();
        mfem::forall(total_gll, [=] MFEM_HOST_DEVICE (int i) {
            d_ik[i] = 1.0 / d_k[i];
        });
    }

    // Copy unrelaxed correction (or fill with 1.0 for pure acoustic).
    {
        real_t* d_dst = unrelaxed_correction_.Write();
        if (unrelaxed_correction) {
            const real_t* d_src = unrelaxed_correction->Read();
            mfem::forall(total_gll, [=] MFEM_HOST_DEVICE (int i) {
                d_dst[i] = d_src[i];
            });
        } else {
            mfem::forall(total_gll, [=] MFEM_HOST_DEVICE (int i) {
                d_dst[i] = 1.0;
            });
        }
    }
}

// =============================================================================
// Reset
// =============================================================================

void IsotropicAcousticSensitivityAD2D::Reset() {
    k_invrho_   = 0.0;
    k_invkappa_ = 0.0;
    // vp_kernel_/rho_kernel_ are re-derived at Save().
}

void IsotropicAcousticSensitivityAD2D::ResetHessian() {
    vp_hessian_  = 0.0;
    rho_hessian_ = 0.0;
}

// =============================================================================
// Accumulate (dispatch to GPU/CPU kernels, NGLL-dispatched)
// =============================================================================

void IsotropicAcousticSensitivityAD2D::Accumulate(
    const Vector& fwd_p, const Vector& fwd_a,
    const Vector& adj_p, real_t dt)
{
    SEM_DISPATCH_NGLL(ngll_, AccumulateVpKernel_AD,  fwd_a, adj_p, dt);
    SEM_DISPATCH_NGLL(ngll_, AccumulateRhoKernel_AD, fwd_p, adj_p, dt);
}

// =============================================================================
// Vp (mass) path via AD on inv_kappa
// =============================================================================
//   mass integrand at (x_i, t_n):  (1/κ) · p̈_fwd · p_adj
//   ∂/∂(1/κ): p̈_fwd · p_adj
//   → K_{1/κ}(x_i) += p̈_fwd · p_adj · dt   (accumulated via AD gradient)
//   Pseudo-Hessian unchanged: H_Vp += 4/(ρ²·Vp⁶) · p̈² · dt = (2/(ρVp³))² · p̈²·dt.

template<int NGLL>
void IsotropicAcousticSensitivityAD2D::AccumulateVpKernel_AD(
    const Vector& fwd_a, const Vector& adj_p, real_t dt)
{
    const int ne = ne_;

    const real_t* d_fwd_a = fwd_a.Read();
    const real_t* d_adj_p = adj_p.Read();
    real_t* d_k_invkappa  = k_invkappa_.ReadWrite();
    real_t* d_hessian     = vp_hessian_.ReadWrite();

    auto gather_map = dofs_.ViewGatherMap();
    auto inv_kappa  = Reshape(inv_kappa_.Read(), ngll_, ngll_, ne_);
    auto inv_rho    = Reshape(inv_rho_.Read(),   ngll_, ngll_, ne_);

    mfem::forall_2D(ne, NGLL, NGLL, [=] MFEM_HOST_DEVICE (int ei)
    {
        MFEM_FOREACH_THREAD(iy, y, NGLL)
        {
            MFEM_FOREACH_THREAD(ix, x, NGLL)
            {
                const int gll_idx = gather_map(ix, iy, ei);
                const int local_idx = ix + iy * NGLL + ei * NGLL * NGLL;

                const real_t ik_val = inv_kappa(ix, iy, ei);
                const real_t a_fwd  = d_fwd_a[gll_idx];
                const real_t p_adj  = d_adj_p[gll_idx];

                // Seed dual on inv_kappa; state a_fwd, p_adj are passive.
                Dual d_ik{ik_val, 1.0};
                Dual integrand = d_ik * a_fwd * p_adj;
                // integrand.value    = (1/κ)·p̈·p_adj   (mass integrand)
                // integrand.gradient = p̈·p_adj          (∂/∂(1/κ))

                d_k_invkappa[local_idx] += integrand.gradient * dt;

                d_hessian[local_idx] += a_fwd * a_fwd * dt;
            }
        }
    });
}

// =============================================================================
// ρ (stiffness) path via AD on inv_rho
// =============================================================================
//   stiffness integrand at (x_i, t_n):  (1/ρ) · ∇φ_fwd · ∇φ_adj   [physical, pointwise]
//   ∂/∂(1/ρ): ∇φ_fwd · ∇φ_adj
//   → K_{1/ρ}(x_i) += ∇φ_fwd · ∇φ_adj · dt
//   Pseudo-Hessian (unchanged): H_ρ += (1/ρ²) · |∇φ_fwd|² · dt.

template<int NGLL>
void IsotropicAcousticSensitivityAD2D::AccumulateRhoKernel_AD(
    const Vector& fwd_p, const Vector& adj_p, real_t dt)
{
    const int ne = ne_;

    const real_t* d_fwd_p = fwd_p.Read();
    const real_t* d_adj_p = adj_p.Read();
    real_t* d_k_invrho    = k_invrho_.ReadWrite();
    real_t* d_rho_hess    = rho_hessian_.ReadWrite();

    auto gather_map = dofs_.ViewGatherMap();
    auto inv_rho    = Reshape(inv_rho_.Read(), ngll_, ngll_, ne_);
    auto invJPacked = geom_.ViewInvJPacked();
    auto dshape     = geom_.ViewDShape();

    mfem::forall_2D(ne, NGLL, NGLL, [=] MFEM_HOST_DEVICE (int ei)
    {
        MFEM_SHARED real_t s_fwd[NGLL][NGLL];
        MFEM_SHARED real_t s_adj[NGLL][NGLL];
        MFEM_SHARED real_t s_hprime[NGLL][NGLL];

        // Load basis derivatives (one row)
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

        // Gather forward + adjoint wavefields
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

        MFEM_FOREACH_THREAD(iy, y, NGLL)
        {
            MFEM_FOREACH_THREAD(ix, x, NGLL)
            {
                // Reference-frame gradients (tensor-product), identical to hand version
                real_t dfwd_xi = 0, dfwd_eta = 0;
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

                // Apply Jacobian: reference → physical
                const real_t j0 = invJPacked(0, ix, iy, ei);
                const real_t j1 = invJPacked(1, ix, iy, ei);
                const real_t j2 = invJPacked(2, ix, iy, ei);
                const real_t j3 = invJPacked(3, ix, iy, ei);

                const real_t dfwd_dx = dfwd_xi * j0 + dfwd_eta * j3;
                const real_t dfwd_dy = dfwd_xi * j2 + dfwd_eta * j1;
                const real_t dadj_dx = dadj_xi * j0 + dadj_eta * j3;
                const real_t dadj_dy = dadj_xi * j2 + dadj_eta * j1;

                // ---- AD: seed dual on inv_rho, evaluate physical flux ----
                const real_t ir_val = inv_rho(ix, iy, ei);
                Dual d_ir{ir_val, 1.0};
                Dual qx, qy;
                AcousticFluxPhysical2D<Dual>(d_ir, dfwd_dx, dfwd_dy, qx, qy);
                // qx.value = (1/ρ)·∂φ/∂x, qx.gradient = ∂φ/∂x  (and same for qy)

                // Contract with adjoint physical gradient: per-point integrand
                Dual integrand = qx * dadj_dx + qy * dadj_dy;
                // integrand.value    = (1/ρ)·∇φ·∇λ   (Lagrangian integrand)
                // integrand.gradient = ∇φ·∇λ          (∂/∂(1/ρ))

                const int local_idx = ix + iy * NGLL + ei * NGLL * NGLL;
                d_k_invrho[local_idx] += integrand.gradient * dt;

                const real_t fwd_grad_sq = dfwd_dx * dfwd_dx + dfwd_dy * dfwd_dy;
                d_rho_hess[local_idx] += fwd_grad_sq * dt;
            }
        }
    });
}

// =============================================================================
// Save — materialize (Vp, ρ) kernels from (1/κ, 1/ρ) accumulators
// =============================================================================
//
// (Vp, ρ) treated as independent → κ-path contribution to K_ρ is NOT added.
//
//   K_Vp(x) = K_{1/κ}(x) · ∂(1/κ)/∂Vp|_ρ = K_{1/κ} · (-2·ρ⁻¹·Vp⁻³·ρ) = -2 K_{1/κ} / (ρ·Vp³)
//     ...  actually simplest: ∂(1/κ)/∂Vp = ∂(1/(ρVp²))/∂Vp = -2·ρ·Vp / (ρVp²)² = -2/(ρ·Vp³)
//   K_ρ(x)  = K_{1/ρ}(x) · ∂(1/ρ)/∂ρ = K_{1/ρ} · (-1/ρ²)
//
// Standard acoustic FWI convention: ρ treated independent of κ despite
// κ = ρ·Vp².

void IsotropicAcousticSensitivityAD2D::FinalizeKernels() const
{
    // Mirrors IsotropicAcousticSensitivity2D::InitCoefficients so hand and AD
    // back-ends agree at machine precision. For viscoacoustic, kappa_ stores
    // the unrelaxed κ_u whereas the user parameterization is Vp_user with
    // κ_user = κ_u / c (c = unrelaxed_correction_ per GLL). The chain rule
    // converts K_{1/κ_u} (the AD accumulator) to K_{Vp_user}:
    //   K_{Vp_user} = −2/(c·ρ·Vp_user³) · K_{1/κ_u}
    // For pure acoustic (c ≡ 1) this reduces to the original formula.
    const int total = ngll_ * ngll_ * ne_;
    const real_t* h_kir  = k_invrho_.HostRead();
    const real_t* h_kik  = k_invkappa_.HostRead();
    const real_t* h_ir   = inv_rho_.HostRead();
    const real_t* h_k    = kappa_.HostRead();
    const real_t* h_corr = unrelaxed_correction_.HostRead();
    real_t* h_vp  = vp_kernel_.HostWrite();
    real_t* h_rho = rho_kernel_.HostWrite();
    for (int i = 0; i < total; i++) {
        const real_t ir     = h_ir[i];
        const real_t rho    = 1.0 / ir;
        const real_t c_corr = h_corr[i];
        const real_t k_user = h_k[i] / c_corr;        // κ_user = κ_u / c
        const real_t vp2    = k_user * ir;            // Vp_user²
        const real_t vp     = std::sqrt(vp2);
        const real_t c_vp   = 2.0 / (c_corr * rho * vp2 * vp);  // 2/(c·ρ·Vp_user³)
        // Chain-rule map:
        //   K_Vp  = -c_vp · K_{1/κ_u}   (κ-path accumulated in κ_u space)
        //   K_ρ   = -ir²  · K_{1/ρ}      (ρ-path unaffected by attenuation)
        h_vp[i]  = -c_vp * h_kik[i];
        h_rho[i] = -(ir * ir) * h_kir[i];
    }
}

void IsotropicAcousticSensitivityAD2D::Save(
    const std::string& dir, ParMesh& mesh, int source_id)
{
    const int total = ngll_ * ngll_ * ne_;

    FinalizeKernels();

    MaterialField rho_field(ne_, ngll_, ngll_);
    MaterialField vp_field(ne_, ngll_, ngll_);
    {
        const real_t* h_src = vp_kernel_.HostRead();
        real_t* h_dst = vp_field.HostWrite();
        for (int i = 0; i < total; i++) h_dst[i] = h_src[i];
    }
    {
        const real_t* h_src = rho_kernel_.HostRead();
        real_t* h_dst = rho_field.HostWrite();
        for (int i = 0; i < total; i++) h_dst[i] = h_src[i];
    }

    std::ostringstream oss;
    oss << "_src" << std::setfill('0') << std::setw(3) << source_id;
    const std::string suffix = oss.str();

    const std::string mesh_path = dir + "/mesh.mesh";
    const std::string rho_path  = dir + "/kernel_rho" + suffix + ".bp";
    const std::string vp_path   = dir + "/kernel_vp" + suffix + ".bp";

    MPI_Comm comm = fes_->GetComm();
    SaveFieldBP(rho_path, "data", rho_field, mesh_path, comm);
    SaveFieldBP(vp_path,  "data", vp_field,  mesh_path, comm);
}

// =============================================================================
// SaveHessian (same output format as hand version)
// =============================================================================

void IsotropicAcousticSensitivityAD2D::SaveHessian(
    const std::string& dir, ParMesh& mesh, int source_id)
{
    const int total = ngll_ * ngll_ * ne_;

    MaterialField vp_hfield(ne_, ngll_, ngll_);
    MaterialField rho_hfield(ne_, ngll_, ngll_);

    const real_t* h_vp_src  = vp_hessian_.HostRead();
    const real_t* h_rho_src = rho_hessian_.HostRead();
    const real_t* h_ik      = inv_kappa_.HostRead();
    const real_t* h_ir      = inv_rho_.HostRead();
    real_t* h_vp_dst  = vp_hfield.HostWrite();
    real_t* h_rho_dst = rho_hfield.HostWrite();
    for (int i = 0; i < total; i++) {
        const real_t ik = h_ik[i];
        const real_t ir = h_ir[i];
        const real_t c_vp = 2.0 * ik * std::sqrt(ik / ir);
        h_vp_dst[i]  = c_vp * c_vp * h_vp_src[i];
        h_rho_dst[i] = ir * ir * h_rho_src[i];
    }

    std::ostringstream oss;
    oss << "_src" << std::setfill('0') << std::setw(3) << source_id;
    const std::string suffix = oss.str();

    const std::string mesh_path = dir + "/mesh.mesh";
    const std::string vp_path   = dir + "/hessian_vp"  + suffix + ".bp";
    const std::string rho_path  = dir + "/hessian_rho" + suffix + ".bp";

    MPI_Comm comm = fes_->GetComm();
    SaveFieldBP(vp_path,  "data", vp_hfield,  mesh_path, comm);
    SaveFieldBP(rho_path, "data", rho_hfield, mesh_path, comm);
}

// =============================================================================
// Explicit NGLL instantiations
// =============================================================================

#define INSTANTIATE_AD_VP_2D(NGLL) \
    template void IsotropicAcousticSensitivityAD2D::AccumulateVpKernel_AD<NGLL>( \
        const Vector&, const Vector&, real_t);

#define INSTANTIATE_AD_RHO_2D(NGLL) \
    template void IsotropicAcousticSensitivityAD2D::AccumulateRhoKernel_AD<NGLL>( \
        const Vector&, const Vector&, real_t);

SEM_INSTANTIATE_NGLL_2D(INSTANTIATE_AD_VP_2D)
SEM_INSTANTIATE_NGLL_2D(INSTANTIATE_AD_RHO_2D)

#undef INSTANTIATE_AD_VP_2D
#undef INSTANTIATE_AD_RHO_2D

}  // namespace SEM
