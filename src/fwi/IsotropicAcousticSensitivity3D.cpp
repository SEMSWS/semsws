/**
 * @file IsotropicAcousticSensitivity3D.cpp
 * @brief Hand-written sensitivity kernel for 3D isotropic acoustic.
 *
 * Mass path (K_Vp): K_Vp -= 2/(ρ·Vp³) · p̈_fwd · p_adj · dt     (pointwise)
 * Stiffness path (K_ρ): K_ρ -= (1/ρ²) · ∇p_fwd · ∇p_adj · dt    (3-component)
 *
 * Same structure as `IsotropicAcousticSensitivity2D.cpp` with `forall_3D`,
 * 3-axis reference gradient, and 10-entry invJPacked.
 */

#include "fwi/IsotropicAcousticSensitivity3D.hpp"
#include "util/FESOrder.hpp"
#include "integ/core/SEMKernelDispatch.hpp"
#include "common/GpuMacros.hpp"
#include <mfem.hpp>
#include "io/ADIOS2IO.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>

namespace SEM {

// =============================================================================
// Constructor
// =============================================================================

IsotropicAcousticSensitivity3D::IsotropicAcousticSensitivity3D(
    ParFiniteElementSpace& fes,
    const MaterialField3D& kappa,
    const MaterialField3D& inv_rho,
    const MaterialField3D* unrelaxed_correction)
    : fes_(&fes)
{
    const int order = SafeFESOrder(fes);
    ngll_ = order + 1;
    ne_   = fes.GetParMesh()->GetNE();

    const int total_gll = ngll_ * ngll_ * ngll_ * ne_;

    geom_.Compute(fes);
    dofs_.ComputeScalar(fes, order);
    geom_.EnableDevice();
    geom_.SyncToDevice();
    dofs_.EnableDevice();
    dofs_.SyncToDevice();

    auto alloc_dev = [&](Vector& v) {
        v.SetSize(total_gll);
        v.UseDevice(true);
    };

    alloc_dev(vp_coeff_);
    alloc_dev(inv_rho_);
    InitCoefficients(kappa, inv_rho, unrelaxed_correction);

    alloc_dev(vp_kernel_);  vp_kernel_  = 0.0;
    alloc_dev(rho_kernel_); rho_kernel_ = 0.0;
    alloc_dev(vp_hessian_); vp_hessian_ = 0.0;
    alloc_dev(rho_hessian_); rho_hessian_ = 0.0;
}

// =============================================================================
// InitCoefficients
// =============================================================================

void IsotropicAcousticSensitivity3D::InitCoefficients(
    const MaterialField3D& kappa, const MaterialField3D& inv_rho,
    const MaterialField3D* unrelaxed_correction)
{
    const int total_gll = ngll_ * ngll_ * ngll_ * ne_;

    // c_vp = 2/(c·ρ·Vp_user³) with Vp_user = sqrt(κ_u/(c·ρ)); for pure acoustic
    // (c ≡ 1) this reduces to 2/(ρ·Vp³). Mirrors the 2D hand implementation.
    {
        const real_t* d_kappa   = kappa.Read();
        const real_t* d_inv_rho = inv_rho.Read();
        const real_t* d_corr    = unrelaxed_correction ? unrelaxed_correction->Read()
                                                         : nullptr;
        real_t* d_vp_coeff      = vp_coeff_.Write();
        mfem::forall(total_gll, [=] MFEM_HOST_DEVICE (int i) {
            const real_t k_u   = d_kappa[i];
            const real_t ir    = d_inv_rho[i];
            const real_t rho   = 1.0 / ir;
            const real_t c_corr = d_corr ? d_corr[i] : 1.0;
            const real_t k_user = k_u / c_corr;
            const real_t vp2    = k_user * ir;
            const real_t vp     = sqrt(vp2);
            const real_t rho_vp3 = rho * vp2 * vp;
            const real_t c      = 2.0 / (c_corr * rho_vp3);
            d_vp_coeff[i] = c;
        });
    }
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

void IsotropicAcousticSensitivity3D::Reset() {
    vp_kernel_  = 0.0;
    rho_kernel_ = 0.0;
}

void IsotropicAcousticSensitivity3D::ResetHessian() {
    vp_hessian_  = 0.0;
    rho_hessian_ = 0.0;
}

// =============================================================================
// Accumulate (dispatch)
// =============================================================================

void IsotropicAcousticSensitivity3D::Accumulate(
    const Vector& fwd_p, const Vector& fwd_a,
    const Vector& adj_p, real_t dt)
{
    SEM_DISPATCH_NGLL_3D(ngll_, AccumulateVpKernel, fwd_a, adj_p, dt);
    SEM_DISPATCH_NGLL_3D(ngll_, AccumulateRhoKernel, fwd_p, adj_p, dt);
}

// =============================================================================
// Vp Kernel: K_Vp -= 2/(ρ·Vp³) · p̈_fwd · p_adj · dt   (pointwise)
// =============================================================================

template <int NGLL>
void IsotropicAcousticSensitivity3D::AccumulateVpKernel(
    const Vector& fwd_a, const Vector& adj_p, real_t dt)
{
    const int ne = ne_;

    const real_t* d_fwd_a = fwd_a.Read();
    const real_t* d_adj_p = adj_p.Read();
    real_t* d_kernel  = vp_kernel_.ReadWrite();
    real_t* d_hessian = vp_hessian_.ReadWrite();

    auto gather_map = dofs_.ViewGatherMap();
    auto vp_coeff   = Reshape(vp_coeff_.Read(), ngll_, ngll_, ngll_, ne_);

    mfem::forall_3D(ne, NGLL, NGLL, NGLL, [=] MFEM_HOST_DEVICE (int ei) {
        MFEM_FOREACH_THREAD(iz, z, NGLL) {
            MFEM_FOREACH_THREAD(iy, y, NGLL) {
                MFEM_FOREACH_THREAD(ix, x, NGLL) {
                    const int gll_idx   = gather_map(ix, iy, iz, ei);
                    const int local_idx =
                        ix + iy * NGLL + iz * NGLL * NGLL
                        + ei * NGLL * NGLL * NGLL;

                    const real_t c_vp  = vp_coeff(ix, iy, iz, ei);
                    const real_t a_fwd = d_fwd_a[gll_idx];
                    const real_t p_adj = d_adj_p[gll_idx];

                    d_kernel[local_idx]  -= c_vp * a_fwd * p_adj * dt;
                    d_hessian[local_idx] += a_fwd * a_fwd * dt;
                }
            }
        }
    });
}

// =============================================================================
// ρ Kernel: K_ρ -= (1/ρ²) · ∇p_fwd · ∇p_adj · dt   (3-axis gradient)
// =============================================================================

template <int NGLL>
void IsotropicAcousticSensitivity3D::AccumulateRhoKernel(
    const Vector& fwd_p, const Vector& adj_p, real_t dt)
{
    const int ne = ne_;

    const real_t* d_fwd_p = fwd_p.Read();
    const real_t* d_adj_p = adj_p.Read();
    real_t* d_kernel     = rho_kernel_.ReadWrite();
    real_t* d_rho_hess   = rho_hessian_.ReadWrite();

    auto gather_map = dofs_.ViewGatherMap();
    auto inv_rho    = Reshape(inv_rho_.Read(), ngll_, ngll_, ngll_, ne_);
    auto invJP      = geom_.ViewInvJPacked();
    auto dshape     = geom_.ViewDShape();

    mfem::forall_3D(ne, NGLL, NGLL, NGLL, [=] MFEM_HOST_DEVICE (int ei) {
        MFEM_SHARED real_t s_fwd[NGLL][NGLL][NGLL];
        MFEM_SHARED real_t s_adj[NGLL][NGLL][NGLL];
        MFEM_SHARED real_t s_hprime[NGLL][NGLL];

        if (MFEM_THREAD_ID(z) == 0) {
            MFEM_FOREACH_THREAD(iy, y, NGLL) {
                MFEM_FOREACH_THREAD(ix, x, NGLL) {
                    s_hprime[ix][iy] = dshape(iy, ix);
                }
            }
        }

        MFEM_FOREACH_THREAD(iz, z, NGLL) {
            MFEM_FOREACH_THREAD(iy, y, NGLL) {
                MFEM_FOREACH_THREAD(ix, x, NGLL) {
                    const int gll_idx = gather_map(ix, iy, iz, ei);
                    s_fwd[iz][iy][ix] = d_fwd_p[gll_idx];
                    s_adj[iz][iy][ix] = d_adj_p[gll_idx];
                }
            }
        }
        MFEM_SYNC_THREAD;

        MFEM_FOREACH_THREAD(iz, z, NGLL) {
            MFEM_FOREACH_THREAD(iy, y, NGLL) {
                MFEM_FOREACH_THREAD(ix, x, NGLL) {
                    real_t dfwd_xi = 0, dfwd_eta = 0, dfwd_zeta = 0;
                    real_t dadj_xi = 0, dadj_eta = 0, dadj_zeta = 0;

                    SEM_NOUNROLL
                    for (int k = 0; k < NGLL; ++k) {
                        const real_t hp_x = s_hprime[ix][k];
                        dfwd_xi  += s_fwd[iz][iy][k] * hp_x;
                        dadj_xi  += s_adj[iz][iy][k] * hp_x;

                        const real_t hp_y = s_hprime[iy][k];
                        dfwd_eta += s_fwd[iz][k][ix] * hp_y;
                        dadj_eta += s_adj[iz][k][ix] * hp_y;

                        const real_t hp_z = s_hprime[iz][k];
                        dfwd_zeta += s_fwd[k][iy][ix] * hp_z;
                        dadj_zeta += s_adj[k][iy][ix] * hp_z;
                    }

                    // Packed invJ: comp 0-8 = invJ row-major, 9 = detJ (unused here).
                    const real_t j0 = invJP(0, ix, iy, iz, ei);
                    const real_t j1 = invJP(1, ix, iy, iz, ei);
                    const real_t j2 = invJP(2, ix, iy, iz, ei);
                    const real_t j3 = invJP(3, ix, iy, iz, ei);
                    const real_t j4 = invJP(4, ix, iy, iz, ei);
                    const real_t j5 = invJP(5, ix, iy, iz, ei);
                    const real_t j6 = invJP(6, ix, iy, iz, ei);
                    const real_t j7 = invJP(7, ix, iy, iz, ei);
                    const real_t j8 = invJP(8, ix, iy, iz, ei);

                    // Physical gradients (same transform as forward kernel).
                    const real_t dfwd_dx = dfwd_xi * j0 + dfwd_eta * j3 + dfwd_zeta * j6;
                    const real_t dfwd_dy = dfwd_xi * j1 + dfwd_eta * j4 + dfwd_zeta * j7;
                    const real_t dfwd_dz = dfwd_xi * j2 + dfwd_eta * j5 + dfwd_zeta * j8;
                    const real_t dadj_dx = dadj_xi * j0 + dadj_eta * j3 + dadj_zeta * j6;
                    const real_t dadj_dy = dadj_xi * j1 + dadj_eta * j4 + dadj_zeta * j7;
                    const real_t dadj_dz = dadj_xi * j2 + dadj_eta * j5 + dadj_zeta * j8;

                    const real_t grad_dot =
                        dfwd_dx * dadj_dx + dfwd_dy * dadj_dy + dfwd_dz * dadj_dz;
                    const real_t grad_sq_f =
                        dfwd_dx * dfwd_dx + dfwd_dy * dfwd_dy + dfwd_dz * dfwd_dz;

                    const int local_idx =
                        ix + iy * NGLL + iz * NGLL * NGLL
                        + ei * NGLL * NGLL * NGLL;
                    const real_t ir = inv_rho(ix, iy, iz, ei);
                    d_kernel[local_idx]   -= ir * ir * grad_dot * dt;
                    d_rho_hess[local_idx] += grad_sq_f * dt;
                }
            }
        }
    });
}

// =============================================================================
// Save
// =============================================================================

void IsotropicAcousticSensitivity3D::Save(
    const std::string& dir, ParMesh& /*mesh*/, int source_id)
{
    const int total = ngll_ * ngll_ * ngll_ * ne_;

    MaterialField3D vp_field(ne_, ngll_, ngll_, ngll_);
    MaterialField3D rho_field(ne_, ngll_, ngll_, ngll_);

    {
        const real_t* h_src = vp_kernel_.HostRead();
        real_t* h_dst = vp_field.HostWrite();
        for (int i = 0; i < total; ++i) h_dst[i] = h_src[i];
    }
    {
        const real_t* h_src = rho_kernel_.HostRead();
        real_t* h_dst = rho_field.HostWrite();
        for (int i = 0; i < total; ++i) h_dst[i] = h_src[i];
    }

    std::ostringstream oss;
    oss << "_src" << std::setfill('0') << std::setw(3) << source_id;
    const std::string suffix = oss.str();
    const std::string mesh_path = dir + "/mesh.mesh";

    MPI_Comm comm = fes_->GetComm();
    SaveFieldBP(dir + "/kernel_vp"  + suffix + ".bp", "data", vp_field,  mesh_path, comm);
    SaveFieldBP(dir + "/kernel_rho" + suffix + ".bp", "data", rho_field, mesh_path, comm);
}

void IsotropicAcousticSensitivity3D::SaveHessian(
    const std::string& dir, ParMesh& /*mesh*/, int source_id)
{
    const int total = ngll_ * ngll_ * ngll_ * ne_;

    MaterialField3D vp_h(ne_, ngll_, ngll_, ngll_);
    MaterialField3D rho_h(ne_, ngll_, ngll_, ngll_);

    const real_t* h_vp_src  = vp_hessian_.HostRead();
    const real_t* h_rho_src = rho_hessian_.HostRead();
    const real_t* h_c_vp    = vp_coeff_.HostRead();
    const real_t* h_ir      = inv_rho_.HostRead();
    real_t* h_vp_dst  = vp_h.HostWrite();
    real_t* h_rho_dst = rho_h.HostWrite();
    for (int i = 0; i < total; ++i) {
        const real_t c  = h_c_vp[i];
        const real_t ir = h_ir[i];
        h_vp_dst[i]  = c * c * h_vp_src[i];
        h_rho_dst[i] = ir * ir * h_rho_src[i];
    }

    std::ostringstream oss;
    oss << "_src" << std::setfill('0') << std::setw(3) << source_id;
    const std::string suffix = oss.str();
    const std::string mesh_path = dir + "/mesh.mesh";

    MPI_Comm comm = fes_->GetComm();
    SaveFieldBP(dir + "/hessian_vp"  + suffix + ".bp", "data", vp_h,  mesh_path, comm);
    SaveFieldBP(dir + "/hessian_rho" + suffix + ".bp", "data", rho_h, mesh_path, comm);
}

// =============================================================================
// NGLL instantiations
// =============================================================================

#define INSTANTIATE_VP_3D(NGLL) \
    template void IsotropicAcousticSensitivity3D::AccumulateVpKernel<NGLL>( \
        const Vector&, const Vector&, real_t);

#define INSTANTIATE_RHO_3D(NGLL) \
    template void IsotropicAcousticSensitivity3D::AccumulateRhoKernel<NGLL>( \
        const Vector&, const Vector&, real_t);

SEM_INSTANTIATE_NGLL_3D(INSTANTIATE_VP_3D)
SEM_INSTANTIATE_NGLL_3D(INSTANTIATE_RHO_3D)

#undef INSTANTIATE_VP_3D
#undef INSTANTIATE_RHO_3D

}  // namespace SEM
