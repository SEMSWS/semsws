/**
 * @file IsotropicAcousticSensitivityAD3D.cpp
 * @brief AD-based implementation of acoustic 3D sensitivity kernels.
 *
 * 3D analogue of IsotropicAcousticSensitivityAD2D: two single-seed AD
 * passes per time step (inv_kappa for the mass path, inv_rho for the
 * stiffness path) with a 3-axis gradient via AcousticFluxPhysical3D<dual>.
 * FinalizeKernels() applies the (Vp, ρ)-independent chain rule to
 * produce (K_Vp, K_ρ).
 */

#include "fwi/IsotropicAcousticSensitivityAD3D.hpp"
#include "integ/kernels/AcousticFluxKernel3D.hpp"
#include "util/FESOrder.hpp"
#include "integ/core/SEMKernelDispatch.hpp"
#include "common/GpuMacros.hpp"
#include <mfem.hpp>
#include "io/ADIOS2IO.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>

namespace SEM {

using Dual = mfem::future::dual<real_t, real_t>;

// =============================================================================
// Constructor / InitMaterialFields
// =============================================================================

IsotropicAcousticSensitivityAD3D::IsotropicAcousticSensitivityAD3D(
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

    auto alloc = [&](Vector& v) { v.SetSize(total_gll); v.UseDevice(true); };
    alloc(inv_rho_); alloc(inv_kappa_); alloc(kappa_);
    alloc(unrelaxed_correction_);
    InitMaterialFields(kappa, inv_rho, unrelaxed_correction);

    alloc(k_invrho_);   k_invrho_   = 0.0;
    alloc(k_invkappa_); k_invkappa_ = 0.0;
    alloc(vp_hessian_); vp_hessian_ = 0.0;
    alloc(rho_hessian_); rho_hessian_ = 0.0;

    alloc(vp_kernel_);
    alloc(rho_kernel_);
}

void IsotropicAcousticSensitivityAD3D::InitMaterialFields(
    const MaterialField3D& kappa, const MaterialField3D& inv_rho,
    const MaterialField3D* unrelaxed_correction)
{
    const int total_gll = ngll_ * ngll_ * ngll_ * ne_;

    {
        const real_t* d_src = inv_rho.Read();
        real_t* d_dst = inv_rho_.Write();
        mfem::forall(total_gll, [=] MFEM_HOST_DEVICE (int i) { d_dst[i] = d_src[i]; });
    }
    {
        const real_t* d_src = kappa.Read();
        real_t* d_dst = kappa_.Write();
        mfem::forall(total_gll, [=] MFEM_HOST_DEVICE (int i) { d_dst[i] = d_src[i]; });
    }
    {
        const real_t* d_k = kappa.Read();
        real_t* d_ik = inv_kappa_.Write();
        mfem::forall(total_gll, [=] MFEM_HOST_DEVICE (int i) { d_ik[i] = 1.0 / d_k[i]; });
    }

    // Unrelaxed correction: copy if provided, else fill with 1.0 (pure acoustic).
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

void IsotropicAcousticSensitivityAD3D::Reset() {
    k_invrho_   = 0.0;
    k_invkappa_ = 0.0;
}

void IsotropicAcousticSensitivityAD3D::ResetHessian() {
    vp_hessian_  = 0.0;
    rho_hessian_ = 0.0;
}

// =============================================================================
// Accumulate (NGLL dispatch)
// =============================================================================

void IsotropicAcousticSensitivityAD3D::Accumulate(
    const Vector& fwd_p, const Vector& fwd_a,
    const Vector& adj_p, real_t dt)
{
    SEM_DISPATCH_NGLL_3D(ngll_, AccumulateVpKernel_AD,  fwd_a, adj_p, dt);
    SEM_DISPATCH_NGLL_3D(ngll_, AccumulateRhoKernel_AD, fwd_p, adj_p, dt);
}

// =============================================================================
// Vp (mass) path via AD on inv_kappa
// =============================================================================

template <int NGLL>
void IsotropicAcousticSensitivityAD3D::AccumulateVpKernel_AD(
    const Vector& fwd_a, const Vector& adj_p, real_t dt)
{
    const int ne = ne_;

    const real_t* d_fwd_a = fwd_a.Read();
    const real_t* d_adj_p = adj_p.Read();
    real_t* d_k_invkappa  = k_invkappa_.ReadWrite();
    real_t* d_hessian     = vp_hessian_.ReadWrite();

    auto gather_map = dofs_.ViewGatherMap();
    auto inv_kappa  = Reshape(inv_kappa_.Read(), ngll_, ngll_, ngll_, ne_);

    mfem::forall_3D(ne, NGLL, NGLL, NGLL, [=] MFEM_HOST_DEVICE (int ei) {
        MFEM_FOREACH_THREAD(iz, z, NGLL) {
            MFEM_FOREACH_THREAD(iy, y, NGLL) {
                MFEM_FOREACH_THREAD(ix, x, NGLL) {
                    const int gll_idx = gather_map(ix, iy, iz, ei);
                    const int local_idx =
                        ix + iy * NGLL + iz * NGLL * NGLL
                        + ei * NGLL * NGLL * NGLL;

                    const real_t ik_val = inv_kappa(ix, iy, iz, ei);
                    const real_t a_fwd  = d_fwd_a[gll_idx];
                    const real_t p_adj  = d_adj_p[gll_idx];

                    Dual d_ik{ik_val, 1.0};
                    Dual integrand = d_ik * a_fwd * p_adj;
                    d_k_invkappa[local_idx] += integrand.gradient * dt;

                    d_hessian[local_idx] += a_fwd * a_fwd * dt;
                }
            }
        }
    });
}

// =============================================================================
// ρ (stiffness) path via AD on inv_rho — 3-axis gradient
// =============================================================================

template <int NGLL>
void IsotropicAcousticSensitivityAD3D::AccumulateRhoKernel_AD(
    const Vector& fwd_p, const Vector& adj_p, real_t dt)
{
    const int ne = ne_;

    const real_t* d_fwd_p = fwd_p.Read();
    const real_t* d_adj_p = adj_p.Read();
    real_t* d_k_invrho    = k_invrho_.ReadWrite();
    real_t* d_rho_hess    = rho_hessian_.ReadWrite();

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
                        dfwd_xi += s_fwd[iz][iy][k] * hp_x;
                        dadj_xi += s_adj[iz][iy][k] * hp_x;

                        const real_t hp_y = s_hprime[iy][k];
                        dfwd_eta += s_fwd[iz][k][ix] * hp_y;
                        dadj_eta += s_adj[iz][k][ix] * hp_y;

                        const real_t hp_z = s_hprime[iz][k];
                        dfwd_zeta += s_fwd[k][iy][ix] * hp_z;
                        dadj_zeta += s_adj[k][iy][ix] * hp_z;
                    }

                    const real_t j0 = invJP(0, ix, iy, iz, ei);
                    const real_t j1 = invJP(1, ix, iy, iz, ei);
                    const real_t j2 = invJP(2, ix, iy, iz, ei);
                    const real_t j3 = invJP(3, ix, iy, iz, ei);
                    const real_t j4 = invJP(4, ix, iy, iz, ei);
                    const real_t j5 = invJP(5, ix, iy, iz, ei);
                    const real_t j6 = invJP(6, ix, iy, iz, ei);
                    const real_t j7 = invJP(7, ix, iy, iz, ei);
                    const real_t j8 = invJP(8, ix, iy, iz, ei);

                    const real_t dfwd_dx = dfwd_xi * j0 + dfwd_eta * j3 + dfwd_zeta * j6;
                    const real_t dfwd_dy = dfwd_xi * j1 + dfwd_eta * j4 + dfwd_zeta * j7;
                    const real_t dfwd_dz = dfwd_xi * j2 + dfwd_eta * j5 + dfwd_zeta * j8;
                    const real_t dadj_dx = dadj_xi * j0 + dadj_eta * j3 + dadj_zeta * j6;
                    const real_t dadj_dy = dadj_xi * j1 + dadj_eta * j4 + dadj_zeta * j7;
                    const real_t dadj_dz = dadj_xi * j2 + dadj_eta * j5 + dadj_zeta * j8;

                    // AD seed on inv_rho → physical flux q = (1/ρ) ∇φ.
                    const real_t ir_val = inv_rho(ix, iy, iz, ei);
                    Dual d_ir{ir_val, 1.0};
                    Dual qx, qy, qz;
                    AcousticFluxPhysical3D<Dual>(d_ir, dfwd_dx, dfwd_dy, dfwd_dz,
                                                 qx, qy, qz);

                    // Integrand: q · ∇λ  →  .gradient = ∇φ·∇λ = ∂/∂(1/ρ).
                    Dual integrand = qx * dadj_dx + qy * dadj_dy + qz * dadj_dz;

                    const int loc =
                        ix + iy * NGLL + iz * NGLL * NGLL
                        + ei * NGLL * NGLL * NGLL;
                    d_k_invrho[loc] += integrand.gradient * dt;

                    // Pseudo-Hessian (same as hand): (1/ρ²) · |∇φ_fwd|² · dt.
                    const real_t fwd_grad_sq =
                        dfwd_dx * dfwd_dx + dfwd_dy * dfwd_dy + dfwd_dz * dfwd_dz;
                    d_rho_hess[loc] += fwd_grad_sq * dt;
                }
            }
        }
    });
}

// =============================================================================
// FinalizeKernels — (Vp, ρ)-independent chain rule with viscoacoustic correction
// =============================================================================
//
// For viscoacoustic, kappa_ stores the unrelaxed κ_u = c · κ_user. The FWI
// parameter is Vp_user = √(κ_user/ρ), so the chain rule maps K_{1/κ_u} → K_Vp
// via
//     K_{Vp_user} = −2/(c · ρ · Vp_user³) · K_{1/κ_u}
// Pure acoustic (c ≡ 1) reduces to the original 2/(ρ·Vp³) formula.
void IsotropicAcousticSensitivityAD3D::FinalizeKernels() const
{
    const int total = ngll_ * ngll_ * ngll_ * ne_;
    const real_t* h_kir  = k_invrho_.HostRead();
    const real_t* h_kik  = k_invkappa_.HostRead();
    const real_t* h_ir   = inv_rho_.HostRead();
    const real_t* h_k    = kappa_.HostRead();
    const real_t* h_corr = unrelaxed_correction_.HostRead();
    real_t* h_vp  = vp_kernel_.HostWrite();
    real_t* h_rho = rho_kernel_.HostWrite();
    for (int i = 0; i < total; ++i) {
        const real_t ir     = h_ir[i];
        const real_t rho    = 1.0 / ir;
        const real_t c_corr = h_corr[i];
        const real_t k_user = h_k[i] / c_corr;
        const real_t vp2    = k_user * ir;
        const real_t vp     = std::sqrt(vp2);
        const real_t c_vp   = 2.0 / (c_corr * rho * vp2 * vp);
        h_vp[i]  = -c_vp * h_kik[i];
        h_rho[i] = -(ir * ir) * h_kir[i];
    }
}

// =============================================================================
// Save / SaveHessian
// =============================================================================

void IsotropicAcousticSensitivityAD3D::Save(
    const std::string& dir, ParMesh& /*mesh*/, int source_id)
{
    const int total = ngll_ * ngll_ * ngll_ * ne_;
    FinalizeKernels();

    MaterialField3D rho_field(ne_, ngll_, ngll_, ngll_);
    MaterialField3D vp_field(ne_, ngll_, ngll_, ngll_);
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

void IsotropicAcousticSensitivityAD3D::SaveHessian(
    const std::string& dir, ParMesh& /*mesh*/, int source_id)
{
    const int total = ngll_ * ngll_ * ngll_ * ne_;

    MaterialField3D vp_h(ne_, ngll_, ngll_, ngll_);
    MaterialField3D rho_h(ne_, ngll_, ngll_, ngll_);

    const real_t* h_vp_src  = vp_hessian_.HostRead();
    const real_t* h_rho_src = rho_hessian_.HostRead();
    const real_t* h_ik      = inv_kappa_.HostRead();
    const real_t* h_ir      = inv_rho_.HostRead();
    real_t* h_vp_dst  = vp_h.HostWrite();
    real_t* h_rho_dst = rho_h.HostWrite();
    for (int i = 0; i < total; ++i) {
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

    MPI_Comm comm = fes_->GetComm();
    SaveFieldBP(dir + "/hessian_vp"  + suffix + ".bp", "data", vp_h,  mesh_path, comm);
    SaveFieldBP(dir + "/hessian_rho" + suffix + ".bp", "data", rho_h, mesh_path, comm);
}

// =============================================================================
// NGLL instantiations
// =============================================================================

#define INSTANTIATE_VP_AD_3D(NGLL) \
    template void IsotropicAcousticSensitivityAD3D::AccumulateVpKernel_AD<NGLL>( \
        const Vector&, const Vector&, real_t);

#define INSTANTIATE_RHO_AD_3D(NGLL) \
    template void IsotropicAcousticSensitivityAD3D::AccumulateRhoKernel_AD<NGLL>( \
        const Vector&, const Vector&, real_t);

SEM_INSTANTIATE_NGLL_3D(INSTANTIATE_VP_AD_3D)
SEM_INSTANTIATE_NGLL_3D(INSTANTIATE_RHO_AD_3D)

#undef INSTANTIATE_VP_AD_3D
#undef INSTANTIATE_RHO_AD_3D

}  // namespace SEM
