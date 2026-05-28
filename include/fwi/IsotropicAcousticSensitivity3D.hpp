/**
 * @file IsotropicAcousticSensitivity3D.hpp
 * @brief Sensitivity kernel for 3D isotropic acoustic media.
 *
 * Dim=3 counterpart to `IsotropicAcousticSensitivity2D`. Same math, same
 * public API; the only differences are the extra gradient component and
 * the 10-entry invJPacked layout (9 invJ entries + detJ).
 *
 *   K_Vp = -Σ_t 2/(ρ·Vp³) · p̈_fwd · p_adj · dt       (mass path)
 *   K_ρ  = -Σ_t (1/ρ²) · ∇p_fwd · ∇p_adj · dt         (stiffness path)
 *
 * Pseudo-Hessian (Shin diagonal, forward-only):
 *   H_Vp += 4/(ρ²·Vp⁶) · p̈_fwd² · dt
 *   H_ρ  += (1/ρ²) · |∇p_fwd|² · dt
 */

#ifndef SEM_ISOTROPIC_ACOUSTIC_SENSITIVITY_3D_HPP
#define SEM_ISOTROPIC_ACOUSTIC_SENSITIVITY_3D_HPP

#include "fwi/SensitivityKernel.hpp"
#include "integ/core/SEMGeometry.hpp"
#include "integ/core/SEMDofOrdering.hpp"
#include "material/MaterialField.hpp"

namespace SEM {

using mfem::ParFiniteElementSpace;

class IsotropicAcousticSensitivity3D : public SensitivityKernelBase3D {
public:
    /// @param unrelaxed_correction Per-GLL c_i = κ_u/κ_user field from
    ///                              viscoacoustic material; pass nullptr for
    ///                              pure acoustic (kernel uses c ≡ 1).
    IsotropicAcousticSensitivity3D(ParFiniteElementSpace& fes,
                                    const MaterialField3D& kappa,
                                    const MaterialField3D& inv_rho,
                                    const MaterialField3D* unrelaxed_correction = nullptr);

    void InitCoefficients(const MaterialField3D& kappa,
                          const MaterialField3D& inv_rho,
                          const MaterialField3D* unrelaxed_correction = nullptr);

    void Accumulate(const Vector& fwd_p, const Vector& fwd_a,
                    const Vector& adj_p, real_t dt) override;

    void Save(const std::string& dir, ParMesh& mesh, int source_id) override;
    void SaveHessian(const std::string& dir, ParMesh& mesh, int source_id) override;
    void Reset() override;
    void ResetHessian() override;
    int Ngll() const override { return ngll_; }

    const Vector& VpKernel() const { return vp_kernel_; }
    const Vector& RhoKernel() const { return rho_kernel_; }

    // GPU kernels (public for CUDA extended lambda)
    template <int NGLL>
    void AccumulateVpKernel(const Vector& fwd_a, const Vector& adj_p, real_t dt);

    template <int NGLL>
    void AccumulateRhoKernel(const Vector& fwd_p, const Vector& adj_p, real_t dt);

private:
    SEMGeometry3D geom_;
    SEMDofOrdering3D dofs_;

    Vector vp_coeff_;
    Vector inv_rho_;

    Vector vp_kernel_;
    Vector rho_kernel_;
    Vector vp_hessian_;
    Vector rho_hessian_;

    ParFiniteElementSpace* fes_;
    int ngll_ = 0;
    int ne_   = 0;
};

}  // namespace SEM

#endif  // SEM_ISOTROPIC_ACOUSTIC_SENSITIVITY_3D_HPP
