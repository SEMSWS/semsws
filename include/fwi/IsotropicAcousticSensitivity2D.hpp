/**
 * @file IsotropicAcousticSensitivity2D.hpp
 * @brief Sensitivity kernel for 2D isotropic acoustic media
 *
 * Accumulates Vp and ρ sensitivity kernels directly from forward and adjoint wavefields:
 *   K_Vp = -Σ_t 2/(ρ·Vp³) · p̈_fwd · p_adj · dt   (velocity kernel)
 *   K_ρ  = -Σ_t (1/ρ²) · ∇p_fwd · ∇p_adj · dt     (density kernel, from d(1/ρ)/dρ = -1/ρ²)
 *
 * Pseudo-Hessian (Shin diagonal approximation):
 *   H_Vp = Σ_t 4/(ρ²·Vp⁶) · p̈_fwd² · dt
 *   H_ρ  = Σ_t (1/ρ²) · |∇p_fwd|² · dt
 *
 * Uses the same forall_2D GPU kernel pattern as IsotropicAcousticKernels2D.
 * Reuses SEMGeometry2D and SEMDofOrdering2D standalone (not inheriting integrator).
 */

#ifndef SEM_ISOTROPIC_ACOUSTIC_SENSITIVITY_2D_HPP
#define SEM_ISOTROPIC_ACOUSTIC_SENSITIVITY_2D_HPP

#include "fwi/SensitivityKernel.hpp"
#include "integ/core/SEMGeometry.hpp"
#include "integ/core/SEMDofOrdering.hpp"
#include "material/MaterialField.hpp"

namespace SEM {

using mfem::ParFiniteElementSpace;

class IsotropicAcousticSensitivity2D : public SensitivityKernelBase2D {
public:
    /**
     * @brief Construct sensitivity kernel for isotropic acoustic 2D
     *
     * @param fes Scalar finite element space (same as forward simulation)
     * @param kappa Bulk modulus field (κ at GLL points)
     * @param inv_rho Inverse density field (1/ρ at GLL points)
     */
    /// @param unrelaxed_correction Per-GLL c_i = κ_u/κ_user field from
    ///                              viscoacoustic material; pass nullptr for
    ///                              pure acoustic (kernel uses c ≡ 1).
    IsotropicAcousticSensitivity2D(ParFiniteElementSpace& fes,
                                    const MaterialField& kappa,
                                    const MaterialField& inv_rho,
                                    const MaterialField* unrelaxed_correction = nullptr);

    /// Initialize device-resident coefficient arrays (separate from ctor for CUDA)
    void InitCoefficients(const MaterialField& kappa, const MaterialField& inv_rho,
                          const MaterialField* unrelaxed_correction = nullptr);

    void Accumulate(const Vector& fwd_p, const Vector& fwd_a,
                    const Vector& adj_p, real_t dt) override;

    void Save(const std::string& dir, ParMesh& mesh, int source_id) override;
    void Reset() override;
    int Ngll() const override { return ngll_; }

    void SaveHessian(const std::string& dir, ParMesh& mesh, int source_id);
    void ResetHessian();

    const Vector& VpKernel() const { return vp_kernel_; }
    const Vector& RhoKernel() const { return rho_kernel_; }

    // GPU kernel methods (public for CUDA extended lambda)
    template<int NGLL>
    void AccumulateVpKernel(const Vector& fwd_a, const Vector& adj_p, real_t dt);

    template<int NGLL>
    void AccumulateRhoKernel(const Vector& fwd_p, const Vector& adj_p, real_t dt);

private:
    // Geometry and DOF ordering (reused from integrator pattern)
    SEMGeometry2D geom_;
    SEMDofOrdering2D dofs_;

    Vector vp_coeff_;
    Vector inv_rho_;

    Vector vp_kernel_;
    Vector rho_kernel_;

    Vector vp_hessian_;
    Vector rho_hessian_;

    // FE space for output
    ParFiniteElementSpace* fes_;

    // Dimensions
    int ngll_ = 0;
    int ne_ = 0;
};

}  // namespace SEM

#endif  // SEM_ISOTROPIC_ACOUSTIC_SENSITIVITY_2D_HPP
