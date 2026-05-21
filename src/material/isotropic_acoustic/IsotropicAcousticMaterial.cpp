/**
 * @file IsotropicAcousticMaterial.cpp
 * @brief Implementation of isotropic acoustic material classes (2D and 3D) with optional attenuation
 */

#include "material/isotropic_acoustic/IsotropicAcousticMaterial.hpp"
#include "material/isotropic_acoustic/IsotropicAcousticLoader.hpp"
#include "material/isotropic_acoustic/IsotropicAcousticBuilder.hpp"
#include "material/InterpolatingCoefficient.hpp"
#include "integ/attenuation/AttenuationCoeffs.hpp"
#include "config/ConfigTypes.hpp"
#include <map>
#include <cmath>
#include <limits>

namespace SEM {

// =============================================================================
// IsotropicAcousticMaterial Implementation (2D)
// =============================================================================

IsotropicAcousticMaterial::IsotropicAcousticMaterial(int ne, int ngllx, int nglly)
    : AcousticMaterialBase2D(ne, ngllx, nglly),
      kappa_(ne, ngllx, nglly),
      inv_rho_(ne, ngllx, nglly)
{
}

IsotropicAcousticMaterial::IsotropicAcousticMaterial(
    Coefficient& vp, Coefficient& rho,
    const ParFiniteElementSpace& fes, const IntegrationRule& ir)
    : AcousticMaterialBase2D(fes.GetNE(),
                             (int)std::sqrt((real_t)ir.GetNPoints()),
                             (int)std::sqrt((real_t)ir.GetNPoints()))
{
    MFEM_VERIFY(ngllx_ * nglly_ == ir.GetNPoints(), "Integration rule must be square");

    kappa_ = MaterialField(ne_, ngllx_, nglly_);
    inv_rho_ = MaterialField(ne_, ngllx_, nglly_);

    InitializeFromVelocity(vp, rho, fes, ir);
}

IsotropicAcousticMaterial::IsotropicAcousticMaterial(
    Coefficient& kappa_c, Coefficient& invrho_c,
    const ParFiniteElementSpace& fes, const IntegrationRule& ir,
    bool /*kappa_invrho*/)
    : AcousticMaterialBase2D(fes.GetNE(),
                             (int)std::sqrt((real_t)ir.GetNPoints()),
                             (int)std::sqrt((real_t)ir.GetNPoints()))
{
    kappa_ = MaterialField(ne_, ngllx_, nglly_);
    inv_rho_ = MaterialField(ne_, ngllx_, nglly_);

    kappa_.ProjectCoefficient(kappa_c, fes, ir);
    inv_rho_.ProjectCoefficient(invrho_c, fes, ir);
}

void IsotropicAcousticMaterial::InitializeFromVelocity(
    Coefficient& vp_coef, Coefficient& rho_coef,
    const ParFiniteElementSpace& fes, const IntegrationRule& ir)
{
    real_t* kappa_data = kappa_.HostWrite();
    real_t* invrho_data = inv_rho_.HostWrite();

    int stride_e = nglly_ * ngllx_;

    for (int e = 0; e < ne_; e++)
    {
        ElementTransformation* Tr = fes.GetElementTransformation(e);

        for (int j = 0; j < nglly_; j++)
        {
            for (int i = 0; i < ngllx_; i++)
            {
                int ip_idx = j * ngllx_ + i;
                const IntegrationPoint& ip = ir.IntPoint(ip_idx);
                Tr->SetIntPoint(&ip);

                real_t vp = vp_coef.Eval(*Tr, ip);
                real_t rho = rho_coef.Eval(*Tr, ip);

                // kappa = rho * vp^2 (bulk modulus)
                real_t kappa = rho * vp * vp;
                real_t invrho = 1.0 / rho;

                int flat_idx = e * stride_e + j * ngllx_ + i;
                kappa_data[flat_idx] = kappa;
                invrho_data[flat_idx] = invrho;
            }
        }
    }
}

std::unique_ptr<IsotropicAcousticMaterial> IsotropicAcousticMaterial::FromConfig(
    const MaterialConfig& config,
    ParFiniteElementSpace& fes,
    const IntegrationRule& ir)
{
    // Use the new Loader/Builder pattern
    IsotropicAcousticInput input = IsotropicAcousticLoader::Load(config, /*is_3d=*/false);
    return IsotropicAcousticBuilder<2>::Build(input, fes, ir);
}

void IsotropicAcousticMaterial::InitializeAttenuationConstant(
    real_t qkappa_val, real_t f0, int n_units)
{
    f0_ = f0;
    n_units_ = n_units;

    // Allocate Q field using dimensions from base class
    qkappa_ = std::make_unique<MaterialField>(ne_, ngllx_, nglly_);

    // Set constant Q value
    qkappa_->SetConstant(qkappa_val);
}

void IsotropicAcousticMaterial::InitializeAttenuationFromCoefficient(
    Coefficient& qkappa_coef,
    const ParFiniteElementSpace& fes, const IntegrationRule& ir,
    real_t f0, int n_units)
{
    f0_ = f0;
    n_units_ = n_units;

    // Allocate Q field using dimensions from base class
    qkappa_ = std::make_unique<MaterialField>(ne_, ngllx_, nglly_);

    // Project coefficient to Q field
    qkappa_->ProjectCoefficient(qkappa_coef, fes, ir);
}

void IsotropicAcousticMaterial::AllocateAttenuationFields(real_t f0, int n_units)
{
    f0_ = f0;
    n_units_ = n_units;

    // Allocate Q field using dimensions from base class
    qkappa_ = std::make_unique<MaterialField>(ne_, ngllx_, nglly_);
}

real_t IsotropicAcousticMaterial::GetMaxVelocity() const
{
    // Vp = sqrt(kappa / rho) = sqrt(kappa * inv_rho)
    // Compute max(Vp) across all GLL points
    const real_t* kappa_data = kappa_.Data().HostRead();
    const real_t* inv_rho_data = inv_rho_.Data().HostRead();

    int total_pts = kappa_.Size();
    real_t max_vp = 0.0;

    for (int idx = 0; idx < total_pts; idx++) {
        real_t vp = std::sqrt(kappa_data[idx] * inv_rho_data[idx]);
        if (vp > max_vp) max_vp = vp;
    }

    return max_vp;
}

real_t IsotropicAcousticMaterial::GetMinVelocity() const
{
    // Vp = sqrt(kappa / rho) = sqrt(kappa * inv_rho)
    // Acoustic only has P-waves, so min velocity = min(Vp)
    const real_t* kappa_data = kappa_.Data().HostRead();
    const real_t* inv_rho_data = inv_rho_.Data().HostRead();

    int total_pts = kappa_.Size();
    real_t min_vp = std::numeric_limits<real_t>::max();

    for (int idx = 0; idx < total_pts; idx++) {
        real_t vp = std::sqrt(kappa_data[idx] * inv_rho_data[idx]);
        if (vp < min_vp) min_vp = vp;
    }

    return min_vp;
}

real_t IsotropicAcousticMaterial::GetElementMinVelocity(int e) const
{
    // Vp = sqrt(kappa * inv_rho)
    // Compute min(Vp) within element e
    auto kappa_view = kappa_.ViewHost();
    auto inv_rho_view = inv_rho_.ViewHost();

    int ngllx = kappa_.NumGLLx();
    int nglly = kappa_.NumGLLy();
    real_t min_vp = std::numeric_limits<real_t>::max();

    for (int j = 0; j < nglly; j++) {
        for (int i = 0; i < ngllx; i++) {
            real_t vp = std::sqrt(kappa_view(i, j, e) * inv_rho_view(i, j, e));
            if (vp < min_vp) min_vp = vp;
        }
    }

    return min_vp;
}

real_t IsotropicAcousticMaterial::GetElementMaxVelocity(int e) const
{
    // Vp = sqrt(kappa * inv_rho)
    // Compute max(Vp) within element e
    auto kappa_view = kappa_.ViewHost();
    auto inv_rho_view = inv_rho_.ViewHost();

    int ngllx = kappa_.NumGLLx();
    int nglly = kappa_.NumGLLy();
    real_t max_vp = 0.0;

    for (int j = 0; j < nglly; j++) {
        for (int i = 0; i < ngllx; i++) {
            real_t vp = std::sqrt(kappa_view(i, j, e) * inv_rho_view(i, j, e));
            if (vp > max_vp) max_vp = vp;
        }
    }

    return max_vp;
}

void IsotropicAcousticMaterial::ApplyAttenuationCorrection()
{
    // Skip if no attenuation or already corrected
    if (!HasAttenuation() || kappa_corrected_) {
        return;
    }

    int total = kappa_.Size();
    real_t* kappa_data = kappa_.Data().HostReadWrite();
    const real_t* qkappa_data = qkappa_->Data().HostRead();

    // Allocate per-GLL correction storage so sensitivity kernels can recover
    // the user-facing kappa (κ_user = κ_u / c_i) for correct K_Vp chain rule.
    unrelaxed_correction_ = std::make_unique<MaterialField>(
        kappa_.NumElements(), kappa_.NumGLLx(), kappa_.NumGLLy());
    real_t* corr_data = unrelaxed_correction_->Data().HostWrite();

    real_t fQmin = 0.1 * f0_;
    real_t fQmax = 10.0 * f0_;

    // Apply unrelaxed correction to each GLL point
    // kappa_unrelaxed = kappa_relaxed * unrelaxed_correction
    for (int i = 0; i < total; i++) {
        AttenuationParams params = ComputeAttenuationCoeffsCached(
            n_units_, qkappa_data[i], fQmin, fQmax);
        corr_data[i] = params.unrelaxed_correction;
        kappa_data[i] *= params.unrelaxed_correction;
    }

    kappa_corrected_ = true;
}


// =============================================================================
// IsotropicAcousticMaterial3D Implementation
// =============================================================================

IsotropicAcousticMaterial3D::IsotropicAcousticMaterial3D(int ne, int ngllx, int nglly, int ngllz)
    : AcousticMaterialBase3D(ne, ngllx, nglly, ngllz),
      kappa_(ne, ngllx, nglly, ngllz),
      inv_rho_(ne, ngllx, nglly, ngllz)
{
}

IsotropicAcousticMaterial3D::IsotropicAcousticMaterial3D(
    Coefficient& vp, Coefficient& rho,
    const ParFiniteElementSpace& fes, const IntegrationRule& ir)
    : AcousticMaterialBase3D(fes.GetNE(),
                             (int)std::cbrt((real_t)ir.GetNPoints()),
                             (int)std::cbrt((real_t)ir.GetNPoints()),
                             (int)std::cbrt((real_t)ir.GetNPoints()))
{
    MFEM_VERIFY(ngllx_ * nglly_ * ngllz_ == ir.GetNPoints(), "Integration rule must be cubic");

    kappa_ = MaterialField3D(ne_, ngllx_, nglly_, ngllz_);
    inv_rho_ = MaterialField3D(ne_, ngllx_, nglly_, ngllz_);

    InitializeFromVelocity(vp, rho, fes, ir);
}

void IsotropicAcousticMaterial3D::InitializeFromVelocity(
    Coefficient& vp, Coefficient& rho,
    const ParFiniteElementSpace& fes, const IntegrationRule& ir)
{
    // kappa = rho * vp^2
    // inv_rho = 1 / rho
    real_t* kappa_data = kappa_.HostWrite();
    real_t* inv_rho_data = inv_rho_.HostWrite();

    int stride_k = nglly_ * ngllx_;
    int stride_e = ngllz_ * stride_k;

    for (int e = 0; e < ne_; e++)
    {
        ElementTransformation* Tr = fes.GetElementTransformation(e);

        for (int k = 0; k < ngllz_; k++)
        {
            for (int j = 0; j < nglly_; j++)
            {
                for (int i = 0; i < ngllx_; i++)
                {
                    int ip_idx = k * nglly_ * ngllx_ + j * ngllx_ + i;
                    const IntegrationPoint& ip = ir.IntPoint(ip_idx);
                    Tr->SetIntPoint(&ip);

                    real_t vp_val = vp.Eval(*Tr, ip);
                    real_t rho_val = rho.Eval(*Tr, ip);

                    int flat_idx = e * stride_e + k * stride_k + j * ngllx_ + i;
                    kappa_data[flat_idx] = rho_val * vp_val * vp_val;
                    inv_rho_data[flat_idx] = 1.0 / rho_val;
                }
            }
        }
    }
}

std::unique_ptr<IsotropicAcousticMaterial3D> IsotropicAcousticMaterial3D::FromConfig(
    const MaterialConfig& config,
    ParFiniteElementSpace& fes,
    const IntegrationRule& ir)
{
    IsotropicAcousticInput input = IsotropicAcousticLoader::Load(config, /*is_3d=*/true);
    return IsotropicAcousticBuilder<3>::Build(input, fes, ir);
}

void IsotropicAcousticMaterial3D::InitializeAttenuationConstant(
    real_t qkappa_val, real_t f0, int n_units)
{
    f0_ = f0;
    n_units_ = n_units;

    // Allocate Q field
    qkappa_ = std::make_unique<MaterialField3D>(ne_, ngllx_, nglly_, ngllz_);

    // Set constant Q value
    qkappa_->SetConstant(qkappa_val);
}

void IsotropicAcousticMaterial3D::InitializeAttenuationFromCoefficient(
    Coefficient& qkappa_coef,
    const ParFiniteElementSpace& fes, const IntegrationRule& ir,
    real_t f0, int n_units)
{
    f0_ = f0;
    n_units_ = n_units;

    // Allocate Q field
    qkappa_ = std::make_unique<MaterialField3D>(ne_, ngllx_, nglly_, ngllz_);

    // Project coefficient to Q field
    qkappa_->ProjectCoefficient(qkappa_coef, fes, ir);
}

void IsotropicAcousticMaterial3D::AllocateAttenuationFields(real_t f0, int n_units)
{
    f0_ = f0;
    n_units_ = n_units;

    // Allocate Q field (values will be set later per-element)
    qkappa_ = std::make_unique<MaterialField3D>(ne_, ngllx_, nglly_, ngllz_);
}

real_t IsotropicAcousticMaterial3D::GetMaxVelocity() const
{
    // Vp = sqrt(kappa / rho) = sqrt(kappa * inv_rho)
    // Compute max(Vp) across all GLL points
    const real_t* kappa_data = kappa_.Data().HostRead();
    const real_t* inv_rho_data = inv_rho_.Data().HostRead();

    int total_pts = kappa_.Size();
    real_t max_vp = 0.0;

    for (int idx = 0; idx < total_pts; idx++) {
        real_t vp = std::sqrt(kappa_data[idx] * inv_rho_data[idx]);
        if (vp > max_vp) max_vp = vp;
    }

    return max_vp;
}

real_t IsotropicAcousticMaterial3D::GetMinVelocity() const
{
    // Vp = sqrt(kappa / rho) = sqrt(kappa * inv_rho)
    // Acoustic only has P-waves, so min velocity = min(Vp)
    const real_t* kappa_data = kappa_.Data().HostRead();
    const real_t* inv_rho_data = inv_rho_.Data().HostRead();

    int total_pts = kappa_.Size();
    real_t min_vp = std::numeric_limits<real_t>::max();

    for (int idx = 0; idx < total_pts; idx++) {
        real_t vp = std::sqrt(kappa_data[idx] * inv_rho_data[idx]);
        if (vp < min_vp) min_vp = vp;
    }

    return min_vp;
}

real_t IsotropicAcousticMaterial3D::GetElementMinVelocity(int e) const
{
    // Vp = sqrt(kappa * inv_rho)
    // Compute min(Vp) within element e
    auto kappa_view = kappa_.ViewHost();
    auto inv_rho_view = inv_rho_.ViewHost();

    int ngllx = kappa_.NumGLLx();
    int nglly = kappa_.NumGLLy();
    int ngllz = kappa_.NumGLLz();
    real_t min_vp = std::numeric_limits<real_t>::max();

    for (int k = 0; k < ngllz; k++) {
        for (int j = 0; j < nglly; j++) {
            for (int i = 0; i < ngllx; i++) {
                real_t vp = std::sqrt(kappa_view(i, j, k, e) * inv_rho_view(i, j, k, e));
                if (vp < min_vp) min_vp = vp;
            }
        }
    }

    return min_vp;
}

real_t IsotropicAcousticMaterial3D::GetElementMaxVelocity(int e) const
{
    // Vp = sqrt(kappa * inv_rho)
    // Compute max(Vp) within element e
    auto kappa_view = kappa_.ViewHost();
    auto inv_rho_view = inv_rho_.ViewHost();

    int ngllx = kappa_.NumGLLx();
    int nglly = kappa_.NumGLLy();
    int ngllz = kappa_.NumGLLz();
    real_t max_vp = 0.0;

    for (int k = 0; k < ngllz; k++) {
        for (int j = 0; j < nglly; j++) {
            for (int i = 0; i < ngllx; i++) {
                real_t vp = std::sqrt(kappa_view(i, j, k, e) * inv_rho_view(i, j, k, e));
                if (vp > max_vp) max_vp = vp;
            }
        }
    }

    return max_vp;
}

void IsotropicAcousticMaterial3D::ApplyAttenuationCorrection()
{
    // Skip if no attenuation or already corrected
    if (!HasAttenuation() || kappa_corrected_) {
        return;
    }

    int total = kappa_.Size();
    real_t* kappa_data = kappa_.Data().HostReadWrite();
    const real_t* qkappa_data = qkappa_->Data().HostRead();

    // Allocate per-GLL correction storage (see 2D analog).
    unrelaxed_correction_ = std::make_unique<MaterialField3D>(
        kappa_.NumElements(), kappa_.NumGLLx(), kappa_.NumGLLy(), kappa_.NumGLLz());
    real_t* corr_data = unrelaxed_correction_->Data().HostWrite();

    real_t fQmin = 0.1 * f0_;
    real_t fQmax = 10.0 * f0_;

    // Apply unrelaxed correction to each GLL point
    // kappa_unrelaxed = kappa_relaxed * unrelaxed_correction
    for (int i = 0; i < total; i++) {
        AttenuationParams params = ComputeAttenuationCoeffsCached(
            n_units_, qkappa_data[i], fQmin, fQmax);
        corr_data[i] = params.unrelaxed_correction;
        kappa_data[i] *= params.unrelaxed_correction;
    }

    kappa_corrected_ = true;
}

}  // namespace SEM
