/**
 * @file IsotropicElasticMaterial.cpp
 * @brief Implementation of isotropic elastic material classes with optional attenuation
 */

#include "material/isotropic_elastic/IsotropicElasticMaterial.hpp"
#include "material/isotropic_elastic/IsotropicElasticLoader.hpp"
#include "material/isotropic_elastic/IsotropicElasticBuilder.hpp"
#include "material/InterpolatingCoefficient.hpp"
#include "integ/attenuation/AttenuationCoeffs.hpp"
#include "config/ConfigTypes.hpp"
#include <map>
#include <fstream>
#include <cmath>
#include <limits>

namespace SEM {

// =============================================================================
// IsotropicElasticMaterial Implementation (2D)
// =============================================================================

IsotropicElasticMaterial::IsotropicElasticMaterial(int ne, int ngllx, int nglly)
    : ElasticMaterialBase2D(ne, ngllx, nglly),
      kappa_(ne, ngllx, nglly),
      lambda_(ne, ngllx, nglly),
      mu_(ne, ngllx, nglly),
      rho_(ne, ngllx, nglly)
{
}

IsotropicElasticMaterial::IsotropicElasticMaterial(
    Coefficient& vp, Coefficient& vs, Coefficient& rho,
    const ParFiniteElementSpace& fes, const IntegrationRule& ir)
    : ElasticMaterialBase2D(fes.GetNE(),
                            (int)std::sqrt((real_t)ir.GetNPoints()),
                            (int)std::sqrt((real_t)ir.GetNPoints()))
{
    // Verify dimensions
    MFEM_VERIFY(ngllx_ * nglly_ == ir.GetNPoints(), "Integration rule must be square");

    kappa_ = MaterialField(ne_, ngllx_, nglly_);
    lambda_ = MaterialField(ne_, ngllx_, nglly_);
    mu_ = MaterialField(ne_, ngllx_, nglly_);
    rho_ = MaterialField(ne_, ngllx_, nglly_);

    InitializeFromVelocities(vp, vs, rho, fes, ir);
    ComputeKappaFromLambdaMu();
}

IsotropicElasticMaterial::IsotropicElasticMaterial(
    Coefficient& lambda_c, Coefficient& mu_c, Coefficient& rho_c,
    const ParFiniteElementSpace& fes, const IntegrationRule& ir,
    bool /*lame_params*/)
    : ElasticMaterialBase2D(fes.GetNE(),
                            (int)std::sqrt((real_t)ir.GetNPoints()),
                            (int)std::sqrt((real_t)ir.GetNPoints()))
{
    MFEM_VERIFY(ngllx_ * nglly_ == ir.GetNPoints(), "Integration rule must be square");

    kappa_ = MaterialField(ne_, ngllx_, nglly_);
    lambda_ = MaterialField(ne_, ngllx_, nglly_);
    mu_ = MaterialField(ne_, ngllx_, nglly_);
    rho_ = MaterialField(ne_, ngllx_, nglly_);

    InitializeFromLame(lambda_c, mu_c, rho_c, fes, ir);
    ComputeKappaFromLambdaMu();
}

void IsotropicElasticMaterial::InitializeFromVelocities(
    Coefficient& vp_coef, Coefficient& vs_coef, Coefficient& rho_coef,
    const ParFiniteElementSpace& fes, const IntegrationRule& ir)
{
    real_t* lambda_data = lambda_.HostWrite();
    real_t* mu_data = mu_.HostWrite();
    real_t* rho_data = rho_.HostWrite();

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
                real_t vs = vs_coef.Eval(*Tr, ip);
                real_t rho = rho_coef.Eval(*Tr, ip);

                real_t mu = rho * vs * vs;
                real_t lambda = rho * vp * vp - 2.0 * mu;

                int flat_idx = e * stride_e + j * ngllx_ + i;
                lambda_data[flat_idx] = lambda;
                mu_data[flat_idx] = mu;
                rho_data[flat_idx] = rho;
            }
        }
    }
}

void IsotropicElasticMaterial::InitializeFromLame(
    Coefficient& lambda_c, Coefficient& mu_c, Coefficient& rho_c,
    const ParFiniteElementSpace& fes, const IntegrationRule& ir)
{
    lambda_.ProjectCoefficient(lambda_c, fes, ir);
    mu_.ProjectCoefficient(mu_c, fes, ir);
    rho_.ProjectCoefficient(rho_c, fes, ir);
}

std::unique_ptr<IsotropicElasticMaterial> IsotropicElasticMaterial::FromConfig(
    const MaterialConfig& config,
    ParFiniteElementSpace& fes,
    const IntegrationRule& ir)
{
    // Use the new Loader/Builder pattern
    // This delegates to IsotropicElasticLoader and IsotropicElasticBuilder<2>
    IsotropicElasticInput input = IsotropicElasticLoader::Load(config, /*is_3d=*/false);
    return IsotropicElasticBuilder<2>::Build(input, fes, ir);
}

void IsotropicElasticMaterial::InitializeAttenuationConstant(
    real_t qkappa_val, real_t qmu_val, real_t f0, int n_units)
{
    f0_ = f0;
    n_units_ = n_units;

    // Allocate Q fields using dimensions from base class
    qkappa_ = std::make_unique<MaterialField>(ne_, ngllx_, nglly_);
    qmu_ = std::make_unique<MaterialField>(ne_, ngllx_, nglly_);

    // Set constant Q values
    qkappa_->SetConstant(qkappa_val);
    qmu_->SetConstant(qmu_val);
}

void IsotropicElasticMaterial::InitializeAttenuationFromCoefficient(
    Coefficient& qkappa_coef, Coefficient& qmu_coef,
    const ParFiniteElementSpace& fes, const IntegrationRule& ir,
    real_t f0, int n_units)
{
    f0_ = f0;
    n_units_ = n_units;

    // Allocate Q fields using dimensions from base class
    qkappa_ = std::make_unique<MaterialField>(ne_, ngllx_, nglly_);
    qmu_ = std::make_unique<MaterialField>(ne_, ngllx_, nglly_);

    // Project coefficients to Q fields
    qkappa_->ProjectCoefficient(qkappa_coef, fes, ir);
    qmu_->ProjectCoefficient(qmu_coef, fes, ir);
}

void IsotropicElasticMaterial::AllocateAttenuationFields(real_t f0, int n_units)
{
    f0_ = f0;
    n_units_ = n_units;

    // Allocate Q fields using dimensions from base class
    qkappa_ = std::make_unique<MaterialField>(ne_, ngllx_, nglly_);
    qmu_ = std::make_unique<MaterialField>(ne_, ngllx_, nglly_);
}

real_t IsotropicElasticMaterial::GetMaxVelocity() const
{
    // Vp = sqrt((lambda + 2*mu) / rho)
    // Compute max(Vp) across all GLL points
    // Use HostRead() to ensure we read from host memory (not GPU)
    const real_t* lambda_data = lambda_.Data().HostRead();
    const real_t* mu_data = mu_.Data().HostRead();
    const real_t* rho_data = rho_.Data().HostRead();

    int total_pts = lambda_.Size();
    real_t max_vp = 0.0;

    for (int idx = 0; idx < total_pts; idx++) {
        real_t modulus = lambda_data[idx] + 2.0 * mu_data[idx];
        real_t vp = std::sqrt(modulus / rho_data[idx]);
        if (vp > max_vp) max_vp = vp;
    }

    return max_vp;
}

real_t IsotropicElasticMaterial::GetMinVelocity() const
{
    // Vs = sqrt(mu / rho)
    // Compute min(Vs) across all GLL points
    // Use HostRead() to ensure we read from host memory (not GPU)
    const real_t* mu_data = mu_.Data().HostRead();
    const real_t* rho_data = rho_.Data().HostRead();

    int total_pts = mu_.Size();
    real_t min_vs = std::numeric_limits<real_t>::max();

    for (int idx = 0; idx < total_pts; idx++) {
        real_t vs = std::sqrt(mu_data[idx] / rho_data[idx]);
        if (vs < min_vs) min_vs = vs;
    }

    return min_vs;
}

real_t IsotropicElasticMaterial::GetElementMinVelocity(int e) const
{
    // Vs = sqrt(mu / rho)
    // Compute min(Vs) within element e
    auto mu_view = mu_.ViewHost();
    auto rho_view = rho_.ViewHost();

    int ngllx = mu_.NumGLLx();
    int nglly = mu_.NumGLLy();
    real_t min_vs = std::numeric_limits<real_t>::max();

    for (int j = 0; j < nglly; j++) {
        for (int i = 0; i < ngllx; i++) {
            real_t vs = std::sqrt(mu_view(i, j, e) / rho_view(i, j, e));
            if (vs < min_vs) min_vs = vs;
        }
    }

    return min_vs;
}

real_t IsotropicElasticMaterial::GetElementMaxVelocity(int e) const
{
    // Vp = sqrt((lambda + 2*mu) / rho)
    // Compute max(Vp) within element e
    auto lambda_view = lambda_.ViewHost();
    auto mu_view = mu_.ViewHost();
    auto rho_view = rho_.ViewHost();

    int ngllx = lambda_.NumGLLx();
    int nglly = lambda_.NumGLLy();
    real_t max_vp = 0.0;

    for (int j = 0; j < nglly; j++) {
        for (int i = 0; i < ngllx; i++) {
            real_t modulus = lambda_view(i, j, e) + 2.0 * mu_view(i, j, e);
            real_t vp = std::sqrt(modulus / rho_view(i, j, e));
            if (vp > max_vp) max_vp = vp;
        }
    }

    return max_vp;
}

void IsotropicElasticMaterial::ComputeKappaFromLambdaMu()
{
    // 2D plane strain: kappa = lambda + mu
    int total = lambda_.Size();

    // Use HostRead/HostWrite for CPU computation
    const real_t* lam = lambda_.Data().HostRead();
    const real_t* m = mu_.Data().HostRead();
    real_t* k = kappa_.Data().HostWrite();

    for (int i = 0; i < total; i++) {
        k[i] = lam[i] + m[i];
    }
}

void IsotropicElasticMaterial::ApplyAttenuationCorrection()
{
    // Skip if no attenuation or already corrected
    if (!HasAttenuation() || moduli_corrected_) {
        return;
    }

    int total = kappa_.Size();
    real_t* kappa_data = kappa_.Data().HostReadWrite();
    real_t* mu_data = mu_.Data().HostReadWrite();
    const real_t* qkappa_data = qkappa_->Data().HostRead();
    const real_t* qmu_data = qmu_->Data().HostRead();

    // Allocate per-GLL correction fields so sensitivity kernels can recover
    // the user-facing (κ_user, μ_user) via chain rule.
    unrelaxed_correction_kappa_ = std::make_unique<MaterialField>(
        kappa_.NumElements(), kappa_.NumGLLx(), kappa_.NumGLLy());
    unrelaxed_correction_mu_ = std::make_unique<MaterialField>(
        kappa_.NumElements(), kappa_.NumGLLx(), kappa_.NumGLLy());
    real_t* corr_kappa_data = unrelaxed_correction_kappa_->Data().HostWrite();
    real_t* corr_mu_data    = unrelaxed_correction_mu_->Data().HostWrite();

    real_t fQmin = 0.1 * f0_;
    real_t fQmax = 10.0 * f0_;

    // Apply unrelaxed correction to each GLL point
    // kappa_unrelaxed = kappa_relaxed * unrelaxed_correction
    // mu_unrelaxed = mu_relaxed * unrelaxed_correction
    for (int i = 0; i < total; i++) {
        AttenuationParams params_kappa = ComputeAttenuationCoeffsCached(
            n_units_, qkappa_data[i], fQmin, fQmax);
        AttenuationParams params_mu = ComputeAttenuationCoeffsCached(
            n_units_, qmu_data[i], fQmin, fQmax);

        corr_kappa_data[i] = params_kappa.unrelaxed_correction;
        corr_mu_data[i]    = params_mu.unrelaxed_correction;
        kappa_data[i] *= params_kappa.unrelaxed_correction;
        mu_data[i]    *= params_mu.unrelaxed_correction;
    }

    moduli_corrected_ = true;
}


// =============================================================================
// IsotropicElasticMaterial3D Implementation
// =============================================================================

IsotropicElasticMaterial3D::IsotropicElasticMaterial3D(int ne, int ngllx, int nglly, int ngllz)
    : ElasticMaterialBase3D(ne, ngllx, nglly, ngllz),
      kappa_(ne, ngllx, nglly, ngllz),
      lambda_(ne, ngllx, nglly, ngllz),
      mu_(ne, ngllx, nglly, ngllz),
      rho_(ne, ngllx, nglly, ngllz)
{
}

IsotropicElasticMaterial3D::IsotropicElasticMaterial3D(
    Coefficient& lambda_c, Coefficient& mu_c, Coefficient& rho_c,
    const ParFiniteElementSpace& fes, const IntegrationRule& ir)
    : ElasticMaterialBase3D(fes.GetNE(),
                            (int)std::cbrt((real_t)ir.GetNPoints()),
                            (int)std::cbrt((real_t)ir.GetNPoints()),
                            (int)std::cbrt((real_t)ir.GetNPoints()))
{
    // Verify dimensions
    MFEM_VERIFY(ngllx_ * nglly_ * ngllz_ == ir.GetNPoints(), "Integration rule must be cubic");

    kappa_ = MaterialField3D(ne_, ngllx_, nglly_, ngllz_);
    lambda_ = MaterialField3D(ne_, ngllx_, nglly_, ngllz_);
    mu_ = MaterialField3D(ne_, ngllx_, nglly_, ngllz_);
    rho_ = MaterialField3D(ne_, ngllx_, nglly_, ngllz_);

    lambda_.ProjectCoefficient(lambda_c, fes, ir);
    mu_.ProjectCoefficient(mu_c, fes, ir);
    rho_.ProjectCoefficient(rho_c, fes, ir);
    ComputeKappaFromLambdaMu();
}

std::unique_ptr<IsotropicElasticMaterial3D> IsotropicElasticMaterial3D::FromConfig(
    const MaterialConfig& config,
    ParFiniteElementSpace& fes,
    const IntegrationRule& ir)
{
    // Use the new Loader/Builder pattern
    // This delegates to IsotropicElasticLoader and IsotropicElasticBuilder<3>
    IsotropicElasticInput input = IsotropicElasticLoader::Load(config, /*is_3d=*/true);
    return IsotropicElasticBuilder<3>::Build(input, fes, ir);
}

void IsotropicElasticMaterial3D::InitializeAttenuationConstant(
    real_t qkappa_val, real_t qmu_val, real_t f0, int n_units)
{
    f0_ = f0;
    n_units_ = n_units;

    // Allocate Q fields using dimensions from base class
    qkappa_ = std::make_unique<MaterialField3D>(ne_, ngllx_, nglly_, ngllz_);
    qmu_ = std::make_unique<MaterialField3D>(ne_, ngllx_, nglly_, ngllz_);

    // Set constant Q values
    qkappa_->SetConstant(qkappa_val);
    qmu_->SetConstant(qmu_val);
}

void IsotropicElasticMaterial3D::InitializeAttenuationFromCoefficient(
    Coefficient& qkappa_coef, Coefficient& qmu_coef,
    const ParFiniteElementSpace& fes, const IntegrationRule& ir,
    real_t f0, int n_units)
{
    f0_ = f0;
    n_units_ = n_units;

    // Allocate Q fields using dimensions from base class
    qkappa_ = std::make_unique<MaterialField3D>(ne_, ngllx_, nglly_, ngllz_);
    qmu_ = std::make_unique<MaterialField3D>(ne_, ngllx_, nglly_, ngllz_);

    // Project coefficients to Q fields
    qkappa_->ProjectCoefficient(qkappa_coef, fes, ir);
    qmu_->ProjectCoefficient(qmu_coef, fes, ir);
}

void IsotropicElasticMaterial3D::AllocateAttenuationFields(real_t f0, int n_units)
{
    f0_ = f0;
    n_units_ = n_units;

    // Allocate Q fields using dimensions from base class
    qkappa_ = std::make_unique<MaterialField3D>(ne_, ngllx_, nglly_, ngllz_);
    qmu_ = std::make_unique<MaterialField3D>(ne_, ngllx_, nglly_, ngllz_);
}

real_t IsotropicElasticMaterial3D::GetMaxVelocity() const
{
    // Vp = sqrt((lambda + 2*mu) / rho)
    // Compute max(Vp) across all GLL points
    // Use HostRead() to ensure we read from host memory (not GPU)
    const real_t* lambda_data = lambda_.Data().HostRead();
    const real_t* mu_data = mu_.Data().HostRead();
    const real_t* rho_data = rho_.Data().HostRead();

    int total_pts = lambda_.Size();
    real_t max_vp = 0.0;

    for (int idx = 0; idx < total_pts; idx++) {
        real_t modulus = lambda_data[idx] + 2.0 * mu_data[idx];
        real_t vp = std::sqrt(modulus / rho_data[idx]);
        if (vp > max_vp) max_vp = vp;
    }

    return max_vp;
}

real_t IsotropicElasticMaterial3D::GetMinVelocity() const
{
    // Vs = sqrt(mu / rho)
    // Compute min(Vs) across all GLL points
    // Use HostRead() to ensure we read from host memory (not GPU)
    const real_t* mu_data = mu_.Data().HostRead();
    const real_t* rho_data = rho_.Data().HostRead();

    int total_pts = mu_.Size();
    real_t min_vs = std::numeric_limits<real_t>::max();

    for (int idx = 0; idx < total_pts; idx++) {
        real_t vs = std::sqrt(mu_data[idx] / rho_data[idx]);
        if (vs < min_vs) min_vs = vs;
    }

    return min_vs;
}

real_t IsotropicElasticMaterial3D::GetElementMinVelocity(int e) const
{
    // Vs = sqrt(mu / rho)
    // Compute min(Vs) within element e
    auto mu_view = mu_.ViewHost();
    auto rho_view = rho_.ViewHost();

    int ngllx = mu_.NumGLLx();
    int nglly = mu_.NumGLLy();
    int ngllz = mu_.NumGLLz();
    real_t min_vs = std::numeric_limits<real_t>::max();

    for (int k = 0; k < ngllz; k++) {
        for (int j = 0; j < nglly; j++) {
            for (int i = 0; i < ngllx; i++) {
                real_t vs = std::sqrt(mu_view(i, j, k, e) / rho_view(i, j, k, e));
                if (vs < min_vs) min_vs = vs;
            }
        }
    }

    return min_vs;
}

real_t IsotropicElasticMaterial3D::GetElementMaxVelocity(int e) const
{
    // Vp = sqrt((lambda + 2*mu) / rho)
    // Compute max(Vp) within element e
    auto lambda_view = lambda_.ViewHost();
    auto mu_view = mu_.ViewHost();
    auto rho_view = rho_.ViewHost();

    int ngllx = lambda_.NumGLLx();
    int nglly = lambda_.NumGLLy();
    int ngllz = lambda_.NumGLLz();
    real_t max_vp = 0.0;

    for (int k = 0; k < ngllz; k++) {
        for (int j = 0; j < nglly; j++) {
            for (int i = 0; i < ngllx; i++) {
                real_t modulus = lambda_view(i, j, k, e) + 2.0 * mu_view(i, j, k, e);
                real_t vp = std::sqrt(modulus / rho_view(i, j, k, e));
                if (vp > max_vp) max_vp = vp;
            }
        }
    }

    return max_vp;
}

void IsotropicElasticMaterial3D::ComputeKappaFromLambdaMu()
{
    // 3D: kappa = lambda + (2/3)*mu
    int total = lambda_.Size();

    // Use HostRead/HostWrite for CPU computation
    const real_t* lam = lambda_.Data().HostRead();
    const real_t* m = mu_.Data().HostRead();
    real_t* k = kappa_.Data().HostWrite();

    for (int i = 0; i < total; i++) {
        k[i] = lam[i] + (2.0 / 3.0) * m[i];
    }
}

void IsotropicElasticMaterial3D::ApplyAttenuationCorrection()
{
    // Skip if no attenuation or already corrected
    if (!HasAttenuation() || moduli_corrected_) {
        return;
    }

    int total = kappa_.Size();
    real_t* kappa_data = kappa_.Data().HostReadWrite();
    real_t* mu_data = mu_.Data().HostReadWrite();
    const real_t* qkappa_data = qkappa_->Data().HostRead();
    const real_t* qmu_data = qmu_->Data().HostRead();

    // Allocate per-GLL correction fields (see 2D analog).
    unrelaxed_correction_kappa_ = std::make_unique<MaterialField3D>(
        kappa_.NumElements(), kappa_.NumGLLx(), kappa_.NumGLLy(), kappa_.NumGLLz());
    unrelaxed_correction_mu_ = std::make_unique<MaterialField3D>(
        kappa_.NumElements(), kappa_.NumGLLx(), kappa_.NumGLLy(), kappa_.NumGLLz());
    real_t* corr_kappa_data = unrelaxed_correction_kappa_->Data().HostWrite();
    real_t* corr_mu_data    = unrelaxed_correction_mu_->Data().HostWrite();

    real_t fQmin = 0.1 * f0_;
    real_t fQmax = 10.0 * f0_;

    // Apply unrelaxed correction to each GLL point
    // kappa_unrelaxed = kappa_relaxed * unrelaxed_correction
    // mu_unrelaxed = mu_relaxed * unrelaxed_correction
    for (int i = 0; i < total; i++) {
        AttenuationParams params_kappa = ComputeAttenuationCoeffsCached(
            n_units_, qkappa_data[i], fQmin, fQmax);
        AttenuationParams params_mu = ComputeAttenuationCoeffsCached(
            n_units_, qmu_data[i], fQmin, fQmax);

        corr_kappa_data[i] = params_kappa.unrelaxed_correction;
        corr_mu_data[i]    = params_mu.unrelaxed_correction;
        kappa_data[i] *= params_kappa.unrelaxed_correction;
        mu_data[i]    *= params_mu.unrelaxed_correction;
    }

    moduli_corrected_ = true;
}

}  // namespace SEM
