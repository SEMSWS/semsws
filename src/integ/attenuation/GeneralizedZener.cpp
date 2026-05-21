/**
 * @file GeneralizedZener.cpp
 * @brief Implementation of Generalized Zener viscoelastic attenuation
 */

#include "integ/attenuation/GeneralizedZener.hpp"
#include <iostream>
#include <cmath>

namespace SEM {

// =============================================================================
// GeneralizedZener2D Implementation
// =============================================================================

void GeneralizedZener2D::EnableAttenuation(
    int ne, int ngllx, int nglly,
    const Vector& Qkappa, const Vector& Qmu,
    Vector& kappa, Vector& mu,
    real_t f0, int n_sls, real_t dt)
{
    dt_ = dt;

    // Initialize memory state
    memory_ = std::make_unique<ViscoelasticMemory2D>(ne, ngllx, nglly, n_sls);

    // Compute frequency band (same as 3D)
    real_t fQmin = 0.1 * f0;
    real_t fQmax = 10.0 * f0;

    // Use HostRead() for CPU-side attenuation coefficient computation
    const real_t* qkappa = Qkappa.HostRead();
    const real_t* qmu = Qmu.HostRead();
    real_t* kappa_data = kappa.HostWrite();
    real_t* mu_data = mu.HostWrite();

    // Packed coefficient array: [4, ngllx, nglly, ne, n_units]
    // Components: 0=alpha_kappa, 1=alpha_mu, 2=strain_coeff_kappa, 3=strain_coeff_mu
    auto coeffs = Reshape(memory_->coeffs_packed.HostWrite(), 4, ngllx, nglly, ne, n_sls);

    for (int ie = 0; ie < ne; ie++)
    {
        for (int iy = 0; iy < nglly; iy++)
        {
            for (int ix = 0; ix < ngllx; ix++)
            {
                int gll_idx = (ie * nglly + iy) * ngllx + ix;
                real_t Qk = qkappa[gll_idx];
                real_t Qm = qmu[gll_idx];

                // Compute attenuation coefficients using cached version for performance
                // Note: unrelaxed correction is now applied in material class
                // via ApplyAttenuationCorrection() before operator setup
                AttenuationParams params_kappa = ComputeAttenuationCoeffsCached(n_sls, Qk, fQmin, fQmax);
                AttenuationParams params_mu = ComputeAttenuationCoeffsCached(n_sls, Qm, fQmin, fQmax);

                // Compute normalization sums (sum of tau_eps/tau_sig)
                real_t sum_beta_kappa = 0.0, sum_beta_mu = 0.0;
                for (int i = 0; i < n_sls; i++)
                {
                    sum_beta_kappa += params_kappa.tau_epsilon[i] / params_kappa.tau_sigma[i];
                    sum_beta_mu += params_mu.tau_epsilon[i] / params_mu.tau_sigma[i];
                }

                // Store parameters for each mechanism
                for (int i = 0; i < n_sls; i++)
                {
                    // beta = tau_eps/tau_sig - 1 (same as 3D)
                    real_t beta_k = params_kappa.tau_epsilon[i] / params_kappa.tau_sigma[i] - 1.0;
                    real_t beta_m = params_mu.tau_epsilon[i] / params_mu.tau_sigma[i] - 1.0;

                    real_t tauinv_pos_k = 1.0 / params_kappa.tau_sigma[i];
                    real_t tauinv_pos_m = 1.0 / params_mu.tau_sigma[i];
                    real_t tauinv_neg_k = -tauinv_pos_k;
                    real_t tauinv_neg_m = -tauinv_pos_m;

                    // norm_factor: NOTE shear has factor of 2 (same as 3D)
                    real_t norm_factor_k = (beta_k * tauinv_pos_k) / sum_beta_kappa;
                    real_t norm_factor_m = (2.0 * beta_m * tauinv_pos_m) / sum_beta_mu;

                    // Crank-Nicolson coefficients (unconditionally stable, 2nd order)
                    real_t half_eta_k = 0.5 * dt * tauinv_pos_k;
                    real_t half_eta_m = 0.5 * dt * tauinv_pos_m;

                    real_t alpha_k = (1.0 - half_eta_k) / (1.0 + half_eta_k);
                    real_t alpha_m = (1.0 - half_eta_m) / (1.0 + half_eta_m);

                    real_t strain_coeff_k = (dt / 2.0) / (1.0 + half_eta_k);
                    real_t strain_coeff_m = (dt / 2.0) / (1.0 + half_eta_m);

                    // Store to packed coefficient array: [4, ngllx, nglly, ne, n_units]
                    coeffs(0, ix, iy, ie, i) = alpha_k;                          // alpha_kappa
                    coeffs(1, ix, iy, ie, i) = alpha_m;                          // alpha_mu
                    coeffs(2, ix, iy, ie, i) = strain_coeff_k * norm_factor_k;   // strain_coeff_kappa
                    coeffs(3, ix, iy, ie, i) = strain_coeff_m * norm_factor_m;   // strain_coeff_mu
                }
            }
        }
    }

    // Force host→device sync for GPU builds
    memory_->SyncToDevice();
}


// =============================================================================
// GeneralizedZener3D Implementation
// =============================================================================

void GeneralizedZener3D::EnableAttenuation(
    int ne, int ngllx, int nglly, int ngllz,
    const Vector& Qkappa, const Vector& Qmu,
    Vector& kappa, Vector& mu,
    real_t f0, int n_sls, real_t dt)
{
    dt_ = dt;

    // Initialize memory state
    memory_ = std::make_unique<ViscoelasticMemory3D>(ne, ngllx, nglly, ngllz, n_sls);

    // Compute frequency band
    real_t fQmin = 0.1 * f0;
    real_t fQmax = 10.0 * f0;

    // Use HostRead() for CPU-side attenuation coefficient computation
    const real_t* qkappa = Qkappa.HostRead();
    const real_t* qmu = Qmu.HostRead();
    real_t* kappa_data = kappa.HostWrite();
    real_t* mu_data = mu.HostWrite();

    // Packed coefficient array: [6, ngllx, nglly, ngllz, ne, n_units]
    // Components: 0=A_kappa, 1=A_mu, 2=B_kappa, 3=B_mu, 4=gamma_kappa, 5=gamma_mu
    real_t* coeffs_packed_data = memory_->coeffs_packed.HostWrite();

    for (int ie = 0; ie < ne; ie++)
    {
        for (int iz = 0; iz < ngllz; iz++)
        {
            for (int iy = 0; iy < nglly; iy++)
            {
                for (int ix = 0; ix < ngllx; ix++)
                {
                    int gll_idx = ((ie * ngllz + iz) * nglly + iy) * ngllx + ix;

                    real_t Qk = qkappa[gll_idx];
                    real_t Qm = qmu[gll_idx];

                    // Compute attenuation coefficients using cached version for performance
                    // Note: unrelaxed correction is now applied in material class
                    // via ApplyAttenuationCorrection() before operator setup
                    AttenuationParams params_kappa = ComputeAttenuationCoeffsCached(n_sls, Qk, fQmin, fQmax);
                    AttenuationParams params_mu = ComputeAttenuationCoeffsCached(n_sls, Qm, fQmin, fQmax);

                    // Compute normalization sums (sum of tau_eps/tau_sig)
                    real_t sum_beta_kappa = 0.0, sum_beta_mu = 0.0;
                    for (int i = 0; i < n_sls; i++)
                    {
                        sum_beta_kappa += params_kappa.tau_epsilon[i] / params_kappa.tau_sigma[i];
                        sum_beta_mu += params_mu.tau_epsilon[i] / params_mu.tau_sigma[i];
                    }

                    // Store parameters for each mechanism
                    for (int i = 0; i < n_sls; i++)
                    {
                        // Zener body: beta = tau_eps/tau_sig - 1
                        real_t beta_k = params_kappa.tau_epsilon[i] / params_kappa.tau_sigma[i] - 1.0;
                        real_t beta_m = params_mu.tau_epsilon[i] / params_mu.tau_sigma[i] - 1.0;

                        real_t tauinv_pos_k = 1.0 / params_kappa.tau_sigma[i];
                        real_t tauinv_pos_m = 1.0 / params_mu.tau_sigma[i];
                        real_t tauinv_neg_k = -tauinv_pos_k;
                        real_t tauinv_neg_m = -tauinv_pos_m;

                        // norm_factor: NOTE shear has factor of 2
                        real_t norm_factor_k = (beta_k * tauinv_pos_k) / sum_beta_kappa;
                        real_t norm_factor_m = (2.0 * beta_m * tauinv_pos_m) / sum_beta_mu;

                        // Crank-Nicolson coefficients (unconditionally stable, 2nd order)
                        real_t half_eta_k = 0.5 * dt * tauinv_pos_k;
                        real_t half_eta_m = 0.5 * dt * tauinv_pos_m;

                        real_t alpha_k = (1.0 - half_eta_k) / (1.0 + half_eta_k);
                        real_t alpha_m = (1.0 - half_eta_m) / (1.0 + half_eta_m);

                        real_t strain_coeff_k = (dt / 2.0) / (1.0 + half_eta_k);
                        real_t strain_coeff_m = (dt / 2.0) / (1.0 + half_eta_m);

                        // Store to packed coefficient array: [6, ngllx, nglly, ngllz, ne, n_units]
                        int packed_idx = 4 * (ix + ngllx * (iy + nglly * (iz + ngllz * (ie + ne * i))));
                        coeffs_packed_data[packed_idx + 0] = alpha_k;                          // alpha_kappa
                        coeffs_packed_data[packed_idx + 1] = alpha_m;                          // alpha_mu
                        coeffs_packed_data[packed_idx + 2] = strain_coeff_k * norm_factor_k;   // strain_coeff_kappa
                        coeffs_packed_data[packed_idx + 3] = strain_coeff_m * norm_factor_m;   // strain_coeff_mu
                    }
                }
            }
        }
    }

    // Force host→device sync for GPU builds
    memory_->SyncToDevice();

}


// =============================================================================
// GeneralizedZenerAcoustic2D Implementation (Crank-Nicolson scheme, matches 2D/3D viscoelastic)
// =============================================================================

void GeneralizedZenerAcoustic2D::EnableAttenuation(
    int ne, int ngllx, int nglly,
    const Vector& Qkappa,
    Vector& kappa,
    real_t f0, int n_sls, real_t dt)
{
    dt_ = dt;

    // Initialize memory state
    memory_ = std::make_unique<ViscoacousticMemory2D>(ne, ngllx, nglly, n_sls);

    // Compute frequency band (same as elastic)
    real_t fQmin = 0.1 * f0;
    real_t fQmax = 10.0 * f0;

    // Use HostRead() for CPU-side attenuation coefficient computation
    const real_t* qkappa = Qkappa.HostRead();
    real_t* kappa_data = kappa.HostWrite();

    // Crank-Nicolson coefficient array: [2, ngllx, nglly, ne, n_sls]
    // Components: 0=alpha, 1=strain_coeff
    auto coeffs = Reshape(memory_->coeffs_packed.HostWrite(), 2, ngllx, nglly, ne, n_sls);

    for (int ie = 0; ie < ne; ie++)
    {
        for (int iy = 0; iy < nglly; iy++)
        {
            for (int ix = 0; ix < ngllx; ix++)
            {
                int gll_idx = (ie * nglly + iy) * ngllx + ix;
                real_t Qk = qkappa[gll_idx];

                // Compute attenuation coefficients using cached version for performance
                // Note: unrelaxed correction is now applied in material class
                // via ApplyAttenuationCorrection() before operator setup
                AttenuationParams params_kappa = ComputeAttenuationCoeffsCached(n_sls, Qk, fQmin, fQmax);

                // Compute normalization sum (sum of tau_eps/tau_sig) - same as viscoelastic
                real_t sum_beta_kappa = 0.0;
                for (int i = 0; i < n_sls; i++)
                {
                    sum_beta_kappa += params_kappa.tau_epsilon[i] / params_kappa.tau_sigma[i];
                }

                // Store Crank-Nicolson coefficients for each mechanism
                // Update: e1_new = alpha × e1_old + beta × strain_old + gamma × strain_new
                for (int i = 0; i < n_sls; i++)
                {
                    // beta = tau_eps/tau_sig - 1 (same as viscoelastic)
                    real_t beta_k = params_kappa.tau_epsilon[i] / params_kappa.tau_sigma[i] - 1.0;

                    real_t tauinv_pos_k = 1.0 / params_kappa.tau_sigma[i];
                    real_t tauinv_neg_k = -tauinv_pos_k;

                    // norm_factor (same as viscoelastic kappa)
                    real_t norm_factor_k = (beta_k * tauinv_pos_k) / sum_beta_kappa;

                    // Crank-Nicolson coefficients (unconditionally stable, 2nd order)
                    real_t half_eta_k = 0.5 * dt * tauinv_pos_k;

                    real_t alpha_k = (1.0 - half_eta_k) / (1.0 + half_eta_k);
                    real_t strain_coeff_k = (dt / 2.0) / (1.0 + half_eta_k);

                    // Store to packed coefficient array: [2, ngllx, nglly, ne, n_sls]
                    coeffs(0, ix, iy, ie, i) = alpha_k;                          // alpha
                    coeffs(1, ix, iy, ie, i) = strain_coeff_k * norm_factor_k;   // strain_coeff
                }
            }
        }
    }

    // Force host→device sync for GPU builds
    memory_->SyncToDevice();
}





// =============================================================================
// GeneralizedZenerAcoustic3D Implementation (Crank-Nicolson scheme, matches 2D/3D viscoelastic)
// =============================================================================

void GeneralizedZenerAcoustic3D::EnableAttenuation(
    int ne, int ngllx, int nglly, int ngllz,
    const Vector& Qkappa,
    Vector& kappa,
    real_t f0, int n_sls, real_t dt)
{
    dt_ = dt;

    // Initialize memory state
    memory_ = std::make_unique<ViscoacousticMemory3D>(ne, ngllx, nglly, ngllz, n_sls);

    // Compute frequency band (same as elastic)
    real_t fQmin = 0.1 * f0;
    real_t fQmax = 10.0 * f0;

    // Use HostRead() for CPU-side attenuation coefficient computation
    const real_t* qkappa = Qkappa.HostRead();
    real_t* kappa_data = kappa.HostWrite();

    // Crank-Nicolson coefficient array: [2, ngllx, nglly, ngllz, ne, n_sls]
    // Components: 0=alpha, 1=strain_coeff
    auto coeffs = Reshape(memory_->coeffs_packed.HostWrite(), 2, ngllx, nglly, ngllz, ne, n_sls);

    for (int ie = 0; ie < ne; ie++)
    {
        for (int iz = 0; iz < ngllz; iz++)
        {
            for (int iy = 0; iy < nglly; iy++)
            {
                for (int ix = 0; ix < ngllx; ix++)
                {
                    int gll_idx = ((ie * ngllz + iz) * nglly + iy) * ngllx + ix;
                    real_t Qk = qkappa[gll_idx];

                    // Compute attenuation coefficients using cached version for performance
                    // Note: unrelaxed correction is now applied in material class
                    // via ApplyAttenuationCorrection() before operator setup
                    AttenuationParams params_kappa = ComputeAttenuationCoeffsCached(n_sls, Qk, fQmin, fQmax);

                    // Compute normalization sum (sum of tau_eps/tau_sig) - same as viscoelastic
                    real_t sum_beta_kappa = 0.0;
                    for (int i = 0; i < n_sls; i++)
                    {
                        sum_beta_kappa += params_kappa.tau_epsilon[i] / params_kappa.tau_sigma[i];
                    }

                    // Store Crank-Nicolson coefficients for each mechanism
                    // Update: e1_new = alpha × e1_old + beta × strain_old + gamma × strain_new
                    for (int i = 0; i < n_sls; i++)
                    {
                        // beta = tau_eps/tau_sig - 1 (same as viscoelastic)
                        real_t beta_k = params_kappa.tau_epsilon[i] / params_kappa.tau_sigma[i] - 1.0;

                        real_t tauinv_pos_k = 1.0 / params_kappa.tau_sigma[i];
                        real_t tauinv_neg_k = -tauinv_pos_k;

                        // norm_factor (same as viscoelastic kappa)
                        real_t norm_factor_k = (beta_k * tauinv_pos_k) / sum_beta_kappa;

                        // Crank-Nicolson coefficients (unconditionally stable, 2nd order)
                        real_t half_eta_k = 0.5 * dt * tauinv_pos_k;

                        real_t alpha_k = (1.0 - half_eta_k) / (1.0 + half_eta_k);
                        real_t strain_coeff_k = (dt / 2.0) / (1.0 + half_eta_k);

                        // Store to packed coefficient array: [2, ngllx, nglly, ngllz, ne, n_sls]
                        coeffs(0, ix, iy, iz, ie, i) = alpha_k;                          // alpha
                        coeffs(1, ix, iy, iz, ie, i) = strain_coeff_k * norm_factor_k;   // strain_coeff
                    }
                }
            }
        }
    }

    // Force host→device sync for GPU builds
    memory_->SyncToDevice();
}

}  // namespace SEM
