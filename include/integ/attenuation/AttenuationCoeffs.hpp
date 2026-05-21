#ifndef SEM_ATTENUATION_COEFFS_HPP
#define SEM_ATTENUATION_COEFFS_HPP

/**
 * @file AttenuationCoeffs.hpp
 * @brief Attenuation coefficient computation for generalized Zener body
 *
 * Computes relaxation parameters (tau_epsilon, tau_sigma) for the
 * Generalized Zener Body viscoelastic model using Emmerich & Korn
 * linear least squares (MFEM DenseMatrix-based normal equations).
 *
 * Reference: Emmerich & Korn, Geophysics 52, 1252-1264 (1987)
 */

#include <mfem.hpp>
#include <string>
#include <vector>
#include <utility>

namespace SEM {

using mfem::real_t;

/**
 * @brief Result structure for attenuation coefficient computation
 */
struct AttenuationParams {
    std::vector<real_t> tau_epsilon;  ///< Relaxation times for strain
    std::vector<real_t> tau_sigma;    ///< Relaxation times for stress
    std::vector<real_t> phi;          ///< Phi coefficients for memory variables
    real_t error;                      ///< Approximation error

    /// Unrelaxed modulus correction factor for physical dispersion
    /// M_unrelaxed = M_relaxed * unrelaxed_correction
    real_t unrelaxed_correction = 1.0;
};

/**
 * @brief Compute attenuation coefficients for a Generalized Zener body model
 *
 * Fits Q(f) ≈ const Qref over [f_min, f_max] via Emmerich & Korn linear
 * least squares: log-spaced relaxation frequencies, weights from normal
 * equations.
 *
 * Internal formulas:
 *   tau_sigma   = 1 / theta
 *   tau_epsilon = tau_sigma * (1 + N * weight)
 *   phi         = (tau_epsilon - tau_sigma) / tau_epsilon
 *
 * @param N Number of relaxation mechanisms (typically 3-5)
 * @param Qref Target Q factor
 * @param f_min Minimum frequency in fitting band (Hz)
 * @param f_max Maximum frequency in fitting band (Hz)
 */
AttenuationParams ComputeAttenuationCoeffs(
    int N,
    real_t Qref,
    real_t f_min,
    real_t f_max
);

/**
 * @brief Compute attenuation coefficients using a reference frequency
 *
 * Convenience wrapper: sets f_min = 0.1*f0, f_max = 10*f0.
 */
AttenuationParams ComputeAttenuationCoeffsFromF0(
    int N,
    real_t Qref,
    real_t f0
);

// ============================================================================
// Caching for Attenuation Coefficients
// ============================================================================

/**
 * @brief Cached version of ComputeAttenuationCoeffs
 *
 * Q is rounded to 0.1 precision for the cache key.
 */
AttenuationParams ComputeAttenuationCoeffsCached(
    int N,
    real_t Qref,
    real_t f_min,
    real_t f_max
);

/// Clear the attenuation coefficient cache
void ClearAttenuationCache();

/// @return pair of (cache_hits, cache_misses)
std::pair<size_t, size_t> GetAttenuationCacheStats();

// ============================================================================
// Q-approximation diagnostic output
// ============================================================================

/**
 * @brief Write Q^{-1}(f) diagnostic file for verifying constant-Q approximation
 *
 * Evaluates the E&K Q^{-1} approximation over [0.1*f0, 10*f0] and writes
 * a tab-separated file. Call from rank 0 only.
 *
 * @param filepath Output file path
 * @param N Number of SLS mechanisms
 * @param Qkappa Q factor for bulk modulus
 * @param Qmu Q factor for shear modulus (0 for acoustic)
 * @param f0 Reference frequency (Hz)
 */
void WriteQApproximationFile(
    const std::string& filepath,
    int N, real_t Qkappa, real_t Qmu,
    real_t f0);

} // namespace SEM

#endif // SEM_ATTENUATION_COEFFS_HPP
