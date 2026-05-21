/**
 * @file AttenuationCoeffs.cpp
 * @brief Attenuation coefficient computation for generalized Zener body
 *
 * Computes relaxation parameters (tau_epsilon, tau_sigma) for the
 * Generalized Zener Body viscoelastic model via Emmerich & Korn linear
 * least squares: log-spaced relaxation frequencies, weights from normal
 * equations solved with MFEM DenseMatrix.
 * Reference: Emmerich & Korn, Geophysics 52, 1252-1264 (1987)
 *
 * Internal math is in double precision; the public interface converts
 * to/from real_t for single-precision builds.
 */

#include "integ/attenuation/AttenuationCoeffs.hpp"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace SEM {

// Internal computations use double precision for numerical stability
static constexpr double TWO_PI = 6.28318530717958;
static constexpr double PI = 3.14159265358979;


// ============================================================================
// Emmerich & Korn least squares
// Reference: Emmerich & Korn, Geophysics 52, 1252-1264 (1987)
// ============================================================================

/**
 * @brief Emmerich & Korn linear least squares for Zener body parameters
 *
 * Fixes relaxation frequencies at log-spaced positions, then solves
 * for optimal weights via linear least squares (normal equations).
 */
/**
 * @brief Result of Emmerich & Korn fitting (internal use)
 */
struct EKResult {
    std::vector<double> tau_sigma;
    std::vector<double> tau_epsilon;
    std::vector<double> theta;   ///< Relaxation angular frequencies
    std::vector<double> kappa;   ///< Fitted weights
};

EKResult EmmerichKornLeastSquares(int N, double Qref, double f_min, double f_max)
{
    double omega_a = TWO_PI * f_min;
    double omega_b = TWO_PI * f_max;

    // Step 1: Log-spaced relaxation angular frequencies
    std::vector<double> theta(N);
    if (N == 1) {
        theta[0] = std::sqrt(omega_a * omega_b);
    } else {
        for (int l = 0; l < N; l++) {
            theta[l] = omega_a * std::pow(omega_b / omega_a,
                       static_cast<double>(l) / (N - 1));
        }
    }

    // Step 2: Build overdetermined system A * kappa = b
    // Basis function (Emmerich & Korn 1987):
    //   Q^{-1}(f) = sum_k kappa_k * f * (theta_k - f/Q) / (theta_k^2 + f^2)
    int M = std::max(2 * N - 1, 3);
    mfem::DenseMatrix A(M, N);
    mfem::Vector b(M);
    double target = 1.0 / Qref;

    for (int i = 0; i < M; i++) {
        double f = omega_a * std::pow(omega_b / omega_a,
                   static_cast<double>(i) / (M - 1));
        b(i) = target;
        for (int k = 0; k < N; k++) {
            A(i, k) = f * (theta[k] - f / Qref) / (theta[k] * theta[k] + f * f);
        }
    }

    // Step 3: Solve normal equations (A^T A) kappa = A^T b
    mfem::DenseMatrix AtA(N, N);
    mfem::Vector Atb(N);
    MultAtB(A, A, AtA);
    A.MultTranspose(b, Atb);

    mfem::DenseMatrixInverse AtA_inv(AtA);
    mfem::Vector kappa_vec(N);
    AtA_inv.Mult(Atb, kappa_vec);

    // Step 4: Pack result
    EKResult res;
    res.theta.resize(N);
    res.kappa.resize(N);
    res.tau_sigma.resize(N);
    res.tau_epsilon.resize(N);
    for (int l = 0; l < N; l++) {
        res.theta[l] = theta[l];
        res.kappa[l] = kappa_vec(l);
        res.tau_sigma[l] = 1.0 / theta[l];
        res.tau_epsilon[l] = res.tau_sigma[l] * (1.0 + N * kappa_vec(l));
    }
    return res;
}

/**
 * @brief Evaluate Q^{-1}(f) using E&K basis function
 *
 * Q^{-1}(f) = sum_k kappa[k] * f * (theta[k] - f/Q) / (theta[k]^2 + f^2)
 */
double EvaluateQinvEK(double omega, double Qref, int N,
                      const std::vector<double>& theta,
                      const std::vector<double>& kappa)
{
    double Qinv = 0.0;
    for (int k = 0; k < N; k++) {
        double num = kappa[k] * omega * (theta[k] - omega / Qref);
        double den = theta[k] * theta[k] + omega * omega;
        Qinv += num / den;
    }
    return Qinv;
}


AttenuationParams ComputeAttenuationCoeffs(
    int N,
    real_t Qref,
    real_t f_min,
    real_t f_max)
{
    if (N < 1) {
        throw std::invalid_argument("Number of mechanisms N must be >= 1");
    }
    if (Qref <= 0) {
        throw std::invalid_argument("Q factor must be positive");
    }
    if (f_min <= 0 || f_max <= 0 || f_min >= f_max) {
        throw std::invalid_argument("Invalid frequency range");
    }

    AttenuationParams result;

    // Emmerich & Korn linear least squares (Geophysics 52, 1987)
    double Q_d = static_cast<double>(Qref);
    double f_min_d = static_cast<double>(f_min);
    double f_max_d = static_cast<double>(f_max);
    EKResult ek = EmmerichKornLeastSquares(N, Q_d, f_min_d, f_max_d);

    result.tau_sigma.resize(N);
    result.tau_epsilon.resize(N);
    result.phi.resize(N);

    for (int i = 0; i < N; i++) {
        result.tau_sigma[i] = static_cast<real_t>(ek.tau_sigma[i]);
        result.tau_epsilon[i] = static_cast<real_t>(ek.tau_epsilon[i]);
        result.phi[i] = (result.tau_epsilon[i] - result.tau_sigma[i])
                      / result.tau_epsilon[i];
    }

    // Compute Q^{-1} approximation error using exact E&K basis function
    int nfreq = 100;
    double target = 1.0 / Q_d;
    double sum_error = 0.0;
    for (int i = 0; i < nfreq; i++) {
        double f = f_min_d * std::pow(f_max_d / f_min_d,
                   static_cast<double>(i) / (nfreq - 1));
        double Qinv = EvaluateQinvEK(TWO_PI * f, Q_d, N, ek.theta, ek.kappa);
        double rel = std::abs(Qinv - target) / target;
        sum_error += rel * rel;
    }
    result.error = static_cast<real_t>(std::sqrt(sum_error / nfreq));


    // Compute unrelaxed_correction (physical dispersion correction)
    double f0 = std::sqrt(static_cast<double>(f_min) * static_cast<double>(f_max));
    double omega0 = TWO_PI * f0;

    double numerator = 1.0;
    double denominator = 1.0;
    for (int i = 0; i < N; i++) {
        double ak = static_cast<double>(result.tau_epsilon[i]) /
                    static_cast<double>(result.tau_sigma[i]) - 1.0;
        double omega_tau = omega0 * static_cast<double>(result.tau_sigma[i]);
        numerator += ak / N;
        denominator += ak / (1.0 + 1.0 / (omega_tau * omega_tau)) / N;
    }
    result.unrelaxed_correction = static_cast<real_t>(numerator / denominator) ;

    return result;
}

AttenuationParams ComputeAttenuationCoeffsFromF0(
    int N,
    real_t Qref,
    real_t f0)
{
    real_t f_min = 0.1 * f0;
    real_t f_max = 10.0 * f0;
    return ComputeAttenuationCoeffs(N, Qref, f_min, f_max);
}

// ============================================================================
// Caching Implementation
// ============================================================================


struct CacheKey {
    int n_sls;
    int q_rounded;      // Q * 10 (0.1 precision)
    int f_min_rounded;  // f_min * 10
    int f_max_rounded;  // f_max * 10

    bool operator==(const CacheKey& other) const {
        return n_sls == other.n_sls &&
               q_rounded == other.q_rounded &&
               f_min_rounded == other.f_min_rounded &&
               f_max_rounded == other.f_max_rounded;
    }
};

struct CacheKeyHash {
    size_t operator()(const CacheKey& k) const {
        size_t h = std::hash<int>()(k.n_sls);
        h ^= std::hash<int>()(k.q_rounded) << 1;
        h ^= std::hash<int>()(k.f_min_rounded) << 2;
        h ^= std::hash<int>()(k.f_max_rounded) << 3;
        return h;
    }
};

static std::unordered_map<CacheKey, AttenuationParams, CacheKeyHash> g_cache;
static size_t g_cache_hits = 0;
static size_t g_cache_misses = 0;
static constexpr size_t MAX_CACHE_SIZE = 10000;

CacheKey MakeKey(int N, real_t Qref, real_t f_min, real_t f_max) {
    return CacheKey{
        N,
        static_cast<int>(std::round(Qref * 10.0)),
        static_cast<int>(std::round(f_min * 10.0)),
        static_cast<int>(std::round(f_max * 10.0))
    };
}


AttenuationParams ComputeAttenuationCoeffsCached(
    int N, real_t Qref, real_t f_min, real_t f_max)
{
    CacheKey key = MakeKey(N, Qref, f_min, f_max);

    auto it = g_cache.find(key);
    if (it != g_cache.end()) {
        g_cache_hits++;
        return it->second;
    }

    g_cache_misses++;
    AttenuationParams result = ComputeAttenuationCoeffs(N, Qref, f_min, f_max);

    if (g_cache.size() < MAX_CACHE_SIZE) {
        g_cache[key] = result;
    }

    return result;
}

void ClearAttenuationCache() {
    g_cache.clear();
    g_cache_hits = 0;
    g_cache_misses = 0;
}

std::pair<size_t, size_t> GetAttenuationCacheStats() {
    return {g_cache_hits, g_cache_misses};
}

// ============================================================================
// Q-approximation diagnostic output
// ============================================================================

void WriteQApproximationFile(
    const std::string& filepath,
    int N, real_t Qkappa, real_t Qmu,
    real_t f0)
{
    real_t f_min = 0.1 * f0;
    real_t f_max = 10.0 * f0;

    double Q_k = static_cast<double>(Qkappa);
    double Q_m = static_cast<double>(Qmu);
    double f_min_d = static_cast<double>(f_min);
    double f_max_d = static_cast<double>(f_max);

    EKResult ek_k = EmmerichKornLeastSquares(N, Q_k, f_min_d, f_max_d);
    AttenuationParams params_k = ComputeAttenuationCoeffs(N, Qkappa, f_min, f_max);

    bool has_mu = (Qmu > 0);
    EKResult ek_m;
    AttenuationParams params_m;
    if (has_mu) {
        ek_m = EmmerichKornLeastSquares(N, Q_m, f_min_d, f_max_d);
        params_m = ComputeAttenuationCoeffs(N, Qmu, f_min, f_max);
    }

    // Frequency range: f_min to f_max
    int nfreq = 200;
    double f_lo = f_min_d;
    double f_hi = f_max_d;

    std::ofstream ofs(filepath);
    if (!ofs.is_open()) return;

    ofs << "# SEMSWS Q-approximation diagnostic\n";
    ofs << "# Method: Emmerich & Korn linear least squares (1987)\n";
    ofs << "# N_SLS=" << N << "  f0=" << f0 << " Hz"
        << "  f_min=" << f_min << " Hz  f_max=" << f_max << " Hz\n";
    ofs << "# Q_kappa=" << Qkappa;
    if (has_mu) ofs << "  Q_mu=" << Qmu;
    ofs << "\n";

    ofs << "# tau_sigma[kappa]:";
    for (int i = 0; i < N; i++) ofs << " " << std::scientific << params_k.tau_sigma[i];
    ofs << "\n# tau_epsilon[kappa]:";
    for (int i = 0; i < N; i++) ofs << " " << std::scientific << params_k.tau_epsilon[i];
    ofs << "\n# RMS_error_kappa=" << std::fixed << std::setprecision(2)
        << params_k.error * 100.0 << "%\n";

    if (has_mu) {
        ofs << "# tau_sigma[mu]:";
        for (int i = 0; i < N; i++) ofs << " " << std::scientific << params_m.tau_sigma[i];
        ofs << "\n# tau_epsilon[mu]:";
        for (int i = 0; i < N; i++) ofs << " " << std::scientific << params_m.tau_epsilon[i];
        ofs << "\n# RMS_error_mu=" << std::fixed << std::setprecision(2)
            << params_m.error * 100.0 << "%\n";
    }

    ofs << "# frequency(Hz)\tQinv_kappa\tQinv_target_kappa";
    if (has_mu) ofs << "\tQinv_mu\tQinv_target_mu";
    ofs << "\n";

    double target_k = 1.0 / Q_k;
    double target_m = has_mu ? 1.0 / Q_m : 0.0;

    ofs << std::scientific << std::setprecision(6);
    for (int i = 0; i < nfreq; i++) {
        double f = f_lo * std::pow(f_hi / f_lo, static_cast<double>(i) / (nfreq - 1));
        double omega = TWO_PI * f;

        double qinv_k = EvaluateQinvEK(omega, Q_k, N, ek_k.theta, ek_k.kappa);
        ofs << f << "\t" << qinv_k << "\t" << target_k;

        if (has_mu) {
            double qinv_m = EvaluateQinvEK(omega, Q_m, N, ek_m.theta, ek_m.kappa);
            ofs << "\t" << qinv_m << "\t" << target_m;
        }
        ofs << "\n";
    }
}

} // namespace SEM
