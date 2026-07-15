/**
 * @file SourceTimeFunction.cpp
 * @brief Source time function (wavelet) generation
 */

#include "srcrecv/Source.hpp"
#include "srcrecv/ObservedResampler.hpp"
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

namespace SEM {

// =============================================================================
// SourceTimeFunction
// =============================================================================

SourceTimeFunction::SourceTimeFunction(const DenseMatrix& stf, real_t dt)
    : stf_(stf), dt_(dt)
{
    nt_ = stf_.Height();
    ncomp_ = stf_.Width();
}

SourceTimeFunction::SourceTimeFunction(const Vector& times, const DenseMatrix& values)
    : stf_(values)
{
    nt_ = stf_.Height();
    ncomp_ = stf_.Width();
    if (nt_ >= 2) {
        dt_ = times[1] - times[0];
    }
}

SourceTimeFunction SourceTimeFunction::Ricker(real_t f0, real_t t0, real_t dt, int nt,
                                               real_t amplitude)
{
    DenseMatrix stf(nt, 1);

    // Use double precision for intermediate calculations to avoid
    // single-precision rounding errors in exp() for large arguments
    const double pi = M_PI;
    const double pi2 = pi * pi;
    const double f0_d = static_cast<double>(f0);
    const double t0_d = static_cast<double>(t0);
    const double dt_d = static_cast<double>(dt);
    const double amp_d = static_cast<double>(amplitude);
    const double f02 = f0_d * f0_d;

    // Warn if t0 is too small for clean vel/accel output
    // const double t0_enough = 2.0 / f0_d;
    // if (t0_d < t0_enough) {
    //     mfem::out << "[WARNING] Ricker wavelet: t0=" << t0_d
    //               << " may be too small for f0=" << f0_d << " Hz.\n"
    //               << "  Consider t0 >= " << t0_enough
    //               << " to reduce high-frequency noise in vel/accel.\n"
    //               << "  Or apply low-pass filter, or use double precision.\n";
    // }

    // Threshold: values below this relative magnitude are set to zero
    // This prevents numerical noise from tiny exp() results ** This does not work
    const double threshold = 1e-6;

    for (int i = 0; i < nt; i++) {
        double t = i * dt_d - t0_d;  // time relative to peak
        double arg = pi2 * f02 * t * t;
        double ricker = (1.0 - 2.0 * arg) * std::exp(-arg);

        // Zero out values below threshold to avoid numerical noise
        if (std::abs(ricker) < threshold) {
            ricker = 0.0;
        }

        stf(i, 0) = static_cast<real_t>(amp_d * ricker);
    }



    //debug output to file
    // std::ofstream ofs("ricker_stf.txt");
    // for (int i = 0; i < nt; i++) {
    //     real_t t = i * dt;
    //     ofs << t << " " << stf(i, 0) << "\n";
    // }
    // ofs.close();

    return SourceTimeFunction(stf, dt);
}

SourceTimeFunction SourceTimeFunction::Gaussian(real_t f0, real_t t0, real_t dt, int nt,
                                                 real_t amplitude)
{
    DenseMatrix stf(nt, 1);

    // Use double precision for intermediate calculations
    const double pi = M_PI;
    const double f0_d = static_cast<double>(f0);
    const double t0_d = static_cast<double>(t0);
    const double dt_d = static_cast<double>(dt);
    const double amp_d = static_cast<double>(amplitude);
    const double sigma = 1.0 / (2.0 * pi * f0_d);
    const double sigma2 = sigma * sigma;

    const double threshold = 1e-10;

    for (int i = 0; i < nt; i++) {
        double t = i * dt_d - t0_d;
        double gauss = std::exp(-0.5 * t * t / sigma2);

        if (gauss < threshold) {
            gauss = 0.0;
        }

        stf(i, 0) = static_cast<real_t>(amp_d * gauss);
    }

    return SourceTimeFunction(stf, dt);
}

void SourceTimeFunction::GetValue(int step, Vector& value) const {
    if (step < 0 || step >= nt_) {
        value.SetSize(ncomp_);
        value = 0.0;
        return;
    }

    value.SetSize(ncomp_);
    for (int c = 0; c < ncomp_; c++) {
        value[c] = stf_(step, c);
    }
}

real_t SourceTimeFunction::GetValue(int step, int component) const {
    if (step < 0 || step >= nt_ || component < 0 || component >= ncomp_) {
        return 0.0;
    }
    return stf_(step, component);
}

namespace {
// Cumulative trapezoidal integral: out[0]=0, out[k]=out[k-1]+dt/2*(x[k]+x[k-1]).
void CumTrapz(const std::vector<double>& x, double dt, std::vector<double>& out) {
    const int n = static_cast<int>(x.size());
    out.assign(static_cast<size_t>(n), 0.0);
    for (int i = 1; i < n; ++i) {
        out[i] = out[i - 1] + 0.5 * dt * (x[i] + x[i - 1]);
    }
}

// Remove the least-squares linear trend (fit over sample index).
void DetrendLinear(std::vector<double>& y) {
    const int n = static_cast<int>(y.size());
    if (n < 2) return;
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    for (int i = 0; i < n; ++i) {
        const double xi = static_cast<double>(i);
        sx += xi; sy += y[i]; sxx += xi * xi; sxy += xi * y[i];
    }
    const double denom = static_cast<double>(n) * sxx - sx * sx;
    if (denom == 0.0) return;
    const double b = (static_cast<double>(n) * sxy - sx * sy) / denom;
    const double a = (sy - b * sx) / static_cast<double>(n);
    for (int i = 0; i < n; ++i) y[i] -= (a + b * static_cast<double>(i));
}
}  // namespace

void SourceTimeFunction::ApplyPhiIntegration() {
    const double dt_d = static_cast<double>(dt_);
    for (int c = 0; c < ncomp_; ++c) {
        // Negate the input, then double-integrate: phi = \int\int (-stf) dt dt.
        std::vector<double> x(static_cast<size_t>(nt_));
        for (int t = 0; t < nt_; ++t) {
            x[static_cast<size_t>(t)] = -static_cast<double>(stf_(t, c));
        }
        std::vector<double> i1, i2;
        CumTrapz(x, dt_d, i1);  DetrendLinear(i1);
        CumTrapz(i1, dt_d, i2); DetrendLinear(i2);
        for (int t = 0; t < nt_; ++t) {
            stf_(t, c) = static_cast<real_t>(i2[static_cast<size_t>(t)]);
        }
    }
}

SourceTimeFunction SourceTimeFunction::FromExternalFile(
    const std::string& filepath,
    real_t dt_sim,
    int nt_sim)
{
    std::ifstream file(filepath);
    MFEM_VERIFY(file.is_open(), "Cannot open STF file: " << filepath);

    std::vector<real_t> times;
    std::vector<real_t> amplitudes;
    std::string line;

    while (std::getline(file, line)) {
        // Skip empty lines
        size_t first_nonspace = line.find_first_not_of(" \t");
        if (first_nonspace == std::string::npos) continue;

        // Skip comments
        if (line[first_nonspace] == '#') continue;

        // Parse two columns: time amplitude
        std::istringstream iss(line);
        real_t t, amp;
        if (!(iss >> t >> amp)) {
            MFEM_ABORT("Invalid STF file format at line: " << line
                       << "\nExpected: time amplitude");
        }

        times.push_back(t);
        amplitudes.push_back(amp);
    }
    file.close();

    int nt_file = static_cast<int>(times.size());
    MFEM_VERIFY(nt_file >= 2, "STF file must have at least 2 data points: " << filepath);

    // Check that file has enough samples for simulation
    MFEM_VERIFY(nt_file >= nt_sim,
        "STF file has " << nt_file << " samples, but simulation requires "
        << nt_sim << " steps. File: " << filepath);

    // Compute dt from file and warn if different from simulation dt
    real_t dt_file = times[1] - times[0];
    MFEM_VERIFY(dt_file > 0, "STF file time must be increasing: " << filepath);

    real_t dt_diff = std::abs(dt_file - dt_sim);
    if (dt_diff > 1e-10 * dt_sim) {
        std::cerr << "[WARNING] STF file dt=" << dt_file
                  << " differs from simulation dt=" << dt_sim
                  << " (file: " << filepath << ")\n";
    }

    // Build STF matrix (use file's dt, truncate to nt_sim)
    DenseMatrix stf(nt_sim, 1);
    for (int i = 0; i < nt_sim; i++) {
        stf(i, 0) = amplitudes[i];
    }

    return SourceTimeFunction(stf, dt_file);
}

SourceTimeFunction SourceTimeFunction::FromConfig(
    const SourceConfig::WaveletConfig& config,
    int nt, real_t dt) {

    // Pre-loaded STF samples (HDF5 input bundle) take precedence over
    // analytic / external-file synthesis. The HDF5 reader stores the STF at
    // the bundle's own sampling (config.stf_dt); the simulation injects one
    // value per step, so resample it onto the simulation grid (nt, dt).
    //
    // The STF is a physical waveform, so the resample must preserve physical
    // time: src_dt is the bundle's true dt, NOT a value back-derived from the
    // sample counts. The old dt*nt/n_src formula silently assumed the bundle
    // and the simulation span the same duration; when they differ (e.g. a 22 s
    // observation bundle vs a 17 s simulation) it rescales the time axis and
    // shifts the whole spectrum (a 3-6 Hz STF emerges near 4-8 Hz). Only fall
    // back to that formula when stf_dt is unknown (0), for backward compat.
    if (!config.stf_samples.empty()) {
        const int n_src = static_cast<int>(config.stf_samples.size());
        const real_t src_dt = (config.stf_dt > real_t{0})
            ? config.stf_dt
            : dt * static_cast<real_t>(nt) / static_cast<real_t>(n_src);
        const bool same_grid = (n_src == nt) &&
            (std::abs(static_cast<double>(src_dt) - static_cast<double>(dt))
                 <= 1e-12 * static_cast<double>(dt));
        DenseMatrix dm(nt, 1);
        if (same_grid) {
            for (int i = 0; i < nt; ++i) {
                dm(i, 0) = config.stf_samples[i];
            }
        } else {
            std::vector<real_t> dst(static_cast<size_t>(nt));
            ObservedResampler::Resample(
                config.stf_samples.data(), n_src, src_dt, /*src_t0=*/real_t{0},
                dst.data(), nt, dt, /*dst_t0=*/real_t{0},
                ObservedResampler::Method::Lanczos, /*lanczos_a=*/3);
            for (int i = 0; i < nt; ++i) {
                dm(i, 0) = dst[static_cast<size_t>(i)];
            }
        }
        return SourceTimeFunction(dm, dt);
    }

    if (config.type == "ricker") {
        return SourceTimeFunction::Ricker(
            config.frequency, config.delay, dt, nt, config.amplitude);
    } else if (config.type == "gaussian") {
        return SourceTimeFunction::Gaussian(
            config.frequency, config.delay, dt, nt, config.amplitude);
    } else if (config.type == "external") {
        MFEM_VERIFY(!config.external_file.empty(),
            "wavelet.type='external' requires 'file' parameter");
        return SourceTimeFunction::FromExternalFile(config.external_file, dt, nt);
    }
    // Default to Ricker
    return SourceTimeFunction::Ricker(
        config.frequency, config.delay, dt, nt, config.amplitude);
}

}  // namespace SEM
