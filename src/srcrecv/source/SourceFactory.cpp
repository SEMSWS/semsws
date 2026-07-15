/**
 * @file SourceFactory.cpp
 * @brief PointSourceCollection::FromConfig factory methods
 *
 * Each factory:
 *   1. Add*() to register all sources (position only, no STF)
 *   2. Setup() for batch PointFinder
 *   3. SetSTF() for local sources only (zero allocation on non-local ranks)
 */

#include "srcrecv/Source.hpp"

namespace SEM {

// =============================================================================
// PointSourceCollection Factory Methods
// =============================================================================

std::unique_ptr<PointSourceCollection> PointSourceCollection::FromConfig(
    const SourceConfig::Config2D& config,
    ParFiniteElementSpace* fes,
    int nt, real_t dt, MPI_Comm comm) {

    auto collection = std::make_unique<PointSourceCollection>(fes, comm);
    constexpr int dim = 2;

    // Phase 1: Register all sources (position only)
    for (size_t i = 0; i < config.forces.size(); i++) {
        const auto& src = config.forces[i];
        Vector pos(dim);
        pos[0] = src.location[0];
        pos[1] = src.location[1];
        collection->AddSingleForce(src.id, pos);
    }

    for (size_t i = 0; i < config.pressures.size(); i++) {
        const auto& src = config.pressures[i];
        Vector pos(dim);
        pos[0] = src.location[0];
        pos[1] = src.location[1];
        collection->AddSingleForce(src.id, pos);
    }

    for (size_t i = 0; i < config.moment_tensors.size(); i++) {
        const auto& src = config.moment_tensors[i];
        Vector pos(dim);
        pos[0] = src.location[0];
        pos[1] = src.location[1];

        DenseMatrix moment(dim, dim);
        moment(0, 0) = src.Mxx;
        moment(1, 1) = src.Myy;
        moment(0, 1) = src.Mxy;
        moment(1, 0) = src.Mxy;  // Symmetric
        collection->AddMomentTensor(src.id, pos, moment);
    }

    // Phase 2: Batch PointFinder
    collection->Setup();

    // Phase 3: Set STF for local sources only
    int src_idx = 0;

    for (size_t i = 0; i < config.forces.size(); i++) {
        const auto& src = config.forces[i];
        if (collection->GetSource(src_idx)->IsLocal()) {
            SourceTimeFunction base_stf = SourceTimeFunction::FromConfig(
                src.wavelet, nt, dt);

            DenseMatrix stf_data(nt, dim);
            for (int t = 0; t < nt; t++) {
                real_t val = base_stf.GetValue(t, 0);
                for (int d = 0; d < dim; d++) {
                    stf_data(t, d) = val * src.direction[d];
                }
            }
            collection->GetSource(src_idx)->SetSTF(SourceTimeFunction(stf_data, dt));
        }
        src_idx++;
    }

    for (size_t i = 0; i < config.pressures.size(); i++) {
        const auto& src = config.pressures[i];
        if (collection->GetSource(src_idx)->IsLocal()) {
            SourceTimeFunction base_stf = SourceTimeFunction::FromConfig(
                src.wavelet, nt, dt);

            DenseMatrix stf_data(nt, dim);
            for (int t = 0; t < nt; t++) {
                real_t val = base_stf.GetValue(t, 0);
                stf_data(t, 0) = 0.0;  // x-component
                stf_data(t, 1) = val;  // y-component (vertical)
            }
            collection->GetSource(src_idx)->SetSTF(SourceTimeFunction(stf_data, dt));
        }
        src_idx++;
    }

    for (size_t i = 0; i < config.moment_tensors.size(); i++) {
        const auto& src = config.moment_tensors[i];
        if (collection->GetSource(src_idx)->IsLocal()) {
            SourceTimeFunction stf = SourceTimeFunction::FromConfig(
                src.wavelet, nt, dt);
            collection->GetSource(src_idx)->SetSTF(std::move(stf));
        }
        src_idx++;
    }

    return collection;
}

std::unique_ptr<PointSourceCollection> PointSourceCollection::FromConfig(
    const SourceConfig::Config3D& config,
    ParFiniteElementSpace* fes,
    int nt, real_t dt, MPI_Comm comm) {

    auto collection = std::make_unique<PointSourceCollection>(fes, comm);
    constexpr int dim = 3;

    // Phase 1: Register all sources (position only)
    for (size_t i = 0; i < config.forces.size(); i++) {
        const auto& src = config.forces[i];
        Vector pos(dim);
        pos[0] = src.location[0];
        pos[1] = src.location[1];
        pos[2] = src.location[2];
        collection->AddSingleForce(src.id, pos);
    }

    for (size_t i = 0; i < config.pressures.size(); i++) {
        const auto& src = config.pressures[i];
        Vector pos(dim);
        pos[0] = src.location[0];
        pos[1] = src.location[1];
        pos[2] = src.location[2];
        collection->AddSingleForce(src.id, pos);
    }

    for (size_t i = 0; i < config.moment_tensors.size(); i++) {
        const auto& src = config.moment_tensors[i];
        Vector pos(dim);
        pos[0] = src.location[0];
        pos[1] = src.location[1];
        pos[2] = src.location[2];

        DenseMatrix moment(dim, dim);
        moment(0, 0) = src.Mxx;
        moment(1, 1) = src.Myy;
        moment(2, 2) = src.Mzz;
        moment(0, 1) = src.Mxy;
        moment(1, 0) = src.Mxy;
        moment(0, 2) = src.Mxz;
        moment(2, 0) = src.Mxz;
        moment(1, 2) = src.Myz;
        moment(2, 1) = src.Myz;
        collection->AddMomentTensor(src.id, pos, moment);
    }

    // Phase 2: Batch PointFinder
    collection->Setup();

    // Phase 3: Set STF for local sources only
    int src_idx = 0;

    for (size_t i = 0; i < config.forces.size(); i++) {
        const auto& src = config.forces[i];
        if (collection->GetSource(src_idx)->IsLocal()) {
            SourceTimeFunction base_stf = SourceTimeFunction::FromConfig(
                src.wavelet, nt, dt);

            DenseMatrix stf_data(nt, dim);
            for (int t = 0; t < nt; t++) {
                real_t val = base_stf.GetValue(t, 0);
                for (int d = 0; d < dim; d++) {
                    stf_data(t, d) = val * src.direction[d];
                }
            }
            collection->GetSource(src_idx)->SetSTF(SourceTimeFunction(stf_data, dt));
        }
        src_idx++;
    }

    for (size_t i = 0; i < config.pressures.size(); i++) {
        const auto& src = config.pressures[i];
        if (collection->GetSource(src_idx)->IsLocal()) {
            SourceTimeFunction base_stf = SourceTimeFunction::FromConfig(
                src.wavelet, nt, dt);

            DenseMatrix stf_data(nt, dim);
            for (int t = 0; t < nt; t++) {
                real_t val = base_stf.GetValue(t, 0);
                stf_data(t, 0) = 0.0;
                stf_data(t, 1) = 0.0;
                stf_data(t, 2) = val;  // z-component (vertical)
            }
            collection->GetSource(src_idx)->SetSTF(SourceTimeFunction(stf_data, dt));
        }
        src_idx++;
    }

    for (size_t i = 0; i < config.moment_tensors.size(); i++) {
        const auto& src = config.moment_tensors[i];
        if (collection->GetSource(src_idx)->IsLocal()) {
            SourceTimeFunction stf = SourceTimeFunction::FromConfig(
                src.wavelet, nt, dt);
            collection->GetSource(src_idx)->SetSTF(std::move(stf));
        }
        src_idx++;
    }

    return collection;
}

std::unique_ptr<PointSourceCollection> PointSourceCollection::FromConfigAcoustic(
    const SourceConfig::Config2D& config,
    ParFiniteElementSpace* fes,
    const MaterialField& kappa,
    int nt, real_t dt, MPI_Comm comm, bool phi_integrate) {

    auto collection = std::make_unique<PointSourceCollection>(fes, comm);

    // Phase 1: Register all sources (position only)
    auto add_source = [&](int id, const std::vector<real_t>& loc) {
        Vector pos(2);
        pos[0] = loc[0];
        pos[1] = loc[1];
        collection->AddSingleForceAcoustic(id, pos, kappa);
    };

    for (const auto& src : config.pressures) {
        add_source(src.id, src.location);
    }
    for (const auto& src : config.forces) {
        add_source(src.id, src.location);
    }
    for (const auto& src : config.moment_tensors) {
        add_source(src.id, src.location);
    }

    // Phase 2: Batch PointFinder
    collection->Setup();

    // Phase 3: Set STF for local sources only
    int src_idx = 0;

    auto set_stf_if_local = [&](const SourceConfig::WaveletConfig& wavelet,
                                bool apply_phi) {
        if (collection->GetSource(src_idx)->IsLocal()) {
            SourceTimeFunction stf = SourceTimeFunction::FromConfig(wavelet, nt, dt);
            if (apply_phi) stf.ApplyPhiIntegration();
            collection->GetSource(src_idx)->SetSTF(std::move(stf));
        }
        src_idx++;
    };

    // phi-integration applies only to the acoustic pressure sources.
    for (const auto& src : config.pressures) {
        set_stf_if_local(src.wavelet, phi_integrate);
    }
    for (const auto& src : config.forces) {
        set_stf_if_local(src.wavelet, false);
    }
    for (const auto& src : config.moment_tensors) {
        set_stf_if_local(src.wavelet, false);
    }

    return collection;
}

std::unique_ptr<PointSourceCollection> PointSourceCollection::FromConfigAcoustic(
    const SourceConfig::Config3D& config,
    ParFiniteElementSpace* fes,
    const MaterialField3D& kappa,
    int nt, real_t dt, MPI_Comm comm, bool phi_integrate) {

    auto collection = std::make_unique<PointSourceCollection>(fes, comm);

    // Phase 1: Register all sources (position only)
    auto add_source = [&](int id, const std::vector<real_t>& loc) {
        Vector pos(3);
        pos[0] = loc[0];
        pos[1] = loc[1];
        pos[2] = loc[2];
        collection->AddSingleForceAcoustic3D(id, pos, kappa);
    };

    for (const auto& src : config.pressures) {
        add_source(src.id, src.location);
    }
    for (const auto& src : config.forces) {
        add_source(src.id, src.location);
    }
    for (const auto& src : config.moment_tensors) {
        add_source(src.id, src.location);
    }

    // Phase 2: Batch PointFinder
    collection->Setup();

    // Phase 3: Set STF for local sources only
    int src_idx = 0;

    auto set_stf_if_local = [&](const SourceConfig::WaveletConfig& wavelet,
                                bool apply_phi) {
        if (collection->GetSource(src_idx)->IsLocal()) {
            SourceTimeFunction stf = SourceTimeFunction::FromConfig(wavelet, nt, dt);
            if (apply_phi) stf.ApplyPhiIntegration();
            collection->GetSource(src_idx)->SetSTF(std::move(stf));
        }
        src_idx++;
    };

    // phi-integration applies only to the acoustic pressure sources.
    for (const auto& src : config.pressures) {
        set_stf_if_local(src.wavelet, phi_integrate);
    }
    for (const auto& src : config.forces) {
        set_stf_if_local(src.wavelet, false);
    }
    for (const auto& src : config.moment_tensors) {
        set_stf_if_local(src.wavelet, false);
    }

    return collection;
}

}  // namespace SEM
