/**
 * @file ObservedData.cpp
 * @brief Observed-data loader (HDF5 canonical format).
 *
 * Each ObservedReceiver holds one physical receiver's data for one channel
 * type, with ALL ncomp vector components packed as columns of `data`. This
 * matches the synthetic ReceiverData layout so AdjointSource can pair
 * observed and synthetic 1:1 by (name, type).
 *
 * Phases:
 *   1. Load()              — rank 0 reads HDF5 catalog, broadcasts metadata
 *   2. ReceiverArray::Setup() [external] marks is_local
 *   3. FetchOwnedData()    — per-rank hyperslab reads (one per component)
 *   4. AlignToSimulation() — per-rank auto-resample on grid mismatch
 */

#include "srcrecv/ObservedData.hpp"
#include "srcrecv/HDF5ObservedReader.hpp"
#include "srcrecv/ObservedResampler.hpp"
#include "srcrecv/ObservedTypes.hpp"
#include "srcrecv/Receiver.hpp"

#include <algorithm>
#include <sstream>
#include <vector>

namespace SEM {

// =============================================================================
// Helpers
// =============================================================================

namespace {

// Expand a hierarchical catalog into ObservedReceiver entries (one per
// physical receiver × channel). For vector channels the entry has
// ncomp = space_dim and all components live in `data`'s columns.
void ExpandCatalog(const HDF5ObservedCatalog& cat,
                   std::vector<ObservedReceiver>& out) {
    out.clear();
    for (const auto& rentry : cat.receivers) {
        for (const auto& ch : rentry.channels) {
            const bool vec = IsVectorObservedType(ch.type);
            const int ncomp = vec ? cat.space_dim : 1;

            ObservedReceiver obs;
            obs.source_receiver = rentry.name;
            obs.position = rentry.position;
            obs.type = ch.type;
            obs.ncomp = ncomp;
            obs.num_samples = cat.n_samples;
            obs.dt = static_cast<real_t>(cat.dt);
            obs.t0 = static_cast<real_t>(cat.t0);
            obs.has_weight = ch.has_weight;
            obs.is_local = false;

            // Name: "<receiver>/<channel>"  (no _cN suffix)
            std::ostringstream nm;
            nm << rentry.name << "/" << ObservedChannelName(ch.type);
            obs.name = nm.str();

            out.push_back(std::move(obs));
        }
    }
}

}  // namespace

// =============================================================================
// Phase 1: Load (rank-0 catalog + metadata broadcast)
// =============================================================================

void ObservedData::Load(const std::string& path, int shot_id, int space_dim,
                        MPI_Comm comm) {
    comm_ = comm;
    space_dim_ = space_dim;
    shot_id_ = shot_id;
    hdf5_path_ = path;

    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        LoadCatalogOnRank0(path, space_dim);
    }
    BroadcastCatalog(comm);
}

void ObservedData::LoadCatalogOnRank0(const std::string& path, int space_dim) {
    receivers_.clear();
    HDF5ObservedCatalog cat =
        HDF5ObservedReader::ReadCatalog(path, space_dim, shot_id_);
    num_samples_ = cat.n_samples;
    dt_ = static_cast<real_t>(cat.dt);
    t0_ = static_cast<real_t>(cat.t0);
    ExpandCatalog(cat, receivers_);
}

void ObservedData::BroadcastCatalog(MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    int nrecv = static_cast<int>(receivers_.size());
    MPI_Bcast(&nrecv, 1, MPI_INT, 0, comm);
    MPI_Bcast(&num_samples_, 1, MPI_INT, 0, comm);
    MPI_Bcast(&space_dim_, 1, MPI_INT, 0, comm);
    MPI_Bcast(&dt_, 1, MFEM_MPI_REAL_T, 0, comm);
    MPI_Bcast(&t0_, 1, MFEM_MPI_REAL_T, 0, comm);

    if (rank != 0) receivers_.resize(nrecv);

    for (int r = 0; r < nrecv; ++r) {
        ObservedReceiver& obs = receivers_[r];

        // name
        int nlen = static_cast<int>(obs.name.size());
        MPI_Bcast(&nlen, 1, MPI_INT, 0, comm);
        if (rank != 0) obs.name.resize(nlen);
        if (nlen > 0) MPI_Bcast(&obs.name[0], nlen, MPI_CHAR, 0, comm);

        // source_receiver (HDF5 group name)
        int slen = static_cast<int>(obs.source_receiver.size());
        MPI_Bcast(&slen, 1, MPI_INT, 0, comm);
        if (rank != 0) obs.source_receiver.resize(slen);
        if (slen > 0) MPI_Bcast(&obs.source_receiver[0], slen, MPI_CHAR, 0, comm);

        // position
        if (rank != 0) obs.position.SetSize(space_dim_);
        MPI_Bcast(obs.position.GetData(), space_dim_, MFEM_MPI_REAL_T, 0, comm);

        // Scalar fields
        int type_int = static_cast<int>(obs.type);
        MPI_Bcast(&type_int, 1, MPI_INT, 0, comm);
        if (rank != 0) obs.type = static_cast<ReceiverType>(type_int);
        MPI_Bcast(&obs.ncomp, 1, MPI_INT, 0, comm);

        int hw = obs.has_weight ? 1 : 0;
        MPI_Bcast(&hw, 1, MPI_INT, 0, comm);
        if (rank != 0) obs.has_weight = (hw != 0);

        if (rank != 0) {
            obs.num_samples = num_samples_;
            obs.dt = dt_;
            obs.t0 = t0_;
            obs.is_local = false;
            // data/weight stay empty until FetchOwnedData
        }
    }
}

// =============================================================================
// Phase 3: Per-rank hyperslab fetch.
//
// For each local obs, queue `ncomp` HDF5 channel requests (one per
// component). After reading, pack each component into the matching column
// of obs.data / obs.weight. The HDF5 reader handles scalar (component=-1)
// and vector (component≥0) channels uniformly.
// =============================================================================

void ObservedData::FetchOwnedData(const ReceiverArray& receivers,
                                   MPI_Comm comm) {
    (void)comm;
    const auto& all_recv = receivers.GetAllReceivers();
    const int nobs = static_cast<int>(receivers_.size());

    // Flat locality assignment: receivers_ order follows CreateForwardReceivers,
    // which calls AddReceiver once per obs in obs_list order. The ReceiverArray
    // stores them grouped by type (std::map<type, vector<ReceiverData>>); we
    // walk receivers_ in the same order and the ReceiverArray walk must mirror
    // it (enum-sorted types, insertion-order within each type, matching
    // ExpandCatalog ordering when the probe order is also type-sorted).
    //
    // To be robust we look up each obs by (name, type) rather than relying on
    // positional pairing.
    std::vector<HDF5ObservedReader::OwnedChannelRequest> reqs;
    struct ReqDest { int obs_idx; int comp; };
    std::vector<ReqDest> req_dest;
    reqs.reserve(nobs);
    req_dest.reserve(nobs);

    // Build a map (name, type) -> ReceiverData for locality lookup.
    std::map<std::pair<std::string, ReceiverType>, const ReceiverData*> syn_map;
    for (const auto& entry : all_recv) {
        for (const auto& rd : entry.second) {
            syn_map[{rd.Name(), entry.first}] = &rd;
        }
    }

    for (int i = 0; i < nobs; ++i) {
        ObservedReceiver& obs = receivers_[i];
        auto it = syn_map.find({obs.source_receiver, obs.type});
        bool local = (it != syn_map.end()) && it->second->IsLocal();
        obs.is_local = local;

        if (!local) {
            obs.data.SetSize(0, 0);
            obs.weight.SetSize(0, 0);
            continue;
        }

        // Queue one hyperslab request per component.
        // Scalar channel (ncomp==1): use component=-1 (legacy convention).
        // Vector channel (ncomp>1):  use component=c for c in 0..ncomp-1.
        for (int c = 0; c < obs.ncomp; ++c) {
            HDF5ObservedReader::OwnedChannelRequest req;
            req.receiver_name = obs.source_receiver;
            req.type = obs.type;
            req.component = (obs.ncomp == 1) ? -1 : c;
            req.want_weight = obs.has_weight;
            reqs.push_back(std::move(req));
            req_dest.push_back({i, c});
        }
    }

    if (reqs.empty()) return;

    auto results = HDF5ObservedReader::ReadOwnedChannels(hdf5_path_, reqs, shot_id_);
    MFEM_VERIFY(results.size() == reqs.size(),
                "HDF5ObservedReader: result/request size mismatch");

    // Allocate (nt, ncomp) matrices on first write to each obs.
    for (size_t k = 0; k < results.size(); ++k) {
        const auto& res = results[k];
        const ReqDest& dst = req_dest[k];
        ObservedReceiver& obs = receivers_[dst.obs_idx];
        const int nt = obs.num_samples;

        if (obs.data.Height() != nt || obs.data.Width() != obs.ncomp) {
            obs.data.SetSize(nt, obs.ncomp);
        }
        if (obs.weight.Height() != nt || obs.weight.Width() != obs.ncomp) {
            obs.weight.SetSize(nt, obs.ncomp);
        }

        MFEM_VERIFY(static_cast<int>(res.data.size()) == nt,
                    "HDF5 slab length mismatch for " << obs.name
                    << " component " << dst.comp);

        const int c = dst.comp;
        for (int s = 0; s < nt; ++s) {
            obs.data(s, c) = static_cast<real_t>(res.data[s]);
        }
        if (obs.has_weight && res.has_weight) {
            for (int s = 0; s < nt; ++s) {
                obs.weight(s, c) = static_cast<real_t>(res.weight[s]);
            }
        } else {
            for (int s = 0; s < nt; ++s) obs.weight(s, c) = 1.0;
        }
    }
}

// =============================================================================
// Phase 4: Align observed grid to simulation grid (auto-resample on mismatch)
// =============================================================================

void ObservedData::AlignToSimulation(int sim_nt, real_t sim_dt) {
    // Bit-exact grid match → no-op fast path.
    constexpr real_t kDtTol = static_cast<real_t>(1e-12);
    if (num_samples_ == sim_nt && std::abs(dt_ - sim_dt) <= kDtTol) {
        return;
    }

    // Window-coverage check (resample cannot fill a missing tail).
    const real_t obs_end = t0_ + dt_ * static_cast<real_t>(num_samples_ - 1);
    const real_t sim_end = t0_ + sim_dt * static_cast<real_t>(sim_nt - 1);
    MFEM_VERIFY(obs_end + kDtTol >= sim_end,
        "Observed t-window [" << t0_ << ", " << obs_end << "]"
        " does not cover simulation [" << t0_ << ", " << sim_end << "]."
        " Regenerate the observed file with a long enough record.");

    int rank;
    MPI_Comm_rank(comm_, &rank);
    if (rank == 0) {
        mfem::out << "[WARN] Observed grid (dt=" << dt_ << ", n=" << num_samples_
                  << ") does not match simulation (dt=" << sim_dt
                  << ", n=" << sim_nt << "). Auto-resampling via Lanczos(a=8)."
                  << " Pre-process the observed file if unintentional." << std::endl;
    }

    const ObservedResampler::Method method = ObservedResampler::Method::Lanczos;
    const int lanczos_a = 8;
    const real_t src_t0 = t0_;

    for (auto& obs : receivers_) {
        if (!obs.is_local) {
            obs.num_samples = sim_nt;
            obs.dt = sim_dt;
            obs.t0 = src_t0;
            continue;
        }

        DenseMatrix new_data(sim_nt, obs.ncomp);
        DenseMatrix new_w(sim_nt, obs.ncomp);

        for (int c = 0; c < obs.ncomp; ++c) {
            std::vector<real_t> src_data(obs.num_samples);
            for (int s = 0; s < obs.num_samples; ++s) {
                src_data[s] = obs.data(s, c);
            }
            std::vector<real_t> dst_data(sim_nt);
            ObservedResampler::Resample(src_data.data(), obs.num_samples,
                                        obs.dt, obs.t0,
                                        dst_data.data(), sim_nt, sim_dt, src_t0,
                                        method, lanczos_a);
            for (int s = 0; s < sim_nt; ++s) new_data(s, c) = dst_data[s];

            if (obs.has_weight && obs.weight.Height() == obs.num_samples) {
                std::vector<real_t> src_w(obs.num_samples);
                for (int s = 0; s < obs.num_samples; ++s) {
                    src_w[s] = obs.weight(s, c);
                }
                std::vector<real_t> dst_w(sim_nt);
                // The weight is a non-negative taper/mask, not a band-limited
                // signal: resample it linearly (no overshoot) and clamp to >=0.
                // Lanczos here would ring below zero at the window edges and
                // make the L2 misfit (0.5*w*r^2*dt) negative.
                ObservedResampler::Resample(src_w.data(), obs.num_samples,
                                            obs.dt, obs.t0,
                                            dst_w.data(), sim_nt, sim_dt, src_t0,
                                            ObservedResampler::Method::Linear,
                                            lanczos_a);
                for (int s = 0; s < sim_nt; ++s)
                    new_w(s, c) = std::max(real_t(0), dst_w[s]);
            } else {
                for (int s = 0; s < sim_nt; ++s) new_w(s, c) = 1.0;
            }
        }

        obs.data = std::move(new_data);
        obs.weight = std::move(new_w);
        obs.num_samples = sim_nt;
        obs.dt = sim_dt;
        obs.t0 = src_t0;
    }
    num_samples_ = sim_nt;
    dt_ = sim_dt;

    std::string err = ValidateCompatibility(sim_nt, sim_dt);
    MFEM_VERIFY(err.empty(),
        "Post-resample validation failed: " << err);
}

// =============================================================================
// Validation
// =============================================================================

std::string ObservedData::ValidateCompatibility(int sim_nt,
                                                real_t sim_dt) const {
    std::ostringstream err;
    if (receivers_.empty()) {
        err << "No observed receivers loaded";
        return err.str();
    }
    if (num_samples_ != sim_nt) {
        err << "Observed data has " << num_samples_ << " samples"
            << " but simulation has " << sim_nt << " steps";
        return err.str();
    }
    real_t dt_diff = std::abs(dt_ - sim_dt);
    if (dt_diff > sim_dt * 0.01) {
        err << "Observed dt=" << dt_ << " differs from simulation dt="
            << sim_dt << " by more than 1%";
        return err.str();
    }
    return "";
}

}  // namespace SEM
