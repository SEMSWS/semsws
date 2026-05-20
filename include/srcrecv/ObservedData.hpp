// ObservedData.hpp — observed-trace storage for FWI. One ObservedReceiver per
// (physical receiver × channel); its data matrix holds the ncomp vector
// components as columns, mirroring synthetic ReceiverData for natural 1:1
// pairing by (name, type). See HDF5ObservedReader for the on-disk format.
//
// MPI phases (per-rank hyperslab read, no bulk broadcast):
//   1. Load()              — rank 0 reads HDF5 catalog, broadcasts metadata
//   2. ReceiverArray::Setup() — GSLIB locator determines is_local
//   3. FetchOwnedData()    — each rank reads its owned slabs from HDF5 into
//                            (nt, ncomp) matrices (ncomp parallel hyperslabs
//                            per physical receiver)
//   4. AlignToSimulation() — per-rank in-place auto-resample on grid mismatch

#ifndef SEM_OBSERVED_DATA_HPP
#define SEM_OBSERVED_DATA_HPP

#include <mfem.hpp>
#include <mpi.h>
#include <string>
#include <vector>
#include "common/Types.hpp"
#include "config/YamlConfig.hpp"

namespace SEM {

using mfem::real_t;
using mfem::Vector;
using mfem::DenseMatrix;

class ReceiverArray;  // forward declaration

/**
 * @brief Observed receiver entry (one per physical receiver × channel).
 *
 * `ncomp` is 1 for scalar channels (PS) and `space_dim` for vector channels
 * (DISP / VEL / ACC). Gradient (GRAD) is not yet supported for observed
 * data; add new cases here if needed. All ncomp components are stored as
 * columns of `data` and `weight`: access via `data(t, c)`.
 */
struct ObservedReceiver {
    std::string name;           // "<receiver>/<channel>"  (no _cN suffix)
    Vector position;
    ReceiverType type;
    int ncomp;                  // 1 scalar, space_dim for vector channels
    int num_samples;
    real_t dt;
    real_t t0;
    DenseMatrix data;           // [nt x ncomp] — only allocated on the owning rank
    DenseMatrix weight;         // [nt x ncomp] — only allocated on the owning rank
    bool has_weight;
    bool is_local;              // set true after FetchOwnedData on owning rank

    // Back-reference to source HDF5 for hyperslab fetch.
    std::string source_receiver;  // group name inside HDF5 (e.g. "R0001")
};

/**
 * @brief Observed data container for a single source.
 */
class ObservedData {
public:
    ObservedData() = default;

    /**
     * @brief Phase 1: load HDF5 catalog, broadcast metadata only.
     *
     * Rank 0 reads `path` (shot `shot_id`) with HDF5ObservedReader::ReadCatalog
     * and expands the hierarchical result into the flat receivers_ list. The
     * catalog (names, positions, types, components, has_weight flags) is
     * broadcast to all ranks. Bulk data/weight buffers stay empty.
     */
    void Load(const std::string& path, int shot_id, int space_dim,
              MPI_Comm comm);

    /**
     * @brief Phase 3: each rank reads only its owned slabs from HDF5.
     *
     * Uses is_local flags set by ReceiverArray::Setup(). Non-owning ranks keep
     * their matrices empty. This replaces the MPI_Send/Recv distribution used
     * by the old SU path — every rank opens the HDF5 file independently.
     */
    void FetchOwnedData(const ReceiverArray& receivers, MPI_Comm comm);

    /**
     * @brief Phase 4: per-rank align to simulation grid.
     *
     * If observed dt/nt match the simulation grid (within 1e-12 on dt and
     * exact on nt), this is a no-op. Otherwise the trace is auto-resampled
     * onto (sim_nt, sim_dt) via Lanczos (a=8); rank 0 emits a single WARN
     * line. Aborts if observed t-window does not cover simulation t-window.
     */
    void AlignToSimulation(int sim_nt, real_t sim_dt);

    int NumReceivers() const { return static_cast<int>(receivers_.size()); }
    int NumSamples() const { return num_samples_; }
    real_t Dt() const { return dt_; }
    real_t T0() const { return t0_; }

    const std::vector<ObservedReceiver>& GetReceivers() const { return receivers_; }

    /**
     * @brief Validate compatibility with simulation parameters
     * @return Empty string if OK, error message otherwise
     */
    std::string ValidateCompatibility(int sim_nt, real_t sim_dt) const;

private:
    std::vector<ObservedReceiver> receivers_;
    int num_samples_ = 0;
    real_t dt_ = 0.0;
    real_t t0_ = 0.0;
    int space_dim_ = 0;
    int shot_id_ = 0;           // shot index (all ranks)
    std::string hdf5_path_;     // source HDF5 file path (all ranks)
    MPI_Comm comm_ = MPI_COMM_NULL;

    void LoadCatalogOnRank0(const std::string& path, int space_dim);
    void BroadcastCatalog(MPI_Comm comm);
};

}  // namespace SEM

#endif  // SEM_OBSERVED_DATA_HPP
