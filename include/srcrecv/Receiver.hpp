/**
 * @file Receiver.hpp
 * @brief Modern receiver system for spectral element method
 *
 * This file provides receiver classes using MFEM data structures:
 * - ReceiverType: Enum for measurement types
 * - ReceiverData: Single receiver with recorded data
 * - ReceiverArray: Manager for multiple receivers
 *
 * Key features:
 * - All data stored in MFEM Vectors/DenseMatrix
 * - Automatic receiver location via FindPointsGSLIB
 * - MPI-aware recording and output
 * - HDF5 and text file I/O
 * - Clean interface with RAII
 */

#ifndef SEM_RECEIVER_HPP
#define SEM_RECEIVER_HPP

#include <mfem.hpp>
#include "common/Types.hpp"
#include "srcrecv/HDF5SourceReceiverWriter.hpp"
#include <memory>
#include <vector>
#include <map>
#include <string>

namespace SEM {


using namespace mfem;

// Forward declarations
namespace ReceiverConfig {
    struct Config;
}

// ReceiverType, DomainType and their conversion functions are now in common/Types.hpp


// =============================================================================
// ReceiverData
// =============================================================================

/**
 * @class ReceiverData
 * @brief Single receiver with recorded data
 *
 * Stores receiver location, metadata, and time series data.
 */
class ReceiverData {
public:
    ReceiverData() = default;

    /**
     * @brief Construct receiver
     * @param name Receiver name/identifier
     * @param position Physical position
     * @param type Measurement type
     * @param nt Number of time steps
     * @param dt Time step
     * @param ncomp Number of components
     */
    ReceiverData(const std::string& name,
                 const Vector& position,
                 ReceiverType type,
                 int nt, real_t dt, int ncomp);

    // -------------------------------------------------------------------------
    // Data access
    // -------------------------------------------------------------------------

    /// Get reference to recorded data [nt x ncomp]
    const DenseMatrix& Data() const { return data_; }
    DenseMatrix& Data() { return data_; }

    /// Set single time sample
    void SetSample(int step, const Vector& sample);

    /// Add to single time sample
    void AddSample(int step, const Vector& sample);

    /// Reset data to zero
    void ResetData();

    /// Resize data storage (called during Setup)
    void Resize(int nt, real_t dt);

    /// Allocate data storage (for local receivers only)
    void AllocateDataStorage();

    /// Check if data storage is allocated
    bool HasDataStorage() const;

    // -------------------------------------------------------------------------
    // Metadata access
    // -------------------------------------------------------------------------

    const std::string& Name() const { return name_; }
    ReceiverType Type() const { return type_; }
    const Vector& Position() const { return position_; }
    const Vector& RefPosition() const { return ref_position_; }

    int NumTimeSteps() const { return nt_; }
    int NumComponents() const { return ncomp_; }
    real_t DeltaT() const { return dt_; }

    /// Get global receiver ID
    int Id() const { return id_; }

    /// Set global receiver ID
    void SetId(int id) { id_ = id; }

    // -------------------------------------------------------------------------
    // Location info
    // -------------------------------------------------------------------------

    bool IsLocal() const { return is_local_; }
    int OwnerRank() const { return owner_rank_; }
    int ElementIndex() const { return elem_; }

    /// Set location info (called by ReceiverArray during setup)
    void SetLocation(int elem, int owner_rank, const Vector& ref_pos, bool is_local);

    // -------------------------------------------------------------------------
    // Cached interpolation data (for efficiency)
    // Shape functions and DOF indices are computed once during setup,
    // not recomputed every time step.
    // -------------------------------------------------------------------------

    /// Get cached shape function values at receiver position
    const Vector& CachedShape() const { return cached_shape_; }

    /// Get cached element DOF indices
    const Array<int>& CachedVDofs() const { return cached_vdofs_; }

    /// Cache interpolation data (shape functions and DOFs)
    void CacheInterpolation(const Vector& shape, const Array<int>& vdofs);

    // -------------------------------------------------------------------------
    // Weight for inversion
    // -------------------------------------------------------------------------

    const DenseMatrix& Weight() const { return weight_; }
    DenseMatrix& Weight() { return weight_; }
    void SetWeight(real_t w);
    void SetWeight(const DenseMatrix& w);

private:
    std::string name_;
    Vector position_;
    Vector ref_position_;
    ReceiverType type_;
    DenseMatrix data_;    // [nt x ncomp]
    DenseMatrix weight_;  // [nt x ncomp] for inversion

    int nt_ = 0;
    int ncomp_ = 0;
    real_t dt_ = 0.0;
    int elem_ = -1;
    int owner_rank_ = -1;
    bool is_local_ = false;
    int id_ = -1;  // Global receiver ID
    real_t initial_weight_ = 1.0;  // Scalar weight (applied when data is allocated)

    // Cached interpolation data (computed once, used every time step)
    Vector cached_shape_;      // Shape function values at ref_position_
    Array<int> cached_vdofs_;  // Element DOF indices
};


// =============================================================================
// ReceiverArray
// =============================================================================

/**
 * @class ReceiverArray
 * @brief Manager for multiple receivers
 *
 * Handles receiver setup, recording, and output.
 */
class ReceiverArray {
public:
    /**
     * @brief Construct receiver array
     * @param fes Finite element space
     * @param comm MPI communicator
     * @param domain Domain type (Solid or Fluid)
     */
    ReceiverArray(const ParFiniteElementSpace& fes,
                  MPI_Comm* comm,
                  DomainType domain);

    // -------------------------------------------------------------------------
    // Factory methods
    // -------------------------------------------------------------------------

    /**
     * @brief Create ReceiverArray from configuration
     * @param config Receiver configuration (Grid already expanded)
     * @param fes Finite element space
     * @param nt Number of time steps
     * @param dt Time step
     * @param comm MPI communicator
     * @param domain Domain type (Solid or Fluid)
     * @return Unique pointer to ReceiverArray
     */
    static std::unique_ptr<ReceiverArray> FromConfig(
        const ReceiverConfig::Config& config,
        const ParFiniteElementSpace& fes,
        int nt, real_t dt,
        MPI_Comm* comm,
        DomainType domain = DomainType::Solid);

    // -------------------------------------------------------------------------
    // Setup from files
    // -------------------------------------------------------------------------

    /**
     * @brief Load receivers from text file
     * @param filepath Path to receiver file
     * @param nt Number of time steps
     * @param dt Time step
     *
     * File format:
     *   num_receivers
     *   type name x y [weight]  (2D)
     *   type name x y z         (3D)
     */
    void LoadFromTextFile(const std::string& filepath, int nt, real_t dt);

    // -------------------------------------------------------------------------
    // Programmatic setup (no file needed)
    // -------------------------------------------------------------------------

    /**
     * @brief Add a receiver programmatically
     * @param name Receiver name/identifier
     * @param position Physical position (2D or 3D)
     * @param type Measurement type
     * @param weight Optional weight (default 1.0)
     *
     * Call Setup() after adding all receivers to finalize.
     */
    void AddReceiver(const std::string& name,
                     const Vector& position,
                     ReceiverType type,
                     real_t weight = 1.0);

    /**
     * @brief Finalize receiver setup after AddReceiver() calls
     * @param nt Number of time steps
     * @param dt Time step
     *
     * Receivers whose position is outside the mesh are silently dropped
     * and logged to filtered_log_ (see SaveFilteredLog). This makes
     * coupled fluid-solid simulations work without per-receiver type
     * overrides — each submesh's array auto-keeps only the receivers
     * physically inside it.
     */
    void Setup(int nt, real_t dt);

    // -------------------------------------------------------------------------
    // Field registration
    // -------------------------------------------------------------------------

    /**
     * @brief Set field pointers for recording
     * @param u Displacement field
     * @param udot Velocity field (optional)
     * @param udot2 Acceleration field (optional)
     */
    void SetFields(ParGridFunction* u,
                   ParGridFunction* udot = nullptr,
                   ParGridFunction* udot2 = nullptr);

    // -------------------------------------------------------------------------
    // Recording
    // -------------------------------------------------------------------------

    /**
     * @brief Record data at time step (auto-detect dimension and device)
     * @param step Current time step
     * @param buffer_steps GPU buffer size (0 = all steps, ignored on CPU)
     */
    void Record(int step, int buffer_steps = 0);

    // -------------------------------------------------------------------------
    // Output
    // -------------------------------------------------------------------------

    /**
     * @brief Set output configuration for Save()
     * @param formats One or more of {"ascii", "hdf5", "su"}
     * @param outdir Output directory
     * @param filename Base filename (e.g., "seismograms"); required if hdf5/su in formats
     */
    void SetOutputConfig(const std::vector<std::string>& formats,
                         const std::string& outdir,
                         const std::string& filename);

    /** @brief Convenience overload for a single output format. */
    void SetOutputConfig(const std::string& format,
                         const std::string& outdir,
                         const std::string& filename);

    /**
     * @brief Register source descriptors to be embedded under
     *        `/shots/0000/sources/...` by the next HDF5 save. Has no
     *        effect on ASCII / SU output. Cleared by Init().
     *
     * Stage-5 self-roundtrip: a per-shot HDF5 file written by SEMSWS can
     * be re-read via the HDF5 input path, since both source metadata and
     * receiver waveforms live in the same file.
     */
    void SetOutputSourceContext(std::vector<HDF5SourceWriteEntry> sources);

    /// Set the input shot_id used for the internal `/shots/<NNNN>/` group key
    /// in HDF5 output. Defaults to 0 if never called.
    void SetShotId(int id) { shot_id_ = id; }

    /**
     * @brief Save receivers using the configured format
     * @param source_id Source identifier (used as filename suffix)
     * @param t0 Start time offset
     * @param source_pos Source position (for SU offset headers)
     */
    void Save(int source_id, real_t t0 = 0.0,
              const Vector* source_pos = nullptr);

    /**
     * @brief Save to text files
     * @param outdir Output directory
     * @param source_id Source identifier
     * @param t0 Start time offset
     */
    void SaveToTextFiles(const std::string& outdir,
                         int source_id,
                         real_t t0 = 0.0);

    /**
     * @brief Save to HDF5 file
     * @param outdir Output directory
     * @param filename Base filename
     * @param source_id Source identifier
     * @param overwrite Overwrite existing file
     * @param t0 Start time offset
     */
    void SaveToHDF5(const std::string& outdir,
                    const std::string& filename,
                    int source_id,
                    bool overwrite = true,
                    real_t t0 = 0.0);

    /**
     * @brief Save to Seismic Unix (SU) format files
     * @param outdir Output directory
     * @param filename Base filename (e.g., "seismo")
     * @param source_id Source identifier
     * @param t0 Start time offset
     * @param source_pos Source position (optional, sets sx/sy/offset headers)
     *
     * Writes one .su file per ReceiverType per component.
     * Each trace in the file corresponds to one receiver.
     */
    void SaveToSU(const std::string& outdir,
                  const std::string& filename,
                  int source_id,
                  real_t t0 = 0.0,
                  const Vector* source_pos = nullptr);

    // -------------------------------------------------------------------------
    // Query
    // -------------------------------------------------------------------------

    /// Total number of receivers
    int NumReceivers() const { return num_total_; }

    /// Number of receiver types
    int NumReceiverTypes() const { return static_cast<int>(receivers_.size()); }

    /// Get all receiver data (map by type)
    std::map<ReceiverType, std::vector<ReceiverData>>& GetAllReceivers();
    const std::map<ReceiverType, std::vector<ReceiverData>>& GetAllReceivers() const;

    // -------------------------------------------------------------------------
    // Reset
    // -------------------------------------------------------------------------

    /// Reset all data to zero
    void ResetData();

    /// Clear all receivers
    void Init();

    // -------------------------------------------------------------------------
    // GPU Recording (public for CUDA extended lambda compatibility)
    // -------------------------------------------------------------------------

    /// Record on GPU (called from Record() when device is enabled)
    void RecordDevice(int step, int max_buffer_steps);

    /// Flush all GPU buffers to host
    void FlushDeviceBuffer();

    /**
     * @brief Initialize GPU recording buffers
     * @param max_buffer_steps Buffer size in time steps (0 = buffer all steps)
     *
     * Must be called before RecordDevice(). Initializes device_data_ with
     * shape functions, DOF indices, and recording buffers.
     */
    void DeviceInit(int max_buffer_steps = 0);

private:
    void LocateReceivers();
    void PrecomputeInterpolation();  // Cache shape functions and DOFs
    void GatherData();  // MPI reduction for output

    void Record2D(int step);
    void Record3D(int step);
    void InitDeviceRecording(int max_buffer_steps);
    void FlushDeviceBufferForType(ReceiverType type);

    void ParseTextFile(const std::string& filepath);
    void RecordSingleReceiver2D(ReceiverData& rec, int step);
    void RecordSingleReceiver3D(ReceiverData& rec, int step);
    std::string GetSingleTraceName(const std::string& outdir,
                                   const ReceiverData& rec,
                                   int comp, int source_id) const;
    std::string GetSUFileName(const std::string& outdir,
                              const std::string& filename,
                              ReceiverType type,
                              int comp, int source_id) const;

    const ParFiniteElementSpace& fes_;
    MPI_Comm* comm_;
    DomainType domain_;
    int local_rank_;
    int num_procs_;
    int space_dim_;

    std::map<ReceiverType, std::vector<ReceiverData>> receivers_;
    int num_total_ = 0;

    // Field pointers
    ParGridFunction* u_ = nullptr;
    ParGridFunction* udot_ = nullptr;
    ParGridFunction* udot2_ = nullptr;

    // Time parameters
    int nt_ = 0;
    real_t dt_ = 0.0;

    // Output configuration (set by SetOutputConfig)
    std::vector<std::string> output_formats_ = {"ascii"};
    std::string output_dir_ = "./results";
    std::string output_filename_ = "seismograms";

    // Source context for self-roundtrip HDF5 output (set by SetOutputSourceContext).
    // Empty → /shots/<shot_id_>/sources/ is omitted (legacy receiver-only output).
    std::vector<HDF5SourceWriteEntry> output_sources_;

    // Input shot_id used for /shots/<NNNN>/ group key in HDF5 output.
    // Set via SetShotId(); default 0 keeps backward-compat with legacy
    // single-shot pipelines.
    int shot_id_ = 0;

    // -------------------------------------------------------------------------
    // GPU Recording Data (per ReceiverType)
    // -------------------------------------------------------------------------
    struct DeviceReceiverData {
        Vector d_shape;           ///< Shape functions [total_ndof]
        Array<int> d_vdofs;       ///< DOF indices [total_ndof * ncomp]
        Array<int> d_offsets;     ///< Receiver offsets [num_local + 1]
        Vector d_buffer;          ///< Recording buffer [num_local * buffer_steps * ncomp]
        int num_local = 0;        ///< Number of local receivers
        int total_ndof = 0;       ///< Total DOFs for all local receivers
        int ncomp = 0;            ///< Components (1 for pressure, space_dim for others)
        int buffer_steps = 0;     ///< Actual buffer size in steps
        int current_pos = 0;      ///< Current position in buffer
        int start_step = 0;       ///< First step in current buffer
        bool initialized = false; ///< Initialization flag
    };
    std::map<ReceiverType, DeviceReceiverData> device_data_;

    // Filtered receivers log (incompatible domain types)
    struct FilteredEntry {
        std::string name;
        std::string type;
        std::string reason;
    };
    std::vector<FilteredEntry> filtered_log_;

public:
    /// Get filtered receivers log
    const std::vector<FilteredEntry>& GetFilteredLog() const { return filtered_log_; }

    /// Save filtered receivers to text file
    void SaveFilteredLog(const std::string& filepath) const;

    /// Check if any receivers were filtered
    bool HasFilteredReceivers() const { return !filtered_log_.empty(); }
};


// =============================================================================
// Utility Functions
// =============================================================================

/// Create 4-digit source ID string
inline std::string MakeSourceIdString(int source_id) {
    char buffer[8];
    snprintf(buffer, sizeof(buffer), "%04d", source_id);
    return std::string(buffer);
}

}  // namespace SEM

#endif  // SEM_RECEIVER_HPP
