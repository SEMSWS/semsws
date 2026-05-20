// Source.hpp — point source classes (single force, moment tensor) and
// collection manager. Sources are batch-located via PointFinder; only the
// owning rank allocates shape functions, DOFs, and STF.

#ifndef SEM_SOURCE_HPP
#define SEM_SOURCE_HPP

#include <mfem.hpp>
#include <memory>
#include <vector>
#include <string>

#include "material/Material.hpp"
#include "config/ConfigTypes.hpp"

namespace SEM {


using namespace mfem;

// =============================================================================
// SourceTimeFunction
// =============================================================================

/**
 * @class SourceTimeFunction
 * @brief Source time function storage
 *
 * Stores source time function values as a DenseMatrix [nt x ncomp]
 * where ncomp is the number of force components.
 */
class SourceTimeFunction {
public:
    SourceTimeFunction() = default;

    /**
     * @brief Construct from dense matrix
     * @param stf Source time function [nt x ncomp]
     * @param dt Time step
     */
    SourceTimeFunction(const DenseMatrix& stf, real_t dt);

    /**
     * @brief Construct from time vector and values
     * @param times Time values [nt]
     * @param values Function values [nt x ncomp]
     */
    SourceTimeFunction(const Vector& times, const DenseMatrix& values);

    /**
     * @brief Construct single-component Ricker wavelet
     * @param f0 Dominant frequency (Hz)
     * @param t0 Time shift
     * @param dt Time step
     * @param nt Number of time steps
     * @param amplitude Amplitude scaling
     */
    static SourceTimeFunction Ricker(real_t f0, real_t t0, real_t dt, int nt,
                                     real_t amplitude = 1.0);

    /**
     * @brief Construct single-component Gaussian wavelet
     */
    static SourceTimeFunction Gaussian(real_t f0, real_t t0, real_t dt, int nt,
                                       real_t amplitude = 1.0);

    /**
     * @brief Factory from wavelet configuration
     * @param config Wavelet configuration
     * @param nt Number of time steps
     * @param dt Time step
     * @return Source time function (single component)
     */
    static SourceTimeFunction FromConfig(
        const SourceConfig::WaveletConfig& config,
        int nt, real_t dt);

    /**
     * @brief Load from external ASCII file
     * @param filepath Path to STF file
     * @param dt_sim Simulation time step
     * @param nt_sim Number of simulation time steps
     * @return Source time function
     *
     * File format (2 columns, whitespace separated):
     *   time [s]    amplitude
     *   0.0000      0.000000e+00
     *   0.0001      1.234568e-05
     *   ...
     *
     * - Lines starting with '#' are comments
     * - Empty lines are ignored
     * - First column: time in seconds
     * - Second column: amplitude value
     * - File must have at least nt_sim samples
     * - Warning if file dt differs from simulation dt
     */
    static SourceTimeFunction FromExternalFile(
        const std::string& filepath,
        real_t dt_sim,
        int nt_sim);

    /// Get value at time step for all components
    void GetValue(int step, Vector& value) const;

    /// Get value at time step for single component
    real_t GetValue(int step, int component) const;

    /// Get reference to underlying data
    const DenseMatrix& Data() const { return stf_; }
    DenseMatrix& Data() { return stf_; }

    /// Query
    int NumTimeSteps() const { return nt_; }
    int NumComponents() const { return ncomp_; }
    real_t DeltaT() const { return dt_; }

    /// Check if initialized
    bool IsValid() const { return nt_ > 0; }

    /// Release STF data (for non-local sources to save memory)
    void Clear() { stf_.SetSize(0, 0); nt_ = 0; ncomp_ = 0; }

private:
    DenseMatrix stf_;  // [nt x ncomp]
    real_t dt_ = 0.0;
    int nt_ = 0;
    int ncomp_ = 0;
};


// =============================================================================
// PointSourceBase
// =============================================================================

/**
 * @class PointSourceBase
 * @brief Base class for point sources
 *
 * Stores source metadata (position, id). Location and shape functions are set by
 * PointSourceCollection::Setup() after batch PointFinder.
 * Only local sources (owning rank) have shape_, source_dofs_, stf_ allocated.
 *
 * Workflow:
 *   1. Constructor — stores only position and id (lightweight, all ranks)
 *   2. SetLocation() — called by Setup() with batch PointFinder results
 *   3. ComputeShapeFunctions() — called by Setup() for local sources only
 *   4. SetSTF() — called by caller for local sources only (zero allocation on non-local)
 */
class PointSourceBase {
public:
    PointSourceBase(int id, ParFiniteElementSpace* fes,
                    const Vector& position);

    virtual ~PointSourceBase() = default;

    /// Assemble contribution at time step into rhs vector
    virtual void Assemble(int step, Vector& rhs) = 0;

    /// Called by PointSourceCollection::Setup() to set location from batch PointFinder
    void SetLocation(int elem, int owner_rank, const Vector& ref_pos, bool is_local);

    /// Compute shape functions at source location (called by Setup for local sources)
    void ComputeShapeFunctions();

    /// Set STF (called by caller for local sources only, after Setup)
    void SetSTF(const SourceTimeFunction& stf) { stf_ = stf; }

    /// Set STF (move version for efficiency)
    void SetSTF(SourceTimeFunction&& stf) { stf_ = std::move(stf); }

    /// Called after Setup() for acoustic sources to scale shape by 1/kappa
    virtual void ApplyAcousticScaling() {}

    /// Get config source ID
    int GetId() const { return id_; }

    /// Check if source is on this rank
    bool IsLocal() const { return is_local_; }

    /// Get physical position
    const Vector& Position() const { return position_; }

    /// Get reference position
    const Vector& RefPosition() const { return ref_position_; }

    /// Get element index
    int ElementIndex() const { return elem_; }

    /// Enable GPU memory for source data (only meaningful for local sources)
    virtual void EnableDevice() {
        if (!is_local_) return;
        shape_.UseDevice(true);
        source_dofs_.GetMemory().UseDevice(true);
    }

    /// Force host→device sync for source data (only meaningful for local sources)
    virtual void SyncToDevice() {
        if (!is_local_) return;
        shape_.Read();
        source_dofs_.Read();
    }

protected:
    int id_;
    ParFiniteElementSpace* fes_;
    Vector position_;
    Vector ref_position_;
    SourceTimeFunction stf_;  // Only allocated for local sources (set via SetSTF after Setup)

    Array<int> source_dofs_;  // Only allocated for local sources
    Vector shape_;            // Only allocated for local sources
    int elem_ = -1;
    int owner_rank_ = -1;
    bool is_found_ = false;
    bool is_local_ = false;
    int dim_ = 0;
    int ndof_ = 0;
};


// =============================================================================
// SingleForceSource
// =============================================================================

/**
 * @class SingleForceSource
 * @brief Single force source for elastic or acoustic media
 *
 * For elastic: applies force in each direction (ncomp = dim)
 * For acoustic: applies pressure source (ncomp = 1)
 */
class SingleForceSource : public PointSourceBase {
public:
    /// Construct single force source (elastic)
    SingleForceSource(int id, ParFiniteElementSpace* fes,
                      const Vector& position);

    /// Construct acoustic source (2D) — kappa stored for post-Setup scaling
    SingleForceSource(int id, ParFiniteElementSpace* fes,
                      const Vector& position,
                      const MaterialField& kappa);

    /// Construct acoustic source (3D) — kappa stored for post-Setup scaling
    SingleForceSource(int id, ParFiniteElementSpace* fes,
                      const Vector& position,
                      const MaterialField3D& kappa);

    void Assemble(int step, Vector& rhs) override;
    void ApplyAcousticScaling() override;

private:
    bool is_acoustic_ = false;
    real_t pressure_scale_ = 1.0;
    const MaterialField* kappa_2d_ = nullptr;    // Non-owning, for post-Setup scaling
    const MaterialField3D* kappa_3d_ = nullptr;  // Non-owning, for post-Setup scaling
};


// =============================================================================
// MomentTensorSource
// =============================================================================

/**
 * @class MomentTensorSource
 * @brief Moment tensor source for earthquake simulation
 *
 * Applies equivalent body forces from moment tensor.
 */
class MomentTensorSource : public PointSourceBase {
public:
    MomentTensorSource(int id, ParFiniteElementSpace* fes,
                       const Vector& position,
                       const DenseMatrix& moment);

    void Assemble(int step, Vector& rhs) override;

    void EnableDevice() override {
        PointSourceBase::EnableDevice();
        if (is_local_) equivalent_force_.UseDevice(true);
    }

    void SyncToDevice() override {
        PointSourceBase::SyncToDevice();
        if (is_local_) equivalent_force_.Read();
    }

    /// Compute equivalent forces (called by Setup for local sources)
    void ComputeEquivalentForces();

private:
    DenseMatrix moment_;
    Vector equivalent_force_;  // Pre-computed force vector (local only)
};


// =============================================================================
// PointSourceCollection
// =============================================================================

/**
 * @class PointSourceCollection
 * @brief Manager for multiple point sources (batch PointFinder pattern)
 *
 * Workflow:
 *   1. Add*() — register sources (stores position + id only, no PointFinder, no STF)
 *   2. Setup() — one batch PointFinder, compute shape/DOF for local sources only
 *   3. Caller sets STF for local sources via GetSource(i)->SetSTF()
 *   4. Assemble() — inject sources into RHS
 *
 * FromConfig*() factory methods handle Add + Setup + SetSTF internally.
 */
class PointSourceCollection {
public:
    /// Constructor with communicator by value (avoids dangling pointer issues)
    explicit PointSourceCollection(ParFiniteElementSpace* fes, MPI_Comm comm);

    // =========================================================================
    // Factory methods from configuration
    // =========================================================================

    static std::unique_ptr<PointSourceCollection> FromConfig(
        const SourceConfig::Config2D& config,
        ParFiniteElementSpace* fes,
        int nt, real_t dt, MPI_Comm comm);

    static std::unique_ptr<PointSourceCollection> FromConfig(
        const SourceConfig::Config3D& config,
        ParFiniteElementSpace* fes,
        int nt, real_t dt, MPI_Comm comm);

    static std::unique_ptr<PointSourceCollection> FromConfigAcoustic(
        const SourceConfig::Config2D& config,
        ParFiniteElementSpace* fes,
        const MaterialField& kappa,
        int nt, real_t dt, MPI_Comm comm);

    static std::unique_ptr<PointSourceCollection> FromConfigAcoustic(
        const SourceConfig::Config3D& config,
        ParFiniteElementSpace* fes,
        const MaterialField3D& kappa,
        int nt, real_t dt, MPI_Comm comm);

    // =========================================================================
    // Source registration (Phase 1: no PointFinder)
    // =========================================================================

    /// Register single force source (elastic)
    void AddSingleForce(int id, const Vector& position);

    /// Register single force source for acoustic (2D)
    void AddSingleForceAcoustic(int id, const Vector& position,
                                const MaterialField& kappa);

    /// Register single force source for acoustic (3D)
    void AddSingleForceAcoustic3D(int id, const Vector& position,
                                  const MaterialField3D& kappa);

    /// Register moment tensor source
    void AddMomentTensor(int id, const Vector& position,
                         const DenseMatrix& moment);

    // =========================================================================
    // Setup (Phase 2: batch PointFinder + local-only initialization)
    // =========================================================================

    /**
     * @brief Locate all registered sources and initialize local ones
     *
     * One batch PointFinder::FindPoints() for all sources.
     * Local sources: compute shape functions.
     * Non-local sources: remain lightweight (position + id only).
     * Must be called after all Add*() calls.
     * After Setup(), caller sets STF for local sources via GetSource(i)->SetSTF().
     */
    void Setup();

    // =========================================================================
    // Runtime methods
    // =========================================================================

    /// Assemble all active sources at time step
    void Assemble(int step, ParGridFunction& rhs);

    /// Reset all sources (clear for new simulation)
    void Reset();

    /// Restrict assembly to a single source (used by AdjointSimulation's
    /// per-source loop). Forward simulation runs all sources together.
    void SetActiveSource(int id);

    /// Clear active source filter so all sources contribute again.
    void ClearActiveSource();

    /// Initialize (alias for Reset for legacy compatibility)
    void Init() { Reset(); }

    /// Number of sources
    int NumSources() const { return static_cast<int>(sources_.size()); }

    /// Get source by index
    PointSourceBase* GetSource(int i) { return sources_[i].get(); }
    const PointSourceBase* GetSource(int i) const { return sources_[i].get(); }

    /// Enable GPU memory for all local sources
    void EnableDevice() {
        for (auto& src : sources_) {
            src->EnableDevice();
        }
    }

    /// Force host→device sync for all local sources
    void SyncToDevice() {
        for (auto& src : sources_) {
            src->SyncToDevice();
        }
    }

private:
    ParFiniteElementSpace* fes_;
    MPI_Comm comm_;  // Store by value to avoid dangling pointer
    std::vector<std::unique_ptr<PointSourceBase>> sources_;
    int active_source_id_ = -1;  // -1 = all sources active (default)
    bool setup_done_ = false;
};


}  // namespace SEM

#endif  // SEM_SOURCE_HPP
