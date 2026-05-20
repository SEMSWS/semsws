/**
 * @file AdjointSimulation.hpp
 * @brief Adjoint simulation for FWI gradient computation
 *
 * AdjointSimulation extends SimulationFacade with:
 * - Forward simulation with Revolve-controlled checkpointing
 * - Adjoint (time-reversed) simulation with kernel accumulation
 * - L2 waveform misfit computation
 * - Per-source loop (independent adjoint sweep per source)
 *
 * Uses Revolve (Griewank & Walther 2000) for optimal checkpointing:
 * minimizes forward recomputations given fixed checkpoint slots.
 */

#ifndef SEM_ADJOINT_SIMULATION_HPP
#define SEM_ADJOINT_SIMULATION_HPP

#include "simulation/SimulationFacade.hpp"
#include "revolve/Revolve.hpp"
#include "fwi/CheckpointStorage.hpp"
#include "srcrecv/AdjointSource.hpp"
#include "srcrecv/ObservedData.hpp"
#include "fwi/SensitivityKernel.hpp"
#include <memory>

namespace SEM {

/**
 * @class AdjointSimulation
 * @brief FWI gradient computation via adjoint method with Revolve checkpointing
 * @tparam Dim Spatial dimension (2 only for now)
 *
 * Supports two modes:
 * - "inversion": full gradient computation (forward + adjoint + kernel)
 * - "misfit_only": forward + misfit computation only (no adjoint/kernel)
 *
 * Workflow per shot (inversion mode), all configured sources fire simultaneously:
 * 1. Load observed data from the bundled HDF5 (sources.file + shot_id)
 * 2. Revolve-controlled forward sweep: record synthetics + save checkpoints
 * 3. At FirstTurn: compute residual → build adjoint sources
 * 4. Revolve-scheduled backward sweep: replay forward segments +
 *    adjoint steps with kernel accumulation
 * 5. Save kernel + per-shot misfit to file
 *
 * Workflow per shot (misfit_only mode):
 * 1. Load observed data from the bundled HDF5
 * 2. Forward sweep (0→T): record synthetics
 * 3. Compute misfit, save to file (no adjoint, no kernel)
 *
 * Multi-shot inversion is handled by the driver (per-shot YAML synthesis)
 * which accumulates kernels across shots externally.
 */
template<int Dim>
class AdjointSimulation : public SimulationFacade<Dim> {
public:
    using Self = AdjointSimulation<Dim>;

    explicit AdjointSimulation(MPI_Comm comm);
    ~AdjointSimulation() = default;

    /**
     * @brief Run FWI computation (inversion or misfit_only mode)
     *
     * Mode is determined by config->GetSimulationMode():
     * - "inversion": full gradient (forward + adjoint + kernel per source)
     * - "misfit_only": forward + misfit only (no adjoint, no kernel)
     */
    void Run();

    /// Misfit value for this shot (populated after Run())
    real_t TotalMisfit() const { return total_misfit_; }

    /// Write summary to file (extends base class with Revolve checkpointing info)
    void WriteSummaryToFile(const std::string& filepath, const TimingInfo& timing);

private:
    // Single-shot FWI workflow (all sources fire simultaneously)
    void RunOneShot(const std::string& kernel_dir);

    // misfit_only: forward sweep + misfit computation only
    void RunMisfitOnly(const std::string& kernel_dir);

    // Revolve-controlled forward + adjoint computation (records receivers during first pass)
    void RevolveAdjoint(const std::string& kernel_dir);

    // Forward stepping (with optional receiver recording)
    void AdvanceForward(int from_step, int to_step,
                        ReceiverArray* receivers = nullptr);
    void SaveCheckpoint(int slot);
    void RestoreCheckpoint(int slot);

    // Adjoint step: accumulate kernel at one time step
    void AdjointStep();

    // Create sensitivity kernel for current material
    void CreateKernel();

    // Members
    int num_checkpoints_ = 10;
    std::unique_ptr<CheckpointStorage> storage_;
    std::unique_ptr<AdjointSource> adjoint_source_;
    /// Dim-specialized sensitivity-kernel base. 2D FWI stores the 2D base,
    /// 3D stores the 3D base — factories return compatible derived types.
    using SensitivityKernelBase =
        std::conditional_t<Dim == 2,
                           SensitivityKernelBase2D,
                           SensitivityKernelBase3D>;
    std::unique_ptr<SensitivityKernelBase> kernel_;
    std::unique_ptr<ReceiverArray> fwd_receivers_;

    // Adjoint simulation state
    std::unique_ptr<WaveOperator> adj_op_;
    std::unique_ptr<ExplicitTimeIntegrator> adj_integrator_;
    ParGridFunction* adj_u_ = nullptr;
    ParGridFunction* adj_v_ = nullptr;
    ParGridFunction* adj_a_ = nullptr;

    // Forward state snapshot for kernel accumulation
    // (saved at each YouTurn step from the current forward state)
    Vector fwd_u_snapshot_;
    Vector fwd_a_snapshot_;

    // Adjoint attenuation state (separate from forward)
    Vector adj_att_state_;

    // Revolve state tracking
    int current_forward_step_ = 0;
    int adjoint_step_ = 0;

    // Per-shot misfit (all sources superposed). total_l2_misfit_ is the
    // diagnostic L2 sidecar (see AdjointSource::L2MisfitValue), never used
    // for adjoint.
    real_t total_misfit_ = 0.0;
    real_t total_l2_misfit_ = 0.0;

    // Mode flag
    bool misfit_only_ = false;

    // Per-phase timing accumulators (seconds, summed over all sources).
    // Reset at the top of Run(); read by WriteSummaryToFile to print a
    // forward / adjoint / IO breakdown after the base summary block.
    double t_forward_advance_ = 0.0;   // first-pass forward Advance
    double t_adjoint_replay_  = 0.0;   // forward Advance during adjoint replay
    double t_adjstep_         = 0.0;   // adjoint integrator Step
    double t_save_            = 0.0;   // checkpoint TakeShot
    double t_restore_         = 0.0;   // checkpoint Restore
    double t_snapshot_        = 0.0;   // forward state snapshot per YouTurn
    double t_io_kernel_       = 0.0;   // kernel BP + Hessian write
    double t_io_synthetic_    = 0.0;   // synthetic HDF5 write
    double t_io_misfit_       = 0.0;   // misfit_shot{N}.txt ASCII write

    // Per-phase counters
    int n_forward_steps_      = 0;     // first-pass forward step count
    int n_replay_steps_       = 0;     // forward steps during adjoint replay
    int n_adjoint_steps_      = 0;     // adjoint integrator step count
    int n_takeshot_           = 0;
    int n_restore_            = 0;
    int n_kernel_writes_      = 0;
    int n_synthetic_writes_   = 0;
    int n_misfit_writes_      = 0;

    void ResetPhaseTimers() {
        t_forward_advance_ = t_adjoint_replay_ = t_adjstep_ = 0.0;
        t_save_ = t_restore_ = t_snapshot_ = 0.0;
        t_io_kernel_ = t_io_synthetic_ = t_io_misfit_ = 0.0;
        n_forward_steps_ = n_replay_steps_ = n_adjoint_steps_ = 0;
        n_takeshot_ = n_restore_ = 0;
        n_kernel_writes_ = n_synthetic_writes_ = n_misfit_writes_ = 0;
    }
};

// Type aliases
using AdjointSimulation2D = AdjointSimulation<2>;
using AdjointSimulation3D = AdjointSimulation<3>;

}  // namespace SEM

#endif  // SEM_ADJOINT_SIMULATION_HPP
