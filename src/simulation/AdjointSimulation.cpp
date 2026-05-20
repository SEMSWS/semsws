/**
 * @file AdjointSimulation.cpp
 * @brief Implementation of AdjointSimulation class
 */

#include "simulation/AdjointSimulation.hpp"
#include "fwi/SensitivityKernelFactory.hpp"
#include "material/AcousticMaterialBase.hpp"
#include "material/ElasticMaterialBase.hpp"
#include "operator/OperatorFactory.hpp"
#include "config/ConfigLoaders.hpp"
#include "config/ConfigTypes.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sys/stat.h>
#include <chrono>

namespace SEM {

// =============================================================================
// Constructor
// =============================================================================

template<int Dim>
AdjointSimulation<Dim>::AdjointSimulation(MPI_Comm comm)
    : SimulationFacade<Dim>(comm) {}

// =============================================================================
// Run — main entry point
// =============================================================================

template<int Dim>
void AdjointSimulation<Dim>::Run() {
    static_assert(Dim == 2 || Dim == 3,
                  "AdjointSimulation supports 2D and 3D");

    MFEM_VERIFY(this->config_, "Config must be loaded before Run()");
    MFEM_VERIFY(this->sources_, "Sources must be set up before Run()");

    num_checkpoints_ = this->config_->GetNumCheckpoints();

    // Determine mode
    std::string sim_mode = this->config_->GetSimulationMode();
    misfit_only_ = (sim_mode == "misfit_only");

    // Per-phase timers reset for this Run() call.
    ResetPhaseTimers();

    // Create kernel output directory (used for misfit files in both modes)
    std::string kernel_dir = this->config_->GetKernelOutputDir();
    if (kernel_dir.empty()) kernel_dir = this->OutputDir() + "/kernels";
    if (this->IsRoot()) {
        mkdir(kernel_dir.c_str(), 0755);
    }
    MPI_Barrier(this->Comm());

    // Create sensitivity kernel (inversion mode only)
    if (!misfit_only_) {
        CreateKernel();
    }

    total_misfit_ = 0.0;
    total_l2_misfit_ = 0.0;

    // Single forward+adjoint per shot. All configured sources fire together
    // (simultaneous superposition), mirroring ForwardSimulation::Run.
    // Multi-shot inversion is handled externally by the driver (per-shot
    // YAML synthesis), which accumulates kernels across shots via SumKernels.
    RunOneShot(kernel_dir);

    using Clock = std::chrono::steady_clock;
    const int shot_id = this->config_->GetSourceShotId();

    // Save kernel (inversion mode only)
    if (!misfit_only_) {
        auto t0 = Clock::now();
        kernel_->Save(kernel_dir, this->Mesh(), shot_id);
        kernel_->SaveHessian(kernel_dir, this->Mesh(), shot_id);
        t_io_kernel_ += std::chrono::duration<double>(Clock::now() - t0).count();
        n_kernel_writes_++;
        kernel_->Reset();
        kernel_->ResetHessian();
    }

    // Save shot misfit to file (both modes)
    if (this->IsRoot()) {
        auto t0 = Clock::now();
        char shot_str[8];
        snprintf(shot_str, sizeof(shot_str), "%04d", shot_id);
        std::string misfit_file = kernel_dir + "/misfit_shot"
            + shot_str + ".txt";
        std::ofstream ofs(misfit_file);
        ofs << std::scientific << std::setprecision(12)
            << total_misfit_ << std::endl;
        // Diagnostic L2 sidecar (parallel-computed, never feeds adjoint).
        std::string l2_file = kernel_dir + "/misfit_l2_shot"
            + shot_str + ".txt";
        std::ofstream ofs_l2(l2_file);
        ofs_l2 << std::scientific << std::setprecision(12)
            << total_l2_misfit_ << std::endl;
        t_io_misfit_ += std::chrono::duration<double>(Clock::now() - t0).count();
        n_misfit_writes_++;
    }
}

// =============================================================================
// RunOneShot — single-shot dispatcher (all sources fire simultaneously)
// =============================================================================

template<int Dim>
void AdjointSimulation<Dim>::RunOneShot(const std::string& kernel_dir) {
    if (misfit_only_) {
        RunMisfitOnly(kernel_dir);
    } else {
        // --- Phase 0: Common setup ---
        this->Reset();
        this->Initialize();

        // Load observed data from the bundled HDF5 (sources.file + shot_id).
        const std::string obs_file = this->config_->GetSourceFile();
        const int shot_id = this->config_->GetSourceShotId();
        MFEM_VERIFY(!obs_file.empty(),
            "sources.file is empty; required for inversion mode");
        ObservedData observed;
        observed.Load(obs_file, shot_id, Dim, this->Comm());

        // Determine domain type
        DomainType domain = GetDomainFromMaterial(this->Material().GetType());

        // Create adjoint source handler
        std::string misfit_type = this->config_->GetMisfitType();
        adjoint_source_ = std::make_unique<AdjointSource>(
            &this->FESpace(), this->Comm(), domain, misfit_type);
        adjoint_source_->SetObservedData(observed);

        // Create forward receivers at observed data positions
        fwd_receivers_ = adjoint_source_->CreateForwardReceivers(
            this->FESpace(), &this->comm_, domain,
            this->NumSteps(), this->Dt());

        // Per-rank hyperslab read of owned observed channels
        observed.FetchOwnedData(*fwd_receivers_, this->Comm());

        // Align observed grid (auto-resample on mismatch).
        observed.AlignToSimulation(this->NumSteps(), this->Dt());

        // Set fields for recording
        fwd_receivers_->SetFields(
            &this->Displacement(), &this->Velocity(), &this->Acceleration());

        // GPU device init for receivers
        if (Device::Allows(Backend::DEVICE_MASK)) {
            int buffer_steps = this->config_ ? this->config_->GetReceiverBufferSteps() : 0;
            fwd_receivers_->DeviceInit(buffer_steps);
        }

        // Create checkpoint storage
        storage_ = std::make_unique<InMemoryStorage>(num_checkpoints_);

        // --- Revolve adjoint (includes forward sweep with receiver recording) ---
        RevolveAdjoint(kernel_dir);

        // Cleanup
        fwd_receivers_.reset();
        adjoint_source_.reset();
        storage_.reset();
    }
}

// =============================================================================
// RunMisfitOnly — forward sweep + misfit computation only (no adjoint/kernel)
// =============================================================================

template<int Dim>
void AdjointSimulation<Dim>::RunMisfitOnly(const std::string& kernel_dir) {
    // Reset simulation state
    this->Reset();
    this->Initialize();

    // Load observed data from the bundled HDF5 (sources.file + shot_id).
    const std::string obs_file = this->config_->GetSourceFile();
    const int shot_id = this->config_->GetSourceShotId();
    MFEM_VERIFY(!obs_file.empty(),
        "sources.file is empty; required for misfit_only mode");
    ObservedData observed;
    observed.Load(obs_file, shot_id, Dim, this->Comm());

    // Determine domain type
    DomainType domain = GetDomainFromMaterial(this->Material().GetType());

    // Create adjoint source handler (for misfit computation)
    std::string misfit_type = this->config_->GetMisfitType();
    adjoint_source_ = std::make_unique<AdjointSource>(
        &this->FESpace(), this->Comm(), domain, misfit_type);
    adjoint_source_->SetObservedData(observed);

    // Create forward receivers at observed data positions
    fwd_receivers_ = adjoint_source_->CreateForwardReceivers(
        this->FESpace(), &this->comm_, domain,
        this->NumSteps(), this->Dt());

    // Per-rank hyperslab read of owned observed channels
    observed.FetchOwnedData(*fwd_receivers_, this->Comm());

    // Align observed grid (auto-resample on mismatch).
    observed.AlignToSimulation(this->NumSteps(), this->Dt());

    // Set fields for recording
    fwd_receivers_->SetFields(
        &this->Displacement(), &this->Velocity(), &this->Acceleration());

    // GPU device init for receivers
    if (Device::Allows(Backend::DEVICE_MASK)) {
        int buffer_steps = this->config_ ? this->config_->GetReceiverBufferSteps() : 0;
        fwd_receivers_->DeviceInit(buffer_steps);
    }

    // --- Forward sweep ---
    int nt = this->NumSteps();
    int buffer_steps = this->config_ ? this->config_->GetReceiverBufferSteps() : 0;
    for (int step = 0; step < nt; step++) {
        fwd_receivers_->Record(step, buffer_steps);
        this->runner_.Step(
            this->Displacement(), this->Velocity(), this->Acceleration());
    }

    // Flush GPU receiver buffers
    if (Device::Allows(Backend::DEVICE_MASK)) {
        fwd_receivers_->FlushDeviceBuffer();
    }

    // Save synthetic seismograms (filename suffix = shot_id, position from first source)
    {
        using Clock = std::chrono::steady_clock;
        fwd_receivers_->SetOutputConfig("hdf5", kernel_dir, "synthetic");
        Vector src_pos(Dim);
        if (this->sources_->NumSources() > 0) {
            const Vector& p = this->sources_->GetSource(0)->Position();
            for (int d = 0; d < Dim; d++) src_pos[d] = p[d];
        } else {
            src_pos = 0.0;
        }
        auto t0 = Clock::now();
        fwd_receivers_->Save(shot_id, this->T0(), &src_pos);
        t_io_synthetic_ += std::chrono::duration<double>(Clock::now() - t0).count();
        n_synthetic_writes_++;
    }

    // --- Compute misfit ---
    total_misfit_ = adjoint_source_->ComputeResidual(*fwd_receivers_);
    total_l2_misfit_ = adjoint_source_->L2MisfitValue();

    // Cleanup
    fwd_receivers_.reset();
    adjoint_source_.reset();
}

// =============================================================================
// RevolveAdjoint — Revolve-controlled forward (with receiver recording) + adjoint
// =============================================================================

template<int Dim>
void AdjointSimulation<Dim>::RevolveAdjoint(const std::string& kernel_dir) {
    int nt = this->NumSteps();
    const int shot_id = this->config_->GetSourceShotId();

    // Determine domain type and kappa pointer (needed at FirstTurn for BuildAdjointSources)
    DomainType domain = GetDomainFromMaterial(this->Material().GetType());
    const MaterialField* kappa_ptr = nullptr;
    if (domain == DomainType::Fluid) {
        const auto& acoustic_mat = static_cast<const AcousticMaterialBase2D&>(
            this->Material());
        kappa_ptr = &acoustic_mat.Kappa();
    }

    // Create adjoint state vectors (separate from forward)
    ParGridFunction adj_u_gf(&this->FESpace());
    ParGridFunction adj_v_gf(&this->FESpace());
    ParGridFunction adj_a_gf(&this->FESpace());
    adj_u_ = &adj_u_gf;
    adj_v_ = &adj_v_gf;
    adj_a_ = &adj_a_gf;

    adj_u_gf = 0.0;
    adj_v_gf = 0.0;
    adj_a_gf = 0.0;

    if (Device::Allows(Backend::DEVICE_MASK)) {
        adj_u_gf.UseDevice(true);
        adj_v_gf.UseDevice(true);
        adj_a_gf.UseDevice(true);
        adj_u_gf.Read();  // sync to device
        adj_v_gf.Read();
        adj_a_gf.Read();
    }

    adjoint_step_ = 0;

    // Enable device memory for forward snapshot vectors used in kernel accumulation.
    // Without this, operator= copies to host only and .Read() in GPU kernels
    // (forall_2D) would return host pointers, causing invalid device access.
    if (Device::Allows(Backend::DEVICE_MASK)) {
        fwd_u_snapshot_.UseDevice(true);
        fwd_a_snapshot_.UseDevice(true);
    }

    // Initialize separate adjoint attenuation state (zero at t=T)
    int att_size = this->Operator().AttenuationStateSize();
    if (att_size > 0) {
        adj_att_state_.SetSize(att_size);
        adj_att_state_ = 0.0;
        if (Device::Allows(Backend::DEVICE_MASK)) {
            adj_att_state_.UseDevice(true);
            adj_att_state_.Read();  // sync zero to device
        }
    }

    current_forward_step_ = 0;
    bool first_pass_done = false;

    // Revolve loop
    Revolve rev(nt, num_checkpoints_, this->IsRoot());

    // Progress file for monitoring (rank 0 only)
    std::string progress_path = kernel_dir + "/progress.txt";
    int revolve_iter = 0;
    auto t_start = std::chrono::steady_clock::now();
    auto t_last_report = t_start;
    int last_report_adj_step = 0;

    // Per-operation cumulative timers (seconds). Member-scoped so the
    // breakdown survives RevolveAdjoint return and reaches WriteSummaryToFile.
    // `t_advance` is split by phase: first-pass forward vs adjoint-replay.
    using Clock = std::chrono::steady_clock;
    int n_advance_count = 0;

    bool done = false;
    while (!done) {
        RevolveAction action = rev.Next();
        revolve_iter++;

        switch (action) {
        case RevolveAction::Advance: {
            n_advance_count++;
            int n_steps = rev.Capo() - rev.OldCapo();
            auto t0 = Clock::now();
            if (!first_pass_done) {
                AdvanceForward(rev.OldCapo(), rev.Capo(), fwd_receivers_.get());
            } else {
                AdvanceForward(rev.OldCapo(), rev.Capo());
            }
            double dt = std::chrono::duration<double>(Clock::now() - t0).count();
            if (!first_pass_done) {
                t_forward_advance_ += dt;
                n_forward_steps_   += n_steps;
            } else {
                t_adjoint_replay_  += dt;
                n_replay_steps_    += n_steps;
            }
            break;
        }

        case RevolveAction::TakeShot: {
            n_takeshot_++;
            auto t0 = Clock::now();
            SaveCheckpoint(rev.Check());
            t_save_ += std::chrono::duration<double>(Clock::now() - t0).count();
            break;
        }

        case RevolveAction::Restore: {
            n_restore_++;
            auto t0 = Clock::now();
            RestoreCheckpoint(rev.Check());
            t_restore_ += std::chrono::duration<double>(Clock::now() - t0).count();
            break;
        }

        case RevolveAction::FirstTurn: {
            // First forward pass complete — process receivers and setup adjoint
            first_pass_done = true;

            // Flush GPU receiver buffers
            if (Device::Allows(Backend::DEVICE_MASK)) {
                fwd_receivers_->FlushDeviceBuffer();
            }

            // Save synthetic seismograms
            {
                fwd_receivers_->SetOutputConfig("hdf5", kernel_dir, "synthetic");
                Vector src_pos(Dim);
                if (this->sources_->NumSources() > 0) {
                    const Vector& p = this->sources_->GetSource(0)->Position();
                    for (int d = 0; d < Dim; d++) src_pos[d] = p[d];
                } else {
                    src_pos = 0.0;
                }
                auto t0 = Clock::now();
                fwd_receivers_->Save(shot_id, this->T0(), &src_pos);
                t_io_synthetic_ += std::chrono::duration<double>(Clock::now() - t0).count();
                n_synthetic_writes_++;
            }

            // Compute residual (and the diagnostic L2 sidecar)
            total_misfit_ = adjoint_source_->ComputeResidual(*fwd_receivers_);
            total_l2_misfit_ = adjoint_source_->L2MisfitValue();

            // Build adjoint sources
            adjoint_source_->BuildAdjointSources(this->NumSteps(), this->Dt(), kappa_ptr);

            // Initialize adjoint integrator
            adj_integrator_ = std::make_unique<NewmarkCentralDifference>();
            PointSourceCollection* adj_sources = adjoint_source_->GetSourceCollection();
            this->Operator().SetupSource(*adj_sources);
            adj_integrator_->Init(this->Operator(), 0.0, this->Dt());

            // Restore forward source on operator for subsequent forward replay
            this->Operator().SetupSource(this->Sources());

            // FirstTurn also acts as the first adjoint step
            fwd_u_snapshot_ = this->Displacement();
            fwd_a_snapshot_ = this->Acceleration();
            {
                auto t0_adj = Clock::now();
                AdjointStep();
                t_adjstep_ += std::chrono::duration<double>(Clock::now() - t0_adj).count();
                n_adjoint_steps_++;
            }
            break;
        }

        case RevolveAction::YouTurn: {
            // Save forward state snapshot for kernel accumulation
            auto t0_snap = Clock::now();
            fwd_u_snapshot_ = this->Displacement();
            fwd_a_snapshot_ = this->Acceleration();
            t_snapshot_ += std::chrono::duration<double>(Clock::now() - t0_snap).count();

            // Adjoint step
            auto t0_adj = Clock::now();
            AdjointStep();
            t_adjstep_ += std::chrono::duration<double>(Clock::now() - t0_adj).count();
            n_adjoint_steps_++;
            // Write progress file every 100 adjoint steps (rank 0 only)
            if (adjoint_step_ % 100 == 0 && adjoint_step_ > 0 && this->IsRoot()) {
                auto t_now = std::chrono::steady_clock::now();
                double elapsed_total = std::chrono::duration<double>(t_now - t_start).count();
                double elapsed_interval = std::chrono::duration<double>(t_now - t_last_report).count();
                int steps_interval = adjoint_step_ - last_report_adj_step;
                double rate = (elapsed_interval > 0) ? steps_interval / elapsed_interval : 0;
                double pct = 100.0 * adjoint_step_ / nt;
                double eta = (rate > 0) ? (nt - adjoint_step_) / rate : 0;
                // Overwrite progress file from beginning
                double t_advance_total = t_forward_advance_ + t_adjoint_replay_;
                double t_other = elapsed_total - t_restore_ - t_save_
                                 - t_advance_total - t_snapshot_ - t_adjstep_;
                std::ofstream pf(progress_path, std::ios::trunc);
                pf << "adj_step: " << adjoint_step_ << " / " << nt
                   << " (" << std::fixed << std::setprecision(1) << pct << "%)\n"
                   << "elapsed: " << std::setprecision(1) << elapsed_total << " s\n"
                   << "rate: " << std::setprecision(1) << rate << " adj_steps/s\n"
                   << "ETA: " << std::setprecision(0) << eta << " s\n"
                   << "--- breakdown (cumulative) ---\n"
                   << "restore:  " << std::setprecision(1) << t_restore_ << " s"
                   << " (" << std::setprecision(0) << (100*t_restore_/elapsed_total) << "%)\n"
                   << "save:     " << std::setprecision(1) << t_save_ << " s"
                   << " (" << std::setprecision(0) << (100*t_save_/elapsed_total) << "%)\n"
                   << "fwd 1st:  " << std::setprecision(1) << t_forward_advance_ << " s"
                   << " (" << std::setprecision(0) << (100*t_forward_advance_/elapsed_total) << "%)\n"
                   << "fwd repl: " << std::setprecision(1) << t_adjoint_replay_ << " s"
                   << " (" << std::setprecision(0) << (100*t_adjoint_replay_/elapsed_total) << "%)\n"
                   << "snapshot: " << std::setprecision(1) << t_snapshot_ << " s"
                   << " (" << std::setprecision(0) << (100*t_snapshot_/elapsed_total) << "%)\n"
                   << "adjstep:  " << std::setprecision(1) << t_adjstep_ << " s"
                   << " (" << std::setprecision(0) << (100*t_adjstep_/elapsed_total) << "%)\n"
                   << "other:    " << std::setprecision(1) << t_other << " s"
                   << " (" << std::setprecision(0) << (100*t_other/elapsed_total) << "%)\n"
                   << "--- revolve stats ---\n"
                   << "takeshots: " << n_takeshot_ << "\n"
                   << "restores:  " << n_restore_ << "\n"
                   << "advances:  " << n_advance_count
                   << " (" << (n_forward_steps_ + n_replay_steps_) << " fwd steps)\n"
                   << "save_ms:   " << std::setprecision(1)
                   << (n_takeshot_ > 0 ? 1000*t_save_/n_takeshot_ : 0) << "\n"
                   << "restore_ms:" << std::setprecision(1)
                   << (n_restore_ > 0 ? 1000*t_restore_/n_restore_ : 0) << "\n";
                t_last_report = t_now;
                last_report_adj_step = adjoint_step_;
            }
            break;
        }

        case RevolveAction::Terminate:
            done = true;
            break;

        case RevolveAction::Error:
            MFEM_ABORT("Revolve error: " << static_cast<int>(rev.GetError()));
            break;
        }
    }

    // Cleanup adjoint state
    adj_u_ = nullptr;
    adj_v_ = nullptr;
    adj_a_ = nullptr;
    adj_integrator_.reset();
}

// =============================================================================
// AdvanceForward — run forward from from_step to to_step
// =============================================================================

template<int Dim>
void AdjointSimulation<Dim>::AdvanceForward(int from_step, int to_step,
                                             ReceiverArray* receivers) {
    int buffer_steps = 0;
    if (receivers) {
        buffer_steps = this->config_ ? this->config_->GetReceiverBufferSteps() : 0;
    }

    for (int s = from_step; s < to_step; s++) {
        if (receivers) {
            receivers->Record(s, buffer_steps);
        }
        this->runner_.Step(
            this->Displacement(), this->Velocity(), this->Acceleration());
        current_forward_step_ = s + 1;
    }
}

// =============================================================================
// SaveCheckpoint / RestoreCheckpoint
// =============================================================================

template<int Dim>
void AdjointSimulation<Dim>::SaveCheckpoint(int slot) {
    // Pack attenuation state
    Vector att_state;
    int att_size = this->Operator().AttenuationStateSize();
    if (att_size > 0) {
        att_state.SetSize(att_size);
        if (Device::Allows(Backend::DEVICE_MASK)) {
            att_state.UseDevice(true);
        }
        this->Operator().GetAttenuationState(att_state);
    }

    storage_->Save(slot,
                   this->Displacement(),
                   this->Velocity(),
                   this->Acceleration(),
                   att_state,
                   current_forward_step_);
}

template<int Dim>
void AdjointSimulation<Dim>::RestoreCheckpoint(int slot) {
    int restored_step = 0;
    Vector att_state;
    if (Device::Allows(Backend::DEVICE_MASK)) {
        att_state.UseDevice(true);
    }

    storage_->Load(slot,
                   this->Displacement(),
                   this->Velocity(),
                   this->Acceleration(),
                   att_state,
                   restored_step);

    // Restore attenuation state
    if (att_state.Size() > 0) {
        this->Operator().SetAttenuationState(att_state);
    }

    current_forward_step_ = restored_step;

    // Reset runner state to match restored step
    this->Runner().SetState(restored_step,
                            this->T0() + restored_step * this->Dt());
}

// =============================================================================
// AdjointStep — one adjoint time step + kernel accumulation
// =============================================================================

template<int Dim>
void AdjointSimulation<Dim>::AdjointStep() {
    // Accumulate kernel using current forward and adjoint wavefields
    // K_kappa += (1/kappa^2) * p_fwd_tt * p_adj * dt
    // K_rho   += (1/rho) * grad(p_fwd) . grad(p_adj) * dt
    kernel_->Accumulate(fwd_u_snapshot_, fwd_a_snapshot_, *adj_u_, this->Dt());

    // Swap to adjoint source for the adjoint time step
    PointSourceCollection* adj_sources = adjoint_source_->GetSourceCollection();
    this->Operator().SetupSource(*adj_sources);

    // Save forward attenuation state, load adjoint attenuation state
    // (forward and adjoint share the same operator, so we must swap states)
    Vector fwd_att_state;
    bool has_attenuation = (adj_att_state_.Size() > 0);
    if (has_attenuation) {
        fwd_att_state.SetSize(adj_att_state_.Size());
        if (Device::Allows(Backend::DEVICE_MASK)) {
            fwd_att_state.UseDevice(true);
        }
        this->Operator().GetAttenuationState(fwd_att_state);
        this->Operator().SetAttenuationState(adj_att_state_);
    }

    // Adjoint time step using adjoint step counter
    // adjoint_step_ counts from 0 (corresponding to forward step nt-1)
    // The adjoint STF is time-reversed: stf(0) = w(T)*residual(T), etc.
    real_t adj_time = adjoint_step_ * this->Dt();
    adj_integrator_->AdjointStep(*adj_u_, *adj_v_, *adj_a_, adj_time, this->Dt());
    adjoint_step_++;

    // Save adjoint attenuation state, restore forward attenuation state
    if (has_attenuation) {
        this->Operator().GetAttenuationState(adj_att_state_);
        this->Operator().SetAttenuationState(fwd_att_state);
    }

    // Restore forward source for subsequent forward replay
    this->Operator().SetupSource(this->Sources());
}

// =============================================================================
// CreateKernel
// =============================================================================

template<int Dim>
void AdjointSimulation<Dim>::CreateKernel() {
    MFEM_VERIFY(this->material_ready_, "Material must be set up before creating kernel");

    // Dispatch via domain (Fluid / Solid) to the matching dim-specific
    // sensitivity-kernel factory. Each factory aborts if a particular
    // (physics, material) combination is not yet implemented, so new cases
    // are added there rather than here.
    const std::string backend = this->config_
        ? this->config_->GetSensitivityBackend()
        : std::string("hand");
    const DomainType domain = this->Material().GetDomainType();
    if constexpr (Dim == 2) {
        if (domain == DomainType::Fluid) {
            kernel_ = CreateAcousticSensitivityKernel2D(
                static_cast<const AcousticMaterialBase2D&>(this->Material()),
                this->FESpace(), backend);
        } else {
            kernel_ = CreateElasticSensitivityKernel2D(
                static_cast<const ElasticMaterialBase2D&>(this->Material()),
                this->FESpace(), backend);
        }
    } else {
        if (domain == DomainType::Fluid) {
            kernel_ = CreateAcousticSensitivityKernel3D(
                static_cast<const AcousticMaterialBase3D&>(this->Material()),
                this->FESpace(), backend);
        } else {
            kernel_ = CreateElasticSensitivityKernel3D(
                static_cast<const ElasticMaterialBase3D&>(this->Material()),
                this->FESpace(), backend);
        }
    }

    MFEM_VERIFY(kernel_,
        "AdjointSimulation::CreateKernel: factory returned nullptr for material "
        << MaterialTypeToString(this->Material().GetType()));
}

// =============================================================================
// WriteSummaryToFile — extends base class with Revolve info
// =============================================================================

template<int Dim>
void AdjointSimulation<Dim>::WriteSummaryToFile(
    const std::string& filepath, const TimingInfo& timing) {
    // Base class summary (MPI collective)
    SimulationFacade<Dim>::WriteSummaryToFile(filepath, timing);

    // Append Revolve info + per-phase breakdown (root only).
    if (this->IsRoot()) {
        std::ofstream ofs(filepath, std::ios::app);

        if (!misfit_only_) {
            int nforw = Revolve::NumForw(this->NumSteps(), num_checkpoints_);
            double expense = Revolve::Expense(this->NumSteps(), num_checkpoints_);
            ofs << "\nCheckpointing (Revolve):\n";
            ofs << "  Checkpoints: " << num_checkpoints_ << "\n";
            ofs << "  Forward recomputations: " << nforw << "\n";
            ofs << "  Expected slowdown: " << std::fixed
                << std::setprecision(2) << expense << "x\n";
        }

        // Per-phase breakdown for this shot's forward+adjoint sweep.
        double t_io_total = t_io_kernel_ + t_io_synthetic_ + t_io_misfit_;
        double t_compute  = t_forward_advance_ + t_adjoint_replay_ + t_adjstep_;
        double t_ckpt_io  = t_save_ + t_restore_ + t_snapshot_;
        double t_accounted = t_compute + t_ckpt_io + t_io_total;
        double t_other = (timing.run_time > t_accounted)
                          ? (timing.run_time - t_accounted) : 0.0;

        auto pct = [&](double x) -> double {
            return (timing.run_time > 0) ? 100.0 * x / timing.run_time : 0.0;
        };

        ofs << std::fixed << std::setprecision(3);
        ofs << "\nPer-phase breakdown (this shot, seconds):\n";
        if (!misfit_only_) {
            ofs << "  Forward (1st pass):    " << t_forward_advance_
                << "  (" << std::setprecision(1) << pct(t_forward_advance_)
                << "%, " << n_forward_steps_ << " steps)\n";
            ofs << std::setprecision(3);
            ofs << "  Adjoint compute:       " << t_adjstep_
                << "  (" << std::setprecision(1) << pct(t_adjstep_)
                << "%, " << n_adjoint_steps_ << " steps)\n";
            ofs << std::setprecision(3);
            ofs << "  Adjoint fwd-replay:    " << t_adjoint_replay_
                << "  (" << std::setprecision(1) << pct(t_adjoint_replay_)
                << "%, " << n_replay_steps_ << " steps)\n";
            ofs << std::setprecision(3);
            ofs << "  Checkpoint IO:         " << t_ckpt_io
                << "  (" << std::setprecision(1) << pct(t_ckpt_io)
                << "%, save=" << n_takeshot_
                << " restore=" << n_restore_ << ")\n";
            ofs << std::setprecision(3);
            ofs << "  Kernel BP write:       " << t_io_kernel_
                << "  (" << std::setprecision(1) << pct(t_io_kernel_)
                << "%, " << n_kernel_writes_ << " writes)\n";
        } else {
            ofs << std::setprecision(3);
            ofs << "  Forward sweep:         " << t_forward_advance_
                << "  (" << std::setprecision(1) << pct(t_forward_advance_)
                << "%, " << n_forward_steps_ << " steps)\n";
        }
        ofs << std::setprecision(3);
        ofs << "  Synthetic HDF5 write:  " << t_io_synthetic_
            << "  (" << std::setprecision(1) << pct(t_io_synthetic_)
            << "%, " << n_synthetic_writes_ << " writes)\n";
        ofs << std::setprecision(3);
        ofs << "  Misfit file write:     " << t_io_misfit_
            << "  (" << std::setprecision(1) << pct(t_io_misfit_)
            << "%, " << n_misfit_writes_ << " writes)\n";
        ofs << std::setprecision(3);
        ofs << "  Other (residual/MPI):  " << t_other
            << "  (" << std::setprecision(1) << pct(t_other) << "%)\n";
    }
}

// =============================================================================
// Explicit Template Instantiations
// =============================================================================

template class AdjointSimulation<2>;
template class AdjointSimulation<3>;

}  // namespace SEM
