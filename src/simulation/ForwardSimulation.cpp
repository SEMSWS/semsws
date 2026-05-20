/**
 * @file ForwardSimulation.cpp
 * @brief Implementation of ForwardSimulation class
 */

#include "simulation/ForwardSimulation.hpp"
#include "config/ConfigLoaders.hpp"
#include "srcrecv/HDF5SourceReceiverWriter.hpp"
#include "util/Profiler.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>

namespace SEM {

// =============================================================================
// Constructor
// =============================================================================

template<int Dim>
ForwardSimulation<Dim>::ForwardSimulation(MPI_Comm comm)
    : SimulationFacade<Dim>(comm) {}

// =============================================================================
// Fluent Configuration
// =============================================================================

template<int Dim>
ForwardSimulation<Dim>& ForwardSimulation<Dim>::SetWavefieldWriter(
    std::unique_ptr<WavefieldWriter> writer) {
    wavefield_writers_.clear();
    wavefield_writers_.push_back(std::move(writer));
    return *this;
}

template<int Dim>
ForwardSimulation<Dim>& ForwardSimulation<Dim>::AddWavefieldWriter(
    std::unique_ptr<WavefieldWriter> writer) {
    wavefield_writers_.push_back(std::move(writer));
    return *this;
}

template<int Dim>
ForwardSimulation<Dim>& ForwardSimulation<Dim>::SetWavefieldWriters(
    std::vector<std::unique_ptr<WavefieldWriter>> writers) {
    wavefield_writers_ = std::move(writers);
    return *this;
}

template<int Dim>
ForwardSimulation<Dim>& ForwardSimulation<Dim>::EnableProgressOutput(int interval) {
    progress_enabled_ = (interval > 0);
    progress_interval_ = interval;

    // Set up log file in output directory
    std::string outdir = this->OutputDir();
    if (!outdir.empty() && progress_enabled_) {
        log_file_ = outdir + "/log.txt";
        // Create/truncate log file (rank 0 only)
        if (this->IsRoot()) {
            std::ofstream ofs(log_file_, std::ios::trunc);
        }
    }

    return *this;
}


// =============================================================================
// Run
// =============================================================================

template<int Dim>
void ForwardSimulation<Dim>::Run() {
    // Initialize if not already done
    const SimulationState& state = this->State();
    if (!state.is_initialized) {
        this->Initialize();
    }

    // Initialize wavefield writers
    for (auto& writer : wavefield_writers_) {
        writer->Init(this->Mesh(), this->NumSteps(), this->Comm());
    }

    double wall_start = MPI_Wtime();

    // Time stepping loop
    while (this->Step()) {
        const SimulationState& current_state = this->State();
        int step = current_state.step;

        // Default progress output with min/max (all ranks participate in MPI collective)
        if (progress_enabled_ && progress_interval_ > 0 && step % progress_interval_ == 0) {
            // MPI collective: all ranks compute local min/max then reduce
            real_t u_min = this->Displacement().Min();
            real_t u_max = this->Displacement().Max();
            real_t u_min_global, u_max_global;
            MPI_Allreduce(&u_min, &u_min_global, 1, HYPRE_MPI_REAL, MPI_MIN, this->Comm());
            MPI_Allreduce(&u_max, &u_max_global, 1, HYPRE_MPI_REAL, MPI_MAX, this->Comm());

            if (this->IsRoot()) {
                double elapsed = MPI_Wtime() - wall_start;
                double rate = (elapsed > 0) ? step / elapsed : 0.0;

                std::ostringstream line;
                line << "  Step " << std::setw(6) << step
                     << "/" << std::setw(6) << this->NumSteps()
                     << " | t=" << std::scientific << std::setprecision(3)
                     << current_state.time << " s"
                     << " | u=[" << std::setprecision(2)
                     << u_min_global << ", " << u_max_global << "]"
                     << std::fixed << std::setprecision(1)
                     << " | " << rate << " steps/s";

                std::cout << line.str() << std::endl;

                // Append to log file
                if (!log_file_.empty()) {
                    std::ofstream ofs(log_file_, std::ios::app);
                    if (ofs.is_open()) {
                        ofs << line.str() << "\n";
                    }
                }
            }
        }

        // Wavefield output at specified intervals.
        // Use input shot_id as the file id (matches receiver output naming).
        const int wf_id = this->config_->GetSourceShotId();
        for (auto& writer : wavefield_writers_) {
            if (writer->ShouldWrite(step)) {
                writer->Write(step, current_state.time,
                              &this->Displacement(),
                              &this->Velocity(),
                              &this->Acceleration(),
                              wf_id);
            }
        }
    }

    // Final wavefield output
    const SimulationState& final_state = this->State();
    const int wf_id_final = this->config_->GetSourceShotId();
    for (auto& writer : wavefield_writers_) {
        if (writer->ShouldWrite(final_state.step)) {
            writer->Write(final_state.step, final_state.time,
                          &this->Displacement(),
                          &this->Velocity(),
                          &this->Acceleration(),
                          wf_id_final);
        }
    }

    // Finalize wavefield writers
    for (auto& writer : wavefield_writers_) {
        writer->Finalize();
    }

    // Save receivers. Filename suffix uses the input shot_id (from
    // sources.shot_id, default 0) so multi-shot pipelines produce
    // distinct files even with all sources globally numbered as id=1.
    // Internal /shots/<NNNN>/ also uses shot_id (set via SetShotId).
    // Embed /shots/<NNNN>/sources/ into the HDF5 output so the file is
    // self-roundtrip — its source metadata can be re-read via the
    // HDF5 input path.
    if (this->receivers_ && this->sources_ && this->sources_->NumSources() > 0) {
        const int shot_id = this->config_->GetSourceShotId();
        this->receivers_->SetShotId(shot_id);

        // Build source descriptors from the live YAML/HDF5 config; STF
        // samples come from SourceTimeFunction::FromConfig (deterministic).
        std::vector<HDF5SourceWriteEntry> src_entries;
        if constexpr (Dim == 2) {
            auto cfg = LoadSourceConfig2D(*this->config_);
            src_entries = BuildSourceWriteEntries(
                cfg, this->NumSteps(), this->Dt());
        } else {
            auto cfg = LoadSourceConfig3D(*this->config_);
            src_entries = BuildSourceWriteEntries(
                cfg, this->NumSteps(), this->Dt());
        }
        this->receivers_->SetOutputSourceContext(std::move(src_entries));

        this->receivers_->Save(shot_id, this->T0(),
                               &this->sources_->GetSource(0)->Position());
    }

    // Record memory usage after simulation completes
    this->memory_report_.Record("RunComplete");

#ifdef SEM_ENABLE_PROFILING
    // Print profiling summary
    SEM::Profiler::Instance().PrintMPISummary(this->Comm());
#endif
}

// =============================================================================
// Explicit Template Instantiations
// =============================================================================

template class ForwardSimulation<2>;
template class ForwardSimulation<3>;

}  // namespace SEM
