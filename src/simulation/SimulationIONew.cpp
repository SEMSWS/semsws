/**
 * @file SimulationIONew.cpp
 * @brief Implementation of SimulationIO template class
 *
 * This is a new implementation for the standalone SimulationIO component.
 * The old SimulationIO.cpp will be removed after migration is complete.
 */

#include "simulation/SimulationIO.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>

namespace SEM {

// =============================================================================
// Constructor
// =============================================================================

template<int Dim>
SimulationIO<Dim>::SimulationIO(MPI_Comm comm)
    : comm_(comm)
{
    MPI_Comm_rank(comm_, &rank_);
}

// =============================================================================
// Output Directory
// =============================================================================

template<int Dim>
void SimulationIO<Dim>::SetOutputDirectory(const std::string& dir) {
    output_dir_ = dir;
}

// =============================================================================
// Receiver Management
// =============================================================================

template<int Dim>
void SimulationIO<Dim>::SetReceivers(ReceiverArray* receivers) {
    receivers_ = receivers;
}

template<int Dim>
void SimulationIO<Dim>::RecordReceivers(int step, int buffer_steps) {
    if (receivers_) {
        receivers_->Record(step, buffer_steps);
    }
}

// =============================================================================
// Wavefield Output
// =============================================================================

template<int Dim>
void SimulationIO<Dim>::SetWavefieldWriter(std::unique_ptr<WavefieldWriter> writer) {
    wavefield_writer_ = std::move(writer);
}

template<int Dim>
void SimulationIO<Dim>::InitWavefieldWriter(ParMesh& mesh, int total_steps) {
    if (wavefield_writer_) {
        wavefield_writer_->Init(mesh, total_steps, comm_);
    }
}

template<int Dim>
void SimulationIO<Dim>::WriteWavefield(int step, real_t time,
                                        const ParGridFunction* u,
                                        const ParGridFunction* v,
                                        const ParGridFunction* a) {
    if (wavefield_writer_) {
        wavefield_writer_->Write(step, time, u, v, a);
    }
}

template<int Dim>
void SimulationIO<Dim>::FinalizeWavefieldWriter() {
    if (wavefield_writer_) {
        wavefield_writer_->Finalize();
    }
}

// =============================================================================
// Diagnostics
// =============================================================================

template<int Dim>
void SimulationIO<Dim>::PrintSummary(std::ostream& os,
                                      const SimulationComponents<Dim>& components,
                                      const MaterialBase* material,
                                      real_t dt, int nt, int num_procs,
                                      const PointSourceCollection* sources,
                                      const ReceiverArray* receivers) const {
    // Gather MPI collective data BEFORE the IsRoot() check
    // GlobalTrueVSize() is an MPI collective - all ranks must call it
    HYPRE_BigInt global_dofs = 0;
    if (components.HasFESpaces()) {
        global_dofs = components.FES().GlobalTrueVSize();
    }
    long long global_ne = 0;
    if (components.HasMesh()) {
        global_ne = components.Mesh().GetGlobalNE();
    }

    // Now only root prints the rest
    if (!IsRoot()) return;

    os << "\n";
    os << "========================================\n";
    os << "        SEMSWS " << Dim << "D Simulation\n";
    os << "========================================\n";
    os << "\n";

    if (material) {
        os << "Material: " << MaterialTypeToString(material->GetType()) << "\n";
        // Velocity range
        real_t vmax = material->GetMaxVelocity();
        real_t vmin = material->GetMinVelocity();
        os << std::fixed << std::setprecision(2);
        os << "  Vmax: " << vmax << " m/s, Vmin: " << vmin << " m/s\n";
        // Attenuation info
        if (material->HasAttenuation()) {
            os << "  Attenuation: enabled (f0=" << material->AttenuationF0()
               << " Hz, " << material->AttenuationNumUnits() << " SLS units)\n";
        } else {
            os << "  Attenuation: disabled\n";
        }
        os << std::defaultfloat;
    }

    os << "\nDiscretization:\n";
    os << "  Order: " << components.Order() << "\n";
    if (components.HasMesh()) {
        os << "  Elements: " << global_ne << "\n";
    }
    if (global_dofs > 0) {
        os << "  DOFs: " << global_dofs << "\n";
    }

    // Sources information
    if (sources && sources->NumSources() > 0) {
        os << "\nSources: " << sources->NumSources() << " total\n";
    }

    // Receivers information
    if (receivers && receivers->NumReceivers() > 0) {
        os << "\nReceivers: " << receivers->NumReceivers() << " total\n";

        // Count by type
        const auto& all_receivers = receivers->GetAllReceivers();
        std::vector<std::pair<std::string, int>> type_counts;
        for (const auto& entry : all_receivers) {
            int count = static_cast<int>(entry.second.size());
            if (count > 0) {
                type_counts.push_back({ReceiverTypeToString(entry.first), count});
            }
        }
        if (!type_counts.empty()) {
            os << "  Types: ";
            for (size_t i = 0; i < type_counts.size(); ++i) {
                if (i > 0) os << ", ";
                os << type_counts[i].second << " " << type_counts[i].first;
            }
            os << "\n";
        }
    }

    os << "\nTime stepping:\n";
    os << "  dt: " << dt << " s\n";
    os << "  nt: " << nt << "\n";
    os << "  Total time: " << dt * nt << " s\n";

    os << "\nMPI:\n";
    os << "  Processes: " << num_procs << "\n";

    os << "========================================\n\n";
}

// =============================================================================
// Explicit Template Instantiations
// =============================================================================

template class SimulationIO<2>;
template class SimulationIO<3>;

}  // namespace SEM
