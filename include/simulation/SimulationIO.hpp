/**
 * @file SimulationIO.hpp
 * @brief I/O operations for simulation: receivers, wavefield output, diagnostics
 *
 * SimulationIO handles:
 * - Receiver recording and output (text, HDF5)
 * - Wavefield output (via WavefieldWriter strategy)
 * - Snapshot saving
 * - PrintSummary diagnostics
 *
 * This class uses non-owning pointers for receivers (owned by SimulationFacade)
 * and owns the WavefieldWriter instance.
 */

#ifndef SEM_SIMULATION_IO_HPP
#define SEM_SIMULATION_IO_HPP

#include <mfem.hpp>
#include <memory>
#include <string>
#include <iostream>

#include "srcrecv/Receiver.hpp"
#include "srcrecv/Source.hpp"
#include "io/WavefieldWriter.hpp"
#include "simulation/SimulationComponents.hpp"
#include "simulation/SimulationRunner.hpp"
#include "material/MaterialBase.hpp"
#include "common/Types.hpp"

namespace SEM {


using namespace mfem;

/**
 * @class SimulationIO
 * @brief Manages I/O operations for simulation
 * @tparam Dim Spatial dimension (2 or 3)
 *
 * Responsibilities:
 * - Receiver management (recording, text/HDF5 output)
 * - Wavefield output (via WavefieldWriter strategy)
 * - Snapshot saving
 * - Simulation summary printing
 *
 * This class does NOT own:
 * - ReceiverArray (receives raw pointer from SimulationFacade)
 *
 * This class OWNS:
 * - WavefieldWriter (via unique_ptr)
 */
template<int Dim>
class SimulationIO {
public:
    /**
     * @brief Construct IO manager with MPI communicator
     * @param comm MPI communicator
     */
    explicit SimulationIO(MPI_Comm comm);

    /// Destructor
    ~SimulationIO() = default;

    // -------------------------------------------------------------------------
    // Output Directory
    // -------------------------------------------------------------------------

    /**
     * @brief Set output directory
     * @param dir Output directory path
     */
    void SetOutputDirectory(const std::string& dir);

    // -------------------------------------------------------------------------
    // Receiver Management
    // -------------------------------------------------------------------------

    /**
     * @brief Set receiver array (non-owning)
     * @param receivers Pointer to ReceiverArray
     */
    void SetReceivers(ReceiverArray* receivers);

    /**
     * @brief Record receivers at current step
     * @param step Current time step
     * @param buffer_steps GPU buffer size (0 = all steps)
     */
    void RecordReceivers(int step, int buffer_steps = 0);

    // -------------------------------------------------------------------------
    // Wavefield Output
    // -------------------------------------------------------------------------

    /**
     * @brief Set wavefield writer (takes ownership)
     * @param writer WavefieldWriter instance
     */
    void SetWavefieldWriter(std::unique_ptr<WavefieldWriter> writer);

    /**
     * @brief Initialize wavefield writer
     * @param mesh Parallel mesh
     * @param total_steps Total number of time steps
     */
    void InitWavefieldWriter(ParMesh& mesh, int total_steps);

    /**
     * @brief Write wavefield at current step
     * @param step Current time step
     * @param time Current simulation time
     * @param u Displacement field (can be nullptr)
     * @param v Velocity field (can be nullptr)
     * @param a Acceleration field (can be nullptr)
     */
    void WriteWavefield(int step, real_t time,
                        const ParGridFunction* u,
                        const ParGridFunction* v = nullptr,
                        const ParGridFunction* a = nullptr);

    /**
     * @brief Finalize wavefield writer
     */
    void FinalizeWavefieldWriter();

    // -------------------------------------------------------------------------
    // Diagnostics
    // -------------------------------------------------------------------------

    /**
     * @brief Print simulation summary
     * @param os Output stream
     * @param components Simulation components (for mesh/FES info)
     * @param material Material (for type info)
     * @param dt Time step
     * @param nt Number of time steps
     * @param num_procs Number of MPI processes
     * @param sources Source collection (optional, for source info)
     * @param receivers Receiver array (optional, for receiver info)
     */
    void PrintSummary(std::ostream& os,
                      const SimulationComponents<Dim>& components,
                      const MaterialBase* material,
                      real_t dt, int nt, int num_procs,
                      const PointSourceCollection* sources = nullptr,
                      const ReceiverArray* receivers = nullptr) const;

    // -------------------------------------------------------------------------
    // Query
    // -------------------------------------------------------------------------

    /// Get MPI communicator
    MPI_Comm Comm() const { return comm_; }

    /// Check if root process
    bool IsRoot() const { return rank_ == 0; }

private:
    MPI_Comm comm_;
    int rank_ = 0;

    std::string output_dir_;

    // Non-owning pointer to receivers
    ReceiverArray* receivers_ = nullptr;

    // Owned wavefield writer
    std::unique_ptr<WavefieldWriter> wavefield_writer_;
};

// Type aliases
using SimulationIO2D = SimulationIO<2>;
using SimulationIO3D = SimulationIO<3>;

}  // namespace SEM

#endif  // SEM_SIMULATION_IO_HPP
