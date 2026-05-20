/**
 * @file ForwardSimulation.hpp
 * @brief High-level forward simulation class with wavefield output support
 *
 * ForwardSimulation extends SimulationBase with:
 * - WavefieldWriter integration for GLVis/HDF5/ADIOS output
 * - Fluent interface for callback configuration
 * - Progress output control
 *
 * Example usage:
 * @code
 *   ForwardSimulation3D sim(MPI_COMM_WORLD);
 *   sim.LoadConfig("config.yaml")
 *      .SetupFromConfig();
 *
 *   if (sim.IsWavefieldOutputEnabled()) {
 *       sim.SetWavefieldWriter(
 *           std::make_unique<GLVisWavefieldWriter>(
 *               sim.OutputDir(), sim.WavefieldInterval()));
 *   }
 *
 *   sim.OnStep([](const SimulationState& s, auto& u, auto& v, auto& a) {
 *       // Custom per-step processing
 *   });
 *
 *   sim.Run();
 * @endcode
 */

#ifndef SEM_FORWARD_SIMULATION_HPP
#define SEM_FORWARD_SIMULATION_HPP

#include "simulation/SimulationFacade.hpp"
#include "io/WavefieldWriter.hpp"
#include <memory>
#include <vector>

namespace SEM {

// =============================================================================
// ForwardSimulation - Forward modeling with wavefield output
// =============================================================================

/**
 * @class ForwardSimulation
 * @brief Forward simulation with integrated wavefield output
 * @tparam Dim Spatial dimension (2 or 3)
 *
 * Extends SimulationBase with:
 * - WavefieldWriter strategy for output
 * - Fluent interface for callbacks
 * - Configurable progress output
 */
template<int Dim>
class ForwardSimulation : public SimulationFacade<Dim> {
public:
    using Self = ForwardSimulation<Dim>;

    /**
     * @brief Construct forward simulation with MPI communicator
     * @param comm MPI communicator
     */
    explicit ForwardSimulation(MPI_Comm comm);

    /// Destructor
    ~ForwardSimulation() = default;

    // -------------------------------------------------------------------------
    // Fluent Configuration
    // -------------------------------------------------------------------------

    /**
     * @brief Set wavefield writer for output (single writer, backward compatible)
     * @param writer Wavefield writer instance (takes ownership)
     * @return Reference to this for chaining
     */
    Self& SetWavefieldWriter(std::unique_ptr<WavefieldWriter> writer);

    /**
     * @brief Add a wavefield writer to the output list
     * @param writer Wavefield writer instance (takes ownership)
     * @return Reference to this for chaining
     */
    Self& AddWavefieldWriter(std::unique_ptr<WavefieldWriter> writer);

    /**
     * @brief Set multiple wavefield writers at once
     * @param writers Vector of wavefield writer instances
     * @return Reference to this for chaining
     */
    Self& SetWavefieldWriters(std::vector<std::unique_ptr<WavefieldWriter>> writers);

    /**
     * @brief Enable/configure progress output
     * @param interval Progress output interval in steps (0 to disable)
     * @return Reference to this for chaining
     */
    Self& EnableProgressOutput(int interval = 100);


    // -------------------------------------------------------------------------
    // Execution
    // -------------------------------------------------------------------------

    /**
     * @brief Run forward simulation
     *
     * Overrides base Run() to integrate wavefield output:
     * 1. Initialize wavefield writer if set
     * 2. Time stepping loop with callbacks
     * 3. Wavefield output at specified intervals
     * 4. Finalize wavefield writer
     */
    void Run();

private:
    std::vector<std::unique_ptr<WavefieldWriter>> wavefield_writers_;
    bool progress_enabled_ = true;
    int progress_interval_ = 100;
    std::string log_file_;  ///< Path to log file (empty = no file logging)

};

// =============================================================================
// Type Aliases
// =============================================================================

using ForwardSimulation2D = ForwardSimulation<2>;
using ForwardSimulation3D = ForwardSimulation<3>;

}  // namespace SEM

#endif  // SEM_FORWARD_SIMULATION_HPP
