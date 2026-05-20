/**
 * @file SimulationFacade.hpp
 * @brief Unified simulation interface composing all components
 *
 * SimulationFacade is the main entry point for simulations. It composes:
 * - SimulationComponents: mesh, FE spaces, state vectors
 * - SimulationRunner: time integration loop
 * - SimulationIO: output operations
 *
 * This class provides backward compatibility with the original SimulationBase API.
 *
 * Example usage:
 * @code
 *   SimulationFacade<3> sim(MPI_COMM_WORLD);
 *   sim.LoadConfig("config.yaml")
 *      .SetupFromConfig();
 *   sim.Run();
 *   sim.SaveReceivers("output/");
 * @endcode
 */

#ifndef SEM_SIMULATION_FACADE_HPP
#define SEM_SIMULATION_FACADE_HPP

#include <mfem.hpp>
#include <memory>
#include <functional>
#include <string>

#include "simulation/SimulationComponents.hpp"
#include "simulation/SimulationRunner.hpp"
#include "simulation/SimulationIO.hpp"
#include "util/MemoryReport.hpp"

#include "operator/OperatorFactory.hpp"
#include "material/Material.hpp"
#include "operator/Operator.hpp"
#include "timeinteg/TimeIntegrator.hpp"
#include "srcrecv/Source.hpp"
#include "srcrecv/Receiver.hpp"
#include "config/YamlConfig.hpp"

namespace SEM {


using namespace mfem;

// Forward declarations for param structs
struct DampingParams;
struct ViscoParams;

/**
 * @struct TimingInfo
 * @brief Timing information for simulation performance reporting
 */
struct TimingInfo {
    double setup_time = 0.0;    ///< Setup phase duration (s)
    double run_time = 0.0;      ///< Simulation run duration (s)
    double io_time = 0.0;       ///< I/O phase duration (s)
    double total_time = 0.0;    ///< Total wall time (s)
};

/**
 * @struct DampingParams
 * @brief Parameters for absorbing boundary conditions
 */
struct DampingParams {
    real_t x_length = 0.0;
    real_t y_length = 0.0;
    real_t z_length = 0.0;
    real_t x_alpha = 0.0;
    real_t y_alpha = 0.0;
    real_t z_alpha = 0.0;
    std::vector<int> attrs;

    bool IsEnabled() const {
        return x_length > 0.0 || y_length > 0.0 || z_length > 0.0;
    }
};

/**
 * @struct ViscoParams
 * @brief Parameters for viscoelastic attenuation
 */
struct ViscoParams {
    bool enabled = false;
    real_t f0 = 1.0;
    int n_units = 3;
    std::string qkappa_file;
    std::string qmu_file;
};

/**
 * @class SimulationFacade
 * @brief Unified simulation interface composing all components
 * @tparam Dim Spatial dimension (2 or 3)
 *
 * This class provides the complete simulation API by composing:
 * - SimulationComponents: mesh, FES, state vectors
 * - SimulationRunner: time integration
 * - SimulationIO: output operations
 *
 * It maintains ownership of physics objects (material, operator, sources, etc.)
 * and passes non-owning pointers to components as needed.
 */
template<int Dim>
class SimulationFacade {
public:
    using DerivedType = SimulationFacade<Dim>;

    /**
     * @brief Construct simulation with MPI communicator
     * @param comm MPI communicator
     */
    explicit SimulationFacade(MPI_Comm comm);

    /// Destructor
    ~SimulationFacade();

    // Prevent copying
    SimulationFacade(const SimulationFacade&) = delete;
    SimulationFacade& operator=(const SimulationFacade&) = delete;

    // Allow moving
    SimulationFacade(SimulationFacade&&) = default;
    SimulationFacade& operator=(SimulationFacade&&) = default;

    // =========================================================================
    // Configuration (fluent interface)
    // =========================================================================

    /**
     * @brief Load configuration from YAML file
     * @param yaml_file Path to YAML configuration file
     * @return Reference to this for chaining
     */
    DerivedType& LoadConfig(const std::string& yaml_file);

    /**
     * @brief Load configuration from YamlConfig object
     * @param config YamlConfig instance (takes ownership)
     * @return Reference to this for chaining
     */
    DerivedType& LoadConfig(std::unique_ptr<YamlConfig> config);

    // =========================================================================
    // Setup methods (fluent interface)
    // =========================================================================

    /// Setup mesh from loaded config
    DerivedType& SetupMeshFromConfig();

    /// Setup material from loaded config
    DerivedType& SetupMaterialFromConfig();

    /// Setup operator from loaded config
    DerivedType& SetupOperatorFromConfig();

    /// Setup sources from loaded config
    DerivedType& SetupSourcesFromConfig();

    /// Setup receivers from loaded config
    DerivedType& SetupReceiversFromConfig();

    /**
     * @brief Setup all components from loaded config
     *
     * Calls SetupMeshFromConfig, SetupMaterialFromConfig, SetupOperatorFromConfig,
     * SetupSourcesFromConfig, SetupReceiversFromConfig in order.
     */
    DerivedType& SetupFromConfig();

    // =========================================================================
    // Execution
    // =========================================================================

    /// Initialize simulation
    void Initialize();

    /**
     * @brief Initialize device memory for all components
     *
     * Call after SetupFromConfig() to enable and sync device memory for:
     * - Solution vectors (u, v, a)
     * - Wave operator (integrators, geometry, DOFs, mass/damping)
     * - Sources
     * - Receivers
     */
    void DeviceInit();

    /// Run single time step, returns true if more steps remain
    bool Step();

    /// Reset simulation to initial state
    void Reset();

    // =========================================================================
    // Callbacks
    // =========================================================================

    /// Set per-step callback
    void SetStepCallback(StepCallback callback);

    /// Set progress callback with interval
    void SetProgressCallback(ProgressCallback callback, int interval = 100);

    // =========================================================================
    // Output
    // =========================================================================

    /// Print simulation summary
    void PrintSummary(std::ostream& os = std::cout) const;

    /**
     * @brief Write full simulation summary to file with timing info
     * @param filepath Output file path
     * @param timing Timing information from the run
     *
     * Same as WriteSummaryToFile() but also includes performance metrics.
     */
    void WriteSummaryToFile(const std::string& filepath, const TimingInfo& timing);

    // =========================================================================
    // State Access (delegated to components)
    // =========================================================================

    /// Get current simulation state
    const SimulationState& State() const { return runner_.State(); }

    /// Get displacement field
    ParGridFunction& Displacement() { return components_.U(); }
    const ParGridFunction& Displacement() const { return components_.U(); }

    /// Get velocity field
    ParGridFunction& Velocity() { return components_.V(); }
    const ParGridFunction& Velocity() const { return components_.V(); }

    /// Get acceleration field
    ParGridFunction& Acceleration() { return components_.A(); }
    const ParGridFunction& Acceleration() const { return components_.A(); }

    /// Get mesh
    ParMesh& Mesh() { return components_.Mesh(); }
    const ParMesh& Mesh() const { return components_.Mesh(); }

    /// Get finite element space
    ParFiniteElementSpace& FESpace() { return components_.FES(); }
    const ParFiniteElementSpace& FESpace() const { return components_.FES(); }

    /// Get scalar finite element space
    ParFiniteElementSpace& FESpaceScalar() { return components_.FESScalar(); }
    const ParFiniteElementSpace& FESpaceScalar() const { return components_.FESScalar(); }

    /// Get wave operator
    WaveOperator& Operator() { return *op_; }
    const WaveOperator& Operator() const { return *op_; }

    /// Get time integrator
    ExplicitTimeIntegrator& Integrator() { return *integrator_; }
    const ExplicitTimeIntegrator& Integrator() const { return *integrator_; }

    /// Get source collection
    PointSourceCollection& Sources() { return *sources_; }
    const PointSourceCollection& Sources() const { return *sources_; }

    /// Get receiver array
    ReceiverArray& Receivers() { return *receivers_; }
    const ReceiverArray& Receivers() const { return *receivers_; }

    /// Get material
    MaterialBase& Material() { return *material_; }
    const MaterialBase& Material() const { return *material_; }

    /// Get configuration
    const YamlConfig& Config() const { return *config_; }

    // =========================================================================
    // CFL Condition Check
    // =========================================================================

    /**
     * @brief Check CFL condition and warn/abort if violated
     *
     * Computes: dt_cfl = cfl_factor * h_min / (N^2 * v_max)
     * where:
     * - h_min = minimum element size (from mesh)
     * - N = polynomial order
     * - v_max = maximum wave velocity (from material)
     *
     * @param cfl_factor CFL safety factor (0 < cfl_factor <= 0.7), required from config
     * @param abort_on_violation If true, MFEM_ABORT on violation; else warn only
     * @return true if CFL is satisfied, false if violated
     */
    bool CheckCFL(real_t cfl_factor, bool abort_on_violation = true);

    /**
     * @brief Get CFL-limited time step
     *
     * @param cfl_factor CFL safety factor (required from config)
     * @return Maximum stable time step
     */
    real_t GetCFLTimeStep(real_t cfl_factor);

    // =========================================================================
    // Wavelength Sampling Check
    // =========================================================================

    /**
     * @brief Check if mesh resolution is sufficient for the source frequency
     *
     * Checks: lambda_min / h_max >= PPW (points per wavelength)
     * where:
     * - lambda_min = v_min / f_max (minimum wavelength)
     * - h_max = maximum element size
     * - PPW = required points per wavelength (typically 5 for N=4)
     *
     * Uses per-element evaluation: computes worst-case ratio
     * across all elements, then takes global max via MPI.
     *
     * @param f_max Maximum source frequency (Hz)
     * @param ppw_required Required points per wavelength (default 5.0)
     * @param warn_on_insufficient If true, print warning if insufficient
     * @return true if resolution is sufficient, false otherwise
     */
    bool CheckWavelengthSampling(real_t f_max, real_t ppw_required = 5.0,
                                  bool warn_on_insufficient = true);

    // =========================================================================
    // Time/Order Parameters
    // =========================================================================

    /// Get time step
    real_t Dt() const { return runner_.Dt(); }

    /// Get simulation start time t0
    real_t T0() const { return config_ ? config_->GetT0() : 0.0; }

    /// Get number of time steps
    int NumSteps() const { return runner_.NumSteps(); }

    /// Get polynomial order
    int Order() const { return components_.Order(); }

    // =========================================================================
    // MPI
    // =========================================================================

    /// Get MPI communicator
    MPI_Comm Comm() const { return comm_; }

    /// Get MPI rank
    int Rank() const { return rank_; }

    /// Check if root process
    bool IsRoot() const { return rank_ == 0; }

    // =========================================================================
    // Convenience Accessors (from config)
    // =========================================================================

    std::string OutputDir() const {
        return config_ ? config_->GetOutputDirectory() : "";
    }

    int LogInterval() const {
        return config_ ? config_->GetLogInterval() : 99999;
    }

    bool IsWavefieldOutputEnabled() const {
        return config_ ? config_->IsWavefieldOutputEnabled() : false;
    }

    int WavefieldInterval() const {
        return config_ ? config_->GetWavefieldInterval() : 100;
    }

    std::vector<std::string> ReceiverOutputFormats() const {
        return config_ ? config_->GetReceiverOutputFormats()
                       : std::vector<std::string>{"hdf5"};
    }

    std::string ReceiverOutputFilename() const {
        return config_ ? config_->GetReceiverOutputFilename() : "seismograms";
    }

    std::string DeviceString() const {
        return config_ ? config_->GetDevice() : "cpu";
    }

    std::string SummaryFile() const {
        if (!config_) return "";
        std::string summary = config_->GetSummaryFile();
        if (summary.empty()) return "";
        // If path is not absolute, prepend output directory
        if (summary[0] != '/') {
            std::string outdir = config_->GetOutputDirectory();
            if (!outdir.empty() && outdir.back() != '/') {
                outdir += '/';
            }
            return outdir + summary;
        }
        return summary;
    }

    int NumSources() const {
        return sources_ ? sources_->NumSources() : 0;
    }

    int NumReceivers() const {
        return receivers_ ? receivers_->NumReceivers() : 0;
    }

    std::string Name() const {
        return config_ ? config_->GetName() : "";
    }

    // =========================================================================
    // Component Access (for advanced use)
    // =========================================================================

    SimulationComponents<Dim>& Components() { return components_; }
    const SimulationComponents<Dim>& Components() const { return components_; }

    SimulationRunner<Dim>& Runner() { return runner_; }
    const SimulationRunner<Dim>& Runner() const { return runner_; }

    SimulationIO<Dim>& IO() { return io_; }
    const SimulationIO<Dim>& IO() const { return io_; }

protected:
    // Internal helpers (implemented in SimulationFacadeSetup.cpp)
    void RecordReceivers();

    // MPI
    MPI_Comm comm_;
    int rank_;
    int num_procs_;

    // Components (owned, composed)
    SimulationComponents<Dim> components_;
    SimulationRunner<Dim> runner_;
    SimulationIO<Dim> io_;

    // Configuration (owned)
    std::unique_ptr<YamlConfig> config_;

    // Physics objects (owned by facade)
    std::unique_ptr<MaterialBase> material_;
    std::unique_ptr<WaveOperator> op_;
    std::unique_ptr<ExplicitTimeIntegrator> integrator_;
    std::unique_ptr<PointSourceCollection> sources_;
    std::unique_ptr<ReceiverArray> receivers_;

    // Damping and viscoelasticity params
    DampingParams damping_;
    ViscoParams visco_;

    // Setup flags
    bool mesh_ready_ = false;
    bool material_ready_ = false;
    bool operator_ready_ = false;
    bool sources_ready_ = false;
    bool receivers_ready_ = false;

    // Memory report (records usage at checkpoints)
    MemoryReport memory_report_;
};

// =============================================================================
// Type Aliases
// =============================================================================

using SimulationFacade2D = SimulationFacade<2>;
using SimulationFacade3D = SimulationFacade<3>;

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Estimate CFL time step
 * @param h_min Minimum element size
 * @param v_max Maximum velocity
 * @param order Polynomial order
 * @param safety Safety factor (default 0.3)
 * @return Estimated stable time step
 */
inline real_t EstimateCFL(real_t h_min, real_t v_max, int order, real_t safety = 0.3) {
    return safety * h_min / (order * v_max);
}

}  // namespace SEM

#endif  // SEM_SIMULATION_FACADE_HPP
