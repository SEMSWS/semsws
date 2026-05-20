/**
 * @file SimulationFacade.cpp
 * @brief Core implementation of SimulationFacade template
 *
 * Contains: constructor, destructor, basic execution methods
 *
 * Setup methods are in SimulationFacadeSetup.cpp
 */

#include "simulation/SimulationFacade.hpp"
#include "util/CFLReport.hpp"
#include "util/ElementMetrics.hpp"

#include <limits>
#include <iomanip>
#include <fstream>
#include <sstream>

// For flush-to-zero (FTZ) and denormals-are-zero (DAZ) mode
#ifdef __x86_64__
#include <xmmintrin.h>
#include <pmmintrin.h>
#endif

using mfem::real_t;
using mfem::ParMesh;
using mfem::ElementTransformation;
using mfem::IntegrationRule;
using mfem::IntegrationRules;
using mfem::IntRules;
using mfem::Quadrature1D;
using mfem::DenseMatrix;

// =============================================================================
// Anonymous namespace for helper functions
// =============================================================================

namespace {

// Compute maximum element size (per process)
// type=2 returns h_max (maximum singular value of Jacobian)
real_t ComputeLocalHMax(ParMesh& mesh) {
    int ne = mesh.GetNE();
    real_t h_max_local = 0.0;
    for (int e = 0; e < ne; e++) {
        real_t h = mesh.GetElementSize(e, 2);  // type=2 for h_max
        if (h > h_max_local) h_max_local = h;
    }
    return h_max_local;
}

// (ComputeElementMinGLLDistance moved to include/util/ElementMetrics.hpp
// so CoupledSimulationFacade's CFL reporter can share the same routine.)

}  // anonymous namespace

namespace SEM {

// =============================================================================
// Constructor / Destructor
// =============================================================================

template<int Dim>
SimulationFacade<Dim>::SimulationFacade(MPI_Comm comm)
    : comm_(comm),
      components_(comm),
      runner_(comm),
      io_(comm)
{
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &num_procs_);

    // Enable flush-to-zero (FTZ) and denormals-are-zero (DAZ) mode
    // This prevents massive slowdown when computing with very small numbers
#ifdef __x86_64__
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
}

template<int Dim>
SimulationFacade<Dim>::~SimulationFacade() = default;

// =============================================================================
// Execution Methods
// =============================================================================

template<int Dim>
void SimulationFacade<Dim>::Initialize() {
    MFEM_VERIFY(mesh_ready_, "Mesh must be set up before initialization");
    MFEM_VERIFY(material_ready_, "Material must be set up before initialization");
    MFEM_VERIFY(operator_ready_, "Operator must be set up before initialization");

    // Setup runner dependencies
    runner_.SetOperator(op_.get());
    runner_.SetIntegrator(integrator_.get());
    if (config_) runner_.SetT0(config_->GetT0());

    // Initialize via runner
    runner_.Initialize(components_.U(), components_.V(), components_.A());
}

template<int Dim>
bool SimulationFacade<Dim>::Step() {
    // Record receivers at current step (0-based) BEFORE time integration
    RecordReceivers();

    // Time integration and step increment
    bool more = runner_.Step(components_.U(), components_.V(), components_.A());

    return more;
}

template<int Dim>
void SimulationFacade<Dim>::Reset() {
    runner_.Reset(components_.U(), components_.V(), components_.A(), op_.get());

    if (receivers_) {
        receivers_->ResetData();
    }
}

// =============================================================================
// Callbacks
// =============================================================================

template<int Dim>
void SimulationFacade<Dim>::SetStepCallback(StepCallback callback) {
    runner_.SetStepCallback(std::move(callback));
}

template<int Dim>
void SimulationFacade<Dim>::SetProgressCallback(ProgressCallback callback, int interval) {
    runner_.SetProgressCallback(std::move(callback), interval);
}

// =============================================================================
// Output Methods
// =============================================================================

template<int Dim>
void SimulationFacade<Dim>::PrintSummary(std::ostream& os) const {
    io_.PrintSummary(os, components_, material_.get(),
                     runner_.Dt(), runner_.NumSteps(), num_procs_,
                     sources_.get(), receivers_.get());

    // CFL and resolution info (MPI collective operations — all ranks must participate)
    real_t dt_cfl = 0.0;
    real_t ppw_achieved = 0.0;
    real_t cfl_ratio = 0.0;
    real_t f_source = 0.0;
    bool has_cfl_info = false;

    if (mesh_ready_ && material_ready_ && config_) {
        // const_cast needed: MFEM GetElementSize/GetElementTransformation are
        // non-const but logically read-only
        auto& mesh = const_cast<ParMesh&>(components_.Mesh());
        int N = components_.Order();
        real_t cfl_factor = config_->GetCflFactor();
        real_t min_dt_local = std::numeric_limits<real_t>::max();
        for (int e = 0; e < mesh.GetNE(); e++) {
            real_t dist_min = ComputeElementMinGLLDistance<Dim>(mesh, e, N);
            real_t v_elem = material_->GetElementMaxVelocity(e);
            real_t dt_max_elem = cfl_factor * dist_min / v_elem;
            if (dt_max_elem < min_dt_local) min_dt_local = dt_max_elem;
        }
        MPI_Allreduce(&min_dt_local, &dt_cfl, 1, MFEM_MPI_REAL_T, MPI_MIN, comm_);
        cfl_ratio = runner_.Dt() / dt_cfl;

        auto all_sources = config_->GetAllSources();
        for (const auto& src : all_sources) {
            if (src.frequency > f_source) f_source = src.frequency;
        }
        if (f_source > 0) {
            int ngll = N + 1;
            real_t min_ppw_local = std::numeric_limits<real_t>::max();
            for (int e = 0; e < mesh.GetNE(); e++) {
                real_t h = mesh.GetElementSize(e, 2);
                real_t v_elem = material_->GetElementMinVelocity(e);
                real_t ppw_elem = v_elem * ngll / (f_source * h);
                if (ppw_elem < min_ppw_local) min_ppw_local = ppw_elem;
            }
            MPI_Allreduce(&min_ppw_local, &ppw_achieved, 1,
                          MFEM_MPI_REAL_T, MPI_MIN, comm_);
        }
        has_cfl_info = true;
    }

    // Only root prints
    if (rank_ == 0 && has_cfl_info) {
        os << "Numerical stability:\n";
        os << std::scientific << std::setprecision(4);
        os << "  CFL dt_max: " << dt_cfl << " s"
           << " (dt/dt_max = " << std::fixed << std::setprecision(2)
           << cfl_ratio << ")\n";
        if (ppw_achieved > 0) {
            os << "  PPW: " << std::fixed << std::setprecision(1)
               << ppw_achieved << " (f=" << f_source << " Hz)\n";
        }
        os << std::defaultfloat;
        os << "\n";
    }

    // Print memory report (MPI collective operation)
    memory_report_.PrintSummary(os, comm_);
}


template<int Dim>
void SimulationFacade<Dim>::WriteSummaryToFile(const std::string& filepath,
                                                const TimingInfo& timing) {
    // Gather MPI collective data BEFORE the IsRoot() check
    // GetGlobalNE() and GlobalTrueVSize() are MPI collectives - all ranks must call
    long long global_ne = 0;
    if (components_.HasMesh()) {
        global_ne = components_.Mesh().GetGlobalNE();
    }
    HYPRE_BigInt global_dofs = 0;
    if (components_.HasFESpaces()) {
        global_dofs = components_.FES().GlobalTrueVSize();
    }

    // Compute CFL and Resolution info (MPI collective operations)
    real_t h_min = 0.0, h_max = 0.0;
    real_t v_max = 0.0, v_min = 0.0;
    real_t dt_cfl = 0.0;
    real_t ppw_achieved = 0.0;
    real_t f_source = 0.0;
    bool has_cfl_info = false;

    if (mesh_ready_ && material_ready_ && config_) {
        // Get h_min and h_max (MPI collective)
        real_t h_min_local = std::numeric_limits<real_t>::max();
        real_t h_max_local = 0.0;
        for (int e = 0; e < components_.Mesh().GetNE(); e++) {
            real_t h = components_.Mesh().GetElementSize(e, 1);
            if (h < h_min_local) h_min_local = h;
            if (h > h_max_local) h_max_local = h;
        }
        MPI_Allreduce(&h_min_local, &h_min, 1, MFEM_MPI_REAL_T, MPI_MIN, comm_);
        MPI_Allreduce(&h_max_local, &h_max, 1, MFEM_MPI_REAL_T, MPI_MAX, comm_);

        // Get velocities (MPI collective)
        real_t v_max_local = material_->GetMaxVelocity();
        real_t v_min_local = material_->GetMinVelocity();
        MPI_Allreduce(&v_max_local, &v_max, 1, MFEM_MPI_REAL_T, MPI_MAX, comm_);
        MPI_Allreduce(&v_min_local, &v_min, 1, MFEM_MPI_REAL_T, MPI_MIN, comm_);

        // CFL: Per-element evaluation using actual GLL point distances
        // dt_max = CFL * min_gll_dist / v_elem
        // Find minimum dt_max across all elements (most restrictive)
        int N = components_.Order();
        real_t cfl_factor = config_->GetCflFactor();
        real_t min_dt_local = std::numeric_limits<real_t>::max();
        for (int e = 0; e < components_.Mesh().GetNE(); e++) {
            real_t dist_min = ComputeElementMinGLLDistance<Dim>(components_.Mesh(), e, N);
            real_t v_elem = material_->GetElementMaxVelocity(e);
            real_t dt_max_elem = cfl_factor * dist_min / v_elem;
            if (dt_max_elem < min_dt_local) min_dt_local = dt_max_elem;
        }
        MPI_Allreduce(&min_dt_local, &dt_cfl, 1, MFEM_MPI_REAL_T, MPI_MIN, comm_);

        // Resolution: Per-element PPW = v_elem * NGLL / (f * h_elem)
        // Find minimum PPW across all elements
        auto all_sources = config_->GetAllSources();
        for (const auto& src : all_sources) {
            if (src.frequency > f_source) {
                f_source = src.frequency;
            }
        }
        if (f_source > 0) {
            int ngll = N + 1;
            real_t min_ppw_local = std::numeric_limits<real_t>::max();
            for (int e = 0; e < components_.Mesh().GetNE(); e++) {
                real_t h = components_.Mesh().GetElementSize(e, 2);
                real_t v_elem = material_->GetElementMinVelocity(e);
                real_t ppw_elem = v_elem * ngll / (f_source * h);
                if (ppw_elem < min_ppw_local) min_ppw_local = ppw_elem;
            }
            MPI_Allreduce(&min_ppw_local, &ppw_achieved, 1, MFEM_MPI_REAL_T, MPI_MIN, comm_);
        }

        has_cfl_info = true;
    }

    // Memory report (MPI collective - must call before IsRoot check)
    std::ostringstream mem_buffer;
    memory_report_.PrintSummary(mem_buffer, comm_);

    // Only root process writes the file
    if (!IsRoot()) return;

    std::ofstream ofs(filepath);
    if (!ofs.is_open()) {
        std::cerr << "[Warning] Could not open summary file: " << filepath << "\n";
        return;
    }

    // Write simulation name if available
    if (config_) {
        ofs << "# Simulation: " << config_->GetName() << "\n";
    }
    ofs << "# Generated by SEMSWS\n\n";

    // PrintSummary content (without MPI collective parts - root only)
    ofs << "========================================\n";
    ofs << "        SEMSWS " << Dim << "D Simulation\n";
    ofs << "========================================\n\n";

    // Material info
    if (material_) {
        ofs << "Material: " << MaterialTypeToString(material_->GetType()) << "\n";
        ofs << std::fixed << std::setprecision(2);
        ofs << "  Vmax: " << material_->GetMaxVelocity() << " m/s, ";
        ofs << "Vmin: " << material_->GetMinVelocity() << " m/s\n";
        if (material_->HasAttenuation()) {
            ofs << "  Attenuation: enabled (f0=" << material_->AttenuationF0()
                << " Hz, " << material_->AttenuationNumUnits() << " SLS units)\n";
        } else {
            ofs << "  Attenuation: disabled\n";
        }
        ofs << std::defaultfloat;
    }

    // Discretization
    ofs << "\nDiscretization:\n";
    ofs << "  Order: " << components_.Order() << "\n";
    if (global_ne > 0) {
        ofs << "  Elements: " << global_ne << "\n";
    }
    if (global_dofs > 0) {
        ofs << "  DOFs: " << global_dofs << "\n";
    }

    // Sources
    if (sources_ && sources_->NumSources() > 0) {
        ofs << "\nSources: " << sources_->NumSources() << " total\n";
    }

    // Receivers
    if (receivers_ && receivers_->NumReceivers() > 0) {
        ofs << "\nReceivers: " << receivers_->NumReceivers() << " total\n";
        const auto& all_receivers = receivers_->GetAllReceivers();
        std::vector<std::pair<std::string, int>> type_counts;
        for (const auto& entry : all_receivers) {
            int count = static_cast<int>(entry.second.size());
            if (count > 0) {
                type_counts.push_back({ReceiverTypeToString(entry.first), count});
            }
        }
        if (!type_counts.empty()) {
            ofs << "  Types: ";
            for (size_t i = 0; i < type_counts.size(); ++i) {
                if (i > 0) ofs << ", ";
                ofs << type_counts[i].second << " " << type_counts[i].first;
            }
            ofs << "\n";
        }
    }

    // Operator info
    if (op_) {
        ofs << "\nOperator:\n";
        std::ostringstream op_oss;
        op_->PrintInfo(op_oss);
        // Indent each line
        std::string line;
        std::istringstream op_iss(op_oss.str());
        while (std::getline(op_iss, line)) {
            ofs << "  " << line << "\n";
        }
    }

    // Time stepping
    ofs << "\nTime stepping:\n";
    ofs << "  dt: " << runner_.Dt() << " s\n";
    ofs << "  nt: " << runner_.NumSteps() << "\n";
    ofs << "  Simulation time: " << runner_.Dt() * runner_.NumSteps() << " s\n";

    // CFL and Resolution
    if (has_cfl_info) {
        ofs << "\nNumerical stability:\n";
        ofs << std::fixed << std::setprecision(2);
        ofs << "  h_min: " << h_min << " m, h_max: " << h_max << " m\n";
        ofs << "  CFL dt_max: " << std::scientific << std::setprecision(4) << dt_cfl << " s";
        real_t cfl_ratio = runner_.Dt() / dt_cfl;
        ofs << " (dt/dt_max = " << std::fixed << std::setprecision(2) << cfl_ratio << ")\n";
        if (ppw_achieved > 0) {
            ofs << "  PPW: " << std::fixed << std::setprecision(1) << ppw_achieved
                << " (f=" << f_source << " Hz)\n";
        }
        ofs << std::defaultfloat;
    }

    // Parallel configuration
    ofs << "\nParallel configuration:\n";
    ofs << "  Device: " << DeviceString() << "\n";
    ofs << "  MPI processes: " << num_procs_ << "\n";

    // Performance metrics
    ofs << "\nPerformance:\n";
    ofs << std::fixed << std::setprecision(4);
    ofs << "  Setup time:  " << timing.setup_time << " s\n";
    ofs << "  Run time:    " << timing.run_time << " s\n";
    ofs << "  IO time:     " << timing.io_time << " s\n";
    ofs << "  Total time:  " << timing.total_time << " s\n";
    if (timing.run_time > 0) {
        int nt = runner_.NumSteps();
        ofs << "  Steps/sec:   " << std::fixed << std::setprecision(4)
            << (nt / timing.run_time) << "\n";
        ofs << "  Time/step:   " << std::fixed << std::setprecision(4)
            << (timing.run_time / nt * 1000.0) << " ms\n";
        // DOFs/sec = (total DOFs * num_steps) / run_time
        double dofs_per_sec = static_cast<double>(global_dofs) * nt / timing.run_time;
        ofs << "  DOFs/sec:    " << std::scientific << std::setprecision(2)
            << dofs_per_sec << "\n";
    }
    ofs << std::defaultfloat;

    // Memory report
    ofs << mem_buffer.str();

    ofs << "========================================\n";

    ofs.close();
}

// =============================================================================
// CFL Condition Check
// =============================================================================

template<int Dim>
real_t SimulationFacade<Dim>::GetCFLTimeStep(real_t cfl_factor) {
    MFEM_VERIFY(mesh_ready_, "Mesh must be set up for CFL calculation");
    MFEM_VERIFY(material_ready_, "Material must be set up for CFL calculation");

    // Per-element CFL evaluation using actual GLL point distances 
    // dt_max = CFL * min_gll_dist / v_elem
    auto& mesh = components_.Mesh();
    int ne = mesh.GetNE();
    int N = components_.Order();

    real_t min_dt_local = std::numeric_limits<real_t>::max();
    for (int e = 0; e < ne; e++) {
        real_t dist_min = ComputeElementMinGLLDistance<Dim>(mesh, e, N);
        real_t v_elem = material_->GetElementMaxVelocity(e);
        real_t dt_max_elem = cfl_factor * dist_min / v_elem;
        if (dt_max_elem < min_dt_local) {
            min_dt_local = dt_max_elem;
        }
    }

    // Get global minimum dt_max
    real_t dt_cfl;
    MPI_Allreduce(&min_dt_local, &dt_cfl, 1, MFEM_MPI_REAL_T, MPI_MIN, comm_);

    return dt_cfl;
}

template<int Dim>
bool SimulationFacade<Dim>::CheckCFL(real_t cfl_factor, bool abort_on_violation) {
    // Delegate to the shared helper used by CoupledSimulationFacade too
    // (include/util/CFLReport.hpp). Label left empty for single-physics
    // runs so the stderr tag stays "[CFL]" exactly as before.
    return CheckCFLOnSubmesh<Dim>(
        components_.Mesh(),
        components_.Order(),
        *material_,
        runner_.Dt(),
        cfl_factor,
        comm_,
        /*label=*/"",
        abort_on_violation);
}

// =============================================================================
// Wavelength Sampling Check
// =============================================================================

template<int Dim>
bool SimulationFacade<Dim>::CheckWavelengthSampling(real_t f_max, real_t ppw_required,
                                                      bool warn_on_insufficient) {
    MFEM_VERIFY(mesh_ready_, "Mesh must be set up for wavelength sampling check");
    MFEM_VERIFY(material_ready_, "Material must be set up for wavelength sampling check");

    // Per-element PPW check using per-element minimum velocity
    auto& mesh = components_.Mesh();
    int ne = mesh.GetNE();
    int N = components_.Order();
    int ngll = N + 1;

    int local_insufficient_count = 0;
    real_t worst_ppw_local = std::numeric_limits<real_t>::max();
    int worst_element_local = -1;
    real_t worst_h_local = 0.0;
    real_t worst_v_local = 0.0;

    for (int e = 0; e < ne; e++) {
        real_t h = mesh.GetElementSize(e, 2);  // type=2 for h_max of element

        // Get minimum velocity for this specific element
        real_t v_elem = material_->GetElementMinVelocity(e);
        real_t lambda_elem = v_elem / f_max;

        // PPW for this element using element-local velocity
        real_t ppw_elem = ngll * lambda_elem / h;

        if (ppw_elem < ppw_required) {
            local_insufficient_count++;
            if (ppw_elem < worst_ppw_local) {
                worst_ppw_local = ppw_elem;
                worst_element_local = e;
                worst_h_local = h;
                worst_v_local = v_elem;
            }
        }
    }

    // Gather global statistics
    int global_insufficient_count = 0;
    MPI_Allreduce(&local_insufficient_count, &global_insufficient_count, 1, MPI_INT, MPI_SUM, comm_);

    bool sufficient = (global_insufficient_count == 0);

    if (!sufficient && warn_on_insufficient) {
        // Find globally worst PPW
        struct { double ppw; int rank; } local_worst, global_worst;
        local_worst.ppw = (worst_ppw_local < std::numeric_limits<real_t>::max())
                          ? worst_ppw_local : 1e30;
        int myrank;
        MPI_Comm_rank(comm_, &myrank);
        local_worst.rank = myrank;
        MPI_Allreduce(&local_worst, &global_worst, 1, MPI_DOUBLE_INT, MPI_MINLOC, comm_);

        if (IsRoot()) {
            std::cerr << "[Resolution] WARNING: Wavelength sampling insufficient!\n";
            std::cerr << "[Resolution]   Elements with PPW < " << ppw_required << ": "
                      << global_insufficient_count << "\n";
            std::cerr << "[Resolution]   Consider: refining mesh or lowering source frequency.\n";
        }

        // Report worst element details from the rank that has it
        if (myrank == global_worst.rank && worst_element_local >= 0) {
            // Get centroid of worst element
            Vector centroid(Dim);
            mesh.GetElementCenter(worst_element_local, centroid);
            std::cerr << "[Resolution]   Worst element: local_id=" << worst_element_local
                      << ", PPW=" << worst_ppw_local
                      << ", h=" << worst_h_local << "m"
                      << ", v_min=" << worst_v_local << "m/s"
                      << ", centroid=(" << centroid(0);
            if (Dim >= 2) std::cerr << ", " << centroid(1);
            if (Dim >= 3) std::cerr << ", " << centroid(2);
            std::cerr << ")\n";
        }
    }

    return sufficient;
}


// =============================================================================
// Internal Helpers
// =============================================================================

template<int Dim>
void SimulationFacade<Dim>::RecordReceivers() {
    if (receivers_ready_ && receivers_) {
        int buffer_steps = config_ ? config_->GetReceiverBufferSteps() : 0;
        receivers_->Record(runner_.CurrentStep(), buffer_steps);
    }
}

// =============================================================================
// Explicit Template Instantiations
// =============================================================================

template class SimulationFacade<2>;
template class SimulationFacade<3>;

}  // namespace SEM
