/**
 * @file test_forward_simulation.cpp
 * @brief Test executable for ForwardSimulation with detailed progress output
 *
 * Demonstrates:
 * - ForwardSimulation fluent interface (2D and 3D)
 * - Custom progress callback with wavefield statistics
 * - GLVisWavefieldWriter integration
 *
 * Usage:
 *   mpirun -np 4 ./test_forward_simulation -config config.yaml
 */

#include "simulation/ForwardSimulation.hpp"
#include "coupling/CoupledSimulationFacade.hpp"
#include "simulation/AdjointSimulation.hpp"
#include "integ/attenuation/AttenuationCoeffs.hpp"
#include "io/WavefieldWriter.hpp"
#include "io/WavefieldWriterFactory.hpp"
#include "io/MaterialWriter.hpp"
#include "config/ConfigTypes.hpp"
#include "util/Profiler.hpp"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <iomanip>
#include <fstream>

using namespace SEM;
using namespace mfem;

// =============================================================================
// Template function to run simulation
// =============================================================================

template<int Dim>
int RunSimulation(std::unique_ptr<YamlConfig> config) {
    double t_start_setup = MPI_Wtime();

    // Device must outlive sim: sim's GPU-backed members free memory in their
    // destructors, which the MFEM Device backend (and Umpire allocators) must
    // still own at that point. Declare device first → destroyed last.
    const std::string device_str =
        config->GetDevice().empty() ? std::string("cpu") : config->GetDevice();
    const bool use_gpu = (device_str.find("cuda") != std::string::npos ||
                          device_str.find("hip") != std::string::npos);

    int device_id = 0;
    if (use_gpu) {
        int num_devices = Device::GetDeviceCount();
        if (num_devices > 0) {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);

            MPI_Comm local_comm;
            MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                MPI_INFO_NULL, &local_comm);
            int local_rank;
            MPI_Comm_rank(local_comm, &local_rank);
            MPI_Comm_free(&local_comm);

            device_id = local_rank % num_devices;
        }
    }

    Device device(device_str, device_id);

    if (use_gpu) {
#ifdef SEM_USE_GPU_AWARE_MPI
        Device::SetGPUAwareMPI(true);
#endif
    }

    ForwardSimulation<Dim> sim(MPI_COMM_WORLD);
    sim.LoadConfig(std::move(config));

    // {
    //     int world_rank, world_size;
    //     MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    //     MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    //     for (int r = 0; r < world_size; ++r) {
    //         if (r == world_rank) {
    //             std::cout << "Rank " << world_rank
    //                       << " (device_id=" << device_id << "): ";
    //             device.Print();
    //         }
    //         MPI_Barrier(MPI_COMM_WORLD);
    //     }
    // }

    


    

    // Setup all components from config
    sim.SetupFromConfig();
    // Initialize device memory for GPU builds
    if (Device::Allows(Backend::DEVICE_MASK)) {
        sim.DeviceInit();
    }

    // Enable progress output using log_interval from config
    sim.EnableProgressOutput(sim.LogInterval());

    // Setup wavefield output if enabled in config
    if (sim.IsWavefieldOutputEnabled()) {
        auto wf_config = sim.Config().GetWavefieldOutputConfig();
        if (!wf_config.formats.empty()) {
            sim.SetWavefieldWriters(
                CreateWavefieldWriters(wf_config, sim.Config().GetVisDirectory()));
        }
    }

    // Material output if enabled in config
    auto mat_config = sim.Config().GetMaterialOutputConfig();
    if (mat_config.enabled) {
        MaterialWriter::Write(sim.Material(), sim.FESpace(),
                              mat_config, sim.Config().GetVisDirectory(),
                              MPI_COMM_WORLD);
    }



    double t_end_setup = MPI_Wtime();

    // Run simulation (receivers are saved internally by Run)
    double t_start = MPI_Wtime();
    sim.Run();
    if (Device::Allows(Backend::DEVICE_MASK)) {
        MFEM_DEVICE_SYNC;  // Wait for GPU to complete before measuring time
    }
    double t_end = MPI_Wtime();

    // Collect timing info
    TimingInfo timing;
    timing.setup_time = t_end_setup - t_start_setup;
    timing.run_time = t_end - t_start;
    timing.io_time = 0.0;
    timing.total_time = t_end - t_start_setup;

    // Always write summary file to output directory
    std::string summary_file = sim.SummaryFile();
    if (summary_file.empty()) {
        // Default: output_dir/summary.txt
        std::string outdir = sim.OutputDir();
        if (!outdir.empty()) {
            summary_file = outdir + "/summary.txt";
        }
    }
    if (!summary_file.empty()) {
        sim.WriteSummaryToFile(summary_file, timing);
    }

    // Write Q-approximation diagnostic if attenuation is enabled (rank 0 only)
    if (sim.Material().HasAttenuation()) {
        int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            std::string qfile = sim.OutputDir() + "/q_approximation.dat";
            SEM::WriteQApproximationFile(qfile,
                sim.Material().AttenuationNumUnits(),
                sim.Material().AttenuationQkappa(),
                sim.Material().AttenuationQmu(),
                sim.Material().AttenuationF0());
        }
    }

    return 0;
}

// =============================================================================
// Template function to run FWI inversion
// =============================================================================

template<int Dim>
int RunCoupled(std::unique_ptr<YamlConfig> config) {
    // Dispatch path for fluid-solid coupled material.type = "coupled".
    // The stateful machinery lives in CoupledSimulationFacade; all this
    // function does is drive it through LoadConfig → SetupFromConfig →
    // Initialize → Run → SaveReceivers and report timings. Kept silent
    // during normal setup so stdout/stderr stays as quiet as the pure
    // RunSimulation path — debugging prints belong in the facade itself
    // (gated by env var when needed).
    double t_start_setup = MPI_Wtime();

    // Device must outlive sim (see RunSimulation for full rationale).
    const std::string device_str =
        config->GetDevice().empty() ? std::string("cpu") : config->GetDevice();
    const bool use_gpu = (device_str.find("cuda") != std::string::npos ||
                          device_str.find("hip")  != std::string::npos);
    int device_id = 0;
    if (use_gpu) {
        const int num_devices = Device::GetDeviceCount();
        if (num_devices > 0) {
            int world_rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
            MPI_Comm local_comm;
            MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                                world_rank, MPI_INFO_NULL, &local_comm);
            int local_rank;
            MPI_Comm_rank(local_comm, &local_rank);
            MPI_Comm_free(&local_comm);
            device_id = local_rank % num_devices;
        }
    }
    Device device(device_str, device_id);
    if (use_gpu) {
#ifdef SEM_USE_GPU_AWARE_MPI
        Device::SetGPUAwareMPI(true);
#endif
        if (!Device::GetGPUAwareMPI() && Mpi::Root()) {
            std::cout << "WARNING: GPU enabled but MPI stack is not "
                         "GPU-aware; fluid-solid coupling will stage "
                         "interface exchange through host memory "
                         "(correct but slower)." << std::endl;
        }
    }

    CoupledSimulationFacade<Dim> sim(MPI_COMM_WORLD);
    sim.LoadConfig(std::move(config));

    sim.SetupFromConfig();

    if (Device::Allows(Backend::DEVICE_MASK)) {
        sim.DeviceInit();
    }

    // CFL sanity check on both sub-meshes. Warn (don't abort) so users on
    // marginal meshes see the number before blowing up in the time loop.
    sim.CheckCFL(sim.Config().GetCflFactor(), /*abort_on_violation=*/false);

    // Wavelength sampling check (points-per-wavelength) on each sub-mesh.
    // Uses mesh.max_freq / mesh.ppw from the YAML, same as the pure path.
    {
        const real_t f_max       = sim.Config().GetMaxFreq();
        const real_t ppw_required = sim.Config().GetPPW();
        sim.CheckWavelengthSampling(f_max, ppw_required, /*warn=*/true);
    }

    // Material output (vp/vs/rho, optional Q). Each submesh gets its
    // own `MaterialOutputConfig` via the per-side overload, so YAMLs
    // can declare `simulation.output.material.fluid: { fields: [vp] }`
    // and `material.solid: { fields: [vp, vs, rho] }` to customise
    // per-domain (or mute one side entirely with `enabled: false`).
    // Absent per-side blocks inherit the top-level defaults.
    {
        const std::string vis_dir = sim.Config().GetVisDirectory();
        auto fluid_cfg = sim.Config().GetMaterialOutputConfig("fluid");
        auto solid_cfg = sim.Config().GetMaterialOutputConfig("solid");
        if (fluid_cfg.enabled) {
            MaterialWriter::Write(sim.FluidMaterial(),
                                  sim.FluidComponents().FESScalar(),
                                  fluid_cfg, vis_dir + "/fluid",
                                  MPI_COMM_WORLD);
        }
        if (solid_cfg.enabled) {
            MaterialWriter::Write(sim.SolidMaterial(),
                                  sim.SolidComponents().FESScalar(),
                                  solid_cfg, vis_dir + "/solid",
                                  MPI_COMM_WORLD);
        }
    }

    // Enable per-step progress output (stdout + <output_dir>/log.txt) at
    // the log_interval cadence from the YAML, mirroring ForwardSimulation.
    sim.EnableProgressOutput(sim.LogInterval());

    // Wavefield output: route through the same factory as the pure side so
    // coupled runs honor every format (glvis / paraview / gmt) declared
    // under `output.wavefield.formats` in the YAML. One writer set per
    // submesh so fluid / solid files don't collide. The config is
    // fetched via the per-side overload so YAMLs can set distinct
    // interval / fields / formats under `output.wavefield.fluid:` and
    // `output.wavefield.solid:`; absent overrides inherit the
    // top-level values.
    if (sim.IsWavefieldOutputEnabled()) {
        const std::string vis_dir = sim.Config().GetVisDirectory();
        auto fluid_cfg = sim.Config().GetWavefieldOutputConfig("fluid");
        auto solid_cfg = sim.Config().GetWavefieldOutputConfig("solid");
        auto build = [&](const WavefieldOutputConfig& cfg,
                         const std::string& side) {
            std::vector<std::unique_ptr<WavefieldWriter>> ws;
            if (cfg.enabled && !cfg.formats.empty()) {
                ws = CreateWavefieldWriters(cfg, vis_dir + "/" + side);
            }
            return ws;
        };
        sim.SetFluidWavefieldWriters(build(fluid_cfg, "fluid"));
        sim.SetSolidWavefieldWriters(build(solid_cfg, "solid"));
    }

    sim.Initialize();
    double t_end_setup = MPI_Wtime();

    double t_start_run = MPI_Wtime();
    sim.Run();
    if (Device::Allows(Backend::DEVICE_MASK)) {
        MFEM_DEVICE_SYNC;  // Wait for GPU to finish before taking end time
    }
    double t_end_run = MPI_Wtime();

    double t_start_save = MPI_Wtime();
    sim.SaveReceivers();
    double t_end_save = MPI_Wtime();

    // Always write summary file mirroring the ForwardSimulation path.
    const std::string summary_path = sim.SummaryFile();
    if (!summary_path.empty()) {
        TimingInfo timing;
        timing.setup_time = t_end_setup - t_start_setup;
        timing.run_time   = t_end_run   - t_start_run;
        timing.io_time    = t_end_save  - t_start_save;
        timing.total_time = t_end_save  - t_start_setup;
        sim.WriteSummaryToFile(summary_path, timing);
    }

    // Q-approximation diagnostic (rank 0). Pure RunSimulation writes
    // q_approximation.dat under OutputDir when attenuation is enabled;
    // coupled has two independent materials so we emit one file per
    // domain that actually carries attenuation, in the corresponding
    // subdirectory (matching the fluid/ and solid/ wavefield layout).
    {
        int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            const std::string outdir = sim.OutputDir();
            if (sim.FluidMaterial().HasAttenuation()) {
                const auto& m = sim.FluidMaterial();
                SEM::WriteQApproximationFile(
                    outdir + "/fluid/q_approximation.dat",
                    m.AttenuationNumUnits(), m.AttenuationQkappa(),
                    m.AttenuationQmu(),      m.AttenuationF0());
            }
            if (sim.SolidMaterial().HasAttenuation()) {
                const auto& m = sim.SolidMaterial();
                SEM::WriteQApproximationFile(
                    outdir + "/solid/q_approximation.dat",
                    m.AttenuationNumUnits(), m.AttenuationQkappa(),
                    m.AttenuationQmu(),      m.AttenuationF0());
            }
        }
    }
    return 0;
}

template<int Dim>
int RunInversion(std::unique_ptr<YamlConfig> config) {
    {
        int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            std::cerr << "[DIAG RunInversion] ENTERED" << std::endl;
            std::cerr.flush();
        }
    }
    double t_start_setup = MPI_Wtime();

    AdjointSimulation<Dim> sim(MPI_COMM_WORLD);
    if (sim.IsRoot()) {
        std::cerr << "[DIAG RunInversion] LoadConfig..." << std::endl;
        std::cerr.flush();
    }
    sim.LoadConfig(std::move(config));

    // Device setup (same as forward)
    const std::string device_str = sim.DeviceString();
    const bool use_gpu = (device_str.find("cuda") != std::string::npos ||
                          device_str.find("hip") != std::string::npos);

    int device_id = 0;
    if (use_gpu) {
        int num_devices = Device::GetDeviceCount();
        if (num_devices > 0) {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm local_comm;
            MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                MPI_INFO_NULL, &local_comm);
            int local_rank;
            MPI_Comm_rank(local_comm, &local_rank);
            MPI_Comm_free(&local_comm);
            device_id = local_rank % num_devices;
        }
    }

    Device device(device_str, device_id);

    if (use_gpu) {
#ifdef SEM_USE_GPU_AWARE_MPI
        Device::SetGPUAwareMPI(true);
#endif
    }

    // Setup all components
    if (sim.IsRoot()) {
        std::cerr << "[DIAG RunInversion] SetupFromConfig..." << std::endl;
        std::cerr.flush();
    }
    sim.SetupFromConfig();
    if (Device::Allows(Backend::DEVICE_MASK)) {
        if (sim.IsRoot()) {
            std::cerr << "[DIAG RunInversion] DeviceInit..." << std::endl;
            std::cerr.flush();
        }
        sim.DeviceInit();
    }

    double t_end_setup = MPI_Wtime();

    if (sim.IsRoot()) {
        std::cerr << "[DIAG RunInversion] Setup done ("
                  << (t_end_setup - t_start_setup) << "s), calling Run()..." << std::endl;
        std::cerr.flush();
    }

    // Run FWI
    double t_start = MPI_Wtime();
    sim.Run();
    if (Device::Allows(Backend::DEVICE_MASK)) {
        MFEM_DEVICE_SYNC;
    }
    double t_end = MPI_Wtime();

    // Always write summary file to output directory (mirrors RunSimulation).
    std::string summary_file = sim.SummaryFile();
    if (summary_file.empty()) {
        std::string outdir = sim.OutputDir();
        if (!outdir.empty()) {
            summary_file = outdir + "/summary.txt";
        }
    }
    if (!summary_file.empty()) {
        TimingInfo timing;
        timing.setup_time = t_end_setup - t_start_setup;
        timing.run_time = t_end - t_start;
        timing.io_time = 0.0;
        timing.total_time = t_end - t_start_setup;
        sim.WriteSummaryToFile(summary_file, timing);
    }

    // Write Q-approximation diagnostic if attenuation is enabled (rank 0 only)
    if (sim.Material().HasAttenuation()) {
        int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            std::string qfile = sim.OutputDir() + "/q_approximation.dat";
            SEM::WriteQApproximationFile(qfile,
                sim.Material().AttenuationNumUnits(),
                sim.Material().AttenuationQkappa(),
                sim.Material().AttenuationQmu(),
                sim.Material().AttenuationF0());
        }
    }

    return 0;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    // Initialize MPI and Hypre
    Mpi::Init(argc, argv);
    Hypre::Init();

    // Parse command line
    std::string config_file;
    OptionsParser args(argc, argv);
    args.AddOption(&config_file, "-config", "--config", "Configuration file (YAML)");
    args.Parse();

    if (!args.Good() || config_file.empty()) {
        if (Mpi::Root()) {
            std::cout << "\nUsage: " << argv[0] << " -config <config.yaml>\n\n";
            args.PrintUsage(std::cout);
        }
        return 1;
    }

    if (Mpi::Root()) {
        std::cerr << "[DIAG main] config=" << config_file << std::endl;
        std::cerr.flush();
    }

    // Parse YAML exactly once. Validation runs in the constructor; if
    // the config is malformed or uses removed legacy syntax, we abort
    // here with a single pointed error message instead of propagating
    // half-parsed state through the dispatcher.
    std::unique_ptr<YamlConfig> config;
    try {
        config = std::make_unique<YamlConfig>(config_file);
    } catch (const std::exception& e) {
        if (Mpi::Root()) {
            std::cerr << "Error reading config: " << e.what() << std::endl;
        }
        return 1;
    }
    if (!config->IsValid()) {
        if (Mpi::Root()) {
            std::cerr << "Error validating config: "
                      << config->GetValidationError() << std::endl;
        }
        return 1;
    }

    const int dimension          = config->GetDimension();
    const std::string mode       = config->GetSimulationMode();
    const std::string mat_type   = config->GetMaterialType();

    if (Mpi::Root()) {
        std::cerr << "[DIAG main] dim=" << dimension
                  << " mode=" << mode << std::endl;
        std::cerr.flush();
    }

    // Mode dispatch: inversion path takes precedence over material-type
    // routing (even `material.type: coupled` + `mode: inversion` runs
    // through the adjoint facade; the coupled FWI path isn't wired up
    // yet, so reject it explicitly rather than silently picking one).
    if (mode == "inversion" || mode == "misfit_only") {
        if (dimension == 2) return RunInversion<2>(std::move(config));
        if (dimension == 3) return RunInversion<3>(std::move(config));
        if (Mpi::Root()) {
            std::cerr << "Error: mode='" << mode
                      << "' requires dim=2 or 3, got " << dimension << ".\n";
        }
        return 1;
    }

    // Forward-mode material-type dispatch. Coupled fluid-solid uses
    // CoupledSimulationFacade (submesh machinery); everything else
    // (isotropic elastic/acoustic, anisotropic) flows through the
    // single-physics ForwardSimulation path.
    if (mat_type == "coupled") {
        if (dimension == 2) return RunCoupled<2>(std::move(config));
        if (dimension == 3) return RunCoupled<3>(std::move(config));
        if (Mpi::Root()) {
            std::cerr << "Error: coupled material requires dim=2 or 3, got "
                      << dimension << "\n";
        }
        return 1;
    }

    if (dimension == 2) return RunSimulation<2>(std::move(config));
    if (dimension == 3) return RunSimulation<3>(std::move(config));
    if (Mpi::Root()) {
        std::cerr << "Error: Invalid dimension " << dimension
                  << " (must be 2 or 3)\n";
    }
    return 1;
}
