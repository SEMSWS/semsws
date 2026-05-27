/**
 * @file smooth_kernel.cpp
 * @brief Helmholtz smoothing tool for FWI sensitivity kernels
 *
 * Reads a kernel from ADIOS2 .bp, smooths via Helmholtz filter, saves result.
 * Mesh and order are read from config.yaml for consistency with forward/adjoint.
 *
 * Usage:
 *   # Constant isotropic smoothing (sigma = alpha directly in meters)
 *   mpirun -np N smooth_kernel \
 *       --config sim.yaml \
 *       --input kernel.bp --output kernel_smooth.bp \
 *       --constant --alpha 150.0 --device cuda
 *
 *   # Variable isotropic smoothing (sigma = alpha * Vp / freq)
 *   mpirun -np N smooth_kernel \
 *       --config sim.yaml \
 *       --input kernel.bp --output kernel_smooth.bp \
 *       --var --vref vp.bp --alpha 1.0 --freq 10.0
 *
 *   # Constant anisotropic smoothing
 *   mpirun -np N smooth_kernel \
 *       --config sim.yaml \
 *       --input kernel.bp --output kernel_smooth.bp \
 *       --constant --alpha-x 50.0 --alpha-y 200.0
 *
 *   # Variable anisotropic smoothing
 *   mpirun -np N smooth_kernel \
 *       --config sim.yaml \
 *       --input kernel.bp --output kernel_smooth.bp \
 *       --var --vref vp.bp --alpha-x 1.0 --alpha-y 3.0 --freq 10.0
 */

#include "io/ADIOS2IO.hpp"
#include "material/MaterialField.hpp"
#include "fwi/HelmholtzSmoothing.hpp"
#include "config/YamlConfig.hpp"
#include "config/ConfigLoaders.hpp"
#include <mfem.hpp>
#include <iostream>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cmath>
#include <sstream>
#include <set>

using namespace SEM;
using namespace mfem;

int main(int argc, char* argv[]) {
    Mpi::Init(argc, argv);
    Hypre::Init();

    // Parse arguments
    std::string config_file, input_file, output_file, vref_file;
    std::string device_str = "cpu";
    real_t alpha = -1.0, freq = -1.0;
    real_t alpha_x = -1.0, alpha_y = -1.0;
    real_t clip_percentile = -1.0;
    int niter = 1;
    bool mode_constant = false;
    bool mode_var = false;
    std::vector<int> mask_attrs;
    std::vector<int> restrict_attrs;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) config_file = argv[++i];
        else if (arg == "--input" && i + 1 < argc) input_file = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_file = argv[++i];
        else if (arg == "--vref" && i + 1 < argc) vref_file = argv[++i];
        else if (arg == "--alpha" && i + 1 < argc) alpha = std::atof(argv[++i]);
        else if (arg == "--alpha-x" && i + 1 < argc) alpha_x = std::atof(argv[++i]);
        else if (arg == "--alpha-y" && i + 1 < argc) alpha_y = std::atof(argv[++i]);
        else if (arg == "--freq" && i + 1 < argc) freq = std::atof(argv[++i]);
        else if (arg == "--clip-percentile" && i + 1 < argc) clip_percentile = std::atof(argv[++i]);
        else if (arg == "--niter" && i + 1 < argc) niter = std::atoi(argv[++i]);
        else if (arg == "--device" && i + 1 < argc) device_str = argv[++i];
        else if (arg == "--mask-attr" && i + 1 < argc) {
            std::string val = argv[++i];
            std::istringstream ss(val);
            std::string token;
            while (std::getline(ss, token, ',')) {
                mask_attrs.push_back(std::stoi(token));
            }
        }
        else if (arg == "--restrict-attr" && i + 1 < argc) {
            std::string val = argv[++i];
            std::istringstream ss(val);
            std::string token;
            while (std::getline(ss, token, ',')) {
                restrict_attrs.push_back(std::stoi(token));
            }
        }
        else if (arg == "--constant") mode_constant = true;
        else if (arg == "--var") mode_var = true;
    }

    // Validate required arguments
    if (config_file.empty() || input_file.empty() || output_file.empty()) {
        if (Mpi::Root()) {
            std::cerr << "Usage: smooth_kernel --config C.yaml --input X.bp --output Y.bp\n"
                      << "  --constant --alpha S              (isotropic, sigma=S meters)\n"
                      << "  --constant --alpha-x Sx --alpha-y Sy  (anisotropic, meters)\n"
                      << "  --var --vref V.bp --alpha A --freq F  (isotropic, sigma=A*Vref/F)\n"
                      << "  --var --vref V.bp --alpha-x Ax --alpha-y Ay --freq F (anisotropic)\n"
                      << "  [--device cpu|cuda]\n";
        }
        return 1;
    }

    if (!mode_constant && !mode_var) {
        if (Mpi::Root()) {
            std::cerr << "Error: Must specify --constant or --var mode\n";
        }
        return 1;
    }

    if (mode_constant && mode_var) {
        if (Mpi::Root()) {
            std::cerr << "Error: Cannot specify both --constant and --var\n";
        }
        return 1;
    }

    // Determine isotropic vs anisotropic
    bool is_aniso = (alpha_x > 0.0 && alpha_y > 0.0);
    bool is_iso = (alpha > 0.0 && !is_aniso);

    if (!is_iso && !is_aniso) {
        if (Mpi::Root()) {
            std::cerr << "Error: Must specify --alpha (isotropic) "
                      << "or --alpha-x and --alpha-y (anisotropic)\n";
        }
        return 1;
    }

    // Validate --var mode requirements
    if (mode_var) {
        if (vref_file.empty()) {
            if (Mpi::Root()) {
                std::cerr << "Error: --var mode requires --vref\n";
            }
            return 1;
        }
        if (freq <= 0.0) {
            if (Mpi::Root()) {
                std::cerr << "Error: --var mode requires --freq > 0\n";
            }
            return 1;
        }
    }

    Device device(device_str);

    MPI_Comm comm = MPI_COMM_WORLD;

    // Load config
    YamlConfig config(config_file);
    if (!config.IsValid()) {
        if (Mpi::Root()) {
            std::cerr << "Config error: " << config.GetValidationError() << std::endl;
        }
        return 1;
    }

    int order = config.GetOrder();
    int dim = config.GetDimension();

    // Load mesh from config (supports external, internal, partitioned)
    if (Mpi::Root()) {
        std::cout << "Loading mesh from config: " << config_file << std::endl;
    }
    std::unique_ptr<ParMesh> pmesh(LoadParMesh(config, comm));
    pmesh->SetCurvature(order, true, dim, Ordering::byNODES);

    H1_FECollection fec(order, dim, BasisType::GaussLobatto);
    ParFiniteElementSpace fes(pmesh.get(), &fec);

    // Load kernel from ADIOS2 (dim-dependent BP reader)
    if (Mpi::Root()) {
        std::cout << "Loading kernel: " << input_file << std::endl;
    }
    ParGridFunction gf(&fes);
    if (dim == 2) {
        MaterialField kernel = LoadFieldBP(input_file, "data", comm);
        kernel.ToParGridFunction(fes, gf);
    } else {
        MaterialField3D kernel = LoadFieldBP3D(input_file, "data", comm);
        kernel.ToParGridFunction(fes, gf);
    }

    // Collect DOFs belonging to masked attributes (before clip and smooth)
    std::set<int> mask_dof_set;
    if (!mask_attrs.empty()) {
        std::set<int> attr_set(mask_attrs.begin(), mask_attrs.end());
        for (int i = 0; i < pmesh->GetNE(); i++) {
            if (attr_set.count(pmesh->GetAttribute(i))) {
                Array<int> dofs;
                fes.GetElementDofs(i, dofs);
                for (int j = 0; j < dofs.Size(); j++) {
                    mask_dof_set.insert(dofs[j] >= 0 ? dofs[j] : -1 - dofs[j]);
                }
            }
        }
        if (Mpi::Root()) {
            std::cout << "Mask attributes: ";
            for (int a : mask_attrs) std::cout << a << " ";
            std::cout << "(" << mask_dof_set.size() << " DOFs masked)"
                      << std::endl;
        }
    }

    // Zero out masked DOFs before clipping (so mask region doesn't affect percentile)
    for (int d : mask_dof_set) {
        gf(d) = 0.0;
    }

    // Percentile clipping (after mask, before smoothing)
    if (clip_percentile > 0.0 && clip_percentile < 100.0) {
        // Gather all |values| locally
        int local_n = gf.Size();
        std::vector<real_t> local_abs(local_n);
        const real_t* gf_data = gf.HostRead();
        for (int i = 0; i < local_n; i++) {
            local_abs[i] = std::abs(gf_data[i]);
        }

        // Gather sizes from all ranks
        int nprocs = Mpi::WorldSize();
        std::vector<int> all_sizes(nprocs);
        MPI_Allgather(&local_n, 1, MPI_INT,
                       all_sizes.data(), 1, MPI_INT, comm);

        int global_n = 0;
        std::vector<int> displs(nprocs);
        for (int r = 0; r < nprocs; r++) {
            displs[r] = global_n;
            global_n += all_sizes[r];
        }

        // Gather all |values| on all ranks
        std::vector<real_t> all_abs(global_n);
        MPI_Allgatherv(local_abs.data(), local_n, MFEM_MPI_REAL_T,
                        all_abs.data(), all_sizes.data(), displs.data(),
                        MFEM_MPI_REAL_T, comm);

        // Sort and find percentile threshold
        std::sort(all_abs.begin(), all_abs.end());
        int idx = static_cast<int>(clip_percentile / 100.0 * (global_n - 1));
        if (idx >= global_n) idx = global_n - 1;
        real_t threshold = all_abs[idx];

        // Clip
        real_t* gf_rw = gf.HostReadWrite();
        int clipped = 0;
        for (int i = 0; i < local_n; i++) {
            if (gf_rw[i] > threshold) { gf_rw[i] = threshold; clipped++; }
            else if (gf_rw[i] < -threshold) { gf_rw[i] = -threshold; clipped++; }
        }

        int total_clipped = 0;
        MPI_Reduce(&clipped, &total_clipped, 1, MPI_INT, MPI_SUM, 0, comm);
        if (Mpi::Root()) {
            std::cout << "Percentile clipping: P" << clip_percentile
                      << " threshold = " << threshold
                      << ", clipped " << total_clipped << " / " << global_n
                      << " values" << std::endl;
        }
    }

    if (Mpi::Root() && !restrict_attrs.empty()) {
        std::cout << "Restrict smoothing to attributes: ";
        for (int a : restrict_attrs) std::cout << a << " ";
        std::cout << std::endl;
    }

    // Scale sigma by 1/sqrt(niter) so total smoothing width is preserved
    real_t scale = 1.0 / std::sqrt(static_cast<real_t>(niter));
    real_t eff_alpha = alpha * scale;
    real_t eff_alpha_x = alpha_x * scale;
    real_t eff_alpha_y = alpha_y * scale;

    // Load Vp once if needed
    std::unique_ptr<ParGridFunction> vp_gf_ptr;
    if (mode_var) {
        if (Mpi::Root()) {
            std::cout << "Loading velocity reference: " << vref_file << std::endl;
        }
        vp_gf_ptr = std::make_unique<ParGridFunction>(&fes);
        if (dim == 2) {
            MaterialField vp_field = LoadFieldBP(vref_file, "data", comm);
            vp_field.ToParGridFunction(fes, *vp_gf_ptr);
        } else {
            MaterialField3D vp_field = LoadFieldBP3D(vref_file, "data", comm);
            vp_field.ToParGridFunction(fes, *vp_gf_ptr);
        }
    }

    if (Mpi::Root() && niter > 1) {
        std::cout << "Repeated smoothing: niter = " << niter
                  << ", sigma scale = " << scale << std::endl;
    }

    // Smooth (repeated niter times)
    for (int n = 0; n < niter; n++) {
        if (Mpi::Root() && niter > 1) {
            std::cout << "  Smoothing pass " << (n + 1) << " / " << niter
                      << std::endl;
        }

        if (mode_constant && is_aniso) {
            if (Mpi::Root() && n == 0) {
                std::cout << "Anisotropic smoothing (constant): sigma_x = "
                          << eff_alpha_x << " m, sigma_y = " << eff_alpha_y
                          << " m" << std::endl;
            }
            HelmholtzSmoothAniso(fes, gf, eff_alpha_x, eff_alpha_y,
                                 restrict_attrs);

        } else if (mode_constant && is_iso) {
            if (Mpi::Root() && n == 0) {
                std::cout << "Isotropic smoothing (constant): sigma = "
                          << eff_alpha << " m" << std::endl;
            }
            HelmholtzSmooth(fes, gf, eff_alpha, restrict_attrs);

        } else if (mode_var && is_aniso) {
            if (Mpi::Root() && n == 0) {
                std::cout << "Anisotropic smoothing (variable): sigma_x = "
                          << eff_alpha_x << " * Vp / " << freq
                          << ", sigma_y = " << eff_alpha_y << " * Vp / "
                          << freq << std::endl;
            }
            HelmholtzSmoothAniso(fes, gf, *vp_gf_ptr, eff_alpha_x,
                                 eff_alpha_y, freq, restrict_attrs);

        } else {
            if (Mpi::Root() && n == 0) {
                std::cout << "Isotropic smoothing (variable): sigma = "
                          << eff_alpha << " * Vp / " << freq << std::endl;
            }
            HelmholtzSmooth(fes, gf, *vp_gf_ptr, eff_alpha, freq,
                            restrict_attrs);
        }
    }

    // Zero out masked DOFs after smoothing
    for (int d : mask_dof_set) {
        gf(d) = 0.0;
    }

    if (Mpi::Root() && !mask_dof_set.empty()) {
        std::cout << "Applied mask: zeroed " << mask_dof_set.size()
                  << " DOFs" << std::endl;
    }

    // Convert back to MaterialField
    // (dim-dependent save below)

    // Save via ADIOS2 (dim-specific BP writer)
    if (Mpi::Root()) {
        std::cout << "Saving smoothed kernel: " << output_file << std::endl;
    }
    if (dim == 2) {
        MaterialField smoothed = MaterialField::FromParGridFunction(fes, gf);
        SaveFieldBP(output_file, "data", smoothed, config_file, comm);
    } else {
        MaterialField3D smoothed = MaterialField3D::FromParGridFunction(fes, gf);
        SaveFieldBP(output_file, "data", smoothed, config_file, comm);
    }

    if (Mpi::Root()) {
        std::cout << "Done." << std::endl;
    }

    return 0;
}
