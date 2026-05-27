/**
 * @file precondition_gradient.cpp
 * @brief Precondition gradient with pseudo-Hessian diagonal approximation
 *
 * Water-level preconditioning with normalized Hessian:
 *   H_norm = H / max(|H|)
 *   g_prec = g / (|H_norm| + delta)
 *
 * This normalizes the Hessian to [0, 1] before division, so the output
 * retains the gradient's scale modulated by the relative Hessian pattern.
 * delta is a water-level fraction (default 1e-2) preventing division by zero.
 *
 * Usage:
 *   mpirun -np N precondition_gradient \
 *       --config sim.yaml \
 *       --gradient kernel_vp_smooth.bp \
 *       --hessian hessian_vp_smooth.bp \
 *       --output precond_vp.bp \
 *       --delta 1e-2 \
 *       --mask-attr 1 \
 *       --device cuda
 */

#include "io/ADIOS2IO.hpp"
#include "material/MaterialField.hpp"
#include "config/YamlConfig.hpp"
#include "config/ConfigLoaders.hpp"
#include <mfem.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <set>

using namespace SEM;
using namespace mfem;

int main(int argc, char* argv[]) {
    Mpi::Init(argc, argv);
    Hypre::Init();

    std::string config_file, gradient_file, hessian_file, output_file;
    std::string device_str = "cpu";
    real_t delta = 1e-2;
    std::vector<int> mask_attrs;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) config_file = argv[++i];
        else if (arg == "--gradient" && i + 1 < argc) gradient_file = argv[++i];
        else if (arg == "--hessian" && i + 1 < argc) hessian_file = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_file = argv[++i];
        else if (arg == "--delta" && i + 1 < argc) delta = std::atof(argv[++i]);
        else if (arg == "--device" && i + 1 < argc) device_str = argv[++i];
        else if (arg == "--mask-attr" && i + 1 < argc) {
            std::string val = argv[++i];
            std::istringstream ss(val);
            std::string token;
            while (std::getline(ss, token, ',')) {
                mask_attrs.push_back(std::stoi(token));
            }
        }
    }

    if (gradient_file.empty() || hessian_file.empty() || output_file.empty()) {
        if (Mpi::Root()) {
            std::cerr << "Usage: precondition_gradient \\\n"
                      << "  --gradient G.bp --hessian H.bp --output O.bp\n"
                      << "  [--config sim.yaml]  (required for --mask-attr)\n"
                      << "  [--delta DELTA]      (water-level fraction, default 1e-2)\n"
                      << "  [--mask-attr 1,2]    (zero gradient for these mesh attributes)\n"
                      << "  [--device cpu|cuda]  (default: cpu)\n";
        }
        return 1;
    }

    Device device(device_str);

    MPI_Comm comm = MPI_COMM_WORLD;

    if (Mpi::Root()) {
        std::cout << "Loading gradient: " << gradient_file << std::endl;
    }
    MaterialField gradient = LoadFieldBP(gradient_file, "data", comm);

    if (Mpi::Root()) {
        std::cout << "Loading Hessian: " << hessian_file << std::endl;
    }
    MaterialField hessian = LoadFieldBP(hessian_file, "data", comm);

    MFEM_VERIFY(gradient.Data().Size() == hessian.Data().Size(),
        "Gradient size (" << gradient.Data().Size()
        << ") != Hessian size (" << hessian.Data().Size() << ")");

    // Mask elements belonging to specified mesh attributes
    // Zero GLL data directly in MaterialField
    if (!mask_attrs.empty()) {
        MFEM_VERIFY(!config_file.empty(),
            "--mask-attr requires --config to load mesh");

        YamlConfig config(config_file);
        MFEM_VERIFY(config.IsValid(),
            "Config error: " << config.GetValidationError());

        std::unique_ptr<ParMesh> pmesh(LoadParMesh(config, comm));

        const int ne = pmesh->GetNE();
        MFEM_VERIFY(ne == gradient.NumElements(),
            "Mesh elements (" << ne << ") != field elements ("
            << gradient.NumElements() << ")");

        std::set<int> attr_set(mask_attrs.begin(), mask_attrs.end());
        const int gll_per_elem = gradient.NumGLLx() * gradient.NumGLLy();
        real_t* g_ptr = gradient.Data().HostReadWrite();
        real_t* h_ptr = hessian.Data().HostReadWrite();

        int masked_elems = 0;
        for (int e = 0; e < ne; e++) {
            if (attr_set.count(pmesh->GetAttribute(e))) {
                const int offset = e * gll_per_elem;
                for (int k = 0; k < gll_per_elem; k++) {
                    g_ptr[offset + k] = 0.0;
                    h_ptr[offset + k] = 0.0;
                }
                masked_elems++;
            }
        }

        if (Mpi::Root()) {
            std::cout << "Mask attributes: ";
            for (int a : mask_attrs) std::cout << a << " ";
            std::cout << "(" << masked_elems << " elements, "
                      << masked_elems * gll_per_elem << " GLL points masked)"
                      << std::endl;
        }
    }

    const int total_size = gradient.Data().Size();

    // Compute max(|H|) across all MPI ranks (after masking)
    // Use HostRead for the reduction — small scalar result via MPI
    const real_t* h_host = hessian.Data().HostRead();
    real_t local_max = 0.0;
    for (int i = 0; i < total_size; i++) {
        local_max = std::max(local_max, std::abs(h_host[i]));
    }
    real_t global_max = 0.0;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (Mpi::Root()) {
        std::cout << "  max(|H|) = " << global_max
                  << ", delta (threshold) = " << delta << std::endl;
    }

    MFEM_VERIFY(global_max > 0.0,
        "Hessian is zero everywhere — cannot precondition. "
        "Check that the adjoint simulation produced valid Hessian output.");

    // Water-level preconditioning with normalized Hessian:
    //   g_prec = g / (|H| / max(|H|) + delta)
    // Normalizing H keeps the output on the same scale as g,
    // modulated by the relative Hessian pattern.
    {
        const real_t inv_hmax = 1.0 / global_max;
        const real_t d = delta;
        real_t* g_data = gradient.Data().ReadWrite();
        const real_t* h_data = hessian.Data().Read();
        mfem::forall(total_size, [=] MFEM_HOST_DEVICE (int i) {
            g_data[i] = g_data[i] / (fabs(h_data[i]) * inv_hmax + d);
        });
    }

    if (Mpi::Root()) {
        std::cout << "  Hessian normalized by max(|H|) = "
                  << global_max << std::endl;
        std::cout << "  Water-level delta = " << delta << std::endl;
        std::cout << "  Max amplification at H=0: 1/delta = "
                  << 1.0 / delta << std::endl;
    }

    if (Mpi::Root()) {
        std::cout << "Saving: " << output_file << std::endl;
    }
    SaveFieldBP(output_file, "data", gradient, gradient_file, comm);

    if (Mpi::Root()) {
        std::cout << "Done." << std::endl;
    }

    return 0;
}
