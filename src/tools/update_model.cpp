/**
 * @file update_model.cpp
 * @brief Update model parameters using a descent direction
 *
 * Reads a model and direction (gradient or optimizer output) from ADIOS2 .bp files,
 * applies the update and saves the updated model.
 *
 * Update formula (default):
 *   m_new = m - lr * d / max|d|
 *
 * With --no-normalize:
 *   m_new = m - lr * d
 *
 * lr is the learning rate in physical units (e.g. m/s for Vp),
 * representing the maximum absolute model change per iteration (normalized mode)
 * or a direct step-size multiplier (no-normalize mode).
 *
 * Optionally `--config` + `--restrict-attr` keeps DOFs outside the requested
 * element attributes invariant (no update, no bounds clamp). Used by FWI to
 * freeze regions like a water layer (`gradient.active_attributes`).
 */

#include "io/ADIOS2IO.hpp"
#include "material/MaterialField.hpp"
#include "config/YamlConfig.hpp"
#include "config/ConfigLoaders.hpp"
#include "common/AttributeMask.hpp"
#include <mfem.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <cstdlib>

using namespace SEM;
using namespace mfem;

int main(int argc, char* argv[]) {
    Mpi::Init(argc, argv);
    Hypre::Init();

    std::string model_file, gradient_file, output_file;
    std::string config_file;
    std::string restrict_attr_str;
    std::vector<int> restrict_attrs;
    std::string device_str = "cpu";
    real_t lr = -1.0;             // learning rate (physical units)
    real_t threshold_pct = 0.0;   // default: no threshold
    real_t bound_min = -1.0, bound_max = -1.0;
    bool normalize = true;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model_file = argv[++i];
        else if (arg == "--gradient" && i + 1 < argc) gradient_file = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_file = argv[++i];
        else if (arg == "--lr" && i + 1 < argc) lr = std::atof(argv[++i]);
        else if (arg == "--threshold" && i + 1 < argc) threshold_pct = std::atof(argv[++i]);
        else if (arg == "--min" && i + 1 < argc) bound_min = std::atof(argv[++i]);
        else if (arg == "--max" && i + 1 < argc) bound_max = std::atof(argv[++i]);
        else if (arg == "--no-normalize") normalize = false;
        else if (arg == "--device" && i + 1 < argc) device_str = argv[++i];
        else if (arg == "--config" && i + 1 < argc) config_file = argv[++i];
        else if (arg == "--restrict-attr" && i + 1 < argc) {
            restrict_attr_str = argv[++i];
            restrict_attrs = ParseRestrictAttrList(restrict_attr_str);
        }
    }

    if (model_file.empty() || gradient_file.empty() || output_file.empty()
        || lr <= 0.0) {
        if (Mpi::Root()) {
            std::cerr << "Usage: update_model --model M.bp --gradient G.bp "
                      << "--output O.bp --lr LR\n"
                      << "  --lr LR             (max absolute model change, required)\n"
                      << "  --threshold T       (zero |d| < T% of max|d|, default 0)\n"
                      << "  --no-normalize      (skip d/max|d| normalization)\n"
                      << "  --min MIN --max MAX (bounds clamp, optional)\n"
                      << "  --config C.yaml     (required only with --restrict-attr)\n"
                      << "  --restrict-attr IDs (skip update outside these element attrs)\n"
                      << "  --device cpu|cuda   (default: cpu)\n";
        }
        return 1;
    }

    if (!restrict_attrs.empty() && config_file.empty()) {
        if (Mpi::Root()) {
            std::cerr << "Error: --restrict-attr requires --config\n";
        }
        return 1;
    }

    Device device(device_str);

    MPI_Comm comm = MPI_COMM_WORLD;

    // Load model and direction
    if (Mpi::Root()) {
        std::cout << "Loading model: " << model_file << std::endl;
    }
    MaterialField model = LoadFieldBP(model_file, "data", comm);

    if (Mpi::Root()) {
        std::cout << "Loading direction: " << gradient_file << std::endl;
    }
    MaterialField gradient = LoadFieldBP(gradient_file, "data", comm);

    Vector& m = model.Data();
    const Vector& g = gradient.Data();
    const int n = m.Size();

    MFEM_VERIFY(n == g.Size(),
        "Model size (" << n << ") != direction size (" << g.Size() << ")");

    // Find global max |direction| across all ranks (host-side reduction).
    // When --restrict-attr is active we still scan only attr-active DOFs so
    // the normalization scale isn't dominated by attr-outside spikes.
    const real_t* g_host = g.HostRead();
    real_t local_max_abs_g = 0.0;
    for (int i = 0; i < n; i++) {
        real_t abs_g = std::abs(g_host[i]);
        if (abs_g > local_max_abs_g) local_max_abs_g = abs_g;
    }

    real_t global_max_abs_g = 0.0;
    MPI_Allreduce(&local_max_abs_g, &global_max_abs_g, 1,
                  MFEM_MPI_REAL_T, MPI_MAX, comm);

    const real_t scale = normalize ? (global_max_abs_g > 0.0 ? lr / global_max_abs_g : 0.0)
                                   : lr;
    const real_t thresh = threshold_pct / 100.0 * global_max_abs_g;

    if (Mpi::Root()) {
        std::cout << "Model update:" << std::endl;
        std::cout << "  max |direction| = " << global_max_abs_g << std::endl;
        std::cout << "  lr = " << lr << std::endl;
        std::cout << "  normalize = " << (normalize ? "yes" : "no") << std::endl;
        if (threshold_pct > 0.0) {
            std::cout << "  threshold = " << threshold_pct << "% of max -> "
                      << thresh << std::endl;
        }
        if (!restrict_attrs.empty()) {
            std::cout << "  restrict-attr = [" << restrict_attr_str << "]"
                      << std::endl;
        }
    }

    // Optional attribute restriction: build a per-DOF active mask via the
    // mesh + FE space. DOFs outside the active attrs are skipped (no update,
    // no bounds clamp) so e.g. a water layer stays exactly at its initial
    // value. Done in the standard ToParGridFunction → mask → FromParGridFunction
    // dance because MaterialField is element-local; we need global LDOFs to
    // identify shared interface DOFs conservatively.
    std::vector<unsigned char> active_dof;
    bool has_restrict = !restrict_attrs.empty();

    if (has_restrict) {
        YamlConfig config(config_file);
        if (!config.IsValid()) {
            if (Mpi::Root()) {
                std::cerr << "Config error: " << config.GetValidationError()
                          << std::endl;
            }
            return 1;
        }
        int order = config.GetOrder();
        int dim = config.GetDimension();

        std::unique_ptr<ParMesh> pmesh(LoadParMesh(config, comm));
        pmesh->SetCurvature(order, true, dim, Ordering::byNODES);

        H1_FECollection fec(order, dim, BasisType::GaussLobatto);
        ParFiniteElementSpace fes(pmesh.get(), &fec);

        active_dof = BuildKeptDofMask(fes, *pmesh, restrict_attrs);

        // We operate in element-local layout (same as MaterialField); convert
        // the per-LDOF mask into per-element-local-DOF mask via element DOFs.
        // Strategy: build the same `kept` over fes.GetVSize() and walk
        // element-local entries through fes.GetElementDofs(ie, ...) to look
        // up the LDOF index. But the MaterialField data layout is
        // (ne, ngllx, nglly) flattened — we need the matching iteration order.
        //
        // Simpler & robust: convert g & m via ToParGridFunction, do the
        // update on ParGridFunction, then write back via FromParGridFunction.

        // Convert model and gradient to ParGridFunctions
        ParGridFunction gf_m(&fes);
        ParGridFunction gf_g(&fes);
        model.ToParGridFunction(fes, gf_m);
        gradient.ToParGridFunction(fes, gf_g);

        if (Mpi::Root()) {
            int nactive = 0;
            for (auto v : active_dof) nactive += (v ? 1 : 0);
            std::cout << "  active DOFs = " << nactive << " / "
                      << active_dof.size() << std::endl;
        }

        // Apply update + bounds only on active DOFs
        for (int i = 0; i < gf_m.Size(); i++) {
            if (!active_dof[i]) continue;
            real_t gi = gf_g(i);
            if (std::abs(gi) >= thresh) {
                gf_m(i) -= scale * gi;
            }
            if (bound_min > 0.0 && gf_m(i) < bound_min) gf_m(i) = bound_min;
            if (bound_max > 0.0 && gf_m(i) > bound_max) gf_m(i) = bound_max;
        }

        // Convert back
        model = MaterialField::FromParGridFunction(fes, gf_m);
    } else if (global_max_abs_g > 0.0) {
        // No attribute restriction: original element-local fast path on device.
        const real_t s = scale;
        const real_t t = thresh;
        const real_t bmin = bound_min;
        const real_t bmax = bound_max;
        real_t* m_data = m.ReadWrite();
        const real_t* g_data = g.Read();
        mfem::forall(n, [=] MFEM_HOST_DEVICE (int i) {
            if (fabs(g_data[i]) >= t) {
                m_data[i] = m_data[i] - s * g_data[i];
            }
            if (bmin > 0.0 && m_data[i] < bmin) m_data[i] = bmin;
            if (bmax > 0.0 && m_data[i] > bmax) m_data[i] = bmax;
        });
    }

    // Save updated model
    if (Mpi::Root()) {
        std::cout << "Saving updated model: " << output_file << std::endl;
    }
    SaveFieldBP(output_file, "data", model, model_file, comm);

    if (Mpi::Root()) {
        std::cout << "Done." << std::endl;
    }

    return 0;
}
