/**
 * @file lbfgs_direction.cpp
 * @brief Compute L-BFGS search direction using two-loop recursion
 *
 * Given the current gradient and (optionally) L-BFGS history,
 * computes the search direction d = H^{-1} g (ascent direction).
 * The output is compatible with update_model which does m - step * g/max|g|.
 *
 * Iteration 0 (no history): outputs gradient as-is (steepest descent fallback).
 * Iteration k > 0: updates history with new {s, y} pair, then two-loop recursion.
 *
 * Usage:
 *   mpirun -np N lbfgs_direction \
 *       --gradient g_k.bp \
 *       --prev-gradient g_{k-1}.bp \
 *       --prev-model m_{k-1}.bp \
 *       --current-model m_k.bp \
 *       --history history.bp \
 *       --output direction.bp \
 *       --output-history history_out.bp \
 *       --memory 10 \
 *       --device cuda
 *
 *   For k=0, only --gradient, --output, --output-history are needed.
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
#include <memory>
#include <vector>

using namespace SEM;
using namespace mfem;

static real_t GlobalDot(const Vector& a, const Vector& b, MPI_Comm comm) {
    MFEM_VERIFY(a.Size() == b.Size(),
        "GlobalDot: size mismatch (" << a.Size() << " vs " << b.Size() << ")");
    const real_t* a_data = a.HostRead();
    const real_t* b_data = b.HostRead();
    real_t local_dot = 0.0;
    for (int i = 0; i < a.Size(); i++) {
        local_dot += a_data[i] * b_data[i];
    }
    real_t global_dot = 0.0;
    MPI_Allreduce(&local_dot, &global_dot, 1, MFEM_MPI_REAL_T, MPI_SUM, comm);
    return global_dot;
}

int main(int argc, char* argv[]) {
    Mpi::Init(argc, argv);
    Hypre::Init();

    std::string gradient_file, prev_gradient_file;
    std::string prev_model_file, current_model_file;
    std::string history_file, output_file, output_history_file;
    std::string config_file;
    std::string restrict_attr_str;
    std::vector<int> restrict_attrs;
    std::string device_str = "cpu";
    int memory = 10;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--gradient" && i + 1 < argc) gradient_file = argv[++i];
        else if (arg == "--prev-gradient" && i + 1 < argc) prev_gradient_file = argv[++i];
        else if (arg == "--prev-model" && i + 1 < argc) prev_model_file = argv[++i];
        else if (arg == "--current-model" && i + 1 < argc) current_model_file = argv[++i];
        else if (arg == "--history" && i + 1 < argc) history_file = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_file = argv[++i];
        else if (arg == "--output-history" && i + 1 < argc) output_history_file = argv[++i];
        else if (arg == "--memory" && i + 1 < argc) memory = std::atoi(argv[++i]);
        else if (arg == "--device" && i + 1 < argc) device_str = argv[++i];
        else if (arg == "--config" && i + 1 < argc) config_file = argv[++i];
        else if (arg == "--restrict-attr" && i + 1 < argc) {
            restrict_attr_str = argv[++i];
            restrict_attrs = ParseRestrictAttrList(restrict_attr_str);
        }
    }

    if (!restrict_attrs.empty() && config_file.empty()) {
        if (Mpi::Root()) {
            std::cerr << "Error: --restrict-attr requires --config\n";
        }
        return 1;
    }

    if (gradient_file.empty() || output_file.empty() || output_history_file.empty()) {
        if (Mpi::Root()) {
            std::cerr << "Usage: lbfgs_direction --gradient G.bp --output D.bp "
                      << "--output-history H.bp\n"
                      << "  [--prev-gradient PG.bp --prev-model PM.bp "
                      << "--current-model CM.bp]\n"
                      << "  [--history H_in.bp] [--memory M] "
                      << "[--device cpu|cuda]\n";
        }
        return 1;
    }

    Device device(device_str);

    MPI_Comm comm = MPI_COMM_WORLD;

    // Load current gradient
    if (Mpi::Root()) {
        std::cout << "Loading gradient: " << gradient_file << std::endl;
    }
    MaterialField grad_field = LoadFieldBP(gradient_file, "data", comm);
    Vector& g_k = grad_field.Data();
    const int n = g_k.Size();

    // Optional attribute restriction. When `--restrict-attr` is provided,
    // every secant pair (s, y) and the final direction get conservatively
    // zeroed outside the active attributes. Defense-in-depth: as long as
    // update_model also honors the same restriction, m and g stay zero
    // there and these masks become no-ops, but explicit masking guards
    // against bounds clamp / floating-point noise leaking into history.
    bool has_restrict = !restrict_attrs.empty();
    std::unique_ptr<ParMesh> pmesh_ptr;
    std::unique_ptr<H1_FECollection> fec_ptr;
    std::unique_ptr<ParFiniteElementSpace> fes_ptr;
    std::vector<unsigned char> active_dof;

    auto mask_vec_inplace = [&](Vector& v) {
        if (!has_restrict) return;
        MaterialField tmp(grad_field);  // copy layout (ne, ngllx, nglly)
        tmp.Data() = v;                   // overwrite with v's values
        ParGridFunction gf(fes_ptr.get());
        tmp.ToParGridFunction(*fes_ptr, gf);
        ApplyKeptDofMask(gf, active_dof);
        MaterialField masked = MaterialField::FromParGridFunction(*fes_ptr, gf);
        v = masked.Data();
    };

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

        pmesh_ptr.reset(LoadParMesh(config, comm));
        pmesh_ptr->SetCurvature(order, true, dim, Ordering::byNODES);
        fec_ptr = std::make_unique<H1_FECollection>(order, dim,
                                                      BasisType::GaussLobatto);
        fes_ptr = std::make_unique<ParFiniteElementSpace>(pmesh_ptr.get(),
                                                            fec_ptr.get());
        active_dof = BuildKeptDofMask(*fes_ptr, *pmesh_ptr, restrict_attrs);

        // Mask incoming gradient defensively (should already be masked by
        // gradient pipeline, but guard against accidental upstream changes).
        mask_vec_inplace(g_k);
    }

    bool have_prev = !prev_gradient_file.empty() &&
                     !prev_model_file.empty() &&
                     !current_model_file.empty();

    // Load or create history
    std::vector<Vector> hist_s, hist_y;
    int hist_iteration = 0;

    if (!history_file.empty()) {
        if (Mpi::Root()) {
            std::cout << "Loading history: " << history_file << std::endl;
        }
        LBFGSHistory hist = LoadLBFGSHistoryBP(history_file, comm);
        hist_s = std::move(hist.s);
        hist_y = std::move(hist.y);
        hist_iteration = hist.iteration;
        // Re-mask any history entries inherited from a run that didn't have
        // active_attributes set. Idempotent if already masked.
        if (has_restrict) {
            for (auto& s : hist_s) mask_vec_inplace(s);
            for (auto& y : hist_y) mask_vec_inplace(y);
        }
        if (Mpi::Root()) {
            std::cout << "  History pairs: " << hist_s.size()
                      << ", iteration: " << hist_iteration << std::endl;
        }
    }

    // If we have previous iteration data, compute new s and y pair
    if (have_prev) {
        if (Mpi::Root()) {
            std::cout << "Computing new s, y pair..." << std::endl;
        }

        // s_{k-1} = m_k - m_{k-1}
        MaterialField prev_model = LoadFieldBP(prev_model_file, "data", comm);
        MaterialField curr_model = LoadFieldBP(current_model_file, "data", comm);
        const Vector& m_prev = prev_model.Data();
        const Vector& m_curr = curr_model.Data();

        MFEM_VERIFY(m_prev.Size() == n && m_curr.Size() == n,
            "Model size mismatch with gradient");

        Vector s_new(n);
        const real_t* m_prev_data = m_prev.HostRead();
        const real_t* m_curr_data = m_curr.HostRead();
        real_t* s_data = s_new.HostWrite();
        for (int i = 0; i < n; i++) {
            s_data[i] = m_curr_data[i] - m_prev_data[i];
        }

        // y_{k-1} = g_k - g_{k-1}
        MaterialField prev_grad = LoadFieldBP(prev_gradient_file, "data", comm);
        const Vector& g_prev = prev_grad.Data();

        MFEM_VERIFY(g_prev.Size() == n,
            "Previous gradient size mismatch");

        Vector y_new(n);
        const real_t* g_k_data = g_k.HostRead();
        const real_t* g_prev_data = g_prev.HostRead();
        real_t* y_data = y_new.HostWrite();
        for (int i = 0; i < n; i++) {
            y_data[i] = g_k_data[i] - g_prev_data[i];
        }

        // Mask new s, y pair so the secant pair stays zero outside the
        // active attributes. This is the primary guard: even if bounds
        // clamping or numerical noise perturbed m at attr-out DOFs,
        // re-zeroing here keeps the L-BFGS history clean.
        mask_vec_inplace(s_new);
        mask_vec_inplace(y_new);

        // Curvature condition: y * s > 0
        real_t ys = GlobalDot(y_new, s_new, comm);
        if (Mpi::Root()) {
            std::cout << "  y * s = " << ys << std::endl;
        }

        if (ys > 0.0) {
            // Add to history (remove oldest if at capacity)
            if (static_cast<int>(hist_s.size()) >= memory) {
                hist_s.erase(hist_s.begin());
                hist_y.erase(hist_y.begin());
            }
            hist_s.push_back(std::move(s_new));
            hist_y.push_back(std::move(y_new));
            if (Mpi::Root()) {
                std::cout << "  Added pair to history (total: "
                          << hist_s.size() << ")" << std::endl;
            }
        } else {
            if (Mpi::Root()) {
                std::cout << "  Skipped: curvature condition not satisfied "
                          << "(y*s = " << ys << " <= 0)" << std::endl;
            }
        }
    }

    // Compute search direction
    Vector direction(n);
    const int m_pairs = static_cast<int>(hist_s.size());

    if (m_pairs == 0) {
        // No history: steepest descent (output gradient as-is)
        if (Mpi::Root()) {
            std::cout << "No history: using gradient (steepest descent)" << std::endl;
        }
        const real_t* g_data = g_k.HostRead();
        real_t* d_data = direction.HostWrite();
        for (int i = 0; i < n; i++) {
            d_data[i] = g_data[i];
        }
    } else {
        // Two-loop recursion
        if (Mpi::Root()) {
            std::cout << "Two-loop recursion with " << m_pairs
                      << " pairs..." << std::endl;
        }

        // q = g_k (copy)
        Vector q(n);
        const real_t* g_data = g_k.HostRead();
        real_t* q_data = q.HostWrite();
        for (int i = 0; i < n; i++) {
            q_data[i] = g_data[i];
        }

        // Forward loop: newest to oldest
        std::vector<real_t> rho(m_pairs);
        std::vector<real_t> alpha(m_pairs);

        for (int j = m_pairs - 1; j >= 0; j--) {
            real_t ys = GlobalDot(hist_y[j], hist_s[j], comm);
            rho[j] = (ys > 0.0) ? 1.0 / ys : 0.0;

            alpha[j] = rho[j] * GlobalDot(hist_s[j], q, comm);

            // q = q - alpha[j] * y[j]
            q_data = q.HostReadWrite();
            const real_t* y_data = hist_y[j].HostRead();
            for (int i = 0; i < n; i++) {
                q_data[i] -= alpha[j] * y_data[i];
            }
        }

        // Initial Hessian scaling: gamma = (y_newest * s_newest) / (y_newest * y_newest)
        real_t ys_newest = GlobalDot(hist_y[m_pairs - 1], hist_s[m_pairs - 1], comm);
        real_t yy_newest = GlobalDot(hist_y[m_pairs - 1], hist_y[m_pairs - 1], comm);
        real_t gamma = (yy_newest > 0.0) ? ys_newest / yy_newest : 1.0;

        if (Mpi::Root()) {
            std::cout << "  Initial Hessian scaling gamma = " << gamma << std::endl;
        }

        // r = gamma * q
        real_t* r_data = direction.HostWrite();
        const real_t* q_read = q.HostRead();
        for (int i = 0; i < n; i++) {
            r_data[i] = gamma * q_read[i];
        }

        // Backward loop: oldest to newest
        for (int j = 0; j < m_pairs; j++) {
            real_t beta = rho[j] * GlobalDot(hist_y[j], direction, comm);

            r_data = direction.HostReadWrite();
            const real_t* s_data = hist_s[j].HostRead();
            for (int i = 0; i < n; i++) {
                r_data[i] += s_data[i] * (alpha[j] - beta);
            }
        }

        if (Mpi::Root()) {
            std::cout << "  Two-loop recursion complete" << std::endl;
        }
    }

    // Defensive final mask: if s, y, g were all masked the recursion can only
    // produce zero in attr-out DOFs by linear combination, but accumulated
    // floating-point error can leave tiny nonzeros — eliminate them here.
    mask_vec_inplace(direction);

    // Save direction as MaterialField (reuse gradient's metadata)
    if (Mpi::Root()) {
        std::cout << "Saving direction: " << output_file << std::endl;
    }
    MaterialField dir_field = grad_field;  // copy metadata
    Vector& dir_data_vec = dir_field.Data();
    const real_t* dir_src = direction.HostRead();
    real_t* dir_dst = dir_data_vec.HostWrite();
    for (int i = 0; i < n; i++) {
        dir_dst[i] = dir_src[i];
    }
    SaveFieldBP(output_file, "data", dir_field, gradient_file, comm);

    // Save updated history
    if (Mpi::Root()) {
        std::cout << "Saving history: " << output_history_file
                  << " (" << hist_s.size() << " pairs)" << std::endl;
    }
    SaveLBFGSHistoryBP(output_history_file, hist_s, hist_y,
                       hist_iteration + 1, comm);

    if (Mpi::Root()) {
        std::cout << "Done." << std::endl;
    }

    return 0;
}
