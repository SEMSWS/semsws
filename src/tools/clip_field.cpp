/**
 * @file clip_field.cpp
 * @brief Clip a scalar BP field's values into a symmetric range to suppress
 *        outliers before downstream processing (e.g. gradient smoothing).
 *
 * Modes:
 *   --mode percentile --percentile P
 *       Threshold = P-th percentile of |x| over all DOFs (MPI-gathered).
 *       Each x is clipped to ±threshold.
 *   --mode absolute --value V
 *       Clip x to [−V, +V].
 *   --mode of_max --fraction F
 *       Threshold = F · max(|x|) over all DOFs (MPI-Allreduce).
 *       Each x is clipped to ±threshold.
 *
 * Usage:
 *   mpirun -np N clip_field --input g.bp --output g_clipped.bp \
 *       --mode percentile --percentile 99.0
 *
 * Notes: when input == output, the file is overwritten in place.
 */

#include "io/ADIOS2IO.hpp"
#include "material/MaterialField.hpp"
#include "config/YamlConfig.hpp"
#include "config/ConfigLoaders.hpp"
#include "common/AttributeMask.hpp"
#include <mfem.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace SEM;
using namespace mfem;

namespace {

real_t ComputePercentileThreshold(const real_t* data, int n_local,
                                    real_t percentile, MPI_Comm comm)
{
    const int nprocs = Mpi::WorldSize();
    std::vector<int> counts(nprocs);
    MPI_Gather(&n_local, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);
    std::vector<int> displs(nprocs);
    long total = 0;
    if (Mpi::Root()) {
        for (int p = 0; p < nprocs; p++) { displs[p] = total; total += counts[p]; }
    }
    std::vector<real_t> local_abs(n_local);
    for (int i = 0; i < n_local; i++) local_abs[i] = std::abs(data[i]);
    std::vector<real_t> all_abs;
    if (Mpi::Root()) all_abs.resize(total);
    MPI_Gatherv(local_abs.data(), n_local, MFEM_MPI_REAL_T,
                all_abs.data(), counts.data(), displs.data(),
                MFEM_MPI_REAL_T, 0, comm);

    real_t threshold = 0.0;
    if (Mpi::Root() && total > 0) {
        std::sort(all_abs.begin(), all_abs.end());
        long k = static_cast<long>(std::round((percentile / 100.0) * (total - 1)));
        if (k < 0) k = 0;
        if (k > total - 1) k = total - 1;
        threshold = all_abs[k];
    }
    MPI_Bcast(&threshold, 1, MFEM_MPI_REAL_T, 0, comm);
    return threshold;
}

real_t ComputeMaxAbs(const real_t* data, int n, MPI_Comm comm) {
    real_t local_max = 0;
    for (int i = 0; i < n; i++) {
        const real_t a = std::abs(data[i]);
        if (a > local_max) local_max = a;
    }
    real_t global_max = 0;
    MPI_Allreduce(&local_max, &global_max, 1, MFEM_MPI_REAL_T, MPI_MAX, comm);
    return global_max;
}

}  // namespace

int main(int argc, char* argv[]) {
    Mpi::Init(argc, argv);
    Hypre::Init();

    std::string input_file, output_file, mode = "percentile";
    std::string config_file;
    std::string restrict_attr_str;
    std::vector<int> restrict_attrs;
    real_t percentile = 99.0;
    real_t abs_value = 0.0;
    real_t fraction = 0.95;
    std::string device_str = "cpu";

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--input"     && i + 1 < argc) input_file = argv[++i];
        else if (a == "--output"    && i + 1 < argc) output_file = argv[++i];
        else if (a == "--mode"      && i + 1 < argc) mode = argv[++i];
        else if (a == "--percentile" && i + 1 < argc) percentile = std::atof(argv[++i]);
        else if (a == "--value"     && i + 1 < argc) abs_value = std::atof(argv[++i]);
        else if (a == "--fraction"  && i + 1 < argc) fraction = std::atof(argv[++i]);
        else if (a == "--device"    && i + 1 < argc) device_str = argv[++i];
        else if (a == "--config"    && i + 1 < argc) config_file = argv[++i];
        else if (a == "--restrict-attr" && i + 1 < argc) {
            restrict_attr_str = argv[++i];
            restrict_attrs = ParseRestrictAttrList(restrict_attr_str);
        }
    }
    if (input_file.empty() || output_file.empty()) {
        if (Mpi::Root()) {
            std::cerr << "Usage: clip_field --input X.bp --output Y.bp "
                      << "--mode percentile|absolute|of_max "
                      << "[--percentile P | --value V | --fraction F]\n"
                      << "  [--config C.yaml --restrict-attr IDs]\n";
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

    MaterialField field = LoadFieldBP(input_file, "data", comm);
    const int n = field.Size();
    real_t* h = field.Data().HostReadWrite();

    // When restricting, build an active-DOF mask in element-local space so
    // the percentile sample (and clip itself) ignore attr-out DOFs. This
    // prevents the threshold from being skewed by the (zero-valued) attr-out
    // population that already got masked upstream.
    bool has_restrict = !restrict_attrs.empty();
    std::vector<unsigned char> active_local(n, 1);  // 1 = active (default)
    if (has_restrict) {
        YamlConfig cfg(config_file);
        if (!cfg.IsValid()) {
            if (Mpi::Root()) {
                std::cerr << "Config error: " << cfg.GetValidationError() << "\n";
            }
            return 1;
        }
        int order = cfg.GetOrder();
        int dim = cfg.GetDimension();
        std::unique_ptr<ParMesh> pmesh(LoadParMesh(cfg, comm));
        pmesh->SetCurvature(order, true, dim, Ordering::byNODES);
        H1_FECollection fec(order, dim, BasisType::GaussLobatto);
        ParFiniteElementSpace fes(pmesh.get(), &fec);
        auto kept = BuildKeptDofMask(fes, *pmesh, restrict_attrs);

        // Round-trip a 0/1 marker through MaterialField to obtain the
        // active mask in the same element-local layout as `field`.
        ParGridFunction gf_mark(&fes);
        for (int i = 0; i < gf_mark.Size(); i++) gf_mark(i) = kept[i] ? 1.0 : 0.0;
        MaterialField marker = MaterialField::FromParGridFunction(fes, gf_mark);
        const real_t* marker_h = marker.Data().HostRead();
        for (int i = 0; i < n; i++) {
            active_local[i] = (marker_h[i] > 0.5) ? 1 : 0;
        }
    }

    // Build a contiguous host buffer of active values for threshold computation.
    std::vector<real_t> active_buf;
    active_buf.reserve(n);
    for (int i = 0; i < n; i++) if (active_local[i]) active_buf.push_back(h[i]);
    const int n_active = static_cast<int>(active_buf.size());
    const real_t* active_data = active_buf.data();

    real_t threshold = 0;
    if (mode == "percentile") {
        threshold = ComputePercentileThreshold(active_data, n_active, percentile, comm);
    } else if (mode == "absolute") {
        threshold = std::abs(abs_value);
    } else if (mode == "of_max") {
        const real_t max_abs = ComputeMaxAbs(active_data, n_active, comm);
        threshold = fraction * max_abs;
    } else {
        if (Mpi::Root())
            std::cerr << "clip_field: unknown --mode '" << mode << "'\n";
        return 2;
    }

    int n_clipped = 0;
    if (threshold > 0) {
        for (int i = 0; i < n; i++) {
            if (!active_local[i]) continue;
            if (h[i] >  threshold) { h[i] =  threshold; n_clipped++; }
            else if (h[i] < -threshold) { h[i] = -threshold; n_clipped++; }
        }
    }

    int total_clipped = 0;
    int total_n = 0;
    MPI_Reduce(&n_clipped, &total_clipped, 1, MPI_INT, MPI_SUM, 0, comm);
    MPI_Reduce(&n, &total_n, 1, MPI_INT, MPI_SUM, 0, comm);

    if (Mpi::Root()) {
        std::cout << "clip_field: mode=" << mode
                  << " threshold=" << std::scientific << std::setprecision(3) << threshold
                  << " clipped " << total_clipped << "/" << total_n << " DOFs\n";
    }

    SaveFieldBP(output_file, "data", field, /*mesh_path=*/input_file, comm);
    return 0;
}
