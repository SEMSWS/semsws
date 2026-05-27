/**
 * @file mask_sponge.cpp
 * @brief Zero out field values in sponge absorbing boundary regions
 *
 * Reads a BP field, identifies DOFs within the sponge layer using the same
 * distance-based criteria as CerjanABC, zeros them out, and saves the result.
 *
 * Usage:
 *   mpirun -np N mask_sponge \
 *       --config sim.yaml \
 *       --input kernel.bp \
 *       --output masked.bp \
 *       --device cuda
 *
 *   mpirun -np N mask_sponge \
 *       --config sim.yaml \
 *       --input kernel.bp \
 *       --overwrite
 */

#include "io/ADIOS2IO.hpp"
#include "material/MaterialField.hpp"
#include "config/YamlConfig.hpp"
#include "config/ConfigLoaders.hpp"
#include "common/BoundaryUtils.hpp"
#include <mfem.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <set>
#include <vector>

using namespace SEM;
using namespace mfem;

int main(int argc, char* argv[]) {
    Mpi::Init(argc, argv);
    Hypre::Init();

    // Parse arguments
    std::string config_file, input_file, output_file;
    std::string device_str = "cpu";
    std::string keep_attr_str;
    bool overwrite = false;
    bool taper = false;
    bool no_sponge = false;
    std::vector<int> keep_attrs;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) config_file = argv[++i];
        else if (arg == "--input" && i + 1 < argc) input_file = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_file = argv[++i];
        else if (arg == "--overwrite") overwrite = true;
        else if (arg == "--taper") taper = true;
        else if (arg == "--no-sponge") no_sponge = true;
        else if (arg == "--device" && i + 1 < argc) device_str = argv[++i];
        else if (arg == "--keep-attr" && i + 1 < argc) {
            keep_attr_str = argv[++i];
            std::stringstream ss(keep_attr_str);
            std::string token;
            while (std::getline(ss, token, ',')) {
                if (!token.empty()) keep_attrs.push_back(std::stoi(token));
            }
        }
    }

    // Validate arguments
    if (config_file.empty() || input_file.empty()) {
        if (Mpi::Root()) {
            std::cerr << "Usage: mask --config C.yaml --input X.bp "
                      << "[--output Y.bp | --overwrite] [--taper|--no-sponge] "
                      << "[--keep-attr ID1,ID2,...] [--device cpu|cuda]\n"
                      << "  --taper:     apply cosine taper instead of zeroing in sponge\n"
                      << "  --no-sponge: skip sponge masking entirely (keep_attrs still applied)\n"
                      << "  --keep-attr: zero DOFs whose containing element attribute "
                                     "is NOT in the list (combines with sponge mask)\n";
        }
        return 1;
    }

    if (output_file.empty() && !overwrite) {
        if (Mpi::Root()) {
            std::cerr << "Error: Must specify --output or --overwrite\n";
        }
        return 1;
    }

    if (overwrite) {
        output_file = input_file;
    }

    Device device(device_str);

    MPI_Comm comm = MPI_COMM_WORLD;

    // Load config
    YamlConfig config(config_file);
    if (!config.IsValid()) {
        if (Mpi::Root()) {
            std::cerr << "Config error: " << config.GetValidationError()
                      << std::endl;
        }
        return 1;
    }

    // Read absorbing boundary config
    ABCConfig abc = config.GetABCConfig();
    const bool sponge_active = !no_sponge && !abc.sides.empty() && abc.thickness > 0.0;
    if (!sponge_active && keep_attrs.empty()) {
        if (Mpi::Root()) {
            std::cout << "Nothing to mask (no sponge, no --keep-attr)." << std::endl;
        }
        // Just copy input to output if different
        if (input_file != output_file) {
            MaterialField field = LoadFieldBP(input_file, "data", comm);
            SaveFieldBP(output_file, "data", field, config_file, comm);
        }
        return 0;
    }

    int order = config.GetOrder();
    int dim = config.GetDimension();

    // Determine active sides
    std::set<int> active_attrs;
    for (const auto& side : abc.sides) {
        int attr = ParseBoundarySide(side, dim);
        if (attr > 0) active_attrs.insert(attr);
    }

    // Load mesh
    std::unique_ptr<ParMesh> pmesh(LoadParMesh(config, comm));
    pmesh->SetCurvature(order, true, dim, Ordering::byNODES);

    H1_FECollection fec(order, dim, BasisType::GaussLobatto);
    ParFiniteElementSpace fes(pmesh.get(), &fec);

    // Load field
    if (Mpi::Root()) {
        std::cout << "Loading field: " << input_file << std::endl;
    }
    MaterialField mat_field = LoadFieldBP(input_file, "data", comm);

    ParGridFunction gf(&fes);
    mat_field.ToParGridFunction(fes, gf);

    // Compute global bounding box (same as CerjanABC)
    Vector bbmin, bbmax;
    pmesh->GetBoundingBox(bbmin, bbmax);

    Vector bbmin_g(dim), bbmax_g(dim);
    MPI_Allreduce(bbmin.GetData(), bbmin_g.GetData(), dim,
                  MPITypeMap<real_t>::mpi_type, MPI_MIN, comm);
    MPI_Allreduce(bbmax.GetData(), bbmax_g.GetData(), dim,
                  MPITypeMap<real_t>::mpi_type, MPI_MAX, comm);

    real_t thickness = sponge_active ? abc.thickness : real_t(0.0);

    // Determine active boundaries by dimension
    bool has_bottom = false, has_top = false;
    bool has_left = false, has_right = false;
    bool has_front = false, has_back = false;

    if (sponge_active) {
        if (dim == 2) {
            has_bottom = active_attrs.count(1);
            has_right  = active_attrs.count(2);
            has_top    = active_attrs.count(3);
            has_left   = active_attrs.count(4);
        } else {
            has_bottom = active_attrs.count(1);
            has_front  = active_attrs.count(2);
            has_right  = active_attrs.count(3);
            has_back   = active_attrs.count(4);
            has_left   = active_attrs.count(5);
            has_top    = active_attrs.count(6);
        }
    }

    // Mask or taper DOFs in sponge region
    int local_masked = 0;
    std::set<int> masked_dofs;

    if (sponge_active)
    for (int ie = 0; ie < fes.GetNE(); ie++) {
        const FiniteElement* fe = fes.GetFE(ie);
        const IntegrationRule& nodes = fe->GetNodes();
        Array<int> dofs;
        fes.GetElementDofs(ie, dofs);

        for (int j = 0; j < nodes.GetNPoints(); j++) {
            Vector pt;
            fes.GetElementTransformation(ie)->Transform(
                nodes.IntPoint(j), pt);

            // Compute minimum distance from any active sponge boundary
            // d = 0 at outer boundary, d = thickness at inner edge
            real_t d = thickness;

            if (dim == 2) {
                if (has_left)   d = std::min(d, pt(0) - bbmin_g(0));
                if (has_right)  d = std::min(d, bbmax_g(0) - pt(0));
                if (has_bottom) d = std::min(d, pt(1) - bbmin_g(1));
                if (has_top)    d = std::min(d, bbmax_g(1) - pt(1));
            } else {
                if (has_left)   d = std::min(d, pt(0) - bbmin_g(0));
                if (has_right)  d = std::min(d, bbmax_g(0) - pt(0));
                if (has_front)  d = std::min(d, pt(1) - bbmin_g(1));
                if (has_back)   d = std::min(d, bbmax_g(1) - pt(1));
                if (has_bottom) d = std::min(d, pt(2) - bbmin_g(2));
                if (has_top)    d = std::min(d, bbmax_g(2) - pt(2));
            }

            bool in_sponge = (d < thickness);

            if (in_sponge) {
                int dof = dofs[j];
                if (dof < 0) dof = -1 - dof;
                if (masked_dofs.insert(dof).second) {
                    if (taper) {
                        // Cosine taper: 0 at boundary, 1 at inner edge
                        real_t d_clamped = (d > 0.0) ? d : real_t(0.0);
                        real_t w = 0.5 * (1.0 - std::cos(M_PI * d_clamped / thickness));
                        gf(dof) *= w;
                    } else {
                        gf(dof) = 0.0;
                    }
                    local_masked++;
                }
            }
        }
    }

    // Report sponge statistics
    if (sponge_active) {
        int total_masked = 0;
        MPI_Reduce(&local_masked, &total_masked, 1, MPI_INT, MPI_SUM, 0, comm);
        if (Mpi::Root()) {
            std::cout << "Sponge " << (taper ? "taper" : "mask")
                      << ": " << (taper ? "tapered " : "zeroed ")
                      << total_masked << " DOFs"
                      << " (thickness=" << thickness << " m, sides:";
            for (const auto& s : abc.sides) std::cout << " " << s;
            std::cout << ")" << std::endl;
        }
    }

    // Attribute keep-mask: zero DOFs whose containing element attribute is
    // NOT in the keep list. Conservative: a DOF is kept if ANY containing
    // element has an attribute in the keep set (so DOFs on attribute
    // boundaries are preserved).
    if (!keep_attrs.empty()) {
        std::set<int> keep_set(keep_attrs.begin(), keep_attrs.end());
        std::vector<unsigned char> kept(gf.Size(), 0);
        for (int ie = 0; ie < fes.GetNE(); ie++) {
            int attr = pmesh->GetAttribute(ie);
            if (!keep_set.count(attr)) continue;
            Array<int> dofs;
            fes.GetElementDofs(ie, dofs);
            for (int j = 0; j < dofs.Size(); j++) {
                int d = dofs[j];
                if (d < 0) d = -1 - d;
                kept[d] = 1;
            }
        }
        int local_zeroed = 0;
        for (int i = 0; i < gf.Size(); i++) {
            if (!kept[i] && gf(i) != 0.0) {
                gf(i) = 0.0;
                local_zeroed++;
            }
        }
        int total_zeroed = 0;
        MPI_Reduce(&local_zeroed, &total_zeroed, 1, MPI_INT, MPI_SUM, 0, comm);
        if (Mpi::Root()) {
            std::cout << "Attribute mask: zeroed " << total_zeroed
                      << " DOFs outside attributes [" << keep_attr_str << "]"
                      << std::endl;
        }
    }

    // Convert back and save
    MaterialField masked = MaterialField::FromParGridFunction(fes, gf);

    if (Mpi::Root()) {
        std::cout << "Saving: " << output_file << std::endl;
    }
    SaveFieldBP(output_file, "data", masked, config_file, comm);

    if (Mpi::Root()) {
        std::cout << "Done." << std::endl;
    }

    return 0;
}
