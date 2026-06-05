/**
 * @file refine_mesh.cpp
 * @brief MPI-parallel mesh refinement tool using YAML config and FromConfig() material pipeline
 *
 * Reads a YAML config (same format as test_forward_simulation) to load mesh
 * and material, computes required element size from min velocity, max_freq,
 * PPW, and polynomial order, then applies conforming uniform refinement
 * via ParMesh::UniformRefinement().
 *
 * Usage:
 *   mpirun -np N ./refine_mesh --config config.yaml [options]
 */

#include <mfem.hpp>
#include "config/YamlConfig.hpp"
#include "config/ConfigLoaders.hpp"
#include "config/ConfigTypes.hpp"
#include "common/Types.hpp"
#include "material/isotropic_elastic/IsotropicElasticMaterial.hpp"
#include "material/isotropic_acoustic/IsotropicAcousticMaterial.hpp"
#include "material/MaterialBase.hpp"

#include <fstream>
#include <functional>
#include <iostream>
#include <cmath>
#include <map>
#include <algorithm>
#include <sys/stat.h>

using namespace mfem;
using namespace SEM;

// =============================================================================
// Helpers (ComputeElementMinGLLDistance copied from SimulationFacade.cpp)
// =============================================================================

template<int Dim>
real_t ComputeElementMinGLLDistance(ParMesh& mesh, int e, int order) {
    const int ngll = order + 1;
    ElementTransformation* T = mesh.GetElementTransformation(e);
    static IntegrationRules gll_rules(0, Quadrature1D::GaussLobatto);
    const IntegrationRule& ir = gll_rules.Get(
        mesh.GetElementGeometry(e), 2 * order - 1);
    DenseMatrix pts(Dim, ir.GetNPoints());
    T->Transform(ir, pts);

    real_t min_dist_sq = std::numeric_limits<real_t>::max();

    if constexpr (Dim == 2) {
        for (int j = 0; j < ngll; j++) {
            for (int i = 0; i < ngll; i++) {
                int idx = i + j * ngll;
                if (i < ngll - 1) {
                    int idx_x = (i + 1) + j * ngll;
                    real_t dist_sq = 0.0;
                    for (int d = 0; d < Dim; d++) {
                        real_t diff = pts(d, idx_x) - pts(d, idx);
                        dist_sq += diff * diff;
                    }
                    min_dist_sq = std::min(min_dist_sq, dist_sq);
                }
                if (j < ngll - 1) {
                    int idx_y = i + (j + 1) * ngll;
                    real_t dist_sq = 0.0;
                    for (int d = 0; d < Dim; d++) {
                        real_t diff = pts(d, idx_y) - pts(d, idx);
                        dist_sq += diff * diff;
                    }
                    min_dist_sq = std::min(min_dist_sq, dist_sq);
                }
            }
        }
    } else {
        for (int k = 0; k < ngll; k++) {
            for (int j = 0; j < ngll; j++) {
                for (int i = 0; i < ngll; i++) {
                    int idx = i + j * ngll + k * ngll * ngll;
                    if (i < ngll - 1) {
                        int idx_x = (i + 1) + j * ngll + k * ngll * ngll;
                        real_t dist_sq = 0.0;
                        for (int d = 0; d < Dim; d++) {
                            real_t diff = pts(d, idx_x) - pts(d, idx);
                            dist_sq += diff * diff;
                        }
                        min_dist_sq = std::min(min_dist_sq, dist_sq);
                    }
                    if (j < ngll - 1) {
                        int idx_y = i + (j + 1) * ngll + k * ngll * ngll;
                        real_t dist_sq = 0.0;
                        for (int d = 0; d < Dim; d++) {
                            real_t diff = pts(d, idx_y) - pts(d, idx);
                            dist_sq += diff * diff;
                        }
                        min_dist_sq = std::min(min_dist_sq, dist_sq);
                    }
                    if (k < ngll - 1) {
                        int idx_z = i + j * ngll + (k + 1) * ngll * ngll;
                        real_t dist_sq = 0.0;
                        for (int d = 0; d < Dim; d++) {
                            real_t diff = pts(d, idx_z) - pts(d, idx);
                            dist_sq += diff * diff;
                        }
                        min_dist_sq = std::min(min_dist_sq, dist_sq);
                    }
                }
            }
        }
    }
    return std::sqrt(min_dist_sq);
}

bool CreateDirectory(const std::string& path) {
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

void PrintUsage(const char* prog) {
    std::cout << "Usage: mpirun -np N " << prog << " --config config.yaml --output-dir <dir> [options]\n\n"
              << "Required:\n"
              << "  --config <file>        YAML config file (mesh, material, order, mesh.max_freq, mesh.ppw)\n"
              << "  --output-dir <dir>     Output directory for partitioned files (mesh.mesh.NNNNNN)\n"
              << "                         Required for parallel runs (np > 1)\n\n"
              << "Optional:\n"
              << "  -n <max_ref>           Override max refinement levels\n"
              << "  --save-as-one <file>   Also save combined single mesh for visualization\n"
              << "  -h, --help             Show this help\n\n"
              << "Refinement criteria:\n"
              << "  h_required = v_min * ngll / (max_freq * ppw)   [ngll = order+1]\n"
              << "  n_refinements = ceil(log2(h_current / h_required))\n\n"
              << "Examples:\n"
              << "  # Serial (single process)\n"
              << "  " << prog << " --config config.yaml --output-dir MESH/\n\n"
              << "  # Parallel\n"
              << "  mpirun -np 8 " << prog << " --config config.yaml --output-dir MESH/\n\n"
              << "  # With visualization mesh\n"
              << "  mpirun -np 8 " << prog << " --config config.yaml --output-dir MESH/ --save-as-one refined.mesh\n";
}

// =============================================================================
// Build material and compute refinement levels
// =============================================================================

template<int Dim>
std::unique_ptr<MaterialBase> BuildMaterial(
    const YamlConfig& config, ParMesh& mesh, int order) {

    // Create FESpace (scalar, for material interpolation)
    H1_FECollection fec(order, Dim);
    ParFiniteElementSpace fes(&mesh, &fec);

    // Create GLL integration rule
    IntegrationRules gll_rules(0, Quadrature1D::GaussLobatto);
    const int exact = 2 * order - 1;
    IntegrationRule ir_1d = gll_rules.Get(Geometry::SEGMENT, exact);
    IntegrationRule ir;
    if constexpr (Dim == 2) {
        ir = IntegrationRule(ir_1d, ir_1d);
    } else {
        ir = IntegrationRule(ir_1d, ir_1d, ir_1d);
    }

    // Load material config
    MaterialConfig mat_config = LoadMaterialConfig(config);
    MaterialType mat_type = StringToMaterialType(mat_config.material_type);

    // Build material via FromConfig()
    if constexpr (Dim == 2) {
        if (mat_type == MaterialType::IsotropicElastic) {
            auto mat = IsotropicElasticMaterial::FromConfig(mat_config, fes, ir);
            return mat;
        } else {
            auto mat = IsotropicAcousticMaterial::FromConfig(mat_config, fes, ir);
            return mat;
        }
    } else {
        if (mat_type == MaterialType::IsotropicElastic) {
            auto mat = IsotropicElasticMaterial3D::FromConfig(mat_config, fes, ir);
            return mat;
        } else {
            auto mat = IsotropicAcousticMaterial3D::FromConfig(mat_config, fes, ir);
            return mat;
        }
    }
}

/// Per-element velocity callback. Lets `ComputeRefinementLevels` and
/// `PrintPostRefinementStats` source velocities from either a single
/// `MaterialBase` (pure-physics path) or from a small attribute→velocity
/// dispatcher (coupled path, where the parent mesh mixes acoustic and
/// elastic elements).
using ElementVelocity = std::function<real_t(int /*elem*/)>;

template<int Dim>
int ComputeRefinementLevels(ParMesh& mesh, const ElementVelocity& v_min_fn,
                             int order, real_t max_freq, real_t ppw,
                             MPI_Comm comm) {
    int ne = mesh.GetNE();
    int local_n_ref = 0;

    for (int e = 0; e < ne; e++) {
        real_t h = mesh.GetElementSize(e, 2);  // h_max
        real_t v_min = v_min_fn(e);
        int ngll = order + 1;
        real_t h_req = v_min * ngll / (max_freq * ppw);
        if (h > h_req) {
            int n_ref = static_cast<int>(std::ceil(std::log2(h / h_req)));
            local_n_ref = std::max(local_n_ref, n_ref);
        }
    }

    int global_n_ref = 0;
    MPI_Allreduce(&local_n_ref, &global_n_ref, 1, MPI_INT, MPI_MAX, comm);
    return global_n_ref;
}

template<int Dim>
void PrintPostRefinementStats(ParMesh& mesh,
                               const ElementVelocity& v_min_fn,
                               const ElementVelocity& v_max_fn,
                               int order, real_t max_freq, real_t ppw,
                               real_t cfl_factor, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    int ne = mesh.GetNE();
    int ngll = order + 1;

    real_t local_h_min = std::numeric_limits<real_t>::max();
    real_t local_h_max = 0.0;
    real_t local_dt_min = std::numeric_limits<real_t>::max();
    real_t local_ppw_min = std::numeric_limits<real_t>::max();

    for (int e = 0; e < ne; e++) {
        real_t h = mesh.GetElementSize(e, 2);
        local_h_min = std::min(local_h_min, mesh.GetElementSize(e, 1));
        local_h_max = std::max(local_h_max, h);

        real_t gll_dist = ComputeElementMinGLLDistance<Dim>(mesh, e, order);
        real_t v_max = v_max_fn(e);
        real_t v_min = v_min_fn(e);

        real_t dt_elem = cfl_factor * gll_dist / v_max;
        local_dt_min = std::min(local_dt_min, dt_elem);

        real_t ppw_elem = ngll * v_min / (max_freq * h);
        local_ppw_min = std::min(local_ppw_min, ppw_elem);
    }

    real_t global_h_min, global_h_max, global_dt_min, global_ppw_min;
    MPI_Allreduce(&local_h_min, &global_h_min, 1, MFEM_MPI_REAL_T, MPI_MIN, comm);
    MPI_Allreduce(&local_h_max, &global_h_max, 1, MFEM_MPI_REAL_T, MPI_MAX, comm);
    MPI_Allreduce(&local_dt_min, &global_dt_min, 1, MFEM_MPI_REAL_T, MPI_MIN, comm);
    MPI_Allreduce(&local_ppw_min, &global_ppw_min, 1, MFEM_MPI_REAL_T, MPI_MIN, comm);

    // Global element count
    long long local_ne = ne;
    long long global_ne = 0;
    MPI_Allreduce(&local_ne, &global_ne, 1, MPI_LONG_LONG, MPI_SUM, comm);

    if (rank == 0) {
        std::cout << "\nRefined mesh:\n";
        std::cout << "  Elements (global): " << global_ne << "\n";
        std::cout << "  h range:           [" << global_h_min << ", " << global_h_max << "]\n";
        std::cout << "  PPW (worst):       " << global_ppw_min << " (target: " << ppw << ")\n";
        std::cout << "  CFL dt_max:        " << global_dt_min << " s (CFL=" << cfl_factor << ")\n";
    }
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    Mpi::Init(argc, argv);
    int rank = Mpi::WorldRank();
    int nprocs = Mpi::WorldSize();

    // --- Parse arguments ---
    std::string config_file;
    std::string output_dir;
    std::string save_as_one_file;
    int max_ref_override = -1;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "--config") && i + 1 < argc) {
            config_file = argv[++i];
        } else if ((arg == "--output-dir") && i + 1 < argc) {
            output_dir = argv[++i];
        } else if ((arg == "--save-as-one") && i + 1 < argc) {
            save_as_one_file = argv[++i];
        } else if ((arg == "-n") && i + 1 < argc) {
            max_ref_override = std::atoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            if (rank == 0) PrintUsage(argv[0]);
            Mpi::Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::cerr << "Unknown option: " << arg << std::endl;
                PrintUsage(argv[0]);
            }
            Mpi::Finalize();
            return 1;
        }
    }

    if (config_file.empty()) {
        if (rank == 0) {
            std::cerr << "Error: --config is required.\n\n";
            PrintUsage(argv[0]);
        }
        Mpi::Finalize();
        return 1;
    }

    // Parallel run requires --output-dir
    if (nprocs > 1 && output_dir.empty()) {
        if (rank == 0) {
            std::cerr << "Error: --output-dir is required for parallel runs (np > 1).\n\n";
            PrintUsage(argv[0]);
        }
        Mpi::Finalize();
        return 1;
    }

    // Default: save as single mesh if no output dir specified (serial only)
    if (output_dir.empty() && save_as_one_file.empty()) {
        save_as_one_file = "refined.mesh";
    }

    // --- Load config ---
    if (rank == 0) std::cout << "Loading config: " << config_file << std::endl;
    YamlConfig config(config_file);
    if (!config.IsValid()) {
        if (rank == 0) {
            std::cerr << "Error: Invalid config: " << config.GetValidationError() << std::endl;
        }
        Mpi::Finalize();
        return 1;
    }

    int order = config.GetOrder();
    int dim = config.GetDimension();
    real_t max_freq = config.GetMaxFreq();
    real_t ppw = config.GetPPW();
    real_t cfl_factor = config.GetCflFactor();

    if (rank == 0) {
        std::cout << "  Dimension:  " << dim << "\n";
        std::cout << "  Order:      " << order << "\n";
        std::cout << "  Max freq:   " << max_freq << " Hz\n";
        std::cout << "  PPW target: " << ppw << "\n";
    }

    // --- Load mesh ---
    if (rank == 0) std::cout << "\nLoading mesh..." << std::endl;
    std::unique_ptr<ParMesh> pmesh(LoadParMesh(config, MPI_COMM_WORLD));
    pmesh->SetCurvature(order, true, dim, Ordering::byNODES);

    long long local_ne = pmesh->GetNE();
    long long global_ne = 0;
    MPI_Allreduce(&local_ne, &global_ne, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "  Elements (global): " << global_ne << "\n";
    }

    // --- Velocity providers (single-material vs coupled) -------------
    // Both paths end up supplying per-element `v_min(e)` / `v_max(e)`
    // callbacks to the refinement-level computation and the post-
    // refinement stats printer. For coupled configs we skip the full
    // `MaterialBase` path entirely — coupled's MaterialConfig isn't a
    // valid scalar material, so calling `BuildMaterial` on it would
    // fall through to `IsotropicAcousticMaterial::FromConfig` and
    // crash. Instead we dispatch by element attribute against the two
    // sub-configs.
    const bool is_coupled = (config.GetMaterialType() == "coupled");

    // single-material state — only populated on the non-coupled path.
    std::unique_ptr<MaterialBase> material;
    // coupled state — only populated on the coupled path. Stored in
    // `main` scope so the `v_*` closures stay valid across
    // `UniformRefinement`.
    CoupledMaterialConfig cc_cfg;
    real_t cc_vp_f = 0.0, cc_vp_s = 0.0, cc_vs_s = 0.0;
    int    cc_fluid_attr = -1;

    ElementVelocity v_min_fn, v_max_fn;
    real_t v_min_global = 0.0, v_max_global = 0.0;

    if (is_coupled) {
        if (rank == 0) {
            std::cout << "\nLoading coupled material config..." << std::endl;
        }
        cc_cfg = LoadCoupledMaterialConfig(config);
        // Fluid side is acoustic → vp only. Solid side is elastic →
        // vp (CFL/dt) and vs (wavelength/PPW).
        cc_vp_f = cc_cfg.fluid.params.at("vp");
        cc_vp_s = cc_cfg.solid.params.at("vp");
        cc_vs_s = cc_cfg.solid.params.at("vs");
        cc_fluid_attr = cc_cfg.fluid_attribute;

        v_min_global = std::min(cc_vp_f, cc_vs_s);
        v_max_global = std::max(cc_vp_f, cc_vp_s);

        // Capture `pmesh` by raw pointer — it outlives the closure
        // (the `unique_ptr` itself stays in main's scope).
        ParMesh* pm = pmesh.get();
        const int fa = cc_fluid_attr;
        const real_t vpf = cc_vp_f, vps = cc_vp_s, vss = cc_vs_s;
        v_min_fn = [pm, fa, vpf, vss](int e) -> real_t {
            return (pm->GetAttribute(e) == fa) ? vpf : vss;
        };
        v_max_fn = [pm, fa, vpf, vps](int e) -> real_t {
            return (pm->GetAttribute(e) == fa) ? vpf : vps;
        };

        if (rank == 0) {
            std::cout << "  fluid_attr=" << cc_fluid_attr
                      << " vp=" << cc_vp_f << "\n";
            std::cout << "  solid_attr=" << cc_cfg.solid_attribute
                      << " vp=" << cc_vp_s << " vs=" << cc_vs_s << "\n";
            std::cout << "  V_min (global): " << v_min_global << "\n";
            std::cout << "  V_max (global): " << v_max_global << "\n";
        }
    } else {
        if (rank == 0) std::cout << "\nLoading material..." << std::endl;
        if (dim == 2) {
            material = BuildMaterial<2>(config, *pmesh, order);
        } else {
            material = BuildMaterial<3>(config, *pmesh, order);
        }
        real_t v_min_local = material->GetMinVelocity();
        real_t v_max_local = material->GetMaxVelocity();
        MPI_Allreduce(&v_min_local, &v_min_global, 1, MFEM_MPI_REAL_T,
                      MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&v_max_local, &v_max_global, 1, MFEM_MPI_REAL_T,
                      MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "  Type: " << config.GetMaterialType() << "\n";
            std::cout << "  V_min (global): " << v_min_global << "\n";
            std::cout << "  V_max (global): " << v_max_global << "\n";
        }
        MaterialBase* m = material.get();
        v_min_fn = [m](int e) -> real_t {
            return m->GetElementMinVelocity(e);
        };
        v_max_fn = [m](int e) -> real_t {
            return m->GetElementMaxVelocity(e);
        };
    }

    // --- Compute refinement levels ---
    int n_ref;
    if (dim == 2) {
        n_ref = ComputeRefinementLevels<2>(*pmesh, v_min_fn, order,
                                            max_freq, ppw, MPI_COMM_WORLD);
    } else {
        n_ref = ComputeRefinementLevels<3>(*pmesh, v_min_fn, order,
                                            max_freq, ppw, MPI_COMM_WORLD);
    }

    if (max_ref_override >= 0) {
        n_ref = max_ref_override;
    }

    if (rank == 0) {
        std::cout << "\nRefinement levels needed: " << n_ref << "\n";
    }

    if (n_ref == 0) {
        if (rank == 0) {
            std::cout << "Mesh already satisfies PPW criterion. No refinement needed.\n";
        }
    }

    // --- Apply parallel uniform refinement ---
    // The single-material path holds a MaterialBase whose FESpace is
    // bound to the pre-refinement mesh; it must be released and
    // rebuilt around UniformRefinement. The coupled path uses plain
    // per-attribute velocity constants (no FESpace involved), so the
    // closures remain valid across refinement without touching them.
    if (!is_coupled) {
        material.reset();
        v_min_fn = nullptr;
        v_max_fn = nullptr;
    }

    for (int r = 0; r < n_ref; r++) {
        pmesh->UniformRefinement();
        local_ne = pmesh->GetNE();
        MPI_Allreduce(&local_ne, &global_ne, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "  Refinement " << (r + 1) << ": " << global_ne << " elements (global)\n";
        }
    }

    // --- Rebuild material on refined mesh for statistics ---
    if (n_ref > 0) {
        pmesh->SetCurvature(order, true, dim, Ordering::byNODES);
    }

    if (!is_coupled) {
        if (dim == 2) {
            material = BuildMaterial<2>(config, *pmesh, order);
        } else {
            material = BuildMaterial<3>(config, *pmesh, order);
        }
        MaterialBase* m = material.get();
        v_min_fn = [m](int e) -> real_t { return m->GetElementMinVelocity(e); };
        v_max_fn = [m](int e) -> real_t { return m->GetElementMaxVelocity(e); };
    }

    // --- Post-refinement statistics ---
    if (dim == 2) {
        PrintPostRefinementStats<2>(*pmesh, v_min_fn, v_max_fn, order,
                                     max_freq, ppw, cfl_factor,
                                     MPI_COMM_WORLD);
    } else {
        PrintPostRefinementStats<3>(*pmesh, v_min_fn, v_max_fn, order,
                                     max_freq, ppw, cfl_factor,
                                     MPI_COMM_WORLD);
    }

    // --- Output ---
    // SaveAsOne: rank 0 writes single combined mesh
    if (!save_as_one_file.empty()) {
        if (rank == 0) std::cout << "\nSaving (SaveAsOne): " << save_as_one_file << std::endl;
        pmesh->SaveAsOne(save_as_one_file);
    }

    // Partitioned output: each rank saves via ParPrint() (parallel MFEM format)
    // ParPrint() includes communication_groups, readable by ParMesh(comm, istream)
    if (!output_dir.empty()) {
        if (rank == 0) {
            CreateDirectory(output_dir);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            std::cout << "\nSaving partitioned mesh to: " << output_dir << "/\n";
            std::cout << "  " << nprocs << " partition files (mesh.mesh.NNNNNN)\n";
        }

        // Write partition files in batches to avoid Lustre MDS overload
        // at large rank counts (2000+)
        const int batch_size = 256;
        int num_batches = (nprocs + batch_size - 1) / batch_size;
        for (int batch = 0; batch < num_batches; batch++) {
            int batch_start = batch * batch_size;
            int batch_end = std::min(batch_start + batch_size, nprocs);

            if (rank >= batch_start && rank < batch_end) {
                std::ostringstream fname;
                fname << output_dir << "/mesh.mesh."
                      << std::setfill('0') << std::setw(6) << rank;
                std::ofstream ofs(fname.str());
                pmesh->ParPrint(ofs);
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        // Write partition_info.yaml (rank 0 only)
        if (rank == 0) {
            std::string info_file = output_dir + "/partition_info.yaml";
            std::ofstream info_ofs(info_file);
            info_ofs << "# Partition info generated by refine_mesh tool\n";
            info_ofs << "partition:\n";
            info_ofs << "  nparts: " << nprocs << "\n";
            info_ofs << "  method: refine\n";
            info_ofs << "\nmesh:\n";
            info_ofs << "  dimension: " << dim << "\n";
            info_ofs << "  num_elements_global: " << global_ne << "\n";
            info_ofs << "  refinement_levels: " << n_ref << "\n";
            info_ofs << "  source_config: " << config_file << "\n";
            info_ofs.close();
            std::cout << "  Wrote: " << info_file << "\n";
        }
    }

    if (rank == 0) std::cout << "\nDone.\n";

    Mpi::Finalize();
    return 0;
}
