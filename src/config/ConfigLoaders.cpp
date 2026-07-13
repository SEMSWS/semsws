/**
 * @file ConfigLoaders.cpp
 * @brief Implementation of configuration loaders for SEMSWS
 */

#include "config/ConfigLoaders.hpp"
#include "common/BoundaryUtils.hpp"
#include "common/Types.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace SEM {

// =============================================================================
// Dimension Validation Helpers
// =============================================================================

/**
 * @brief Check if source definitions have dimension mismatches
 *
 * For 2D simulations, warns if z-coordinate is non-zero (will be ignored).
 * For 3D simulations, warns if z-coordinate is zero for all sources (might be 2D config).
 */
static void ValidateSourceDimensions(const std::vector<SourceDef>& defs, int dim, int rank) {
    if (rank != 0) return;  // Only print from rank 0

    if (dim == 2) {
        // Check if any source has non-zero z-coordinate
        for (size_t i = 0; i < defs.size(); i++) {
            const auto& def = defs[i];
            if (std::abs(def.location[2]) > 1e-10) {
                std::cerr << "WARNING: Source " << i << " (" << def.name
                          << ") has z-coordinate " << def.location[2]
                          << " but simulation is 2D. Z-coordinate will be ignored.\n";
            }
            if (def.type == "force" && std::abs(def.direction[2]) > 1e-10) {
                std::cerr << "WARNING: Source " << i << " (" << def.name
                          << ") has z-direction " << def.direction[2]
                          << " but simulation is 2D. Z-direction will be ignored.\n";
            }
        }
    }
    // Note: For 3D, z=0 is a valid coordinate, so no warning needed
}

/**
 * @brief Check if receiver definitions have dimension mismatches
 *
 * For 2D simulations, warns if z-coordinate is non-zero (will be ignored).
 */
static void ValidateReceiverDimensions(const std::vector<ReceiverDef>& defs, int dim, int rank) {
    if (rank != 0) return;  // Only print from rank 0

    if (dim == 2) {
        for (size_t i = 0; i < defs.size(); i++) {
            const auto& def = defs[i];
            if (std::abs(def.location[2]) > 1e-10) {
                std::cerr << "WARNING: Receiver " << i << " (" << def.name
                          << ") has z-coordinate " << def.location[2]
                          << " but simulation is 2D. Z-coordinate will be ignored.\n";
            }
        }
    }
}

// =============================================================================
// Configuration Reading (Pure data extraction)
// =============================================================================

// =============================================================================
// Source Configuration Reading (Pure data extraction)
// =============================================================================

SourceConfig::Config2D LoadSourceConfig2D(const YamlConfig& config) {
    SourceConfig::Config2D result;
    std::vector<SourceDef> defs = config.GetAllSources();

    for (const auto& def : defs) {
        // Common wavelet config
        SourceConfig::WaveletConfig wavelet;
        wavelet.type = def.wavelet_type;
        wavelet.frequency = def.frequency;
        wavelet.amplitude = def.amplitude;
        wavelet.delay = def.delay;
        wavelet.external_file = def.external_file;
        wavelet.stf_samples = def.stf_samples;
        wavelet.stf_dt = def.stf_dt;

        if (def.type == "force") {
            SourceConfig::ForceSource src;
            src.id = def.id;
            src.location = {def.location[0], def.location[1]};
            src.direction = {def.direction[0], def.direction[1]};
            src.wavelet = wavelet;
            result.forces.push_back(src);
        }
        else if (def.type == "pressure") {
            SourceConfig::PressureSource src;
            src.id = def.id;
            src.location = {def.location[0], def.location[1]};
            src.wavelet = wavelet;
            result.pressures.push_back(src);
        }
        else if (def.type == "moment_tensor") {
            SourceConfig::MomentTensorSource2D src;
            src.id = def.id;
            src.location = {def.location[0], def.location[1]};
            src.Mxx = def.M[0];
            src.Myy = def.M[1];
            src.Mxy = def.M[3];  // M[3] is Mxy in the 6-component format
            src.wavelet = wavelet;
            result.moment_tensors.push_back(src);
        }
    }
    return result;
}

SourceConfig::Config3D LoadSourceConfig3D(const YamlConfig& config) {
    SourceConfig::Config3D result;
    std::vector<SourceDef> defs = config.GetAllSources();

    for (const auto& def : defs) {
        SourceConfig::WaveletConfig wavelet;
        wavelet.type = def.wavelet_type;
        wavelet.frequency = def.frequency;
        wavelet.amplitude = def.amplitude;
        wavelet.delay = def.delay;
        wavelet.external_file = def.external_file;
        wavelet.stf_samples = def.stf_samples;
        wavelet.stf_dt = def.stf_dt;

        if (def.type == "force") {
            SourceConfig::ForceSource src;
            src.id = def.id;
            src.location = {def.location[0], def.location[1], def.location[2]};
            src.direction = {def.direction[0], def.direction[1], def.direction[2]};
            src.wavelet = wavelet;
            result.forces.push_back(src);
        }
        else if (def.type == "pressure") {
            SourceConfig::PressureSource src;
            src.id = def.id;
            src.location = {def.location[0], def.location[1], def.location[2]};
            src.wavelet = wavelet;
            result.pressures.push_back(src);
        }
        else if (def.type == "moment_tensor") {
            SourceConfig::MomentTensorSource3D src;
            src.id = def.id;
            src.location = {def.location[0], def.location[1], def.location[2]};
            src.Mxx = def.M[0];
            src.Myy = def.M[1];
            src.Mzz = def.M[2];
            src.Mxy = def.M[3];
            src.Mxz = def.M[4];
            src.Myz = def.M[5];
            src.wavelet = wavelet;
            result.moment_tensors.push_back(src);
        }
    }
    return result;
}


// =============================================================================
// Material Configuration Reading (Pure data extraction)
// =============================================================================

CoupledMaterialConfig LoadCoupledMaterialConfig(const YamlConfig& config) {
    MFEM_VERIFY(config.IsCoupledMaterial(),
                "LoadCoupledMaterialConfig called but material.type != 'coupled'");
    return config.GetCoupledMaterialConfig();
}

MaterialConfig LoadMaterialConfig(const YamlConfig& config) {
    MaterialConfig mat_config;

    // Get material type first (isotropic_acoustic, isotropic_elastic, etc.)
    mat_config.material_type = config.GetMaterialType();

    // Get format (constant, grid, by_attribute, by_attribute_mixed, adios2)
    std::string format = config.GetMaterialFormat();
    mat_config.format = format;


    if (format == "constant") {
        // Read constant material values based on material type
        // Store in generic params map for flexibility with different material types
        auto type = StringToMaterialType(mat_config.material_type);
        if (type == MaterialType::IsotropicAcoustic) {
            real_t vp, rho;
            config.GetConstantMaterialVpRho(&vp, &rho);
            mat_config.params["vp"] = vp;
            mat_config.params["rho"] = rho;
        }
        else if (type == MaterialType::IsotropicElastic) {
            real_t vp, vs, rho;
            config.GetConstantMaterialVpVsRho(&vp, &vs, &rho);
            mat_config.params["vp"] = vp;
            mat_config.params["vs"] = vs;
            mat_config.params["rho"] = rho;
        }
        else {
            MFEM_ABORT("Constant material format not yet implemented for material type: "
                   + mat_config.material_type);
        }
    }
    
    else if (format == "grid") {
        // Get the unified material file path
        mat_config.material_file = config.GetMaterialFile();
    }
    else if (format == "by_attribute") {
        // by_attribute only reads from file
        mat_config.by_attribute_file = config.GetMaterialByAttributeFile();
    }
    else if (format == "by_attribute_mixed") {
        // by_attribute_mixed reads from external YAML file
        mat_config.by_attribute_file = config.GetMaterialByAttributeFile();
        LoadAttributeMixedFromYaml(mat_config.by_attribute_file, mat_config.attribute_entries);
    }
    else if (format == "adios2") {
        // ADIOS2 pre-computed GLL data (FWI iterations)
        mat_config.adios2_vp_file = config.GetADIOS2VpFile();
        mat_config.adios2_vs_file = config.GetADIOS2VsFile();
        mat_config.adios2_rho_file = config.GetADIOS2RhoFile();
        mat_config.adios2_qkappa_file = config.GetADIOS2QkappaFile();
        mat_config.adios2_qmu_file = config.GetADIOS2QmuFile();
    }

    // Load attenuation config
    // Note: enabled, f0, n_units apply to all formats
    // Q values (qkappa, qmu) are only read from YAML for constant format
    // For grid/by_attribute formats, Q values come from the material file
    mat_config.attenuation.enabled = config.IsAttenuationEnabled();
    if (mat_config.attenuation.enabled) {
        mat_config.attenuation.f0 = config.GetAttenuationF0();
        mat_config.attenuation.n_units = config.GetAttenuationNumUnits();

        // Q values only for constant format - for grid/by_attribute they come from file
        if (format == "constant") {
            mat_config.attenuation.qkappa = config.GetConstantQkappa();
            // Qmu only meaningful for elastic materials (acoustic ignores it)
            mat_config.attenuation.qmu = (mat_config.material_type == "isotropic_acoustic")
                ? -1.0 : config.GetConstantQmu();
        }
    }

    return mat_config;
}

// =============================================================================
// Mesh Loading
// =============================================================================

Mesh* LoadMesh(const YamlConfig& config) {
    std::string mesh_type = config.GetMeshType();
    int dim = config.GetDimension();

    if (mesh_type == "internal") {
        real_t origin[3], size[3];
        int elements[3];

        config.GetMeshOrigin(origin);
        config.GetMeshSize(size);
        config.GetMeshElements(elements);

        Mesh* mesh = nullptr;

        if (dim == 2) {
            mesh = new Mesh(Mesh::MakeCartesian2D(
                elements[0], elements[1],
                Element::QUADRILATERAL,
                true, size[0], size[1]));

            if (origin[0] != 0.0 || origin[1] != 0.0) {
                for (int vi = 0; vi < mesh->GetNV(); vi++) {
                    real_t* v = mesh->GetVertex(vi);
                    v[0] += origin[0];
                    v[1] += origin[1];
                }
            }
        } else {
            mesh = new Mesh(Mesh::MakeCartesian3D(
                elements[0], elements[1], elements[2],
                Element::HEXAHEDRON,
                size[0], size[1], size[2]));

            if (origin[0] != 0.0 || origin[1] != 0.0 || origin[2] != 0.0) {
                for (int vi = 0; vi < mesh->GetNV(); vi++) {
                    real_t* v = mesh->GetVertex(vi);
                    v[0] += origin[0];
                    v[1] += origin[1];
                    v[2] += origin[2];
                }
            }
        }

        return mesh;

    } else {
        std::string mesh_file = config.GetMeshFile();
        Mesh* mesh = new Mesh(mesh_file.c_str(), 1, 1);
        return mesh;
    }
}

// Helper function to get current memory usage (RSS) in MB
static double GetMemoryUsageMB() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) {
            long kb = 0;
            sscanf(line.c_str(), "VmRSS: %ld kB", &kb);
            return kb / 1024.0;
        }
    }
    return -1.0;
}

ParMesh* LoadParMeshFromPartition(const YamlConfig& config, MPI_Comm comm) {
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    double mem_before = GetMemoryUsageMB();

    // Get partition configuration
    std::string directory = config.GetPartitionDirectory();
    int nparts = config.GetPartitionCount();

    // Validate: nparts must equal nprocs
    MFEM_VERIFY(nparts == nprocs,
        "Partition count (" << nparts << ") must equal number of MPI ranks ("
        << nprocs << "). Please re-partition the mesh with -n " << nprocs);

    // Construct filename for this rank (e.g., mesh.mesh.000000)
    std::ostringstream oss;
    oss << directory << "/mesh.mesh." << std::setfill('0') << std::setw(6) << rank;
    std::string filename = oss.str();

    // Read file in batches to avoid overwhelming Lustre MDS
    // at large rank counts (2000+). Each rank reads its file into
    // a string buffer; ParMesh construction happens after all reads.
    const int batch_size = 256;
    int num_batches = (nprocs + batch_size - 1) / batch_size;
    std::string mesh_data;

    for (int batch = 0; batch < num_batches; batch++) {
        int batch_start = batch * batch_size;
        int batch_end = std::min(batch_start + batch_size, nprocs);

        if (rank >= batch_start && rank < batch_end) {
            std::ifstream ifs(filename);
            MFEM_VERIFY(ifs.good(),
                "Cannot open partition file: " << filename
                << " (rank " << rank << ")");
            std::ostringstream buf;
            buf << ifs.rdbuf();
            mesh_data = buf.str();
        }

        MPI_Barrier(comm);
    }

    // Construct ParMesh from buffered data (all ranks simultaneously)
    std::istringstream iss(mesh_data);
    ParMesh* pmesh = new ParMesh(comm, iss);

    // Report memory usage
    double mem_after = GetMemoryUsageMB();
    double mem_used = mem_after - mem_before;
    double max_mem, min_mem, sum_mem;
    MPI_Reduce(&mem_used, &max_mem, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&mem_used, &min_mem, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&mem_used, &sum_mem, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

    // if (rank == 0) {
    //     std::cout << "LoadParMeshFromPartition memory (MB): min=" << min_mem
    //               << ", max=" << max_mem << ", total=" << sum_mem
    //               << " [partitioned files, nprocs=" << nprocs << "]" << std::endl;
    // }

    return pmesh;
}

ParMesh* LoadParMesh(const YamlConfig& config, MPI_Comm comm) {
    // Check for pre-partitioned mesh first (most memory efficient)
    if (config.UsePartitionedMesh()) {
        return LoadParMeshFromPartition(config, comm);
    }

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    double mem_before = GetMemoryUsageMB();

    // Rank 0 distribution: Only rank 0 loads serial mesh, partitions,
    // and distributes via MPI. This saves memory for large meshes.
    // Based on https://github.com/mfem/mfem/pull/2669
    std::string mesh_data;

    if (rank == 0) {
        Mesh* serial_mesh = LoadMesh(config);

        std::string partition_method = config.GetMeshPartitionMethod();
        int* partitioning = nullptr;

        if (partition_method == "cartesian") {
            int nxyz[3] = {1, 1, 1};
            config.GetPartitionGrid(nxyz);

            int dim = serial_mesh->Dimension();
            int total = 1;
            for (int i = 0; i < dim; i++) total *= nxyz[i];
            MFEM_VERIFY(total == nprocs,
                "partition_grid product (" << total << ") must equal nprocs (" << nprocs << ")");

            partitioning = serial_mesh->CartesianPartitioning(nxyz);
        }

        MeshPartitioner partitioner(*serial_mesh, nprocs, partitioning);

        for (int r = 0; r < nprocs; r++) {
            MeshPart mesh_part;
            partitioner.ExtractPart(r, mesh_part);

            std::ostringstream oss;
            mesh_part.Print(oss);
            std::string data = oss.str();

            if (r == 0) {
                mesh_data = std::move(data);
            } else {
                int size = static_cast<int>(data.size());
                MPI_Send(&size, 1, MPI_INT, r, 0, comm);
                MPI_Send(data.data(), size, MPI_CHAR, r, 1, comm);
            }
        }

        delete[] partitioning;
        delete serial_mesh;
    } else {
        int size;
        MPI_Recv(&size, 1, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
        mesh_data.resize(size);
        MPI_Recv(&mesh_data[0], size, MPI_CHAR, 0, 1, comm, MPI_STATUS_IGNORE);
    }

    std::istringstream iss(mesh_data);
    ParMesh* pmesh = new ParMesh(comm, iss);

    // Report memory usage
    double mem_after = GetMemoryUsageMB();
    double mem_used = mem_after - mem_before;
    double max_mem, min_mem, sum_mem;
    MPI_Reduce(&mem_used, &max_mem, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&mem_used, &min_mem, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&mem_used, &sum_mem, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

    return pmesh;
}


// =============================================================================
// Utility Functions
// =============================================================================

// ParseBoundarySide is now in common/BoundaryUtils.hpp

Array<int> GetABCBoundaryAttributes(const ABCConfig& abc, int dim) {
    Array<int> attrs;

    for (const auto& side : abc.sides) {
        int attr = ParseBoundarySide(side, dim);
        if (attr > 0) {
            attrs.Append(attr);
        }
    }

    return attrs;
}


Array<int> GetDirichletBoundaryAttributes(const YamlConfig& config, int dim) {
    Array<int> attrs;

    std::vector<std::string> values = config.GetDirichletAttributes();

    for (const auto& value : values) {
        int attr = ParseBoundaryAttributeValue(value, dim);
        if (attr > 0) {
            attrs.Append(attr);
        }
    }

    return attrs;
}

Array<int> GetDirichletTrueDofs(ParFiniteElementSpace& fes,
                                const Array<int>& dirichlet_attrs) {
    Array<int> ess_tdof_list;

    if (dirichlet_attrs.Size() == 0) {
        return ess_tdof_list;
    }

    ParMesh* pmesh = fes.GetParMesh();
    int max_bdr_attr = pmesh->bdr_attributes.Max();

    // Create boundary marker array
    Array<int> ess_bdr(max_bdr_attr);
    ess_bdr = 0;

    // Mark Dirichlet boundaries
    for (int i = 0; i < dirichlet_attrs.Size(); i++) {
        int attr = dirichlet_attrs[i];
        if (attr > 0 && attr <= max_bdr_attr) {
            ess_bdr[attr - 1] = 1;  // Attributes are 1-based
        }
    }

    // Get essential true DOFs
    fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    return ess_tdof_list;
}

// =============================================================================
// Mixed Attribute YAML Loading
// =============================================================================

void LoadAttributeMixedFromYaml(
    const std::string& yaml_path,
    std::vector<AttributeMaterialEntry>& entries)
{
    YAML::Node root;
    try {
        root = YAML::LoadFile(yaml_path);
    } catch (const YAML::Exception& e) {
        MFEM_ABORT("Failed to load mixed attribute YAML file: " << yaml_path
                   << "\nError: " << e.what());
    }

    YAML::Node attrs = root["attributes"];
    MFEM_VERIFY(attrs && attrs.IsMap(),
        "Mixed attribute YAML must contain 'attributes' map: " << yaml_path);

    // Get directory of the YAML file for relative path resolution
    std::string yaml_dir;
    size_t last_slash = yaml_path.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        yaml_dir = yaml_path.substr(0, last_slash + 1);
    }

    entries.clear();

    for (auto it = attrs.begin(); it != attrs.end(); ++it) {
        AttributeMaterialEntry entry;
        entry.attribute = it->first.as<int>();

        YAML::Node val = it->second;
        MFEM_VERIFY(val["mode"],
            "Attribute " << entry.attribute << " missing 'mode' field");

        entry.mode = val["mode"].as<std::string>();

        if (entry.mode == "constant") {
            // Read all numeric fields into params map (material_type agnostic)
            // Common fields: vp, vs, rho, qkappa, qmu
            // VTI fields: vp0, vs0, epsilon, delta, gamma (future)
            // The Loader will extract the fields it needs based on material_type
            for (auto field_it = val.begin(); field_it != val.end(); ++field_it) {
                std::string key = field_it->first.as<std::string>();
                if (key == "mode") continue;  // Skip mode field

                // Try to read as real_t (skip strings like "file")
                try {
                    entry.params[key] = field_it->second.as<real_t>();
                } catch (const YAML::BadConversion&) {
                    // Not a numeric field, skip
                }
            }

            // Basic validation: rho is always required
            MFEM_VERIFY(entry.params.count("rho") > 0,
                "Attribute " << entry.attribute
                << " (constant) must have 'rho' field");
        }
        else if (entry.mode == "grid") {
            MFEM_VERIFY(val["file"],
                "Attribute " << entry.attribute
                << " (grid) must have 'file' field");

            std::string grid_file = val["file"].as<std::string>();
            // Resolve relative path from YAML directory
            if (!grid_file.empty() && grid_file[0] != '/') {
                entry.grid_file = yaml_dir + grid_file;
            } else {
                entry.grid_file = grid_file;
            }
        }
        else if (entry.mode == "adios2") {
            // Per-field BP file paths. vp_file/rho_file mandatory; others optional.
            // material_type-aware fields (vs only matters for elastic, qmu likewise).
            auto resolve_path = [&yaml_dir](const std::string& p) {
                if (p.empty() || p[0] == '/') return p;
                return yaml_dir + p;
            };
            MFEM_VERIFY(val["vp_file"],
                "Attribute " << entry.attribute
                << " (adios2) must have 'vp_file' field");
            MFEM_VERIFY(val["rho_file"],
                "Attribute " << entry.attribute
                << " (adios2) must have 'rho_file' field");
            entry.adios2_vp_file  = resolve_path(val["vp_file"].as<std::string>());
            entry.adios2_rho_file = resolve_path(val["rho_file"].as<std::string>());
            if (val["vs_file"])
                entry.adios2_vs_file = resolve_path(val["vs_file"].as<std::string>());
            if (val["qkappa_file"])
                entry.adios2_qkappa_file =
                    resolve_path(val["qkappa_file"].as<std::string>());
            if (val["qmu_file"])
                entry.adios2_qmu_file =
                    resolve_path(val["qmu_file"].as<std::string>());
        }
        else {
            MFEM_ABORT("Unknown mode '" << entry.mode
                       << "' for attribute " << entry.attribute
                       << ". Supported: 'constant', 'grid', 'adios2'");
        }

        entries.push_back(entry);
    }

    MFEM_VERIFY(!entries.empty(),
        "No attribute entries found in YAML file: " << yaml_path);
}

}  // namespace SEM
