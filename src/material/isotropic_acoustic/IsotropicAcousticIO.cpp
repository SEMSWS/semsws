/**
 * @file IsotropicAcousticIO.cpp
 * @brief I/O implementation for isotropic acoustic material data
 */

#include "material/isotropic_acoustic/IsotropicAcousticIO.hpp"
#include "material/AcousticMaterialBase.hpp"
#include "io/ADIOS2IO.hpp"
#include <fstream>
#include <iostream>
#include <sstream>

namespace SEM {

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

/// Skip comment lines (starting with #) and empty lines
bool GetNextDataLine(std::ifstream& file, std::string& line) {
    while (std::getline(file, line)) {
        size_t first = line.find_first_not_of(" \t");
        if (first == std::string::npos) continue;  // empty line
        if (line[first] == '#') continue;          // comment
        return true;
    }
    return false;
}

/// Count values in a string
int CountValues(const std::string& line) {
    std::istringstream iss(line);
    int count = 0;
    real_t val;
    while (iss >> val) count++;
    return count;
}

}  // anonymous namespace

// =============================================================================
// 2D Grid I/O
// =============================================================================

bool ReadIsotropicAcousticMaterialData2D(const std::string& filename,
                                         IsotropicAcousticMaterialData2D& mat,
                                         bool read_Q) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open material file: " << filename << std::endl;
        return false;
    }

    std::string line;

    // Read nx ny
    if (!GetNextDataLine(file, line)) {
        std::cerr << "Error: Missing grid dimensions (nx ny) in: " << filename << std::endl;
        return false;
    }
    {
        std::istringstream iss(line);
        if (!(iss >> mat.nx >> mat.ny)) {
            std::cerr << "Error: Invalid grid dimensions in: " << filename << std::endl;
            return false;
        }
    }

    // Read dx dy
    if (!GetNextDataLine(file, line)) {
        std::cerr << "Error: Missing grid spacing (dx dy) in: " << filename << std::endl;
        return false;
    }
    {
        std::istringstream iss(line);
        if (!(iss >> mat.dx >> mat.dy)) {
            std::cerr << "Error: Invalid grid spacing in: " << filename << std::endl;
            return false;
        }
    }

    // Read x0 y0
    if (!GetNextDataLine(file, line)) {
        std::cerr << "Error: Missing grid origin (x0 y0) in: " << filename << std::endl;
        return false;
    }
    {
        std::istringstream iss(line);
        if (!(iss >> mat.x0 >> mat.y0)) {
            std::cerr << "Error: Invalid grid origin in: " << filename << std::endl;
            return false;
        }
    }

    // Validate header
    if (mat.nx <= 0 || mat.ny <= 0 || mat.dx <= 0 || mat.dy <= 0) {
        std::cerr << "Error: Invalid header values in: " << filename << std::endl;
        return false;
    }

    int total = mat.nx * mat.ny;

    // Read first data line to check columns
    if (!GetNextDataLine(file, line)) {
        std::cerr << "Error: No data found in: " << filename << std::endl;
        return false;
    }

    int ncols = CountValues(line);
    int required_cols = read_Q ? 3 : 2;
    if (ncols < required_cols) {
        std::cerr << "Error: Acoustic material needs " << required_cols
                  << " columns in: " << filename
                  << " (got " << ncols << ")" << std::endl;
        return false;
    }
    if (ncols > required_cols) {
        std::cerr << "[WARN] " << filename << " has " << ncols
                  << " columns but only " << required_cols
                  << " will be used (extra columns ignored). "
                  << "Enable material.attenuation if the extras are Qkappa."
                  << std::endl;
    }
    mat.has_Q = read_Q;

    // Allocate vectors
    mat.vp.SetSize(total);
    mat.rho.SetSize(total);
    if (mat.has_Q) {
        mat.Qkappa.SetSize(total);
    }

    // Parse first data line
    {
        std::istringstream iss(line);
        iss >> mat.vp(0) >> mat.rho(0);
        if (mat.has_Q) {
            iss >> mat.Qkappa(0);
        }
    }

    // Read remaining data
    for (int i = 1; i < total; i++) {
        if (!GetNextDataLine(file, line)) {
            std::cerr << "Error: Insufficient data in: " << filename
                      << " (expected " << total << " points, got " << i << ")" << std::endl;
            return false;
        }
        std::istringstream iss(line);
        iss >> mat.vp(i) >> mat.rho(i);
        if (mat.has_Q) {
            iss >> mat.Qkappa(i);
        }
    }

    return true;
}

// =============================================================================
// 3D Grid I/O
// =============================================================================

bool ReadIsotropicAcousticMaterialData3D(const std::string& filename,
                                         IsotropicAcousticMaterialData3D& mat,
                                         bool read_Q) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open material file: " << filename << std::endl;
        return false;
    }

    std::string line;

    // Read nx ny nz
    if (!GetNextDataLine(file, line)) {
        std::cerr << "Error: Missing grid dimensions (nx ny nz) in: " << filename << std::endl;
        return false;
    }
    {
        std::istringstream iss(line);
        if (!(iss >> mat.nx >> mat.ny >> mat.nz)) {
            std::cerr << "Error: Invalid grid dimensions in: " << filename << std::endl;
            return false;
        }
    }

    // Read dx dy dz
    if (!GetNextDataLine(file, line)) {
        std::cerr << "Error: Missing grid spacing (dx dy dz) in: " << filename << std::endl;
        return false;
    }
    {
        std::istringstream iss(line);
        if (!(iss >> mat.dx >> mat.dy >> mat.dz)) {
            std::cerr << "Error: Invalid grid spacing in: " << filename << std::endl;
            return false;
        }
    }

    // Read x0 y0 z0
    if (!GetNextDataLine(file, line)) {
        std::cerr << "Error: Missing grid origin (x0 y0 z0) in: " << filename << std::endl;
        return false;
    }
    {
        std::istringstream iss(line);
        if (!(iss >> mat.x0 >> mat.y0 >> mat.z0)) {
            std::cerr << "Error: Invalid grid origin in: " << filename << std::endl;
            return false;
        }
    }

    // Validate header
    if (mat.nx <= 0 || mat.ny <= 0 || mat.nz <= 0 ||
        mat.dx <= 0 || mat.dy <= 0 || mat.dz <= 0) {
        std::cerr << "Error: Invalid header values in: " << filename << std::endl;
        return false;
    }

    int total = mat.nx * mat.ny * mat.nz;

    // Read first data line to check columns
    if (!GetNextDataLine(file, line)) {
        std::cerr << "Error: No data found in: " << filename << std::endl;
        return false;
    }

    int ncols = CountValues(line);
    int required_cols = read_Q ? 3 : 2;
    if (ncols < required_cols) {
        std::cerr << "Error: Acoustic material needs " << required_cols
                  << " columns in: " << filename
                  << " (got " << ncols << ")" << std::endl;
        return false;
    }
    if (ncols > required_cols) {
        std::cerr << "[WARN] " << filename << " has " << ncols
                  << " columns but only " << required_cols
                  << " will be used (extra columns ignored). "
                  << "Enable material.attenuation if the extras are Qkappa."
                  << std::endl;
    }
    mat.has_Q = read_Q;

    // Allocate vectors
    mat.vp.SetSize(total);
    mat.rho.SetSize(total);
    if (mat.has_Q) {
        mat.Qkappa.SetSize(total);
    }

    // Parse first data line
    {
        std::istringstream iss(line);
        iss >> mat.vp(0) >> mat.rho(0);
        if (mat.has_Q) {
            iss >> mat.Qkappa(0);
        }
    }

    // Read remaining data
    for (int i = 1; i < total; i++) {
        if (!GetNextDataLine(file, line)) {
            std::cerr << "Error: Insufficient data in: " << filename
                      << " (expected " << total << " points, got " << i << ")" << std::endl;
            return false;
        }
        std::istringstream iss(line);
        iss >> mat.vp(i) >> mat.rho(i);
        if (mat.has_Q) {
            iss >> mat.Qkappa(i);
        }
    }

    return true;
}

// =============================================================================
// Attribute-Based I/O
// =============================================================================

bool ReadIsotropicAcousticAttributeMaterial(const std::string& filename,
                                            std::vector<IsotropicAcousticAttributeEntry>& entries,
                                            bool read_Q) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open isotropic acoustic attribute material file: " << filename << std::endl;
        return false;
    }

    entries.clear();
    std::string line;

    while (GetNextDataLine(file, line)) {
        int ncols = CountValues(line);
        int required_cols = read_Q ? 4 : 3;
        if (ncols < required_cols) {
            std::cerr << "Error: Isotropic acoustic attribute material needs " << required_cols
                      << " columns in: " << filename
                      << " (got " << ncols << ")" << std::endl;
            return false;
        }

        IsotropicAcousticAttributeEntry entry;
        std::istringstream iss(line);
        iss >> entry.attribute >> entry.vp >> entry.rho;

        if (read_Q) {
            iss >> entry.Qkappa;
        }

        entries.push_back(entry);
    }

    if (entries.empty()) {
        std::cerr << "Error: No attribute material data found in: " << filename << std::endl;
        return false;
    }

    return true;
}

// =============================================================================
// ADIOS2 Export
// =============================================================================

void ExportAcousticMaterialBP(const AcousticMaterialBase2D& material,
                              const std::string& export_dir,
                              const std::string& mesh_file,
                              MPI_Comm comm)
{
    const MaterialField& kappa = material.Kappa();
    const MaterialField& inv_rho = material.InvRho();

    int ne = kappa.NumElements();
    int ngllx = kappa.NumGLLx();
    int nglly = kappa.NumGLLy();
    int total = ne * ngllx * nglly;

    // Convert kappa/inv_rho -> vp/rho
    MaterialField vp_field(ne, ngllx, nglly);
    MaterialField rho_field(ne, ngllx, nglly);

    const real_t* k_data = kappa.Data().HostRead();
    const real_t* ir_data = inv_rho.Data().HostRead();
    real_t* vp_data = vp_field.Data().HostWrite();
    real_t* rho_data = rho_field.Data().HostWrite();

    for (int i = 0; i < total; i++) {
        rho_data[i] = 1.0 / ir_data[i];
        vp_data[i] = std::sqrt(k_data[i] * ir_data[i]);
    }

    SaveFieldBP(export_dir + "/vp.bp", "data", vp_field, mesh_file, comm);
    SaveFieldBP(export_dir + "/rho.bp", "data", rho_field, mesh_file, comm);

    // Export Qkappa if attenuation is enabled
    if (material.HasAttenuation()) {
        SaveFieldBP(export_dir + "/qkappa.bp", "data",
                    material.Qkappa(), mesh_file, comm);
    }
}

void ExportAcousticMaterialBP(const AcousticMaterialBase3D& material,
                              const std::string& export_dir,
                              const std::string& mesh_file,
                              MPI_Comm comm)
{
    const MaterialField3D& kappa = material.Kappa();
    const MaterialField3D& inv_rho = material.InvRho();

    int ne = kappa.NumElements();
    int ngllx = kappa.NumGLLx();
    int nglly = kappa.NumGLLy();
    int ngllz = kappa.NumGLLz();
    int total = ne * ngllx * nglly * ngllz;

    // Convert kappa/inv_rho -> vp/rho as 2D MaterialField (flattened)
    int ngll_total = ngllx * nglly * ngllz;
    MaterialField vp_field(ne, ngll_total, 1);
    MaterialField rho_field(ne, ngll_total, 1);

    const real_t* k_data = kappa.Data().HostRead();
    const real_t* ir_data = inv_rho.Data().HostRead();
    real_t* vp_data = vp_field.Data().HostWrite();
    real_t* rho_data = rho_field.Data().HostWrite();

    for (int i = 0; i < total; i++) {
        rho_data[i] = 1.0 / ir_data[i];
        vp_data[i] = std::sqrt(k_data[i] * ir_data[i]);
    }

    SaveFieldBP(export_dir + "/vp.bp", "data", vp_field, mesh_file, comm);
    SaveFieldBP(export_dir + "/rho.bp", "data", rho_field, mesh_file, comm);

    // Export Qkappa if attenuation is enabled
    if (material.HasAttenuation()) {
        // Flatten 3D Qkappa to 2D MaterialField for SaveFieldBP
        const MaterialField3D& qkappa3d = material.Qkappa();
        MaterialField qkappa_field(ne, ngll_total, 1);
        const real_t* q_src = qkappa3d.Data().HostRead();
        real_t* q_dst = qkappa_field.Data().HostWrite();
        for (int i = 0; i < total; i++) {
            q_dst[i] = q_src[i];
        }
        SaveFieldBP(export_dir + "/qkappa.bp", "data",
                    qkappa_field, mesh_file, comm);
    }
}

}  // namespace SEM
