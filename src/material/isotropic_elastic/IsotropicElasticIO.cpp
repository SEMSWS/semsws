/**
 * @file IsotropicElasticIO.cpp
 * @brief I/O implementation for isotropic elastic material data
 */

#include "material/isotropic_elastic/IsotropicElasticIO.hpp"
#include "material/MaterialBase.hpp"
#include "material/ElasticMaterialBase.hpp"
#include "material/MaterialField.hpp"
#include "io/ADIOS2IO.hpp"
#include <cmath>
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

bool ReadIsotropicElasticMaterialData2D(const std::string& filename,
                                        IsotropicElasticMaterialData2D& mat,
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
    int required_cols = read_Q ? 5 : 3;
    if (ncols < required_cols) {
        std::cerr << "Error: Isotropic material needs " << required_cols
                  << " columns in: " << filename
                  << " (got " << ncols << ")" << std::endl;
        return false;
    }
    if (ncols > required_cols) {
        std::cerr << "[WARN] " << filename << " has " << ncols
                  << " columns but only " << required_cols
                  << " will be used (extra columns ignored). "
                  << "Enable material.attenuation if the extras are Qkappa/Qmu."
                  << std::endl;
    }
    mat.has_Q = read_Q;

    // Allocate vectors
    mat.vp.SetSize(total);
    mat.vs.SetSize(total);
    mat.rho.SetSize(total);
    if (mat.has_Q) {
        mat.Qkappa.SetSize(total);
        mat.Qmu.SetSize(total);
    }

    // Parse first data line
    {
        std::istringstream iss(line);
        iss >> mat.vp(0) >> mat.vs(0) >> mat.rho(0);
        if (mat.has_Q) {
            iss >> mat.Qkappa(0) >> mat.Qmu(0);
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
        iss >> mat.vp(i) >> mat.vs(i) >> mat.rho(i);
        if (mat.has_Q) {
            iss >> mat.Qkappa(i) >> mat.Qmu(i);
        }
    }

    return true;
}

// =============================================================================
// 3D Grid I/O
// =============================================================================

bool ReadIsotropicElasticMaterialData3D(const std::string& filename,
                                        IsotropicElasticMaterialData3D& mat,
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
    int required_cols = read_Q ? 5 : 3;
    if (ncols < required_cols) {
        std::cerr << "Error: Isotropic material needs " << required_cols
                  << " columns in: " << filename
                  << " (got " << ncols << ")" << std::endl;
        return false;
    }
    if (ncols > required_cols) {
        std::cerr << "[WARN] " << filename << " has " << ncols
                  << " columns but only " << required_cols
                  << " will be used (extra columns ignored). "
                  << "Enable material.attenuation if the extras are Qkappa/Qmu."
                  << std::endl;
    }
    mat.has_Q = read_Q;

    // Allocate vectors
    mat.vp.SetSize(total);
    mat.vs.SetSize(total);
    mat.rho.SetSize(total);
    if (mat.has_Q) {
        mat.Qkappa.SetSize(total);
        mat.Qmu.SetSize(total);
    }

    // Parse first data line
    {
        std::istringstream iss(line);
        iss >> mat.vp(0) >> mat.vs(0) >> mat.rho(0);
        if (mat.has_Q) {
            iss >> mat.Qkappa(0) >> mat.Qmu(0);
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
        iss >> mat.vp(i) >> mat.vs(i) >> mat.rho(i);
        if (mat.has_Q) {
            iss >> mat.Qkappa(i) >> mat.Qmu(i);
        }
    }

    return true;
}

// =============================================================================
// Write 2D Grid
// =============================================================================

bool WriteIsotropicElasticMaterialData2D(const std::string& filename,
                                         const IsotropicElasticMaterialData2D& mat,
                                         bool include_Q) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create material file: " << filename << std::endl;
        return false;
    }

    file.precision(8);

    // Write header with comments
    file << "# SEMSWS Isotropic Material File (2D)\n";
    file << "# nx ny\n";
    file << mat.nx << " " << mat.ny << "\n";
    file << "# dx dy\n";
    file << mat.dx << " " << mat.dy << "\n";
    file << "# x0 y0\n";
    file << mat.x0 << " " << mat.y0 << "\n";

    if (include_Q && mat.has_Q) {
        file << "# Data: vp(m/s) vs(m/s) rho(kg/m^3) Qkappa Qmu\n";
        file << "# x varies fastest (lexicographic order)\n";
        int total = mat.nx * mat.ny;
        for (int i = 0; i < total; i++) {
            file << mat.vp(i) << " " << mat.vs(i) << " " << mat.rho(i)
                 << " " << mat.Qkappa(i) << " " << mat.Qmu(i) << "\n";
        }
    } else {
        file << "# Data: vp(m/s) vs(m/s) rho(kg/m^3)\n";
        file << "# x varies fastest (lexicographic order)\n";
        int total = mat.nx * mat.ny;
        for (int i = 0; i < total; i++) {
            file << mat.vp(i) << " " << mat.vs(i) << " " << mat.rho(i) << "\n";
        }
    }

    return true;
}

// =============================================================================
// Attribute-Based I/O
// =============================================================================

bool ReadIsotropicElasticAttributeMaterial(const std::string& filename,
                                           std::vector<IsotropicElasticAttributeEntry>& entries,
                                           bool read_Q) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open isotropic elastic attribute material file: " << filename << std::endl;
        return false;
    }

    entries.clear();
    std::string line;

    while (GetNextDataLine(file, line)) {
        int ncols = CountValues(line);
        int required_cols = read_Q ? 6 : 4;
        if (ncols < required_cols) {
            std::cerr << "Error: Isotropic elastic attribute material needs " << required_cols
                      << " columns in: " << filename
                      << " (got " << ncols << ")" << std::endl;
            return false;
        }

        IsotropicElasticAttributeEntry entry;
        std::istringstream iss(line);
        iss >> entry.attribute >> entry.vp >> entry.vs >> entry.rho;

        if (read_Q) {
            iss >> entry.Qkappa >> entry.Qmu;
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

void ExportElasticMaterialBP(const ElasticMaterialBase2D& material,
                             const std::string& export_dir,
                             const std::string& mesh_file,
                             MPI_Comm comm)
{
    const MaterialField& kappa = material.Kappa();
    const MaterialField& mu    = material.Mu();
    const MaterialField& rho   = material.Rho();

    int ne = kappa.NumElements();
    int ngllx = kappa.NumGLLx();
    int nglly = kappa.NumGLLy();
    int total = ne * ngllx * nglly;

    MaterialField vp_field(ne, ngllx, nglly);
    MaterialField vs_field(ne, ngllx, nglly);

    const real_t* k_data = kappa.Data().HostRead();
    const real_t* m_data = mu.Data().HostRead();
    const real_t* r_data = rho.Data().HostRead();
    real_t* vp_data = vp_field.Data().HostWrite();
    real_t* vs_data = vs_field.Data().HostWrite();

    // 2D plane strain: kappa = lambda + mu, so vp = sqrt((kappa + mu) / rho).
    for (int i = 0; i < total; i++) {
        real_t vp2 = (k_data[i] + m_data[i]) / r_data[i];
        real_t vs2 = m_data[i] / r_data[i];
        vp_data[i] = std::sqrt(vp2);
        vs_data[i] = std::sqrt(vs2);
    }

    SaveFieldBP(export_dir + "/vp.bp",  "data", vp_field, mesh_file, comm);
    SaveFieldBP(export_dir + "/vs.bp",  "data", vs_field, mesh_file, comm);
    SaveFieldBP(export_dir + "/rho.bp", "data", rho,      mesh_file, comm);

    if (material.HasAttenuation()) {
        SaveFieldBP(export_dir + "/qkappa.bp", "data",
                    material.Qkappa(), mesh_file, comm);
        SaveFieldBP(export_dir + "/qmu.bp", "data",
                    material.Qmu(), mesh_file, comm);
    }
}

void ExportElasticMaterialBP(const ElasticMaterialBase3D& material,
                             const std::string& export_dir,
                             const std::string& mesh_file,
                             MPI_Comm comm)
{
    const MaterialField3D& kappa = material.Kappa();
    const MaterialField3D& mu    = material.Mu();
    const MaterialField3D& rho   = material.Rho();

    int ne = kappa.NumElements();
    int ngllx = kappa.NumGLLx();
    int nglly = kappa.NumGLLy();
    int ngllz = kappa.NumGLLz();
    int total = ne * ngllx * nglly * ngllz;

    // Mirror the acoustic 3D export: pack as 2D MaterialField (ne, ngll_total, 1)
    int ngll_total = ngllx * nglly * ngllz;
    MaterialField vp_field(ne, ngll_total, 1);
    MaterialField vs_field(ne, ngll_total, 1);
    MaterialField rho_field(ne, ngll_total, 1);

    const real_t* k_data = kappa.Data().HostRead();
    const real_t* m_data = mu.Data().HostRead();
    const real_t* r_data = rho.Data().HostRead();
    real_t* vp_data = vp_field.Data().HostWrite();
    real_t* vs_data = vs_field.Data().HostWrite();
    real_t* rho_data = rho_field.Data().HostWrite();

    for (int i = 0; i < total; i++) {
        real_t vp2 = (k_data[i] + (4.0 / 3.0) * m_data[i]) / r_data[i];
        real_t vs2 = m_data[i] / r_data[i];
        vp_data[i] = std::sqrt(vp2);
        vs_data[i] = std::sqrt(vs2);
        rho_data[i] = r_data[i];
    }

    SaveFieldBP(export_dir + "/vp.bp",  "data", vp_field,  mesh_file, comm);
    SaveFieldBP(export_dir + "/vs.bp",  "data", vs_field,  mesh_file, comm);
    SaveFieldBP(export_dir + "/rho.bp", "data", rho_field, mesh_file, comm);

    if (material.HasAttenuation()) {
        // Flatten 3D Q fields to 2D MaterialField for SaveFieldBP
        const MaterialField3D& qk3d = material.Qkappa();
        const MaterialField3D& qm3d = material.Qmu();
        MaterialField qk_field(ne, ngll_total, 1);
        MaterialField qm_field(ne, ngll_total, 1);
        const real_t* qk_src = qk3d.Data().HostRead();
        const real_t* qm_src = qm3d.Data().HostRead();
        real_t* qk_dst = qk_field.Data().HostWrite();
        real_t* qm_dst = qm_field.Data().HostWrite();
        for (int i = 0; i < total; i++) {
            qk_dst[i] = qk_src[i];
            qm_dst[i] = qm_src[i];
        }
        SaveFieldBP(export_dir + "/qkappa.bp", "data", qk_field, mesh_file, comm);
        SaveFieldBP(export_dir + "/qmu.bp",    "data", qm_field, mesh_file, comm);
    }
}

}  // namespace SEM
