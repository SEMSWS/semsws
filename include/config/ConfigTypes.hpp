/**
 * @file ConfigTypes.hpp
 * @brief Simple data structures for configuration values
 *
 * These structures hold values read from YAML configuration.
 * They are pure data containers with no logic - the actual object
 * creation is done by the respective classes (Material, Source, etc.)
 * via their FromConfig() methods.
 */

#ifndef SEM_CONFIG_TYPES_HPP
#define SEM_CONFIG_TYPES_HPP

#include <mfem.hpp>
#include <string>
#include <vector>
#include <array>
#include <map>

namespace SEM {

using namespace mfem;

// =============================================================================
// Material Configuration
// =============================================================================

/**
 * @brief Material configuration from YAML
 *
 * Holds the raw values from YAML. File parsing and Lame parameter
 * conversion are done by Material::FromConfig().
 *
 * The material_type determines how the data is interpreted:
 * - "isotropic": vp, vs, rho (optionally Qkappa, Qmu)
 * - "acoustic": vp, rho only
 * - "anisotropic": Full elastic tensor (future)
 *
 * The format determines where the data comes from:
 * - "constant": Values directly in YAML
 * - "grid": Values from structured grid ASCII file
 * - "by_attribute": Per-region values (from YAML or file)
 * - "by_attribute_mixed": Per-region values from external mix YAML
 * - "adios2": Pre-computed GLL data from .bp files
 */
/**
 * @brief Attenuation configuration from YAML
 *
 * Used by both isotropic (Qkappa + Qmu) and acoustic (Qkappa only) materials.
 */
struct AttenuationConfig {
    bool enabled = false;         ///< Whether attenuation is enabled
    real_t f0 = 1.0;              ///< Reference frequency (Hz) for Q fitting
    int n_units = 3;              ///< Number of SLS units (N_SLS)
    real_t qkappa = -1.0;         ///< Q factor for bulk modulus (-1 = from file)
    real_t qmu = -1.0;            ///< Q factor for shear modulus (-1 = from file, acoustic ignores)
};

struct MaterialConfig {
    /// Material type: "isotropic_elastic", "isotropic_acoustic", "vti_elastic" (future), etc.
    /// Determines how parameters are interpreted
    std::string material_type = "isotropic_elastic";

    /// Format: "constant", "grid", "by_attribute", "by_attribute_mixed", "adios2"
    /// Determines where parameters come from
    std::string format;

    // ===== format="constant" =====
    /// Generic parameter map for material properties (material_type specific)
    /// - isotropic_elastic: vp, vs, rho
    /// - isotropic_acoustic: vp, rho
    /// - vti_elastic: vp0, vs0, rho, epsilon, delta, gamma (future)
    std::map<std::string, real_t> params;

    // ===== Attenuation (optional, applies to all formats) =====
    AttenuationConfig attenuation;

    // ===== format="grid" =====
    /// Path to material file (contains all properties for material_type)
    /// File format interpretation depends on material_type
    std::string material_file;

    // ===== format="by_attribute" =====
    /// File containing attribute-material mapping
    /// File is read by Material::FromConfig() based on material_type
    std::string by_attribute_file;

    // ===== format="by_attribute_mixed" =====
    /// Entries loaded from external YAML file (by_attribute_file)
    /// Each entry can be constant or grid (heterogeneous)
    std::vector<struct AttributeMaterialEntry> attribute_entries;

    // ===== format="adios2" =====
    /// ADIOS2 .bp file paths for pre-computed GLL data (FWI iterations)
    std::string adios2_vp_file;      ///< Path to Vp .bp file
    std::string adios2_vs_file;      ///< Path to Vs .bp file (elastic only)
    std::string adios2_rho_file;     ///< Path to rho .bp file
    std::string adios2_qkappa_file;  ///< Path to Qkappa .bp file (visco)
    std::string adios2_qmu_file;     ///< Path to Qmu .bp file (visco-elastic)
};

/**
 * @brief Fluid-solid coupled material configuration (YAML `material.type: coupled`).
 *
 * Holds two nested MaterialConfigs (one per submesh) plus the parent-mesh
 * element attribute that identifies each domain. The interface boundary
 * attribute is NOT stored here: FluidSolidInterface::Setup detects it at
 * run time via set-difference on ParSubMesh bdr_attributes.
 */
struct CoupledMaterialConfig {
    int fluid_attribute = 1;       ///< parent-mesh element attr for the fluid domain
    int solid_attribute = 2;       ///< parent-mesh element attr for the solid domain
    MaterialConfig fluid;          ///< nested config; must resolve to an acoustic material
    MaterialConfig solid;          ///< nested config; must resolve to an elastic material
};

/**
 * @brief Entry for mixed attribute material configuration
 *
 * Each entry defines material properties for one mesh attribute.
 * mode="constant": properties specified directly in params map
 * mode="grid": properties loaded from ascii grid file
 *
 * The params map is material_type agnostic:
 * - isotropic_elastic: vp, vs, rho, qkappa, qmu
 * - isotropic_acoustic: vp, rho, qkappa
 * - vti_elastic: vp0, vs0, rho, epsilon, delta, gamma (future)
 */
struct AttributeMaterialEntry {
    int attribute = 0;           ///< Mesh element attribute number
    std::string mode;            ///< "constant" | "grid" | "adios2"

    // mode="constant" parameters (generic key-value)
    std::map<std::string, real_t> params;

    // mode="grid" parameters
    std::string grid_file;       ///< Path to ascii grid file (relative to YAML)

    // mode="adios2" parameters (paths to .bp files)
    /// Per-field ADIOS2 .bp file paths (resolved against the mixed YAML's dir)
    std::string adios2_vp_file;
    std::string adios2_vs_file;       ///< Optional (acoustic ignores)
    std::string adios2_rho_file;
    std::string adios2_qkappa_file;   ///< Optional (visco)
    std::string adios2_qmu_file;      ///< Optional (visco-elastic only)
};

// =============================================================================
// Source Configuration (Namespace-based design)
// =============================================================================

/**
 * @brief Source configuration namespace
 *
 * Contains pure data structures for source definition.
 * Object creation is done by PointSourceCollection::FromConfig().
 */
namespace SourceConfig {

/// Common wavelet/STF parameters
struct WaveletConfig {
    std::string type = "ricker";   ///< "ricker", "gaussian", "external"
    real_t frequency = 0.0;        ///< Dominant frequency [Hz]
    real_t amplitude = 1.0;        ///< Amplitude scaling
    real_t delay = 0.0;            ///< Time delay [s]
    std::string external_file;     ///< Path to external STF file
    std::vector<real_t> stf_samples; ///< Pre-loaded samples (type="hdf5"); length == nt
};

/// Force source (elastic media)
struct ForceSource {
    int id = 0;                     ///< Source identifier from config
    std::vector<real_t> location;   ///< Position [x,y] or [x,y,z]
    std::vector<real_t> direction;  ///< Force direction [dim]
    WaveletConfig wavelet;
};

/// Pressure source (acoustic media)
struct PressureSource {
    int id = 0;                     ///< Source identifier from config
    std::vector<real_t> location;   ///< Position [x,y] or [x,y,z]
    WaveletConfig wavelet;
};

/// 2D moment tensor source (3 independent components)
struct MomentTensorSource2D {
    int id = 0;                     ///< Source identifier from config
    std::vector<real_t> location;   ///< Position [x, y]
    real_t Mxx = 0.0;
    real_t Myy = 0.0;
    real_t Mxy = 0.0;
    WaveletConfig wavelet;
};

/// 3D moment tensor source (6 independent components)
struct MomentTensorSource3D {
    int id = 0;                     ///< Source identifier from config
    std::vector<real_t> location;   ///< Position [x, y, z]
    real_t Mxx = 0.0;  //MΘΘ (harvard CMT)　//Mpp
    real_t Myy = 0.0;  //MΦΦ                //Mtt
    real_t Mzz = 0.0;  //Mrr                //Mrr
    real_t Mxy = 0.0;  //-MΘΦ               //-Mtp
    real_t Mxz = 0.0;  //MΘr                //Mrp
    real_t Myz = 0.0;  //-MΦr               //-Mrt
    WaveletConfig wavelet;
};

/// 2D simulation source configuration
struct Config2D {
    std::vector<ForceSource> forces;
    std::vector<PressureSource> pressures;
    std::vector<MomentTensorSource2D> moment_tensors;
};

/// 3D simulation source configuration
struct Config3D {
    std::vector<ForceSource> forces;
    std::vector<PressureSource> pressures;
    std::vector<MomentTensorSource3D> moment_tensors;
};

}  // namespace SourceConfig

// =============================================================================
// Receiver Configuration (Namespace-based design)
// =============================================================================

/**
 * @brief Receiver configuration namespace
 *
 * Contains pure data structures for receiver definition.
 * Object creation is done by ReceiverArray::FromConfig().
 * Grid expansion is done in YamlConfig::ParseReceivers().
 */
namespace ReceiverConfig {

/// Single receiver definition (types as string list for ConfigTypes independence)
struct SingleReceiver {
    std::string name;
    std::vector<real_t> location;       ///< [x,y] or [x,y,z]
    std::vector<std::string> types;     ///< ["velocity", "displacement", ...] multiple types
    real_t weight = 1.0;
};

/// Configuration (Grid already expanded to receivers by YamlConfig)
struct Config {
    std::vector<SingleReceiver> receivers;  ///< All receivers (including Grid-expanded)
    std::string output_format = "hdf5";     ///< "hdf5", "ascii", or "su"
};

}  // namespace ReceiverConfig

// =============================================================================
// Visualization Output Configuration
// =============================================================================

/**
 * @brief GMT cross-section positions for 3D simulations
 *
 * Each vector contains positions where orthogonal slices are taken.
 * Slice extents are automatically determined from mesh bounding box.
 * Points outside the mesh are written as NaN.
 */
struct GMTCrossSections {
    std::vector<real_t> yz;   ///< yz-plane slice positions (x coordinates)
    std::vector<real_t> xz;   ///< xz-plane slice positions (y coordinates)
    std::vector<real_t> xy;   ///< xy-plane slice positions (z coordinates)

    bool Empty() const { return yz.empty() && xz.empty() && xy.empty(); }
};

/**
 * @brief Configuration for a single output format
 *
 * Used in wavefield and material output format lists.
 * Only fields relevant to the specified type are used.
 */
struct OutputFormatConfig {
    std::string type;                ///< "glvis", "paraview", "gmt"

    // ParaView options (type="paraview" only)
    int refinement = 1;              ///< Element subdivision level (default: 1)
    std::string data_format = "binary32"; ///< "ascii", "binary"(64bit), "binary32"(32bit)
    int compression = -1;            ///< zlib compression level (-1=default, 0-9)

    // GMT options (type="gmt" only)
    std::array<int,2> resolution = {100, 100}; ///< Interpolation grid size [nx, ny]
    std::vector<int> components = {-1}; ///< Vector components (-1=mag, 0=x, 1=y, 2=z)
    GMTCrossSections cross_sections; ///< 3D cross-section positions
};

/**
 * @brief Wavefield snapshot output configuration
 *
 * Supports multiple output formats simultaneously.
 * Fields use ReceiverType naming: DISP, VEL, ACC, PS.
 */
struct WavefieldOutputConfig {
    bool enabled = false;
    int interval = 100;
    std::vector<std::string> fields = {"DISP"};  ///< DISP, VEL, ACC, PS
    std::vector<OutputFormatConfig> formats;
};

/**
 * @brief Material field output configuration
 *
 * Written once at simulation startup. Fields: vp, vs, rho.
 */
struct MaterialOutputConfig {
    bool enabled = false;
    std::vector<std::string> fields = {"vp"};   ///< vp, vs, rho
    std::vector<OutputFormatConfig> formats;
};

}  // namespace SEM

#endif  // SEM_CONFIG_TYPES_HPP
