/**
 * @file YamlConfig.cpp
 * @brief Implementation of YAML configuration parser for SEMSWS
 */

#include "config/YamlConfig.hpp"
#include "common/Types.hpp"
#include "common/BoundaryUtils.hpp"
#include "srcrecv/HDF5SourceReceiverReader.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <vector>

namespace SEM {

namespace {

/// Reject any key in `node` that is not present in `allowed`. Used as a
/// strict-key guard by block-level parsers/validators so typos, renamed
/// keys, and removed-legacy forms become hard errors instead of silent
/// no-ops. Returns false and writes to `error_msg` on the first unknown
/// key; leaves `error_msg` untouched on success. When `node` is not a
/// mapping (scalar / null / sequence) there are no keys to check and the
/// function trivially succeeds.
bool CheckKnownKeys(const YAML::Node& node,
                    const std::set<std::string>& allowed,
                    const std::string& context,
                    std::string& error_msg) {
    if (!node || !node.IsMap()) return true;
    for (const auto& kv : node) {
        const std::string key = kv.first.as<std::string>();
        if (!allowed.count(key)) {
            std::ostringstream oss;
            oss << "Unknown key `" << key << "` in " << context << " (allowed: ";
            bool first = true;
            for (const auto& k : allowed) {
                if (!first) oss << ", ";
                oss << k;
                first = false;
            }
            oss << ")";
            error_msg = oss.str();
            return false;
        }
    }
    return true;
}

/// MFEM_ABORT-on-failure variant for parsers that are already in the
/// abort-on-error path (ApplyWavefieldNode, ParseOutputFormat, etc.).
void RequireKnownKeys(const YAML::Node& node,
                      const std::set<std::string>& allowed,
                      const std::string& context) {
    std::string err;
    if (!CheckKnownKeys(node, allowed, context, err)) {
        MFEM_ABORT(err);
    }
}

/// Parse a single material sub-node (as used under `material.fluid` and
/// `material.solid` for `type: coupled`). Mirrors the flat LoadMaterialConfig
/// path in ConfigLoaders.cpp but drives off a YAML::Node rather than the
/// flat YamlConfig getters, so the same logic can be reused per-submesh.
MaterialConfig ParseMaterialSubNode(const YAML::Node& node,
                                    const std::string& sub_label)
{
    MFEM_VERIFY(node, "material." << sub_label << " block is missing");

    MaterialConfig mc;

    MFEM_VERIFY(node["type"],   "material." << sub_label << ".type is required");
    MFEM_VERIFY(node["format"], "material." << sub_label << ".format is required");
    mc.material_type = node["type"].as<std::string>();
    mc.format        = node["format"].as<std::string>();

    const auto mat_type = StringToMaterialType(mc.material_type);  // aborts if unknown
    MFEM_VERIFY(mat_type != MaterialType::Coupled,
                "nested material." << sub_label << ".type cannot itself be 'coupled'");

    if (mc.format == "constant") {
        if (mat_type == MaterialType::IsotropicAcoustic) {
            MFEM_VERIFY(node["vp"] && node["rho"],
                        "material." << sub_label << " (constant acoustic) requires vp, rho");
            mc.params["vp"]  = node["vp"].as<real_t>();
            mc.params["rho"] = node["rho"].as<real_t>();
        } else if (mat_type == MaterialType::IsotropicElastic) {
            MFEM_VERIFY(node["vp"] && node["vs"] && node["rho"],
                        "material." << sub_label << " (constant elastic) requires vp, vs, rho");
            mc.params["vp"]  = node["vp"].as<real_t>();
            mc.params["vs"]  = node["vs"].as<real_t>();
            mc.params["rho"] = node["rho"].as<real_t>();
        } else {
            MFEM_ABORT("material." << sub_label << ".format=constant is not supported "
                       "for material_type=" << mc.material_type);
        }
    } else if (mc.format == "grid" || mc.format == "by_attribute" ||
               mc.format == "by_attribute_mixed") {
        MFEM_VERIFY(node["file"],
                    "material." << sub_label << ".format=" << mc.format
                    << " requires 'file'");
        if (mc.format == "grid") {
            mc.material_file = node["file"].as<std::string>();
        } else {
            mc.by_attribute_file = node["file"].as<std::string>();
        }
    } else if (mc.format == "adios2") {
        MFEM_ABORT("material." << sub_label
                   << ".format=adios2 is not yet supported in coupled"
                      " (fluid-solid) configurations. Coming in the next"
                      " release. Use a non-coupled config or another format"
                      " for now.");
    } else {
        MFEM_ABORT("material." << sub_label << ".format=" << mc.format
                   << " is not yet supported for coupled materials");
    }

    // Parse nested attenuation block, if any. Same shape as the top-level
    // `material.attenuation` block handled by `Validate()` below — f0 +
    // n_units mandatory, qkappa/qmu required for the `constant` format
    // (read from the Q files for grid / by_attribute modes).
    if (node["attenuation"] &&
        node["attenuation"]["enabled"] &&
        node["attenuation"]["enabled"].as<bool>())
    {
        const YAML::Node atten = node["attenuation"];
        MFEM_VERIFY(atten["f0"],
                    "material." << sub_label << ".attenuation.f0 is required");
        MFEM_VERIFY(atten["n_units"],
                    "material." << sub_label << ".attenuation.n_units is required");

        mc.attenuation.enabled = true;
        mc.attenuation.f0      = atten["f0"].as<real_t>();
        mc.attenuation.n_units = atten["n_units"].as<int>();

        if (mc.format == "constant") {
            MFEM_VERIFY(atten["qkappa"],
                        "material." << sub_label
                        << ".attenuation.qkappa is required for format=constant");
            mc.attenuation.qkappa = atten["qkappa"].as<real_t>();
            if (mat_type == MaterialType::IsotropicElastic) {
                MFEM_VERIFY(atten["qmu"],
                            "material." << sub_label
                            << ".attenuation.qmu is required for format=constant elastic");
                mc.attenuation.qmu = atten["qmu"].as<real_t>();
            }
        }
        // For grid / by_attribute{,_mixed}: Q fields are read from the
        // material file alongside vp/vs/rho, so no extra parsing here.
    }

    return mc;
}

/// Validate that a nested material sub-block (under `material.fluid` or
/// `material.solid`) has the fields that ParseMaterialSubNode will read.
/// Writes the error message on failure and returns false.
bool ValidateMaterialSubNode(const YAML::Node& node,
                             const std::string& sub_label,
                             std::string& error_msg)
{
    if (!node) {
        error_msg = "Missing required section: material." + sub_label
                  + " (required when material.type = coupled)";
        return false;
    }
    if (!CheckKnownKeys(node, {
            "attribute", "type", "format",
            "vp", "vs", "rho", "file",
            "attenuation"
        }, "material." + sub_label, error_msg)) {
        return false;
    }
    if (node["attenuation"] &&
        !CheckKnownKeys(node["attenuation"], {
            "enabled", "f0", "n_units",
            "qkappa", "qmu", "files"
        }, "material." + sub_label + ".attenuation", error_msg)) {
        return false;
    }
    if (node["attenuation"] && node["attenuation"]["files"] &&
        !CheckKnownKeys(node["attenuation"]["files"],
            {"qkappa", "qmu"},
            "material." + sub_label + ".attenuation.files", error_msg)) {
        return false;
    }
    if (!node["attribute"]) {
        error_msg = "Missing required parameter: material." + sub_label + ".attribute";
        return false;
    }
    if (!node["type"]) {
        error_msg = "Missing required parameter: material." + sub_label + ".type";
        return false;
    }
    if (!node["format"]) {
        error_msg = "Missing required parameter: material." + sub_label + ".format";
        return false;
    }
    const auto mat_type = StringToMaterialType(node["type"].as<std::string>());
    if (mat_type == MaterialType::Coupled) {
        error_msg = "material." + sub_label
                  + ".type must be a concrete material, not 'coupled'";
        return false;
    }
    const std::string fmt = node["format"].as<std::string>();
    if (fmt == "constant") {
        if (mat_type == MaterialType::IsotropicAcoustic) {
            if (!node["vp"] || !node["rho"]) {
                error_msg = "material." + sub_label
                          + " (constant acoustic) requires vp and rho";
                return false;
            }
        } else if (mat_type == MaterialType::IsotropicElastic) {
            if (!node["vp"] || !node["vs"] || !node["rho"]) {
                error_msg = "material." + sub_label
                          + " (constant elastic) requires vp, vs and rho";
                return false;
            }
        } else {
            error_msg = "material." + sub_label
                      + ".format=constant not supported for type="
                      + node["type"].as<std::string>();
            return false;
        }
    }
    // grid / by_attribute / by_attribute_mixed: file is required
    else if (fmt == "grid" || fmt == "by_attribute" || fmt == "by_attribute_mixed") {
        if (!node["file"]) {
            error_msg = "material." + sub_label + ".format=" + fmt
                      + " requires 'file'";
            return false;
        }
    }
    else if (fmt == "adios2") {
        // Schema-allowed; runtime fails until the coupled adios2 loader lands.
    } else {
        error_msg = "material." + sub_label + ".format=" + fmt
                  + " is not yet supported for coupled materials";
        return false;
    }

    // Attenuation block (optional). When enabled, f0 / n_units are
    // always required; Qkappa (+ Qmu for elastic) required for format=constant.
    if (node["attenuation"] &&
        node["attenuation"]["enabled"] &&
        node["attenuation"]["enabled"].as<bool>())
    {
        const YAML::Node atten = node["attenuation"];
        if (!atten["f0"]) {
            error_msg = "material." + sub_label
                      + ".attenuation.f0 is required when enabled";
            return false;
        }
        if (!atten["n_units"]) {
            error_msg = "material." + sub_label
                      + ".attenuation.n_units is required when enabled";
            return false;
        }
        if (fmt == "constant") {
            if (!atten["qkappa"]) {
                error_msg = "material." + sub_label
                          + ".attenuation.qkappa is required for format=constant";
                return false;
            }
            if (mat_type == MaterialType::IsotropicElastic && !atten["qmu"]) {
                error_msg = "material." + sub_label
                          + ".attenuation.qmu is required for format=constant elastic";
                return false;
            }
        } else if (atten["qkappa"] || atten["qmu"]) {
            error_msg = "material." + sub_label
                      + ".attenuation.qkappa/qmu may only be set with"
                        " format=constant. With format='" + fmt
                      + "', Q values come from the data file.";
            return false;
        }
    }
    return true;
}

}  // namespace

// =============================================================================
// Constructor
// =============================================================================

YamlConfig::YamlConfig()
    : valid_(false), sources_parsed_(false), receivers_parsed_(false) {
    error_msg_ = "No configuration file loaded";
}

YamlConfig::YamlConfig(const std::string& filepath)
    : filepath_(filepath), valid_(false), sources_parsed_(false), receivers_parsed_(false) {
    try {
        root_ = YAML::LoadFile(filepath);
        Validate();
    } catch (const YAML::Exception& e) {
        valid_ = false;
        // Enhance error message: if yaml-cpp reports a mark (line/col),
        // re-open the file and dump a ±2-line context window with a caret
        // pointing at the offending column. Dramatically shortens the
        // debug loop for typos / wrong indentation.
        std::ostringstream oss;
        oss << "YAML parsing error: " << e.msg;
        if (e.mark.line >= 0) {
            oss << " (line " << (e.mark.line + 1)
                << ", col " << (e.mark.column + 1) << ")";
            std::ifstream ifs(filepath);
            if (ifs) {
                std::vector<std::string> lines;
                std::string ln;
                while (std::getline(ifs, ln)) lines.push_back(ln);
                const int err = e.mark.line;
                const int beg = std::max(0, err - 2);
                const int end = std::min(static_cast<int>(lines.size()),
                                         err + 3);
                oss << "\n";
                for (int i = beg; i < end; ++i) {
                    oss << (i == err ? "--> " : "    ")
                        << std::setw(4) << (i + 1) << " | "
                        << lines[i] << "\n";
                    if (i == err) {
                        oss << "         " << std::string(e.mark.column, ' ')
                            << "^\n";
                    }
                }
            }
        }
        error_msg_ = oss.str();
    } catch (const std::exception& e) {
        valid_ = false;
        error_msg_ = "Error loading config: " + std::string(e.what());
    }
}

// =============================================================================
// Validation
// =============================================================================

void YamlConfig::Validate() {
    valid_ = true;
    error_msg_.clear();

    // Root-level keys are warn-only: external drivers (e.g. semsws-driver /
    // semsws-fwi-driver) inject their own top-level blocks (`run`, …) into the
    // same YAML and we want the C++ binary to ignore them rather than fail.
    // Typos in actual SEMSWS sections are still caught by the strict-key
    // checks on each nested block below.
    {
        const std::set<std::string> known_root = {
            "name",
            "simulation", "mesh", "material", "boundary",
            "sources", "receivers", "device", "inversion", "vis"
        };
        if (root_ && root_.IsMap()) {
            for (const auto& kv : root_) {
                const std::string key = kv.first.as<std::string>();
                if (!known_root.count(key)) {
                    std::cerr << "[YamlConfig] warning: ignoring unknown root key `"
                              << key << "` (allowed: name, simulation, mesh, material, "
                              "boundary, sources, receivers, device, inversion, vis)\n";
                }
            }
        }
    }

    if (!root_["name"]) {
        root_["name"] = "Seismic wave simulation";
    }

    //=============================================================================
    // Validate Simulation Section
    //=============================================================================
    YAML::Node sim = root_["simulation"];
    if (!root_["simulation"]) {
        error_msg_ = "Missing required section: simulation";
        valid_ = false;
        return;
    }

    // Strict-key validation for `simulation`.
    if (!CheckKnownKeys(sim, {
            "dimension", "order", "mode",
            "dt", "steps", "cfl_factor", "t0",
            "directory", "log_interval", "summary_file"
        }, "simulation", error_msg_)) {
        valid_ = false;
        return;
    }

    // simulation.dimension
    if (!sim["dimension"]) {
        error_msg_ = "Missing required parameter: simulation.dimension";
        valid_ = false;
        return;
    }

    int dim = sim["dimension"].as<int>();
    if (dim != 2 && dim != 3) {
        error_msg_ = "Invalid dimension: must be 2 or 3";
        valid_ = false;
        return;
    }

    // simulation.order
    if (!sim["order"]) {
        error_msg_ = "Missing required parameter: simulation.order";
        valid_ = false;
        return;
    }
    int order = sim["order"].as<int>();
    if (order < 1) {
        error_msg_ = "Invalid simulation.order: must be >= 1";
        valid_ = false;
        return;
    }

    if (!sim["dt"]) {
        error_msg_ = "Missing required parameter: simulation.dt";
        valid_ = false;
        return;
    }
    auto dt = sim["dt"].as<real_t>();
    if (dt <= 0.0) {
        error_msg_ = "Invalid simulation.dt: must be positive";
        valid_ = false;
        return;
    }
    if (!sim["steps"]) {
        error_msg_ = "Missing required parameter: simulation.steps";
        valid_ = false;
        return;
    }
    auto steps = sim["steps"].as<int>();
    if (steps <= 0) {
        error_msg_ = "Invalid simulation.steps: must be positive";
        valid_ = false;
        return;
    }
    // cfl_factor is optional (default 0.3 in GetCflFactor())
    if (sim["cfl_factor"]) {
        auto cfl = sim["cfl_factor"].as<real_t>();
        if (cfl <= 0.0) {
            error_msg_ = "Invalid simulation.cfl_factor: must be > 0";
            valid_ = false;
            return;
        }
    }

    // simulation.directory / log_interval (flat, was simulation.output.*)
    if (!sim["directory"]) {
        error_msg_ = "Missing required parameter: simulation.directory";
        valid_ = false;
        return;
    }
    // simulation.log_interval is optional (default in GetLogInterval())

    // Top-level `vis:` consolidates wavefield / material / mesh visualization.
    if (root_["vis"]) {
        YAML::Node vis = root_["vis"];
        if (!CheckKnownKeys(vis,
                {"directory", "mesh", "wavefield", "material"},
                "vis", error_msg_)) {
            valid_ = false;
            return;
        }
        if (vis["wavefield"]) {
            if (!CheckKnownKeys(vis["wavefield"],
                    {"enabled", "interval", "fields", "formats", "components",
                     "fluid", "solid"},
                    "vis.wavefield", error_msg_)) {
                valid_ = false;
                return;
            }
            for (const char* side : {"fluid", "solid"}) {
                if (vis["wavefield"][side] &&
                    !CheckKnownKeys(vis["wavefield"][side],
                        {"enabled", "interval", "fields", "formats", "components"},
                        std::string("vis.wavefield.") + side,
                        error_msg_)) {
                    valid_ = false;
                    return;
                }
            }
        }
        if (vis["material"]) {
            if (!CheckKnownKeys(vis["material"],
                    {"enabled", "fields", "formats", "fluid", "solid"},
                    "vis.material", error_msg_)) {
                valid_ = false;
                return;
            }
            for (const char* side : {"fluid", "solid"}) {
                if (vis["material"][side] &&
                    !CheckKnownKeys(vis["material"][side],
                        {"enabled", "fields", "formats"},
                        std::string("vis.material.") + side,
                        error_msg_)) {
                    valid_ = false;
                    return;
                }
            }
        }
        if (vis["mesh"]) {
            if (!vis["mesh"].IsMap()) {
                error_msg_ = "vis.mesh must be a mapping (`{enabled: true}`)";
                valid_ = false;
                return;
            }
            if (!CheckKnownKeys(vis["mesh"], {"enabled"},
                    "vis.mesh", error_msg_)) {
                valid_ = false;
                return;
            }
        }
    }
    // Validate wavefield output parameters when enabled
    if (root_["vis"] && root_["vis"]["wavefield"] &&
        root_["vis"]["wavefield"]["enabled"] &&
        root_["vis"]["wavefield"]["enabled"].as<bool>()) {
        YAML::Node wf = root_["vis"]["wavefield"];
        if (!wf["interval"]) {
            error_msg_ = "Missing required parameter: vis.wavefield.interval (when wavefield enabled)";
            valid_ = false;
            return;
        }
        if (!wf["formats"]) {
            error_msg_ = "Missing required parameter: "
                         "vis.wavefield.formats "
                         "(when wavefield enabled)";
            valid_ = false;
            return;
        }
    }

    //=============================================================================
    // Validate Mesh Section
    //=============================================================================
    if (!root_["mesh"]) {
        error_msg_ = "Missing required section: mesh";
        valid_ = false;
        return;
    }
    YAML::Node mesh = root_["mesh"];
    if (!CheckKnownKeys(mesh, {
            "type", "file",
            "origin", "size", "elements",
            "max_freq", "ppw",
            "partition_method", "partition_grid", "prepart_mfem"
        }, "mesh", error_msg_)) {
        valid_ = false;
        return;
    }

    // mesh.type
    if (!mesh["type"]) {
        error_msg_ = "Missing required parameter: mesh.type";
        valid_ = false;
        return;
    }

    std::string mesh_type = mesh["type"].as<std::string>();

    if (mesh_type == "external") {
        if (!mesh["file"]) {
            error_msg_ = "External mesh requires 'file' parameter";
            valid_ = false;
            return;
        }
        // mesh.format is optional - MFEM auto-detects from file header
    } else if (mesh_type == "internal") {
        if (!mesh["origin"] || !mesh["size"] || !mesh["elements"]) {
            error_msg_ = "Internal mesh requires 'origin', 'size', and 'elements' parameters";
            valid_ = false;
            return;
        }
        else{
            //check the dimensions of origin, size, elements
            YAML::Node origin = mesh["origin"];
            YAML::Node size = mesh["size"];
            YAML::Node elements = mesh["elements"];
            if(origin.size() != dim || size.size() != dim || elements.size() != dim){
                error_msg_ = "Internal mesh 'origin', 'size', and 'elements' must have length equal to dimension";
                valid_ = false;
                return;
            }
        }
    } else if (mesh_type == "prepart_mfem") {
        // Pre-partitioned (MFEM) mesh files for memory-efficient loading
        YAML::Node prepart = mesh["prepart_mfem"];
        if (!prepart) {
            error_msg_ = "mesh.type=prepart_mfem requires a 'prepart_mfem' sub-block";
            valid_ = false;
            return;
        }
        if (!CheckKnownKeys(prepart,
                {"directory", "nparts"},
                "mesh.prepart_mfem", error_msg_)) {
            valid_ = false;
            return;
        }
        if (!prepart["directory"]) {
            error_msg_ = "Missing required parameter: mesh.prepart_mfem.directory";
            valid_ = false;
            return;
        }
        if (!prepart["nparts"]) {
            error_msg_ = "Missing required parameter: mesh.prepart_mfem.nparts";
            valid_ = false;
            return;
        }
        int nparts = prepart["nparts"].as<int>();
        if (nparts < 1) {
            error_msg_ = "mesh.prepart_mfem.nparts must be >= 1";
            valid_ = false;
            return;
        }
    } else {
        error_msg_ = "Invalid mesh.type: " + mesh_type
                   + " (must be 'internal', 'external', or 'prepart_mfem')";
        valid_ = false;
        return;
    }


    // mesh.max_freq (required)
    if (!mesh["max_freq"]) {
        error_msg_ = "Missing required parameter: mesh.max_freq";
        valid_ = false;
        return;
    }
    if (mesh["max_freq"].as<real_t>() <= 0.0) {
        error_msg_ = "mesh.max_freq must be > 0";
        valid_ = false;
        return;
    }

    // mesh.ppw (required)
    if (!mesh["ppw"]) {
        error_msg_ = "Missing required parameter: mesh.ppw";
        valid_ = false;
        return;
    }
    if (mesh["ppw"].as<real_t>() <= 0.0) {
        error_msg_ = "mesh.ppw must be > 0";
        valid_ = false;
        return;
    }

    //=============================================================================
    // Validate Material Section
    //=============================================================================
    YAML::Node mat = root_["material"];
    if (!mat) {
        error_msg_ = "Missing required section: material";
        valid_ = false;
        return;
    }
    // Material strict-key validation accepts the union of flat and
    // coupled keys; the specific flat/coupled path below enforces which
    // of those keys are required for the chosen type.
    if (!CheckKnownKeys(mat, {
            "type", "format", "file",
            "vp", "vs", "rho",
            "vp_file", "vs_file", "rho_file",
            "attenuation",
            "fluid", "solid"
        }, "material", error_msg_)) {
        valid_ = false;
        return;
    }

    // material.type
    if (!mat["type"]) {
        error_msg_ = "Missing required parameter: material.type";
        valid_ = false;
        return;
    }
    auto mat_type_str = mat["type"].as<std::string>();
    auto mat_type = StringToMaterialType(mat_type_str); // validate - will be abort if invalid type

    // Coupled (fluid-solid) materials validate nested fluid/solid blocks
    // instead of a flat format+params block. Short-circuit the rest of the
    // flat-format validation on success.
    if (mat_type == MaterialType::Coupled) {
        // Top-level keys reserved for flat materials must not be present
        // alongside fluid/solid; attenuation in particular must live inside
        // each sub-block to make the per-domain Q semantics explicit.
        if (mat["attenuation"]) {
            error_msg_ = "material.attenuation is not allowed at top level"
                         " when material.type = coupled. Move the block into"
                         " material.fluid.attenuation and/or"
                         " material.solid.attenuation.";
            valid_ = false;
            return;
        }
        if (mat["format"] || mat["file"] || mat["vp"] || mat["vs"] || mat["rho"]
            || mat["vp_file"] || mat["vs_file"] || mat["rho_file"]) {
            error_msg_ = "material.{format,file,vp,vs,rho,vp_file,vs_file,"
                         "rho_file} are not allowed at top level when"
                         " material.type = coupled. Put them under"
                         " material.fluid / material.solid.";
            valid_ = false;
            return;
        }
        if (!ValidateMaterialSubNode(mat["fluid"], "fluid", error_msg_)) {
            valid_ = false;
            return;
        }
        if (!ValidateMaterialSubNode(mat["solid"], "solid", error_msg_)) {
            valid_ = false;
            return;
        }
        const int fa = mat["fluid"]["attribute"].as<int>();
        const int sa = mat["solid"]["attribute"].as<int>();
        if (fa == sa) {
            error_msg_ = "material.fluid.attribute and material.solid.attribute "
                         "must differ (got " + std::to_string(fa) + " for both)";
            valid_ = false;
            return;
        }
        // Physical consistency: fluid must be acoustic, solid must be elastic.
        const auto ft = StringToMaterialType(mat["fluid"]["type"].as<std::string>());
        const auto st = StringToMaterialType(mat["solid"]["type"].as<std::string>());
        if (ft != MaterialType::IsotropicAcoustic) {
            error_msg_ = "material.fluid.type must be acoustic (got "
                       + mat["fluid"]["type"].as<std::string>() + ")";
            valid_ = false;
            return;
        }
        if (st != MaterialType::IsotropicElastic &&
            st != MaterialType::AnisotropicElastic) {
            error_msg_ = "material.solid.type must be elastic (got "
                       + mat["solid"]["type"].as<std::string>() + ")";
            valid_ = false;
            return;
        }
        // No flat format/attenuation checks apply when type=coupled.
    } else {

    // material.format
    if (!mat["format"]) {
        error_msg_ = "Missing required parameter: material.format";
        valid_ = false;
        return;
    }
    std::string mat_format = mat["format"].as<std::string>();

    // Validate based on material format
    if (mat_format == "constant") {
        
        if (mat_type == MaterialType::IsotropicElastic) {
            if (!mat["vp"] || !mat["vs"] || !mat["rho"]) {
                error_msg_ = "Constant elastic material requires vp, vs, and rho parameters";
                valid_ = false;
                return;
            }
        } else if (mat_type == MaterialType::IsotropicAcoustic) {
            if (!mat["vp"] || !mat["rho"]) {
                error_msg_ = "Constant acoustic material requires vp and rho parameters";
                valid_ = false;
                return;
            }
        } else {
            error_msg_ = "Constant format not supported for anisotropic materials";
            valid_ = false;
            return;
        }

    } else if (mat_format == "grid") {
        if (!mat["file"]) {
            error_msg_ = "Grid material format requires 'file' parameter";
            valid_ = false;
            return;
        }
    } else if (mat_format == "by_attribute") {
        if (!mat["file"]) {
            error_msg_ = "by_attribute material format requires 'file' parameter";
            valid_ = false;
            return;
        }
    } else if (mat_format == "by_attribute_mixed") {
        if (!mat["file"]) {
            error_msg_ = "by_attribute_mixed material format requires 'file' parameter";
            valid_ = false;
            return;
        }
    } else if (mat_format == "adios2") {
        if (!mat["vp_file"]) {
            error_msg_ = "ADIOS2 material format requires 'vp_file' parameter";
            valid_ = false;
            return;
        }
        if (!mat["rho_file"]) {
            error_msg_ = "ADIOS2 material format requires 'rho_file' parameter";
            valid_ = false;
            return;
        }
        if (mat_type == MaterialType::IsotropicElastic && !mat["vs_file"]) {
            error_msg_ = "ADIOS2 elastic material format requires 'vs_file' parameter";
            valid_ = false;
            return;
        }
    } else {
        error_msg_ = "Invalid material.format: " + mat_format;
        valid_ = false;
        return;
    }

    // Validate attenuation when enabled
    YAML::Node atten = mat["attenuation"];
    if (atten && !CheckKnownKeys(atten, {
            "enabled", "f0", "n_units",
            "qkappa", "qmu", "files"
        }, "material.attenuation", error_msg_)) {
        valid_ = false;
        return;
    }
    if (atten && atten["files"] &&
        !CheckKnownKeys(atten["files"],
            {"qkappa", "qmu"},
            "material.attenuation.files", error_msg_)) {
        valid_ = false;
        return;
    }
    bool atten_enabled = atten && atten["enabled"] && atten["enabled"].as<bool>();
    if (atten_enabled) {
        // f0 is required
        if (!atten["f0"]) {
            error_msg_ = "Missing required parameter: material.attenuation.f0 (when attenuation enabled)";
            valid_ = false;
            return;
        }
        // n_units is required
        if (!atten["n_units"]) {
            error_msg_ = "Missing required parameter: material.attenuation.n_units (when attenuation enabled)";
            valid_ = false;
            return;
        }
        // Qkappa / Qmu YAML scalars are only consumed for format=constant.
        // For other formats the Q values come from the data file (grid, .bp,
        // by_attribute table). Reject mixed specification to avoid silent
        // misuse.
        if (mat_format == "constant") {
            if (!atten["qkappa"]) {
                error_msg_ = "Missing required parameter: material.attenuation.qkappa (when attenuation enabled with constant format)";
                valid_ = false;
                return;
            }
            // Qmu required for elastic
            if (mat_type == MaterialType::IsotropicElastic) { //should be added for anisotropic, others.
                if (!atten["qmu"]) {
                    error_msg_ = "Missing required parameter: material.attenuation.qmu (when attenuation enabled with constant elastic format)";
                    valid_ = false;
                    return;
                }
            }
        } else {
            if (atten["qkappa"] || atten["qmu"]) {
                error_msg_ = "material.attenuation.qkappa/qmu may only be set"
                             " when material.format = constant. With format='"
                             + mat_format + "', Q values come from the data file."
                             " Remove qkappa/qmu from material.attenuation.";
                valid_ = false;
                return;
            }
        }
    }

    }  // end else (flat material validation)

    //=============================================================================
    // Validate boundary Conditions Section
    //=============================================================================
    YAML::Node bc = root_["boundary"];
    if (!bc) {
        error_msg_ = "Missing required section: boundary";
        valid_ = false;
        return;
    }
    if (!CheckKnownKeys(bc,
            {"absorbing", "dirichlet"},
            "boundary", error_msg_)) {
        valid_ = false;
        return;
    }
    // boundary.absorbing is optional. When omitted, no absorbing BC is
    // applied (equivalent to sides: []).
    YAML::Node abc = bc["absorbing"];
    if (abc) {
        if (!CheckKnownKeys(abc,
                {"sides", "thickness", "alpha"},
                "boundary.absorbing", error_msg_)) {
            valid_ = false;
            return;
        }
        // sides is optional; absent or [] means no absorbing layer.
        std::vector<std::string> sides;
        if (abc["sides"]) {
            sides = abc["sides"].as<std::vector<std::string>>();
            for (const auto& side : sides) {
                if (ParseBoundarySide(side, dim) < 0) {
                    error_msg_ = "Invalid boundary side: " + side;
                    valid_ = false;
                    return;
                }
            }
        }
        // thickness/alpha only required (and only consumed) when sides
        // is non-empty. Empty sides → absorbing effectively disabled.
        if (!sides.empty()) {
            if (!abc["thickness"]) {
                error_msg_ = "Missing required parameter: boundary.absorbing.thickness"
                             " (when sides is non-empty)";
                valid_ = false;
                return;
            }
            if (!abc["alpha"]) {
                error_msg_ = "Missing required parameter: boundary.absorbing.alpha"
                             " (when sides is non-empty)";
                valid_ = false;
                return;
            }
            if (abc["alpha"].as<real_t>() < 0) {
                error_msg_ = "Invalid parameter: boundary.absorbing.alpha (must be non-negative)";
                valid_ = false;
                return;
            }
        }
    }

    // Dirichlet BC section (optional). Canonical form is a plain string
    // list (`dirichlet: [bottom, left]`); legacy `{attributes: [...]}`
    // wrapper is rejected with a migration message.
    YAML::Node dirichlet = bc["dirichlet"];
    if (dirichlet) {
        if (dirichlet.IsMap()) {
            error_msg_ = "boundary.dirichlet is now a plain string list "
                         "(`dirichlet: [bottom, left]`). Remove the "
                         "`attributes:` wrapper.";
            valid_ = false;
            return;
        }
        if (!dirichlet.IsSequence()) {
            error_msg_ = "boundary.dirichlet must be a list of side names or "
                         "attribute ids";
            valid_ = false;
            return;
        }
        std::vector<std::string> attrs = dirichlet.as<std::vector<std::string>>();
        for (const auto& attr : attrs) {
            int parsed = ParseBoundaryAttributeValue(attr, dim);
            if (parsed < 0) {
                error_msg_ = "Invalid boundary attribute: " + attr;
                valid_ = false;
                return;
            }
        }
    }


    //=============================================================================
    // Validate Sources Section
    //=============================================================================
    if (!root_["sources"]) {
        error_msg_ = "Missing required section: sources";
        valid_ = false;
        return;
    }
    YAML::Node sources = root_["sources"];
    if (!CheckKnownKeys(sources, {"file", "shot_id", "list"},
            "sources", error_msg_)) {
        valid_ = false;
        return;
    }

    // Input mode is inferred from key presence:
    //   `list:` only            → inline YAML sources
    //   `file:` only            → HDF5 bundle (/shots/<shot_id>/sources/)
    // Both present → ambiguous → ABORT.
    const bool has_list = static_cast<bool>(sources["list"]);
    const bool has_file = static_cast<bool>(sources["file"]);
    if (has_list && has_file) {
        error_msg_ = "sources.list and sources.file are mutually exclusive "
                     "(inline list OR HDF5 bundle, not both)";
        valid_ = false;
        return;
    }
    if (!has_list && !has_file) {
        error_msg_ = "sources must define either `list:` (inline) or "
                     "`file:` (HDF5 bundle)";
        valid_ = false;
        return;
    }
    if (sources["shot_id"]) {
        const int sid = sources["shot_id"].as<int>();
        if (sid < 0) {
            error_msg_ = "Invalid parameter: sources.shot_id must be >= 0";
            valid_ = false;
            return;
        }
    }

    YAML::Node src_list = sources["list"];

    if (has_file) {
        // HDF5 bundle path: just verify the file exists. Schema details
        // are checked inside HDF5SourceReceiverReader at parse time.
        std::string src_file = sources["file"].as<std::string>();
        std::ifstream infile(src_file);
        if (!infile.good()) {
            error_msg_ = "Source file not found: " + src_file;
            valid_ = false;
            return;
        }
    }
    else {
        if (src_list && src_list.IsSequence()) {
            for (size_t i = 0; i < src_list.size(); i++) {
                YAML::Node src = src_list[i];
                std::string src_idx = "sources.list[" + std::to_string(i) + "]";

                if (!CheckKnownKeys(src, {
                        "id", "name", "type", "location", "direction",
                        "moment_tensor", "wavelet"
                    }, src_idx, error_msg_)) {
                    valid_ = false;
                    return;
                }

                // type required
                if (!src["type"]) {
                    error_msg_ = "Missing required parameter: " + src_idx + ".type";
                    valid_ = false;
                    return;
                }
                std::string src_type = src["type"].as<std::string>();
                if (src_type == "force") {
                    if(!src["direction"]) {
                        error_msg_ = "Missing required parameter: " + src_idx + ".direction (for force source)";
                        valid_ = false;
                        return;
                    }
                    std::vector<real_t> dir = src["direction"].as<std::vector<real_t>>();
                    if (dir.size() != dim) {
                        error_msg_ = "Invalid parameter: " + src_idx + ".direction (must have length equal to dimension)";
                        valid_ = false;
                        return;
                    }
                    if (src["moment_tensor"]) {
                        error_msg_ = "Invalid parameter: " + src_idx
                                   + ".moment_tensor not allowed for type=force";
                        valid_ = false;
                        return;
                    }
                }
                else if (src_type == "moment_tensor") {
                    if (!src["moment_tensor"]) {
                        error_msg_ = "Missing required parameter: " + src_idx + ".moment_tensor (for moment_tensor source)";
                        valid_ = false;
                        return;
                    }
                    YAML::Node mt = src["moment_tensor"];
                    const std::set<std::string> allowed = (dim == 2)
                        ? std::set<std::string>{"Mxx", "Myy", "Mxy"}
                        : std::set<std::string>{"Mxx", "Myy", "Mzz", "Mxy", "Mxz", "Myz"};
                    if (!CheckKnownKeys(mt, allowed,
                            src_idx + ".moment_tensor", error_msg_)) {
                        valid_ = false;
                        return;
                    }
                    for (const auto& k : allowed) {
                        if (!mt[k]) {
                            error_msg_ = "Missing required moment tensor component '" + k
                                       + "' for " + std::to_string(dim) + "D moment_tensor source at " + src_idx;
                            valid_ = false;
                            return;
                        }
                    }
                    if (src["direction"]) {
                        error_msg_ = "Invalid parameter: " + src_idx
                                   + ".direction not allowed for type=moment_tensor";
                        valid_ = false;
                        return;
                    }
                }
                else if (src_type == "pressure") {
                    if (src["direction"]) {
                        error_msg_ = "Invalid parameter: " + src_idx
                                   + ".direction not allowed for type=pressure";
                        valid_ = false;
                        return;
                    }
                    if (src["moment_tensor"]) {
                        error_msg_ = "Invalid parameter: " + src_idx
                                   + ".moment_tensor not allowed for type=pressure";
                        valid_ = false;
                        return;
                    }
                }
                else {
                    error_msg_ = "Invalid source.type at " + src_idx + " (must be 'force', 'moment_tensor', or 'pressure')";
                    valid_ = false;
                    return;
                }

                // location required
                if (!src["location"]) {
                    error_msg_ = "Missing required parameter: " + src_idx + ".location";
                    valid_ = false;
                    return;
                }
                std::vector<real_t> loc = src["location"].as<std::vector<real_t>>();
                if (loc.size() != dim) {
                    error_msg_ = "Invalid parameter: " + src_idx + ".location (must have length equal to dimension)";
                    valid_ = false;
                    return;
                }


                // wavelet section required
                if (!src["wavelet"]) {
                    error_msg_ = "Missing required section: " + src_idx + ".wavelet";
                    valid_ = false;
                    return;
                }
                YAML::Node wv = src["wavelet"];
                if (!CheckKnownKeys(wv,
                        {"type", "frequency", "amplitude", "delay", "file"},
                        src_idx + ".wavelet", error_msg_)) {
                    valid_ = false;
                    return;
                }

                // wavelet.type required
                if (!wv["type"]) {
                    error_msg_ = "Missing required parameter: " + src_idx + ".wavelet.type";
                    valid_ = false;
                    return;
                }

                std::string wv_type = wv["type"].as<std::string>();

                if (wv_type == "ricker" || wv_type == "gaussian") {
                    // frequency required for analytic wavelets (PPW check)
                    if (!wv["frequency"]) {
                        error_msg_ = "Missing required parameter: " + src_idx + ".wavelet.frequency";
                        valid_ = false;
                        return;
                    }
                    // wavelet.amplitude required
                    if (!wv["amplitude"]) {
                        error_msg_ = "Missing required parameter: " + src_idx + ".wavelet.amplitude";
                        valid_ = false;
                        return;
                    }
                    // wavelet.delay is optional (default 0.0 in parser)
                }
                else if (wv_type == "external") {
                    // wavelet.file required
                    // amplitude and delay are NOT required for external
                    if (!wv["file"]) {
                        error_msg_ = "Missing required parameter: " + src_idx + ".wavelet.file";
                        valid_ = false;
                        return;
                    }
                    std::string wv_file = wv["file"].as<std::string>();
                    std::ifstream infile(wv_file);
                    if (!infile.good()) {
                        error_msg_ = "Wavelet file not found: " + wv_file + " (for external wavelet at " + src_idx + ")";
                        valid_ = false;
                        return;
                    }
                }
                else {
                    error_msg_ = "Invalid wavelet.type at " + src_idx + " (must be 'ricker', 'gaussian', or 'external')";
                    valid_ = false;
                    return;
                }
            }
        }
    }

    //=============================================================================
    // Validate Receivers Section
    // (optional for inversion mode — receiver geometry comes from sources.file)
    //=============================================================================
    // `receivers:` is optional. Forward sims that only want wavefield
    // snapshots can omit it entirely; FWI auto-uses the bundled HDF5
    // receivers when sources.file is given.
    if (root_["receivers"]) {
    YAML::Node receivers = root_["receivers"];
    if (!CheckKnownKeys(receivers,
            {"type", "output", "list", "line"},
            "receivers", error_msg_)) {
        valid_ = false;
        return;
    }

    // receivers.type is required UNLESS sources.file is set — in that case
    // the per-receiver `@types` attribute in the HDF5 bundle is the canonical
    // source of truth and an empty parent default is allowed (the runtime
    // reader aborts cleanly when a given receiver has neither).
    const bool has_sources_file =
        root_["sources"] && root_["sources"]["file"];
    if (!receivers["type"] && !has_sources_file) {
        error_msg_ = "Missing required parameter: receivers.type "
                     "(or set sources.file to inherit types from the HDF5 bundle)";
        valid_ = false;
        return;
    }
    if (receivers["type"]) {
        std::vector<std::string> types =
            receivers["type"].as<std::vector<std::string>>();
        for (const auto& type_str : types) {
            auto rec_type = StringToReceiverType(type_str); // validate
        }
    }

    // receivers.output section
    if (!receivers["output"]) {
        error_msg_ = "Missing required section: receivers.output";
        valid_ = false;
        return;
    }
    YAML::Node recv_output = receivers["output"];
    if (!CheckKnownKeys(recv_output, {"formats", "filename", "buffer_steps"},
            "receivers.output", error_msg_)) {
        valid_ = false;
        return;
    }
    if (recv_output["buffer_steps"]
            && recv_output["buffer_steps"].as<int>() < 0) {
        error_msg_ = "Invalid parameter: receivers.output.buffer_steps must be >= 0";
        valid_ = false;
        return;
    }
    if (!recv_output["formats"]) {
        error_msg_ = "Missing required parameter: receivers.output.formats";
        valid_ = false;
        return;
    }

    YAML::Node fmt_node = recv_output["formats"];
    std::vector<std::string> recv_formats;
    if (!fmt_node.IsSequence() || fmt_node.size() == 0) {
        error_msg_ = "Invalid parameter: receivers.output.formats must be a "
                     "non-empty list (`[ascii, hdf5, ...]`).";
        valid_ = false;
        return;
    }
    for (const auto& item : fmt_node) {
        if (!item.IsScalar()) {
            error_msg_ = "Invalid parameter: receivers.output.formats "
                         "entries must be plain strings.";
            valid_ = false;
            return;
        }
        recv_formats.push_back(item.as<std::string>());
    }

    std::set<std::string> seen;
    for (const auto& f : recv_formats) {
        if (f != "ascii" && f != "hdf5" && f != "su") {
            error_msg_ = "Invalid parameter: receivers.output.format entry '" + f +
                         "' (must be 'ascii', 'hdf5', or 'su')";
            valid_ = false;
            return;
        }
        if (!seen.insert(f).second) {
            error_msg_ = "Invalid parameter: receivers.output.format (duplicate entry '" + f + "')";
            valid_ = false;
            return;
        }
    }
    // `filename:` is optional; default `shot` is used as the bundle name
    // for hdf5/su (ascii filenames are derived per-receiver and ignore
    // this key entirely).





    // Validate receiver line if present
    YAML::Node line = receivers["line"];
    if (line) {
        if (!CheckKnownKeys(line,
                {"start", "end", "count"},
                "receivers.line", error_msg_)) {
            valid_ = false;
            return;
        }
        if (!line["start"]) {
            error_msg_ = "Missing required parameter: receivers.line.start";
            valid_ = false;
            return;
        }
        if (!line["end"]) {
            error_msg_ = "Missing required parameter: receivers.line.end";
            valid_ = false;
            return;
        }
        if (!line["count"]) {
            error_msg_ = "Missing required parameter: receivers.line.count";
            valid_ = false;
            return;
        }
    }

    // Validate inline receivers if present
    YAML::Node recv_list = receivers["list"];
    if (recv_list && recv_list.IsSequence()) {
        for (size_t i = 0; i < recv_list.size(); i++) {
            YAML::Node rec = recv_list[i];
            std::string rec_idx = "receivers.list[" + std::to_string(i) + "]";

            if (!CheckKnownKeys(rec,
                    {"name", "location", "weight"},
                    rec_idx, error_msg_)) {
                valid_ = false;
                return;
            }

            // location required
            if (!rec["location"]) {
                error_msg_ = "Missing required parameter: " + rec_idx + ".location";
                valid_ = false;
                return;
            }

            std::vector<real_t> loc = rec["location"].as<std::vector<real_t>>();
            if (loc.size() != dim) {
                error_msg_ = "Invalid parameter: " + rec_idx + ".location (must have length equal to dimension)";
                valid_ = false;
                return;
            }

            if (rec["weight"] && rec["weight"].as<real_t>() < 0.0) {
                error_msg_ = "Invalid parameter: " + rec_idx + ".weight must be >= 0";
                valid_ = false;
                return;
            }
        }
    }
    }  // if (root_["receivers"])

    //=============================================================================
    // Validate Device Section
    //=============================================================================
    if (!root_["device"].IsDefined()) {
        error_msg_ = "Missing required section: device";
        valid_ = false;
        return;
    }

    YAML::Node device_node = root_["device"];
    if (!CheckKnownKeys(device_node,
            {"type"},
            "device", error_msg_)) {
        valid_ = false;
        return;
    }
    if (!device_node["type"]) {
        error_msg_ = "Missing required parameter: device.type";
        valid_ = false;
        return;
    }
    {
        const std::string dev = device_node["type"].as<std::string>();
        static const std::vector<std::string> allowed_devices = {
            "cpu", "omp", "cuda", "hip",
            "raja-cpu", "raja-omp", "raja-cuda", "raja-hip"
        };
        if (std::find(allowed_devices.begin(), allowed_devices.end(), dev)
                == allowed_devices.end()) {
            error_msg_ = "Invalid parameter: device.type='" + dev
                       + "' (must be one of cpu, omp, cuda, hip,"
                         " raja-cpu, raja-omp, raja-cuda, raja-hip)";
            valid_ = false;
            return;
        }
    }

    //=============================================================================
    // Validate Simulation Mode and Inversion Parameters
    //=============================================================================
    std::string sim_mode = "forward";  // default
    if (sim["mode"]) {
        sim_mode = sim["mode"].as<std::string>();
        if (sim_mode != "forward" && sim_mode != "inversion"
            && sim_mode != "misfit_only") {
            error_msg_ = "Invalid simulation.mode: " + sim_mode
                         + " (must be 'forward', 'inversion', or 'misfit_only')";
            valid_ = false;
            return;
        }
    }

    if (sim_mode == "inversion" || sim_mode == "misfit_only") {
        YAML::Node inv = root_["inversion"];
        if (inv) {
            if (!CheckKnownKeys(inv, {
                    "misfit_type", "kernel_dir", "checkpoints", "sensitivity_backend"
                }, "inversion", error_msg_)) {
                valid_ = false;
                return;
            }
            if (inv["misfit_type"]) {
                std::string mt = inv["misfit_type"].as<std::string>();
                if (mt != "l2_waveform" && mt != "normalized_correlation") {
                    error_msg_ = "Invalid inversion.misfit_type: " + mt
                                 + " (must be 'l2_waveform' or 'normalized_correlation')";
                    valid_ = false;
                    return;
                }
            }
            YAML::Node ckpt = inv["checkpoints"];
            if (ckpt) {
                if (!CheckKnownKeys(ckpt, {"n", "storage", "device", "dir"},
                        "inversion.checkpoints", error_msg_)) {
                    valid_ = false;
                    return;
                }
                if (ckpt["n"] && ckpt["n"].as<int>() < 1) {
                    error_msg_ = "Invalid inversion.checkpoints.n: must be >= 1";
                    valid_ = false;
                    return;
                }
                if (ckpt["storage"]) {
                    std::string cs = ckpt["storage"].as<std::string>();
                    if (cs != "memory" && cs != "disk") {
                        error_msg_ = "Invalid inversion.checkpoints.storage:"
                                     " must be 'memory' or 'disk'";
                        valid_ = false;
                        return;
                    }
                    if (cs == "disk" && !ckpt["dir"]) {
                        error_msg_ = "Missing required parameter:"
                                     " inversion.checkpoints.dir"
                                     " (when storage is 'disk')";
                        valid_ = false;
                        return;
                    }
                }
                if (ckpt["device"]) {
                    std::string cd = ckpt["device"].as<std::string>();
                    if (cd != "auto" && cd != "host" && cd != "device") {
                        error_msg_ = "Invalid inversion.checkpoints.device:"
                                     " must be 'auto', 'host', or 'device'";
                        valid_ = false;
                        return;
                    }
                }
            }
        }

        // Inversion / misfit_only mode reads observed data from the bundled
        // HDF5 file pointed to by `sources.file` (with `sources.shot_id`).
        if (!sources["file"]) {
            error_msg_ = "Missing required parameter: sources.file"
                         " (HDF5 bundle with observed data) when"
                         " simulation.mode is '" + sim_mode + "'";
            valid_ = false;
            return;
        }
    }
}

// =============================================================================
// Helper Methods
// =============================================================================

std::string YamlConfig::ReadString(const YAML::Node& node, const std::string& key) const {
    if (!node || !node[key]) {
        MFEM_ABORT("Required string parameter '" << key << "' not found in config");
    }
    return node[key].as<std::string>();
}

real_t YamlConfig::Readreal_t(const YAML::Node& node, const std::string& key) const {
    if (!node || !node[key]) {
        MFEM_ABORT("Required real parameter '" << key << "' not found in config");
    }
    return node[key].as<real_t>();
}

int YamlConfig::ReadInt(const YAML::Node& node, const std::string& key) const {
    if (!node || !node[key]) {
        MFEM_ABORT("Required int parameter '" << key << "' not found in config");
    }
    return node[key].as<int>();
}

bool YamlConfig::ReadBool(const YAML::Node& node, const std::string& key) const {
    if (!node || !node[key]) {
        MFEM_ABORT("Required bool parameter '" << key << "' not found in config");
    }
    return node[key].as<bool>();
}

void YamlConfig::ReadRealArray(const YAML::Node& node, const std::string& key,
                                real_t* arr, int size) const {
    if (!node || !node[key] || !node[key].IsSequence()) {
        MFEM_ABORT("Required real array parameter '" << key << "' not found in config");
    }
    int n = std::min(size, static_cast<int>(node[key].size()));
    for (int i = 0; i < n; i++) {
        arr[i] = static_cast<real_t>(node[key][i].as<real_t>());
    }
    // Zero-fill remaining elements
    for (int i = n; i < size; i++) {
        arr[i] = 0.0;
    }
}

void YamlConfig::ReadIntArray(const YAML::Node& node, const std::string& key,
                               int* arr, int size) const {
    if (!node || !node[key] || !node[key].IsSequence()) {
        MFEM_ABORT("Required int array parameter '" << key << "' not found in config");
    }
    int n = std::min(size, static_cast<int>(node[key].size()));
    for (int i = 0; i < n; i++) {
        arr[i] = node[key][i].as<int>();
    }
    // Zero-fill remaining elements
    for (int i = n; i < size; i++) {
        arr[i] = 0;
    }
}

std::vector<std::string> YamlConfig::ReadStringArray(const YAML::Node& node,
                                                      const std::string& key) const {
    std::vector<std::string> result;
    if (node && node[key] && node[key].IsSequence()) {
        for (size_t i = 0; i < node[key].size(); i++) {
            result.push_back(node[key][i].as<std::string>());
        }
    }
    return result;
}

// =============================================================================
// Simulation Section
// =============================================================================

std::string YamlConfig::GetName() const {
    // Name is at root level, with default set in Validate()
    if (root_["name"]) {
        return root_["name"].as<std::string>();
    }
    return "Seismic wave simulation";
}

// GetPhysics() removed - physics type is inferred from MaterialType
// GetBackend() removed - unused

int YamlConfig::GetDimension() const {
    // Required - validated in Validate()
    return root_["simulation"]["dimension"].as<int>();
}

int YamlConfig::GetOrder() const {
    // order is required (validated in Validate())
    YAML::Node sim = root_["simulation"];
    // if (!sim || !sim["order"]) {
    //     MFEM_ABORT("simulation.order is required but not found in config");
    // }
    return sim["order"].as<int>();
}

int YamlConfig::GetNumSteps() const {
    return root_["simulation"]["steps"].as<int>();
}

real_t YamlConfig::GetDt() const {
    return root_["simulation"]["dt"].as<real_t>();
}

real_t YamlConfig::GetT0() const {
    YAML::Node sim = root_["simulation"];
    if (sim && sim["t0"]) {
        return sim["t0"].as<real_t>();
    }
    return 0.0;
}

real_t YamlConfig::GetCflFactor() const {
    YAML::Node sim = root_["simulation"];
    if (sim && sim["cfl_factor"]) {
        return sim["cfl_factor"].as<real_t>();
    }
    return real_t(0.3);
}

std::string YamlConfig::GetOutputDirectory() const {
    // Required - validated in Validate()
    return root_["simulation"]["directory"].as<std::string>();
}

// GetSeismogramFormat() removed - unused

bool YamlConfig::IsWavefieldOutputEnabled() const {
    if (!root_["vis"] || !root_["vis"]["wavefield"] ||
        !root_["vis"]["wavefield"]["enabled"]) {
        return false;
    }
    return root_["vis"]["wavefield"]["enabled"].as<bool>();
}

int YamlConfig::GetWavefieldInterval() const {
    // Required when wavefield enabled - validated in Validate()
    // Caller should check IsWavefieldOutputEnabled() first
    YAML::Node wf_node = root_["vis"]["wavefield"];
    return wf_node["interval"].as<int>();
}

// NOTE: GetWavefieldFormat() / WavefieldFormat() have been removed.
// Use GetWavefieldOutputConfig() and iterate its `formats` list.

// =============================================================================
// Multi-format visualization output configuration
// =============================================================================

namespace {

/// Parse a single output format entry from YAML.
///
/// Accepts two syntaxes per list entry:
///   * Scalar  (short):  `"gmt"`  — equivalent to `{type: "gmt"}` with
///                       all per-format options at defaults.
///   * Mapping (full):   `{type: "gmt", resolution: [100, 100], …}` —
///                       use this to override per-format options.
OutputFormatConfig ParseOutputFormat(const YAML::Node& node) {
    OutputFormatConfig fmt;
    if (node.IsScalar()) {
        fmt.type = node.as<std::string>();
        // `gmt` always needs per-format options (resolution, cross_sections
        // in 3D, components) — defaults alone produce a 100×100 magnitude
        // image (2D) or NO output at all (3D, cross_sections empty). Force
        // mapping form so users see and configure the knobs.
        if (fmt.type == "gmt") {
            MFEM_ABORT("output format `gmt` requires the mapping form "
                       "(e.g. `{type: gmt, resolution: [200,200], "
                       "cross_sections: {xy: [0.0]}}`) — plain-string "
                       "shorthand is not allowed for gmt.");
        }
        return fmt;
    }
    if (!node.IsMap()) {
        MFEM_ABORT("output format entry must be either a plain string "
                   "(e.g., `gmt`) or a mapping (`{type: gmt, ...}`).");
    }
    if (!node["type"]) {
        MFEM_ABORT("output format entry is missing the required `type:` key.");
    }
    // Strict-key validation on format entries — catches renamed or
    // mistyped keys that would otherwise be silently ignored.
    RequireKnownKeys(node, {
        "type",
        "refinement", "data_format", "compression",
        "resolution", "cross_sections"
    }, "output format entry");
    fmt.type = node["type"].as<std::string>();

    // ParaView options
    if (node["refinement"]) {
        fmt.refinement = node["refinement"].as<int>();
    }
    if (node["data_format"]) {
        fmt.data_format = node["data_format"].as<std::string>();
    }
    if (node["compression"]) {
        fmt.compression = node["compression"].as<int>();
    }

    // GMT options
    if (node["resolution"]) {
        auto res = node["resolution"];
        if (res.IsSequence() && res.size() == 2) {
            fmt.resolution[0] = res[0].as<int>();
            fmt.resolution[1] = res[1].as<int>();
        }
    }
    if (node["cross_sections"]) {
        auto cs = node["cross_sections"];
        if (cs["yz"]) {
            for (size_t i = 0; i < cs["yz"].size(); ++i) {
                fmt.cross_sections.yz.push_back(cs["yz"][i].as<real_t>());
            }
        }
        if (cs["xz"]) {
            for (size_t i = 0; i < cs["xz"].size(); ++i) {
                fmt.cross_sections.xz.push_back(cs["xz"][i].as<real_t>());
            }
        }
        if (cs["xy"]) {
            for (size_t i = 0; i < cs["xy"].size(); ++i) {
                fmt.cross_sections.xy.push_back(cs["xy"][i].as<real_t>());
            }
        }
    }

    return fmt;
}

}  // anonymous namespace

namespace {

/// Apply an output.wavefield(-like) YAML node's keys onto an existing
/// WavefieldOutputConfig, overriding whichever fields are present in
/// the node and leaving the rest untouched. Shared between the
/// top-level accessor and the per-side override accessor so both paths
/// follow identical parsing rules.
void ApplyWavefieldNode(const YAML::Node& wf, WavefieldOutputConfig& config) {
    if (wf["enabled"]) {
        config.enabled = wf["enabled"].as<bool>();
    }
    if (wf["interval"]) {
        config.interval = wf["interval"].as<int>();
    }
    if (wf["fields"]) {
        config.fields.clear();
        static const std::vector<std::string> valid_wf_fields =
            {"DISP", "VEL", "ACC", "PS"};
        for (size_t i = 0; i < wf["fields"].size(); ++i) {
            std::string f = wf["fields"][i].as<std::string>();
            if (std::find(valid_wf_fields.begin(), valid_wf_fields.end(), f)
                == valid_wf_fields.end()) {
                MFEM_ABORT("Unknown wavefield field '" + f
                           + "' (valid: DISP, VEL, ACC, PS)");
            }
            config.fields.push_back(f);
        }
    }
    if (wf["formats"]) {
        // Canonical form — replaces any previously accumulated formats
        // (per-side override replaces the inherited base).
        config.formats.clear();
        for (size_t i = 0; i < wf["formats"].size(); ++i) {
            config.formats.push_back(ParseOutputFormat(wf["formats"][i]));
        }
    }

    // Block-level `components:` is the single source of truth — applied
    // uniformly to every format entry. Only GMT consumes it; ParaView /
    // glvis ignore it.
    if (wf["components"]) {
        std::vector<int> comps;
        for (size_t i = 0; i < wf["components"].size(); ++i) {
            comps.push_back(wf["components"][i].as<int>());
        }
        for (auto& fmt : config.formats) {
            fmt.components = comps;
        }
    }
}

}  // anonymous namespace

WavefieldOutputConfig YamlConfig::GetWavefieldOutputConfig() const {
    WavefieldOutputConfig config;

    if (!root_["vis"] || !root_["vis"]["wavefield"]) {
        return config;
    }

    YAML::Node wf = root_["vis"]["wavefield"];
    ApplyWavefieldNode(wf, config);
    if (!config.enabled) {
        // Drop any formats we may have parsed before enabled=false was
        // applied, to preserve the pre-refactor early-return behavior.
        config.formats.clear();
        config.fields.clear();
    }
    return config;
}

WavefieldOutputConfig YamlConfig::GetWavefieldOutputConfig(
    const std::string& side) const {
    // Start from the top-level defaults (inherits interval / fields /
    // formats). A per-side block under `vis.wavefield.<side>` then
    // overrides whatever keys it sets; unspecified keys stay inherited.
    WavefieldOutputConfig config = GetWavefieldOutputConfig();

    if (!config.enabled) return config;
    if (!root_["vis"] || !root_["vis"]["wavefield"]) {
        return config;
    }
    YAML::Node wf = root_["vis"]["wavefield"];
    if (!wf[side]) return config;

    YAML::Node side_node = wf[side];
    ApplyWavefieldNode(side_node, config);

    // A side-level `enabled: false` explicitly mutes one domain's
    // wavefield output even when the global toggle is on.
    if (!config.enabled) {
        config.formats.clear();
        config.fields.clear();
    }
    return config;
}

namespace {

/// Shared parser for an `output.material`-like node. Mirrors
/// `ApplyWavefieldNode` — any key set in `mat` overrides the matching
/// field in `config`; unset keys leave `config` untouched. Used by both
/// the top-level accessor and the per-side override accessor so the
/// two paths follow identical rules.
void ApplyMaterialNode(const YAML::Node& mat, MaterialOutputConfig& config) {
    if (mat["enabled"]) {
        config.enabled = mat["enabled"].as<bool>();
    }
    if (mat["fields"]) {
        config.fields.clear();
        static const std::vector<std::string> valid_mat_fields =
            {"vp", "vs", "rho", "qkappa", "qmu"};
        for (size_t i = 0; i < mat["fields"].size(); ++i) {
            std::string f = mat["fields"][i].as<std::string>();
            if (std::find(valid_mat_fields.begin(), valid_mat_fields.end(), f)
                == valid_mat_fields.end()) {
                MFEM_ABORT("Unknown material field '" + f
                           + "' (valid: vp, vs, rho, qkappa, qmu)");
            }
            config.fields.push_back(f);
        }
    }
    if (mat["formats"]) {
        // Explicit formats list — replaces any previously accumulated
        // formats (per-side override should not inherit base formats
        // that the user is explicitly replacing).
        config.formats.clear();
        for (size_t i = 0; i < mat["formats"].size(); ++i) {
            config.formats.push_back(ParseOutputFormat(mat["formats"][i]));
        }
    }
}

}  // anonymous namespace

MaterialOutputConfig YamlConfig::GetMaterialOutputConfig() const {
    MaterialOutputConfig config;

    if (!root_["vis"] || !root_["vis"]["material"]) {
        return config;
    }

    YAML::Node mat = root_["vis"]["material"];
    ApplyMaterialNode(mat, config);
    if (!config.enabled) {
        config.formats.clear();
        config.fields.clear();
    }
    return config;
}

MaterialOutputConfig YamlConfig::GetMaterialOutputConfig(
    const std::string& side) const {
    // Start from the top-level defaults (inherits fields / formats).
    // A per-side block under `vis.material.<side>` then overrides
    // whatever keys it sets; unspecified keys stay inherited.
    MaterialOutputConfig config = GetMaterialOutputConfig();

    if (!config.enabled) return config;
    if (!root_["vis"] || !root_["vis"]["material"]) {
        return config;
    }
    YAML::Node mat = root_["vis"]["material"];
    if (!mat[side]) return config;

    ApplyMaterialNode(mat[side], config);
    if (!config.enabled) {
        config.formats.clear();
        config.fields.clear();
    }
    return config;
}

bool YamlConfig::GetMeshSave() const {
    YAML::Node vis = root_["vis"];
    if (!vis || !vis["mesh"] || !vis["mesh"]["enabled"]) return false;
    return vis["mesh"]["enabled"].as<bool>();
}

std::string YamlConfig::GetVisDirectory() const {
    // `vis.directory:` (explicit) overrides the default
    // `<simulation.output.directory>/vis` derivation.
    YAML::Node vis = root_["vis"];
    if (vis && vis["directory"]) {
        return vis["directory"].as<std::string>();
    }
    return GetOutputDirectory() + "/vis";
}

int YamlConfig::GetLogInterval() const {
    YAML::Node sim = root_["simulation"];
    if (sim && sim["log_interval"]) {
        return sim["log_interval"].as<int>();
    }
    return 99999;
}

std::string YamlConfig::GetSummaryFile() const {
    // Optional - returns empty string if not specified
    YAML::Node sim = root_["simulation"];
    if (sim && sim["summary_file"]) {
        return sim["summary_file"].as<std::string>();
    }
    return "";
}

// =============================================================================
// Mesh Section
// =============================================================================

real_t YamlConfig::GetMaxFreq() const {
    // Required - validated in Validate()
    return root_["mesh"]["max_freq"].as<real_t>();
}

real_t YamlConfig::GetPPW() const {
    // Required - validated in Validate()
    return root_["mesh"]["ppw"].as<real_t>();
}

std::string YamlConfig::GetMeshType() const {
    // Required - validated in Validate()
    return root_["mesh"]["type"].as<std::string>();
}

std::string YamlConfig::GetMeshFile() const {
    // Required for external mesh - validated in Validate()
    return root_["mesh"]["file"].as<std::string>();
}

void YamlConfig::GetMeshOrigin(real_t* origin) const {
    // Required for internal mesh - validated in Validate()
    int dim = GetDimension();
    YAML::Node orig = root_["mesh"]["origin"];
    for (int i = 0; i < dim; i++) {
        origin[i] = orig[i].as<real_t>();
    }
}

void YamlConfig::GetMeshSize(real_t* size) const {
    // Required for internal mesh - validated in Validate()
    int dim = GetDimension();
    YAML::Node sz = root_["mesh"]["size"];
    for (int i = 0; i < dim; i++) {
        size[i] = sz[i].as<real_t>();
    }
}

void YamlConfig::GetMeshElements(int* nel) const {
    // Required for internal mesh - validated in Validate()
    int dim = GetDimension();
    YAML::Node el = root_["mesh"]["elements"];
    for (int i = 0; i < dim; i++) {
        nel[i] = el[i].as<int>();
    }
}

std::string YamlConfig::GetMeshPartitionMethod() const {
    // Optional - defaults to "metis"
    YAML::Node mesh = root_["mesh"];
    if (mesh["partition_method"]) {
        return mesh["partition_method"].as<std::string>();
    }
    return "metis";
}

void YamlConfig::GetPartitionGrid(int* nxyz) const {
    // Required when partition: cartesian
    int dim = GetDimension();
    YAML::Node grid = root_["mesh"]["partition_grid"];
    for (int i = 0; i < dim; i++) {
        nxyz[i] = grid[i].as<int>();
    }
}

// -----------------------------------------------------------------------------
// Partitioned Mesh (pre-partitioned files)
// -----------------------------------------------------------------------------

bool YamlConfig::UsePartitionedMesh() const {
    std::string mesh_type = GetMeshType();
    return (mesh_type == "prepart_mfem");
}

std::string YamlConfig::GetPartitionDirectory() const {
    YAML::Node prepart = root_["mesh"]["prepart_mfem"];
    if (!prepart || !prepart["directory"]) {
        MFEM_ABORT("mesh.prepart_mfem.directory is required when mesh.type is 'prepart_mfem'");
    }
    return prepart["directory"].as<std::string>();
}

int YamlConfig::GetPartitionCount() const {
    YAML::Node prepart = root_["mesh"]["prepart_mfem"];
    if (!prepart || !prepart["nparts"]) {
        MFEM_ABORT("mesh.prepart_mfem.nparts is required when mesh.type is 'prepart_mfem'");
    }
    return prepart["nparts"].as<int>();
}

// =============================================================================
// Material Section
// =============================================================================

std::string YamlConfig::GetMaterialType() const {
    // Required - validated in Validate()
    auto mat = root_["material"]["type"].as<std::string>();
    StringToMaterialType(mat); // validate - will MFEM_ABORT if invalid
    return mat;
}

std::string YamlConfig::GetMaterialFormat() const {
    // Required - validated in Validate()
    return root_["material"]["format"].as<std::string>();
}

bool YamlConfig::IsCoupledMaterial() const {
    if (!root_["material"] || !root_["material"]["type"]) return false;
    return root_["material"]["type"].as<std::string>() == "coupled";
}

CoupledMaterialConfig YamlConfig::GetCoupledMaterialConfig() const {
    MFEM_VERIFY(IsCoupledMaterial(),
                "GetCoupledMaterialConfig called on a non-coupled material "
                "(material.type must be 'coupled')");
    CoupledMaterialConfig cc;
    YAML::Node mat = root_["material"];
    cc.fluid_attribute = mat["fluid"]["attribute"].as<int>();
    cc.solid_attribute = mat["solid"]["attribute"].as<int>();
    cc.fluid = ParseMaterialSubNode(mat["fluid"], "fluid");
    cc.solid = ParseMaterialSubNode(mat["solid"], "solid");
    return cc;
}

std::string YamlConfig::GetMaterialFile() const {
    // Required for hdf5/ascii format - validated in Validate()
    return root_["material"]["file"].as<std::string>();
}



void YamlConfig::GetConstantMaterialVpVsRho(real_t* vp, real_t* vs, real_t* rho) const {
    // Required for constant format - validated in Validate()
    YAML::Node mat = root_["material"];
    *vp = mat["vp"].as<real_t>();
    *vs = mat["vs"].as<real_t>();
    *rho = mat["rho"].as<real_t>();
}

void YamlConfig::GetConstantMaterialVpRho(real_t* vp, real_t* rho) const {
    // Required for constant acoustic format - validated in Validate()
    YAML::Node mat = root_["material"];
    *vp = mat["vp"].as<real_t>();
    *rho = mat["rho"].as<real_t>();
}

bool YamlConfig::IsAttenuationEnabled() const {
    YAML::Node atten = root_["material"]["attenuation"];
    if (!atten || !atten["enabled"]) {
        return false;
    }
    return atten["enabled"].as<bool>();
}

real_t YamlConfig::GetAttenuationF0() const {
    // Required when attenuation enabled - validated in Validate()
    return root_["material"]["attenuation"]["f0"].as<real_t>();
}

int YamlConfig::GetAttenuationNumUnits() const {
    // Required when attenuation enabled - validated in Validate()
    return root_["material"]["attenuation"]["n_units"].as<int>();
}

real_t YamlConfig::GetConstantQkappa() const {
    YAML::Node atten = root_["material"]["attenuation"];
    // Required when attenuation enabled with constant format - validated in Validate()
    return atten["qkappa"].as<real_t>();
}

real_t YamlConfig::GetConstantQmu() const {
    YAML::Node atten = root_["material"]["attenuation"];
    // For acoustic materials, Qmu may not be specified - return -1 as sentinel
    if (!atten || !atten["qmu"]) {
        return -1.0;
    }
    // Required when attenuation enabled with constant elastic format - validated in Validate()
    return atten["qmu"].as<real_t>();
}

std::string YamlConfig::GetMaterialByAttributeFile() const {
    // Optional - returns empty string if not specified (sentinel value OK)
    YAML::Node mat = root_["material"];
    if (mat && mat["file"]) {
        return mat["file"].as<std::string>();
    }
    return "";
}

std::string YamlConfig::GetADIOS2VpFile() const {
    YAML::Node mat = root_["material"];
    if (mat && mat["vp_file"]) {
        return mat["vp_file"].as<std::string>();
    }
    return "";
}

std::string YamlConfig::GetADIOS2VsFile() const {
    YAML::Node mat = root_["material"];
    if (mat && mat["vs_file"]) {
        return mat["vs_file"].as<std::string>();
    }
    return "";
}

std::string YamlConfig::GetADIOS2RhoFile() const {
    YAML::Node mat = root_["material"];
    if (mat && mat["rho_file"]) {
        return mat["rho_file"].as<std::string>();
    }
    return "";
}

std::string YamlConfig::GetADIOS2QkappaFile() const {
    YAML::Node files = root_["material"]["attenuation"]["files"];
    if (files && files["qkappa"]) {
        return files["qkappa"].as<std::string>();
    }
    return "";
}

std::string YamlConfig::GetADIOS2QmuFile() const {
    YAML::Node files = root_["material"]["attenuation"]["files"];
    if (files && files["qmu"]) {
        return files["qmu"].as<std::string>();
    }
    return "";
}

// =============================================================================
// Boundary Section
// =============================================================================

ABCConfig YamlConfig::GetABCConfig() const {
    ABCConfig config;
    YAML::Node abc = root_["boundary"]["absorbing"];

    if (!abc) {
        return config;  // No absorbing boundary defined
    }

    if (abc["sides"]) {
        config.sides = ReadStringArray(abc, "sides");
    }
    // thickness/alpha required only when sides non-empty (validated in Validate())
    if (!config.sides.empty()) {
        config.thickness = abc["thickness"].as<real_t>();
        config.alpha = abc["alpha"].as<real_t>();
    }
    return config;
}

std::vector<std::string> YamlConfig::GetDirichletAttributes() const {
    YAML::Node section = root_["boundary"]["dirichlet"];
    if (!section || !section.IsSequence()) return {};
    return section.as<std::vector<std::string>>();
}

// =============================================================================
// Sources Section
// =============================================================================

std::string YamlConfig::GetSourceFile() const {
    // Optional - returns empty string if not specified (sentinel value OK)
    YAML::Node src = root_["sources"];
    if (src && src["file"]) {
        return src["file"].as<std::string>();
    }
    return "";
}

int YamlConfig::GetSourceShotId() const {
    YAML::Node src = root_["sources"];
    if (src && src["shot_id"]) {
        return src["shot_id"].as<int>();
    }
    return 0;
}

void YamlConfig::ParseSources() const {
    if (sources_parsed_) return;

    sources_cache_.clear();

    // HDF5 input branch: delegate to HDF5SourceReceiverReader and translate
    // each HDF5SourceEntry back into SourceDef so downstream
    // (LoadSourceConfig{2D,3D}) is unchanged.
    // HDF5 mode is signaled by `sources.file:` (inline list takes the
    // alternative path below).
    const std::string h5_path_top = GetSourceFile();
    if (!h5_path_top.empty()) {
        const int shot_id = GetSourceShotId();
        const int dim = GetDimension();
        HDF5SourceCatalog cat =
            HDF5SourceReceiverReader::ReadSources(h5_path_top, shot_id, dim);
        for (const auto& s : cat.sources) {
            SourceDef def;
            def.id = s.id;
            def.name = s.label.empty()
                ? "source_" + std::to_string(s.id)
                : s.label;
            def.type = s.type;
            for (int d = 0; d < 3; ++d) {
                def.location[d] = (d < static_cast<int>(s.position.size()))
                                      ? s.position[d] : real_t{0};
                def.direction[d] = (d < static_cast<int>(s.direction.size()))
                                       ? s.direction[d] : real_t{0};
            }
            // Moment tensor components in canonical order from the reader.
            // 3D fills M[0..5]; 2D fills M[0]=Mxx, M[1]=Myy, M[3]=Mxy
            // (matches the slot convention used by
            // ConfigLoaders.cpp::LoadSourceConfig2D).
            for (int i = 0; i < 6; ++i) def.M[i] = real_t{0};
            if (s.type == "moment_tensor") {
                if (s.moment_tensor.size() == 6) {
                    for (int i = 0; i < 6; ++i) def.M[i] = s.moment_tensor[i];
                } else if (s.moment_tensor.size() == 3) {
                    // 2D order from reader: {Mxx, Myy, Mxy}
                    def.M[0] = s.moment_tensor[0];
                    def.M[1] = s.moment_tensor[1];
                    def.M[3] = s.moment_tensor[2];
                }
            }
            // STF samples are pre-loaded from HDF5. SourceTimeFunction
            // sees non-empty stf_samples and bypasses analytic synthesis.
            def.wavelet_type = "ricker";  // unused (stf_samples wins)
            def.frequency = real_t{0};
            def.amplitude = real_t{1};
            def.delay = real_t{0};
            def.stf_samples = s.stf;
            sources_cache_.push_back(std::move(def));
        }
        sources_parsed_ = true;
        return;
    }

    // Check if sources come from external file. `ext_root` must outlive
    // `sources_node` — see the note in ParseReceivers() for the lifetime
    // rationale (yaml-cpp child nodes keep a shared reference to the
    // document, and dropping that document mid-iteration manifests as
    // mysterious waveform drift on GPU builds).
    std::string source_file = GetSourceFile();
    YAML::Node ext_root;
    YAML::Node sources_node;

    if (!source_file.empty()) {
        try {
            ext_root = YAML::LoadFile(source_file);
        } catch (const YAML::Exception& e) {
            std::cerr << "Error loading source file: " << e.what() << std::endl;
            sources_parsed_ = true;
            return;
        }
        sources_node = ext_root["sources"];
    } else {
        sources_node = root_["sources"]["list"];
    }

    if (!sources_node || !sources_node.IsSequence()) {
        sources_parsed_ = true;
        return;
    }

    int dim = GetDimension();

    for (size_t i = 0; i < sources_node.size(); i++) {
        YAML::Node src = sources_node[i];
        SourceDef def;

        // id and name are optional
        def.id = src["id"] ? src["id"].as<int>() : static_cast<int>(i + 1);
        def.name = src["name"] ? src["name"].as<std::string>() : "source_" + std::to_string(def.id);
        // type is required - validated in Validate()
        def.type = src["type"].as<std::string>();

        // Location required - validated in Validate()
        YAML::Node loc = src["location"];
        for (int d = 0; d < 3; d++) {
            def.location[d] = (d < static_cast<int>(loc.size())) ? loc[d].as<real_t>() : 0.0;
        }

        // Direction (for force type) - optional, default vertical
        if (src["direction"]) {
            YAML::Node dir = src["direction"];
            for (int d = 0; d < 3; d++) {
                def.direction[d] = (d < static_cast<int>(dir.size())) ? dir[d].as<real_t>() : 0.0;
            }
        } else {
            def.direction[0] = 0.0;
            def.direction[1] = 0.0;
            def.direction[2] = 0.0;
            def.direction[dim - 1] = 1.0;  // Default: vertical
        }

        // Wavelet parameters - required - validated in Validate()
        YAML::Node wv = src["wavelet"];
        def.wavelet_type = wv["type"].as<std::string>();

        if (def.wavelet_type == "external") {
            // External STF file: only `file` is required. `frequency` and
            // `amplitude` are determined by the file content itself; if the
            // user supplies them they're used as a PPW-check hint, otherwise
            // the PPW check is skipped for this source.
            MFEM_VERIFY(wv["file"],
                "Source '" << def.name << "': wavelet.type='external' requires 'file' parameter");
            def.external_file = wv["file"].as<std::string>();
            def.frequency = wv["frequency"] ? wv["frequency"].as<real_t>() : 0.0;
            def.amplitude = 1.0;  // Not used for external
        } else {
            // Ricker or Gaussian - require frequency (PPW check) and amplitude
            def.frequency = wv["frequency"].as<real_t>();
            def.amplitude = wv["amplitude"].as<real_t>();
        }
        // delay is optional (0.0 is valid default)
        def.delay = wv["delay"] ? wv["delay"].as<real_t>() : 0.0;

        // Moment tensor (if applicable) - 0.0 is valid default
        if (def.type == "moment_tensor" && src["moment_tensor"]) {
            YAML::Node mt = src["moment_tensor"];
            def.M[0] = mt["Mxx"] ? mt["Mxx"].as<real_t>() : 0.0;
            def.M[1] = mt["Myy"] ? mt["Myy"].as<real_t>() : 0.0;
            def.M[2] = mt["Mzz"] ? mt["Mzz"].as<real_t>() : 0.0;
            def.M[3] = mt["Mxy"] ? mt["Mxy"].as<real_t>() : 0.0;
            def.M[4] = mt["Mxz"] ? mt["Mxz"].as<real_t>() : 0.0;
            def.M[5] = mt["Myz"] ? mt["Myz"].as<real_t>() : 0.0;
        }

        sources_cache_.push_back(def);
    }

    sources_parsed_ = true;
}

std::vector<SourceDef> YamlConfig::GetAllSources() const {
    ParseSources();
    return sources_cache_;
}

// =============================================================================
// Receivers Section
// =============================================================================

bool YamlConfig::HasReceivers() const {
    return root_["receivers"].IsDefined();
}

void YamlConfig::ParseReceivers() const {
    if (receivers_parsed_) return;

    receivers_cache_.clear();

    // `receivers:` block is optional (forward sims may only want wavefield
    // snapshots). Empty section → empty cache, no receivers configured.
    if (!root_["receivers"]) {
        receivers_parsed_ = true;
        return;
    }

    // Optional: when sources.file is set, receivers may inherit types from
    // the HDF5 per-receiver `@types` attribute (validated downstream).
    std::vector<std::string> default_types;
    YAML::Node type_node = root_["receivers"]["type"];
    if (type_node) {
        if (type_node.IsSequence()) {
            for (size_t i = 0; i < type_node.size(); i++) {
                default_types.push_back(type_node[i].as<std::string>());
            }
        } else {
            default_types.push_back(type_node.as<std::string>());
        }
    }
    int dim = GetDimension();

    // When no inline list/line is given, receivers come from sources.file
    // (the bundled HDF5 input). This is the canonical FWI / driver path.
    YAML::Node rec_root = root_["receivers"];
    const std::string src_file = GetSourceFile();
    if (!rec_root["list"] && !rec_root["line"] && !src_file.empty()) {
        ReceiverConfig::Config rc =
            HDF5SourceReceiverReader::ReadReceivers(src_file,
                                                    GetSourceShotId(),
                                                    default_types, dim);
        for (const auto& r : rc.receivers) {
            ReceiverDef def;
            def.name = r.name;
            for (int d = 0; d < 3; ++d) {
                def.location[d] = (d < static_cast<int>(r.location.size()))
                                      ? r.location[d] : 0.0;
            }
            def.types = r.types;
            def.weight = r.weight;
            receivers_cache_.push_back(std::move(def));
        }
        receivers_parsed_ = true;
        return;
    }

    // Inline path: receiver list / line under receivers: block in this YAML.
    YAML::Node rec_list_node = root_["receivers"]["list"];
    YAML::Node rec_line_node = root_["receivers"]["line"];

    // 2. Parse receivers list
    if (rec_list_node && rec_list_node.IsSequence()) {
        for (size_t i = 0; i < rec_list_node.size(); i++) {
            YAML::Node rec = rec_list_node[i];
            ReceiverDef def;

            // name is optional
            def.name = rec["name"] ? rec["name"].as<std::string>() : "R" + std::to_string(i);
            // location required - validated in Validate() for inline, checked here for external
            YAML::Node loc = rec["location"];
            if (!loc) {
                std::cerr << "Receiver list[" << i << "] missing location" << std::endl;
                continue;
            }
            for (int d = 0; d < 3; d++) {
                def.location[d] = (d < static_cast<int>(loc.size())) ? loc[d].as<real_t>() : 0.0;
            }
            // All receivers inherit the parent-level `receivers.type` list.
            // Coupled fluid-solid simulations rely on the domain-aware
            // filtering inside LoadReceivers (PS routes to fluid submesh,
            // DISP/VEL/ACC to solid) — no per-receiver override needed.
            def.types = default_types;
            // weight is optional (1.0 is valid default for weight)
            def.weight = rec["weight"] ? rec["weight"].as<real_t>() : 1.0;

            receivers_cache_.push_back(def);
        }
    }

    // 3. Line expansion
    if (rec_line_node) {
        YAML::Node start_node = rec_line_node["start"];
        YAML::Node end_node = rec_line_node["end"];
        if (start_node && end_node && rec_line_node["count"]) {
            real_t start[3] = {0.0, 0.0, 0.0};
            real_t end_[3] = {0.0, 0.0, 0.0};
            for (int d = 0; d < 3; d++) {
                start[d] = (d < static_cast<int>(start_node.size())) ? start_node[d].as<real_t>() : 0.0;
                end_[d] = (d < static_cast<int>(end_node.size())) ? end_node[d].as<real_t>() : 0.0;
            }
            const int count = rec_line_node["count"].as<int>();

            for (int i = 0; i < count; i++) {
                ReceiverDef def;
                char buf[32];
                snprintf(buf, sizeof(buf), "R%03d", i + 1);
                def.name = buf;
                def.types = default_types;
                def.weight = 1.0;

                real_t t = (count > 1) ?
                           static_cast<real_t>(i) / (count - 1) : 0.0;
                for (int d = 0; d < dim; d++) {
                    def.location[d] = start[d] + t * (end_[d] - start[d]);
                }
                receivers_cache_.push_back(def);
            }
        }
    }

    receivers_parsed_ = true;
}

std::vector<ReceiverDef> YamlConfig::GetAllReceivers() const {
    ParseReceivers();
    return receivers_cache_;
}

std::vector<std::string> YamlConfig::GetReceiverOutputFormats() const {
    std::vector<std::string> formats;
    YAML::Node fmt_node = root_["receivers"]["output"]["formats"];
    for (const auto& item : fmt_node) {
        formats.push_back(item.as<std::string>());
    }
    return formats;
}

std::string YamlConfig::GetReceiverOutputFilename() const {
    // Optional; default `shot` is used as the bundle name for hdf5/su
    // formats (e.g. shot0001.h5). ASCII derives filenames per-receiver
    // and ignores this key.
    if (root_["receivers"]["output"]["filename"]) {
        return root_["receivers"]["output"]["filename"].as<std::string>();
    }
    return "shot";
}

// =============================================================================
// Device Section (replaces Parallel Section)
// =============================================================================

std::string YamlConfig::GetDevice() const {
    // Required - validated in Validate()
    return root_["device"]["type"].as<std::string>();
}

int YamlConfig::GetReceiverBufferSteps() const {
    // GPU-only knob; 0 = flush all steps at end of run.
    YAML::Node out = root_["receivers"]["output"];
    if (out && out["buffer_steps"]) {
        return out["buffer_steps"].as<int>();
    }
    return 0;
}

// =============================================================================
// Simulation Mode
// =============================================================================

std::string YamlConfig::GetSimulationMode() const {
    YAML::Node sim = root_["simulation"];
    if (sim && sim["mode"]) {
        return sim["mode"].as<std::string>();
    }
    return "forward";
}

// =============================================================================
// Inversion Parameters (simulation direct)
// =============================================================================

std::string YamlConfig::GetMisfitType() const {
    YAML::Node inv = root_["inversion"];
    if (inv && inv["misfit_type"]) {
        return inv["misfit_type"].as<std::string>();
    }
    return "l2_waveform";
}

int YamlConfig::GetNumCheckpoints() const {
    YAML::Node inv = root_["inversion"];
    if (inv && inv["checkpoints"] && inv["checkpoints"]["n"]) {
        return inv["checkpoints"]["n"].as<int>();
    }
    return 10;
}

std::string YamlConfig::GetCheckpointStorage() const {
    YAML::Node inv = root_["inversion"];
    if (inv && inv["checkpoints"] && inv["checkpoints"]["storage"]) {
        return inv["checkpoints"]["storage"].as<std::string>();
    }
    return "memory";
}

std::string YamlConfig::GetCheckpointDevice() const {
    YAML::Node inv = root_["inversion"];
    if (inv && inv["checkpoints"] && inv["checkpoints"]["device"]) {
        return inv["checkpoints"]["device"].as<std::string>();
    }
    return "auto";
}

std::string YamlConfig::GetCheckpointDir() const {
    // Explicit override; default = <simulation.directory>/checkpoints
    // (mirrors kernel_dir derivation).
    YAML::Node inv = root_["inversion"];
    if (inv && inv["checkpoints"] && inv["checkpoints"]["dir"]) {
        return inv["checkpoints"]["dir"].as<std::string>();
    }
    return GetOutputDirectory() + "/checkpoints";
}

std::string YamlConfig::GetKernelOutputDir() const {
    // Explicit override allowed; default = <simulation.directory>/kernels.
    YAML::Node inv = root_["inversion"];
    if (inv && inv["kernel_dir"]) {
        return inv["kernel_dir"].as<std::string>();
    }
    return GetOutputDirectory() + "/kernels";
}

std::string YamlConfig::GetSensitivityBackend() const {
    YAML::Node inv = root_["inversion"];
    if (inv && inv["sensitivity_backend"]) {
        std::string b = inv["sensitivity_backend"].as<std::string>();
        if (b != "hand" && b != "ad") {
            MFEM_ABORT("Invalid inversion.sensitivity_backend: '" << b
                       << "' (valid: hand, ad)");
        }
        return b;
    }
    return "hand";  // default to production hand path
}

}  // namespace SEM
