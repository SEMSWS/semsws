// HDF5SourceReceiverReader — forward-input source/receiver reader.
// Uses the HDF5 C API directly (matches HDF5ObservedReader.cpp's reasoning
// about the runtime header / library version mismatch).

#include "srcrecv/HDF5SourceReceiverReader.hpp"
#include "srcrecv/HDF5IOSchema.hpp"
#include "common/Types.hpp"

#include <hdf5.h>
#include <mfem.hpp>
#include <cstring>
#include <string>
#include <vector>

namespace SEM {

namespace {

// -----------------------------------------------------------------------------
// Local HDF5 RAII helpers. Mirrors HDF5ObservedReader.cpp; intentionally
// duplicated until a shared HDF5IOUtils header is extracted.
// -----------------------------------------------------------------------------

struct H5Id {
    hid_t id = -1;
    enum Kind { File, Group, Dataset, Attr, Space, Type, Plist, None } kind = None;
    H5Id() = default;
    H5Id(hid_t i, Kind k) : id(i), kind(k) {}
    H5Id(const H5Id&) = delete;
    H5Id& operator=(const H5Id&) = delete;
    H5Id(H5Id&& o) noexcept : id(o.id), kind(o.kind) { o.id = -1; o.kind = None; }
    H5Id& operator=(H5Id&& o) noexcept {
        reset();
        id = o.id; kind = o.kind; o.id = -1; o.kind = None;
        return *this;
    }
    ~H5Id() { reset(); }
    void reset() {
        if (id < 0) return;
        switch (kind) {
            case File:    H5Fclose(id); break;
            case Group:   H5Gclose(id); break;
            case Dataset: H5Dclose(id); break;
            case Attr:    H5Aclose(id); break;
            case Space:   H5Sclose(id); break;
            case Type:    H5Tclose(id); break;
            case Plist:   H5Pclose(id); break;
            case None: break;
        }
        id = -1; kind = None;
    }
    operator hid_t() const { return id; }
};

struct ErrStackGuard {
    H5E_auto2_t old_func = nullptr;
    void* old_data = nullptr;
    ErrStackGuard() {
        H5Eget_auto2(H5E_DEFAULT, &old_func, &old_data);
        H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    }
    ~ErrStackGuard() {
        H5Eset_auto2(H5E_DEFAULT, old_func, old_data);
    }
};

[[noreturn]] void Abort(const std::string& msg) {
    MFEM_ABORT(msg);
}

// Forward decls — definitions further down so the source-side helpers can
// call AttrExists / LinkExists / ReadStringAttr / TryReadStringListAttr
// without reordering.
bool LinkExists(hid_t loc, const char* name);
bool AttrExists(hid_t loc, const char* name);
std::string ReadStringAttr(hid_t loc, const char* name,
                           const std::string& ctx);
bool TryReadStringListAttr(hid_t loc, const char* name,
                           std::vector<std::string>& out,
                           const std::string& ctx);

/// SEMSWS canonical receiver-type strings (PS / VEL / DISP / ACC) are
/// shared across YAML and HDF5, so the per-receiver `@types` attribute is
/// validated as-is. Helper kept for symmetry with future translations
/// (e.g. if the spec ever introduces fullword aliases).
const std::string& ValidateTypeAttr(const std::string& s,
                                    const std::string& ctx) {
    if (s == HDF5Schema::kChannelPressure ||
        s == HDF5Schema::kChannelVelocity ||
        s == HDF5Schema::kChannelDisplacement ||
        s == HDF5Schema::kChannelAcceleration) {
        return s;
    }
    Abort("HDF5 source/receiver: unknown type '" + s + "' on " + ctx +
          " (valid: PS, VEL, DISP, ACC)");
}

void ReadDirectionAttr(hid_t g, int space_dim, std::vector<real_t>& out,
                       const std::string& ctx) {
    if (!AttrExists(g, HDF5Schema::kAttrDirection)) {
        Abort("HDF5 source/receiver: missing 'direction' attr on " + ctx +
              " (required for type=force)");
    }
    H5Id a(H5Aopen(g, HDF5Schema::kAttrDirection, H5P_DEFAULT), H5Id::Attr);
    H5Id sp(H5Aget_space(a), H5Id::Space);
    int ndims = H5Sget_simple_extent_ndims(sp);
    hsize_t dims[2] = {0, 0};
    if (ndims >= 1) H5Sget_simple_extent_dims(sp, dims, nullptr);
    hsize_t n = (ndims >= 1) ? dims[0] : 1;
    if (static_cast<int>(n) != space_dim) {
        Abort("HDF5 source/receiver: 'direction' on " + ctx +
              " has size " + std::to_string(n) +
              ", expected " + std::to_string(space_dim));
    }
    std::vector<double> buf(n);
    if (H5Aread(a, H5T_NATIVE_DOUBLE, buf.data()) < 0) {
        Abort("HDF5 source/receiver: failed reading 'direction' on " + ctx);
    }
    out.assign(space_dim, real_t{0});
    for (int i = 0; i < space_dim; ++i) {
        out[i] = static_cast<real_t>(buf[i]);
    }
}

/// Read `/stf` from a source group. Enforces F2 (no @layout), F4 (length
/// == n_samples), F5 (dtype f64). Always 1-D.
void ReadScalarStf(hid_t g, int n_samples, std::vector<real_t>& out,
                   const std::string& ctx) {
    if (!LinkExists(g, HDF5Schema::kDatasetStf)) {
        Abort("HDF5 source/receiver: missing '" +
              std::string(HDF5Schema::kDatasetStf) + "' dataset on " + ctx);
    }
    H5Id ds(H5Dopen2(g, HDF5Schema::kDatasetStf, H5P_DEFAULT), H5Id::Dataset);
    if (ds < 0) {
        Abort("HDF5 source/receiver: cannot open " + ctx + "/" +
              HDF5Schema::kDatasetStf);
    }
    // F2: no @layout attribute on /stf in v2.0.
    if (AttrExists(ds, "layout")) {
        Abort("HDF5 source/receiver: '/stf/@layout' attribute is forbidden "
              "in v2.0 (F2: scalar layout only). Found on " + ctx);
    }
    H5Id sp(H5Dget_space(ds), H5Id::Space);
    int ndims = H5Sget_simple_extent_ndims(sp);
    hsize_t dims[2] = {0, 0};
    if (ndims >= 1) H5Sget_simple_extent_dims(sp, dims, nullptr);
    if (ndims != 1 ||
        static_cast<int>(dims[0]) != n_samples) {
        Abort("HDF5 source/receiver: '/stf' on " + ctx +
              " has shape mismatch (expected (" +
              std::to_string(n_samples) + ",) f64, got ndims=" +
              std::to_string(ndims) + ", n=" +
              std::to_string(ndims >= 1 ? dims[0] : 0) + ")");
    }
    H5Id dtype(H5Dget_type(ds), H5Id::Type);
    if (H5Tget_class(dtype) != H5T_FLOAT || H5Tget_size(dtype) != 8) {
        Abort("HDF5 source/receiver: '/stf' on " + ctx +
              " must be f64 (F5)");
    }
    std::vector<double> buf(static_cast<size_t>(n_samples));
    if (H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                buf.data()) < 0) {
        Abort("HDF5 source/receiver: failed reading '/stf' on " + ctx);
    }
    out.assign(n_samples, real_t{0});
    for (int i = 0; i < n_samples; ++i) {
        out[i] = static_cast<real_t>(buf[i]);
    }
}

/// Read /moment_tensor and remap to canonical SEMSWS order:
///   3D → [Mxx, Myy, Mzz, Mxy, Mxz, Myz]
///   2D → [Mxx, Myy, Mxy]
/// The file's `@component_order` may list the components in any order;
/// reader respects it (important for Harvard CMT / USGS-style writers).
void ReadMomentTensor(hid_t g, int space_dim,
                      std::vector<real_t>& out_canonical,
                      const std::string& ctx) {
    if (!LinkExists(g, HDF5Schema::kDatasetMomentTensor)) {
        Abort("HDF5 source/receiver: missing '" +
              std::string(HDF5Schema::kDatasetMomentTensor) +
              "' dataset on " + ctx + " (required for type=moment_tensor)");
    }
    H5Id ds(H5Dopen2(g, HDF5Schema::kDatasetMomentTensor, H5P_DEFAULT),
            H5Id::Dataset);
    if (ds < 0) {
        Abort("HDF5 source/receiver: cannot open " + ctx + "/" +
              HDF5Schema::kDatasetMomentTensor);
    }

    // Shape: (6,) for 3D, (3,) for 2D (R5 / 2D MT spec, 3 components only).
    H5Id sp(H5Dget_space(ds), H5Id::Space);
    int ndims = H5Sget_simple_extent_ndims(sp);
    hsize_t dims[2] = {0, 0};
    if (ndims >= 1) H5Sget_simple_extent_dims(sp, dims, nullptr);
    const int expected_n = (space_dim == 3) ? 6 : 3;
    if (ndims != 1 || static_cast<int>(dims[0]) != expected_n) {
        Abort("HDF5 source/receiver: '/moment_tensor' on " + ctx +
              " expected shape (" + std::to_string(expected_n) +
              ",) f64, got ndims=" + std::to_string(ndims) + ", n=" +
              std::to_string(ndims >= 1 ? dims[0] : 0));
    }
    H5Id dtype(H5Dget_type(ds), H5Id::Type);
    if (H5Tget_class(dtype) != H5T_FLOAT || H5Tget_size(dtype) != 8) {
        Abort("HDF5 source/receiver: '/moment_tensor' on " + ctx +
              " must be f64");
    }

    // @component_order is required.
    std::vector<std::string> order;
    if (!TryReadStringListAttr(ds, HDF5Schema::kAttrComponentOrder,
                               order, ctx + "/" +
                               HDF5Schema::kDatasetMomentTensor)) {
        Abort("HDF5 source/receiver: '/moment_tensor' on " + ctx +
              " is missing required '@component_order' attribute");
    }
    if (static_cast<int>(order.size()) != expected_n) {
        Abort("HDF5 source/receiver: '@component_order' size " +
              std::to_string(order.size()) +
              " on " + ctx + " does not match dataset size " +
              std::to_string(expected_n));
    }

    // @coord_system: only "xyz" supported in v2.0 (default if absent).
    if (AttrExists(ds, HDF5Schema::kAttrCoordSystem)) {
        const std::string coord = ReadStringAttr(
            ds, HDF5Schema::kAttrCoordSystem, ctx);
        if (coord != "xyz") {
            Abort("HDF5 source/receiver: '/moment_tensor/@coord_system'='"
                  + coord + "' on " + ctx +
                  " — only 'xyz' is supported in v2.0 (RTP / spherical "
                  "is reserved for v2.1+).");
        }
    }

    // Canonical SEMSWS order. Linear search is fine for n ≤ 6.
    static const char* kCanon3D[6] =
        {"Mxx", "Myy", "Mzz", "Mxy", "Mxz", "Myz"};
    static const char* kCanon2D[3] = {"Mxx", "Myy", "Mxy"};
    const char** canon = (space_dim == 3)
        ? kCanon3D : kCanon2D;
    auto canon_index = [&](const std::string& s) -> int {
        for (int i = 0; i < expected_n; ++i) {
            if (s == canon[i]) return i;
        }
        return -1;
    };

    std::vector<int> file_to_canon(expected_n, -1);
    std::vector<bool> seen(expected_n, false);
    for (int i = 0; i < expected_n; ++i) {
        const int idx = canon_index(order[i]);
        if (idx < 0) {
            std::string valid = "{";
            for (int j = 0; j < expected_n; ++j) {
                if (j > 0) valid += ", ";
                valid += canon[j];
            }
            valid += "}";
            Abort("HDF5 source/receiver: '@component_order[" +
                  std::to_string(i) + "]'='" + order[i] + "' on " + ctx +
                  " is not a valid name; valid: " + valid);
        }
        if (seen[idx]) {
            Abort("HDF5 source/receiver: '@component_order' on " + ctx +
                  " has duplicate '" + order[i] + "'");
        }
        seen[idx] = true;
        file_to_canon[i] = idx;
    }

    // Read raw values and remap.
    std::vector<double> raw(static_cast<size_t>(expected_n));
    if (H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                raw.data()) < 0) {
        Abort("HDF5 source/receiver: failed reading '/moment_tensor' on "
              + ctx);
    }
    out_canonical.assign(static_cast<size_t>(expected_n), real_t{0});
    for (int i = 0; i < expected_n; ++i) {
        out_canonical[file_to_canon[i]] = static_cast<real_t>(raw[i]);
    }
}

void ForbidPerSourceTimeAttrs(hid_t g, const std::string& ctx) {
    // F7: per-source @t0 / @hdur are not allowed in v2.0. Time shifts must
    // be expressed by zero-padding the head of /stf.
    if (AttrExists(g, "t0")) {
        Abort("HDF5 source/receiver: per-source '@t0' is forbidden (F7). "
              "Express time shifts via leading zero samples in /stf. "
              "Offending source: " + ctx);
    }
    if (AttrExists(g, "hdur")) {
        Abort("HDF5 source/receiver: per-source '@hdur' is forbidden (F7) "
              "on " + ctx);
    }
}

bool LinkExists(hid_t loc, const char* name) {
    return H5Lexists(loc, name, H5P_DEFAULT) > 0;
}
bool AttrExists(hid_t loc, const char* name) {
    return H5Aexists(loc, name) > 0;
}

std::string ReadStringAttr(hid_t loc, const char* name,
                           const std::string& ctx) {
    if (!AttrExists(loc, name)) {
        Abort("HDF5 source/receiver: missing required attribute '"
              + std::string(name) + "' on " + ctx);
    }
    H5Id attr(H5Aopen(loc, name, H5P_DEFAULT), H5Id::Attr);
    H5Id t(H5Aget_type(attr), H5Id::Type);
    if (H5Tget_class(t) != H5T_STRING) {
        Abort("HDF5 source/receiver: attr '" + std::string(name) + "' on "
              + ctx + " is not a string");
    }
    if (H5Tis_variable_str(t) > 0) {
        char* buf = nullptr;
        H5Id mtype(H5Tcopy(H5T_C_S1), H5Id::Type);
        H5Tset_size(mtype, H5T_VARIABLE);
        H5Tset_cset(mtype, H5T_CSET_UTF8);
        if (H5Aread(attr, mtype, &buf) < 0 || buf == nullptr) {
            Abort("HDF5 source/receiver: failed reading string attr '"
                  + std::string(name) + "' on " + ctx);
        }
        std::string out(buf);
        H5Dvlen_reclaim(mtype, H5Aget_space(attr), H5P_DEFAULT, &buf);
        return out;
    } else {
        size_t sz = H5Tget_size(t);
        std::string out(sz, '\0');
        if (H5Aread(attr, t, out.data()) < 0) {
            Abort("HDF5 source/receiver: failed reading string attr '"
                  + std::string(name) + "' on " + ctx);
        }
        while (!out.empty() && out.back() == '\0') out.pop_back();
        return out;
    }
}

double ReadDoubleAttr(hid_t loc, const char* name, const std::string& ctx) {
    if (!AttrExists(loc, name)) {
        Abort("HDF5 source/receiver: missing required attribute '"
              + std::string(name) + "' on " + ctx);
    }
    H5Id a(H5Aopen(loc, name, H5P_DEFAULT), H5Id::Attr);
    double v = 0.0;
    if (H5Aread(a, H5T_NATIVE_DOUBLE, &v) < 0) {
        Abort("HDF5 source/receiver: failed reading double attr '"
              + std::string(name) + "' on " + ctx);
    }
    return v;
}

int ReadInt32Attr(hid_t loc, const char* name, const std::string& ctx) {
    if (!AttrExists(loc, name)) {
        Abort("HDF5 source/receiver: missing required attribute '"
              + std::string(name) + "' on " + ctx);
    }
    H5Id a(H5Aopen(loc, name, H5P_DEFAULT), H5Id::Attr);
    int v = 0;
    if (H5Aread(a, H5T_NATIVE_INT32, &v) < 0) {
        Abort("HDF5 source/receiver: failed reading int32 attr '"
              + std::string(name) + "' on " + ctx);
    }
    return v;
}

int64_t ReadInt64Attr(hid_t loc, const char* name, const std::string& ctx) {
    if (!AttrExists(loc, name)) {
        Abort("HDF5 source/receiver: missing required attribute '"
              + std::string(name) + "' on " + ctx);
    }
    H5Id a(H5Aopen(loc, name, H5P_DEFAULT), H5Id::Attr);
    int64_t v = 0;
    if (H5Aread(a, H5T_NATIVE_INT64, &v) < 0) {
        Abort("HDF5 source/receiver: failed reading int64 attr '"
              + std::string(name) + "' on " + ctx);
    }
    return v;
}

void ReadPositionAttr(hid_t g, int space_dim, std::vector<real_t>& out,
                      const std::string& ctx) {
    if (!AttrExists(g, HDF5Schema::kAttrPosition)) {
        Abort("HDF5 source/receiver: missing 'position' attr on " + ctx);
    }
    H5Id a(H5Aopen(g, HDF5Schema::kAttrPosition, H5P_DEFAULT), H5Id::Attr);
    H5Id sp(H5Aget_space(a), H5Id::Space);
    int ndims = H5Sget_simple_extent_ndims(sp);
    hsize_t dims[2] = {0, 0};
    if (ndims >= 1) H5Sget_simple_extent_dims(sp, dims, nullptr);
    hsize_t n = (ndims >= 1) ? dims[0] : 1;
    if (static_cast<int>(n) != space_dim) {
        Abort("HDF5 source/receiver: 'position' on " + ctx +
              " has size " + std::to_string(n) +
              ", expected " + std::to_string(space_dim));
    }
    std::vector<double> buf(n);
    if (H5Aread(a, H5T_NATIVE_DOUBLE, buf.data()) < 0) {
        Abort("HDF5 source/receiver: failed reading 'position' on " + ctx);
    }
    out.assign(space_dim, real_t{0});
    for (int i = 0; i < space_dim; ++i) {
        out[i] = static_cast<real_t>(buf[i]);
    }
}

/// Reads an optional variable-length string-array attribute. Returns true
/// when present (and fills `out`); false when absent.
bool TryReadStringListAttr(hid_t loc, const char* name,
                           std::vector<std::string>& out,
                           const std::string& ctx) {
    if (!AttrExists(loc, name)) return false;
    H5Id a(H5Aopen(loc, name, H5P_DEFAULT), H5Id::Attr);
    H5Id t(H5Aget_type(a), H5Id::Type);
    if (H5Tget_class(t) != H5T_STRING) {
        Abort("HDF5 source/receiver: attr '" + std::string(name) + "' on "
              + ctx + " is not a string");
    }
    H5Id sp(H5Aget_space(a), H5Id::Space);
    int ndims = H5Sget_simple_extent_ndims(sp);
    if (ndims != 1) {
        Abort("HDF5 source/receiver: attr '" + std::string(name) + "' on "
              + ctx + " expected 1-D string array, got ndims="
              + std::to_string(ndims));
    }
    hsize_t dims[1] = {0};
    H5Sget_simple_extent_dims(sp, dims, nullptr);
    const size_t n = static_cast<size_t>(dims[0]);
    if (H5Tis_variable_str(t) > 0) {
        std::vector<char*> bufs(n, nullptr);
        H5Id mtype(H5Tcopy(H5T_C_S1), H5Id::Type);
        H5Tset_size(mtype, H5T_VARIABLE);
        H5Tset_cset(mtype, H5T_CSET_UTF8);
        if (H5Aread(a, mtype, bufs.data()) < 0) {
            Abort("HDF5 source/receiver: failed reading string-list attr '"
                  + std::string(name) + "' on " + ctx);
        }
        out.clear();
        out.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            out.emplace_back(bufs[i] ? bufs[i] : "");
        }
        H5Dvlen_reclaim(mtype, sp, H5P_DEFAULT, bufs.data());
    } else {
        size_t sz = H5Tget_size(t);
        std::vector<char> raw(n * sz, '\0');
        if (H5Aread(a, t, raw.data()) < 0) {
            Abort("HDF5 source/receiver: failed reading fixed-string-list "
                  "attr '" + std::string(name) + "' on " + ctx);
        }
        out.clear();
        out.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            std::string s(&raw[i * sz], sz);
            while (!s.empty() && s.back() == '\0') s.pop_back();
            out.emplace_back(std::move(s));
        }
    }
    return true;
}

}  // namespace

// -----------------------------------------------------------------------------

ReceiverConfig::Config
HDF5SourceReceiverReader::ReadReceivers(
    const std::string& path,
    int shot_id,
    const std::vector<std::string>& default_types,
    int expected_space_dim) {

    ReceiverConfig::Config cfg;
    ErrStackGuard eg;

    H5Id file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Id::File);
    if (file < 0) {
        Abort("HDF5 source/receiver: cannot open file " + path);
    }

    // ---- Root attrs ----
    const std::string fv = ReadStringAttr(file, HDF5Schema::kAttrFormatVersion,
                                          path);
    const std::string major = fv.substr(0, fv.find('.'));
    if (major != HDF5Schema::kFormatVersionMajor) {
        Abort("HDF5 source/receiver: unsupported format_version '" + fv +
              "' in " + path + " (expected major " +
              HDF5Schema::kFormatVersionMajor +
              "). Run driver/tools/migrate_observed_v1_to_v2.py to upgrade.");
    }
    const int space_dim = ReadInt32Attr(file, HDF5Schema::kAttrSpaceDim, path);
    if (space_dim != expected_space_dim) {
        Abort("HDF5 source/receiver: " + path +
              " has space_dim=" + std::to_string(space_dim) +
              ", simulation Dim=" + std::to_string(expected_space_dim));
    }
    // dt / t0 / n_samples are required by the schema even for geometry-only
    // input — read them so a malformed file is rejected eagerly.
    (void)ReadDoubleAttr(file, HDF5Schema::kAttrDt, path);
    (void)ReadDoubleAttr(file, HDF5Schema::kAttrT0, path);
    (void)ReadInt64Attr(file, HDF5Schema::kAttrNSamples, path);

    // ---- /shots/<NNNN>/receivers/ ----
    const std::string shot_path = HDF5Schema::ShotGroupPath(shot_id);
    if (!LinkExists(file, HDF5Schema::kGroupShots)) {
        Abort("HDF5 source/receiver: file " + path + " has no /" +
              HDF5Schema::kGroupShots + " group (v2.0 schema required)");
    }
    if (!LinkExists(file, shot_path.c_str())) {
        Abort("HDF5 source/receiver: file " + path + " has no /" + shot_path +
              " group (shot_id=" + std::to_string(shot_id) + ")");
    }
    H5Id gshot(H5Gopen2(file, shot_path.c_str(), H5P_DEFAULT), H5Id::Group);
    if (gshot < 0) {
        Abort("HDF5 source/receiver: cannot open /" + shot_path + " in " + path);
    }
    if (!LinkExists(gshot, HDF5Schema::kGroupReceivers)) {
        Abort("HDF5 source/receiver: " + path + ":/" + shot_path +
              " has no '" + HDF5Schema::kGroupReceivers + "' group");
    }
    H5Id grecv(H5Gopen2(gshot, HDF5Schema::kGroupReceivers, H5P_DEFAULT),
               H5Id::Group);
    if (grecv < 0) {
        Abort("HDF5 source/receiver: cannot open /" + shot_path + "/" +
              HDF5Schema::kGroupReceivers + " in " + path);
    }

    // Per-receiver default types must be valid (validate eagerly so the
    // failure surfaces at config load, not at receiver creation time).
    auto validate_types = [](const std::vector<std::string>& types,
                             const std::string& ctx) {
        for (const auto& t : types) {
            (void)StringToReceiverType(t);  // aborts on unknown
            // domain-compat (Solid vs Fluid) is checked in
            // ReceiverArray::AddReceiver, where we know the simulation
            // domain. Don't duplicate that here.
            (void)ctx;  // currently unused; reserved for richer messages
        }
    };
    validate_types(default_types, path);

    // ---- Iterate group children in name order ----
    H5G_info_t ginfo{};
    H5Gget_info(grecv, &ginfo);
    cfg.receivers.reserve(static_cast<size_t>(ginfo.nlinks));

    for (hsize_t i = 0; i < ginfo.nlinks; ++i) {
        char namebuf[256];
        ssize_t len = H5Lget_name_by_idx(grecv, ".", H5_INDEX_NAME, H5_ITER_INC,
                                         i, namebuf, sizeof(namebuf),
                                         H5P_DEFAULT);
        if (len <= 0) continue;
        const std::string key(namebuf, static_cast<size_t>(len));
        const std::string ctx = path + ":/" + shot_path + "/" +
                                HDF5Schema::kGroupReceivers + "/" + key;

        H5Id g(H5Gopen2(grecv, key.c_str(), H5P_DEFAULT), H5Id::Group);
        if (g < 0) Abort("HDF5 source/receiver: cannot open " + ctx);

        ReceiverConfig::SingleReceiver rec;
        // Prefer @label as the user-facing name when present; fall back to
        // the group key when the writer didn't emit @label.
        if (AttrExists(g, HDF5Schema::kAttrLabel)) {
            rec.name = ReadStringAttr(g, HDF5Schema::kAttrLabel, ctx);
            if (rec.name.empty()) rec.name = key;
        } else {
            rec.name = key;
        }
        ReadPositionAttr(g, space_dim, rec.location, ctx);

        std::vector<std::string> per_recv_types_raw;
        if (TryReadStringListAttr(g, "types", per_recv_types_raw, ctx)) {
            if (per_recv_types_raw.empty()) {
                Abort("HDF5 source/receiver: '@types' on " + ctx +
                      " is an empty list (omit the attribute to inherit "
                      "the parent-level receivers.type)");
            }
            for (const auto& t : per_recv_types_raw) {
                (void)ValidateTypeAttr(t, ctx);
            }
            validate_types(per_recv_types_raw, ctx);  // domain-side validator
            rec.types = std::move(per_recv_types_raw);
        } else {
            if (default_types.empty()) {
                Abort("HDF5 source/receiver: " + ctx + " has no `@types` "
                      "attribute and YAML provides no `receivers.type` "
                      "fallback — set one or the other");
            }
            rec.types = default_types;
        }
        rec.weight = real_t{1.0};

        cfg.receivers.push_back(std::move(rec));
    }

    if (cfg.receivers.empty()) {
        Abort("HDF5 source/receiver: file " + path + " /" + shot_path +
              "/" + HDF5Schema::kGroupReceivers +
              " contains no receivers");
    }
    return cfg;
}

// -----------------------------------------------------------------------------

HDF5SourceCatalog
HDF5SourceReceiverReader::ReadSources(
    const std::string& path,
    int shot_id,
    int expected_space_dim) {

    HDF5SourceCatalog out;
    ErrStackGuard eg;

    H5Id file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Id::File);
    if (file < 0) {
        Abort("HDF5 source/receiver: cannot open file " + path);
    }

    // ---- Root attrs (same prologue as ReadReceivers) ----
    const std::string fv = ReadStringAttr(file, HDF5Schema::kAttrFormatVersion,
                                          path);
    const std::string major = fv.substr(0, fv.find('.'));
    if (major != HDF5Schema::kFormatVersionMajor) {
        Abort("HDF5 source/receiver: unsupported format_version '" + fv +
              "' in " + path + " (expected major " +
              HDF5Schema::kFormatVersionMajor +
              "). Run driver/tools/migrate_observed_v1_to_v2.py to upgrade.");
    }
    out.space_dim = ReadInt32Attr(file, HDF5Schema::kAttrSpaceDim, path);
    if (out.space_dim != expected_space_dim) {
        Abort("HDF5 source/receiver: " + path +
              " has space_dim=" + std::to_string(out.space_dim) +
              ", simulation Dim=" + std::to_string(expected_space_dim));
    }
    out.dt = static_cast<real_t>(
        ReadDoubleAttr(file, HDF5Schema::kAttrDt, path));
    out.t0 = static_cast<real_t>(
        ReadDoubleAttr(file, HDF5Schema::kAttrT0, path));
    out.n_samples = static_cast<int>(
        ReadInt64Attr(file, HDF5Schema::kAttrNSamples, path));
    if (out.n_samples <= 0) {
        Abort("HDF5 source/receiver: " + path + " has n_samples=" +
              std::to_string(out.n_samples));
    }

    // ---- /shots/<NNNN>/sources/ ----
    const std::string shot_path = HDF5Schema::ShotGroupPath(shot_id);
    if (!LinkExists(file, HDF5Schema::kGroupShots)) {
        Abort("HDF5 source/receiver: file " + path + " has no /" +
              HDF5Schema::kGroupShots + " group (v2.0 schema required)");
    }
    if (!LinkExists(file, shot_path.c_str())) {
        Abort("HDF5 source/receiver: file " + path + " has no /" +
              shot_path + " group (shot_id=" + std::to_string(shot_id) + ")");
    }
    H5Id gshot(H5Gopen2(file, shot_path.c_str(), H5P_DEFAULT), H5Id::Group);
    if (gshot < 0) {
        Abort("HDF5 source/receiver: cannot open /" + shot_path + " in " + path);
    }
    if (!LinkExists(gshot, HDF5Schema::kGroupSources)) {
        Abort("HDF5 source/receiver: " + path + ":/" + shot_path +
              " has no '" + HDF5Schema::kGroupSources + "' group");
    }
    H5Id gsrc(H5Gopen2(gshot, HDF5Schema::kGroupSources, H5P_DEFAULT),
              H5Id::Group);
    if (gsrc < 0) {
        Abort("HDF5 source/receiver: cannot open /" + shot_path + "/" +
              HDF5Schema::kGroupSources + " in " + path);
    }

    // ---- Iterate group children in name order ----
    H5G_info_t ginfo{};
    H5Gget_info(gsrc, &ginfo);
    out.sources.reserve(static_cast<size_t>(ginfo.nlinks));

    // For F9 (per-shot @id uniqueness).
    std::vector<int> seen_ids;

    for (hsize_t i = 0; i < ginfo.nlinks; ++i) {
        char namebuf[256];
        ssize_t len = H5Lget_name_by_idx(gsrc, ".", H5_INDEX_NAME, H5_ITER_INC,
                                         i, namebuf, sizeof(namebuf),
                                         H5P_DEFAULT);
        if (len <= 0) continue;
        const std::string key(namebuf, static_cast<size_t>(len));
        const std::string ctx = path + ":/" + shot_path + "/" +
                                HDF5Schema::kGroupSources + "/" + key;

        H5Id g(H5Gopen2(gsrc, key.c_str(), H5P_DEFAULT), H5Id::Group);
        if (g < 0) Abort("HDF5 source/receiver: cannot open " + ctx);

        HDF5SourceEntry entry;
        // F9: @id must be a positive int and unique within the shot.
        const int id = ReadInt32Attr(g, HDF5Schema::kAttrId, ctx);
        if (id <= 0) {
            Abort("HDF5 source/receiver: source '@id' must be a positive "
                  "integer (got " + std::to_string(id) + ") on " + ctx);
        }
        for (int prev : seen_ids) {
            if (prev == id) {
                Abort("HDF5 source/receiver: duplicate '@id'=" +
                      std::to_string(id) + " within shot on " + ctx);
            }
        }
        seen_ids.push_back(id);
        entry.id = id;

        // F8: group key should match S{id:04d}. Warn-via-error rather than
        // silently accepting random keys, so writers stay aligned.
        if (key != HDF5Schema::SourceKey(id)) {
            Abort("HDF5 source/receiver: source group key '" + key +
                  "' on " + ctx + " does not match expected '" +
                  HDF5Schema::SourceKey(id) + "' (F8)");
        }

        if (AttrExists(g, HDF5Schema::kAttrLabel)) {
            entry.label = ReadStringAttr(g, HDF5Schema::kAttrLabel, ctx);
        }

        // @type
        const std::string type = ReadStringAttr(g, HDF5Schema::kAttrType, ctx);
        if (type != HDF5Schema::kSourceTypeForce &&
            type != HDF5Schema::kSourceTypePressure &&
            type != HDF5Schema::kSourceTypeMomentTensor) {
            Abort("HDF5 source/receiver: unknown source @type='" + type +
                  "' on " + ctx +
                  " (valid: 'force', 'pressure', 'moment_tensor')");
        }
        entry.type = type;

        // F10: position size == space_dim.
        ReadPositionAttr(g, out.space_dim, entry.position, ctx);

        // F7: forbid per-source @t0 / @hdur.
        ForbidPerSourceTimeAttrs(g, ctx);

        // direction is required for force; ignored (but accepted) for
        // pressure and moment_tensor — store zero sentinel.
        if (type == HDF5Schema::kSourceTypeForce) {
            ReadDirectionAttr(g, out.space_dim, entry.direction, ctx);
        } else {
            entry.direction.assign(out.space_dim, real_t{0});
        }

        // moment_tensor dataset is required for type=moment_tensor; rejected
        // (well, just absent) for the other types.
        if (type == HDF5Schema::kSourceTypeMomentTensor) {
            ReadMomentTensor(g, out.space_dim, entry.moment_tensor, ctx);
        }

        // /stf (scalar, F2 / F4 / F5 enforced).
        ReadScalarStf(g, out.n_samples, entry.stf, ctx);

        out.sources.push_back(std::move(entry));
    }

    // F11: at least one source per shot.
    if (out.sources.empty()) {
        Abort("HDF5 source/receiver: file " + path + " /" + shot_path +
              "/" + HDF5Schema::kGroupSources + " contains no sources (F11)");
    }
    return out;
}

}  // namespace SEM
