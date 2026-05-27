/**
 * @file AttributeMask.hpp
 * @brief Element-attribute-based DOF masking helpers shared by FWI tools
 *
 * Reused by mask, update_model, lbfgs_direction, adam_direction, clip_field
 * to honor `gradient.active_attributes` (CLI: `--restrict-attr ID1,ID2,...`).
 *
 * Convention: a DOF is "kept" if ANY containing element has its attribute in
 * the keep set (interface DOFs between attr-1 and attr-2 elements are kept on
 * both sides). DOFs touching only non-keep elements are zeroed by callers.
 */
#pragma once

#include <mfem.hpp>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace SEM {

/// Parse "ID1,ID2,..." comma-separated list into vector<int>.
inline std::vector<int> ParseRestrictAttrList(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) out.push_back(std::stoi(token));
    }
    return out;
}

/// Build a per-DOF kept mask (size = fes.GetVSize()): 1 = keep, 0 = zero.
/// Empty keep_attrs => all kept.
inline std::vector<unsigned char> BuildKeptDofMask(
    mfem::ParFiniteElementSpace& fes,
    mfem::ParMesh& pmesh,
    const std::vector<int>& keep_attrs)
{
    std::vector<unsigned char> kept(fes.GetVSize(), 0);
    if (keep_attrs.empty()) {
        std::fill(kept.begin(), kept.end(), 1);
        return kept;
    }
    std::set<int> keep_set(keep_attrs.begin(), keep_attrs.end());
    for (int ie = 0; ie < fes.GetNE(); ie++) {
        int attr = pmesh.GetAttribute(ie);
        if (!keep_set.count(attr)) continue;
        mfem::Array<int> dofs;
        fes.GetElementDofs(ie, dofs);
        for (int j = 0; j < dofs.Size(); j++) {
            int d = dofs[j];
            if (d < 0) d = -1 - d;
            kept[d] = 1;
        }
    }
    return kept;
}

/// Zero entries of `gf` where kept[i] == 0. Returns count zeroed.
inline int ApplyKeptDofMask(mfem::ParGridFunction& gf,
                             const std::vector<unsigned char>& kept)
{
    int zeroed = 0;
    for (int i = 0; i < gf.Size(); i++) {
        if (!kept[i] && gf(i) != 0.0) {
            gf(i) = 0.0;
            zeroed++;
        }
    }
    return zeroed;
}

} // namespace SEM
