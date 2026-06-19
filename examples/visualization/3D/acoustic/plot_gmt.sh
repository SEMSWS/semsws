#!/bin/bash
# =============================================================================
# GMT plot script for 3D acoustic visualization test
#
# Plots cross-section wavefield snapshots and material fields.
# Reads grid_info.txt for exact grid parameters.
#
# Usage:
#   bash plot_gmt.sh [results_dir]
#
# Requirements: GMT 6+
# =============================================================================
set -e

RESULTS=${1:-./results}
OUTDIR=${RESULTS}/figures
mkdir -p ${OUTDIR}

# =============================================================================
# Helper: read grid_info.txt
# =============================================================================
read_grid_info() {
    local infofile=$1
    eval $(grep -v '^#' ${infofile} | grep '=')
}

# =============================================================================
# Plot a single cross-section directory
# =============================================================================
plot_cross_section() {
    local GMT_DIR=$1      # e.g., results/wavefield/gmt or results/material/gmt
    local SLICE_DIR=$2    # e.g., yz_x12000
    local OUTPUT_PREFIX=$3 # e.g., wavefield or material
    local IS_WAVEFIELD=$4 # 1 for wavefield (symmetric cmap), 0 for material

    local FULL_DIR="${GMT_DIR}/${SLICE_DIR}"
    local INFO="${FULL_DIR}/grid_info.txt"

    if [ ! -f "${INFO}" ]; then
        echo "  No grid_info.txt in ${FULL_DIR}"
        return
    fi

    read_grid_info ${INFO}
    echo "  ${SLICE_DIR}: ${nx}x${ny}, region=${region}"

    mkdir -p ${OUTDIR}/${OUTPUT_PREFIX}/${SLICE_DIR}

    for XYZFILE in $(ls ${FULL_DIR}/*.xyz 2>/dev/null | sort); do
        local BASENAME=$(basename ${XYZFILE} .xyz)
        local GRD="${OUTDIR}/${OUTPUT_PREFIX}/${SLICE_DIR}/${BASENAME}.grd"
        local FIG="${OUTDIR}/${OUTPUT_PREFIX}/${SLICE_DIR}/${BASENAME}"

        gmt xyz2grd ${XYZFILE} -G${GRD} -R${region} -I${spacing}

        # Check constant field
        local ZMIN=$(gmt grdinfo ${GRD} -Cn -o4)
        local ZMAX=$(gmt grdinfo ${GRD} -Cn -o5)
        local IS_CONST=$(awk -v zmax="${ZMAX}" -v zmin="${ZMIN}" 'BEGIN{m=(zmax+zmin)/2; s=zmax-zmin; print (m!=0 && s/m<0.001) ? 1 : 0}')
        local ZMEAN=$(awk "BEGIN{printf \"%.6g\", (${ZMAX}+${ZMIN})/2}")

        # Extract axis labels from slice name
        local AXIS1=$(echo ${SLICE_DIR} | cut -c1)
        local AXIS2=$(echo ${SLICE_DIR} | cut -c2)

        gmt begin ${FIG} png
            if [ "${IS_WAVEFIELD}" = "1" ] && [ "${IS_CONST}" = "0" ]; then
                # Wavefield: symmetric colormap
                local VMAX=$(awk -v zmax="${ZMAX}" -v zmin="${ZMIN}" 'BEGIN{a=zmin<0?-zmin:zmin; b=zmax<0?-zmax:zmax; print (a>b?a:b)}')
                gmt makecpt -Cpolar -T-${VMAX}/${VMAX} -Z
                gmt grdimage ${GRD} -R${region} -JX12c/12c -C
            elif [ "${IS_CONST}" = "1" ]; then
                # Constant field
                local CVAL=$(awk "BEGIN{printf \"%.6g\", ${ZMEAN}-1}")
                local CVAL2=$(awk "BEGIN{printf \"%.6g\", ${ZMEAN}+1}")
                gmt makecpt -Cviridis -T${CVAL}/${CVAL2}
                gmt grdimage ${GRD} -R${region} -JX12c/12c -C
            else
                # Variable material field
                gmt grdimage ${GRD} -R${region} -JX12c/12c -Cviridis
            fi
            gmt basemap -Bxaf+l"${AXIS1} (m)" -Byaf+l"${AXIS2} (m)" -BWSne
            gmt colorbar -DJBC+w10c/0.4c+o0/1c -Baf+l"${BASENAME}"
        gmt end

        echo "    ${BASENAME} -> ${FIG}.png"
    done
}

# =============================================================================
# Material fields
# =============================================================================
echo "Plotting material cross-sections..."
GMT_MAT="${RESULTS}/material/gmt"
if [ -d "${GMT_MAT}" ]; then
    for SLICE in $(ls -d ${GMT_MAT}/*/ 2>/dev/null | xargs -I{} basename {}); do
        plot_cross_section "${GMT_MAT}" "${SLICE}" "material" 0
    done
fi

# =============================================================================
# Wavefield snapshots
# =============================================================================
echo "Plotting wavefield cross-sections..."
GMT_WF="${RESULTS}/wavefield/gmt"
if [ -d "${GMT_WF}" ]; then
    for SLICE in $(ls -d ${GMT_WF}/*/ 2>/dev/null | xargs -I{} basename {}); do
        plot_cross_section "${GMT_WF}" "${SLICE}" "wavefield" 1
    done
fi

echo "Done. Figures saved to ${OUTDIR}/"
