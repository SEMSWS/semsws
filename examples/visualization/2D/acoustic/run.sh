#!/bin/bash
# 2D acoustic visualization demo: short simulation that exercises every
# wavefield output format (glvis / paraview / gmt).

set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../../../.." && pwd)"
progdir="${REPO_ROOT}/build/src"
plotdir="${REPO_ROOT}/scripts/plot"
NP="${1:-2}"

cd "${HERE}"

echo "=== Forward simulation (NP=${NP}) ==="
rm -rf results
mpirun -np "${NP}" "${progdir}/semsws" --config config.yaml

echo "=== Matplotlib wavefield figures ==="
python3 "${plotdir}/plot_wavefield_2d.py" --results-dir ./results/vis \
    || echo "  (plot_wavefield_2d failed — need matplotlib + numpy)"

echo "=== GMT wavefield figures (requires GMT 6 CLI) ==="
bash plot_gmt.sh ./results/vis || echo "  (plot_gmt.sh failed — install GMT 6 to enable)"
