#!/bin/bash -l
# ============================================================
# 3D Elastic Analytic Benchmark
#
# Validates SEMSWS against the Aki & Richards analytical solution
# for a Mrp double-couple point source in a homogeneous elastic full-space.
#
# Usage:
#   bash run_benchmark.sh [NPROCS]         # default NPROCS=4
#
# Prereqs:
#   - SEMSWS build at ../../build/src/semsws
#   - Python (numpy) for generate_analytical.py
# ============================================================

set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
PROGDIR="${REPO_ROOT}/build/src"
NPROCS="${1:-4}"

cd "${HERE}"

echo "=========================================="
echo "3D Elastic Analytic Benchmark"
echo "  NPROCS=${NPROCS}"
echo "Job started at: $(date)"
echo "=========================================="

# Step 1: Generate analytical references
echo ""
echo "=== Step 1: Generate analytical references ==="
python3 generate_analytical.py

# Step 2: Run SEMSWS simulation
echo ""
echo "=== Step 2: Run SEMSWS simulation ==="
rm -rf results
mpirun -np "${NPROCS}" "${PROGDIR}/semsws" --config config.yaml

# Step 3: Compare results (inline: normalized L2 error per trace)
echo ""
echo "=== Step 3: Compare results ==="
echo "Comparing results vs ref..."
for station in Z1 Z2 Z3 Z4 Z5 Z6; do
    for comp in x y z; do
        COMP_UPPER=$(echo "$comp" | tr 'a-z' 'A-Z')
        ref_file="ref/XX.${station}.FX${COMP_UPPER}.semd"
        res_file="results/${station}_${comp}_0000.d"
        if [ ! -f "$res_file" ]; then
            echo "  MISSING: ${res_file}"
            continue
        fi
        err=$(paste "$ref_file" "$res_file" | awk '{
            d = ($2 - $4); sum_d2 += d*d; sum_r2 += $2*$2;
        } END {
            if (sum_r2 > 0) printf "%.6f", sqrt(sum_d2/sum_r2);
            else printf "0.000000";
        }')
        echo "  ${station}.FX${COMP_UPPER}: L2_err=${err}"
    done
done

echo ""
echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="
