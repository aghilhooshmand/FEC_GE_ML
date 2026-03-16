#!/usr/bin/env bash

set -euo pipefail

# Simple launcher that runs FEC_runs_simple.py many times,
# each time with a different --run-index. The Python code
# derives the seed from base-seed, run-index, method and fraction.
#
# Usage:
#   chmod +x run_FEC_simple_batch.sh
#   ./run_FEC_simple_batch.sh           # uses defaults below
#   ./run_FEC_simple_batch.sh 5         # 5 runs per (method, fraction)
#   ./run_FEC_simple_batch.sh 30 4      # 30 runs per config, max 4 parallel

# Edit these arrays to choose what to run:
SAMPLING_METHODS=("farthest_point" "kmeans") #("farthest_point" "kmeans")
FRACTIONS=(0.1 0.2)
FAKE_HIT_THRESHOLDS=(0 1e-5)

RUNS_PER_CONFIG="${1:-30}"
MAX_PARALLEL="${2:-30}"
BASE_SEED=42

echo "FEC_simple batch configuration:"
echo "  methods:   ${SAMPLING_METHODS[*]}"
echo "  fractions: ${FRACTIONS[*]}"
echo "  fake-hit thresholds: ${FAKE_HIT_THRESHOLDS[*]}"
echo "  runs per (method, frac, th): ${RUNS_PER_CONFIG}"
echo "  max parallel: ${MAX_PARALLEL}"
echo "  base seed: ${BASE_SEED}"
echo

for method in "${SAMPLING_METHODS[@]}"; do
  for frac in "${FRACTIONS[@]}"; do
    for th in "${FAKE_HIT_THRESHOLDS[@]}"; do
      echo "=== Launching ${RUNS_PER_CONFIG} FEC_simple runs for method=${method}, frac=${frac}, th=${th} (max ${MAX_PARALLEL} at a time) ==="
      for ((i=1; i<=RUNS_PER_CONFIG; i++)); do
        (( i > MAX_PARALLEL )) && wait -n
        echo "  -> python FEC_runs_simple.py --run-index ${i} --base-seed ${BASE_SEED} --sample-fraction ${frac} --sampling-method ${method} --fake-hit-threshold ${th}"
        python FEC_runs_simple.py \
          --run-index "${i}" \
          --base-seed "${BASE_SEED}" \
          --sample-fraction "${frac}" \
          --sampling-method "${method}" \
          --fake-hit-threshold "${th}" &
      done
      wait
      echo
    done
  done
done

echo "All FEC_simple runs finished."

