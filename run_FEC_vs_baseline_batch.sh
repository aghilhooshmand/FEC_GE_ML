#!/usr/bin/env bash

set -euo pipefail

# Simple batch launcher for FEC_vs_baseline.py.
# Usage:
#   chmod +x run_FEC_vs_baseline_batch.sh
#   # Use default folder name (dataset_Gen_#_Pop_#_Run_#):
#   ./run_FEC_vs_baseline_batch.sh 8
#   # Or explicit folder name suffix (under results/FEC_vs_baseline/):
#   ./run_FEC_vs_baseline_batch.sh 3 "wine_Gen_50_Pop_500_Run_10"
#
# All runs share a single experiment directory under results/FEC_vs_baseline/,
# so that FEC_report.py can aggregate them later.

# Number of independent runs (global run-indices) to launch.
# You can override by passing a number as the first argument.
NUM_RUNS="${1:-10}"

echo "Preparing to launch ${NUM_RUNS} runs of FEC_vs_baseline.py ..."

# Determine experiment directory name (for display only).
EXP_DIR=$(python - << 'PY'
from pathlib import Path
from config import CONFIG

dataset_stem = Path(CONFIG.get("dataset.file", "data")).stem
n_gen = CONFIG.get("evolution.generations", 0)
pop = CONFIG.get("evolution.population", 0)
n_runs = CONFIG.get("evolution.n_runs", 1)
folder_name = f"{dataset_stem}_Gen_{n_gen}_Pop_{pop}_Run_{n_runs}"
print(str(Path("results") / "FEC_vs_baseline" / folder_name))
PY
)

echo "Experiment directory (computed by FEC_vs_baseline.py): ${EXP_DIR}"
mkdir -p "${EXP_DIR}"

# Launch runs in parallel, each with a different --run-index but the same experiment directory.
for i in $(seq 1 "${NUM_RUNS}"); do
  echo "Launching run-index ${i} ..."
  # If you want to pin each run to a specific CPU core, you can replace the next line with:
  #   taskset -c $((i - 1)) python FEC_vs_baseline.py --run-index "${i}" &
  python FEC_vs_baseline.py --run-index "${i}" &
done

echo "All ${NUM_RUNS} runs launched. Waiting for completion ..."
wait
echo "All runs finished. Aggregating later with:"
echo "  python FEC_report.py \"${EXP_DIR}\""

