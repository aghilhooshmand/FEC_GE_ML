#!/usr/bin/env bash

set -euo pipefail

# Reduce nested BLAS/OpenMP threading per process for more stable timing.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Unified batch launcher:
# - baseline and FEC run at the same time
# - separate parallel limits per pipeline
# Example below gives total ~= 60 parallel jobs (30 baseline + 30 FEC).

# ----------------------- Baseline batch config -----------------------
BASELINE_NUM_RUNS=10
BASELINE_MAX_PARALLEL=5

# ------------------------- FEC batch config --------------------------
FEC_RUNS_PER_CONFIG=10
FEC_MAX_PARALLEL=5
FEC_SAMPLING_METHODS=("farthest_point" "stratified")
FEC_FRACTIONS=(0.1 0.2 0.3)
FEC_FAKE_HIT_THRESHOLDS=(0)
# Run both modes:
# false -> noFake folder, true -> withFake folder
FEC_EVALUATE_FAKE_MODES=("true")

# Datasets come from config.json DATA object as "file|label" lines.
mapfile -t DATASETS < <(python3 - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path("config.json").read_text(encoding="utf-8"))
data = payload.get("DATA", {})
for dataset_file, label_col in data.items():
    print(f"{dataset_file}|{label_col}")
PY
)

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "No datasets found in config.json DATA."
  exit 1
fi


run_baseline_batch() {
  local dataset_file="$1"
  local label_col="$2"
  echo "=== Baseline batch: dataset=${dataset_file}, label=${label_col}, runs=${BASELINE_NUM_RUNS}, max_parallel=${BASELINE_MAX_PARALLEL} ==="
  local active=0
  for ((i=1; i<=BASELINE_NUM_RUNS; i++)); do
    if (( active >= BASELINE_MAX_PARALLEL )); then
      wait -n
      active=$((active-1))
    fi
    echo "  -> python3 baseline_runs_simple.py --run-index ${i} --dataset-file ${dataset_file} --label-column ${label_col}"
    python3 baseline_runs_simple.py --run-index "${i}" --dataset-file "${dataset_file}" --label-column "${label_col}" &
    active=$((active+1))
  done
  wait
  echo "=== Baseline batch finished ==="
}


run_fec_batch() {
  local dataset_file="$1"
  local label_col="$2"
  echo "=== FEC batch: dataset=${dataset_file}, label=${label_col}, runs_per_config=${FEC_RUNS_PER_CONFIG}, max_parallel=${FEC_MAX_PARALLEL} ==="
  local active=0
  local mode method frac th i
  for mode in "${FEC_EVALUATE_FAKE_MODES[@]}"; do
    for method in "${FEC_SAMPLING_METHODS[@]}"; do
      for frac in "${FEC_FRACTIONS[@]}"; do
        for th in "${FEC_FAKE_HIT_THRESHOLDS[@]}"; do
          echo "  --- FEC config: dataset=${dataset_file}, label=${label_col}, mode=${mode}, method=${method}, frac=${frac}, th=${th}"
          for ((i=1; i<=FEC_RUNS_PER_CONFIG; i++)); do
            if (( active >= FEC_MAX_PARALLEL )); then
              wait -n
              active=$((active-1))
            fi
            if [[ "${mode}" == "true" ]]; then
              echo "  -> python3 FEC_runs_simple.py --run-index ${i} --dataset-file ${dataset_file} --label-column ${label_col} --sample-fraction ${frac} --sampling-method ${method} --fake-hit-threshold ${th} --evaluate-fake-hits"
              python3 FEC_runs_simple.py \
                --run-index "${i}" \
                --dataset-file "${dataset_file}" \
                --label-column "${label_col}" \
                --sample-fraction "${frac}" \
                --sampling-method "${method}" \
                --fake-hit-threshold "${th}" \
                --evaluate-fake-hits &
            else
              echo "  -> python3 FEC_runs_simple.py --run-index ${i} --dataset-file ${dataset_file} --label-column ${label_col} --sample-fraction ${frac} --sampling-method ${method} --fake-hit-threshold ${th}"
              python3 FEC_runs_simple.py \
                --run-index "${i}" \
                --dataset-file "${dataset_file}" \
                --label-column "${label_col}" \
                --sample-fraction "${frac}" \
                --sampling-method "${method}" \
                --fake-hit-threshold "${th}" &
            fi
            active=$((active+1))
          done
        done
      done
    done
  done
  wait
  echo "=== FEC batch finished ==="
}


echo "Starting unified parallel batch..."
echo "Total target parallelism ~= BASELINE_MAX_PARALLEL + FEC_MAX_PARALLEL = $((BASELINE_MAX_PARALLEL + FEC_MAX_PARALLEL))"
echo

for item in "${DATASETS[@]}"; do
  dataset_file="${item%%|*}"
  label_col="${item#*|}"
  echo
  echo "### Dataset loop: file=${dataset_file} label=${label_col}"

  # Run both pipelines in parallel for this dataset.
  run_baseline_batch "${dataset_file}" "${label_col}" &
  pid_baseline=$!

  run_fec_batch "${dataset_file}" "${label_col}" &
  pid_fec=$!

  wait "${pid_baseline}"
  wait "${pid_fec}"
done

echo
echo "All batches completed."
