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

# ---------------------- CPU pinning (taskset) ------------------------
# If true: each launched process is pinned to one core (round-robin).
CPU_PINNING_ENABLED=true
# You can use ranges and commas, e.g. "0-39" or "0-19,40-59".
BASELINE_CORE_RANGE="0-3"
FEC_CORE_RANGE="4-9"

# ----------------------- Baseline batch config -----------------------
BASELINE_NUM_RUNS=20
BASELINE_MAX_PARALLEL=5

# ------------------------- FEC batch config --------------------------
FEC_RUNS_PER_CONFIG=20
FEC_MAX_PARALLEL=5
FEC_SAMPLING_METHODS=("farthest_point" "stratified")
FEC_FRACTIONS=(0.1 0.2 0.3)
# Run both modes:
# false -> noFake folder, true -> withFake folder
FEC_EVALUATE_FAKE_MODES=("false" "true")

expand_core_range() {
  local spec="$1"
  local part s e c
  local -a parts
  IFS=',' read -r -a parts <<< "${spec}"
  for part in "${parts[@]}"; do
    if [[ "${part}" == *-* ]]; then
      s="${part%-*}"
      e="${part#*-}"
      for ((c=s; c<=e; c++)); do
        echo "${c}"
      done
    elif [[ -n "${part}" ]]; then
      echo "${part}"
    fi
  done
}

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
  local core_idx=0
  local core=""
  local -a baseline_cores=()
  if [[ "${CPU_PINNING_ENABLED}" == "true" ]]; then
    mapfile -t baseline_cores < <(expand_core_range "${BASELINE_CORE_RANGE}")
    if [[ ${#baseline_cores[@]} -eq 0 ]]; then
      echo "No baseline cores parsed from BASELINE_CORE_RANGE=${BASELINE_CORE_RANGE}"
      exit 1
    fi
  fi
  for ((i=1; i<=BASELINE_NUM_RUNS; i++)); do
    if (( active >= BASELINE_MAX_PARALLEL )); then
      wait -n
      active=$((active-1))
    fi
    if [[ "${CPU_PINNING_ENABLED}" == "true" ]]; then
      core="${baseline_cores[$((core_idx % ${#baseline_cores[@]}))]}"
      core_idx=$((core_idx+1))
      echo "  -> taskset -c ${core} python3 baseline_runs_simple.py --run-index ${i} --dataset-file ${dataset_file} --label-column ${label_col}"
      taskset -c "${core}" python3 baseline_runs_simple.py --run-index "${i}" --dataset-file "${dataset_file}" --label-column "${label_col}" &
    else
      echo "  -> python3 baseline_runs_simple.py --run-index ${i} --dataset-file ${dataset_file} --label-column ${label_col}"
      python3 baseline_runs_simple.py --run-index "${i}" --dataset-file "${dataset_file}" --label-column "${label_col}" &
    fi
    active=$((active+1))
  done
  wait
  echo "=== Baseline batch finished ==="
}


run_fec_config_batch() {
  local dataset_file="$1"
  local label_col="$2"
  local mode="$3"
  local method="$4"
  local frac="$5"

  echo "=== FEC config batch: dataset=${dataset_file}, label=${label_col}, mode=${mode}, method=${method}, frac=${frac}, runs=${FEC_RUNS_PER_CONFIG}, max_parallel=${FEC_MAX_PARALLEL} ==="
  local active=0
  local i
  local core_idx=0
  local core=""
  local -a fec_cores=()
  if [[ "${CPU_PINNING_ENABLED}" == "true" ]]; then
    mapfile -t fec_cores < <(expand_core_range "${FEC_CORE_RANGE}")
    if [[ ${#fec_cores[@]} -eq 0 ]]; then
      echo "No FEC cores parsed from FEC_CORE_RANGE=${FEC_CORE_RANGE}"
      exit 1
    fi
  fi
  for ((i=1; i<=FEC_RUNS_PER_CONFIG; i++)); do
    if (( active >= FEC_MAX_PARALLEL )); then
      wait -n
      active=$((active-1))
    fi
    if [[ "${CPU_PINNING_ENABLED}" == "true" ]]; then
      core="${fec_cores[$((core_idx % ${#fec_cores[@]}))]}"
      core_idx=$((core_idx+1))
    fi
    if [[ "${mode}" == "true" ]]; then
      if [[ "${CPU_PINNING_ENABLED}" == "true" ]]; then
        echo "  -> taskset -c ${core} python3 FEC_runs_simple.py --run-index ${i} --dataset-file ${dataset_file} --label-column ${label_col} --sample-fraction ${frac} --sampling-method ${method} --evaluate-fake-hits"
        taskset -c "${core}" python3 FEC_runs_simple.py \
          --run-index "${i}" \
          --dataset-file "${dataset_file}" \
          --label-column "${label_col}" \
          --sample-fraction "${frac}" \
          --sampling-method "${method}" \
          --evaluate-fake-hits &
      else
        echo "  -> python3 FEC_runs_simple.py --run-index ${i} --dataset-file ${dataset_file} --label-column ${label_col} --sample-fraction ${frac} --sampling-method ${method} --evaluate-fake-hits"
        python3 FEC_runs_simple.py \
          --run-index "${i}" \
          --dataset-file "${dataset_file}" \
          --label-column "${label_col}" \
          --sample-fraction "${frac}" \
          --sampling-method "${method}" \
          --evaluate-fake-hits &
      fi
    else
      if [[ "${CPU_PINNING_ENABLED}" == "true" ]]; then
        echo "  -> taskset -c ${core} python3 FEC_runs_simple.py --run-index ${i} --dataset-file ${dataset_file} --label-column ${label_col} --sample-fraction ${frac} --sampling-method ${method}"
        taskset -c "${core}" python3 FEC_runs_simple.py \
          --run-index "${i}" \
          --dataset-file "${dataset_file}" \
          --label-column "${label_col}" \
          --sample-fraction "${frac}" \
          --sampling-method "${method}" &
      else
        echo "  -> python3 FEC_runs_simple.py --run-index ${i} --dataset-file ${dataset_file} --label-column ${label_col} --sample-fraction ${frac} --sampling-method ${method}"
        python3 FEC_runs_simple.py \
          --run-index "${i}" \
          --dataset-file "${dataset_file}" \
          --label-column "${label_col}" \
          --sample-fraction "${frac}" \
          --sampling-method "${method}" &
      fi
    fi
    active=$((active+1))
  done
  wait
  echo "=== FEC config batch finished ==="
}


echo "Starting unified parallel batch..."
echo "Total target parallelism ~= BASELINE_MAX_PARALLEL + FEC_MAX_PARALLEL = $((BASELINE_MAX_PARALLEL + FEC_MAX_PARALLEL))"
echo

for item in "${DATASETS[@]}"; do
  dataset_file="${item%%|*}"
  label_col="${item#*|}"
  echo
  echo "### Dataset loop: file=${dataset_file} label=${label_col}"

  # Baseline runs once per dataset.
  echo
  echo "### Baseline run (once for dataset)"
  run_baseline_batch "${dataset_file}" "${label_col}"

  # Then run all FEC configs for this dataset.
  for mode in "${FEC_EVALUATE_FAKE_MODES[@]}"; do
    for method in "${FEC_SAMPLING_METHODS[@]}"; do
      for frac in "${FEC_FRACTIONS[@]}"; do
        echo
        echo "### FEC config run (mode=${mode}, method=${method}, frac=${frac})"
        run_fec_config_batch "${dataset_file}" "${label_col}" "${mode}" "${method}" "${frac}"
      done
    done
  done
done

echo
echo "All batches completed."
