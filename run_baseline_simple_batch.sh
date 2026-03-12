#!/usr/bin/env bash

set -euo pipefail

# Simple launcher that runs baseline_runs_simple.py many times,
# each time with a different --run-index. The Python code
# derives the seed from base-seed and run-index.
#
# Usage:
#   chmod +x run_baseline_simple_batch.sh
#   ./run_baseline_simple_batch.sh          # default: 20 runs, max 4 at a time
#   ./run_baseline_simple_batch.sh 5       # 5 runs
#   ./run_baseline_simple_batch.sh 20 8    # 20 runs, max 8 parallel

NUM_RUNS="${1:-20}"
MAX_PARALLEL="${2:-4}"
BASE_SEED=42

echo "Launching ${NUM_RUNS} baseline_simple runs (max ${MAX_PARALLEL} at a time)."

for ((i=1; i<=NUM_RUNS; i++)); do
  (( i > MAX_PARALLEL )) && wait -n
  echo "  -> python baseline_runs_simple.py --run-index ${i} --base-seed ${BASE_SEED}"
  python baseline_runs_simple.py --run-index "${i}" --base-seed "${BASE_SEED}" &
done
wait
echo "All baseline_simple runs finished."

#!/usr/bin/env bash

set -euo pipefail

# Simple launcher that runs baseline_runs_simple.py many times,
# each time with a different --run-index. The Python code
# derives the seed from base-seed and run-index.
#
# Usage:
#   chmod +x run_baseline_simple_batch.sh
#   ./run_baseline_simple_batch.sh          # default: 20 runs, max 4 at a time
#   ./run_baseline_simple_batch.sh 5       # 5 runs
#   ./run_baseline_simple_batch.sh 20 8    # 20 runs, max 8 parallel

NUM_RUNS="${1:-20}"
MAX_PARALLEL="${2:-4}"
BASE_SEED=42

echo "Launching ${NUM_RUNS} baseline_simple runs (max ${MAX_PARALLEL} at a time)."

for ((i=1; i<=NUM_RUNS; i++)); do
  (( i > MAX_PARALLEL )) && wait -n
  echo "  -> python baseline_runs_simple.py --run-index ${i} --base-seed ${BASE_SEED}"
  python baseline_runs_simple.py --run-index "${i}" --base-seed "${BASE_SEED}" &
done
wait
echo "All baseline_simple runs finished."

