  #!/usr/bin/env bash

  set -euo pipefail

  # Run FEC_report_simple.py for all experiment folders in results_simple.
  # An experiment folder is expected to match: <dataset>_Gen_<G>_Pop_<P>

  RESULTS_ROOT="${1:-results_simple}"

  if [[ ! -d "${RESULTS_ROOT}" ]]; then
    echo "Results directory not found: ${RESULTS_ROOT}"
    exit 1
  fi

  shopt -s nullglob globstar
  count=0

  for exp_dir in "${RESULTS_ROOT}"/*_Gen_*_Pop_*; do
    [[ -d "${exp_dir}" ]] || continue

    # Only run report if baseline and FEC subfolders exist.
    if [[ ! -d "${exp_dir}/baseline" || ! -d "${exp_dir}/FEC" ]]; then
      echo "Skipping ${exp_dir} (missing baseline/ or FEC/)"
      continue
    fi

    # Also require at least one CSV somewhere under baseline/ and FEC/.
    baseline_csv=( "${exp_dir}/baseline"/**/*.csv "${exp_dir}/baseline"/*.csv )
    fec_csv=( "${exp_dir}/FEC"/**/*.csv "${exp_dir}/FEC"/*.csv )
    if [[ ${#baseline_csv[@]} -eq 0 || ${#fec_csv[@]} -eq 0 ]]; then
      echo "Skipping ${exp_dir} (baseline/ or FEC/ has no CSV files)"
      continue
    fi

    echo "=== Running report: ${exp_dir} ==="
    python3 FEC_report_simple.py "${exp_dir}"
    count=$((count + 1))
  done

  echo "Done. Generated reports for ${count} experiment folder(s)."
