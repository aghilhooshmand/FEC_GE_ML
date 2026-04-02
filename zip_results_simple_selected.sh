#!/usr/bin/env bash

set -euo pipefail

# Create one ZIP per whitelisted experiment folder under results_simple, including only selected files.
# Folder structure inside each ZIP is preserved (starts with the experiment folder name).
#
# Included filenames:
# - generation_stats_aggregated.csv
# - summary_aggregated.csv
# - fec_individual_cache_aggregated_by_gen.csv
# - fec_individual_cache_aggregated_summary.csv
# - generation_stats_aggregated_FEC.csv
# - summary_aggregated_FEC.csv
# - FEC_report_simple_noFake.html
# - FEC_report_simple_withFake.html
# - summary_baseline_vs_FEC_noFake.csv
# - summary_baseline_vs_FEC_withFake.csv

RESULTS_ROOT="${1:-results_simple}"
OUT_DIR="${2:-results_simple_zips}"

if [[ ! -d "${RESULTS_ROOT}" ]]; then
  echo "Results directory not found: ${RESULTS_ROOT}"
  exit 1
fi

mkdir -p "${OUT_DIR}"

python3 - "$RESULTS_ROOT" "$OUT_DIR" <<'PY'
from pathlib import Path
import sys
import zipfile

results_root = Path(sys.argv[1]).resolve()
out_dir = Path(sys.argv[2]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

allowed_names = {
    "generation_stats_aggregated.csv",
    "summary_aggregated.csv",
    "fec_individual_cache_aggregated_by_gen.csv",
    "fec_individual_cache_aggregated_summary.csv",
    "generation_stats_aggregated_FEC.csv",
    "summary_aggregated_FEC.csv",
    "FEC_report_simple_noFake.html",
    "FEC_report_simple_withFake.html",
    "summary_baseline_vs_FEC_noFake.csv",
    "summary_baseline_vs_FEC_withFake.csv",
}

allowed_experiments = (
    "saheart_Gen_50_Pop_1000",
    "sonar_Gen_50_Pop_1000",
    "spectf_Gen_50_Pop_1000",
    "spect_Gen_50_Pop_1000",
    "threeOf9_Gen_50_Pop_1000",
    "wine_Gen_50_Pop_1000",
    "wine_smoth_class_Gen_50_Pop_1000",
    "Wisconsin_Breast_Cancer_without_ID_Gen_50_Pop_1000",
)

root_parent = results_root.parent
count_archives = 0
count_files = 0

for exp_name in allowed_experiments:
    exp_dir = results_root / exp_name
    if not exp_dir.is_dir():
        print(f"Skipping {exp_name} (folder not found under {results_root})")
        continue

    selected = [p for p in exp_dir.rglob("*") if p.is_file() and p.name in allowed_names]
    if not selected:
        print(f"Skipping {exp_dir.name} (no matching files)")
        continue

    zip_path = out_dir / f"{exp_dir.name}.zip"
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for file_path in selected:
            # Keep folder structure unchanged from results_simple parent:
            # e.g. results_simple/<exp>/baseline/... -> <exp>/baseline/...
            arcname = file_path.relative_to(root_parent)
            zf.write(file_path, arcname=arcname.as_posix())

    count_archives += 1
    count_files += len(selected)
    print(f"Created {zip_path} ({len(selected)} files)")

print(f"Done. Created {count_archives} archive(s), total files added: {count_files}")
PY
