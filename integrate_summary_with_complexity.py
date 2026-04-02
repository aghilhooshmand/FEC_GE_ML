#!/usr/bin/env python3
"""
Join aggregated experiment summaries with datasets_complexity_summary.csv.

experiment_folder names look like: sonar_Gen_50_Pop_1000
dataset_file in complexity table:   sonar.csv  (stem is the part before _Gen_)

Reads (defaults):
  - results_simple/summary_baseline_vs_FEC_all_combined.csv
  - data/datasets_complexity_summary.csv

Writes:
  - results_simple/summary_baseline_vs_FEC_all_with_complexity.csv

Usage:
  python3 integrate_summary_with_complexity.py
  python3 integrate_summary_with_complexity.py --summary path/to/combined.csv --complexity path/to/summary.csv -o out.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# Match ..._Gen_<digits>_Pop_<digits> at end of experiment folder name.
_EXPERIMENT_TAIL = re.compile(r"_Gen_\d+_Pop_\d+$")


def experiment_folder_to_dataset_file(folder: str) -> str:
    """
    Map results_simple subfolder name to dataset_file for complexity join.

    Examples:
      sonar_Gen_50_Pop_1000 -> sonar.csv
      processed.cleveland_Gen_50_Pop_1000 -> processed.cleveland.csv
    """
    name = folder.strip()
    if "_Gen_" in name:
        stem = name.split("_Gen_", 1)[0]
    elif _EXPERIMENT_TAIL.search(name):
        stem = _EXPERIMENT_TAIL.sub("", name)
    else:
        stem = name
    return f"{stem}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge summary_baseline_vs_FEC_all_combined.csv with datasets_complexity_summary.csv."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results_simple/summary_baseline_vs_FEC_all_combined.csv"),
        help="Aggregated combined summary CSV (must have experiment_folder column).",
    )
    parser.add_argument(
        "--complexity",
        type=Path,
        default=Path("data/datasets_complexity_summary.csv"),
        help="Wide complexity summary CSV (dataset_file key).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("results_simple/summary_baseline_vs_FEC_all_with_complexity.csv"),
        help="Output merged CSV path.",
    )
    args = parser.parse_args()

    summary_path = args.summary.resolve()
    complexity_path = args.complexity.resolve()
    if not summary_path.is_file():
        raise SystemExit(f"Summary file not found: {summary_path}")
    if not complexity_path.is_file():
        raise SystemExit(f"Complexity file not found: {complexity_path}")

    summary = pd.read_csv(summary_path)
    if "experiment_folder" not in summary.columns:
        raise SystemExit("Summary CSV must contain column 'experiment_folder'.")

    complexity = pd.read_csv(complexity_path)
    if "dataset_file" not in complexity.columns:
        raise SystemExit("Complexity CSV must contain column 'dataset_file'.")

    summary = summary.copy()
    summary["dataset_file"] = summary["experiment_folder"].map(experiment_folder_to_dataset_file)

    # Place dataset_file next to experiment_folder for readability.
    cols = list(summary.columns)
    cols.remove("dataset_file")
    insert_at = cols.index("experiment_folder") + 1
    cols = cols[:insert_at] + ["dataset_file"] + cols[insert_at:]
    summary = summary[cols]

    merged = summary.merge(complexity, on="dataset_file", how="left", suffixes=("", "_cx_dup"))
    dup_cols = [c for c in merged.columns if c.endswith("_cx_dup")]
    if dup_cols:
        merged = merged.drop(columns=dup_cols)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    n_miss = merged["n_rows"].isna().sum() if "n_rows" in merged.columns else 0
    print(f"Wrote {len(merged)} rows -> {args.output}")
    if n_miss:
        print(f"Warning: {int(n_miss)} rows had no complexity match (check dataset_file / complexity table).")


if __name__ == "__main__":
    main()
