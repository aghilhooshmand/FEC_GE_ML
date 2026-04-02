#!/usr/bin/env python3
"""
Concatenate summary_baseline_vs_FEC_noFake.csv and summary_baseline_vs_FEC_withFake.csv
from every experiment folder under results_simple, adding the folder name as a column.

Writes:
  - <out-dir>/summary_baseline_vs_FEC_all_noFake.csv
  - <out-dir>/summary_baseline_vs_FEC_all_withFake.csv
  - <out-dir>/summary_baseline_vs_FEC_all_combined.csv  (both modes + fec_mode column)

Usage:
  python3 aggregate_summary_baseline_vs_fec.py
  python3 aggregate_summary_baseline_vs_fec.py --results-dir results_simple --output-dir results_simple
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

EXPERIMENT_GLOB = "*_Gen_*_Pop_*"


def _collect(
    results_root: Path,
    filename: str,
    column_name: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for exp_dir in sorted(results_root.glob(EXPERIMENT_GLOB)):
        if not exp_dir.is_dir():
            continue
        path = exp_dir / filename
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        df.insert(0, column_name, exp_dir.name)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate summary_baseline_vs_FEC_*.csv from all results_simple experiment folders."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results_simple"),
        help="Root directory containing <dataset>_Gen_*_Pop_* folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write aggregated CSVs (default: same as --results-dir).",
    )
    parser.add_argument(
        "--folder-column",
        type=str,
        default="experiment_folder",
        help="Column name for the results_simple subfolder name.",
    )
    args = parser.parse_args()

    results_root = args.results_dir.resolve()
    out_dir = (args.output_dir or args.results_dir).resolve()
    if not results_root.is_dir():
        raise SystemExit(f"Results directory not found: {results_root}")

    out_dir.mkdir(parents=True, exist_ok=True)
    col = args.folder_column

    nofake = _collect(results_root, "summary_baseline_vs_FEC_noFake.csv", col)
    withfake = _collect(results_root, "summary_baseline_vs_FEC_withFake.csv", col)

    out_nofake = out_dir / "summary_baseline_vs_FEC_all_noFake.csv"
    out_withfake = out_dir / "summary_baseline_vs_FEC_all_withFake.csv"
    out_combined = out_dir / "summary_baseline_vs_FEC_all_combined.csv"

    if not nofake.empty:
        nofake.to_csv(out_nofake, index=False)
        print(f"Wrote {len(nofake)} rows -> {out_nofake}")
    else:
        print("No summary_baseline_vs_FEC_noFake.csv files found.")

    if not withfake.empty:
        withfake.to_csv(out_withfake, index=False)
        print(f"Wrote {len(withfake)} rows -> {out_withfake}")
    else:
        print("No summary_baseline_vs_FEC_withFake.csv files found.")

    parts: list[pd.DataFrame] = []
    if not nofake.empty:
        tmp = nofake.copy()
        tmp.insert(1, "fec_mode", "noFake")
        parts.append(tmp)
    if not withfake.empty:
        tmp = withfake.copy()
        tmp.insert(1, "fec_mode", "withFake")
        parts.append(tmp)
    if parts:
        combined = pd.concat(parts, ignore_index=True)
        combined.to_csv(out_combined, index=False)
        print(f"Wrote {len(combined)} rows -> {out_combined}")
    else:
        print("Nothing to write for combined file.")


if __name__ == "__main__":
    main()
