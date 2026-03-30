#!/usr/bin/env python3
"""
Merge baseline-vs-FEC summary CSVs (speedup, test metrics, …) with dataset complexity
from data/datasets_complexity_summary.csv.

Scans results directories named <dataset_stem>_Gen_<G>_Pop_<P>/ for:
  summary_baseline_vs_FEC_noFake.csv, summary_baseline_vs_FEC_withFake.csv, or legacy
  summary_baseline_vs_FEC.csv

Join key: dataset_file = <dataset_stem>.csv (must match the dataset_file column in the
complexity table).

Example:
  python merge_summary_with_complexity.py
  python merge_summary_with_complexity.py --fec-mode noFake -o results_simple/summary_plus_complexity_noFake.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

EXPERIMENT_DIR_RE = re.compile(r"^(.+)_Gen_(\d+)_Pop_(\d+)$")


def _parse_experiment_dir(name: str) -> tuple[str, int, int] | None:
    m = EXPERIMENT_DIR_RE.match(name)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def _summary_candidates(subdir: Path, fec_mode: str) -> list[tuple[Path, str]]:
    """Return list of (path, fec_mode_label) to load."""
    if fec_mode == "noFake":
        p = subdir / "summary_baseline_vs_FEC_noFake.csv"
        return [(p, "noFake")] if p.is_file() else []
    if fec_mode == "withFake":
        p = subdir / "summary_baseline_vs_FEC_withFake.csv"
        return [(p, "withFake")] if p.is_file() else []
    if fec_mode == "legacy":
        p = subdir / "summary_baseline_vs_FEC.csv"
        return [(p, "legacy")] if p.is_file() else []
    # both
    out: list[tuple[Path, str]] = []
    for fname, label in (
        ("summary_baseline_vs_FEC_noFake.csv", "noFake"),
        ("summary_baseline_vs_FEC_withFake.csv", "withFake"),
    ):
        p = subdir / fname
        if p.is_file():
            out.append((p, label))
    if not out:
        p = subdir / "summary_baseline_vs_FEC.csv"
        if p.is_file():
            out.append((p, "legacy"))
    return out


def collect_merged(
    results_root: Path,
    complexity_path: Path,
    fec_mode: str,
) -> pd.DataFrame:
    complexity = pd.read_csv(complexity_path)
    if "dataset_file" not in complexity.columns:
        raise SystemExit(f"{complexity_path} has no 'dataset_file' column")

    frames: list[pd.DataFrame] = []
    for subdir in sorted(results_root.iterdir()):
        if not subdir.is_dir():
            continue
        parsed = _parse_experiment_dir(subdir.name)
        if not parsed:
            continue
        stem, generations, population = parsed
        dataset_file = f"{stem}.csv"
        for csv_path, mode_label in _summary_candidates(subdir, fec_mode):
            df = pd.read_csv(csv_path)
            df.insert(0, "experiment_dir", subdir.name)
            df.insert(1, "dataset_file", dataset_file)
            df.insert(2, "evolution_generations", generations)
            df.insert(3, "evolution_population", population)
            df.insert(4, "fec_mode", mode_label)
            merged = df.merge(complexity, on="dataset_file", how="left")
            frames.append(merged)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine summary_baseline_vs_FEC*.csv with datasets_complexity_summary.csv."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results_simple"),
        help="Directory containing <dataset>_Gen_*_Pop_* experiment folders (default: results_simple).",
    )
    parser.add_argument(
        "--complexity",
        type=Path,
        default=Path("data/datasets_complexity_summary.csv"),
        help="Wide complexity table (default: data/datasets_complexity_summary.csv).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("results_simple/summary_plus_complexity.csv"),
        help="Output CSV path (default: results_simple/summary_plus_complexity.csv).",
    )
    parser.add_argument(
        "--fec-mode",
        choices=("both", "noFake", "withFake", "legacy"),
        default="both",
        help="Which summary file(s) to read per experiment (default: both noFake and withFake if present).",
    )
    args = parser.parse_args()

    results_root = args.results_dir.resolve()
    complexity_path = args.complexity.resolve()
    if not complexity_path.is_file():
        raise SystemExit(f"Complexity file not found: {complexity_path}")
    if not results_root.is_dir():
        raise SystemExit(f"Results directory not found: {results_root}")

    out = collect_merged(results_root, complexity_path, args.fec_mode)
    if out.empty:
        print("No matching summary_baseline_vs_FEC*.csv files found; nothing written.", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output.resolve()}")


if __name__ == "__main__":
    main()
