#!/usr/bin/env python3
"""
Build a ranked dataset-complexity table from ``datasets_complexity_long.csv``.

Steps
-----
1. Pivot long format to one row per dataset.
2. Merge general descriptors from ``datasets_complexity_summary.csv`` (same ``dataset_file``).
3. Align each PyMFE complexity metric so that **larger = more difficult** (see
   ``METRICS_HIGHER_IS_EASIER``): for those, use ``x' = max - x`` on the column
   (same ranking as ``1 - x`` when values lie in ``[0, 1]``).
4. Min–max scale each metric column to ``[0, 1]`` across datasets (ignoring NaNs).
5. ``avg_all_metrics`` = row mean of aligned scaled metrics (ignoring NaNs).
6. Sort rows by ``avg_all_metrics`` **descending** (most complex first).

Usage
-----
  python3 build_datasets_complexity_ranked.py
  python3 build_datasets_complexity_ranked.py --long-csv data/datasets_complexity_long.csv \\
      --summary-csv data/datasets_complexity_summary.csv --output data/datasets_complexity_ranked.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# PyMFE / Lorena-style: "efficiency" measures — higher value ⇒ easier separation ⇒ invert.
METRICS_HIGHER_IS_EASIER = frozenset(
    {
        "f3.mean",
        "f4.mean",
    }
)

# Descriptor columns copied from summary (not averaged into complexity score).
META_COLUMNS_PREFERRED = [
    "dataset_file",
    "n_rows",
    "n_columns_total",
    "n_features_raw",
    "n_features_numeric_raw",
    "n_features_categorical_raw",
    "n_missing_cells",
    "missing_pct",
    "label_column",
    "n_classes",
    "majority_class_fraction",
    "is_binary",
    "n_rows_used_after_cleaning",
    "n_features_used_after_encoding",
]


def _pivot_long(long_df: pd.DataFrame) -> pd.DataFrame:
    if not {"dataset_file", "metric", "value"}.issubset(long_df.columns):
        raise SystemExit("Long CSV must have columns: dataset_file, metric, value")
    wide = long_df.pivot_table(
        index="dataset_file",
        columns="metric",
        values="value",
        aggfunc="first",
    )
    wide = wide.reset_index()
    wide.columns.name = None
    return wide


def _metric_columns(wide: pd.DataFrame) -> list[str]:
    skip = {"dataset_file", "error"}
    return [c for c in wide.columns if c not in skip]


def _align_and_scale(
    wide: pd.DataFrame,
    metric_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Return copy with aligned + min-max scaled columns; column names unchanged."""
    out = wide.copy()
    for c in metric_cols:
        s = pd.to_numeric(out[c], errors="coerce").astype(float)
        if c in METRICS_HIGHER_IS_EASIER:
            valid = s.dropna()
            if valid.size:
                lo, hi = float(valid.min()), float(valid.max())
                if hi > lo:
                    s = hi - s
                else:
                    s = pd.Series(0.5, index=s.index)
        out[c] = s

    scaled = out.copy()
    for c in metric_cols:
        col = scaled[c]
        lo = np.nanmin(col.to_numpy(dtype=float))
        hi = np.nanmax(col.to_numpy(dtype=float))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            scaled[c] = 0.5
        else:
            scaled[c] = (col - lo) / (hi - lo)
    return scaled, metric_cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank datasets by aligned mean complexity.")
    parser.add_argument(
        "--long-csv",
        type=Path,
        default=Path("data/datasets_complexity_long.csv"),
        help="Long-format complexity metrics.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("data/datasets_complexity_summary.csv"),
        help="Wide summary (used for dataset descriptors).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/datasets_complexity_ranked.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    long_df = pd.read_csv(args.long_csv)
    wide = _pivot_long(long_df)
    # Only PyMFE / long-file metric names — not summary descriptors (n_rows, etc.).
    metric_cols = _metric_columns(wide)

    summary = pd.read_csv(args.summary_csv)
    meta_available = [c for c in META_COLUMNS_PREFERRED if c in summary.columns]
    meta = summary[meta_available].copy()

    merged = meta.merge(wide, on="dataset_file", how="right")

    scaled, metric_cols = _align_and_scale(merged, metric_cols)
    m = scaled[metric_cols].to_numpy(dtype=float)
    row_mean = np.nanmean(m, axis=1)
    scaled["avg_all_metrics"] = row_mean

    meta_only = [c for c in meta_available if c != "dataset_file"]
    ordered_cols = (
        ["dataset_file"]
        + meta_only
        + [c for c in metric_cols if c not in meta_only]
        + ["avg_all_metrics"]
    )
    ordered_cols = [c for c in ordered_cols if c in scaled.columns]
    out = scaled[ordered_cols].sort_values("avg_all_metrics", ascending=False, na_position="last")
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(out)} rows), sorted by avg_all_metrics (desc = most complex first).")


if __name__ == "__main__":
    main()
