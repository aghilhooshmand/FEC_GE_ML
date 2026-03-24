"""
Compute PyMFE complexity metrics and basic descriptors for CSV datasets.

How to run
----------
From the project root (or any directory; use absolute paths if needed)::

    python3 compute_data_complexity.py

Requirements: Python 3.10+, ``pandas``, ``numpy``, ``pymfe``. For HTML reports,
also install ``plotly`` (``pip install plotly``).

Outputs (defaults)
~~~~~~~~~~~~~~~~~~
- ``data/datasets_complexity_summary.csv`` — wide table (one row per dataset).
- ``data/datasets_complexity_long.csv`` — long format (metric/value rows).
- ``data/datasets_complexity.html`` — interactive Plotly heatmap (if plotly OK).

Common options
~~~~~~~~~~~~~~
- **Only small datasets** — compute complexity only for CSVs with **fewer than N**
  rows (after ``read_csv``); larger files are skipped::

    python3 compute_data_complexity.py --max-rows 1000

- **One dataset only**::

    python3 compute_data_complexity.py --dataset wine.csv

- **Batch of files (sorted list)**::

    python3 compute_data_complexity.py --start-index 0 --end-index 10 --append

- **Regenerate HTML from existing summary CSV**::

    python3 compute_data_complexity.py --html-only

Label column: optional ``config.json`` key ``DATA`` maps each dataset filename
to the target column name; otherwise ``target`` / ``class`` / ``label`` / ``y``
or the last column is used.

See ``python3 compute_data_complexity.py --help`` for all flags.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

# Columns treated as dataset metadata (not PyMFE complexity scores for heatmap).
_META_COLUMNS = {
    "dataset_file",
    "error",
    "label_column",
    "n_rows",
    "n_columns_total",
    "n_features_raw",
    "n_features_numeric_raw",
    "n_features_categorical_raw",
    "n_missing_cells",
    "missing_pct",
    "n_classes",
    "majority_class_fraction",
    "is_binary",
    "n_rows_used_after_cleaning",
    "n_features_used_after_encoding",
}


def _load_label_map(config_json: Path) -> Dict[str, str]:
    """Load dataset->label mapping from config.json (DATA section), if available."""
    if not config_json.exists():
        return {}
    try:
        payload = json.loads(config_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    data_map = payload.get("DATA", {})
    if not isinstance(data_map, dict):
        return {}
    return {str(k): str(v) for k, v in data_map.items()}


def _pick_label_column(df: pd.DataFrame, dataset_name: str, label_map: Dict[str, str]) -> str:
    """Pick target column by config map, common names, then fallback to last column."""
    if dataset_name in label_map and label_map[dataset_name] in df.columns:
        return label_map[dataset_name]
    for cand in ("target", "class", "label", "y"):
        if cand in df.columns:
            return cand
    return str(df.columns[-1])


def _prepare_xy(
    df: pd.DataFrame,
    label_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare X, y for PyMFE:
    - one-hot encode categorical features
    - coerce numerics
    - drop rows with missing values
    - encode y to integer labels
    """
    y_raw = df[label_col].copy()
    x_df = df.drop(columns=[label_col]).copy()

    # One-hot encode object/category features.
    obj_cols = [c for c in x_df.columns if x_df[c].dtype == "object" or str(x_df[c].dtype).startswith("category")]
    if obj_cols:
        x_df = pd.get_dummies(x_df, columns=obj_cols, dtype=float)

    # Ensure numeric feature matrix.
    for col in x_df.columns:
        x_df[col] = pd.to_numeric(x_df[col], errors="coerce")

    # Align and drop NA rows.
    merged = x_df.copy()
    merged["__y__"] = y_raw
    merged = merged.dropna(axis=0).reset_index(drop=True)

    y_vals = merged["__y__"]
    # Use float32 to reduce memory footprint.
    x_vals = merged.drop(columns=["__y__"]).to_numpy(dtype=np.float32)
    y_codes, _ = pd.factorize(y_vals, sort=True)

    return x_vals, y_codes.astype(int)


def _base_features(df: pd.DataFrame, label_col: str) -> Dict[str, Any]:
    """Compute basic dataset descriptors."""
    n_rows, n_cols_total = df.shape
    n_features_raw = max(0, n_cols_total - 1)
    n_missing = int(df.isna().sum().sum())
    missing_pct = float(100.0 * n_missing / (n_rows * n_cols_total)) if (n_rows * n_cols_total) > 0 else 0.0

    feat_df = df.drop(columns=[label_col])
    n_num = int(feat_df.select_dtypes(include=[np.number]).shape[1])
    n_cat = int(n_features_raw - n_num)

    y = df[label_col]
    cls_counts = y.value_counts(dropna=True)
    n_classes = int(cls_counts.shape[0])
    majority_frac = float((cls_counts.max() / cls_counts.sum()) if cls_counts.sum() > 0 else np.nan)

    out: Dict[str, Any] = {
        "n_rows": int(n_rows),
        "n_columns_total": int(n_cols_total),
        "n_features_raw": int(n_features_raw),
        "n_features_numeric_raw": n_num,
        "n_features_categorical_raw": n_cat,
        "n_missing_cells": n_missing,
        "missing_pct": missing_pct,
        "label_column": label_col,
        "n_classes": n_classes,
        "majority_class_fraction": majority_frac,
        "is_binary": bool(n_classes == 2),
    }
    return out


def _compute_complexity(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Compute all pyMFE complexity features (summary='mean' over multi-valued metrics)."""
    from pymfe.mfe import MFE

    mfe = MFE(groups=["complexity"], summary=("mean",))
    mfe.fit(X=x, y=y)
    ft_names, ft_vals = mfe.extract()

    out: Dict[str, Any] = {}
    for name, val in zip(ft_names, ft_vals):
        # cast np scalars/lists to plain python types for CSV robustness
        if isinstance(val, (np.floating, np.integer)):
            out[str(name)] = float(val)
        else:
            out[str(name)] = val
    return out


def _complexity_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Numeric columns excluding metadata (PyMFE complexity + any other numeric scores)."""
    cols: List[str] = []
    for c in df.columns:
        if c in _META_COLUMNS:
            continue
        if c == "error":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _minmax_per_column(mat: np.ndarray) -> np.ndarray:
    """Scale each column to [0, 1] for comparable coloring across metrics."""
    out = np.full_like(mat, np.nan, dtype=float)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        ok = np.isfinite(col)
        if not np.any(ok):
            continue
        lo, hi = np.nanmin(col), np.nanmax(col)
        if hi > lo:
            out[:, j] = (col - lo) / (hi - lo)
        else:
            out[:, j] = 0.5
    return out


def write_complexity_html(summary_csv: Path, out_html: Path, title: str = "Dataset complexity (PyMFE)") -> None:
    """Build interactive Plotly HTML: heatmap (per-column color scale via min-max norm) + meta table."""
    if not _HAS_PLOTLY:
        raise RuntimeError("plotly is required for HTML output. Install: pip install plotly")

    df = pd.read_csv(summary_csv)
    if df.empty:
        raise ValueError(f"Empty or missing summary: {summary_csv}")

    id_col = "dataset_file" if "dataset_file" in df.columns else df.columns[0]
    comp_cols = _complexity_numeric_columns(df)
    if not comp_cols:
        raise ValueError("No numeric complexity columns found in summary CSV.")

    # Drop rows with only errors if complexity missing
    plot_df = df.dropna(subset=comp_cols, how="all").copy()
    if plot_df.empty:
        plot_df = df.copy()

    Z = plot_df[comp_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    Z_norm = _minmax_per_column(Z)

    heat = go.Heatmap(
        z=Z_norm,
        x=comp_cols,
        y=plot_df[id_col].astype(str).tolist(),
        customdata=Z,
        hovertemplate=(
            "Dataset: %{y}<br>"
            "Metric: %{x}<br>"
            "Value: %{customdata:.6g}<br>"
            "(color = min-max normalized within column)"
            "<extra></extra>"
        ),
        colorscale="Viridis",
        colorbar=dict(title="Norm.<br>per column"),
    )

    fig1 = go.Figure(data=[heat])
    fig1.update_layout(
        title=title + " — heatmap (each column scaled 0–1 for color)",
        xaxis_title="Complexity metric",
        yaxis_title="Dataset",
        height=max(400, 28 * len(plot_df)),
        margin=dict(l=120, r=40, t=60, b=120),
        xaxis=dict(tickangle=-45),
    )

    meta_available = [c for c in sorted(_META_COLUMNS) if c in df.columns and c != "error"]
    if meta_available:
        meta_df = df[[id_col] + [c for c in meta_available if c in df.columns]].copy()
        tbl = go.Table(
            header=dict(values=list(meta_df.columns), fill_color="paleturquoise", align="left"),
            cells=dict(
                # Use iloc so duplicate column names in CSV still yield one column each.
                values=[meta_df.iloc[:, i].astype(str).tolist() for i in range(meta_df.shape[1])],
                align="left",
            ),
        )
        final = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.62, 0.38],
            vertical_spacing=0.08,
            subplot_titles=("Complexity metrics (color = per-column min–max)", "Base features"),
            specs=[[{"type": "heatmap"}], [{"type": "table"}]],
        )
        final.add_trace(heat, row=1, col=1)
        final.add_trace(tbl, row=2, col=1)
        final.update_layout(
            title=title,
            height=max(700, 28 * len(plot_df) + 320),
            showlegend=False,
        )
        final.update_xaxes(tickangle=-45, row=1, col=1)
        html_str = final.to_html(include_plotlyjs="cdn", full_html=True)
    else:
        html_str = fig1.to_html(include_plotlyjs="cdn", full_html=True)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_str, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read CSV datasets from a directory (root-level *.csv only), compute "
            "pyMFE complexity metrics and basic dataset descriptors, write CSV "
            "summaries and optionally a Plotly HTML report."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s\n"
            "  %(prog)s --data-dir data --output-csv data/out.csv\n"
            "  %(prog)s --max-rows 1000          # only datasets with <1000 rows\n"
            "  %(prog)s --dataset iris.csv\n"
            "  %(prog)s --html-only              # build HTML from existing --output-csv\n"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing dataset CSVs (root only, no recursive scan).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/datasets_complexity_summary.csv",
        help="Output CSV path for wide summary table.",
    )
    parser.add_argument(
        "--output-long-csv",
        type=str,
        default="data/datasets_complexity_long.csv",
        help="Output CSV path for long-format complexity metrics.",
    )
    parser.add_argument(
        "--config-json",
        type=str,
        default="config.json",
        help="Optional config file containing DATA mapping (dataset -> label column).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Process only one dataset filename (e.g., wine.csv).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index in sorted dataset list (for part-by-part runs).",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Exclusive end index in sorted dataset list (for part-by-part runs).",
    )
    parser.add_argument(
        "--max-rows",
        "--head-rows",
        dest="max_rows",
        type=int,
        default=None,
        metavar="N",
        help=(
            "If set, each dataset uses only the first N rows in file order (after "
            "read_csv) for complexity extraction—no random sampling. Omit to use all "
            "rows. Useful to cap memory/runtime (e.g. --max-rows 1000)."
        ),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output CSVs (useful for part-by-part runs).",
    )
    parser.add_argument(
        "--output-html",
        type=str,
        default="data/datasets_complexity.html",
        help="Output HTML path (Plotly heatmap + base-feature table).",
    )
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Only build HTML from existing --output-csv (no dataset processing).",
    )
    args = parser.parse_args()

    if args.max_rows is not None and args.max_rows < 1:
        raise SystemExit("--max-rows must be >= 1 when set.")

    out_html = Path(args.output_html)

    if args.html_only:
        write_complexity_html(Path(args.output_csv), out_html)
        print(f"Saved HTML: {out_html}")
        return

    data_dir = Path(args.data_dir)
    out_csv = Path(args.output_csv)
    out_long_csv = Path(args.output_long_csv)
    label_map = _load_label_map(Path(args.config_json))

    csv_files = sorted([p for p in data_dir.glob("*.csv") if p.is_file()])
    if not csv_files:
        raise SystemExit(f"No CSV files found in {data_dir}")

    if args.dataset:
        csv_files = [p for p in csv_files if p.name == args.dataset]
        if not csv_files:
            raise SystemExit(f"Dataset not found in {data_dir}: {args.dataset}")

    start_idx = max(0, int(args.start_index))
    end_idx = len(csv_files) if args.end_index is None else int(args.end_index)
    csv_files = csv_files[start_idx:end_idx]
    if not csv_files:
        raise SystemExit("No datasets selected after applying --dataset/--start-index/--end-index filters.")

    rows: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []

    for csv_path in csv_files:
        ds_name = csv_path.name
        print(f"Processing: {ds_name}")
        try:
            df = pd.read_csv(csv_path)
            n_loaded = int(df.shape[0])
            if args.max_rows is not None and n_loaded >= args.max_rows:
                msg = (
                    f"skipped: {n_loaded} rows (only datasets with < {args.max_rows} rows)"
                )
                print(f"  {msg}")
                rows.append({"dataset_file": ds_name, "error": msg})
                continue

            label_col = _pick_label_column(df, ds_name, label_map)

            base = _base_features(df, label_col)
            x, y = _prepare_xy(df, label_col)
            if x.size == 0 or y.size == 0:
                raise ValueError("No usable rows after preprocessing.")

            complexity = _compute_complexity(x, y)

            row = {"dataset_file": ds_name}
            row.update(base)
            row.update(
                {
                    "n_rows_used_after_cleaning": int(x.shape[0]),
                    "n_features_used_after_encoding": int(x.shape[1]),
                }
            )
            row.update(complexity)
            rows.append(row)

            for k, v in complexity.items():
                long_rows.append(
                    {
                        "dataset_file": ds_name,
                        "metric": k,
                        "value": v,
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "dataset_file": ds_name,
                    "error": str(exc),
                }
            )
        finally:
            # Reduce memory pressure between datasets.
            gc.collect()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(rows)
    long_df = pd.DataFrame(long_rows)

    if args.append and out_csv.exists():
        summary_df.to_csv(out_csv, mode="a", header=False, index=False)
    else:
        summary_df.to_csv(out_csv, index=False)

    if args.append and out_long_csv.exists():
        long_df.to_csv(out_long_csv, mode="a", header=False, index=False)
    else:
        long_df.to_csv(out_long_csv, index=False)

    print(f"\nSaved summary: {out_csv}")
    print(f"Saved long metrics: {out_long_csv}")

    # HTML uses the full on-disk summary (important when --append merged batches).
    try:
        write_complexity_html(out_csv, out_html)
        print(f"Saved HTML: {out_html}")
    except Exception as exc:
        print(f"Warning: could not write HTML ({exc}). Install plotly or run: python3 compute_data_complexity.py --html-only")


if __name__ == "__main__":
    main()

