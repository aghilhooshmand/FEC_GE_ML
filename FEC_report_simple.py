from __future__ import annotations

"""
Simple report for baseline_runs_simple.py and FEC_runs_simple.py.

Reads:
  results_simple/<dataset>_Gen_<G>_Pop_<P>/baseline/summary_run*.csv
  results_simple/<dataset>_Gen_<G>_Pop_<P>/FEC/summary_*_run*.csv

Outputs:
  results_simple/<dataset>_Gen_<G>_Pop_<P>/FEC_report_simple.csv
  results_simple/<dataset>_Gen_<G>_Pop_<P>/FEC_report_simple.html

Each row:
  - source: "baseline" or "FEC"
  - mode
  - sample_fraction
  - fake_hit_threshold (NaN for baseline)
  - final_test_mae_mean/std
  - total_time_sec_mean/std
  - total_wall_time_sec_mean/std
  - fake_eval_time_sec_mean/std (FEC only)
  - speedup_total_time     (baseline_total_time_sec_mean / fair_fec_time_sec_mean)
  - speedup_wall_time      (baseline_total_wall_time_sec_mean / fair_fec_wall_time_sec_mean)
  - delta_mae_vs_baseline

Here "fair" FEC time excludes fake-hit evaluation time:
  fair_fec_time_sec_mean      = total_time_sec_mean - fake_eval_time_sec_mean
  fair_fec_wall_time_sec_mean = total_wall_time_sec_mean - fake_eval_time_sec_mean
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


def _load_baseline_simple(baseline_dir: Path) -> pd.DataFrame:
    files = sorted(baseline_dir.glob("summary_run*.csv"))
    if not files:
        raise SystemExit(f"No summary_run*.csv found in {baseline_dir}")
    frames = [pd.read_csv(p) for p in files]
    return pd.concat(frames, ignore_index=True)


def _load_fec_simple(fec_dir: Path) -> pd.DataFrame:
    files = sorted(fec_dir.glob("summary_*_run*.csv"))
    if not files:
        raise SystemExit(f"No summary_*_run*.csv found in {fec_dir}")
    frames = [pd.read_csv(p) for p in files]
    return pd.concat(frames, ignore_index=True)


def _aggregate_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate baseline_simple summaries across runs.
    Baseline has a single configuration: mode='baseline_simple', sample_fraction=1.0.
    """
    if df.empty:
        return pd.DataFrame()

    grouping_keys = ["mode", "sample_fraction"]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    metrics = [c for c in numeric_cols if c not in (["run"] + grouping_keys)]

    if not metrics:
        return df.copy()

    grouped = df.groupby(grouping_keys, as_index=False)[metrics]
    agg_mean = grouped.mean(numeric_only=True)
    agg_std = grouped.std(ddof=0, numeric_only=True)

    mean_cols = {c: f"{c}_mean" for c in metrics}
    std_cols = {c: f"{c}_std" for c in metrics}
    agg_mean = agg_mean.rename(columns=mean_cols)
    agg_std = agg_std.rename(columns=std_cols)

    return pd.merge(agg_mean, agg_std, on=grouping_keys, how="left")


def _aggregate_fec(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate FEC_simple summaries across runs.
    Group by (mode, sample_fraction, fake_hit_threshold).
    """
    if df.empty:
        return pd.DataFrame()

    grouping_keys = ["mode", "sample_fraction", "fake_hit_threshold"]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    metrics = [c for c in numeric_cols if c not in (["run"] + grouping_keys)]

    if not metrics:
        return df.copy()

    grouped = df.groupby(grouping_keys, as_index=False)[metrics]
    agg_mean = grouped.mean(numeric_only=True)
    agg_std = grouped.std(ddof=0, numeric_only=True)

    mean_cols = {c: f"{c}_mean" for c in metrics}
    std_cols = {c: f"{c}_std" for c in metrics}
    agg_mean = agg_mean.rename(columns=mean_cols)
    agg_std = agg_std.rename(columns=std_cols)

    return pd.merge(agg_mean, agg_std, on=grouping_keys, how="left")


def _build_simple_comparison(
    base_agg: pd.DataFrame,
    fec_agg: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a simple comparison table (one baseline row + one row per FEC config).
    Uses 'fair' FEC times that subtract fake_eval_time_sec_mean.
    """
    if base_agg.empty or fec_agg.empty:
        return pd.DataFrame()

    base_row = base_agg.iloc[0]

    baseline_mae = float(base_row.get("final_test_mae_mean", np.nan))
    baseline_time = float(base_row.get("total_time_sec_mean", np.nan))
    baseline_wall = float(base_row.get("total_wall_time_sec_mean", np.nan))

    rows: List[dict] = []

    # Baseline row
    rows.append(
        {
            "source": "baseline",
            "mode": str(base_row.get("mode", "baseline_simple")),
            "sample_fraction": float(base_row.get("sample_fraction", 1.0)),
            "fake_hit_threshold": np.nan,
            "final_test_mae_mean": baseline_mae,
            "final_test_mae_std": float(base_row.get("final_test_mae_std", np.nan)),
            "total_time_sec_mean": baseline_time,
            "total_time_sec_std": float(base_row.get("total_time_sec_std", np.nan)),
            "total_wall_time_sec_mean": baseline_wall,
            "total_wall_time_sec_std": float(
                base_row.get("total_wall_time_sec_std", np.nan)
            ),
            "fake_eval_time_sec_mean": np.nan,
            "fake_eval_time_sec_std": np.nan,
            "speedup_total_time": np.nan,
            "speedup_wall_time": np.nan,
            "delta_mae_vs_baseline": 0.0,
        }
    )

    # FEC rows
    for _, row in fec_agg.iterrows():
        mae = float(row.get("final_test_mae_mean", np.nan))
        time_sec = float(row.get("total_time_sec_mean", np.nan))
        wall_sec = float(row.get("total_wall_time_sec_mean", np.nan))
        fake_time = float(row.get("fake_eval_time_sec_mean", 0.0))

        fair_time = (
            time_sec - fake_time
            if np.isfinite(time_sec) and np.isfinite(fake_time)
            else np.nan
        )
        fair_wall = (
            wall_sec - fake_time
            if np.isfinite(wall_sec) and np.isfinite(fake_time)
            else np.nan
        )

        if np.isfinite(baseline_time) and np.isfinite(fair_time) and fair_time > 0:
            speedup_total = baseline_time / fair_time
        else:
            speedup_total = np.nan

        if np.isfinite(baseline_wall) and np.isfinite(fair_wall) and fair_wall > 0:
            speedup_wall = baseline_wall / fair_wall
        else:
            speedup_wall = np.nan
        delta_mae = (
            mae - baseline_mae
            if np.isfinite(mae) and np.isfinite(baseline_mae)
            else np.nan
        )

        rows.append(
            {
                "source": "FEC",
                "mode": row.get("mode"),
                "sample_fraction": float(row.get("sample_fraction", np.nan)),
                "fake_hit_threshold": float(row.get("fake_hit_threshold", np.nan)),
                "final_test_mae_mean": mae,
                "final_test_mae_std": float(row.get("final_test_mae_std", np.nan)),
                "total_time_sec_mean": time_sec,
                "total_time_sec_std": float(row.get("total_time_sec_std", np.nan)),
                "total_wall_time_sec_mean": wall_sec,
                "total_wall_time_sec_std": float(
                    row.get("total_wall_time_sec_std", np.nan)
                ),
                "fake_eval_time_sec_mean": fake_time,
                "fake_eval_time_sec_std": float(
                    row.get("fake_eval_time_sec_std", np.nan)
                ),
                "speedup_total_time": speedup_total,
                "speedup_wall_time": speedup_wall,
                "delta_mae_vs_baseline": delta_mae,
            }
        )

    return pd.DataFrame.from_records(rows)


def _build_speedup_heatmap(df: pd.DataFrame) -> str:
    """Heatmap: x = threshold, y = fraction, colour = speedup_total_time."""
    fec = df[df["source"] == "FEC"].copy()
    if fec.empty:
        return "<p>No FEC rows for speedup heatmap.</p>"

    fractions = sorted(fec["sample_fraction"].dropna().unique().tolist())
    thresholds = sorted(fec["fake_hit_threshold"].dropna().unique().tolist())

    if not fractions or not thresholds:
        return "<p>No fractions or thresholds found for speedup heatmap.</p>"

    th_labels = [str(t) for t in thresholds]
    frac_labels = [f"{f:.0%}" for f in fractions]

    z = []
    text = []
    for frac in fractions:
        row_vals = []
        row_txt = []
        for th in thresholds:
            match = fec[
                (np.isclose(fec["sample_fraction"], frac))
                & (np.isclose(fec["fake_hit_threshold"], th))
            ]
            if not match.empty and "speedup_total_time" in match.columns:
                val = float(match["speedup_total_time"].iloc[0])
                row_vals.append(val if np.isfinite(val) else np.nan)
                row_txt.append(f"{val:.2f}" if np.isfinite(val) else "")
            else:
                row_vals.append(np.nan)
                row_txt.append("")
        z.append(row_vals)
        text.append(row_txt)

    finite_vals = [v for row in z for v in row if np.isfinite(v)]
    if not finite_vals:
        return "<p>No finite speedup values for heatmap.</p>"

    vmin = min(finite_vals)
    vmax = max(finite_vals)
    if vmax == vmin:
        vmin = max(0.0, vmin - 0.1)
        vmax = vmax + 0.1
    else:
        pad = 0.05 * (vmax - vmin)
        vmin = max(0.0, vmin - pad)
        vmax = vmax + pad

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=th_labels,
            y=frac_labels,
            text=text,
            texttemplate="%{text}",
            colorscale="Viridis",
            colorbar=dict(title="Speedup (baseline_time / FEC_time_fair)"),
            zmin=vmin,
            zmax=vmax,
        )
    )
    fig.update_layout(
        title=(
            "Speedup heatmap (simple pipeline)<br>"
            "x = fake-hit threshold, y = sample fraction"
        ),
        xaxis_title="Fake-hit threshold",
        yaxis_title="Sample fraction",
        template="plotly_white",
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def _build_speedup_lines(df: pd.DataFrame) -> str:
    """Speedup vs sample fraction, one line per threshold."""
    fec = df[df["source"] == "FEC"].copy()
    if fec.empty:
        return "<p>No FEC rows for speedup lines.</p>"

    fractions = sorted(fec["sample_fraction"].dropna().unique().tolist())
    thresholds = sorted(fec["fake_hit_threshold"].dropna().unique().tolist())

    fig = go.Figure()
    for th in thresholds:
        y_vals = []
        for frac in fractions:
            match = fec[
                (np.isclose(fec["sample_fraction"], frac))
                & (np.isclose(fec["fake_hit_threshold"], th))
            ]
            if not match.empty and "speedup_total_time" in match.columns:
                val = float(match["speedup_total_time"].iloc[0])
                y_vals.append(val if np.isfinite(val) else np.nan)
            else:
                y_vals.append(np.nan)
        if any(np.isfinite(v) for v in y_vals):
            fig.add_trace(
                go.Scatter(
                    x=[f"{f:.0%}" for f in fractions],
                    y=y_vals,
                    mode="lines+markers",
                    name=f"th={th}",
                )
            )

    if not fig.data:
        return "<p>No finite speedup values for lines.</p>"

    fig.update_layout(
        title=(
            "Speedup vs sample fraction (simple pipeline)<br>"
            "speedup = baseline_time_sec / FEC_time_sec_fair"
        ),
        xaxis_title="Sample fraction",
        yaxis_title="Speedup (>1 = FEC faster)",
        template="plotly_white",
        hovermode="x unified",
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Simple baseline vs FEC comparison using baseline_runs_simple.py and "
            "FEC_runs_simple.py outputs under results_simple/."
        )
    )
    parser.add_argument(
        "experiment_root",
        type=str,
        help=(
            "Path to results_simple/<dataset>_Gen_<G>_Pop_<P>/ "
            "containing 'baseline' and 'FEC' subdirectories."
        ),
    )
    args = parser.parse_args()

    root = Path(args.experiment_root)
    baseline_dir = root / "baseline"
    fec_dir = root / "FEC"

    if not baseline_dir.is_dir() or not fec_dir.is_dir():
        raise SystemExit(
            f"Expected 'baseline' and 'FEC' subdirectories under {root}, "
            "run baseline_runs_simple.py and FEC_runs_simple.py first."
        )

    base_all = _load_baseline_simple(baseline_dir)
    fec_all = _load_fec_simple(fec_dir)

    base_agg = _aggregate_baseline(base_all)
    fec_agg = _aggregate_fec(fec_all)

    comparison_df = _build_simple_comparison(base_agg, fec_agg)
    if comparison_df.empty:
        print("No comparison rows produced (missing baseline or FEC data).")
        return

    csv_path = root / "FEC_report_simple.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"Saved simple baseline vs FEC comparison to {csv_path}")

    # Build HTML report (summary table + plots)
    summary_html = comparison_df.to_html(
        index=False, float_format=lambda x: f"{x:.5g}"
    )
    heatmap_html = _build_speedup_heatmap(comparison_df)
    lines_html = _build_speedup_lines(comparison_df)

    html = (
        "<html><head>"
        "<meta charset='utf-8' />"
        "<title>Baseline vs FEC (Simple Pipeline Report)</title>"
        "<script src='https://cdn.plot.ly/plotly-2.27.0.min.js'></script>"
        "</head><body>"
        "<h1>Baseline vs FEC (Simple Pipeline Report)</h1>"
        "<h2>Summary table</h2>"
        f"{summary_html}"
        "<h2>Speedup heatmap</h2>"
        f"{heatmap_html}"
        "<h2>Speedup vs sample fraction</h2>"
        f"{lines_html}"
        "</body></html>"
    )

    html_path = root / "FEC_report_simple.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"Saved simple HTML report to {html_path}")


if __name__ == "__main__":
    main()

