from __future__ import annotations

"""
Aggregate results across runs for a single *folder*.

This script supports two kinds of folders:

1) **Baseline folder** (no FEC):

       results/<dataset>_Gen_<G>_Pop_<P>/baseline/
           generation_stats_run<run>.csv
           summary_run<run>.csv

   It produces, in the same folder:

       generation_stats_aggregated.csv
       summary_aggregated.csv

   Both are averages (and std) across runs.

2) **FEC folder** for one experiment:

       results/<dataset>_Gen_<G>_Pop_<P>/FEC/
           generation_stats_<method>_frac_<p>_run<run>.csv
           summary_<method>_frac_<p>_run<run>.csv

   It produces, in the same folder:

       generation_stats_aggregated_FEC.csv
       summary_aggregated_FEC.csv

   Here, aggregation is **separate for each (sampling method, fraction)**:
   - Per-generation aggregation groups by (mode, sample_fraction, gen).
   - Per-summary aggregation groups by (mode, sample_fraction),
     so each row is one (sampling method, fraction).

Usage:
    # Baseline only
    python FEC_report.py results/<dataset>_Gen_<G>_Pop_<P>/baseline

    # FEC only
    python FEC_report.py results/<dataset>_Gen_<G>_Pop_<P>/FEC

    # Full comparison (baseline vs FEC) and HTML report
    python FEC_report.py results/<dataset>_Gen_<G>_Pop_<P>/
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from config import CONFIG


def _load_baseline_csvs(baseline_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all baseline per-run CSVs from a single folder."""
    gen_files = sorted(baseline_dir.glob("generation_stats_run*.csv"))
    summary_files = sorted(baseline_dir.glob("summary_run*.csv"))

    if not gen_files:
        raise SystemExit(
            f"No generation_stats_run*.csv files found under {baseline_dir}. "
            "Run baseline_runs.py (or run_baseline_batch.sh) first."
        )
    if not summary_files:
        raise SystemExit(
            f"No summary_run*.csv files found under {baseline_dir}. "
            "Run baseline_runs.py (or run_baseline_batch.sh) first."
        )

    gen_frames = [pd.read_csv(p) for p in gen_files]
    summary_frames = [pd.read_csv(p) for p in summary_files]

    gen_all = pd.concat(gen_frames, ignore_index=True)
    summary_all = pd.concat(summary_frames, ignore_index=True)
    return gen_all, summary_all


def _aggregate_generation_stats(gen_all: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-generation metrics across baseline runs.

    Group by (to keep columns consistent with FEC):
        - mode
        - sample_fraction
        - gen

    For all numeric columns (except 'run'), compute mean and std;
    column names become <metric>_mean and <metric>_std.
    """
    grouping_keys = ["mode", "sample_fraction", "gen"]
    numeric_cols = gen_all.select_dtypes(include="number").columns.tolist()

    exclude = set(["run"] + grouping_keys)
    metrics = [c for c in numeric_cols if c not in exclude]

    if not metrics:
        return gen_all.copy()

    grouped = gen_all.groupby(grouping_keys, as_index=False)[metrics]
    agg_mean = grouped.mean(numeric_only=True)
    agg_std = grouped.std(ddof=0, numeric_only=True)

    mean_cols = {c: f"{c}_mean" for c in metrics}
    std_cols = {c: f"{c}_std" for c in metrics}
    agg_mean = agg_mean.rename(columns=mean_cols)
    agg_std = agg_std.rename(columns=std_cols)

    agg = pd.merge(agg_mean, agg_std, on=grouping_keys, how="left")
    return agg


def _aggregate_summary_stats(summary_all: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-run summary metrics across baseline runs.

    Group by (to keep columns consistent with FEC):
        - mode
        - sample_fraction

    For all numeric columns (except 'run'), compute mean and std; column names
    become <metric>_mean and <metric>_std. Each row is one (mode, fraction),
    which for baseline will just be ('baseline', 1.0).
    """
    grouping_keys = ["mode", "sample_fraction"]
    numeric_cols = summary_all.select_dtypes(include="number").columns.tolist()
    metrics = [c for c in numeric_cols if c not in (["run"] + grouping_keys)]

    if not metrics:
        return summary_all.copy()

    grouped = summary_all.groupby(grouping_keys, as_index=False)[metrics]
    agg_mean = grouped.mean(numeric_only=True)
    agg_std = grouped.std(ddof=0, numeric_only=True)

    mean_cols = {c: f"{c}_mean" for c in metrics}
    std_cols = {c: f"{c}_std" for c in metrics}
    agg_mean = agg_mean.rename(columns=mean_cols)
    agg_std = agg_std.rename(columns=std_cols)

    agg = pd.merge(agg_mean, agg_std, on=grouping_keys, how="left")
    return agg


def _load_fec_csvs(fec_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all FEC per-run CSVs from a single folder."""
    gen_files = sorted(fec_dir.glob("generation_stats_*_run*.csv"))
    summary_files = sorted(fec_dir.glob("summary_*_run*.csv"))

    if not gen_files:
        raise SystemExit(
            f"No generation_stats_*_run*.csv files found under {fec_dir}. "
            "Run FEC_runs.py (or run_FEC_batch.sh) first."
        )
    if not summary_files:
        raise SystemExit(
            f"No summary_*_run*.csv files found under {fec_dir}. "
            "Run FEC_runs.py (or run_FEC_batch.sh) first."
        )

    def _parse_threshold_from_fec_filename(path: Path) -> float | None:
        """Extract fake_hit_threshold from filename ..._th_<tag>_run<n>.csv . Returns None if no th_ segment."""
        stem = path.stem
        if "_th_" not in stem or "_run" not in stem:
            return None
        after_th = stem.split("_th_", 1)[1]
        tag = after_th.split("_run", 1)[0]
        if not tag:
            return None
        # Convert tag to float: 0p1 -> 0.1, 1em05 -> 1e-5
        s = tag.replace("p", ".", 1).replace("m", "e-", 1)
        try:
            return float(s)
        except ValueError:
            return None

    gen_frames = []
    for p in gen_files:
        df = pd.read_csv(p)
        th = _parse_threshold_from_fec_filename(p)
        df["fake_hit_threshold"] = th if th is not None else np.nan
        gen_frames.append(df)
    summary_frames = []
    for p in summary_files:
        df = pd.read_csv(p)
        th = _parse_threshold_from_fec_filename(p)
        df["fake_hit_threshold"] = th if th is not None else np.nan
        summary_frames.append(df)

    gen_all = pd.concat(gen_frames, ignore_index=True)
    summary_all = pd.concat(summary_frames, ignore_index=True)
    return gen_all, summary_all


def _aggregate_fec_generation_stats(gen_all: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-generation metrics across FEC runs.

    Group by:
        - mode               (e.g. 'fec_kmeans')
        - sample_fraction    (e.g. 0.1, 0.2, ...)
        - fake_hit_threshold (e.g. 0, 1e-3, ...)
        - gen

    For all numeric columns (except 'run'), compute mean and std;
    column names become <metric>_mean and <metric>_std.
    """
    grouping_keys = ["mode", "sample_fraction", "fake_hit_threshold", "gen"]
    numeric_cols = gen_all.select_dtypes(include="number").columns.tolist()

    exclude = set(["run"] + grouping_keys)
    metrics = [c for c in numeric_cols if c not in exclude]

    if not metrics:
        return gen_all.copy()

    grouped = gen_all.groupby(grouping_keys, as_index=False)[metrics]
    agg_mean = grouped.mean(numeric_only=True)
    agg_std = grouped.std(ddof=0, numeric_only=True)

    mean_cols = {c: f"{c}_mean" for c in metrics}
    std_cols = {c: f"{c}_std" for c in metrics}
    agg_mean = agg_mean.rename(columns=mean_cols)
    agg_std = agg_std.rename(columns=std_cols)

    agg = pd.merge(agg_mean, agg_std, on=grouping_keys, how="left")
    return agg


def _aggregate_fec_summary_stats(summary_all: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-run summary metrics across FEC runs.

    Group by:
        - mode
        - sample_fraction
        - fake_hit_threshold

    For all numeric columns (except 'run'), compute mean and std;
    column names become <metric>_mean and <metric>_std.

    Result: each row is one (sampling method, fraction).
    """
    grouping_keys = ["mode", "sample_fraction", "fake_hit_threshold"]
    numeric_cols = summary_all.select_dtypes(include="number").columns.tolist()

    exclude = set(["run"] + grouping_keys)
    metrics = [c for c in numeric_cols if c not in exclude]

    if not metrics:
        return summary_all.copy()

    grouped = summary_all.groupby(grouping_keys, as_index=False)[metrics]
    agg_mean = grouped.mean(numeric_only=True)
    agg_std = grouped.std(ddof=0, numeric_only=True)

    mean_cols = {c: f"{c}_mean" for c in metrics}
    std_cols = {c: f"{c}_std" for c in metrics}
    agg_mean = agg_mean.rename(columns=mean_cols)
    agg_std = agg_std.rename(columns=std_cols)

    agg = pd.merge(agg_mean, agg_std, on=grouping_keys, how="left")
    return agg


# ---------------------------------------------------------------------------
# HTML report helpers (baseline vs FEC, using aggregated CSVs)
# ---------------------------------------------------------------------------


def _format_method_display(method: str) -> str:
    """Convert mode identifier to display name, e.g. fec_kmeans -> k-means, fec_farthest_point -> farthest_point."""
    s = method
    if s.startswith("fec_"):
        s = s[4:]
    if s == "kmeans":
        return "k-means"
    return s.replace("_", "-")


def _format_legend_fec(method: str, frac: float) -> str:
    """Legend label for FEC series: e.g. 'farthest_point (10%)'."""
    return f"{_format_method_display(method)} ({frac:.0%})"


def _format_threshold_label(threshold: float) -> str:
    """Display label for fake_hit_threshold, e.g. 0.1 -> '0.1', 1e-5 -> '1e-5'."""
    if np.isnan(threshold):
        return "N/A"
    if threshold >= 0.01 or threshold == 0:
        return str(threshold)
    return f"{threshold:.0e}"


def _legend_method_plus_threshold(method: str, threshold: float) -> str:
    """Legend for Across fractions: e.g. 'farthest-point (th 0.1)'."""
    return f"{_format_method_display(method)} (th {_format_threshold_label(threshold)})"


def _legend_method_plus_fraction(method: str, fraction: float) -> str:
    """Legend for Across thresholds: e.g. 'farthest-point (10%)'."""
    return f"{_format_method_display(method)} ({fraction:.0%})"


def _scale_fake_hit_rate_yaxis(fig: go.Figure) -> None:
    """
    When fake-hit rate values are very small, fix y-axis range and tick format
    so points are visible (not stuck at 0). Call after adding traces.
    """
    if not fig.data:
        return
    all_y: List[float] = []
    for trace in fig.data:
        y = getattr(trace, "y", None)
        if y is not None:
            for v in np.atleast_1d(y).flatten():
                if v is not None and np.isfinite(v):
                    all_y.append(float(v))
    if not all_y:
        return
    max_y = max(all_y)
    if max_y <= 0:
        fig.update_layout(yaxis_range=[0, 0.01], yaxis_tickformat=".2%")
        return
    if max_y < 0.1:
        top = max(0.01, max_y * 1.15)
        fig.update_layout(yaxis_range=[0, top], yaxis_tickformat=".2%")


def _get_thresholds_and_fractions(fec_gen_agg: pd.DataFrame) -> tuple[List[float], dict]:
    """
    Return (sorted list of unique thresholds, mapping threshold -> sorted list of fractions).
    NaN threshold is placed last.
    """
    if "fake_hit_threshold" not in fec_gen_agg.columns:
        th_series = fec_gen_agg.get("sample_fraction")  # fallback: no threshold
        if th_series is not None:
            thresholds = [np.nan]
            fractions = sorted(
                float(f)
                for f in fec_gen_agg["sample_fraction"].dropna().unique().tolist()
            )
            return thresholds, {np.nan: fractions}
        return [], {}

    th_vals = fec_gen_agg["fake_hit_threshold"].dropna().unique().tolist()
    nan_has_data = fec_gen_agg["fake_hit_threshold"].isna().any()
    thresholds = sorted(
        (float(t) for t in th_vals if np.isfinite(t)),
        key=lambda x: (x, 0),
    )
    if nan_has_data:
        thresholds.append(np.nan)

    frac_per_th: dict = {}
    for th in thresholds:
        if np.isnan(th):
            mask = fec_gen_agg["fake_hit_threshold"].isna()
        else:
            mask = np.isclose(fec_gen_agg["fake_hit_threshold"].astype(float), th)
        sub = fec_gen_agg.loc[mask, "sample_fraction"].dropna().unique().tolist()
        frac_per_th[th] = sorted(float(f) for f in sub)
    return thresholds, frac_per_th


def _filter_fec_by_threshold_and_fraction(
    fec_gen_agg: pd.DataFrame,
    threshold: float,
    fraction: float,
) -> pd.DataFrame:
    """Filter FEC aggregated gen stats to one (threshold, fraction)."""
    mask_frac = np.isclose(fec_gen_agg["sample_fraction"].astype(float), fraction)
    if "fake_hit_threshold" not in fec_gen_agg.columns:
        return fec_gen_agg.loc[mask_frac].copy()
    if np.isnan(threshold):
        mask_th = fec_gen_agg["fake_hit_threshold"].isna()
    else:
        mask_th = np.isclose(
            fec_gen_agg["fake_hit_threshold"].astype(float), threshold
        )
    return fec_gen_agg.loc[mask_th & mask_frac].copy()


def _build_training_and_test_figs(
    base_gen_agg: pd.DataFrame,
    fec_gen_agg: pd.DataFrame,
    threshold: float,
    fraction: float,
    methods: List[str],
) -> List[str]:
    """Training / test MAE (and optional best) for one (threshold, fraction). Returns list of HTML fragments (no headers)."""
    frac_rows = _filter_fec_by_threshold_and_fraction(
        fec_gen_agg, threshold, fraction
    )
    if frac_rows.empty:
        return []

    blocks: List[str] = []

    # Training MAE
    fig_train = go.Figure()
    base_rows = base_gen_agg.sort_values("gen")
    if not base_rows.empty and "avg_mean" in base_rows.columns:
        fig_train.add_trace(
            go.Scatter(
                x=base_rows["gen"],
                y=base_rows["avg_mean"],
                mode="lines+markers",
                name="Baseline",
                error_y=dict(
                    type="data",
                    array=base_rows.get("avg_std", None),
                    visible=bool("avg_std" in base_rows.columns),
                ),
            )
        )

    for method in methods:
        rows = frac_rows[frac_rows["mode"] == method].sort_values("gen")
        if rows.empty or "avg_mean" not in rows.columns:
            continue
        fig_train.add_trace(
            go.Scatter(
                x=rows["gen"],
                y=rows["avg_mean"],
                mode="lines+markers",
                name=_format_legend_fec(method, fraction),
                error_y=dict(
                    type="data",
                    array=rows.get("avg_std", None),
                    visible=bool("avg_std" in rows.columns),
                ),
            )
        )

    if fig_train.data:
        fig_train.update_layout(
            title=f"Training MAE — Fraction {fraction:.0%}",
            xaxis_title="Generation",
            yaxis_title="Training MAE (lower is better)",
            template="plotly_white",
            hovermode="x unified",
        )
        blocks.append(
            pio.to_html(fig_train, include_plotlyjs="cdn", full_html=False)
        )

    # Test MAE (average)
    fig_test = go.Figure()
    if not base_rows.empty and "fitness_test_mean" in base_rows.columns:
        fig_test.add_trace(
            go.Scatter(
                x=base_rows["gen"],
                y=base_rows["fitness_test_mean"],
                mode="lines+markers",
                name="Baseline",
                error_y=dict(
                    type="data",
                    array=base_rows.get("fitness_test_std", None),
                    visible=bool("fitness_test_std" in base_rows.columns),
                ),
            )
        )

    for method in methods:
        rows = frac_rows[frac_rows["mode"] == method].sort_values("gen")
        if rows.empty or "fitness_test_mean" not in rows.columns:
            continue
        fig_test.add_trace(
            go.Scatter(
                x=rows["gen"],
                y=rows["fitness_test_mean"],
                mode="lines+markers",
                name=_format_legend_fec(method, fraction),
                error_y=dict(
                    type="data",
                    array=rows.get("fitness_test_std", None),
                    visible=bool("fitness_test_std" in rows.columns),
                ),
            )
        )

    if fig_test.data:
        fig_test.update_layout(
            title=f"Test MAE — Fraction {fraction:.0%}",
            xaxis_title="Generation",
            yaxis_title="Test MAE (lower is better)",
            template="plotly_white",
            hovermode="x unified",
        )
        blocks.append(
            pio.to_html(fig_test, include_plotlyjs="cdn", full_html=False)
        )

    # Optional: best test across runs over generations, if *_min_mean exists
    if (
        "fitness_test_min_mean" in base_gen_agg.columns
        or "fitness_test_min_mean" in fec_gen_agg.columns
    ):
        fig_best = go.Figure()
        if "fitness_test_min_mean" in base_gen_agg.columns:
            fig_best.add_trace(
                go.Scatter(
                    x=base_rows["gen"],
                    y=base_rows["fitness_test_min_mean"],
                    mode="lines+markers",
                    name="Baseline",
                )
            )
        for method in methods:
            rows = frac_rows[frac_rows["mode"] == method].sort_values("gen")
            if rows.empty or "fitness_test_min_mean" not in rows.columns:
                continue
            fig_best.add_trace(
                go.Scatter(
                    x=rows["gen"],
                    y=rows["fitness_test_min_mean"],
                    mode="lines+markers",
                    name=_format_legend_fec(method, fraction),
                )
            )
        if fig_best.data:
            fig_best.update_layout(
                title=f"Best test MAE — Fraction {fraction:.0%}",
                xaxis_title="Generation",
                yaxis_title="Best test MAE (lower is better)",
                template="plotly_white",
                hovermode="x unified",
            )
            blocks.append(
                pio.to_html(fig_best, include_plotlyjs="cdn", full_html=False)
            )

    return blocks


def _build_fec_hit_and_fake_figs(
    fec_gen_agg: pd.DataFrame,
    threshold: float,
    fraction: float,
    methods: List[str],
) -> List[str]:
    """Hit / fake-hit rates for one (threshold, fraction). Returns list of HTML fragments (no headers)."""
    frac_rows = _filter_fec_by_threshold_and_fraction(
        fec_gen_agg, threshold, fraction
    )
    if frac_rows.empty:
        return []

    blocks: List[str] = []
    fig_hit = go.Figure()
    fig_fake_rate = go.Figure()

    for method in methods:
        rows = frac_rows[frac_rows["mode"] == method].sort_values("gen")
        if rows.empty:
            continue

        hits = rows.get("cache_hits_mean")
        misses = rows.get("cache_misses_mean")
        hs = rows.get("hits_just_structural_mean")
        hb = rows.get("hits_behavioural_without_structural_mean")
        fake_hits = rows.get("cache_fake_hits_mean")
        label = _format_legend_fec(method, fraction)

        if hits is not None and misses is not None:
            total = hits + misses
            rate_all = hits / total.replace(0, np.nan)
            if CONFIG.get("fec.modes.fec_enabled_total", True):
                fig_hit.add_trace(
                    go.Scatter(
                        x=rows["gen"],
                        y=rate_all,
                        mode="lines+markers",
                        name=f"{label} (total)",
                    )
                )
        if hs is not None and hits is not None:
            rate_struct = hs / hits.replace(0, np.nan)
            if CONFIG.get("fec.modes.fec_enabled_structural_only", True):
                fig_hit.add_trace(
                    go.Scatter(
                        x=rows["gen"],
                        y=rate_struct,
                        mode="lines+markers",
                        name=f"{label} (structural)",
                    )
                )
        if hb is not None and hits is not None:
            rate_behav = hb / hits.replace(0, np.nan)
            if CONFIG.get(
                "fec.modes.fec_enabled_behaviour_without_structural", True
            ):
                fig_hit.add_trace(
                    go.Scatter(
                        x=rows["gen"],
                        y=rate_behav,
                        mode="lines+markers",
                        name=f"{label} (behavioural-only)",
                    )
                )
        if fake_hits is not None and hits is not None:
            safe_hits = hits.replace(0, np.nan)
            fake_rate = fake_hits / safe_hits
            fig_fake_rate.add_trace(
                go.Scatter(
                    x=rows["gen"],
                    y=fake_rate,
                    mode="lines+markers",
                    name=label,
                )
            )

    if fig_hit.data:
        fig_hit.update_layout(
            title=f"Hit rate — Fraction {fraction:.0%}",
            xaxis_title="Generation",
            yaxis_title="Hit rate",
            yaxis_tickformat=".0%",
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
        )
        blocks.append(
            pio.to_html(fig_hit, include_plotlyjs="cdn", full_html=False)
        )
    if fig_fake_rate.data:
        fig_fake_rate.update_layout(
            title=f"Fake-hit rate — Fraction {fraction:.0%}",
            xaxis_title="Generation",
            yaxis_title="Fake-hit rate (fake / hits)",
            yaxis_tickformat=".0%",
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
        )
        _scale_fake_hit_rate_yaxis(fig_fake_rate)
        blocks.append(
            pio.to_html(fig_fake_rate, include_plotlyjs="cdn", full_html=False)
        )

    return blocks


def _build_cross_fraction_figs(
    fec_summary_agg: pd.DataFrame,
    base_summary_agg: pd.DataFrame,
    fractions: List[float],
    methods: List[str],
) -> List[str]:
    """Cross-fraction plots: final test MAE, hit-rate, fake-hit-rate, runtime. Legend = method + threshold."""
    sections: List[str] = []
    thresholds = (
        _get_sorted_thresholds_from_summary(fec_summary_agg)
        if "fake_hit_threshold" in fec_summary_agg.columns
        else [np.nan]
    )
    if not thresholds:
        thresholds = [np.nan]

    def _match(method: str, frac: float, th: float):
        mask_frac = np.isclose(fec_summary_agg["sample_fraction"].astype(float), frac)
        if "fake_hit_threshold" not in fec_summary_agg.columns:
            mask_th = np.ones(len(fec_summary_agg), dtype=bool)
        elif np.isnan(th):
            mask_th = fec_summary_agg["fake_hit_threshold"].isna()
        else:
            mask_th = np.isclose(
                fec_summary_agg["fake_hit_threshold"].astype(float), th
            )
        return fec_summary_agg[
            (fec_summary_agg["mode"] == method) & mask_frac & mask_th
        ]

    # Final test MAE vs sample fraction — Baseline + (method, threshold)
    fig_final = go.Figure()
    if (
        not base_summary_agg.empty
        and "final_test_mae_mean" in base_summary_agg.columns
    ):
        baseline_val = float(base_summary_agg["final_test_mae_mean"].iloc[0])
        fig_final.add_trace(
            go.Scatter(
                x=fractions,
                y=[baseline_val] * len(fractions),
                mode="lines",
                name="Baseline",
            )
        )
    for method in methods:
        for th in thresholds:
            y_vals = []
            for f in fractions:
                match = _match(method, f, th)
                if not match.empty and "final_test_mae_mean" in match.columns:
                    y_vals.append(float(match["final_test_mae_mean"].iloc[0]))
                else:
                    y_vals.append(np.nan)
            if any(np.isfinite(y) for y in y_vals):
                fig_final.add_trace(
                    go.Scatter(
                        x=fractions,
                        y=y_vals,
                        mode="lines+markers",
                        name=_legend_method_plus_threshold(method, th),
                    )
                )
    if fig_final.data:
        fig_final.update_layout(
            title="Final test MAE across sample fractions (baseline vs FEC)",
            xaxis_title="Sample fraction",
            yaxis_title="Final test MAE (lower is better)",
            template="plotly_white",
            hovermode="x unified",
        )
        sections.append(
            pio.to_html(fig_final, include_plotlyjs="cdn", full_html=False)
        )

    # Overall hit rate vs sample fraction — (method, threshold)
    fig_hit_overall = go.Figure()
    for method in methods:
        for th in thresholds:
            y_vals = []
            for f in fractions:
                match = _match(method, f, th)
                if not match.empty and "hit_rate_overall_mean" in match.columns:
                    y_vals.append(float(match["hit_rate_overall_mean"].iloc[0]))
                else:
                    y_vals.append(np.nan)
            if any(np.isfinite(y) for y in y_vals):
                fig_hit_overall.add_trace(
                    go.Scatter(
                        x=fractions,
                        y=y_vals,
                        mode="lines+markers",
                        name=_legend_method_plus_threshold(method, th),
                    )
                )
    if fig_hit_overall.data:
        fig_hit_overall.update_layout(
            title="Overall hit rate across sample fractions (FEC only)",
            xaxis_title="Sample fraction",
            yaxis_title="Hit rate",
            yaxis_tickformat=".0%",
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
        )
        sections.append(
            pio.to_html(fig_hit_overall, include_plotlyjs="cdn", full_html=False)
        )

    # Overall fake-hit rate vs sample fraction — (method, threshold)
    fig_fake_overall = go.Figure()
    for method in methods:
        for th in thresholds:
            y_vals = []
            for f in fractions:
                match = _match(method, f, th)
                if not match.empty and "fake_hit_rate_overall_mean" in match.columns:
                    y_vals.append(float(match["fake_hit_rate_overall_mean"].iloc[0]))
                else:
                    y_vals.append(np.nan)
            if any(np.isfinite(y) for y in y_vals):
                fig_fake_overall.add_trace(
                    go.Scatter(
                        x=fractions,
                        y=y_vals,
                        mode="lines+markers",
                        name=_legend_method_plus_threshold(method, th),
                    )
                )
    if fig_fake_overall.data:
        fig_fake_overall.update_layout(
            title="Overall fake-hit rate across sample fractions (FEC only)",
            xaxis_title="Sample fraction",
            yaxis_title="Fake-hit rate",
            yaxis_tickformat=".0%",
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
        )
        _scale_fake_hit_rate_yaxis(fig_fake_overall)
        sections.append(
            pio.to_html(fig_fake_overall, include_plotlyjs="cdn", full_html=False)
        )

    # Cache hit rate vs sample fraction — (method, threshold)
    fig_speedup = go.Figure()
    for method in methods:
        for th in thresholds:
            y_vals = []
            for f in fractions:
                match = _match(method, f, th)
                if not match.empty and "hit_rate_overall_mean" in match.columns:
                    y_vals.append(float(match["hit_rate_overall_mean"].iloc[0]))
                else:
                    y_vals.append(np.nan)
            if any(np.isfinite(y) for y in y_vals):
                fig_speedup.add_trace(
                    go.Scatter(
                        x=fractions,
                        y=y_vals,
                        mode="lines+markers",
                        name=_legend_method_plus_threshold(method, th),
                    )
                )
    if fig_speedup.data:
        fig_speedup.update_layout(
            title="Overall cache hit rate across sample fractions (FEC only)",
            xaxis_title="Sample fraction",
            yaxis_title="Hit rate",
            yaxis_tickformat=".0%",
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
        )
        sections.append(
            pio.to_html(fig_speedup, include_plotlyjs="cdn", full_html=False)
        )

    # Runtime ratio vs sample fraction — Baseline + (method, threshold)
    fig_runtime = go.Figure()
    baseline_time = None
    if (
        not base_summary_agg.empty
        and "total_time_sec_mean" in base_summary_agg.columns
    ):
        baseline_time = float(base_summary_agg["total_time_sec_mean"].iloc[0])
        fig_runtime.add_trace(
            go.Scatter(
                x=fractions,
                y=[1.0] * len(fractions),
                mode="lines",
                name="Baseline",
            )
        )
    if baseline_time and baseline_time > 0:
        for method in methods:
            for th in thresholds:
                y_vals = []
                for f in fractions:
                    match = _match(method, f, th)
                    if not match.empty and "total_time_sec_mean" in match.columns:
                        t_fec = float(match["total_time_sec_mean"].iloc[0])
                        y_vals.append(t_fec / baseline_time)
                    else:
                        y_vals.append(np.nan)
                if any(np.isfinite(y) for y in y_vals):
                    fig_runtime.add_trace(
                        go.Scatter(
                            x=fractions,
                            y=y_vals,
                            mode="lines+markers",
                            name=_legend_method_plus_threshold(method, th),
                        )
                    )
    if fig_runtime.data:
        fig_runtime.update_layout(
            title="Runtime ratio across sample fractions (time_FEC / time_baseline)",
            xaxis_title="Sample fraction",
            yaxis_title="Runtime ratio (FEC / baseline)",
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
        )
        sections.append(
            pio.to_html(fig_runtime, include_plotlyjs="cdn", full_html=False)
        )

    # Speedup across sample fractions — x = fraction, y = speedup, legend = (method + threshold), Baseline = 1
    fig_speedup_frac = go.Figure()
    if baseline_time and baseline_time > 0:
        fig_speedup_frac.add_trace(
            go.Scatter(
                x=fractions,
                y=[1.0] * len(fractions),
                mode="lines",
                name="Baseline",
            )
        )
        for method in methods:
            for th in thresholds:
                y_vals = []
                for f in fractions:
                    match = _match(method, f, th)
                    if not match.empty and "total_time_sec_mean" in match.columns:
                        t_fec = float(match["total_time_sec_mean"].iloc[0])
                        y_vals.append(
                            baseline_time / t_fec if t_fec > 0 else np.nan
                        )
                    else:
                        y_vals.append(np.nan)
                if any(np.isfinite(y) for y in y_vals):
                    fig_speedup_frac.add_trace(
                        go.Scatter(
                            x=fractions,
                            y=y_vals,
                            mode="lines+markers",
                            name=_legend_method_plus_threshold(method, th),
                        )
                    )
    if fig_speedup_frac.data:
        fig_speedup_frac.update_layout(
            title="Speedup across sample fractions (baseline_time / FEC_time)",
            xaxis_title="Sample fraction",
            yaxis_title="Speedup (>1 = FEC faster)",
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
        )
        sections.append(
            pio.to_html(fig_speedup_frac, include_plotlyjs="cdn", full_html=False)
        )

    return sections


def _get_sorted_thresholds_from_summary(fec_summary_agg: pd.DataFrame) -> List[float]:
    """Unique fake_hit_threshold values sorted 0 to higher, NaN last."""
    if fec_summary_agg.empty or "fake_hit_threshold" not in fec_summary_agg.columns:
        return []
    th_vals = fec_summary_agg["fake_hit_threshold"].dropna().unique()
    finite = sorted(float(t) for t in th_vals if np.isfinite(t))
    if fec_summary_agg["fake_hit_threshold"].isna().any():
        finite.append(np.nan)
    return finite


def _build_cross_threshold_figs(
    fec_summary_agg: pd.DataFrame,
    base_summary_agg: pd.DataFrame,
    thresholds: List[float],
    methods: List[str],
    fractions: List[float],
) -> List[str]:
    """Charts across thresholds (0 to higher): hit rate, fake-hit rate, runtime ratio."""
    sections: List[str] = []
    if not thresholds or fec_summary_agg.empty:
        return sections

    # For x-axis display: use numeric order; NaN shown as last.
    x_vals = [t if np.isfinite(t) else float("inf") for t in thresholds]
    x_label = [_format_threshold_label(t) for t in thresholds]

    baseline_time = None
    if (
        not base_summary_agg.empty
        and "total_time_sec_mean" in base_summary_agg.columns
    ):
        baseline_time = float(base_summary_agg["total_time_sec_mean"].iloc[0])

    # Hit rate vs threshold — one series per (method, fraction), legend = method + fraction
    fig_hit_th = go.Figure()
    for method in methods:
        for frac in fractions:
            y_vals: List[float] = []
            for th in thresholds:
                if np.isnan(th):
                    mask_th = fec_summary_agg["fake_hit_threshold"].isna()
                else:
                    mask_th = np.isclose(
                        fec_summary_agg["fake_hit_threshold"].astype(float), th
                    )
                match = fec_summary_agg[
                    (fec_summary_agg["mode"] == method)
                    & np.isclose(fec_summary_agg["sample_fraction"].astype(float), frac)
                    & mask_th
                ]
                if not match.empty and "hit_rate_overall_mean" in match.columns:
                    y_vals.append(float(match["hit_rate_overall_mean"].iloc[0]))
                else:
                    y_vals.append(np.nan)
            if any(np.isfinite(y) for y in y_vals):
                fig_hit_th.add_trace(
                    go.Scatter(
                        x=x_label,
                        y=y_vals,
                        mode="lines+markers",
                        name=_legend_method_plus_fraction(method, frac),
                    )
                )
    if fig_hit_th.data:
        fig_hit_th.update_layout(
            title="Overall hit rate across thresholds (FEC only)",
            xaxis_title="Fake-hit threshold (0 to higher)",
            yaxis_title="Hit rate",
            yaxis_tickformat=".0%",
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
        )
        sections.append(
            pio.to_html(fig_hit_th, include_plotlyjs="cdn", full_html=False)
        )

    # Fake-hit rate vs threshold
    fig_fake_th = go.Figure()
    for method in methods:
        for frac in fractions:
            y_vals = []
            for th in thresholds:
                if np.isnan(th):
                    mask_th = fec_summary_agg["fake_hit_threshold"].isna()
                else:
                    mask_th = np.isclose(
                        fec_summary_agg["fake_hit_threshold"].astype(float), th
                    )
                match = fec_summary_agg[
                    (fec_summary_agg["mode"] == method)
                    & np.isclose(fec_summary_agg["sample_fraction"].astype(float), frac)
                    & mask_th
                ]
                if not match.empty and "fake_hit_rate_overall_mean" in match.columns:
                    y_vals.append(float(match["fake_hit_rate_overall_mean"].iloc[0]))
                else:
                    y_vals.append(np.nan)
            if any(np.isfinite(y) for y in y_vals):
                fig_fake_th.add_trace(
                    go.Scatter(
                        x=x_label,
                        y=y_vals,
                        mode="lines+markers",
                        name=_legend_method_plus_fraction(method, frac),
                    )
                )
    if fig_fake_th.data:
        fig_fake_th.update_layout(
            title="Overall fake-hit rate across thresholds (FEC only)",
            xaxis_title="Fake-hit threshold (0 to higher)",
            yaxis_title="Fake-hit rate",
            yaxis_tickformat=".0%",
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
        )
        _scale_fake_hit_rate_yaxis(fig_fake_th)
        sections.append(
            pio.to_html(fig_fake_th, include_plotlyjs="cdn", full_html=False)
        )

    # Runtime ratio vs threshold — Baseline + (method, fraction)
    if baseline_time is not None and baseline_time > 0:
        fig_runtime_th = go.Figure()
        fig_runtime_th.add_trace(
            go.Scatter(
                x=x_label,
                y=[1.0] * len(x_label),
                mode="lines",
                name="Baseline",
            )
        )
        for method in methods:
            for frac in fractions:
                y_vals = []
                for th in thresholds:
                    if np.isnan(th):
                        mask_th = fec_summary_agg["fake_hit_threshold"].isna()
                    else:
                        mask_th = np.isclose(
                            fec_summary_agg["fake_hit_threshold"].astype(float), th
                        )
                    match = fec_summary_agg[
                        (fec_summary_agg["mode"] == method)
                        & np.isclose(
                            fec_summary_agg["sample_fraction"].astype(float), frac
                        )
                        & mask_th
                    ]
                    if not match.empty and "total_time_sec_mean" in match.columns:
                        t_fec = float(match["total_time_sec_mean"].iloc[0])
                        y_vals.append(t_fec / baseline_time)
                    else:
                        y_vals.append(np.nan)
                if any(np.isfinite(y) for y in y_vals):
                    fig_runtime_th.add_trace(
                        go.Scatter(
                            x=x_label,
                            y=y_vals,
                            mode="lines+markers",
                            name=_legend_method_plus_fraction(method, frac),
                        )
                    )
        if fig_runtime_th.data:
            fig_runtime_th.update_layout(
                title="Runtime ratio across thresholds (time_FEC / time_baseline)",
                xaxis_title="Fake-hit threshold (0 to higher)",
                yaxis_title="Runtime ratio (FEC / baseline)",
                template="plotly_white",
                hovermode="x unified",
                showlegend=True,
            )
            sections.append(
                pio.to_html(fig_runtime_th, include_plotlyjs="cdn", full_html=False)
            )

        # Speedup across thresholds — x = threshold, y = speedup, legend = (method + fraction), Baseline = 1
        fig_speedup_th = go.Figure()
        fig_speedup_th.add_trace(
            go.Scatter(
                x=x_label,
                y=[1.0] * len(x_label),
                mode="lines",
                name="Baseline",
            )
        )
        for method in methods:
            for frac in fractions:
                y_vals = []
                for th in thresholds:
                    if np.isnan(th):
                        mask_th = fec_summary_agg["fake_hit_threshold"].isna()
                    else:
                        mask_th = np.isclose(
                            fec_summary_agg["fake_hit_threshold"].astype(float), th
                        )
                    match = fec_summary_agg[
                        (fec_summary_agg["mode"] == method)
                        & np.isclose(
                            fec_summary_agg["sample_fraction"].astype(float), frac
                        )
                        & mask_th
                    ]
                    if not match.empty and "total_time_sec_mean" in match.columns:
                        t_fec = float(match["total_time_sec_mean"].iloc[0])
                        y_vals.append(
                            baseline_time / t_fec if t_fec > 0 else np.nan
                        )
                    else:
                        y_vals.append(np.nan)
                if any(np.isfinite(y) for y in y_vals):
                    fig_speedup_th.add_trace(
                        go.Scatter(
                            x=x_label,
                            y=y_vals,
                            mode="lines+markers",
                            name=_legend_method_plus_fraction(method, frac),
                        )
                    )
        if fig_speedup_th.data:
            fig_speedup_th.update_layout(
                title="Speedup across thresholds (baseline_time / FEC_time)",
                xaxis_title="Fake-hit threshold (0 to higher)",
                yaxis_title="Speedup (>1 = FEC faster)",
                template="plotly_white",
                hovermode="x unified",
                showlegend=True,
            )
            sections.append(
                pio.to_html(fig_speedup_th, include_plotlyjs="cdn", full_html=False)
            )

    return sections


def _build_comparison_table(
    fec_summary_agg: pd.DataFrame,
    base_summary_agg: pd.DataFrame,
) -> pd.DataFrame:
    """
    Table with rows = (sampling method, fraction) and columns:
        - baseline vs FEC summary (MAE and runtime)
        - delta MAE vs baseline
        - speed-up vs baseline (baseline_time / fec_time)
        - a simple "meaningful_speedup" flag
    """
    if (
        base_summary_agg.empty
        or "final_test_mae_mean" not in base_summary_agg.columns
        or "total_time_sec_mean" not in base_summary_agg.columns
    ):
        return pd.DataFrame()

    baseline_mae = float(base_summary_agg["final_test_mae_mean"].iloc[0])
    baseline_time = float(base_summary_agg["total_time_sec_mean"].iloc[0])

    records: List[dict] = []
    for _, row in fec_summary_agg.iterrows():
        mode = row["mode"]
        frac = float(row["sample_fraction"])
        th = float(row.get("fake_hit_threshold", np.nan))
        mae = float(row.get("final_test_mae_mean", np.nan))
        time_sec = float(row.get("total_time_sec_mean", np.nan))

        delta_mae = mae - baseline_mae if np.isfinite(mae) else np.nan
        speedup = (
            baseline_time / time_sec
            if np.isfinite(time_sec) and time_sec > 0
            else np.nan
        )

        # Heuristic flag: call speedup "meaningful" if
        #  - FEC is at least 10% faster (speedup >= 1.1), AND
        #  - MAE is not worse than baseline by more than 0.01.
        meaningful_speedup = bool(
            np.isfinite(speedup)
            and speedup >= 1.1
            and (
                not np.isfinite(delta_mae)
                or delta_mae <= 0.01
            )
        )

        records.append(
            {
                "mode": mode,
                "sample_fraction": frac,
                "fake_hit_threshold": th,
                "baseline_final_test_mae_mean": baseline_mae,
                "baseline_total_time_sec_mean": baseline_time,
                "fec_final_test_mae_mean": mae,
                "fec_total_time_sec_mean": time_sec,
                "delta_mae_vs_baseline": delta_mae,
                "speedup_vs_baseline": speedup,
                "meaningful_speedup": meaningful_speedup,
            }
        )

    return pd.DataFrame.from_records(records)


def _build_combined_summary(
    base_summary_agg: pd.DataFrame,
    fec_summary_agg: pd.DataFrame,
) -> pd.DataFrame:
    """
    Integrate baseline and FEC summaries into one table for comparison.

    - One row for baseline; one row per FEC (mode, sample_fraction, fake_hit_threshold).
    - Columns: source, mode, sample_fraction, fake_hit_threshold, accuracy/MAE/time
      (mean and std), speedup (baseline_time / time for FEC), delta_mae_vs_baseline,
      delta_accuracy_vs_baseline; then any FEC-only columns (empty for baseline).
    """
    if base_summary_agg.empty:
        return pd.DataFrame()

    base_row = base_summary_agg.iloc[0]
    baseline_mae = float(base_row.get("final_test_mae_mean", np.nan))
    baseline_accuracy = float(base_row.get("final_test_accuracy_mean", np.nan))
    baseline_time = float(base_row.get("total_time_sec_mean", np.nan))

    # Column order: identity, key metrics, comparison, then rest (FEC-only for baseline)
    identity_cols = ["source", "mode", "sample_fraction", "fake_hit_threshold"]
    key_metrics = [
        "final_test_mae_mean", "final_test_mae_std",
        "final_test_accuracy_mean", "final_test_accuracy_std",
        "total_time_sec_mean", "total_time_sec_std",
    ]
    comparison_cols = [
        "speedup",
        "delta_mae_vs_baseline",
        "delta_accuracy_vs_baseline",
    ]

    all_base_cols = set(base_summary_agg.columns) - set(identity_cols)
    all_fec_cols = set(fec_summary_agg.columns) - set(identity_cols)
    rest_cols = sorted((all_fec_cols | all_base_cols) - set(key_metrics))

    rows: List[dict] = []

    # Baseline row
    bline: dict = {
        "source": "baseline",
        "mode": "baseline",
        "sample_fraction": 1.0,
        "fake_hit_threshold": np.nan,
        "speedup": np.nan,
        "delta_mae_vs_baseline": 0.0,
        "delta_accuracy_vs_baseline": 0.0,
    }
    for c in key_metrics:
        bline[c] = base_row.get(c, np.nan)
    for c in rest_cols:
        bline[c] = base_row.get(c, np.nan) if c in base_row.index else np.nan
    rows.append(bline)

    # FEC rows
    for _, row in fec_summary_agg.iterrows():
        mae = float(row.get("final_test_mae_mean", np.nan))
        acc = float(row.get("final_test_accuracy_mean", np.nan))
        time_sec = float(row.get("total_time_sec_mean", np.nan))
        speedup = (
            baseline_time / time_sec
            if np.isfinite(time_sec) and time_sec > 0 and np.isfinite(baseline_time)
            else np.nan
        )
        delta_mae = mae - baseline_mae if np.isfinite(mae) and np.isfinite(baseline_mae) else np.nan
        delta_acc = (
            acc - baseline_accuracy
            if np.isfinite(acc) and np.isfinite(baseline_accuracy)
            else np.nan
        )
        rec: dict = {
            "source": "FEC",
            "mode": row["mode"],
            "sample_fraction": float(row["sample_fraction"]),
            "fake_hit_threshold": float(row.get("fake_hit_threshold", np.nan)),
            "speedup": speedup,
            "delta_mae_vs_baseline": delta_mae,
            "delta_accuracy_vs_baseline": delta_acc,
        }
        for c in key_metrics:
            rec[c] = row.get(c, np.nan)
        for c in rest_cols:
            rec[c] = row.get(c, np.nan) if c in row.index else np.nan
        rows.append(rec)

    col_order = identity_cols + key_metrics + comparison_cols + rest_cols
    df = pd.DataFrame.from_records(rows)
    return df[[c for c in col_order if c in df.columns]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate results across runs.\n\n"
            "Three modes:\n"
            "  1) Baseline-only folder (generation_stats_run*.csv)\n"
            "  2) FEC-only folder (generation_stats_*_run*.csv)\n"
            "  3) Experiment root folder containing 'baseline' and 'FEC' subfolders,\n"
            "     in which case an HTML comparison report is generated.\n\n"
            "Usage:\n"
            "  # Baseline only:\n"
            "  python FEC_report.py results/<dataset>_Gen_<G>_Pop_<P>/baseline\n\n"
            "  # FEC only:\n"
            "  python FEC_report.py results/<dataset>_Gen_<G>_Pop_<P>/FEC\n\n"
            "  # Full comparison + HTML report:\n"
            "  python FEC_report.py results/<dataset>_Gen_<G>_Pop_<P>/\n"
        )
    )
    parser.add_argument(
        "target_dir",
        type=str,
        help=(
            "Path to: (a) a baseline folder, (b) a FEC folder, or (c) an experiment "
            "root folder that contains 'baseline' and 'FEC' subdirectories."
        ),
    )
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    #target_dir="Wisconsin_Breast_Cancer_without_ID_Gen_50_Pop_5000_without_overhead"

    # Mode 3: experiment root with both baseline and FEC subfolders
    baseline_subdir = target_dir / "baseline"
    fec_subdir = target_dir / "FEC"
    if baseline_subdir.is_dir() and fec_subdir.is_dir():
        print(f"Detected experiment root: {target_dir}")

        # Always recompute baseline aggregates from raw per-run CSVs to avoid
        # stale or partially-written aggregated files.
        print("Computing baseline aggregates from per-run CSVs ...")
        gen_all_base, sum_all_base = _load_baseline_csvs(baseline_subdir)
        gen_agg_base = _aggregate_generation_stats(gen_all_base)
        base_gen_agg_path = baseline_subdir / "generation_stats_aggregated.csv"
        gen_agg_base.to_csv(base_gen_agg_path, index=False)
        sum_agg_base = _aggregate_summary_stats(sum_all_base)
        base_sum_agg_path = baseline_subdir / "summary_aggregated.csv"
        sum_agg_base.to_csv(base_sum_agg_path, index=False)

        # Always recompute FEC aggregates from raw per-run CSVs for the same reason.
        print("Computing FEC aggregates from per-run CSVs ...")
        gen_all_fec, sum_all_fec = _load_fec_csvs(fec_subdir)
        gen_agg_fec = _aggregate_fec_generation_stats(gen_all_fec)
        fec_gen_agg_path = fec_subdir / "generation_stats_aggregated_FEC.csv"
        gen_agg_fec.to_csv(fec_gen_agg_path, index=False)
        sum_agg_fec = _aggregate_fec_summary_stats(sum_all_fec)
        fec_sum_agg_path = fec_subdir / "summary_aggregated_FEC.csv"
        sum_agg_fec.to_csv(fec_sum_agg_path, index=False)

        methods = sorted(gen_agg_fec["mode"].dropna().unique().tolist())
        thresholds, fractions_per_threshold = _get_thresholds_and_fractions(
            gen_agg_fec
        )

        print("Building HTML report ...")
        sections: List[str] = []
        for th in thresholds:
            th_label = _format_threshold_label(th)
            sections.append(f'<h2>Threshold {th_label}</h2>')
            for frac in fractions_per_threshold.get(th, []):
                sections.append(f'<h3>Fraction {frac:.0%}</h3>')
                sections.extend(
                    _build_training_and_test_figs(
                        gen_agg_base, gen_agg_fec, th, frac, methods
                    )
                )
                sections.extend(
                    _build_fec_hit_and_fake_figs(
                        gen_agg_fec, th, frac, methods
                    )
                )
        # Cross-fraction summary (all thresholds/fractions)
        all_fractions = sorted(
            float(f)
            for f in gen_agg_fec["sample_fraction"].dropna().unique().tolist()
        )
        sections.append("<h2>Across fractions</h2>")
        sections.extend(
            _build_cross_fraction_figs(
                sum_agg_fec,
                sum_agg_base,
                all_fractions,
                methods,
            )
        )
        # Across thresholds (0 to higher): hit rate, fake-hit rate, runtime ratio
        summary_thresholds = _get_sorted_thresholds_from_summary(sum_agg_fec)
        if summary_thresholds:
            sections.append("<h2>Across thresholds</h2>")
            sections.extend(
                _build_cross_threshold_figs(
                    sum_agg_fec,
                    sum_agg_base,
                    summary_thresholds,
                    methods,
                    all_fractions,
                )
            )

        comparison_df = _build_comparison_table(sum_agg_fec, sum_agg_base)

        # Combined baseline + FEC summary CSV (accuracy, MAE, time, speedup) next to HTML
        combined_summary = _build_combined_summary(sum_agg_base, sum_agg_fec)
        if not combined_summary.empty:
            summary_csv_path = target_dir / "summary_baseline_vs_FEC.csv"
            combined_summary.to_csv(summary_csv_path, index=False)
            print(f"Saved combined summary to {summary_csv_path}")

        config_json = CONFIG.copy()
        html = (
            "<html><head>"
            "<meta charset='utf-8' />"
            "<title>Baseline vs FEC (Aggregated Report)</title>"
            "</head><body>"
            "<h1>Baseline vs FEC (Aggregated Report)</h1>"
            "<h2>Configuration (from config.py)</h2>"
            "<pre style='font-size: 12px; background:#f7f7f7; "
            "padding:8px; border:1px solid #ddd;'>"
            + str(config_json)
            + "</pre>"
        )

        # Comparison table
        html += "<h2>FEC vs baseline comparison table</h2>"
        if not comparison_df.empty:
            html += comparison_df.to_html(index=False)
        else:
            html += "<p>No comparison table (missing baseline or FEC summary rows).</p>"

        html += "".join(sections) + "</body></html>"

        report_path = target_dir / "FEC_report.html"
        report_path.write_text(html, encoding="utf-8")
        print(f"Saved aggregated HTML report to {report_path}")
        return

    # Modes 1 & 2: single-folder baseline or FEC aggregation only
    # Detect whether this is a baseline or FEC folder based on filenames.
    has_baseline_pattern = any(target_dir.glob("generation_stats_run*.csv"))
    has_fec_pattern = any(target_dir.glob("generation_stats_*_run*.csv"))

    if has_baseline_pattern and not has_fec_pattern:
        # Baseline aggregation
        print(f"Detected baseline folder. Loading CSVs from {target_dir} ...")
        gen_all, summary_all = _load_baseline_csvs(target_dir)

        print("Aggregating baseline per-generation stats across runs ...")
        gen_agg = _aggregate_generation_stats(gen_all)
        gen_agg_path = target_dir / "generation_stats_aggregated.csv"
        gen_agg.to_csv(gen_agg_path, index=False)
        print(f"Saved aggregated per-generation stats to {gen_agg_path}")

        print("Aggregating baseline summary stats across runs ...")
        summary_agg = _aggregate_summary_stats(summary_all)
        summary_agg_path = target_dir / "summary_aggregated.csv"
        summary_agg.to_csv(summary_agg_path, index=False)
        print(f"Saved aggregated summary stats to {summary_agg_path}")

    elif has_fec_pattern and not has_baseline_pattern:
        # FEC aggregation
        print(f"Detected FEC folder. Loading CSVs from {target_dir} ...")
        gen_all, summary_all = _load_fec_csvs(target_dir)

        print("Aggregating FEC per-generation stats across runs "
              "for each (mode, sample_fraction) ...")
        gen_agg = _aggregate_fec_generation_stats(gen_all)
        gen_agg_path = target_dir / "generation_stats_aggregated_FEC.csv"
        gen_agg.to_csv(gen_agg_path, index=False)
        print(f"Saved aggregated FEC per-generation stats to {gen_agg_path}")

        print("Aggregating FEC summary stats across runs "
              "for each (mode, sample_fraction) ...")
        summary_agg = _aggregate_fec_summary_stats(summary_all)
        summary_agg_path = target_dir / "summary_aggregated_FEC.csv"
        summary_agg.to_csv(summary_agg_path, index=False)
        print(f"Saved aggregated FEC summary stats to {summary_agg_path}")

    else:
        raise SystemExit(
            f"Could not determine folder type for {target_dir}. "
            "Expected baseline files 'generation_stats_run*.csv' OR "
            "FEC files 'generation_stats_*_run*.csv'."
        )


if __name__ == "__main__":
    main()

