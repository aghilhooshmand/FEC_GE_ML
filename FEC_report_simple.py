from __future__ import annotations

"""
Simple report for baseline_runs_simple.py and FEC_runs_simple.py.
Same structure as FEC_report.py but:
  - Reads from results_simple/<dataset>_Gen_<G>_Pop_<P>/baseline/ and .../FEC/
  - FEC filenames: generation_stats_<method>_frac_<pct>_th_<tag>_run<n> (e.g. th 0p01 -> 0.01)
  - Speedup = baseline total_time_sec / FEC total_time_fair_sec (fair excludes fake-hit eval time)
  - Outputs:
      - baseline/generation_stats_aggregated.csv   (evolution: per-generation metrics aggregated across runs)
      - baseline/summary_aggregated.csv            (summary stats aggregated across baseline runs)
      - FEC/generation_stats_aggregated_FEC.csv   (evolution: per-generation metrics by mode/fraction/threshold, aggregated across runs)
      - FEC/summary_aggregated_FEC.csv            (summary stats by mode/fraction/threshold, aggregated across runs)
      - summary_baseline_vs_FEC.csv               (combined comparison; columns include sampling_method, sampling_fraction, threshold)
      - FEC_report_simple.html
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from scipy import stats

# Tolerance for float comparison of threshold and sample_fraction (e.g. 0.01 vs 0.0100001)
_FLOAT_RTOL = 1e-5


def _safe_plot_series(series) -> list:
    """Convert a pandas Series to a list safe for Plotly JSON (NaN -> None, numeric types to Python float/int)."""
    if series is None or (hasattr(series, "empty") and series.empty):
        return []
    arr = np.asarray(series, dtype=float)
    return [None if np.isnan(x) else float(x) for x in arr]


def _safe_plot_x(series) -> list:
    """Generation values for x-axis (prefer int for display)."""
    if series is None or (hasattr(series, "empty") and series.empty):
        return []
    return [int(x) if np.isfinite(x) and x == np.trunc(x) else float(x) for x in np.asarray(series)]


def _parse_threshold_tag(tag: str) -> float | None:
    """Convert threshold tag to float: 0p01 -> 0.01, 1em5 -> 1e-5."""
    if not tag:
        return None
    s = tag.replace("p", ".", 1).replace("m", "e-", 1)
    try:
        return float(s)
    except ValueError:
        return None


def _parse_fec_filename(path: Path) -> tuple[str | None, float | None, float | None]:
    """
    Parse FEC filename template: generation_stats_<method>_frac_<pct>_th_<tag>_run<n>
    or summary_<method>_frac_<pct>_th_<tag>_run<n>.
    Returns (sampling_method, fraction, threshold); fraction = pct/100, threshold from tag (0p01 -> 0.01).
    """
    stem = path.stem
    if "_frac_" not in stem or "_th_" not in stem or "_run" not in stem:
        return None, None, None
    prefix, rest = stem.split("_frac_", 1)
    method = None
    if prefix.startswith("generation_stats_"):
        method = prefix[len("generation_stats_"):]
    elif prefix.startswith("summary_"):
        method = prefix[len("summary_"):]
    if not method:
        return None, None, None
    part_before_th = rest.split("_th_", 1)[0]
    try:
        fraction = int(part_before_th.strip()) / 100.0
    except ValueError:
        fraction = None
    after_th = rest.split("_th_", 1)[1]
    tag = after_th.split("_run", 1)[0].strip()
    threshold = _parse_threshold_tag(tag)
    return method, fraction, threshold


def _parse_threshold_from_fec_filename(path: Path) -> float | None:
    """Extract fake_hit_threshold from filename ..._th_<tag>_run<n>.csv."""
    _, _, th = _parse_fec_filename(path)
    return th


def _load_baseline_csvs(baseline_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all baseline per-run CSVs."""
    gen_files = sorted(baseline_dir.glob("generation_stats_run*.csv"))
    summary_files = sorted(baseline_dir.glob("summary_run*.csv"))
    if not gen_files:
        raise SystemExit(f"No generation_stats_run*.csv in {baseline_dir}")
    if not summary_files:
        raise SystemExit(f"No summary_run*.csv in {baseline_dir}")
    gen_all = pd.concat([pd.read_csv(p) for p in gen_files], ignore_index=True)
    summary_all = pd.concat([pd.read_csv(p) for p in summary_files], ignore_index=True)
    return gen_all, summary_all


def _load_fec_csvs(fec_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all FEC per-run CSVs. Filename template: generation_stats_<method>_frac_<pct>_th_<tag>_run<n>.
    Add/overwrite fake_hit_threshold from filename (0p01 -> 0.01) so grouping and filtering are consistent.
    """
    gen_files = sorted(fec_dir.glob("generation_stats_*_run*.csv"))
    summary_files = sorted(fec_dir.glob("summary_*_run*.csv"))
    if not gen_files:
        raise SystemExit(f"No generation_stats_*_run*.csv in {fec_dir}")
    if not summary_files:
        raise SystemExit(f"No summary_*_run*.csv in {fec_dir}")
    gen_frames = []
    for p in gen_files:
        df = pd.read_csv(p)
        method, fraction, th = _parse_fec_filename(p)
        df["fake_hit_threshold"] = th if th is not None else np.nan
        if fraction is not None:
            df["sample_fraction"] = fraction
        gen_frames.append(df)
    summary_frames = []
    for p in summary_files:
        df = pd.read_csv(p)
        method, fraction, th = _parse_fec_filename(p)
        df["fake_hit_threshold"] = th if th is not None else np.nan
        if fraction is not None:
            df["sample_fraction"] = fraction
        summary_frames.append(df)
    gen_all = pd.concat(gen_frames, ignore_index=True)
    summary_all = pd.concat(summary_frames, ignore_index=True)
    return gen_all, summary_all


def _aggregate_generation_stats(gen_all: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-generation metrics; group by mode, sample_fraction, gen."""
    grouping_keys = ["mode", "sample_fraction", "gen"]
    numeric_cols = gen_all.select_dtypes(include="number").columns.tolist()
    metrics = [c for c in numeric_cols if c not in (["run"] + grouping_keys)]
    if not metrics:
        return gen_all.copy()
    grouped = gen_all.groupby(grouping_keys, as_index=False)[metrics]
    agg_mean = grouped.mean(numeric_only=True)
    agg_std = grouped.std(ddof=0, numeric_only=True)
    agg_mean = agg_mean.rename(columns={c: f"{c}_mean" for c in metrics})
    agg_std = agg_std.rename(columns={c: f"{c}_std" for c in metrics})
    return pd.merge(agg_mean, agg_std, on=grouping_keys, how="left")


def _aggregate_summary_stats(summary_all: pd.DataFrame) -> pd.DataFrame:
    """Aggregate baseline summary; group by mode, sample_fraction."""
    grouping_keys = ["mode", "sample_fraction"]
    numeric_cols = summary_all.select_dtypes(include="number").columns.tolist()
    metrics = [c for c in numeric_cols if c not in (["run"] + grouping_keys)]
    if not metrics:
        return summary_all.copy()
    grouped = summary_all.groupby(grouping_keys, as_index=False)[metrics]
    agg_mean = grouped.mean(numeric_only=True)
    agg_std = grouped.std(ddof=0, numeric_only=True)
    agg_mean = agg_mean.rename(columns={c: f"{c}_mean" for c in metrics})
    agg_std = agg_std.rename(columns={c: f"{c}_std" for c in metrics})
    return pd.merge(agg_mean, agg_std, on=grouping_keys, how="left")


def _aggregate_fec_generation_stats(gen_all: pd.DataFrame) -> pd.DataFrame:
    """Aggregate FEC per-generation; group by mode, sample_fraction, fake_hit_threshold, gen."""
    grouping_keys = ["mode", "sample_fraction", "fake_hit_threshold", "gen"]
    numeric_cols = gen_all.select_dtypes(include="number").columns.tolist()
    metrics = [c for c in numeric_cols if c not in (["run"] + grouping_keys)]
    if not metrics:
        return gen_all.copy()
    grouped = gen_all.groupby(grouping_keys, as_index=False)[metrics]
    agg_mean = grouped.mean(numeric_only=True)
    agg_std = grouped.std(ddof=0, numeric_only=True)
    agg_mean = agg_mean.rename(columns={c: f"{c}_mean" for c in metrics})
    agg_std = agg_std.rename(columns={c: f"{c}_std" for c in metrics})
    return pd.merge(agg_mean, agg_std, on=grouping_keys, how="left")


def _aggregate_fec_summary_stats(summary_all: pd.DataFrame) -> pd.DataFrame:
    """Aggregate FEC summary; group by mode, sample_fraction, fake_hit_threshold. Add fake_hit_rate_overall_mean."""
    grouping_keys = ["mode", "sample_fraction", "fake_hit_threshold"]
    numeric_cols = summary_all.select_dtypes(include="number").columns.tolist()
    metrics = [c for c in numeric_cols if c not in (["run"] + grouping_keys)]
    if not metrics:
        return summary_all.copy()
    grouped = summary_all.groupby(grouping_keys, as_index=False)[metrics]
    agg_mean = grouped.mean(numeric_only=True)
    agg_std = grouped.std(ddof=0, numeric_only=True)
    agg_mean = agg_mean.rename(columns={c: f"{c}_mean" for c in metrics})
    agg_std = agg_std.rename(columns={c: f"{c}_std" for c in metrics})
    agg = pd.merge(agg_mean, agg_std, on=grouping_keys, how="left")
    if "hits_mean" in agg.columns and "fake_hits_mean" in agg.columns:
        agg["fake_hit_rate_overall_mean"] = np.where(
            agg["hits_mean"] > 0,
            agg["fake_hits_mean"] / agg["hits_mean"],
            np.nan,
        )
    return agg


def _get_thresholds_and_fractions(fec_gen_agg: pd.DataFrame) -> tuple[List[float], dict]:
    """Return (sorted thresholds, mapping threshold -> sorted fractions)."""
    if "fake_hit_threshold" not in fec_gen_agg.columns:
        fractions = sorted(fec_gen_agg["sample_fraction"].dropna().unique().tolist())
        return ([np.nan], {np.nan: [float(f) for f in fractions]}) if fractions else ([], {})
    th_vals = fec_gen_agg["fake_hit_threshold"].dropna().unique().tolist()
    nan_has = fec_gen_agg["fake_hit_threshold"].isna().any()
    thresholds = sorted((float(t) for t in th_vals if np.isfinite(t)), key=lambda x: (x, 0))
    if nan_has:
        thresholds.append(np.nan)
    frac_per_th = {}
    for th in thresholds:
        if np.isnan(th):
            mask = fec_gen_agg["fake_hit_threshold"].isna()
        else:
            mask = np.isclose(fec_gen_agg["fake_hit_threshold"].astype(float), th, rtol=_FLOAT_RTOL)
        frac_per_th[th] = sorted(float(f) for f in fec_gen_agg.loc[mask, "sample_fraction"].dropna().unique().tolist())
    return thresholds, frac_per_th


def _get_sorted_thresholds_from_summary(fec_summary_agg: pd.DataFrame) -> List[float]:
    if fec_summary_agg.empty or "fake_hit_threshold" not in fec_summary_agg.columns:
        return []
    th_vals = fec_summary_agg["fake_hit_threshold"].dropna().unique()
    finite = sorted(float(t) for t in th_vals if np.isfinite(t))
    if fec_summary_agg["fake_hit_threshold"].isna().any():
        finite.append(np.nan)
    return finite


def _format_method_display(method: str) -> str:
    s = method
    if s.startswith("fec_simple_"):
        s = s[len("fec_simple_"):]
    elif s.startswith("fec_"):
        s = s[4:]
    if s == "kmeans":
        return "k-means"
    return s.replace("_", "-")


def _format_legend_fec(method: str, frac: float) -> str:
    return f"{_format_method_display(method)} ({frac:.0%})"


def _format_threshold_label(threshold: float) -> str:
    if np.isnan(threshold):
        return "N/A"
    if threshold >= 0.01 or threshold == 0:
        return str(threshold)
    return f"{threshold:.0e}"


def _legend_method_plus_threshold(method: str, threshold: float) -> str:
    return f"{_format_method_display(method)} (th {_format_threshold_label(threshold)})"


def _legend_method_plus_fraction(method: str, fraction: float) -> str:
    return f"{_format_method_display(method)} ({fraction:.0%})"


def _scale_fake_hit_rate_yaxis(fig: go.Figure) -> None:
    if not fig.data:
        return
    all_y = [float(v) for t in fig.data for v in (getattr(t, "y", None) or []) if v is not None and np.isfinite(v)]
    if not all_y:
        return
    max_y = max(all_y)
    if max_y <= 0:
        fig.update_layout(yaxis_range=[0, 0.01], yaxis_tickformat=".2%")
    elif max_y < 0.1:
        fig.update_layout(yaxis_range=[0, max(0.01, max_y * 1.15)], yaxis_tickformat=".2%")


def _filter_fec_by_threshold_and_fraction(
    fec_gen_agg: pd.DataFrame, threshold: float, fraction: float
) -> pd.DataFrame:
    mask_frac = np.isclose(fec_gen_agg["sample_fraction"].astype(float), fraction, rtol=_FLOAT_RTOL)
    if "fake_hit_threshold" not in fec_gen_agg.columns:
        return fec_gen_agg.loc[mask_frac].copy()
    if np.isnan(threshold):
        mask_th = fec_gen_agg["fake_hit_threshold"].isna()
    else:
        mask_th = np.isclose(fec_gen_agg["fake_hit_threshold"].astype(float), threshold, rtol=_FLOAT_RTOL)
    return fec_gen_agg.loc[mask_th & mask_frac].copy()


def _legend_evolution(method: str, fraction: float, threshold: float) -> str:
    """Legend label for evolution charts: method, fraction, threshold."""
    method_d = _format_method_display(method)
    th_l = _format_threshold_label(threshold)
    return f"{method_d} {fraction:.0%} th {th_l}"


def _build_aggregated_evolution_figs(
    base_gen_agg: pd.DataFrame,
    fec_gen_agg: pd.DataFrame,
    methods: List[str],
    thresholds: List[float],
    fractions_per_threshold: dict,
) -> List[str]:
    """
    Build two charts: Training MAE and Test MAE over generations with all (fraction, threshold)
    configurations as separate series (Baseline + each FEC config). Aggregated evolution.
    """
    blocks = []
    base_rows = base_gen_agg.sort_values("gen")

    # Training MAE — all configs
    fig_train = go.Figure()
    if not base_rows.empty and "avg_mean" in base_rows.columns:
        x_b = _safe_plot_x(base_rows["gen"])
        y_b = _safe_plot_series(base_rows["avg_mean"])
        if x_b and y_b and any(v is not None for v in y_b):
            err = _safe_plot_series(base_rows.get("avg_std")) if "avg_std" in base_rows.columns else None
            fig_train.add_trace(
                go.Scatter(
                    x=x_b, y=y_b,
                    mode="lines+markers", name="Baseline",
                    error_y=dict(type="data", array=err, visible=bool(err and any(v is not None for v in err))) if err else {},
                )
            )
    for th in thresholds:
        for frac in fractions_per_threshold.get(th, []):
            frac_rows = _filter_fec_by_threshold_and_fraction(fec_gen_agg, th, frac)
            for method in methods:
                rows = frac_rows[frac_rows["mode"] == method].sort_values("gen")
                if not rows.empty and "avg_mean" in rows.columns:
                    x_f = _safe_plot_x(rows["gen"])
                    y_f = _safe_plot_series(rows["avg_mean"])
                    if x_f and y_f and any(v is not None for v in y_f):
                        err = _safe_plot_series(rows.get("avg_std")) if "avg_std" in rows.columns else None
                        fig_train.add_trace(
                            go.Scatter(
                                x=x_f, y=y_f,
                                mode="lines+markers", name=_legend_evolution(method, frac, th),
                                error_y=dict(type="data", array=err, visible=bool(err and any(v is not None for v in err))) if err else {},
                            )
                        )
    if fig_train.data:
        fig_train.update_layout(
            title="Aggregated evolution — Training MAE (all fractions and thresholds)",
            xaxis_title="Generation", yaxis_title="Training MAE (lower is better)",
            template="plotly_white", hovermode="x unified", showlegend=True,
        )
        blocks.append(pio.to_html(fig_train, include_plotlyjs=False, full_html=False))

    # Test MAE — all configs
    fig_test = go.Figure()
    if not base_rows.empty and "fitness_test_mean" in base_rows.columns:
        x_b = _safe_plot_x(base_rows["gen"])
        y_b = _safe_plot_series(base_rows["fitness_test_mean"])
        if x_b and y_b and any(v is not None for v in y_b):
            err = _safe_plot_series(base_rows.get("fitness_test_std")) if "fitness_test_std" in base_rows.columns else None
            fig_test.add_trace(
                go.Scatter(
                    x=x_b, y=y_b,
                    mode="lines+markers", name="Baseline",
                    error_y=dict(type="data", array=err, visible=bool(err and any(v is not None for v in err))) if err else {},
                )
            )
    for th in thresholds:
        for frac in fractions_per_threshold.get(th, []):
            frac_rows = _filter_fec_by_threshold_and_fraction(fec_gen_agg, th, frac)
            for method in methods:
                rows = frac_rows[frac_rows["mode"] == method].sort_values("gen")
                if not rows.empty and "fitness_test_mean" in rows.columns:
                    x_f = _safe_plot_x(rows["gen"])
                    y_f = _safe_plot_series(rows["fitness_test_mean"])
                    if x_f and y_f and any(v is not None for v in y_f):
                        err = _safe_plot_series(rows.get("fitness_test_std")) if "fitness_test_std" in rows.columns else None
                        fig_test.add_trace(
                            go.Scatter(
                                x=x_f, y=y_f,
                                mode="lines+markers", name=_legend_evolution(method, frac, th),
                                error_y=dict(type="data", array=err, visible=bool(err and any(v is not None for v in err))) if err else {},
                            )
                        )
    if fig_test.data:
        fig_test.update_layout(
            title="Aggregated evolution — Test MAE (all fractions and thresholds)",
            xaxis_title="Generation", yaxis_title="Test MAE (lower is better)",
            template="plotly_white", hovermode="x unified", showlegend=True,
        )
        blocks.append(pio.to_html(fig_test, include_plotlyjs=False, full_html=False))

    return blocks


def _speedup_pvalue(baseline_times: np.ndarray, fec_fair_times: np.ndarray) -> float:
    """
    One-sided t-test: H0 mean_baseline <= mean_FEC, H1 mean_baseline > mean_FEC.
    Returns p-value; small p means speedup (baseline > FEC) is significant.
    """
    baseline_times = np.asarray(baseline_times, dtype=float)
    fec_fair_times = np.asarray(fec_fair_times, dtype=float)
    baseline_times = baseline_times[np.isfinite(baseline_times)]
    fec_fair_times = fec_fair_times[np.isfinite(fec_fair_times)]
    if baseline_times.size < 2 or fec_fair_times.size < 2:
        return np.nan
    try:
        result = stats.ttest_ind(
            baseline_times, fec_fair_times, alternative="greater", equal_var=False
        )
        return float(result.pvalue)
    except Exception:
        return np.nan


def _get_fec_fair_time(row: pd.Series) -> float:
    """FEC time for fair comparison: total_time_fair_sec_mean if present else total_time_sec_mean - fake_eval_time_sec_mean."""
    fair = row.get("total_time_fair_sec_mean")
    if fair is not None and np.isfinite(fair):
        return float(fair)
    t = float(row.get("total_time_sec_mean", np.nan))
    f = float(row.get("fake_eval_time_sec_mean", 0) or 0)
    if np.isfinite(t) and np.isfinite(f):
        return max(0.0, t - f)
    return np.nan


def _build_training_and_test_figs(
    base_gen_agg: pd.DataFrame,
    fec_gen_agg: pd.DataFrame,
    threshold: float,
    fraction: float,
    methods: List[str],
) -> List[str]:
    frac_rows = _filter_fec_by_threshold_and_fraction(fec_gen_agg, threshold, fraction)
    blocks = []
    base_rows = base_gen_agg.sort_values("gen")

    # Training MAE
    fig_train = go.Figure()
    if not base_rows.empty and "avg_mean" in base_rows.columns:
        x_b = _safe_plot_x(base_rows["gen"])
        y_b = _safe_plot_series(base_rows["avg_mean"])
        if x_b and y_b and any(v is not None for v in y_b):
            err = _safe_plot_series(base_rows.get("avg_std")) if "avg_std" in base_rows.columns else None
            fig_train.add_trace(
                go.Scatter(
                    x=x_b, y=y_b,
                    mode="lines+markers", name="Baseline",
                    error_y=dict(type="data", array=err, visible=bool(err and any(v is not None for v in err))) if err else {},
                )
            )
    for method in methods:
        rows = frac_rows[frac_rows["mode"] == method].sort_values("gen")
        if not rows.empty and "avg_mean" in rows.columns:
            x_f = _safe_plot_x(rows["gen"])
            y_f = _safe_plot_series(rows["avg_mean"])
            if x_f and y_f and any(v is not None for v in y_f):
                err = _safe_plot_series(rows.get("avg_std")) if "avg_std" in rows.columns else None
                fig_train.add_trace(
                    go.Scatter(
                        x=x_f, y=y_f,
                        mode="lines+markers", name=_format_legend_fec(method, fraction),
                        error_y=dict(type="data", array=err, visible=bool(err and any(v is not None for v in err))) if err else {},
                    )
                )
    th_label = _format_threshold_label(threshold)
    if fig_train.data:
        fig_train.update_layout(
            title=f"Training MAE — Threshold {th_label}, Fraction {fraction:.0%}",
            xaxis_title="Generation", yaxis_title="Training MAE (lower is better)",
            template="plotly_white", hovermode="x unified",
        )
        blocks.append(pio.to_html(fig_train, include_plotlyjs=False, full_html=False))

    # Test MAE
    fig_test = go.Figure()
    if not base_rows.empty and "fitness_test_mean" in base_rows.columns:
        x_b = _safe_plot_x(base_rows["gen"])
        y_b = _safe_plot_series(base_rows["fitness_test_mean"])
        if x_b and y_b and any(v is not None for v in y_b):
            err = _safe_plot_series(base_rows.get("fitness_test_std")) if "fitness_test_std" in base_rows.columns else None
            fig_test.add_trace(
                go.Scatter(
                    x=x_b, y=y_b,
                    mode="lines+markers", name="Baseline",
                    error_y=dict(type="data", array=err, visible=bool(err and any(v is not None for v in err))) if err else {},
                )
            )
    for method in methods:
        rows = frac_rows[frac_rows["mode"] == method].sort_values("gen")
        if not rows.empty and "fitness_test_mean" in rows.columns:
            x_f = _safe_plot_x(rows["gen"])
            y_f = _safe_plot_series(rows["fitness_test_mean"])
            if x_f and y_f and any(v is not None for v in y_f):
                err = _safe_plot_series(rows.get("fitness_test_std")) if "fitness_test_std" in rows.columns else None
                fig_test.add_trace(
                    go.Scatter(
                        x=x_f, y=y_f,
                        mode="lines+markers", name=_format_legend_fec(method, fraction),
                        error_y=dict(type="data", array=err, visible=bool(err and any(v is not None for v in err))) if err else {},
                    )
                )
    if fig_test.data:
        fig_test.update_layout(
            title=f"Test MAE — Threshold {th_label}, Fraction {fraction:.0%}",
            xaxis_title="Generation", yaxis_title="Test MAE (lower is better)",
            template="plotly_white", hovermode="x unified",
        )
        blocks.append(pio.to_html(fig_test, include_plotlyjs=False, full_html=False))

    return blocks


def _build_cross_fraction_figs(
    fec_summary_agg: pd.DataFrame,
    base_summary_agg: pd.DataFrame,
    fractions: List[float],
    methods: List[str],
) -> List[str]:
    sections = []
    thresholds = _get_sorted_thresholds_from_summary(fec_summary_agg) or [np.nan]

    def _match(method: str, frac: float, th: float):
        mask_frac = np.isclose(fec_summary_agg["sample_fraction"].astype(float), frac, rtol=_FLOAT_RTOL)
        if "fake_hit_threshold" not in fec_summary_agg.columns:
            mask_th = np.ones(len(fec_summary_agg), dtype=bool)
        elif np.isnan(th):
            mask_th = fec_summary_agg["fake_hit_threshold"].isna()
        else:
            mask_th = np.isclose(fec_summary_agg["fake_hit_threshold"].astype(float), th, rtol=_FLOAT_RTOL)
        return fec_summary_agg[(fec_summary_agg["mode"] == method) & mask_frac & mask_th]

    baseline_mae = float(base_summary_agg["final_test_mae_mean"].iloc[0]) if not base_summary_agg.empty and "final_test_mae_mean" in base_summary_agg.columns else np.nan
    baseline_time = float(base_summary_agg["total_time_sec_mean"].iloc[0]) if not base_summary_agg.empty and "total_time_sec_mean" in base_summary_agg.columns else np.nan

    # Final test MAE
    fig_final = go.Figure()
    if np.isfinite(baseline_mae):
        fig_final.add_trace(go.Scatter(x=fractions, y=[baseline_mae] * len(fractions), mode="lines", name="Baseline"))
    for method in methods:
        for th in thresholds:
            y_vals = []
            for f in fractions:
                m = _match(method, f, th)
                y_vals.append(float(m["final_test_mae_mean"].iloc[0]) if not m.empty and "final_test_mae_mean" in m.columns else np.nan)
            if any(np.isfinite(y) for y in y_vals):
                fig_final.add_trace(go.Scatter(x=fractions, y=y_vals, mode="lines+markers", name=_legend_method_plus_threshold(method, th)))
    if fig_final.data:
        fig_final.update_layout(title="Final test MAE across sample fractions", xaxis_title="Sample fraction", yaxis_title="Final test MAE", template="plotly_white", hovermode="x unified")
        sections.append(pio.to_html(fig_final, include_plotlyjs=False, full_html=False))

    # Hit rate across fractions
    fig_hit = go.Figure()
    for method in methods:
        for th in thresholds:
            y_vals = []
            for f in fractions:
                m = _match(method, f, th)
                y_vals.append(float(m["hit_rate_overall_mean"].iloc[0]) if not m.empty and "hit_rate_overall_mean" in m.columns else np.nan)
            if any(np.isfinite(y) for y in y_vals):
                fig_hit.add_trace(go.Scatter(x=fractions, y=y_vals, mode="lines+markers", name=_legend_method_plus_threshold(method, th)))
    if fig_hit.data:
        fig_hit.update_layout(title="Overall hit rate across sample fractions", xaxis_title="Sample fraction", yaxis_title="Hit rate", yaxis_tickformat=".0%", template="plotly_white", hovermode="x unified")
        sections.append(pio.to_html(fig_hit, include_plotlyjs=False, full_html=False))

    # Fake-hit rate across fractions
    if "fake_hit_rate_overall_mean" in fec_summary_agg.columns:
        fig_fake = go.Figure()
        for method in methods:
            for th in thresholds:
                y_vals = []
                for f in fractions:
                    m = _match(method, f, th)
                    y_vals.append(float(m["fake_hit_rate_overall_mean"].iloc[0]) if not m.empty and "fake_hit_rate_overall_mean" in m.columns else np.nan)
                if any(np.isfinite(y) for y in y_vals):
                    fig_fake.add_trace(go.Scatter(x=fractions, y=y_vals, mode="lines+markers", name=_legend_method_plus_threshold(method, th)))
        if fig_fake.data:
            fig_fake.update_layout(title="Overall fake-hit rate across sample fractions", xaxis_title="Sample fraction", yaxis_title="Fake-hit rate", yaxis_tickformat=".0%", template="plotly_white", hovermode="x unified")
            _scale_fake_hit_rate_yaxis(fig_fake)
            sections.append(pio.to_html(fig_fake, include_plotlyjs=False, full_html=False))

    # Runtime ratio (FEC_fair_time / baseline_time) — baseline = 1
    fig_rt = go.Figure()
    if np.isfinite(baseline_time) and baseline_time > 0:
        fig_rt.add_trace(go.Scatter(x=fractions, y=[1.0] * len(fractions), mode="lines", name="Baseline"))
        for method in methods:
            for th in thresholds:
                y_vals = []
                for f in fractions:
                    m = _match(method, f, th)
                    if not m.empty:
                        fair_t = _get_fec_fair_time(m.iloc[0])
                        y_vals.append(fair_t / baseline_time if np.isfinite(fair_t) else np.nan)
                    else:
                        y_vals.append(np.nan)
                if any(np.isfinite(y) for y in y_vals):
                    fig_rt.add_trace(go.Scatter(x=fractions, y=y_vals, mode="lines+markers", name=_legend_method_plus_threshold(method, th)))
        if fig_rt.data:
            fig_rt.update_layout(title="Runtime ratio across fractions (FEC_fair_time / baseline_time)", xaxis_title="Sample fraction", yaxis_title="Runtime ratio", template="plotly_white", hovermode="x unified")
            sections.append(pio.to_html(fig_rt, include_plotlyjs=False, full_html=False))

    # Speedup across fractions: baseline_time / FEC_total_time_fair_sec
    fig_speed = go.Figure()
    if np.isfinite(baseline_time) and baseline_time > 0:
        fig_speed.add_trace(go.Scatter(x=fractions, y=[1.0] * len(fractions), mode="lines", name="Baseline"))
        for method in methods:
            for th in thresholds:
                y_vals = []
                for f in fractions:
                    m = _match(method, f, th)
                    if not m.empty:
                        fair_t = _get_fec_fair_time(m.iloc[0])
                        y_vals.append(baseline_time / fair_t if np.isfinite(fair_t) and fair_t > 0 else np.nan)
                    else:
                        y_vals.append(np.nan)
                if any(np.isfinite(y) for y in y_vals):
                    fig_speed.add_trace(go.Scatter(x=fractions, y=y_vals, mode="lines+markers", name=_legend_method_plus_threshold(method, th)))
        if fig_speed.data:
            fig_speed.update_layout(title="Speedup across fractions (baseline total_time_sec / FEC total_time_fair_sec)", xaxis_title="Sample fraction", yaxis_title="Speedup (>1 = FEC faster)", template="plotly_white", hovermode="x unified")
            sections.append(pio.to_html(fig_speed, include_plotlyjs=False, full_html=False))

    return sections


def _build_cross_threshold_figs(
    fec_summary_agg: pd.DataFrame,
    base_summary_agg: pd.DataFrame,
    thresholds: List[float],
    methods: List[str],
    fractions: List[float],
) -> List[str]:
    sections = []
    if not thresholds or fec_summary_agg.empty:
        return sections
    x_label = [_format_threshold_label(t) for t in thresholds]
    baseline_time = float(base_summary_agg["total_time_sec_mean"].iloc[0]) if not base_summary_agg.empty and "total_time_sec_mean" in base_summary_agg.columns else None

    # Hit rate vs threshold
    fig_hit = go.Figure()
    for method in methods:
        for frac in fractions:
            y_vals = []
            for th in thresholds:
                if np.isnan(th):
                    mask_th = fec_summary_agg["fake_hit_threshold"].isna()
                else:
                    mask_th = np.isclose(fec_summary_agg["fake_hit_threshold"].astype(float), th, rtol=_FLOAT_RTOL)
                m = fec_summary_agg[(fec_summary_agg["mode"] == method) & np.isclose(fec_summary_agg["sample_fraction"].astype(float), frac, rtol=_FLOAT_RTOL) & mask_th]
                y_vals.append(float(m["hit_rate_overall_mean"].iloc[0]) if not m.empty and "hit_rate_overall_mean" in m.columns else np.nan)
            if any(np.isfinite(y) for y in y_vals):
                fig_hit.add_trace(go.Scatter(x=x_label, y=y_vals, mode="lines+markers", name=_legend_method_plus_fraction(method, frac)))
    if fig_hit.data:
        fig_hit.update_layout(title="Overall hit rate across thresholds", xaxis_title="Fake-hit threshold", yaxis_title="Hit rate", yaxis_tickformat=".0%", template="plotly_white", hovermode="x unified")
        sections.append(pio.to_html(fig_hit, include_plotlyjs=False, full_html=False))

    # Fake-hit rate vs threshold
    if "fake_hit_rate_overall_mean" in fec_summary_agg.columns:
        fig_fake = go.Figure()
        for method in methods:
            for frac in fractions:
                y_vals = []
                for th in thresholds:
                    if np.isnan(th):
                        mask_th = fec_summary_agg["fake_hit_threshold"].isna()
                    else:
                        mask_th = np.isclose(fec_summary_agg["fake_hit_threshold"].astype(float), th, rtol=_FLOAT_RTOL)
                    m = fec_summary_agg[(fec_summary_agg["mode"] == method) & np.isclose(fec_summary_agg["sample_fraction"].astype(float), frac, rtol=_FLOAT_RTOL) & mask_th]
                    y_vals.append(float(m["fake_hit_rate_overall_mean"].iloc[0]) if not m.empty and "fake_hit_rate_overall_mean" in m.columns else np.nan)
                if any(np.isfinite(y) for y in y_vals):
                    fig_fake.add_trace(go.Scatter(x=x_label, y=y_vals, mode="lines+markers", name=_legend_method_plus_fraction(method, frac)))
        if fig_fake.data:
            fig_fake.update_layout(title="Overall fake-hit rate across thresholds", xaxis_title="Fake-hit threshold", yaxis_title="Fake-hit rate", yaxis_tickformat=".0%", template="plotly_white", hovermode="x unified")
            _scale_fake_hit_rate_yaxis(fig_fake)
            sections.append(pio.to_html(fig_fake, include_plotlyjs=False, full_html=False))

    # Runtime ratio and Speedup across thresholds (using fair time)
    if baseline_time is not None and baseline_time > 0:
        fig_rt = go.Figure()
        fig_rt.add_trace(go.Scatter(x=x_label, y=[1.0] * len(x_label), mode="lines", name="Baseline"))
        for method in methods:
            for frac in fractions:
                y_vals = []
                for th in thresholds:
                    if np.isnan(th):
                        mask_th = fec_summary_agg["fake_hit_threshold"].isna()
                    else:
                        mask_th = np.isclose(fec_summary_agg["fake_hit_threshold"].astype(float), th, rtol=_FLOAT_RTOL)
                    m = fec_summary_agg[(fec_summary_agg["mode"] == method) & np.isclose(fec_summary_agg["sample_fraction"].astype(float), frac, rtol=_FLOAT_RTOL) & mask_th]
                    if not m.empty:
                        fair_t = _get_fec_fair_time(m.iloc[0])
                        y_vals.append(fair_t / baseline_time if np.isfinite(fair_t) else np.nan)
                    else:
                        y_vals.append(np.nan)
                if any(np.isfinite(y) for y in y_vals):
                    fig_rt.add_trace(go.Scatter(x=x_label, y=y_vals, mode="lines+markers", name=_legend_method_plus_fraction(method, frac)))
        if fig_rt.data:
            fig_rt.update_layout(title="Runtime ratio across thresholds (FEC_fair_time / baseline_time)", xaxis_title="Fake-hit threshold", yaxis_title="Runtime ratio", template="plotly_white", hovermode="x unified")
            sections.append(pio.to_html(fig_rt, include_plotlyjs=False, full_html=False))

        fig_speed = go.Figure()
        fig_speed.add_trace(go.Scatter(x=x_label, y=[1.0] * len(x_label), mode="lines", name="Baseline"))
        for method in methods:
            for frac in fractions:
                y_vals = []
                for th in thresholds:
                    if np.isnan(th):
                        mask_th = fec_summary_agg["fake_hit_threshold"].isna()
                    else:
                        mask_th = np.isclose(fec_summary_agg["fake_hit_threshold"].astype(float), th, rtol=_FLOAT_RTOL)
                    m = fec_summary_agg[(fec_summary_agg["mode"] == method) & np.isclose(fec_summary_agg["sample_fraction"].astype(float), frac, rtol=_FLOAT_RTOL) & mask_th]
                    if not m.empty:
                        fair_t = _get_fec_fair_time(m.iloc[0])
                        y_vals.append(baseline_time / fair_t if np.isfinite(fair_t) and fair_t > 0 else np.nan)
                    else:
                        y_vals.append(np.nan)
                if any(np.isfinite(y) for y in y_vals):
                    fig_speed.add_trace(go.Scatter(x=x_label, y=y_vals, mode="lines+markers", name=_legend_method_plus_fraction(method, frac)))
        if fig_speed.data:
            fig_speed.update_layout(title="Speedup across thresholds (baseline total_time_sec / FEC total_time_fair_sec)", xaxis_title="Fake-hit threshold", yaxis_title="Speedup (>1 = FEC faster)", template="plotly_white", hovermode="x unified")
            sections.append(pio.to_html(fig_speed, include_plotlyjs=False, full_html=False))

    return sections


def _build_speedup_heatmaps(
    fec_summary_agg: pd.DataFrame,
    base_summary_agg: pd.DataFrame,
    methods: List[str],
    fractions: List[float],
    thresholds: List[float],
) -> List[str]:
    sections = []
    if fec_summary_agg.empty or base_summary_agg.empty or not fractions or not thresholds:
        return sections
    baseline_mae = float(base_summary_agg["final_test_mae_mean"].iloc[0])
    baseline_time = float(base_summary_agg["total_time_sec_mean"].iloc[0])
    if not np.isfinite(baseline_time) or baseline_time <= 0:
        return sections
    th_labels = [_format_threshold_label(t) for t in thresholds]

    for method in methods:
        rows_m = fec_summary_agg[fec_summary_agg["mode"] == method]
        if rows_m.empty:
            continue
        z_speed, text_annot = [], []
        for frac in fractions:
            row_speed, row_txt = [], []
            for th in thresholds:
                mask_frac = np.isclose(rows_m["sample_fraction"].astype(float), frac, rtol=_FLOAT_RTOL)
                if np.isnan(th):
                    mask_th = rows_m["fake_hit_threshold"].isna()
                else:
                    mask_th = np.isclose(rows_m["fake_hit_threshold"].astype(float), th, rtol=_FLOAT_RTOL)
                m = rows_m[mask_frac & mask_th]
                if not m.empty and "final_test_mae_mean" in m.columns:
                    fair_t = _get_fec_fair_time(m.iloc[0])
                    mae_fec = float(m["final_test_mae_mean"].iloc[0])
                    speed = baseline_time / fair_t if np.isfinite(fair_t) and fair_t > 0 else np.nan
                    d_mae = mae_fec - baseline_mae if np.isfinite(mae_fec) else np.nan
                    row_speed.append(speed)
                    row_txt.append(f"S={speed:.2f}\nΔMAE={d_mae:.3f}" if np.isfinite(speed) and np.isfinite(d_mae) else (f"S={speed:.2f}" if np.isfinite(speed) else ""))
                else:
                    row_speed.append(np.nan)
                    row_txt.append("")
            z_speed.append(row_speed)
            text_annot.append(row_txt)
        if not any(np.isfinite(v) for row in z_speed for v in row):
            continue
        fig = go.Figure(
            data=go.Heatmap(
                z=z_speed, x=th_labels, y=[f"{f:.0%}" for f in fractions],
                text=text_annot, texttemplate="%{text}", colorscale="Viridis",
                colorbar=dict(title="Speedup (baseline_time / FEC_time_fair)"), zmin=0,
            )
        )
        fig.update_layout(
            title=f"Speedup heatmap — {_format_method_display(method)} (Speedup = baseline total_time_sec / FEC total_time_fair_sec)",
            xaxis_title="Fake-hit threshold", yaxis_title="Sample fraction", template="plotly_white",
        )
        sections.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))
    return sections


def _build_hit_and_fake_heatmaps(
    fec_summary_agg: pd.DataFrame,
    methods: List[str],
    fractions: List[float],
    thresholds: List[float],
) -> List[str]:
    sections = []
    if fec_summary_agg.empty or not fractions or not thresholds:
        return sections
    if "hit_rate_overall_mean" not in fec_summary_agg.columns:
        return sections
    has_fake = "fake_hit_rate_overall_mean" in fec_summary_agg.columns
    th_labels = [_format_threshold_label(t) for t in thresholds]
    frac_labels = [f"{f:.0%}" for f in fractions]

    for method in methods:
        rows_m = fec_summary_agg[fec_summary_agg["mode"] == method]
        if rows_m.empty:
            continue
        z_hit, text_hit = [], []
        z_fake, text_fake = [], []
        for frac in fractions:
            rh, th, rf, tf = [], [], [], []
            for th_v in thresholds:
                mask_frac = np.isclose(rows_m["sample_fraction"].astype(float), frac, rtol=_FLOAT_RTOL)
                if np.isnan(th_v):
                    mask_th = rows_m["fake_hit_threshold"].isna()
                else:
                    mask_th = np.isclose(rows_m["fake_hit_threshold"].astype(float), th_v, rtol=_FLOAT_RTOL)
                m = rows_m[mask_frac & mask_th]
                if not m.empty:
                    hr = float(m["hit_rate_overall_mean"].iloc[0])
                    fr = float(m["fake_hit_rate_overall_mean"].iloc[0]) if has_fake else np.nan
                    rh.append(hr if np.isfinite(hr) else np.nan)
                    rf.append(fr if np.isfinite(fr) else np.nan)
                    th.append(f"{hr:.2%}" if np.isfinite(hr) else "")
                    tf.append(f"{fr:.2%}" if np.isfinite(fr) else "")
                else:
                    rh.append(np.nan)
                    rf.append(np.nan)
                    th.append("")
                    tf.append("")
            z_hit.append(rh)
            text_hit.append(th)
            z_fake.append(rf)
            text_fake.append(tf)

        hit_vals = [v for row in z_hit for v in row if np.isfinite(v)]
        if hit_vals:
            hmin, hmax = min(hit_vals), max(hit_vals)
            pad = 0.05 * (hmax - hmin) if hmax != hmin else 0.01
            fig_h = go.Figure(data=go.Heatmap(z=z_hit, x=th_labels, y=frac_labels, text=text_hit, texttemplate="%{text}", colorscale="Blues", colorbar=dict(title="Hit rate"), zmin=max(0, hmin - pad), zmax=min(1, hmax + pad)))
            fig_h.update_layout(title=f"Hit-rate heatmap — {_format_method_display(method)}", xaxis_title="Fake-hit threshold", yaxis_title="Sample fraction", template="plotly_white")
            sections.append(pio.to_html(fig_h, include_plotlyjs=False, full_html=False))

        if has_fake:
            fake_vals = [v for row in z_fake for v in row if np.isfinite(v)]
            if fake_vals:
                fmin, fmax = min(fake_vals), max(fake_vals)
                pad = 0.05 * (fmax - fmin) if fmax != fmin else 0.01
                fig_f = go.Figure(data=go.Heatmap(z=z_fake, x=th_labels, y=frac_labels, text=text_fake, texttemplate="%{text}", colorscale="Reds", colorbar=dict(title="Fake-hit rate"), zmin=max(0, fmin - pad), zmax=min(1, fmax + pad)))
                fig_f.update_layout(title=f"Fake-hit-rate heatmap — {_format_method_display(method)}", xaxis_title="Fake-hit threshold", yaxis_title="Sample fraction", template="plotly_white")
                sections.append(pio.to_html(fig_f, include_plotlyjs=False, full_html=False))
    return sections


def _sampling_method_from_mode(mode: str) -> str:
    """Extract sampling method for display: 'fec_simple_farthest_point' -> 'farthest_point'; baseline unchanged."""
    if not isinstance(mode, str):
        return ""
    if mode == "baseline" or mode.startswith("baseline"):
        return ""
    if mode.startswith("fec_simple_"):
        return mode[len("fec_simple_"):]
    if mode.startswith("fec_"):
        return mode[len("fec_"):]
    return mode


def _build_combined_summary(
    base_summary_agg: pd.DataFrame,
    fec_summary_agg: pd.DataFrame,
    sum_all_base: pd.DataFrame | None = None,
    sum_all_fec: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combined baseline + FEC table; speedup = baseline total_time_sec / FEC total_time_fair_sec. Optional per-run data for speedup_pvalue.
    Includes columns: sampling_method, sampling_fraction, threshold (empty/NaN for baseline)."""
    if base_summary_agg.empty:
        return pd.DataFrame()
    base_row = base_summary_agg.iloc[0]
    baseline_mae = float(base_row.get("final_test_mae_mean", np.nan))
    baseline_acc = float(base_row.get("final_test_accuracy_mean", np.nan))
    baseline_time = float(base_row.get("total_time_sec_mean", np.nan))

    identity = ["source", "sampling_method", "mode", "sampling_fraction", "sample_fraction", "threshold", "fake_hit_threshold"]
    key_metrics = ["final_test_mae_mean", "final_test_mae_std", "final_test_accuracy_mean", "final_test_accuracy_std", "total_time_sec_mean", "total_time_sec_std"]
    comparison = ["speedup", "speedup_pvalue", "delta_mae_vs_baseline", "delta_accuracy_vs_baseline"]
    rows = []

    bline = {
        "source": "baseline",
        "sampling_method": "",
        "mode": base_row.get("mode", "baseline"),
        "sampling_fraction": np.nan,
        "sample_fraction": 1.0,
        "threshold": np.nan,
        "fake_hit_threshold": np.nan,
        "speedup": np.nan,
        "speedup_pvalue": np.nan,
        "delta_mae_vs_baseline": 0.0,
        "delta_accuracy_vs_baseline": 0.0,
    }
    for c in key_metrics:
        bline[c] = base_row.get(c, np.nan)
    rows.append(bline)

    baseline_times = None
    if sum_all_base is not None and "total_time_sec" in sum_all_base.columns:
        baseline_times = np.asarray(sum_all_base["total_time_sec"].dropna(), dtype=float)

    for _, row in fec_summary_agg.iterrows():
        mae = float(row.get("final_test_mae_mean", np.nan))
        acc = float(row.get("final_test_accuracy_mean", np.nan))
        fair_time = _get_fec_fair_time(row)
        speedup = baseline_time / fair_time if np.isfinite(fair_time) and fair_time > 0 and np.isfinite(baseline_time) else np.nan
        delta_mae = mae - baseline_mae if np.isfinite(mae) and np.isfinite(baseline_mae) else np.nan
        delta_acc = acc - baseline_acc if np.isfinite(acc) and np.isfinite(baseline_acc) else np.nan

        speedup_pvalue = np.nan
        if baseline_times is not None and sum_all_fec is not None and baseline_times.size >= 2:
            mode, frac, th = row["mode"], float(row["sample_fraction"]), row.get("fake_hit_threshold", np.nan)
            mask_m = sum_all_fec["mode"] == mode
            mask_f = np.isclose(sum_all_fec["sample_fraction"].astype(float), frac, rtol=_FLOAT_RTOL)
            if np.isnan(th):
                mask_th = sum_all_fec["fake_hit_threshold"].isna()
            else:
                mask_th = np.isclose(sum_all_fec["fake_hit_threshold"].astype(float), th, rtol=_FLOAT_RTOL)
            sub = sum_all_fec.loc[mask_m & mask_f & mask_th]
            if sub.shape[0] >= 2:
                if "total_time_fair_sec" in sub.columns:
                    fec_times = np.asarray(sub["total_time_fair_sec"].dropna(), dtype=float)
                else:
                    t = np.asarray(sub["total_time_sec"], dtype=float)
                    f = np.asarray(sub.get("fake_eval_time_sec", 0), dtype=float)
                    fec_times = np.maximum(0.0, t - f)
                    fec_times = fec_times[np.isfinite(fec_times)]
                if fec_times.size >= 2:
                    speedup_pvalue = _speedup_pvalue(baseline_times, fec_times)

        frac_val = float(row["sample_fraction"])
        th_val = row.get("fake_hit_threshold", np.nan)
        try:
            th_val = float(th_val) if (th_val is not None and not (isinstance(th_val, float) and np.isnan(th_val))) else np.nan
        except (TypeError, ValueError):
            th_val = np.nan
        rec = {
            "source": "FEC",
            "sampling_method": _sampling_method_from_mode(row["mode"]),
            "mode": row["mode"],
            "sampling_fraction": frac_val,
            "sample_fraction": frac_val,
            "threshold": th_val,
            "fake_hit_threshold": th_val,
            "speedup": speedup,
            "speedup_pvalue": speedup_pvalue,
            "delta_mae_vs_baseline": delta_mae,
            "delta_accuracy_vs_baseline": delta_acc,
        }
        for c in key_metrics:
            rec[c] = row.get(c, np.nan)
        rows.append(rec)

    return pd.DataFrame.from_records(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple baseline vs FEC report (same structure as FEC_report.py). Speedup = baseline total_time_sec / FEC total_time_fair_sec.")
    parser.add_argument("experiment_root", type=str, help="Path to results_simple/<dataset>_Gen_<G>_Pop_<P>/ with 'baseline' and 'FEC' subdirs.")
    args = parser.parse_args()

    root = Path(args.experiment_root)
    baseline_dir = root / "baseline"
    fec_dir = root / "FEC"
    if not baseline_dir.is_dir() or not fec_dir.is_dir():
        raise SystemExit(f"Need 'baseline' and 'FEC' under {root}")

    print("Loading baseline CSVs ...")
    gen_all_base, sum_all_base = _load_baseline_csvs(baseline_dir)
    print("Loading FEC CSVs ...")
    gen_all_fec, sum_all_fec = _load_fec_csvs(fec_dir)

    print("Aggregating baseline ...")
    gen_agg_base = _aggregate_generation_stats(gen_all_base)
    sum_agg_base = _aggregate_summary_stats(sum_all_base)
    print("Aggregating FEC ...")
    gen_agg_fec = _aggregate_fec_generation_stats(gen_all_fec)
    sum_agg_fec = _aggregate_fec_summary_stats(sum_all_fec)

    methods = sorted(gen_agg_fec["mode"].dropna().unique().tolist())
    thresholds, fractions_per_threshold = _get_thresholds_and_fractions(gen_agg_fec)
    all_fractions = sorted(set(f for fracs in fractions_per_threshold.values() for f in fracs)) if fractions_per_threshold else []
    summary_thresholds = _get_sorted_thresholds_from_summary(sum_agg_fec) or []

    # Aggregated evolution CSVs (per-generation metrics, mean/std across runs)
    gen_agg_base_path = baseline_dir / "generation_stats_aggregated.csv"
    gen_agg_base.to_csv(gen_agg_base_path, index=False)
    print(f"Saved {gen_agg_base_path}")

    sum_agg_base_path = baseline_dir / "summary_aggregated.csv"
    sum_agg_base.to_csv(sum_agg_base_path, index=False)
    print(f"Saved {sum_agg_base_path}")

    gen_agg_fec_path = fec_dir / "generation_stats_aggregated_FEC.csv"
    gen_agg_fec.to_csv(gen_agg_fec_path, index=False)
    print(f"Saved {gen_agg_fec_path}")

    sum_agg_fec_path = fec_dir / "summary_aggregated_FEC.csv"
    sum_agg_fec.to_csv(sum_agg_fec_path, index=False)
    print(f"Saved {sum_agg_fec_path}")

    # Reload aggregated evolution CSVs for charting (ensures charts use exact same data as CSV; fixes dtype/JSON issues)
    gen_agg_base = pd.read_csv(gen_agg_base_path)
    gen_agg_fec = pd.read_csv(gen_agg_fec_path)
    if "sample_fraction" in gen_agg_fec.columns:
        gen_agg_fec["sample_fraction"] = gen_agg_fec["sample_fraction"].astype(float)
    if "fake_hit_threshold" in gen_agg_fec.columns:
        gen_agg_fec["fake_hit_threshold"] = pd.to_numeric(gen_agg_fec["fake_hit_threshold"], errors="coerce")

    # Combined summary CSV (speedup = baseline_time / FEC_total_time_fair_sec; speedup_pvalue from t-test)
    combined = _build_combined_summary(
        sum_agg_base, sum_agg_fec,
        sum_all_base=sum_all_base, sum_all_fec=sum_all_fec,
    )
    if not combined.empty:
        combined.to_csv(root / "summary_baseline_vs_FEC.csv", index=False)
        print(f"Saved {root / 'summary_baseline_vs_FEC.csv'}")

    # HTML: Summary table then aggregated evolution, then per threshold/fraction, then across-fraction/threshold
    sections = []
    sections.append("<h2>Aggregated evolution (training and test by fraction and threshold)</h2>")
    sections.append("<p>Mean across runs; each line is one (fraction, threshold) configuration.</p>")
    sections.extend(_build_aggregated_evolution_figs(gen_agg_base, gen_agg_fec, methods, thresholds, fractions_per_threshold))

    sections.append("<h2>Training and test MAE per threshold and fraction</h2>")
    for th in thresholds:
        th_label = _format_threshold_label(th)
        sections.append(f"<h3>Threshold {th_label}</h3>")
        for frac in fractions_per_threshold.get(th, []):
            sections.append(f"<h4>Fraction {frac:.0%}</h4>")
            sections.extend(_build_training_and_test_figs(gen_agg_base, gen_agg_fec, th, frac, methods))

    sections.append("<h2>Across fractions</h2>")
    sections.extend(_build_cross_fraction_figs(sum_agg_fec, sum_agg_base, all_fractions, methods))

    if summary_thresholds:
        sections.append("<h2>Across thresholds</h2>")
        sections.extend(_build_cross_threshold_figs(sum_agg_fec, sum_agg_base, summary_thresholds, methods, all_fractions))
        sections.append("<h2>Speedup heatmaps (fraction × threshold)</h2>")
        sections.extend(_build_speedup_heatmaps(sum_agg_fec, sum_agg_base, methods, all_fractions, summary_thresholds))
        sections.append("<h2>Hit-rate and fake-hit-rate heatmaps (fraction × threshold)</h2>")
        sections.extend(_build_hit_and_fake_heatmaps(sum_agg_fec, methods, all_fractions, summary_thresholds))

    html = (
        "<html><head><meta charset='utf-8' /><title>Baseline vs FEC (Simple Pipeline Report)</title>"
        "<script src='https://cdn.plot.ly/plotly-2.27.0.min.js'></script></head><body>"
        "<h1>Baseline vs FEC (Simple Pipeline Report)</h1>"
        "<p>Speedup = baseline total_time_sec / FEC total_time_fair_sec (fair time excludes fake-hit evaluation). "
        "speedup_pvalue: one-sided t-test (H1: baseline mean time &gt; FEC mean time); p &lt; 0.05 indicates significant speedup.</p>"
        "<h2>Summary table</h2>"
        + (combined.to_html(index=False, float_format=lambda x: f"{x:.5g}") if not combined.empty else "<p>No data.</p>")
        + "".join(sections)
        + "</body></html>"
    )
    html_path = root / "FEC_report_simple.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"Saved {html_path}")


if __name__ == "__main__":
    main()
