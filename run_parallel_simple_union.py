#!/usr/bin/env python3
"""
Launch multiple independent runs of simple_union_vs_baseline.py in parallel
OS processes, each with a different base seed, and then aggregate the
resulting fec_vs_baseline_summary.csv files into a single combined CSV.

This keeps each evolutionary run single-process (one core), but allows you to
use many cores on a server by running several experiments at once.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from config import CONFIG


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_BASE = PROJECT_ROOT / "results" / "simple_union_vs_baseline"


def _discover_existing_summaries() -> set[Path]:
    """Return set of all fec_vs_baseline_summary.csv paths currently present."""
    if not RESULTS_BASE.exists():
        return set()
    return set(RESULTS_BASE.glob("*/fec_vs_baseline_summary.csv"))


def _run_one_simple_union(run_index: int, seed: int, python_exec: str) -> Tuple[int, int]:
    """
    Run one instance of simple_union_vs_baseline.py with a specific seed.

    Returns (run_index, exit_code).
    """
    env = os.environ.copy()
    env["FEC_GE_SEED"] = str(seed)

    cmd = [python_exec, "simple_union_vs_baseline.py"]
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    log_dir = RESULTS_BASE / "batch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{run_index:03d}.log"
    with log_path.open("w", encoding="utf-8") as f:
        f.write("COMMAND: " + " ".join(cmd) + "\n")
        f.write(f"SEED: {seed}\n\n")
        f.write("STDOUT:\n")
        f.write(proc.stdout)
        f.write("\n\nSTDERR:\n")
        f.write(proc.stderr)

    return run_index, proc.returncode


def _aggregate_new_summaries(
    old_paths: set[Path],
    batch_tag: str,
) -> Path | None:
    """
    Find all new fec_vs_baseline_summary.csv files created since old_paths,
    concatenate them with an extra 'batch_run' column, and write a combined CSV.
    """
    if not RESULTS_BASE.exists():
        return None

    all_paths = set(RESULTS_BASE.glob("*/fec_vs_baseline_summary.csv"))
    new_paths = sorted(all_paths - old_paths)
    if not new_paths:
        return None

    dfs: List[pd.DataFrame] = []
    for idx, path in enumerate(new_paths, start=1):
        try:
            df = pd.read_csv(path)
            df.insert(0, "batch_run", idx)
            df.insert(1, "source_dir", path.parent.name)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: could not read {path}: {e}", file=sys.stderr)

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    out_dir = RESULTS_BASE / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fec_vs_baseline_summary_combined_{batch_tag}.csv"
    combined.to_csv(out_path, index=False)
    return out_path


def _build_batch_charts(combined_csv: Path, batch_tag: str) -> Path | None:
    """
    Build aggregated charts over all runs from the combined summary CSV.

    Produces one HTML file with:
      - Accuracy vs sample_fraction (mean ± std) per mode.
      - Speedup vs sample_fraction (mean ± std) per mode.
    """
    try:
        df = pd.read_csv(combined_csv)
    except Exception as e:
        print(f"Warning: could not read combined CSV for charts: {e}", file=sys.stderr)
        return None

    required_cols = {"mode", "sample_fraction", "accuracy", "speedup_vs_baseline"}
    if not required_cols.issubset(df.columns):
        print(
            f"Combined CSV missing required columns for charts: "
            f"{required_cols - set(df.columns)}",
            file=sys.stderr,
        )
        return None

    grouped = (
        df.groupby(["mode", "sample_fraction"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_speedup=("speedup_vs_baseline", "mean"),
            std_speedup=("speedup_vs_baseline", "std"),
            n_runs=("accuracy", "size"),
        )
        .sort_values(["mode", "sample_fraction"])
    )

    # Accuracy vs sample_fraction
    fig_acc = go.Figure()
    for mode, sub in grouped.groupby("mode"):
        x = sub["sample_fraction"]
        y = sub["mean_accuracy"]
        err = sub["std_accuracy"].fillna(0.0)
        fig_acc.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=str(mode),
                error_y=dict(type="data", array=err, visible=True),
            )
        )
    fig_acc.update_layout(
        title="Final Accuracy vs Sample Fraction (aggregated over runs)",
        xaxis_title="Sample fraction",
        yaxis_title="Accuracy",
        template="plotly_white",
        hovermode="x unified",
    )

    # Speedup vs sample_fraction
    fig_speed = go.Figure()
    for mode, sub in grouped.groupby("mode"):
        x = sub["sample_fraction"]
        y = sub["mean_speedup"]
        err = sub["std_speedup"].fillna(0.0)
        fig_speed.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=str(mode),
                error_y=dict(type="data", array=err, visible=True),
            )
        )
    fig_speed.update_layout(
        title="Speedup vs Sample Fraction (aggregated over runs)",
        xaxis_title="Sample fraction",
        yaxis_title="Speedup vs baseline",
        template="plotly_white",
        hovermode="x unified",
    )

    sections = [
        pio.to_html(fig_acc, include_plotlyjs="cdn", full_html=False),
        pio.to_html(fig_speed, include_plotlyjs="cdn", full_html=False),
    ]

    out_dir = RESULTS_BASE / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"batch_charts_{batch_tag}.html"

    html = (
        "<html><head>"
        "<meta charset='utf-8' />"
        "<title>simple_union_vs_baseline - Aggregated Results</title>"
        "</head><body>"
        "<h1>simple_union_vs_baseline - Aggregated Results</h1>"
        f"<p>Combined CSV: {combined_csv.name}</p>"
        + "".join(sections)
        + "</body></html>"
    )
    html_path.write_text(html, encoding="utf-8")
    return html_path


def main() -> None:
    # Read batch parameters from CONFIG only.
    total_runs_cfg = CONFIG.get("evolution.n_runs", 1)
    max_parallel_cfg = CONFIG.get("parallel.batch.total_runs")
    base_seed_cfg = CONFIG.get("parallel.batch.base_seed")

    try:
        total_runs = max(1, int(total_runs_cfg))
    except (TypeError, ValueError):
        total_runs = 1

    if max_parallel_cfg is None:
        available = os.cpu_count() or 1
        max_parallel = min(total_runs, available)
    else:
        try:
            max_parallel = max(1, int(max_parallel_cfg))
        except (TypeError, ValueError):
            available = os.cpu_count() or 1
            max_parallel = min(total_runs, available)

    # Safety note if user requested more parallel processes than cores.
    available_cores = os.cpu_count() or 1
    if max_parallel > available_cores:
        print(
            f"Warning: requested {max_parallel} parallel runs but only "
            f"{available_cores} CPU cores detected. Processes will oversubscribe the CPU."
        )

    if base_seed_cfg is None:
        base_seed = int(CONFIG.get("evolution.random_seed", 42))
    else:
        try:
            base_seed = int(base_seed_cfg)
        except (TypeError, ValueError):
            base_seed = int(CONFIG.get("evolution.random_seed", 42))

    python_exec = sys.executable

    print(
        f"Launching {total_runs} runs of simple_union_vs_baseline.py "
        f"with up to {max_parallel} in parallel..."
    )

    existing = _discover_existing_summaries()
    batch_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    futures = []
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        for run_idx in range(total_runs):
            seed = base_seed + 1000 * run_idx
            futures.append(
                executor.submit(_run_one_simple_union, run_idx + 1, seed, python_exec)
            )

        for fut in as_completed(futures):
            run_index, code = fut.result()
            status = "OK" if code == 0 else f"FAILED (exit={code})"
            print(f"[run {run_index:03d}] {status}")

    combined_path = _aggregate_new_summaries(existing, batch_tag)
    if combined_path is not None:
        print(f"Combined summary written to: {combined_path}")
        charts_path = _build_batch_charts(combined_path, batch_tag)
        if charts_path is not None:
            print(f"Aggregated charts written to: {charts_path}")
        else:
            print("Skipped aggregated charts (see warnings above).")
    else:
        print("No new fec_vs_baseline_summary.csv files found to combine.")


if __name__ == "__main__":
    main()

