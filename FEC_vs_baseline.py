from __future__ import annotations

"""
Single-run baseline vs FEC script.

This script is a lightweight companion to ``simple_union_vs_baseline.py``.
Instead of running many independent runs and aggregating them internally,
it runs **one** full evolution run for:

    - Baseline (no FEC, full training set)
    - Each enabled FEC sampling configuration in ``config.CONFIG``
      across all fractions in ``CONFIG["fec.sample_sizes"]``

The intent is that you can:
    - Launch this script many times in parallel on different CPU cores
      (each with a different ``--run-index``), and then
    - Use ``FEC_report.py`` to aggregate all per-run CSVs into final
      summary tables and charts (mean/std across runs).

Outputs (per invocation / run-index) are written under:

    results/FEC_vs_baseline/

Files created:
    - generation_stats_run{run}.csv
        Per-generation metrics for baseline + all FEC configs for this run.
    - summary_run{run}.csv
        One row per (mode, sample_fraction) summarising final test fitness,
        accuracy, total runtime, and overall cache hit/fake-hit statistics.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import grape.grape as grape

from config import CONFIG
from util import (
    ExperimentResult,
    load_dataset,
    load_operators,
    run_configured_experiment,
)


def _collect_generation_stats_for_run(
    exp_result: ExperimentResult,
    mode: str,
    sample_fraction: float | None,
    global_run_index: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Take an ExperimentResult with a single run and return:
      - Per-generation table (DataFrame) tagged with mode/sample_fraction/run
      - Per-config summary dict (final fitness, accuracy, runtime, cache stats)
    """
    if not exp_result.logbooks:
        raise RuntimeError(f"No logbooks found for mode={mode!r}")

    logbook = exp_result.logbooks[0]
    cache_stats = exp_result.cache_stats[0] if exp_result.cache_stats else {}

    # Per-generation DEAP stats table for this run
    if not exp_result.per_run_tables:
        raise RuntimeError(f"No per-run table found for mode={mode!r}")
    df = exp_result.per_run_tables[0].copy()

    # Overwrite run index with the global run index provided via CLI
    if "run" in df.columns:
        df["run"] = int(global_run_index)
    else:
        df.insert(0, "run", int(global_run_index))

    # Tag configuration-level info
    df["mode"] = mode
    df["sample_fraction"] = float(sample_fraction) if sample_fraction is not None else np.nan

    # ----------------------------------------------------------------------
    # Cache-derived, per-generation hit/fake-hit metrics (FEC only)
    # ----------------------------------------------------------------------
    gens = list(df["gen"]) if "gen" in df.columns else list(range(len(df)))

    gen_hits = np.asarray(cache_stats.get("gen_hits", [0] * len(gens)), dtype=float)
    gen_misses = np.asarray(cache_stats.get("gen_misses", [0] * len(gens)), dtype=float)
    gen_fake = np.asarray(cache_stats.get("gen_fake_hits", [0] * len(gens)), dtype=float)
    gen_hits_just_struct = np.asarray(
        cache_stats.get("gen_hits_just_structural", [0] * len(gens)), dtype=float
    )
    gen_hits_behav_no_struct = np.asarray(
        cache_stats.get("gen_hits_behavioural_without_structural", [0] * len(gens)),
        dtype=float,
    )

    # Ensure all arrays have the same length as df
    n_gen = len(df)
    def _pad(arr: np.ndarray) -> np.ndarray:
        if arr.size >= n_gen:
            return arr[:n_gen]
        out = np.zeros(n_gen, dtype=float)
        out[: arr.size] = arr
        return out

    gen_hits = _pad(gen_hits)
    gen_misses = _pad(gen_misses)
    gen_fake = _pad(gen_fake)
    gen_hits_just_struct = _pad(gen_hits_just_struct)
    gen_hits_behav_no_struct = _pad(gen_hits_behav_no_struct)

    hit_rate = np.zeros(n_gen, dtype=float)
    fake_hit_rate = np.zeros(n_gen, dtype=float)
    rate_behav_or_both = np.zeros(n_gen, dtype=float)
    rate_behav_no_struct = np.zeros(n_gen, dtype=float)
    rate_just_struct = np.zeros(n_gen, dtype=float)

    for i in range(n_gen):
        h = float(gen_hits[i])
        m = float(gen_misses[i])
        f = float(gen_fake[i])
        j = float(gen_hits_just_struct[i])
        b = float(gen_hits_behav_no_struct[i])
        lookups = h + m
        if lookups > 0.0:
            hit_rate[i] = h / lookups
            rate_behav_or_both[i] = h / lookups
            rate_behav_no_struct[i] = b / lookups
            rate_just_struct[i] = j / lookups
        else:
            hit_rate[i] = 0.0
            rate_behav_or_both[i] = 0.0
            rate_behav_no_struct[i] = 0.0
            rate_just_struct[i] = 0.0
        if h > 0.0:
            fake_hit_rate[i] = f / h
        else:
            fake_hit_rate[i] = 0.0

    df["hit_rate"] = hit_rate
    df["fake_hit_rate"] = fake_hit_rate
    df["rate_behavioural_or_both"] = rate_behav_or_both
    df["rate_behavioural_without_structural"] = rate_behav_no_struct
    df["rate_just_structural"] = rate_just_struct

    # ----------------------------------------------------------------------
    # Per-config summary (one row per (mode, sample_fraction))
    # ----------------------------------------------------------------------
    # Total runtime for this run (sum of generation_time in logbook)
    total_time_sec = None
    try:
        gen_times = np.asarray(logbook.select("generation_time"), dtype=float)
        if gen_times.size > 0:
            total_time_sec = float(np.nansum(gen_times))
    except (KeyError, IndexError):
        total_time_sec = None

    final_test_fitness = None
    if "fitness_test" in df.columns and not df["fitness_test"].empty:
        final_test_fitness = float(df["fitness_test"].iloc[-1])
    accuracy = 1.0 - final_test_fitness if final_test_fitness is not None else None

    hits_total = float(cache_stats.get("hits", 0.0) or 0.0)
    misses_total = float(cache_stats.get("misses", 0.0) or 0.0)
    fake_hits_total = float(cache_stats.get("fake_hits", 0.0) or 0.0)
    total_lookups = hits_total + misses_total

    overall_hit_rate = (hits_total / total_lookups) if total_lookups > 0.0 else 0.0
    overall_fake_hit_rate = (fake_hits_total / hits_total) if hits_total > 0.0 else 0.0

    hits_just_struct_total = float(cache_stats.get("hits_just_structural", 0.0) or 0.0)
    hits_behav_no_struct_total = float(
        cache_stats.get("hits_behavioural_without_structural", 0.0) or 0.0
    )
    rate_behav_or_both_overall = (
        hits_total / total_lookups if total_lookups > 0.0 else 0.0
    )
    rate_behav_no_struct_overall = (
        hits_behav_no_struct_total / total_lookups if total_lookups > 0.0 else 0.0
    )
    rate_just_struct_overall = (
        hits_just_struct_total / total_lookups if total_lookups > 0.0 else 0.0
    )

    summary_row: Dict[str, Any] = {
        "run": int(global_run_index),
        "mode": mode,
        "sample_fraction": float(sample_fraction) if sample_fraction is not None else np.nan,
        "final_test_mae": final_test_fitness,
        "final_test_accuracy": accuracy,
        "total_time_sec": total_time_sec,
        "hits": hits_total,
        "misses": misses_total,
        "fake_hits": fake_hits_total,
        "hit_rate_overall": overall_hit_rate,
        "fake_hit_rate_overall": overall_fake_hit_rate,
        "rate_behavioural_or_both_overall": rate_behav_or_both_overall,
        "rate_behavioural_without_structural_overall": rate_behav_no_struct_overall,
        "rate_just_structural_overall": rate_just_struct_overall,
    }

    return df, summary_row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single baseline + FEC evolution run and save detailed CSVs for later aggregation."
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=1,
        help="Global run index (1-based). Different values should be used on different cores.",
    )
    args = parser.parse_args()

    run_index = int(args.run_index)
    if run_index < 1:
        raise SystemExit("run-index must be >= 1")

    # Derive the seed for this global run index so that each invocation
    # on different cores gets different data splits but baseline and FEC
    # within the same invocation share the same splits.
    base_seed = int(CONFIG.get("evolution.random_seed", 42))
    run_seed = base_seed + (run_index - 1)

    print(f"Running FEC_vs_baseline for global run-index={run_index}, seed={run_seed}")

    # ------------------------------------------------------------------
    # Determine experiment directory (shared folder for this config)
    # ------------------------------------------------------------------
    root_base = Path("results") / "FEC_vs_baseline"
    dataset_stem = Path(CONFIG.get("dataset.file", "data")).stem
    n_gen = CONFIG.get("evolution.generations", 0)
    pop = CONFIG.get("evolution.population", 0)
    total_runs = CONFIG.get("evolution.n_runs", 1)
    # Default folder pattern: dataset_Gen_#_Pop_#_Run_#
    folder_name = f"{dataset_stem}_Gen_{n_gen}_Pop_{pop}_Run_{total_runs}"
    experiment_dir = root_base / folder_name

    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment directory: {experiment_dir}")

    # ------------------------------------------------------------------
    # Shared resources: dataset, operators, grammar
    # ------------------------------------------------------------------
    cfg_base = CONFIG.copy()
    cfg_base["evolution.n_runs"] = 1
    cfg_base["evolution.random_seed"] = run_seed

    X, y = load_dataset(cfg_base)
    operators = load_operators(Path("operators"))
    grammar_file = Path("grammars") / cfg_base["grammar.file"]
    grammar = grape.Grammar(str(grammar_file))

    # ------------------------------------------------------------------
    # Baseline (no FEC)
    # ------------------------------------------------------------------
    baseline_cfg = dict(cfg_base)
    baseline_cfg["fec.enabled"] = False
    # Ensure any old-style FEC keys do not accidentally enable behaviour
    baseline_cfg["fec.sample_fraction"] = None
    baseline_cfg["fec.sample_size"] = 0

    baseline_result = run_configured_experiment(
        baseline_cfg,
        run_name_suffix=f"baseline_run{run_index}",
        X=X,
        y=y,
        grammar=grammar,
        operators=operators,
        results_root=experiment_dir,
    )
    baseline_df, baseline_summary = _collect_generation_stats_for_run(
        baseline_result, mode="baseline", sample_fraction=1.0, global_run_index=run_index
    )

    # ------------------------------------------------------------------
    # FEC configurations: from CONFIG["fec.sampling_methods.enabled"]
    # and CONFIG["fec.sample_sizes"]
    # ------------------------------------------------------------------
    enabled_cfg = cfg_base.get("fec.sampling_methods.enabled", {})
    sample_fractions: List[float] = list(cfg_base.get("fec.sample_sizes", []))
    sample_fractions = [float(f) for f in sample_fractions]

    method_labels: List[Tuple[str, str]] = []
    for method_name, flag in enabled_cfg.items():
        if not flag:
            continue
        if method_name == "union":
            union_list = cfg_base.get("fec.sampling_methods.union", [])
            if not union_list:
                continue
            label = "fec_union(" + "+".join(union_list) + ")"
            method_labels.append((label, "union"))
        else:
            method_labels.append((f"fec_{method_name}", method_name))

    all_gen_rows: List[pd.DataFrame] = [baseline_df]
    all_summary_rows: List[Dict[str, Any]] = [baseline_summary]

    for method_label, method_name in method_labels:
        for frac in sample_fractions:
            fec_cfg = dict(cfg_base)
            fec_cfg["fec.enabled"] = True
            fec_cfg["fec.sampling_method"] = method_name
            # Old API compatibility: we drive via fraction, sample_size is derived inside util.run_configured_experiment
            fec_cfg["fec.sample_fraction"] = float(frac)

            run_suffix = f"{method_label}_frac_{int(round(frac * 100))}_run{run_index}"
            print(
                f"\n=== Running FEC config: mode={method_label}, "
                f"sampling_method={method_name}, fraction={frac:.2f}, run={run_index} ==="
            )
            fec_result = run_configured_experiment(
                fec_cfg,
                run_name_suffix=run_suffix,
                X=X,
                y=y,
                grammar=grammar,
                operators=operators,
                results_root=experiment_dir,
            )
            fec_df, fec_summary = _collect_generation_stats_for_run(
                fec_result,
                mode=method_label,
                sample_fraction=frac,
                global_run_index=run_index,
            )
            all_gen_rows.append(fec_df)
            all_summary_rows.append(fec_summary)

    # ------------------------------------------------------------------
    # Write per-run outputs
    # ------------------------------------------------------------------
    gen_all = pd.concat(all_gen_rows, ignore_index=True)
    gen_path = experiment_dir / f"generation_stats_run{run_index}.csv"
    gen_all.to_csv(gen_path, index=False)
    print(f"Saved per-generation stats for run {run_index} to {gen_path}")

    summary_df = pd.DataFrame(all_summary_rows)
    summary_path = experiment_dir / f"summary_run{run_index}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved per-config summary for run {run_index} to {summary_path}")


if __name__ == "__main__":
    main()

