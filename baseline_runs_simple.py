from __future__ import annotations

"""
Minimal baseline runner (no FEC) for fair runtime comparison.

This script is intentionally simple:
  - Same GE pipeline as FEC, but with caching disabled.
  - Measures:
      * total_time_sec          = sum of generation_time from the logbook
      * total_wall_time_sec     = wall-clock time around the whole experiment

Outputs go under:
  results_simple/<dataset>_Gen_<G>_Pop_<P>/baseline/
    generation_stats_run<run>.csv
    summary_run<run>.csv
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import grape.grape as grape

from util_simple import (
    SimpleExperimentResult,
    load_dataset,
    load_operators,
    run_baseline_experiment_simple,
)

CONFIG_JSON_PATH = Path(__file__).with_name("config.json")


def _load_config_baseline() -> Dict[str, object]:
    payload = json.loads(CONFIG_JSON_PATH.read_text(encoding="utf-8"))
    common = dict(payload.get("COMMON_CONFIG", {}))
    baseline_only = dict(payload.get("BASELINE_SPECIFIC_CONFIG", {}))
    return {**common, **baseline_only}


def _run_one_baseline(
    run_index: int,
    seed: int,
    cfg_template: Dict[str, object],
    experiment_dir: Path,
) -> None:
    """Run a single baseline experiment (no FEC) with a given seed."""
    cfg = cfg_template.copy()
    cfg["evolution.random_seed"] = seed

    X, y = load_dataset(cfg)
    operators = load_operators(Path("operators"))
    grammar_file = Path("grammars") / cfg["grammar.file"]
    grammar = grape.Grammar(str(grammar_file))

    t0 = time.perf_counter()
    result: SimpleExperimentResult = run_baseline_experiment_simple(
        cfg=cfg,
        run_name_suffix=f"baseline_simple_run{run_index}",
        X=X,
        y=y,
        grammar=grammar,
        operators=operators,
        results_root=experiment_dir,
    )
    t1 = time.perf_counter()
    total_wall_time_sec = float(t1 - t0)

    # Simple pipeline returns a single logbook and per-run table.
    if result.logbook is None or result.per_run_table is None or result.per_run_table.empty:
        return

    logbook = result.logbook
    df = result.per_run_table.copy()

    # Tag run/mode/fraction (run column may already exist, so just overwrite)
    df["run"] = run_index
    df["mode"] = "baseline_simple"
    df["sample_fraction"] = 1.0

    # Total time from logbook (evolution loop only)
    total_time_sec = None
    try:
        gen_times = np.asarray(logbook.select("generation_time"), dtype=float)
        if gen_times.size > 0:
            total_time_sec = float(np.nansum(gen_times))
    except (KeyError, IndexError):
        total_time_sec = None

    final_test_mae = None
    if "fitness_test" in df.columns and not df["fitness_test"].empty:
        final_test_mae = float(df["fitness_test"].iloc[-1])
    final_test_accuracy = 1.0 - final_test_mae if final_test_mae is not None else None

    summary_row: Dict[str, object] = {
        "run": run_index,
        "mode": "baseline_simple",
        "sample_fraction": 1.0,
        "final_test_mae": final_test_mae,
        "final_test_accuracy": final_test_accuracy,
        # total_time_sec = sum of generation_time (evolution only)
        "total_time_sec": total_time_sec,
        # total_wall_time_sec = wall-clock around the whole baseline pipeline
        "total_wall_time_sec": total_wall_time_sec,
    }

    # Save per-generation and summary CSVs
    gen_path = experiment_dir / f"generation_stats_run{run_index}.csv"
    df.to_csv(gen_path, index=False)

    summary_df = pd.DataFrame([summary_row])
    summary_path = experiment_dir / f"summary_run{run_index}.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved baseline_simple per-generation stats to {gen_path}")
    print(f"Saved baseline_simple summary to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one baseline (no FEC) GE run on full training data (simple pipeline)."
    )
    parser.add_argument(
        "--run-index",
        type=int,
        required=True,
        help="1-based run index used to derive the seed and output filenames.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help="Base RNG seed (defaults from config); run_seed = base_seed + (run_index-1).",
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        default=None,
        help="Optional dataset CSV filename under ./data to override config default.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default=None,
        help="Optional label column name to override config default.",
    )
    args = parser.parse_args()

    cfg = _load_config_baseline()
    run_index = args.run_index
    base_seed = int(cfg.get("evolution.random_seed", 42)) if args.base_seed is None else int(args.base_seed)
    if args.dataset_file:
        cfg["dataset.file"] = args.dataset_file
    if args.label_column:
        cfg["dataset.label_column"] = args.label_column
    if run_index < 1:
        raise SystemExit("run-index must be >= 1")

    dataset_stem = Path(str(cfg.get("dataset.file", "data"))).stem
    if bool(cfg.get("dataset.smoth_balance", False)):
        dataset_stem = f"{dataset_stem}_SMOTH"
    n_gen = int(cfg.get("evolution.generations", 0))
    pop = int(cfg.get("evolution.population", 0))

    # Root for simple comparison: results_simple/<dataset>_Gen_<G>_Pop_<P>/baseline/
    results_root = Path("results_simple") / f"{dataset_stem}_Gen_{n_gen}_Pop_{pop}"
    experiment_dir = results_root / "baseline"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"Baseline_simple experiment directory: {experiment_dir}")

    seed = base_seed + (run_index - 1)
    _run_one_baseline(run_index=run_index, seed=seed, cfg_template=cfg, experiment_dir=experiment_dir)


if __name__ == "__main__":
    main()

