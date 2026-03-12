from __future__ import annotations

"""
Run one baseline (no FEC) GE run on the full training data.

CLI:
    python baseline_runs.py --run-index 1 --base-seed 42

Seed used for this run:
    seed = base_seed + (run_index - 1)

Each run:
    - uses the full train set (no sampling, no cache)
    - computes per-generation stats and final test MAE

Outputs (all runs share the same experiment directory):
    results/baseline/<dataset>_Gen_<G>_Pop_<P>_Run_<N_from_config>/
        generation_stats_run<run_index>.csv
        summary_run<run_index>.csv
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import grape.grape as grape

from config import CONFIG
from util import ExperimentResult, load_dataset, load_operators, run_baseline_experiment


def _run_one_baseline(
    run_index: int,
    seed: int,
    experiment_dir: Path,
) -> None:
    """Run a single baseline experiment (no FEC) with a given seed."""
    cfg = CONFIG.copy()
    cfg["evolution.n_runs"] = 1
    cfg["evolution.random_seed"] = seed
    cfg["fec.enabled"] = False
    cfg["fec.sample_fraction"] = None
    cfg["fec.sample_size"] = 0

    X, y = load_dataset(cfg)
    operators = load_operators(Path("operators"))
    grammar_file = Path("grammars") / cfg["grammar.file"]
    grammar = grape.Grammar(str(grammar_file))

    print(
        f"\n=== Baseline run {run_index} / seed={seed} "
        f"dataset={cfg['dataset.file']} G={cfg['evolution.generations']} "
        f"Pop={cfg['evolution.population']} ==="
    )

    result: ExperimentResult = run_baseline_experiment(
        cfg=cfg,
        run_name_suffix=f"baseline_run{run_index}",
        X=X,
        y=y,
        grammar=grammar,
        operators=operators,
        results_root=experiment_dir,
    )

    # Build per-generation DataFrame and summary (similar to FEC_vs_baseline).
    if not result.logbooks or not result.per_run_tables:
        return

    logbook = result.logbooks[0]
    df = result.per_run_tables[0].copy()

    # Tag run/mode/fraction (run column may already exist, so just overwrite)
    df["run"] = run_index
    df["mode"] = "baseline"
    df["sample_fraction"] = 1.0

    # Total time
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
        "mode": "baseline",
        "sample_fraction": 1.0,
        "final_test_mae": final_test_mae,
        "final_test_accuracy": final_test_accuracy,
        "total_time_sec": total_time_sec,
        "hits": 0,
        "misses": 0,
        "fake_hits": 0,
        "hit_rate_overall": 0.0,
    }

    # Save per-generation and summary CSVs
    gen_path = experiment_dir / f"generation_stats_run{run_index}.csv"
    df.to_csv(gen_path, index=False)

    summary_df = pd.DataFrame([summary_row])
    summary_path = experiment_dir / f"summary_run{run_index}.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved baseline per-generation stats to {gen_path}")
    print(f"Saved baseline summary to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one baseline (no FEC) GE run on full training data."
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
        default=int(CONFIG.get("evolution.random_seed", 42)),
        help="Base RNG seed; run_seed = base_seed + (run_index-1).",
    )
    args = parser.parse_args()

    run_index = args.run_index
    base_seed = args.base_seed
    if run_index < 1:
        raise SystemExit("run-index must be >= 1")

    dataset_stem = Path(CONFIG.get("dataset.file", "data")).stem
    n_gen = CONFIG.get("evolution.generations", 0)
    pop = CONFIG.get("evolution.population", 0)

    # Common root: results/<dataset>_Gen_<G>_Pop_<P>/baseline
    results_root = Path("results") / f"{dataset_stem}_Gen_{n_gen}_Pop_{pop}"
    experiment_dir = results_root / "baseline"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"Baseline experiment directory: {experiment_dir}")

    seed = base_seed + (run_index - 1)
    _run_one_baseline(run_index=run_index, seed=seed, experiment_dir=experiment_dir)


if __name__ == "__main__":
    main()

