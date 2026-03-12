from __future__ import annotations

"""
Minimal FEC runner for fair runtime comparison.

This script mirrors baseline_runs_simple.py but enables caching:
  - Same GE pipeline.
  - FEC is enabled (behaviour-only key).
  - Fake-hit analysis and detailed event logging are turned OFF.
  - Measures:
      * total_time_sec          = sum of generation_time from the logbook
      * total_wall_time_sec     = wall-clock time around the whole experiment

Outputs go under:
  results_simple/<dataset>_Gen_<G>_Pop_<P>/FEC/
    generation_stats_<method>_frac_<p>_run<run>.csv
    summary_<method>_frac_<p>_run<run>.csv
"""

import argparse
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import grape.grape as grape

from config import CONFIG
from util import ExperimentResult, load_dataset, load_operators, run_fec_experiment


VALID_SAMPLING_METHODS = {
    "kmeans",
    "kmedoids",
    "farthest_point",
    "stratified",
    "random",
    "union",
}


def _run_one_fec_simple(
    run_index: int,
    base_seed: int,
    sample_fraction: float,
    sampling_method: str,
    fake_hit_threshold: float,
    experiment_dir: Path,
) -> None:
    """Run a single FEC-enabled experiment for one sampling method and fraction (simple, no extra analysis)."""
    combo = (int(base_seed), int(run_index), str(sampling_method), float(sample_fraction))
    seed = abs(hash(combo)) % (2**31 - 1)

    cfg = CONFIG.copy()
    cfg["evolution.n_runs"] = 1
    cfg["evolution.random_seed"] = seed

    # FEC configuration (minimal: cache only, no fake-hit re-eval, no detailed events).
    cfg["fec.enabled"] = True
    cfg["fec.sampling_method"] = sampling_method
    cfg["fec.sample_fraction"] = float(sample_fraction)
    cfg["fec.structural_similarity"] = False
    cfg["fec.behavior_similarity"] = True
    cfg["fec.evaluate_fake_hits"] = False
    cfg["fec.record_detailed_events"] = False
    cfg["fec.fake_hit_threshold"] = float(fake_hit_threshold)

    # Avoid any tracking / plotting overhead during the run.
    cfg["output.track_individuals"] = False
    cfg["output.plot"] = False
    cfg["output.save_individuals_csv"] = False

    X, y = load_dataset(cfg)
    operators = load_operators(Path("operators"))
    grammar_file = Path("grammars") / cfg["grammar.file"]
    grammar = grape.Grammar(str(grammar_file))

    print(
        f"\n=== FEC_simple run {run_index} / seed={seed} "
        f"method={sampling_method} frac={sample_fraction:.3f} "
        f"dataset={cfg['dataset.file']} G={cfg['evolution.generations']} "
        f"Pop={cfg['evolution.population']} ==="
    )

    th_str = f"{fake_hit_threshold:g}"
    th_tag = th_str.replace("-", "m").replace(".", "p")
    frac_pct = int(round(sample_fraction * 100))
    file_tag = f"{sampling_method}_frac_{frac_pct}_th_{th_tag}"
    run_suffix = f"FEC_simple_{file_tag}_run{run_index}"

    t0 = time.perf_counter()
    result: ExperimentResult = run_fec_experiment(
        cfg=cfg,
        run_name_suffix=run_suffix,
        X=X,
        y=y,
        grammar=grammar,
        operators=operators,
        results_root=experiment_dir,
    )
    t1 = time.perf_counter()
    total_wall_time_sec = float(t1 - t0)

    if not result.logbooks or not result.per_run_tables:
        return

    logbook = result.logbooks[0]
    df = result.per_run_tables[0].copy()

    mode_label = f"fec_simple_{sampling_method}"
    df["run"] = run_index
    df["mode"] = mode_label
    df["sample_fraction"] = float(sample_fraction)

    # Total time from logbook (evolution + caching only)
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

    # We can optionally still use cache_stats for basic hit/miss counts if present.
    cache_stats = result.cache_stats[0] if result.cache_stats else {}
    hits_total = float(cache_stats.get("hits", 0.0) or 0.0)
    misses_total = float(cache_stats.get("misses", 0.0) or 0.0)
    total_lookups = hits_total + misses_total
    overall_hit_rate = hits_total / total_lookups if total_lookups > 0.0 else 0.0
    fake_eval_time_sec = float(cache_stats.get("fake_eval_time_sec", 0.0) or 0.0)

    summary_row: Dict[str, object] = {
        "run": run_index,
        "mode": mode_label,
        "sample_fraction": float(sample_fraction),
        "fake_hit_threshold": float(fake_hit_threshold),
        "final_test_mae": final_test_mae,
        "final_test_accuracy": final_test_accuracy,
        # total_time_sec = sum of generation_time (evolution + caching only)
        "total_time_sec": total_time_sec,
        # total_wall_time_sec = wall-clock around the whole FEC pipeline
        "total_wall_time_sec": total_wall_time_sec,
        # Time spent ONLY on fake-hit evaluation (extra full evals on hits)
        "fake_eval_time_sec": fake_eval_time_sec,
        "hits": hits_total,
        "misses": misses_total,
        "hit_rate_overall": overall_hit_rate,
    }

    gen_path = experiment_dir / f"generation_stats_{file_tag}_run{run_index}.csv"
    df.to_csv(gen_path, index=False)

    summary_df = pd.DataFrame([summary_row])
    summary_path = experiment_dir / f"summary_{file_tag}_run{run_index}.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved FEC_simple per-generation stats to {gen_path}")
    print(f"Saved FEC_simple summary to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one FEC-enabled GE run (simple pipeline) for a single sampling method and fraction."
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
        help="Base RNG seed used together with run-index and sample fraction.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        required=True,
        help="Fraction of training data to use for FEC sampling (0 < f <= 1).",
    )
    parser.add_argument(
        "--sampling-method",
        type=str,
        required=True,
        help=f"Sampling method name: one of {sorted(VALID_SAMPLING_METHODS)}.",
    )
    parser.add_argument(
        "--fake-hit-threshold",
        type=float,
        default=float(CONFIG.get("fec.fake_hit_threshold", 1e-5)),
        help=(
            "Fake-hit threshold parameter (no re-eval in simple mode, "
            "recorded for analysis and filenames only)."
        ),
    )
    args = parser.parse_args()

    run_index = args.run_index
    base_seed = args.base_seed
    sample_fraction = float(args.sample_fraction)
    sampling_method = args.sampling_method.strip()
    fake_hit_threshold = float(args.fake_hit_threshold)

    if run_index < 1:
        raise SystemExit("run-index must be >= 1")
    if not (0.0 < sample_fraction <= 1.0):
        raise SystemExit("sample-fraction must be in (0, 1].")
    if sampling_method not in VALID_SAMPLING_METHODS:
        raise SystemExit(f"sampling-method must be one of {sorted(VALID_SAMPLING_METHODS)}.")

    dataset_stem = Path(CONFIG.get("dataset.file", "data")).stem
    n_gen = CONFIG.get("evolution.generations", 0)
    pop = CONFIG.get("evolution.population", 0)

    # Root for simple comparison: results_simple/<dataset>_Gen_<G>_Pop_<P>/FEC/
    results_root = Path("results_simple") / f"{dataset_stem}_Gen_{n_gen}_Pop_{pop}"
    experiment_dir = results_root / "FEC"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"FEC_simple experiment directory: {experiment_dir}")

    _run_one_fec_simple(
        run_index=run_index,
        base_seed=base_seed,
        sample_fraction=sample_fraction,
        sampling_method=sampling_method,
        fake_hit_threshold=fake_hit_threshold,
        experiment_dir=experiment_dir,
    )


if __name__ == "__main__":
    main()

