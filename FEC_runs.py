from __future__ import annotations

"""
Run one FEC-enabled GE run for a single sampling method and fraction.

CLI example:
    python FEC_runs.py \
        --run-index 1 \
        --base-seed 42 \
        --sample-fraction 0.1 \
        --sampling-method farthest_point

Seed used for this run:
    seed = f(base_seed, run_index, sampling_method, sample_fraction)

This script:
    - Enables FEC.
    - Uses the given sampling method and fraction for the centroids.
    - Writes per-generation and per-run summary CSVs under:
        results/FEC/<dataset>_Gen_<G>_Pop_<P>_Run_<n_runs>_frac_<p>_meth_<name>/
"""

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

import grape.grape as grape

from config import CONFIG
from util import ExperimentResult, load_dataset, load_operators, run_fec_experiment


VALID_SAMPLING_METHODS = {"kmeans", "kmedoids", "farthest_point", "stratified", "random", "union"}


def _run_one_fec(
    run_index: int,
    base_seed: int,
    sample_fraction: float,
    sampling_method: str,
    fake_hit_threshold: float,
    experiment_dir: Path,
) -> None:
    """Run a single FEC-enabled experiment for one sampling method and fraction."""
    # Derive a seed that depends on base_seed, run_index, method, and fraction.
    combo = (int(base_seed), int(run_index), str(sampling_method), float(sample_fraction))
    seed = abs(hash(combo)) % (2**31 - 1)

    cfg = CONFIG.copy()
    cfg["evolution.n_runs"] = 1
    cfg["evolution.random_seed"] = seed

    # FEC configuration
    cfg["fec.enabled"] = True
    cfg["fec.sampling_method"] = sampling_method
    cfg["fec.sample_fraction"] = float(sample_fraction)
    # Behaviour-key only (no structural component) as agreed.
    cfg["fec.structural_similarity"] = False
    cfg["fec.behavior_similarity"] = True
    # Respect config: when False, no re-evaluation on cache hits (faster, fake_hits stay 0).
    # When True, we re-evaluate on hits to measure fake-hit rate (slower, for analysis).
    cfg["fec.evaluate_fake_hits"] = bool(CONFIG.get("fec.evaluate_fake_hits", False))
    cfg["fec.fake_hit_threshold"] = float(fake_hit_threshold)

    X, y = load_dataset(cfg)
    operators = load_operators(Path("operators"))
    grammar_file = Path("grammars") / cfg["grammar.file"]
    grammar = grape.Grammar(str(grammar_file))

    print(
        f"\n=== FEC run {run_index} / seed={seed} "
        f"method={sampling_method} frac={sample_fraction:.3f} "
        f"dataset={cfg['dataset.file']} G={cfg['evolution.generations']} "
        f"Pop={cfg['evolution.population']} ==="
    )

    th_str = f"{fake_hit_threshold:g}"
    th_tag = th_str.replace("-", "m").replace(".", "p")
    run_suffix = (
        f"FEC_{sampling_method}_frac_{int(round(sample_fraction * 100))}_"
        f"th_{th_tag}_run{run_index}"
    )
    result: ExperimentResult = run_fec_experiment(
        cfg=cfg,
        run_name_suffix=run_suffix,
        X=X,
        y=y,
        grammar=grammar,
        operators=operators,
        results_root=experiment_dir,
    )

    if not result.logbooks or not result.per_run_tables:
        return

    logbook = result.logbooks[0]
    df = result.per_run_tables[0].copy()

    # Tag run/mode/fraction
    mode_label = f"fec_{sampling_method}"
    # per_run_tables already include a 'run' column; just overwrite to be safe
    df["run"] = run_index
    df["mode"] = mode_label
    df["sample_fraction"] = float(sample_fraction)
    df["fake_hit_threshold"] = float(fake_hit_threshold)

    # Cache stats (overall)
    cache_stats = result.cache_stats[0] if result.cache_stats else {}
    hits_total = float(cache_stats.get("hits", 0.0) or 0.0)
    misses_total = float(cache_stats.get("misses", 0.0) or 0.0)
    fake_hits_total = float(cache_stats.get("fake_hits", 0.0) or 0.0)
    full_evals_total = float(cache_stats.get("full_evals", 0.0) or 0.0)
    sample_evals_total = float(cache_stats.get("sample_evals", 0.0) or 0.0)
    total_lookups = hits_total + misses_total

    overall_hit_rate = hits_total / total_lookups if total_lookups > 0.0 else 0.0
    overall_fake_hit_rate = fake_hits_total / hits_total if hits_total > 0.0 else 0.0
    overall_full_eval_rate = float(cache_stats.get("full_eval_rate", 0.0) or 0.0)
    overall_sample_eval_rate = float(cache_stats.get("sample_eval_rate", 0.0) or 0.0)

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
        "mode": mode_label,
        "sample_fraction": float(sample_fraction),
        "fake_hit_threshold": float(fake_hit_threshold),
        "final_test_mae": final_test_mae,
        "final_test_accuracy": final_test_accuracy,
        "total_time_sec": total_time_sec,
        "hits": hits_total,
        "misses": misses_total,
        "fake_hits": fake_hits_total,
        "hit_rate_overall": overall_hit_rate,
        "fake_hit_rate_overall": overall_fake_hit_rate,
        "full_evals_total": full_evals_total,
        "sample_evals_total": sample_evals_total,
        "full_eval_rate_overall": overall_full_eval_rate,
        "sample_eval_rate_overall": overall_sample_eval_rate,
    }

    # Per-generation cache stats to make plotting easier later.
    # If the lengths do not match the DF rows, we skip attaching them.
    n_rows = len(df)
    gen_hits = np.asarray(cache_stats.get("gen_hits", []), dtype=float)
    gen_misses = np.asarray(cache_stats.get("gen_misses", []), dtype=float)
    gen_fake_hits = np.asarray(cache_stats.get("gen_fake_hits", []), dtype=float)
    gen_full_evals = np.asarray(cache_stats.get("gen_full_evals", []), dtype=float)
    gen_sample_evals = np.asarray(cache_stats.get("gen_sample_evals", []), dtype=float)
    gen_hits_just_structural = np.asarray(
        cache_stats.get("gen_hits_just_structural", []), dtype=float
    )
    gen_hits_behavioural_without_structural = np.asarray(
        cache_stats.get("gen_hits_behavioural_without_structural", []), dtype=float
    )

    if gen_hits.size == n_rows:
        df["cache_hits"] = gen_hits
    if gen_misses.size == n_rows:
        df["cache_misses"] = gen_misses
    if gen_fake_hits.size == n_rows:
        df["cache_fake_hits"] = gen_fake_hits
    if gen_full_evals.size == n_rows:
        df["full_evals"] = gen_full_evals
    if gen_sample_evals.size == n_rows:
        df["sample_evals"] = gen_sample_evals
    if gen_hits_just_structural.size == n_rows:
        df["hits_just_structural"] = gen_hits_just_structural
    if gen_hits_behavioural_without_structural.size == n_rows:
        df["hits_behavioural_without_structural"] = (
            gen_hits_behavioural_without_structural
        )

    # Save per-generation and summary CSVs.
    # Include method and fraction in filenames so different configs don't overwrite each other.
    frac_pct = int(round(sample_fraction * 100))
    file_tag = f"{sampling_method}_frac_{frac_pct}_th_{th_tag}"

    gen_path = experiment_dir / f"generation_stats_{file_tag}_run{run_index}.csv"
    df.to_csv(gen_path, index=False)

    summary_df = pd.DataFrame([summary_row])
    summary_path = experiment_dir / f"summary_{file_tag}_run{run_index}.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved FEC per-generation stats to {gen_path}")
    print(f"Saved FEC summary to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one FEC-enabled GE run for a single sampling method and fraction."
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
        help="Base RNG seed used together with run-index, sampling method and fraction.",
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
            "Fake-hit threshold used when evaluating cached hits; "
            "a hit is marked fake if |cached - full| > threshold."
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

    # Common root: results/<dataset>_Gen_<G>_Pop_<P>/FEC
    results_root = Path("results") / f"{dataset_stem}_Gen_{n_gen}_Pop_{pop}"
    experiment_dir = results_root / "FEC"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"FEC experiment directory: {experiment_dir}")

    _run_one_fec(
        run_index=run_index,
        base_seed=base_seed,
        sample_fraction=sample_fraction,
        sampling_method=sampling_method,
        fake_hit_threshold=fake_hit_threshold,
        experiment_dir=experiment_dir,
    )


if __name__ == "__main__":
    main()

