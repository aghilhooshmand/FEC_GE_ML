from __future__ import annotations

"""
Minimal, self-contained script to compare:

  1) Baseline (no FEC, full training set)
  2) Union sampling with a simple cache over kmeans + stratified + farthest_point

It:
  - Loads the Cleveland heart dataset.
  - Runs a small GP (using grape) multiple times.
  - Records per-generation TRAIN and TEST fitness for:
        - baseline (no FEC)
        - union + simple cache
  - Records simple cache hit / fake-hit statistics for union across sample fractions.
  - Produces three Plotly HTML charts:
        1) Training fitness vs generation   (baseline vs union at one fraction)
        2) Testing fitness vs generation    (baseline vs union at one fraction)
        3) Hit rate and fake-hit rate vs sample fraction (union only)

The goal is clarity, not feature completeness.
"""

import math
import os
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from deap import base, creator, tools
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from scipy import stats
from multiprocessing.pool import ThreadPool

# Ensure project root (containing 'grape', 'sampling_methods', 'operators', 'util') is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import grape.grape as grape
import grape.algorithms as ga

from sampling_methods import get_sampling_function
from operators import OPERATORS as OP_OPERATORS, UNARIES as OP_UNARIES
from util import FECCache, PhenotypeTracker, create_fitness_eval, load_operators as util_load_operators
from config import CONFIG


# ---------------------------------------------------------------------------
# Simple configuration (all from CONFIG)
# ---------------------------------------------------------------------------

DATA_FILE = Path("data") / CONFIG["dataset.file"]
GRAMMAR_FILE = Path("grammars") / CONFIG["grammar.file"]

POP_SIZE = CONFIG["evolution.population"]
N_GEN = CONFIG["evolution.generations"]
N_RUNS = CONFIG["evolution.n_runs"]
TEST_SIZE = CONFIG["dataset.test_size"]
RANDOM_SEED = CONFIG["evolution.random_seed"]

SAMPLE_FRACTIONS = CONFIG["fec.sample_sizes"]
UNION_METHODS = CONFIG["fec.sampling_methods.union"]

RESULTS_BASE = Path("results") / "simple_union_vs_baseline"
RESULTS_BASE.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers: data, operators, fitness
# ---------------------------------------------------------------------------


def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the Cleveland dataset with a header row.
    Assumes:
      - last column is the label
      - all feature columns are numeric
    """
    df = pd.read_csv(path)
    df = df.replace("?", np.nan).dropna(axis=0)
    y = df.iloc[:, -1].astype(int).to_numpy()
    X = df.iloc[:, :-1].to_numpy(dtype=float)
    return X, y


def baseline_fitness(individual: Any, points: Tuple[np.ndarray, np.ndarray], operators: Dict[str, Any]) -> Tuple[float]:
    x, y_true = points
    if getattr(individual, "invalid", False):
        return (float("nan"),)

    env: Dict[str, Any] = {"np": np, "x": x}
    env.update(operators)
    try:
        prediction = eval(individual.phenotype, env)
    except Exception:
        return (float("nan"),)

    if not np.isrealobj(prediction):
        return (float("nan"),)

    prediction = np.asarray(prediction, dtype=np.float64).flatten()
    if prediction.shape[0] != y_true.shape[0]:
        return (float("nan"),)

    try:
        y_pred = (prediction > 0).astype(int)
        fitness = 1.0 - np.mean(np.equal(y_true, y_pred))  # 1 - accuracy
    except Exception:
        return (float("nan"),)

    return (float(fitness),)


def prepare_toolbox(ops: Dict[str, Any], grammar: grape.Grammar, tournsize: int) -> base.Toolbox:
    toolbox = base.Toolbox()

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", grape.Individual, fitness=creator.FitnessMin)

    toolbox.register("populationCreator", grape.sensible_initialisation, creator.Individual)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    toolbox.register("mate", grape.crossover_onepoint)
    toolbox.register("mutate", grape.mutation_int_flip_per_codon)

    toolbox.population_kwargs = dict(
        bnf_grammar=grammar,
        min_init_depth=3,
        max_init_depth=7,
        codon_size=255,
        codon_consumption="lazy",
        genome_representation="list",
    )
    return toolbox


def run_one_experiment(
    mode: str,
    X: np.ndarray,
    y: np.ndarray,
    sample_fraction: float | None,
    rng_seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Run one multi-run experiment for:
      mode = "baseline"            → no cache, full dataset
      mode = "union_cache"         → simple cache + union sampling
    Returns:
      - aggregated per-generation statistics (DataFrame with columns gen, avg, min, max, fitness_test)
      - union cache stats (only for union mode; otherwise empty dict)
      - per-individual tracking DataFrame
      - cache-events DataFrame (run, gen, phenotype, hit_count, fake_count, fitness values, generation_time_sec)
    """
    # Use the same operator set as the main project (from util.load_operators)
    ops = util_load_operators(Path("operators"))
    grammar = grape.Grammar(str(GRAMMAR_FILE))

    all_logbooks: List[tools.Logbook] = []
    union_cache_stats_all_runs: List[Dict[str, Any]] = []
    tracker = PhenotypeTracker()
    cache_event_rows: List[Dict[str, Any]] = []

    for run_idx in range(N_RUNS):
        run_seed = rng_seed + run_idx
        np.random.seed(run_seed)
        tracker.start_run(run_idx + 1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=run_seed
        )
        X_train = X_train.T
        X_test = X_test.T

        toolbox = prepare_toolbox(ops, grammar, tournsize=7)

        if mode == "baseline":
            # Full training set, no cache
            def eval_fn(ind, pts):
                return baseline_fitness(ind, pts, ops)

            sample_points = (X_train, y_train)
            run_cache: FECCache | None = None
        elif mode == "union_cache":
            assert sample_fraction is not None
            train_n = X_train.shape[1]
            sample_size = max(1, min(int(round(sample_fraction * train_n)), train_n))  # cap at train_n (K-means needs n_clusters <= n_samples)

            # Build union of indices from multiple methods
            indices_all: List[int] = []
            for m_name in UNION_METHODS:
                sampling_fn = get_sampling_function(m_name)
                _, _, idx = sampling_fn(X_train, y_train, sample_size, run_seed)
                indices_all.extend(list(idx))
            union_idx = np.unique(np.asarray(indices_all, dtype=int))
            X_sample = X_train[:, union_idx]
            y_sample = y_train[union_idx]

            # Real FEC-based fitness using project utilities (same behaviour as main code)
            run_cache = FECCache()
            run_cache.clear()
            eval_fn = create_fitness_eval(
                centroid_X=X_sample,
                centroid_y=y_sample,
                centroid_indices=union_idx,
                cache=run_cache,
                fec_enabled=True,
                evaluate_fake_hits=bool(CONFIG.get("fec.evaluate_fake_hits", False)),
                fake_hit_threshold=float(CONFIG.get("fec.fake_hit_threshold", 1e-6)),
                structural_similarity=bool(CONFIG.get("fec.structural_similarity", True)),
                behavior_similarity=bool(CONFIG.get("fec.behavior_similarity", True)),
                operators=ops,
                X_test_ref=X_test,
                y_test_ref=y_test,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Register evaluate
        toolbox.register("evaluate", lambda ind, pts, f=eval_fn: f(ind, pts))

        # Population, hall of fame, stats
        pop = toolbox.populationCreator(pop_size=POP_SIZE, **toolbox.population_kwargs)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.nanmean)
        stats.register("std", np.nanstd)
        stats.register("min", np.nanmin)
        stats.register("max", np.nanmax)

        # Prepare points_train / points_test
        # For union, training still uses full training set; the sampled centroids
        # (X_sample, y_sample) are passed separately to create_fitness_eval above.
        points_train = [X_train, y_train]
        points_test = [X_test, y_test]

        # Optional parallel evaluation across individuals using a thread pool.
        # Threads share the same FECCache and numpy releases the GIL, so this can
        # exploit multiple CPU cores on a server.
        n_workers_cfg = CONFIG.get("parallel.n_workers")
        try:
            n_workers = int(n_workers_cfg) if n_workers_cfg is not None else (os.cpu_count() or 1)
        except (TypeError, ValueError):
            n_workers = os.cpu_count() or 1
        n_workers = max(1, n_workers)

        pool = ThreadPool(processes=n_workers)
        toolbox.register("map", pool.map)

        try:
            # Run evolutionary algorithm (always using FEC-aware variant so we can track phenotypes)
            pop, logbook = ga.ge_eaSimpleWithElitism_fec(
                pop,
                toolbox,
                cxpb=0.8,
                mutpb=0.05,
                ngen=N_GEN,
                elite_size=1,
                bnf_grammar=grammar,
                codon_size=255,
                max_tree_depth=35,
                max_genome_length=None,
                points_train=points_train,
                points_test=points_test,
                codon_consumption="lazy",
                report_items=["gen", "invalid", "avg", "std", "min", "max", "fitness_test"],
                genome_representation="list",
                stats=stats,
                halloffame=hof,
                verbose=False,
                run_id=None,
                fec_cache=run_cache,
                phenotype_tracker=tracker,
            )
        finally:
            pool.close()
            pool.join()

        all_logbooks.append(logbook)
        if mode == "union_cache" and run_cache is not None:
            stats_cache = run_cache.get_stats()
            union_cache_stats_all_runs.append(stats_cache)

            # Generation time (sec) per gen for this run
            gen_list = list(logbook.select("gen"))
            try:
                time_list = list(logbook.select("generation_time"))
            except (KeyError, IndexError):
                time_list = [None] * len(gen_list)

            # Flatten detailed cache events with run index and generation_time_sec
            for event_list_name, event_type in [
                ("detailed_hits", "hit"),
                ("detailed_fake_hits", "fake_hit"),
                ("detailed_misses", "miss"),
            ]:
                for ev in stats_cache.get(event_list_name, []):
                    row = dict(ev)
                    row["run"] = run_idx + 1
                    row["cache_event_type"] = event_type
                    g = row.get("generation")
                    if g is not None and g in gen_list:
                        idx = gen_list.index(g)
                        row["generation_time_sec"] = time_list[idx] if idx < len(time_list) else None
                    else:
                        row["generation_time_sec"] = None
                    cache_event_rows.append(row)

    # Aggregate per-generation stats across runs
    # All logbooks share the same gens; we focus on avg, min, max, fitness_test
    gen = np.array(all_logbooks[0].select("gen"))
    keys = ["avg", "min", "max", "fitness_test"]
    agg: Dict[str, np.ndarray] = {"gen": gen}
    for key in keys:
        mats = [np.array(lb.select(key), dtype=float) for lb in all_logbooks]
        stacked = np.vstack(mats)
        agg[key] = np.nanmean(stacked, axis=0)
        agg[f"{key}_std"] = np.nanstd(stacked, axis=0)

    df = pd.DataFrame(agg)

    # Per-run final test fitness (for statistical significance tests in the summary)
    final_test_values: List[float] = []
    for lb in all_logbooks:
        try:
            test_vals = np.array(lb.select("fitness_test"), dtype=float)
        except (KeyError, IndexError):
            continue
        if test_vals.size == 0:
            continue
        final_val = float(test_vals[-1])
        if not np.isnan(final_val):
            final_test_values.append(final_val)

    # Aggregate cache / runtime stats over runs
    union_summary: Dict[str, Any] = {}
    if final_test_values:
        union_summary["final_test_values"] = final_test_values
        union_summary["final_test_mean"] = float(np.mean(final_test_values))
        union_summary["final_test_std"] = float(np.std(final_test_values, ddof=0))
    # Cache-only stats (union-cache mode)
    if mode == "union_cache" and union_cache_stats_all_runs:
        hit_rates: List[float] = []
        fake_rates: List[float] = []
        for st in union_cache_stats_all_runs:
            hit_rates.append(float(st.get("hit_rate", 0.0)))
            fake_rates.append(float(st.get("fake_hit_rate", 0.0)))
        union_summary["hit_rate"] = float(np.mean(hit_rates)) if hit_rates else 0.0
        union_summary["fake_hit_rate"] = float(np.mean(fake_rates)) if fake_rates else 0.0

        # Per-generation hit/fake-hit rates across runs (for plotting)
        gen_hits_list = [np.asarray(st.get("gen_hits", []), dtype=float) for st in union_cache_stats_all_runs]
        gen_misses_list = [np.asarray(st.get("gen_misses", []), dtype=float) for st in union_cache_stats_all_runs]
        gen_fake_list = [np.asarray(st.get("gen_fake_hits", []), dtype=float) for st in union_cache_stats_all_runs]
        max_len = max(len(arr) for arr in gen_hits_list) if gen_hits_list else 0

        hit_rate_gen_mean: List[float] = []
        hit_rate_gen_std: List[float] = []
        fake_rate_gen_mean: List[float] = []
        fake_rate_gen_std: List[float] = []

        for i in range(max_len):
            per_run_hit_rates: List[float] = []
            per_run_fake_rates: List[float] = []
            for hits_arr, misses_arr, fake_arr in zip(gen_hits_list, gen_misses_list, gen_fake_list):
                if i >= len(hits_arr) or i >= len(misses_arr) or i >= len(fake_arr):
                    continue
                h = float(hits_arr[i])
                m = float(misses_arr[i])
                f = float(fake_arr[i])
                denom = h + m
                if denom > 0:
                    per_run_hit_rates.append(h / denom)
                if h > 0:
                    per_run_fake_rates.append(f / h)
            # Aggregate across runs for this generation index
            if per_run_hit_rates:
                hit_rate_gen_mean.append(float(np.mean(per_run_hit_rates)))
                hit_rate_gen_std.append(float(np.std(per_run_hit_rates, ddof=0)))
            else:
                hit_rate_gen_mean.append(float("nan"))
                hit_rate_gen_std.append(float("nan"))
            if per_run_fake_rates:
                fake_rate_gen_mean.append(float(np.mean(per_run_fake_rates)))
                fake_rate_gen_std.append(float(np.std(per_run_fake_rates, ddof=0)))
            else:
                fake_rate_gen_mean.append(float("nan"))
                fake_rate_gen_std.append(float("nan"))

        union_summary["gen_hit_rate_mean"] = hit_rate_gen_mean
        union_summary["gen_hit_rate_std"] = hit_rate_gen_std
        union_summary["gen_fake_hit_rate_mean"] = fake_rate_gen_mean
        union_summary["gen_fake_hit_rate_std"] = fake_rate_gen_std

    # Runtime stats (both baseline and union-cache)
    run_total_times: List[float] = []
    for lb in all_logbooks:
        try:
            gen_times = np.array(lb.select("generation_time"), dtype=float)
        except (KeyError, IndexError):
            continue
        if gen_times.size == 0:
            continue
        run_total_times.append(float(np.nansum(gen_times)))
    if run_total_times:
        union_summary["total_time_mean"] = float(np.mean(run_total_times))
        union_summary["total_time_std"] = float(np.std(run_total_times, ddof=0))

    # Build per-individual tracking dataframe, optionally enriched with cache info
    tracking_rows = tracker.get_tracking_data()
    individuals_df = pd.DataFrame(tracking_rows) if tracking_rows else pd.DataFrame()

    if mode == "union_cache" and cache_event_rows and not individuals_df.empty:
        cache_df = pd.DataFrame(cache_event_rows)
        # Merge cache events onto individuals by (run, generation, phenotype)
        merged = individuals_df.merge(
            cache_df,
            on=["run", "generation", "phenotype"],
            how="left",
            suffixes=("", "_cache"),
        )
        merged["cache_event_type"] = merged["cache_event_type"].fillna("none")
        merged["cache_hit"] = merged["cache_event_type"] == "hit"
        merged["cache_fake_hit"] = merged["cache_event_type"] == "fake_hit"
        merged["cache_miss"] = merged["cache_event_type"] == "miss"
        individuals_df = merged
    elif not individuals_df.empty:
        individuals_df["cache_event_type"] = "none"
        individuals_df["cache_hit"] = False
        individuals_df["cache_fake_hit"] = False
        individuals_df["cache_miss"] = False

    # Tag configuration-level info
    if not individuals_df.empty:
        individuals_df["mode"] = mode
        individuals_df["sample_fraction"] = sample_fraction

    # Build cache-events CSV: run, gen, phenotype, hit_count, fake_count, fitness values, generation_time_sec
    cache_events_df = pd.DataFrame()
    if cache_event_rows:
        cache_df = pd.DataFrame(cache_event_rows)
        actual = cache_df.get("actual_fitness")
        if actual is None:
            actual = cache_df.get("current_full_fitness")
        if actual is None:
            actual = cache_df.get("fitness")
        cache_events_df = pd.DataFrame({
            "run": cache_df["run"],
            "gen": cache_df["generation"],
            "phenotype": cache_df["phenotype"],
            "hit_count": (cache_df["cache_event_type"] == "hit").astype(int),
            "fake_count": (cache_df["cache_event_type"] == "fake_hit").astype(int),
            "cached_fitness": cache_df.get("cached_fitness"),
            "actual_fitness": actual,
            "generation_time_sec": cache_df.get("generation_time_sec"),
        })
        if mode == "union_cache" and sample_fraction is not None:
            cache_events_df["sample_fraction"] = sample_fraction

    return df, union_summary, individuals_df, cache_events_df


# ---------------------------------------------------------------------------
# Main: run experiments and plot
# ---------------------------------------------------------------------------

def main() -> None:
    # Unique folder for this experiment run (date and time)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = RESULTS_BASE / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment output directory: {experiment_dir}")

    X, y = load_dataset(DATA_FILE)

    # Baseline: full dataset, no cache
    baseline_df, baseline_summary, baseline_individuals, _ = run_one_experiment(
        "baseline", X, y, None, RANDOM_SEED
    )

    # Union with cache: one run per fraction
    union_results: Dict[float, pd.DataFrame] = {}
    union_summaries: Dict[float, Dict[str, Any]] = {}
    union_hit_rates: List[float] = []
    union_fake_rates: List[float] = []
    used_fractions: List[float] = []
    all_individual_rows: List[pd.DataFrame] = []
    all_cache_event_rows: List[pd.DataFrame] = []

    if not baseline_individuals.empty:
        all_individual_rows.append(baseline_individuals)

    for frac in SAMPLE_FRACTIONS:
        df_union, stats_union, union_individuals, cache_events_df = run_one_experiment(
            "union_cache", X, y, frac, RANDOM_SEED + 1000
        )
        union_results[frac] = df_union
        union_summaries[frac] = stats_union
        union_hit_rates.append(stats_union.get("hit_rate", 0.0))
        union_fake_rates.append(stats_union.get("fake_hit_rate", 0.0))
        used_fractions.append(frac)
        if not union_individuals.empty:
            all_individual_rows.append(union_individuals)
        if not cache_events_df.empty:
            all_cache_event_rows.append(cache_events_df)

    # Save detailed per-individual CSV (run 1 only, to save disk space) into this experiment's folder
    if all_individual_rows:
        all_individuals_df = pd.concat(all_individual_rows, ignore_index=True)
        if "run" in all_individuals_df.columns:
            all_individuals_df = all_individuals_df[all_individuals_df["run"] == 1].reset_index(drop=True)
        individuals_csv_path = experiment_dir / "individuals.csv"
        all_individuals_df.to_csv(individuals_csv_path, index=False)
        print(f"Saved per-individual evolution + cache details (run 1 only) to {individuals_csv_path}")

    # Save cache-events CSV into this experiment's folder
    cache_events_combined = None
    if all_cache_event_rows:
        cache_events_combined = pd.concat(all_cache_event_rows, ignore_index=True)
        cache_csv_path = experiment_dir / "cache_events.csv"
        cache_events_combined.to_csv(cache_csv_path, index=False)
        print(
            f"Saved cache-events (run, gen, phenotype, hit_count, fake_count, fitness, generation_time_sec) to {cache_csv_path}"
        )

        # Per-(run, gen, sample_fraction) cache aggregation (each fraction = separate full evolution)
        group_cols = ["run", "gen"]
        if "sample_fraction" in cache_events_combined.columns:
            group_cols.append("sample_fraction")
        cache_agg = (
            cache_events_combined.groupby(group_cols, as_index=False)[
                ["hit_count", "fake_count"]
            ]
            .sum()
        )
        cache_agg_path = experiment_dir / "cache_per_generation.csv"
        cache_agg.to_csv(cache_agg_path, index=False)
        print(
            f"Saved per-generation cache aggregates (run, gen, sample_fraction, hit_count, fake_count) to {cache_agg_path}"
        )

    # Build summary table: baseline vs FEC (final test fitness / accuracy vs speedup and hit rate)
    summary_rows: List[Dict[str, Any]] = []
    baseline_time = baseline_summary.get("total_time_mean")
    baseline_final_test = (
        float(baseline_df["fitness_test"].iloc[-1]) if "fitness_test" in baseline_df else None
    )
    baseline_accuracy = 1.0 - baseline_final_test if baseline_final_test is not None else None
    baseline_hit_rate = float(baseline_summary.get("hit_rate", 0.0) or 0.0)
    baseline_fake_hit_rate = float(baseline_summary.get("fake_hit_rate", 0.0) or 0.0)
    baseline_final_tests_raw = baseline_summary.get("final_test_values")
    baseline_final_tests = (
        np.asarray(baseline_final_tests_raw, dtype=float)
        if baseline_final_tests_raw is not None
        else np.array([])
    )

    summary_rows.append(
        {
            "mode": "baseline",
            "sample_fraction": 1.0,
            "final_test_fitness_mean": baseline_final_test,
            "accuracy": baseline_accuracy,
            "total_time_mean_sec": baseline_time,
            "speedup_vs_baseline": 1.0,
            "hit_rate": baseline_hit_rate,
            "fake_hit_rate": baseline_fake_hit_rate,
            "delta_accuracy_vs_baseline": 0.0,
            "p_value_vs_baseline": None,
            "significant_vs_baseline_0.05": None,
            "p_value_baseline_better": None,
            "accuracy_comparable": True,
            "test_name": None,
        }
    )

    # Derive a human-readable FEC mode label from config
    enabled_methods_cfg = CONFIG.get("fec.sampling_methods.enabled", {})
    if enabled_methods_cfg.get("union", False):
        label_methods = CONFIG.get("fec.sampling_methods.union", UNION_METHODS)
        fec_mode_label = "fec_union(" + "+".join(label_methods) + ")"
    else:
        enabled_non_union = [
            m for m, flag in enabled_methods_cfg.items() if flag and m != "union"
        ]
        if enabled_non_union:
            fec_mode_label = "fec_" + "+".join(enabled_non_union)
        else:
            fec_mode_label = "fec"

    for frac in used_fractions:
        df_union = union_results[frac]
        stats_union = union_summaries.get(frac, {})
        union_time = stats_union.get("total_time_mean")
        union_final_test = (
            float(df_union["fitness_test"].iloc[-1]) if "fitness_test" in df_union else None
        )
        union_accuracy = 1.0 - union_final_test if union_final_test is not None else None
        speedup = None
        if baseline_time and union_time and union_time > 0:
            speedup = float(baseline_time / union_time)
        delta_acc = None
        if union_accuracy is not None and baseline_accuracy is not None:
            delta_acc = union_accuracy - baseline_accuracy
        union_hit_rate = float(stats_union.get("hit_rate", 0.0) or 0.0)
        union_fake_hit_rate = float(stats_union.get("fake_hit_rate", 0.0) or 0.0)

        # Statistical significance vs baseline on final test fitness (1 - accuracy, lower is better)
        union_final_tests_raw = stats_union.get("final_test_values")
        union_final_tests = (
            np.asarray(union_final_tests_raw, dtype=float)
            if union_final_tests_raw is not None
            else np.array([])
        )
        p_value = None
        significant = None
        p_value_baseline_better = None
        accuracy_comparable = None
        test_name = None
        # Use Mann-Whitney U when we have at least 2 runs per method
        if baseline_final_tests.size >= 2 and union_final_tests.size >= 2:
            try:
                # Two-sided: is there any difference?
                _, p_val = stats.mannwhitneyu(
                    baseline_final_tests,
                    union_final_tests,
                    alternative="two-sided",
                )
                p_value = float(p_val)
                significant = bool(p_value < 0.05)
                # One-sided: is baseline significantly better (lower fitness) than FEC?
                # alternative="greater" => baseline (1st) stochastically greater than FEC (2nd)
                # (higher fitness = worse). So small p => baseline has worse fitness; large p => no evidence baseline is better => FEC comparable
                _, p_one = stats.mannwhitneyu(
                    baseline_final_tests,
                    union_final_tests,
                    alternative="greater",
                )
                p_value_baseline_better = float(p_one)
                accuracy_comparable = p_value_baseline_better > 0.05
                test_name = "mannwhitneyu_two_sided"
            except Exception:
                p_value = None
                significant = None
                p_value_baseline_better = None
                accuracy_comparable = None
                test_name = None

        summary_rows.append(
            {
                "mode": fec_mode_label,
                "sample_fraction": frac,
                "final_test_fitness_mean": union_final_test,
                "accuracy": union_accuracy,
                "total_time_mean_sec": union_time,
                "speedup_vs_baseline": speedup,
                "hit_rate": union_hit_rate,
                "fake_hit_rate": union_fake_hit_rate,
                "delta_accuracy_vs_baseline": delta_acc,
                "p_value_vs_baseline": p_value,
                "significant_vs_baseline_0.05": significant,
                "p_value_baseline_better": p_value_baseline_better,
                "accuracy_comparable": accuracy_comparable,
                "test_name": test_name,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = experiment_dir / "fec_vs_baseline_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved baseline vs FEC trade-off summary table to {summary_path}")

    # Collect all charts into a single HTML string
    sections: List[str] = []

    # 1) For each sample fraction: compare baseline vs FEC (training + testing)
    for frac in used_fractions:
        df_union_frac = union_results[frac]
        enabled_methods_cfg = CONFIG.get("fec.sampling_methods.enabled", {})
        if enabled_methods_cfg.get("union", False):
            label_methods = CONFIG.get("fec.sampling_methods.union", UNION_METHODS)
            union_label_base = "FEC (union: " + "+".join(label_methods) + ")"
        else:
            # Use whichever individual sampling methods are enabled (excluding 'union')
            enabled_non_union = [
                m for m, flag in enabled_methods_cfg.items() if flag and m != "union"
            ]
            if enabled_non_union:
                union_label_base = "FEC (" + "+".join(enabled_non_union) + ")"
            else:
                # Fallback to union methods list
                union_label_base = "FEC (union: " + "+".join(UNION_METHODS) + ")"
        stats_union = union_summaries.get(frac, {})
        gen_seq = df_union_frac["gen"]

        # Training
        fig_train = go.Figure()
        fig_train.add_trace(
            go.Scatter(
                x=baseline_df["gen"],
                y=baseline_df["avg"],
                mode="lines+markers",
                name="Baseline (no FEC)",
                error_y=dict(
                    type="data",
                    array=baseline_df.get("avg_std"),
                    visible=True,
                ),
            )
        )
        fig_train.add_trace(
            go.Scatter(
                x=df_union_frac["gen"],
                y=df_union_frac["avg"],
                mode="lines+markers",
                name=f"{union_label_base}, frac={frac:.0%}",
                error_y=dict(
                    type="data",
                    array=df_union_frac.get("avg_std"),
                    visible=True,
                ),
            )
        )
        fig_train.update_layout(
            title=f"Training Fitness (Average) vs Generation (frac={frac:.0%})",
            xaxis_title="Generation",
            yaxis_title="Fitness (1 - accuracy, lower is better)",
            template="plotly_white",
            hovermode="x unified",
        )
        sections.append(pio.to_html(fig_train, include_plotlyjs="cdn", full_html=False))

        # Testing
        fig_test = go.Figure()
        fig_test.add_trace(
            go.Scatter(
                x=baseline_df["gen"],
                y=baseline_df["fitness_test"],
                mode="lines+markers",
                name="Baseline (no FEC)",
                error_y=dict(
                    type="data",
                    array=baseline_df.get("fitness_test_std"),
                    visible=True,
                ),
            )
        )
        fig_test.add_trace(
            go.Scatter(
                x=df_union_frac["gen"],
                y=df_union_frac["fitness_test"],
                mode="lines+markers",
                name=f"{union_label_base}, frac={frac:.0%}",
                error_y=dict(
                    type="data",
                    array=df_union_frac.get("fitness_test_std"),
                    visible=True,
                ),
            )
        )
        fig_test.update_layout(
            title=f"Testing Fitness vs Generation (frac={frac:.0%})",
            xaxis_title="Generation",
            yaxis_title="Fitness (1 - accuracy, lower is better)",
            template="plotly_white",
            hovermode="x unified",
        )
        sections.append(pio.to_html(fig_test, include_plotlyjs="cdn", full_html=False))

        # Hit-rate vs generation (union only, with STD across runs)
        gen_hit_mean = stats_union.get("gen_hit_rate_mean")
        gen_hit_std = stats_union.get("gen_hit_rate_std")
        if gen_hit_mean is not None:
            fig_hit_gen = go.Figure()
            fig_hit_gen.add_trace(
                go.Scatter(
                    x=gen_seq,
                    y=gen_hit_mean,
                    mode="lines+markers",
                    name=f"{union_label_base}, frac={frac:.0%}",
                    error_y=dict(
                        type="data",
                        array=gen_hit_std,
                        visible=True,
                    ),
                )
            )
            fig_hit_gen.update_layout(
                title=f"Cache Hit Rate vs Generation ({union_label_base}, frac={frac:.0%})",
                xaxis_title="Generation",
                yaxis_title="Hit rate",
                yaxis_tickformat=".0%",
                template="plotly_white",
                hovermode="x unified",
                showlegend=True,
            )
            sections.append(
                pio.to_html(fig_hit_gen, include_plotlyjs="cdn", full_html=False)
            )

        # Fake-hit vs generation (union only, with STD across runs)
        gen_fake_mean = stats_union.get("gen_fake_hit_rate_mean")
        gen_fake_std = stats_union.get("gen_fake_hit_rate_std")
        if gen_fake_mean is not None:
            fig_fake_gen = go.Figure()
            fig_fake_gen.add_trace(
                go.Scatter(
                    x=gen_seq,
                    y=gen_fake_mean,
                    mode="lines+markers",
                    name=f"{union_label_base}, frac={frac:.0%}",
                    error_y=dict(
                        type="data",
                        array=gen_fake_std,
                        visible=True,
                    ),
                )
            )
            fig_fake_gen.update_layout(
                title=f"Cache Fake-Hit Rate vs Generation ({union_label_base}, frac={frac:.0%})",
                xaxis_title="Generation",
                yaxis_title="Fake-hit rate",
                yaxis_tickformat=".0%",
                template="plotly_white",
                hovermode="x unified",
                showlegend=True,
            )
            sections.append(
                pio.to_html(fig_fake_gen, include_plotlyjs="cdn", full_html=False)
            )

    # 2) One chart: final-generation testing fitness vs sample fraction
    baseline_final_test = float(baseline_df["fitness_test"].iloc[-1])
    union_final_tests = [float(union_results[f]["fitness_test"].iloc[-1]) for f in used_fractions]

    fig_final = go.Figure()
    # Baseline as horizontal reference line
    fig_final.add_trace(
        go.Scatter(
            x=used_fractions,
            y=[baseline_final_test] * len(used_fractions),
            mode="lines",
            name="Baseline (final test)",
        )
    )
    # Build label base consistent with sampling config
    enabled_methods_cfg = CONFIG.get("fec.sampling_methods.enabled", {})
    if enabled_methods_cfg.get("union", False):
        label_methods = CONFIG.get("fec.sampling_methods.union", UNION_METHODS)
        fec_label_base = "FEC (union: " + "+".join(label_methods) + ")"
    else:
        enabled_non_union = [
            m for m, flag in enabled_methods_cfg.items() if flag and m != "union"
        ]
        if enabled_non_union:
            fec_label_base = "FEC (" + "+".join(enabled_non_union) + ")"
        else:
            fec_label_base = "FEC (union: " + "+".join(UNION_METHODS) + ")"
    fig_final.add_trace(
        go.Scatter(
            x=used_fractions,
            y=union_final_tests,
            mode="lines+markers",
            name=f"{fec_label_base} (final test)",
        )
    )
    fig_final.update_layout(
        title="Final Testing Fitness vs Sample Fraction (Baseline vs Union)",
        xaxis_title="Sample fraction",
        yaxis_title="Final test fitness (1 - accuracy, lower is better)",
        template="plotly_white",
        hovermode="x unified",
    )
    sections.append(pio.to_html(fig_final, include_plotlyjs="cdn", full_html=False))

    # 3) Hit rate vs sample fraction (union only)
    fig_hit = go.Figure()
    fig_hit.add_trace(
        go.Scatter(
            x=used_fractions,
            y=union_hit_rates,
            mode="lines+markers",
            name="Hit rate",
        )
    )
    fig_hit.update_layout(
        title="Union Cache Hit Rate vs Sample Fraction",
        xaxis_title="Sample fraction",
        yaxis_title="Hit rate",
        yaxis_tickformat=".0%",
        template="plotly_white",
        hovermode="x unified",
        showlegend=True,
    )
    sections.append(pio.to_html(fig_hit, include_plotlyjs="cdn", full_html=False))

    # 4) Fake-hit rate vs sample fraction (union only)
    fig_fake = go.Figure()
    fig_fake.add_trace(
        go.Scatter(
            x=used_fractions,
            y=union_fake_rates,
            mode="lines+markers",
            name="Fake-hit rate",
        )
    )
    fig_fake.update_layout(
        title="Union Cache Fake-Hit Rate vs Sample Fraction",
        xaxis_title="Sample fraction",
        yaxis_title="Fake-hit rate",
        yaxis_tickformat=".0%",
        template="plotly_white",
        hovermode="x unified",
        showlegend=True,
    )
    sections.append(pio.to_html(fig_fake, include_plotlyjs="cdn", full_html=False))

    # Combine into one HTML file
    config_json = json.dumps(CONFIG, indent=2, sort_keys=True)

    # Build a human-readable dataset description based on CONFIG.
    dataset_source = (CONFIG.get("dataset.source") or "csv").lower()
    dataset_desc = ""
    if dataset_source in ("uci", "uci_openml"):
        uci_id = CONFIG.get("dataset.uci_id")
        label_col = CONFIG.get("dataset.label_column")
        dataset_name = None
        try:
            if uci_id is not None:
                ds = fetch_openml(data_id=uci_id)
                # Try several possible name attributes
                dataset_name = getattr(ds, "name", None) or ds.details.get("name")
        except Exception:
            dataset_name = None
        if dataset_name is None:
            dataset_name = f"OpenML data_id={uci_id}"
        dataset_desc = f"Dataset: {dataset_name} (source=uci_openml, id={uci_id}), label={label_col}"
    else:
        dataset_file = CONFIG.get("dataset.file")
        label_col = CONFIG.get("dataset.label_column")
        dataset_desc = f"Dataset: file={dataset_file}, label={label_col}"

    grammar_file = CONFIG.get("grammar.file")
    if grammar_file:
        dataset_desc += f"; Grammar: {grammar_file}"

    html = (
        "<html><head>"
        "<meta charset='utf-8' />"
        "<title>Baseline vs FEC (Simple Experiment)</title>"
        "</head><body>"
        "<h1>Baseline vs FEC (Simple Experiment)</h1>"
        "<h2>Configuration</h2>"
        "<pre style='font-size: 12px; background:#f7f7f7; padding:8px; border:1px solid #ddd;'>"
        + config_json +
        "</pre>"
        f"<p>{dataset_desc}</p>"
        + "".join(sections) +
        "</body></html>"
    )
    output_path = experiment_dir / "baseline_vs_union_all_charts.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"Saved all charts to {output_path}")


if __name__ == "__main__":
    main()


