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
from scipy import stats

# Ensure project root (containing 'grape', 'sampling_methods', 'operators', 'util') is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import grape.grape as grape
import grape.algorithms as ga

from sampling_methods import get_sampling_function
from operators import OPERATORS as OP_OPERATORS, UNARIES as OP_UNARIES
from util import (
    FECCache,
    PhenotypeTracker,
    create_fitness_eval,
    load_operators as util_load_operators,
    load_dataset as util_load_dataset,
)
from config import CONFIG


# ---------------------------------------------------------------------------
# Simple configuration (all from CONFIG)
# ---------------------------------------------------------------------------

def _fec_show_behaviour_without_structural() -> bool:
    return bool(CONFIG.get("fec.modes.fec_enabled_behaviour_without_structural", True))


def _fec_show_structural_only() -> bool:
    return bool(CONFIG.get("fec.modes.fec_enabled_structural_only", True))


def _fec_show_total() -> bool:
    return bool(CONFIG.get("fec.modes.fec_enabled_total", True))


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
        # MAE (classification error) = 1 - accuracy
        fitness = 1.0 - np.mean(np.equal(y_true, y_pred))
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
    fec_sampling_methods: List[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run one multi-run experiment for:
      mode = "baseline"            → no cache, full dataset
      mode = "union_cache"         → FEC cache + sampling (single method or union of methods)
    fec_sampling_methods: for union_cache, list of method names (e.g. ["kmeans"] or ["kmeans", "farthest_point"]).
      One method = use that sampling only; multiple = take union of indices.
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
    tracker = PhenotypeTracker(track_individuals=bool(CONFIG.get("output.track_individuals", True)))
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

            # Which methods to use: one = single method; several = union of indices
            methods = fec_sampling_methods if fec_sampling_methods else list(UNION_METHODS)
            indices_all: List[int] = []
            for m_name in methods:
                sampling_fn = get_sampling_function(m_name)
                _, _, idx = sampling_fn(X_train, y_train, sample_size, run_seed)
                indices_all.extend(list(idx))
            union_idx = np.unique(np.asarray(indices_all, dtype=int))
            X_sample = X_train[:, union_idx]
            y_sample = y_train[union_idx]

            # Real FEC-based fitness using project utilities (same behaviour as main code)
            run_cache = FECCache()
            run_cache.clear()
            run_cache.record_detailed_events = bool(CONFIG.get("fec.record_detailed_events", True))
            eval_fn = create_fitness_eval(
                centroid_X=X_sample,
                centroid_y=y_sample,
                centroid_indices=union_idx,
                cache=run_cache,
                fec_enabled=True,
                evaluate_fake_hits=bool(CONFIG.get("fec.evaluate_fake_hits", False)),
                fake_hit_threshold=float(CONFIG.get("fec.fake_hit_threshold", 1e-6)),
                structural_similarity=bool(CONFIG.get("fec.structural_similarity", False)),
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

    # Per-run, per-generation logbook table (one row per run per gen) for export
    logbook_columns = [
        "gen", "invalid", "avg", "std", "min", "max", "fitness_test",
        "best_ind_length", "avg_length", "best_ind_nodes", "avg_nodes",
        "best_ind_depth", "avg_depth", "avg_used_codons", "best_ind_used_codons",
        "selection_time", "generation_time",
    ]
    logbook_rows: List[Dict[str, Any]] = []
    for run_idx, lb in enumerate(all_logbooks):
        for rec in list(lb):
            row: Dict[str, Any] = {"run": run_idx + 1}
            for k in logbook_columns:
                row[k] = rec.get(k, float("nan"))
            logbook_rows.append(row)
    logbook_df = pd.DataFrame(logbook_rows)
    if not logbook_df.empty:
        logbook_df = logbook_df[["run"] + [c for c in logbook_columns if c in logbook_df.columns]]

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

        # Hit breakdown: rates (per lookups) = behavioural (includes structural), behavioural without structural, just structural
        gen_just_struct = [np.asarray(st.get("gen_hits_just_structural", []), dtype=float) for st in union_cache_stats_all_runs]
        gen_behav_no_struct = [np.asarray(st.get("gen_hits_behavioural_without_structural", []), dtype=float) for st in union_cache_stats_all_runs]
        rate_behav_or_both_mean: List[float] = []
        rate_behav_or_both_std: List[float] = []
        rate_behav_no_struct_mean: List[float] = []
        rate_behav_no_struct_std: List[float] = []
        rate_just_struct_mean: List[float] = []
        rate_just_struct_std: List[float] = []
        for i in range(max_len):
            per_run_behav: List[float] = []
            per_run_behav_no: List[float] = []
            per_run_just: List[float] = []
            for h_arr, m_arr, j_arr, b_arr in zip(gen_hits_list, gen_misses_list, gen_just_struct, gen_behav_no_struct):
                if i >= len(h_arr) or i >= len(m_arr):
                    continue
                lookups = float(h_arr[i]) + float(m_arr[i])
                if lookups <= 0:
                    continue
                per_run_behav.append(float(h_arr[i]) / lookups)
                if i < len(j_arr):
                    per_run_just.append(float(j_arr[i]) / lookups)
                if i < len(b_arr):
                    per_run_behav_no.append(float(b_arr[i]) / lookups)
            rate_behav_or_both_mean.append(float(np.mean(per_run_behav)) if per_run_behav else float("nan"))
            rate_behav_or_both_std.append(float(np.std(per_run_behav, ddof=0)) if per_run_behav else float("nan"))
            rate_behav_no_struct_mean.append(float(np.mean(per_run_behav_no)) if per_run_behav_no else float("nan"))
            rate_behav_no_struct_std.append(float(np.std(per_run_behav_no, ddof=0)) if per_run_behav_no else float("nan"))
            rate_just_struct_mean.append(float(np.mean(per_run_just)) if per_run_just else float("nan"))
            rate_just_struct_std.append(float(np.std(per_run_just, ddof=0)) if per_run_just else float("nan"))
        union_summary["gen_rate_behavioural_or_both_mean"] = rate_behav_or_both_mean
        union_summary["gen_rate_behavioural_or_both_std"] = rate_behav_or_both_std
        union_summary["gen_rate_behavioural_without_structural_mean"] = rate_behav_no_struct_mean
        union_summary["gen_rate_behavioural_without_structural_std"] = rate_behav_no_struct_std
        union_summary["gen_rate_just_structural_mean"] = rate_just_struct_mean
        union_summary["gen_rate_just_structural_std"] = rate_just_struct_std

        # Scalar rates for across-fractions chart
        total_hits = sum(st.get("hits", 0) for st in union_cache_stats_all_runs)
        total_misses = sum(st.get("misses", 0) for st in union_cache_stats_all_runs)
        total_lookups = total_hits + total_misses
        if total_lookups > 0:
            j = sum(st.get("hits_just_structural", 0) for st in union_cache_stats_all_runs)
            b = sum(st.get("hits_behavioural_without_structural", 0) for st in union_cache_stats_all_runs)
            union_summary["rate_behavioural_or_both"] = total_hits / total_lookups
            union_summary["rate_behavioural_without_structural"] = b / total_lookups
            union_summary["rate_just_structural"] = j / total_lookups

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

    return df, union_summary, individuals_df, cache_events_df, logbook_df


# ---------------------------------------------------------------------------
# Main: run experiments and plot
# ---------------------------------------------------------------------------

def main() -> None:
    # Folder name: timestamp + dataset + Gen_Pop_Run for quick identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_stem = Path(CONFIG.get("dataset.file", "data")).stem
    folder_name = f"{timestamp}_{dataset_stem}_Gen_{N_GEN}_Pop_{POP_SIZE}_Run_{N_RUNS}"
    experiment_dir = RESULTS_BASE / folder_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment output directory: {experiment_dir}")

    # Use the shared util.load_dataset so that CSV vs UCI/OpenML and
    # dataset.sample_fraction behave consistently with the main pipeline.
    X, y = util_load_dataset(CONFIG)

    # Baseline: full dataset, no cache
    baseline_df, baseline_summary, baseline_individuals, _, baseline_logbook_df = run_one_experiment(
        "baseline", X, y, None, RANDOM_SEED
    )

    # FEC experiments: first, one experiment per enabled (non-union) method; then, only if union is True, one experiment for union (methods from fec.sampling_methods.union).
    enabled_cfg = CONFIG.get("fec.sampling_methods.enabled", {})
    fec_configs: List[Tuple[str, List[str]]] = []
    # 1) For each method in enabled that is True (except "union"), run one experiment for that method only.
    for method_name, flag in enabled_cfg.items():
        if method_name == "union" or not flag:
            continue
        fec_configs.append((f"fec_{method_name}", [method_name]))
    # 2) If "union" is True in enabled, run one additional experiment using the union of methods listed in fec.sampling_methods.union.
    if enabled_cfg.get("union", False):
        union_list = CONFIG.get("fec.sampling_methods.union", [])
        if union_list:
            fec_configs.append((f"fec_union(" + "+".join(union_list) + ")", list(union_list)))

    fec_results: Dict[str, Dict[float, pd.DataFrame]] = {}
    fec_summaries: Dict[str, Dict[float, Dict[str, Any]]] = {}
    used_fractions: List[float] = list(SAMPLE_FRACTIONS)
    all_individual_rows: List[pd.DataFrame] = []
    all_cache_event_rows: List[pd.DataFrame] = []
    all_logbook_dfs: List[pd.DataFrame] = []
    min_fraction = min(SAMPLE_FRACTIONS)  # save cache_events only for smallest fraction to limit file size

    if not baseline_individuals.empty:
        baseline_individuals = baseline_individuals.copy()
        baseline_individuals["mode"] = "baseline"
        all_individual_rows.append(baseline_individuals)

    for method_label, method_list in fec_configs:
        fec_results[method_label] = {}
        fec_summaries[method_label] = {}
        for frac in SAMPLE_FRACTIONS:
            df_fec, stats_fec, fec_individuals, cache_events_df, logbook_df = run_one_experiment(
                "union_cache", X, y, frac, RANDOM_SEED + 1000, fec_sampling_methods=method_list
            )
            fec_results[method_label][frac] = df_fec
            fec_summaries[method_label][frac] = stats_fec
            if not fec_individuals.empty:
                fec_individuals = fec_individuals.copy()
                fec_individuals["mode"] = method_label
                all_individual_rows.append(fec_individuals)
            # Only save cache events for the minimum (smallest) fraction to keep file size down
            if frac == min_fraction and not cache_events_df.empty:
                cache_events_df = cache_events_df.copy()
                cache_events_df["sampling_mode"] = method_label
                all_cache_event_rows.append(cache_events_df)
            if not logbook_df.empty:
                logbook_df = logbook_df.copy()
                logbook_df["mode"] = method_label
                logbook_df["sample_fraction"] = frac
                all_logbook_dfs.append(logbook_df)

    # One CSV: per-run, per-generation stats (gen, invalid, avg, std, min, max, fitness_test, best_ind_*, selection_time, generation_time) for baseline + all FEC experiments
    if not baseline_logbook_df.empty:
        baseline_logbook_df = baseline_logbook_df.copy()
        baseline_logbook_df["mode"] = "baseline"
        baseline_logbook_df["sample_fraction"] = 1.0
        all_logbook_dfs.insert(0, baseline_logbook_df)
    if all_logbook_dfs:
        combined_logbook = pd.concat(all_logbook_dfs, ignore_index=True)
        # Column order: mode, sample_fraction, run, gen, then the rest
        lead_cols = ["mode", "sample_fraction", "run", "gen"]
        rest = [c for c in combined_logbook.columns if c not in lead_cols]
        combined_logbook = combined_logbook[lead_cols + rest]
        logbook_csv_path = experiment_dir / "generation_stats.csv"
        combined_logbook.to_csv(logbook_csv_path, index=False)
        print(f"Saved per-run per-generation stats (gen, invalid, avg, std, min, max, fitness_test, best_ind_*, ...) to {logbook_csv_path}")

    # Save detailed per-individual CSV (run 1 only, to save disk space) into this experiment's folder
    if CONFIG.get("output.save_individuals_csv", False) and all_individual_rows:
        all_individuals_df = pd.concat(all_individual_rows, ignore_index=True)
        if "run" in all_individuals_df.columns:
            all_individuals_df = all_individuals_df[all_individuals_df["run"] == 1].reset_index(drop=True)
        individuals_csv_path = experiment_dir / "individuals.csv"
        all_individuals_df.to_csv(individuals_csv_path, index=False)
        print(f"Saved per-individual evolution + cache details (run 1 only) to {individuals_csv_path}")

    # Save cache-events CSV only for the minimum fraction (to limit file size)
    cache_events_combined = None
    if all_cache_event_rows:
        cache_events_combined = pd.concat(all_cache_event_rows, ignore_index=True)
        cache_csv_path = experiment_dir / "cache_events.csv"
        cache_events_combined.to_csv(cache_csv_path, index=False)
        print(
            f"Saved cache-events (frac={min_fraction:.0%} only; run, gen, phenotype, hit_count, fake_count, fitness, generation_time_sec) to {cache_csv_path}"
        )

        # Per-(run, gen, sample_fraction, sampling_mode) cache aggregation
        group_cols = ["run", "gen"]
        if "sample_fraction" in cache_events_combined.columns:
            group_cols.append("sample_fraction")
        if "sampling_mode" in cache_events_combined.columns:
            group_cols.append("sampling_mode")
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

    # One summary row per (method_label, fraction)
    for method_label in fec_results:
        for frac in used_fractions:
            if frac not in fec_results[method_label]:
                continue
            df_fec = fec_results[method_label][frac]
            stats_fec = fec_summaries[method_label].get(frac, {})
            fec_time = stats_fec.get("total_time_mean")
            fec_final_test = (
                float(df_fec["fitness_test"].iloc[-1]) if "fitness_test" in df_fec else None
            )
            fec_accuracy = 1.0 - fec_final_test if fec_final_test is not None else None
            speedup = None
            if baseline_time and fec_time and fec_time > 0:
                speedup = float(baseline_time / fec_time)
            delta_acc = None
            if fec_accuracy is not None and baseline_accuracy is not None:
                delta_acc = fec_accuracy - baseline_accuracy
            fec_hit_rate = float(stats_fec.get("hit_rate", 0.0) or 0.0)
            fec_fake_hit_rate = float(stats_fec.get("fake_hit_rate", 0.0) or 0.0)

            fec_final_tests_raw = stats_fec.get("final_test_values")
            fec_final_tests = (
                np.asarray(fec_final_tests_raw, dtype=float)
                if fec_final_tests_raw is not None
                else np.array([])
            )
            p_value = None
            significant = None
            p_value_baseline_better = None
            accuracy_comparable = None
            test_name = None
            if baseline_final_tests.size >= 2 and fec_final_tests.size >= 2:
                try:
                    _, p_val = stats.mannwhitneyu(
                        baseline_final_tests,
                        fec_final_tests,
                        alternative="two-sided",
                    )
                    p_value = float(p_val)
                    significant = bool(p_value < 0.05)
                    _, p_one = stats.mannwhitneyu(
                        baseline_final_tests,
                        fec_final_tests,
                        alternative="greater",
                    )
                    p_value_baseline_better = float(p_one)
                    accuracy_comparable = p_value_baseline_better > 0.05
                    test_name = "mannwhitneyu_two_sided"
                except Exception:
                    pass

            summary_rows.append(
                {
                    "mode": method_label,
                    "sample_fraction": frac,
                    "final_test_fitness_mean": fec_final_test,
                    "accuracy": fec_accuracy,
                    "total_time_mean_sec": fec_time,
                    "speedup_vs_baseline": speedup,
                    "hit_rate": fec_hit_rate,
                    "fake_hit_rate": fec_fake_hit_rate,
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

    # 1) For each sample fraction: compare baseline vs all FEC methods (training + testing)
    for frac in used_fractions:
        fig_train = go.Figure()
        fig_train.add_trace(
            go.Scatter(
                x=baseline_df["gen"],
                y=baseline_df["avg"],
                mode="lines+markers",
                name="Baseline (no FEC)",
                error_y=dict(type="data", array=baseline_df.get("avg_std"), visible=True),
            )
        )
        for method_label in fec_results:
            if frac not in fec_results[method_label]:
                continue
            df_fec = fec_results[method_label][frac]
            fig_train.add_trace(
                go.Scatter(
                    x=df_fec["gen"],
                    y=df_fec["avg"],
                    mode="lines+markers",
                    name=f"{method_label}, frac={frac:.0%}",
                    error_y=dict(type="data", array=df_fec.get("avg_std"), visible=True),
                )
            )
        fig_train.update_layout(
            title=f"Training MAE over generations ({frac:.0%} sample)",
            xaxis_title="Generation",
            yaxis_title="MAE (classification error, lower is better)",
            template="plotly_white",
            hovermode="x unified",
        )
        sections.append(pio.to_html(fig_train, include_plotlyjs="cdn", full_html=False))

        fig_test = go.Figure()
        fig_test.add_trace(
            go.Scatter(
                x=baseline_df["gen"],
                y=baseline_df["fitness_test"],
                mode="lines+markers",
                name="Baseline (no FEC)",
                error_y=dict(type="data", array=baseline_df.get("fitness_test_std"), visible=True),
            )
        )
        for method_label in fec_results:
            if frac not in fec_results[method_label]:
                continue
            df_fec = fec_results[method_label][frac]
            fig_test.add_trace(
                go.Scatter(
                    x=df_fec["gen"],
                    y=df_fec["fitness_test"],
                    mode="lines+markers",
                    name=f"{method_label}, frac={frac:.0%}",
                    error_y=dict(type="data", array=df_fec.get("fitness_test_std"), visible=True),
                )
            )
        fig_test.update_layout(
            title=f"Test MAE over generations ({frac:.0%} sample)",
            xaxis_title="Generation",
            yaxis_title="MAE (classification error, lower is better)",
            template="plotly_white",
            hovermode="x unified",
        )
        sections.append(pio.to_html(fig_test, include_plotlyjs="cdn", full_html=False))

        # Hit rate vs generation: add series only when corresponding fec.modes flag is True
        fig_hit_gen = go.Figure()
        fig_fake_gen = go.Figure()
        show_total = _fec_show_total()
        show_behav_no_struct = _fec_show_behaviour_without_structural()
        show_just_struct = _fec_show_structural_only()
        for method_label in fec_results:
            if frac not in fec_results[method_label]:
                continue
            stats_fec = fec_summaries[method_label].get(frac, {})
            gen_fake_mean = stats_fec.get("gen_fake_hit_rate_mean")
            gen_fake_std = stats_fec.get("gen_fake_hit_rate_std")
            gen_seq = fec_results[method_label][frac]["gen"]
            r_behav = stats_fec.get("gen_rate_behavioural_or_both_mean")
            r_behav_no = stats_fec.get("gen_rate_behavioural_without_structural_mean")
            r_just = stats_fec.get("gen_rate_just_structural_mean")
            if r_behav is not None:
                if show_total:
                    fig_hit_gen.add_trace(
                        go.Scatter(
                            x=gen_seq,
                            y=r_behav,
                            mode="lines+markers",
                            name=f"{method_label} (behavioural), frac={frac:.0%}",
                            error_y=dict(type="data", array=stats_fec.get("gen_rate_behavioural_or_both_std") or [], visible=True),
                        )
                    )
                if show_behav_no_struct:
                    fig_hit_gen.add_trace(
                        go.Scatter(
                            x=gen_seq,
                            y=r_behav_no if r_behav_no is not None else [float("nan")] * len(gen_seq),
                            mode="lines+markers",
                            name=f"{method_label} (behavioural without structural), frac={frac:.0%}",
                            error_y=dict(type="data", array=stats_fec.get("gen_rate_behavioural_without_structural_std") or [], visible=True),
                        )
                    )
                if show_just_struct:
                    fig_hit_gen.add_trace(
                        go.Scatter(
                            x=gen_seq,
                            y=r_just if r_just is not None else [float("nan")] * len(gen_seq),
                            mode="lines+markers",
                            name=f"{method_label} (just structural), frac={frac:.0%}",
                            error_y=dict(type="data", array=stats_fec.get("gen_rate_just_structural_std") or [], visible=True),
                        )
                    )
            if gen_fake_mean is not None:
                fig_fake_gen.add_trace(
                    go.Scatter(
                        x=gen_seq,
                        y=gen_fake_mean,
                        mode="lines+markers",
                        name=f"{method_label}, frac={frac:.0%}",
                        error_y=dict(type="data", array=gen_fake_std or [], visible=True),
                    )
                )
        if len(fig_hit_gen.data) > 0:
            ytitle = "Hit rate (per lookups): behavioural | behav without struct | just structural"
            fig_hit_gen.update_layout(
                title=f"Hit rate over generations ({frac:.0%} sample)",
                xaxis_title="Generation",
                yaxis_title=ytitle,
                yaxis_tickformat=".0%",
                template="plotly_white",
                hovermode="x unified",
                showlegend=True,
            )
            sections.append(pio.to_html(fig_hit_gen, include_plotlyjs="cdn", full_html=False))

        # Hit rate (overall: hit vs miss) over generations — only when fec_enabled_total
        fig_hit_overall_gen = go.Figure()
        if show_total:
            for method_label in fec_results:
                if frac not in fec_results[method_label]:
                    continue
                stats_fec = fec_summaries[method_label].get(frac, {})
                gen_hit_mean = stats_fec.get("gen_hit_rate_mean")
                gen_hit_std = stats_fec.get("gen_hit_rate_std")
                gen_seq = fec_results[method_label][frac]["gen"]
                if gen_hit_mean is not None:
                    fig_hit_overall_gen.add_trace(
                        go.Scatter(
                            x=gen_seq,
                            y=gen_hit_mean,
                            mode="lines+markers",
                            name=f"{method_label}, frac={frac:.0%}",
                            error_y=dict(type="data", array=gen_hit_std or [], visible=True),
                        )
                    )
        if len(fig_hit_overall_gen.data) > 0:
            fig_hit_overall_gen.update_layout(
                title=f"Hit rate (overall: hit vs miss) over generations ({frac:.0%} sample)",
                xaxis_title="Generation",
                yaxis_title="Hit rate (hits / lookups)",
                yaxis_tickformat=".0%",
                template="plotly_white",
                hovermode="x unified",
                showlegend=True,
            )
            sections.append(pio.to_html(fig_hit_overall_gen, include_plotlyjs="cdn", full_html=False))

        if len(fig_fake_gen.data) > 0:
            fig_fake_gen.update_layout(
                title=f"Fake-hit rate over generations ({frac:.0%} sample)",
                xaxis_title="Generation",
                yaxis_title="Fake-hit rate",
                yaxis_tickformat=".0%",
                template="plotly_white",
                hovermode="x unified",
                showlegend=True,
            )
            sections.append(pio.to_html(fig_fake_gen, include_plotlyjs="cdn", full_html=False))

    # 2) Final test MAE vs sample fraction: baseline + one trace per sampling method
    baseline_final_test = float(baseline_df["fitness_test"].iloc[-1])
    fig_final = go.Figure()
    fig_final.add_trace(
        go.Scatter(
            x=used_fractions,
            y=[baseline_final_test] * len(used_fractions),
            mode="lines",
            name="Baseline (final test)",
        )
    )
    for method_label in fec_results:
        y_vals = []
        for f in used_fractions:
            if f in fec_results[method_label] and "fitness_test" in fec_results[method_label][f]:
                y_vals.append(float(fec_results[method_label][f]["fitness_test"].iloc[-1]))
            else:
                y_vals.append(float("nan"))
        fig_final.add_trace(
            go.Scatter(
                x=used_fractions,
                y=y_vals,
                mode="lines+markers",
                name=method_label,
            )
        )
    fig_final.update_layout(
        title="Test MAE across sample fractions",
        xaxis_title="Sample fraction",
        yaxis_title="Final test MAE (classification error, lower is better)",
        template="plotly_white",
        hovermode="x unified",
    )
    sections.append(pio.to_html(fig_final, include_plotlyjs="cdn", full_html=False))

    # 3) Hit rate vs sample fraction (breakdown): only series whose fec.modes flag is True
    show_total_g = _fec_show_total()
    show_behav_no_struct_g = _fec_show_behaviour_without_structural()
    show_just_struct_g = _fec_show_structural_only()
    fig_hit = go.Figure()
    for method_label in fec_results:
        rate_behav = [float(fec_summaries[method_label].get(f, {}).get("rate_behavioural_or_both", 0.0) or 0.0) for f in used_fractions]
        rate_behav_no = [float(fec_summaries[method_label].get(f, {}).get("rate_behavioural_without_structural", 0.0) or 0.0) for f in used_fractions]
        rate_just = [float(fec_summaries[method_label].get(f, {}).get("rate_just_structural", 0.0) or 0.0) for f in used_fractions]
        if show_total_g:
            fig_hit.add_trace(go.Scatter(x=used_fractions, y=rate_behav, mode="lines+markers", name=f"{method_label} (behavioural)"))
        if show_behav_no_struct_g:
            fig_hit.add_trace(go.Scatter(x=used_fractions, y=rate_behav_no, mode="lines+markers", name=f"{method_label} (behavioural without structural)"))
        if show_just_struct_g:
            fig_hit.add_trace(go.Scatter(x=used_fractions, y=rate_just, mode="lines+markers", name=f"{method_label} (just structural)"))
    if len(fig_hit.data) > 0:
        ytitle_hit = "Hit rate (per lookups): behavioural | behav without struct | just structural"
        fig_hit.update_layout(
            title="Hit rate across sample fractions",
            xaxis_title="Sample fraction",
            yaxis_title=ytitle_hit,
            yaxis_tickformat=".0%",
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
        )
        sections.append(pio.to_html(fig_hit, include_plotlyjs="cdn", full_html=False))

    # 3b) Hit rate (overall: hit vs miss) across sample fractions — only when fec_enabled_total
    fig_hit_overall = go.Figure()
    if show_total_g:
        for method_label in fec_results:
            hit_rates = [
                float(fec_summaries[method_label].get(f, {}).get("hit_rate", 0.0) or 0.0)
                for f in used_fractions
            ]
            fig_hit_overall.add_trace(
                go.Scatter(x=used_fractions, y=hit_rates, mode="lines+markers", name=method_label)
            )
    if len(fig_hit_overall.data) > 0:
        fig_hit_overall.update_layout(
            title="Hit rate (overall: hit vs miss) across sample fractions",
            xaxis_title="Sample fraction",
            yaxis_title="Hit rate (hits / lookups)",
            yaxis_tickformat=".0%",
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
        )
        sections.append(pio.to_html(fig_hit_overall, include_plotlyjs="cdn", full_html=False))

    # 4) Fake-hit rate vs sample fraction: one trace per sampling method
    fig_fake = go.Figure()
    for method_label in fec_results:
        fake_rates = [
            float(fec_summaries[method_label].get(f, {}).get("fake_hit_rate", 0.0) or 0.0)
            for f in used_fractions
        ]
        fig_fake.add_trace(
            go.Scatter(x=used_fractions, y=fake_rates, mode="lines+markers", name=method_label)
        )
    fig_fake.update_layout(
        title="Fake-hit rate across sample fractions",
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

    # Build a human-readable dataset description based on CONFIG (local CSV only).
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


