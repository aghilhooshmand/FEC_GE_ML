from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from deap import base, creator, tools
from sklearn.model_selection import train_test_split

import grape.grape as grape
import grape.algorithms
from sampling_methods import get_sampling_function

# Current run / generation for optional per-evaluation CSV logging (set by grape.algorithms.ge_eaSimpleWithElitism).
_FEC_LOG_RUN: Any = None
_FEC_LOG_GEN: int = 0


def set_fec_eval_context(run_id: Any, gen: int) -> None:
    """Called from ge_eaSimpleWithElitism so create_fec_fitness can log run and gen."""
    global _FEC_LOG_RUN, _FEC_LOG_GEN
    _FEC_LOG_RUN = run_id
    _FEC_LOG_GEN = int(gen)


def _fec_log_run_gen() -> tuple[Any, int]:
    return _FEC_LOG_RUN, _FEC_LOG_GEN


@dataclass
class SimpleExperimentResult:
    config: Dict[str, Any]
    logbook: tools.Logbook
    per_run_table: pd.DataFrame
    cache_stats: Dict[str, Any]


def load_dataset(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Self-contained dataset loader for the simple pipeline.
    - Reads CSV from ./data/<dataset.file>
    - Applies simple preprocessing (drop missing)
    - Optional one-time simple oversampling (dataset.smoth_balance)
    - Optional stratified subsampling (dataset.sample_fraction)
    """
    label_column = cfg.get("dataset.label_column")

    data_dir = Path.cwd() / "data"
    csv_path = data_dir / str(cfg["dataset.file"])

    df = pd.read_csv(csv_path)
    df = df.replace("?", np.nan).dropna(axis=0).reset_index(drop=True)

    # Determine label column.
    if label_column and label_column in df.columns:
        label_col = str(label_column)
    else:
        label_col = str(df.columns[-1])
    cfg["dataset.label_column"] = label_col

    # Optional simple oversampling cache (kept for parity with prior behavior).
    if bool(cfg.get("dataset.smoth_balance", False)):
        smoth_out_path = data_dir / f"{Path(str(cfg['dataset.file'])).stem}_smoth_{label_col}.csv"
        if smoth_out_path.exists():
            df = pd.read_csv(smoth_out_path)
            df = df.replace("?", np.nan).dropna(axis=0).reset_index(drop=True)
        else:
            y_all = df[label_col].astype(int)
            classes, counts = np.unique(y_all, return_counts=True)
            if classes.size == 2:
                n_target = int(counts.max())
                n_min = int(counts.min())
                min_class = int(classes[np.argmin(counts)])
                if n_min < n_target:
                    rng = np.random.default_rng(42)
                    pos = np.arange(len(df))
                    pos_min = pos[y_all == min_class]
                    n_to_add = n_target - n_min
                    add_pos = rng.choice(pos_min, size=n_to_add, replace=True)
                    new_pos = np.concatenate([pos, add_pos])
                    perm = rng.permutation(new_pos.shape[0])
                    df = df.iloc[new_pos[perm]].reset_index(drop=True)
            smoth_out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(smoth_out_path, index=False)

    # Optional stratified subsampling.
    sample_fraction = cfg.get("dataset.sample_fraction")
    if sample_fraction is not None:
        try:
            frac = float(sample_fraction)
        except (TypeError, ValueError):
            frac = 1.0
        if 0.0 < frac < 1.0:
            y_all = df[label_col]
            df, _ = train_test_split(
                df,
                train_size=frac,
                stratify=y_all,
                random_state=int(cfg.get("evolution.random_seed", 42)),
            )

    y = df[label_col].astype(int).to_numpy()
    X = df.drop(columns=[label_col]).to_numpy(dtype=float)
    return X, y


def load_operators(operators_dir: Path) -> Dict[str, Any]:
    with operators_dir.joinpath("custom_operators.json").open("r", encoding="utf-8") as handle:
        custom_ops = json.load(handle)

    namespace: Dict[str, Any] = {}
    for _, op_data in custom_ops.get("operators", {}).items():
        exec(op_data.get("function_code", ""), {"np": np}, namespace)
    print(f"Loaded {len(namespace)} operators from {operators_dir}")
    return namespace


def baseline_fitness_eval(individual: Any, points: Sequence[np.ndarray], operators: Dict[str, Any]) -> Tuple[float]:
    x = np.asarray(points[0], dtype=np.float64)
    y_true = np.asarray(points[1], dtype=np.int64)
    if getattr(individual, "invalid", False):
        return (float("nan"),)

    env = {"np": np, "x": x}
    env.update(operators)
    try:
        prediction = eval(individual.phenotype, env)
    except (FloatingPointError, ZeroDivisionError, OverflowError, MemoryError, IndexError):
        return (float("nan"),)

    if not np.isrealobj(prediction):
        return (float("nan"),)

    prediction = np.asarray(prediction, dtype=np.float64).flatten()
    if prediction.shape[0] != y_true.shape[0]:
        return (float("nan"),)

    try:
        y_pred = (prediction > 0).astype(int)
        fitness = 1 - np.mean(np.equal(y_true, y_pred))
    except (IndexError, TypeError):
        return (float("nan"),)

    return (fitness,)


def prepare_toolbox(cfg: Dict[str, Any], operators: Dict[str, Any], grammar: grape.Grammar) -> base.Toolbox:
    toolbox = base.Toolbox()

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", grape.Individual, fitness=creator.FitnessMin)

    toolbox.register("populationCreator", grape.sensible_initialisation, creator.Individual)

    toolbox.population_kwargs = dict(
        bnf_grammar=grammar,
        min_init_depth=int(cfg["ge_parameters.min_init_tree_depth"]),
        max_init_depth=int(cfg["ge_parameters.max_init_tree_depth"]),
        codon_size=int(cfg["ge_parameters.codon_size"]),
        codon_consumption=cfg["ge_parameters.codon_consumption"],
        genome_representation=cfg["ge_parameters.genome_representation"],
    )

    toolbox.register("select", tools.selTournament, tournsize=int(cfg["ga_parameters.tournsize"]))
    toolbox.register("mate", grape.crossover_onepoint)
    toolbox.register("mutate", grape.mutation_int_flip_per_codon)

    return toolbox


class SimpleFECCache:
    def __init__(self) -> None:
        # Simple in-memory hash table (Python dict)
        self.cache: Dict[object, float] = {}
        # Phenotype string stored on first insert for a key (for analysis CSV).
        self.cache_phenotype: Dict[object, str] = {}
        self.hits = 0
        self.misses = 0
        self.fake_hits = 0
        self.fake_eval_time_sec = 0.0

    def lookup(self, key: object) -> Tuple[bool, float, str]:
        if key in self.cache:
            self.hits += 1
            return True, self.cache[key], self.cache_phenotype.get(key, "")
        self.misses += 1
        return False, 0.0, ""

    def store(self, key: object, fitness: float, phenotype: str = "") -> None:
        self.cache[key] = fitness
        if key not in self.cache_phenotype and phenotype:
            self.cache_phenotype[key] = phenotype


def _hash_key_bytes(data: bytes) -> str:
    """Stable small key for caching (faster and smaller than repr())."""
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def _make_cache_key(
    key_mode: str,
    *,
    individual: Any,
    pred_sample: np.ndarray,
    centroid_y: np.ndarray,
) -> object:
    km = str(key_mode or "behavior_hash").lower()
    if km == "phenotype":
        return str(getattr(individual, "phenotype", ""))
    if km == "phenotype_hash":
        return _hash_key_bytes(str(getattr(individual, "phenotype", "")).encode("utf-8", errors="ignore"))

    # Behaviour-based key (default)
    # Round predictions for stability then hash the bytes.
    pred_r = np.round(np.asarray(pred_sample, dtype=np.float64), 6)
    y_b = np.asarray(centroid_y, dtype=np.int64)
    if km == "behavior_repr":
        return repr((tuple(pred_r.tolist()), tuple(y_b.tolist())))
    # behavior_hash
    payload = pred_r.tobytes() + b"|" + y_b.tobytes()
    return _hash_key_bytes(payload)


def create_fec_fitness(
    centroid_X: np.ndarray,
    centroid_y: np.ndarray,
    operators: Dict[str, Any],
    cache: SimpleFECCache,
    evaluate_fake_hits: bool,
    fake_hit_threshold: float,
    cache_key: str = "behavior_hash",
    individual_log_rows: Optional[List[Dict[str, Any]]] = None,
) -> Any:
    # fake_hit_threshold kept for API/config parity; fake-hit detection uses exact matching only.

    def _phenotype_str(ind: Any) -> str:
        return str(getattr(ind, "phenotype", "") or "")

    def _append_log(
        *,
        individual: Any,
        hit: bool,
        cache_key_val: object,
        phenotype_in_cache: str,
        fitness_cached: float,
        fitness_full: float,
        fake_hit_val: Any,
        fitness_diff: Any,
    ) -> None:
        if individual_log_rows is None:
            return
        run, gen = _fec_log_run_gen()
        row: Dict[str, Any] = {
            "run": run if run is not None else "",
            "gen": gen,
            "phenotype": _phenotype_str(individual),
            "hit": hit,
            "phenotype_in_cache": phenotype_in_cache if hit else "",
            "fitness_cached": fitness_cached,
            "cache_key": str(cache_key_val),
            "fitness_full_train": fitness_full,
            "fake_hit": fake_hit_val,
            "fitness_diff": fitness_diff,
        }
        individual_log_rows.append(row)

    def fitness_eval(individual: Any, points: Sequence[np.ndarray], dataset_type: str = "train") -> Tuple[float]:
        x = np.asarray(points[0], dtype=np.float64)
        y_true = np.asarray(points[1], dtype=np.int64)

        if getattr(individual, "invalid", False):
            return (float("nan"),)

        is_training = dataset_type.lower() != "test"

        # Behaviour key on sampled points
        env_sample = {"np": np, "x": centroid_X}
        env_sample.update(operators)
        try:
            pred_sample = eval(individual.phenotype, env_sample)
        except Exception:
            return (float("nan"),)
        pred_sample = np.asarray(pred_sample, dtype=np.float64).flatten()
        if pred_sample.shape[0] != centroid_y.shape[0]:
            return (float("nan"),)
        key = _make_cache_key(
            cache_key,
            individual=individual,
            pred_sample=pred_sample,
            centroid_y=centroid_y,
        )

        use_cache = is_training and cache is not None
        hit = False
        cached = 0.0
        phen_cached = ""

        if use_cache:
            hit, cached, phen_cached = cache.lookup(key)
            if hit and not evaluate_fake_hits:
                if individual_log_rows is not None:
                    _append_log(
                        individual=individual,
                        hit=True,
                        cache_key_val=key,
                        phenotype_in_cache=phen_cached,
                        fitness_cached=float(cached),
                        fitness_full=float("nan"),
                        fake_hit_val="",
                        fitness_diff=float("nan"),
                    )
                return (cached,)

        env = {"np": np, "x": x}
        env.update(operators)

        import time as _t
        t0_fake = _t.perf_counter() if use_cache and hit and evaluate_fake_hits else None

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
            fitness_full = 1 - np.mean(np.equal(y_true, y_pred))
        except Exception:
            return (float("nan"),)

        if use_cache and hit and evaluate_fake_hits and t0_fake is not None:
            dt = float(_t.perf_counter() - t0_fake)
            cache.fake_eval_time_sec += dt
            # Exact match: cached == full fitness (float-safe via np.isclose with atol=rtol=0).
            is_fake = not np.isclose(float(cached), float(fitness_full), atol=0.0, rtol=0.0, equal_nan=True)
            if is_fake:
                cache.fake_hits += 1
            # Signed difference: full_train_mae - cached (positive => full eval higher error than cache)
            diff_ft = float(fitness_full) - float(cached)
            if individual_log_rows is not None:
                _append_log(
                    individual=individual,
                    hit=True,
                    cache_key_val=key,
                    phenotype_in_cache=phen_cached,
                    fitness_cached=float(cached),
                    fitness_full=float(fitness_full),
                    fake_hit_val=is_fake,
                    fitness_diff=diff_ft,
                )
            return (cached,)

        if use_cache and not hit:
            cache.store(key, fitness_full, _phenotype_str(individual))
            if individual_log_rows is not None:
                _append_log(
                    individual=individual,
                    hit=False,
                    cache_key_val=key,
                    phenotype_in_cache="",
                    fitness_cached=float("nan"),
                    fitness_full=float(fitness_full),
                    fake_hit_val=False,
                    fitness_diff=0.0,
                )

        return (fitness_full,)

    return fitness_eval


def run_baseline_experiment_simple(
    cfg: Dict[str, Any],
    run_name_suffix: str | None,
    X: np.ndarray,
    y: np.ndarray,
    grammar: grape.Grammar,
    operators: Dict[str, Any],
    results_root: Path,
) -> SimpleExperimentResult:
    toolbox = prepare_toolbox(cfg, operators, grammar)
    toolbox.register("evaluate", lambda ind, pts: baseline_fitness_eval(ind, pts, operators))

    results_root.mkdir(parents=True, exist_ok=True)

    test_size = float(cfg.get("dataset.test_size", 0.2))
    # load_dataset returns X in (samples, features) format.
    # We split in that format, then transpose for the GE operators,
    # matching the behaviour of the full pipeline in util.py.
    X_full = X
    y_full = y
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=test_size, random_state=int(cfg["evolution.random_seed"])
    )
    X_train = X_train.T
    X_test = X_test.T

    pop_size = int(cfg["evolution.population"])
    ngen = int(cfg["evolution.generations"])

    population = toolbox.populationCreator(pop_size=pop_size, **toolbox.population_kwargs)
    hof = tools.HallOfFame(int(cfg["ga_parameters.halloffame_size"]))
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.nanmean)
    stats.register("std", np.nanstd)
    stats.register("min", np.nanmin)
    stats.register("max", np.nanmax)

    report_items = cfg.get("report_items", [])

    population, logbook = grape.algorithms.ge_eaSimpleWithElitism(
        population,
        toolbox,
        cxpb=float(cfg["ga_parameters.p_crossover"]),
        mutpb=float(cfg["ga_parameters.p_mutation"]),
        ngen=ngen,
        elite_size=int(cfg["ga_parameters.elite_size"]),
        bnf_grammar=grammar,
        codon_size=int(cfg["ge_parameters.codon_size"]),
        max_tree_depth=int(cfg["ge_parameters.max_tree_depth"]),
        max_genome_length=cfg.get("ge_parameters.max_genome_length"),
        points_train=[X_train, y_train],
        points_test=[X_test, y_test],
        codon_consumption=cfg["ge_parameters.codon_consumption"],
        report_items=report_items,
        genome_representation=cfg["ge_parameters.genome_representation"],
        stats=stats,
        halloffame=hof,
        verbose=False,
    )

    if report_items and len(logbook) > 0:
        table_data = list(zip(*[logbook.select(col) for col in report_items]))
        per_run_df = pd.DataFrame(table_data, columns=report_items)
    else:
        per_run_df = pd.DataFrame()

    return SimpleExperimentResult(
        config=cfg,
        logbook=logbook,
        per_run_table=per_run_df,
        cache_stats={},
    )


def run_fec_experiment_simple(
    cfg: Dict[str, Any],
    run_name_suffix: str | None,
    X: np.ndarray,
    y: np.ndarray,
    grammar: grape.Grammar,
    operators: Dict[str, Any],
    results_root: Path,
) -> SimpleExperimentResult:
    toolbox = prepare_toolbox(cfg, operators, grammar)

    results_root.mkdir(parents=True, exist_ok=True)

    test_size = float(cfg.get("dataset.test_size", 0.2))
    # Same convention as baseline_simple: split in (samples, features),
    # then transpose to (features, samples) for GE and sampling.
    X_full = X
    y_full = y
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=test_size, random_state=int(cfg["evolution.random_seed"])
    )
    X_train = X_train.T
    X_test = X_test.T

    # Sampling for behaviour key
    sample_fraction = float(cfg.get("fec.sample_fraction", 0.1))
    train_set_size = X_train.shape[1]
    sample_size = max(1, int(round(sample_fraction * train_set_size)))

    sampling_method = cfg.get("fec.sampling_method", "farthest_point")
    sampling_func = get_sampling_function(sampling_method)
    centroid_X, centroid_y, centroid_indices = sampling_func(
        X_train, y_train, sample_size, int(cfg["evolution.random_seed"])
    )

    cache = SimpleFECCache()
    save_ind_csv = bool(cfg.get("output.save_individuals_csv", False))
    individual_log_rows: Optional[List[Dict[str, Any]]] = [] if save_ind_csv else None
    eval_fn = create_fec_fitness(
        centroid_X=centroid_X,
        centroid_y=centroid_y,
        operators=operators,
        cache=cache,
        evaluate_fake_hits=bool(cfg.get("fec.evaluate_fake_hits", False)),
        fake_hit_threshold=float(cfg.get("fec.fake_hit_threshold", 1e-5)),
        cache_key=str(cfg.get("fec.cache_key", "behavior_hash")),
        individual_log_rows=individual_log_rows,
    )
    toolbox.register("evaluate", eval_fn)

    pop_size = int(cfg["evolution.population"])
    ngen = int(cfg["evolution.generations"])

    population = toolbox.populationCreator(pop_size=pop_size, **toolbox.population_kwargs)
    hof = tools.HallOfFame(int(cfg["ga_parameters.halloffame_size"]))
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.nanmean)
    stats.register("std", np.nanstd)
    stats.register("min", np.nanmin)
    stats.register("max", np.nanmax)

    report_items = cfg.get("report_items", [])

    run_id_for_log = cfg.get("fec._individual_log_run")

    population, logbook = grape.algorithms.ge_eaSimpleWithElitism(
        population,
        toolbox,
        cxpb=float(cfg["ga_parameters.p_crossover"]),
        mutpb=float(cfg["ga_parameters.p_mutation"]),
        ngen=ngen,
        elite_size=int(cfg["ga_parameters.elite_size"]),
        bnf_grammar=grammar,
        codon_size=int(cfg["ge_parameters.codon_size"]),
        max_tree_depth=int(cfg["ge_parameters.max_tree_depth"]),
        max_genome_length=cfg.get("ge_parameters.max_genome_length"),
        points_train=[X_train, y_train],
        points_test=[X_test, y_test],
        codon_consumption=cfg["ge_parameters.codon_consumption"],
        report_items=report_items,
        genome_representation=cfg["ge_parameters.genome_representation"],
        stats=stats,
        halloffame=hof,
        verbose=False,
        run_id=run_id_for_log,
    )

    if report_items and len(logbook) > 0:
        table_data = list(zip(*[logbook.select(col) for col in report_items]))
        per_run_df = pd.DataFrame(table_data, columns=report_items)
    else:
        per_run_df = pd.DataFrame()

    cache_stats = {
        "hits": cache.hits,
        "misses": cache.misses,
        "fake_hits": cache.fake_hits,
        "fake_eval_time_sec": cache.fake_eval_time_sec,
        "hit_rate_overall": (
            cache.hits / (cache.hits + cache.misses)
            if (cache.hits + cache.misses) > 0
            else 0.0
        ),
    }

    if individual_log_rows is not None and len(individual_log_rows) > 0:
        tag = str(cfg.get("fec._individual_log_file_tag", "fec_run"))
        out_csv = results_root / f"fec_individual_cache_{tag}.csv"
        _cols = [
            "run",
            "gen",
            "phenotype",
            "hit",
            "phenotype_in_cache",
            "fitness_cached",
            "cache_key",
            "fitness_full_train",
            "fake_hit",
            "fitness_diff",
        ]
        df_log = pd.DataFrame(individual_log_rows)
        df_log = df_log[[c for c in _cols if c in df_log.columns]]
        df_log.to_csv(out_csv, index=False)
        print(f"Saved FEC individual cache log: {out_csv}")

    return SimpleExperimentResult(
        config=cfg,
        logbook=logbook,
        per_run_table=per_run_df,
        cache_stats=cache_stats,
    )

