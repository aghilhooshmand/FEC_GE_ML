from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import html

import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from deap import base, creator, tools
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.spatial.distance import cdist

import grape.grape as grape
import grape.algorithms
from config import CONFIG
from sampling_methods import get_sampling_function


def get_large_font_layout() -> Dict[str, Any]:
    """
    Returns font configuration for larger, more readable charts (useful for screenshots).

    IMPORTANT:
        This helper is merged into existing layout dictionaries using::

            layout_updates.update(get_large_font_layout())

        Therefore it MUST NOT overwrite the existing ``title`` text. We only
        provide font settings (via ``title_font`` and axis / legend font keys),
        so chart titles remain visible.
    """
    return {
        # Use title_font so we don't clobber the existing title string
        "title_font": dict(size=20),
        "xaxis": dict(
            title=dict(font=dict(size=16)),
            tickfont=dict(size=14),
        ),
        "yaxis": dict(
            title=dict(font=dict(size=16)),
            tickfont=dict(size=14),
        ),
        "legend": dict(font=dict(size=14)),
    }


def apply_large_fonts_to_layout(layout_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges large font settings into an existing layout dictionary.
    Use this to update any chart layout with larger fonts.
    """
    result = layout_dict.copy()
    large_fonts = get_large_font_layout()
    
    # Merge title font
    if "title" in result:
        if isinstance(result["title"], dict):
            result["title"].update(large_fonts["title"])
        else:
            result["title_font"] = large_fonts["title"]["font"]
    else:
        result.update(large_fonts["title"])
    
    # Merge xaxis and yaxis fonts
    for axis in ["xaxis", "yaxis"]:
        if axis in result:
            if isinstance(result[axis], dict):
                result[axis].update(large_fonts[axis])
            else:
                # If it's a string (like xaxis_title), we need to handle differently
                pass
        else:
            result[axis] = large_fonts[axis].copy()
    
    # Merge legend font
    if "legend" in result:
        if isinstance(result["legend"], dict):
            result["legend"].update(large_fonts["legend"])
    else:
        result["legend"] = large_fonts["legend"].copy()
    
    return result

__all__ = [
    "derive_behavior_key",
    "validate_config",
    "load_operators",
    "load_dataset",
    "configure_mlflow",
    "log_config_to_mlflow",
    "FECCache",
    "PhenotypeTracker",
    "create_fitness_eval",
    "baseline_fitness_eval",
    "prepare_toolbox",
    "ExperimentResult",
    "run_configured_experiment",
    "plot_comparison",
    "compare_fec_modes",
    "generate_sample_size_comparison",
    "generate_consolidated_html_report",
    "auto_select_sampling_method",
    "compute_statistical_distance",
]


# ---------------------------------------------------------------------------
# Behaviour key helpers
# ---------------------------------------------------------------------------


def derive_behavior_key(
    individual: Any,
    centroid_X: Optional[np.ndarray],
    centroid_y: Optional[np.ndarray],
    centroid_indices: Optional[Sequence[int]],
    operators: Dict[str, Any],
    cache: Optional["FECCache"] = None,
) -> Optional[str]:
    """Return a behaviour signature string for centroid evaluation, or ``None`` if it cannot be built."""
    if centroid_X is None or centroid_y is None or centroid_indices is None:
        return None

    centroid_X_copy = np.asarray(centroid_X, dtype=np.float64).copy()
    centroid_y_copy = np.asarray(centroid_y, dtype=np.int64).copy()
    eval_env_centroid = {"np": np, "x": centroid_X_copy}
    eval_env_centroid.update(operators)

    try:
        pred_centroid = eval(individual.phenotype, eval_env_centroid)
        if not np.isrealobj(pred_centroid):
            raise TypeError
        pred_centroid = np.asarray(pred_centroid, dtype=np.float64).flatten().copy()
        if len(pred_centroid) != len(centroid_y_copy):
            return None

        # --- Behaviour key sampling configuration -------------------------
        # How many centroid points to use in the behaviour fingerprint?
        # Controlled by:
        #   - fec.behavior_key.sample_fraction  (0-1], default 1.0
        #   - fec.behavior_key.max_points      (int > 0, default 100)
        #   - fec.behavior_key.random_subset   (bool, default False)
        total_points = len(centroid_y_copy)
        sample_fraction = float(CONFIG.get("fec.behavior_key.sample_fraction", 1.0) or 1.0)
        max_points_cfg = CONFIG.get("fec.behavior_key.max_points", 100)
        random_subset = bool(CONFIG.get("fec.behavior_key.random_subset", False))

        # Clamp fraction to (0, 1]; anything outside is treated as full.
        if not (0.0 < sample_fraction <= 1.0):
            sample_fraction = 1.0

        n_by_fraction = max(1, int(round(total_points * sample_fraction)))

        if isinstance(max_points_cfg, int) and max_points_cfg > 0:
            n_points = min(total_points, n_by_fraction, max_points_cfg)
        else:
            n_points = min(total_points, n_by_fraction)

        if n_points >= total_points or total_points == 0:
            # Use all points (or empty)
            indices = np.arange(total_points, dtype=int)
        else:
            if random_subset:
                # Deterministic random subset based on phenotype string,
                # so the same individual + centroids give the same subset.
                seed = abs(hash(individual.phenotype)) % (2**32)
                rng = np.random.default_rng(seed)
                indices = rng.choice(total_points, size=n_points, replace=False)
                indices.sort()
            else:
                # Take the first n_points (fast, deterministic)
                indices = np.arange(n_points, dtype=int)

        pred_selected = pred_centroid[indices]
        labels_selected = centroid_y_copy[indices]

        pred_str = ",".join(f"{value:.6f}" for value in pred_selected)
        label_str = ",".join(str(label) for label in labels_selected)
        sample_str = ",".join(str(idx) for idx in centroid_indices) if centroid_indices is not None else "full"
        if cache is not None:
            cache.record_sample_eval()
        return f"{pred_str}|{label_str}|{sample_str}"
    except (FloatingPointError, ZeroDivisionError, OverflowError, MemoryError, IndexError, TypeError):
        return None



# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------




def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_fingerprint(
    phenotype_component: Optional[str],
    behavior_key: Optional[str],
) -> str:
    parts = []
    if phenotype_component is not None:
        parts.append(f"phen:{phenotype_component}")
    if behavior_key is not None:
        parts.append(f"beh:{behavior_key}")
    if not parts:
        raise ValueError("Fingerprint requires at least one component.")
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()


# def validate_config(cfg: Dict[str, Any]) -> None:
#     """Validate mandatory configuration sections and raise an informative error if invalid."""
#     errors: List[str] = []

#     dataset_cfg = cfg.get("dataset", {})
#     dataset_file = dataset_cfg.get("file")
#     if not dataset_file:
#         errors.append("dataset.file must point to a CSV under ./data/")
#     else:
#         if not (Path.cwd() / "data" / dataset_file).exists():
#             errors.append(f"dataset file '{dataset_file}' not found under ./data/")
#     if dataset_cfg.get("test_size") is None:
#         errors.append("dataset.test_size must be specified (float between 0 and 1).")

#     grammar_cfg = cfg.get("grammar", {})
#     grammar_file = grammar_cfg.get("file")
#     if not grammar_file:
#         errors.append("grammar.file must point to a .bnf grammar under ./grammars/")
#     else:
#         if not (Path.cwd() / "grammars" / grammar_file).exists():
#             errors.append(f"grammar file '{grammar_file}' not found under ./grammars/")

#     evolution_cfg = cfg.get("evolution", {})
#     for key in ("population", "generations", "random_seed", "n_runs"):
#         if evolution_cfg.get(key) is None:
#             errors.append(f"evolution.{key} must be set (integer).")

#     fec_cfg = cfg.get("fec", {})
#     if fec_cfg.get("enabled", False):
#         sample_size = fec_cfg.get("sample_size")
#         sample_sizes = fec_cfg.get("sample_sizes")
#         if sample_sizes is not None:
#             if not isinstance(sample_sizes, (list, tuple)) or len(sample_sizes) == 0:
#                 errors.append("fec.sample_sizes must be a non-empty list when provided.")
#             else:
#                 for value in sample_sizes:
#                     if not isinstance(value, (int, float, np.integer, np.floating)):
#                         errors.append("fec.sample_sizes must contain numeric values (int or float).")
#                         break
#                     numeric_value = float(value)
#                     if numeric_value <= 0:
#                         errors.append("fec.sample_sizes values must be > 0.")
#                         break
#         if (not sample_size or sample_size <= 0) and not sample_sizes:
#             errors.append("fec.sample_size must be > 0 when FEC is enabled (unless fec.sample_sizes is set).")
#         if not (fec_cfg.get("structural_similarity") or fec_cfg.get("behavior_similarity")):
#             errors.append("At least one of structural_similarity / behavior_similarity must be True when FEC is enabled.")

#     mlflow_cfg = cfg.get("mlflow", {})
#     if mlflow_cfg.get("enabled", False) and not mlflow_cfg.get("tracking_uri"):
#         errors.append("mlflow.tracking_uri must be set when MLflow tracking is enabled.")

#     if errors:
#         joined = "\n - ".join(errors)
#         raise ValueError(f"Configuration validation failed:\n - {joined}")


def load_operators(operators_dir: Path) -> Dict[str, Any]:
    with operators_dir.joinpath("custom_operators.json").open("r", encoding="utf-8") as handle:
        custom_ops = json.load(handle)

    namespace: Dict[str, Any] = {}
    for name, op_data in custom_ops.get("operators", {}).items():
        exec(op_data.get("function_code", ""), {"np": np}, namespace)
    print(f"Loaded {len(namespace)} operators from {operators_dir}")
    return namespace


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple preprocessing:
    - Treat "?" as missing (NaN) if still present as strings.
    - Drop any rows that contain missing values.
    """
    # Replace any lingering "?" strings with NaN, then drop rows with NaN
    df = df.replace("?", np.nan)
    df = df.dropna(axis=0).reset_index(drop=True)
    return df


def _build_config_summary_html(cfg: Dict[str, Any]) -> str:
    """Render a simple HTML table with all config key/value pairs."""
    rows = []
    for key, value in sorted(cfg.items()):
        key_esc = html.escape(str(key))
        val_esc = html.escape(str(value))
        rows.append(f"<tr><td><code>{key_esc}</code></td><td>{val_esc}</td></tr>")
    table_html = (
        "<h2>Configuration Summary</h2>"
        "<table border='1' cellspacing='0' cellpadding='4'>"
        "<thead><tr><th>Key</th><th>Value</th></tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table><hr>"
    )
    return table_html


def load_dataset(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from CSV, then apply a very simple preprocessing step
    (handled by preprocess_dataframe).
    """
    data_dir = Path.cwd() / "data"
    csv_path = data_dir / cfg["dataset.file"]
    label_column = cfg.get("dataset.label_column")

    # Read CSV (raw)
    df = pd.read_csv(csv_path)
    # Apply simple preprocessing
    df = preprocess_dataframe(df)

    if label_column in df.columns:
        y = df[label_column].astype(int).to_numpy()
        X = df.drop(columns=[label_column]).to_numpy(dtype=float)
    else:
        y = df.iloc[:, -1].astype(int).to_numpy()
        X = df.iloc[:, :-1].to_numpy(dtype=float)
    return X, y


# ---------------------------------------------------------------------------
# Auto-selection of sampling methods
# ---------------------------------------------------------------------------


def _compute_ks_distance(X_full: np.ndarray, X_sample: np.ndarray) -> float:
    """Compute Kolmogorov-Smirnov distance (average over features)."""
    distances = []
    for feat_idx in range(X_full.shape[0]):
        full_feat = X_full[feat_idx, :]
        sample_feat = X_sample[feat_idx, :]
        ks_stat, _ = stats.ks_2samp(full_feat, sample_feat)
        distances.append(ks_stat)
    return np.mean(distances) if distances else 0.0


def _compute_wasserstein_distance(X_full: np.ndarray, X_sample: np.ndarray) -> float:
    """Compute Wasserstein distance (multivariate, using 1D per feature then average)."""
    distances = []
    for feat_idx in range(X_full.shape[0]):
        full_feat = X_full[feat_idx, :]
        sample_feat = X_sample[feat_idx, :]
        # Use scipy's wasserstein_distance for 1D
        wd = stats.wasserstein_distance(full_feat, sample_feat)
        distances.append(wd)
    return np.mean(distances) if distances else 0.0


def _compute_energy_distance(X_full: np.ndarray, X_sample: np.ndarray) -> float:
    """Compute Energy distance (E-test style, simplified version)."""
    # Simplified: compute pairwise distances within and between sets
    # Full implementation would use energy distance formula
    # For now, use a proxy: mean distance between sample and full set
    n_full = X_full.shape[1]
    n_sample = X_sample.shape[1]
    
    # Sample a subset of full for efficiency if too large
    if n_full > 1000:
        indices = np.random.choice(n_full, size=min(1000, n_full), replace=False)
        X_full_subset = X_full[:, indices]
    else:
        X_full_subset = X_full
    
    # Compute mean distance from each sample point to nearest full point
    distances = []
    for i in range(n_sample):
        sample_point = X_sample[:, i:i+1]  # Column vector
        dists = np.linalg.norm(X_full_subset.T - sample_point.T, axis=1)
        distances.append(np.mean(dists))
    
    return np.mean(distances) if distances else 0.0


def _compute_mmd_distance(X_full: np.ndarray, X_sample: np.ndarray) -> float:
    """Compute Maximum Mean Discrepancy (simplified, using RBF kernel approximation)."""
    # Simplified MMD: use mean pairwise distances as proxy
    # Full MMD would use kernel functions
    n_full = X_full.shape[1]
    n_sample = X_sample.shape[1]
    
    # Sample subsets for efficiency
    if n_full > 500:
        full_indices = np.random.choice(n_full, size=500, replace=False)
        X_full_subset = X_full[:, full_indices]
    else:
        X_full_subset = X_full
    
    # Compute mean pairwise distances within each set and between sets
    # MMD proxy: difference between within-set and cross-set distances
    full_dists = []
    sample_dists = []
    cross_dists = []
    
    # Within full set (sample pairs)
    for _ in range(min(100, X_full_subset.shape[1])):
        i, j = np.random.choice(X_full_subset.shape[1], size=2, replace=False)
        full_dists.append(np.linalg.norm(X_full_subset[:, i] - X_full_subset[:, j]))
    
    # Within sample set
    for _ in range(min(100, n_sample)):
        i, j = np.random.choice(n_sample, size=2, replace=False)
        sample_dists.append(np.linalg.norm(X_sample[:, i] - X_sample[:, j]))
    
    # Cross distances
    for _ in range(min(200, X_full_subset.shape[1] * n_sample)):
        i = np.random.randint(X_full_subset.shape[1])
        j = np.random.randint(n_sample)
        cross_dists.append(np.linalg.norm(X_full_subset[:, i] - X_sample[:, j]))
    
    mean_full = np.mean(full_dists) if full_dists else 0.0
    mean_sample = np.mean(sample_dists) if sample_dists else 0.0
    mean_cross = np.mean(cross_dists) if cross_dists else 0.0
    
    # MMD-like metric: how different are the distributions
    mmd = abs(mean_full + mean_sample - 2 * mean_cross)
    return mmd


def _compute_frechet_distance(X_full: np.ndarray, X_sample: np.ndarray) -> float:
    """Compute Fréchet distance (FID-style, using mean and covariance)."""
    # Fréchet distance between multivariate Gaussians
    # Approximate distributions by their mean and covariance
    mu_full = np.mean(X_full, axis=1)
    mu_sample = np.mean(X_sample, axis=1)
    
    # Compute covariance matrices
    cov_full = np.cov(X_full)
    cov_sample = np.cov(X_sample)
    
    # Ensure matrices are 2D
    if cov_full.ndim == 0:
        cov_full = np.array([[cov_full]])
    if cov_sample.ndim == 0:
        cov_sample = np.array([[cov_sample]])
    
    # Mean difference
    mean_diff = np.linalg.norm(mu_full - mu_sample) ** 2
    
    # Covariance term (trace of product)
    try:
        cov_product = np.dot(cov_full, cov_sample)
        cov_term = np.trace(cov_full) + np.trace(cov_sample) - 2 * np.trace(
            np.real(np.linalg.matrix_power(cov_product, 0.5))
        )
    except (np.linalg.LinAlgError, ValueError):
        # Fallback: use trace difference
        cov_term = np.linalg.norm(cov_full - cov_sample, 'fro')
    
    frechet = mean_diff + cov_term
    return float(frechet)


def compute_statistical_distance(
    X_full: np.ndarray,
    X_sample: np.ndarray,
    metric: str = "average",
) -> float:
    """
    Compute statistical distance between full dataset and sample.
    
    Args:
        X_full: Full dataset (features x samples)
        X_sample: Sampled dataset (features x samples)
        metric: Distance metric name or "average" for average over all metrics
    
    Returns:
        Distance value (lower is better)
    """
    # Transpose for easier feature-wise operations if needed
    # X_full and X_sample are (n_features, n_samples)
    
    if metric == "average":
        # Compute all metrics and average
        metrics_to_compute = ["ks", "wasserstein", "energy", "mmd", "frechet"]
        distances = []
        for m in metrics_to_compute:
            try:
                dist = compute_statistical_distance(X_full, X_sample, m)
                if not np.isnan(dist) and not np.isinf(dist):
                    distances.append(dist)
            except Exception:
                continue
        return np.mean(distances) if distances else float('inf')
    
    elif metric == "ks":
        return _compute_ks_distance(X_full, X_sample)
    elif metric == "wasserstein":
        return _compute_wasserstein_distance(X_full, X_sample)
    elif metric == "energy":
        return _compute_energy_distance(X_full, X_sample)
    elif metric == "mmd":
        return _compute_mmd_distance(X_full, X_sample)
    elif metric == "frechet":
        return _compute_frechet_distance(X_full, X_sample)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")


def auto_select_sampling_method(
    cfg: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
) -> str:
    """
    Automatically select the best sampling method by evaluating how well
    each method's samples represent the full dataset.
    
    Args:
        cfg: Configuration dictionary
        X: Full dataset features (samples x features) - will be transposed internally
        y: Full dataset labels
    
    Returns:
        Name of the best sampling method
    """
    print("\n" + "=" * 72)
    print("Auto-selecting best sampling method...")
    print("=" * 72)
    
    # Get configuration
    n_repetitions = int(cfg.get("fec.auto_select.n_repetitions", 20))
    sample_fraction = float(cfg.get("fec.auto_select.sample_fraction", 0.1))
    distance_metric = cfg.get("fec.auto_select.distance_metric", "average")
    test_size = float(cfg.get("dataset.test_size", 0.2))
    base_seed = int(cfg.get("evolution.random_seed", 42))
    
    # Get candidate methods (exclude "union" as it's composite)
    enabled_flags = cfg.get("fec.sampling_methods.enabled", {})
    candidate_methods = [
        method for method, enabled in enabled_flags.items()
        if enabled and method != "union"
    ]
    
    if not candidate_methods:
        print("Warning: No candidate sampling methods found. Using 'kmeans' as default.")
        return "kmeans"
    
    print(f"Candidate methods: {candidate_methods}")
    print(f"Evaluating with {n_repetitions} repetitions, {sample_fraction:.1%} sample fraction")
    print(f"Distance metric: {distance_metric}")
    
    # Store results for each method
    method_scores: Dict[str, List[float]] = {method: [] for method in candidate_methods}
    
    # Ensure X is in (samples, features) format (load_dataset returns this)
    # Sampling functions expect (features, samples), so we'll transpose when calling them
    # For distance computation, we'll work with (features, samples) format
    
    # Run repetitions
    for rep in range(n_repetitions):
        rep_seed = base_seed + rep * 1000
        
        # Split dataset (X is in samples x features format from load_dataset)
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_size, random_state=rep_seed
        )
        # Convert to (features, samples) for sampling functions
        X_train_full_T = X_train_full.T
        
        # Calculate sample size
        train_size = X_train_full.shape[0]  # Number of samples
        n_samples = max(1, int(round(sample_fraction * train_size)))
        
        # Evaluate each method
        for method in candidate_methods:
            try:
                sampling_func = get_sampling_function(method)
                # Sampling functions expect (features, samples) format
                sample_X, sample_y, sample_indices = sampling_func(
                    X_train_full_T, y_train_full, n_samples, rep_seed
                )
                
                # Compute distance (both should be in features x samples format)
                distance = compute_statistical_distance(
                    X_train_full_T, sample_X, distance_metric
                )
                
                if not np.isnan(distance) and not np.isinf(distance):
                    method_scores[method].append(distance)
            except Exception as e:
                print(f"  Warning: Error evaluating {method} in repetition {rep+1}: {e}")
                continue
        
        if (rep + 1) % 5 == 0:
            print(f"  Completed {rep + 1}/{n_repetitions} repetitions...")
    
    # Compute average distance for each method
    method_averages: Dict[str, float] = {}
    for method, scores in method_scores.items():
        if scores:
            method_averages[method] = np.mean(scores)
            print(f"  {method}: average distance = {method_averages[method]:.6f} ({len(scores)} valid repetitions)")
        else:
            method_averages[method] = float('inf')
            print(f"  {method}: no valid scores")
    
    # Select method with smallest distance
    best_method = min(method_averages.items(), key=lambda x: x[1])[0]
    best_score = method_averages[best_method]
    
    print(f"\n{'='*72}")
    print(f"Selected best sampling method: {best_method} (distance: {best_score:.6f})")
    print(f"{'='*72}\n")
    
    return best_method


def configure_mlflow(cfg: Dict[str, Any]) -> None:
    if not cfg.get("mlflow.enabled", False):
        return

    tracking_uri = cfg.get("mlflow.tracking_uri")
    if tracking_uri and tracking_uri.startswith("sqlite:///"):
        db_path = tracking_uri.replace("sqlite:///", "", 1)
        db_abs = Path.cwd() / db_path if not os.path.isabs(db_path) else Path(db_path)
        _ensure_dir(db_abs.parent)
        mlflow.set_tracking_uri(f"sqlite:///{db_abs}")

    mlflow.set_experiment(cfg.get("mlflow.experiment_name", "FEC Experiments"))
    
    if mlflow.active_run() is not None:
        mlflow.end_run()


def log_config_to_mlflow(cfg: Dict[str, Any], run_prefix: str, variant_tag: str) -> None:
    if not cfg.get("mlflow.enabled", False):
        return

    if mlflow.active_run() is None:
        mlflow.start_run(run_name=f"{run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    mlflow.set_tags({"variant": variant_tag})
    
    # Convert lists to comma-separated strings for MLflow
    params = {k: (",".join(map(str, v)) if isinstance(v, (list, tuple, set)) else v) for k, v in cfg.items()}
    mlflow.log_params(params)


# ---------------------------------------------------------------------------
# Cache and tracking utilities
# ---------------------------------------------------------------------------


class FECCache:
    """Fitness Evaluation Cache with per-generation statistics."""

    def __init__(self) -> None:
        self.cache: Dict[str, float] = {}
        self.hits = 0
        self.misses = 0
        self.fake_hits = 0
        self.gen_hits: List[int] = []
        self.gen_fake_hits: List[int] = []
        self.gen_misses: List[int] = []
        self.gen_full_evals: List[int] = []
        self.gen_sample_evals: List[int] = []
        self.gen_hits_count = 0
        self.gen_fake_hits_count = 0
        self.gen_misses_count = 0
        self.gen_full_evals_count = 0
        self.gen_sample_evals_count = 0
        self.full_evals = 0
        self.sample_evals = 0
        self.current_gen = 0
        self.detailed_hits: List[Dict[str, Any]] = []
        self.detailed_misses: List[Dict[str, Any]] = []
        self.detailed_fake_hits: List[Dict[str, Any]] = []
        self.cache_metadata: Dict[str, Dict[str, Any]] = {}

    def clear(self) -> None:
        self.__init__()

    def start_generation(self) -> None:
        self.gen_hits_count = 0
        self.gen_fake_hits_count = 0
        self.gen_misses_count = 0
        self.gen_full_evals_count = 0
        self.gen_sample_evals_count = 0

    def end_generation(self) -> None:
        self.gen_hits.append(self.gen_hits_count)
        self.gen_fake_hits.append(self.gen_fake_hits_count)
        self.gen_misses.append(self.gen_misses_count)
        self.gen_full_evals.append(self.gen_full_evals_count)
        self.gen_sample_evals.append(self.gen_sample_evals_count)
        self.current_gen += 1

    def record_sample_eval(self) -> None:
        self.sample_evals += 1
        self.gen_sample_evals_count += 1

    def record_full_eval(self) -> None:
        self.full_evals += 1
        self.gen_full_evals_count += 1

    def record_hit(self, phenotype: str, fingerprint: str, cached_fitness: float, actual_fitness: Optional[float]) -> None:
        meta = self.cache_metadata.get(fingerprint, {})
        self.detailed_hits.append(
            {
                "generation": self.current_gen,
                "phenotype": phenotype,
                "fingerprint": fingerprint,
                "cached_fitness": cached_fitness,
                "actual_fitness": actual_fitness,
                "fitness_difference": None if actual_fitness is None else abs(cached_fitness - actual_fitness),
                "cached_first_stored_at_gen": meta.get("first_stored_gen"),
                "cached_phenotype_when_stored": meta.get("phenotype"),
                "is_same_phenotype": phenotype == meta.get("phenotype"),
            }
        )

    def record_miss(self, phenotype: str, fingerprint: str, fitness: float) -> None:
        self.detailed_misses.append(
            {
                "generation": self.current_gen,
                "phenotype": phenotype,
                "fingerprint": fingerprint,
                "fitness": fitness,
                "cache_stored_at_gen": self.current_gen,
            }
        )
        if fingerprint not in self.cache_metadata:
            self.cache_metadata[fingerprint] = {
                "first_stored_gen": self.current_gen,
                "phenotype": phenotype,
                "fitness_stored": fitness,
            }

    def record_fake_hit(self, phenotype: str, fingerprint: str, cached_fitness: float, current_full_fitness: float) -> None:
        meta = self.cache_metadata.get(fingerprint, {})
        diff = abs(cached_fitness - current_full_fitness)
        percent = diff / max(abs(cached_fitness), 1e-10) * 100 if cached_fitness else 0.0
        self.detailed_fake_hits.append(
            {
                "generation": self.current_gen,
                "phenotype": phenotype,
                "fingerprint": fingerprint,
                "cached_fitness": cached_fitness,
                "current_full_fitness": current_full_fitness,
                "fitness_difference": diff,
                "fitness_difference_percent": percent,
                "cached_first_stored_at_gen": meta.get("first_stored_gen"),
                "cached_phenotype_when_stored": meta.get("phenotype"),
                "cached_fitness_when_stored": meta.get("fitness_stored"),
                "is_same_phenotype": phenotype == meta.get("phenotype"),
            }
        )

    def get_stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = self.hits / total if total else 0.0
        fake_hit_rate = self.fake_hits / self.hits if self.hits else 0.0
        full_eval_rate = self.full_evals / total if total else 0.0

        sample_eval_rate = self.sample_evals / total if total else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "fake_hits": self.fake_hits,
            "hit_rate": hit_rate,
            "fake_hit_rate": fake_hit_rate,
            "size": len(self.cache),
            "full_evals": self.full_evals,
            "sample_evals": self.sample_evals,
            "full_eval_rate": full_eval_rate,
            "sample_eval_rate": sample_eval_rate,
            "gen_hits": self.gen_hits,
            "gen_fake_hits": self.gen_fake_hits,
            "gen_misses": self.gen_misses,
            "gen_full_evals": self.gen_full_evals,
            "gen_sample_evals": self.gen_sample_evals,
            "detailed_hits": self.detailed_hits,
            "detailed_misses": self.detailed_misses,
            "detailed_fake_hits": self.detailed_fake_hits,
            "cache_metadata": self.cache_metadata,
        }


class PhenotypeTracker:
    """Collect phenotype-level information for debugging and analytics."""

    def __init__(self) -> None:
        self.tracking: List[Dict[str, Any]] = []
        self.current_run = 0
        self.current_gen = 0

    def clear(self) -> None:
        self.tracking.clear()
        self.current_run = 0
        self.current_gen = 0

    def start_run(self, run_idx: int) -> None:
        self.current_run = run_idx
        self.current_gen = 0

    def record_generation(self, population: Sequence[Any], hof: Optional[tools.HallOfFame], elite_size: int) -> None:
        elite_phenotypes = set()
        if hof is not None and len(hof) > 0:
            for i in range(min(elite_size, len(hof))):
                elite = hof.items[i]
                if hasattr(elite, "phenotype"):
                    elite_phenotypes.add(elite.phenotype)

        for idx, individual in enumerate(population):
            if getattr(individual, "invalid", False):
                continue
            phenotype = getattr(individual, "phenotype", None)
            if phenotype is None:
                continue
            fitness_val = float(individual.fitness.values[0]) if individual.fitness.valid else None

            genome: List[int] = []
            if hasattr(individual, "genome") and individual.genome is not None:
                try:
                    genome = list(individual.genome)
                except TypeError:
                    genome = [int(g) for g in individual.genome]

            used_codons_attr = getattr(individual, "used_codons", None)
            if isinstance(used_codons_attr, (int, np.integer)):
                used_codons = int(used_codons_attr)
            elif isinstance(used_codons_attr, float) and used_codons_attr.is_integer():
                used_codons = int(used_codons_attr)
            else:
                used_codons = used_codons_attr

            active_codons = None
            if genome:
                if isinstance(used_codons, int) and used_codons >= 0:
                    active_codons = genome[:used_codons]
                else:
                    active_codons = genome

            self.tracking.append(
                {
                    "run": self.current_run,
                    "generation": self.current_gen,
                    "individual_index": idx,
                    "phenotype": phenotype,
                    "genome_length": len(genome),
                    "genome": genome or None,
                    "used_codons": used_codons,
                    "active_codons": active_codons or None,
                    "fitness": fitness_val,
                    "is_elite": phenotype in elite_phenotypes,
                    "is_invalid": getattr(individual, "invalid", False),
                }
            )
        self.current_gen += 1

    def get_tracking_data(self) -> List[Dict[str, Any]]:
        return self.tracking


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------


def create_fitness_eval(
    centroid_X: Optional[np.ndarray],
    centroid_y: Optional[np.ndarray],
    centroid_indices: Optional[Sequence[int]],
    cache: Optional[FECCache],
    fec_enabled: bool,
    evaluate_fake_hits: bool,
    fake_hit_threshold: float,
    structural_similarity: bool,
    behavior_similarity: bool,
    operators: Dict[str, Any],
    X_test_ref: Optional[np.ndarray] = None,
    y_test_ref: Optional[np.ndarray] = None,
) -> Any:
    if fec_enabled and not (structural_similarity or behavior_similarity):
        raise ValueError("FEC cache requires structural and/or behaviour similarity when enabled.")

    def fitness_eval(individual: Any, points: Sequence[np.ndarray], dataset_type: str = "train") -> Tuple[float]:
        x = np.asarray(points[0], dtype=np.float64).copy()
        y_true = np.asarray(points[1], dtype=np.int64).copy()

        if getattr(individual, "invalid", False):
            return (float("nan"),)

        # Auto-detect if this is test set evaluation by comparing with stored test set reference
        # This handles the case where algorithm calls evaluate() without dataset_type parameter
        # CRITICAL: When evaluating on test set, we must NOT use the cache (which was built on sampled training data)
        # and we must evaluate on the actual test points, not the sampled training points
        is_test_set = False
        if X_test_ref is not None and y_test_ref is not None:
            try:
                # Check if points match test set (allowing for small numerical differences)
                # Both x and X_test_ref should be shape (features, samples) after transpose
                x_match = False
                y_match = False
                if x.shape == X_test_ref.shape:
                    x_match = np.allclose(x, X_test_ref, rtol=1e-10, atol=1e-10)
                if y_true.shape == y_test_ref.shape:
                    y_match = np.array_equal(y_true, y_test_ref)
                is_test_set = x_match and y_match
            except Exception:
                # If comparison fails, fall back to dataset_type parameter
                is_test_set = dataset_type.lower() == "test"
        else:
            # Fall back to dataset_type parameter if no test reference provided
            is_test_set = dataset_type.lower() == "test"
        
        is_training = not is_test_set
        use_cache = is_training and fec_enabled and cache is not None

        phenotype_component = getattr(individual, "phenotype", None) if structural_similarity else None
        behavior_key = None
        fingerprint = None
        cached_fitness = None
        use_cached_value = False
        register_cache_after_eval = False

        if use_cache and behavior_similarity:
            behavior_key = derive_behavior_key(individual, centroid_X, centroid_y, centroid_indices, operators, cache)
            if behavior_key is None:
                use_cache = False
            if not structural_similarity:
                phenotype_component = None

        if use_cache:
            try:
                fingerprint = build_fingerprint(phenotype_component, behavior_key)
            except ValueError:
                use_cache = False

        if use_cache and fingerprint:
            if fingerprint in cache.cache:
                cache.hits += 1
                cache.gen_hits_count += 1
                cached_fitness = cache.cache[fingerprint]
                if not evaluate_fake_hits:
                    cache.record_hit(individual.phenotype, fingerprint, cached_fitness, None)
                    return (cached_fitness,)
                use_cached_value = True
            else:
                cache.misses += 1
                cache.gen_misses_count += 1
                register_cache_after_eval = True

        eval_env = {"np": np, "x": x}
        eval_env.update(operators)
        try:
            prediction = eval(individual.phenotype, eval_env)
        except (FloatingPointError, ZeroDivisionError, OverflowError, MemoryError, IndexError):
            # Any numerical/array indexing error during phenotype evaluation
            # makes this individual invalid for this dataset.
            return (float("nan"),)

        if not np.isrealobj(prediction):
            return (float("nan"),)

        prediction = np.asarray(prediction, dtype=np.float64).flatten()
        if prediction.shape[0] != y_true.shape[0]:
            return (float("nan"),)

        try:
            y_pred = (prediction > 0).astype(int)
            fitness_full = 1 - np.mean(np.equal(y_true, y_pred))
        except (IndexError, TypeError):
            return (float("nan"),)

        if cache is not None and is_training:
            cache.record_full_eval()

        if use_cached_value and fingerprint:
            cache.record_hit(individual.phenotype, fingerprint, cached_fitness, fitness_full)
            if evaluate_fake_hits and abs(cached_fitness - fitness_full) > fake_hit_threshold:
                cache.fake_hits += 1
                cache.gen_fake_hits_count += 1
                cache.record_fake_hit(individual.phenotype, fingerprint, cached_fitness, fitness_full)
            return (cached_fitness,)

        if register_cache_after_eval and cache is not None and fingerprint is not None:
            cache.record_miss(individual.phenotype, fingerprint, fitness_full)
            cache.cache[fingerprint] = fitness_full

        return (fitness_full,)

    return fitness_eval


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
        # Invalid numerical/array access during evaluation → treat as invalid individual
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


# ---------------------------------------------------------------------------
# Core experiment orchestration
# ---------------------------------------------------------------------------


def prepare_toolbox(cfg: Dict[str, Any], operators: Dict[str, Any], grammar: grape.Grammar) -> base.Toolbox:
    toolbox = base.Toolbox()

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", grape.Individual, fitness=creator.FitnessMin)

    toolbox.register("populationCreator", grape.sensible_initialisation, creator.Individual)
    toolbox.register("evaluate", lambda ind, pts: baseline_fitness_eval(ind, pts, operators))
    toolbox.register("select", tools.selTournament, tournsize=int(cfg["ga_parameters.tournsize"]))
    toolbox.register("mate", grape.crossover_onepoint)
    toolbox.register("mutate", grape.mutation_int_flip_per_codon)

    toolbox.population_kwargs = dict(
        bnf_grammar=grammar,
        min_init_depth=int(cfg["ge_parameters.min_init_tree_depth"]),
        max_init_depth=int(cfg["ge_parameters.max_init_tree_depth"]),
        codon_size=int(cfg["ge_parameters.codon_size"]),
        codon_consumption=cfg["ge_parameters.codon_consumption"],
        genome_representation=cfg["ge_parameters.genome_representation"],
    )
    return toolbox


@dataclass
class ExperimentResult:
    config: Dict[str, Any]
    run_prefix: str
    logbooks: List[tools.Logbook]
    hofs: List[tools.HallOfFame]
    cache_stats: List[Dict[str, Any]]
    per_run_tables: List[pd.DataFrame]
    combined_results: Optional[pd.DataFrame]
    aggregated_means: Optional[pd.DataFrame]
    aggregated_std: Optional[pd.DataFrame]
    tracker: PhenotypeTracker
    results_dir: Path


def run_configured_experiment(
    cfg: Dict[str, Any],
    run_name_suffix: Optional[str],
    X: np.ndarray,
    y: np.ndarray,
    grammar: grape.Grammar,
    operators: Dict[str, Any],
    results_root: Optional[Path] = None,
) -> ExperimentResult:
    mlflow_enabled = cfg.get("mlflow.enabled", False)
    run_prefix_base = cfg.get("mlflow.run_name_prefix", "setup")
    run_prefix = f"{run_prefix_base}_{run_name_suffix}" if run_name_suffix else run_prefix_base
    variant_tag = run_name_suffix or "baseline"

    if mlflow_enabled:
        configure_mlflow(cfg)
        log_config_to_mlflow(cfg, run_prefix, variant_tag)

    run_batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_runs = int(cfg["evolution.n_runs"])
    base_seed = int(cfg["evolution.random_seed"])
    test_size = float(cfg["dataset.test_size"])
    fec_enabled = bool(cfg.get("fec.enabled", False))
    fec_sample_size = int(cfg.get("fec.sample_size", 0) or 0)
    evaluate_fake_hits = bool(cfg.get("fec.evaluate_fake_hits", False))
    fake_hit_threshold = float(cfg.get("fec.fake_hit_threshold", 1e-6))
    structural_similarity = bool(cfg.get("fec.structural_similarity", True))
    behavior_similarity = bool(cfg.get("fec.behavior_similarity", True))

    toolbox = prepare_toolbox(cfg, operators, grammar)

    results_dir = _ensure_dir(results_root if results_root is not None else Path.cwd() / "results")
    tracker = PhenotypeTracker()
    tracker.clear()

    all_logbooks: List[tools.Logbook] = []
    all_hofs: List[tools.HallOfFame] = []
    all_cache_stats: List[Dict[str, Any]] = []
    per_run_tables: List[pd.DataFrame] = []

    # Sequential execution
    for run_idx in range(n_runs):
            run_seed = base_seed + run_idx
            random.seed(run_seed)
            np.random.seed(run_seed)

            X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size, random_state=run_seed)
            X_train_full = X_train_full.T
            X_test = X_test.T

            centroid_X = None
            centroid_y = None
            centroid_indices = None
            run_cache = None

            # Get sampling method info for display
            if fec_enabled:
                sampling_method_display = cfg.get("fec.sampling_method", "kmeans")
                sample_fraction_display = cfg.get("fec.sample_fraction")
                sample_fraction_str = f" ({sample_fraction_display:.1%})" if sample_fraction_display is not None else ""
            else:
                sampling_method_display = "N/A (FEC disabled)"
                sample_fraction_str = ""
            
            print("\n" + "=" * 60)
            print(f"Run {run_idx + 1}/{n_runs} [{run_prefix}]")
            print(f"Sampling Method: {sampling_method_display}{sample_fraction_str}")
            print("=" * 60)

            if fec_enabled:
                # Calculate sample size based on training set size (after split)
                train_set_size = X_train_full.shape[1]  # Number of samples in training set
                sample_fraction = cfg.get("fec.sample_fraction")
                
                if sample_fraction is not None:
                    # Use fraction of training set
                    actual_sample_size = max(1, int(round(sample_fraction * train_set_size)))
                else:
                    # Fallback to configured sample_size, but cap to training set size
                    actual_sample_size = min(fec_sample_size, train_set_size)
                    if actual_sample_size < fec_sample_size:
                        print(f"Warning: Requested sample size {fec_sample_size} exceeds training set size {train_set_size}. Using {actual_sample_size} instead.")
                
                # Get sampling method from config (default to kmeans for backward compatibility)
                sampling_method = cfg.get("fec.sampling_method", "kmeans")
                # Incorporate sample size into seed for ALL methods to ensure true independence
                # across different sample sizes (even with same run index).
                # This prevents any potential correlation or overlapping samples between different
                # sample size configurations, ensuring each sample size gets independent results.
                # Combine run_seed and sample_size to create unique seed per sample size
                sampling_seed = abs(hash((run_seed, actual_sample_size))) % (2**31 - 1)

                # ------------------------------------------------------------------
                # Handle special "union" sampling method
                # ------------------------------------------------------------------
                if sampling_method == "union":
                    union_cfg = cfg.get("fec.sampling_methods.union", False)
                    if not union_cfg:
                        raise ValueError(
                            "fec.sampling_method is set to 'union' but "
                            "'fec.sampling_methods.union' is not configured. "
                            "Set it to a list of base methods, e.g. "
                            "['random', 'stratified']."
                        )
                    if not isinstance(union_cfg, (list, tuple)):
                        raise ValueError(
                            "fec.sampling_methods.union must be False or a list "
                            f"of method names, got: {type(union_cfg)}"
                        )

                    all_indices: List[int] = []
                    method_sample_sizes: Dict[str, int] = {}
                    for base_method in union_cfg:
                        base_sampling_func = get_sampling_function(base_method)
                        # Derive a per-method seed so methods are decorrelated
                        base_seed = abs(hash((sampling_seed, base_method))) % (2**31 - 1)
                        _, _, base_indices = base_sampling_func(
                            X_train_full, y_train_full, actual_sample_size, base_seed
                        )
                        base_indices_array = np.asarray(base_indices, dtype=int)
                        method_sample_sizes[base_method] = len(base_indices_array)
                        all_indices.extend(list(base_indices_array))

                    if not all_indices:
                        raise ValueError(
                            "Union sampling produced an empty index set. "
                            "Check 'fec.sampling_methods.union' configuration."
                        )

                    # Calculate union statistics
                    total_samples_before_union = len(all_indices)
                    union_indices = np.unique(np.asarray(all_indices, dtype=int))
                    unique_samples_after_union = len(union_indices)
                    final_union_fraction = unique_samples_after_union / len(y_train_full) if len(y_train_full) > 0 else 0.0
                    
                    # Store union statistics in config for later use
                    if "union_stats" not in cfg:
                        cfg["union_stats"] = {}
                    cfg["union_stats"][f"run_{run_idx}"] = {
                        "total_samples_before_union": total_samples_before_union,
                        "unique_samples_after_union": unique_samples_after_union,
                        "method_sample_sizes": method_sample_sizes,
                        "original_sample_size": actual_sample_size,
                        "original_sample_fraction": actual_sample_size / len(y_train_full) if len(y_train_full) > 0 else 0.0,
                        "final_union_fraction": final_union_fraction,
                        "union_methods": list(union_cfg),
                    }
                    
                    centroid_indices = union_indices
                    centroid_X = X_train_full[:, union_indices]
                    centroid_y = y_train_full[union_indices]
                    
                    print(f"Union sampling: {total_samples_before_union} total samples from {len(union_cfg)} methods, "
                          f"{unique_samples_after_union} unique samples ({final_union_fraction:.1%} of training set)")
                else:
                    # Standard single-method sampling
                    sampling_func = get_sampling_function(sampling_method)
                    centroid_X, centroid_y, centroid_indices = sampling_func(
                        X_train_full, y_train_full, actual_sample_size, sampling_seed
                    )
                
                run_cache = FECCache()
                run_cache.clear()
                if sampling_method != "union":
                    print(f"✓ Sampling complete: {actual_sample_size} samples selected using '{sampling_method}' method")
                else:
                    print(f"✓ Union sampling complete: {unique_samples_after_union} unique samples from {len(union_cfg)} methods")

            if fec_enabled and run_cache is not None:
                # Store references to test set for detection
                X_test_ref = X_test.copy()
                y_test_ref = y_test.copy()
                eval_fn = create_fitness_eval(
                    centroid_X,
                    centroid_y,
                    centroid_indices,
                    run_cache,
                    fec_enabled,
                    evaluate_fake_hits,
                    fake_hit_threshold,
                    structural_similarity,
                    behavior_similarity,
                    operators,
                    X_test_ref=X_test_ref,
                    y_test_ref=y_test_ref,
                )
            else:
                eval_fn = lambda ind, pts: baseline_fitness_eval(ind, pts, operators)

            if "evaluate" in toolbox.__dict__:
                del toolbox.__dict__["evaluate"]
            toolbox.register("evaluate", eval_fn)

            population = toolbox.populationCreator(pop_size=int(cfg["evolution.population"]), **toolbox.population_kwargs)
            hof = tools.HallOfFame(int(cfg["ga_parameters.halloffame_size"]))
            stats = tools.Statistics(key=lambda ind: ind.fitness.values)
            stats.register("avg", np.nanmean)
            stats.register("std", np.nanstd)
            stats.register("min", np.nanmin)
            stats.register("max", np.nanmax)

            tracker.start_run(run_idx)

            if fec_enabled and run_cache is not None:
                population, logbook = grape.algorithms.ge_eaSimpleWithElitism_fec(
                    population,
                    toolbox,
                    cxpb=float(cfg["ga_parameters.p_crossover"]),
                    mutpb=float(cfg["ga_parameters.p_mutation"]),
                    ngen=int(cfg["evolution.generations"]),
                    elite_size=int(cfg["ga_parameters.elite_size"]),
                    bnf_grammar=grammar,
                    codon_size=int(cfg["ge_parameters.codon_size"]),
                    max_tree_depth=int(cfg["ge_parameters.max_tree_depth"]),
                    max_genome_length=cfg.get("ge_parameters.max_genome_length"),
                    points_train=[X_train_full, y_train_full],
                    points_test=[X_test, y_test],
                    codon_consumption=cfg["ge_parameters.codon_consumption"],
                    report_items=cfg.get("report_items", []),
                    genome_representation=cfg["ge_parameters.genome_representation"],
                    stats=stats,
                    halloffame=hof,
                    verbose=False,
                    fec_cache=run_cache,
                    phenotype_tracker=tracker,
                )
            else:
                population, logbook = grape.algorithms.ge_eaSimpleWithElitism(
                    population,
                    toolbox,
                    cxpb=float(cfg["ga_parameters.p_crossover"]),
                    mutpb=float(cfg["ga_parameters.p_mutation"]),
                    ngen=int(cfg["evolution.generations"]),
                    elite_size=int(cfg["ga_parameters.elite_size"]),
                    bnf_grammar=grammar,
                    codon_size=int(cfg["ge_parameters.codon_size"]),
                    max_tree_depth=int(cfg["ge_parameters.max_tree_depth"]),
                    max_genome_length=cfg.get("ge_parameters.max_genome_length"),
                    points_train=[X_train_full, y_train_full],
                    points_test=[X_test, y_test],
                    codon_consumption=cfg["ge_parameters.codon_consumption"],
                    report_items=cfg.get("report_items", []),
                    genome_representation=cfg["ge_parameters.genome_representation"],
                    stats=stats,
                    halloffame=hof,
                    verbose=False,
                )

            all_logbooks.append(logbook)
            all_hofs.append(hof)

            if fec_enabled and run_cache is not None:
                cache_stats = run_cache.get_stats()
                cache_stats.setdefault("gen_full_evals", run_cache.gen_full_evals)
                cache_stats.setdefault("gen_sample_evals", run_cache.gen_sample_evals)
            else:
                generations = int(cfg["evolution.generations"])
                cache_stats = {
                    "hits": 0,
                    "misses": 0,
                    "fake_hits": 0,
                    "gen_hits": [0] * generations,
                    "gen_fake_hits": [0] * generations,
                    "gen_misses": [0] * generations,
                    "gen_full_evals": [0] * generations,
                    "gen_sample_evals": [0] * generations,
                    "detailed_hits": [],
                    "detailed_misses": [],
                    "detailed_fake_hits": [],
                    "cache_metadata": {},
                    "hit_rate": 0.0,
                    "fake_hit_rate": 0.0,
                    "full_evals": 0,
                    "sample_evals": 0,
                    "full_eval_rate": 0.0,
                    "sample_eval_rate": 0.0,
                    "size": 0,
                }
            all_cache_stats.append(cache_stats)

            if mlflow_enabled:
                def _log_metric_sequence(metric_name: str, values: Sequence[Any]) -> None:
                    for gen_idx, value in enumerate(values):
                        try:
                            numeric_value = float(value)
                        except (TypeError, ValueError):
                            continue
                        mlflow.log_metric(metric_name, numeric_value, step=int(gen_idx))

                gen_hits = cache_stats.get("gen_hits", [])
                gen_misses = cache_stats.get("gen_misses", [])
                gen_fake_hits = cache_stats.get("gen_fake_hits", [])
                gen_full = cache_stats.get("gen_full_evals", [])
                gen_sample = cache_stats.get("gen_sample_evals", [])

                hit_rates: List[float] = []
                cumul_hit_rates: List[float] = []
                if gen_hits and gen_misses:
                    cum_hits_total = 0.0
                    cum_eval_total = 0.0
                    for hits_val, miss_val in zip(gen_hits, gen_misses):
                        total = hits_val + miss_val
                        hit_rates.append((hits_val / total) if total else 0.0)
                        cum_hits_total += hits_val
                        cum_eval_total += total
                        cumul_hit_rates.append((cum_hits_total / cum_eval_total) if cum_eval_total else 0.0)
                    _log_metric_sequence("gen_hit_rate", hit_rates)
                    _log_metric_sequence("gen_hit_rate_cumulative", cumul_hit_rates)
                if gen_fake_hits and gen_hits:
                    fake_rates = []
                    for fake_val, hits_val in zip(gen_fake_hits, gen_hits):
                        fake_rates.append((fake_val / hits_val) if hits_val else 0.0)
                    _log_metric_sequence("gen_fake_hit_rate", fake_rates)
                if gen_hits:
                    _log_metric_sequence("gen_hits", gen_hits)
                if gen_misses:
                    _log_metric_sequence("gen_misses", gen_misses)
                if gen_full:
                    _log_metric_sequence("gen_full_evals", gen_full)
                if gen_sample:
                    _log_metric_sequence("gen_sample_evals", gen_sample)

            report_columns = cfg.get("report_items", [])
            if report_columns and len(logbook) > 0:
                try:
                    table_data = list(zip(*[logbook.select(col) for col in report_columns]))
                    results_df = pd.DataFrame(table_data, columns=report_columns)
                    results_df.insert(0, "run", run_idx + 1)
                    per_run_tables.append(results_df)
                except (KeyError, IndexError):
                    print("Missing fields in logbook; skipping per-run table generation.")

            best_fitness = hof.items[0].fitness.values[0] if len(hof) > 0 else float("nan")
            print(f"Run {run_idx + 1} completed. Best fitness: {best_fitness}")

            if mlflow_enabled:
                mlflow.log_metric("best_train_fitness", float(best_fitness), step=run_idx)
                mlflow.log_metric("run_hit_rate", cache_stats.get("hit_rate", 0.0), step=run_idx)
                mlflow.log_metric("run_fake_hit_rate", cache_stats.get("fake_hit_rate", 0.0), step=run_idx)
                mlflow.log_metric("run_hits", cache_stats.get("hits", 0), step=run_idx)
                mlflow.log_metric("run_misses", cache_stats.get("misses", 0), step=run_idx)
                mlflow.log_metric("run_fake_hits", cache_stats.get("fake_hits", 0), step=run_idx)

    combined_results: Optional[pd.DataFrame] = None
    aggregated_means: Optional[pd.DataFrame] = None
    aggregated_std: Optional[pd.DataFrame] = None
    if per_run_tables:
        combined_results = pd.concat(per_run_tables, ignore_index=True)
        # Only save aggregated means, not detailed per-run CSV
        
        # Compute generation-wise averages across runs for report columns.
        numeric_subset = combined_results.select_dtypes(include="number").copy()
        if "run" in numeric_subset.columns:
            numeric_subset = numeric_subset.drop(columns=["run"])
        if not numeric_subset.empty and "gen" in numeric_subset.columns:
            gen_series = numeric_subset["gen"]
            value_columns = numeric_subset.drop(columns=["gen"])
            grouped = value_columns.groupby(gen_series)
            aggregated_means = grouped.mean(numeric_only=True)
            aggregated_means.index.name = "gen"
            aggregated_std = grouped.std(ddof=0)
            aggregated_std.index.name = "gen"
            # Don't save individual CSV files - will be appended to single CSV instead
            if mlflow_enabled:
                for metric_name in ["avg", "std", "min", "max", "fitness_test", "selection_time", "generation_time"]:
                    if metric_name in aggregated_means.columns:
                        for gen, value in aggregated_means[metric_name].items():
                            mlflow.log_metric(metric_name, float(value), step=int(gen))

    if mlflow_enabled and mlflow.active_run() is not None:
        mlflow.end_run()

    return ExperimentResult(
        config=cfg,
        run_prefix=run_prefix,
        logbooks=all_logbooks,
        hofs=all_hofs,
        cache_stats=all_cache_stats,
        per_run_tables=per_run_tables,
        combined_results=combined_results,
        aggregated_means=aggregated_means,
        aggregated_std=aggregated_std,
        tracker=tracker,
        results_dir=results_dir,
    )


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------


def plot_comparison(results: List[Tuple[str, ExperimentResult]]) -> None:
    fig = go.Figure()
    for mode, output in results:
        if not output.logbooks:
            continue
        gens = np.array(output.logbooks[0].select("gen"))
        min_matrix = np.array([log.select("min") for log in output.logbooks], dtype=float)
        mean_min = np.nanmean(min_matrix, axis=0)
        fig.add_trace(
            go.Scatter(
                x=gens,
                y=mean_min,
                mode="lines+markers",
                name=mode,
                hovertemplate="Generation %{x}<br>Best fitness: %{y:.4f}",
            )
        )

    fig.update_layout(
        title="Mean Best Fitness per Generation across FEC Configurations",
        xaxis_title="Generation",
        yaxis_title="Best Fitness (lower is better)",
        template="plotly_white",
        hovermode="x unified",
    )

    target_dir = results[0][1].results_dir if results and results[0][1].results_dir else Path.cwd() / "results"
    output_path = target_dir / "fec_comparison.html"
    _ensure_dir(output_path.parent)
    pio.write_html(fig, file=str(output_path), auto_open=False)
    print(f"Saved comparison plot to {output_path}")


def _mean_cache_rate(
    cache_stats: List[Dict[str, Any]],
    numerator_key: str,
    denominator_keys: Sequence[str],
) -> Optional[np.ndarray]:
    if not cache_stats:
        return None
    rates: List[np.ndarray] = []
    for stats in cache_stats:
        numerator = np.asarray(stats.get(numerator_key, []), dtype=float)
        if numerator.size == 0:
            continue
        denominator = sum(np.asarray(stats.get(key, []), dtype=float) for key in denominator_keys)
        safe_denominator = np.where(denominator == 0, np.nan, denominator)
        rates.append(numerator / safe_denominator)
    if not rates:
        return None
    stacked = np.vstack([r[: min(r.shape[0] for r in rates)] for r in rates])
    return np.nanmean(stacked, axis=0)


def _mean_cache_ratio(cache_stats: List[Dict[str, Any]], numerator_key: str, denominator_key: str) -> Optional[np.ndarray]:
    if not cache_stats:
        return None
    ratios: List[np.ndarray] = []
    for stats in cache_stats:
        numerator = np.asarray(stats.get(numerator_key, []), dtype=float)
        denominator = np.asarray(stats.get(denominator_key, []), dtype=float)
        if numerator.size == 0 or denominator.size == 0:
            continue
        safe_denominator = np.where(denominator == 0, np.nan, denominator)
        ratios.append(numerator / safe_denominator)
    if not ratios:
        return None
    stacked = np.vstack([r[: min(r.shape[0] for r in ratios)] for r in ratios])
    return np.nanmean(stacked, axis=0)


def generate_html_report(
    results: List[Tuple[str, ExperimentResult]],
    report_name: Optional[str] = None,
    title: str = "Fitness Evaluation Cache Comparison",
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    if not results:
        return None

    if output_dir is not None:
        report_dir = _ensure_dir(output_dir)
    else:
        report_dir = results[0][1].results_dir if results[0][1].results_dir else Path.cwd() / "results"
    _ensure_dir(report_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if report_name:
        output_name = report_name
    else:
        sample_size = results[0][1].config.get("fec.sample_size") if results[0][1].config else None
        output_name = "fec_report"
        if sample_size:
            output_name += f"_size{int(sample_size)}"
    output_path = report_dir / f"{output_name}_{timestamp}.html"

    sections: List[str] = []

    def add_metric_section(metric_key: str, title: str, yaxis: str) -> None:
        """
        Add a metric section chart.
        Note: exp.aggregated_means already contains the average across runs for each generation.
        This accounts for stochasticity by averaging results from num_of_run independent runs.
        """
        fig = go.Figure()
        added = False
        for mode, exp in results:
            # aggregated_means: average across runs for each generation
            df = exp.aggregated_means
            # aggregated_std: standard deviation across runs for each generation (for error bars)
            df_std = getattr(exp, "aggregated_std", None)
            if df is None or metric_key not in df.columns:
                continue
            df_reset = df.reset_index()
            err_array = None
            if df_std is not None and metric_key in df_std.columns:
                err_array = df_std.reset_index()[metric_key].fillna(0.0).to_numpy()
            trace_kwargs: Dict[str, Any] = {
                "x": df_reset["gen"],
                "y": df_reset[metric_key],  # This is already averaged across runs per generation
                "mode": "lines+markers",
                "name": mode,
                "hovertemplate": "Generation %{x}<br>%{y:.4f}<extra></extra>",
            }
            if err_array is not None:
                trace_kwargs["error_y"] = {
                    "type": "data",
                    "array": err_array,
                    "visible": True,
                }
            fig.add_trace(go.Scatter(**trace_kwargs))
            added = True
        if added:
            layout_updates = {
                "title": title,
                "xaxis_title": "Generation",
                "yaxis_title": yaxis,
                "template": "plotly_white",
                "hovermode": "x unified",
                "legend_title": "Configuration",
            }
            layout_updates.update(get_large_font_layout())
            fig.update_layout(**layout_updates)
            sections.append(pio.to_html(fig, include_plotlyjs="cdn", full_html=False))

    # Note: aggregated_means already contains the average across runs for each generation
    # This accounts for stochasticity by averaging results from num_of_run independent runs
    add_metric_section("avg", "Training Fitness (Average) - Averaged Across Runs", "Fitness (lower is better)")
    add_metric_section("fitness_test", "Testing Fitness - Averaged Across Runs", "Fitness (lower is better)")
    add_metric_section("min", "Training Fitness (Best per generation) - Averaged Across Runs", "Fitness (lower is better)")
    add_metric_section("avg_length", "Average Individual Length", "Length")
    add_metric_section("avg_nodes", "Average Node Count", "Nodes")
    add_metric_section("generation_time", "Generation Time per Generation", "Seconds")
    add_metric_section("selection_time", "Selection Time per Generation", "Seconds")

    # Hit rate per generation (averaged across runs)
    fig_hits = go.Figure()
    fig_fake = go.Figure()
    fig_hits_cumul = go.Figure()
    hits_added = False
    fake_added = False
    cumul_added = False
    for mode, exp in results:
        hit_rates = _mean_cache_rate(exp.cache_stats, "gen_hits", ["gen_hits", "gen_misses"])
        fake_rates = _mean_cache_ratio(exp.cache_stats, "gen_fake_hits", "gen_hits")
        cumul_rates = None
        generations = range(len(hit_rates)) if hit_rates is not None else ()
        if hit_rates is not None and len(hit_rates) > 0:
            fig_hits.add_trace(
                go.Scatter(
                    x=list(range(len(hit_rates))),
                    y=hit_rates,
                    mode="lines+markers",
                    name=mode,
                    hovertemplate="Generation %{x}<br>Hit Rate %{y:.2%}",
                )
            )
            hits_added = True
        gen_hits = [np.asarray(stats.get("gen_hits", []), dtype=float) for stats in exp.cache_stats]
        gen_misses = [np.asarray(stats.get("gen_misses", []), dtype=float) for stats in exp.cache_stats]
        cumul_series: List[np.ndarray] = []
        for hits_array, misses_array in zip(gen_hits, gen_misses):
            if hits_array.size == 0 or misses_array.size == 0:
                continue
            min_length = min(hits_array.shape[0], misses_array.shape[0])
            hits_trimmed = hits_array[:min_length]
            misses_trimmed = misses_array[:min_length]
            cum_hits = np.cumsum(hits_trimmed)
            cum_total = np.cumsum(hits_trimmed + misses_trimmed)
            cumul_series.append(np.divide(
                cum_hits,
                np.where(cum_total == 0, np.nan, cum_total),
            ))
        if cumul_series:
            min_len = min(arr.shape[0] for arr in cumul_series)
            stacked = np.vstack([arr[:min_len] for arr in cumul_series])
            cumul_rates = np.nanmean(stacked, axis=0)
        if cumul_rates is not None and len(cumul_rates) > 0:
            fig_hits_cumul.add_trace(
                go.Scatter(
                    x=list(range(len(cumul_rates))),
                    y=cumul_rates,
                    mode="lines+markers",
                    name=mode,
                    hovertemplate="Generation %{x}<br>Cumulative Hit Rate %{y:.2%}",
                )
            )
            cumul_added = True
        if fake_rates is not None and len(fake_rates) > 0:
            fig_fake.add_trace(
                go.Scatter(
                    x=list(range(len(fake_rates))),
                    y=fake_rates,
                    mode="lines+markers",
                    name=mode,
                    hovertemplate="Generation %{x}<br>Fake Hit Rate %{y:.2%}",
                )
            )
            fake_added = True

    if hits_added:
        layout_updates = {
            "title": "Cache Hit Rate per Generation",
            "xaxis_title": "Generation",
            "yaxis_title": "Hit Rate",
            "yaxis_tickformat": ".0%",
            "template": "plotly_white",
            "hovermode": "x unified",
            "legend_title": "Configuration",
        }
        layout_updates.update(get_large_font_layout())
        fig_hits.update_layout(**layout_updates)
        sections.append(pio.to_html(fig_hits, include_plotlyjs="cdn", full_html=False))
    if cumul_added:
        layout_updates = {
            "title": "Cumulative Cache Hit Rate",
            "xaxis_title": "Generation",
            "yaxis_title": "Cumulative Hit Rate",
            "yaxis_tickformat": ".0%",
            "template": "plotly_white",
            "hovermode": "x unified",
            "legend_title": "Configuration",
        }
        layout_updates.update(get_large_font_layout())
        fig_hits_cumul.update_layout(**layout_updates)
        sections.append(pio.to_html(fig_hits_cumul, include_plotlyjs="cdn", full_html=False))
    if fake_added:
        layout_updates = {
            "title": "Fake Hit Rate per Generation",
            "xaxis_title": "Generation",
            "yaxis_title": "Fake Hit Rate",
            "yaxis_tickformat": ".0%",
            "template": "plotly_white",
            "hovermode": "x unified",
            "legend_title": "Configuration",
        }
        layout_updates.update(get_large_font_layout())
        fig_fake.update_layout(**layout_updates)
        sections.append(pio.to_html(fig_fake, include_plotlyjs="cdn", full_html=False))

    summary_rows = []
    for mode, exp in results:
        population_size = exp.config.get("evolution.population", 0)
        sample_size_val = exp.config.get("fec.sample_size")
        sample_fraction_val = exp.config.get("fec.sample_fraction")
        if exp.config.get("fec.enabled", False):
            sample_size_text = str(sample_size_val) if sample_size_val else "-"
            sample_fraction_text = f"{sample_fraction_val:.2%}" if sample_fraction_val is not None else "-"
        else:
            sample_size_text = "-"
            sample_fraction_text = "-"
        summary_rows.append(
            f"<tr><td>{mode}</td>"
            f"<td>{exp.config.get('evolution.n_runs', 0)}</td>"
            f"<td>{exp.config.get('evolution.generations', 0)}</td>"
            f"<td>{population_size}</td>"
            f"<td>{sample_size_text}</td>"
            f"<td>{sample_fraction_text}</td>"
            "</tr>"
        )
    summary_table = (
        "<h2>Experiment Summary</h2>"
        "<table border='1' cellpadding='6' cellspacing='0'>"
        "<tr><th>Configuration</th><th>Runs</th><th>Generations</th><th>Population</th>"
        "<th>Sample Size</th><th>Sample Fraction</th></tr>"
        f"{''.join(summary_rows)}</table>"
    )

    html = (
        "<html><head><meta charset='utf-8'><title>FEC Report</title></head><body>"
        f"<h1>{title}</h1>"
        f"{summary_table}"
        + "".join(sections)
        + "</body></html>"
    )

    output_path.write_text(html, encoding="utf-8")
    print(f"Saved interactive report to {output_path}")
    return output_path


def compare_fec_modes(
    cfg: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    grammar: grape.Grammar,
    operators: Dict[str, Any],
    sample_size: Optional[int] = None,
    sample_fraction: Optional[float] = None,
    sampling_method: Optional[str] = None,
    batch_prefix: Optional[str] = None,
    results_root: Optional[Path] = None,
    results_csv_path: Optional[Path] = None,
    completed_configs: Optional[Set[Tuple[str, str, float]]] = None,
) -> Tuple[pd.DataFrame, List[Tuple[str, ExperimentResult]]]:
    # Check if FEC is globally disabled
    fec_globally_disabled = not cfg.get("fec.enabled", True)

    # Special case: FEC globally disabled → run a single baseline experiment,
    # completely ignoring individual fec.modes.* flags.
    if fec_globally_disabled:
        if completed_configs is None:
            completed_configs = set()

        # Use a fixed mode name for baseline
        mode_name = "fec_disabled"

        # Skip if this exact configuration was already completed (if we have keys)
        if sample_fraction is not None and sampling_method is not None:
            config_key = (mode_name, sampling_method, sample_fraction)
            if config_key in completed_configs:
                print(f"  SKIPPING baseline mode '{mode_name}' (already completed)")
                empty_df = pd.DataFrame(
                    columns=["mode", "best_train_fitness", "hit_rate", "sample_size", "sample_fraction", "batch", "runs"]
                )
                return empty_df, []

        cfg_baseline = cfg.copy()
        cfg_baseline["fec.enabled"] = False
        cfg_baseline["fec.evaluate_fake_hits"] = False
        # Ensure sample size / fraction are consistent if provided
        if sample_size is not None:
            cfg_baseline["fec.sample_size"] = int(sample_size)
            cfg_baseline["fec.sample_fraction"] = float(sample_fraction) if sample_fraction is not None else None

        suffix = "_".join(
            filter(None, [batch_prefix, mode_name, f"s{int(sample_size)}" if sample_size else None])
        )
        result = run_configured_experiment(
            cfg_baseline,
            suffix,
            X,
            y,
            grammar,
            operators,
            results_root=results_root,
        )

        # Append summary row to main CSV if requested
        if results_csv_path is not None:
            _append_experiment_to_csv(result, mode_name, cfg_baseline, sample_size, sample_fraction, sampling_method, results_csv_path)

        # Build summary DataFrame
        logbooks = result.logbooks
        summaries: List[Dict[str, Any]] = []
        if logbooks:
            best_train_values = [log.select("min")[-1] for log in logbooks]
            best_train = float(np.nanmin(best_train_values)) if best_train_values else float("nan")
            cache_stats = result.cache_stats
            total_hits = sum(stat.get("hits", 0) for stat in cache_stats)
            total_misses = sum(stat.get("misses", 0) for stat in cache_stats)
            hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) else 0.0
            summaries.append(
                {
                    "mode": mode_name,
                    "best_train_fitness": best_train,
                    "hit_rate": hit_rate,
                    "sample_size": sample_size if sample_size is not None else result.config.get("fec.sample_size"),
                    "sample_fraction": sample_fraction if sample_fraction is not None else result.config.get("fec.sample_fraction"),
                    "batch": batch_prefix,
                    "runs": len(logbooks),
                }
            )

        summary_df = (
            pd.DataFrame(summaries).sort_values("mode").reset_index(drop=True)
            if summaries
            else pd.DataFrame(columns=["mode", "best_train_fitness", "hit_rate", "sample_size", "sample_fraction", "batch", "runs"])
        )
        return summary_df, [(mode_name, result)]

    # ----------------------------------------------------------------------
    # Standard case: FEC enabled → run whichever modes are enabled in cfg
    # ----------------------------------------------------------------------

    # Define all possible modes
    all_modes = [
        ("fec_disabled", {"enabled": False, "behavior_similarity": False, "structural_similarity": False}, "fec.modes.fec_disabled"),
        ("fec_enabled_same_behaviour", {"enabled": True, "behavior_similarity": True, "structural_similarity": False}, "fec.modes.fec_enabled_behaviour"),
        ("fec_enabled_same_structural", {"enabled": True, "behavior_similarity": False, "structural_similarity": True}, "fec.modes.fec_enabled_structural"),
        ("fec_enabled_same_behaviour_structural", {"enabled": True, "behavior_similarity": True, "structural_similarity": True}, "fec.modes.fec_enabled_behaviour_structural"),
    ]

    # Filter modes based on config flags (only relevant when FEC is enabled)
    modes = []
    for mode_name, overrides, config_key in all_modes:
        if cfg.get(config_key, True):  # Default to True if config key doesn't exist
            modes.append((mode_name, overrides))

    if not modes:
        raise ValueError("At least one FEC mode must be enabled. Check your config settings for fec.modes.*")

    if completed_configs is None:
        completed_configs = set()
    
    results: List[Tuple[str, ExperimentResult]] = []
    skipped_count = 0
    for mode_name, overrides in modes:
        # Check if this mode is already completed
        if sample_fraction is not None and sampling_method is not None:
            config_key = (mode_name, sampling_method, sample_fraction)
            if config_key in completed_configs:
                print(f"  SKIPPING mode '{mode_name}' (already completed)")
                skipped_count += 1
                continue
        
        cfg_variant = cfg.copy()
        for key, value in overrides.items():
            cfg_variant[f"fec.{key}"] = value
        if sample_size is not None:
            cfg_variant["fec.sample_size"] = int(sample_size)
            cfg_variant["fec.sample_fraction"] = float(sample_fraction) if sample_fraction is not None else None
        if not cfg_variant.get("fec.enabled", False):
            cfg_variant["fec.evaluate_fake_hits"] = False
        suffix = "_".join(filter(None, [batch_prefix, mode_name, f"s{int(sample_size)}" if sample_size else None]))
        result = run_configured_experiment(
            cfg_variant,
            suffix,
            X,
            y,
            grammar,
            operators,
            results_root=results_root,
        )
        results.append((mode_name, result))
        
        # Append to single CSV file
        if results_csv_path is not None:
            _append_experiment_to_csv(result, mode_name, cfg_variant, sample_size, sample_fraction, sampling_method, results_csv_path)
    
    # Check if all modes were skipped
    if skipped_count == len(modes) and len(results) == 0:
        print(f"  All {len(modes)} modes were already completed. Skipping this configuration.")
    
    summaries: List[Dict[str, Any]] = []
    for mode_name, result in results:
        logbooks = result.logbooks
        if not logbooks:
            continue
        best_train_values = [log.select("min")[-1] for log in logbooks]
        best_train = float(np.nanmin(best_train_values)) if best_train_values else float("nan")
        cache_stats = result.cache_stats
        total_hits = sum(stat.get("hits", 0) for stat in cache_stats)
        total_misses = sum(stat.get("misses", 0) for stat in cache_stats)
        hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) else 0.0
        summaries.append(
            {
                "mode": mode_name,
                "best_train_fitness": best_train,
                "hit_rate": hit_rate,
                "sample_size": sample_size if sample_size is not None else result.config.get("fec.sample_size"),
                "sample_fraction": sample_fraction if sample_fraction is not None else result.config.get("fec.sample_fraction"),
                "batch": batch_prefix,
                "runs": len(logbooks),
            }
        )

    # Don't generate HTML reports here - will be done in one consolidated file at the end
    if summaries:
        summary_df = pd.DataFrame(summaries).sort_values("mode").reset_index(drop=True)
    else:
        # Return empty DataFrame with expected columns if all modes were skipped
        summary_df = pd.DataFrame(columns=["mode", "best_train_fitness", "hit_rate", "sample_size", "sample_fraction", "batch", "runs"])
    return summary_df, results


def _append_experiment_to_csv(
    exp_result: ExperimentResult,
    mode_name: str,
    cfg: Dict[str, Any],
    sample_size: Optional[int],
    sample_fraction: Optional[float],
    sampling_method: Optional[str],
    csv_path: Path,
) -> None:
    """Append experiment results to the accumulating CSV file (one row per configuration)."""
    if exp_result.aggregated_means is None or len(exp_result.aggregated_means) == 0:
        return
    
    # Extract configuration
    fec_enabled = cfg.get("fec.enabled", False)
    behavior_similarity = cfg.get("fec.behavior_similarity", False)
    structural_similarity = cfg.get("fec.structural_similarity", False)
    
    # Get last generation metrics
    last_gen_idx = exp_result.aggregated_means.index.max()
    last_gen_data = exp_result.aggregated_means.loc[last_gen_idx]
    
    # Calculate hit rate and fake hit rate using BOTH methods:
    # 1. Cumulative: total_hits / total_evaluations for each run, then average across runs
    # 2. Avg per generation: average of generation hit rates for each run, then average across runs
    cache_stats = exp_result.cache_stats
    if cache_stats:
        # Method 1: Cumulative hit rate (total hits / total evaluations per run, then average)
        run_cumulative_hit_rates = []
        run_cumulative_fake_hit_rates = []
        
        for stat in cache_stats:
            total_hits = stat.get("hits", 0)
            total_misses = stat.get("misses", 0)
            total_fake_hits = stat.get("fake_hits", 0)
            
            # Cumulative hit rate for this run
            run_hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) else 0.0
            run_cumulative_hit_rates.append(run_hit_rate)
            
            # Cumulative fake hit rate for this run
            run_fake_hit_rate = total_fake_hits / total_hits if total_hits > 0 else 0.0
            run_cumulative_fake_hit_rates.append(run_fake_hit_rate)
        
        # Average cumulative hit rates across all runs
        hit_rate_cumulative = np.mean(run_cumulative_hit_rates) if run_cumulative_hit_rates else 0.0
        fake_hit_rate_cumulative = np.mean(run_cumulative_fake_hit_rates) if run_cumulative_fake_hit_rates else 0.0
        
        # Method 2: Average per generation (for each run: avg of generation hit rates, then avg across runs)
        gen_hits_list = [np.asarray(stat.get("gen_hits", []), dtype=float) for stat in cache_stats]
        gen_fake_hits_list = [np.asarray(stat.get("gen_fake_hits", []), dtype=float) for stat in cache_stats]
        gen_misses_list = [np.asarray(stat.get("gen_misses", []), dtype=float) for stat in cache_stats]
        
        run_avg_per_gen_hit_rates = []
        run_avg_per_gen_fake_hit_rates = []
        
        if gen_hits_list and all(len(arr) > 0 for arr in gen_hits_list):
            for run_idx in range(len(cache_stats)):
                gen_hits = gen_hits_list[run_idx] if run_idx < len(gen_hits_list) else np.array([])
                gen_misses = gen_misses_list[run_idx] if (run_idx < len(gen_misses_list) and len(gen_misses_list[run_idx]) > 0) else np.array([])
                gen_fake_hits = gen_fake_hits_list[run_idx] if (run_idx < len(gen_fake_hits_list) and len(gen_fake_hits_list[run_idx]) > 0) else np.array([])
                
                if len(gen_hits) > 0:
                    # Calculate hit rate for each generation
                    gen_hit_rates = []
                    for gen_idx in range(len(gen_hits)):
                        hits = gen_hits[gen_idx]
                        misses = gen_misses[gen_idx] if gen_idx < len(gen_misses) else 0.0
                        total = hits + misses
                        gen_hit_rates.append(hits / total if total > 0 else 0.0)
                    
                    if gen_hit_rates:
                        run_avg_per_gen_hit_rates.append(np.mean(gen_hit_rates))
                    
                    # Calculate fake hit rate for each generation
                    gen_fake_hit_rates = []
                    for gen_idx in range(len(gen_hits)):
                        hits = gen_hits[gen_idx]
                        fake_hits = gen_fake_hits[gen_idx] if gen_idx < len(gen_fake_hits) else 0.0
                        gen_fake_hit_rates.append(fake_hits / hits if hits > 0 else 0.0)
                    
                    if gen_fake_hit_rates:
                        run_avg_per_gen_fake_hit_rates.append(np.mean(gen_fake_hit_rates))
        
        hit_rate_avg_per_generation = np.mean(run_avg_per_gen_hit_rates) if run_avg_per_gen_hit_rates else 0.0
        fake_hit_rate_avg_per_generation = np.mean(run_avg_per_gen_fake_hit_rates) if run_avg_per_gen_fake_hit_rates else 0.0
    else:
        hit_rate_cumulative = 0.0
        fake_hit_rate_cumulative = 0.0
        hit_rate_avg_per_generation = 0.0
        fake_hit_rate_avg_per_generation = 0.0
    
    # Calculate union statistics if this is a union sampling experiment
    union_stats = None
    if sampling_method == "union":
        union_stats_dict = cfg.get("union_stats", {})
        if union_stats_dict:
            # Aggregate statistics across all runs
            total_before_list = []
            unique_after_list = []
            final_fraction_list = []
            method_sizes_dict: Dict[str, List[int]] = {}
            
            for run_key, stats in union_stats_dict.items():
                total_before_list.append(stats.get("total_samples_before_union", 0))
                unique_after_list.append(stats.get("unique_samples_after_union", 0))
                final_fraction_list.append(stats.get("final_union_fraction", 0.0))
                
                method_sizes = stats.get("method_sample_sizes", {})
                for method, size in method_sizes.items():
                    if method not in method_sizes_dict:
                        method_sizes_dict[method] = []
                    method_sizes_dict[method].append(size)
            
            # Calculate averages
            union_stats = {
                "union_total_samples_avg": np.mean(total_before_list) if total_before_list else 0.0,
                "union_unique_samples_avg": np.mean(unique_after_list) if unique_after_list else 0.0,
                "union_final_fraction_avg": np.mean(final_fraction_list) if final_fraction_list else 0.0,
                "union_methods": list(union_stats_dict.get("run_0", {}).get("union_methods", [])) if "run_0" in union_stats_dict else [],
            }
            
            # Add per-method average sample sizes
            for method, sizes in method_sizes_dict.items():
                union_stats[f"union_{method}_sample_size_avg"] = np.mean(sizes) if sizes else 0.0
    
    # Prepare row data
    row_data = {
        "batch": cfg.get("mlflow.run_name_prefix", "setup"),
        "mode": mode_name,
        "sampling_method": sampling_method if sampling_method is not None else cfg.get("fec.sampling_method", "kmeans"),
        "fec_enabled": fec_enabled,
        "behavior_similarity": behavior_similarity,
        "structural_similarity": structural_similarity,
        "sample_fraction": sample_fraction if sample_fraction is not None else cfg.get("fec.sample_fraction"),
        "sample_size": sample_size if sample_size is not None else cfg.get("fec.sample_size"),
        "n_runs": len(exp_result.logbooks),
        "n_generations": int(last_gen_idx) + 1,
        "hit_rate_cumulative": hit_rate_cumulative,
        "fake_hit_rate_cumulative": fake_hit_rate_cumulative,
        "hit_rate_avg_per_generation": hit_rate_avg_per_generation,
        "fake_hit_rate_avg_per_generation": fake_hit_rate_avg_per_generation,
        "train_fitness_last_gen": float(last_gen_data.get("min", float("nan"))),
        "test_fitness_last_gen": float(last_gen_data.get("fitness_test", float("nan"))),
        "avg_fitness_last_gen": float(last_gen_data.get("avg", float("nan"))),
        "complexity_avg_nodes": float(last_gen_data.get("avg_nodes", float("nan"))),
        "avg_length": float(last_gen_data.get("avg_length", float("nan"))),
        "avg_depth": float(last_gen_data.get("avg_depth", float("nan"))),
    }
    
    # Always add union statistics columns (even if None for non-union methods)
    # This ensures consistent CSV structure
    if union_stats:
        row_data.update({
            "union_total_samples_avg": union_stats["union_total_samples_avg"],
            "union_unique_samples_avg": union_stats["union_unique_samples_avg"],
            "union_final_fraction_avg": union_stats["union_final_fraction_avg"],
            "union_methods": ",".join(union_stats["union_methods"]) if union_stats["union_methods"] else "",
        })
        # Add per-method sample sizes
        for key, value in union_stats.items():
            if key.startswith("union_") and key.endswith("_sample_size_avg"):
                row_data[key] = value
    else:
        # Add empty union columns for non-union methods (use empty string for consistency)
        row_data.update({
            "union_total_samples_avg": "",
            "union_unique_samples_avg": "",
            "union_final_fraction_avg": "",
            "union_methods": "",
        })
    
    # Append to CSV (one row per configuration)
    # Ensure consistent column order by reading existing CSV first if it exists
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    if file_exists:
        try:
            # Read existing CSV to get column order
            existing_df = pd.read_csv(csv_path, nrows=0)  # Just read header
            existing_cols = list(existing_df.columns)
            # Add any new columns that don't exist
            for col in row_data.keys():
                if col not in existing_cols:
                    existing_cols.append(col)
            # Reorder row_data to match existing columns, fill missing with empty string
            ordered_row_data = {col: row_data.get(col, "") for col in existing_cols}
            df = pd.DataFrame([ordered_row_data])
            df.to_csv(csv_path, mode="a", header=False, index=False)
        except Exception as e:
            # If reading fails (e.g., inconsistent columns), just append with current structure
            print(f"Warning: Could not read existing CSV structure: {e}. Appending with current columns.")
            df = pd.DataFrame([row_data])
            df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        # New file - write with header
        df = pd.DataFrame([row_data])
        df.to_csv(csv_path, mode="a", header=True, index=False)
    print(f"Appended results to {csv_path}")


def generate_sample_size_comparison(
    all_sample_results: List[Tuple[float, int, str, List[Tuple[str, ExperimentResult]]]],
    output_dir: Path,
    batch_prefix: str,
) -> None:
    """
    Generate CSV and charts comparing metrics across different sample sizes and sampling methods.
    
    Args:
        all_sample_results: List of (sample_fraction, sample_size, sampling_method, run_results) tuples
        output_dir: Directory to save outputs
        batch_prefix: Prefix for output files
    """
    comparison_data = []
    
    for sample_fraction, sample_size, sampling_method, run_results in all_sample_results:
        for mode_name, exp_result in run_results:
            # Extract last generation metrics (averaged across runs)
            if exp_result.aggregated_means is None or len(exp_result.aggregated_means) == 0:
                continue
            
            last_gen_idx = exp_result.aggregated_means.index.max()
            last_gen_data = exp_result.aggregated_means.loc[last_gen_idx]
            
            # Get cache stats - average across runs
            cache_stats = exp_result.cache_stats
            if cache_stats:
                # Calculate average hit rate and fake hit rate across runs
                total_hits = sum(stat.get("hits", 0) for stat in cache_stats)
                total_misses = sum(stat.get("misses", 0) for stat in cache_stats)
                total_fake_hits = sum(stat.get("fake_hits", 0) for stat in cache_stats)
                
                hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) else 0.0
                fake_hit_rate = total_fake_hits / total_hits if total_hits > 0 else 0.0
                
                # Calculate average hit rate and fake hit rate across all generations and runs
                # For each run: calculate hit_rate for each generation, then average all generations
                # Then average across all runs
                gen_hits_list = [np.asarray(stat.get("gen_hits", []), dtype=float) for stat in cache_stats]
                gen_fake_hits_list = [np.asarray(stat.get("gen_fake_hits", []), dtype=float) for stat in cache_stats]
                gen_misses_list = [np.asarray(stat.get("gen_misses", []), dtype=float) for stat in cache_stats]
                
                if gen_hits_list and all(len(arr) > 0 for arr in gen_hits_list):
                    # For each run, calculate hit rate per generation, then average
                    run_hit_rates = []
                    run_fake_hit_rates = []
                    
                    for run_idx in range(len(cache_stats)):
                        gen_hits = gen_hits_list[run_idx] if run_idx < len(gen_hits_list) else np.array([])
                        gen_misses = gen_misses_list[run_idx] if (run_idx < len(gen_misses_list) and len(gen_misses_list[run_idx]) > 0) else np.array([])
                        gen_fake_hits = gen_fake_hits_list[run_idx] if (run_idx < len(gen_fake_hits_list) and len(gen_fake_hits_list[run_idx]) > 0) else np.array([])
                        
                        if len(gen_hits) > 0:
                            # Calculate hit rate for each generation in this run
                            gen_hit_rates = []
                            for gen_idx in range(len(gen_hits)):
                                hits = gen_hits[gen_idx]
                                misses = gen_misses[gen_idx] if gen_idx < len(gen_misses) else 0.0
                                total = hits + misses
                                gen_hit_rate = hits / total if total > 0 else 0.0
                                gen_hit_rates.append(gen_hit_rate)
                            
                            # Average hit rates across all generations for this run
                            if gen_hit_rates:
                                run_hit_rates.append(np.mean(gen_hit_rates))
                            
                            # Calculate fake hit rate for each generation in this run
                            gen_fake_hit_rates = []
                            for gen_idx in range(len(gen_hits)):
                                hits = gen_hits[gen_idx]
                                fake_hits = gen_fake_hits[gen_idx] if gen_idx < len(gen_fake_hits) else 0.0
                                gen_fake_hit_rate = fake_hits / hits if hits > 0 else 0.0
                                gen_fake_hit_rates.append(gen_fake_hit_rate)
                            
                            # Average fake hit rates across all generations for this run
                            if gen_fake_hit_rates:
                                run_fake_hit_rates.append(np.mean(gen_fake_hit_rates))
                    
                    # Average across all runs
                    last_gen_hit_rate = np.mean(run_hit_rates) if run_hit_rates else 0.0
                    last_gen_fake_hit_rate = np.mean(run_fake_hit_rates) if run_fake_hit_rates else 0.0
                else:
                    last_gen_hit_rate = hit_rate
                    last_gen_fake_hit_rate = fake_hit_rate
            else:
                hit_rate = 0.0
                fake_hit_rate = 0.0
                last_gen_hit_rate = 0.0
                last_gen_fake_hit_rate = 0.0
            
            # Extract metrics from last generation
            train_fitness = float(last_gen_data.get("min", float("nan")))
            test_fitness = float(last_gen_data.get("fitness_test", float("nan")))
            complexity = float(last_gen_data.get("avg_nodes", float("nan")))  # Using avg_nodes as complexity metric
            
            comparison_data.append({
                "sample_fraction": sample_fraction,
                "sample_size": sample_size,
                "sampling_method": sampling_method,
                "mode": mode_name,
                "hit_rate_cumulative": hit_rate,
                "fake_hit_rate_cumulative": fake_hit_rate,
                "hit_rate_avg_per_generation": last_gen_hit_rate,
                "fake_hit_rate_avg_per_generation": last_gen_fake_hit_rate,
                "train_fitness": train_fitness,
                "test_fitness": test_fitness,
                "complexity": complexity,
            })
    
    if not comparison_data:
        print("No data collected for sample size comparison")
        return
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Don't save separate comparison CSV - all data is already in the accumulating CSV
    # Just generate charts from the comparison data
    _generate_sample_size_charts(comparison_df, all_sample_results, output_dir, batch_prefix)
    
    # Generate method comparison charts (sample sizes across all methods)
    csv_files = list(output_dir.glob(f"{batch_prefix}_all_experiments_*.csv"))
    if csv_files:
        latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
        try:
            df = pd.read_csv(latest_csv, on_bad_lines='skip', engine='python')
            _generate_method_sample_size_charts(df, output_dir, batch_prefix)
        except Exception as e:
            print(f"Warning: Could not load CSV for method comparison charts: {e}")
    
    # Generate union analysis charts if union method is used
    # Check if union method appears in any results
    union_results = [r for r in all_sample_results if r[2] == "union"]
    if union_results:
        # Load CSV to get union statistics
        csv_files = list(output_dir.glob(f"{batch_prefix}_all_experiments_*.csv"))
        if csv_files:
            latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
            try:
                # Read CSV with error handling for inconsistent columns (some rows may have union columns, some may not)
                df = pd.read_csv(latest_csv, on_bad_lines='skip', engine='python')
                # Filter to union rows and check if union columns exist
                union_df = df[df["sampling_method"] == "union"]
                if not union_df.empty and "union_total_samples_avg" in union_df.columns:
                    _generate_union_analysis_charts(union_df, output_dir, batch_prefix)
            except Exception as e:
                print(f"Warning: Could not load CSV for union analysis: {e}")
                print("Union statistics may be incomplete. Continuing without union charts.")


def _generate_sample_size_charts(
    comparison_df: pd.DataFrame,
    all_sample_results: Optional[List[Tuple[float, int, str, List[Tuple[str, ExperimentResult]]]]],
    output_dir: Path,
    batch_prefix: str,
) -> None:
    """Generate charts comparing metrics across sample sizes and sampling methods."""
    
    # Filter to only include behaviour and structural modes (exclude fec_disabled and fec_structural_behaviour)
    allowed_modes = ["fec_enabled_same_behaviour", "fec_enabled_same_structural"]
    fec_enabled_df = comparison_df[comparison_df["mode"].isin(allowed_modes)].copy()
    
    if fec_enabled_df.empty:
        print("No allowed FEC modes found for comparison")
        return
    
    html_sections: List[str] = []
    
    # For sampling method comparison charts, only use fec_enabled_same_behaviour mode
    # (hit and fake hit rates are 0 for other modes, so they're not meaningful)
    behaviour_only_df = fec_enabled_df[fec_enabled_df["mode"] == "fec_enabled_same_behaviour"].copy()
    
    # Get unique sampling methods from behaviour-only data
    sampling_methods = sorted(behaviour_only_df["sampling_method"].unique())
    
    # ------------------------------------------------------------------
    # Line charts: Sampling method curves across sample sizes
    # ------------------------------------------------------------------
    # Chart: Compare sampling methods across sample sizes (using only fec_enabled_same_behaviour)
    html_sections.append("<h2>Sampling Method Comparison Across Sample Sizes</h2>")
    
    # Hit Rate vs Sample Size (comparing sampling methods)
    fig_hit_rate_sampling = go.Figure()
    for sampling_method in sampling_methods:
        method_df = behaviour_only_df[behaviour_only_df["sampling_method"] == sampling_method].copy()
        if method_df.empty:
            continue
        
        # Sort by sample_fraction (no averaging needed since we only have one mode)
        method_df_sorted = method_df.sort_values("sample_fraction")
        
        fig_hit_rate_sampling.add_trace(
            go.Scatter(
                x=method_df_sorted["sample_fraction"],
                y=method_df_sorted["hit_rate_avg_per_generation"],
                mode="lines+markers",
                name=sampling_method,
                hovertemplate="Sample Fraction: %{x:.2%}<br>Hit Rate: %{y:.2%}<br>Method: %{fullData.name}<extra></extra>",
            )
        )
    
    if len(fig_hit_rate_sampling.data) > 0:
        fig_hit_rate_sampling.update_layout(
            title="Hit Rate vs Sample Size (Comparing Sampling Methods)",
            xaxis_title="Sample Fraction",
            yaxis_title="Hit Rate (%)",
            yaxis_tickformat=".0%",
            xaxis_tickformat=".0%",
            template="plotly_white",
            hovermode="x unified",
            legend_title="Sampling Method",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.0,
            ),
        )
        html_sections.append(pio.to_html(fig_hit_rate_sampling, include_plotlyjs="cdn", full_html=False))
    
    # Fake Hit Rate vs Sample Size (comparing sampling methods)
    fig_fake_hit_rate_sampling = go.Figure()
    for sampling_method in sampling_methods:
        method_df = behaviour_only_df[behaviour_only_df["sampling_method"] == sampling_method].copy()
        if method_df.empty:
            continue
        
        # Sort by sample_fraction (no averaging needed since we only have one mode)
        method_df_sorted = method_df.sort_values("sample_fraction")
        
        fig_fake_hit_rate_sampling.add_trace(
            go.Scatter(
                x=method_df_sorted["sample_fraction"],
                y=method_df_sorted["fake_hit_rate_avg_per_generation"],
                mode="lines+markers",
                name=sampling_method,
                hovertemplate="Sample Fraction: %{x:.2%}<br>Fake Hit Rate: %{y:.2%}<br>Method: %{fullData.name}<extra></extra>",
            )
        )
    
    if len(fig_fake_hit_rate_sampling.data) > 0:
        fig_fake_hit_rate_sampling.update_layout(
            title="Fake Hit Rate vs Sample Size (Comparing Sampling Methods)",
            xaxis_title="Sample Fraction",
            yaxis_title="Fake Hit Rate (%)",
            yaxis_tickformat=".0%",
            xaxis_tickformat=".0%",
            template="plotly_white",
            hovermode="x unified",
            legend_title="Sampling Method",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.0,
            ),
        )
        html_sections.append(pio.to_html(fig_fake_hit_rate_sampling, include_plotlyjs="cdn", full_html=False))
    
    # Training Fitness vs Sample Size (comparing sampling methods)
    fig_train_fitness_sampling = go.Figure()
    for sampling_method in sampling_methods:
        method_df = behaviour_only_df[behaviour_only_df["sampling_method"] == sampling_method].copy()
        if method_df.empty:
            continue
        
        method_df_sorted = method_df.sort_values("sample_fraction")
        
        fig_train_fitness_sampling.add_trace(
            go.Scatter(
                x=method_df_sorted["sample_fraction"],
                y=method_df_sorted["train_fitness"],
                mode="lines+markers",
                name=sampling_method,
                hovertemplate="Sample Fraction: %{x:.2%}<br>Training Fitness: %{y:.4f}<br>Method: %{fullData.name}<extra></extra>",
            )
        )
    
    if len(fig_train_fitness_sampling.data) > 0:
        fig_train_fitness_sampling.update_layout(
            title="Training Fitness vs Sample Size (Comparing Sampling Methods)",
            xaxis_title="Sample Fraction",
            yaxis_title="Training Fitness (lower is better)",
            xaxis_tickformat=".0%",
            template="plotly_white",
            hovermode="x unified",
            legend_title="Sampling Method",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.0,
            ),
        )
        html_sections.append(pio.to_html(fig_train_fitness_sampling, include_plotlyjs="cdn", full_html=False))
    
    # Test Fitness vs Sample Size (comparing sampling methods)
    fig_test_fitness_sampling = go.Figure()
    for sampling_method in sampling_methods:
        method_df = behaviour_only_df[behaviour_only_df["sampling_method"] == sampling_method].copy()
        if method_df.empty:
            continue
        
        method_df_sorted = method_df.sort_values("sample_fraction")
        
        fig_test_fitness_sampling.add_trace(
            go.Scatter(
                x=method_df_sorted["sample_fraction"],
                y=method_df_sorted["test_fitness"],
                mode="lines+markers",
                name=sampling_method,
                hovertemplate="Sample Fraction: %{x:.2%}<br>Test Fitness: %{y:.4f}<br>Method: %{fullData.name}<extra></extra>",
            )
        )
    
    if len(fig_test_fitness_sampling.data) > 0:
        fig_test_fitness_sampling.update_layout(
            title="Test Fitness vs Sample Size (Comparing Sampling Methods)",
            xaxis_title="Sample Fraction",
            yaxis_title="Test Fitness (lower is better)",
            xaxis_tickformat=".0%",
            template="plotly_white",
            hovermode="x unified",
            legend_title="Sampling Method",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.0,
            ),
        )
        html_sections.append(pio.to_html(fig_test_fitness_sampling, include_plotlyjs="cdn", full_html=False))
    
    # Complexity vs Sample Size (comparing sampling methods)
    fig_complexity_sampling = go.Figure()
    for sampling_method in sampling_methods:
        method_df = behaviour_only_df[behaviour_only_df["sampling_method"] == sampling_method].copy()
        if method_df.empty:
            continue
        
        method_df_sorted = method_df.sort_values("sample_fraction")
        
        fig_complexity_sampling.add_trace(
            go.Scatter(
                x=method_df_sorted["sample_fraction"],
                y=method_df_sorted["complexity"],
                mode="lines+markers",
                name=sampling_method,
                hovertemplate="Sample Fraction: %{x:.2%}<br>Complexity (avg_nodes): %{y:.2f}<br>Method: %{fullData.name}<extra></extra>",
            )
        )
    
    if len(fig_complexity_sampling.data) > 0:
        fig_complexity_sampling.update_layout(
            title="Complexity (Avg Nodes) vs Sample Size (Comparing Sampling Methods)",
            xaxis_title="Sample Fraction",
            yaxis_title="Complexity (avg_nodes)",
            xaxis_tickformat=".0%",
            template="plotly_white",
            hovermode="x unified",
            legend_title="Sampling Method",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.0,
            ),
        )
        html_sections.append(pio.to_html(fig_complexity_sampling, include_plotlyjs="cdn", full_html=False))
    
    html_sections.append("<hr>")  # Separator before per-sampling-method charts
    
    # For each sampling method, create charts showing different FEC modes
    for sampling_method in sampling_methods:
        method_df = fec_enabled_df[fec_enabled_df["sampling_method"] == sampling_method].copy()
        
        if method_df.empty:
            continue
        
        html_sections.append(f"<h2>Sampling Method: {sampling_method}</h2>")
        
        # Chart 1: Hit Rate (Cumulative) vs Sample Size by FEC Mode (for this sampling method)
        fig_hit_rate_cumulative = go.Figure()
        for mode in method_df["mode"].unique():
            mode_data = method_df[method_df["mode"] == mode].sort_values("sample_fraction")
            fig_hit_rate_cumulative.add_trace(
                go.Scatter(
                    x=mode_data["sample_fraction"],
                    y=mode_data["hit_rate_cumulative"],
                    mode="lines+markers",
                    name=mode,
                    hovertemplate="Sample Fraction: %{x:.2%}<br>Hit Rate (Cumulative): %{y:.2%}<br>Mode: %{fullData.name}<extra></extra>",
                )
            )
        
        fig_hit_rate_cumulative.update_layout(
            title=f"Hit Rate (Cumulative) vs Sample Size - {sampling_method}",
            xaxis_title="Sample Fraction",
            yaxis_title="Hit Rate (Cumulative)",
            yaxis_tickformat=".0%",
            xaxis_tickformat=".0%",
            template="plotly_white",
            hovermode="x unified",
            legend_title="FEC Mode",
        )
        html_sections.append(pio.to_html(fig_hit_rate_cumulative, include_plotlyjs="cdn", full_html=False))
        
        # Chart 2: Hit Rate (Avg Per Generation) vs Sample Size by FEC Mode (for this sampling method)
        fig_hit_rate_avg = go.Figure()
        for mode in method_df["mode"].unique():
            mode_data = method_df[method_df["mode"] == mode].sort_values("sample_fraction")
            fig_hit_rate_avg.add_trace(
                go.Scatter(
                    x=mode_data["sample_fraction"],
                    y=mode_data["hit_rate_avg_per_generation"],
                    mode="lines+markers",
                    name=mode,
                    hovertemplate="Sample Fraction: %{x:.2%}<br>Hit Rate (Avg Per Generation): %{y:.2%}<br>Mode: %{fullData.name}<extra></extra>",
                )
            )
        
        fig_hit_rate_avg.update_layout(
            title=f"Hit Rate (Avg Per Generation) vs Sample Size - {sampling_method}",
            xaxis_title="Sample Fraction",
            yaxis_title="Hit Rate (Avg Per Generation)",
            yaxis_tickformat=".0%",
            xaxis_tickformat=".0%",
            template="plotly_white",
            hovermode="x unified",
            legend_title="FEC Mode",
        )
        html_sections.append(pio.to_html(fig_hit_rate_avg, include_plotlyjs="cdn", full_html=False))
        
        # Chart 3: Fake Hit Rate (Cumulative) vs Sample Size by FEC Mode (for this sampling method)
        fig_fake_hit_rate_cumulative = go.Figure()
        for mode in method_df["mode"].unique():
            mode_data = method_df[method_df["mode"] == mode].sort_values("sample_fraction")
            fig_fake_hit_rate_cumulative.add_trace(
                go.Scatter(
                    x=mode_data["sample_fraction"],
                    y=mode_data["fake_hit_rate_cumulative"],
                    mode="lines+markers",
                    name=mode,
                    hovertemplate="Sample Fraction: %{x:.2%}<br>Fake Hit Rate (Cumulative): %{y:.2%}<br>Mode: %{fullData.name}<extra></extra>",
                )
            )
        
        fig_fake_hit_rate_cumulative.update_layout(
            title=f"Fake Hit Rate (Cumulative) vs Sample Size - {sampling_method}",
            xaxis_title="Sample Fraction",
            yaxis_title="Fake Hit Rate (Cumulative)",
            yaxis_tickformat=".0%",
            xaxis_tickformat=".0%",
            template="plotly_white",
            hovermode="x unified",
            legend_title="FEC Mode",
        )
        html_sections.append(pio.to_html(fig_fake_hit_rate_cumulative, include_plotlyjs="cdn", full_html=False))
        
        # Chart 4: Fake Hit Rate (Avg Per Generation) vs Sample Size by FEC Mode (for this sampling method)
        fig_fake_hit_rate_avg = go.Figure()
        for mode in method_df["mode"].unique():
            mode_data = method_df[method_df["mode"] == mode].sort_values("sample_fraction")
            fig_fake_hit_rate_avg.add_trace(
                go.Scatter(
                    x=mode_data["sample_fraction"],
                    y=mode_data["fake_hit_rate_avg_per_generation"],
                    mode="lines+markers",
                    name=mode,
                    hovertemplate="Sample Fraction: %{x:.2%}<br>Fake Hit Rate (Avg Per Generation): %{y:.2%}<br>Mode: %{fullData.name}<extra></extra>",
                )
            )
        
        fig_fake_hit_rate_avg.update_layout(
            title=f"Fake Hit Rate (Avg Per Generation) vs Sample Size - {sampling_method}",
            xaxis_title="Sample Fraction",
            yaxis_title="Fake Hit Rate (Avg Per Generation)",
            yaxis_tickformat=".0%",
            xaxis_tickformat=".0%",
            template="plotly_white",
            hovermode="x unified",
            legend_title="FEC Mode",
        )
        html_sections.append(pio.to_html(fig_fake_hit_rate_avg, include_plotlyjs="cdn", full_html=False))
        
        # Charts 5-7: Training Fitness, Test Fitness, and Complexity over Generations by FEC Mode
        # These require in-memory ExperimentResult objects (all_sample_results). When we are
        # generating reports from CSV only, all_sample_results will be None and we skip them.
        if all_sample_results is not None:
            # Get all results for this sampling method (only allowed modes)
            method_results: List[Tuple[str, ExperimentResult, float]] = []  # (mode, exp_result, sample_fraction)
            for sample_fraction, sample_size, sm, run_results in all_sample_results:
                if sm == sampling_method:
                    for mode_name, exp_result in run_results:
                        if mode_name in allowed_modes and exp_result.aggregated_means is not None and len(exp_result.aggregated_means) > 0:
                            method_results.append((mode_name, exp_result, sample_fraction))
            
            if method_results:
                # Group by mode and use a fixed representative sample size for all modes (median of all sample fractions)
                from collections import defaultdict
                by_mode: Dict[str, List[Tuple[ExperimentResult, float]]] = defaultdict(list)
                for mode_name, exp_result, sample_fraction in method_results:
                    by_mode[mode_name].append((exp_result, sample_fraction))
                
                # Find the median sample fraction across all results for this sampling method
                all_sample_fractions = sorted([sf for _, _, sf in method_results])
                median_sample_fraction = all_sample_fractions[len(all_sample_fractions) // 2] if all_sample_fractions else None
                
                # For each mode, select the result closest to the median sample fraction
                selected_results: List[Tuple[str, ExperimentResult]] = []
                selected_sample_fraction = None
                if median_sample_fraction is not None:
                    for mode_name, results_list in by_mode.items():
                        # Find the result with sample fraction closest to median
                        closest_result = min(results_list, key=lambda x: abs(x[1] - median_sample_fraction))
                        selected_results.append((mode_name, closest_result[0]))
                        if selected_sample_fraction is None:
                            selected_sample_fraction = closest_result[1]
                
                # Chart 5: Training Fitness over Generations
                fig_train = go.Figure()
                for mode_name, exp_result in selected_results:
                    df = exp_result.aggregated_means
                    if df is not None and "min" in df.columns:
                        df_reset = df.reset_index()
                        fig_train.add_trace(
                            go.Scatter(
                                x=df_reset["gen"],
                                y=df_reset["min"],
                                mode="lines+markers",
                                name=mode_name,
                                hovertemplate="Generation %{x}<br>Training Fitness: %{y:.4f}<extra></extra>",
                            )
                        )
                
                if len(fig_train.data) > 0:
                    sample_size_note = f" (Sample Fraction: {selected_sample_fraction:.2%})" if selected_sample_fraction is not None else ""
                    fig_train.update_layout(
                        title=f"Training Fitness over Generations - {sampling_method}{sample_size_note}",
                        xaxis_title="Generation",
                        yaxis_title="Training Fitness (lower is better)",
                        template="plotly_white",
                        hovermode="x unified",
                        legend_title="FEC Mode",
                    )
                    html_sections.append(pio.to_html(fig_train, include_plotlyjs="cdn", full_html=False))
                
                # Chart 6: Test Fitness over Generations
                fig_test = go.Figure()
                for mode_name, exp_result in selected_results:
                    df = exp_result.aggregated_means
                    if df is not None and "fitness_test" in df.columns:
                        df_reset = df.reset_index()
                        fig_test.add_trace(
                            go.Scatter(
                                x=df_reset["gen"],
                                y=df_reset["fitness_test"],
                                mode="lines+markers",
                                name=mode_name,
                                hovertemplate="Generation %{x}<br>Test Fitness: %{y:.4f}<extra></extra>",
                            )
                        )
                
                if len(fig_test.data) > 0:
                    sample_size_note = f" (Sample Fraction: {selected_sample_fraction:.2%})" if selected_sample_fraction is not None else ""
                    fig_test.update_layout(
                        title=f"Test Fitness over Generations - {sampling_method}{sample_size_note}",
                        xaxis_title="Generation",
                        yaxis_title="Test Fitness (lower is better)",
                        template="plotly_white",
                        hovermode="x unified",
                        legend_title="FEC Mode",
                    )
                    html_sections.append(pio.to_html(fig_test, include_plotlyjs="cdn", full_html=False))
                
                # Chart 7: Complexity over Generations
                fig_complexity = go.Figure()
                for mode_name, exp_result in selected_results:
                    df = exp_result.aggregated_means
                    if df is not None and "avg_nodes" in df.columns:
                        df_reset = df.reset_index()
                        fig_complexity.add_trace(
                            go.Scatter(
                                x=df_reset["gen"],
                                y=df_reset["avg_nodes"],
                                mode="lines+markers",
                                name=mode_name,
                                hovertemplate="Generation %{x}<br>Complexity (avg_nodes): %{y:.2f}<extra></extra>",
                            )
                        )
                
                if len(fig_complexity.data) > 0:
                    sample_size_note = f" (Sample Fraction: {selected_sample_fraction:.2%})" if selected_sample_fraction is not None else ""
                    fig_complexity.update_layout(
                        title=f"Complexity (Avg Nodes) over Generations - {sampling_method}{sample_size_note}",
                        xaxis_title="Generation",
                        yaxis_title="Average Nodes",
                        template="plotly_white",
                        hovermode="x unified",
                        legend_title="FEC Mode",
                    )
                    html_sections.append(pio.to_html(fig_complexity, include_plotlyjs="cdn", full_html=False))
        
        html_sections.append("<hr>")  # Separator between sampling methods
    
    # Combine charts into one HTML file
    html_content = (
        "<html><head><meta charset='utf-8'><title>Sample Size & Sampling Method Comparison</title></head><body>"
        f"<h1>Sample Size & Sampling Method Comparison - {batch_prefix}</h1>"
        f"<p>Comparing the effect of different sample sizes and sampling methods on FEC cache performance metrics</p>"
        + _build_config_summary_html(CONFIG)
        + "".join(html_sections)
        + "</body></html>"
    )
    
    html_path = output_dir / f"{batch_prefix}_sample_size_comparison.html"
    html_path.write_text(html_content, encoding="utf-8")
    print(f"Saved sample size comparison charts to {html_path}")


def generate_consolidated_html_report(
    all_sample_results: List[Tuple[float, int, str, List[Tuple[str, ExperimentResult]]]],
    output_dir: Path,
    batch_prefix: str,
    fec_disabled_result: Optional[List[Tuple[str, ExperimentResult]]] = None,
    csv_path: Optional[Path] = None,
) -> None:
    """
    Generate one consolidated HTML file with all charts organized by sample size.
    Each sample size section contains charts for that sample size across all sampling methods and modes.
    If fec_disabled_result is provided, it will be included in all training and testing fitness charts.
    """
    from collections import defaultdict
    
    html_sections: List[str] = []
    
    # Extract fec_disabled result if available
    fec_disabled_exp_result = None
    if fec_disabled_result:
        for mode_name, exp_result in fec_disabled_result:
            if mode_name == "fec_disabled":
                fec_disabled_exp_result = exp_result
                break
    
    # Check if fec_disabled is enabled in config (to show note if it's missing)
    fec_disabled_enabled = CONFIG.get("fec.modes.fec_disabled", False)
    if fec_disabled_enabled and fec_disabled_exp_result is None:
        # If not in memory, try to load from CSV (though we can't plot over generations from CSV)
        if csv_path and csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                fec_disabled_rows = df[(df["mode"] == "fec_disabled") & (df["sampling_method"] == "full_dataset")]
                if not fec_disabled_rows.empty:
                    print("Note: fec_disabled is enabled but ExperimentResult not in memory. "
                          "Cannot include in generation-by-generation charts (CSV only has final metrics). "
                          "Re-run with fec_disabled enabled to include in all charts.")
            except Exception as e:
                print(f"Warning: Could not check CSV for fec_disabled: {e}")
        else:
            print("Note: fec_disabled is enabled in config but no result available. "
                  "Make sure fec_disabled was run to include it in charts.")
    
    # Group results by sample size
    by_sample_size: Dict[Tuple[float, int], List[Tuple[str, List[Tuple[str, ExperimentResult]]]]] = defaultdict(list)
    
    for sample_fraction, sample_size, sampling_method, run_results in all_sample_results:
        by_sample_size[(sample_fraction, sample_size)].append((sampling_method, run_results))
    
    # Sort by sample fraction
    sorted_sample_sizes = sorted(by_sample_size.keys(), key=lambda x: x[0])
    
    for sample_fraction, sample_size in sorted_sample_sizes:
        sample_results = by_sample_size[(sample_fraction, sample_size)]
        
        html_sections.append(f"<h2>Sample Size: {sample_size} ({sample_fraction:.2%} of training set)</h2>")
        
        # Collect all results for this sample size (across all sampling methods)
        all_results_for_size: List[Tuple[str, ExperimentResult]] = []
        for sampling_method, run_results in sample_results:
            for mode_name, exp_result in run_results:
                # Add sampling method to the mode name for clarity
                combined_name = f"{mode_name}_{sampling_method}"
                all_results_for_size.append((combined_name, exp_result))
        
        if not all_results_for_size:
            continue
        
        # Generate charts for this sample size using the existing generate_html_report logic
        sections_for_size: List[str] = []
        
        def add_metric_section(metric_key: str, title: str, yaxis: str, include_fec_disabled: bool = False) -> None:
            """
            Add a metric section chart.
            Note: exp.aggregated_means already contains the average across runs for each generation.
            This accounts for stochasticity by averaging results from num_of_run independent runs.
            """
            fig = go.Figure()
            added = False
            for mode_name, exp in all_results_for_size:
                # aggregated_means: average across runs for each generation
                df = exp.aggregated_means
                # aggregated_std: standard deviation across runs for each generation (for error bars)
                df_std = getattr(exp, "aggregated_std", None)
                if df is None or metric_key not in df.columns:
                    continue
                df_reset = df.reset_index()
                err_array = None
                if df_std is not None and metric_key in df_std.columns:
                    err_array = df_std.reset_index()[metric_key].fillna(0.0).to_numpy()
                trace_kwargs: Dict[str, Any] = {
                    "x": df_reset["gen"],
                    "y": df_reset[metric_key],  # This is already averaged across runs per generation
                    "mode": "lines+markers",
                    "name": mode_name,
                    "hovertemplate": "Generation %{x}<br>%{y:.4f}<extra></extra>",
                }
                if err_array is not None:
                    trace_kwargs["error_y"] = {
                        "type": "data",
                        "array": err_array,
                        "visible": True,
                    }
                fig.add_trace(go.Scatter(**trace_kwargs))
                added = True
            
            # Add fec_disabled if requested and available
            if include_fec_disabled and fec_disabled_exp_result is not None:
                df_fec_disabled = fec_disabled_exp_result.aggregated_means
                if df_fec_disabled is not None and metric_key in df_fec_disabled.columns:
                    df_fec_disabled_reset = df_fec_disabled.reset_index()
                    df_fec_disabled_std = getattr(fec_disabled_exp_result, "aggregated_std", None)
                    err_array_fec = None
                    if df_fec_disabled_std is not None and metric_key in df_fec_disabled_std.columns:
                        err_array_fec = df_fec_disabled_std.reset_index()[metric_key].fillna(0.0).to_numpy()
                    
                    trace_kwargs_fec = {
                        "x": df_fec_disabled_reset["gen"],
                        "y": df_fec_disabled_reset[metric_key],
                        "mode": "lines+markers",
                        "name": "fec_disabled (full_dataset)",
                        "line": dict(color="red", width=2, dash="dash"),
                        "marker": dict(size=10, symbol="diamond"),
                        "hovertemplate": "Generation %{x}<br>fec_disabled: %{y:.4f}",
                    }
                    if err_array_fec is not None:
                        trace_kwargs_fec["error_y"] = {
                            "type": "data",
                            "array": err_array_fec,
                            "visible": True,
                        }
                    fig.add_trace(go.Scatter(**trace_kwargs_fec))
                    added = True
            
            if added:
                layout_updates = {
                    "title": title,
                    "xaxis_title": "Generation",
                    "yaxis_title": yaxis,
                    "template": "plotly_white",
                    "hovermode": "x unified",
                    "legend_title": "Mode & Sampling Method",
                }
                layout_updates.update(get_large_font_layout())
                fig.update_layout(**layout_updates)
                sections_for_size.append(pio.to_html(fig, include_plotlyjs="cdn", full_html=False))
        
        # Add metric sections - include fec_disabled in training and testing fitness charts
        # Note: aggregated_means already contains the average across runs for each generation
        # "avg" = average fitness across population, averaged across all runs per generation
        # "fitness_test" = test fitness, averaged across all runs per generation
        add_metric_section("avg", "Training Fitness (Average) - Averaged Across Runs", "Fitness (lower is better)", include_fec_disabled=True)
        add_metric_section("fitness_test", "Testing Fitness - Averaged Across Runs", "Fitness (lower is better)", include_fec_disabled=True)
        add_metric_section("min", "Training Fitness (Best per generation) - Averaged Across Runs", "Fitness (lower is better)")
        add_metric_section("avg_length", "Average Individual Length", "Length")
        add_metric_section("avg_nodes", "Average Node Count", "Nodes")
        add_metric_section("avg_depth", "Average Tree Depth", "Depth")
        add_metric_section("generation_time", "Generation Time per Generation", "Seconds")
        add_metric_section("selection_time", "Selection Time per Generation", "Seconds")
        
        # Add cache hit rate charts for this sample size
        fig_hits = go.Figure()
        fig_fake = go.Figure()
        hits_added = False
        fake_added = False
        
        for mode_name, exp in all_results_for_size:
            hit_rates = _mean_cache_rate(exp.cache_stats, "gen_hits", ["gen_hits", "gen_misses"])
            fake_rates = _mean_cache_ratio(exp.cache_stats, "gen_fake_hits", "gen_hits")
            
            if hit_rates is not None and len(hit_rates) > 0:
                fig_hits.add_trace(
                    go.Scatter(
                        x=list(range(len(hit_rates))),
                        y=hit_rates,
                        mode="lines+markers",
                        name=mode_name,
                        hovertemplate="Generation %{x}<br>Hit Rate %{y:.2%}",
                    )
                )
                hits_added = True
            
            if fake_rates is not None and len(fake_rates) > 0:
                fig_fake.add_trace(
                    go.Scatter(
                        x=list(range(len(fake_rates))),
                        y=fake_rates,
                        mode="lines+markers",
                        name=mode_name,
                        hovertemplate="Generation %{x}<br>Fake Hit Rate %{y:.2%}",
                    )
                )
                fake_added = True
        
        if hits_added:
            fig_hits.update_layout(
                title="Cache Hit Rate per Generation",
                xaxis_title="Generation",
                yaxis_title="Hit Rate",
                yaxis_tickformat=".0%",
                template="plotly_white",
                hovermode="x unified",
                legend_title="Mode & Sampling Method",
            )
            sections_for_size.append(pio.to_html(fig_hits, include_plotlyjs="cdn", full_html=False))
        
        if fake_added:
            fig_fake.update_layout(
                title="Fake Hit Rate per Generation",
                xaxis_title="Generation",
                yaxis_title="Fake Hit Rate",
                yaxis_tickformat=".0%",
                template="plotly_white",
                hovermode="x unified",
                legend_title="Mode & Sampling Method",
            )
            sections_for_size.append(pio.to_html(fig_fake, include_plotlyjs="cdn", full_html=False))
        
        # Add all sections for this sample size
        html_sections.extend(sections_for_size)
        html_sections.append("<hr>")  # Separator between sample sizes
    
    # Combine all sections into one HTML file
    html_content = (
        "<html><head><meta charset='utf-8'><title>Consolidated FEC Report</title></head><body>"
        f"<h1>Consolidated FEC Report - {batch_prefix}</h1>"
        f"<p>All charts organized by sample size. Each section shows charts for that sample size across all sampling methods and FEC modes.</p>"
        + _build_config_summary_html(CONFIG)
        + "".join(html_sections)
        + "</body></html>"
    )
    
    html_path = output_dir / f"{batch_prefix}_consolidated_report.html"
    html_path.write_text(html_content, encoding="utf-8")
    print(f"Saved consolidated HTML report to {html_path}")


def _generate_union_analysis_charts(
    union_df: pd.DataFrame,
    output_dir: Path,
    batch_prefix: str,
) -> None:
    """
    Generate charts analyzing union sampling statistics.
    
    Args:
        union_df: DataFrame containing union sampling results with union statistics columns
        output_dir: Directory to save charts
        batch_prefix: Prefix for output files
    """
    if union_df.empty:
        return
    
    # Check if union statistics columns exist
    required_cols = ["union_total_samples_avg", "union_unique_samples_avg", "union_final_fraction_avg", "sample_fraction"]
    missing_cols = [col for col in required_cols if col not in union_df.columns]
    if missing_cols:
        print(f"Warning: Missing union statistics columns: {missing_cols}. Skipping union analysis charts.")
        return
    
    # Group by sample_fraction to get one row per fraction (take first mode if multiple)
    union_summary = union_df.groupby("sample_fraction").first().reset_index()
    
    # Chart 1: Union sample size vs original sample fraction
    fig1 = go.Figure()
    
    # Original sample size (what each method was asked to sample)
    fig1.add_trace(go.Scatter(
        x=union_summary["sample_fraction"],
        y=union_summary["sample_size"],
        mode="lines+markers",
        name="Original Sample Size (per method)",
        line=dict(color="blue", dash="dash"),
        marker=dict(size=8),
    ))
    
    # Total samples before union (sum of all methods)
    if "union_total_samples_avg" in union_summary.columns:
        fig1.add_trace(go.Scatter(
            x=union_summary["sample_fraction"],
            y=union_summary["union_total_samples_avg"],
            mode="lines+markers",
            name="Total Samples Before Union",
            line=dict(color="orange"),
            marker=dict(size=8),
        ))
    
    # Unique samples after union
    if "union_unique_samples_avg" in union_summary.columns:
        fig1.add_trace(go.Scatter(
            x=union_summary["sample_fraction"],
            y=union_summary["union_unique_samples_avg"],
            mode="lines+markers",
            name="Unique Samples After Union",
            line=dict(color="green"),
            marker=dict(size=8),
        ))
    
    fig1.update_layout(
        title="Union Sampling: Sample Size Comparison",
        xaxis_title="Original Sample Fraction",
        yaxis_title="Number of Samples",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    
    # Chart 2: Union fraction vs original fraction
    fig2 = go.Figure()
    
    # Original fraction (what was requested)
    fig2.add_trace(go.Scatter(
        x=union_summary["sample_fraction"],
        y=union_summary["sample_fraction"],
        mode="lines+markers",
        name="Original Sample Fraction",
        line=dict(color="blue", dash="dash"),
        marker=dict(size=8),
    ))
    
    # Final union fraction
    if "union_final_fraction_avg" in union_summary.columns:
        fig2.add_trace(go.Scatter(
            x=union_summary["sample_fraction"],
            y=union_summary["union_final_fraction_avg"],
            mode="lines+markers",
            name="Final Union Fraction",
            line=dict(color="red"),
            marker=dict(size=8),
        ))
    
    fig2.update_layout(
        title="Union Sampling: Fraction Comparison",
        xaxis_title="Original Sample Fraction",
        yaxis_title="Sample Fraction",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    
    # Chart 3: Per-method sample sizes (if available)
    method_cols = [col for col in union_summary.columns if col.startswith("union_") and col.endswith("_sample_size_avg")]
    if method_cols:
        fig3 = go.Figure()
        
        for col in method_cols:
            method_name = col.replace("union_", "").replace("_sample_size_avg", "")
            fig3.add_trace(go.Scatter(
                x=union_summary["sample_fraction"],
                y=union_summary[col],
                mode="lines+markers",
                name=f"{method_name} sample size",
                marker=dict(size=8),
            ))
        
        fig3.update_layout(
            title="Union Sampling: Per-Method Sample Sizes",
            xaxis_title="Original Sample Fraction",
            yaxis_title="Sample Size",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
    else:
        fig3 = None
    
    # Chart 4: Unique vs Total samples ratio
    if "union_total_samples_avg" in union_summary.columns and "union_unique_samples_avg" in union_summary.columns:
        union_summary["uniqueness_ratio"] = (
            union_summary["union_unique_samples_avg"] / union_summary["union_total_samples_avg"]
        )
        
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=union_summary["sample_fraction"],
            y=union_summary["uniqueness_ratio"],
            mode="lines+markers",
            name="Uniqueness Ratio (Unique/Total)",
            line=dict(color="purple"),
            marker=dict(size=8),
        ))
        
        fig4.update_layout(
            title="Union Sampling: Uniqueness Ratio",
            xaxis_title="Original Sample Fraction",
            yaxis_title="Uniqueness Ratio (Unique Samples / Total Samples)",
            template="plotly_white",
            hovermode="x unified",
            yaxis=dict(range=[0, 1.1]),
        )
    else:
        fig4 = None
    
    # Chart 5: Unique samples across fractions with detailed hover info
    if "union_unique_samples_avg" in union_summary.columns:
        # Calculate full dataset size estimate (from sample_size / sample_fraction)
        # Use the first row to estimate, assuming consistent dataset size
        first_row = union_summary.iloc[0]
        if first_row["sample_fraction"] > 0:
            estimated_full_dataset_size = first_row["sample_size"] / first_row["sample_fraction"]
        else:
            estimated_full_dataset_size = None
        
        # Sort by sample_fraction to ensure correct ordering
        union_summary_sorted = union_summary.sort_values("sample_fraction").reset_index(drop=True)
        
        # Create fraction labels: "Fraction 1 (10%)", "Fraction 2 (20%)", etc.
        fraction_labels = []
        for frac_num, (idx, row) in enumerate(union_summary_sorted.iterrows(), start=1):
            frac_pct = row["sample_fraction"] * 100
            fraction_labels.append(f"Fraction {frac_num}<br>({frac_pct:.0f}%)")
        
        fig5 = go.Figure()
        
        # Prepare hover text with all information
        hover_texts = []
        for idx, row in union_summary_sorted.iterrows():
            unique_samples = row["union_unique_samples_avg"]
            total_samples = row.get("union_total_samples_avg", 0)
            final_fraction = row.get("union_final_fraction_avg", 0) * 100
            
            hover_text = (
                f"Fraction: {row['sample_fraction']:.1%}<br>"
                f"Unique Samples: {unique_samples:.0f}<br>"
                f"All Samples (with duplicates): {total_samples:.0f}<br>"
                f"Percent of Full Dataset: {final_fraction:.2f}%"
            )
            hover_texts.append(hover_text)
        
        fig5.add_trace(go.Scatter(
            x=fraction_labels,
            y=union_summary_sorted["union_unique_samples_avg"],
            mode="lines+markers",
            name="Unique Samples",
            line=dict(color="green", width=3),
            marker=dict(size=10, color="green"),
            hovertemplate=(
                "%{hovertext}<extra></extra>"
            ),
            hovertext=hover_texts,
        ))
        
        fig5.update_layout(
            title="Unique Samples Across Different Sample Fractions",
            xaxis_title="Sample Fraction",
            yaxis_title="Number of Unique Samples",
            template="plotly_white",
            hovermode="closest",
            xaxis=dict(tickangle=-45),
        )
    else:
        fig5 = None
    
    # Combine all charts into HTML
    html_sections = ["<h2>Union Sampling Analysis</h2>"]
    html_sections.append("<h3>Sample Size Comparison</h3>")
    html_sections.append(pio.to_html(fig1, include_plotlyjs="cdn", full_html=False))
    
    html_sections.append("<h3>Fraction Comparison</h3>")
    html_sections.append(pio.to_html(fig2, include_plotlyjs="cdn", full_html=False))
    
    if fig3:
        html_sections.append("<h3>Per-Method Sample Sizes</h3>")
        html_sections.append(pio.to_html(fig3, include_plotlyjs="cdn", full_html=False))
    
    if fig4:
        html_sections.append("<h3>Uniqueness Ratio</h3>")
        html_sections.append(pio.to_html(fig4, include_plotlyjs="cdn", full_html=False))
    
    if fig5:
        html_sections.append("<h3>Unique Samples Across Fractions</h3>")
        html_sections.append("<p>This chart shows the number of unique samples (after removing duplicates) for each sample fraction. Hover over points to see detailed information including total samples with duplicates and percentage of full dataset.</p>")
        html_sections.append(pio.to_html(fig5, include_plotlyjs="cdn", full_html=False))
    
    html_content = (
        "<html><head><meta charset='utf-8'><title>Union Sampling Analysis</title></head><body>"
        f"<h1>Union Sampling Analysis - {batch_prefix}</h1>"
        + "".join(html_sections)
        + "</body></html>"
    )
    
    html_path = output_dir / f"{batch_prefix}_union_analysis.html"
    html_path.write_text(html_content, encoding="utf-8")
    print(f"Saved union analysis charts to {html_path}")
    
    # Also save individual chart files
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(exist_ok=True)
    
    pio.write_html(fig1, chart_dir / f"{batch_prefix}_union_sample_size_comparison.html")
    pio.write_html(fig2, chart_dir / f"{batch_prefix}_union_fraction_comparison.html")
    if fig3:
        pio.write_html(fig3, chart_dir / f"{batch_prefix}_union_per_method_sizes.html")
    if fig4:
        pio.write_html(fig4, chart_dir / f"{batch_prefix}_union_uniqueness_ratio.html")
    if fig5:
        pio.write_html(fig5, chart_dir / f"{batch_prefix}_union_unique_samples_across_fractions.html")


def _generate_method_sample_size_charts(
    df: pd.DataFrame,
    output_dir: Path,
    batch_prefix: str,
) -> None:
    """
    Generate charts comparing sample sizes across all sampling methods.
    
    Args:
        df: DataFrame containing all experiment results
        output_dir: Directory to save charts
        batch_prefix: Prefix for output files
    """
    if df.empty:
        return
    
    # Group by sample_fraction and sampling_method, take first row for each combination
    # (assuming same sample_size for same fraction and method across modes)
    summary = df.groupby(["sample_fraction", "sampling_method"]).first().reset_index()
    
    # Get unique methods and fractions
    methods = summary["sampling_method"].unique()
    fractions = sorted(summary["sample_fraction"].unique())
    
    if len(fractions) == 0:
        return
    
    # Chart 1: Number of samples vs sample fraction for all methods
    fig1 = go.Figure()
    
    # Color palette for different methods
    colors = {
        "kmeans": "blue",
        "kmedoids": "green",
        "farthest_point": "orange",
        "stratified": "purple",
        "random": "red",
        "union": "brown",
    }
    
    for method in methods:
        method_data = summary[summary["sampling_method"] == method].sort_values("sample_fraction")
        
        if method == "union":
            # For union, use unique samples after union
            if "union_unique_samples_avg" in method_data.columns:
                y_values = method_data["union_unique_samples_avg"].fillna(0)
            else:
                continue
        else:
            # For other methods, use sample_size
            y_values = method_data["sample_size"]
        
        fig1.add_trace(go.Scatter(
            x=method_data["sample_fraction"],
            y=y_values,
            mode="lines+markers",
            name=method,
            line=dict(color=colors.get(method, "gray"), width=2),
            marker=dict(size=8),
        ))
    
    fig1.update_layout(
        title="Sample Size Comparison Across All Methods",
        xaxis_title="Sample Fraction",
        yaxis_title="Number of Samples",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        xaxis=dict(tickformat=".0%"),
    )
    
    # Chart 2: Fraction of dataset vs requested fraction for all methods
    fig2 = go.Figure()
    
    # Add reference line (y = x) showing what it should be for non-union methods
    fig2.add_trace(go.Scatter(
        x=fractions,
        y=fractions,
        mode="lines",
        name="Expected (y=x)",
        line=dict(color="black", dash="dash", width=1),
        showlegend=True,
    ))
    
    for method in methods:
        method_data = summary[summary["sampling_method"] == method].sort_values("sample_fraction")
        
        if method == "union":
            # For union, use final union fraction
            if "union_final_fraction_avg" in method_data.columns:
                y_values = method_data["union_final_fraction_avg"].fillna(0)
            else:
                continue
        else:
            # For other methods, calculate actual fraction from sample_size
            # We need the full dataset size - estimate from sample_size / sample_fraction
            # Or use sample_fraction directly if it's already the fraction
            y_values = method_data["sample_fraction"]  # For non-union, it should match
        
        fig2.add_trace(go.Scatter(
            x=method_data["sample_fraction"],
            y=y_values,
            mode="lines+markers",
            name=method,
            line=dict(color=colors.get(method, "gray"), width=2),
            marker=dict(size=8),
        ))
    
    fig2.update_layout(
        title="Actual Fraction of Dataset vs Requested Fraction",
        xaxis_title="Requested Sample Fraction",
        yaxis_title="Actual Fraction of Dataset",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        xaxis=dict(tickformat=".0%"),
        yaxis=dict(tickformat=".0%"),
    )
    
    # Combine charts into HTML
    html_content = (
        "<html><head><meta charset='utf-8'><title>Method Sample Size Comparison</title></head><body>"
        f"<h1>Method Sample Size Comparison - {batch_prefix}</h1>"
        "<h2>Number of Samples vs Sample Fraction</h2>"
        "<p>This chart shows how many samples each method uses for different sample fractions. "
        "For non-union methods, this equals the requested fraction of the dataset. "
        "For union method, this shows the unique samples after combining all base methods.</p>"
        + pio.to_html(fig1, include_plotlyjs="cdn", full_html=False)
        + "<h2>Actual Fraction vs Requested Fraction</h2>"
        "<p>This chart shows the actual fraction of the dataset used by each method. "
        "For non-union methods, this should match the requested fraction (y=x line). "
        "For union method, this shows the final fraction after removing duplicates.</p>"
        + pio.to_html(fig2, include_plotlyjs="cdn", full_html=False)
        + "</body></html>"
    )
    
    html_path = output_dir / f"{batch_prefix}_method_sample_size_comparison.html"
    html_path.write_text(html_content, encoding="utf-8")
    print(f"Saved method sample size comparison charts to {html_path}")
    
    # Also save individual chart files
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(exist_ok=True)
    
    pio.write_html(fig1, chart_dir / f"{batch_prefix}_method_sample_sizes.html")
    pio.write_html(fig2, chart_dir / f"{batch_prefix}_method_fraction_comparison.html")


def generate_fec_disabled_vs_union_comparison(
    all_sample_results: List[Tuple[float, int, str, List[Tuple[str, ExperimentResult]]]],
    fec_disabled_result: Optional[List[Tuple[str, ExperimentResult]]],
    output_dir: Path,
    batch_prefix: str,
    csv_path: Optional[Path] = None,
) -> None:
    """
    Generate specific comparison charts:
    1. Training (avg) through generations: fec_disabled vs union for each fraction (one chart per fraction)
    2. Test through generations: fec_disabled vs union for each fraction (one chart per fraction)
    3. Hit rate and fake hit rate for different fraction sizes in union method
    
    Args:
        all_sample_results: List of (sample_fraction, sample_size, sampling_method, run_results) tuples
        fec_disabled_result: List of (mode_name, ExperimentResult) tuples for fec_disabled
        output_dir: Directory to save charts
        batch_prefix: Prefix for output files
        csv_path: Optional CSV path to load fec_disabled from if not in memory
    """
    html_sections: List[str] = []
    
    # Extract fec_disabled result
    fec_disabled_exp_result = None
    if fec_disabled_result:
        for mode_name, exp_result in fec_disabled_result:
            if mode_name == "fec_disabled":
                fec_disabled_exp_result = exp_result
                break
    
    # If not in memory, try to load from CSV (but we can't get generation-by-generation data from CSV)
    if fec_disabled_exp_result is None and csv_path and csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            fec_disabled_rows = df[(df["mode"] == "fec_disabled") & (df["sampling_method"] == "full_dataset")]
            if not fec_disabled_rows.empty:
                print("Warning: fec_disabled found in CSV but generation-by-generation data not available. "
                      "Cannot include in generation charts. Re-run with fec_disabled enabled.")
        except Exception as e:
            print(f"Warning: Could not check CSV for fec_disabled: {e}")
    
    if fec_disabled_exp_result is None:
        print("Warning: fec_disabled result not available. Skipping fec_disabled vs union comparison.")
        return
    
    # Extract union results only
    union_results = [r for r in all_sample_results if r[2] == "union"]
    if not union_results:
        print("No union results found for comparison")
        return
    
    # Sort union results by sample fraction
    union_results.sort(key=lambda x: x[0])
    
    # Get fec_disabled aggregated means
    fec_disabled_df = fec_disabled_exp_result.aggregated_means
    fec_disabled_df_std = getattr(fec_disabled_exp_result, "aggregated_std", None)
    
    if fec_disabled_df is None:
        print("fec_disabled has no aggregated means data")
        return
    
    fec_disabled_df_reset = fec_disabled_df.reset_index()
    
    # -------------------------------------------------------------------------
    # 1. Training (avg) through generations: fec_disabled vs union for each fraction
    # Note: aggregated_means contains the average across runs for each generation
    # -------------------------------------------------------------------------
    html_sections.append("<h2>Training Fitness (Average) Through Generations: FEC Disabled vs Union</h2>")
    html_sections.append("<p>One chart per sample fraction comparing training fitness (average) over generations. "
                        "Values shown are averaged across all runs for each generation (accounting for stochasticity).</p>")
    
    for sample_fraction, sample_size, sampling_method, run_results in union_results:
        # Find fec_enabled_same_behaviour mode in union results (for training avg)
        union_exp_result = None
        for mode_name, exp_result in run_results:
            if mode_name == "fec_enabled_same_behaviour":
                union_exp_result = exp_result
                break
        
        if union_exp_result is None or union_exp_result.aggregated_means is None:
            continue
        
        union_df = union_exp_result.aggregated_means
        union_df_std = getattr(union_exp_result, "aggregated_std", None)
        
        if "avg" not in union_df.columns:
            continue
        
        union_df_reset = union_df.reset_index()
        
        # Create chart for this fraction
        fig_train_avg = go.Figure()
        
        # Union data
        union_err_array = None
        if union_df_std is not None and "avg" in union_df_std.columns:
            union_err_array = union_df_std.reset_index()["avg"].fillna(0.0).to_numpy()
        
        trace_kwargs_union = {
            "x": union_df_reset["gen"],
            "y": union_df_reset["avg"],
            "mode": "lines+markers",
            "name": f"Union (fraction={sample_fraction:.2%})",
            "line": dict(color="blue", width=2),
            "marker": dict(size=8),
            "hovertemplate": "Generation %{x}<br>Union Training (avg): %{y:.4f}<extra></extra>",
        }
        if union_err_array is not None:
            trace_kwargs_union["error_y"] = dict(type="data", array=union_err_array, visible=True)
        fig_train_avg.add_trace(go.Scatter(**trace_kwargs_union))
        
        # FEC Disabled data
        fec_err_array = None
        if fec_disabled_df_std is not None and "avg" in fec_disabled_df_std.columns:
            fec_err_array = fec_disabled_df_std.reset_index()["avg"].fillna(0.0).to_numpy()
        
        trace_kwargs_fec = {
            "x": fec_disabled_df_reset["gen"],
            "y": fec_disabled_df_reset["avg"],
            "mode": "lines+markers",
            "name": "FEC Disabled (full_dataset)",
            "line": dict(color="red", width=2, dash="dash"),
            "marker": dict(size=10, symbol="diamond"),
            "hovertemplate": "Generation %{x}<br>FEC Disabled Training (avg): %{y:.4f}<extra></extra>",
        }
        if fec_err_array is not None:
            trace_kwargs_fec["error_y"] = dict(type="data", array=fec_err_array, visible=True)
        fig_train_avg.add_trace(go.Scatter(**trace_kwargs_fec))
        
        layout_updates = {
            "title": f"Training Fitness (Average) Through Generations - Fraction {sample_fraction:.2%}",
            "xaxis_title": "Generation",
            "yaxis_title": "Training Fitness (Average, lower is better)",
            "template": "plotly_white",
            "hovermode": "x unified",
            "legend": dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        }
        layout_updates.update(get_large_font_layout())
        fig_train_avg.update_layout(**layout_updates)
        
        html_sections.append(pio.to_html(fig_train_avg, include_plotlyjs="cdn", full_html=False))
    
    # -------------------------------------------------------------------------
    # 2. Test through generations: fec_disabled vs union for each fraction
    # Note: aggregated_means contains the average across runs for each generation
    # -------------------------------------------------------------------------
    html_sections.append("<hr>")
    html_sections.append("<h2>Testing Fitness Through Generations: FEC Disabled vs Union</h2>")
    html_sections.append("<p>One chart per sample fraction comparing testing fitness over generations. "
                        "Values shown are averaged across all runs for each generation (accounting for stochasticity).</p>")
    
    for sample_fraction, sample_size, sampling_method, run_results in union_results:
        # Find fec_enabled_same_behaviour mode in union results (for testing)
        union_exp_result = None
        for mode_name, exp_result in run_results:
            if mode_name == "fec_enabled_same_behaviour":
                union_exp_result = exp_result
                break
        
        if union_exp_result is None or union_exp_result.aggregated_means is None:
            continue
        
        union_df = union_exp_result.aggregated_means
        union_df_std = getattr(union_exp_result, "aggregated_std", None)
        
        if "fitness_test" not in union_df.columns:
            continue
        
        union_df_reset = union_df.reset_index()
        
        # Create chart for this fraction
        fig_test = go.Figure()
        
        # Union data
        union_err_array = None
        if union_df_std is not None and "fitness_test" in union_df_std.columns:
            union_err_array = union_df_std.reset_index()["fitness_test"].fillna(0.0).to_numpy()
        
        trace_kwargs_union_test = {
            "x": union_df_reset["gen"],
            "y": union_df_reset["fitness_test"],
            "mode": "lines+markers",
            "name": f"Union (fraction={sample_fraction:.2%})",
            "line": dict(color="blue", width=2),
            "marker": dict(size=8),
            "hovertemplate": "Generation %{x}<br>Union Testing: %{y:.4f}<extra></extra>",
        }
        if union_err_array is not None:
            trace_kwargs_union_test["error_y"] = dict(type="data", array=union_err_array, visible=True)
        fig_test.add_trace(go.Scatter(**trace_kwargs_union_test))
        
        # FEC Disabled data
        fec_err_array = None
        if fec_disabled_df_std is not None and "fitness_test" in fec_disabled_df_std.columns:
            fec_err_array = fec_disabled_df_std.reset_index()["fitness_test"].fillna(0.0).to_numpy()
        
        trace_kwargs_fec_test = {
            "x": fec_disabled_df_reset["gen"],
            "y": fec_disabled_df_reset["fitness_test"],
            "mode": "lines+markers",
            "name": "FEC Disabled (full_dataset)",
            "line": dict(color="red", width=2, dash="dash"),
            "marker": dict(size=10, symbol="diamond"),
            "hovertemplate": "Generation %{x}<br>FEC Disabled Testing: %{y:.4f}<extra></extra>",
        }
        if fec_err_array is not None:
            trace_kwargs_fec_test["error_y"] = dict(type="data", array=fec_err_array, visible=True)
        fig_test.add_trace(go.Scatter(**trace_kwargs_fec_test))
        
        layout_updates = {
            "title": f"Testing Fitness Through Generations - Fraction {sample_fraction:.2%}",
            "xaxis_title": "Generation",
            "yaxis_title": "Testing Fitness (lower is better)",
            "template": "plotly_white",
            "hovermode": "x unified",
            "legend": dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        }
        layout_updates.update(get_large_font_layout())
        fig_test.update_layout(**layout_updates)
        
        html_sections.append(pio.to_html(fig_test, include_plotlyjs="cdn", full_html=False))
    
    # -------------------------------------------------------------------------
    # 3. Hit rate and fake hit rate for different fraction sizes in union method
    # -------------------------------------------------------------------------
    html_sections.append("<hr>")
    html_sections.append("<h2>Hit Rate and Fake Hit Rate: Union Method Across Different Fractions</h2>")
    html_sections.append("<p>Hit rate and fake hit rate over generations for union method at different sample fractions.</p>")
    
    # Chart 1: Hit Rate
    fig_hit_rate = go.Figure()
    
    for sample_fraction, sample_size, sampling_method, run_results in union_results:
        # Find fec_enabled_same_behaviour mode in union results
        union_exp_result = None
        for mode_name, exp_result in run_results:
            if mode_name == "fec_enabled_same_behaviour":
                union_exp_result = exp_result
                break
        
        if union_exp_result is None:
            continue
        
        # Calculate hit rate per generation
        hit_rates = _mean_cache_rate(union_exp_result.cache_stats, "gen_hits", ["gen_hits", "gen_misses"])
        
        if hit_rates is not None and len(hit_rates) > 0:
            fig_hit_rate.add_trace(go.Scatter(
                x=list(range(len(hit_rates))),
                y=hit_rates,
                mode="lines+markers",
                name=f"Union (fraction={sample_fraction:.2%})",
                line=dict(width=2),
                marker=dict(size=8),
                hovertemplate="Generation %{x}<br>Hit Rate: %{y:.2%}<extra></extra>",
            ))
    
    if len(fig_hit_rate.data) > 0:
        layout_updates = {
            "title": "Cache Hit Rate Through Generations - Union Method",
            "xaxis_title": "Generation",
            "yaxis_title": "Hit Rate",
            "yaxis_tickformat": ".0%",
            "template": "plotly_white",
            "hovermode": "x unified",
            "legend": dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        }
        layout_updates.update(get_large_font_layout())
        fig_hit_rate.update_layout(**layout_updates)
        html_sections.append(pio.to_html(fig_hit_rate, include_plotlyjs="cdn", full_html=False))
    
    # Chart 2: Fake Hit Rate
    fig_fake_hit_rate = go.Figure()
    
    for sample_fraction, sample_size, sampling_method, run_results in union_results:
        # Find fec_enabled_same_behaviour mode in union results
        union_exp_result = None
        for mode_name, exp_result in run_results:
            if mode_name == "fec_enabled_same_behaviour":
                union_exp_result = exp_result
                break
        
        if union_exp_result is None:
            continue
        
        # Calculate fake hit rate per generation
        fake_rates = _mean_cache_ratio(union_exp_result.cache_stats, "gen_fake_hits", "gen_hits")
        
        if fake_rates is not None and len(fake_rates) > 0:
            fig_fake_hit_rate.add_trace(go.Scatter(
                x=list(range(len(fake_rates))),
                y=fake_rates,
                mode="lines+markers",
                name=f"Union (fraction={sample_fraction:.2%})",
                line=dict(width=2),
                marker=dict(size=8),
                hovertemplate="Generation %{x}<br>Fake Hit Rate: %{y:.2%}<extra></extra>",
            ))
    
    if len(fig_fake_hit_rate.data) > 0:
        layout_updates = {
            "title": "Cache Fake Hit Rate Through Generations - Union Method",
            "xaxis_title": "Generation",
            "yaxis_title": "Fake Hit Rate",
            "yaxis_tickformat": ".0%",
            "template": "plotly_white",
            "hovermode": "x unified",
            "legend": dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        }
        layout_updates.update(get_large_font_layout())
        fig_fake_hit_rate.update_layout(**layout_updates)
        html_sections.append(pio.to_html(fig_fake_hit_rate, include_plotlyjs="cdn", full_html=False))
    
    # Build final HTML
    html_content = (
        "<html><head><meta charset='utf-8'><title>FEC Disabled vs Union Comparison</title></head><body>"
        f"<h1>FEC Disabled vs Union Comparison - {batch_prefix}</h1>"
        "<p>This report provides detailed comparisons between FEC Disabled (full dataset) and Union sampling method.</p>"
        + "".join(html_sections)
        + "</body></html>"
    )
    
    html_path = output_dir / f"{batch_prefix}_fec_disabled_vs_union_comparison.html"
    html_path.write_text(html_content, encoding="utf-8")
    print(f"Saved FEC Disabled vs Union comparison to {html_path}")
    
    # Also save individual chart files
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(exist_ok=True)
    
    # Save individual charts (we'll save them with unique names based on fraction)
    # For now, the HTML contains all charts, individual files can be extracted if needed


def generate_methods_comparison_charts(
    all_sample_results: List[Tuple[float, int, str, List[Tuple[str, ExperimentResult]]]],
    fec_disabled_result: Optional[List[Tuple[str, ExperimentResult]]],
    output_dir: Path,
    batch_prefix: str,
    csv_path: Optional[Path] = None,
) -> None:
    """
    Generate comparison charts:
    1. Compare different methods (kmeans, kmedoids, stratified, union, etc.) in hit rate and fake hit rate through different fraction sizes
    2. Compare training and testing between each method with fec_disabled (fec_disabled as baseline)
    
    Args:
        all_sample_results: List of (sample_fraction, sample_size, sampling_method, run_results) tuples
        fec_disabled_result: List of (mode_name, ExperimentResult) tuples for fec_disabled
        output_dir: Directory to save charts
        batch_prefix: Prefix for output files
        csv_path: Optional CSV path to load fec_disabled from if not in memory
    """
    html_sections: List[str] = []
    
    # Extract fec_disabled result
    fec_disabled_exp_result = None
    if fec_disabled_result:
        for mode_name, exp_result in fec_disabled_result:
            if mode_name == "fec_disabled":
                fec_disabled_exp_result = exp_result
                break
    
    if fec_disabled_exp_result is None:
        print("Warning: fec_disabled result not available. Some comparisons will be incomplete.")
    
    # Get fec_disabled aggregated means if available
    fec_disabled_df = None
    fec_disabled_df_std = None
    if fec_disabled_exp_result:
        fec_disabled_df = fec_disabled_exp_result.aggregated_means
        fec_disabled_df_std = getattr(fec_disabled_exp_result, "aggregated_std", None)
    
    # Group results by sampling method
    from collections import defaultdict
    by_method: Dict[str, List[Tuple[float, int, List[Tuple[str, ExperimentResult]]]]] = defaultdict(list)
    
    for sample_fraction, sample_size, sampling_method, run_results in all_sample_results:
        by_method[sampling_method].append((sample_fraction, sample_size, run_results))
    
    # Sort methods and fractions
    methods = sorted(by_method.keys())
    
    # -------------------------------------------------------------------------
    # 1. Hit Rate and Fake Hit Rate: Compare different methods across fractions
    # -------------------------------------------------------------------------
    html_sections.append("<h2>Hit Rate and Fake Hit Rate: Methods Comparison Across Fractions</h2>")
    html_sections.append("<p>Comparing hit rate and fake hit rate for different sampling methods across different sample fractions.</p>")
    
    # Chart 1: Hit Rate vs Sample Fraction (all methods)
    fig_hit_rate_fractions = go.Figure()
    
    # Chart 2: Fake Hit Rate vs Sample Fraction (all methods)
    fig_fake_hit_rate_fractions = go.Figure()
    
    # Colors for different methods
    method_colors = {
        "kmeans": "blue",
        "kmedoids": "green",
        "farthest_point": "orange",
        "stratified": "purple",
        "random": "brown",
        "union": "red",
    }
    
    for method in methods:
        method_results = sorted(by_method[method], key=lambda x: x[0])  # Sort by fraction
        
        hit_rates_per_fraction = []
        fake_rates_per_fraction = []
        fractions = []
        
        for sample_fraction, sample_size, run_results in method_results:
            # Find fec_enabled_same_behaviour mode
            method_exp_result = None
            for mode_name, exp_result in run_results:
                if mode_name == "fec_enabled_same_behaviour":
                    method_exp_result = exp_result
                    break
            
            if method_exp_result is None:
                continue
            
            # Calculate average hit rate and fake hit rate across all generations
            cache_stats = method_exp_result.cache_stats
            if cache_stats:
                # Calculate cumulative hit rate
                total_hits = sum(stat.get("hits", 0) for stat in cache_stats)
                total_misses = sum(stat.get("misses", 0) for stat in cache_stats)
                total_fake_hits = sum(stat.get("fake_hits", 0) for stat in cache_stats)
                
                hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) else 0.0
                fake_hit_rate = total_fake_hits / total_hits if total_hits > 0 else 0.0
                
                hit_rates_per_fraction.append(hit_rate)
                fake_rates_per_fraction.append(fake_hit_rate)
                fractions.append(sample_fraction)
        
        if hit_rates_per_fraction:
            color = method_colors.get(method, "gray")
            fig_hit_rate_fractions.add_trace(go.Scatter(
                x=fractions,
                y=hit_rates_per_fraction,
                mode="lines+markers",
                name=method,
                line=dict(color=color, width=2),
                marker=dict(size=8),
                hovertemplate="Fraction: %{x:.2%}<br>Hit Rate: %{y:.2%}<extra></extra>",
            ))
            
            fig_fake_hit_rate_fractions.add_trace(go.Scatter(
                x=fractions,
                y=fake_rates_per_fraction,
                mode="lines+markers",
                name=method,
                line=dict(color=color, width=2),
                marker=dict(size=8),
                hovertemplate="Fraction: %{x:.2%}<br>Fake Hit Rate: %{y:.2%}<extra></extra>",
            ))
    
    if len(fig_hit_rate_fractions.data) > 0:
        layout_updates = {
            "title": "Hit Rate vs Sample Fraction - All Methods",
            "xaxis_title": "Sample Fraction",
            "yaxis_title": "Cumulative Hit Rate",
            "yaxis_tickformat": ".0%",
            "xaxis_tickformat": ".0%",
            "template": "plotly_white",
            "hovermode": "x unified",
            "legend": dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        }
        layout_updates.update(get_large_font_layout())
        fig_hit_rate_fractions.update_layout(**layout_updates)
        html_sections.append(pio.to_html(fig_hit_rate_fractions, include_plotlyjs="cdn", full_html=False))
    
    if len(fig_fake_hit_rate_fractions.data) > 0:
        layout_updates = {
            "title": "Fake Hit Rate vs Sample Fraction - All Methods",
            "xaxis_title": "Sample Fraction",
            "yaxis_title": "Cumulative Fake Hit Rate",
            "yaxis_tickformat": ".0%",
            "xaxis_tickformat": ".0%",
            "template": "plotly_white",
            "hovermode": "x unified",
            "legend": dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        }
        layout_updates.update(get_large_font_layout())
        fig_fake_hit_rate_fractions.update_layout(**layout_updates)
        html_sections.append(pio.to_html(fig_fake_hit_rate_fractions, include_plotlyjs="cdn", full_html=False))
    
    # -------------------------------------------------------------------------
    # 2. Training and Testing: Each method vs fec_disabled (fec_disabled as baseline)
    # Note: aggregated_means contains the average across runs for each generation
    # -------------------------------------------------------------------------
    html_sections.append("<hr>")
    html_sections.append("<h2>Training and Testing Fitness: Each Method vs FEC Disabled (Baseline)</h2>")
    html_sections.append("<p>Comparing training and testing fitness for each sampling method against FEC Disabled (shown as baseline with black solid line). "
                        "All values are averaged across all runs for each generation (accounting for stochasticity).</p>")
    
    if fec_disabled_df is not None:
        fec_disabled_df_reset = fec_disabled_df.reset_index()
        fec_err_array_train = None
        fec_err_array_test = None
        if fec_disabled_df_std is not None:
            if "avg" in fec_disabled_df_std.columns:
                fec_err_array_train = fec_disabled_df_std.reset_index()["avg"].fillna(0.0).to_numpy()
            if "fitness_test" in fec_disabled_df_std.columns:
                fec_err_array_test = fec_disabled_df_std.reset_index()["fitness_test"].fillna(0.0).to_numpy()
        
        # For each method, create one chart showing all fractions vs fec_disabled
        for method in methods:
            method_results = sorted(by_method[method], key=lambda x: x[0])
            
            # Training chart for this method
            fig_train_method = go.Figure()
            
            # Add fec_disabled as baseline first (so it appears at the bottom of legend)
            trace_kwargs_fec_train = {
                "x": fec_disabled_df_reset["gen"],
                "y": fec_disabled_df_reset["avg"],
                "mode": "lines+markers",
                "name": "FEC Disabled (baseline)",
                "line": dict(color="black", width=3, dash="solid"),
                "marker": dict(size=12, symbol="diamond"),
                "hovertemplate": "Generation %{x}<br>FEC Disabled (baseline): %{y:.4f}<extra></extra>",
            }
            if fec_err_array_train is not None:
                trace_kwargs_fec_train["error_y"] = dict(type="data", array=fec_err_array_train, visible=True)
            fig_train_method.add_trace(go.Scatter(**trace_kwargs_fec_train))
            
            # Add each fraction for this method
            for sample_fraction, sample_size, run_results in method_results:
                # Find fec_enabled_same_behaviour mode
                method_exp_result = None
                for mode_name, exp_result in run_results:
                    if mode_name == "fec_enabled_same_behaviour":
                        method_exp_result = exp_result
                        break
                
                if method_exp_result is None or method_exp_result.aggregated_means is None:
                    continue
                
                method_df = method_exp_result.aggregated_means
                method_df_std = getattr(method_exp_result, "aggregated_std", None)
                
                if "avg" not in method_df.columns:
                    continue
                
                method_df_reset = method_df.reset_index()
                method_err_array = None
                if method_df_std is not None and "avg" in method_df_std.columns:
                    method_err_array = method_df_std.reset_index()["avg"].fillna(0.0).to_numpy()
                
                color = method_colors.get(method, "gray")
                trace_kwargs_method = {
                    "x": method_df_reset["gen"],
                    "y": method_df_reset["avg"],
                    "mode": "lines+markers",
                    "name": f"{method} (fraction={sample_fraction:.2%})",
                    "line": dict(color=color, width=2),
                    "marker": dict(size=8),
                    "hovertemplate": f"Generation %{{x}}<br>{method} (f={sample_fraction:.2%}): %{{y:.4f}}<extra></extra>",
                }
                if method_err_array is not None:
                    trace_kwargs_method["error_y"] = dict(type="data", array=method_err_array, visible=True)
                fig_train_method.add_trace(go.Scatter(**trace_kwargs_method))
            
            if len(fig_train_method.data) > 1:  # More than just baseline
                layout_updates = {
                    "title": f"Training Fitness (Average): {method.capitalize()} vs FEC Disabled (Baseline)",
                    "xaxis_title": "Generation",
                    "yaxis_title": "Training Fitness (Average, lower is better)",
                    "template": "plotly_white",
                    "hovermode": "x unified",
                    "legend": dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                }
                layout_updates.update(get_large_font_layout())
                fig_train_method.update_layout(**layout_updates)
                html_sections.append(pio.to_html(fig_train_method, include_plotlyjs="cdn", full_html=False))
            
            # Testing chart for this method
            fig_test_method = go.Figure()
            
            # Add fec_disabled as baseline first
            trace_kwargs_fec_test = {
                "x": fec_disabled_df_reset["gen"],
                "y": fec_disabled_df_reset["fitness_test"],
                "mode": "lines+markers",
                "name": "FEC Disabled (baseline)",
                "line": dict(color="black", width=3, dash="solid"),
                "marker": dict(size=12, symbol="diamond"),
                "hovertemplate": "Generation %{x}<br>FEC Disabled (baseline): %{y:.4f}<extra></extra>",
            }
            if fec_err_array_test is not None:
                trace_kwargs_fec_test["error_y"] = dict(type="data", array=fec_err_array_test, visible=True)
            fig_test_method.add_trace(go.Scatter(**trace_kwargs_fec_test))
            
            # Add each fraction for this method
            for sample_fraction, sample_size, run_results in method_results:
                # Find fec_enabled_same_behaviour mode
                method_exp_result = None
                for mode_name, exp_result in run_results:
                    if mode_name == "fec_enabled_same_behaviour":
                        method_exp_result = exp_result
                        break
                
                if method_exp_result is None or method_exp_result.aggregated_means is None:
                    continue
                
                method_df = method_exp_result.aggregated_means
                method_df_std = getattr(method_exp_result, "aggregated_std", None)
                
                if "fitness_test" not in method_df.columns:
                    continue
                
                method_df_reset = method_df.reset_index()
                method_err_array = None
                if method_df_std is not None and "fitness_test" in method_df_std.columns:
                    method_err_array = method_df_std.reset_index()["fitness_test"].fillna(0.0).to_numpy()
                
                color = method_colors.get(method, "gray")
                trace_kwargs_method_test = {
                    "x": method_df_reset["gen"],
                    "y": method_df_reset["fitness_test"],
                    "mode": "lines+markers",
                    "name": f"{method} (fraction={sample_fraction:.2%})",
                    "line": dict(color=color, width=2),
                    "marker": dict(size=8),
                    "hovertemplate": f"Generation %{{x}}<br>{method} (f={sample_fraction:.2%}): %{{y:.4f}}<extra></extra>",
                }
                if method_err_array is not None:
                    trace_kwargs_method_test["error_y"] = dict(type="data", array=method_err_array, visible=True)
                fig_test_method.add_trace(go.Scatter(**trace_kwargs_method_test))
            
            if len(fig_test_method.data) > 1:  # More than just baseline
                layout_updates = {
                    "title": f"Testing Fitness: {method.capitalize()} vs FEC Disabled (Baseline)",
                    "xaxis_title": "Generation",
                    "yaxis_title": "Testing Fitness (lower is better)",
                    "template": "plotly_white",
                    "hovermode": "x unified",
                    "legend": dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                }
                layout_updates.update(get_large_font_layout())
                fig_test_method.update_layout(**layout_updates)
                html_sections.append(pio.to_html(fig_test_method, include_plotlyjs="cdn", full_html=False))
    else:
        html_sections.append("<p><em>FEC Disabled results not available. Cannot generate method vs baseline comparisons.</em></p>")
    
    # Build final HTML
    html_content = (
        "<html><head><meta charset='utf-8'><title>Methods Comparison Charts</title></head><body>"
        f"<h1>Methods Comparison Charts - {batch_prefix}</h1>"
        "<p>This report compares different sampling methods in terms of hit rates, fake hit rates, and fitness performance against FEC Disabled baseline.</p>"
        + "".join(html_sections)
        + "</body></html>"
    )
    
    html_path = output_dir / f"{batch_prefix}_methods_comparison.html"
    html_path.write_text(html_content, encoding="utf-8")
    print(f"Saved methods comparison charts to {html_path}")


def generate_union_vs_fec_disabled_comparison(
    all_sample_results: List[Tuple[float, int, str, List[Tuple[str, ExperimentResult]]]],
    fec_disabled_result: Optional[List[Tuple[str, ExperimentResult]]],
    expected_train_size: int,
    output_dir: Path,
    batch_prefix: str,
    csv_path: Optional[Path] = None,
) -> None:
    """
    Generate comparison charts between union sampling method and fec_disabled mode.
    
    For union method: uses effective fraction (unique samples after union / total training set)
    For fec_disabled: single point at 1.0 (full dataset)
    
    Charts include:
    - Training fitness comparison
    - Testing fitness comparison
    - Hit rate (union only, fec_disabled has no cache)
    - Fake hit rate (union only)
    
    Args:
        all_sample_results: List of (sample_fraction, sample_size, sampling_method, run_results) tuples
        fec_disabled_result: List of (mode_name, ExperimentResult) tuples for fec_disabled
        expected_train_size: Total training set size
        output_dir: Directory to save charts
        batch_prefix: Prefix for output files
    """
    # Extract union results
    union_results = [r for r in all_sample_results if r[2] == "union"]
    if not union_results:
        print("No union results found for comparison")
        return
    
    # Extract fec_disabled result (from memory or CSV)
    fec_disabled_exp_result = None
    fec_disabled_train_fitness = None
    fec_disabled_test_fitness = None
    
    if fec_disabled_result:
        for mode_name, exp_result in fec_disabled_result:
            if mode_name == "fec_disabled":
                fec_disabled_exp_result = exp_result
                break
    
    # If not in memory, try to load from CSV
    if fec_disabled_exp_result is None and csv_path and csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            fec_disabled_rows = df[(df["mode"] == "fec_disabled") & (df["sampling_method"] == "full_dataset")]
            if not fec_disabled_rows.empty:
                # Take the first row (should be only one)
                row = fec_disabled_rows.iloc[0]
                fec_disabled_train_fitness = float(row.get("train_fitness_last_gen", float("nan")))
                fec_disabled_test_fitness = float(row.get("test_fitness_last_gen", float("nan")))
                print("Loaded fec_disabled metrics from CSV")
        except Exception as e:
            print(f"Warning: Could not load fec_disabled from CSV: {e}")
    
    # If we have the ExperimentResult object, extract metrics from it
    if fec_disabled_exp_result is not None:
        if fec_disabled_exp_result.aggregated_means is None or len(fec_disabled_exp_result.aggregated_means) == 0:
            print("fec_disabled result has no aggregated means")
            return
        
        last_gen_idx = fec_disabled_exp_result.aggregated_means.index.max()
        last_gen_data = fec_disabled_exp_result.aggregated_means.loc[last_gen_idx]
        
        fec_disabled_train_fitness = float(last_gen_data.get("min", float("nan")))
        fec_disabled_test_fitness = float(last_gen_data.get("fitness_test", float("nan")))
    
    if fec_disabled_train_fitness is None or fec_disabled_test_fitness is None:
        print("No fec_disabled result found for comparison (neither in memory nor CSV)")
        return
    
    # Prepare data for charts
    union_data = []
    
    for sample_fraction, sample_size, sampling_method, run_results in union_results:
        # Find fec_enabled_same_behaviour mode in union results
        union_exp_result = None
        for mode_name, exp_result in run_results:
            if mode_name == "fec_enabled_same_behaviour":
                union_exp_result = exp_result
                break
        
        if union_exp_result is None or union_exp_result.aggregated_means is None:
            continue
        
        # Get effective fraction from union statistics
        union_stats = union_exp_result.config.get("union_stats", {})
        if union_stats:
            # Calculate average effective fraction across runs
            effective_fractions = []
            for run_key, stats in union_stats.items():
                eff_frac = stats.get("final_union_fraction", 0.0)
                if eff_frac > 0:
                    effective_fractions.append(eff_frac)
            
            if effective_fractions:
                effective_fraction = np.mean(effective_fractions)
            else:
                # Fallback: use sample_size / expected_train_size
                effective_fraction = sample_size / expected_train_size if expected_train_size > 0 else 0.0
        else:
            # Fallback: use sample_size / expected_train_size
            effective_fraction = sample_size / expected_train_size if expected_train_size > 0 else 0.0
        
        # Extract metrics from last generation
        last_gen_idx = union_exp_result.aggregated_means.index.max()
        last_gen_data = union_exp_result.aggregated_means.loc[last_gen_idx]
        
        # Calculate hit rates
        cache_stats = union_exp_result.cache_stats
        hit_rate_cumulative = 0.0
        fake_hit_rate_cumulative = 0.0
        
        if cache_stats:
            run_cumulative_hit_rates = []
            run_cumulative_fake_hit_rates = []
            
            for stat in cache_stats:
                total_hits = stat.get("hits", 0)
                total_misses = stat.get("misses", 0)
                total_fake_hits = stat.get("fake_hits", 0)
                
                run_hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) else 0.0
                run_cumulative_hit_rates.append(run_hit_rate)
                
                run_fake_hit_rate = total_fake_hits / total_hits if total_hits > 0 else 0.0
                run_cumulative_fake_hit_rates.append(run_fake_hit_rate)
            
            hit_rate_cumulative = np.mean(run_cumulative_hit_rates) if run_cumulative_hit_rates else 0.0
            fake_hit_rate_cumulative = np.mean(run_cumulative_fake_hit_rates) if run_cumulative_fake_hit_rates else 0.0
        
        union_data.append({
            "effective_fraction": effective_fraction,
            "train_fitness": float(last_gen_data.get("min", float("nan"))),
            "test_fitness": float(last_gen_data.get("fitness_test", float("nan"))),
            "hit_rate": hit_rate_cumulative,
            "fake_hit_rate": fake_hit_rate_cumulative,
        })
    
    if not union_data:
        print("No valid union data found for comparison")
        return
    
    # fec_disabled metrics are already extracted above
    
    # Sort union data by effective fraction
    union_data.sort(key=lambda x: x["effective_fraction"])
    
    # Create charts
    html_sections: List[str] = []
    
    # Chart 1: Training Fitness Comparison
    fig_train = go.Figure()
    
    # Union data
    union_eff_fracs = [d["effective_fraction"] for d in union_data]
    union_train_fits = [d["train_fitness"] for d in union_data]
    
    fig_train.add_trace(go.Scatter(
        x=union_eff_fracs,
        y=union_train_fits,
        mode="lines+markers",
        name="Union (FEC Enabled)",
        line=dict(color="blue", width=2),
        marker=dict(size=10),
    ))
    
    # FEC Disabled (single point at 1.0)
    fig_train.add_trace(go.Scatter(
        x=[1.0],
        y=[fec_disabled_train_fitness],
        mode="markers",
        name="FEC Disabled (Full Dataset)",
        marker=dict(size=15, color="red", symbol="diamond"),
    ))
    
    fig_train.update_layout(
        title="Training Fitness: Union vs FEC Disabled",
        xaxis_title="Effective Fraction of Dataset (Union) / Full Dataset (FEC Disabled)",
        yaxis_title="Training Fitness (Best)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        xaxis=dict(tickformat=".0%"),
    )
    
    html_sections.append("<h2>Training Fitness Comparison</h2>")
    html_sections.append(pio.to_html(fig_train, include_plotlyjs="cdn", full_html=False))
    
    # Chart 2: Testing Fitness Comparison
    fig_test = go.Figure()
    
    union_test_fits = [d["test_fitness"] for d in union_data]
    
    fig_test.add_trace(go.Scatter(
        x=union_eff_fracs,
        y=union_test_fits,
        mode="lines+markers",
        name="Union (FEC Enabled)",
        line=dict(color="blue", width=2),
        marker=dict(size=10),
    ))
    
    fig_test.add_trace(go.Scatter(
        x=[1.0],
        y=[fec_disabled_test_fitness],
        mode="markers",
        name="FEC Disabled (Full Dataset)",
        marker=dict(size=15, color="red", symbol="diamond"),
    ))
    
    fig_test.update_layout(
        title="Testing Fitness: Union vs FEC Disabled",
        xaxis_title="Effective Fraction of Dataset (Union) / Full Dataset (FEC Disabled)",
        yaxis_title="Testing Fitness (Best)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        xaxis=dict(tickformat=".0%"),
    )
    
    html_sections.append("<h2>Testing Fitness Comparison</h2>")
    html_sections.append(pio.to_html(fig_test, include_plotlyjs="cdn", full_html=False))
    
    # Chart 3: Hit Rate (Union only, FEC Disabled has no cache)
    fig_hit_rate = go.Figure()
    
    union_hit_rates = [d["hit_rate"] for d in union_data]
    
    fig_hit_rate.add_trace(go.Scatter(
        x=union_eff_fracs,
        y=union_hit_rates,
        mode="lines+markers",
        name="Union Hit Rate",
        line=dict(color="green", width=2),
        marker=dict(size=10),
    ))
    
    fig_hit_rate.update_layout(
        title="Cache Hit Rate: Union Method",
        xaxis_title="Effective Fraction of Dataset (after duplicate removal)",
        yaxis_title="Cumulative Hit Rate",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        xaxis=dict(tickformat=".0%"),
        yaxis=dict(tickformat=".0%"),
    )
    
    html_sections.append("<h2>Cache Hit Rate (Union Method Only)</h2>")
    html_sections.append("<p><em>Note: FEC Disabled mode does not use caching, so hit rate is not applicable.</em></p>")
    html_sections.append(pio.to_html(fig_hit_rate, include_plotlyjs="cdn", full_html=False))
    
    # Chart 4: Fake Hit Rate (Union only)
    fig_fake_hit_rate = go.Figure()
    
    union_fake_hit_rates = [d["fake_hit_rate"] for d in union_data]
    
    fig_fake_hit_rate.add_trace(go.Scatter(
        x=union_eff_fracs,
        y=union_fake_hit_rates,
        mode="lines+markers",
        name="Union Fake Hit Rate",
        line=dict(color="orange", width=2),
        marker=dict(size=10),
    ))
    
    fig_fake_hit_rate.update_layout(
        title="Cache Fake Hit Rate: Union Method",
        xaxis_title="Effective Fraction of Dataset (after duplicate removal)",
        yaxis_title="Cumulative Fake Hit Rate",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        xaxis=dict(tickformat=".0%"),
        yaxis=dict(tickformat=".0%"),
    )
    
    html_sections.append("<h2>Cache Fake Hit Rate (Union Method Only)</h2>")
    html_sections.append("<p><em>Note: FEC Disabled mode does not use caching, so fake hit rate is not applicable.</em></p>")
    html_sections.append(pio.to_html(fig_fake_hit_rate, include_plotlyjs="cdn", full_html=False))
    
    # Combined comparison chart
    fig_combined = go.Figure()
    
    # Training fitness
    fig_combined.add_trace(go.Scatter(
        x=union_eff_fracs,
        y=union_train_fits,
        mode="lines+markers",
        name="Union - Training Fitness",
        line=dict(color="blue", width=2),
        marker=dict(size=8),
        yaxis="y",
    ))
    
    fig_combined.add_trace(go.Scatter(
        x=[1.0],
        y=[fec_disabled_train_fitness],
        mode="markers",
        name="FEC Disabled - Training Fitness",
        marker=dict(size=12, color="red", symbol="diamond"),
        yaxis="y",
    ))
    
    # Testing fitness (secondary y-axis)
    fig_combined.add_trace(go.Scatter(
        x=union_eff_fracs,
        y=union_test_fits,
        mode="lines+markers",
        name="Union - Testing Fitness",
        line=dict(color="green", width=2, dash="dash"),
        marker=dict(size=8),
        yaxis="y2",
    ))
    
    fig_combined.add_trace(go.Scatter(
        x=[1.0],
        y=[fec_disabled_test_fitness],
        mode="markers",
        name="FEC Disabled - Testing Fitness",
        marker=dict(size=12, color="orange", symbol="diamond"),
        yaxis="y2",
    ))
    
    fig_combined.update_layout(
        title="Combined Comparison: Training & Testing Fitness",
        xaxis_title="Effective Fraction of Dataset (Union) / Full Dataset (FEC Disabled)",
        yaxis=dict(title="Training Fitness", side="left"),
        yaxis2=dict(title="Testing Fitness", side="right", overlaying="y"),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        xaxis=dict(tickformat=".0%"),
    )
    
    html_sections.append("<h2>Combined Fitness Comparison</h2>")
    html_sections.append(pio.to_html(fig_combined, include_plotlyjs="cdn", full_html=False))
    
    # Build final HTML
    html_content = (
        "<html><head><meta charset='utf-8'><title>Union vs FEC Disabled Comparison</title></head><body>"
        f"<h1>Union Sampling vs FEC Disabled Comparison - {batch_prefix}</h1>"
        "<p>This report compares the Union sampling method (with FEC enabled) against FEC Disabled mode (using full dataset).</p>"
        "<p><strong>Key Points:</strong></p>"
        "<ul>"
        "<li><strong>Union Method:</strong> Uses effective fraction (unique samples after union / total training set) on x-axis</li>"
        "<li><strong>FEC Disabled:</strong> Uses full dataset (1.0 = 100%) - shown as a single point</li>"
        "<li><strong>Hit Rate & Fake Hit Rate:</strong> Only applicable to Union method (FEC Disabled has no cache)</li>"
        "</ul>"
        + "".join(html_sections)
        + "</body></html>"
    )
    
    html_path = output_dir / f"{batch_prefix}_union_vs_fec_disabled_comparison.html"
    html_path.write_text(html_content, encoding="utf-8")
    print(f"Saved Union vs FEC Disabled comparison to {html_path}")
    
    # Also save individual chart files
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(exist_ok=True)
    
    pio.write_html(fig_train, chart_dir / f"{batch_prefix}_union_vs_fec_disabled_train.html")
    pio.write_html(fig_test, chart_dir / f"{batch_prefix}_union_vs_fec_disabled_test.html")
    pio.write_html(fig_hit_rate, chart_dir / f"{batch_prefix}_union_vs_fec_disabled_hit_rate.html")
    pio.write_html(fig_fake_hit_rate, chart_dir / f"{batch_prefix}_union_vs_fec_disabled_fake_hit_rate.html")
    pio.write_html(fig_combined, chart_dir / f"{batch_prefix}_union_vs_fec_disabled_combined.html")