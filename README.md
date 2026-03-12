# FEC vs Baseline: GE with Fitness Evaluation Cache

This project compares **baseline** Grammatical Evolution (GE) runs (full training set, no cache) with **FEC** (Fitness Evaluation Cache) runs that use a sampled subset of the training data for cache lookups. The goal is to measure **speedup** and **accuracy/MAE** when using FEC under different sampling methods, sample fractions, and fake-hit thresholds.

---

## 1. About the system

### Components

| Component | Role |
|-----------|------|
| **baseline_runs.py** | Runs one GE run **without** FEC on the full training set. Outputs per-generation stats and a one-row summary (final test MAE, accuracy, total time). |
| **FEC_runs.py** | Runs one GE run **with** FEC: uses a sampling method and fraction to build a cache key (e.g. behaviour on centroids). Outputs per-generation stats and summary including cache hit/miss/fake-hit rates and total time. |
| **FEC_report.py** | Aggregates per-run CSVs (mean/std across runs), then (when given an experiment root) builds an HTML report and a combined CSV comparing baseline vs FEC. |
| **run_baseline_batch.sh** | Launches many baseline runs in parallel (same config, different `--run-index`). |
| **run_FEC_batch.sh** | Launches many FEC runs over a grid of sampling methods, fractions, and fake-hit thresholds. |

### Concepts

- **Baseline**: Full training set, every individual is fully evaluated. No cache.
- **FEC**: A subset of the training data is used to build a “fingerprint” (e.g. predictions on sampled points). If the cache has that fingerprint, we return the cached fitness (when `fec.evaluate_fake_hits` is False) and skip the full evaluation.
- **Sample fraction**: Fraction of the training set used for the cache (e.g. 0.1 = 10%).
- **Sampling method**: How that subset is chosen (e.g. `farthest_point`, `kmeans`, `stratified`, `random`, `union`).
- **Fake-hit threshold**: Only used when `fec.evaluate_fake_hits` is True: a cache hit is counted as “fake” if the full fitness would differ from the cached value by more than this threshold.

Global settings (dataset, generations, population, FEC on/off, etc.) come from **config.py**.

---

## 2. Directory layout

Paths are derived from **config.py**: `dataset.file`, `evolution.generations`, `evolution.population`.

```
results/
└── <dataset_stem>_Gen_<G>_Pop_<P>/     # e.g. Wisconsin_Breast_Cancer_without_ID_Gen_50_Pop_5000
    ├── baseline/                         # Baseline runs
    │   ├── generation_stats_run1.csv
    │   ├── summary_run1.csv
    │   ├── generation_stats_run2.csv
    │   ├── ...
    │   ├── generation_stats_aggregated.csv   # Created by FEC_report.py
    │   └── summary_aggregated.csv
    ├── FEC/                              # FEC runs (all methods/fractions/thresholds in one folder)
    │   ├── generation_stats_<method>_frac_<p>_th_<tag>_run1.csv
    │   ├── summary_<method>_frac_<p>_th_<tag>_run1.csv
    │   ├── ...
    │   ├── generation_stats_aggregated_FEC.csv
    │   └── summary_aggregated_FEC.csv
    ├── FEC_report.html                  # Full comparison report (when FEC_report.py is run on this folder)
    └── summary_baseline_vs_FEC.csv      # Combined baseline + FEC table (accuracy, MAE, time, speedup)
```

- **baseline/** and **FEC/** are created by the batch scripts / Python entry points.
- **FEC_report.html** and **summary_baseline_vs_FEC.csv** are created when you run **FEC_report.py** on the experiment root (the `..._Gen_<G>_Pop_<P>/` directory).

---

## 3. Configuration (config.py)

Main knobs used by this workflow:

- **Dataset**: `dataset.file`, `dataset.label_column`, `dataset.test_size`
- **Evolution**: `evolution.population`, `evolution.generations`, `evolution.random_seed`
- **FEC (runtime comparison, no overhead)**:
  - `fec.enabled`: True for FEC runs.
  - `fec.evaluate_fake_hits`: False → no re-eval on cache hit (fast, fake_hits = 0).
  - `fec.record_detailed_events`: False → no per-event lists in the cache.
  - `fec.structural_similarity`: False, `fec.behavior_similarity`: True → behaviour-only fingerprint.
- **Output**: `output.plot`, `output.track_individuals`, `output.save_individuals_csv` (all False for minimal overhead).

Seeds:

- **Baseline**: `seed = base_seed + (run_index - 1)`.
- **FEC**: `seed = hash(base_seed, run_index, sampling_method, sample_fraction)` so each (method, fraction, run) is independent.

---

## 4. How to use

### Typical workflow

1. **Configure**  
   Edit **config.py** (dataset, generations, population, FEC options). Optionally edit **run_FEC_batch.sh** to set `SAMPLING_METHODS`, `FRACTIONS`, `FAKE_HIT_THRESHOLDS`, `RUNS_PER_CONFIG`, `MAX_PARALLEL`.

2. **Run baseline**  
   From the project root:
   ```bash
   chmod +x run_baseline_batch.sh
   ./run_baseline_batch.sh          # default: 20 runs, max 4 in parallel
   ./run_baseline_batch.sh 30 8     # 30 runs, max 8 in parallel
   ```
   This writes into `results/<dataset>_Gen_<G>_Pop_<P>/baseline/`.

3. **Run FEC**  
   ```bash
   chmod +x run_FEC_batch.sh
   ./run_FEC_batch.sh               # uses defaults in the script (methods, fractions, thresholds, runs per config)
   ./run_FEC_batch.sh 5             # 5 runs per (method, fraction, threshold)
   ./run_FEC_batch.sh 30 4          # 30 runs per config, max 4 in parallel
   ```
   This writes into `results/<dataset>_Gen_<G>_Pop_<P>/FEC/`.

4. **Generate report**  
   Point **FEC_report.py** at the **experiment root** (the folder that contains both `baseline` and `FEC`):
   ```bash
   python FEC_report.py results/Wisconsin_Breast_Cancer_without_ID_Gen_50_Pop_5000/
   ```
   This creates/updates:
   - `FEC_report.html` (charts: per threshold/fraction, across fractions, across thresholds, speedup, etc.)
   - `summary_baseline_vs_FEC.csv` (one row per config: baseline + FEC metrics and speedup)
   - Aggregated CSVs in `baseline/` and `FEC/`.

5. **Inspect**  
   Open `FEC_report.html` in a browser; use `summary_baseline_vs_FEC.csv` for tables or further analysis.

---

## 5. Script reference

### baseline_runs.py — single baseline run

**Purpose**: One GE run without FEC on the full training set.

**Usage**:
```bash
python baseline_runs.py --run-index <n> [--base-seed <seed>]
```

**Arguments**:

| Argument      | Required | Default (config) | Description |
|---------------|----------|-------------------|-------------|
| `--run-index` | Yes      | —                 | 1-based run index (seed = base_seed + run_index - 1). |
| `--base-seed` | No       | `evolution.random_seed` | Base RNG seed. |

**Output**:  
`results/<dataset>_Gen_<G>_Pop_<P>/baseline/generation_stats_run<n>.csv`, `summary_run<n>.csv`.

---

### FEC_runs.py — single FEC run

**Purpose**: One GE run with FEC for one (sampling method, sample fraction, fake-hit threshold).

**Usage**:
```bash
python FEC_runs.py --run-index <n> --base-seed <seed> --sample-fraction <f> --sampling-method <name> [--fake-hit-threshold <th>]
```

**Arguments**:

| Argument               | Required | Default (config)   | Description |
|------------------------|----------|--------------------|-------------|
| `--run-index`          | Yes      | —                  | 1-based run index. |
| `--base-seed`          | No       | `evolution.random_seed` | Base seed (combined with run-index, method, fraction for RNG). |
| `--sample-fraction`   | Yes      | —                  | Fraction of training data for cache (e.g. 0.1, 0.2). |
| `--sampling-method`    | Yes      | —                  | One of: `kmeans`, `kmedoids`, `farthest_point`, `stratified`, `random`, `union`. |
| `--fake-hit-threshold` | No       | `fec.fake_hit_threshold` | Threshold for fake-hit classification when `fec.evaluate_fake_hits` is True. |

**Output**:  
`results/.../FEC/generation_stats_<method>_frac_<p>_th_<tag>_run<n>.csv`, `summary_...csv`.

---

### FEC_report.py — aggregate and report

**Purpose**: Aggregate per-run CSVs and (when given an experiment root) produce the comparison report and combined summary CSV.

**Usage**:
```bash
# Full comparison (baseline + FEC → HTML report and summary CSV)
python FEC_report.py results/<dataset>_Gen_<G>_Pop_<P>/

# Baseline-only aggregation (no HTML comparison)
python FEC_report.py results/<dataset>_Gen_<G>_Pop_<P>/baseline

# FEC-only aggregation (no HTML comparison)
python FEC_report.py results/<dataset>_Gen_<G>_Pop_<P>/FEC
```

**Argument**:

| Argument     | Description |
|-------------|-------------|
| `target_dir` | Path to (a) experiment root (contains `baseline/` and `FEC/`), (b) `baseline/` only, or (c) `FEC/` only. |

**Output when target is experiment root**:

- `target_dir/FEC_report.html`: interactive charts (by threshold → fraction → metrics; across fractions; across thresholds; speedup).
- `target_dir/summary_baseline_vs_FEC.csv`: combined table (baseline row + one row per FEC config, with accuracy, MAE, time, speedup).
- `target_dir/baseline/generation_stats_aggregated.csv`, `summary_aggregated.csv`.
- `target_dir/FEC/generation_stats_aggregated_FEC.csv`, `summary_aggregated_FEC.csv`.

---

### run_baseline_batch.sh — many baseline runs

**Purpose**: Run `baseline_runs.py` multiple times with different `--run-index`, in parallel.

**Usage**:
```bash
./run_baseline_batch.sh [NUM_RUNS] [MAX_PARALLEL]
```

**Parameters** (defaults in script):

| Parameter       | Default | Description |
|-----------------|---------|-------------|
| `NUM_RUNS`      | 20      | Total number of runs (run-index 1..NUM_RUNS). |
| `MAX_PARALLEL`  | 4       | Max number of runs in parallel. |

**Example**: `./run_baseline_batch.sh 30 8` → 30 runs, up to 8 at a time.

---

### run_FEC_batch.sh — many FEC runs (grid)

**Purpose**: Run `FEC_runs.py` over a grid of sampling methods, fractions, and fake-hit thresholds, with multiple runs per cell.

**Usage**:
```bash
./run_FEC_batch.sh [RUNS_PER_CONFIG] [MAX_PARALLEL]
```

**Parameters** (defaults in script):

| Parameter          | Default | Description |
|--------------------|---------|-------------|
| `RUNS_PER_CONFIG`  | 30      | Number of runs per (method, fraction, threshold). |
| `MAX_PARALLEL`     | 20      | Max parallel jobs. |
| `BASE_SEED`        | 42      | Base seed (edit in script). |

**Grid** (edit arrays in the script):

- `SAMPLING_METHODS`: e.g. `("farthest_point")` or `("farthest_point" "kmeans")`.
- `FRACTIONS`: e.g. `(0.1 0.2 0.3 0.4)`.
- `FAKE_HIT_THRESHOLDS`: e.g. `(0 1e-1 1e-3 1e-4 1e-5)`.

**Example**: `./run_FEC_batch.sh 5 4` → 5 runs per (method, fraction, threshold), max 4 in parallel.

---

## 6. Report contents (FEC_report.html)

When you run **FEC_report.py** on the experiment root, the HTML report includes:

- **Config** (from config.py).
- **Comparison table**: FEC vs baseline (MAE, time, speedup, meaningful_speedup flag).
- **Per threshold / per fraction**: Sections by fake-hit threshold, then by sample fraction; for each fraction, charts: Training MAE, Test MAE, Best test MAE, Hit rate, Fake-hit rate (baseline vs sampling methods for that fraction).
- **Across fractions**: Final test MAE, hit rate, fake-hit rate, runtime ratio, **speedup** vs baseline (legend: method + threshold).
- **Across thresholds**: Hit rate, fake-hit rate, runtime ratio, **speedup** vs baseline (legend: method + fraction).
- **summary_baseline_vs_FEC.csv**: One row for baseline, one per FEC config; columns include accuracy, MAE, time, speedup, delta_mae_vs_baseline, delta_accuracy_vs_baseline; FEC-only columns are empty for the baseline row.

---

## 7. Help (quick reference)

| Task | Command |
|------|--------|
| Run 20 baseline runs (4 at a time) | `./run_baseline_batch.sh` |
| Run 30 baseline runs, 8 at a time | `./run_baseline_batch.sh 30 8` |
| Run one baseline run | `python baseline_runs.py --run-index 1 --base-seed 42` |
| Run FEC grid (defaults in script) | `./run_FEC_batch.sh` |
| Run one FEC run | `python FEC_runs.py --run-index 1 --base-seed 42 --sample-fraction 0.1 --sampling-method farthest_point --fake-hit-threshold 1e-5` |
| Generate full report | `python FEC_report.py results/<dataset>_Gen_<G>_Pop_<P>/` |
| Aggregate baseline only | `python FEC_report.py results/<dataset>_Gen_<G>_Pop_<P>/baseline` |
| Aggregate FEC only | `python FEC_report.py results/<dataset>_Gen_<G>_Pop_<P>/FEC` |
| Python help | `python baseline_runs.py -h`, `python FEC_runs.py -h`, `python FEC_report.py -h` |

Replace `<dataset>_Gen_<G>_Pop_<P>` with the actual folder name under `results/` (e.g. `Wisconsin_Breast_Cancer_without_ID_Gen_50_Pop_5000`).

---

## 8. Dependencies and config

- **config.py** is the single source for dataset, evolution, and FEC settings; batch scripts and report use the same `results/` layout keyed by dataset stem, generations, and population.
- **FEC_runs.py** reads `fec.evaluate_fake_hits` (and related FEC options) from **config.py** so that “no overhead” runtime comparison is controlled from config (no re-eval on cache hit, no detailed event lists, etc.).

For a minimal runtime comparison: set in config `fec.evaluate_fake_hits: False`, `fec.record_detailed_events: False`, `output.track_individuals: False`, `output.plot: False`, then run baseline and FEC batches and generate the report as above.
