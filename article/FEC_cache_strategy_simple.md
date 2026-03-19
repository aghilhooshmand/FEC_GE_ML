# FEC Simple Cache Strategy (What it stores, how it keys, and where it lives)

This document explains the cache system used in the *simple* FEC pipeline (the one used by `FEC_runs_simple.py` / `FEC_report_simple.py`).
It focuses on three things:

1. What data is cached
2. How the cache key is computed
3. What parts of the code use the cache (and why it can improve speedup)

---

## 1) Where the cache is configured (configuration knobs)

The cache strategy for the simple FEC pipeline is controlled in:

- `config_FEC_simple.py`

The relevant keys are:

- `fec.cache_backend`
  - `dict`: unbounded in-memory cache (fastest lookups, but can grow in size)
  - `lru`: bounded cache using LRU eviction (limits memory, but adds overhead)
- `fec.cache_max_entries`
  - Only used when `fec.cache_backend == "lru"`
- `fec.cache_key`
  - Determines how the cache key is computed:
    - `behavior_repr`: large `repr(...)` key of sampled predictions and centroid labels
    - `behavior_hash` (recommended): compact hash key of the sampled behavior
    - `phenotype`: exact `individual.phenotype` string as the key
    - `phenotype_hash`: hash of the phenotype string

Your optimized default is:

- `fec.cache_backend = "dict"`
- `fec.cache_key = "behavior_hash"`

This choice is specifically aimed at reducing key size and lookup/store overhead.

---

## 2) What the cache stores (cache values)

In the simple FEC pipeline, the cache stores a mapping:

- `cache[key] = fitness_full`

Where `fitness_full` is the fitness computed after evaluating the individual’s phenotype on the input points (the “full evaluation” for that call).

The cache also tracks statistics:

- `hits`: number of times `key` was already present
- `misses`: number of times `key` was absent
- `fake_hits`: only used when “fake-hit evaluation” is enabled
- `fake_eval_time_sec`: accumulated time spent on fake-hit evaluation

These stats are returned by the simple FEC run and written to the per-run summary CSV.

---

## 3) How the cache key is computed (cache keys)

The cache key is created in `util_simple.py` using:

- `SimpleFECCache`
- `_make_cache_key(...)`
- `create_fec_fitness(...)` (the function that uses caching during evaluation)

### 3.1 Cache key inputs

To build the key, the code first computes a “behavior” signal on a sampled subset:

- `pred_sample` is the prediction of the individual phenotype on the sampled features (`centroid_X`)
- `centroid_y` is the sampled labels corresponding to `centroid_X`

So the cache key depends on:

- the individual’s behavior on the sample points
- the sampled labels (centroid labels)

### 3.2 Key stability (rounding)

When using behavior-based keys, the code quantizes the sampled predictions:

- it rounds sampled predictions to 6 decimal places before key creation

This increases cache hit rate by making small floating-point differences less likely to produce a different key.

### 3.3 `behavior_hash` key

With `fec.cache_key = "behavior_hash"`:

- the quantized sampled predictions and centroid labels are converted to bytes
- the code computes a compact digest using `blake2b` (16 bytes, hex output)

This produces a small constant-size key instead of a long `repr(...)` string.

### 3.4 Other key modes (when you want to compare later)

- `behavior_repr`
  - produces a large string key containing tuples of rounded prediction values and centroid labels
  - accurate but expensive (string construction and memory overhead)
- `phenotype`
  - uses the raw `individual.phenotype` string directly
  - key may be very large, but it can be more exact depending on phenotype format
- `phenotype_hash`
  - hashes the phenotype string so the key becomes smaller

---

## 4) Where the cache is used (exact code path)

The cache is used inside the fitness evaluation function created by:

- `create_fec_fitness(...)`

In `create_fec_fitness`, the inner function `fitness_eval(...)` performs the following steps:

1. Determine whether the evaluation is on training or test data
2. Compute sampled behavior (`pred_sample`) and build the cache key (`key`)
3. Look up `key` in the cache
4. If there is a hit, return cached fitness (unless fake-hit evaluation is requested)
5. If there is a miss, compute full fitness and store it in the cache

Key point: cache hits skip the full evaluation.

The “hit” behavior is controlled by:

- `evaluate_fake_hits`
  - in your experiments it is `False`, so on a hit it returns cached fitness immediately

---

## 5) Cache backend details (dict vs LRU)

### `dict` backend

- Implemented as a Python dictionary
- Python dictionaries are hash tables internally
- Lookup and store are typically very fast
- There is no eviction; memory can grow with the number of unique keys

### `lru` backend

- Uses `OrderedDict`
- On each hit, it marks the key as “recent”
- When size exceeds `fec.cache_max_entries`, it evicts the oldest entries
- This limits memory but adds overhead (move-to-end and eviction checks)

That’s why the default recommendation for speed testing is `dict` first, then `lru` if memory becomes an issue.

---

## 6) Why these cache settings can increase speedup

Speedup improves when the FEC run avoids repeated expensive work.

In your simple cache system:

- On a cache hit, the code returns the cached `fitness_full`
- Because fitness evaluation is expensive (GE expression evaluation + fitness computation on points), skipping it reduces generation runtime

Switching from `behavior_repr` to `behavior_hash` typically improves:

- key construction overhead (less string building)
- lookup speed (smaller keys)
- memory usage (smaller cached keys)

Switching from `lru` to `dict` further improves:

- lookup/store overhead (no eviction / no move-to-end bookkeeping)

Together, these two changes reduce the per-evaluation overhead and increase the wall-clock “time saved” by cache hits, which shows up as better speedup in the report.

---

## 7) Where to cite in your report

You can cite these pieces directly:

- Cache configuration (keys/backend/key_mode)
  - `config_FEC_simple.py` (look for `fec.cache_backend` and `fec.cache_key`)
- Cache implementation and key creation
  - `util_simple.py`
    - `class SimpleFECCache`
    - `_make_cache_key`
    - `create_fec_fitness` (the cache-hit path)

---

## 8) Important note (training vs test evaluation)

The cache key/value is built using a sampled behavior (`centroid_X`, `centroid_y`).
The code intends to use caching for training evaluations and disable caching for test evaluations.

In `create_fec_fitness`, caching is controlled by a `dataset_type` argument inside `fitness_eval`.
If the caller does not pass `dataset_type="test"` when evaluating on `points_test`, caching may also be applied during fitness-test evaluation.

If you want strict separation later, you can wrap the evaluate function so that:

- training calls pass `dataset_type="train"`
- test calls pass `dataset_type="test"`

This note can be useful for a supervisor if you want to ensure the cached value is used only in the intended place.

