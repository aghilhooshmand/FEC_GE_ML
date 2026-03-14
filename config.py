# Configuration file - all parameters are flattened (dot notation) for simplicity
# Example: "dataset.file" instead of nested {"dataset": {"file": ...}}
# This makes the code simpler and eliminates the need for flattening functions

CONFIG = {
    # --- Dataset inputs -----------------------------------------------------
    # Local CSV under ./data
    "dataset.file": "Wisconsin_Breast_Cancer_without_ID.csv",
    "dataset.label_column": "diagnosis",  # Target column name (uses last column if missing)
    # Optional subsampling of the dataset before train/test split.
    # - If None or 1.0: use the full dataset.
    # - If 0 < dataset.sample_fraction < 1.0: randomly select that fraction of
    #   rows, preserving the class ratio (stratified on the label column).
    "dataset.sample_fraction": None,
    "dataset.test_size": 0.2,  # Fraction of data used for testing (0.0-1.0)

    # --- Grammar configuration ----------------------------------------------
    "grammar.file": "heartDisease.bnf",  # BNF grammar file in ./grammars directory

    # --- Evolution loop parameters ------------------------------------------
    "evolution.population": 10,  # Number of individuals per generation
    "evolution.generations": 3,  # Number of evolutionary generations per run
    "evolution.random_seed": 42,  # Base RNG seed; each run offsets by +run index
    "evolution.n_runs": 1,  # Number of independent runs for statistics/averaging

    # --- GA operator tuning -------------------------------------------------
    "ga_parameters.p_crossover": 0.8,  # Probability of crossover between parents
    "ga_parameters.p_mutation": 0.05,  # Probability of mutating a genome
    "ga_parameters.tournsize": 4,  # Tournament size for selection
    "ga_parameters.elite_size": 1,  # Number of elites preserved each generation
    "ga_parameters.halloffame_size": 1,  # Best individuals retained at the end

    # --- GE (mapping) parameters --------------------------------------------
    "ge_parameters.min_init_tree_depth": 3,  # Minimum depth when initialising individuals
    "ge_parameters.max_init_tree_depth": 7,  # Maximum depth when initialising individuals
    "ge_parameters.max_tree_depth": 35,  # Cut-off depth for any generated phenotype tree
    "ge_parameters.codon_size": 255,  # Upper bound for codon values
    "ge_parameters.genome_representation": "list",  # Storage type: "list" or "array"
    "ge_parameters.codon_consumption": "lazy",  # How codons are consumed: "lazy" or "eager"
    "ge_parameters.max_init_genome_length": None,  # Optional cap on genome length on creation
    "ge_parameters.max_genome_length": None,  # Optional cap enforced during evolution
    "ge_parameters.max_wraps": 0,  # Number of grammar wraps allowed during mapping

    # --- Reporting controls -------------------------------------------------
    # Metrics extracted from DEAP logbook that end up in CSV export (list of strings)
    "report_items": [
        "gen", "invalid", "avg", "std", "min", "max",
        "fitness_test",
        "best_ind_length", "avg_length",
        "best_ind_nodes", "avg_nodes",
        "best_ind_depth", "avg_depth",
        "avg_used_codons", "best_ind_used_codons",
        "selection_time", "generation_time",
    ],

    # --- Fitness Evaluation Cache (FEC) knobs -------------------------------
    "fec.enabled": True,  # Master switch for the cache
    
    # ------------------------------------------------------------------------
    # FEC ANALYSIS vs. RUNTIME-COMPARISON MODES
    #
    # 1) Analysis mode (for understanding behaviour, fake hits, breakdowns):
    #    - fec.structural_similarity: True
    #    - fec.behavior_similarity: True
    #    - fec.evaluate_fake_hits: True
    #    - fec.record_detailed_events: True (if RAM allows)
    #    - output.track_individuals: True (if needed)
    #
    # 2) Runtime-comparison mode (NO OVERHEAD – for accurate FEC vs baseline time):
    #    - fec.structural_similarity: False     (behaviour-key only, no phenotype in fingerprint)
    #    - fec.behavior_similarity: True
    #    - fec.evaluate_fake_hits: False        (no re-eval on cache hit; fake_hits stay 0)
    #    - fec.record_detailed_events: False   (no detailed_* lists; FECCache uses config)
    #    - output.track_individuals: False     (PhenotypeTracker does not store per-individual data)
    #    - output.plot: False                  (no Plotly charts during run)
    #    - output.save_individuals_csv: False
    # The current config below is set up for RUNTIME-COMPARISON (no overhead).
    # ------------------------------------------------------------------------
    
    # FEC mode enable/disable flags: when True, compute and show the corresponding series in hit-rate charts
    "fec.modes.fec_disabled": False,
    "fec.modes.fec_enabled_behaviour_without_structural": True,  # show "behavioural without structural" in breakdown chart
    "fec.modes.fec_enabled_structural_only": True,              # show "just structural" in breakdown chart
    "fec.modes.fec_enabled_total": True,                      # show total hit rate (hit vs miss) in overall chart and "behavioural" in breakdown
    
    "fec.sample_sizes": [round(i * 0.1, 2) for i in range(1, 5)],  # Sweep values: fractions of dataset (0.0-1.0) or absolute counts, from 0.02 to 0.9 in 0.02 increments
    
    # Sampling method enable/disable flags (set to True to enable, False to disable)
    "fec.sampling_methods.enabled": {
        "kmeans": False,
        "kmedoids": False,
        "farthest_point": True,
        "stratified": False,
        "random": False,
        # Special composite method: "union"
        # - When enabled here AND configured via "fec.sampling_methods.union",
        #   the system will build a sample that is the union of several base
        #   sampling methods (see "fec.sampling_methods.union" below).
        "union": False,
    },
    # Union sampling configuration:
    # - False (default): union sampling is disabled.
    # - List of method names: e.g. ["random", "stratified"].
    #   For each experiment configuration, the system will:
    #     * run each listed base sampling method with the same sample size,
    #     * take the union of all selected samples (by index),
    #     * use that union as the final sampled dataset for FEC.
    "fec.sampling_methods.union": ["kmeans", "farthest_point","stratified"],
    
    # --- Auto-selection of sampling methods (disabled in this refactor) ------
    # Kept here as comments for possible future use.
    # "fec.auto_select_sampling_method": False,
    # "fec.auto_select.n_repetitions": 10,
    # "fec.auto_select.sample_fraction": 0.1,
    # "fec.auto_select.distance_metric": "ks",
    
    # When False, FEC cache does not store per-event lists (detailed_hits/misses/fake_hits); saves RAM and avoids overhead.
    "fec.record_detailed_events": False,
    # When False, on cache HIT we return cached fitness without re-evaluating (no fake-hit measurement, no extra work).
    "fec.evaluate_fake_hits": True,
    "fec.fake_hit_threshold": 1e-5	,  # Kept for completeness (ignored when evaluate_fake_hits is False)
    "fec.structural_similarity": False,  # Use behaviour-key only in fingerprints (no phenotype component)
    "fec.behavior_similarity": True,  # Include centroid predictions + labels in key
    # Behaviour key sampling for cache fingerprints (now simplified to use all centroids):
    # "fec.behavior_key.sample_fraction": 1.0,
    # "fec.behavior_key.max_points": 0,
    # "fec.behavior_key.random_subset": False,

    # --- Output artefacts ---------------------------------------------------
    # Set False for runtime-comparison (no overhead). Set True if you want Plotly charts during run.
    "output.plot": False,
    "output.save_individuals_csv": False,
    # When False, PhenotypeTracker does not store per-individual data (saves RAM, no overhead).
    "output.track_individuals": False,
    # When True, skip running new experiments and ONLY generate HTML reports from an existing CSV
    "reports.from_csv_only": False,

    # --- MLflow tracking (disabled / commented out) -------------------------
    # "mlflow.enabled": False,  # Toggle experiment tracking entirely
    # "mlflow.experiment_name": "Sample size effect - RFC breast cancer Experiments V1.0",
    # "mlflow.run_name_prefix": "setup",
    # "mlflow.tracking_uri": "sqlite:///results/mlflow/mlflow.db",
    # "mlflow.artifact_paths.charts": "charts",
    # "mlflow.artifact_paths.reports": "reports",
    # "mlflow.artifact_paths.cache_stats": "cache_stats",

    # --- Resume / report from previous run ----------------------------------
    # Set to a CSV file path (relative to results/ or absolute) to base reports on that file.
    # Example: "resume.from_csv": "run002/run002_all_experiments_20251123_152010.csv"
    # - When reports.from_csv_only = False: this CSV is used ONLY to skip already-completed configs.
    # - When reports.from_csv_only = True: this CSV is used as the data source for HTML reports,
    #   and NO new experiments are executed.
    "resume.from_csv": None,  # None = auto-detect latest when reports.from_csv_only is True

    # --- Config validation --------------------------------------------------
    "check_config": True,  # When True, validates mandatory fields before executing
}
