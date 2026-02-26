# Configuration file - all parameters are flattened (dot notation) for simplicity
# Example: "dataset.file" instead of nested {"dataset": {"file": ...}}
# This makes the code simpler and eliminates the need for flattening functions

CONFIG = {
    # --- Dataset inputs -----------------------------------------------------
    # Local CSV under ./data
    "dataset.file": "clinical_breast_cancer_RFC_preprocessed.csv",
    "dataset.label_column": "RFS_STATUS",  # Target column name (uses last column if missing)
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
    "evolution.generations": 2,  # Number of evolutionary generations per run
    "evolution.random_seed": 42,  # Base RNG seed; each run offsets by +run index
    "evolution.n_runs": 2,  # Number of independent runs for statistics/averaging

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

    # --- Parallel evaluation --------------------------------------------------
    # Number of worker threads used to evaluate individuals in parallel.
    # None = use os.cpu_count() on the server.
    "parallel.n_workers": None,

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
    
    # FEC mode enable/disable flags (set to True to enable, False to disable)
    "fec.modes.fec_disabled": False,
    "fec.modes.fec_enabled_behaviour": True,
    "fec.modes.fec_enabled_structural": False,
    "fec.modes.fec_enabled_behaviour_structural": False,
    
    "fec.sample_sizes": [round(i * 0.1, 2) for i in range(1, 7)],  # Sweep values: fractions of dataset (0.0-1.0) or absolute counts, from 0.02 to 0.9 in 0.02 increments
    
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
    
    # --- Auto-selection of sampling methods -----------------------------------
    # When True: automatically selects the best sampling method by comparing
    # how well each method's samples represent the full dataset using statistical
    # distance metrics. Only the winning method is used for all subsequent experiments.
    # When False: all enabled sampling methods are run independently (default behavior).
    "fec.auto_select_sampling_method": False,
    # Number of repetitions for auto-selection evaluation (default: 20)
    "fec.auto_select.n_repetitions": 10,
    # Sample fraction to use during auto-selection evaluation (default: 0.1 = 10%)
    "fec.auto_select.sample_fraction": 0.1,
    # Statistical distance metric(s) to use for comparison:
    # - "ks": Kolmogorov-Smirnov distance (univariate, per feature)
    # - "wasserstein": Wasserstein distance (multivariate)
    # - "energy": Energy distance (E-test)
    # - "mmd": Maximum Mean Discrepancy
    # - "frechet": Fréchet distance (FID-style)
    # - "average": Average over all available metrics (recommended)
    # Can be a single string or a list of strings
    "fec.auto_select.distance_metric": "ks",
    
    "fec.evaluate_fake_hits": True,  # Re-score cached individuals to detect drift
    "fec.fake_hit_threshold": 0,  # Allowed delta before a cached hit is flagged, 0 means no threshold ( must exactly the same,if not fake happened)
    "fec.structural_similarity": False,  # Include phenotype string in the cache key #if False means phenotype dosent include when hash fingerprint for looking in cach is creating 
    "fec.behavior_similarity": True,  # Include centroid predictions + labels in key
    # Behaviour key sampling for cache fingerprints:
    # - fec.behavior_key.sample_fraction: fraction of centroid points to use in the key (0-1]. 1.0 = all points.
    # - fec.behavior_key.max_points: hard cap on number of points in the key (set to 0 or None for no cap).
    # - fec.behavior_key.random_subset: when True, pick a random subset of points for the key; when False, use the first N.
    "fec.behavior_key.sample_fraction": 1.0,
    "fec.behavior_key.max_points": 0,
    "fec.behavior_key.random_subset": False,

    # --- Output artefacts ---------------------------------------------------
    "output.plot": True,  # Whether to emit Plotly charts (saved under ./results/)
    # When True, skip running new experiments and ONLY generate HTML reports from an existing CSV
    "reports.from_csv_only": False,

    # --- MLflow tracking ----------------------------------------------------
    "mlflow.enabled": True,  # Toggle experiment tracking entirely
    "mlflow.experiment_name": "Sample size effect - RFC breast cancer Experiments V1.0",  # MLflow experiment bucket
    "mlflow.run_name_prefix": "setup",  # Prefix for auto-generated run names
    "mlflow.tracking_uri": "sqlite:///results/mlflow/mlflow.db",  # Tracking backend
    "mlflow.artifact_paths.charts": "charts",  # Subdirectory for chart artifacts
    "mlflow.artifact_paths.reports": "reports",  # Subdirectory for report artifacts
    "mlflow.artifact_paths.cache_stats": "cache_stats",  # Subdirectory for cache stats

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
