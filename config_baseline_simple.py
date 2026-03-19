from __future__ import annotations

"""
Minimal config for the simple baseline pipeline (no FEC).

Used only by baseline_runs_simple.py. All run-specific values (seed) are
injected from the command line; this module just defines static defaults
for dataset, grammar, evolution, GA/GE parameters, and report_items.
"""

from typing import Dict


CONFIG_BASELINE_SIMPLE: Dict[str, object] = {
    # --- Dataset inputs -----------------------------------------------------
    "dataset.file": "clinical_breast_cancer_RFC_preprocessed.csv",
    "dataset.label_column": "RFS_Status",
    "dataset.sample_fraction": None,
    "dataset.test_size": 0.2,
    # --- Optional balancing ("SMOTH") -------------------------------------
    # If True, oversample the minority class (binary labels) before any split
    # to make the dataset class-balanced.
    "dataset.smoth_balance": True,

    # --- Grammar configuration ----------------------------------------------
    "grammar.file": "heartDisease.bnf",

    # --- Evolution loop parameters ------------------------------------------
    "evolution.population": 1000,
    "evolution.generations": 50,
    "evolution.random_seed": 30,  # will be overwritten per run

    # --- GA operator tuning -------------------------------------------------
    "ga_parameters.p_crossover": 0.8,
    "ga_parameters.p_mutation": 0.05,
    "ga_parameters.tournsize": 4,
    "ga_parameters.elite_size": 1,
    "ga_parameters.halloffame_size": 1,

    # --- GE parameters ------------------------------------------------------
    "ge_parameters.min_init_tree_depth": 3,
    "ge_parameters.max_init_tree_depth": 7,
    "ge_parameters.max_tree_depth": 35,
    "ge_parameters.codon_size": 255,
    "ge_parameters.genome_representation": "list",
    "ge_parameters.codon_consumption": "lazy",
    "ge_parameters.max_init_genome_length": None,
    "ge_parameters.max_genome_length": None,
    "ge_parameters.max_wraps": 0,

    # --- Reporting (full set, same as main config) --------------------------
    "report_items": [
        "gen",
        "invalid",
        "avg",
        "std",
        "min",
        "max",
        "fitness_test",
        "best_ind_length",
        "avg_length",
        "best_ind_nodes",
        "avg_nodes",
        "best_ind_depth",
        "avg_depth",
        "avg_used_codons",
        "best_ind_used_codons",
        "selection_time",
        "generation_time",
    ],

}

