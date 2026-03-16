from __future__ import annotations

"""
Minimal config for the simple FEC pipeline.

Used only by FEC_runs_simple.py. Run-specific values (seed, sample_fraction,
sampling_method, fake_hit_threshold) are injected from the command line.
"""

from typing import Dict


CONFIG_FEC_SIMPLE: Dict[str, object] = {
    # --- Dataset inputs -----------------------------------------------------
    "dataset.file": "Wisconsin_Breast_Cancer_without_ID.csv",
    "dataset.label_column": "diagnosis",
    "dataset.sample_fraction": None,
    "dataset.test_size": 0.2,

    # --- Grammar configuration ----------------------------------------------
    "grammar.file": "heartDisease.bnf",

    # --- Evolution loop parameters ------------------------------------------
    "evolution.population": 100,
    "evolution.generations": 10,
    "evolution.random_seed": 42,  # will be overwritten per run

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

    # --- FEC defaults (will be overwritten per run) -------------------------
    "fec.enabled": True,
    "fec.sampling_method": "farthest_point",
    "fec.sample_fraction": 0.1,
    "fec.structural_similarity": False,
    "fec.behavior_similarity": True,
    "fec.evaluate_fake_hits": False,
    "fec.record_detailed_events": False,
    "fec.fake_hit_threshold": 1e-5,

    # --- Output controls (no extra overhead) --------------------------------
    "output.plot": False,
    "output.save_individuals_csv": False,
    "output.track_individuals": False,
}

