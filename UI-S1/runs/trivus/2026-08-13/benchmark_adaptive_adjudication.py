import json
import sys
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
PRIOR_DIR = RUN_DIR.parent / "2026-08-12"
sys.path.insert(0, str(PRIOR_DIR))

from context_common import atomic_json_file, sha256_file
from finalize_trivus import (
    CELL_ORDER, frozen_baselines, load_configs, load_public, merge_outers,
    paired_samples,
)


FAMILIES = ("mind2web", "screenspot_pro", "androidcontrol")
COMPARISON_OFFSETS = {
    "primary": 400,
    "strongest": 500,
    "joint3": 600,
    "direct_strongest": 700,
    "oracle_strongest": 800,
}
OUTPUT_PATH = RUN_DIR / "BENCHMARK_ADAPTIVE_DIAGNOSTIC.json"
PRIOR_ADJUDICATION_PATH = PRIOR_DIR / "TRIVUS_ADJUDICATION.json"
EXPECTED_PRIOR_OUTCOME = "TRIVUS_NOT_PROMOTED"


def family_comparison(public, left, right, family, comparison, config):
    cells = [(name, cell) for name, cell in CELL_ORDER if name == family]
    reports = {}
    replicates = []
    for index, (_, cell) in enumerate(cells):
        keys = sorted(
            key for key, row in public.items()
            if row["benchmark"] == family
            and (row["setting"] if family == "androidcontrol" else row["arm"]) == cell
        )
        report, samples = paired_samples(
            public,
            keys,
            {key: left[key] for key in keys},
            {key: right[key] for key in keys},
            config["statistics"]["resamples"],
            config["statistics"]["bootstrap_seed_base"]
            + COMPARISON_OFFSETS[comparison]
            + FAMILIES.index(family) * 10
            + index,
        )
        reports[cell] = report
        replicates.append(samples)
    family_samples = np.mean(np.stack(replicates), axis=0)
    return {
        "comparison": comparison,
        "family": family,
        "cells": reports,
        "equal_cell_family": {
            "point_delta": float(np.mean([
                report["point_delta"] for report in reports.values()
            ])),
            "ci_99": [
                float(np.quantile(family_samples, 0.005)),
                float(np.quantile(family_samples, 0.995)),
            ],
        },
    }


def diagnose(outputs, public, primary, strongest, config):
    target = outputs["TARGET_ONLY"]["safe"]
    direct = outputs["TARGET_ONLY"]["direct"]
    joint = outputs["JOINT3"]["safe"]
    if any(set(values) != set(public) for values in (target, direct, joint)):
        raise ValueError("Benchmark-adaptive held-out coverage mismatch")
    oracle = {
        key: bool(direct[key] or strongest[key])
        for key in public
    }
    diagnostics = {}
    for family in FAMILIES:
        comparisons = {
            "primary": family_comparison(
                public, target, primary, family, "primary", config
            ),
            "strongest": family_comparison(
                public, target, strongest, family, "strongest", config
            ),
            "target_minus_joint3": family_comparison(
                public, target, joint, family, "joint3", config
            ),
            "direct_minus_strongest": family_comparison(
                public, direct, strongest, family, "direct_strongest", config
            ),
            "oracle_direct_or_strongest_minus_strongest": family_comparison(
                public, oracle, strongest, family, "oracle_strongest", config
            ),
        }
        mde = config["thresholds"]["mde"][family]
        gates = {
            "primary_cell_safety": all(
                row["ci_99"][0] > -mde
                for row in comparisons["primary"]["cells"].values()
            ),
            "strongest_cell_safety": all(
                row["ci_99"][0] > -mde
                for row in comparisons["strongest"]["cells"].values()
            ),
            "primary_family_improvement": (
                comparisons["primary"]["equal_cell_family"]["ci_99"][0] > 0
            ),
        }
        gates["benchmark_ready_diagnostic"] = all(gates.values())
        diagnostics[family] = {
            "mde": mde,
            "incremental_utility_headroom": (
                comparisons["oracle_direct_or_strongest_minus_strongest"]
                ["equal_cell_family"]
            ),
            "comparisons": comparisons,
            "gates": gates,
        }
    return diagnostics


def main():
    prior = json.loads(PRIOR_ADJUDICATION_PATH.read_text())
    if prior.get("outcome") != EXPECTED_PRIOR_OUTCOME:
        raise PermissionError("Prior TriVUS outcome boundary mismatch")
    config, training_config = load_configs()
    public = load_public()
    outputs, _ = merge_outers(public)
    primary, strongest = frozen_baselines(public, training_config)
    diagnostics = diagnose(outputs, public, primary, strongest, config)
    result = {
        "schema_version": 1,
        "status": "PASS_POSTHOC_BENCHMARK_ADAPTIVE_DIAGNOSTIC",
        "outcome": "EXPLORATORY_ONLY_NO_PROMOTION",
        "method": "TARGET_ONLY_BENCHMARK_SPECIFIC_TRAINING_AND_CALIBRATION",
        "prior_outcome": EXPECTED_PRIOR_OUTCOME,
        "prior_adjudication_sha256": sha256_file(PRIOR_ADJUDICATION_PATH),
        "bootstrap": {
            "resamples": config["statistics"]["resamples"],
            "confidence": config["statistics"]["confidence"],
            "seed_base": config["statistics"]["bootstrap_seed_base"],
            "comparison_offsets": COMPARISON_OFFSETS,
            "equal_cell_composition": True,
        },
        "diagnostics": diagnostics,
        "claim_boundary": {
            "confirmatory": False,
            "promotion_allowed": False,
            "reason": "outer_labels_opened_before_amendment_015",
            "next_confirmation_requires_untouched_labels": True,
        },
    }
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    atomic_json_file(OUTPUT_PATH, result)
    print(json.dumps({
        "outcome": result["outcome"],
        "benchmark_ready": {
            family: value["gates"]["benchmark_ready_diagnostic"]
            for family, value in diagnostics.items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()