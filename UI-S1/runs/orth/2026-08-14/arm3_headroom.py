import json
import math
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
MASK_STAGE1_PATH = RUN_DIR.parents[2] / "runs/mask/2026-08-14/STAGE1.json"
OUTPUT_PATH = RUN_DIR / "ARM3.json"
ACCURACIES = tuple(round(value, 2) for value in np.arange(0.50, 0.951, 0.05))
KAPPAS = tuple(round(value, 1) for value in np.arange(-0.2, 0.801, 0.1))
VISUAL_WEIGHTS = (12.0, 1.5936767669403409, 1.0)


def coupling(visual_error, new_error, requested_kappa):
    expected_agreement = visual_error * new_error + (1 - visual_error) * (1 - new_error)
    both_error_unconstrained = (
        visual_error * new_error
        + 0.5 * requested_kappa * (1 - expected_agreement)
    )
    lower = max(0.0, visual_error + new_error - 1)
    upper = min(visual_error, new_error)
    both_error = min(max(both_error_unconstrained, lower), upper)
    observed_agreement = 1 - visual_error - new_error + 2 * both_error
    achieved_kappa = (
        (observed_agreement - expected_agreement) / (1 - expected_agreement)
        if not math.isclose(expected_agreement, 1.0) else None
    )
    return {
        "both_correct": 1 - visual_error - new_error + both_error,
        "visual_only_correct": new_error - both_error,
        "new_only_correct": visual_error - both_error,
        "both_wrong": both_error,
        "requested_kappa": requested_kappa,
        "achieved_kappa": achieved_kappa,
        "projected_to_feasible": not math.isclose(both_error, both_error_unconstrained, abs_tol=1e-15),
        "feasible_both_error_interval": [lower, upper],
    }


def simulate_coupling(probabilities, seeds=100, rows=1581):
    categories = np.asarray([
        probabilities["both_correct"], probabilities["visual_only_correct"],
        probabilities["new_only_correct"], probabilities["both_wrong"],
    ], dtype=np.float64)
    values = []
    for seed in range(seeds):
        counts = np.random.default_rng(20260814 + seed).multinomial(rows, categories)
        both_correct, visual_only, new_only, both_wrong = counts / rows
        visual_error = new_only + both_wrong
        new_error = visual_only + both_wrong
        agreement = both_correct + both_wrong
        expected = visual_error * new_error + (1 - visual_error) * (1 - new_error)
        kappa = (agreement - expected) / (1 - expected) if not math.isclose(expected, 1.0) else None
        values.append({
            "visual_accuracy": both_correct + visual_only,
            "new_accuracy": both_correct + new_only,
            "oracle_accuracy": 1 - both_wrong,
            "kappa": kappa,
        })
    return {
        key: {
            "mean": float(np.mean([row[key] for row in values])),
            "range": [float(np.min([row[key] for row in values])), float(np.max([row[key] for row in values]))],
        }
        for key in ("visual_accuracy", "new_accuracy", "oracle_accuracy", "kappa")
    }


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    stage1 = json.loads(MASK_STAGE1_PATH.read_text())
    visual_accuracy = float(stage1["base_rates_and_masks"]["density_B3_accuracy"])
    visual_error = 1 - visual_accuracy
    table = []
    for accuracy in ACCURACIES:
        for kappa in KAPPAS:
            value = coupling(visual_error, 1 - accuracy, kappa)
            value.update({
                "requested_new_accuracy": accuracy,
                "visual_accuracy": visual_accuracy,
                "forced_new_accuracy": accuracy,
                "random_disagreement_accuracy": (
                    value["both_correct"]
                    + 0.5 * (value["visual_only_correct"] + value["new_only_correct"])
                ),
                "oracle_selector_accuracy": 1 - value["both_wrong"],
                "oracle_gain_over_visual": visual_error - value["both_wrong"],
                "disagreement_probability": value["visual_only_correct"] + value["new_only_correct"],
                "weight_rules": {
                    str(weight): {
                        "visual_weight": weight,
                        "new_weight": 1.0,
                        "disagreement_choice": "visual" if weight > 1 else "random_tie",
                        "accuracy_without_row_confidence": (
                            visual_accuracy if weight > 1
                            else value["both_correct"] + 0.5 * (
                                value["visual_only_correct"] + value["new_only_correct"]
                            )
                        ),
                    }
                    for weight in VISUAL_WEIGHTS
                },
                "multinomial_coupling_simulation_100_seeds": simulate_coupling(value),
            })
            table.append(value)
    result = {
        "schema_version": 1,
        "status": "PASS_ORTH_ARM3_IDENTIFIABLE_HEADROOM_COMPLETE",
        "visual_accuracy": visual_accuracy,
        "visual_error": visual_error,
        "accuracy_grid": list(ACCURACIES),
        "kappa_grid": list(KAPPAS),
        "visual_weights": list(VISUAL_WEIGHTS),
        "table": table,
        "identifiability_boundary": {
            "bayes_fused_grounding_accuracy_identified": False,
            "reason": "accuracy_and_error_kappa_do_not_define_candidate_identity_or_row_level_likelihoods",
            "reported_metrics": "joint_error_coupling_and_selector_upper_bounds_only",
        },
        "claim_boundary": {"design_headroom_only": True, "runtime_rule_allowed": False},
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "cells": len(table),
        "projected_cells": sum(row["projected_to_feasible"] for row in table),
        "oracle_gain_range": [min(row["oracle_gain_over_visual"] for row in table), max(row["oracle_gain_over_visual"] for row in table)],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()