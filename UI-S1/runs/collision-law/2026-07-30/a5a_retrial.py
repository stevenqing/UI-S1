import argparse
import json
from pathlib import Path

from aggregators import pka_medoid, pka_medoid_leave_one_out, plurality_then_density
from w1_run import (
    POOLS,
    deployable_models,
    dev_priority,
    fold_map,
    load_pool,
    model_step_sr,
    prediction_from_row,
    score_prediction,
    split_rows,
)


METHODS = ("A2_sequential_density", "A3_joint_original", "A5a_joint_leave_one_out")


def aggregate(method, bench, model_rows, models, priority):
    predictions = [prediction_from_row(model_rows[model]) for model in models]
    if method == "A2_sequential_density":
        return plurality_then_density(bench, predictions, priority).prediction
    if method == "A3_joint_original":
        return pka_medoid(bench, predictions).prediction
    if method == "A5a_joint_leave_one_out":
        return pka_medoid_leave_one_out(bench, predictions).prediction
    raise ValueError(method)


def evaluate_pool(bench, setting):
    identities, available_models, pivot = load_pool(bench, setting)
    models = deployable_models(identities, available_models, pivot)
    mapping = fold_map(f"{bench}/{setting}")
    folds = []
    totals = {method: 0 for method in METHODS}
    total_rows = 0
    changed_from_a3 = 0
    comparisons = {
        "A5a_vs_A3": {"wins": 0, "losses": 0},
        "A5a_vs_A2": {"wins": 0, "losses": 0},
    }
    for test_fold in range(5):
        dev_ids, test_ids = split_rows(identities, pivot, mapping, test_fold)
        priority = dev_priority(dev_ids, models, pivot)
        heldout_model = priority[0]
        successes = {method: 0 for method in METHODS}
        fold_changed = 0
        fold_comparisons = {
            "A5a_vs_A3": {"wins": 0, "losses": 0},
            "A5a_vs_A2": {"wins": 0, "losses": 0},
        }
        for row_id in test_ids:
            reference = next(iter(pivot[row_id].values()))
            predictions = {
                method: aggregate(method, bench, pivot[row_id], models, priority)
                for method in METHODS
            }
            outcomes = {
                method: score_prediction(reference, prediction)
                for method, prediction in predictions.items()
            }
            for method, success in outcomes.items():
                successes[method] += int(success)
            if predictions["A5a_joint_leave_one_out"] != predictions["A3_joint_original"]:
                fold_changed += 1
            for label, baseline in (
                ("A5a_vs_A3", "A3_joint_original"),
                ("A5a_vs_A2", "A2_sequential_density"),
            ):
                a5a = outcomes["A5a_joint_leave_one_out"]
                base = outcomes[baseline]
                fold_comparisons[label]["wins"] += int(a5a and not base)
                fold_comparisons[label]["losses"] += int(base and not a5a)
        for method in METHODS:
            totals[method] += successes[method]
        total_rows += len(test_ids)
        changed_from_a3 += fold_changed
        for label in comparisons:
            for key in comparisons[label]:
                comparisons[label][key] += fold_comparisons[label][key]
        folds.append({
            "fold": test_fold,
            "dev_rows": len(dev_ids),
            "test_rows": len(test_ids),
            "heldout_best_model": heldout_model,
            "priority": priority,
            "step_sr": {
                "A0_heldout_best": model_step_sr(test_ids, heldout_model, pivot),
                **{method: successes[method] / len(test_ids) for method in METHODS},
            },
            "a3_to_a5a_changed_rows": fold_changed,
            "comparisons": fold_comparisons,
        })
    aggregate_metrics = {method: totals[method] / total_rows for method in METHODS}
    aggregate_metrics["A0_heldout_best"] = sum(
        fold["step_sr"]["A0_heldout_best"] * fold["test_rows"] for fold in folds
    ) / total_rows
    return {
        "models": models,
        "rows": total_rows,
        "folds": folds,
        "aggregate_step_sr": aggregate_metrics,
        "a3_to_a5a_changed_rows": changed_from_a3,
        "comparisons": comparisons,
        "k3_retrial": {
            "a5a_exceeds_sequential_density": (
                aggregate_metrics["A5a_joint_leave_one_out"] > aggregate_metrics["A2_sequential_density"]
            ),
            "delta_a5a_minus_a2": (
                aggregate_metrics["A5a_joint_leave_one_out"] - aggregate_metrics["A2_sequential_density"]
            ),
        },
        "collision_tax": {
            "delta_a2_minus_a0": aggregate_metrics["A2_sequential_density"] - aggregate_metrics["A0_heldout_best"],
            "independent_of_a5a": True,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "status": "PASS",
        "contract": {
            "scope": "deployable",
            "folds": "runs/complementarity/2026-07-30/folds.json",
            "only_change_from_a3": "exclude candidate self-vote i=j",
            "candidate_order_and_tie_break_unchanged": True,
            "test_tuning": False,
        },
        "pools": {},
    }
    for bench, setting in POOLS:
        result["pools"][f"{bench}/{setting}"] = evaluate_pool(bench, setting)
    passes = sum(pool["k3_retrial"]["a5a_exceeds_sequential_density"] for pool in result["pools"].values())
    result["k3_retrial_summary"] = {
        "directional_passes": passes,
        "pools": len(result["pools"]),
        "all_pools_pass": passes == len(result["pools"]),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "k3_retrial": result["k3_retrial_summary"],
        "aggregate_step_sr": {
            pool: values["aggregate_step_sr"] for pool, values in result["pools"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()