import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from common import (
    auc_roc,
    geometric_median,
    load_rows,
    micro,
    pivot_rows,
    score_prediction,
    split_identities,
)
from scoring import token_f1


POOLS = (("androidcontrol", "low"), ("androidcontrol", "high"), ("mind2web", "visual"))


def model_priority(identities, models, pivot):
    return sorted(models, key=lambda model: (-micro(pivot[row_id][model]["success"] for row_id in identities), model))


def grounding_weights(identities, models, pivot):
    output = {}
    for model in models:
        values = []
        for row_id in identities:
            row = pivot[row_id][model]
            if row["bench"] == "androidcontrol" and not math.isnan(row["ground_dist"]):
                values.append(row["ground_dist"] < 0.14)
            elif row["bench"] == "mind2web" and not math.isnan(row["bbox_dist"]):
                values.append(row["bbox_dist"] == 0)
        output[model] = max(micro(values), 1e-6) if values else 1e-6
    return output


def text_medoid(candidates):
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0][1]
    best = None
    for rank, (_, value) in enumerate(candidates):
        score = sum(token_f1(value.lower(), other.lower()) for _, other in candidates) / len(candidates)
        key = (score, -rank)
        if best is None or key > best[0]:
            best = (key, value)
    return best[1]


def ensemble_row(model_rows, selected, priority, weights, mode):
    valid = [model for model in selected if model_rows[model]["parse_ok"] and model_rows[model]["pred_action"]]
    if not valid:
        return {"success": False, "vote_margin": 0.0, "dispersion": 10.0}
    votes = Counter(model_rows[model]["pred_action"] for model in valid)
    highest = max(votes.values())
    tied = {action for action, count in votes.items() if count == highest}
    winning_action = next(
        model_rows[model]["pred_action"] for model in priority
        if model in valid and model_rows[model]["pred_action"] in tied
    )
    vote_counts = sorted(votes.values(), reverse=True)
    runner_up = vote_counts[1] if len(vote_counts) > 1 else 0
    vote_margin = (highest - runner_up) / len(valid)
    winners = [model for model in priority if model in valid and model_rows[model]["pred_action"] == winning_action]
    coordinate_candidates = [
        (model, model_rows[model]["pred_x"], model_rows[model]["pred_y"])
        for model in winners
        if not math.isnan(model_rows[model]["pred_x"]) and not math.isnan(model_rows[model]["pred_y"])
    ]
    if mode == "stage_a" or not coordinate_candidates:
        chosen = winners[0]
        x, y = model_rows[chosen]["pred_x"], model_rows[chosen]["pred_y"]
    else:
        points = [(x, y) for _, x, y in coordinate_candidates]
        point_weights = [weights[model] for model, _, _ in coordinate_candidates] if mode == "weighted_full" else None
        point = geometric_median(points, point_weights)
        x, y = point if point is not None else (float("nan"), float("nan"))
    if len(coordinate_candidates) < 2:
        dispersion = 0.0
    else:
        distances = [
            math.dist((left[1], left[2]), (right[1], right[2]))
            for index, left in enumerate(coordinate_candidates)
            for right in coordinate_candidates[index + 1:]
        ]
        dispersion = float(np.median(distances)) / 0.14
    parameter_candidates = [(model, model_rows[model]["pred_param"]) for model in winners]
    if mode in {"full", "weighted_full"}:
        parameter = text_medoid(parameter_candidates)
    else:
        parameter = model_rows[winners[0]]["pred_param"]
    reference = next(iter(model_rows.values()))
    return {
        "success": score_prediction(reference, winning_action, x, y, parameter),
        "vote_margin": vote_margin,
        "dispersion": dispersion,
        "winning_action": winning_action,
    }


def evaluate(identities, selected, pivot, priority, weights, mode):
    return {
        row_id: ensemble_row(pivot[row_id], selected, priority, weights, mode)
        for row_id in identities
    }


def select_subset(train_ids, models, pivot, priority, weights):
    best_single = priority[0]
    selected = [best_single]
    current = micro(pivot[row_id][best_single]["success"] for row_id in train_ids)
    remaining = set(models) - set(selected)
    trace = [{"models": selected.copy(), "dev_step_micro": current}]
    score_cache = {}
    while remaining:
        candidates = []
        for model in sorted(remaining):
            trial = selected + [model]
            cache_key = tuple(sorted(trial))
            if cache_key not in score_cache:
                result = evaluate(train_ids, trial, pivot, priority, weights, "weighted_full")
                score_cache[cache_key] = micro(item["success"] for item in result.values())
            candidates.append((score_cache[cache_key], model))
        score, model = max(candidates)
        if score <= current:
            break
        selected.append(model)
        remaining.remove(model)
        current = score
        trace.append({"models": selected.copy(), "dev_step_micro": current})
    return selected, trace


def logistic_confidence(train_signals, train_labels, test_signals):
    train_values = np.asarray(train_signals, dtype=np.float64)
    labels = np.asarray(train_labels, dtype=np.float64)
    means = train_values.mean(axis=0)
    scales = train_values.std(axis=0)
    scales[scales == 0] = 1
    normalized = (train_values - means) / scales
    design = np.column_stack([np.ones(len(normalized)), normalized])

    def loss(coefficients):
        logits = np.clip(design @ coefficients, -30, 30)
        probabilities = 1 / (1 + np.exp(-logits))
        return -np.sum(labels * np.log(probabilities + 1e-12) + (1 - labels) * np.log(1 - probabilities + 1e-12))

    fit = minimize(loss, np.zeros(design.shape[1]), method="BFGS")
    test_values = (np.asarray(test_signals, dtype=np.float64) - means) / scales
    logits = np.column_stack([np.ones(len(test_values)), test_values]) @ fit.x
    return (1 / (1 + np.exp(-np.clip(logits, -30, 30)))).tolist(), fit.x.tolist()


def run_pool(bench, setting):
    rows = load_rows(bench, setting)
    identities, models, pivot = pivot_rows(rows)
    pool = f"{bench}/{setting}"
    folds = []
    for test_fold in range(5):
        train_ids, test_ids = split_identities(pool, identities, pivot, test_fold)
        priority = model_priority(train_ids, models, pivot)
        weights = grounding_weights(train_ids, models, pivot)
        selected, selection_trace = select_subset(train_ids, models, pivot, priority, weights)
        best_single = priority[0]
        metrics = {
            "best_single": micro(pivot[row_id][best_single]["success"] for row_id in test_ids),
            "full_oracle": micro(any(pivot[row_id][model]["success"] for model in models) for row_id in test_ids),
        }
        outputs = {}
        for mode in ("stage_a", "stage_ab", "full", "weighted_full"):
            result = evaluate(test_ids, selected, pivot, priority, weights, mode)
            outputs[mode] = result
            metrics[mode] = micro(item["success"] for item in result.values())
        train_full = evaluate(train_ids, selected, pivot, priority, weights, "weighted_full")
        test_full = outputs["weighted_full"]
        train_signals = [(item["vote_margin"], -item["dispersion"]) for item in train_full.values()]
        test_signals = [(item["vote_margin"], -item["dispersion"]) for item in test_full.values()]
        logistic_scores, coefficients = logistic_confidence(
            train_signals, [item["success"] for item in train_full.values()], test_signals
        )
        labels = [item["success"] for item in test_full.values()]
        confidence = {
            "vote_margin_auroc": auc_roc(labels, [item["vote_margin"] for item in test_full.values()]),
            "negative_geometric_dispersion_auroc": auc_roc(labels, [-item["dispersion"] for item in test_full.values()]),
            "logistic_combination_auroc": auc_roc(labels, logistic_scores),
            "logistic_coefficients_intercept_margin_negative_dispersion": coefficients,
        }
        folds.append({
            "fold": test_fold, "train_rows": len(train_ids), "test_rows": len(test_ids),
            "best_single": best_single, "selected_models": selected,
            "selection_trace": selection_trace, "grounding_weights": weights,
            "metrics": metrics, "confidence": confidence,
        })
    aggregate = {}
    for metric in folds[0]["metrics"]:
        values = [fold["metrics"][metric] for fold in folds]
        aggregate[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "folds": values}
    for signal in ("vote_margin_auroc", "negative_geometric_dispersion_auroc", "logistic_combination_auroc"):
        values = [fold["confidence"][signal] for fold in folds]
        aggregate[signal] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "folds": values}
    aggregate["weighted_full_delta_over_best_single"] = (
        aggregate["weighted_full"]["mean"] - aggregate["best_single"]["mean"]
    )
    return {"models": models, "folds": folds, "aggregate": aggregate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "status": "PASS",
        "contract": {
            "folds": "shared deterministic group folds from folds.json",
            "subset_selection": "greedy weighted-full dev improvement only",
            "stage_a": "plurality action plus dev-best winning-model coordinate and parameter",
            "stage_ab": "plurality action plus unweighted geometric median plus dev-best parameter",
            "full": "stage_ab plus token-F1 text medoid",
            "weighted_full": "dev-grounding-weighted geometric median plus text medoid",
        },
        "pools": {},
    }
    for bench, setting in POOLS:
        result["pools"][f"{bench}/{setting}"] = run_pool(bench, setting)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({pool: {
        "best_single": value["aggregate"]["best_single"]["mean"],
        "weighted_full": value["aggregate"]["weighted_full"]["mean"],
        "delta": value["aggregate"]["weighted_full_delta_over_best_single"],
        "logistic_auroc": value["aggregate"]["logistic_combination_auroc"]["mean"],
    } for pool, value in result["pools"].items()}, indent=2))


if __name__ == "__main__":
    main()