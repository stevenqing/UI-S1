import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from utility_common import (
    ARMS,
    BENCHMARKS,
    OBJECTIVES,
    evaluation_rows,
    ids_for_folds,
    load_banks,
    load_cev,
    metadata,
    reliability_by_arm,
    training_matrix,
)
from behavior_policy import apply_policy, fit_final_policies, fit_inner_policies, load_cev_config


RUN_DIR = Path(__file__).resolve().parent
CONFIG_PATH = RUN_DIR / "configs/utility_prereg.yaml"
MODEL_IDS = ("H1", "H2", "H3")


def make_model(model_id, config):
    values = config["models"][model_id]
    return HistGradientBoostingRegressor(
        learning_rate=values["learning_rate"],
        max_iter=values["max_iter"],
        max_leaf_nodes=values["max_leaf_nodes"],
        min_samples_leaf=values["min_samples_leaf"],
        l2_regularization=values["l2_regularization"],
        loss="squared_error",
        early_stopping=False,
        random_state=config["models"]["random_state"],
    )


def fit_model(model_id, objective, config, banks, ids, reliability, policies, feature_mode="pair"):
    values, targets, weights, active = training_matrix(
        banks, ids, reliability, policies, objective, feature_mode
    )
    model = make_model(model_id, config)
    with threadpool_limits(limits=1):
        model.fit(values, targets, sample_weight=weights)
    return model, {"candidates": len(targets), "active_groups": active}


def predict(model, evaluation):
    output = {benchmark: {arm: {} for arm in ARMS} for benchmark in BENCHMARKS}
    for benchmark in BENCHMARKS:
        for arm in ARMS:
            for row_id, row in evaluation[benchmark][arm].items():
                scores = model.predict(row["features"])
                learned = int(np.argmax(scores))
                fallback = int(row["fallback_index"])
                output[benchmark][arm][row_id] = {
                    "scores": scores,
                    "labels": row["labels"],
                    "learned_index": learned,
                    "fallback_index": fallback,
                    "margin": float(scores[learned] - scores[fallback]),
                    "learned_score": float(scores[learned]),
                    "changed": learned != fallback,
                    "direct_success": bool(row["labels"][learned]),
                    "fallback_success": bool(row["labels"][fallback]),
                }
    return output


def threshold_candidates(predictions):
    margins = [
        row["margin"]
        for benchmark in predictions.values()
        for arm in benchmark.values()
        for row in arm.values()
        if row["changed"] and row["learned_score"] > 0 and row["margin"] > 0
    ]
    values = {float("inf"), 0.0}
    if margins:
        values.update(float(np.quantile(margins, quantile)) for quantile in np.linspace(0, 1, 11))
    return sorted(values)


def apply_threshold(rows, threshold):
    outputs = {}
    wins = losses = overrides = 0
    for row_id, row in rows.items():
        override = row["changed"] and row["learned_score"] > 0 and row["margin"] >= threshold
        success = row["direct_success"] if override else row["fallback_success"]
        outputs[row_id] = bool(success)
        wins += int(override and row["direct_success"] and not row["fallback_success"])
        losses += int(override and row["fallback_success"] and not row["direct_success"])
        overrides += int(override)
    return outputs, {"wins": wins, "losses": losses, "overrides": overrides, "override_rate": overrides / len(rows)}


def select_configuration(oof, config):
    candidates = []
    objective_order = config["safe_policy"]["tie_order"]["objective"]
    model_order = config["safe_policy"]["tie_order"]["model"]
    for objective in OBJECTIVES:
        for model_id in MODEL_IDS:
            predictions = oof[(objective, model_id)]
            for threshold in threshold_candidates(predictions):
                reports = {benchmark: {} for benchmark in BENCHMARKS}
                benchmark_means = []
                eligible = True
                for benchmark in BENCHMARKS:
                    cell_deltas = []
                    for arm in ARMS:
                        safe, override = apply_threshold(predictions[benchmark][arm], threshold)
                        fallback = [row["fallback_success"] for row in predictions[benchmark][arm].values()]
                        delta = float(np.mean(list(safe.values())) - np.mean(fallback))
                        reports[benchmark][arm] = {"point_delta": delta, **override}
                        cell_deltas.append(delta)
                        eligible &= delta >= -0.5 * config["mde"][benchmark] - 1e-15
                    equal_arm = float(np.mean(cell_deltas))
                    reports[benchmark]["equal_arm_mean"] = equal_arm
                    eligible &= equal_arm >= -0.25 * config["mde"][benchmark] - 1e-15
                    benchmark_means.append(equal_arm / config["mde"][benchmark])
                if eligible:
                    candidates.append({
                        "objective": objective,
                        "model_id": model_id,
                        "threshold": threshold,
                        "selection_objective": float(np.mean(benchmark_means)),
                        "OOF": reports,
                    })
    if not candidates:
        raise AssertionError("infinity threshold must be eligible")
    return max(candidates, key=lambda value: (
        value["selection_objective"],
        value["threshold"],
        -objective_order.index(value["objective"]),
        -model_order.index(value["model_id"]),
    ))


def validate_frozen_fallbacks(banks, policies, cev, outer_fold):
    report = {benchmark: {} for benchmark in BENCHMARKS}
    for benchmark in BENCHMARKS:
        for arm in ARMS:
            mismatches = []
            indices = {}
            for row_id, row in banks[arm][benchmark].items():
                if row.fold != outer_fold:
                    continue
                index = apply_policy(row, policies[benchmark][arm])
                indices[row_id] = index
                observed = bool(row.candidates[index].success)
                expected = bool(cev["outputs"][benchmark][arm]["CEV_A"][row_id])
                if observed != expected:
                    mismatches.append(row_id)
            if mismatches:
                raise ValueError(f"UR-K1 fallback mismatch: {benchmark}/{arm}/{len(mismatches)}")
            report[benchmark][arm] = {"rows": len(indices), "mismatches": 0}
    return report


def run(config, feature_mode="pair"):
    banks = load_banks()
    cev = load_cev()
    cev_config = load_cev_config()
    outputs = {
        benchmark: {arm: {"safe": {}, "direct": {}, "fallback": {}} for arm in ARMS}
        for benchmark in BENCHMARKS
    }
    folds = []
    for outer_fold in range(5):
        dev_folds = [fold for fold in range(5) if fold != outer_fold]
        oof = {
            (objective, model_id): {benchmark: {arm: {} for arm in ARMS} for benchmark in BENCHMARKS}
            for objective in OBJECTIVES for model_id in MODEL_IDS
        }
        inner_reports = []
        for holdout_fold in dev_folds:
            train_folds = [fold for fold in dev_folds if fold != holdout_fold]
            train_ids = ids_for_folds(banks, train_folds)
            holdout_ids = ids_for_folds(banks, [holdout_fold])
            reliability = reliability_by_arm(banks, train_ids)
            policies, policy_report = fit_inner_policies(banks, train_folds, holdout_fold, cev_config)
            evaluation = evaluation_rows(banks, holdout_ids, reliability, policies, feature_mode)
            report = {"holdout_fold": holdout_fold, "train_folds": train_folds, "behavior_policy": policy_report, "configurations": {}}
            for objective in OBJECTIVES:
                for model_id in MODEL_IDS:
                    model, training = fit_model(model_id, objective, config, banks, train_ids, reliability, policies, feature_mode)
                    predictions = predict(model, evaluation)
                    for benchmark in BENCHMARKS:
                        for arm in ARMS:
                            oof[(objective, model_id)][benchmark][arm].update(predictions[benchmark][arm])
                    report["configurations"][f"{objective}/{model_id}"] = training
            inner_reports.append(report)
        selected = select_configuration(oof, config)
        dev_ids = ids_for_folds(banks, dev_folds)
        test_ids = ids_for_folds(banks, [outer_fold])
        reliability = reliability_by_arm(banks, dev_ids)
        final_policies = fit_final_policies(banks, outer_fold, cev)
        fallback_validation = validate_frozen_fallbacks(banks, final_policies, cev, outer_fold)
        model, training = fit_model(selected["model_id"], selected["objective"], config, banks, dev_ids, reliability, final_policies, feature_mode)
        evaluation = evaluation_rows(banks, test_ids, reliability, final_policies, feature_mode)
        predictions = predict(model, evaluation)
        fold_report = {
            "outer_fold": outer_fold,
            "selected": selected,
            "training": training,
            "fallback_validation": fallback_validation,
            "inner_OOF": inner_reports,
            "test": {benchmark: {} for benchmark in BENCHMARKS},
        }
        for benchmark in BENCHMARKS:
            for arm in ARMS:
                safe, override = apply_threshold(predictions[benchmark][arm], selected["threshold"])
                direct = {row_id: row["direct_success"] for row_id, row in predictions[benchmark][arm].items()}
                fallback = {row_id: row["fallback_success"] for row_id, row in predictions[benchmark][arm].items()}
                frozen = cev["outputs"][benchmark][arm]["CEV_A"]
                if fallback != {row_id: frozen[row_id] for row_id in fallback}:
                    mismatch = sum(fallback[row_id] != frozen[row_id] for row_id in fallback)
                    raise ValueError(f"UR-K1 fallback mismatch: {benchmark}/{arm}/fold{outer_fold}/{mismatch}")
                outputs[benchmark][arm]["safe"].update(safe)
                outputs[benchmark][arm]["direct"].update(direct)
                outputs[benchmark][arm]["fallback"].update(fallback)
                fold_report["test"][benchmark][arm] = {
                    "safe_accuracy": float(np.mean(list(safe.values()))),
                    "direct_accuracy": float(np.mean(list(direct.values()))),
                    "fallback_accuracy": float(np.mean(list(fallback.values()))),
                    **override,
                }
        folds.append(fold_report)
        print(f"completed utility outer_fold={outer_fold} objective={selected['objective']} model={selected['model_id']} threshold={selected['threshold']}", flush=True)
    return {
        "feature_mode": feature_mode,
        "fallback_validation": [fold["fallback_validation"] for fold in folds],
        "folds": folds,
        "outputs": outputs,
        "accuracy": {
            benchmark: {
                arm: {method: float(np.mean(list(values.values()))) for method, values in methods.items()}
                for arm, methods in arms.items()
            }
            for benchmark, arms in outputs.items()
        },
        "metadata": metadata(banks),
    }


def main():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if config["status"] != "FROZEN_BEFORE_UTILITY_RESULTS":
        raise ValueError("utility protocol is not frozen")
    result = run(config)
    output = {
        "schema_version": 1,
        "status": "PASS_TRAINING_COMPLETE",
        "config": "configs/utility_prereg.yaml",
        "main": result,
    }
    (RUN_DIR / "utility_main.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accuracy": result["accuracy"], "fold_selections": [{"fold": fold["outer_fold"], **fold["selected"]} for fold in result["folds"]]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()