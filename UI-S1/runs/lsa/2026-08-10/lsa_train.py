import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from lsa_common import (
    evaluation_rows,
    feature_names,
    load_rows,
    reliability_statistics,
    training_matrix,
)


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CONFIG_PATH = RUN_DIR / "configs/lsa_prereg.yaml"
CEV_RESULT = ROOT / "runs/cev/2026-08-09/cev_main.json"
BENCHMARKS = ("mind2web", "screenspot_pro")
MODEL_IDS = ("H1", "H2", "H3", "H4")


def make_model(model_id, config):
    values = config["models"][model_id]
    return HistGradientBoostingClassifier(
        learning_rate=values["learning_rate"],
        max_iter=values["max_iter"],
        max_leaf_nodes=values["max_leaf_nodes"],
        min_samples_leaf=values["min_samples_leaf"],
        l2_regularization=values["l2_regularization"],
        early_stopping=False,
        random_state=config["models"]["random_state"],
    )


def ids_for_folds(banks, benchmarks, included_folds):
    return {
        benchmark: [row_id for row_id, row in banks[benchmark].items() if row.fold in included_folds]
        for benchmark in benchmarks
    }


def reliability_for_ids(banks, ids_by_benchmark):
    return {
        benchmark: reliability_statistics(banks[benchmark], ids)
        for benchmark, ids in ids_by_benchmark.items()
    }


def fit_estimator(model_id, config, banks, ids_by_benchmark, feature_indices=None):
    reliability = reliability_for_ids(banks, ids_by_benchmark)
    values, labels, weights, _, mixed = training_matrix(
        banks, ids_by_benchmark, reliability, feature_indices
    )
    model = make_model(model_id, config)
    with threadpool_limits(limits=1):
        model.fit(values, labels, sample_weight=weights)
    return model, reliability, {"candidate_rows": len(labels), "mixed_rows": mixed}


def predict_rows(model, evaluation):
    output = {}
    for benchmark, rows in evaluation.items():
        output[benchmark] = {}
        for row_id, row in rows.items():
            probabilities = model.predict_proba(row["features"])[:, 1]
            learned_index = int(np.argmax(probabilities))
            fallback_index = int(row["fallback_index"])
            output[benchmark][row_id] = {
                "probabilities": probabilities,
                "labels": row["labels"],
                "learned_index": learned_index,
                "fallback_index": fallback_index,
                "margin": float(probabilities[learned_index] - probabilities[fallback_index]),
                "changed": learned_index != fallback_index,
                "direct_success": bool(row["labels"][learned_index]),
                "fallback_success": bool(row["labels"][fallback_index]),
            }
    return output


def threshold_candidates(predictions):
    margins = [
        row["margin"]
        for benchmark in predictions.values()
        for row in benchmark.values()
        if row["changed"] and row["margin"] > 0
    ]
    values = {float("inf"), 0.0}
    if margins:
        values.update(float(np.quantile(margins, quantile)) for quantile in np.linspace(0, 1, 11))
    return sorted(values)


def apply_threshold(rows, threshold):
    output = {}
    overrides = 0
    wins = 0
    losses = 0
    for row_id, row in rows.items():
        override = row["changed"] and row["margin"] >= threshold
        success = row["direct_success"] if override else row["fallback_success"]
        output[row_id] = bool(success)
        overrides += int(override)
        wins += int(override and row["direct_success"] and not row["fallback_success"])
        losses += int(override and row["fallback_success"] and not row["direct_success"])
    return output, {"overrides": overrides, "override_rate": overrides / len(rows), "wins": wins, "losses": losses}


def select_model_threshold(oof_by_model, config, benchmarks):
    candidates = []
    for model_rank, model_id in enumerate(MODEL_IDS):
        predictions = oof_by_model[model_id]
        for threshold in threshold_candidates(predictions):
            reports = {}
            eligible = True
            standardized = []
            for benchmark in benchmarks:
                outputs, override = apply_threshold(predictions[benchmark], threshold)
                baseline = [row["fallback_success"] for row in predictions[benchmark].values()]
                values = list(outputs.values())
                delta = float(np.mean(values) - np.mean(baseline))
                reports[benchmark] = {
                    "accuracy": float(np.mean(values)),
                    "fallback_accuracy": float(np.mean(baseline)),
                    "point_delta": delta,
                    **override,
                }
                eligible &= delta >= -1e-15
                standardized.append(delta / config["benchmarks"][benchmark]["mde"])
            if eligible:
                candidates.append({
                    "model_id": model_id,
                    "model_rank": model_rank,
                    "threshold": threshold,
                    "objective": float(np.mean(standardized)),
                    "benchmarks": reports,
                })
    if not candidates:
        raise AssertionError("infinity threshold must be eligible")
    return max(
        candidates,
        key=lambda value: (
            value["objective"],
            value["threshold"],
            -value["model_rank"],
        ),
    )


def run_variant(config, banks, benchmarks, feature_indices=None):
    outputs = {
        benchmark: {"safe": {}, "direct": {}, "fallback": {}}
        for benchmark in benchmarks
    }
    folds = []
    for outer_fold in range(5):
        development_folds = [fold for fold in range(5) if fold != outer_fold]
        oof_by_model = {
            model_id: {benchmark: {} for benchmark in benchmarks}
            for model_id in MODEL_IDS
        }
        for holdout_fold in development_folds:
            train_folds = [fold for fold in development_folds if fold != holdout_fold]
            train_ids = ids_for_folds(banks, benchmarks, train_folds)
            holdout_ids = ids_for_folds(banks, benchmarks, [holdout_fold])
            holdout_reliability = reliability_for_ids(banks, train_ids)
            evaluation = evaluation_rows(banks, holdout_ids, holdout_reliability, feature_indices)
            for model_id in MODEL_IDS:
                model, _, _ = fit_estimator(model_id, config, banks, train_ids, feature_indices)
                predictions = predict_rows(model, evaluation)
                for benchmark in benchmarks:
                    oof_by_model[model_id][benchmark].update(predictions[benchmark])
        selected = select_model_threshold(oof_by_model, config, benchmarks)
        development_ids = ids_for_folds(banks, benchmarks, development_folds)
        test_ids = ids_for_folds(banks, benchmarks, [outer_fold])
        model, reliability, training = fit_estimator(selected["model_id"], config, banks, development_ids, feature_indices)
        test_evaluation = evaluation_rows(banks, test_ids, reliability, feature_indices)
        predictions = predict_rows(model, test_evaluation)
        fold_report = {
            "outer_fold": outer_fold,
            "selected": selected,
            "training": training,
            "test": {},
        }
        for benchmark in benchmarks:
            safe, override = apply_threshold(predictions[benchmark], selected["threshold"])
            direct = {row_id: row["direct_success"] for row_id, row in predictions[benchmark].items()}
            fallback = {row_id: row["fallback_success"] for row_id, row in predictions[benchmark].items()}
            outputs[benchmark]["safe"].update(safe)
            outputs[benchmark]["direct"].update(direct)
            outputs[benchmark]["fallback"].update(fallback)
            fold_report["test"][benchmark] = {
                "safe_accuracy": float(np.mean(list(safe.values()))),
                "direct_accuracy": float(np.mean(list(direct.values()))),
                "fallback_accuracy": float(np.mean(list(fallback.values()))),
                "candidate_auroc": candidate_auc(predictions[benchmark]),
                "override": override,
            }
        folds.append(fold_report)
        print(f"completed LSA variant benchmarks={','.join(benchmarks)} outer_fold={outer_fold}", flush=True)
    return {
        "benchmarks": list(benchmarks),
        "feature_indices": feature_indices,
        "folds": folds,
        "outputs": outputs,
        "accuracy": {
            benchmark: {
                method: float(np.mean(list(values.values())))
                for method, values in methods.items()
            }
            for benchmark, methods in outputs.items()
        },
    }


def candidate_auc(predictions):
    labels = []
    scores = []
    for row in predictions.values():
        labels.extend(row["labels"].tolist())
        scores.extend(row["probabilities"].tolist())
    return float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else None


def feature_blocks(names):
    return {
        "reliability": [index for index, name in enumerate(names) if name == "source_reliability"],
        "action": [index for index, name in enumerate(names) if "action" in name],
        "geometry": [index for index, name in enumerate(names) if "coordinate" in name or "lineage_support" in name],
        "parameter": [index for index, name in enumerate(names) if "parameter" in name],
        "generic": [index for index, name in enumerate(names) if not any(token in name for token in ("action", "coordinate", "lineage_support", "parameter", "source_reliability"))],
    }


def permutation_auc_drop(model, evaluation, seed):
    names = feature_names()
    blocks = feature_blocks(names)
    arrays = []
    labels = []
    for row in evaluation.values():
        arrays.append(row["features"])
        labels.append(row["labels"])
    values = np.concatenate(arrays)
    y = np.concatenate(labels)
    baseline = float(roc_auc_score(y, model.predict_proba(values)[:, 1]))
    rng = np.random.default_rng(seed)
    report = {}
    for name, indices in blocks.items():
        permuted = values.copy()
        order = rng.permutation(len(values))
        permuted[:, indices] = permuted[order][:, indices]
        report[name] = baseline - float(roc_auc_score(y, model.predict_proba(permuted)[:, 1]))
    return {"baseline_candidate_auroc": baseline, "permutation_auroc_drop": report}


def run_pooled(config, banks, feature_indices=None):
    outputs = {
        benchmark: {"safe": {}, "direct": {}, "fallback": {}}
        for benchmark in BENCHMARKS
    }
    folds = []
    for outer_fold in range(5):
        development_folds = [fold for fold in range(5) if fold != outer_fold]
        oof_by_model = {
            model_id: {benchmark: {} for benchmark in BENCHMARKS}
            for model_id in MODEL_IDS
        }
        inner_reports = []
        for holdout_fold in development_folds:
            train_folds = [fold for fold in development_folds if fold != holdout_fold]
            train_ids = ids_for_folds(banks, BENCHMARKS, train_folds)
            holdout_ids = ids_for_folds(banks, BENCHMARKS, [holdout_fold])
            holdout_reliability = reliability_for_ids(banks, train_ids)
            evaluation = evaluation_rows(banks, holdout_ids, holdout_reliability, feature_indices)
            inner_report = {"holdout_fold": holdout_fold, "train_folds": train_folds, "models": {}}
            for model_id in MODEL_IDS:
                model, _, training = fit_estimator(model_id, config, banks, train_ids, feature_indices)
                predictions = predict_rows(model, evaluation)
                for benchmark in BENCHMARKS:
                    oof_by_model[model_id][benchmark].update(predictions[benchmark])
                inner_report["models"][model_id] = {
                    "training": training,
                    "candidate_auroc": {
                        benchmark: candidate_auc(predictions[benchmark]) for benchmark in BENCHMARKS
                    },
                }
            inner_reports.append(inner_report)
        selected = select_model_threshold(oof_by_model, config, BENCHMARKS)
        development_ids = ids_for_folds(banks, BENCHMARKS, development_folds)
        test_ids = ids_for_folds(banks, BENCHMARKS, [outer_fold])
        model, reliability, training = fit_estimator(selected["model_id"], config, banks, development_ids, feature_indices)
        test_evaluation = evaluation_rows(banks, test_ids, reliability, feature_indices)
        predictions = predict_rows(model, test_evaluation)
        fold_report = {
            "outer_fold": outer_fold,
            "development_folds": development_folds,
            "selected": selected,
            "training": training,
            "inner_OOF": inner_reports,
            "test": {},
        }
        for benchmark in BENCHMARKS:
            safe, override = apply_threshold(predictions[benchmark], selected["threshold"])
            direct = {row_id: row["direct_success"] for row_id, row in predictions[benchmark].items()}
            fallback = {row_id: row["fallback_success"] for row_id, row in predictions[benchmark].items()}
            outputs[benchmark]["safe"].update(safe)
            outputs[benchmark]["direct"].update(direct)
            outputs[benchmark]["fallback"].update(fallback)
            fold_report["test"][benchmark] = {
                "rows": len(safe),
                "safe_accuracy": float(np.mean(list(safe.values()))),
                "direct_accuracy": float(np.mean(list(direct.values()))),
                "fallback_accuracy": float(np.mean(list(fallback.values()))),
                "candidate_auroc": candidate_auc(predictions[benchmark]),
                "override": override,
                "feature_importance": permutation_auc_drop(model, test_evaluation[benchmark], 20260810 + outer_fold),
            }
        folds.append(fold_report)
        print(f"completed LSA pooled outer_fold={outer_fold} model={selected['model_id']} threshold={selected['threshold']}", flush=True)
    return {
        "folds": folds,
        "outputs": outputs,
        "accuracy": {
            benchmark: {
                method: float(np.mean(list(values.values())))
                for method, values in methods.items()
            }
            for benchmark, methods in outputs.items()
        },
    }


def main():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if config["status"] != "FROZEN_BEFORE_LSA_RESULTS":
        raise ValueError("LSA preregistration is not frozen")
    oracle = json.loads((RUN_DIR / "lsa_oracle.json").read_text())
    if oracle["LSA_K1"]:
        raise ValueError("LSA-K1 blocks training")
    banks = load_rows()
    result = run_pooled(config, banks)
    cev = json.loads(CEV_RESULT.read_text())
    for benchmark in BENCHMARKS:
        expected = cev["outputs"][benchmark]["C_uni"]["CEV_A"]
        if result["outputs"][benchmark]["fallback"] != expected:
            mismatches = sum(result["outputs"][benchmark]["fallback"][row_id] != expected[row_id] for row_id in expected)
            raise ValueError(f"frozen fallback mismatch: {benchmark}/{mismatches}")
    output = {
        "schema_version": 1,
        "status": "PASS_TRAINING_COMPLETE",
        "config": "configs/lsa_prereg.yaml",
        "feature_names": feature_names(),
        "pooled": result,
    }
    (RUN_DIR / "lsa_main.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accuracy": result["accuracy"], "fold_selections": [{"fold": fold["outer_fold"], **fold["selected"]} for fold in result["folds"]]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()