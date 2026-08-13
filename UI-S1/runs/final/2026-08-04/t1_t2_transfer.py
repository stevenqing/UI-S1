import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
COMPLEMENTARITY = ROOT / "runs/complementarity/2026-07-30"
ROWS_PATH = COMPLEMENTARITY / "rows.parquet"
CONFIG_PATH = RUN_DIR / "configs/t1_t2_pools.yaml"
sys.path.insert(0, str(COMPLEMENTARITY))
from common import load_rows, micro, pivot_rows, split_identities
from e1_ensemble import evaluate, grounding_weights, model_priority


SEED = 20260804
RESAMPLES = 10000


def cohen_kappa(left, right):
    left = np.asarray(left, dtype=np.bool_)
    right = np.asarray(right, dtype=np.bool_)
    observed = float(np.mean(left == right))
    left_rate = float(np.mean(left))
    right_rate = float(np.mean(right))
    expected = left_rate * right_rate + (1 - left_rate) * (1 - right_rate)
    return 1.0 if math.isclose(expected, 1.0) else (observed - expected) / (1 - expected)


def mean_failure_kappa(identities, models, pivot):
    values = []
    for index, left in enumerate(models):
        for right in models[index + 1:]:
            left_failure = [not pivot[row_id][left]["success"] for row_id in identities]
            right_failure = [not pivot[row_id][right]["success"] for row_id in identities]
            values.append(cohen_kappa(left_failure, right_failure))
    return float(np.mean(values)) if values else None


def paired_bootstrap(rows_by_id, folds, left, right):
    by_fold_group = defaultdict(lambda: defaultdict(list))
    for row_id, row in rows_by_id.items():
        by_fold_group[folds[row_id]][row["group_key"]].append(row_id)
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(RESAMPLES):
        selected = []
        for fold in sorted(by_fold_group):
            groups = sorted(by_fold_group[fold])
            for group in rng.choice(groups, size=len(groups), replace=True):
                selected.extend(by_fold_group[fold][group])
        values.append(float(np.mean([int(left[row_id]) - int(right[row_id]) for row_id in selected])))
    point = float(np.mean([int(left[row_id]) - int(right[row_id]) for row_id in rows_by_id]))
    return {
        "point_delta": point,
        "ci_99": [float(np.quantile(values, 0.005)), float(np.quantile(values, 0.995))],
        "p_one_sided_delta_le_zero": float((1 + sum(value <= 0 for value in values)) / (RESAMPLES + 1)),
        "resamples": RESAMPLES,
        "seed": SEED,
    }


def evaluate_fixed_pool(bench, setting, pool_name, models):
    rows = load_rows(bench, setting)
    identities, available_models, pivot = pivot_rows(rows)
    if not set(models).issubset(available_models):
        raise ValueError(f"{pool_name} missing models: {sorted(set(models)-set(available_models))}")
    outputs = {metric: {} for metric in ("pass_at_n", "stage_ab", "weighted_full")}
    fold_for_row = {}
    folds = []
    for test_fold in range(5):
        train_ids, test_ids = split_identities(f"{bench}/{setting}", identities, pivot, test_fold)
        priority = model_priority(train_ids, models, pivot)
        weights = grounding_weights(train_ids, models, pivot)
        stage_ab = evaluate(test_ids, models, pivot, priority, weights, "stage_ab")
        weighted = evaluate(test_ids, models, pivot, priority, weights, "weighted_full")
        for row_id in test_ids:
            fold_for_row[row_id] = test_fold
            outputs["pass_at_n"][row_id] = any(pivot[row_id][model]["success"] for model in models)
            outputs["stage_ab"][row_id] = bool(stage_ab[row_id]["success"])
            outputs["weighted_full"][row_id] = bool(weighted[row_id]["success"])
        folds.append({
            "fold": test_fold,
            "train_rows": len(train_ids),
            "test_rows": len(test_ids),
            "dev_mean_pairwise_failure_kappa": mean_failure_kappa(train_ids, models, pivot),
            "dev_priority": priority,
            "test_accuracy": {metric: micro(values[row_id] for row_id in test_ids) for metric, values in outputs.items()},
        })
    model_accuracy = {model: micro(pivot[row_id][model]["success"] for row_id in identities) for model in models}
    reference_rows = {row_id: next(iter(pivot[row_id].values())) for row_id in identities}
    return {
        "pool": pool_name,
        "models": models,
        "rows": len(identities),
        "member_accuracy": model_accuracy,
        "mean_member_accuracy": float(np.mean(list(model_accuracy.values()))),
        "strongest_member_accuracy": max(model_accuracy.values()),
        "folds": folds,
        "accuracy": {metric: micro(values.values()) for metric, values in outputs.items()},
        "outputs": outputs,
        "fold_for_row": fold_for_row,
        "reference_rows": reference_rows,
    }


def compare(left, right):
    if left["reference_rows"].keys() != right["reference_rows"].keys():
        raise ValueError("T1/T2 comparison identity mismatch")
    comparisons = {}
    for metric in ("pass_at_n", "stage_ab", "weighted_full"):
        result = paired_bootstrap(
            left["reference_rows"], left["fold_for_row"], left["outputs"][metric], right["outputs"][metric]
        )
        result.update({
            "metric": metric,
            "left_accuracy": left["accuracy"][metric],
            "right_accuracy": right["accuracy"][metric],
        })
        comparisons[metric] = result
    return comparisons


def compact(pool):
    return {key: value for key, value in pool.items() if key not in {"outputs", "fold_for_row", "reference_rows"}}


def preflight():
    return {
        "status": "READY" if ROWS_PATH.is_file() else "BLOCKED_MISSING_ROWS_PARQUET",
        "required": str(ROWS_PATH.relative_to(ROOT)),
    }


def run(output_path):
    check = preflight()
    if check["status"] != "READY":
        raise FileNotFoundError(json.dumps(check, sort_keys=True))
    config = yaml.safe_load(CONFIG_PATH.read_text())
    results = {}
    for section in ("T1_mind2web", "T2_androidcontrol"):
        spec = config[section]
        results[section] = {
            pool_name: evaluate_fixed_pool(spec["benchmark"], spec["setting"], pool_name, pool["models"])
            for pool_name, pool in spec["pools"].items()
        }
    comparisons = {
        "T1_M_cross_3_vs_M_same_3": compare(results["T1_mind2web"]["M_cross_3"], results["T1_mind2web"]["M_same_3"]),
        "T2_A_cross_2_vs_A_same_2_agile": compare(results["T2_androidcontrol"]["A_cross_2"], results["T2_androidcontrol"]["A_same_2_agile"]),
        "T2_A_cross_2_vs_A_same_2_gui": compare(results["T2_androidcontrol"]["A_cross_2"], results["T2_androidcontrol"]["A_same_2_gui"]),
    }
    result = {
        "schema_version": 1,
        "status": "PASS",
        "pools": {section: {name: compact(pool) for name, pool in pools.items()} for section, pools in results.items()},
        "comparisons": comparisons,
        "android_quality_confound": config["T2_androidcontrol"]["confound"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RUN_DIR / "t1_t2_transfer.json")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    result = preflight() if args.preflight else run(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()