import argparse
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from scipy.stats import spearmanr

from aggregators import (
    best_single,
    pka_continuous,
    pka_medoid,
    plurality_then_density,
    plurality_then_median,
)
from pka import Prediction
from scoring import ACTION_TO_ID, GROUNDING_ACTIONS, SIMPLE_ACTIONS, TEXT_ACTIONS, text_f1, token_f1


RUN_DIR = Path(__file__).resolve().parent
ROWS_PATH = RUN_DIR / "rows.parquet"
FOLDS_PATH = RUN_DIR.parents[1] / "complementarity/2026-07-30/folds.json"
STRATA_PATH = RUN_DIR / "configs/strata.yaml"
BANDS_PATH = RUN_DIR / "configs/bands.yaml"
POOLS = (("androidcontrol", "low"), ("androidcontrol", "high"), ("mind2web", "visual"))
KAPPA_SEED = 20260730
KAPPA_PERMUTATIONS = 1000


def load_pool(bench: str, setting: str):
    table = pq.read_table(ROWS_PATH, filters=[("bench", "=", bench), ("setting", "=", setting)])
    table = table.filter(pc.invert(table["quarantine"]))
    rows = table.to_pylist()
    models = sorted({row["model"] for row in rows})
    pivot = defaultdict(dict)
    for row in rows:
        if row["model"] in pivot[row["row_id"]]:
            raise ValueError(f"duplicate row/model: {row['row_id']}/{row['model']}")
        pivot[row["row_id"]][row["model"]] = row
    if any(set(model_rows) != set(models) for model_rows in pivot.values()):
        raise ValueError(f"model coverage mismatch: {bench}/{setting}")
    return sorted(pivot), models, dict(pivot)


def prediction_from_row(row: dict) -> Prediction:
    x = None if row["pred_x"] is None or math.isnan(row["pred_x"]) else row["pred_x"]
    y = None if row["pred_y"] is None or math.isnan(row["pred_y"]) else row["pred_y"]
    return Prediction(
        action=row["pred_action"], x=x, y=y, parameter=row["pred_param"],
        source=row["model"], parse_ok=row["parse_ok"],
    )


def score_prediction(reference: dict, prediction: Prediction | None) -> bool:
    if prediction is None or not prediction.parse_ok:
        return False
    if reference["bench"] == "androidcontrol":
        if prediction.action != reference["gt_action"]:
            return False
        if reference["gt_action"] in GROUNDING_ACTIONS:
            return prediction.coordinate is not None and math.dist(
                prediction.coordinate, (reference["gt_x"], reference["gt_y"])
            ) < 0.14
        if reference["gt_action"] in TEXT_ACTIONS:
            return text_f1(prediction.parameter, reference["gt_param"]) >= 0.5
        if reference["gt_action"] in SIMPLE_ACTIONS:
            return True
        raise ValueError(f"unknown AndroidControl action: {reference['gt_action']}")
    if reference["bench"] != "mind2web":
        raise ValueError(reference["bench"])
    if prediction.action not in ACTION_TO_ID or prediction.coordinate is None:
        return False
    x, y = prediction.coordinate
    x0, y0, x1, y1 = reference["gt_bbox"]
    element = x0 <= x <= x1 and y0 <= y <= y1
    predicted_operation = str(ACTION_TO_ID[prediction.action])
    if prediction.action in {"TYPE", "SELECT"}:
        predicted_operation += " " + prediction.parameter.lower()
    expected_operation = str(ACTION_TO_ID[reference["gt_action"]])
    if reference["gt_action"] in {"TYPE", "SELECT"}:
        expected_operation += " " + reference["gt_param"].lower()
    return element and token_f1(predicted_operation, expected_operation) == 1.0


def fold_map(pool: str) -> dict[str, int]:
    folds = json.loads(FOLDS_PATH.read_text())
    return folds["pools"][pool]["group_to_fold"]


def split_rows(identities, pivot, mapping, test_fold):
    dev, test = [], []
    for row_id in identities:
        reference = next(iter(pivot[row_id].values()))
        (test if mapping[reference["group_key"]] == test_fold else dev).append(row_id)
    return dev, test


def model_step_sr(identities, model, pivot):
    return sum(pivot[row_id][model]["success"] for row_id in identities) / len(identities)


def parse_rate(identities, model, pivot):
    return sum(pivot[row_id][model]["parse_ok"] for row_id in identities) / len(identities)


def deployable_models(identities, models, pivot):
    bands = yaml.safe_load(BANDS_PATH.read_text())["deployable_subset"]
    parse_lower = bands["parse_rate"]["lower_inclusive"]
    parse_upper = bands["parse_rate"]["upper_inclusive"]
    step_lower = bands["step_success_rate"]["lower_inclusive"]
    step_upper = bands["step_success_rate"]["upper_inclusive"]
    selected = [
        model for model in models
        if parse_lower <= parse_rate(identities, model, pivot) <= parse_upper
        and step_lower <= model_step_sr(identities, model, pivot) <= step_upper
    ]
    if not selected:
        raise ValueError("deployability band selected no models")
    return selected


def dev_priority(dev_ids, models, pivot):
    return sorted(models, key=lambda model: (-model_step_sr(dev_ids, model, pivot), model))


def grounding_weights(dev_ids, models, pivot):
    output = {}
    for model in models:
        values = []
        for row_id in dev_ids:
            row = pivot[row_id][model]
            if row["bench"] == "androidcontrol" and row["gt_action"] in GROUNDING_ACTIONS:
                values.append(row["ground_dist"] < 0.14)
            elif row["bench"] == "mind2web" and row["bbox_dist"] is not None and not math.isnan(row["bbox_dist"]):
                values.append(row["bbox_dist"] == 0.0)
        output[model] = sum(values) / len(values) if values else 1.0
    return output


def aggregate_row(method, bench, model_rows, models, priority, weights):
    predictions = [prediction_from_row(model_rows[model]) for model in models]
    if method == "A1_plurality_median":
        return plurality_then_median(bench, predictions, priority, weights).prediction
    if method == "A2_plurality_density":
        return plurality_then_density(bench, predictions, priority).prediction
    if method == "A3_pka_joint":
        return pka_medoid(bench, predictions).prediction
    if method == "A4_pka_continuous":
        return pka_continuous(bench, predictions).prediction
    raise ValueError(method)


def evaluate_scope(bench, setting, identities, models, pivot, limit=None):
    pool = f"{bench}/{setting}"
    mapping = fold_map(pool)
    methods = ("A1_plurality_median", "A2_plurality_density", "A3_pka_joint", "A4_pka_continuous")
    folds = []
    row_outputs = {method: {} for method in methods}
    heldout_outputs = {}
    insample_outputs = {}
    for test_fold in range(5):
        dev_ids, test_ids = split_rows(identities, pivot, mapping, test_fold)
        if limit is not None:
            test_ids = test_ids[:limit]
        priority = dev_priority(dev_ids, models, pivot)
        weights = grounding_weights(dev_ids, models, pivot)
        heldout_model = priority[0]
        insample_model = max(models, key=lambda model: (sum(pivot[row_id][model]["success"] for row_id in test_ids), model))
        metrics = {
            "A0_heldout_best": model_step_sr(test_ids, heldout_model, pivot),
            "A0_insample_best": model_step_sr(test_ids, insample_model, pivot),
            "oracle": sum(any(pivot[row_id][model]["success"] for model in models) for row_id in test_ids) / len(test_ids),
        }
        for row_id in test_ids:
            heldout_outputs[row_id] = bool(pivot[row_id][heldout_model]["success"])
            insample_outputs[row_id] = bool(pivot[row_id][insample_model]["success"])
        for method in methods:
            successes = []
            for row_id in test_ids:
                reference = next(iter(pivot[row_id].values()))
                prediction = aggregate_row(method, bench, pivot[row_id], models, priority, weights)
                success = score_prediction(reference, prediction)
                successes.append(success)
                row_outputs[method][row_id] = success
            metrics[method] = sum(successes) / len(successes)
        folds.append({
            "fold": test_fold, "dev_rows": len(dev_ids), "test_rows": len(test_ids),
            "heldout_best_model": heldout_model, "insample_best_model": insample_model,
            "priority": priority, "grounding_weights": weights, "metrics": metrics,
        })
    aggregate = {}
    for metric in folds[0]["metrics"]:
        numerator = sum(fold["metrics"][metric] * fold["test_rows"] for fold in folds)
        denominator = sum(fold["test_rows"] for fold in folds)
        aggregate[metric] = numerator / denominator
    return {
        "models": models,
        "folds": folds,
        "aggregate": aggregate,
    }, {**row_outputs, "A0_heldout_best": heldout_outputs, "A0_insample_best": insample_outputs}


def cohen_kappa(left, right):
    left = np.asarray(left, dtype=np.int8)
    right = np.asarray(right, dtype=np.int8)
    observed = np.mean(left == right)
    left_rate, right_rate = left.mean(), right.mean()
    expected = left_rate * right_rate + (1 - left_rate) * (1 - right_rate)
    return float((observed - expected) / (1 - expected)) if expected < 1 else 1.0


def pairwise_kappa(identities, models, pivot, permutations=KAPPA_PERMUTATIONS):
    rng = np.random.default_rng(KAPPA_SEED)
    output = []
    for pair_index, (left, right) in enumerate(combinations(models, 2)):
        left_failure = np.asarray([not pivot[row_id][left]["success"] for row_id in identities], dtype=np.int8)
        right_failure = np.asarray([not pivot[row_id][right]["success"] for row_id in identities], dtype=np.int8)
        observed = cohen_kappa(left_failure, right_failure)
        null = np.asarray([cohen_kappa(left_failure, rng.permutation(right_failure)) for _ in range(permutations)])
        output.append({
            "left": left, "right": right, "rows": len(identities), "observed_kappa": observed,
            "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "p_greater_equal": float((1 + np.count_nonzero(null >= observed)) / (permutations + 1)),
            "permutations": permutations,
        })
    return output


def dominant_error(model_rows):
    if any(row["success"] for row in model_rows.values()):
        return None
    counts = Counter(row["err_label"] for row in model_rows.values())
    highest = max(counts.values())
    winners = [label for label, count in counts.items() if count == highest]
    return winners[0] if len(winners) == 1 and highest > len(model_rows) / 2 else None


def stratum_ids(stratum_id, pools):
    selected = []
    if stratum_id.startswith("mind2web"):
        identities, _, pivot = pools[("mind2web", "visual")]
        allowed = {"CLICK"} if stratum_id == "mind2web_click" else {"SELECT", "TYPE"}
        for row_id in identities:
            if next(iter(pivot[row_id].values()))["gt_action"] in allowed:
                selected.append(("mind2web/visual", row_id))
        return selected
    target = "grounding_miss" if stratum_id == "androidcontrol_grounding_dominant" else "action_mismatch"
    for setting in ("low", "high"):
        identities, _, pivot = pools[("androidcontrol", setting)]
        for row_id in identities:
            if dominant_error(pivot[row_id]) == target:
                selected.append((f"androidcontrol/{setting}", row_id))
    return selected


def build_strata(pools, scope_outputs, scope_models):
    config = yaml.safe_load(STRATA_PATH.read_text())
    output = {}
    collisions = []
    gains = []
    for definition in config["strata"]:
        stratum_id = definition["id"]
        members = stratum_ids(stratum_id, pools)
        by_pool = defaultdict(list)
        for pool, row_id in members:
            by_pool[pool].append(row_id)
        kappa_values = []
        method_counts = Counter()
        denominator = 0
        for pool, row_ids in by_pool.items():
            bench, setting = pool.split("/", 1)
            identities, _, pivot = pools[(bench, setting)]
            models = scope_models[pool]
            kappas = pairwise_kappa(row_ids, models, pivot)
            kappa_values.extend(item["observed_kappa"] for item in kappas)
            denominator += len(row_ids)
            for method, values in scope_outputs[pool].items():
                method_counts[method] += sum(values[row_id] for row_id in row_ids)
        rates = {method: count / denominator for method, count in method_counts.items()}
        gain = rates["A3_pka_joint"] - rates["A0_heldout_best"]
        collision = float(np.mean(kappa_values)) if kappa_values else None
        output[stratum_id] = {
            "rows": denominator, "mean_pairwise_failure_kappa": collision,
            "step_sr": rates, "a3_gain_over_heldout_best": gain,
        }
        collisions.append(collision)
        gains.append(gain)
    correlation = spearmanr(collisions, gains)
    return {
        "status": "PASS",
        "contract": config,
        "strata": output,
        "p1_reverse_order_test": {
            "spearman_collision_vs_gain": float(correlation.statistic),
            "p_value": float(correlation.pvalue),
            "prediction_satisfied": bool(correlation.statistic < 0),
        },
    }


def cross_lineage_kappa(pools):
    result = {"status": "PASS", "seed": KAPPA_SEED, "permutations": KAPPA_PERMUTATIONS, "pools": {}}
    m2w_models = ["tongui-7b", "cogagent-18b", "ui-tars-72b", "seeclick-9.6b"]
    identities, _, pivot = pools[("mind2web", "visual")]
    result["pools"]["mind2web/visual"] = pairwise_kappa(identities, m2w_models, pivot)
    for setting in ("low", "high"):
        identities, _, pivot = pools[("androidcontrol", setting)]
        models = ["ui-agile-3b", "ui-agile-7b", "gui-r1-3b", "gui-r1-7b", "ui-r1-e-3b"]
        family = {
            "ui-agile-3b": "ui-agile", "ui-agile-7b": "ui-agile",
            "gui-r1-3b": "gui-r1", "gui-r1-7b": "gui-r1", "ui-r1-e-3b": "ui-r1-e",
        }
        all_pairs = pairwise_kappa(identities, models, pivot)
        result["pools"][f"androidcontrol/{setting}"] = [
            item for item in all_pairs if family[item["left"]] != family[item["right"]]
        ]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregators", type=Path, required=True)
    parser.add_argument("--strata", type=Path, required=True)
    parser.add_argument("--kappa", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    pools = {}
    for bench, setting in POOLS:
        pools[(bench, setting)] = load_pool(bench, setting)

    result = {
        "status": "PASS",
        "primary_scope": "deployable",
        "scopes": {"full": {}, "deployable": {}},
        "contract": {
            "folds": str(FOLDS_PATH.relative_to(RUN_DIR.parents[2])),
            "uniform_pka_weights": True,
            "a1_dev_grounding_weights": True,
            "test_tuning": False,
        },
    }
    outputs = {"full": {}, "deployable": {}}
    scope_models = {"full": {}, "deployable": {}}
    for bench, setting in POOLS:
        identities, models, pivot = pools[(bench, setting)]
        pool = f"{bench}/{setting}"
        scopes = {"full": models, "deployable": deployable_models(identities, models, pivot)}
        for scope, selected_models in scopes.items():
            summary, row_outputs = evaluate_scope(
                bench, setting, identities, selected_models, pivot, args.limit
            )
            result["scopes"][scope][pool] = summary
            outputs[scope][pool] = row_outputs
            scope_models[scope][pool] = selected_models
    args.aggregators.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    if args.limit is not None:
        smoke = {"status": "SMOKE_PASS", "limit_per_fold": args.limit}
        args.strata.write_text(json.dumps(smoke, indent=2) + "\n")
        args.kappa.write_text(json.dumps(smoke, indent=2) + "\n")
        print(json.dumps({"status": result["status"], "mode": "smoke"}, indent=2))
        return

    strata = build_strata(pools, outputs["deployable"], scope_models["deployable"])
    kappa = cross_lineage_kappa(pools)
    args.strata.write_text(json.dumps(strata, indent=2, sort_keys=True) + "\n")
    args.kappa.write_text(json.dumps(kappa, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "deployable": {pool: values["aggregate"] for pool, values in result["scopes"]["deployable"].items()},
        "p1": strata["p1_reverse_order_test"],
    }, indent=2))


if __name__ == "__main__":
    main()