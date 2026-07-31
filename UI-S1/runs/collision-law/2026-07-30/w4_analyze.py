import argparse
import contextlib
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

from scipy.stats import kendalltau

from aggregators import pka_continuous, pka_medoid, plurality_then_density, plurality_then_median
from pka import Prediction
from w4_curated import DATA_CONFIG, DATA_DIR, IMAGE_ROOT, load_official_utils, read_jsonl


RUN_DIR = Path(__file__).resolve().parent
MODELS = ("ui-agile-3b", "ui-agile-7b", "ui-r1-e-3b", "gui-r1-3b", "gui-r1-7b")
RADII = tuple(round(0.06 + 0.02 * index, 2) for index in range(13))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def assign_folds(rows, n_splits=5):
    group_counts = Counter(str(Path(row["image"]).parent) for row in rows)
    loads = [0] * n_splits
    mapping = {}
    for group, count in sorted(group_counts.items(), key=lambda item: (-item[1], sha256_text(item[0]))):
        fold = min(range(n_splits), key=lambda index: (loads[index], index))
        mapping[group] = fold
        loads[fold] += count
    return mapping, loads


def normalize_type(type_value, utils):
    if not isinstance(type_value, str) or not type_value.strip():
        return None, ""
    normalized = utils.correct_type(type_value.strip())
    action, separator, parameter = normalized.partition(":")
    if action == "input_text":
        action = "type"
    elif action == "swipe" and separator:
        action = f"swipe:{parameter.strip()}"
        parameter = ""
    return action, parameter.strip()


def prediction_from_response(row, utils):
    bbox, type_value = utils.parse_response(row["response"])
    action, parameter = normalize_type(type_value, utils)
    coordinate = None
    if isinstance(bbox, (list, tuple)) and len(bbox) in (2, 4):
        values = [float(value) for value in bbox]
        if len(values) == 4:
            coordinate = ((values[0] + values[2]) / 2, (values[1] + values[3]) / 2)
        else:
            coordinate = (values[0], values[1])
    width, height = row["image_size"]
    return Prediction(
        action=action,
        x=coordinate[0] / width if coordinate else None,
        y=coordinate[1] / height if coordinate else None,
        parameter=parameter,
        source=row["model"],
        parse_ok=action is not None,
    )


def prediction_response(prediction, width, height):
    if prediction is None or not prediction.parse_ok:
        return ""
    action = prediction.action
    if action == "type":
        action = f"input_text:{prediction.parameter}"
    elif action == "open_app" and prediction.parameter:
        action = f"open_app:{prediction.parameter}"
    result = {"action_type": action}
    if prediction.coordinate is not None and prediction.action in {"click", "long_press"}:
        x = prediction.x * width
        y = prediction.y * height
        result["bbox_2d"] = [x, y, x, y]
    return f"<answer>{json.dumps(result, ensure_ascii=True)}</answer>"


def expected_actions(row):
    actions = [(row["gt_action"], row.get("gt_max_bbox"), row.get("gt_coordinate"))]
    for candidate in row.get("candidate_actions", []):
        actions.append((candidate.get("action_type", ""), candidate.get("action_bounds"), None))
    return actions


def score_response(row, response, utils):
    width, height = row["image_size"]
    actions = [[action, bbox] for action, bbox, _ in expected_actions(row)]
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        if row["setting"] == "high":
            bbox, action, step = utils.calculate_multi_android(actions, response, width, height, use_distance=False)
        else:
            bbox, action, step = utils.calculate_single_android(actions[0], response, width, height, use_distance=False)
    return {"bbox": bbox, "action": bool(action), "step": bool(step)}


def score_prediction(row, prediction, utils):
    width, height = row["image_size"]
    return score_response(row, prediction_response(prediction, width, height), utils)


def aggregate(method, predictions, priority, weights):
    if method == "A1_plurality_median":
        return plurality_then_median("androidcontrol", predictions, priority, weights).prediction
    if method == "A2_plurality_density":
        return plurality_then_density("androidcontrol", predictions, priority).prediction
    if method == "A3_pka_joint":
        return pka_medoid("androidcontrol", predictions).prediction
    if method == "A4_pka_continuous":
        return pka_continuous("androidcontrol", predictions).prediction
    raise ValueError(method)


def score_aggregate(row, prediction, method, utils):
    if method == "A3_pka_joint" and prediction.source in row["responses"]:
        return score_response(row, row["responses"][prediction.source], utils)
    return score_prediction(row, prediction, utils)


def threshold_success(row, prediction, radius, utils):
    if prediction is None or not prediction.parse_ok:
        return False
    width, height = row["image_size"]
    for action, bounds, point in expected_actions(row):
        expected_action, expected_parameter = normalize_type(action, utils)
        if prediction.action != expected_action:
            continue
        if expected_action in {"click", "long_press"}:
            if prediction.coordinate is None:
                continue
            if point is None and isinstance(bounds, list) and len(bounds) == 4:
                point = [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2]
            if point is None:
                continue
            if math.dist(prediction.coordinate, (point[0] / width, point[1] / height)) >= radius:
                continue
        elif expected_parameter and not utils.calculate_f1_score(expected_parameter, prediction.parameter):
            continue
        return True
    return False


def load_setting(setting, utils):
    filename, expected = DATA_CONFIG[setting]
    metadata = json.loads((DATA_DIR / filename).read_text())
    if len(metadata) != expected:
        raise ValueError(f"W4 metadata mismatch: {setting}")
    by_model = {}
    for model in MODELS:
        path = RUN_DIR / "w4_artifacts" / model / setting / "predictions.jsonl"
        rows = read_jsonl(path)
        if len(rows) != expected or [row["index"] for row in rows] != list(range(expected)):
            return None
        by_model[model] = rows
    combined = []
    for index, source in enumerate(metadata):
        model_rows = {model: by_model[model][index] for model in MODELS}
        exemplar = model_rows[MODELS[0]]
        if any(row["image"] != source["image"] for row in model_rows.values()):
            raise ValueError(f"W4 model identity mismatch: {setting}/{index}")
        combined.append({
            **source,
            "index": index,
            "setting": setting,
            "image_size": exemplar["image_size"],
            "candidate_actions": exemplar.get("candidate_actions", []),
            "predictions": {model: prediction_from_response(row, utils) for model, row in model_rows.items()},
            "responses": {model: row["response"] for model, row in model_rows.items()},
        })
    return combined


def analyze_setting(setting, rows, utils):
    mapping, fold_rows = assign_folds(rows)
    methods = ("A1_plurality_median", "A2_plurality_density", "A3_pka_joint", "A4_pka_continuous")
    individual = {
        model: [score_response(row, row["responses"][model], utils) for row in rows]
        for model in MODELS
    }
    folds = []
    aggregate_success = {method: 0 for method in methods}
    heldout_success = 0
    for fold in range(5):
        dev = [index for index, row in enumerate(rows) if mapping[str(Path(row["image"]).parent)] != fold]
        test = [index for index, row in enumerate(rows) if mapping[str(Path(row["image"]).parent)] == fold]
        priority = sorted(MODELS, key=lambda model: (-sum(individual[model][index]["step"] for index in dev), model))
        weights = {}
        for model in MODELS:
            grounding = [individual[model][index]["bbox"] for index in dev if individual[model][index]["bbox"] is not None]
            weights[model] = sum(grounding) / len(grounding) if grounding else 1.0
        heldout = priority[0]
        fold_counts = Counter()
        for index in test:
            row = rows[index]
            heldout_success += individual[heldout][index]["step"]
            predictions = [row["predictions"][model] for model in MODELS]
            for method in methods:
                prediction = aggregate(method, predictions, priority, weights)
                success = score_aggregate(row, prediction, method, utils)["step"]
                aggregate_success[method] += success
                fold_counts[method] += success
        folds.append({
            "fold": fold,
            "dev_rows": len(dev),
            "test_rows": len(test),
            "heldout_best_model": heldout,
            "priority": priority,
            "grounding_weights": weights,
            "step_sr": {method: fold_counts[method] / len(test) for method in methods},
        })
    metrics = {
        "A0_heldout_best": heldout_success / len(rows),
        **{method: aggregate_success[method] / len(rows) for method in methods},
        "oracle": sum(any(individual[model][index]["step"] for model in MODELS) for index in range(len(rows))) / len(rows),
    }
    model_metrics = {
        model: {
            "step_sr": sum(value["step"] for value in individual[model]) / len(rows),
            "type_accuracy": sum(value["action"] for value in individual[model]) / len(rows),
            "parse_rate": sum(rows[index]["predictions"][model].parse_ok for index in range(len(rows))) / len(rows),
        }
        for model in MODELS
    }
    hard_core = sum(not any(individual[model][index]["step"] for model in MODELS) for index in range(len(rows)))
    thresholds = {}
    base_values = None
    for radius in RADII:
        model_values = {
            model: sum(threshold_success(row, row["predictions"][model], radius, utils) for row in rows) / len(rows)
            for model in MODELS
        }
        oracle = sum(any(threshold_success(row, row["predictions"][model], radius, utils) for model in MODELS) for row in rows) / len(rows)
        if radius == 0.14:
            base_values = model_values
        thresholds[str(radius)] = {"model_step_sr": model_values, "oracle_step_sr": oracle}
    for radius in RADII:
        values = thresholds[str(radius)]["model_step_sr"]
        thresholds[str(radius)]["kendall_tau_vs_0.14_ranking"] = float(kendalltau(
            [base_values[model] for model in MODELS], [values[model] for model in MODELS]
        ).statistic)
    return {
        "rows": len(rows),
        "groups": len(mapping),
        "fold_method": "deterministic_group_balance_v1_on_image_parent",
        "fold_rows": fold_rows,
        "models": model_metrics,
        "hard_core_rows": hard_core,
        "hard_core_rate": hard_core / len(rows),
        "folds": folds,
        "aggregate_step_sr": metrics,
    }, thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated", type=Path, required=True)
    parser.add_argument("--threshold", type=Path, required=True)
    args = parser.parse_args()
    utils = load_official_utils()
    loaded = {setting: load_setting(setting, utils) for setting in ("low", "high")}
    if any(rows is None for rows in loaded.values()):
        pending = {"status": "PENDING_INFERENCE", "reason": "requires all ten W4 model-setting cells"}
        args.curated.write_text(json.dumps(pending, indent=2, sort_keys=True) + "\n")
        args.threshold.write_text(json.dumps(pending, indent=2, sort_keys=True) + "\n")
        print(json.dumps(pending, indent=2))
        return
    curated = {"status": "PASS", "settings": {}}
    threshold = {
        "status": "PASS",
        "contract": {"radii": list(RADII), "reference": 0.14, "target": "original gt_coordinate"},
        "settings": {},
    }
    for setting, rows in loaded.items():
        curated_result, threshold_result = analyze_setting(setting, rows, utils)
        curated["settings"][setting] = curated_result
        threshold["settings"][setting] = threshold_result
    args.curated.write_text(json.dumps(curated, indent=2, sort_keys=True) + "\n")
    args.threshold.write_text(json.dumps(threshold, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"curated": curated["status"], "threshold": threshold["status"]}, indent=2))


if __name__ == "__main__":
    main()
