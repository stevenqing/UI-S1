import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
XFER = ROOT / "runs/xfer/2026-08-07"
COLLISION = ROOT / "runs/collision-law/2026-07-30"
sys.path.insert(0, str(COLLISION))

from aggregators import pka_continuous, pka_medoid, plurality_then_density, plurality_then_median
from pka import Prediction
from scoring import GROUNDING_ACTIONS, SIMPLE_ACTIONS, TEXT_ACTIONS, text_f1, token_f1

from aggmatch_common import atomic_json, paired_bootstrap, sha256_file


CONFIG_PATH = RUN_DIR / "configs/f3_rows.yaml"
METHODS = ("majority", "sequential", "A1", "A2", "A3", "A4")
HISTORICAL_DIRS = {
    "UI-AGILE-7B": "ui-agile-7b",
    "GUI-R1-7B": "gui-r1-7b",
    "UI-R1-E-3B": "ui-r1-e-3b",
}


def load_lane(directory, setting, required_fields, required_prediction_fields):
    rows = {}
    paths = sorted((XFER / "raw/ac-stage1" / directory / setting).glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"{directory}/{setting}")
    for path in paths:
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in rows:
                raise ValueError(f"duplicate row: {path}:{line_number}/{row['id']}")
            if not set(required_fields).issubset(row):
                raise ValueError(f"incomplete row: {path}:{line_number}")
            if not set(required_prediction_fields).issubset(row["prediction"]):
                raise ValueError(f"incomplete prediction: {path}:{line_number}")
            rows[row["id"]] = row
    return rows, paths


def reference_rows(setting):
    path = XFER / f"data/androidcontrol/{setting}_sample.jsonl"
    return {row["id"]: row for row in map(json.loads, path.read_text().splitlines())}


def to_reference(row):
    width, height = row["image_size"]
    point = row["gt_bbox"]
    parameter = "" if row["gt_input_text"] == "no input text" else row["gt_input_text"]
    return {
        "action": row["gt_action"],
        "x": point[0] / width if point[0] >= 0 else None,
        "y": point[1] / height if point[1] >= 0 else None,
        "parameter": parameter,
        "episode_id": row["episode_id"],
    }


def to_prediction(row):
    value = row["prediction"]
    position = value["position"]
    return Prediction(
        action=str(value.get("action") or ""),
        x=position[0] if position is not None else None,
        y=position[1] if position is not None else None,
        parameter=str(value.get("value") or ""),
        source=row["model_id"],
        parse_ok=bool(value.get("parse_ok")),
    )


def score_prediction(reference, prediction):
    if prediction is None or not prediction.parse_ok or prediction.action != reference["action"]:
        return False
    if reference["action"] in GROUNDING_ACTIONS:
        return prediction.coordinate is not None and math.dist(prediction.coordinate, (reference["x"], reference["y"])) < 0.14
    if reference["action"] in TEXT_ACTIONS:
        return text_f1(prediction.parameter, reference["parameter"]) >= 0.5
    if reference["action"] in SIMPLE_ACTIONS:
        return True
    raise ValueError(f"unknown AndroidControl action: {reference['action']}")


def majority_prediction(predictions, priority):
    parsed = [prediction for prediction in predictions if prediction.parse_ok]
    if not parsed:
        return None
    counts = Counter(prediction.action for prediction in parsed)
    highest = max(counts.values())
    tied = {action for action, count in counts.items() if count == highest}
    return next(prediction for source in priority for prediction in parsed if prediction.source == source and prediction.action in tied)


def text_medoid(predictions):
    values = [prediction.parameter for prediction in predictions if prediction.parameter]
    if not values:
        return ""
    scores = [sum(token_f1(value.lower(), other.lower()) for other in values) for value in values]
    return values[max(range(len(values)), key=lambda index: (scores[index], -index))]


def sequential_prediction(predictions, priority, reliability, image_size):
    representative = majority_prediction(predictions, priority)
    if representative is None:
        return None
    retained = [prediction for prediction in predictions if prediction.parse_ok and prediction.action == representative.action]
    parameter = text_medoid(retained) if representative.action in TEXT_ACTIONS else ""
    if representative.action not in GROUNDING_ACTIONS:
        return Prediction(action=representative.action, parameter=parameter, source="sequential")
    width, height = image_size
    coordinate_predictions = [prediction for prediction in retained if prediction.coordinate is not None]
    if not coordinate_predictions:
        return Prediction(action=representative.action, parameter=parameter, source="sequential")
    points = [(prediction.x * width, prediction.y * height) for prediction in coordinate_predictions]
    groups = []
    assigned = set()
    for index in range(len(points)):
        if index in assigned:
            continue
        group = [index]
        assigned.add(index)
        for candidate in range(len(points)):
            if candidate in assigned:
                continue
            if all(abs(points[member][0] - points[candidate][0]) <= 14 and abs(points[member][1] - points[candidate][1]) <= 14 for member in group):
                group.append(candidate)
                assigned.add(candidate)
        groups.append(group)
    groups.sort(key=lambda group: (-len(group), min(group)))
    winner = max(
        groups[0],
        key=lambda index: (reliability[coordinate_predictions[index].source], -priority.index(coordinate_predictions[index].source), -index),
    )
    selected = coordinate_predictions[winner]
    return Prediction(action=representative.action, x=selected.x, y=selected.y, parameter=parameter, source="sequential")


def aggregate(method, predictions, priority, weights, reliability, image_size):
    if method == "majority":
        return majority_prediction(predictions, priority)
    if method == "sequential":
        return sequential_prediction(predictions, priority, reliability, image_size)
    if method == "A1":
        return plurality_then_median("androidcontrol", predictions, priority, weights).prediction
    if method == "A2":
        return plurality_then_density("androidcontrol", predictions, priority).prediction
    if method == "A3":
        return pka_medoid("androidcontrol", predictions).prediction
    if method == "A4":
        return pka_continuous("androidcontrol", predictions).prediction
    raise ValueError(method)


def dev_statistics(dev_ids, candidates, references, model_ids):
    reliability = {
        model: float(np.mean([score_prediction(references[row_id], candidates[row_id][model]) for row_id in dev_ids]))
        for model in model_ids
    }
    grounding_ids = [row_id for row_id in dev_ids if references[row_id]["action"] in GROUNDING_ACTIONS]
    weights = {}
    for model in model_ids:
        correct = []
        for row_id in grounding_ids:
            coordinate = candidates[row_id][model].coordinate
            correct.append(coordinate is not None and math.dist(coordinate, (references[row_id]["x"], references[row_id]["y"])) < 0.14)
        weights[model] = max(float(np.mean(correct)), 1e-6)
    priority = sorted(model_ids, key=lambda model: (-reliability[model], model))
    return priority, weights, reliability


def historical_scores(setting):
    output = {}
    for model, directory in HISTORICAL_DIRS.items():
        path = ROOT / f"runs/androidcontrol-rft/2026-07-29/artifacts/{directory}/{setting}/score.json"
        output[model] = {
            "step_sr": json.loads(path.read_text())["metrics"]["step_success"]["accuracy"],
            "source": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
        }
    return output


def analyze_setting(setting, config):
    required = config["eligibility"]["require_fields"]
    prediction_required = config["eligibility"]["require_prediction_fields"]
    lane_rows = {}
    source_files = {}
    for model in config["models"]:
        lane_rows[model["id"]], paths = load_lane(model["directory"], setting, required, prediction_required)
        source_files[model["id"]] = [{"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)} for path in paths]
    row_ids = sorted(set.intersection(*(set(rows) for rows in lane_rows.values())))
    frozen = config["eligibility"]["frozen_intersections"][setting]
    identity_sha = hashlib.sha256("\n".join(row_ids).encode()).hexdigest()
    if len(row_ids) != frozen["rows"] or identity_sha != frozen["row_ids_sha256"]:
        raise ValueError(f"frozen F3 intersection mismatch: {setting}")
    if len(row_ids) < config["eligibility"]["minimum_rows_per_setting"]:
        return {"status": "CANCELLED_INTERSECTION_BELOW_800", "rows": len(row_ids), "row_ids_sha256": identity_sha}
    source_references = reference_rows(setting)
    references = {row_id: to_reference(source_references[row_id]) for row_id in row_ids}
    model_ids = [model["id"] for model in config["models"]]
    candidates = {}
    for row_id in row_ids:
        rows = [lane_rows[model][row_id] for model in model_ids]
        if len({row["source_sha256"] for row in rows}) != 1 or len({row["image_sha256"] for row in rows}) != 1 or len({tuple(row["image_size"]) for row in rows}) != 1:
            raise ValueError(f"cross-model provenance mismatch: {setting}/{row_id}")
        candidates[row_id] = {model: to_prediction(lane_rows[model][row_id]) for model in model_ids}
    folds = json.loads((ROOT / "runs/complementarity/2026-07-30/folds.json").read_text())["pools"][f"androidcontrol/{setting}"]["group_to_fold"]
    outputs = {method: {} for method in METHODS}
    fold_records = []
    for test_fold in range(5):
        dev_ids = [row_id for row_id in row_ids if folds[references[row_id]["episode_id"]] != test_fold]
        test_ids = [row_id for row_id in row_ids if folds[references[row_id]["episode_id"]] == test_fold]
        priority, weights, reliability = dev_statistics(dev_ids, candidates, references, model_ids)
        for row_id in test_ids:
            predictions = [candidates[row_id][model] for model in model_ids]
            image_size = lane_rows[model_ids[0]][row_id]["image_size"]
            for method in METHODS:
                prediction = aggregate(method, predictions, priority, weights, reliability, image_size)
                outputs[method][row_id] = score_prediction(references[row_id], prediction)
        fold_records.append({"fold": test_fold, "dev_rows": len(dev_ids), "test_rows": len(test_ids), "priority": priority, "weights": weights, "reliability": reliability})
    metadata = {row_id: {"fold": folds[references[row_id]["episode_id"]], "group": references[row_id]["episode_id"]} for row_id in row_ids}
    comparisons = {}
    for method in METHODS[1:]:
        differences = {row_id: int(outputs["majority"][row_id]) - int(outputs[method][row_id]) for row_id in row_ids}
        comparisons[method] = paired_bootstrap(metadata, differences, config["bootstrap"]["resamples"], config["bootstrap"]["seed"] + (0 if setting == "low" else 1))
    subset_scores = {
        model: float(np.mean([score_prediction(references[row_id], candidates[row_id][model]) for row_id in row_ids]))
        for model in model_ids
    }
    historical = historical_scores(setting)
    bias = {
        model: {
            "subset_step_sr": subset_scores[model],
            "historical_full_step_sr": historical[model]["step_sr"],
            "delta_subset_minus_historical": subset_scores[model] - historical[model]["step_sr"],
            "absolute_delta": abs(subset_scores[model] - historical[model]["step_sr"]),
            "historical_source": historical[model]["source"],
            "historical_source_sha256": historical[model]["sha256"],
        }
        for model in model_ids
    }
    return {
        "status": "PASS",
        "rows": len(row_ids),
        "row_ids_sha256": identity_sha,
        "episodes": len({references[row_id]["episode_id"] for row_id in row_ids}),
        "pool": {"models": model_ids, "model_count": 3, "views_per_model": 1, "forwards_per_row": 3, "stages": ["stage1"], "not_Mind2Web_12_forward_pool": True},
        "source_files": source_files,
        "accuracy": {method: float(np.mean(list(values.values()))) for method, values in outputs.items()},
        "majority_minus_density": comparisons,
        "single_model_subset_bias": bias,
        "max_absolute_bias": max(value["absolute_delta"] for value in bias.values()),
        "direction_matches_Mind2Web": comparisons["sequential"]["point_delta"] > 0,
        "folds": fold_records,
    }


def main():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if config["status"] != "FROZEN_BEFORE_RESULTS":
        raise ValueError("F3 row contract is not frozen")
    low = analyze_setting("low", config)
    high = analyze_setting("high", config)
    intersection_failure = any(result["rows"] < config["eligibility"]["minimum_rows_per_setting"] for result in (low, high))
    bias_failure = any(result.get("max_absolute_bias", 0) > config["bias"]["appendix_threshold_absolute"] for result in (low, high))
    f_k3 = intersection_failure or bias_failure
    result = {
        "schema_version": 1,
        "status": "PASS_APPENDIX_ONLY" if f_k3 and not intersection_failure else "CANCELLED" if intersection_failure else "PASS_MAIN_TEXT_ELIGIBLE",
        "analysis_scope": "SAME_POOL_AGGREGATOR_COMPARISON_ONLY",
        "config": "configs/f3_rows.yaml",
        "config_sha256": sha256_file(CONFIG_PATH),
        "collection_context": "partial_completed_subset_from_a_cancelled_protocol_with_sampling_bias_risk",
        "androidcontrol_action_space": "product_action_type_x_coordinate_x_parameter_including_parameterless_actions",
        "low": low,
        "high": high,
        "gates": {"F_K3": f_k3, "intersection_failure": intersection_failure, "bias_over_2pp": bias_failure},
        "four_arm_status": "CANCELLED_UNCHANGED",
        "C_cond_conclusion": "PROHIBITED_NOT_EVALUATED",
        "limitation": "Rows are a partially completed subset collected under a cancelled protocol; subset-vs-historical single-model deltas estimate selection and rerun bias.",
    }
    atomic_json(RUN_DIR / "f3_androidcontrol_aggregator.json", result)
    print(json.dumps({"status": result["status"], "gates": result["gates"], "low": {key: low[key] for key in ("rows", "accuracy", "majority_minus_density", "max_absolute_bias", "direction_matches_Mind2Web")}, "high": {key: high[key] for key in ("rows", "accuracy", "majority_minus_density", "max_absolute_bias", "direction_matches_Mind2Web")}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()