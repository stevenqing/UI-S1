import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
COLLISION = ROOT / "runs/collision-law/2026-07-30"
sys.path.insert(0, str(RUN_DIR))
sys.path.insert(0, str(COLLISION))

from aggregators import pka_continuous, pka_medoid, plurality_then_density, plurality_then_median
from pka import Prediction
from xf_mind2web import MODEL_DIRS, load_unique, load_unique_file, paired_episode_bootstrap, score_prediction
from xfer_common import aggregate


METHODS = (
    "ours_sequential_cluster",
    "A0_heldout_best",
    "majority_exact_candidate",
    "A1_plurality_median",
    "A2_plurality_density",
    "A3_pka_joint",
    "A4_pka_continuous",
)


def crop_prediction(row, set_name, crop_index=0):
    return row["predictions"][set_name][crop_index]["prediction"]


def candidate_slots(rows_by_id, full_lanes, view1_lanes, stage2):
    output = {}
    for row_id in rows_by_id:
        slots = []
        for view_index in (0, 1):
            for model in full_lanes:
                prediction = (
                    full_lanes[model][row_id]["prediction"]
                    if view_index == 0
                    else crop_prediction(view1_lanes[model][row_id], "view1")
                )
                slots.append((f"stage1_{model}_view{view_index}", model, prediction))
        for crop_index in range(2):
            for model in full_lanes:
                prediction = crop_prediction(stage2[model][row_id], "C_cond", crop_index)
                slots.append((f"stage2_{model}_crop{crop_index}", model, prediction))
        if len(slots) != 12:
            raise ValueError(f"baseline candidate budget mismatch: {row_id}")
        output[row_id] = slots
    return output


def to_collision(slot, prediction):
    position = prediction.get("position")
    return Prediction(
        action=prediction.get("action") or "",
        x=position[0] if position is not None else None,
        y=position[1] if position is not None else None,
        parameter=str(prediction.get("value") or ""),
        source=slot,
        parse_ok=bool(prediction.get("parse_ok")),
    )


def from_collision(prediction):
    if prediction is None:
        return {"action": None, "value": None, "position": None, "parse_ok": False}
    return {
        "action": prediction.action,
        "value": prediction.parameter or None,
        "position": list(prediction.coordinate) if prediction.coordinate is not None else None,
        "parse_ok": prediction.parse_ok,
    }


def dev_slot_statistics(dev_ids, slots_by_id, rows_by_id, image_sizes):
    success = defaultdict(list)
    grounding = defaultdict(list)
    for row_id in dev_ids:
        for slot, _, prediction in slots_by_id[row_id]:
            success[slot].append(score_prediction(rows_by_id[row_id], prediction, image_sizes[row_id]))
            if prediction.get("position") is not None:
                bbox = rows_by_id[row_id]["step"]["bbox"]
                width, height = image_sizes[row_id]
                x, y = prediction["position"]
                grounding[slot].append(
                    bbox["x"] / width <= x <= (bbox["x"] + bbox["width"]) / width
                    and bbox["y"] / height <= y <= (bbox["y"] + bbox["height"]) / height
                )
    reliability = {slot: float(np.mean(values)) for slot, values in success.items()}
    weights = {slot: max(float(np.mean(grounding.get(slot, [False]))), 1e-6) for slot in success}
    priority = sorted(reliability, key=lambda slot: (-reliability[slot], slot))
    return priority, weights, reliability


def majority_exact(predictions, priority):
    parsed = [prediction for prediction in predictions if prediction.parse_ok]
    if not parsed:
        return None
    counts = defaultdict(int)
    for prediction in parsed:
        counts[prediction.action] += 1
    highest = max(counts.values())
    tied = {action for action, count in counts.items() if count == highest}
    for slot in priority:
        for prediction in parsed:
            if prediction.source == slot and prediction.action in tied:
                return prediction
    raise AssertionError("majority tie-break failed")


def evaluate_method(method, row_id, slots_by_id, priority, weights, rows_by_id, image_sizes, model_order):
    source = slots_by_id[row_id]
    collision_predictions = [to_collision(slot, prediction) for slot, _, prediction in source]
    if method == "A0_heldout_best":
        by_slot = {prediction.source: prediction for prediction in collision_predictions}
        result = next((by_slot[slot] for slot in priority if by_slot[slot].parse_ok), None)
        output = from_collision(result)
    elif method == "majority_exact_candidate":
        output = from_collision(majority_exact(collision_predictions, priority))
    elif method == "A1_plurality_median":
        output = from_collision(plurality_then_median("mind2web", collision_predictions, priority, weights).prediction)
    elif method == "A2_plurality_density":
        output = from_collision(plurality_then_density("mind2web", collision_predictions, priority).prediction)
    elif method == "A3_pka_joint":
        output = from_collision(pka_medoid("mind2web", collision_predictions).prediction)
    elif method == "A4_pka_continuous":
        output = from_collision(pka_continuous("mind2web", collision_predictions).prediction)
    elif method == "ours_sequential_cluster":
        candidates = [
            {**prediction, "model": model}
            for slot, model, prediction in source
        ]
        output = aggregate(candidates, model_order, image_sizes[row_id])
    else:
        raise ValueError(method)
    return score_prediction(rows_by_id[row_id], output, image_sizes[row_id])


def main():
    rows = [json.loads(line) for line in (RUN_DIR / "data/mind2web/mind2web_test_task.jsonl").read_text().splitlines() if line.strip()]
    rows_by_id = {row["id"]: row for row in rows}
    image_sizes = {row["id"]: Image.open(ROOT / row["image"]).size for row in rows}
    full_lanes = {model: load_unique(RUN_DIR / "raw/stage1" / directory) for model, directory in MODEL_DIRS.items()}
    view1_lanes = {model: load_unique(RUN_DIR / "raw/stage1/view1" / directory) for model, directory in MODEL_DIRS.items()}
    stage2 = {model: load_unique(RUN_DIR / "raw/stage2" / directory) for model, directory in MODEL_DIRS.items()}
    slots = candidate_slots(rows_by_id, full_lanes, view1_lanes, stage2)
    folds = json.loads((ROOT / "runs/complementarity/2026-07-30/folds.json").read_text())["pools"]["mind2web/visual"]["group_to_fold"]
    model_order = list(MODEL_DIRS)
    outputs = {method: {} for method in METHODS}
    fold_reports = []
    for test_fold in range(5):
        dev_ids = [row_id for row_id, row in rows_by_id.items() if folds[row["website"]] != test_fold]
        test_ids = [row_id for row_id, row in rows_by_id.items() if folds[row["website"]] == test_fold]
        priority, weights, reliability = dev_slot_statistics(dev_ids, slots, rows_by_id, image_sizes)
        fold_accuracy = {}
        for method in METHODS:
            for row_id in test_ids:
                outputs[method][row_id] = evaluate_method(
                    method, row_id, slots, priority, weights,
                    rows_by_id, image_sizes, model_order,
                )
            fold_accuracy[method] = float(np.mean([outputs[method][row_id] for row_id in test_ids]))
        fold_reports.append({
            "fold": test_fold,
            "dev_rows": len(dev_ids),
            "test_rows": len(test_ids),
            "priority": priority,
            "slot_reliability": reliability,
            "grounding_weights": weights,
            "accuracy": fold_accuracy,
        })
    if any(set(values) != set(rows_by_id) for values in outputs.values()):
        raise ValueError("baseline output coverage mismatch")
    accuracy = {method: float(np.mean(list(values.values()))) for method, values in outputs.items()}
    expected_ours = 0.3158653846153846
    if abs(accuracy["ours_sequential_cluster"] - expected_ours) > 1e-15:
        raise ValueError(
            f"published C-cond anchor mismatch: {accuracy['ours_sequential_cluster']} != {expected_ours}"
        )
    comparisons = {
        method: paired_episode_bootstrap(rows_by_id, folds, outputs["ours_sequential_cluster"], outputs[method])
        for method in METHODS if method != "ours_sequential_cluster"
    }
    result = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "same C-cond 12-candidate pool, fold-local priorities and weights",
        "rows": len(rows),
        "episodes": len({row["episode_id"] for row in rows}),
        "methods": list(METHODS),
        "accuracy": accuracy,
        "ours_minus_baseline": comparisons,
        "folds": fold_reports,
        "claim_gate": {
            method: comparisons[method]["point_delta"] > 0 and comparisons[method]["ci_99"][0] > 0
            for method in comparisons
        },
    }
    output = RUN_DIR / "baseline_mind2web.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "accuracy": accuracy,
        "ours_minus_baseline": comparisons,
        "claim_gate": result["claim_gate"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()