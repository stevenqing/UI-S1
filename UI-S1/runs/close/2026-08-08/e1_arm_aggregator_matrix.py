import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
XFER = ROOT / "runs/xfer/2026-08-07"
CONSOLIDATE = ROOT / "runs/consolidate/2026-08-06"
COLLISION = ROOT / "runs/collision-law/2026-07-30"
SOURCEBIAS = ROOT / "runs/sourcebias/2026-08-03"
sys.path.insert(0, str(XFER))
sys.path.insert(0, str(COLLISION))
sys.path.insert(0, str(SOURCEBIAS))

from aggregators import pka_continuous, pka_medoid, plurality_then_density, plurality_then_median
from pka import Prediction
from sourcebias_common import b3_select_index
from xf_mind2web import MODEL_DIRS, load_unique, paired_episode_bootstrap, score_prediction
from xfer_common import aggregate


ARMS = ("C_uni", "C_cond", "C_rand", "C_self")
METHODS = ("majority", "A0", "ours", "A1", "A2", "A3", "A4")
SCREEN_MODELS = ("GTA1-7B", "Qwen3-VL-8B-Instruct", "UI-TARS-7B-SFT")
SCREEN_DIRECTORIES = {
    "GTA1-7B": "q1-gta1",
    "Qwen3-VL-8B-Instruct": "q1-qwen3",
    "UI-TARS-7B-SFT": "q1-uitars",
}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def crop_prediction(row, arm, crop_index):
    return row["predictions"][arm][crop_index]["prediction"]


def mind_slots(rows_by_id, full_lanes, view1_lanes, stage2, arm):
    output = {}
    for row_id in rows_by_id:
        slots = []
        for view_index in (0, 1):
            for model in full_lanes:
                prediction = full_lanes[model][row_id]["prediction"] if view_index == 0 else view1_lanes[model][row_id]["predictions"]["view1"][0]["prediction"]
                slots.append((f"stage1_{model}_view{view_index}", model, prediction))
        for crop_index in range(2):
            for model in full_lanes:
                prediction = crop_prediction(stage2[model][row_id], arm, crop_index)
                slots.append((f"stage2_{model}_crop{crop_index}", model, prediction))
        if len(slots) != 12:
            raise ValueError(f"Mind2Web E1 candidate budget mismatch: {arm}/{row_id}")
        output[row_id] = slots
    return output


def to_prediction(slot, prediction):
    position = prediction.get("position")
    return Prediction(
        action=prediction.get("action") or "",
        x=position[0] if position is not None else None,
        y=position[1] if position is not None else None,
        parameter=str(prediction.get("value") or ""),
        source=slot,
        parse_ok=bool(prediction.get("parse_ok")),
    )


def from_prediction(prediction):
    if prediction is None:
        return {"action": None, "value": None, "position": None, "parse_ok": False}
    return {
        "action": prediction.action,
        "value": prediction.parameter or None,
        "position": list(prediction.coordinate) if prediction.coordinate is not None else None,
        "parse_ok": prediction.parse_ok,
    }


def dev_mind_statistics(dev_ids, slots, rows_by_id, image_sizes):
    success = defaultdict(list)
    grounding = defaultdict(list)
    for row_id in dev_ids:
        for slot, _, prediction in slots[row_id]:
            success[slot].append(score_prediction(rows_by_id[row_id], prediction, image_sizes[row_id]))
            if prediction.get("position") is not None:
                bbox = rows_by_id[row_id]["step"]["bbox"]
                width, height = image_sizes[row_id]
                x, y = prediction["position"]
                grounding[slot].append(bbox["x"] / width <= x <= (bbox["x"] + bbox["width"]) / width and bbox["y"] / height <= y <= (bbox["y"] + bbox["height"]) / height)
    reliability = {slot: float(np.mean(values)) for slot, values in success.items()}
    priority = sorted(reliability, key=lambda slot: (-reliability[slot], slot))
    weights = {slot: max(float(np.mean(grounding.get(slot, [False]))), 1e-6) for slot in reliability}
    return priority, weights


def majority_exact(predictions, priority):
    parsed = [prediction for prediction in predictions if prediction.parse_ok]
    if not parsed:
        return None
    counts = defaultdict(int)
    for prediction in parsed:
        counts[prediction.action] += 1
    highest = max(counts.values())
    tied = {action for action, count in counts.items() if count == highest}
    return next(prediction for slot in priority for prediction in parsed if prediction.source == slot and prediction.action in tied)


def evaluate_mind_method(method, row_id, slots, priority, weights, rows_by_id, image_sizes, model_order):
    source = slots[row_id]
    predictions = [to_prediction(slot, prediction) for slot, _, prediction in source]
    if method in {"majority", "A0"}:
        result = majority_exact(predictions, priority) if method == "majority" else next((prediction for slot in priority for prediction in predictions if prediction.source == slot and prediction.parse_ok), None)
        output = from_prediction(result)
    elif method == "ours":
        output = aggregate([{**prediction, "model": model} for _, model, prediction in source], model_order, image_sizes[row_id])
    elif method == "A1":
        output = from_prediction(plurality_then_median("mind2web", predictions, priority, weights).prediction)
    elif method == "A2":
        output = from_prediction(plurality_then_density("mind2web", predictions, priority).prediction)
    elif method == "A3":
        output = from_prediction(pka_medoid("mind2web", predictions).prediction)
    elif method == "A4":
        output = from_prediction(pka_continuous("mind2web", predictions).prediction)
    else:
        raise ValueError(method)
    return score_prediction(rows_by_id[row_id], output, image_sizes[row_id])


def mind2web_matrix(config):
    rows = [json.loads(line) for line in (XFER / "data/mind2web/mind2web_test_task.jsonl").read_text().splitlines() if line.strip()]
    rows_by_id = {row["id"]: row for row in rows}
    image_sizes = {row["id"]: Image.open(ROOT / row["image"]).size for row in rows}
    full = {model: load_unique(XFER / "raw/stage1" / directory) for model, directory in MODEL_DIRS.items()}
    view1 = {model: load_unique(XFER / "raw/stage1/view1" / directory) for model, directory in MODEL_DIRS.items()}
    stage2 = {model: load_unique(XFER / "raw/stage2" / directory) for model, directory in MODEL_DIRS.items()}
    fold_map = json.loads((ROOT / "runs/complementarity/2026-07-30/folds.json").read_text())["pools"]["mind2web/visual"]["group_to_fold"]
    outputs = {arm: {method: {} for method in METHODS} for arm in ARMS}
    folds = []
    for test_fold in range(5):
        dev_ids = [row_id for row_id, row in rows_by_id.items() if fold_map[row["website"]] != test_fold]
        test_ids = [row_id for row_id, row in rows_by_id.items() if fold_map[row["website"]] == test_fold]
        fold_result = {"fold": test_fold, "dev_rows": len(dev_ids), "test_rows": len(test_ids), "arms": {}}
        for arm in ARMS:
            slots = mind_slots(rows_by_id, full, view1, stage2, arm)
            priority, weights = dev_mind_statistics(dev_ids, slots, rows_by_id, image_sizes)
            accuracy = {}
            for method in METHODS:
                for row_id in test_ids:
                    outputs[arm][method][row_id] = evaluate_mind_method(method, row_id, slots, priority, weights, rows_by_id, image_sizes, list(MODEL_DIRS))
                accuracy[method] = float(np.mean([outputs[arm][method][row_id] for row_id in test_ids]))
            fold_result["arms"][arm] = {"priority": priority, "grounding_weights": weights, "accuracy": accuracy}
        folds.append(fold_result)
    accuracy = {arm: {method: float(np.mean(list(outputs[arm][method].values()))) for method in METHODS} for arm in ARMS}
    if abs(accuracy["C_cond"]["ours"] - 0.3158653846153846) > 1e-15 or abs(accuracy["C_cond"]["majority"] - 0.3230769230769231) > 1e-15:
        raise ValueError(f"Mind2Web E1 anchor mismatch: {accuracy['C_cond']}")
    comparisons = {}
    for reference in ("C_uni", "C_rand", "C_self"):
        comparisons[reference] = paired_episode_bootstrap(
            rows_by_id, fold_map, outputs["C_cond"]["majority"], outputs[reference]["majority"],
            resamples=config["mind2web"]["bootstrap"]["resamples"], seed=config["mind2web"]["bootstrap"]["seed"],
        )
    return {"rows": len(rows), "accuracy": accuracy, "majority_comparisons": comparisons, "folds": folds, "outputs": outputs}


def load_screen_q1(model):
    rows = {}
    for path in sorted((CONSOLIDATE / "raw" / SCREEN_DIRECTORIES[model]).glob("shard-*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in rows:
                raise ValueError(f"duplicate ScreenSpot Q1 row: {model}/{row['id']}")
            rows[row["id"]] = row
    if len(rows) != 1581:
        raise ValueError(f"ScreenSpot Q1 coverage mismatch: {model}/{len(rows)}")
    return rows


def screen_slots(context, regions, q1, arm, row_id):
    if arm == "C_uni":
        actions = [(model, view) for view in range(4) for model in SCREEN_MODELS]
        return [(f"{model}_view{view}", dict(context["bank"][(model, view)][row_id])) for model, view in actions]
    source = regions[row_id]
    slots = []
    for model, view in source["stage1_actions"]:
        slots.append((f"{model}_view{view}", dict(context["bank"][(model, view)][row_id])))
    for crop_index in range(2):
        for model in SCREEN_MODELS:
            prediction = q1[model][row_id]["predictions"][arm][crop_index]
            slots.append((f"{model}_{arm}_crop{crop_index}", {
                "model": model,
                "view_index": crop_index + 2,
                "point": prediction["point"],
                "region": prediction["region"],
                "coverage": 0,
            }))
    if len(slots) != 12:
        raise ValueError(f"ScreenSpot E1 candidate budget mismatch: {arm}/{row_id}")
    return slots


def geometric_median(points, iterations=100):
    estimate = np.mean(np.asarray(points, dtype=np.float64), axis=0)
    for _ in range(iterations):
        distances = np.linalg.norm(np.asarray(points) - estimate, axis=1)
        if np.any(distances < 1e-12):
            return tuple(points[int(np.argmin(distances))])
        weights = 1 / distances
        updated = np.sum(np.asarray(points) * weights[:, None], axis=0) / weights.sum()
        if np.linalg.norm(updated - estimate) < 1e-9:
            return tuple(updated)
        estimate = updated
    return tuple(estimate)


def screen_similarity(left, right):
    return math.exp(-(math.dist(left, right) ** 2) / (2 * 14.0**2))


def density_medoid(points):
    scores = [sum(screen_similarity(point, candidate) for point in points) for candidate in points]
    return points[max(range(len(points)), key=lambda index: (scores[index], -index))]


def density_mode(points, iterations=8):
    candidates = []
    for seed in points:
        estimate = np.asarray(seed, dtype=np.float64)
        for _ in range(iterations):
            weights = np.asarray([screen_similarity(point, estimate) for point in points])
            if weights.sum() <= 0:
                break
            updated = np.sum(np.asarray(points) * weights[:, None], axis=0) / weights.sum()
            if np.linalg.norm(updated - estimate) < 1e-9:
                estimate = updated
                break
            estimate = updated
        density = sum(screen_similarity(point, estimate) for point in points)
        candidates.append((density, tuple(estimate)))
    return max(candidates, key=lambda item: item[0])[1]


def screen_dev_priority(dev_ids, slots_by_id, targets):
    reliability = {}
    for index, (slot, _) in enumerate(next(iter(slots_by_id.values()))):
        reliability[slot] = float(np.mean([point_in_bbox(slots_by_id[row_id][index][1]["point"], targets[row_id]) for row_id in dev_ids]))
    return sorted(reliability, key=lambda slot: (-reliability[slot], slot)), reliability


def point_in_bbox(point, bbox):
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def evaluate_screen_method(method, row_id, slots, priority, target):
    candidates = [candidate for _, candidate in slots[row_id]]
    by_slot = {slot: candidate for slot, candidate in slots[row_id]}
    points = [candidate["point"] for candidate in candidates]
    if method in {"majority", "A0"}:
        point = by_slot[priority[0]]["point"]
    elif method == "ours":
        point = candidates[b3_select_index(candidates)[0]]["point"]
    elif method == "A1":
        point = geometric_median(points)
    elif method in {"A2", "A3"}:
        point = density_medoid(points)
    elif method == "A4":
        point = density_mode(points)
    else:
        raise ValueError(method)
    return point_in_bbox(point, target)


def screenspot_matrix(config):
    consolidate_common = load_module(CONSOLIDATE / "common.py", "close_consolidate_common")
    context = consolidate_common.load_context()
    regions = {row["id"]: row for row in map(json.loads, (CONSOLIDATE / "raw/q1_regions.jsonl").read_text().splitlines())}
    q1 = {model: load_screen_q1(model) for model in SCREEN_MODELS}
    row_ids = context["row_ids"]
    targets = {row_id: context["metadata"][row_id]["target_bbox"] for row_id in row_ids}
    fold_for_group = context["fold_for_group"]
    outputs = {arm: {method: {} for method in METHODS} for arm in ARMS}
    folds = []
    for test_fold in range(5):
        dev_ids = [row_id for row_id in row_ids if fold_for_group[context["metadata"][row_id]["application"]] != test_fold]
        test_ids = [row_id for row_id in row_ids if fold_for_group[context["metadata"][row_id]["application"]] == test_fold]
        fold_result = {"fold": test_fold, "dev_rows": len(dev_ids), "test_rows": len(test_ids), "arms": {}}
        for arm in ARMS:
            slots = {row_id: screen_slots(context, regions, q1, arm, row_id) for row_id in row_ids}
            priority, reliability = screen_dev_priority(dev_ids, slots, targets)
            accuracy = {}
            for method in METHODS:
                for row_id in test_ids:
                    outputs[arm][method][row_id] = evaluate_screen_method(method, row_id, slots, priority, targets[row_id])
                accuracy[method] = float(np.mean([outputs[arm][method][row_id] for row_id in test_ids]))
            fold_result["arms"][arm] = {"priority": priority, "slot_reliability": reliability, "accuracy": accuracy}
        folds.append(fold_result)
    accuracy = {arm: {method: float(np.mean(list(outputs[arm][method].values()))) for method in METHODS} for arm in ARMS}
    expected = {"C_uni": 0.6369386464263125, "C_cond": 0.6590765338393422, "C_rand": 0.6053130929791272, "C_self": 0.6457938013915243}
    for arm, value in expected.items():
        if abs(accuracy[arm]["ours"] - value) > 1e-15:
            raise ValueError(f"ScreenSpot E1 B3 anchor mismatch: {arm}/{accuracy[arm]['ours']} != {value}")
    metadata = {row_id: {"outer_fold": fold_for_group[context["metadata"][row_id]["application"]], "application": context["metadata"][row_id]["application"]} for row_id in row_ids}
    comparisons = {
        reference: consolidate_common.paired_group_bootstrap(metadata, outputs["C_cond"]["majority"], outputs[reference]["majority"], resamples=config["screenspot_pro"]["bootstrap"]["resamples"], seed=config["screenspot_pro"]["bootstrap"]["seed"])
        for reference in ("C_uni", "C_rand", "C_self")
    }
    return {"rows": len(row_ids), "accuracy": accuracy, "majority_comparisons": comparisons, "folds": folds, "outputs": outputs}


def summarize_matrix(result):
    accuracy = result["accuracy"]
    row_best = {arm: max(values, key=lambda method: (values[method], method)) for arm, values in accuracy.items()}
    column_best = {method: max(ARMS, key=lambda arm: (accuracy[arm][method], arm)) for method in METHODS}
    cells = [(accuracy[arm][method], arm, method) for arm in ARMS for method in METHODS]
    best_value, best_arm, best_method = max(cells)
    global_best_cells = [
        {"arm": arm, "aggregator": method, "accuracy": value}
        for value, arm, method in cells if abs(value - best_value) <= 1e-15
    ]
    best_arm_by_max = max(ARMS, key=lambda arm: max(accuracy[arm].values()))
    best_method_by_max = max(METHODS, key=lambda method: max(accuracy[arm][method] for arm in ARMS))
    return {
        "row_best_aggregator": row_best,
        "column_best_arm": column_best,
        "global_best": {"arm": best_arm, "aggregator": best_method, "accuracy": best_value},
        "global_best_ties": global_best_cells,
        "interaction": {
            "best_arm_by_row_max": best_arm_by_max,
            "best_aggregator_by_column_max": best_method_by_max,
            "their_combination_is_global_best": abs(accuracy[best_arm_by_max][best_method_by_max] - best_value) <= 1e-15,
            "row_best_varies": len(set(row_best.values())) > 1,
            "column_best_varies": len(set(column_best.values())) > 1,
        },
    }


def strip_outputs(value):
    return {key: item for key, item in value.items() if key != "outputs"}


def main():
    config = yaml.safe_load((RUN_DIR / "configs/aggregator_map.yaml").read_text())
    if config["status"] != "RESULT_BLIND_BEFORE_E1":
        raise ValueError("E1 aggregator map is not frozen")
    mind = mind2web_matrix(config)
    screen = screenspot_matrix(config)
    for result in (mind, screen):
        result["matrix_summary"] = summarize_matrix(result)
    mde = {"mind2web": config["mind2web"]["mde"], "screenspot_pro": config["screenspot_pro"]["mde"]}
    gates = {}
    for name, result in (("mind2web", mind), ("screenspot_pro", screen)):
        primary = result["majority_comparisons"]["C_uni"]
        gates[name] = {
            "primary_pass": primary["point_delta"] > mde[name] and primary["ci_99"][0] > 0,
            "rand_control_pass": result["majority_comparisons"]["C_rand"]["ci_99"][0] > 0,
            "self_control_pass": result["majority_comparisons"]["C_self"]["ci_99"][0] > 0,
        }
    result = {
        "schema_version": 1,
        "status": "PASS",
        "config": "configs/aggregator_map.yaml",
        "mind2web": strip_outputs(mind),
        "screenspot_pro": strip_outputs(screen),
        "gates": gates,
        "E_K1": not (gates["mind2web"]["primary_pass"] and gates["screenspot_pro"]["primary_pass"]),
        "downstream_decision": "START_E2" if gates["mind2web"]["primary_pass"] and gates["screenspot_pro"]["primary_pass"] else "CANCEL_E2_AND_ANDROIDCONTROL",
    }
    output = RUN_DIR / "e1_arm_aggregator_matrix.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": gates, "E_K1": result["E_K1"], "decision": result["downstream_decision"], "mind2web": {"accuracy": mind["accuracy"], "comparisons": mind["majority_comparisons"], "summary": mind["matrix_summary"]}, "screenspot_pro": {"accuracy": screen["accuracy"], "comparisons": screen["majority_comparisons"], "summary": screen["matrix_summary"]}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
