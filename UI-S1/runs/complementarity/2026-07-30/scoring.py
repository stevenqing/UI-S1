import json
import math
from pathlib import Path


GROUNDING_ACTIONS = {"click", "long_press", "moveto", "doubleclick", "rightclick"}
TEXT_ACTIONS = {"type", "open_app", "scroll", "select"}
SIMPLE_ACTIONS = {
    "press_back", "wait", "navigate_back", "press_home", "complete", "impossible",
    "press_space", "press_enter", "press_down", "hotkey", "press_tab", "press_pgdn",
}
ACTION_TO_ID = {"CLICK": 4, "SELECT": 2, "TYPE": 3}


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def token_f1(prediction: str, reference: str) -> float:
    predicted = set(prediction.strip().split())
    expected = set(reference.strip().split())
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = len(predicted & expected)
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def text_f1(prediction, reference) -> float:
    if not isinstance(prediction, str) or not isinstance(reference, str):
        return 0.0
    return token_f1(prediction.lower(), reference.lower())


def label_android_row(row: dict, grounding_radius: float = 0.14) -> dict:
    action_correct = row["pred_action"] == row["gt_action"]
    grounding_correct = None
    parameter_f1 = None
    parameter_correct = None
    normalized_distance = None
    if row["gt_action"] in GROUNDING_ACTIONS:
        predicted_x, predicted_y = row["pred_coord"][:2]
        expected_x, expected_y = row["gt_bbox"][:2]
        width, height = row["image_size"]
        normalized_distance = math.sqrt(
            ((expected_x - predicted_x) / width) ** 2
            + ((expected_y - predicted_y) / height) ** 2
        )
        grounding_correct = normalized_distance < grounding_radius
        step_success = action_correct and grounding_correct
    elif row["gt_action"] in TEXT_ACTIONS:
        parameter_f1 = text_f1(row["pred_input_text"], row["gt_input_text"])
        parameter_correct = parameter_f1 >= 0.5
        step_success = action_correct and parameter_correct
    elif row["gt_action"] in SIMPLE_ACTIONS:
        step_success = action_correct
    else:
        raise ValueError(f"unclassified GT action: {row['gt_action']}")

    if step_success:
        error_type = "success"
    elif row["pred_action"] is None:
        error_type = "parse_failure"
    elif not action_correct:
        error_type = "action_mismatch"
    elif grounding_correct is False:
        error_type = "grounding_miss"
    elif parameter_correct is False:
        error_type = "parameter_miss"
    else:
        error_type = "simple_action_miss"
    return {
        "step_success": bool(step_success),
        "action_correct": action_correct,
        "grounding_correct": grounding_correct,
        "parameter_f1": parameter_f1,
        "parameter_correct": parameter_correct,
        "normalized_distance": normalized_distance,
        "error_type": error_type,
    }


def android_metric_counts(labels: list[dict], rows: list[dict]) -> dict:
    output = {
        "action": {"correct": 0, "total": len(rows)},
        "grounding": {"correct": 0, "total": 0},
        "text": {"correct": 0, "total": 0},
        "step_success": {"correct": 0, "total": len(rows)},
    }
    for row, label in zip(rows, labels):
        output["action"]["correct"] += int(label["action_correct"])
        output["step_success"]["correct"] += int(label["step_success"])
        if row["gt_action"] in GROUNDING_ACTIONS:
            output["grounding"]["total"] += 1
            output["grounding"]["correct"] += int(label["grounding_correct"])
        elif row["gt_action"] in TEXT_ACTIONS:
            output["text"]["total"] += 1
            output["text"]["correct"] += int(label["parameter_correct"])
    return output


def transition(left: list[dict], right: list[dict], indices=None) -> dict:
    indices = range(len(left)) if indices is None else indices
    counts = {"both_success": 0, "left_only": 0, "right_only": 0, "both_fail": 0}
    for index in indices:
        left_success = left[index]["step_success"]
        right_success = right[index]["step_success"]
        name = (
            "both_success" if left_success and right_success
            else "left_only" if left_success
            else "right_only" if right_success
            else "both_fail"
        )
        counts[name] += 1
    counts = {key: value for key, value in counts.items() if value}
    counts["compared"] = sum(counts.values())
    counts["net_right_gain"] = counts.get("right_only", 0) - counts.get("left_only", 0)
    return counts


def normalized_bbox(row: dict, rounded: bool) -> list[float]:
    width, height = row["image_size"]
    bbox = row["bbox"]
    values = [
        bbox["x"] / width,
        bbox["y"] / height,
        (bbox["x"] + bbox["width"]) / width,
        (bbox["y"] + bbox["height"]) / height,
    ]
    return [round(value, 3) for value in values] if rounded else values


def distance_to_bbox(position, bbox) -> float | None:
    if position is None:
        return None
    delta_x = max(bbox[0] - position[0], 0.0, position[0] - bbox[2])
    delta_y = max(bbox[1] - position[1], 0.0, position[1] - bbox[3])
    return math.sqrt(delta_x**2 + delta_y**2)


def score_mind2web_row(row: dict, parser_kind: str, parse_prediction) -> dict:
    result = {
        "parse_ok": False,
        "supported_action": False,
        "element": 0.0,
        "operation_f1": 0.0,
        "step_success": False,
        "pred_action": None,
        "position": None,
        "semantic_error": None,
        "error_type": None,
    }
    try:
        prediction = parse_prediction(row["response"])
        result["parse_ok"] = True
    except (IndexError, KeyError, TypeError, ValueError):
        result["semantic_error"] = "parse_failure"
        result["error_type"] = "parse_failure"
        return result

    action = prediction.get("action")
    result["pred_action"] = action
    result["pred_param"] = prediction.get("value")
    result["position"] = prediction.get("position")
    if action not in ACTION_TO_ID:
        result["semantic_error"] = "unsupported_action"
        result["error_type"] = "unsupported_action"
        return result
    result["supported_action"] = True
    if parser_kind in {"showui", "tongui", "cogagent"} and result["position"] is None:
        result["semantic_error"] = "missing_position"
        result["error_type"] = "missing_position"
        return result

    bbox = normalized_bbox(row, rounded=parser_kind in {"showui", "tongui"})
    position = result["position"]
    if position is not None:
        result["element"] = float(
            bbox[0] <= position[0] <= bbox[2] and bbox[1] <= position[1] <= bbox[3]
        )
    predicted_operation = str(ACTION_TO_ID[action])
    if action in {"TYPE", "SELECT"}:
        value = prediction.get("value")
        if not isinstance(value, str):
            result["semantic_error"] = "missing_parameter"
            result["error_type"] = "missing_parameter"
            return result
        predicted_operation += " " + value.lower()
    answer = row["answer"]
    reference_operation = str(ACTION_TO_ID[answer["action"]])
    if answer["action"] in {"TYPE", "SELECT"}:
        reference_operation += " " + answer["value"].lower()
    result["operation_f1"] = token_f1(predicted_operation, reference_operation)
    result["step_success"] = bool(result["operation_f1"] == 1.0 and result["element"] == 1.0)
    result["bbox_distance"] = distance_to_bbox(position, bbox)
    if result["step_success"]:
        error_type = "success"
    elif action != answer["action"]:
        error_type = "action_mismatch"
    elif result["operation_f1"] != 1.0:
        error_type = "parameter_miss"
    elif position is None:
        error_type = "missing_position"
    else:
        error_type = "element_miss"
    result["error_type"] = error_type
    return result