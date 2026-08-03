import argparse
import hashlib
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

from scoring import ACTION_TO_ID, token_f1


ROOT = Path(__file__).resolve().parents[4]
SCORE_PATH = ROOT / "runs/mind2web-tongui/2026-07-28/score.py"


def load_parser():
    spec = importlib.util.spec_from_file_location("collision_w2_tongui_score", SCORE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(SCORE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_prediction


def score_row(row, parse_prediction):
    result = {
        "parse_ok": False, "pred_action": None, "pred_param": "",
        "pred_x": None, "pred_y": None, "element": 0.0,
        "operation_f1": 0.0, "success": False, "error_label": "parse_failure",
    }
    try:
        prediction = parse_prediction(row["response"])
        action = prediction["action"]
        position = prediction["position"]
        if action not in ACTION_TO_ID:
            result["error_label"] = "unsupported_action"
            return result
        if position is None:
            result["error_label"] = "missing_position"
            return result
        result["parse_ok"] = True
        result["pred_action"] = action
        result["pred_param"] = prediction.get("value") or ""
        view_width, view_height = row["view_size"]
        offset_x, offset_y = row["view_offset"]
        original_width, original_height = row["image_size"]
        original_x = (position[0] * view_width + offset_x) / original_width
        original_y = (position[1] * view_height + offset_y) / original_height
        result["pred_x"], result["pred_y"] = original_x, original_y
        bbox = row["bbox"]
        normalized_bbox = [
            round(bbox["x"] / original_width, 3),
            round(bbox["y"] / original_height, 3),
            round((bbox["x"] + bbox["width"]) / original_width, 3),
            round((bbox["y"] + bbox["height"]) / original_height, 3),
        ]
        result["element"] = float(
            normalized_bbox[0] <= original_x <= normalized_bbox[2]
            and normalized_bbox[1] <= original_y <= normalized_bbox[3]
        )
        predicted_operation = str(ACTION_TO_ID[action])
        if action in {"TYPE", "SELECT"}:
            predicted_operation += " " + result["pred_param"].lower()
        answer = row["answer"]
        expected_operation = str(ACTION_TO_ID[answer["action"]])
        if answer["action"] in {"TYPE", "SELECT"}:
            expected_operation += " " + answer["value"].lower()
        result["operation_f1"] = token_f1(predicted_operation, expected_operation)
        result["success"] = bool(result["element"] == 1.0 and result["operation_f1"] == 1.0)
        result["error_label"] = (
            "success" if result["success"]
            else "action_mismatch" if action != answer["action"]
            else "parameter_miss" if result["operation_f1"] != 1.0
            else "element_miss"
        )
    except (IndexError, KeyError, TypeError, ValueError):
        pass
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scored-rows", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    if args.require_complete and (len(rows) != 2080 or [row["index"] for row in rows] != list(range(2080))):
        raise ValueError("complete W2 Mind2Web scoring requires ordered indices 0..2079")
    parse_prediction = load_parser()
    scores = [score_row(row, parse_prediction) for row in rows]
    args.scored_rows.parent.mkdir(parents=True, exist_ok=True)
    with args.scored_rows.open("w") as output:
        for row, score in zip(rows, scores):
            output.write(json.dumps({
                "index": row["index"], "annot_id": row["annot_id"], "action_uid": row["action_uid"],
                "model": row["model"], "view_id": row["view_id"], **score,
            }, ensure_ascii=True) + "\n")
    result = {
        "status": "PASS", "coverage": "COMPLETE" if len(rows) == 2080 else "PARTIAL",
        "rows": len(rows), "model": rows[0]["model"] if rows else None,
        "view_id": rows[0]["view_id"] if rows else None,
        "parse_successes": sum(score["parse_ok"] for score in scores),
        "element_accuracy": sum(score["element"] for score in scores) / len(scores),
        "operation_f1": sum(score["operation_f1"] for score in scores) / len(scores),
        "step_successes": sum(score["success"] for score in scores),
        "step_sr": sum(score["success"] for score in scores) / len(scores),
        "error_counts": dict(sorted(Counter(score["error_label"] for score in scores).items())),
        "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "scored_rows_sha256": hashlib.sha256(args.scored_rows.read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()