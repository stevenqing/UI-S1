import ast
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from PIL import Image


ROOT = Path(__file__).resolve().parent
ANNOTATIONS = ROOT / "data/mind2web_data_test_task.json"
IMAGES = ROOT / "data/ming2web_images"
NATIVE_PREDICTIONS = ROOT / "artifacts/gate1_cross_task/predictions.jsonl"
NATIVE_SUMMARY = ROOT / "artifacts/gate1_cross_task/summary.json"
SUPPLEMENT_PREDICTIONS = ROOT / "artifacts/gate1_corrected_missing/predictions.jsonl"
OUTPUT = ROOT / "artifacts/gate1_audit.json"
ANCHOR_PERCENT = {
    "macro_element_accuracy": 28.3,
    "macro_operation_f1": 87.0,
    "macro_step_success": 25.5,
}


def read_jsonl(path):
    with path.open() as predictions_file:
        return [json.loads(line) for line in predictions_file if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_f1(prediction, reference):
    predicted_tokens = set(prediction.strip().split())
    reference_tokens = set(reference.strip().split())
    if not predicted_tokens and not reference_tokens:
        return 1.0
    if not predicted_tokens or not reference_tokens:
        return 0.0
    true_positives = len(predicted_tokens & reference_tokens)
    precision = true_positives / len(predicted_tokens)
    recall = true_positives / len(reference_tokens)
    if precision == 0 or recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def build_references(episodes):
    references = {}
    no_bbox = []
    for episode in episodes:
        annotation_id = episode["annotation_id"]
        for action in episode["actions"]:
            image_name = f'{annotation_id}-{action["action_uid"]}.jpg'
            if "bbox" not in action:
                no_bbox.append(image_name)
                continue
            image_path = IMAGES / image_name
            with Image.open(image_path) as image:
                width, height = image.size
            bbox = action["bbox"]
            operation = action["operation"]
            original_op = operation["original_op"]
            action_type = 4 if original_op in {"CLICK", "HOVER", "ENTER"} else 2 if original_op == "SELECT" else 3
            references[image_name] = {
                "annotation_id": annotation_id,
                "action_uid": action["action_uid"],
                "original_op": original_op,
                "value": operation.get("value"),
                "action_type": action_type,
                "bbox": [
                    round(bbox["x"] / width, 3),
                    round(bbox["y"] / height, 3),
                    round((bbox["x"] + bbox["width"]) / width, 3),
                    round((bbox["y"] + bbox["height"]) / height, 3),
                ],
            }
    return references, no_bbox


def score_row(row, reference):
    parse_ok = False
    element_match = False
    operation_f1 = 0.0
    try:
        prediction = ast.literal_eval(row["sentence"])
        parse_ok = True
        click_point = prediction["click_point"]
        bbox = reference["bbox"]
        element_match = bbox[0] <= click_point[0] <= bbox[2] and bbox[1] <= click_point[1] <= bbox[3]
        prediction_string = str(prediction["action_type"])
        if prediction["action_type"] in {2, 3}:
            prediction_string += " " + prediction["value"].lower()
        reference_string = str(reference["action_type"])
        if reference["action_type"] in {2, 3}:
            reference_string += " " + reference["value"].lower()
        operation_f1 = token_f1(prediction_string, reference_string)
    except (KeyError, SyntaxError, TypeError, ValueError):
        pass
    return {
        "parse_ok": parse_ok,
        "element_match": element_match,
        "operation_f1": operation_f1,
        "step_success": element_match and operation_f1 == 1.0,
    }


def evaluate(rows, references, episode_ids):
    episode_scores = defaultdict(list)
    field_mismatches = []
    for row in rows:
        image_name = Path(row["img_path"]).name
        score = score_row(row, references[image_name])
        episode_scores[references[image_name]["annotation_id"]].append(score)
        recorded = {
            "parse_ok": row["parse_ok"],
            "element_match": row["Ele_match"],
            "operation_f1": row["Op_F1"][0],
        }
        recomputed = {key: score[key] for key in recorded}
        if recorded != recomputed:
            field_mismatches.append({"image": image_name, "recorded": recorded, "recomputed": recomputed})
    if set(episode_scores) != set(episode_ids):
        raise RuntimeError("prediction rows do not cover every episode")
    flat_scores = [score for scores in episode_scores.values() for score in scores]
    return {
        "num_steps": len(flat_scores),
        "num_episodes": len(episode_scores),
        "parse_success": sum(score["parse_ok"] for score in flat_scores),
        "parse_rate": mean(score["parse_ok"] for score in flat_scores),
        "micro_element_accuracy": mean(score["element_match"] for score in flat_scores),
        "micro_operation_f1": mean(score["operation_f1"] for score in flat_scores),
        "micro_step_success": mean(score["step_success"] for score in flat_scores),
        "macro_element_accuracy": mean(mean(score["element_match"] for score in episode_scores[episode_id]) for episode_id in episode_ids),
        "macro_operation_f1": mean(mean(score["operation_f1"] for score in episode_scores[episode_id]) for episode_id in episode_ids),
        "macro_step_success": mean(mean(score["step_success"] for score in episode_scores[episode_id]) for episode_id in episode_ids),
        "recorded_field_mismatches": field_mismatches,
    }


def main():
    episodes = json.loads(ANNOTATIONS.read_text())
    references, no_bbox = build_references(episodes)
    native_rows = read_jsonl(NATIVE_PREDICTIONS)
    supplement_rows = read_jsonl(SUPPLEMENT_PREDICTIONS)
    episode_ids = [episode["annotation_id"] for episode in episodes]
    native_names = [Path(row["img_path"]).name for row in native_rows]
    supplement_names = [Path(row["img_path"]).name for row in supplement_rows]
    missing_native = sorted(set(references) - set(native_names))
    duplicate_native = len(native_names) - len(set(native_names))
    if duplicate_native != 0 or len(missing_native) != 1 or supplement_names != missing_native:
        raise RuntimeError("native/supplement prediction coverage does not match the diagnosed skip")

    native_metrics = evaluate(native_rows, references, episode_ids)
    corrected_metrics = evaluate(native_rows + supplement_rows, references, episode_ids)
    native_summary = json.loads(NATIVE_SUMMARY.read_text())
    comparable_summary_keys = [
        "num_steps",
        "num_episodes",
        "parse_success",
        "parse_rate",
        "micro_element_accuracy",
        "micro_step_success",
        "macro_element_accuracy",
        "macro_operation_f1",
        "macro_step_success",
    ]
    summary_deltas = {
        key: native_metrics[key] - native_summary[key]
        for key in comparable_summary_keys
    }
    audit = {
        "status": "PASS",
        "input_counts": {
            "episodes": len(episodes),
            "actions": sum(len(episode["actions"]) for episode in episodes),
            "bbox_actions": len(references),
            "no_bbox_actions": len(no_bbox),
            "native_predictions": len(native_rows),
            "native_unique_images": len(set(native_names)),
            "native_duplicates": duplicate_native,
            "native_missing_bbox_actions": missing_native,
            "supplement_predictions": len(supplement_rows),
            "corrected_predictions": len(native_rows) + len(supplement_rows),
        },
        "diagnosed_skip": {
            "image": missing_native[0],
            "operation": references[missing_native[0]]["original_op"],
            "value": references[missing_native[0]]["value"],
            "cause": "Upstream action2step interpolated an unescaped double quote into a Python literal; ast.literal_eval raised SyntaxError and the broad except silently continued.",
        },
        "native_metrics_independent": native_metrics,
        "native_summary_deltas": summary_deltas,
        "native_summary_exact_within_1e_12": all(abs(delta) <= 1e-12 for delta in summary_deltas.values()),
        "corrected_metrics_independent": corrected_metrics,
        "paper_anchor_percent": ANCHOR_PERCENT,
        "corrected_anchor_delta_percentage_points": {
            key: corrected_metrics[key] * 100 - anchor
            for key, anchor in ANCHOR_PERCENT.items()
        },
        "within_one_percentage_point": all(
            abs(corrected_metrics[key] * 100 - anchor) <= 1.0
            for key, anchor in ANCHOR_PERCENT.items()
        ),
        "artifact_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [NATIVE_PREDICTIONS, NATIVE_SUMMARY, SUPPLEMENT_PREDICTIONS]
        },
    }
    if native_metrics["recorded_field_mismatches"] or corrected_metrics["recorded_field_mismatches"]:
        audit["status"] = "FAIL"
    if not audit["native_summary_exact_within_1e_12"] or not audit["within_one_percentage_point"]:
        audit["status"] = "FAIL"
    OUTPUT.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()