import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ACTION_TO_ID = {"CLICK": 4, "SELECT": 2, "TYPE": 3}
PREDICTION_PATTERN = re.compile(
    r"'action':\s*'(.*?)',\s*'value':\s*(None|'(.*?)'),\s*"
    r"'position':\s*(None|\[([0-9.]+),\s*([0-9.]+)\])"
)


def parse_prediction(prediction: str) -> dict:
    match = PREDICTION_PATTERN.search(prediction.replace('"', "'"))
    if not match:
        raise ValueError("prediction does not match the released ShowUI parser")
    return {
        "action": match.group(1),
        "value": None if match.group(2) == "None" else match.group(3),
        "position": None
        if match.group(4) == "None"
        else [float(match.group(5)), float(match.group(6))],
    }


def token_f1(prediction: str, reference: str) -> float:
    pred = set(prediction.strip().split())
    ref = set(reference.strip().split())
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    overlap = len(pred & ref)
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    if precision == 0 or recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return sum(values) / len(values)


def score_rows(rows: list[dict]) -> dict:
    if len(rows) != 2080:
        raise ValueError(f"expected 2080 rows, found {len(rows)}")
    if [row["index"] for row in rows] != list(range(2080)):
        raise ValueError("rows are not in exact prepared order")

    episodes = defaultdict(list)
    category_f1 = defaultdict(list)
    parse_success = 0
    supported_action_count = 0
    parsed_action_counts = Counter()
    for row in rows:
        answer = row["answer"]
        result = {"element": 0.0, "operation_f1": 0.0, "step": 0.0}
        try:
            prediction = parse_prediction(row["response"])
            parse_success += 1
            action = prediction["action"]
            parsed_action_counts[action] += 1
            if action not in ACTION_TO_ID:
                raise ValueError(f"unsupported action {action}")
            supported_action_count += 1
            position = prediction["position"]
            if position is None:
                raise ValueError("position is required by Mind2Web")
            width, height = row["image_size"]
            bbox = row["bbox"]
            normalized_bbox = [
                round(bbox["x"] / width, 3),
                round(bbox["y"] / height, 3),
                round((bbox["x"] + bbox["width"]) / width, 3),
                round((bbox["y"] + bbox["height"]) / height, 3),
            ]
            result["element"] = float(
                normalized_bbox[0] <= position[0] <= normalized_bbox[2]
                and normalized_bbox[1] <= position[1] <= normalized_bbox[3]
            )

            pred_operation = str(ACTION_TO_ID[action])
            if action in {"TYPE", "SELECT"}:
                if prediction["value"] is None:
                    raise ValueError("value is required for TYPE/SELECT")
                pred_operation += " " + prediction["value"].lower()
            ref_operation = str(ACTION_TO_ID[answer["action"]])
            if answer["action"] in {"TYPE", "SELECT"}:
                ref_operation += " " + answer["value"].lower()
            result["operation_f1"] = token_f1(pred_operation, ref_operation)
            result["step"] = float(
                result["operation_f1"] == 1.0 and result["element"] == 1.0
            )
        except (KeyError, TypeError, ValueError):
            pass
        category_f1[answer["action"]].append(result["operation_f1"])
        episodes[row["annot_id"]].append(result)

    if set(category_f1) != set(ACTION_TO_ID):
        raise ValueError(f"unexpected GT action categories: {sorted(category_f1)}")
    flat = [result for episode in episodes.values() for result in episode]
    metrics = {
        "rows": len(rows),
        "episodes": len(episodes),
        "parse_success": parse_success,
        "parse_rate": parse_success / len(rows),
        "supported_action_count": supported_action_count,
        "supported_action_rate": supported_action_count / len(rows),
        "parsed_action_counts": dict(sorted(parsed_action_counts.items())),
        "element_accuracy_micro": mean([item["element"] for item in flat]),
        "step_success_micro": mean([item["step"] for item in flat]),
        "operation_f1_action_category_macro": mean(
            [mean(category_f1[action]) for action in ACTION_TO_ID]
        ),
        "operation_f1_by_category": {
            action: mean(category_f1[action]) for action in ACTION_TO_ID
        },
        "macro_element_accuracy": mean(
            [mean([item["element"] for item in episode]) for episode in episodes.values()]
        ),
        "macro_operation_f1": mean(
            [
                mean([item["operation_f1"] for item in episode])
                for episode in episodes.values()
            ]
        ),
        "macro_step_success_rate": mean(
            [mean([item["step"] for item in episode]) for episode in episodes.values()]
        ),
        "released_evaluator_identity_diagnostic": {
            "behavior": "released dataset overwrites anno_id with step index",
            "pseudo_macro_element_accuracy": mean(
                [item["element"] for item in flat]
            ),
            "pseudo_macro_operation_f1": mean(
                [item["operation_f1"] for item in flat]
            ),
            "pseudo_macro_step_success_rate": mean(
                [item["step"] for item in flat]
            ),
        },
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.predictions.open() as handle:
        rows = [json.loads(line) for line in handle]
    metrics = score_rows(rows)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
