import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import parse_prediction, read_jsonl


ACTION_TO_ID = {"CLICK": 4, "SELECT": 2, "TYPE": 3}


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
    return 0.0 if not overlap else 2 * precision * recall / (precision + recall)


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average empty values")
    return sum(values) / len(values)


def score_rows(rows: list[dict], require_complete: bool) -> dict:
    indices = [row["index"] for row in rows]
    if len(indices) != len(set(indices)) or indices != sorted(indices):
        raise ValueError("prediction indices must be unique and ordered")
    if require_complete and indices != list(range(2080)):
        raise ValueError("complete scoring requires exactly indices 0..2079")

    episodes = defaultdict(list)
    category_f1 = defaultdict(list)
    parsed_actions = Counter()
    parse_errors = Counter()
    parse_success = 0
    for row in rows:
        answer = row["answer"]
        result = {"element": 0.0, "operation_f1": 0.0, "step": 0.0}
        try:
            prediction = parse_prediction(row["response"])
            parse_success += 1
            action = prediction["action"]
            parsed_actions[action] += 1
            position = prediction["position"]
            width, height = row["image_size"]
            bbox = row["bbox"]
            normalized_bbox = [
                bbox["x"] / width,
                bbox["y"] / height,
                (bbox["x"] + bbox["width"]) / width,
                (bbox["y"] + bbox["height"]) / height,
            ]
            result["element"] = float(
                normalized_bbox[0] <= position[0] <= normalized_bbox[2]
                and normalized_bbox[1] <= position[1] <= normalized_bbox[3]
            )
            predicted_operation = str(ACTION_TO_ID[action])
            if action in {"TYPE", "SELECT"}:
                predicted_operation += " " + prediction["value"].lower()
            reference_operation = str(ACTION_TO_ID[answer["action"]])
            if answer["action"] in {"TYPE", "SELECT"}:
                reference_operation += " " + answer["value"].lower()
            result["operation_f1"] = token_f1(predicted_operation, reference_operation)
            result["step"] = float(result["operation_f1"] == 1.0 and result["element"] == 1.0)
        except (KeyError, TypeError, ValueError) as error:
            parse_errors[str(error)] += 1
        category_f1[answer["action"]].append(result["operation_f1"])
        episodes[row["annot_id"]].append(result)

    flat = [item for episode in episodes.values() for item in episode]
    return {
        "coverage": "COMPLETE" if len(rows) == 2080 else "PARTIAL",
        "rows": len(rows),
        "episodes": len(episodes),
        "parse_success": parse_success,
        "parse_rate": parse_success / len(rows),
        "parsed_action_counts": dict(sorted(parsed_actions.items())),
        "parse_errors": dict(sorted(parse_errors.items())),
        "element_accuracy_micro": mean([item["element"] for item in flat]),
        "operation_f1_micro": mean([item["operation_f1"] for item in flat]),
        "step_success_micro": mean([item["step"] for item in flat]),
        "operation_f1_by_category": {
            action: mean(category_f1[action])
            for action in ACTION_TO_ID
            if category_f1[action]
        },
        "macro_element_accuracy": mean(
            [mean([item["element"] for item in episode]) for episode in episodes.values()]
        ),
        "macro_operation_f1": mean(
            [mean([item["operation_f1"] for item in episode]) for episode in episodes.values()]
        ),
        "macro_step_success_rate": mean(
            [mean([item["step"] for item in episode]) for episode in episodes.values()]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    metrics = score_rows(read_jsonl(args.predictions), args.require_complete)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()