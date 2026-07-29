import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from common import parse_prediction, read_jsonl


def token_f1(predicted: str, reference: str) -> float:
    predicted_tokens = set(predicted.lower().split())
    reference_tokens = set(reference.lower().split())
    common = predicted_tokens & reference_tokens
    precision = len(common) / len(predicted_tokens) if predicted_tokens else 0
    recall = len(common) / len(reference_tokens) if reference_tokens else 0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0


def argument(action: str) -> str:
    match = re.search(r"\[(.*)\]", action)
    return match.group(1) if match else ""


def evaluate(rows: list[dict], require_complete: bool) -> dict:
    indices = [row["index"] for row in rows]
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError("indices must be unique and ordered")
    if require_complete and indices != list(range(7708)):
        raise ValueError("complete scoring requires indices 0..7707")
    counts = Counter()
    parse_errors = Counter()
    per_type = Counter()
    per_type_success = Counter()
    coordinate_type_matches = 0
    coordinate_success = 0
    for row in rows:
        reference = row["gt_action"]
        reference_type = reference.split()[0]
        per_type[reference_type] += 1
        counts["rows"] += 1
        try:
            prediction = parse_prediction(row["response"])
            counts["parsed"] += 1
        except (KeyError, TypeError, ValueError) as error:
            parse_errors[str(error)] += 1
            continue
        prediction_type = prediction.split()[0]
        if prediction_type != reference_type:
            continue
        counts["type_match"] += 1
        success = False
        if reference_type in {"CLICK", "LONG_PRESS"}:
            predicted_numbers = re.findall(r"\d+", prediction)
            reference_numbers = re.findall(r"\d+", reference)
            if len(predicted_numbers) >= 2 and len(reference_numbers) >= 2:
                distance = math.hypot(
                    int(predicted_numbers[0]) - int(reference_numbers[0]),
                    int(predicted_numbers[1]) - int(reference_numbers[1]),
                )
                success = distance <= 140
            coordinate_type_matches += 1
            coordinate_success += success
        elif reference_type in {"OPEN_APP", "TYPE"}:
            success = prediction == reference or token_f1(argument(prediction), argument(reference)) > 0.5
        else:
            success = prediction == reference
        if success:
            counts["success"] += 1
            counts[f"{reference_type}_success"] += 1
            per_type_success[reference_type] += 1
    total = counts["rows"]
    click_type_matches = sum(
        1
        for row in rows
        if row["gt_action"].startswith("CLICK ")
        and _type_matches(row, "CLICK")
    )
    return {
        "coverage": "COMPLETE" if total == 7708 else "PARTIAL",
        "rows": total,
        "episodes": len({row["episode_id"] for row in rows}),
        "parse_success": counts["parsed"],
        "parse_rate": counts["parsed"] / total,
        "parse_errors": dict(sorted(parse_errors.items())),
        "type_matches": counts["type_match"],
        "type_accuracy": counts["type_match"] / total,
        "step_successes": counts["success"],
        "step_success_rate": counts["success"] / total,
        "click_only_grounding": counts["CLICK_success"] / click_type_matches if click_type_matches else 0,
        "coordinate_grounding_including_long_press": coordinate_success / coordinate_type_matches if coordinate_type_matches else 0,
        "per_type_count": dict(sorted(per_type.items())),
        "per_type_success": dict(sorted(per_type_success.items())),
    }


def _type_matches(row: dict, expected: str) -> bool:
    try:
        return parse_prediction(row["response"]).split()[0] == expected
    except (KeyError, TypeError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = evaluate(read_jsonl(args.predictions), args.require_complete)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()