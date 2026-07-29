import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


def token_f1(predicted, reference):
    predicted_tokens = set(predicted.lower().split())
    reference_tokens = set(reference.lower().split())
    common = predicted_tokens & reference_tokens
    precision = len(common) / len(predicted_tokens) if predicted_tokens else 0
    recall = len(common) / len(reference_tokens) if reference_tokens else 0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0


def upstream_action(response):
    try:
        return response.split("actions:\n")[1].strip("<|im_end|>"), True
    except (IndexError, AttributeError):
        return "invalid action", False


def flexible_action(response):
    match = re.search(r"actions\s*:\s*\n?(.+)", response, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return "invalid action", False
    return match.group(1).replace("<|im_end|>", "").strip(), True


def evaluate(predictions, parser):
    counts = Counter()
    parse_success = 0
    per_type = Counter()
    per_type_success = Counter()
    coordinate_type_matches = 0
    coordinate_success = 0
    runtime_errors = []
    for row in predictions:
        prediction, parsed = parser(row["response"])
        parse_success += parsed
        reference = row["gt_action"]
        reference_type = reference.split()[0]
        prediction_type = prediction.split()[0]
        per_type[reference_type] += 1
        counts["full"] += 1
        if prediction_type != reference_type:
            continue
        counts["type_match"] += 1
        counts[f"{reference_type}_type_match"] += 1
        success = False
        try:
            if reference_type in {"CLICK", "LONG_PRESS"}:
                predicted_numbers = re.findall(r"\d+", prediction)
                reference_numbers = re.findall(r"\d+", reference)
                predicted_x, predicted_y = int(predicted_numbers[0]), int(predicted_numbers[1])
                reference_x, reference_y = int(reference_numbers[0]), int(reference_numbers[1])
                success = math.hypot(predicted_x - reference_x, predicted_y - reference_y) <= 140
                coordinate_type_matches += 1
                coordinate_success += success
            elif reference_type in {"OPEN_APP", "TYPE"}:
                success = prediction == reference or token_f1(prediction.split()[1], reference.split()[1]) > 0.5
            else:
                success = prediction == reference
        except (IndexError, TypeError, ValueError) as error:
            runtime_errors.append({"identity": row["identity"], "error": str(error)})
        if success:
            counts["success"] += 1
            counts[f"{reference_type}_all_match"] += 1
            per_type_success[reference_type] += 1

    total = counts["full"]
    click_type_matches = counts["CLICK_type_match"]
    return {
        "rows": total,
        "parse_success": parse_success,
        "parse_rate": parse_success / total,
        "type_matches": counts["type_match"],
        "type_accuracy": counts["type_match"] / total,
        "step_successes": counts["success"],
        "step_success_rate": counts["success"] / total,
        "upstream_click_only_grounding": counts["CLICK_all_match"] / click_type_matches if click_type_matches else 0,
        "coordinate_grounding_including_long_press": coordinate_success / coordinate_type_matches if coordinate_type_matches else 0,
        "per_type_count": dict(per_type),
        "per_type_success": dict(per_type_success),
        "runtime_errors": runtime_errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.predictions.open() as predictions_file:
        predictions = [json.loads(line) for line in predictions_file if line.strip()]
    identities = [row["identity"] for row in predictions]
    summary = {
        "status": "PASS" if len(identities) == len(set(identities)) else "FAIL",
        "duplicates": len(identities) - len(set(identities)),
        "upstream_exact_parser": evaluate(predictions, upstream_action),
        "flexible_parser_diagnostic": evaluate(predictions, flexible_action),
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()