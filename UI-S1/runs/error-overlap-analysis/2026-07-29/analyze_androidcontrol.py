import argparse
import json
from collections import Counter, defaultdict, deque
from itertools import combinations
from pathlib import Path
import sys

SCORING_DIR = Path(__file__).resolve().parents[2] / "complementarity/2026-07-30"
sys.path.insert(0, str(SCORING_DIR))
from scoring import android_metric_counts, label_android_row, read_jsonl, transition
MODELS = (
    "ui-agile-3b",
    "ui-agile-7b",
    "ui-r1-e-3b",
    "gui-r1-3b",
    "gui-r1-7b",
)
SETTINGS = ("low", "high")
SIZE_PAIRS = (
    ("ui-agile-3b", "ui-agile-7b"),
    ("gui-r1-3b", "gui-r1-7b"),
)


def pairwise_failure_overlap(model_labels: dict[str, list[dict]]) -> list[dict]:
    output = []
    for left, right in combinations(model_labels, 2):
        left_fail = {i for i, item in enumerate(model_labels[left]) if not item["step_success"]}
        right_fail = {i for i, item in enumerate(model_labels[right]) if not item["step_success"]}
        intersection = len(left_fail & right_fail)
        union = len(left_fail | right_fail)
        shared_error_types = Counter(
            model_labels[left][index]["error_type"]
            for index in left_fail & right_fail
            if model_labels[left][index]["error_type"]
            == model_labels[right][index]["error_type"]
        )
        output.append({
            "left": left,
            "right": right,
            "left_fail": len(left_fail),
            "right_fail": len(right_fail),
            "intersection": intersection,
            "jaccard": intersection / union if union else 1.0,
            "same_error_type": sum(shared_error_types.values()),
            "same_error_type_rate_within_intersection": (
                sum(shared_error_types.values()) / intersection if intersection else 1.0
            ),
            "shared_error_type_counts": dict(sorted(shared_error_types.items())),
        })
    return output


def overlap_summary(rows: list[dict], model_labels: dict[str, list[dict]]) -> dict:
    success_counts = [
        sum(model_labels[model][index]["step_success"] for model in model_labels)
        for index in range(len(rows))
    ]
    histogram = Counter(success_counts)
    hard_core_dominant = Counter()
    hard_core_unanimous = Counter()
    for index, success_count in enumerate(success_counts):
        if success_count:
            continue
        votes = Counter(model_labels[model][index]["error_type"] for model in model_labels)
        highest = max(votes.values())
        winners = sorted(error_type for error_type, count in votes.items() if count == highest)
        hard_core_dominant[winners[0] if len(winners) == 1 else "tie"] += 1
        if len(votes) == 1:
            hard_core_unanimous[next(iter(votes))] += 1
    by_action = {}
    for action in sorted({row["gt_action"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["gt_action"] == action]
        action_counts = [success_counts[index] for index in indices]
        by_action[action] = {
            "rows": len(indices),
            "all_models_fail": sum(count == 0 for count in action_counts),
            "oracle_success": sum(count > 0 for count in action_counts),
            "all_models_success": sum(count == len(model_labels) for count in action_counts),
        }
    hardest = sorted(
        range(len(rows)),
        key=lambda index: (success_counts[index], rows[index]["gt_action"], index),
    )[:100]
    return {
        "rows": len(rows),
        "models": len(model_labels),
        "all_models_fail": histogram[0],
        "oracle_success": len(rows) - histogram[0],
        "all_models_success": histogram[len(model_labels)],
        "success_count_histogram": {str(key): histogram[key] for key in sorted(histogram)},
        "hard_core_dominant_error_type": dict(sorted(hard_core_dominant.items())),
        "hard_core_unanimous_error_type": dict(sorted(hard_core_unanimous.items())),
        "by_gt_action": by_action,
        "pairwise_failure_overlap": pairwise_failure_overlap(model_labels),
        "hardest_examples": [
            {
                "index": index,
                "image_sha256": rows[index]["image_sha256"],
                "gt_action": rows[index]["gt_action"],
                "gt_input_text": rows[index]["gt_input_text"],
                "successful_models": [
                    model for model in model_labels
                    if model_labels[model][index]["step_success"]
                ],
            }
            for index in hardest
        ],
    }


def cross_setting_pairs(low_rows: list[dict], high_rows: list[dict]) -> tuple[list[tuple[int, int]], list[dict]]:
    def base_key(row: dict):
        return row["image_sha256"], row["gt_action"], tuple(row["gt_bbox"])

    low_groups = defaultdict(list)
    high_groups = defaultdict(list)
    for index, row in enumerate(low_rows):
        low_groups[base_key(row)].append(index)
    for index, row in enumerate(high_rows):
        high_groups[base_key(row)].append(index)
    if Counter({key: len(value) for key, value in low_groups.items()}) != Counter(
        {key: len(value) for key, value in high_groups.items()}
    ):
        raise ValueError("Low/High base-key multisets differ")

    matched = []
    conflicts = []
    for key in sorted(low_groups, key=repr):
        high_by_text = defaultdict(deque)
        for high_index in high_groups[key]:
            high_by_text[high_rows[high_index]["gt_input_text"]].append(high_index)
        unmatched_low = []
        for low_index in low_groups[key]:
            candidates = high_by_text[low_rows[low_index]["gt_input_text"]]
            if candidates:
                matched.append((low_index, candidates.popleft()))
            else:
                unmatched_low.append(low_index)
        unmatched_high = [index for queue in high_by_text.values() for index in queue]
        if len(unmatched_low) != len(unmatched_high):
            raise ValueError("duplicate-aware Low/High matching failed")
        for low_index, high_index in zip(sorted(unmatched_low), sorted(unmatched_high)):
            conflicts.append({
                "low_index": low_index,
                "high_index": high_index,
                "image_sha256": low_rows[low_index]["image_sha256"],
                "gt_action": low_rows[low_index]["gt_action"],
                "low_gt_input_text": low_rows[low_index]["gt_input_text"],
                "high_gt_input_text": high_rows[high_index]["gt_input_text"],
            })
    if len(matched) + len(conflicts) != 7708:
        raise ValueError("Low/High matching does not cover 7,708 rows")
    return matched, conflicts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/androidcontrol-rft/2026-07-29"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows_by_setting = {}
    labels = defaultdict(dict)
    model_results = {}
    for model in MODELS:
        model_results[model] = {}
        for setting in SETTINGS:
            artifact_dir = args.run_dir / "artifacts" / model / setting
            rows = read_jsonl(artifact_dir / "predictions.jsonl")
            if len(rows) != 7708 or [row["index"] for row in rows] != list(range(7708)):
                raise ValueError(f"incomplete or unordered trace: {model}/{setting}")
            if setting not in rows_by_setting:
                rows_by_setting[setting] = rows
            else:
                reference = rows_by_setting[setting]
                source_fields = ("image_sha256", "gt_action", "gt_bbox", "gt_input_text")
                if any(
                    tuple(row[field] for field in source_fields)
                    != tuple(reference[index][field] for field in source_fields)
                    for index, row in enumerate(rows)
                ):
                    raise ValueError(f"within-setting source mismatch: {model}/{setting}")
            row_labels = [label_android_row(row) for row in rows]
            labels[setting][model] = row_labels
            recomputed = android_metric_counts(row_labels, rows)
            score = json.loads((artifact_dir / "score.json").read_text())
            expected = {
                metric: {
                    "correct": score["metrics"][metric]["correct"],
                    "total": score["metrics"][metric]["total"],
                }
                for metric in recomputed
            }
            if recomputed != expected:
                raise ValueError(f"evaluator parity failure: {model}/{setting}")
            error_counts = Counter(label["error_type"] for label in row_labels)
            grounding_misses = [
                label["normalized_distance"] for label in row_labels
                if label["error_type"] == "grounding_miss"
            ]
            model_results[model][setting] = {
                "evaluator_parity": "PASS",
                "metric_counts": recomputed,
                "error_type_counts": dict(sorted(error_counts.items())),
                "grounding_miss_distance": {
                    "under_0.28": sum(distance < 0.28 for distance in grounding_misses),
                    "at_least_0.28": sum(distance >= 0.28 for distance in grounding_misses),
                },
            }

    matched, conflicts = cross_setting_pairs(rows_by_setting["low"], rows_by_setting["high"])
    low_to_high = {}
    comparable_low_indices = [low for low, _ in matched]
    for model in MODELS:
        remapped_high = [None] * 7708
        for low_index, high_index in matched:
            remapped_high[low_index] = labels["high"][model][high_index]
        low_to_high[model] = transition(
            labels["low"][model], remapped_high, comparable_low_indices
        )

    size_effects = {}
    for setting in SETTINGS:
        size_effects[setting] = {
            f"{left}_to_{right}": transition(labels[setting][left], labels[setting][right])
            for left, right in SIZE_PAIRS
        }

    result = {
        "status": "PASS",
        "contract": {
            "rows_per_setting": 7708,
            "models": list(MODELS),
            "grounding_radius": 0.14,
            "parameter_f1_threshold": 0.5,
            "cross_setting_identity": ["image_sha256", "gt_action", "gt_bbox"],
            "cross_setting_parameter_conflicts_excluded": len(conflicts),
        },
        "models": model_results,
        "within_setting_overlap": {
            setting: overlap_summary(rows_by_setting[setting], labels[setting])
            for setting in SETTINGS
        },
        "cross_setting": {
            "exact_parameter_matches": len(matched),
            "gt_parameter_conflicts": conflicts,
            "low_to_high_transitions_excluding_conflicts": low_to_high,
        },
        "size_effects": size_effects,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "low_all_fail": result["within_setting_overlap"]["low"]["all_models_fail"],
        "high_all_fail": result["within_setting_overlap"]["high"]["all_models_fail"],
        "parameter_conflicts": len(conflicts),
    }, indent=2))


if __name__ == "__main__":
    main()