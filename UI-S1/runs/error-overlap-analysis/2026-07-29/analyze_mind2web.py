import argparse
import ast
import importlib.util
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

SCORING_DIR = Path(__file__).resolve().parents[2] / "complementarity/2026-07-30"
sys.path.insert(0, str(SCORING_DIR))
from scoring import ACTION_TO_ID, read_jsonl, score_mind2web_row, token_f1, transition
MODEL_SPECS = {
    "seeclick-9.6b": ("seeclick", None, None),
    "showui-2b": (
        "showui", "runs/mind2web-showui/2026-07-28/artifacts/merged/predictions.jsonl",
        "runs/mind2web-showui/2026-07-28/artifacts/merged/score.json",
    ),
    "cogagent-18b": (
        "cogagent", "runs/mind2web-cogagent/2026-07-28/artifacts/merged/predictions.jsonl",
        "runs/mind2web-cogagent/2026-07-28/artifacts/merged/score.json",
    ),
    "qwen2.5-vl-3b": (
        "tongui", "runs/mind2web-tongui/2026-07-28/artifacts/qwen-3b/merged/predictions.jsonl",
        "runs/mind2web-tongui/2026-07-28/artifacts/qwen-3b/merged/score.json",
    ),
    "qwen2.5-vl-7b": (
        "tongui", "runs/mind2web-tongui/2026-07-28/artifacts/qwen-7b/merged/predictions.jsonl",
        "runs/mind2web-tongui/2026-07-28/artifacts/qwen-7b/merged/score.json",
    ),
    "tongui-3b": (
        "tongui", "runs/mind2web-tongui/2026-07-28/artifacts/tongui-3b/merged/predictions.jsonl",
        "runs/mind2web-tongui/2026-07-28/artifacts/tongui-3b/merged/score.json",
    ),
    "tongui-7b": (
        "tongui", "runs/mind2web-tongui/2026-07-28/artifacts/tongui-7b/merged/predictions.jsonl",
        "runs/mind2web-tongui/2026-07-28/artifacts/tongui-7b/merged/score.json",
    ),
    "tongui-32b": (
        "tongui", "runs/mind2web-tongui/2026-07-28/artifacts/tongui-32b/full/predictions.jsonl",
        "runs/mind2web-tongui/2026-07-28/artifacts/tongui-32b/full/score.json",
    ),
    "ui-tars-2b": (
        "uitars", "runs/mind2web-uitars/2026-07-28/artifacts/2b/merged/predictions.jsonl",
        "runs/mind2web-uitars/2026-07-28/artifacts/2b/merged/score.json",
    ),
    "ui-tars-7b": (
        "uitars", "runs/mind2web-uitars/2026-07-28/artifacts/7b/merged/predictions.jsonl",
        "runs/mind2web-uitars/2026-07-28/artifacts/7b/merged/score.json",
    ),
    "ui-tars-72b": (
        "uitars", "runs/mind2web-uitars/2026-07-28/artifacts/72b/full/predictions.jsonl",
        "runs/mind2web-uitars/2026-07-28/artifacts/72b/full/score.json",
    ),
}
FAMILIES = {
    "SeeClick": ("seeclick-9.6b",),
    "ShowUI": ("showui-2b",),
    "CogAgent": ("cogagent-18b",),
    "Qwen2.5-VL": ("qwen2.5-vl-3b", "qwen2.5-vl-7b"),
    "TongUI": ("tongui-3b", "tongui-7b", "tongui-32b"),
    "UI-TARS": ("ui-tars-2b", "ui-tars-7b", "ui-tars-72b"),
}
SCALE_PAIRS = (
    ("qwen2.5-vl-3b", "qwen2.5-vl-7b"),
    ("tongui-3b", "tongui-7b"),
    ("tongui-7b", "tongui-32b"),
    ("tongui-3b", "tongui-32b"),
    ("ui-tars-2b", "ui-tars-7b"),
    ("ui-tars-7b", "ui-tars-72b"),
    ("ui-tars-2b", "ui-tars-72b"),
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity(row: dict) -> tuple[str, str]:
    return row["annot_id"], row["action_uid"]


def load_seeclick(root: Path, reference_rows: list[dict]) -> list[dict]:
    run_dir = root / "runs/mind2web/2026-07-27"
    paths = (
        run_dir / "artifacts/gate1_cross_task/predictions.jsonl",
        run_dir / "artifacts/gate1_corrected_missing/predictions.jsonl",
    )
    by_identity = {}
    for path in paths:
        for row in read_jsonl(path):
            stem = Path(row["img_path"]).stem
            annot_id, action_uid = stem.split("-", 5)[:5], None
            annot_id = "-".join(annot_id)
            action_uid = stem[len(annot_id) + 1:]
            key = (annot_id, action_uid)
            if key in by_identity:
                raise ValueError(f"duplicate SeeClick identity: {key}")
            by_identity[key] = row
    output = []
    for reference in reference_rows:
        row = by_identity[identity(reference)]
        try:
            prediction = ast.literal_eval(row["sentence"])
            pred_action = {4: "CLICK", 2: "SELECT", 3: "TYPE"}.get(prediction["action_type"])
        except (KeyError, SyntaxError, TypeError, ValueError):
            pred_action = None
        operation_f1 = float(row["Op_F1"][0])
        element = float(row["Ele_match"])
        step_success = bool(element == 1.0 and operation_f1 == 1.0)
        if step_success:
            error_type = "success"
        elif not row["parse_ok"]:
            error_type = "parse_failure"
        elif pred_action != reference["answer"]["action"]:
            error_type = "action_mismatch"
        elif operation_f1 != 1.0:
            error_type = "parameter_miss"
        else:
            error_type = "element_miss"
        output.append({
            "parse_ok": bool(row["parse_ok"]),
            "supported_action": pred_action in ACTION_TO_ID,
            "element": element,
            "operation_f1": operation_f1,
            "step_success": step_success,
            "pred_action": pred_action,
            "position": None,
            "semantic_error": None,
            "error_type": error_type,
            "bbox_distance": None,
        })
    if len(by_identity) != len(output):
        raise ValueError("SeeClick identity set differs from visual reference")
    return output


def load_mindact(root: Path, reference_rows: list[dict]) -> tuple[list[dict], dict]:
    run_dir = root / "runs/mindact/2026-07-29"
    source_dir = root / "runs/mind2web/2026-07-27/repos/Mind2Web/src"
    sys.path.insert(0, str((source_dir / "action_prediction").resolve()))
    sys.path.insert(0, str(source_dir.resolve()))
    from dataloader import get_data_split

    with (run_dir / "data/source/scores_all_data.pkl").open("rb") as handle:
        candidate_results = pickle.load(handle)
    source_rows = get_data_split(
        str((run_dir / "data/Mind2Web").resolve()),
        "test_task/*.json",
        candidate_results=candidate_results,
    )
    predictions = json.loads(
        (run_dir / "artifacts/full/test_task_predictions_top50.json").read_text()
    )
    if len(source_rows) != 2094 or len(predictions) != 2094:
        raise ValueError("MindAct trace is incomplete")
    by_identity = {}
    all_labels = []
    for source, prediction in zip(source_rows, predictions):
        key = (source["annotation_id"], source["action_uid"])
        if prediction[0] != f"{key[0]}_{key[1]}" or key in by_identity:
            raise ValueError(f"MindAct identity/order mismatch: {key}")
        positive_ids = {
            candidate["backend_node_id"]
            for candidate in source["pos_candidates"]
            if candidate["rank"] < 50
        }
        element = float(prediction[1] in positive_ids)
        reference_action = source["operation"]["op"]
        reference_operation = reference_action
        if reference_action != "CLICK":
            reference_operation += " " + source["operation"]["value"]
        operation_f1 = token_f1(prediction[2], reference_operation)
        step_success = bool(element == 1.0 and operation_f1 == 1.0)
        predicted_action = prediction[2].strip().split(maxsplit=1)[0] if prediction[2].strip() else None
        if step_success:
            error_type = "success"
        elif predicted_action != reference_action:
            error_type = "action_mismatch"
        elif operation_f1 != 1.0:
            error_type = "parameter_miss"
        else:
            error_type = "element_miss"
        label = {
            "element": element,
            "operation_f1": operation_f1,
            "step_success": step_success,
            "pred_action": predicted_action,
            "error_type": error_type,
        }
        by_identity[key] = label
        all_labels.append(label)
    visual_identities = [identity(row) for row in reference_rows]
    if not set(visual_identities) < set(by_identity) or len(set(by_identity) - set(visual_identities)) != 14:
        raise ValueError("MindAct/visual identity relationship is not the expected 2,080 + 14")
    audit = json.loads((run_dir / "artifacts/full/audit.json").read_text())
    recomputed = {
        "element_acc": sum(item["element"] for item in all_labels) / 2094,
        "action_f1": sum(item["operation_f1"] for item in all_labels) / 2094,
        "step_acc": sum(item["step_success"] for item in all_labels) / 2094,
    }
    for metric, value in recomputed.items():
        if not math.isclose(value, audit["metrics"][metric], abs_tol=1e-12):
            raise ValueError(f"MindAct evaluator parity failure: {metric}")
    return [by_identity[key] for key in visual_identities], {
        "coverage": 2094,
        "shared_visual_identities": 2080,
        "non_visual_actions": 14,
        "evaluator_parity": "PASS",
        "full_metrics": recomputed,
        "shared_metrics": {
            "element_accuracy": sum(by_identity[key]["element"] for key in visual_identities) / 2080,
            "operation_f1": sum(by_identity[key]["operation_f1"] for key in visual_identities) / 2080,
            "step_success": sum(by_identity[key]["step_success"] for key in visual_identities) / 2080,
        },
        "shared_error_type_counts": dict(sorted(Counter(
            by_identity[key]["error_type"] for key in visual_identities
        ).items())),
    }


def overlap_summary(rows: list[dict], labels: dict[str, list[dict]], include_pairs=True) -> dict:
    success_counts = [
        sum(labels[model][index]["step_success"] for model in labels)
        for index in range(len(rows))
    ]
    histogram = Counter(success_counts)
    hard_core_dominant = Counter()
    hard_core_unanimous = Counter()
    for index, success_count in enumerate(success_counts):
        if success_count:
            continue
        votes = Counter(labels[model][index]["error_type"] for model in labels)
        highest = max(votes.values())
        winners = sorted(error_type for error_type, count in votes.items() if count == highest)
        hard_core_dominant[winners[0] if len(winners) == 1 else "tie"] += 1
        if len(votes) == 1:
            hard_core_unanimous[next(iter(votes))] += 1
    by_action = {}
    for action in ACTION_TO_ID:
        indices = [index for index, row in enumerate(rows) if row["answer"]["action"] == action]
        counts = [success_counts[index] for index in indices]
        by_action[action] = {
            "rows": len(indices),
            "all_models_fail": sum(count == 0 for count in counts),
            "oracle_success": sum(count > 0 for count in counts),
            "all_models_success": sum(count == len(labels) for count in counts),
        }
    output = {
        "models": list(labels),
        "rows": len(rows),
        "all_models_fail": histogram[0],
        "oracle_success": len(rows) - histogram[0],
        "all_models_success": histogram[len(labels)],
        "success_count_histogram": {str(key): histogram[key] for key in sorted(histogram)},
        "hard_core_dominant_error_type": dict(sorted(hard_core_dominant.items())),
        "hard_core_unanimous_error_type": dict(sorted(hard_core_unanimous.items())),
        "by_gt_action": by_action,
    }
    if include_pairs:
        pairs = []
        for left, right in combinations(labels, 2):
            left_fail = {i for i, item in enumerate(labels[left]) if not item["step_success"]}
            right_fail = {i for i, item in enumerate(labels[right]) if not item["step_success"]}
            intersection = len(left_fail & right_fail)
            union = len(left_fail | right_fail)
            shared_error_types = Counter(
                labels[left][index]["error_type"]
                for index in left_fail & right_fail
                if labels[left][index]["error_type"] == labels[right][index]["error_type"]
            )
            pairs.append({
                "left": left, "right": right, "intersection": intersection,
                "jaccard": intersection / union if union else 1.0,
                "same_error_type": sum(shared_error_types.values()),
                "same_error_type_rate_within_intersection": (
                    sum(shared_error_types.values()) / intersection if intersection else 1.0
                ),
                "shared_error_type_counts": dict(sorted(shared_error_types.items())),
            })
        output["pairwise_failure_overlap"] = pairs
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    parsers = {
        "showui": load_module(root / "runs/mind2web-showui/2026-07-28/score.py", "showui_score").parse_prediction,
        "tongui": load_module(root / "runs/mind2web-tongui/2026-07-28/score.py", "tongui_score").parse_prediction,
        "cogagent": load_module(root / "runs/mind2web-cogagent/2026-07-28/common.py", "cogagent_common").parse_prediction,
        "uitars": load_module(root / "runs/mind2web-uitars/2026-07-28/common.py", "uitars_common").parse_prediction,
    }
    reference_path = root / MODEL_SPECS["showui-2b"][1]
    rows = read_jsonl(reference_path)
    if len(rows) != 2080 or len({identity(row) for row in rows}) != 2080:
        raise ValueError("reference trace is incomplete or duplicated")
    labels = {}
    model_results = {}
    for model, (kind, prediction_path, score_path) in MODEL_SPECS.items():
        if kind == "seeclick":
            model_labels = load_seeclick(root, rows)
            audit = json.loads((root / "runs/mind2web/2026-07-27/artifacts/gate1_audit.json").read_text())
            expected_metrics = audit["corrected_metrics_independent"]
        else:
            model_rows = read_jsonl(root / prediction_path)
            if [identity(row) for row in model_rows] != [identity(row) for row in rows]:
                raise ValueError(f"identity/order mismatch: {model}")
            source_fields = ("answer", "bbox", "image_size")
            if any(
                tuple(row[field] for field in source_fields)
                != tuple(rows[index][field] for field in source_fields)
                for index, row in enumerate(model_rows)
            ):
                raise ValueError(f"reference mismatch: {model}")
            model_labels = [score_mind2web_row(row, kind, parsers[kind]) for row in model_rows]
            expected_metrics = json.loads((root / score_path).read_text())
        labels[model] = model_labels
        recomputed = {
            "parse_success": sum(item["parse_ok"] for item in model_labels),
            "element_accuracy_micro": sum(item["element"] for item in model_labels) / 2080,
            "operation_f1_micro": sum(item["operation_f1"] for item in model_labels) / 2080,
            "step_success_micro": sum(item["step_success"] for item in model_labels) / 2080,
        }
        expected_keys = {
            "parse_success": "parse_success",
            "element_accuracy_micro": "micro_element_accuracy" if kind == "seeclick" else "element_accuracy_micro",
            "operation_f1_micro": "micro_operation_f1" if kind == "seeclick" else "operation_f1_micro",
            "step_success_micro": "micro_step_success" if kind == "seeclick" else "step_success_micro",
        }
        for metric, expected_key in expected_keys.items():
            if model == "showui-2b" and metric == "operation_f1_micro":
                expected_value = expected_metrics["released_evaluator_identity_diagnostic"][
                    "pseudo_macro_operation_f1"
                ]
            else:
                expected_value = expected_metrics[expected_key]
            if not math.isclose(recomputed[metric], expected_value, abs_tol=1e-12):
                raise ValueError(f"evaluator parity failure: {model}/{metric}")
        error_counts = Counter(item["error_type"] for item in model_labels)
        miss_distances = [
            item.get("bbox_distance") for item in model_labels
            if item["error_type"] == "element_miss" and item.get("bbox_distance") is not None
        ]
        model_results[model] = {
            "family": next(family for family, members in FAMILIES.items() if model in members),
            "evaluator_parity": "PASS",
            "metrics": recomputed,
            "error_type_counts": dict(sorted(error_counts.items())),
            "element_miss_distance": {
                "at_most_0.02": sum(distance <= 0.02 for distance in miss_distances),
                "0.02_to_0.05": sum(0.02 < distance <= 0.05 for distance in miss_distances),
                "over_0.05": sum(distance > 0.05 for distance in miss_distances),
            },
        }

    overall = overlap_summary(rows, labels)
    mindact_labels, mindact_summary = load_mindact(root, rows)
    best_model = max(labels, key=lambda model: model_results[model]["metrics"]["step_success_micro"])
    hard_indices = [
        index for index in range(2080)
        if not any(labels[model][index]["step_success"] for model in labels)
    ]
    disagreement_indices = [
        index for index in range(2080)
        if 0 < sum(labels[model][index]["step_success"] for model in labels) < len(labels)
    ]
    visual_oracle = [
        any(labels[model][index]["step_success"] for model in labels)
        for index in range(2080)
    ]
    cross_modal = Counter()
    for visual_success, mindact_item in zip(visual_oracle, mindact_labels):
        mindact_success = mindact_item["step_success"]
        key = (
            "both_success" if visual_success and mindact_success
            else "visual_only" if visual_success
            else "mindact_only" if mindact_success
            else "both_fail"
        )
        cross_modal[key] += 1
    cross_modal["combined_oracle_success"] = 2080 - cross_modal["both_fail"]
    result = {
        "status": "PASS",
        "contract": {
            "visual_rows": 2080,
            "episodes": 252,
            "identity": ["annot_id", "action_uid"],
            "models": list(MODEL_SPECS),
            "parser_policy": "each model uses its released local parser and scorer semantics",
        },
        "models": model_results,
        "mindact_html": mindact_summary,
        "overlap": overall,
        "family_overlap": {
            family: overlap_summary(rows, {model: labels[model] for model in members}, False)
            for family, members in FAMILIES.items()
        },
        "scale_effects": {
            f"{left}_to_{right}": transition(labels[left], labels[right])
            for left, right in SCALE_PAIRS
        },
        "cross_modal_overlap": dict(cross_modal),
        "learnable_pools": {
            "best_single_model": best_model,
            "best_single_successes": sum(item["step_success"] for item in labels[best_model]),
            "oracle_successes": overall["oracle_success"],
            "oracle_gain_over_best": overall["oracle_success"]
            - sum(item["step_success"] for item in labels[best_model]),
            "consensus_hard_count": len(hard_indices),
            "model_disagreement_count": len(disagreement_indices),
            "consensus_hard_examples": [
                {
                    "annot_id": rows[index]["annot_id"],
                    "action_uid": rows[index]["action_uid"],
                    "gt_action": rows[index]["answer"]["action"],
                    "instruction": rows[index].get("instruction"),
                }
                for index in hard_indices[:100]
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "models": len(labels),
        "all_models_fail": overall["all_models_fail"],
        "oracle_success": overall["oracle_success"],
        "best_single_model": best_model,
        "oracle_gain_over_best": result["learnable_pools"]["oracle_gain_over_best"],
    }, indent=2))


if __name__ == "__main__":
    main()