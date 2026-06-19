#!/usr/bin/env python3
"""Analyze candidate-repair scorer errors for next feature design."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_counterfactual_memory_utility import is_positive, load_split, routed_value, score_rows  # noqa: E402


JsonDict = dict[str, Any]


def action_type(action: JsonDict | None) -> str:
    if not action:
        return "missing"
    return str(action.get("action", "missing"))


def action_value(action: JsonDict | None) -> str:
    if not action:
        return ""
    if action.get("text") is not None:
        return str(action.get("text"))
    if action.get("button") is not None:
        return str(action.get("button"))
    if action.get("coordinate") is not None:
        return json.dumps(action.get("coordinate"), ensure_ascii=False)
    return json.dumps(action, ensure_ascii=False)


def transition(row: JsonDict, first: str, second: str) -> str:
    actions = row.get("pred_actions", {}) or {}
    return f"{action_type(actions.get(first))}->{action_type(actions.get(second))}"


def same_candidate(row: JsonDict, first: str, second: str) -> bool:
    actions = row.get("pred_actions", {}) or {}
    first_action = actions.get(first)
    second_action = actions.get(second)
    return action_type(first_action) == action_type(second_action) and action_value(first_action) == action_value(second_action)


def same_action_type(row: JsonDict, first: str, second: str) -> bool:
    actions = row.get("pred_actions", {}) or {}
    return action_type(actions.get(first)) == action_type(actions.get(second))


def routing_metrics(rows: list[JsonDict], labels: np.ndarray, pred: np.ndarray) -> JsonDict:
    true_positive = int(np.sum(labels & pred))
    false_positive = int(np.sum((~labels) & pred))
    false_negative = int(np.sum(labels & (~pred)))
    routed_acc = float(np.mean([routed_value(row, bool(use_memory)) for row, use_memory in zip(rows, pred)])) if rows else 0.0
    regressions = sum(
        bool(use_memory)
        and bool(row.get("condition_value_match", {}).get("no_history"))
        and not bool(row.get("condition_value_match", {}).get("segment_summary"))
        for row, use_memory in zip(rows, pred)
    )
    return {
        "predicted_memory": int(np.sum(pred)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0,
        "recall": true_positive / int(np.sum(labels)) if int(np.sum(labels)) else 0.0,
        "routed_acc": routed_acc,
        "regressions": int(regressions),
    }


def summarize_bucket(rows: list[JsonDict]) -> JsonDict:
    return {
        "n": len(rows),
        "utility_labels": Counter(row.get("utility_label") for row in rows),
        "condition_value_match_pattern": Counter(json.dumps(row.get("condition_value_match", {}), sort_keys=True) for row in rows),
        "no_to_segment_transition": Counter(transition(row, "no_history", "segment_summary") for row in rows),
        "no_to_wrong_transition": Counter(transition(row, "no_history", "wrong_summary") for row in rows),
        "segment_matches_wrong_candidate": Counter(str(same_candidate(row, "segment_summary", "wrong_summary")) for row in rows),
        "segment_matches_no_history_candidate": Counter(str(same_candidate(row, "segment_summary", "no_history")) for row in rows),
        "dominant_capability": Counter(row.get("metadata", {}).get("dominant_capability", "unknown") for row in rows),
        "gt_action_type": Counter(row.get("metadata", {}).get("gt_action_type", "unknown") for row in rows),
    }


def counter_to_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.most_common()}


def normalize_summary(value: Any) -> Any:
    if isinstance(value, Counter):
        return counter_to_dict(value)
    if isinstance(value, dict):
        return {str(key): normalize_summary(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_summary(item) for item in value]
    return value


def example_row(row: JsonDict, score: float) -> JsonDict:
    return {
        "score": float(score),
        "utility_label": row.get("utility_label"),
        "episode_id": row.get("metadata", {}).get("episode_id"),
        "case_id": row.get("metadata", {}).get("case_id"),
        "model_key": row.get("metadata", {}).get("model_key"),
        "thinking_mode": row.get("metadata", {}).get("thinking_mode"),
        "case_kind": row.get("metadata", {}).get("case_kind"),
        "dominant_capability": row.get("metadata", {}).get("dominant_capability"),
        "gt_action_type": row.get("metadata", {}).get("gt_action_type"),
        "condition_value_match": row.get("condition_value_match"),
        "current_instruction": row.get("current_state_parts", {}).get("instruction"),
        "true_memory_text": row.get("true_memory_text"),
        "wrong_memory_text": row.get("wrong_memory_text"),
        "pred_actions": row.get("pred_actions"),
        "no_to_segment_transition": transition(row, "no_history", "segment_summary"),
        "segment_matches_wrong_candidate": same_candidate(row, "segment_summary", "wrong_summary"),
        "segment_matches_no_history_candidate": same_candidate(row, "segment_summary", "no_history"),
    }


def write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, summary: JsonDict) -> None:
    lines = ["# Candidate Repair Error Analysis", ""]
    lines.append(f"Split: `{summary['split']}`")
    lines.append("")
    for threshold_result in summary["thresholds"]:
        lines.append(f"## Threshold {threshold_result['threshold']:.2f}")
        lines.append("")
        lines.append(f"- predicted_memory: {threshold_result['predicted_memory']}")
        lines.append(f"- true_positive: {threshold_result['true_positive']}")
        lines.append(f"- false_positive: {threshold_result['false_positive']}")
        lines.append(f"- false_negative: {threshold_result['false_negative']}")
        lines.append(f"- precision: {threshold_result['precision']:.4f}")
        lines.append(f"- recall: {threshold_result['recall']:.4f}")
        lines.append(f"- routed_acc: {threshold_result['routed_acc']:.4f}")
        lines.append(f"- regressions: {threshold_result['regressions']}")
        lines.append("")
        lines.append("Specificity filter variants:")
        lines.append("")
        lines.append("| filter | predicted | precision | recall | routed_acc | regressions |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for filter_name, metrics in threshold_result["specificity_filters"].items():
            lines.append(
                f"| {filter_name} | {metrics['predicted_memory']} | {metrics['precision']:.4f} | "
                f"{metrics['recall']:.4f} | {metrics['routed_acc']:.4f} | {metrics['regressions']} |"
            )
        lines.append("")
        for bucket_name in ["true_positives", "false_positives", "false_negatives"]:
            bucket = threshold_result[bucket_name]
            lines.append(f"### {bucket_name}")
            lines.append("")
            lines.append("Utility labels:")
            for key, value in list(bucket["utility_labels"].items())[:10]:
                lines.append(f"- {key}: {value}")
            lines.append("")
            lines.append("Dominant capabilities:")
            for key, value in list(bucket["dominant_capability"].items())[:10]:
                lines.append(f"- {key}: {value}")
            lines.append("")
            lines.append("Condition value-match patterns:")
            for key, value in list(bucket["condition_value_match_pattern"].items())[:6]:
                lines.append(f"- `{key}`: {value}")
            lines.append("")
            lines.append("Top no_history -> segment_summary transitions:")
            for key, value in list(bucket["no_to_segment_transition"].items())[:10]:
                lines.append(f"- {key}: {value}")
            lines.append("")
            lines.append("segment_summary matches wrong_summary candidate:")
            for key, value in bucket["segment_matches_wrong_candidate"].items():
                lines.append(f"- {key}: {value}")
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze candidate repair scorer errors")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.9, 0.97, 0.99])
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_split(Path(args.data_dir), args.split)
    model = joblib.load(args.model)
    scores = score_rows(model, rows)
    labels = np.array([is_positive(row) for row in rows], dtype=bool)
    summary: JsonDict = {"split": args.split, "thresholds": []}
    for threshold in args.thresholds:
        pred = scores >= threshold
        true_positive_indices = [index for index, value in enumerate(pred) if value and labels[index]]
        false_positive_indices = [index for index, value in enumerate(pred) if value and not labels[index]]
        false_negative_indices = [index for index, value in enumerate(pred) if (not value) and labels[index]]
        exact_filtered = np.array([
            bool(use_memory) and not same_candidate(row, "segment_summary", "wrong_summary")
            for row, use_memory in zip(rows, pred)
        ], dtype=bool)
        type_filtered = np.array([
            bool(use_memory) and not same_action_type(row, "segment_summary", "wrong_summary")
            for row, use_memory in zip(rows, pred)
        ], dtype=bool)
        threshold_dir = output_dir / f"threshold_{threshold:.2f}"
        threshold_dir.mkdir(exist_ok=True)
        buckets = {
            "true_positives": true_positive_indices,
            "false_positives": false_positive_indices,
            "false_negatives": false_negative_indices,
        }
        threshold_result: JsonDict = {"threshold": threshold, **routing_metrics(rows, labels, pred)}
        threshold_result["specificity_filters"] = {
            "none": routing_metrics(rows, labels, pred),
            "reject_exact_segment_equals_wrong": routing_metrics(rows, labels, exact_filtered),
            "reject_type_segment_equals_wrong": routing_metrics(rows, labels, type_filtered),
        }
        for bucket_name, indices in buckets.items():
            bucket_rows = [rows[index] for index in indices]
            threshold_result[bucket_name] = normalize_summary(summarize_bucket(bucket_rows))
            ordered = sorted(indices, key=lambda index: scores[index], reverse=True)
            write_jsonl(threshold_dir / f"{bucket_name}.jsonl", [example_row(rows[index], float(scores[index])) for index in ordered[: args.top_k]])
        summary["thresholds"].append(threshold_result)
    normalized = normalize_summary(summary)
    (output_dir / "summary.json").write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(output_dir / "error_report.md", normalized)
    print(f"wrote candidate repair error analysis to {output_dir}")


if __name__ == "__main__":
    main()
