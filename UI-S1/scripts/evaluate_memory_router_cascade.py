#!/usr/bin/env python3
"""Evaluate memory utility scorer cascades with candidate-validity filters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_counterfactual_memory_utility import is_positive, load_split, routed_value, score_rows  # noqa: E402


JsonDict = dict[str, Any]
FilterFn = Callable[[JsonDict], bool]


def action_type(action: JsonDict | None) -> str:
    if not action:
        return "missing"
    return str(action.get("action", "missing"))


def action_value(action: JsonDict | None) -> str:
    if not action:
        return ""
    if action.get("text") is not None:
        return str(action.get("text")).lower().strip()
    if action.get("button") is not None:
        return str(action.get("button")).lower().strip()
    if action.get("coordinate") is not None:
        return json.dumps(action.get("coordinate"), ensure_ascii=False)
    return json.dumps(action, ensure_ascii=False, sort_keys=True)


def candidate(row: JsonDict, condition: str) -> JsonDict | None:
    return (row.get("pred_actions", {}) or {}).get(condition)


def same_type(row: JsonDict, first: str, second: str) -> bool:
    return action_type(candidate(row, first)) == action_type(candidate(row, second))


def same_exact(row: JsonDict, first: str, second: str) -> bool:
    return same_type(row, first, second) and action_value(candidate(row, first)) == action_value(candidate(row, second))


def exists(row: JsonDict, condition: str) -> bool:
    return candidate(row, condition) is not None


def no_filter(row: JsonDict) -> bool:
    return True


def segment_exists(row: JsonDict) -> bool:
    return exists(row, "segment_summary")


def segment_full_same_type(row: JsonDict) -> bool:
    return same_type(row, "segment_summary", "full_history")


def segment_full_exact(row: JsonDict) -> bool:
    return same_exact(row, "segment_summary", "full_history")


def segment_full_same_type_not_wrong_type(row: JsonDict) -> bool:
    return same_type(row, "segment_summary", "full_history") and not same_type(row, "segment_summary", "wrong_summary")


def segment_full_exact_not_wrong_exact(row: JsonDict) -> bool:
    return same_exact(row, "segment_summary", "full_history") and not same_exact(row, "segment_summary", "wrong_summary")


def filter_bank() -> dict[str, FilterFn]:
    return {
        "none": no_filter,
        "segment_exists": segment_exists,
        "segment_full_same_type": segment_full_same_type,
        "segment_full_exact": segment_full_exact,
        "segment_full_same_type_not_wrong_type": segment_full_same_type_not_wrong_type,
        "segment_full_exact_not_wrong_exact": segment_full_exact_not_wrong_exact,
    }


def labels_for(rows: list[JsonDict]) -> np.ndarray:
    return np.array([is_positive(row) for row in rows], dtype=bool)


def condition_ok(row: JsonDict, condition: str) -> bool:
    return bool((row.get("condition_value_match", {}) or {}).get(condition))


def evaluate_policy(rows: list[JsonDict], scores: np.ndarray, threshold: float, filter_fn: FilterFn) -> JsonDict:
    labels = labels_for(rows)
    base_pred = scores >= threshold
    accept_memory = np.array([bool(base) and filter_fn(row) for row, base in zip(rows, base_pred)], dtype=bool)
    rejected_by_filter = base_pred & (~accept_memory)
    true_positive = int(np.sum(labels & accept_memory))
    false_positive = int(np.sum((~labels) & accept_memory))
    false_negative = int(np.sum(labels & (~accept_memory)))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    routed_acc = float(np.mean([routed_value(row, bool(use_memory)) for row, use_memory in zip(rows, accept_memory)])) if rows else 0.0
    regressions = sum(
        bool(use_memory)
        and condition_ok(row, "no_history")
        and not condition_ok(row, "segment_summary")
        for row, use_memory in zip(rows, accept_memory)
    )
    rejected_rows = [row for row, rejected in zip(rows, rejected_by_filter) if rejected]
    rejected_labels: dict[str, int] = {}
    rejected_patterns: dict[str, int] = {}
    for row in rejected_rows:
        label = str(row.get("utility_label", "unknown"))
        rejected_labels[label] = rejected_labels.get(label, 0) + 1
        pattern = json.dumps(row.get("condition_value_match", {}), sort_keys=True)
        rejected_patterns[pattern] = rejected_patterns.get(pattern, 0) + 1
    unresolved_rejects = sum(1 for row in rejected_rows if row.get("utility_label") == "unresolved")
    no_history_wrong_rejects = sum(1 for row in rejected_rows if not condition_ok(row, "no_history"))
    full_history_only_rejects = sum(
        1
        for row in rejected_rows
        if condition_ok(row, "full_history") and not condition_ok(row, "no_history") and not condition_ok(row, "segment_summary")
    )
    return {
        "n": len(rows),
        "threshold": threshold,
        "base_predicted_memory": int(np.sum(base_pred)),
        "accepted_memory": int(np.sum(accept_memory)),
        "rejected_by_filter": int(np.sum(rejected_by_filter)),
        "positive": int(np.sum(labels)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "routed_acc": routed_acc,
        "regressions": int(regressions),
        "rejected_unresolved": int(unresolved_rejects),
        "rejected_no_history_wrong": int(no_history_wrong_rejects),
        "rejected_full_history_only": int(full_history_only_rejects),
        "rejected_labels": dict(sorted(rejected_labels.items(), key=lambda item: (-item[1], item[0]))),
        "rejected_condition_patterns": dict(sorted(rejected_patterns.items(), key=lambda item: (-item[1], item[0]))),
    }


def write_report(path: Path, report: JsonDict) -> None:
    lines = ["# Memory Router Cascade Evaluation", ""]
    lines.append(f"Model: `{report['model']}`")
    lines.append(f"Data dir: `{report['data_dir']}`")
    lines.append("")
    for split, split_result in report["splits"].items():
        lines.append(f"## {split.title()}")
        lines.append("")
        lines.append("| threshold | filter | base predicted | accepted | rejected | precision | recall | routed_acc | regressions | rejected unresolved | rejected no-history wrong | rejected full-history-only |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for threshold_result in split_result:
            threshold = threshold_result["threshold"]
            for filter_name, metrics in threshold_result["filters"].items():
                lines.append(
                    f"| {threshold:.2f} | {filter_name} | {metrics['base_predicted_memory']} | {metrics['accepted_memory']} | "
                    f"{metrics['rejected_by_filter']} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                    f"{metrics['routed_acc']:.4f} | {metrics['regressions']} | {metrics['rejected_unresolved']} | "
                    f"{metrics['rejected_no_history_wrong']} | {metrics['rejected_full_history_only']} |"
                )
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("The cascade separates memory utility scoring from candidate-validity verification. Full-history consistency filters are not meant to replace the memory scorer; they test whether a segment-memory candidate is supported by a stronger raw-history context before committing to memory.")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate memory utility scorer cascades with candidate-validity filters")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.7, 0.9, 0.99])
    parser.add_argument("--splits", nargs="+", default=["dev", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = joblib.load(args.model)
    filters = filter_bank()
    report: JsonDict = {"data_dir": args.data_dir, "model": args.model, "thresholds": args.thresholds, "splits": {}}
    for split in args.splits:
        rows = load_split(Path(args.data_dir), split)
        scores = score_rows(model, rows)
        split_results = []
        for threshold in args.thresholds:
            split_results.append({
                "threshold": threshold,
                "filters": {name: evaluate_policy(rows, scores, threshold, filter_fn) for name, filter_fn in filters.items()},
            })
        report["splits"][split] = split_results
    (output_dir / "cascade_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(output_dir / "cascade_report.md", report)
    print(f"wrote cascade evaluation to {output_dir}")


if __name__ == "__main__":
    main()
