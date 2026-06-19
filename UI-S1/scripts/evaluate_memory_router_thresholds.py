#!/usr/bin/env python3
"""Evaluate global and per-group thresholds for a memory utility scorer."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_counterfactual_memory_utility import is_positive, load_split, routed_value, score_rows  # noqa: E402


JsonDict = dict[str, Any]


def threshold_grid() -> list[float]:
    base = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 1.01]
    dense = [round(float(value), 4) for value in np.linspace(0.01, 0.99, 99)]
    return sorted(set(base + dense))


def group_value(row: JsonDict, group_key: str) -> str:
    if group_key.startswith("metadata."):
        return str(row.get("metadata", {}).get(group_key.split(".", 1)[1], "unknown"))
    return str(row.get(group_key, "unknown"))


def labels_for(rows: list[JsonDict]) -> np.ndarray:
    return np.array([is_positive(row) for row in rows], dtype=bool)


def metrics_for(rows: list[JsonDict], labels: np.ndarray, pred: np.ndarray) -> JsonDict:
    true_positive = int(np.sum(labels & pred))
    false_positive = int(np.sum((~labels) & pred))
    false_negative = int(np.sum(labels & (~pred)))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    routed_acc = float(np.mean([routed_value(row, bool(use_memory)) for row, use_memory in zip(rows, pred)])) if rows else 0.0
    regressions = sum(
        bool(use_memory)
        and bool(row.get("condition_value_match", {}).get("no_history"))
        and not bool(row.get("condition_value_match", {}).get("segment_summary"))
        for row, use_memory in zip(rows, pred)
    )
    return {
        "n": len(rows),
        "positive": int(np.sum(labels)),
        "predicted_memory": int(np.sum(pred)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "routed_acc": routed_acc,
        "regressions": int(regressions),
    }


def evaluate_threshold(rows: list[JsonDict], scores: np.ndarray, threshold: float) -> JsonDict:
    labels = labels_for(rows)
    pred = scores >= threshold
    result = metrics_for(rows, labels, pred)
    result["threshold"] = threshold
    result["meets_threshold"] = threshold <= 1.0
    return result


def select_threshold(rows: list[JsonDict], scores: np.ndarray, target_precision: float, min_predictions: int) -> JsonDict:
    candidates = []
    for threshold in threshold_grid():
        result = evaluate_threshold(rows, scores, threshold)
        if result["predicted_memory"] >= min_predictions and result["precision"] >= target_precision:
            candidates.append(result)
    if candidates:
        candidates.sort(key=lambda item: (item["recall"], item["routed_acc"], item["precision"], -item["threshold"]), reverse=True)
        selected = candidates[0]
        selected["met_target_precision"] = True
        return selected
    all_results = [evaluate_threshold(rows, scores, threshold) for threshold in threshold_grid()]
    all_results.sort(key=lambda item: (item["precision"], item["recall"], item["routed_acc"], -item["threshold"]), reverse=True)
    selected = all_results[0]
    selected["met_target_precision"] = False
    return selected


def group_indices(rows: list[JsonDict], group_key: str) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[group_value(row, group_key)].append(index)
    return dict(groups)


def take_rows(rows: list[JsonDict], indices: list[int]) -> list[JsonDict]:
    return [rows[index] for index in indices]


def take_scores(scores: np.ndarray, indices: list[int]) -> np.ndarray:
    return np.array([scores[index] for index in indices])


def build_per_group_thresholds(
    dev_rows: list[JsonDict],
    dev_scores: np.ndarray,
    group_key: str,
    target_precision: float,
    min_dev_positives: int,
    min_predictions: int,
    fallback_threshold: float,
) -> tuple[dict[str, float], dict[str, JsonDict]]:
    thresholds: dict[str, float] = {}
    diagnostics: dict[str, JsonDict] = {}
    labels = labels_for(dev_rows)
    for name, indices in sorted(group_indices(dev_rows, group_key).items()):
        group_rows = take_rows(dev_rows, indices)
        group_scores = take_scores(dev_scores, indices)
        positive = int(np.sum(labels[indices]))
        if positive < min_dev_positives:
            thresholds[name] = fallback_threshold
            diagnostics[name] = {
                "selection": "fallback_global",
                "reason": f"positive<{min_dev_positives}",
                "n": len(group_rows),
                "positive": positive,
                "threshold": fallback_threshold,
            }
            continue
        selected = select_threshold(group_rows, group_scores, target_precision, min_predictions)
        thresholds[name] = float(selected["threshold"])
        diagnostics[name] = {"selection": "per_group", **selected}
    return thresholds, diagnostics


def evaluate_per_group(rows: list[JsonDict], scores: np.ndarray, group_key: str, thresholds: dict[str, float], fallback_threshold: float) -> JsonDict:
    pred = np.zeros(len(rows), dtype=bool)
    applied_thresholds = []
    for index, row in enumerate(rows):
        threshold = thresholds.get(group_value(row, group_key), fallback_threshold)
        applied_thresholds.append(threshold)
        pred[index] = scores[index] >= threshold
    labels = labels_for(rows)
    result = metrics_for(rows, labels, pred)
    result["fallback_threshold"] = fallback_threshold
    result["groups"] = {}
    for name, indices in sorted(group_indices(rows, group_key).items()):
        group_rows = take_rows(rows, indices)
        group_labels = labels[indices]
        group_pred = pred[indices]
        group_result = metrics_for(group_rows, group_labels, group_pred)
        group_result["threshold"] = thresholds.get(name, fallback_threshold)
        result["groups"][name] = group_result
    return result


def write_report(path: Path, report: JsonDict) -> None:
    lines = ["# Memory Router Threshold Evaluation", ""]
    lines.append(f"Model: `{report['model']}`")
    lines.append(f"Data dir: `{report['data_dir']}`")
    lines.append(f"Group key: `{report['group_key']}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| target precision | policy | selected on | threshold | predicted | precision | recall | routed_acc | regressions |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for target in report["targets"]:
        target_text = f"{target['target_precision']:.2f}"
        global_dev = target["global_dev"]
        global_test = target["global_test"]
        per_test = target["per_group_test"]
        lines.append(
            f"| {target_text} | global | dev | {global_dev['threshold']:.4f} | {global_test['predicted_memory']} | "
            f"{global_test['precision']:.4f} | {global_test['recall']:.4f} | {global_test['routed_acc']:.4f} | {global_test['regressions']} |"
        )
        lines.append(
            f"| {target_text} | per_group | dev groups | mixed | {per_test['predicted_memory']} | "
            f"{per_test['precision']:.4f} | {per_test['recall']:.4f} | {per_test['routed_acc']:.4f} | {per_test['regressions']} |"
        )
    lines.append("")
    for target in report["targets"]:
        lines.append(f"## Target Precision {target['target_precision']:.2f}")
        lines.append("")
        lines.append("### Per-Group Test Metrics")
        lines.append("")
        lines.append("| group | threshold | n | positive | predicted | precision | recall | regressions |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for name, item in target["per_group_test"]["groups"].items():
            lines.append(
                f"| {name} | {item['threshold']:.4f} | {item['n']} | {item['positive']} | {item['predicted_memory']} | "
                f"{item['precision']:.4f} | {item['recall']:.4f} | {item['regressions']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate global and per-group thresholds for a memory utility scorer")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-key", default="metadata.dominant_capability")
    parser.add_argument("--target-precision", nargs="+", type=float, default=[0.5, 0.6, 0.7])
    parser.add_argument("--min-dev-positives", type=int, default=2)
    parser.add_argument("--min-predictions", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = joblib.load(args.model)
    rows = {split: load_split(data_dir, split) for split in ["dev", "test"]}
    scores = {split: score_rows(model, rows[split]) for split in rows}
    report: JsonDict = {
        "data_dir": str(data_dir),
        "model": str(args.model),
        "group_key": args.group_key,
        "min_dev_positives": args.min_dev_positives,
        "min_predictions": args.min_predictions,
        "targets": [],
    }
    for target_precision in args.target_precision:
        global_dev = select_threshold(rows["dev"], scores["dev"], target_precision, args.min_predictions)
        fallback_threshold = float(global_dev["threshold"])
        global_test = evaluate_threshold(rows["test"], scores["test"], fallback_threshold)
        thresholds, diagnostics = build_per_group_thresholds(
            rows["dev"],
            scores["dev"],
            args.group_key,
            target_precision,
            args.min_dev_positives,
            args.min_predictions,
            fallback_threshold,
        )
        per_group_dev = evaluate_per_group(rows["dev"], scores["dev"], args.group_key, thresholds, fallback_threshold)
        per_group_test = evaluate_per_group(rows["test"], scores["test"], args.group_key, thresholds, fallback_threshold)
        report["targets"].append({
            "target_precision": target_precision,
            "global_dev": global_dev,
            "global_test": global_test,
            "per_group_thresholds": thresholds,
            "per_group_selection_diagnostics": diagnostics,
            "per_group_dev": per_group_dev,
            "per_group_test": per_group_test,
        })
    (output_dir / "threshold_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(output_dir / "threshold_report.md", report)
    print(f"wrote threshold evaluation to {output_dir}")


if __name__ == "__main__":
    main()
