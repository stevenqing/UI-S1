#!/usr/bin/env python3
"""Evaluate multi-route memory router policies from behavior-intervention tables."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_counterfactual_memory_utility import is_positive, load_split, score_rows  # noqa: E402


JsonDict = dict[str, Any]
PolicyFn = Callable[[JsonDict, float, float], str]

CONTEXT_ROUTES = {"no_history", "segment_summary", "full_history"}
ALL_ROUTES = ["no_history", "segment_summary", "full_history", "replan"]
ROUTE_COSTS = {"no_history": 1.0, "segment_summary": 1.2, "full_history": 2.0, "replan": 3.0}


def candidate(row: JsonDict, condition: str) -> JsonDict | None:
    return (row.get("pred_actions", {}) or {}).get(condition)


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


def same_type(row: JsonDict, first: str, second: str) -> bool:
    return action_type(candidate(row, first)) == action_type(candidate(row, second))


def same_exact(row: JsonDict, first: str, second: str) -> bool:
    return same_type(row, first, second) and action_value(candidate(row, first)) == action_value(candidate(row, second))


def exists(row: JsonDict, condition: str) -> bool:
    return candidate(row, condition) is not None


def ok(row: JsonDict, condition: str) -> bool:
    return bool((row.get("condition_value_match", {}) or {}).get(condition))


def oracle_route(row: JsonDict, nonspecific_policy: str) -> str:
    if ok(row, "no_history"):
        return "no_history"
    if ok(row, "segment_summary") and not ok(row, "wrong_summary"):
        return "segment_summary"
    if ok(row, "segment_summary"):
        return nonspecific_policy
    if ok(row, "full_history"):
        return "full_history"
    return "replan"


def all_candidate_types(row: JsonDict) -> set[str]:
    return {action_type(candidate(row, condition)) for condition in ["no_history", "segment_summary", "full_history", "wrong_summary"]}


def full_history_specific(row: JsonDict) -> bool:
    return exists(row, "full_history") and not same_type(row, "full_history", "no_history") and not same_type(row, "full_history", "wrong_summary")


def candidates_unstable(row: JsonDict) -> bool:
    types = all_candidate_types(row)
    non_missing = {item for item in types if item != "missing"}
    return len(non_missing) >= 3 or ("missing" in types and len(non_missing) >= 2)


def policy_base(row: JsonDict, score: float, threshold: float) -> str:
    return "segment_summary" if score >= threshold else "no_history"


def policy_verified_no_history(row: JsonDict, score: float, threshold: float) -> str:
    if score >= threshold and same_type(row, "segment_summary", "full_history"):
        return "segment_summary"
    return "no_history"


def policy_verified_full_history(row: JsonDict, score: float, threshold: float) -> str:
    if score < threshold:
        return "no_history"
    if same_type(row, "segment_summary", "full_history"):
        return "segment_summary"
    if full_history_specific(row):
        return "full_history"
    return "no_history"


def policy_verified_replan(row: JsonDict, score: float, threshold: float) -> str:
    if score < threshold:
        return "no_history"
    if same_type(row, "segment_summary", "full_history"):
        return "segment_summary"
    if candidates_unstable(row):
        return "replan"
    return "no_history"


def policy_verified_full_or_replan(row: JsonDict, score: float, threshold: float) -> str:
    if score < threshold:
        return "no_history"
    if same_type(row, "segment_summary", "full_history"):
        return "segment_summary"
    if full_history_specific(row):
        return "full_history"
    if candidates_unstable(row):
        return "replan"
    return "no_history"


def policies() -> dict[str, PolicyFn]:
    return {
        "base_segment_else_no_history": policy_base,
        "verified_else_no_history": policy_verified_no_history,
        "verified_else_full_history": policy_verified_full_history,
        "verified_else_replan": policy_verified_replan,
        "verified_else_full_history_or_replan": policy_verified_full_or_replan,
    }


def action_accuracy(row: JsonDict, route: str) -> bool:
    if route in CONTEXT_ROUTES:
        return ok(row, route)
    return False


def evaluate_policy(rows: list[JsonDict], scores: np.ndarray, threshold: float, policy: PolicyFn, nonspecific_policy: str) -> JsonDict:
    route_counts: Counter[str] = Counter()
    oracle_counts: Counter[str] = Counter()
    route_confusion: Counter[str] = Counter()
    action_correct = 0
    route_correct = 0
    segment_tp = segment_fp = segment_fn = 0
    full_tp = full_fp = full_fn = 0
    replan_tp = replan_fp = replan_fn = 0
    segment_regressions = 0
    full_history_rescues = 0
    total_cost = 0.0
    for row, score in zip(rows, scores):
        route = policy(row, float(score), threshold)
        total_cost += ROUTE_COSTS.get(route, 1.0)
        oracle = oracle_route(row, nonspecific_policy)
        route_counts[route] += 1
        oracle_counts[oracle] += 1
        route_confusion[f"{oracle}->{route}"] += 1
        action_correct += int(action_accuracy(row, route))
        route_correct += int(route == oracle)
        if route == "segment_summary" and oracle == "segment_summary":
            segment_tp += 1
        elif route == "segment_summary" and oracle != "segment_summary":
            segment_fp += 1
        elif route != "segment_summary" and oracle == "segment_summary":
            segment_fn += 1
        if route == "full_history" and oracle == "full_history":
            full_tp += 1
        elif route == "full_history" and oracle != "full_history":
            full_fp += 1
        elif route != "full_history" and oracle == "full_history":
            full_fn += 1
        if route == "replan" and oracle == "replan":
            replan_tp += 1
        elif route == "replan" and oracle != "replan":
            replan_fp += 1
        elif route != "replan" and oracle == "replan":
            replan_fn += 1
        if route == "segment_summary" and ok(row, "no_history") and not ok(row, "segment_summary"):
            segment_regressions += 1
        if route == "full_history" and ok(row, "full_history") and not ok(row, "no_history") and not ok(row, "segment_summary"):
            full_history_rescues += 1
    def prf(tp: int, fp: int, fn: int) -> JsonDict:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    return {
        "n": len(rows),
        "threshold": threshold,
        "action_accuracy": action_correct / len(rows) if rows else 0.0,
        "route_accuracy": route_correct / len(rows) if rows else 0.0,
        "route_counts": dict(route_counts),
        "avg_route_cost": total_cost / len(rows) if rows else 0.0,
        "non_no_history_rate": 1.0 - (route_counts.get("no_history", 0) / len(rows) if rows else 0.0),
        "full_history_rate": route_counts.get("full_history", 0) / len(rows) if rows else 0.0,
        "replan_rate": route_counts.get("replan", 0) / len(rows) if rows else 0.0,
        "oracle_counts": dict(oracle_counts),
        "route_confusion": dict(route_confusion.most_common()),
        "segment_memory": prf(segment_tp, segment_fp, segment_fn),
        "full_history": prf(full_tp, full_fp, full_fn),
        "replan": prf(replan_tp, replan_fp, replan_fn),
        "segment_regressions": int(segment_regressions),
        "full_history_rescues": int(full_history_rescues),
    }


def static_policy(route: str) -> PolicyFn:
    return lambda row, score, threshold: route


def write_report(path: Path, report: JsonDict) -> None:
    lines = ["# Memory Router Multi-Route Policy Evaluation", ""]
    lines.append(f"Model: `{report['model']}`")
    lines.append(f"Data dir: `{report['data_dir']}`")
    lines.append(f"Nonspecific policy: `{report['nonspecific_policy']}`")
    lines.append("")
    for split, split_items in report["splits"].items():
        lines.append(f"## {split.title()}")
        lines.append("")
        lines.append("| threshold | policy | action_acc | route_acc | avg cost | non-default | segment P/R | full P/R | replan P/R | segment regressions | full rescues | route counts |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for threshold_item in split_items:
            threshold = threshold_item["threshold"]
            for policy_name, metrics in threshold_item["policies"].items():
                seg = metrics["segment_memory"]
                full = metrics["full_history"]
                replan = metrics["replan"]
                route_counts = ", ".join(f"{key}:{value}" for key, value in sorted(metrics["route_counts"].items()))
                lines.append(
                    f"| {threshold:.2f} | {policy_name} | {metrics['action_accuracy']:.4f} | {metrics['route_accuracy']:.4f} | "
                    f"{metrics['avg_route_cost']:.4f} | {metrics['non_no_history_rate']:.4f} | "
                    f"{seg['precision']:.4f}/{seg['recall']:.4f} | {full['precision']:.4f}/{full['recall']:.4f} | "
                    f"{replan['precision']:.4f}/{replan['recall']:.4f} | {metrics['segment_regressions']} | "
                    f"{metrics['full_history_rescues']} | {route_counts} |"
                )
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("This report separates context-action accuracy from route-decision quality. Replan routes are counted as route decisions, not as successful next actions, because they require another candidate source or verifier in prospective evaluation.")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multi-route memory router policies")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.7, 0.9])
    parser.add_argument("--splits", nargs="+", default=["dev", "test"])
    parser.add_argument("--nonspecific-policy", choices=["no_history", "segment_summary", "replan"], default="no_history")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = joblib.load(args.model)
    policy_bank = {
        "always_no_history": static_policy("no_history"),
        "always_segment_summary": static_policy("segment_summary"),
        "always_full_history": static_policy("full_history"),
        **policies(),
    }
    report: JsonDict = {
        "data_dir": args.data_dir,
        "model": args.model,
        "thresholds": args.thresholds,
        "nonspecific_policy": args.nonspecific_policy,
        "splits": {},
    }
    for split in args.splits:
        rows = load_split(Path(args.data_dir), split)
        scores = score_rows(model, rows)
        split_results = []
        for threshold in args.thresholds:
            split_results.append({
                "threshold": threshold,
                "policies": {
                    name: evaluate_policy(rows, scores, threshold, policy, args.nonspecific_policy)
                    for name, policy in policy_bank.items()
                },
            })
        report["splits"][split] = split_results
    (output_dir / "policy_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(output_dir / "policy_report.md", report)
    print(f"wrote multi-route policy evaluation to {output_dir}")


if __name__ == "__main__":
    main()
