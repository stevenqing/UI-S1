#!/usr/bin/env python3
"""Evaluate verifier-agent route decisions from packets or rule baselines."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]
DECISIONS = ["use_no_history", "commit_segment", "use_full_history", "replan"]


def iter_jsonl(path: Path) -> list[JsonDict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def action_type(agent_packet: JsonDict, agent_name: str) -> str:
    return str((agent_packet.get("candidate_agents", {}).get(agent_name, {}) or {}).get("action_type", "missing"))


def full_history_specific(packet: JsonDict) -> bool:
    full_type = action_type(packet, "full_history_agent")
    no_type = action_type(packet, "no_history_agent")
    wrong_type = action_type(packet, "distractor_memory_agent")
    return full_type != "missing" and full_type != no_type and full_type != wrong_type


def unstable_candidates(packet: JsonDict) -> bool:
    types = {
        action_type(packet, "no_history_agent"),
        action_type(packet, "segment_memory_agent"),
        action_type(packet, "full_history_agent"),
        action_type(packet, "distractor_memory_agent"),
    }
    non_missing = {item for item in types if item != "missing"}
    return len(non_missing) >= 3 or ("missing" in types and len(non_missing) >= 2)


def rule_decision(packet: JsonDict, mode: str, threshold: float) -> JsonDict:
    evidence = packet.get("computed_evidence", {}) or {}
    score = float(evidence.get("memory_proposal_score", 0.0))
    if score < threshold:
        return {"decision": "use_no_history", "reason_codes": ["proposal_score_below_threshold"]}
    if mode == "commit_if_segment_full_same_type":
        if evidence.get("segment_vs_full_same_type"):
            return {"decision": "commit_segment", "reason_codes": ["segment_full_same_type"]}
        return {"decision": "use_no_history", "reason_codes": ["full_history_does_not_support_segment"]}
    if mode == "commit_if_segment_full_same_type_else_full_or_replan":
        if evidence.get("segment_vs_full_same_type"):
            return {"decision": "commit_segment", "reason_codes": ["segment_full_same_type"]}
        if full_history_specific(packet):
            return {"decision": "use_full_history", "reason_codes": ["full_history_specific"]}
        if unstable_candidates(packet):
            return {"decision": "replan", "reason_codes": ["candidate_instability"]}
        return {"decision": "use_no_history", "reason_codes": ["reject_segment"]}
    if mode == "commit_if_specific_progress_and_full_support":
        if (
            evidence.get("segment_vs_full_same_type")
            and not evidence.get("segment_vs_wrong_same_type")
            and evidence.get("segment_candidate_matches_instruction")
        ):
            return {"decision": "commit_segment", "reason_codes": ["specific_progress_full_support"]}
        if full_history_specific(packet):
            return {"decision": "use_full_history", "reason_codes": ["full_history_specific"]}
        if unstable_candidates(packet):
            return {"decision": "replan", "reason_codes": ["candidate_instability"]}
        return {"decision": "use_no_history", "reason_codes": ["reject_segment"]}
    if mode == "three_way_commit_full_or_replan":
        if evidence.get("segment_vs_full_same_type"):
            return {"decision": "commit_segment", "reason_codes": ["segment_full_same_type"]}
        if full_history_specific(packet):
            return {"decision": "use_full_history", "reason_codes": ["full_history_specific"]}
        return {"decision": "replan", "reason_codes": ["segment_rejected_hard_state"]}
    raise ValueError(f"unknown rule mode: {mode}")


def parse_decision_text(text: str) -> str:
    text = text.strip()
    try:
        data = json.loads(text)
        return str(data.get("decision", "invalid"))
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            return str(data.get("decision", "invalid"))
        except json.JSONDecodeError:
            return "invalid"
    return "invalid"


def load_prediction_decisions(path: Path) -> list[str]:
    decisions = []
    for row in iter_jsonl(path):
        if "decision" in row:
            decisions.append(str(row["decision"]))
        elif "assistant" in row:
            decisions.append(parse_decision_text(str(row["assistant"])))
        elif "content" in row:
            decisions.append(parse_decision_text(str(row["content"])))
        else:
            decisions.append(parse_decision_text(json.dumps(row, ensure_ascii=False)))
    return decisions


def metrics(gold: list[str], pred: list[str]) -> JsonDict:
    confusion = Counter(f"{g}->{p}" for g, p in zip(gold, pred))
    gold_counts = Counter(gold)
    pred_counts = Counter(pred)
    correct = sum(int(g == p) for g, p in zip(gold, pred))
    per_class = {}
    f1_values = []
    for decision in DECISIONS:
        tp = confusion.get(f"{decision}->{decision}", 0)
        fp = sum(count for key, count in confusion.items() if key.endswith(f"->{decision}") and not key.startswith(f"{decision}->"))
        fn = sum(count for key, count in confusion.items() if key.startswith(f"{decision}->") and not key.endswith(f"->{decision}"))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[decision] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    return {
        "n": len(gold),
        "accuracy": correct / len(gold) if gold else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "gold_counts": dict(gold_counts),
        "pred_counts": dict(pred_counts),
        "per_class": per_class,
        "confusion": dict(confusion.most_common()),
    }


def write_report(path: Path, result: JsonDict) -> None:
    lines = ["# Verifier Agent Evaluation", ""]
    lines.append(f"Mode: `{result['mode']}`")
    lines.append(f"Threshold: `{result['threshold']}`")
    lines.append("")
    lines.append(f"- n: {result['metrics']['n']}")
    lines.append(f"- accuracy: {result['metrics']['accuracy']:.4f}")
    lines.append(f"- macro_f1: {result['metrics']['macro_f1']:.4f}")
    lines.append("")
    lines.append("## Per Decision")
    lines.append("")
    lines.append("| decision | precision | recall | f1 | tp | fp | fn |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for decision, item in result["metrics"]["per_class"].items():
        lines.append(
            f"| {decision} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} | "
            f"{item['tp']} | {item['fp']} | {item['fn']} |"
        )
    lines.append("")
    lines.append("## Confusion")
    lines.append("")
    for key, count in result["metrics"]["confusion"].items():
        lines.append(f"- `{key}`: {count}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate verifier-agent decisions")
    parser.add_argument("--data", required=True, help="Verifier-agent JSONL with target decisions")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["predictions", "commit_if_segment_full_same_type", "commit_if_segment_full_same_type_else_full_or_replan", "commit_if_specific_progress_and_full_support", "three_way_commit_full_or_replan"], default="commit_if_segment_full_same_type")
    parser.add_argument("--predictions", default="", help="Optional JSONL of model outputs for mode=predictions")
    parser.add_argument("--threshold", type=float, default=0.70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = iter_jsonl(Path(args.data))
    gold = [str(row.get("target", {}).get("decision", "invalid")) for row in rows]
    if args.mode == "predictions":
        pred = load_prediction_decisions(Path(args.predictions))
        if len(pred) != len(gold):
            raise SystemExit(f"prediction count mismatch: {len(pred)} != {len(gold)}")
    else:
        pred = [rule_decision(row.get("packet", {}), args.mode, args.threshold)["decision"] for row in rows]
    result = {"mode": args.mode, "threshold": args.threshold, "data": args.data, "metrics": metrics(gold, pred)}
    (output_dir / "verifier_eval_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(output_dir / "verifier_eval_report.md", result)
    print(json.dumps({"mode": args.mode, "accuracy": result["metrics"]["accuracy"], "macro_f1": result["metrics"]["macro_f1"], "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
