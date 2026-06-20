#!/usr/bin/env python3
"""Evaluate Verifier Agent decisions as Execution Coordinator policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from verifier_agent_runtime import (
    coordinator_command,
    iter_jsonl,
    summarize_commands,
    target_decision,
    write_jsonl,
)


JsonDict = dict[str, Any]


def static_prediction(decision: str, index: int) -> JsonDict:
    return {"index": index, "decision": decision, "assistant": json.dumps({"decision": decision})}


def oracle_prediction(data_row: JsonDict, index: int) -> JsonDict:
    decision = target_decision(data_row)
    return {"index": index, "decision": decision, "assistant": json.dumps({"decision": decision})}


def commands_for_predictions(data_rows: list[JsonDict], prediction_rows: list[JsonDict]) -> list[JsonDict]:
    if len(data_rows) != len(prediction_rows):
        raise ValueError(f"prediction count mismatch: {len(prediction_rows)} != {len(data_rows)}")
    return [
        coordinator_command(data_row, prediction_row)
        for data_row, prediction_row in zip(data_rows, prediction_rows, strict=True)
    ]


def policy_report(name: str, commands: list[JsonDict]) -> JsonDict:
    return {"name": name, **summarize_commands(commands)}


def write_report(path: Path, data_path: str, predictions_path: str, policies: list[JsonDict]) -> None:
    lines = ["# Verifier Agent Coordinator Evaluation", ""]
    lines.append(f"Data: `{data_path}`")
    lines.append(f"Predictions: `{predictions_path}`")
    lines.append("")
    lines.append("| policy | execute rate | action acc all | executed acc | unsafe exec | replan rate | replan abstain recall | missed executable | selected counts |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for item in policies:
        selected_counts = ", ".join(f"{key}:{value}" for key, value in sorted(item["selected_counts"].items()))
        lines.append(
            f"| {item['name']} | {item['execute_rate']:.4f} | {item['action_accuracy_all']:.4f} | "
            f"{item['executed_action_accuracy']:.4f} | {item['unsafe_execution_rate']:.4f} | "
            f"{item['replan_rate']:.4f} | {item['replan_abstain_recall']:.4f} | "
            f"{item['missed_executable_count']} | {selected_counts} |"
        )
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append("`executed_acc` is correctness among commands the coordinator actually executes. `replan_abstain_recall` is the fraction of oracle-replan states that the coordinator withholds instead of forcing an unsafe fallback.")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate verifier coordinator policies")
    parser.add_argument("--data", required=True, help="Verifier Agent JSONL with packets and metadata")
    parser.add_argument("--predictions", required=True, help="Verifier prediction JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_rows = iter_jsonl(Path(args.data))
    prediction_rows = iter_jsonl(Path(args.predictions))
    if args.limit > 0:
        data_rows = data_rows[: args.limit]
        prediction_rows = prediction_rows[: args.limit]
    if len(data_rows) != len(prediction_rows):
        raise SystemExit(f"prediction count mismatch: {len(prediction_rows)} != {len(data_rows)}")

    static_no_history = [static_prediction("use_no_history", index) for index in range(len(data_rows))]
    static_segment = [static_prediction("commit_segment", index) for index in range(len(data_rows))]
    static_full_history = [static_prediction("use_full_history", index) for index in range(len(data_rows))]
    oracle_rows = [oracle_prediction(data_row, index) for index, data_row in enumerate(data_rows)]

    verifier_commands = commands_for_predictions(data_rows, prediction_rows)
    policies = [
        policy_report("always_no_history", commands_for_predictions(data_rows, static_no_history)),
        policy_report("always_segment", commands_for_predictions(data_rows, static_segment)),
        policy_report("always_full_history", commands_for_predictions(data_rows, static_full_history)),
        policy_report("verifier_safety_gate", verifier_commands),
        policy_report("oracle_coordinator", commands_for_predictions(data_rows, oracle_rows)),
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "verifier_safety_gate_commands.jsonl", verifier_commands)
    report = {"data": args.data, "predictions": args.predictions, "policies": policies}
    (output_dir / "coordinator_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(output_dir / "coordinator_report.md", args.data, args.predictions, policies)
    print(json.dumps({"output_dir": str(output_dir), "policies": policies}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()