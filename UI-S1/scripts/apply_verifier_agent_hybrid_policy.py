#!/usr/bin/env python3
"""Apply a production-style hybrid verifier policy.

Easy states execute the no-history candidate directly. Hard states use Verifier
Agent predictions and can optionally apply a stricter runtime safety filter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from verifier_agent_runtime import coordinator_command, iter_jsonl, summarize_commands, write_jsonl


JsonDict = dict[str, Any]


def row_key(row: JsonDict) -> tuple[Any, Any, Any]:
    metadata = row.get("metadata", {}) or {}
    return (metadata.get("case_id"), metadata.get("model_key"), metadata.get("thinking_mode"))


def action_type(row: JsonDict, agent_name: str) -> str:
    return str(((row.get("packet", {}) or {}).get("candidate_agents", {}) or {}).get(agent_name, {}).get("action_type", "missing"))


def evidence(row: JsonDict) -> JsonDict:
    return ((row.get("packet", {}) or {}).get("computed_evidence", {}) or {})


def no_history_prediction(index: int) -> JsonDict:
    payload = {
        "decision": "use_no_history",
        "selected_condition": "no_history",
        "confidence": "high",
        "reason_codes": ["hybrid_easy_state_default"],
        "rationale": "The hard-state detector did not trigger, so the current-screen candidate is used directly.",
    }
    return {"index": index, "decision": "use_no_history", "assistant": json.dumps(payload, ensure_ascii=False)}


def replan_prediction(index: int, original_prediction: JsonDict, reason_code: str) -> JsonDict:
    payload = {
        "decision": "replan",
        "selected_condition": None,
        "confidence": "medium",
        "reason_codes": [reason_code],
        "rationale": "The verifier selected an executable route, but the runtime safety filter rejected it.",
        "original_verifier_output": original_prediction.get("assistant"),
    }
    return {"index": index, "decision": "replan", "assistant": json.dumps(payload, ensure_ascii=False)}


def pass_safety_filter(row: JsonDict, prediction: JsonDict, mode: str) -> bool:
    decision = str(prediction.get("decision", "invalid"))
    if mode == "raw":
        return True
    if mode == "balanced":
        item = evidence(row)
        if decision == "commit_segment":
            segment_type = action_type(row, "segment_memory_agent")
            if segment_type == "system_button" and item.get("segment_vs_wrong_same_type"):
                return False
            if item.get("segment_vs_wrong_same_type") and segment_type != "swipe":
                return False
            return True
        if decision == "use_full_history":
            full_type = action_type(row, "full_history_agent")
            return bool(
                full_type in {"swipe", "click", "long_press"}
                and (item.get("full_history_candidate_matches_instruction") or full_type in {"swipe", "click"})
            )
        return decision == "use_no_history"
    if decision == "commit_segment":
        item = evidence(row)
        return bool(
            item.get("segment_candidate_matches_instruction")
            and not item.get("segment_vs_wrong_exact")
            and not item.get("segment_vs_wrong_same_type")
        )
    if decision == "use_full_history":
        item = evidence(row)
        return bool(
            item.get("full_history_candidate_matches_instruction")
            and action_type(row, "full_history_agent") != action_type(row, "distractor_memory_agent")
        )
    return decision == "use_no_history"


def condition_value_match(row: JsonDict, condition: str) -> bool:
    return bool(((row.get("metadata", {}) or {}).get("condition_value_match", {}) or {}).get(condition))


def write_summary(path: Path, all_rows: list[JsonDict], hard_keys: set[tuple[Any, Any, Any]], commands: list[JsonDict], mode: str) -> JsonDict:
    base_summary = summarize_commands(commands)
    baseline_no_history_correct = sum(condition_value_match(row, "no_history") for row in all_rows)
    hard_commands = [command for command in commands if command.get("hybrid_hard_state")]
    hard_execute = [command for command in hard_commands if command.get("status") == "execute"]
    hard_correct = [command for command in hard_execute if command.get("known_correct")]
    summary = {
        **base_summary,
        "policy": "hybrid_no_history_then_verifier",
        "safety_mode": mode,
        "hard_state_count": len(hard_keys),
        "easy_state_count": len(all_rows) - len(hard_keys),
        "baseline_no_history_correct": baseline_no_history_correct,
        "baseline_no_history_action_accuracy": baseline_no_history_correct / len(all_rows) if all_rows else 0.0,
        "delta_correct_vs_no_history": len([command for command in commands if command.get("known_correct")]) - baseline_no_history_correct,
        "hard_execute_count": len(hard_execute),
        "hard_executed_action_accuracy": len(hard_correct) / len(hard_execute) if hard_execute else 0.0,
        "hard_unsafe_execution_count": len(hard_execute) - len(hard_correct),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a hybrid no-history/verifier policy over full-distribution rows")
    parser.add_argument("--all-data", required=True, help="Full-distribution verifier-agent JSONL")
    parser.add_argument("--hard-data", required=True, help="Hard-only verifier-agent JSONL used for predictions")
    parser.add_argument("--hard-predictions", required=True, help="Predictions for rows in --hard-data")
    parser.add_argument("--output", required=True, help="Output JSONL of hybrid coordinator commands")
    parser.add_argument("--summary", required=True, help="Output JSON summary")
    parser.add_argument("--safety-mode", choices=["raw", "balanced", "high_precision"], default="balanced")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_rows = iter_jsonl(Path(args.all_data))
    hard_rows = iter_jsonl(Path(args.hard_data))
    hard_predictions = iter_jsonl(Path(args.hard_predictions))
    if args.limit > 0:
        all_rows = all_rows[: args.limit]
    if len(hard_rows) != len(hard_predictions):
        raise SystemExit(f"hard prediction count mismatch: {len(hard_predictions)} != {len(hard_rows)}")

    hard_by_key = {row_key(row): row for row in hard_rows}
    pred_by_key = {row_key(row): prediction for row, prediction in zip(hard_rows, hard_predictions, strict=True)}
    hard_keys = set(hard_by_key)
    commands = []
    for index, row in enumerate(all_rows):
        key = row_key(row)
        if key not in hard_keys:
            prediction = no_history_prediction(index)
            command = coordinator_command(row, prediction)
            command["hybrid_hard_state"] = False
            command["hybrid_policy"] = "default_no_history"
            commands.append(command)
            continue

        prediction = pred_by_key[key]
        if not pass_safety_filter(row, prediction, args.safety_mode):
            prediction = replan_prediction(index, prediction, f"{args.safety_mode}_runtime_filter")
        else:
            prediction = {**prediction, "index": index}
        command = coordinator_command(row, prediction)
        command["hybrid_hard_state"] = True
        command["hybrid_policy"] = f"verifier_{args.safety_mode}"
        commands.append(command)

    write_jsonl(Path(args.output), commands)
    summary = write_summary(Path(args.summary), all_rows, hard_keys, commands, args.safety_mode)
    print(json.dumps({"output": args.output, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()