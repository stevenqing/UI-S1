#!/usr/bin/env python3
"""Runtime helpers for applying Verifier Agent decisions in a GUI coordinator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluate_verifier_agent import parse_decision_text


JsonDict = dict[str, Any]

DECISION_TO_AGENT = {
    "use_no_history": "no_history_agent",
    "commit_segment": "segment_memory_agent",
    "use_full_history": "full_history_agent",
}

DECISION_TO_CONDITION = {
    "use_no_history": "no_history",
    "commit_segment": "segment_summary",
    "use_full_history": "full_history",
}


def iter_jsonl(path: Path) -> list[JsonDict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prediction_decision(prediction_row: JsonDict) -> str:
    if "decision" in prediction_row:
        return str(prediction_row["decision"])
    if "assistant" in prediction_row:
        return parse_decision_text(str(prediction_row["assistant"]))
    if "content" in prediction_row:
        return parse_decision_text(str(prediction_row["content"]))
    return "invalid"


def condition_value_match(data_row: JsonDict, condition: str | None) -> bool:
    if condition is None:
        return False
    metadata = data_row.get("metadata", {}) or {}
    matches = metadata.get("condition_value_match", {}) or {}
    return bool(matches.get(condition))


def target_decision(data_row: JsonDict) -> str:
    return str((data_row.get("target", {}) or {}).get("decision", "invalid"))


def target_condition(data_row: JsonDict) -> str | None:
    return DECISION_TO_CONDITION.get(target_decision(data_row))


def command_target_condition(command: JsonDict) -> str | None:
    return DECISION_TO_CONDITION.get(str(command.get("target_decision", "invalid")))


def candidate_summary(data_row: JsonDict, agent_name: str | None) -> JsonDict:
    if agent_name is None:
        return {}
    packet = data_row.get("packet", {}) or {}
    candidates = packet.get("candidate_agents", {}) or {}
    return candidates.get(agent_name, {}) or {}


def replan_request(data_row: JsonDict, prediction_row: JsonDict, reason: str, decision: str) -> JsonDict:
    packet = data_row.get("packet", {}) or {}
    return {
        "reason": reason,
        "verifier_decision": decision,
        "verifier_output": prediction_row.get("assistant"),
        "task": packet.get("task", {}),
        "memory": packet.get("memory", {}),
        "candidate_agents": packet.get("candidate_agents", {}),
        "computed_evidence": packet.get("computed_evidence", {}),
        "recommended_next_steps": [
            "generate_alternative_candidate",
            "recover_missing_carried_value",
            "rewrite_current_instruction",
            "rerun_verifier_on_new_packet",
        ],
    }


def coordinator_command(data_row: JsonDict, prediction_row: JsonDict) -> JsonDict:
    decision = prediction_decision(prediction_row)
    selected_agent = DECISION_TO_AGENT.get(decision)
    selected_condition = DECISION_TO_CONDITION.get(decision)
    selected_candidate = candidate_summary(data_row, selected_agent)
    raw_action = selected_candidate.get("raw")
    metadata = data_row.get("metadata", {}) or {}
    command_index = prediction_row.get("index", metadata.get("case_id"))

    base = {
        "index": command_index,
        "verifier_decision": decision,
        "target_decision": target_decision(data_row),
        "selected_agent": selected_agent,
        "selected_condition": selected_condition,
        "metadata": metadata,
    }
    if decision in {"replan", "invalid"}:
        reason = "invalid_verifier_decision" if decision == "invalid" else "verifier_requested_replan"
        return {
            **base,
            "status": "replan",
            "action": None,
            "known_correct": False,
            "replan_request": replan_request(data_row, prediction_row, reason, decision),
        }
    if selected_agent is None or selected_condition is None:
        return {
            **base,
            "status": "replan",
            "action": None,
            "known_correct": False,
            "replan_request": replan_request(data_row, prediction_row, "unknown_verifier_route", decision),
        }
    if raw_action is None:
        return {
            **base,
            "status": "replan",
            "action": None,
            "known_correct": False,
            "replan_request": replan_request(data_row, prediction_row, "selected_candidate_missing_action", decision),
        }
    return {
        **base,
        "status": "execute",
        "action": raw_action,
        "candidate_summary": selected_candidate,
        "known_correct": condition_value_match(data_row, selected_condition),
        "replan_request": None,
    }


def summarize_commands(commands: list[JsonDict]) -> JsonDict:
    total = len(commands)
    execute_commands = [command for command in commands if command.get("status") == "execute"]
    replan_commands = [command for command in commands if command.get("status") == "replan"]
    correct_execute = [command for command in execute_commands if command.get("known_correct")]
    target_replans = [command for command in commands if command.get("target_decision") == "replan"]
    abstained_target_replans = [
        command
        for command in replan_commands
        if command.get("target_decision") == "replan"
    ]
    unsafe_execute = [command for command in execute_commands if not command.get("known_correct")]
    missed_executable = [
        command
        for command in replan_commands
        if command_target_condition(command) is not None
    ]
    status_counts: dict[str, int] = {}
    decision_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    for command in commands:
        status = str(command.get("status", "unknown"))
        decision = str(command.get("verifier_decision", "invalid"))
        selected = str(command.get("selected_condition") or "abstain")
        status_counts[status] = status_counts.get(status, 0) + 1
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        selected_counts[selected] = selected_counts.get(selected, 0) + 1
    return {
        "n": total,
        "status_counts": status_counts,
        "decision_counts": decision_counts,
        "selected_counts": selected_counts,
        "execute_rate": len(execute_commands) / total if total else 0.0,
        "replan_rate": len(replan_commands) / total if total else 0.0,
        "action_accuracy_all": len(correct_execute) / total if total else 0.0,
        "executed_action_accuracy": len(correct_execute) / len(execute_commands) if execute_commands else 0.0,
        "unsafe_execution_rate": len(unsafe_execute) / len(execute_commands) if execute_commands else 0.0,
        "replan_abstain_recall": len(abstained_target_replans) / len(target_replans) if target_replans else 0.0,
        "missed_executable_count": len(missed_executable),
    }