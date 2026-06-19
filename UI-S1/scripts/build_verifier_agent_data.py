#!/usr/bin/env python3
"""Build SFT/eval packets for an agentic memory-route verifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_counterfactual_memory_utility import (  # noqa: E402
    action_matches_any_intent,
    action_text,
    action_type,
    candidate_action,
    distractor_action,
    instruction_intents,
    instruction_text,
    load_split,
    score_rows,
)
from evaluate_memory_router_cascade import same_exact, same_type  # noqa: E402


JsonDict = dict[str, Any]

VERIFIER_SYSTEM_PROMPT = """You are a Candidate Verification Agent in a multi-agent GUI controller.
Your job is not to predict a new GUI action. Your job is to inspect candidate actions proposed by context agents and choose the safest route.

Available routes:
- use_no_history: current-screen candidate is sufficient or memory is unsafe.
- commit_segment: segment-memory candidate is memory-specific, instruction-progressing, and valid enough to execute.
- use_full_history: compact segment memory appears insufficient but raw history provides a supported candidate.
- replan: candidates are unstable, nonspecific, missing, or all known contexts appear unreliable.

Return exactly one JSON object with keys:
{"decision": route, "selected_condition": condition_or_null, "confidence": "high|medium|low", "reason_codes": [strings], "rationale": short_string}
"""


def ok(row: JsonDict, condition: str) -> bool:
    return bool((row.get("condition_value_match", {}) or {}).get(condition))


def decision_label(row: JsonDict) -> JsonDict:
    no_ok = ok(row, "no_history")
    seg_ok = ok(row, "segment_summary")
    full_ok = ok(row, "full_history")
    wrong_ok = ok(row, "wrong_summary")
    if no_ok:
        reason_codes = ["current_screen_sufficient"]
        if not seg_ok:
            reason_codes.append("avoid_segment_regression")
        return {
            "decision": "use_no_history",
            "selected_condition": "no_history",
            "confidence": "high" if no_ok and not seg_ok else "medium",
            "reason_codes": reason_codes,
            "rationale": "The no-history candidate already succeeds, so segment memory is unnecessary or risky.",
        }
    if seg_ok and not wrong_ok:
        return {
            "decision": "commit_segment",
            "selected_condition": "segment_summary",
            "confidence": "high",
            "reason_codes": ["memory_specific_rescue", "wrong_memory_rejected"],
            "rationale": "Segment memory succeeds while no-history and wrong-memory do not, so the repair is memory-specific.",
        }
    if full_ok and not seg_ok:
        return {
            "decision": "use_full_history",
            "selected_condition": "full_history",
            "confidence": "medium",
            "reason_codes": ["segment_summary_insufficient", "raw_history_rescue"],
            "rationale": "Compact segment memory is insufficient, but raw history rescues the action.",
        }
    if seg_ok and wrong_ok:
        return {
            "decision": "replan",
            "selected_condition": None,
            "confidence": "medium",
            "reason_codes": ["nonspecific_context_success", "wrong_memory_also_succeeds"],
            "rationale": "The segment candidate is not memory-specific because wrong memory also succeeds.",
        }
    return {
        "decision": "replan",
        "selected_condition": None,
        "confidence": "high",
        "reason_codes": ["all_contexts_failed", "candidate_unreliable"],
        "rationale": "All known context variants fail, so another candidate source or replanning is needed.",
    }


def action_summary(action: JsonDict | None) -> JsonDict:
    return {
        "action_type": action_type(action),
        "action_text": action_text(action),
        "raw": action,
    }


def packet_for_row(row: JsonDict, score: float) -> JsonDict:
    intents = sorted(instruction_intents(instruction_text(row)))
    segment_action = candidate_action(row, "true_memory")
    no_action = candidate_action(row, "no_memory")
    wrong_action = distractor_action(row, "true_memory")
    full_action = (row.get("pred_actions", {}) or {}).get("full_history")
    return {
        "task": {
            "goal": (row.get("current_state_parts", {}) or {}).get("goal", ""),
            "current_instruction": (row.get("current_state_parts", {}) or {}).get("instruction", ""),
            "current_observation": (row.get("current_state_parts", {}) or {}).get("observation", ""),
            "current_segment": (row.get("current_state_parts", {}) or {}).get("current_segment", ""),
            "instruction_intents": intents,
        },
        "memory": {
            "segment_memory": row.get("true_memory_text", ""),
            "distractor_memory": row.get("wrong_memory_text", ""),
        },
        "candidate_agents": {
            "no_history_agent": action_summary(no_action),
            "segment_memory_agent": action_summary(segment_action),
            "full_history_agent": action_summary(full_action),
            "distractor_memory_agent": action_summary(wrong_action),
        },
        "computed_evidence": {
            "memory_proposal_score": float(score),
            "segment_vs_no_same_type": same_type(row, "segment_summary", "no_history"),
            "segment_vs_no_exact": same_exact(row, "segment_summary", "no_history"),
            "segment_vs_full_same_type": same_type(row, "segment_summary", "full_history"),
            "segment_vs_full_exact": same_exact(row, "segment_summary", "full_history"),
            "segment_vs_wrong_same_type": same_type(row, "segment_summary", "wrong_summary"),
            "segment_vs_wrong_exact": same_exact(row, "segment_summary", "wrong_summary"),
            "segment_candidate_matches_instruction": action_matches_any_intent(segment_action, set(intents)),
            "no_history_candidate_matches_instruction": action_matches_any_intent(no_action, set(intents)),
            "full_history_candidate_matches_instruction": action_matches_any_intent(full_action, set(intents)),
            "wrong_candidate_matches_instruction": action_matches_any_intent(wrong_action, set(intents)),
        },
    }


def user_prompt(packet: JsonDict) -> str:
    return "Verify this multi-agent candidate packet and choose a route:\n" + json.dumps(packet, ensure_ascii=False, indent=2)


def row_record(row: JsonDict, score: float) -> JsonDict:
    packet = packet_for_row(row, score)
    target = decision_label(row)
    return {
        "messages": [
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(packet)},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ],
        "packet": packet,
        "target": target,
        "metadata": {
            **(row.get("metadata", {}) or {}),
            "utility_label": row.get("utility_label"),
            "condition_value_match": row.get("condition_value_match"),
        },
    }


def write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: list[JsonDict]) -> JsonDict:
    decisions: dict[str, int] = {}
    labels: dict[str, int] = {}
    capabilities: dict[str, int] = {}
    for row in rows:
        decision = row["target"]["decision"]
        decisions[decision] = decisions.get(decision, 0) + 1
        label = str(row.get("metadata", {}).get("utility_label", "unknown"))
        labels[label] = labels.get(label, 0) + 1
        capability = str(row.get("metadata", {}).get("dominant_capability", "unknown"))
        capabilities[capability] = capabilities.get(capability, 0) + 1
    return {
        "rows": len(rows),
        "target_decisions": dict(sorted(decisions.items(), key=lambda item: (-item[1], item[0]))),
        "utility_labels": dict(sorted(labels.items(), key=lambda item: (-item[1], item[0]))),
        "capabilities": dict(sorted(capabilities.items(), key=lambda item: (-item[1], item[0]))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build verifier-agent candidate packets from CMU rows")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--proposal-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--proposal-threshold", type=float, default=0.0, help="Keep all rows with proposal score >= this value")
    parser.add_argument("--hard-only", action="store_true", help="Keep rows where at least one non-no-history route is the target or no_history fails")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = joblib.load(args.proposal_model)
    stats: dict[str, JsonDict] = {}
    for split in ["train", "dev", "test"]:
        source_rows = load_split(data_dir, split)
        scores = score_rows(model, source_rows)
        output_rows = []
        for row, score in zip(source_rows, scores):
            target = decision_label(row)
            no_ok = ok(row, "no_history")
            if score < args.proposal_threshold:
                continue
            if args.hard_only and target["decision"] == "use_no_history" and no_ok:
                continue
            output_rows.append(row_record(row, float(score)))
        write_jsonl(output_dir / f"{split}.jsonl", output_rows)
        stats[split] = summarize(output_rows)
    (output_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "verifier_agent_system_prompt.txt").write_text(VERIFIER_SYSTEM_PROMPT, encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
