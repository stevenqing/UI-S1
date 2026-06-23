#!/usr/bin/env python3
"""Build on-policy state-repair probes from full-rollout failures."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


JsonDict = dict[str, Any]


def iter_jsonl(path: Path) -> Iterable[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl_by_id(path: Path) -> dict[str, JsonDict]:
    return {str(row["episode_id"]): row for row in iter_jsonl(path)}


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def unique_strings(values: Iterable[Any]) -> list[str]:
    output = []
    seen = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        key = normalize_text(text)
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


def segment_for_step(episode: JsonDict | None, step_index: int) -> JsonDict | None:
    if not episode:
        return None
    for segment in episode.get("segments", []) or []:
        if int(segment.get("start_step", 0)) <= step_index <= int(segment.get("end_step", -1)):
            return segment
    return None


def step_family(step: JsonDict) -> str:
    if step.get("extract_match"):
        return "ok"
    if not step.get("parse_ok", True):
        return "parse_or_no_action"
    gt = step.get("gt_action_type")
    pred = step.get("pred_action") or {}
    pred_type = pred.get("action") if isinstance(pred, dict) else None
    if not step.get("type_match"):
        if gt == "terminate" or pred_type == "terminate":
            return "terminate_status_or_timing"
        if pred_type is None:
            return "parse_or_no_action"
        return "wrong_action_type"
    if gt == "click":
        return "click_grounding_wrong_target"
    if gt == "type":
        return "text_value_mismatch"
    if gt == "swipe":
        return "swipe_direction_or_context"
    if gt == "system_button":
        return "system_button_mismatch"
    if gt == "long_press":
        return "long_press_grounding"
    if gt == "terminate":
        return "terminate_status_or_timing"
    return "semantic_mismatch_other"


def segment_state(record: JsonDict | None, step_index: int) -> JsonDict:
    if not record:
        return {
            "num_steps": None,
            "num_segments": None,
            "current_segment": None,
            "completed_segments": [],
            "upcoming_segments": [],
        }
    current = segment_for_step(record, step_index)
    completed = [segment for segment in record.get("segments", []) or [] if int(segment.get("end_step", -1)) < step_index]
    upcoming = [segment for segment in record.get("segments", []) or [] if current and int(segment.get("segment_id", -1)) > int(current.get("segment_id", -1))]
    return {
        "num_steps": record.get("num_steps"),
        "num_segments": record.get("num_segments"),
        "current_segment": compact_segment(current),
        "completed_segments": [compact_segment(segment) for segment in completed[-4:]],
        "upcoming_segments": [compact_segment(segment) for segment in upcoming[:3]],
    }


def compact_segment(segment: JsonDict | None) -> JsonDict | None:
    if not segment:
        return None
    memory = segment.get("memory_need", {}) or {}
    return {
        "segment_id": segment.get("segment_id"),
        "start_step": segment.get("start_step"),
        "end_step": segment.get("end_step"),
        "summary": segment.get("summary", ""),
        "dominant_capability": segment.get("dominant_capability", "unknown"),
        "memory_strength": memory.get("strength", "unknown"),
        "memory_reasons": memory.get("reasons", []),
        "carried_values": segment.get("carried_values", []) or [],
    }


def state_text(state: JsonDict, label: str) -> str:
    current = state.get("current_segment") or {}
    completed = state.get("completed_segments") or []
    upcoming = state.get("upcoming_segments") or []
    lines = [f"Task State ({label}):"]
    lines.append(f"- Total steps: {state.get('num_steps')}; total segments: {state.get('num_segments')}")
    lines.append(f"- Current segment: S{current.get('segment_id')} steps {current.get('start_step')}-{current.get('end_step')}")
    lines.append(f"- Current phase/capability: {current.get('dominant_capability', 'unknown')}")
    lines.append(f"- Current segment summary: {current.get('summary', '')}")
    lines.append(f"- Memory strength: {current.get('memory_strength', 'unknown')}; reasons: {current.get('memory_reasons', [])}")
    lines.append(f"- Active carried values: {current.get('carried_values', [])}")
    if completed:
        lines.append("- Recently completed subgoals:")
        for segment in completed:
            lines.append(f"  - S{segment.get('segment_id')} steps {segment.get('start_step')}-{segment.get('end_step')}: {segment.get('summary', '')}")
    else:
        lines.append("- Recently completed subgoals: none")
    if upcoming:
        lines.append("- Upcoming high-level subgoals:")
        for segment in upcoming:
            lines.append(f"  - S{segment.get('segment_id')} steps {segment.get('start_step')}-{segment.get('end_step')}: {segment.get('summary', '')}")
    else:
        lines.append("- Upcoming high-level subgoals: unknown or none")
    return "\n".join(lines)


def prior_available_text(episode: JsonDict | None, step_index: int, goal: str) -> str:
    parts = [goal]
    if episode:
        for index in range(min(step_index, len(episode.get("steps", []) or []))):
            step = episode["steps"][index]
            parts.append(step.get("step_instruction", ""))
            parts.append(json.dumps(step.get("action_content"), ensure_ascii=False))
    return normalize_text("\n".join(parts))


def split_available_values(values: Iterable[Any], available_text: str) -> tuple[list[str], list[str]]:
    legal = []
    withheld = []
    for value in unique_strings(values):
        normalized = normalize_text(value)
        if normalized and normalized in available_text:
            legal.append(value)
        else:
            withheld.append(value)
    return legal, withheld


def prior_steps_inside_current_segment_text(episode: JsonDict | None, segment: JsonDict | None, step_index: int) -> list[str]:
    if not episode or not segment:
        return ["- Prior steps in current segment: unavailable"]
    start_step = int(segment.get("start_step", 0) or 0)
    lines = ["- Prior steps in current segment:"]
    if step_index <= start_step:
        lines.append("  - none")
        return lines
    for index in range(start_step, min(step_index, len(episode.get("steps", []) or []))):
        step = episode["steps"][index]
        lines.append(
            f"  - Step {index}: instruction={step.get('step_instruction', '')}; action={json.dumps(step.get('action_content'), ensure_ascii=False)}"
        )
    return lines


def prefix_only_state_text(
    state: JsonDict,
    label: str,
    episode: JsonDict | None,
    step_index: int,
    goal: str,
) -> tuple[str, JsonDict]:
    current = state.get("current_segment") or {}
    completed = state.get("completed_segments") or []
    available_text = prior_available_text(episode, step_index, goal)
    carried_candidates = []
    for segment in completed:
        carried_candidates.extend(segment.get("carried_values", []) or [])
    carried_candidates.extend(current.get("carried_values", []) or [])
    legal_values, withheld_values = split_available_values(carried_candidates, available_text)

    lines = [f"Task State ({label}; prefix-only/no-future):"]
    lines.append(f"- Total steps: {state.get('num_steps')}; total segments: {state.get('num_segments')}")
    lines.append(f"- Current step index: {step_index}")
    if current:
        lines.append(f"- Current segment: S{current.get('segment_id')} started at step {current.get('start_step')}")
    else:
        lines.append("- Current segment: unknown")
    lines.append(f"- Active carried values already available from goal/prior steps: {legal_values}")
    if completed:
        lines.append("- Completed subgoals:")
        for segment in completed:
            lines.append(f"  - S{segment.get('segment_id')} steps {segment.get('start_step')}-{segment.get('end_step')}: {segment.get('summary', '')}")
    else:
        lines.append("- Completed subgoals: none")
    lines.extend(prior_steps_inside_current_segment_text(episode, current, step_index))
    lines.append("- Upcoming subgoals: withheld")
    lines.append("- Current-segment future summary: withheld")
    audit = {
        "legal_carried_values": legal_values,
        "withheld_future_or_unavailable_values": withheld_values,
    }
    return "\n".join(lines), audit


def no_upcoming_state_text(state: JsonDict, label: str) -> str:
    trimmed = dict(state)
    trimmed["upcoming_segments"] = []
    return state_text(trimmed, label)


def history_text(episode: JsonDict, step_index: int, max_steps: int = 12) -> str:
    start = max(0, step_index - max_steps)
    lines = ["Ground-truth prior progress before the current screenshot:"]
    for index in range(start, step_index):
        step = episode["steps"][index]
        lines.append(
            f"- Step {index}: instruction={step.get('step_instruction', '')}; action={json.dumps(step.get('action_content'), ensure_ascii=False)}"
        )
    if step_index == 0:
        lines.append("- No previous steps.")
    return "\n".join(lines)


def first_error_step(result: JsonDict) -> JsonDict | None:
    for step in result.get("step_results", []) or []:
        if not step.get("extract_match"):
            return step
    return None


def build_probe(
    result: JsonDict,
    step: JsonDict,
    episode: JsonDict,
    segment_record: JsonDict | None,
    wrong_state_record: JsonDict | None,
    wrong_episode: JsonDict | None,
    wrong_step_index: int,
    probe_kind: str,
    state_mode: str,
) -> JsonDict:
    step_index = int(step["step_num"])
    correct_state = segment_state(segment_record, step_index)
    wrong_state = segment_state(wrong_state_record, wrong_step_index)
    gt_step = episode["steps"][step_index]
    correct_text = state_text(correct_state, "correct")
    wrong_text = state_text(wrong_state, "wrong/mismatched")
    audit = None
    if state_mode == "no_upcoming":
        correct_text = no_upcoming_state_text(correct_state, "correct")
        wrong_text = no_upcoming_state_text(wrong_state, "wrong/mismatched")
    elif state_mode == "prefix_only_no_future":
        correct_text, audit = prefix_only_state_text(correct_state, "correct", episode, step_index, result.get("goal", ""))
        wrong_text, _wrong_audit = prefix_only_state_text(
            wrong_state,
            "wrong/mismatched",
            wrong_episode,
            wrong_step_index,
            (wrong_episode or {}).get("goal", ""),
        )
    return {
        "probe_id": f"{result['episode_id']}:{step_index}:{probe_kind}",
        "probe_kind": probe_kind,
        "episode_id": str(result["episode_id"]),
        "step_index": step_index,
        "num_steps": int(result["num_steps"]),
        "category": result.get("category", ""),
        "goal": result.get("goal", ""),
        "screenshot": gt_step.get("screenshot"),
        "gt_action": gt_step.get("action_content"),
        "check_options": gt_step.get("check_options"),
        "rollout_pred_action": step.get("pred_action"),
        "rollout_raw_response": step.get("raw_response", ""),
        "rollout_failure_family": step_family(step),
        "rollout_type_match": bool(step.get("type_match")),
        "rollout_value_match": bool(step.get("extract_match")),
        "correct_state": correct_state,
        "wrong_state": wrong_state,
        "screen_only_state_text": "No additional task state is provided. Use only the global task and current screenshot.",
        "correct_task_state_text": correct_text,
        "wrong_task_state_text": wrong_text,
        "full_history_text": history_text(episode, step_index),
        "state_mode": state_mode,
        "correct_state_no_future_audit": audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build on-policy state repair probes from corrected full-rollout failures")
    parser.add_argument("--trajectory-results", type=Path, required=True)
    parser.add_argument("--jsonl-file", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-probes", type=int, default=400)
    parser.add_argument("--selection-mode", choices=["priority", "first_error_all"], default="priority")
    parser.add_argument("--state-mode", choices=["full", "no_upcoming", "prefix_only_no_future"], default="full")
    parser.add_argument("--min-num-steps", type=int, default=-1, help="Minimum episode length. Default preserves old priority behavior.")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    episodes = load_jsonl_by_id(args.jsonl_file)
    segments = load_jsonl_by_id(args.segments)
    segment_records = list(segments.values())
    results = list(iter_jsonl(args.trajectory_results))
    min_num_steps = args.min_num_steps if args.min_num_steps >= 0 else (11 if args.selection_mode == "priority" else 0)

    candidates: list[tuple[int, JsonDict, JsonDict, str]] = []
    seen = set()
    population = {
        "dataset_episodes": len(episodes),
        "rollout_results": len(results),
        "selection_mode": args.selection_mode,
        "state_mode": args.state_mode,
        "min_num_steps": min_num_steps,
        "max_probes": args.max_probes,
        "skipped_by_min_num_steps": 0,
        "rollout_success_episodes": 0,
        "failed_episodes_with_first_error": 0,
    }
    for result in results:
        num_steps = int(result.get("num_steps") or 0)
        if num_steps < min_num_steps:
            population["skipped_by_min_num_steps"] += 1
            continue
        first = first_error_step(result)
        if not first:
            population["rollout_success_episodes"] += 1
            continue
        population["failed_episodes_with_first_error"] += 1
        if args.selection_mode == "first_error_all":
            key = (str(result["episode_id"]), int(first["step_num"]))
            if key not in seen:
                seen.add(key)
                candidates.append((0, result, first, "first_error_all"))
            continue
        if first:
            key = (str(result["episode_id"]), int(first["step_num"]))
            if key not in seen:
                seen.add(key)
                priority = 0
                family = step_family(first)
                if family in {"wrong_action_type", "click_grounding_wrong_target", "terminate_status_or_timing"}:
                    priority += 10
                if int(result.get("num_steps") or 0) > 15:
                    priority += 5
                candidates.append((priority, result, first, "first_error_long"))
        for step in result.get("step_results", []) or []:
            family = step_family(step)
            if family not in {"terminate_status_or_timing", "wrong_action_type"}:
                continue
            step_index = int(step["step_num"])
            if step_index < 6:
                continue
            key = (str(result["episode_id"]), step_index)
            if key in seen:
                continue
            seen.add(key)
            candidates.append((7 if family == "terminate_status_or_timing" else 5, result, step, family))

    if args.selection_mode == "priority":
        candidates.sort(key=lambda item: (item[0], int(item[1].get("num_steps") or 0), int(item[2].get("step_num") or 0)), reverse=True)
    selected = candidates if args.max_probes <= 0 else candidates[: args.max_probes]
    probes = []
    for _priority, result, step, kind in selected:
        episode = episodes[str(result["episode_id"])]
        segment_record = segments.get(str(result["episode_id"]))
        wrong_record = random.choice(segment_records)
        while wrong_record.get("episode_id") == result.get("episode_id") and len(segment_records) > 1:
            wrong_record = random.choice(segment_records)
        wrong_step_index = min(int(step["step_num"]), int(wrong_record.get("num_steps") or 1) - 1)
        wrong_episode = episodes.get(str(wrong_record.get("episode_id")))
        probes.append(build_probe(result, step, episode, segment_record, wrong_record, wrong_episode, wrong_step_index, kind, args.state_mode))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for probe in probes:
            handle.write(json.dumps(probe, ensure_ascii=False) + "\n")

    family_counts = {}
    for probe in probes:
        family_counts[probe["rollout_failure_family"]] = family_counts.get(probe["rollout_failure_family"], 0) + 1
    population["candidate_steps_before_cap"] = len(candidates)
    population["probes_written"] = len(probes)
    population["selected_failure_families"] = family_counts
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(population, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"probes": len(probes), "families": family_counts, "output": str(args.output), "manifest": str(manifest_path), "population": population}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()