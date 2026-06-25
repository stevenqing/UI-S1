#!/usr/bin/env python3
"""Mine offline hard states from GUI-360 eval_results JSON.

The evaluator runs on GT screenshots with predicted history. Under stop-on-error,
the first failed step is the high-leverage training target for offline TSR.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional


def load_episode_jsonl(path: str) -> Dict[str, Dict[str, Any]]:
    episodes: Dict[str, Dict[str, Any]] = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            episode = json.loads(line)
            episodes[str(episode.get("episode_id"))] = episode
    return episodes


def load_eval_results(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path) as handle:
        data = json.load(handle)
    return {str(key): value for key, value in data.items()}


def action_family(action_type: Optional[str]) -> str:
    action_type = (action_type or "").strip().lower()
    aliases = {"drag": "swipe", "wheel_mouse_input": "swipe", "scroll": "swipe", "tap": "click"}
    action_type = aliases.get(action_type, action_type)
    if action_type in {"click", "long_press"}:
        return "click_grounding"
    if action_type in {"type", "open", "answer", "key"}:
        return "text_value_carry"
    if action_type in {"swipe", "drag"}:
        return "swipe_scroll"
    if action_type in {"system_button", "wait"}:
        return "navigation_phase"
    if action_type == "terminate":
        return "terminal_step"
    return "other"


def failure_kind(step_result: Dict[str, Any]) -> str:
    pred_action = step_result.get("pred_action")
    format_reward = float(step_result.get("format_reward") or 0.0)
    type_reward = float(step_result.get("type_reward") or 0.0)
    content_reward = float(step_result.get("content_reward") or 0.0)
    pred_type = step_result.get("pred_type")
    gt_type = step_result.get("gt_type")
    pred_text = str(step_result.get("pred_text") or "").lower()

    if pred_action is None or format_reward <= 0.0:
        if "finish" in pred_text or "terminate" in pred_text:
            return "premature_finish_or_unparsed_terminate"
        return "parse_or_format_failure"
    if type_reward <= 0.0:
        return f"type_mismatch:{pred_type}->{gt_type}"
    if content_reward < 0.5:
        return "content_mismatch_severe"
    return "content_mismatch_near"


def depth_bin(step_idx: int, num_steps: int) -> str:
    if step_idx == 0:
        return "0"
    if step_idx == 1:
        return "1"
    if step_idx == 2:
        return "2"
    if step_idx <= 4:
        return "3-4"
    if step_idx <= 7:
        return "5-7"
    if step_idx <= 10:
        return "8-10"
    return "11+"


def hard_weight(step_idx: int, num_steps: int, kind: str, family: str, is_first_error: bool) -> float:
    early = 1.0 + (num_steps - step_idx - 1) / max(num_steps, 1)
    first = 1.5 if is_first_error else 1.0
    kind_mult = 1.3 if "premature" in kind or "type_mismatch" in kind else 1.0
    family_mult = {
        "click_grounding": 1.15,
        "text_value_carry": 1.25,
        "swipe_scroll": 1.15,
    }.get(family, 1.0)
    return round(early * first * kind_mult * family_mult, 4)


def mine_hard_states(
    eval_results: Dict[str, Dict[str, Any]],
    episodes: Dict[str, Dict[str, Any]],
    include_previous: bool = True,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    counters: Dict[str, Counter[str]] = defaultdict(Counter)
    totals = Counter()

    for episode_id, result in eval_results.items():
        episode = episodes.get(str(episode_id))
        if not episode:
            counters["missing_episode"][episode_id] += 1
            continue

        num_steps = int(result.get("num_steps") or len(episode.get("steps", [])))
        first_error_step = result.get("first_error_step")
        step_results = result.get("steps", [])

        if not first_error_step:
            totals["successful_episodes"] += 1
            continue

        failed_idx = int(first_error_step) - 1
        candidate_indices = [failed_idx]
        if include_previous and failed_idx > 0:
            candidate_indices.append(failed_idx - 1)

        for idx in candidate_indices:
            if idx < 0 or idx >= len(episode.get("steps", [])):
                continue
            step = episode["steps"][idx]
            step_result = step_results[idx] if idx < len(step_results) else {}
            is_first_error = idx == failed_idx
            kind = failure_kind(step_result) if is_first_error else "pre_first_error_context"
            gt_type = step_result.get("gt_type") or (step.get("action", {}) or {}).get("action")
            family = action_family(gt_type)
            row = {
                "episode_id": episode_id,
                "goal": episode.get("goal", ""),
                "step_idx": idx,
                "num_steps": num_steps,
                "depth_bin": depth_bin(idx, num_steps),
                "is_first_error": is_first_error,
                "failure_kind": kind,
                "family": family,
                "screenshot": step.get("screenshot"),
                "gt_action": step.get("action"),
                "pred_action": step_result.get("pred_action"),
                "pred_text": step_result.get("pred_text"),
                "reward": step_result.get("reward"),
                "format_reward": step_result.get("format_reward"),
                "type_reward": step_result.get("type_reward"),
                "content_reward": step_result.get("content_reward"),
                "weight": hard_weight(idx, num_steps, kind, family, is_first_error),
            }
            rows.append(row)
            counters["failure_kind"][kind] += 1
            counters["family"][family] += 1
            counters["depth_bin"][row["depth_bin"]] += 1
            totals["hard_rows"] += 1
            if is_first_error:
                totals["failed_episodes"] += 1

    summary = {
        "num_eval_episodes": len(eval_results),
        "num_hard_rows": len(rows),
        "totals": dict(totals),
        "failure_kind": dict(counters["failure_kind"].most_common()),
        "family": dict(counters["family"].most_common()),
        "depth_bin": dict(counters["depth_bin"].most_common()),
        "missing_episode": dict(counters["missing_episode"].most_common()),
    }
    return rows, summary


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine GUI-360 hard states from eval_results JSON")
    parser.add_argument("--eval_results", required=True)
    parser.add_argument("--episode_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--no_previous", action="store_true", help="Do not include pre-first-error context rows")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    eval_results = load_eval_results(args.eval_results)
    episodes = load_episode_jsonl(args.episode_data)
    rows, summary = mine_hard_states(eval_results, episodes, include_previous=not args.no_previous)

    hard_path = os.path.join(args.output_dir, "hard_states.jsonl")
    summary_path = os.path.join(args.output_dir, "hard_state_summary.json")
    write_jsonl(hard_path, rows)
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {hard_path}")


if __name__ == "__main__":
    main()