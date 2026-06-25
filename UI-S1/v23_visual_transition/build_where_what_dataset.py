#!/usr/bin/env python3
"""Build where/what decomposition data for GUI-360 cooperative LoRA experts.

The goal is to train two specialization slots with different supervision:

- WHAT: choose function, status, and non-location arguments such as text.
- WHERE: given the WHAT decision, predict only location arguments.

This script does not train a model. It creates explicit datasets that a later
trainer can route to expert 1 / expert 2 or use for separate losses.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from v13_gui_360.reward import parse_action_from_text  # noqa: E402
from v23_visual_transition.prepare_offline_data import (  # noqa: E402
    action_family,
    as_int_pair,
    build_tool_call,
    format_action_for_history,
    normalize_action_type,
    survival_weight,
)


WHAT_PROMPT = """<image>
You are the WHAT expert for a GUI automation policy.

Decide the next action's function, status, and non-location arguments from the screenshot, task, and action history.
Do not output coordinates, start coordinates, or end coordinates.

The instruction is:
{goal}

The history of actions are:
{history}

Output exactly one JSON object inside <what_call></what_call>.
"""


WHERE_PROMPT = """<image>
You are the WHERE expert for a GUI automation policy.

The WHAT expert has already chosen the action semantics below. Predict only the missing location arguments from the screenshot.

The instruction is:
{goal}

The history of actions are:
{history}

The WHAT decision is:
{what_json}

Output exactly one JSON object inside <where_args></where_args> containing only coordinate, start_coordinate, or end_coordinate fields that are needed.
"""


def read_jsonl(path: str, max_rows: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def load_episodes(path: str, max_episodes: int = 0) -> Dict[str, Dict[str, Any]]:
    episodes: Dict[str, Dict[str, Any]] = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            episode = json.loads(line)
            episodes[str(episode.get("episode_id"))] = episode
            if max_episodes and len(episodes) >= max_episodes:
                break
    return episodes


def build_history(steps: List[Dict[str, Any]], step_idx: int) -> List[str]:
    return [
        format_action_for_history(steps[idx].get("action", {}) or {}, idx + 1)
        for idx in range(step_idx)
    ]


def strip_location_args(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    stripped = {
        "function": tool_call.get("function", ""),
        "args": dict(tool_call.get("args") or {}),
        "status": tool_call.get("status", "CONTINUE"),
    }
    for key in ("coordinate", "start_coordinate", "end_coordinate"):
        stripped["args"].pop(key, None)
    return stripped


def what_target_text(tool_call: Dict[str, Any]) -> str:
    payload = strip_location_args(tool_call)
    return "<what_call>\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n</what_call>"


def where_target(action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    action_type = normalize_action_type(action.get("action", ""))
    if action_type in {"click", "long_press", "type"}:
        coord = as_int_pair(action.get("coordinate"))
        if coord:
            return {"coordinate": coord}
        return None
    if action_type == "swipe":
        start = as_int_pair(action.get("coordinate") or action.get("startCoordinate"))
        end = as_int_pair(action.get("endCoordinate"))
        target: Dict[str, Any] = {}
        if start:
            target["start_coordinate"] = start
        if end:
            target["end_coordinate"] = end
        return target or None
    return None


def where_target_text(target: Dict[str, Any]) -> str:
    return "<where_args>\n" + json.dumps(target, ensure_ascii=False, indent=2) + "\n</where_args>"


def prompt_history(history: List[str]) -> str:
    return "\n".join(history) if history else "None"


def build_what_prompt(goal: str, history: List[str]) -> str:
    return WHAT_PROMPT.format(goal=goal, history=prompt_history(history))


def build_where_prompt(goal: str, history: List[str], what_call: Dict[str, Any]) -> str:
    what_json = json.dumps(strip_location_args(what_call), ensure_ascii=False, indent=2)
    return WHERE_PROMPT.format(
        goal=goal,
        history=prompt_history(history),
        what_json=what_json,
    )


def make_examples_for_action(
    *,
    episode_id: str,
    step_idx: int,
    num_steps: int,
    goal: str,
    screenshot: str,
    image_w: int,
    image_h: int,
    history: List[str],
    action: Dict[str, Any],
    source: str,
    source_reward: Optional[float] = None,
    source_weight: Optional[float] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any]]:
    is_last_step = step_idx == num_steps - 1
    tool_call = build_tool_call(action, is_last_step)
    family = action_family(action, is_last_step)
    weight = float(source_weight) if source_weight is not None else survival_weight(step_idx, num_steps, family)

    metadata = {
        "episode_id": episode_id,
        "step_idx": step_idx,
        "num_steps": num_steps,
        "source": source,
        "source_reward": source_reward,
        "family": family,
        "action_type": normalize_action_type(action.get("action", "")),
        "image_w": image_w,
        "image_h": image_h,
    }

    what_row = {
        "conversations": [
            {"from": "human", "value": build_what_prompt(goal, history)},
            {"from": "gpt", "value": what_target_text(tool_call)},
        ],
        "images": [screenshot],
        "weight": weight,
        "metadata": {**metadata, "expert": "what", "loss_target": "function_status_non_location_args"},
    }

    loc_target = where_target(action)
    where_row = None
    if loc_target is not None:
        where_row = {
            "conversations": [
                {"from": "human", "value": build_where_prompt(goal, history, tool_call)},
                {"from": "gpt", "value": where_target_text(loc_target)},
            ],
            "images": [screenshot],
            "weight": weight,
            "metadata": {**metadata, "expert": "where", "loss_target": "location_args"},
        }

    pair_row = {
        "episode_id": episode_id,
        "step_idx": step_idx,
        "goal": goal,
        "history": history,
        "image": screenshot,
        "weight": weight,
        "what_target": strip_location_args(tool_call),
        "where_target": loc_target,
        "full_tool_call": tool_call,
        "metadata": metadata,
    }
    return what_row, where_row, pair_row


def iter_gt_actions(episodes: Dict[str, Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for episode_id, episode in episodes.items():
        goal = episode.get("goal", "")
        steps = episode.get("steps", [])
        num_steps = len(steps)
        for step_idx, step in enumerate(steps):
            screenshot = step.get("screenshot")
            if not screenshot or not os.path.exists(screenshot):
                continue
            yield {
                "episode_id": episode_id,
                "step_idx": step_idx,
                "num_steps": num_steps,
                "goal": goal,
                "screenshot": screenshot,
                "image_w": int(step.get("image_w", 1040)),
                "image_h": int(step.get("image_h", 736)),
                "history": build_history(steps, step_idx),
                "action": step.get("action", {}) or {},
                "source": "gt",
                "source_reward": 1.0,
                "source_weight": None,
            }


def parse_candidate_action(candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    action = candidate.get("pred_action")
    if isinstance(action, dict):
        return action
    text = candidate.get("text") or ""
    return parse_action_from_text(text)


def iter_success_candidates(
    candidate_rows: Iterable[Dict[str, Any]],
    episodes: Dict[str, Dict[str, Any]],
    reward_threshold: float,
    max_per_state: int,
) -> Iterable[Dict[str, Any]]:
    for row in candidate_rows:
        episode_id = str(row.get("episode_id"))
        episode = episodes.get(episode_id)
        if not episode:
            continue
        steps = episode.get("steps", [])
        step_idx = int(row.get("step_idx", 0))
        if step_idx < 0 or step_idx >= len(steps):
            continue

        step = steps[step_idx]
        screenshot = row.get("screenshot") or step.get("screenshot")
        if not screenshot or not os.path.exists(screenshot):
            continue

        selected = []
        seen = set()
        for candidate in sorted(row.get("candidates", []), key=lambda item: float(item.get("reward") or 0.0), reverse=True):
            reward = float(candidate.get("reward") or 0.0)
            if reward < reward_threshold:
                continue
            action = parse_candidate_action(candidate)
            if not action:
                continue
            key = json.dumps(action, sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            selected.append((candidate, action, reward))
            if len(selected) >= max_per_state:
                break

        for rank, (candidate, action, reward) in enumerate(selected):
            yield {
                "episode_id": episode_id,
                "step_idx": step_idx,
                "num_steps": len(steps),
                "goal": row.get("goal") or episode.get("goal", ""),
                "screenshot": screenshot,
                "image_w": int(step.get("image_w", 1040)),
                "image_h": int(step.get("image_h", 736)),
                "history": build_history(steps, step_idx),
                "action": action,
                "source": f"candidate_success_rank_{rank}",
                "source_reward": reward,
                "source_weight": float((row.get("hard_state") or {}).get("weight") or row.get("source_weight") or 1.0),
            }


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GUI-360 where/what expert datasets")
    parser.add_argument("--episode_data", required=True)
    parser.add_argument("--candidate_data", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--max_candidate_rows", type=int, default=0)
    parser.add_argument("--candidate_reward_threshold", type=float, default=0.5)
    parser.add_argument("--max_success_candidates_per_state", type=int, default=1)
    parser.add_argument("--no_candidate_successes", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    episodes = load_episodes(args.episode_data, args.max_episodes)

    action_rows = list(iter_gt_actions(episodes))
    if args.candidate_data and not args.no_candidate_successes:
        candidate_rows = read_jsonl(args.candidate_data, args.max_candidate_rows)
        action_rows.extend(iter_success_candidates(
            candidate_rows,
            episodes,
            args.candidate_reward_threshold,
            args.max_success_candidates_per_state,
        ))

    what_rows: List[Dict[str, Any]] = []
    where_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    counts = Counter()
    source_counts = Counter()
    family_counts = Counter()

    for item in action_rows:
        what_row, where_row, pair_row = make_examples_for_action(**item)
        what_rows.append(what_row)
        if where_row is not None:
            where_rows.append(where_row)
            counts["with_where"] += 1
        else:
            counts["without_where"] += 1
        pair_rows.append(pair_row)
        source_counts[pair_row["metadata"]["source"]] += 1
        family_counts[pair_row["metadata"]["family"]] += 1

    what_path = os.path.join(args.output_dir, "what_examples.jsonl")
    where_path = os.path.join(args.output_dir, "where_examples.jsonl")
    pair_path = os.path.join(args.output_dir, "paired_where_what_examples.jsonl")
    write_jsonl(what_path, what_rows)
    write_jsonl(where_path, where_rows)
    write_jsonl(pair_path, pair_rows)

    summary = {
        "episode_data": args.episode_data,
        "candidate_data": args.candidate_data,
        "num_episodes": len(episodes),
        "num_action_sources": len(action_rows),
        "num_what_examples": len(what_rows),
        "num_where_examples": len(where_rows),
        "num_paired_examples": len(pair_rows),
        "counts": dict(counts),
        "source_counts": dict(source_counts),
        "family_counts": dict(family_counts),
        "candidate_reward_threshold": args.candidate_reward_threshold,
        "max_success_candidates_per_state": args.max_success_candidates_per_state,
        "outputs": {
            "what": what_path,
            "where": where_path,
            "paired": pair_path,
        },
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()