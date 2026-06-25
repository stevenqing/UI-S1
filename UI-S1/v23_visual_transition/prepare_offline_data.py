#!/usr/bin/env python3
"""Prepare offline GUI-360 action and transition examples.

This is intentionally offline-only: every screenshot is the expert/GT screen.
The output is meant for survival-weighted post-training and lightweight
visual-transition auxiliary losses, not online recovery.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


SYSTEM_PROMPT = (
    "You are a helpful assistant. Given a screenshot of the current screen, "
    "user instruction and history of actions, you need to decide the next "
    "action to take."
)

SUPPORTED_ACTIONS = """<action>
- click
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to click at.
- type
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to type at.
    - keys: str, the text or key sequence to input.
- drag
  - Args:
    - start_coordinate: [x, y], where the drag starts.
    - end_coordinate: [x, y], where the drag ends.
- wheel_mouse_input
  - Args:
    - coordinate: [x, y], position on the screen to scroll.
    - wheel_dist: int, wheel notches. Positive=up, negative=down.
</action>"""

ACTION_PROMPT_TEMPLATE = """<image>
{system_prompt}

The instruction is:
{instruction}

The history of actions are:
{history}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen.

First, briefly reason about the current screen and task. Then output exactly one action within <tool_call></tool_call>.
"""

TRANSITION_PROMPT_TEMPLATE = """<image>
<image>
You are checking whether a GUI action is visually consistent with a task.

The instruction is:
{instruction}

The previous actions are:
{history}

The proposed action is:
{action_text}

Compare the first screenshot before the action and the second screenshot after the action. Predict whether the transition is plausible and whether it makes progress toward the instruction. Output compact JSON with keys: plausible, progress, terminate_valid.
"""


def read_episode_jsonl(path: str, max_episodes: int = 0) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            episode = json.loads(line)
            if episode.get("steps"):
                episodes.append(episode)
            if max_episodes and len(episodes) >= max_episodes:
                break
    return episodes


def valid_coord(coord: Any) -> bool:
    return (
        isinstance(coord, (list, tuple))
        and len(coord) >= 2
        and coord[0] is not None
        and coord[1] is not None
    )


def as_int_pair(coord: Any) -> Optional[List[int]]:
    if not valid_coord(coord):
        return None
    return [int(round(float(coord[0]))), int(round(float(coord[1])))]


def normalize_action_type(action_type: str) -> str:
    aliases = {
        "drag": "swipe",
        "scroll": "swipe",
        "wheel_mouse_input": "swipe",
        "tap": "click",
        "left_click": "click",
        "double_click": "click",
        "input": "type",
    }
    key = str(action_type or "").strip().lower()
    return aliases.get(key, key)


def action_family(action: Dict[str, Any], is_last_step: bool = False) -> str:
    action_type = normalize_action_type(action.get("action", ""))
    if action_type in {"click", "long_press"}:
        return "click_grounding"
    if action_type in {"type", "open", "answer", "key"}:
        return "text_value_carry"
    if action_type in {"swipe", "drag"}:
        return "swipe_scroll"
    if action_type == "terminate" or is_last_step:
        return "terminal_step"
    if action_type in {"system_button", "wait"}:
        return "navigation_phase"
    return "other"


def format_action_for_history(action: Dict[str, Any], step_id: int) -> str:
    action_type = normalize_action_type(action.get("action", ""))
    coord = as_int_pair(action.get("coordinate"))

    if action_type == "click":
        if coord:
            return f"Step {step_id}: click(coordinate=[{coord[0]}, {coord[1]}])"
        return f"Step {step_id}: click()"
    if action_type == "type":
        text = str(action.get("text", ""))
        short = text[:30] + "..." if len(text) > 30 else text
        if coord and short:
            return f"Step {step_id}: type(coordinate=[{coord[0]}, {coord[1]}], keys='{short}')"
        if short:
            return f"Step {step_id}: type(keys='{short}')"
        if coord:
            return f"Step {step_id}: type(coordinate=[{coord[0]}, {coord[1]}])"
        return f"Step {step_id}: type()"
    if action_type == "swipe":
        start = as_int_pair(action.get("coordinate") or action.get("startCoordinate"))
        end = as_int_pair(action.get("endCoordinate"))
        if start and end:
            return (
                f"Step {step_id}: drag(start_coordinate=[{start[0]}, {start[1]}], "
                f"end_coordinate=[{end[0]}, {end[1]}])"
            )
        return f"Step {step_id}: drag()"
    return f"Step {step_id}: {action_type}()"


def build_tool_call(action: Dict[str, Any], is_last_step: bool) -> Dict[str, Any]:
    action_type = normalize_action_type(action.get("action", ""))
    function_name = "drag" if action_type == "swipe" else action_type
    args: Dict[str, Any] = {}

    coord = as_int_pair(action.get("coordinate"))
    if action_type == "click":
        if coord:
            args["coordinate"] = coord
    elif action_type == "type":
        if coord:
            args["coordinate"] = coord
        text = action.get("text")
        if text is not None:
            args["keys"] = text
    elif action_type == "swipe":
        start = as_int_pair(action.get("coordinate") or action.get("startCoordinate"))
        end = as_int_pair(action.get("endCoordinate"))
        if start:
            args["start_coordinate"] = start
        if end:
            args["end_coordinate"] = end
    elif action_type == "system_button":
        button = action.get("button")
        if button is not None:
            args["button"] = button

    return {
        "function": function_name,
        "args": args,
        "status": "FINISH" if is_last_step else "CONTINUE",
    }


def tool_call_text(action: Dict[str, Any], is_last_step: bool) -> str:
    payload = json.dumps(build_tool_call(action, is_last_step), ensure_ascii=False, indent=2)
    return f"<tool_call>\n{payload}\n</tool_call>"


def survival_weight(step_idx: int, num_steps: int, family: str) -> float:
    if num_steps <= 0:
        return 1.0
    early = 1.0 + (num_steps - step_idx - 1) / max(num_steps, 1)
    long_task = 1.0 + min(math.log1p(num_steps) / 4.0, 0.8)
    family_mult = {
        "click_grounding": 1.15,
        "text_value_carry": 1.25,
        "swipe_scroll": 1.15,
        "terminal_step": 1.10,
        "navigation_phase": 1.05,
    }.get(family, 1.0)
    return round(early * long_task * family_mult, 4)


def build_action_prompt(goal: str, history: List[str]) -> str:
    return ACTION_PROMPT_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        instruction=goal,
        history="\n".join(history) if history else "None",
        actions=SUPPORTED_ACTIONS,
    )


def iter_examples(episodes: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    action_examples: List[Dict[str, Any]] = []
    transition_examples: List[Dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    missing_images = 0

    for episode in episodes:
        episode_id = str(episode.get("episode_id", ""))
        goal = episode.get("goal", "")
        steps = episode.get("steps", [])
        num_steps = len(steps)
        history: List[str] = []

        for step_idx, step in enumerate(steps):
            action = step.get("action", {}) or {}
            is_last_step = step_idx == num_steps - 1
            action_type = normalize_action_type(action.get("action", ""))
            family = action_family(action, is_last_step)
            family_counts[family] += 1
            action_counts[action_type] += 1

            screenshot = step.get("screenshot")
            if screenshot and not os.path.exists(screenshot):
                missing_images += 1

            weight = survival_weight(step_idx, num_steps, family)
            response_text = tool_call_text(action, is_last_step)
            metadata = {
                "episode_id": episode_id,
                "step_idx": step_idx,
                "num_steps": num_steps,
                "normalized_depth": round(step_idx / max(num_steps - 1, 1), 4),
                "action_type": action_type,
                "family": family,
                "survival_weight": weight,
                "is_last_step": is_last_step,
                "image_w": step.get("image_w", 1040),
                "image_h": step.get("image_h", 736),
            }

            action_examples.append({
                "conversations": [
                    {"from": "human", "value": build_action_prompt(goal, history)},
                    {"from": "gpt", "value": response_text},
                ],
                "images": [screenshot],
                "weight": weight,
                "metadata": metadata,
            })

            if step_idx + 1 < num_steps:
                next_step = steps[step_idx + 1]
                action_text = format_action_for_history(action, step_idx + 1)
                transition_examples.append({
                    "conversations": [
                        {
                            "from": "human",
                            "value": TRANSITION_PROMPT_TEMPLATE.format(
                                instruction=goal,
                                history="\n".join(history) if history else "None",
                                action_text=action_text,
                            ),
                        },
                        {
                            "from": "gpt",
                            "value": json.dumps({
                                "plausible": True,
                                "progress": "positive",
                                "terminate_valid": False,
                                "action_type": action_type,
                            }, ensure_ascii=False),
                        },
                    ],
                    "images": [screenshot, next_step.get("screenshot")],
                    "weight": weight,
                    "metadata": {
                        **metadata,
                        "next_screenshot": next_step.get("screenshot"),
                        "next_step_idx": step_idx + 1,
                    },
                })

            history.append(format_action_for_history(action, step_idx + 1))

    summary = {
        "num_action_examples": len(action_examples),
        "num_transition_examples": len(transition_examples),
        "family_counts": dict(sorted(family_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "missing_images": missing_images,
    }
    return action_examples, transition_examples, summary


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare offline GUI-360 transition/action examples")
    parser.add_argument("--input", required=True, help="Episode JSONL path")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_episodes", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    episodes = read_episode_jsonl(args.input, args.max_episodes)
    action_rows, transition_rows, summary = iter_examples(episodes)
    summary["num_episodes"] = len(episodes)
    summary["input"] = args.input

    action_path = os.path.join(args.output_dir, "offline_action_examples.jsonl")
    transition_path = os.path.join(args.output_dir, "offline_transition_examples.jsonl")
    summary_path = os.path.join(args.output_dir, "summary.json")
    write_jsonl(action_path, action_rows)
    write_jsonl(transition_path, transition_rows)
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {action_path}")
    print(f"Wrote {transition_path}")


if __name__ == "__main__":
    main()