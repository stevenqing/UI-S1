#!/usr/bin/env python3
"""V17.1 Dual-Aux SFT Data Preparation.

Converts episode-format training data to ShareGPT JSONL with dual-aux format.
Template-based aux content teaches the format structure without overfitting content.

Input: v12_gui_360/data/gui360_train_2000_balanced.jsonl (episode format)
Output: ShareGPT JSONL for v15_gui_360/train_cooperative_sft.py

Aux content is intentionally generic and low-information — no GT action
type is leaked into aux content. Only observational descriptions are used.

This teaches the model the <aux_a>...<aux_b>...<decision> structure.
RL will learn meaningful aux content.

Usage:
  python v17_self_evolve/prepare_dual_aux_sft.py \
    --input_data v12_gui_360/data/gui360_train_2000_balanced.jsonl \
    --output_data v17_self_evolve/data/dual_aux_sft_train.jsonl \
    --output_val v17_self_evolve/data/dual_aux_sft_val.jsonl \
    --val_episodes 50
"""

import argparse
import json
import os
import sys
from typing import Dict, List

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


SUPPORTED_ACTIONS = """<action>
- click
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to click at.
    - button: str, One of 'left', 'right', 'middle' or 'x' (Default: 'left')
    - double: bool, Whether to perform a double click (Default: False)
    - pressed: str|None, Keyboard key to press while clicking (Default: None)
  - Example: click(coordinate=[100, 100], button='left', double=False, pressed=None)
- type
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to type at.
    - keys: str, The key to input.
    - clear_current_text: bool, Whether to clear the current text (Default: False)
    - control_focus: bool, Whether to focus on selected control before typing (Default: True)
  - Example: type(coordinate=[100, 100], keys='Hello')
- drag
  - Args:
    - start_coordinate: [x, y], where the drag starts.
    - end_coordinate: [x, y], where the drag ends.
    - button: str, 'left' or 'right' (Default: 'left')
    - duration: float, Duration in seconds (Default: 1.0)
  - Example: drag(start_coordinate=[100, 100], end_coordinate=[200, 200])
- wheel_mouse_input
  - Args:
    - coordinate: [x, y], position on the screen to scroll.
    - wheel_dist: int, Wheel notches. Positive=up, negative=down.
  - Example: wheel_mouse_input(coordinate=[100, 100], wheel_dist=-5)
</action>"""


USER_PROMPT_TEMPLATE = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

In <aux_a></aux_a>, analyze from perspective A.
In <aux_b></aux_b>, analyze from perspective B, considering perspective A's analysis.
Then output your action inside <decision></decision> tags, with a <tool_call> block:
<decision>
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>
</decision>

If you think the task is finished, set status to "FINISH" and take no action:
<decision>
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>
</decision>

Only **ONE** action should be taken at a time.

<aux_a>
"""


def format_gt_action_for_history(gt_action: Dict, step_id: int) -> str:
    """Convert GT action dict to history format."""
    atype = gt_action.get("action", "")
    coord = gt_action.get("coordinate")

    if atype == "click":
        if _safe_coord(coord):
            return f"Step {step_id}: click(coordinate=[{int(coord[0])}, {int(coord[1])}])"
        return f"Step {step_id}: click()"
    elif atype == "type":
        text = gt_action.get("text", "")
        if _safe_coord(coord) and text:
            t = text[:30] + "..." if len(text) > 30 else text
            return f"Step {step_id}: type(coordinate=[{int(coord[0])}, {int(coord[1])}], keys='{t}')"
        elif text:
            t = text[:30] + "..." if len(text) > 30 else text
            return f"Step {step_id}: type(keys='{t}')"
        elif _safe_coord(coord):
            return f"Step {step_id}: type(coordinate=[{int(coord[0])}, {int(coord[1])}])"
        return f"Step {step_id}: type()"
    elif atype in ("swipe", "drag"):
        start = gt_action.get("coordinate")
        end = gt_action.get("endCoordinate")
        if _safe_coord(start) and _safe_coord(end):
            return (f"Step {step_id}: drag(start_coordinate=[{int(start[0])}, {int(start[1])}], "
                    f"end_coordinate=[{int(end[0])}, {int(end[1])}])")
        return f"Step {step_id}: drag()"
    else:
        return f"Step {step_id}: {atype}()"


def _safe_coord(coord) -> bool:
    """Check if coordinate is valid (non-None list with numeric elements)."""
    return (coord is not None
            and isinstance(coord, (list, tuple))
            and len(coord) >= 2
            and coord[0] is not None
            and coord[1] is not None)


def build_gt_tool_call(gt_action: Dict) -> str:
    """Build the GT tool_call JSON from action dict."""
    atype = gt_action.get("action", "click")
    coord = gt_action.get("coordinate")

    if atype == "click":
        args = {}
        if _safe_coord(coord):
            args["coordinate"] = [int(coord[0]), int(coord[1])]
        args["button"] = "left"
        args["double"] = False
        args["pressed"] = None
        return json.dumps({
            "function": "click",
            "args": args,
            "status": "CONTINUE",
        }, indent=2)

    elif atype == "type":
        args = {}
        if _safe_coord(coord):
            args["coordinate"] = [int(coord[0]), int(coord[1])]
        args["keys"] = gt_action.get("text", "")
        args["clear_current_text"] = False
        args["control_focus"] = True
        return json.dumps({
            "function": "type",
            "args": args,
            "status": "CONTINUE",
        }, indent=2)

    elif atype in ("swipe", "drag"):
        args = {}
        start = gt_action.get("coordinate")
        end = gt_action.get("endCoordinate")
        if _safe_coord(start):
            args["start_coordinate"] = [int(start[0]), int(start[1])]
        if _safe_coord(end):
            args["end_coordinate"] = [int(end[0]), int(end[1])]
        return json.dumps({
            "function": "drag",
            "args": args,
            "status": "CONTINUE",
        }, indent=2)

    else:
        return json.dumps({
            "function": atype,
            "args": {},
            "status": "CONTINUE",
        }, indent=2)


def generate_template_aux(goal: str, step_idx: int) -> tuple:
    """Generate template-based aux_a and aux_b content.

    Intentionally generic — teaches format structure, not content.
    Does NOT include GT action type to avoid information leakage.
    """
    # Truncate goal for aux
    goal_short = goal[:80] + "..." if len(goal) > 80 else goal

    if step_idx == 0:
        aux_a = (f"I observe the current screen. The task requires: {goal_short}. "
                 f"I need to identify the relevant UI element and determine the appropriate action.")
    else:
        aux_a = (f"I observe the current screen after {step_idx} previous actions. "
                 f"The task requires: {goal_short}. "
                 f"Based on the progress so far, I need to determine the next step.")

    aux_b = (f"Considering perspective A's analysis, I examine the screen layout "
             f"and identify the target element for the next action.")

    return aux_a, aux_b


def convert_episode_to_sharegpt(episode: Dict) -> List[Dict]:
    """Convert one episode to list of ShareGPT samples (one per step)."""
    goal = episode["goal"]
    steps = episode["steps"]
    samples = []
    action_history: List[str] = []

    for si, step in enumerate(steps):
        screenshot = step["screenshot"]
        gt_action = step["action"]

        if not os.path.exists(screenshot):
            action_history.append(format_gt_action_for_history(gt_action, si + 1))
            continue

        history_text = "\n".join(action_history) if action_history else "None"

        # Build user prompt
        prompt_text = USER_PROMPT_TEMPLATE.format(
            instruction=goal,
            history=history_text,
            actions=SUPPORTED_ACTIONS,
        )

        # Build assistant response with dual-aux format
        aux_a, aux_b = generate_template_aux(goal, si)
        tool_call_json = build_gt_tool_call(gt_action)

        assistant_text = (
            f"{aux_a}\n"
            f"</aux_a>\n"
            f"<aux_b>\n"
            f"{aux_b}\n"
            f"</aux_b>\n"
            f"<decision>\n"
            f"<tool_call>\n"
            f"{tool_call_json}\n"
            f"</tool_call>\n"
            f"</decision>"
        )

        sample = {
            "conversations": [
                {"from": "human", "value": prompt_text},
                {"from": "gpt", "value": assistant_text},
            ],
            "images": [screenshot],
        }
        samples.append(sample)

        action_history.append(format_gt_action_for_history(gt_action, si + 1))

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="V17.1 Dual-Aux SFT Data Preparation")
    parser.add_argument("--input_data", type=str, required=True,
                        help="Episode JSONL input")
    parser.add_argument("--output_data", type=str, required=True,
                        help="ShareGPT JSONL output (train)")
    parser.add_argument("--output_val", type=str, default="",
                        help="ShareGPT JSONL output (val)")
    parser.add_argument("--val_episodes", type=int, default=50,
                        help="Number of episodes to hold out for validation")
    args = parser.parse_args()

    # Load episodes
    episodes = []
    with open(args.input_data) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episodes.append(json.loads(line))

    print(f"Loaded {len(episodes)} episodes from {args.input_data}")

    # Split train/val
    if args.output_val and args.val_episodes > 0:
        val_episodes = episodes[:args.val_episodes]
        train_episodes = episodes[args.val_episodes:]
    else:
        val_episodes = []
        train_episodes = episodes

    # Convert
    os.makedirs(os.path.dirname(args.output_data), exist_ok=True)

    train_count = 0
    with open(args.output_data, "w") as f:
        for ep in train_episodes:
            samples = convert_episode_to_sharegpt(ep)
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
                train_count += 1

    print(f"Wrote {train_count} train samples to {args.output_data}")

    if args.output_val and val_episodes:
        os.makedirs(os.path.dirname(args.output_val), exist_ok=True)
        val_count = 0
        with open(args.output_val, "w") as f:
            for ep in val_episodes:
                samples = convert_episode_to_sharegpt(ep)
                for s in samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
                    val_count += 1
        print(f"Wrote {val_count} val samples to {args.output_val}")


if __name__ == "__main__":
    main()
