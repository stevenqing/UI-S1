#!/usr/bin/env python3
"""Evaluate models using GUI-360's native prompt template on our balanced test set.

Uses the GUI-360 prompt format (<tool_call> output) but reads from our JSONL
and uses compute_step_reward for consistent metrics across all models.

Usage:
    python v13_gui_360/eval_gui360_template.py \
        --test_data v12_gui_360/data/gui360_test_1000_balanced.jsonl \
        --api_url http://localhost:8000/v1 \
        --model_name gui_action \
        --output_dir v12_gui_360/outputs/gui_action_balanced \
        --threads 128
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v13_gui_360.reward import compute_step_reward, _normalize_action_type


# ═══════════════════════════════════════════════════════════════════════
# GUI-360 Prompt Template (from GUI-360-eval)
# ═══════════════════════════════════════════════════════════════════════

USER_PROMPT_TEMPLATE = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

First, explain your reasoning process—describe how you analyze the screenshot, understand the current state, and determine what action should be taken next based on the instruction and previous actions.

Then output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished, you can output status as "FINISH" and take no action. Like:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time. If the instruction could apply to multiple elements, choose the most relevant one based on the context provided by the screenshot and previous actions.
"""

# IMPORTANT: Must be identical to SUPPORTED_ACTIONS in train_trajectory_rl.py
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


# ═══════════════════════════════════════════════════════════════════════
# Parse <tool_call> output → V12 action format
# ═══════════════════════════════════════════════════════════════════════

def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse <tool_call> output and convert to V12 action format."""
    m = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
    if not m:
        # Fallback: try to find JSON with "function" key
        m = re.search(r'\{[^{}]*"function"[^{}]*\}', text)
    if not m:
        return None

    try:
        tc = json.loads(m.group(1) if '<tool_call>' in text[:m.start()+20] else m.group(0))
    except json.JSONDecodeError:
        return None

    func = tc.get("function", "")
    args = tc.get("args", {})

    if not func:
        return None

    # Handle malformed args (e.g. model outputs args as a list instead of dict)
    if isinstance(args, list):
        if func in ("click", "double_click") and len(args) == 2:
            args = {"coordinate": args}
        else:
            args = {}
    elif not isinstance(args, dict):
        args = {}

    # Convert to V12 action format
    action = {}

    if func == "click":
        action["action"] = "click"
        action["coordinate"] = args.get("coordinate")
    elif func == "type":
        action["action"] = "type"
        action["text"] = args.get("keys", args.get("text", ""))
        if args.get("coordinate"):
            action["coordinate"] = args["coordinate"]
    elif func == "drag":
        action["action"] = "swipe"
        action["coordinate"] = args.get("start_coordinate")
        action["endCoordinate"] = args.get("end_coordinate")
    elif func == "wheel_mouse_input":
        action["action"] = "swipe"
        coord = args.get("coordinate", [0, 0])
        dist = args.get("wheel_dist", -3)
        # Approximate: scroll = vertical movement
        action["coordinate"] = coord
        action["endCoordinate"] = [coord[0], coord[1] - dist * 50]
    elif func == "double_click":
        action["action"] = "click"
        action["coordinate"] = args.get("coordinate")
    else:
        # Unknown function, try to map
        action["action"] = func
        if args.get("coordinate"):
            action["coordinate"] = args["coordinate"]

    return action


def _format_action_for_history(action: Optional[Dict], step_id: int) -> str:
    """Format a predicted action dict for history, matching training's format.

    Mirrors format_gt_action_for_history() in train_trajectory_rl.py so that
    the prompt seen at eval time is consistent with training.
    """
    if action is None:
        return f"Step {step_id}: (no action parsed)"

    atype = action.get("action", "")
    coord = action.get("coordinate")

    # Guard against [None, None] or partial coords
    def _valid_coord(c):
        return c and isinstance(c, (list, tuple)) and len(c) >= 2 and c[0] is not None and c[1] is not None

    if atype == "click":
        if _valid_coord(coord):
            return f"Step {step_id}: click(coordinate=[{int(float(coord[0]))}, {int(float(coord[1]))}])"
        return f"Step {step_id}: click()"

    elif atype == "type":
        text = action.get("text", "")
        if _valid_coord(coord) and text:
            t = text[:30] + "..." if len(text) > 30 else text
            return f"Step {step_id}: type(coordinate=[{int(float(coord[0]))}, {int(float(coord[1]))}], keys='{t}')"
        elif text:
            t = text[:30] + "..." if len(text) > 30 else text
            return f"Step {step_id}: type(keys='{t}')"
        elif _valid_coord(coord):
            return f"Step {step_id}: type(coordinate=[{int(float(coord[0]))}, {int(float(coord[1]))}])"
        return f"Step {step_id}: type()"

    elif atype in ("swipe", "drag"):
        start = action.get("coordinate")
        end = action.get("endCoordinate")
        if _valid_coord(start) and _valid_coord(end):
            return (f"Step {step_id}: drag(start_coordinate=[{int(float(start[0]))}, {int(float(start[1]))}], "
                    f"end_coordinate=[{int(float(end[0]))}, {int(float(end[1]))}])")
        return f"Step {step_id}: drag()"

    else:
        return f"Step {step_id}: {atype}()"


def build_step_prompt(goal: str, screenshot_path: str, step_idx: int, history: List[str]) -> List[dict]:
    """Build messages for a single step using GUI-360 template."""
    history_text = "\n".join(history) if history else "None"

    prompt_text = USER_PROMPT_TEMPLATE.format(
        instruction=goal,
        history=history_text,
        actions=SUPPORTED_ACTIONS,
    )

    # Build multimodal message
    user_content = []

    # Add screenshot as base64
    img = Image.open(screenshot_path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    user_content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64}"}
    })

    user_content.append({"type": "text", "text": prompt_text})

    return [
        {"role": "user", "content": user_content},
    ]


def evaluate_episode(
    client: OpenAI,
    model_name: str,
    episode: Dict,
    stop_on_error: bool = True,
    match_threshold: float = 0.5,
    gt_history: bool = False,
) -> Dict[str, Any]:
    """Evaluate a single episode autoregressively (or with GT history)."""
    episode_id = episode["episode_id"]
    goal = episode["goal"]
    steps = episode["steps"]
    num_steps = len(steps)

    history = []
    step_results = []
    first_error_step = None
    correct_steps = 0

    for i, step in enumerate(steps):
        gt_action = step["action"]
        screenshot = step["screenshot"]
        image_w = step.get("image_w", 1040)
        image_h = step.get("image_h", 736)

        messages = build_step_prompt(goal, screenshot, i, history)

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=1024,
                temperature=0.0,
            )
            pred_text = response.choices[0].message.content or ""
        except Exception as e:
            pred_text = ""
            print(f"  [ep {episode_id}] step {i+1} API error: {e}")

        # Parse <tool_call> output and convert to V12 format
        pred_action = parse_tool_call(pred_text)

        # Also try <action> format as fallback (in case model uses it)
        if pred_action is None:
            m = re.search(r'<action>\s*(\{.*?\})\s*</action>', pred_text, re.DOTALL)
            if m:
                try:
                    pred_action = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass

        # Build a fake pred_text in <action> format for compute_step_reward
        if pred_action:
            fake_text = f"<action>{json.dumps(pred_action)}</action>"
        else:
            fake_text = pred_text  # let reward function try to parse it

        reward, info = compute_step_reward(
            fake_text, gt_action, image_w, image_h
        )

        success = reward >= match_threshold

        step_results.append({
            "step_idx": i,
            "success": success,
            "reward": reward,
            "pred_text": pred_text[:300],
            "pred_action": info.get("pred_action"),
            "pred_type": info.get("pred_type"),
            "gt_type": info.get("gt_type"),
            "format_reward": info.get("format_reward", 0),
            "type_reward": info.get("type_reward", 0),
            "content_reward": info.get("content_reward", 0),
        })

        # Update history: GT history uses ground-truth actions (teacher-forced),
        # autoregressive uses predicted actions
        if gt_history:
            history.append(_format_action_for_history(gt_action, i + 1))
        else:
            history.append(_format_action_for_history(pred_action, i + 1))

        if success:
            correct_steps += 1
        else:
            if first_error_step is None:
                first_error_step = i + 1
            if stop_on_error and not gt_history:
                break

    progress = (first_error_step - 1) / num_steps if first_error_step else 1.0
    task_success = first_error_step is None and len(step_results) == num_steps

    return {
        "episode_id": episode_id,
        "goal": goal,
        "num_steps": num_steps,
        "steps_evaluated": len(step_results),
        "correct_steps": correct_steps,
        "task_success": task_success,
        "progress": progress,
        "first_error_step": first_error_step,
        "steps": step_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--stop_on_error", action="store_true", default=True)
    parser.add_argument("--gt_history", action="store_true", default=False,
                        help="Use GT actions for history (teacher-forced). Evaluates all steps.")
    parser.add_argument("--start", type=int, default=0,
                        help="Start index for episode slicing (for parallel sharding)")
    parser.add_argument("--end", type=int, default=None,
                        help="End index for episode slicing (for parallel sharding)")
    args = parser.parse_args()

    # Load test data
    episodes = []
    with open(args.test_data) as f:
        for line in f:
            episodes.append(json.loads(line))
    total_loaded = len(episodes)
    episodes = episodes[args.start:args.end]
    print(f"Loaded {len(episodes)} episodes from {args.test_data} "
          f"(shard [{args.start}:{args.end}] of {total_loaded})")

    client = OpenAI(base_url=args.api_url, api_key="dummy")

    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluate
    results = {}
    total_success = 0
    total_progress = 0.0
    total_steps = 0
    total_correct = 0
    total_reward = 0.0

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(
                evaluate_episode, client, args.model_name, ep,
                args.stop_on_error, args.match_threshold, args.gt_history
            ): ep["episode_id"]
            for ep in episodes
        }

        pbar = tqdm(total=len(episodes), desc="Evaluating")
        for future in as_completed(futures):
            result = future.result()
            eid = result["episode_id"]
            results[eid] = result

            if result["task_success"]:
                total_success += 1
            total_progress += result["progress"]
            total_steps += result["steps_evaluated"]
            total_correct += result["correct_steps"]
            total_reward += sum(s["reward"] for s in result["steps"])

            n = len(results)
            pbar.update(1)
            pbar.set_postfix({
                "TSR": f"{total_success/n:.3f}",
                "Progress": f"{total_progress/n:.3f}",
                "StepSR": f"{total_correct/total_steps:.3f}" if total_steps > 0 else "0",
            })
        pbar.close()

    n = len(results)
    summary = {
        "num_episodes": n,
        "tsr": total_success / n if n > 0 else 0,
        "avg_progress": total_progress / n if n > 0 else 0,
        "step_sr": total_correct / total_steps if total_steps > 0 else 0,
        "mean_reward": total_reward / total_steps if total_steps > 0 else 0,
        "total_steps_evaluated": total_steps,
        "total_steps_correct": total_correct,
        "match_threshold": args.match_threshold,
        "stop_on_error": args.stop_on_error,
        "gt_history": args.gt_history,
        "prompt_format": "gui360_template",
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.output_dir, f"eval_summary_{ts}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, f"eval_results_{ts}.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Results: {args.model_name}")
    print(f"  TSR:      {summary['tsr']*100:.1f}%")
    print(f"  Progress: {summary['avg_progress']*100:.1f}%")
    print(f"  Step SR:  {summary['step_sr']*100:.1f}%")
    print(f"  Mean Rwd: {summary['mean_reward']:.4f}")
    print(f"  Episodes: {n}")
    if args.gt_history:
        print(f"  Mode:     GT History (teacher-forced)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
