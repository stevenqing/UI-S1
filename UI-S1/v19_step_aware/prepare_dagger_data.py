#!/usr/bin/env python3
"""DAgger data preparation: rollout base model on training data with pred history.

Standard SFT trains on (GT_history, GT_action) pairs, but at test time the model
sees its own imperfect predictions in the history. This distribution mismatch
causes error accumulation on long tasks:
  - 1-step tasks: TSR 72.3%  (no history needed)
  - 6+ step tasks: TSR 6.6%  (compounding errors)

DAgger (Dataset Aggregation) fixes this by training on (pred_history, GT_action):
  1. Roll out the model on training episodes using its own predictions for history
  2. At each step, save: screenshot + pred_history (model's actual state)
  3. Target: GT action (expert supervision)
  4. SFT on this data teaches the model to recover from its own mistakes

Output format (JSONL, one line per step):
{
  "episode_id": 42,
  "step_idx": 0,
  "goal": "...",
  "screenshot": "/path/to/img.png",
  "pred_history": "None",                    # Model's predicted history
  "gt_history": "None",                      # GT history (for comparison)
  "gt_action": {"action": "click", ...},     # Expert action (training target)
  "gt_response": "<tool_call>...</tool_call>",# GT response text
  "model_response": "...",                   # Model's actual prediction
  "model_action": {...},                     # Parsed model action
  "step_reward": 0.85,                       # How close model was
  "step_correct": true,                      # reward >= threshold
  "image_w": 1040, "image_h": 736,
  "num_steps": 5,                            # Total steps in episode
  "history_diverged": false,                 # Whether pred != GT history at this step
}

Usage:
    python v19_step_aware/prepare_dagger_data.py \
        --train_data v12_gui_360/data/gui360_train_2000_balanced.jsonl \
        --api_url http://localhost:8000/v1 --model_name base_sft \
        --output v19_step_aware/data/dagger_rollouts.jsonl \
        --threads 48
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v13_gui_360.eval_gui360_template import (
    parse_tool_call, _format_action_for_history, SUPPORTED_ACTIONS,
    _encode_screenshot, USER_PROMPT_TEMPLATE,
)
from v13_gui_360.reward import compute_step_reward


# Use the type_focused prompt (best performing variant from Phase 1)
PREDICT_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

Before acting, determine:
1. What type of interaction is needed next? (click to select/navigate, type to enter text, drag to move, scroll to view more)
2. Which UI element should receive this interaction?
3. Output the action with precise coordinates.

After your reasoning, output your action within <tool_call></tool_call> tag:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time."""


def _build_gt_response(gt_action: Dict) -> str:
    """Build a tool_call response string from GT action dict."""
    atype = gt_action.get("action", "click")
    args = {}

    if atype in ("click", "long_press"):
        coord = gt_action.get("coordinate")
        if coord:
            args["coordinate"] = [int(float(coord[0])), int(float(coord[1]))]
        func = "click"
    elif atype in ("type", "input"):
        text = gt_action.get("text", "")
        args["keys"] = text
        coord = gt_action.get("coordinate")
        if coord:
            args["coordinate"] = [int(float(coord[0])), int(float(coord[1]))]
        func = "type"
    elif atype in ("swipe", "drag"):
        start = gt_action.get("coordinate") or gt_action.get("startCoordinate")
        end = gt_action.get("endCoordinate")
        if start:
            args["start_coordinate"] = [int(float(start[0])), int(float(start[1]))]
        if end:
            args["end_coordinate"] = [int(float(end[0])), int(float(end[1]))]
        func = "drag"
    elif atype == "scroll":
        coord = gt_action.get("coordinate")
        if coord:
            args["coordinate"] = [int(float(coord[0])), int(float(coord[1]))]
        args["wheel_dist"] = gt_action.get("wheel_dist", -3)
        func = "wheel_mouse_input"
    else:
        func = atype
        for k, v in gt_action.items():
            if k != "action":
                args[k] = v

    tc = {"function": func, "args": args, "status": "CONTINUE"}
    return f"<tool_call>\n{json.dumps(tc, indent=2)}\n</tool_call>"


def rollout_episode(
    client: OpenAI,
    model_name: str,
    episode: Dict,
    match_threshold: float = 0.5,
    image_max_pixels: Optional[int] = None,
) -> List[Dict]:
    """Roll out model on one episode with predicted history, returning DAgger samples."""
    episode_id = episode["episode_id"]
    goal = episode["goal"]
    steps = episode["steps"]
    num_steps = len(steps)

    pred_history = []  # Model's predicted actions
    gt_history = []    # Ground truth actions
    results = []
    history_diverged = False

    for i, step in enumerate(steps):
        gt_action = step["action"]
        screenshot = step["screenshot"]
        image_w = step.get("image_w", 1040)
        image_h = step.get("image_h", 736)

        pred_history_text = "\n".join(pred_history) if pred_history else "None"
        gt_history_text = "\n".join(gt_history) if gt_history else "None"

        b64 = _encode_screenshot(screenshot, image_max_pixels)

        prompt_text = PREDICT_PROMPT.format(
            instruction=goal,
            history=pred_history_text,
            actions=SUPPORTED_ACTIONS,
        )

        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt_text},
        ]}]

        model_response = ""
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=1024,
                temperature=0.0,
            )
            model_response = response.choices[0].message.content or ""
        except Exception as e:
            print(f"  [ep {episode_id}] step {i+1} error: {e}")

        # Parse model's action
        model_action = parse_tool_call(model_response)
        if model_action is None:
            m = re.search(r'<action>\s*(\{.*?\})\s*</action>', model_response, re.DOTALL)
            if m:
                try:
                    model_action = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass

        # Compute reward for model's prediction
        if model_action:
            fake_text = f"<action>{json.dumps(model_action)}</action>"
        else:
            fake_text = model_response
        reward, info = compute_step_reward(fake_text, gt_action, image_w, image_h)
        step_correct = reward >= match_threshold

        # Check if history has diverged
        if not step_correct and not history_diverged:
            history_diverged = True

        # Build GT response text
        gt_response = _build_gt_response(gt_action)

        results.append({
            "episode_id": episode_id,
            "step_idx": i,
            "goal": goal,
            "screenshot": screenshot,
            "pred_history": pred_history_text,
            "gt_history": gt_history_text,
            "gt_action": gt_action,
            "gt_response": gt_response,
            "model_response": model_response[:1000],
            "model_action": model_action,
            "step_reward": reward,
            "step_correct": step_correct,
            "image_w": image_w,
            "image_h": image_h,
            "num_steps": num_steps,
            "history_diverged": history_diverged,
        })

        # Update histories
        pred_history.append(_format_action_for_history(model_action, i + 1))
        gt_history.append(_format_action_for_history(gt_action, i + 1))

    return results


def main():
    parser = argparse.ArgumentParser(description="DAgger rollout data preparation")
    parser.add_argument("--train_data", required=True,
                        help="Training episodes JSONL")
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output", required=True,
                        help="Output JSONL path")
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--image_max_pixels", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    # Load episodes
    episodes = []
    with open(args.train_data) as f:
        for line in f:
            episodes.append(json.loads(line))
    total = len(episodes)
    episodes = episodes[args.start:args.end]
    print(f"Loaded {len(episodes)} episodes (shard [{args.start}:{args.end}] of {total})")

    client = OpenAI(base_url=args.api_url, api_key="dummy")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    all_results = []
    stats = {
        "total_steps": 0, "correct_steps": 0, "diverged_steps": 0,
        "total_episodes": 0, "successful_episodes": 0,
    }

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(
                rollout_episode, client, args.model_name, ep,
                args.match_threshold, args.image_max_pixels,
            ): ep["episode_id"]
            for ep in episodes
        }

        pbar = tqdm(total=len(episodes), desc="DAgger rollout")
        for future in as_completed(futures):
            try:
                results = future.result()
            except Exception as e:
                print(f"Episode error: {e}")
                pbar.update(1)
                continue

            all_results.extend(results)
            stats["total_episodes"] += 1

            ep_correct = all(r["step_correct"] for r in results)
            if ep_correct:
                stats["successful_episodes"] += 1

            for r in results:
                stats["total_steps"] += 1
                if r["step_correct"]:
                    stats["correct_steps"] += 1
                if r["history_diverged"]:
                    stats["diverged_steps"] += 1

            pbar.update(1)
            n = stats["total_episodes"]
            pbar.set_postfix({
                "TSR": f"{stats['successful_episodes']/n:.3f}",
                "StepSR": f"{stats['correct_steps']/stats['total_steps']:.3f}" if stats['total_steps'] > 0 else "0",
            })
        pbar.close()

    # Sort by episode_id, step_idx for determinism
    all_results.sort(key=lambda r: (r["episode_id"], r["step_idx"]))

    # Write output
    with open(args.output, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    # Stats
    print(f"\n{'='*60}")
    print(f"DAgger Rollout Summary")
    print(f"{'='*60}")
    print(f"  Episodes: {stats['total_episodes']}")
    print(f"  TSR: {stats['successful_episodes']/max(stats['total_episodes'],1)*100:.1f}%")
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Step SR: {stats['correct_steps']/max(stats['total_steps'],1)*100:.1f}%")
    print(f"  Diverged steps: {stats['diverged_steps']} "
          f"({stats['diverged_steps']/max(stats['total_steps'],1)*100:.1f}%)")
    print(f"  Output: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
