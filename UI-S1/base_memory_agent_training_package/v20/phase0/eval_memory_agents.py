#!/usr/bin/env python3
"""Evaluate memory-specialized proposal agents with GT history.

For each agent checkpoint, evaluates standalone StepSR/TSR using GT history
on the balanced 1000-episode test set. Also saves per-step predictions
for downstream oracle reranking.

Usage:
    python v20/phase0/eval_memory_agents.py \
        --agent_type type_recovery \
        --test_data v12_gui_360/data/gui360_test_1000_balanced.jsonl \
        --api_url http://localhost:8000/v1 \
        --output_dir v20/phase0/eval_results/type_recovery \
        --threads 128
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v13_gui_360.eval_gui360_template import (
    parse_tool_call, _format_action_for_history, SUPPORTED_ACTIONS,
    _encode_screenshot,
)
from v13_gui_360.reward import compute_step_reward


# ═══════════════════════════════════════════════════════════════════════
# Agent Specialization Descriptions
# ═══════════════════════════════════════════════════════════════════════

AGENT_DESCRIPTIONS = {
    "type_recovery": "type_recovery (focus on text-entry, field focus, keyboard input, and type-action alternatives)",
    "click_recovery": "click_recovery (focus on alternative grounded click targets and nearby-target recovery)",
    "swipe_navigation": "swipe_navigation (focus on scroll, drag, navigation, and actions that reveal hidden targets)",
    "minimal_next_step": "minimal_next_step (focus on the simplest valid next action that advances the task)",
    "escape_finish": "escape_finish (focus on modal escape, finish guards, and avoiding premature FINISH)",
    "spreadsheet_formula": "spreadsheet_formula (focus on spreadsheet cells, formulas, table editing, and office-document structure)",
}


# ═══════════════════════════════════════════════════════════════════════
# Memory Agent Prompt Template
# ═══════════════════════════════════════════════════════════════════════

MEMORY_AGENT_PROMPT = """You are a trainable memory-specialized proposal agent for GUI control.
Your goal is to generate one candidate action for a multi-agent candidate pool.
Do not imitate a generic base policy when a useful specialized alternative exists.
Agent specialization: {agent_desc}.
Task: {instruction}
Step: {step_num} / {total_steps}
History:
{history}
Supported actions:
<action>
{actions}
</action>
Return exactly one action in <tool_call></tool_call> format."""


def format_gt_history_brief(gt_actions: List[Dict], up_to: int) -> str:
    """Format GT history in the brief style used during training."""
    if up_to == 0:
        return "None"
    lines = []
    for i in range(up_to):
        act = gt_actions[i]
        # Support both formats: {"function": ..., "args": ...} and {"action": ..., "coordinate": ...}
        func = act.get("function") or act.get("action", "unknown")
        args = act.get("args", act)  # fallback to act itself for flat format
        # Brief format matching training data
        if func == "click":
            coord = act.get("coordinate", args.get("coordinate", [0, 0]))
            lines.append(f"Step {i+1}: click(coordinate={coord})")
        elif func == "type":
            coord = act.get("coordinate", args.get("coordinate", [0, 0]))
            keys = act.get("keys", args.get("keys", ""))
            if len(str(keys)) > 30:
                keys = str(keys)[:27] + "..."
            lines.append(f"Step {i+1}: type(coordinate={coord}, keys='{keys}')")
        elif func == "drag":
            start = act.get("start_coordinate", args.get("start_coordinate", [0, 0]))
            end = act.get("end_coordinate", args.get("end_coordinate", [0, 0]))
            lines.append(f"Step {i+1}: drag(start_coordinate={start}, end_coordinate={end})")
        elif func in ("wheel_mouse_input", "scroll"):
            coord = act.get("coordinate", args.get("coordinate", [0, 0]))
            dist = act.get("wheel_dist", args.get("wheel_dist", 0))
            lines.append(f"Step {i+1}: wheel_mouse_input(coordinate={coord}, wheel_dist={dist})")
        else:
            lines.append(f"Step {i+1}: {func}()")
    return "\n".join(lines)


def evaluate_episode(
    client: OpenAI,
    model_name: str,
    episode: Dict,
    agent_type: str,
    match_threshold: float = 0.5,
    image_max_pixels: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate one episode with GT history."""
    episode_id = episode["episode_id"]
    goal = episode["goal"]
    steps = episode["steps"]
    num_steps = len(steps)
    agent_desc = AGENT_DESCRIPTIONS[agent_type]

    # Parse GT actions for history
    gt_actions = []
    for step in steps:
        act = step["action"]
        if isinstance(act, dict):
            gt_actions.append(act)
        else:
            gt_actions.append(parse_tool_call(act))

    step_results = []
    correct_steps = 0
    first_error_step = None

    for i, step in enumerate(steps):
        gt_action = step["action"]
        screenshot = step["screenshot"]
        image_w = step.get("image_w", 1040)
        image_h = step.get("image_h", 736)

        # Build GT history up to current step
        history_text = format_gt_history_brief(gt_actions, i)
        b64 = _encode_screenshot(screenshot, image_max_pixels)

        prompt_text = MEMORY_AGENT_PROMPT.format(
            agent_desc=agent_desc,
            instruction=goal,
            step_num=i + 1,
            total_steps=num_steps,
            history=history_text,
            actions=SUPPORTED_ACTIONS,
        )

        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt_text},
        ]}]

        pred_text = ""
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=512,
                temperature=0.0,
            )
            pred_text = response.choices[0].message.content or ""
        except Exception as e:
            print(f"  [ep {episode_id}] step {i+1} error: {e}")

        pred_action = parse_tool_call(pred_text)
        # compute_step_reward expects (pred_text: str, gt_action: dict)
        gt_parsed = gt_action if isinstance(gt_action, dict) else parse_tool_call(gt_action)
        reward, _ = compute_step_reward(pred_text, gt_parsed, image_w, image_h)
        is_correct = reward >= match_threshold

        if is_correct:
            correct_steps += 1
        elif first_error_step is None:
            first_error_step = i + 1

        step_results.append({
            "step": i + 1,
            "gt_action": gt_action,
            "pred_text": pred_text,
            "pred_action": pred_action,
            "reward": reward,
            "correct": is_correct,
        })

    all_correct = correct_steps == num_steps
    return {
        "episode_id": episode_id,
        "goal": goal,
        "num_steps": num_steps,
        "correct_steps": correct_steps,
        "step_sr": correct_steps / num_steps if num_steps > 0 else 0,
        "task_success": all_correct,
        "first_error_step": first_error_step,
        "steps": step_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_type", required=True, choices=list(AGENT_DESCRIPTIONS.keys()))
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--model_name", default="memory_agent")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--image_max_pixels", type=int, default=None)
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data
    episodes = []
    with open(args.test_data) as f:
        for line in f:
            episodes.append(json.loads(line))
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
    print(f"Loaded {len(episodes)} episodes")

    from httpx import Timeout
    client = OpenAI(
        base_url=args.api_url, api_key="empty",
        timeout=Timeout(120.0, connect=30.0),  # 120s read timeout, 30s connect
    )

    # Evaluate all episodes in parallel
    results = []
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = {
            pool.submit(
                evaluate_episode, client, args.model_name, ep,
                args.agent_type, args.match_threshold, args.image_max_pixels,
            ): ep["episode_id"]
            for ep in episodes
        }
        pbar = tqdm(total=len(futures), desc=f"Eval {args.agent_type}")
        for future in as_completed(futures):
            try:
                result = future.result(timeout=600)  # 10 min max per episode
            except Exception as e:
                eid = futures[future]
                print(f"  [ep {eid}] TIMEOUT/ERROR: {e}")
                result = {
                    "episode_id": eid, "goal": "", "num_steps": 0,
                    "correct_steps": 0, "step_sr": 0, "task_success": False,
                    "first_error_step": 1, "steps": [],
                }
            results.append(result)
            pbar.update(1)
        pbar.close()

    # Sort by episode_id
    results.sort(key=lambda r: r["episode_id"])

    # Compute aggregate metrics
    total_episodes = len(results)
    successful = sum(1 for r in results if r["task_success"])
    total_steps = sum(r["num_steps"] for r in results)
    correct_steps = sum(r["correct_steps"] for r in results)

    tsr = successful / total_episodes * 100 if total_episodes > 0 else 0
    step_sr = correct_steps / total_steps * 100 if total_steps > 0 else 0

    summary = {
        "agent_type": args.agent_type,
        "total_episodes": total_episodes,
        "successful_episodes": successful,
        "TSR": round(tsr, 2),
        "total_steps": total_steps,
        "correct_steps": correct_steps,
        "StepSR": round(step_sr, 2),
    }

    print(f"\n{'='*50}")
    print(f"Agent: {args.agent_type}")
    print(f"TSR: {tsr:.1f}% ({successful}/{total_episodes})")
    print(f"StepSR: {step_sr:.1f}% ({correct_steps}/{total_steps})")
    print(f"{'='*50}")

    # Save results
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, "episode_results.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
