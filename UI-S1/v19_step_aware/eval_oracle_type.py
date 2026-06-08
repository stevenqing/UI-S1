#!/usr/bin/env python3
"""Oracle action-type experiment: validate that action type planning is the bottleneck.

Gives the model a GT action type hint at each step (e.g., "Your next action
should be a CLICK") while the model must still predict coordinates/text on its own.

Experiments:
  oracle_type:       GT action type hint every step
  oracle_type_noisy: GT type with 80% probability, random type 20% (simulates
                     a realistic type predictor with ~80% accuracy)
  no_oracle:         No hint (baseline, same as type_focused prompt)

If oracle_type dramatically improves 6+ step TSR → action type planning is the
bottleneck → train a lightweight type predictor as planning agent.

Usage:
    python v19_step_aware/eval_oracle_type.py \
        --mode oracle_type \
        --test_data v12_gui_360/data/gui360_test_1000_balanced.jsonl \
        --api_url http://localhost:8000/v1 \
        --model_name base_sft \
        --output_dir v19_step_aware/outputs/oracle_type_pred \
        --threads 128
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v13_gui_360.eval_gui360_template import (
    parse_tool_call, _format_action_for_history, SUPPORTED_ACTIONS,
    _encode_screenshot,
)
from v13_gui_360.reward import compute_step_reward


# ═══════════════════════════════════════════════════════════════════════
# Action Type Mapping
# ═══════════════════════════════════════════════════════════════════════

# Map GT action types to natural language hints
ACTION_TYPE_HINTS = {
    "click": "Your next action should be a **CLICK**. Identify the correct UI element to click on.",
    "type": "Your next action should be to **TYPE** text. Find the appropriate input field and enter the correct text.",
    "swipe": "Your next action should be a **SCROLL/DRAG**. Determine the direction and distance to scroll or drag.",
    "drag": "Your next action should be a **SCROLL/DRAG**. Determine the direction and distance to scroll or drag.",
}

# All possible types for noisy oracle
ALL_TYPES = ["click", "type", "swipe"]


def _normalize_gt_type(action: Dict) -> str:
    """Normalize GT action type to one of: click, type, swipe."""
    atype = action.get("action", "").lower().strip()
    if atype in ("click", "double_click"):
        return "click"
    elif atype in ("type", "input", "text"):
        return "type"
    elif atype in ("swipe", "drag", "scroll", "wheel_mouse_input"):
        return "swipe"
    return atype


def _get_type_hint(gt_type: str, mode: str, rng: random.Random) -> Optional[str]:
    """Get action type hint based on oracle mode."""
    if mode == "no_oracle":
        return None
    elif mode == "oracle_type":
        return ACTION_TYPE_HINTS.get(gt_type, f"Your next action type is: {gt_type}.")
    elif mode == "oracle_type_noisy":
        # 80% correct, 20% random (simulates ~80% accurate type predictor)
        if rng.random() < 0.8:
            chosen_type = gt_type
        else:
            others = [t for t in ALL_TYPES if t != gt_type]
            chosen_type = rng.choice(others)
        return ACTION_TYPE_HINTS.get(chosen_type, f"Your next action type is: {chosen_type}.")
    return None


# ═══════════════════════════════════════════════════════════════════════
# Prompt Template with Oracle Hint
# ═══════════════════════════════════════════════════════════════════════

# Based on type_focused (the best-performing prompt variant)
ORACLE_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

{type_hint}

Based on this guidance:
1. Which UI element should receive this interaction?
2. Output the action with precise coordinates.

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

# No-oracle baseline (same as type_focused)
NO_ORACLE_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

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


def build_prompt(goal: str, history_text: str,
                 type_hint: Optional[str]) -> str:
    """Build prompt with or without oracle type hint."""
    if type_hint is None:
        return NO_ORACLE_PROMPT.format(
            instruction=goal,
            history=history_text,
            actions=SUPPORTED_ACTIONS,
        )
    else:
        return ORACLE_PROMPT.format(
            instruction=goal,
            history=history_text,
            actions=SUPPORTED_ACTIONS,
            type_hint=type_hint,
        )


# ═══════════════════════════════════════════════════════════════════════
# Episode Evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate_episode(
    client: OpenAI,
    model_name: str,
    episode: Dict,
    mode: str,
    gt_history: bool = False,
    match_threshold: float = 0.5,
    image_max_pixels: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate one episode with oracle type hints."""
    episode_id = episode["episode_id"]
    goal = episode["goal"]
    steps = episode["steps"]
    num_steps = len(steps)

    rng = random.Random(seed + hash(episode_id))
    history = []
    step_results = []
    first_error_step = None
    correct_steps = 0

    for i, step in enumerate(steps):
        gt_action = step["action"]
        screenshot = step["screenshot"]
        image_w = step.get("image_w", 1040)
        image_h = step.get("image_h", 736)

        gt_type = _normalize_gt_type(gt_action)
        history_text = "\n".join(history) if history else "None"
        b64 = _encode_screenshot(screenshot, image_max_pixels)

        # Get type hint once (important: single rng call for consistency)
        type_hint = _get_type_hint(gt_type, mode, rng)
        prompt_text = build_prompt(goal, history_text, type_hint)

        # Track what hint was given
        if type_hint is None:
            hinted_type = None
            hint_correct = None
        elif "CLICK" in type_hint:
            hinted_type = "click"
            hint_correct = (hinted_type == gt_type)
        elif "TYPE" in type_hint:
            hinted_type = "type"
            hint_correct = (hinted_type == gt_type)
        elif "SCROLL" in type_hint:
            hinted_type = "swipe"
            hint_correct = (hinted_type == gt_type)
        else:
            hinted_type = gt_type
            hint_correct = True

        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt_text},
        ]}]

        pred_text = ""
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=1024,
                temperature=0.0,
            )
            pred_text = response.choices[0].message.content or ""
        except Exception as e:
            print(f"  [ep {episode_id}] step {i+1} error: {e}")

        # Parse action
        pred_action = parse_tool_call(pred_text)
        if pred_action is None:
            m = re.search(r'<action>\s*(\{.*?\})\s*</action>', pred_text, re.DOTALL)
            if m:
                try:
                    pred_action = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass

        if pred_action:
            fake_text = f"<action>{json.dumps(pred_action)}</action>"
        else:
            fake_text = pred_text

        reward, info = compute_step_reward(fake_text, gt_action, image_w, image_h)
        success = reward >= match_threshold

        step_results.append({
            "step_idx": i,
            "success": success,
            "reward": reward,
            "gt_type": gt_type,
            "pred_type": info.get("pred_type"),
            "hinted_type": hinted_type,
            "hint_correct": hint_correct,
            "type_reward": info.get("type_reward", 0),
            "content_reward": info.get("content_reward", 0),
            "pred_text": pred_text[:500],
        })

        if gt_history:
            history.append(_format_action_for_history(gt_action, i + 1))
        else:
            history.append(_format_action_for_history(pred_action, i + 1))

        if success:
            correct_steps += 1
        else:
            if first_error_step is None:
                first_error_step = i + 1

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


# ═══════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════

def compute_breakdown(results: Dict[str, Dict]) -> Dict[str, Any]:
    """Compute per-task-length and per-step-position breakdowns, plus
    action-type transition analysis."""

    length_buckets = {
        "1": (1, 1),
        "2-3": (2, 3),
        "4-5": (4, 5),
        "6+": (6, 999),
    }

    # Per-task-length
    length_success = defaultdict(int)
    length_total = defaultdict(int)
    length_progress = defaultdict(float)
    length_steps_correct = defaultdict(int)
    length_steps_total = defaultdict(int)

    # Per-step-position
    step_pos_correct = defaultdict(int)
    step_pos_total = defaultdict(int)
    step_pos_type_correct = defaultdict(int)

    # Action type transition analysis
    transition_correct = defaultdict(int)
    transition_total = defaultdict(int)

    # Type hint effectiveness
    hint_correct_and_step_correct = 0
    hint_correct_and_step_wrong = 0
    hint_wrong_and_step_correct = 0
    hint_wrong_and_step_wrong = 0

    for eid, result in results.items():
        num_steps = result["num_steps"]
        bucket = None
        for bname, (lo, hi) in length_buckets.items():
            if lo <= num_steps <= hi:
                bucket = bname
                break
        if bucket is None:
            bucket = "6+"

        length_total[bucket] += 1
        if result["task_success"]:
            length_success[bucket] += 1
        length_progress[bucket] += result["progress"]

        prev_type = None
        for step in result["steps"]:
            idx = step["step_idx"]
            step_pos_total[idx] += 1
            if step["success"]:
                step_pos_correct[idx] += 1
            if step.get("type_reward", 0) >= 1.0:
                step_pos_type_correct[idx] += 1

            length_steps_total[bucket] += 1
            if step["success"]:
                length_steps_correct[bucket] += 1

            # Transition analysis
            gt_type = step.get("gt_type")
            if prev_type and gt_type:
                trans_key = f"{prev_type}->{gt_type}"
                transition_total[trans_key] += 1
                if step["success"]:
                    transition_correct[trans_key] += 1
            prev_type = gt_type

            # Hint effectiveness (for noisy oracle)
            hc = step.get("hint_correct")
            if hc is not None:
                if hc and step["success"]:
                    hint_correct_and_step_correct += 1
                elif hc and not step["success"]:
                    hint_correct_and_step_wrong += 1
                elif not hc and step["success"]:
                    hint_wrong_and_step_correct += 1
                else:
                    hint_wrong_and_step_wrong += 1

    # Format per-step-position
    max_step = max(step_pos_total.keys()) if step_pos_total else 0
    step_position_acc = {}
    for idx in range(min(max_step + 1, 15)):
        total = step_pos_total.get(idx, 0)
        correct = step_pos_correct.get(idx, 0)
        type_correct = step_pos_type_correct.get(idx, 0)
        if total > 0:
            step_position_acc[f"step_{idx}"] = {
                "accuracy": correct / total,
                "type_accuracy": type_correct / total,
                "total": total,
                "correct": correct,
            }

    # Format per-task-length
    task_length_metrics = {}
    for bname in length_buckets:
        total = length_total.get(bname, 0)
        if total > 0:
            s_total = length_steps_total.get(bname, 0)
            task_length_metrics[bname] = {
                "tsr": length_success[bname] / total,
                "avg_progress": length_progress[bname] / total,
                "step_sr": length_steps_correct[bname] / s_total if s_total > 0 else 0,
                "num_episodes": total,
                "num_success": length_success[bname],
            }

    # Format transition analysis
    transition_metrics = {}
    for trans, total in sorted(transition_total.items(), key=lambda x: -x[1]):
        transition_metrics[trans] = {
            "accuracy": transition_correct[trans] / total,
            "total": total,
            "correct": transition_correct[trans],
        }

    # Hint effectiveness
    hint_metrics = {}
    total_hint = (hint_correct_and_step_correct + hint_correct_and_step_wrong +
                  hint_wrong_and_step_correct + hint_wrong_and_step_wrong)
    if total_hint > 0:
        hint_metrics = {
            "correct_hint_step_correct": hint_correct_and_step_correct,
            "correct_hint_step_wrong": hint_correct_and_step_wrong,
            "wrong_hint_step_correct": hint_wrong_and_step_correct,
            "wrong_hint_step_wrong": hint_wrong_and_step_wrong,
            "correct_hint_accuracy": (hint_correct_and_step_correct /
                                       (hint_correct_and_step_correct + hint_correct_and_step_wrong))
                                      if (hint_correct_and_step_correct + hint_correct_and_step_wrong) > 0 else 0,
            "wrong_hint_accuracy": (hint_wrong_and_step_correct /
                                     (hint_wrong_and_step_correct + hint_wrong_and_step_wrong))
                                    if (hint_wrong_and_step_correct + hint_wrong_and_step_wrong) > 0 else 0,
        }

    return {
        "step_position_accuracy": step_position_acc,
        "task_length_metrics": task_length_metrics,
        "transition_metrics": transition_metrics,
        "hint_effectiveness": hint_metrics,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Oracle action-type experiment")
    parser.add_argument("--mode", required=True,
                        choices=["oracle_type", "oracle_type_noisy", "no_oracle"],
                        help="Oracle mode")
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--gt_history", action="store_true", default=False)
    parser.add_argument("--image_max_pixels", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    episodes = []
    with open(args.test_data) as f:
        for line in f:
            episodes.append(json.loads(line))
    total_loaded = len(episodes)
    episodes = episodes[args.start:args.end]
    print(f"Loaded {len(episodes)} episodes (shard [{args.start}:{args.end}] of {total_loaded})")
    print(f"Oracle mode: {args.mode}")
    print(f"GT history: {args.gt_history}")

    client = OpenAI(base_url=args.api_url, api_key="dummy")
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    total_success = 0
    total_progress = 0.0
    total_steps = 0
    total_correct = 0

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(
                evaluate_episode, client, args.model_name, ep,
                args.mode, args.gt_history, args.match_threshold,
                args.image_max_pixels, args.seed,
            ): ep["episode_id"]
            for ep in episodes
        }

        pbar = tqdm(total=len(episodes), desc=f"Oracle {args.mode}")
        for future in as_completed(futures):
            result = future.result()
            eid = result["episode_id"]
            results[eid] = result

            if result["task_success"]:
                total_success += 1
            total_progress += result["progress"]
            total_steps += result["steps_evaluated"]
            total_correct += result["correct_steps"]

            n = len(results)
            pbar.update(1)
            pbar.set_postfix({
                "TSR": f"{total_success/n:.3f}",
                "StepSR": f"{total_correct/total_steps:.3f}" if total_steps > 0 else "0",
            })
        pbar.close()

    # Compute breakdowns
    breakdown = compute_breakdown(results)

    n = len(results)
    summary = {
        "num_episodes": n,
        "tsr": total_success / n if n > 0 else 0,
        "avg_progress": total_progress / n if n > 0 else 0,
        "step_sr": total_correct / total_steps if total_steps > 0 else 0,
        "total_steps_evaluated": total_steps,
        "total_steps_correct": total_correct,
        "mode": args.mode,
        "gt_history": args.gt_history,
        "match_threshold": args.match_threshold,
        **breakdown,
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.output_dir, f"eval_summary_{ts}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, f"eval_results_{ts}.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Print results
    hist_mode = "GT history" if args.gt_history else "Pred history"
    print(f"\n{'='*65}")
    print(f"Oracle Type Experiment: {args.mode} ({hist_mode})")
    print(f"{'='*65}")
    print(f"  TSR:      {summary['tsr']*100:.1f}%")
    print(f"  Step SR:  {summary['step_sr']*100:.1f}%")
    print(f"  Progress: {summary['avg_progress']*100:.1f}%")
    print(f"  (Baselines: standard=21.9%, type_focused=23.6%)")

    print(f"\n  --- By Task Length ---")
    for bname in ["1", "2-3", "4-5", "6+"]:
        metrics = summary.get("task_length_metrics", {}).get(bname)
        if metrics:
            print(f"  {bname:>5s} steps: TSR={metrics['tsr']*100:5.1f}%  "
                  f"StepSR={metrics['step_sr']*100:5.1f}%  "
                  f"Progress={metrics['avg_progress']*100:5.1f}%  "
                  f"(n={metrics['num_episodes']})")

    print(f"\n  --- By Step Position ---")
    for sname, metrics in sorted(summary.get("step_position_accuracy", {}).items(),
                                  key=lambda x: int(x[0].split("_")[1])):
        print(f"  {sname:>8s}: acc={metrics['accuracy']*100:5.1f}%  "
              f"type_acc={metrics['type_accuracy']*100:5.1f}%  "
              f"(n={metrics['total']})")

    print(f"\n  --- Action Type Transitions ---")
    for trans, metrics in sorted(summary.get("transition_metrics", {}).items(),
                                  key=lambda x: -x[1]["total"]):
        print(f"  {trans:>15s}: acc={metrics['accuracy']*100:5.1f}%  "
              f"(n={metrics['total']}, correct={metrics['correct']})")

    hint_eff = summary.get("hint_effectiveness", {})
    if hint_eff:
        print(f"\n  --- Hint Effectiveness ---")
        print(f"  Correct hint → step correct: {hint_eff['correct_hint_step_correct']}  "
              f"(acc={hint_eff['correct_hint_accuracy']*100:.1f}%)")
        print(f"  Correct hint → step wrong:   {hint_eff['correct_hint_step_wrong']}")
        print(f"  Wrong hint → step correct:   {hint_eff['wrong_hint_step_correct']}  "
              f"(acc={hint_eff['wrong_hint_accuracy']*100:.1f}%)")
        print(f"  Wrong hint → step wrong:     {hint_eff['wrong_hint_step_wrong']}")

    print(f"{'='*65}")


if __name__ == "__main__":
    main()
