#!/usr/bin/env python3
"""Step-aware prompt variant evaluation for long-horizon GUI navigation.

Evaluates prompt variants designed to fix CoT's "jumping ahead" problem
on long tasks while preserving its grounding/type benefits on short tasks.

Variants:
  - standard:     Baseline GUI-360 template (USER_PROMPT_TEMPLATE)
  - cot:          Existing CoT prompt (from eval_cot_step.py)
  - focused_cot:  History-anchored reasoning with anti-jump instruction
  - adaptive_cot: CoT for steps 0-1, standard for steps 2+
  - step_context: Standard + step-position metadata ("Progress: N actions done")
  - type_focused: Forces explicit action-type reasoning before coordinates
  - subtask:      Oracle GT thought as step instruction + compressed history

Usage:
    python v19_step_aware/eval_step_aware.py \
        --prompt_variant focused_cot \
        --test_data v12_gui_360/data/gui360_test_1000_balanced.jsonl \
        --api_url http://localhost:8000/v1 \
        --model_name base_sft \
        --output_dir v19_step_aware/outputs/eval_focused_cot_pred \
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v13_gui_360.eval_gui360_template import (
    parse_tool_call, _format_action_for_history, SUPPORTED_ACTIONS,
    _encode_screenshot, USER_PROMPT_TEMPLATE,
)
from v13_gui_360.reward import compute_step_reward
from v15_gui_360.eval_cot_step import COT_PROMPT


# ═══════════════════════════════════════════════════════════════════════
# Prompt Variants
# ═══════════════════════════════════════════════════════════════════════

# Variant A: Focused CoT (anti-jump)
# Replaces screen-global reasoning with history-anchored reasoning.
# Explicit "Do NOT skip ahead" instruction to prevent jumping.
FOCUSED_COT_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

Think about the IMMEDIATE next action:
1. Based on the history, what has been done so far toward completing the instruction?
2. What is the SINGLE next step needed? Do NOT skip ahead — only the very next action.
3. Which exact UI element on the current screen should you interact with?
4. Output the action.

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

# Variant C: Step-Position Context
# Minimal change: adds progress counter and "RIGHT NOW" emphasis.
STEP_CONTEXT_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

Progress: You have completed {n_done} action(s) so far.
The history of actions are:
{history}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

Based on the current screen and the actions already taken, predict the next action.
Focus on what needs to happen RIGHT NOW to make progress on the instruction.

Then output your action within <tool_call></tool_call> tag:
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

# Variant D: Type-Focused CoT
# Forces explicit action-type reasoning before coordinates.
# Targets the #1 error source (49.2% wrong action type).
TYPE_FOCUSED_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

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

# Variant E: Subtask (Oracle)
# Uses GT thought as oracle step instruction + compressed history.
# Mirrors GUI-360-eval's action_prediction_ar_subtask mode.
# Requires test data with 'thought' field in each step.
SUBTASK_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

Overall Task:
{instruction}

Current Step Instruction:
{step_instruction}

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

Only **ONE** action should be taken at a time."""

# Registry of all prompt variants
PROMPT_VARIANTS = {
    "standard": USER_PROMPT_TEMPLATE,
    "cot": COT_PROMPT,
    "focused_cot": FOCUSED_COT_PROMPT,
    "adaptive_cot": None,  # Special handling in get_prompt()
    "step_context": STEP_CONTEXT_PROMPT,
    "type_focused": TYPE_FOCUSED_PROMPT,
    "subtask": SUBTASK_PROMPT,
    "standard_compressed": USER_PROMPT_TEMPLATE,  # Standard prompt + compressed history
    "subtask_verbose": SUBTASK_PROMPT,  # Subtask prompt (oracle thought) + verbose history
}


def _format_action_brief(pred_action: Optional[Dict]) -> str:
    """Create a brief text summary of a predicted action for compressed history."""
    if pred_action is None:
        return "unknown action"
    func = pred_action.get("action", pred_action.get("function", ""))
    if func == "click":
        coord = pred_action.get("coordinate", [0, 0])
        return f"click({coord})"
    elif func == "type":
        text = pred_action.get("text", pred_action.get("keys", ""))
        if len(text) > 30:
            text = text[:30] + "..."
        return f"type('{text}')"
    elif func in ("swipe", "drag"):
        coord = pred_action.get("coordinate", [0, 0])
        end = pred_action.get("endCoordinate", coord)
        return f"drag({coord} -> {end})"
    elif func == "scroll":
        coord = pred_action.get("coordinate", [0, 0])
        return f"scroll({coord})"
    else:
        return f"{func}({pred_action})"


def get_prompt(variant: str, step_idx: int, goal: str, history_text: str,
               step_thought: str = "") -> str:
    """Build prompt text for a given variant and step position.

    Args:
        variant: One of PROMPT_VARIANTS keys.
        step_idx: 0-indexed step position.
        goal: Task instruction text.
        history_text: Formatted history string.
        step_thought: GT thought text (used by subtask variant as oracle step instruction).

    Returns:
        Formatted prompt string.
    """
    if variant == "adaptive_cot":
        if step_idx < 2:
            template = COT_PROMPT
        else:
            template = USER_PROMPT_TEMPLATE
        return template.format(
            instruction=goal,
            history=history_text,
            actions=SUPPORTED_ACTIONS,
        )
    elif variant == "step_context":
        return STEP_CONTEXT_PROMPT.format(
            instruction=goal,
            n_done=step_idx,
            history=history_text,
            actions=SUPPORTED_ACTIONS,
        )
    elif variant in ("subtask", "subtask_verbose"):
        return SUBTASK_PROMPT.format(
            instruction=goal,
            step_instruction=step_thought if step_thought else goal,
            history=history_text,
            actions=SUPPORTED_ACTIONS,
        )
    elif variant == "standard_compressed":
        return USER_PROMPT_TEMPLATE.format(
            instruction=goal,
            history=history_text,
            actions=SUPPORTED_ACTIONS,
        )
    else:
        template = PROMPT_VARIANTS[variant]
        return template.format(
            instruction=goal,
            history=history_text,
            actions=SUPPORTED_ACTIONS,
        )


# ═══════════════════════════════════════════════════════════════════════
# Episode Evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate_episode(
    client: OpenAI,
    model_name: str,
    episode: Dict,
    variant: str,
    gt_history: bool = True,
    match_threshold: float = 0.5,
    image_max_pixels: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate one episode with a given prompt variant."""
    episode_id = episode["episode_id"]
    goal = episode["goal"]
    steps = episode["steps"]
    num_steps = len(steps)

    history = []
    compressed_history = []  # Brief action descriptions for subtask mode
    step_results = []
    first_error_step = None
    correct_steps = 0

    for i, step in enumerate(steps):
        gt_action = step["action"]
        screenshot = step["screenshot"]
        image_w = step.get("image_w", 1040)
        image_h = step.get("image_h", 736)
        step_thought = step.get("thought", "")

        if variant in ("subtask", "standard_compressed"):
            # Use compressed history (brief action descriptions)
            history_text = "\n".join(compressed_history) if compressed_history else "None"
        else:
            # Use verbose history (full formatted actions)
            history_text = "\n".join(history) if history else "None"
        b64 = _encode_screenshot(screenshot, image_max_pixels)

        prompt_text = get_prompt(variant, i, goal, history_text, step_thought)

        # Determine which prompt was actually used for this step
        if variant == "adaptive_cot":
            step_prompt_used = "cot" if i < 2 else "standard"
        else:
            step_prompt_used = variant

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

        # Extract reasoning (text before tool_call)
        reasoning = ""
        tc_match = re.search(r'<tool_call>', pred_text)
        if tc_match:
            reasoning = pred_text[:tc_match.start()].strip()

        step_results.append({
            "step_idx": i,
            "success": success,
            "reward": reward,
            "prompt_used": step_prompt_used,
            "reasoning": reasoning[:300],
            "pred_text": pred_text[:500],
            "pred_action": info.get("pred_action"),
            "pred_type": info.get("pred_type"),
            "gt_type": info.get("gt_type"),
            "type_reward": info.get("type_reward", 0),
            "content_reward": info.get("content_reward", 0),
        })

        if gt_history:
            history.append(_format_action_for_history(gt_action, i + 1))
        else:
            history.append(_format_action_for_history(pred_action, i + 1))

        # Also track compressed history for subtask mode
        action_brief = _format_action_brief(pred_action)
        compressed_history.append(f"Step {i + 1}: {action_brief}")

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
# Analysis: per-step-position and per-task-length breakdowns
# ═══════════════════════════════════════════════════════════════════════

def compute_breakdown(results: Dict[str, Dict]) -> Dict[str, Any]:
    """Compute per-step-position and per-task-length metrics."""

    # Per-step-position accuracy (step 0, 1, 2, ...)
    step_pos_correct = defaultdict(int)
    step_pos_total = defaultdict(int)
    step_pos_type_correct = defaultdict(int)

    # Per-task-length TSR
    length_buckets = {
        "1": (1, 1),
        "2-3": (2, 3),
        "4-5": (4, 5),
        "6+": (6, 999),
    }
    length_success = defaultdict(int)
    length_total = defaultdict(int)
    length_progress = defaultdict(float)
    length_steps_correct = defaultdict(int)
    length_steps_total = defaultdict(int)

    for eid, result in results.items():
        num_steps = result["num_steps"]

        # Determine length bucket
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

    # Format per-step-position
    max_step = max(step_pos_total.keys()) if step_pos_total else 0
    step_position_acc = {}
    for idx in range(min(max_step + 1, 15)):  # Cap at 15 for readability
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

    return {
        "step_position_accuracy": step_position_acc,
        "task_length_metrics": task_length_metrics,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Step-aware prompt variant evaluation")
    parser.add_argument("--prompt_variant", required=True,
                        choices=list(PROMPT_VARIANTS.keys()),
                        help="Prompt variant to evaluate")
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--gt_history", action="store_true", default=False)
    parser.add_argument("--image_max_pixels", type=int, default=None)
    parser.add_argument("--start", type=int, default=0,
                        help="Start index for episode slicing")
    parser.add_argument("--end", type=int, default=None,
                        help="End index for episode slicing")
    args = parser.parse_args()

    episodes = []
    with open(args.test_data) as f:
        for line in f:
            episodes.append(json.loads(line))
    total_loaded = len(episodes)
    episodes = episodes[args.start:args.end]
    print(f"Loaded {len(episodes)} episodes (shard [{args.start}:{args.end}] of {total_loaded})")
    print(f"Prompt variant: {args.prompt_variant}")
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
                args.prompt_variant, args.gt_history,
                args.match_threshold, args.image_max_pixels,
            ): ep["episode_id"]
            for ep in episodes
        }

        pbar = tqdm(total=len(episodes), desc=f"Eval {args.prompt_variant}")
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
        "prompt_variant": args.prompt_variant,
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
    print(f"\n{'='*60}")
    print(f"Step-Aware Eval: {args.prompt_variant}")
    print(f"{'='*60}")
    print(f"  TSR:      {summary['tsr']*100:.1f}%")
    print(f"  Step SR:  {summary['step_sr']*100:.1f}%")
    print(f"  Progress: {summary['avg_progress']*100:.1f}%")
    hist_mode = "GT history" if args.gt_history else "Pred history"
    print(f"  Mode:     {hist_mode}")
    print(f"  (Baselines: standard pred=21.9%, CoT pred=23.1%)")

    print(f"\n  --- By Task Length ---")
    for bname, metrics in sorted(summary.get("task_length_metrics", {}).items()):
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

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
