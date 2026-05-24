#!/usr/bin/env python3
"""Decomposed evaluation: separate action decision from visual grounding.

Tests whether separating "what to do" from "where to do it" improves accuracy.

Two-pass approach using the SAME model:
  Pass 1 (Decision): Given screenshot + goal + history, describe what action to take
          and which element to interact with (NO coordinates)
  Pass 2 (Grounding): Given screenshot + element description, output coordinates

This diagnoses whether failures are from:
  (a) Wrong decision (model doesn't know WHAT to do) → Pass 1 fails
  (b) Wrong grounding (model knows what but can't find it) → Pass 1 right, Pass 2 wrong

Usage:
    python v15_gui_360/eval_decomposed.py \
        --test_data v12_gui_360/data/gui360_test_1000_balanced.jsonl \
        --api_url http://localhost:8000/v1 \
        --model_name balanced_sft_step272 \
        --output_dir v15_gui_360/outputs/preliminary_tests/decomposed \
        --threads 64 --gt_history
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
from v13_gui_360.reward import compute_step_reward
from v13_gui_360.eval_gui360_template import (
    parse_tool_call, _format_action_for_history, SUPPORTED_ACTIONS,
    _encode_screenshot,
)

# ═══════════════════════════════════════════════════════════════════════
# Pass 1: Decision prompt — describe action + target element, NO coordinates
# ═══════════════════════════════════════════════════════════════════════

DECISION_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, describe what action should be taken next.

The instruction is:
{instruction}

The history of actions are:
{history}

The actions supported are: click, type, drag, wheel_mouse_input

DO NOT output coordinates or a tool_call. Instead, answer these two questions:
1. What ACTION TYPE should be taken? (click / type / drag / scroll)
2. WHICH ELEMENT on the screen should be interacted with? Describe it precisely (e.g., "the 'Insert' tab in the top ribbon menu", "the search text box at the top right", "the 'OK' button in the dialog box").

If the action is 'type', also specify what text should be typed.

Format your answer as:
ACTION: <click|type|drag|scroll>
TARGET: <precise description of the UI element>
TEXT: <text to type, only if action is type>
"""

# ═══════════════════════════════════════════════════════════════════════
# Pass 2: Grounding prompt — find element by description
# ═══════════════════════════════════════════════════════════════════════

GROUNDING_PROMPT = """You are a helpful assistant that locates UI elements on screen.

Look at the screenshot and find this element:
{element_description}

Output the EXACT pixel coordinates [x, y] of this element within a <tool_call> tag:
<tool_call>
{{
  "function": "{action_type}",
  "args": {{"coordinate": [x, y]}},
  "status": "CONTINUE"
}}
</tool_call>

If the action is 'type', use this format:
<tool_call>
{{
  "function": "type",
  "args": {{"coordinate": [x, y], "keys": "{text}"}},
  "status": "CONTINUE"
}}
</tool_call>

Output ONLY the tool_call, nothing else."""


def _parse_decision(text: str) -> Dict[str, str]:
    """Parse the decision output into action_type, target, text."""
    result = {"action_type": "", "target": "", "text": ""}

    # Try structured format first
    action_m = re.search(r'ACTION:\s*(\w+)', text, re.IGNORECASE)
    target_m = re.search(r'TARGET:\s*(.+?)(?:\n|TEXT:|$)', text, re.IGNORECASE | re.DOTALL)
    text_m = re.search(r'TEXT:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)

    if action_m:
        result["action_type"] = action_m.group(1).strip().lower()
    if target_m:
        result["target"] = target_m.group(1).strip()
    if text_m:
        result["text"] = text_m.group(1).strip()

    # Normalize action type
    if result["action_type"] in ("scroll", "wheel", "wheel_mouse_input"):
        result["action_type"] = "scroll"
    elif result["action_type"] in ("drag", "swipe"):
        result["action_type"] = "drag"

    return result


def evaluate_episode_decomposed(
    client: OpenAI,
    model_name: str,
    episode: Dict,
    gt_history: bool = True,
    match_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate a single episode with decomposed decision + grounding."""
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

        history_text = "\n".join(history) if history else "None"
        b64 = _encode_screenshot(screenshot)

        # ── Pass 1: Decision ──
        decision_text = ""
        try:
            decision_prompt = DECISION_PROMPT.format(
                instruction=goal,
                history=history_text,
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": decision_prompt},
                ]}],
                max_tokens=256,
                temperature=0.0,
            )
            decision_text = response.choices[0].message.content or ""
        except Exception as e:
            print(f"  [ep {episode_id}] step {i+1} decision error: {e}")

        decision = _parse_decision(decision_text)

        # ── Pass 2: Grounding ──
        pred_text = ""
        if decision["target"]:
            try:
                grounding_prompt = GROUNDING_PROMPT.format(
                    element_description=decision["target"],
                    action_type=decision["action_type"] or "click",
                    text=decision.get("text", ""),
                )
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": grounding_prompt},
                    ]}],
                    max_tokens=256,
                    temperature=0.0,
                )
                pred_text = response.choices[0].message.content or ""
            except Exception as e:
                print(f"  [ep {episode_id}] step {i+1} grounding error: {e}")

        # Parse grounding output
        pred_action = parse_tool_call(pred_text)
        if pred_action is None:
            m = re.search(r'<action>\s*(\{.*?\})\s*</action>', pred_text, re.DOTALL)
            if m:
                try:
                    pred_action = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass

        # Compute reward
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
            "decision_text": decision_text[:200],
            "decision_action_type": decision["action_type"],
            "decision_target": decision["target"][:150],
            "decision_type_text": decision.get("text", "")[:100],
            "grounding_text": pred_text[:200],
            "pred_action": info.get("pred_action"),
            "pred_type": info.get("pred_type"),
            "gt_type": info.get("gt_type"),
            "type_reward": info.get("type_reward", 0),
            "content_reward": info.get("content_reward", 0),
        })

        # Update history with GT
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--gt_history", action="store_true", default=False)
    args = parser.parse_args()

    episodes = []
    with open(args.test_data) as f:
        for line in f:
            episodes.append(json.loads(line))
    print(f"Loaded {len(episodes)} episodes")

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
                evaluate_episode_decomposed, client, args.model_name, ep,
                args.gt_history, args.match_threshold,
            ): ep["episode_id"]
            for ep in episodes
        }

        pbar = tqdm(total=len(episodes), desc="Decomposed eval")
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

    n = len(results)
    summary = {
        "num_episodes": n,
        "tsr": total_success / n if n > 0 else 0,
        "avg_progress": total_progress / n if n > 0 else 0,
        "step_sr": total_correct / total_steps if total_steps > 0 else 0,
        "total_steps_evaluated": total_steps,
        "total_steps_correct": total_correct,
        "mode": "decomposed",
        "gt_history": args.gt_history,
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.output_dir, f"eval_summary_{ts}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, f"eval_results_{ts}.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Decomposed Eval Results")
    print(f"  TSR:      {summary['tsr']*100:.1f}%")
    print(f"  Step SR:  {summary['step_sr']*100:.1f}%")
    print(f"  Progress: {summary['avg_progress']*100:.1f}%")
    print(f"  Episodes: {n}")
    print(f"{'='*50}")

    # Diagnostic: decision accuracy
    decision_type_correct = 0
    decision_total = 0
    from v13_gui_360.reward import _normalize_action_type
    for eid, ep in results.items():
        for s in ep["steps"]:
            decision_total += 1
            gt_type = s.get("gt_type", "")
            dec_type = _normalize_action_type(s.get("decision_action_type", ""))
            if dec_type == gt_type:
                decision_type_correct += 1

    if decision_total > 0:
        print(f"\n  Decision type accuracy: {decision_type_correct}/{decision_total} = "
              f"{decision_type_correct/decision_total*100:.1f}%")
        print(f"  (vs baseline end-to-end type accuracy: ~56%)")


if __name__ == "__main__":
    main()
