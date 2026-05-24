#!/usr/bin/env python3
"""Self-verification & retry evaluation.

After predicting an action, the model verifies whether the predicted
action targets the correct element. If verification fails, retry with
a corrected prompt.

Two modes:
  (a) verify_only: predict → verify → if NO, mark as uncertain
  (b) verify_retry: predict → verify → if NO, retry up to max_retries times
      with the verification feedback included in the retry prompt

This targets the binary grounding problem: model either perfectly finds
the element (0.5px) or completely misidentifies it (333px). Verification
is easier than generation — the model just needs to check if the element
at the predicted coordinate matches the intended target.

Usage:
    python v15_gui_360/eval_self_verify.py \
        --test_data v12_gui_360/data/gui360_test_1000_balanced.jsonl \
        --api_url http://localhost:8000/v1 \
        --model_name balanced_sft_step272 \
        --output_dir v15_gui_360/outputs/preliminary_tests/self_verify \
        --threads 64 --gt_history --max_retries 2
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

# ═══════════════════════════════════════════════════════════════════════
# Verification prompt
# ═══════════════════════════════════════════════════════════════════════

VERIFY_PROMPT = """You are a helpful assistant that verifies UI actions.

The user instruction is:
{instruction}

The history of previous actions:
{history}

The proposed next action is:
{proposed_action}

Look at the screenshot carefully. Check whether the proposed action targets the CORRECT UI element for this instruction.

Consider:
1. Is the action TYPE correct (click vs type vs drag vs scroll)?
2. For click/type actions: does the coordinate point to the RIGHT element?
3. Does this action make logical sense as the next step?

Answer with ONLY one of:
- CORRECT: The action targets the right element and is appropriate.
- WRONG: The action targets the wrong element or is inappropriate. Briefly explain what element should be targeted instead."""

RETRY_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

Your previous attempt was INCORRECT:
Previous action: {previous_action}
Reason: {rejection_reason}

Please try again. Look at the screenshot more carefully and find the CORRECT element to interact with.

Output your action within <tool_call></tool_call> tag:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

Only **ONE** action should be taken at a time."""


def _format_pred_action_text(pred_action):
    """Format a predicted action dict into readable text for verification."""
    if not pred_action:
        return "(no action)"
    atype = pred_action.get("action", "")
    coord = pred_action.get("coordinate")
    if atype == "click" and coord:
        return f"click at coordinate [{coord[0]}, {coord[1]}]"
    elif atype == "type":
        text = pred_action.get("text", "")
        if coord:
            return f"type '{text[:50]}' at coordinate [{coord[0]}, {coord[1]}]"
        return f"type '{text[:50]}'"
    elif atype in ("swipe", "drag"):
        start = pred_action.get("coordinate")
        end = pred_action.get("endCoordinate")
        if start and end:
            return f"drag from [{start[0]}, {start[1]}] to [{end[0]}, {end[1]}]"
        return "drag"
    return str(pred_action)


def _parse_verification(text):
    """Parse verification response. Returns (is_correct, reason)."""
    text_lower = text.strip().lower()
    if text_lower.startswith("correct"):
        return True, ""
    if text_lower.startswith("wrong"):
        # Extract reason after "WRONG:"
        reason = text.strip()
        if ":" in reason:
            reason = reason.split(":", 1)[1].strip()
        return False, reason
    # Fallback: look for keywords
    if "correct" in text_lower and "wrong" not in text_lower:
        return True, ""
    if "wrong" in text_lower or "incorrect" in text_lower:
        reason = text.strip()
        return False, reason
    # Default: assume correct (conservative)
    return True, ""


def evaluate_episode_verify(
    client: OpenAI,
    model_name: str,
    episode: Dict,
    gt_history: bool = True,
    match_threshold: float = 0.5,
    max_retries: int = 2,
    image_max_pixels: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate one episode with self-verification and retry."""
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
        b64 = _encode_screenshot(screenshot, image_max_pixels)

        # ── Initial prediction ──
        prompt_text = USER_PROMPT_TEMPLATE.format(
            instruction=goal,
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
                max_tokens=1024,
                temperature=0.0,
            )
            pred_text = response.choices[0].message.content or ""
        except Exception as e:
            print(f"  [ep {episode_id}] step {i+1} predict error: {e}")

        pred_action = parse_tool_call(pred_text)
        if pred_action is None:
            m = re.search(r'<action>\s*(\{.*?\})\s*</action>', pred_text, re.DOTALL)
            if m:
                try:
                    pred_action = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass

        # ── Verification loop ──
        attempt = 0
        verified = False
        rejection_reasons = []

        while attempt <= max_retries:
            if pred_action is None:
                break

            # Verify
            verify_text = VERIFY_PROMPT.format(
                instruction=goal,
                history=history_text,
                proposed_action=_format_pred_action_text(pred_action),
            )
            verify_messages = [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": verify_text},
            ]}]

            verify_response_text = ""
            try:
                vr = client.chat.completions.create(
                    model=model_name,
                    messages=verify_messages,
                    max_tokens=256,
                    temperature=0.0,
                )
                verify_response_text = vr.choices[0].message.content or ""
            except Exception as e:
                print(f"  [ep {episode_id}] step {i+1} verify error: {e}")
                break

            is_correct, reason = _parse_verification(verify_response_text)

            if is_correct:
                verified = True
                break

            # Verification says WRONG → retry
            rejection_reasons.append(reason)
            attempt += 1

            if attempt > max_retries:
                break

            # Retry with feedback
            retry_text = RETRY_PROMPT.format(
                instruction=goal,
                history=history_text,
                actions=SUPPORTED_ACTIONS,
                previous_action=_format_pred_action_text(pred_action),
                rejection_reason=reason,
            )
            retry_messages = [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": retry_text},
            ]}]

            try:
                rr = client.chat.completions.create(
                    model=model_name,
                    messages=retry_messages,
                    max_tokens=1024,
                    temperature=0.0,
                )
                pred_text = rr.choices[0].message.content or ""
            except Exception as e:
                print(f"  [ep {episode_id}] step {i+1} retry error: {e}")
                break

            pred_action = parse_tool_call(pred_text)
            if pred_action is None:
                m = re.search(r'<action>\s*(\{.*?\})\s*</action>', pred_text, re.DOTALL)
                if m:
                    try:
                        pred_action = json.loads(m.group(1))
                    except json.JSONDecodeError:
                        pass

        # ── Compute final reward ──
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
            "attempts": attempt + 1,
            "verified": verified,
            "rejection_reasons": rejection_reasons[:3],
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
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--image_max_pixels", type=int, default=None)
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
                evaluate_episode_verify, client, args.model_name, ep,
                args.gt_history, args.match_threshold, args.max_retries,
                args.image_max_pixels,
            ): ep["episode_id"]
            for ep in episodes
        }

        pbar = tqdm(total=len(episodes), desc="Self-verify eval")
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
    # Retry statistics
    total_retries = 0
    retry_helped = 0
    retry_hurt = 0
    verify_reject_count = 0
    for eid, ep in results.items():
        for s in ep["steps"]:
            if s["attempts"] > 1:
                total_retries += 1
                verify_reject_count += s["attempts"] - 1
            # Can't directly measure helped/hurt without baseline per step,
            # but we can log how often verification rejected

    summary = {
        "num_episodes": n,
        "tsr": total_success / n if n > 0 else 0,
        "avg_progress": total_progress / n if n > 0 else 0,
        "step_sr": total_correct / total_steps if total_steps > 0 else 0,
        "total_steps_evaluated": total_steps,
        "total_steps_correct": total_correct,
        "mode": "self_verify",
        "max_retries": args.max_retries,
        "gt_history": args.gt_history,
        "steps_with_retries": total_retries,
        "total_verify_rejections": verify_reject_count,
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.output_dir, f"eval_summary_{ts}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, f"eval_results_{ts}.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Self-Verification & Retry Results (max_retries={args.max_retries})")
    print(f"{'='*60}")
    print(f"  TSR:      {summary['tsr']*100:.1f}%")
    print(f"  Step SR:  {summary['step_sr']*100:.1f}%")
    print(f"  Progress: {summary['avg_progress']*100:.1f}%")
    print(f"  Steps with retries:     {total_retries}/{total_steps}")
    print(f"  Total rejections:       {verify_reject_count}")
    print(f"  (Baseline greedy: TSR=22.5%, Step SR=58.8%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
