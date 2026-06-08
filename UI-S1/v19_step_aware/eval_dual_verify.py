#!/usr/bin/env python3
"""Dual-model self-verification: separate predictor and verifier.

Uses base_sft (port 8000) for action prediction and a trained verifier
(port 8001) for verification. This avoids the catastrophic forgetting
issue of using a single merged model for both tasks.

Usage:
    python v19_step_aware/eval_dual_verify.py \
        --test_data v12_gui_360/data/gui360_test_1000_balanced.jsonl \
        --predict_url http://localhost:8000/v1 --predict_model base_sft \
        --verify_url http://localhost:8001/v1 --verify_model verifier \
        --output_dir v19_step_aware/outputs/dual_verify_pred \
        --threads 48
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
    _encode_screenshot,
)
from v13_gui_360.reward import compute_step_reward


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

VERIFY_PROMPT = """You are verifying whether a GUI action was executed correctly.

You will see two screenshots:
- Screenshot 1 (BEFORE): The screen before the action
- Screenshot 2 (AFTER): The screen after the action was executed

The action that was performed:
{action_description}

The task instruction: {instruction}

Carefully compare the two screenshots. Does Screenshot 2 show the expected result of performing the described action on Screenshot 1?

Consider:
- If it was a click action: Did the clicked element respond? (e.g., button pressed, menu opened, field focused)
- If it was a type action: Does the typed text appear in the correct field?
- If it was a scroll/drag: Did the content move in the expected direction?

Answer ONLY with one word: YES or NO"""


def _describe_action(action):
    if action is None:
        return "(no action predicted)"
    atype = action.get("action", "unknown")
    coord = action.get("coordinate")
    def _c(c):
        if not c or not isinstance(c, (list, tuple)) or len(c) < 2 or c[0] is None:
            return None
        return [int(float(c[0])), int(float(c[1]))]
    coord = _c(coord)
    if atype == "click":
        return f"Click at coordinates [{coord[0]}, {coord[1]}]" if coord else "Click"
    elif atype == "type":
        text = action.get("text", "")
        if coord and text:
            return f"Type '{text[:50]}' at coordinates [{coord[0]}, {coord[1]}]"
        elif text:
            return f"Type '{text[:50]}'"
        return "Type"
    elif atype in ("swipe", "drag"):
        start = _c(action.get("coordinate"))
        end = _c(action.get("endCoordinate"))
        if start and end:
            return f"Drag from [{start[0]}, {start[1]}] to [{end[0]}, {end[1]}]"
        return "Drag/scroll"
    return f"{atype}"


def evaluate_episode(
    pred_client: OpenAI,
    pred_model: str,
    verify_client: OpenAI,
    verify_model: str,
    episode: Dict,
    gt_history: bool = False,
    match_threshold: float = 0.5,
    image_max_pixels: Optional[int] = None,
) -> Dict[str, Any]:
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
        b64_before = _encode_screenshot(screenshot, image_max_pixels)

        # ── Predict (using predictor model) ──
        prompt_text = PREDICT_PROMPT.format(
            instruction=goal, history=history_text, actions=SUPPORTED_ACTIONS)
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_before}"}},
            {"type": "text", "text": prompt_text},
        ]}]

        pred_text = ""
        try:
            response = pred_client.chat.completions.create(
                model=pred_model, messages=messages,
                max_tokens=1024, temperature=0.0)
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

        if pred_action:
            fake_text = f"<action>{json.dumps(pred_action)}</action>"
        else:
            fake_text = pred_text

        reward, info = compute_step_reward(fake_text, gt_action, image_w, image_h)
        step_correct = reward >= match_threshold

        # ── Verify (using verifier model) ──
        verify_result = None
        verify_correct = None

        if i + 1 < num_steps:
            next_screenshot = steps[i + 1]["screenshot"]
            b64_after = _encode_screenshot(next_screenshot, image_max_pixels)
            action_desc = _describe_action(pred_action)

            v_prompt = VERIFY_PROMPT.format(
                action_description=action_desc, instruction=goal)
            v_messages = [{"role": "user", "content": [
                {"type": "text", "text": "Screenshot 1 (BEFORE the action):"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_before}"}},
                {"type": "text", "text": "Screenshot 2 (AFTER the action):"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_after}"}},
                {"type": "text", "text": v_prompt},
            ]}]

            try:
                v_response = verify_client.chat.completions.create(
                    model=verify_model, messages=v_messages,
                    max_tokens=32, temperature=0.0)
                v_text = (v_response.choices[0].message.content or "").strip().upper()
                if v_text.startswith("YES"):
                    verify_result = "YES"
                elif v_text.startswith("NO"):
                    verify_result = "NO"
                else:
                    verify_result = v_text[:20]

                if step_correct and verify_result == "YES":
                    verify_correct = True
                elif not step_correct and verify_result == "NO":
                    verify_correct = True
                elif step_correct and verify_result == "NO":
                    verify_correct = False
                elif not step_correct and verify_result == "YES":
                    verify_correct = False
            except Exception as e:
                print(f"  [ep {episode_id}] step {i+1} verify error: {e}")

        step_results.append({
            "step_idx": i,
            "success": step_correct,
            "reward": reward,
            "pred_type": info.get("pred_type"),
            "gt_type": info.get("gt_type"),
            "type_reward": info.get("type_reward", 0),
            "content_reward": info.get("content_reward", 0),
            "verify_result": verify_result,
            "verify_correct": verify_correct,
            "pred_text": pred_text[:300],
        })

        if gt_history:
            history.append(_format_action_for_history(gt_action, i + 1))
        else:
            history.append(_format_action_for_history(pred_action, i + 1))

        if step_correct:
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


def compute_breakdown(results):
    """Same breakdown as eval_self_verify.py."""
    tp = fn = fp = tn = 0
    type_tp = defaultdict(int)
    type_fn = defaultdict(int)
    type_fp = defaultdict(int)
    type_tn = defaultdict(int)
    pos_tp = defaultdict(int)
    pos_fn = defaultdict(int)
    pos_fp = defaultdict(int)
    pos_tn = defaultdict(int)

    length_buckets = {"1": (1,1), "2-3": (2,3), "4-5": (4,5), "6+": (6,999)}
    length_success = defaultdict(int)
    length_total = defaultdict(int)
    length_progress = defaultdict(float)

    for eid, result in results.items():
        num_steps = result["num_steps"]
        bucket = "6+"
        for bname, (lo, hi) in length_buckets.items():
            if lo <= num_steps <= hi:
                bucket = bname
                break
        length_total[bucket] += 1
        if result["task_success"]:
            length_success[bucket] += 1
        length_progress[bucket] += result["progress"]

        for step in result["steps"]:
            vr = step.get("verify_result")
            sc = step["success"]
            gt_type = step.get("gt_type", "unknown")
            idx = step["step_idx"]
            if vr is None:
                continue
            if not sc and vr == "NO":
                tp += 1; type_tp[gt_type] += 1; pos_tp[idx] += 1
            elif not sc and vr == "YES":
                fn += 1; type_fn[gt_type] += 1; pos_fn[idx] += 1
            elif sc and vr == "NO":
                fp += 1; type_fp[gt_type] += 1; pos_fp[idx] += 1
            elif sc and vr == "YES":
                tn += 1; type_tn[gt_type] += 1; pos_tn[idx] += 1

    total = tp + fn + fp + tn
    total_wrong = tp + fn
    total_correct = fp + tn

    vm = {
        "total_verified_steps": total,
        "total_wrong_actions": total_wrong,
        "total_correct_actions": total_correct,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "detection_rate_TPR": tp / total_wrong if total_wrong > 0 else 0,
        "miss_rate_FNR": fn / total_wrong if total_wrong > 0 else 0,
        "false_alarm_FPR": fp / total_correct if total_correct > 0 else 0,
        "specificity_TNR": tn / total_correct if total_correct > 0 else 0,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "accuracy": (tp + tn) / total if total > 0 else 0,
    }

    vbt = {}
    for at in set(list(type_tp) + list(type_fn)):
        tw = type_tp[at] + type_fn[at]
        tc = type_fp[at] + type_tn[at]
        vbt[at] = {
            "TPR": type_tp[at] / tw if tw > 0 else 0,
            "FPR": type_fp[at] / tc if tc > 0 else 0,
            "wrong_actions": tw, "correct_actions": tc,
        }

    vbp = {}
    max_pos = max(list(pos_tp) + list(pos_fn) + [0])
    for idx in range(min(max_pos + 1, 12)):
        pw = pos_tp[idx] + pos_fn[idx]
        pc = pos_fp[idx] + pos_tn[idx]
        if pw + pc > 0:
            vbp[f"step_{idx}"] = {
                "TPR": pos_tp[idx] / pw if pw > 0 else 0,
                "FPR": pos_fp[idx] / pc if pc > 0 else 0,
                "wrong": pw, "correct": pc,
            }

    p_base = total_correct / total if total > 0 else 0.5
    tpr = vm["detection_rate_TPR"]
    fpr = vm["false_alarm_FPR"]
    p_retry = p_base
    eff_p = p_base*(1-fpr) + (1-p_base)*tpr*p_retry + p_base*fpr*p_retry

    sim = {
        "base_step_accuracy": p_base,
        "effective_step_accuracy_1retry": eff_p,
        "simulated_TSR_6step_base": p_base ** 6,
        "simulated_TSR_6step_1retry": eff_p ** 6,
    }

    tlm = {}
    for bname in length_buckets:
        t = length_total.get(bname, 0)
        if t > 0:
            tlm[bname] = {
                "tsr": length_success[bname] / t,
                "avg_progress": length_progress[bname] / t,
                "num_episodes": t,
            }

    return {
        "verification_metrics": vm,
        "verify_by_action_type": vbt,
        "verify_by_position": vbp,
        "simulated_self_correction": sim,
        "task_length_metrics": tlm,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--predict_url", required=True)
    parser.add_argument("--predict_model", required=True)
    parser.add_argument("--verify_url", required=True)
    parser.add_argument("--verify_model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--gt_history", action="store_true", default=False)
    parser.add_argument("--image_max_pixels", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    episodes = []
    with open(args.test_data) as f:
        for line in f:
            episodes.append(json.loads(line))
    episodes = episodes[args.start:args.end]
    print(f"Loaded {len(episodes)} episodes")
    print(f"Predict: {args.predict_url} / {args.predict_model}")
    print(f"Verify:  {args.verify_url} / {args.verify_model}")

    pred_client = OpenAI(base_url=args.predict_url, api_key="dummy")
    verify_client = OpenAI(base_url=args.verify_url, api_key="dummy")
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    total_success = 0
    total_progress = 0.0
    total_steps = 0
    total_correct = 0

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(
                evaluate_episode, pred_client, args.predict_model,
                verify_client, args.verify_model, ep,
                args.gt_history, args.match_threshold, args.image_max_pixels,
            ): ep["episode_id"]
            for ep in episodes
        }
        pbar = tqdm(total=len(episodes), desc="DualVerify")
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

    breakdown = compute_breakdown(results)
    n = len(results)
    summary = {
        "num_episodes": n,
        "tsr": total_success / n if n > 0 else 0,
        "avg_progress": total_progress / n if n > 0 else 0,
        "step_sr": total_correct / total_steps if total_steps > 0 else 0,
        "gt_history": args.gt_history,
        "predict_model": args.predict_model,
        "verify_model": args.verify_model,
        **breakdown,
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.output_dir, f"eval_summary_{ts}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, f"eval_results_{ts}.json"), "w") as f:
        json.dump(results, f, indent=2)

    vm = breakdown["verification_metrics"]
    sim = breakdown["simulated_self_correction"]
    hist = "GT history" if args.gt_history else "Pred history"
    print(f"\n{'='*65}")
    print(f"Dual-Model Verification ({hist})")
    print(f"{'='*65}")
    print(f"  TSR:      {summary['tsr']*100:.1f}%")
    print(f"  Step SR:  {summary['step_sr']*100:.1f}%")
    print(f"  TPR:      {vm['detection_rate_TPR']*100:.1f}%")
    print(f"  FPR:      {vm['false_alarm_FPR']*100:.1f}%")
    print(f"  Precision:{vm['precision']*100:.1f}%")
    print(f"  Accuracy: {vm['accuracy']*100:.1f}%")
    print(f"  Simulated: {sim['base_step_accuracy']*100:.1f}% → {sim['effective_step_accuracy_1retry']*100:.1f}%")
    print(f"  6-step TSR: {sim['simulated_TSR_6step_base']*100:.1f}% → {sim['simulated_TSR_6step_1retry']*100:.1f}%")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
