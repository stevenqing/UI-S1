#!/usr/bin/env python3
"""Self-verification diagnostic: Can the model detect its own errors?

For each step:
  1. Model predicts action (normal evaluation)
  2. Model sees: current screenshot + its predicted action + GT next screenshot
  3. Model answers: "Is the screen change consistent with my action?" (YES/NO)

Key metrics:
  - Detection rate (TPR): % of wrong predictions correctly flagged
  - False alarm rate (FPR): % of correct predictions incorrectly flagged
  - If TPR > 50%, self-correction via retry is feasible

Mathematical impact:
  With per-step p=0.60, detection d, and R retries:
    effective_p ≈ 1 - (1-p) * (1-d)^R
    d=0.5, R=2: effective_p ≈ 0.85 → TSR(6) = 37.7%

Usage:
    python v19_step_aware/eval_self_verify.py \
        --test_data v12_gui_360/data/gui360_test_1000_balanced.jsonl \
        --api_url http://localhost:8000/v1 \
        --model_name base_sft \
        --output_dir v19_step_aware/outputs/self_verify_pred \
        --threads 64
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v13_gui_360.eval_gui360_template import (
    parse_tool_call, _format_action_for_history, SUPPORTED_ACTIONS,
    _encode_screenshot, USER_PROMPT_TEMPLATE,
)
from v13_gui_360.reward import compute_step_reward


# ═══════════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════════

# Action prediction prompt (type_focused, our best variant)
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

# Verification prompt: two screenshots + action description
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


def _describe_action(action: Optional[Dict]) -> str:
    """Convert action dict to natural language description."""
    if action is None:
        return "(no action predicted)"

    atype = action.get("action", "unknown")
    coord = action.get("coordinate")

    if atype == "click":
        if coord:
            return f"Click at coordinates [{coord[0]}, {coord[1]}]"
        return "Click (no coordinates)"
    elif atype == "type":
        text = action.get("text", "")
        if coord and text:
            return f"Type '{text[:50]}' at coordinates [{coord[0]}, {coord[1]}]"
        elif text:
            return f"Type '{text[:50]}'"
        elif coord:
            return f"Type at coordinates [{coord[0]}, {coord[1]}]"
        return "Type (no text/coordinates)"
    elif atype in ("swipe", "drag"):
        start = action.get("coordinate")
        end = action.get("endCoordinate")
        if start and end:
            return f"Drag from [{start[0]}, {start[1]}] to [{end[0]}, {end[1]}]"
        return "Drag/scroll"
    else:
        return f"{atype} action"


# ═══════════════════════════════════════════════════════════════════════
# Episode Evaluation with Verification
# ═══════════════════════════════════════════════════════════════════════

def evaluate_episode(
    client: OpenAI,
    model_name: str,
    episode: Dict,
    gt_history: bool = False,
    match_threshold: float = 0.5,
    image_max_pixels: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate one episode: predict actions + verify each step."""
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

        # ── Step 1: Predict action ──
        prompt_text = PREDICT_PROMPT.format(
            instruction=goal,
            history=history_text,
            actions=SUPPORTED_ACTIONS,
        )

        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_before}"}},
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

        if pred_action:
            fake_text = f"<action>{json.dumps(pred_action)}</action>"
        else:
            fake_text = pred_text

        reward, info = compute_step_reward(fake_text, gt_action, image_w, image_h)
        step_correct = reward >= match_threshold

        # ── Step 2: Verify action (if next screenshot available) ──
        verify_result = None
        verify_correct = None

        if i + 1 < num_steps:
            next_screenshot = steps[i + 1]["screenshot"]
            b64_after = _encode_screenshot(next_screenshot, image_max_pixels)
            action_desc = _describe_action(pred_action)

            verify_prompt = VERIFY_PROMPT.format(
                action_description=action_desc,
                instruction=goal,
            )

            verify_messages = [{"role": "user", "content": [
                {"type": "text", "text": "Screenshot 1 (BEFORE the action):"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_before}"}},
                {"type": "text", "text": "Screenshot 2 (AFTER the action):"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_after}"}},
                {"type": "text", "text": verify_prompt},
            ]}]

            try:
                v_response = client.chat.completions.create(
                    model=model_name,
                    messages=verify_messages,
                    max_tokens=32,
                    temperature=0.0,
                )
                v_text = (v_response.choices[0].message.content or "").strip().upper()
                # Parse YES/NO
                if v_text.startswith("YES"):
                    verify_result = "YES"
                elif v_text.startswith("NO"):
                    verify_result = "NO"
                else:
                    verify_result = v_text[:20]

                # Was verification correct?
                if step_correct and verify_result == "YES":
                    verify_correct = True  # True negative (correct action, verified OK)
                elif not step_correct and verify_result == "NO":
                    verify_correct = True  # True positive (wrong action, detected)
                elif step_correct and verify_result == "NO":
                    verify_correct = False  # False positive (correct action, flagged)
                elif not step_correct and verify_result == "YES":
                    verify_correct = False  # False negative (wrong action, missed)

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

        # Update history
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


# ═══════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════

def compute_breakdown(results: Dict[str, Dict]) -> Dict[str, Any]:
    """Compute verification metrics + standard breakdowns."""

    # Verification confusion matrix
    tp = 0  # wrong action, detected (NO)
    fn = 0  # wrong action, missed (YES)
    fp = 0  # correct action, flagged (NO)
    tn = 0  # correct action, verified (YES)

    # By action type
    type_tp = defaultdict(int)
    type_fn = defaultdict(int)
    type_fp = defaultdict(int)
    type_tn = defaultdict(int)

    # By step position
    pos_tp = defaultdict(int)
    pos_fn = defaultdict(int)
    pos_fp = defaultdict(int)
    pos_tn = defaultdict(int)

    # Per-task-length
    length_buckets = {"1": (1, 1), "2-3": (2, 3), "4-5": (4, 5), "6+": (6, 999)}
    length_success = defaultdict(int)
    length_total = defaultdict(int)
    length_progress = defaultdict(float)

    # Simulated self-correction TSR
    # If model retries when verification says NO, using different temperature
    # We simulate: if verify=NO and step was wrong, the retry "succeeds" with prob p_retry
    # This gives us expected TSR under self-correction

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

        for step in result["steps"]:
            vr = step.get("verify_result")
            sc = step["success"]
            gt_type = step.get("gt_type", "unknown")
            idx = step["step_idx"]

            if vr is None:
                continue  # Last step (no next screenshot)

            if not sc and vr == "NO":
                tp += 1
                type_tp[gt_type] += 1
                pos_tp[idx] += 1
            elif not sc and vr == "YES":
                fn += 1
                type_fn[gt_type] += 1
                pos_fn[idx] += 1
            elif sc and vr == "NO":
                fp += 1
                type_fp[gt_type] += 1
                pos_fp[idx] += 1
            elif sc and vr == "YES":
                tn += 1
                type_tn[gt_type] += 1
                pos_tn[idx] += 1

    total_verified = tp + fn + fp + tn
    total_wrong = tp + fn
    total_correct = fp + tn

    verify_metrics = {
        "total_verified_steps": total_verified,
        "total_wrong_actions": total_wrong,
        "total_correct_actions": total_correct,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "detection_rate_TPR": tp / total_wrong if total_wrong > 0 else 0,
        "miss_rate_FNR": fn / total_wrong if total_wrong > 0 else 0,
        "false_alarm_FPR": fp / total_correct if total_correct > 0 else 0,
        "specificity_TNR": tn / total_correct if total_correct > 0 else 0,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "accuracy": (tp + tn) / total_verified if total_verified > 0 else 0,
    }

    # By action type
    verify_by_type = {}
    for atype in set(list(type_tp.keys()) + list(type_fn.keys())):
        t_wrong = type_tp[atype] + type_fn[atype]
        t_correct = type_fp[atype] + type_tn[atype]
        verify_by_type[atype] = {
            "TPR": type_tp[atype] / t_wrong if t_wrong > 0 else 0,
            "FPR": type_fp[atype] / t_correct if t_correct > 0 else 0,
            "wrong_actions": t_wrong,
            "correct_actions": t_correct,
            "tp": type_tp[atype], "fn": type_fn[atype],
            "fp": type_fp[atype], "tn": type_tn[atype],
        }

    # By step position
    verify_by_position = {}
    max_pos = max(list(pos_tp.keys()) + list(pos_fn.keys()) + [0])
    for idx in range(min(max_pos + 1, 12)):
        p_wrong = pos_tp[idx] + pos_fn[idx]
        p_correct = pos_fp[idx] + pos_tn[idx]
        if p_wrong + p_correct > 0:
            verify_by_position[f"step_{idx}"] = {
                "TPR": pos_tp[idx] / p_wrong if p_wrong > 0 else 0,
                "FPR": pos_fp[idx] / p_correct if p_correct > 0 else 0,
                "wrong": p_wrong,
                "correct": p_correct,
            }

    # Simulated self-correction impact
    # If we retry when verify=NO, with retry success prob = base per-step accuracy
    # effective_p = p + (1-p)*TPR*p - p*FPR*(1-p)  (simplified)
    # More precisely: P(step_ok) = P(correct) * P(not_flagged|correct) + retry_contribution
    p_base = total_correct / total_verified if total_verified > 0 else 0.5
    tpr = verify_metrics["detection_rate_TPR"]
    fpr = verify_metrics["false_alarm_FPR"]

    # P(accept correct) = p * (1 - FPR)
    # P(reject wrong, retry succeeds) = (1-p) * TPR * p_retry
    # P(reject correct, retry) = p * FPR * p_retry  (might get it right again)
    # Simplified: effective_p ≈ p*(1-FPR) + (1-p)*TPR*p + p*FPR*p
    p_retry = p_base  # assume retry has same accuracy
    effective_p = (p_base * (1 - fpr) +
                   (1 - p_base) * tpr * p_retry +
                   p_base * fpr * p_retry)

    simulated = {
        "base_step_accuracy": p_base,
        "effective_step_accuracy_1retry": effective_p,
        "simulated_TSR_6step_base": p_base ** 6,
        "simulated_TSR_6step_1retry": effective_p ** 6,
    }

    # Task length metrics
    task_length_metrics = {}
    for bname in length_buckets:
        total = length_total.get(bname, 0)
        if total > 0:
            task_length_metrics[bname] = {
                "tsr": length_success[bname] / total,
                "avg_progress": length_progress[bname] / total,
                "num_episodes": total,
            }

    return {
        "verification_metrics": verify_metrics,
        "verify_by_action_type": verify_by_type,
        "verify_by_position": verify_by_position,
        "simulated_self_correction": simulated,
        "task_length_metrics": task_length_metrics,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Self-verification diagnostic")
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threads", type=int, default=64)
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
    total_loaded = len(episodes)
    episodes = episodes[args.start:args.end]
    print(f"Loaded {len(episodes)} episodes (shard [{args.start}:{args.end}] of {total_loaded})")
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
                args.gt_history, args.match_threshold, args.image_max_pixels,
            ): ep["episode_id"]
            for ep in episodes
        }

        pbar = tqdm(total=len(episodes), desc="SelfVerify")
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
        "gt_history": args.gt_history,
        **breakdown,
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.output_dir, f"eval_summary_{ts}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, f"eval_results_{ts}.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Print results
    hist_mode = "GT history" if args.gt_history else "Pred history"
    vm = breakdown["verification_metrics"]
    sim = breakdown["simulated_self_correction"]

    print(f"\n{'='*65}")
    print(f"Self-Verification Diagnostic ({hist_mode})")
    print(f"{'='*65}")
    print(f"  TSR:      {summary['tsr']*100:.1f}%")
    print(f"  Step SR:  {summary['step_sr']*100:.1f}%")

    print(f"\n  --- Verification Confusion Matrix ---")
    print(f"  Total verified steps: {vm['total_verified_steps']}")
    print(f"                     Verify=YES    Verify=NO")
    print(f"  Action WRONG:     {vm['fn']:>6d} (FN)   {vm['tp']:>6d} (TP)")
    print(f"  Action CORRECT:   {vm['tn']:>6d} (TN)   {vm['fp']:>6d} (FP)")
    print(f"")
    print(f"  Detection Rate (TPR):  {vm['detection_rate_TPR']*100:.1f}%  "
          f"← Can detect {vm['detection_rate_TPR']*100:.0f}% of errors")
    print(f"  Miss Rate (FNR):       {vm['miss_rate_FNR']*100:.1f}%")
    print(f"  False Alarm (FPR):     {vm['false_alarm_FPR']*100:.1f}%  "
          f"← Wrongly flags {vm['false_alarm_FPR']*100:.0f}% of correct actions")
    print(f"  Precision:             {vm['precision']*100:.1f}%")
    print(f"  Overall Accuracy:      {vm['accuracy']*100:.1f}%")

    print(f"\n  --- By Action Type ---")
    for atype, m in sorted(breakdown["verify_by_action_type"].items()):
        print(f"  {atype:>8s}: TPR={m['TPR']*100:5.1f}%  FPR={m['FPR']*100:5.1f}%  "
              f"(wrong={m['wrong_actions']}, correct={m['correct_actions']})")

    print(f"\n  --- By Step Position ---")
    for sname, m in sorted(breakdown["verify_by_position"].items(),
                            key=lambda x: int(x[0].split("_")[1])):
        print(f"  {sname:>8s}: TPR={m['TPR']*100:5.1f}%  FPR={m['FPR']*100:5.1f}%  "
              f"(wrong={m['wrong']}, correct={m['correct']})")

    print(f"\n  --- Simulated Self-Correction Impact ---")
    print(f"  Base step accuracy:     {sim['base_step_accuracy']*100:.1f}%")
    print(f"  With 1 retry:           {sim['effective_step_accuracy_1retry']*100:.1f}%")
    print(f"  6-step TSR (base):      {sim['simulated_TSR_6step_base']*100:.1f}%")
    print(f"  6-step TSR (1 retry):   {sim['simulated_TSR_6step_1retry']*100:.1f}%")

    print(f"\n  --- By Task Length ---")
    for bname in ["1", "2-3", "4-5", "6+"]:
        metrics = breakdown.get("task_length_metrics", {}).get(bname)
        if metrics:
            print(f"  {bname:>5s}: TSR={metrics['tsr']*100:5.1f}%  "
                  f"Progress={metrics['avg_progress']*100:5.1f}%  "
                  f"(n={metrics['num_episodes']})")

    print(f"{'='*65}")


if __name__ == "__main__":
    main()
