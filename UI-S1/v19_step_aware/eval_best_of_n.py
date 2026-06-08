#!/usr/bin/env python3
"""Best-of-N evaluation with verifier selection.

Generates N candidate actions per step, uses a trained verifier to select
the best one. This breaks the compound error trap by:
1. Increasing P(at least one correct) through multiple samples
2. Using verifier to filter out wrong candidates
3. Keeping correct actions in history to prevent cascade errors

Usage:
    python v19_step_aware/eval_best_of_n.py \
        --test_data v12_gui_360/data/gui360_test_1000_balanced.jsonl \
        --predict_url http://localhost:8000/v1 --predict_model base_sft \
        --verify_url http://localhost:8001/v1 --verify_model verifier \
        --output_dir v19_step_aware/outputs/best_of_3_pred \
        --n_candidates 3 --sample_temperature 0.7 \
        --threads 48
"""

import argparse
import json
import math
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


def _parse_prediction(pred_text: str) -> Optional[Dict]:
    """Parse a prediction response into an action dict."""
    pred_action = parse_tool_call(pred_text)
    if pred_action is None:
        m = re.search(r'<action>\s*(\{.*?\})\s*</action>', pred_text, re.DOTALL)
        if m:
            try:
                pred_action = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    return pred_action


def _generate_candidate(
    pred_client: OpenAI,
    pred_model: str,
    messages: List[Dict],
    temperature: float,
) -> Tuple[str, Optional[Dict]]:
    """Generate one candidate prediction."""
    try:
        response = pred_client.chat.completions.create(
            model=pred_model, messages=messages,
            max_tokens=1024, temperature=temperature)
        pred_text = response.choices[0].message.content or ""
    except Exception as e:
        return f"ERROR: {e}", None
    pred_action = _parse_prediction(pred_text)
    return pred_text, pred_action


def _verify_candidate(
    verify_client: OpenAI,
    verify_model: str,
    b64_before: str,
    b64_after: str,
    action: Optional[Dict],
    instruction: str,
    extract_logprobs: bool = False,
) -> Tuple[Optional[str], Optional[float]]:
    """Verify one candidate action.

    Returns:
        (binary_result, logprob_score)
        binary_result: 'YES', 'NO', or None
        logprob_score: logP(YES) - logP(NO) if extract_logprobs, else None
    """
    action_desc = _describe_action(action)
    v_prompt = VERIFY_PROMPT.format(
        action_description=action_desc, instruction=instruction)
    v_messages = [{"role": "user", "content": [
        {"type": "text", "text": "Screenshot 1 (BEFORE the action):"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_before}"}},
        {"type": "text", "text": "Screenshot 2 (AFTER the action):"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_after}"}},
        {"type": "text", "text": v_prompt},
    ]}]
    try:
        kwargs = dict(model=verify_model, messages=v_messages,
                      max_tokens=1, temperature=0.0)
        if extract_logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 20
        v_response = verify_client.chat.completions.create(**kwargs)

        v_text = (v_response.choices[0].message.content or "").strip().upper()
        binary_result = None
        if v_text.startswith("YES") or v_text == "Y":
            binary_result = "YES"
        elif v_text.startswith("NO") or v_text == "N":
            binary_result = "NO"
        else:
            binary_result = v_text[:20]

        logprob_score = None
        if extract_logprobs and v_response.choices[0].logprobs:
            content_logprobs = v_response.choices[0].logprobs.content
            if content_logprobs and len(content_logprobs) > 0:
                top_lps = content_logprobs[0].top_logprobs
                yes_lp = -100.0
                no_lp = -100.0
                for tlp in top_lps:
                    tok = tlp.token.strip().upper()
                    if tok in ("YES", "Y", "Yes", "yes"):
                        yes_lp = max(yes_lp, tlp.logprob)
                    elif tok in ("NO", "N", "No", "no"):
                        no_lp = max(no_lp, tlp.logprob)
                if yes_lp > -100 or no_lp > -100:
                    logprob_score = yes_lp - no_lp

        return binary_result, logprob_score
    except Exception:
        return None, None


def evaluate_episode(
    pred_client: OpenAI,
    pred_model: str,
    verify_client: OpenAI,
    verify_model: str,
    episode: Dict,
    n_candidates: int = 3,
    sample_temperature: float = 0.7,
    selection_strategy: str = "first_yes",
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
    total_retries = 0

    for i, step in enumerate(steps):
        gt_action = step["action"]
        screenshot = step["screenshot"]
        image_w = step.get("image_w", 1040)
        image_h = step.get("image_h", 736)

        history_text = "\n".join(history) if history else "None"
        b64_before = _encode_screenshot(screenshot, image_max_pixels)

        prompt_text = PREDICT_PROMPT.format(
            instruction=goal, history=history_text, actions=SUPPORTED_ACTIONS)
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_before}"}},
            {"type": "text", "text": prompt_text},
        ]}]

        # ── Generate N candidates ──
        # Candidate 0: greedy (temperature=0)
        # Candidates 1..N-1: sampled (temperature=sample_temperature)
        candidates = []

        greedy_text, greedy_action = _generate_candidate(
            pred_client, pred_model, messages, temperature=0.0)
        candidates.append((greedy_text, greedy_action, 0.0))

        for k in range(1, n_candidates):
            s_text, s_action = _generate_candidate(
                pred_client, pred_model, messages, temperature=sample_temperature)
            candidates.append((s_text, s_action, sample_temperature))

        # ── Verify candidates against GT next screenshot ──
        can_verify = (i + 1 < num_steps)
        b64_after = None
        if can_verify:
            next_screenshot = steps[i + 1]["screenshot"]
            b64_after = _encode_screenshot(next_screenshot, image_max_pixels)

        selected_idx = 0  # default to greedy
        candidate_results = []

        for k, (c_text, c_action, c_temp) in enumerate(candidates):
            # Compute GT reward for this candidate
            if c_action:
                fake_text = f"<action>{json.dumps(c_action)}</action>"
            else:
                fake_text = c_text
            reward, info = compute_step_reward(fake_text, gt_action, image_w, image_h)
            is_correct = reward >= match_threshold

            # Verify against GT next screenshot
            v_result = None
            v_score = None
            use_logprobs = (selection_strategy == "logprob_rank")
            if can_verify and c_action is not None:
                v_result, v_score = _verify_candidate(
                    verify_client, verify_model,
                    b64_before, b64_after, c_action, goal,
                    extract_logprobs=use_logprobs)

            candidate_results.append({
                "idx": k,
                "temperature": c_temp,
                "action": _describe_action(c_action) if c_action else "(parse error)",
                "reward": reward,
                "is_correct": is_correct,
                "verify_result": v_result,
                "verify_score": v_score,
                "pred_type": info.get("pred_type"),
                "gt_type": info.get("gt_type"),
            })

        # ── Selection ──
        if can_verify:
            if selection_strategy == "logprob_rank":
                # Pick candidate with highest logP(YES) - logP(NO)
                best_score = -float('inf')
                for k, cr in enumerate(candidate_results):
                    s = cr.get("verify_score")
                    if s is not None and s > best_score:
                        best_score = s
                        selected_idx = k
            else:
                # first_yes: pick first candidate verified YES
                for k, cr in enumerate(candidate_results):
                    if cr["verify_result"] == "YES":
                        selected_idx = k
                        break
                # If none verified YES, fall back to greedy (idx=0)

        selected = candidate_results[selected_idx]
        sel_text, sel_action, _ = candidates[selected_idx]
        step_correct = selected["is_correct"]
        retries_used = selected_idx  # 0 means greedy was accepted or no retry

        if retries_used > 0:
            total_retries += 1

        # ── Compute oracle: was any candidate correct? ──
        any_correct = any(cr["is_correct"] for cr in candidate_results)
        # Did we pick a correct one?
        selection_correct = selected["is_correct"]
        # Could we have done better?
        oracle_correct = any_correct

        step_results.append({
            "step_idx": i,
            "success": step_correct,
            "reward": selected["reward"],
            "selected_idx": selected_idx,
            "retries_used": retries_used,
            "n_candidates": len(candidates),
            "any_candidate_correct": any_correct,
            "selection_correct": selection_correct,
            "verify_result": selected["verify_result"],
            "pred_type": selected["pred_type"],
            "gt_type": selected["gt_type"],
            "candidates": candidate_results,
        })

        # ── Update history with SELECTED action ──
        if gt_history:
            history.append(_format_action_for_history(gt_action, i + 1))
        else:
            history.append(_format_action_for_history(sel_action, i + 1))

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
        "total_retries": total_retries,
        "steps": step_results,
    }


def compute_breakdown(results: Dict[str, Dict], n_candidates: int) -> Dict[str, Any]:
    """Compute verification and selection metrics."""
    # Verification confusion matrix
    tp = fn = fp = tn = 0
    type_tp = defaultdict(int)
    type_fn = defaultdict(int)
    type_fp = defaultdict(int)
    type_tn = defaultdict(int)

    # Selection quality
    total_steps_with_candidates = 0
    steps_any_correct = 0
    steps_selected_correct = 0
    steps_greedy_correct = 0
    steps_retry_helped = 0  # selected correct but greedy was wrong
    steps_retry_hurt = 0    # selected wrong but greedy was correct

    # Retry distribution
    retry_counts = defaultdict(int)

    # Task length buckets
    length_buckets = {"1": (1,1), "2-3": (2,3), "4-5": (4,5), "6+": (6,999)}
    length_success = defaultdict(int)
    length_total = defaultdict(int)
    length_progress = defaultdict(float)
    length_steps_correct = defaultdict(int)
    length_steps_total = defaultdict(int)

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
            cands = step.get("candidates", [])
            gt_type = step.get("gt_type", "unknown")
            sel_idx = step["selected_idx"]
            retry_counts[sel_idx] += 1

            length_steps_total[bucket] += 1
            if step["success"]:
                length_steps_correct[bucket] += 1

            if not cands:
                continue

            total_steps_with_candidates += 1
            greedy_correct = cands[0]["is_correct"] if cands else False
            any_correct = step.get("any_candidate_correct", False)
            sel_correct = step["selection_correct"]

            if any_correct:
                steps_any_correct += 1
            if sel_correct:
                steps_selected_correct += 1
            if greedy_correct:
                steps_greedy_correct += 1
            if sel_correct and not greedy_correct:
                steps_retry_helped += 1
            if not sel_correct and greedy_correct:
                steps_retry_hurt += 1

            # Verification confusion for ALL candidates
            for cr in cands:
                vr = cr.get("verify_result")
                sc = cr["is_correct"]
                if vr is None:
                    continue
                if not sc and vr == "NO":
                    tp += 1; type_tp[gt_type] += 1
                elif not sc and vr == "YES":
                    fn += 1; type_fn[gt_type] += 1
                elif sc and vr == "NO":
                    fp += 1; type_fp[gt_type] += 1
                elif sc and vr == "YES":
                    tn += 1; type_tn[gt_type] += 1

    total = tp + fn + fp + tn
    total_wrong = tp + fn
    total_correct = fp + tn

    vm = {
        "total_verified": total,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "TPR": tp / total_wrong if total_wrong > 0 else 0,
        "FPR": fp / total_correct if total_correct > 0 else 0,
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
            "wrong": tw, "correct": tc,
        }

    n = total_steps_with_candidates or 1
    sm = {
        "total_steps": total_steps_with_candidates,
        "greedy_accuracy": steps_greedy_correct / n,
        "oracle_accuracy": steps_any_correct / n,
        "selected_accuracy": steps_selected_correct / n,
        "retry_helped": steps_retry_helped,
        "retry_hurt": steps_retry_hurt,
        "net_gain": steps_retry_helped - steps_retry_hurt,
        "retry_distribution": dict(retry_counts),
    }

    # Simulated TSR from selected accuracy
    p_sel = sm["selected_accuracy"]
    p_greedy = sm["greedy_accuracy"]
    p_oracle = sm["oracle_accuracy"]
    sim = {
        "greedy_6step_TSR": p_greedy ** 6,
        "selected_6step_TSR": p_sel ** 6,
        "oracle_6step_TSR": p_oracle ** 6,
    }

    tlm = {}
    for bname in length_buckets:
        t = length_total.get(bname, 0)
        if t > 0:
            st = length_steps_total.get(bname, 0)
            tlm[bname] = {
                "tsr": length_success[bname] / t,
                "avg_progress": length_progress[bname] / t,
                "step_sr": length_steps_correct[bname] / st if st > 0 else 0,
                "num_episodes": t,
                "num_success": length_success[bname],
            }

    return {
        "verification_metrics": vm,
        "verify_by_type": vbt,
        "selection_metrics": sm,
        "simulated_tsr": sim,
        "task_length_metrics": tlm,
    }


def main():
    parser = argparse.ArgumentParser(description="Best-of-N eval with verifier selection")
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--predict_url", required=True)
    parser.add_argument("--predict_model", required=True)
    parser.add_argument("--verify_url", required=True)
    parser.add_argument("--verify_model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_candidates", type=int, default=3,
                        help="Number of candidates per step (1 greedy + N-1 sampled)")
    parser.add_argument("--sample_temperature", type=float, default=0.7)
    parser.add_argument("--selection_strategy", default="first_yes",
                        choices=["first_yes", "logprob_rank"],
                        help="first_yes: pick first YES; logprob_rank: pick highest logP(YES)-logP(NO)")
    parser.add_argument("--threads", type=int, default=32)
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
    print(f"N={args.n_candidates}, temp={args.sample_temperature}, strategy={args.selection_strategy}")
    print(f"GT history: {args.gt_history}")

    pred_client = OpenAI(base_url=args.predict_url, api_key="dummy")
    verify_client = OpenAI(base_url=args.verify_url, api_key="dummy")
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    total_success = 0
    total_progress = 0.0
    total_steps = 0
    total_correct = 0
    total_retries = 0

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(
                evaluate_episode, pred_client, args.predict_model,
                verify_client, args.verify_model, ep,
                args.n_candidates, args.sample_temperature, args.selection_strategy,
                args.gt_history, args.match_threshold, args.image_max_pixels,
            ): ep["episode_id"]
            for ep in episodes
        }
        pbar = tqdm(total=len(episodes), desc=f"BoN-{args.n_candidates}")
        for future in as_completed(futures):
            result = future.result()
            eid = result["episode_id"]
            results[eid] = result
            if result["task_success"]:
                total_success += 1
            total_progress += result["progress"]
            total_steps += result["steps_evaluated"]
            total_correct += result["correct_steps"]
            total_retries += result["total_retries"]
            n = len(results)
            pbar.update(1)
            pbar.set_postfix({
                "TSR": f"{total_success/n:.3f}",
                "StepSR": f"{total_correct/total_steps:.3f}" if total_steps > 0 else "0",
                "retries": total_retries,
            })
        pbar.close()

    breakdown = compute_breakdown(results, args.n_candidates)
    n = len(results)
    summary = {
        "num_episodes": n,
        "tsr": total_success / n if n > 0 else 0,
        "avg_progress": total_progress / n if n > 0 else 0,
        "step_sr": total_correct / total_steps if total_steps > 0 else 0,
        "n_candidates": args.n_candidates,
        "sample_temperature": args.sample_temperature,
        "selection_strategy": args.selection_strategy,
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

    # ── Print results ──
    vm = breakdown["verification_metrics"]
    sm = breakdown["selection_metrics"]
    sim = breakdown["simulated_tsr"]
    hist = "GT history" if args.gt_history else "Pred history"

    print(f"\n{'='*65}")
    print(f"Best-of-{args.n_candidates} [{args.selection_strategy}] ({hist})")
    print(f"{'='*65}")
    print(f"  TSR:           {summary['tsr']*100:.1f}%")
    print(f"  Step SR:       {summary['step_sr']*100:.1f}%")
    print(f"  Avg Progress:  {summary['avg_progress']*100:.1f}%")
    print()
    print(f"  --- Selection Quality ---")
    print(f"  Greedy accuracy:   {sm['greedy_accuracy']*100:.1f}%")
    print(f"  Oracle accuracy:   {sm['oracle_accuracy']*100:.1f}% (best of {args.n_candidates})")
    print(f"  Selected accuracy: {sm['selected_accuracy']*100:.1f}% (verifier-picked)")
    print(f"  Retry helped: {sm['retry_helped']} steps")
    print(f"  Retry hurt:   {sm['retry_hurt']} steps")
    print(f"  Net gain:     {sm['net_gain']} steps")
    print(f"  Retry distribution: {sm['retry_distribution']}")
    print()
    print(f"  --- Simulated 6-step TSR ---")
    print(f"  Greedy:   {sim['greedy_6step_TSR']*100:.1f}%")
    print(f"  Selected: {sim['selected_6step_TSR']*100:.1f}%")
    print(f"  Oracle:   {sim['oracle_6step_TSR']*100:.1f}%")
    print()
    print(f"  --- Verifier (across all candidates) ---")
    print(f"  TPR: {vm['TPR']*100:.1f}%  FPR: {vm['FPR']*100:.1f}%  Precision: {vm['precision']*100:.1f}%")
    print()
    print(f"  --- By Task Length ---")
    for bname, metrics in sorted(breakdown.get("task_length_metrics", {}).items()):
        print(f"  {bname:>5s} steps: TSR={metrics['tsr']*100:5.1f}%  "
              f"StepSR={metrics['step_sr']*100:5.1f}%  "
              f"Progress={metrics['avg_progress']*100:5.1f}%  "
              f"(n={metrics['num_episodes']}, success={metrics['num_success']})")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
