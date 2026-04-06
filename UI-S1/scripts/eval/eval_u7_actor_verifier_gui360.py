#!/usr/bin/env python3
"""
Exp U7: Actor-Verifier AR Evaluation on GUI-360

Architecture per step:
  1. [Actor] V2 greedy action prediction (temp=0)
  2. [V3 Grounding] Parallel coordinate prediction (same as D1/Eval A)
  3. [Verifier] V2 judges (screenshot, instruction, pred_action) -> PASS/FAIL
  4. If PASS -> use Actor's greedy action (+ V3 coord replacement)
  5. If FAIL -> resample K=5 (temp=0.6), majority vote on function type
     -> replace V3 coord on voted action
  6. Evaluate with stop-on-error

Compares against Eval A (V2+V3, no verifier) and D1 (Observer).

Usage:
    python scripts/eval/eval_u7_actor_verifier_gui360.py \
        --v2_endpoint http://localhost:19816/v1 \
        --v3_endpoint http://localhost:19815/v1 \
        --K 5 --temperature 0.6 --max_workers 8
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "train_GUI_360" / "GUI-360-eval"))

from scripts.exp1.grounding_utils import (
    preprocess_image,
    parse_coordinate_response,
    transform_coord_to_original,
    GROUNDING_PROMPT,
)
from prompts.prompt_action_prediction import (
    SUPPORTED_ACTIONS_WORD,
    SUPPORTED_ACTIONS_EXCEL,
    SUPPORTED_ACTIONS_PPT,
)
from evaluator.tool_definitions import normalize_tool_args as _normalize_tool_args


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
ACTION_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

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

Only **ONE** action should be taken at a time. If the instruction could apply to multiple elements, choose the most relevant one based on the context provided by the screenshot and previous actions."""

VERIFIER_PROMPT = """You are a verification agent for a desktop GUI automation task.
The user's goal is: {instruction}
The actor agent predicted this action: {action_json}

Evaluate whether this action is correct:
1. Is the action TYPE (click/type/scroll/etc.) appropriate for the current screen?
2. Is the TARGET (coordinate/text/element) reasonable given what's visible?
3. Does this action make progress toward the goal?

Output a JSON: {{"verdict": "PASS" or "FAIL", "reason": "brief explanation"}}"""


# ---------------------------------------------------------------------------
# Data loading (same as D1)
# ---------------------------------------------------------------------------
def load_trajectories(dataset_root, max_trajectories=0):
    data_path = os.path.join(dataset_root, "data")
    count = 0
    for domain in sorted(os.listdir(data_path)):
        domain_path = os.path.join(data_path, domain)
        if not os.path.isdir(domain_path):
            continue
        for category in sorted(os.listdir(domain_path)):
            success_path = os.path.join(domain_path, category, "success")
            if not os.path.isdir(success_path):
                continue
            for jsonl_file in sorted(os.listdir(success_path)):
                if not jsonl_file.endswith(".jsonl"):
                    continue
                file_path = os.path.join(success_path, jsonl_file)
                file_stem = os.path.splitext(jsonl_file)[0]
                trajectory_id = f"{domain}_{category}_{file_stem}"
                try:
                    steps = []
                    with open(file_path, "r") as f:
                        for line_num, line in enumerate(f, 1):
                            if not line.strip():
                                continue
                            data = json.loads(line.strip())
                            if "action_prediction" not in data["step"].get("tags", []):
                                continue
                            clean_img = os.path.join(
                                dataset_root, "image", domain, category,
                                data["step"]["screenshot_clean"],
                            )
                            if not os.path.exists(clean_img):
                                continue
                            status = data["step"]["status"]
                            if status == "OVERALL_FINISH":
                                status = "FINISH"
                            elif status == "FINISH":
                                status = "CONTINUE"
                            action = data["step"]["action"]
                            if action.get("function", "") == "drag" or not action.get("rectangle", {}):
                                continue
                            sample_id = f"{trajectory_id}_{line_num}"
                            steps.append({
                                "sample_id": sample_id, "line_num": line_num,
                                "request": data["request"],
                                "screenshot_clean": clean_img,
                                "thought": data["step"]["thought"],
                                "action": action, "status": status,
                                "domain": domain, "category": category,
                            })
                    if steps:
                        count += 1
                        yield {
                            "trajectory_id": trajectory_id,
                            "request": steps[0]["request"],
                            "domain": domain, "category": category,
                            "steps": steps,
                        }
                        if max_trajectories > 0 and count >= max_trajectories:
                            return
                except Exception:
                    continue


# ---------------------------------------------------------------------------
# Action parsing & comparison (same as D1)
# ---------------------------------------------------------------------------
def parse_action(response):
    if not response:
        return None, None, None
    match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', response, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            obj = json.loads(match.group(1))
            return obj.get("function"), obj.get("args", {}), obj.get("status")
        except json.JSONDecodeError:
            pass
    match2 = re.search(
        r'\{\s*"function":\s*"([^"]*)",\s*"args":\s*(\{.*?\}),\s*"status":\s*"([^"]+)"\s*\}',
        response, re.DOTALL,
    )
    if match2:
        try:
            return match2.group(1), json.loads(match2.group(2)), match2.group(3)
        except json.JSONDecodeError:
            return match2.group(1), {}, match2.group(3)
    return None, None, None


def is_coord_in_rect(coord, rect):
    if not coord or not rect:
        return False
    try:
        x, y = float(coord[0]), float(coord[1])
        return rect["left"] <= x <= rect["right"] and rect["top"] <= y <= rect["bottom"]
    except (TypeError, ValueError, KeyError, IndexError):
        return False


def compare_actions(pred_function, pred_args, pred_status,
                    gt_function, gt_args, gt_status, gt_rect):
    func_match = pred_function == gt_function if pred_function else False
    status_match = pred_status == gt_status if pred_status else False
    args_match = coord_match = False
    if pred_args and gt_args:
        p_norm = _normalize_tool_args(pred_function or "", pred_args)
        g_norm = _normalize_tool_args(gt_function or "", gt_args)
        p_coord = p_norm.get("coordinate")
        if p_coord and isinstance(p_coord, (list, tuple)) and len(p_coord) >= 2:
            coord_match = is_coord_in_rect(p_coord, gt_rect)
        all_match = True
        for key in p_norm:
            if key == "coordinate":
                if not coord_match:
                    all_match = False
            elif key in ("start_coordinate", "end_coordinate"):
                continue
            else:
                p_str = str(p_norm.get(key)).lower() if p_norm.get(key) is not None else "none"
                g_str = str(g_norm.get(key)).lower() if g_norm.get(key) is not None else "none"
                if g_str and g_str != "none" and p_str != g_str:
                    all_match = False
        args_match = all_match
    return func_match, args_match, status_match, coord_match


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def _call_v2(client, model, messages, max_tokens=4096, temperature=0.0):
    """Single V2 call with retry."""
    for retry in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            if retry == 2:
                return ""
            time.sleep(2)
    return ""


def _call_v3(client, model, messages):
    """Single V3 call with retry."""
    for retry in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.0, max_tokens=512,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            if retry == 2:
                return ""
            time.sleep(2)
    return ""


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------
def parse_verifier_response(response_text):
    """Parse verifier response to extract verdict and reason.
    Defaults to PASS on failure (conservative: trust the actor).
    """
    # Try JSON parse
    try:
        parsed = json.loads(response_text)
        verdict = parsed.get("verdict", "PASS").upper().strip()
        reason = parsed.get("reason", "")
        if verdict in ("PASS", "FAIL"):
            return verdict, reason
    except (json.JSONDecodeError, AttributeError):
        pass

    # Try extracting JSON from within the response
    json_match = re.search(
        r'\{[^{}]*"verdict"\s*:\s*"(PASS|FAIL)"[^{}]*\}',
        response_text, re.IGNORECASE,
    )
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            verdict = parsed.get("verdict", "PASS").upper().strip()
            reason = parsed.get("reason", "")
            if verdict in ("PASS", "FAIL"):
                return verdict, reason
        except (json.JSONDecodeError, AttributeError):
            pass

    # Fallback: regex search for PASS or FAIL keywords
    if re.search(r'\bFAIL\b', response_text, re.IGNORECASE):
        return "FAIL", response_text.strip()[:200]
    if re.search(r'\bPASS\b', response_text, re.IGNORECASE):
        return "PASS", response_text.strip()[:200]

    # Default: trust the actor
    return "PASS", "verifier_parse_fallback"


def call_verifier(v2_client, v2_model, data_url, instruction, action_json_str):
    """Call verifier agent with screenshot and predicted action."""
    prompt = VERIFIER_PROMPT.format(
        instruction=instruction,
        action_json=action_json_str,
    )
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": prompt},
    ]}]
    response = _call_v2(v2_client, v2_model, messages, max_tokens=512)
    if not response:
        return "PASS", "verifier_call_failed"
    return parse_verifier_response(response)


# ---------------------------------------------------------------------------
# Resample & majority vote
# ---------------------------------------------------------------------------
def resample_and_vote(v2_client, v2_model, messages, K=5, temperature=0.6):
    """Generate K samples and majority vote on function type.

    Returns:
        (function, args, status, raw_response, agreement)
    """
    samples = []
    for _ in range(K):
        raw = _call_v2(v2_client, v2_model, messages, temperature=temperature)
        func, args, status = parse_action(raw)
        samples.append((func, args, status, raw))

    # Majority vote on function
    func_counts = Counter(s[0] for s in samples if s[0])
    if not func_counts:
        if samples:
            return samples[0] + (0.0,)
        return None, None, None, "", 0.0

    voted_func = func_counts.most_common(1)[0][0]
    agreement = func_counts[voted_func] / len(samples)

    # Pick first sample matching voted function
    for s in samples:
        if s[0] == voted_func:
            return s + (agreement,)
    return samples[0] + (0.0,)


# ---------------------------------------------------------------------------
# Trajectory evaluation
# ---------------------------------------------------------------------------
def evaluate_trajectory(trajectory, v2_client, v2_model, v3_client, v3_model,
                        K=5, temperature=0.6, stop_on_error=True):
    traj_id = trajectory["trajectory_id"]
    steps = trajectory["steps"]
    domain = trajectory["domain"]

    step_results = []
    stopped_early = False
    action_history = []

    actions_str = {"word": SUPPORTED_ACTIONS_WORD, "excel": SUPPORTED_ACTIONS_EXCEL,
                   "ppt": SUPPORTED_ACTIONS_PPT}.get(domain, SUPPORTED_ACTIONS_WORD)
    coord_actions = {"click", "right_click", "double_click"}

    for step_idx, step in enumerate(steps):
        t0 = time.time()
        sample_id = step["sample_id"]
        step_num = step_idx + 1
        clean_img = step["screenshot_clean"]

        action = step["action"]
        action_args = dict(action.get("args", {}))
        gt_rect = action.get("rectangle", {})
        action_args.pop("x", None)
        action_args.pop("y", None)
        if "coordinate_x" in action and action["coordinate_x"]:
            action_args["coordinate"] = [action["coordinate_x"], action["coordinate_y"]]

        gt_function = action.get("function", "")
        gt_args = action_args
        gt_status = step["status"]

        if stopped_early:
            step_results.append({
                "sample_id": sample_id, "step_num": step_num,
                "gt_function": gt_function,
                "pred_function": None, "success": False,
                "func_match": False, "args_match": False,
                "coord_match": False, "status_match": False,
                "verdict": None, "resampled": False,
                "agreement": None, "v3_coord": None,
                "execution_time": 0,
            })
            continue

        data_url, orig_wh, resized_wh = preprocess_image(clean_img)

        # --- Build actor prompt ---
        history_str = "\n".join(action_history) if action_history else "(none)"
        user_prompt = ACTION_PROMPT.format(
            instruction=step["request"], history=history_str, actions=actions_str)
        actor_messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": user_prompt},
        ]}]

        # --- Parallel: Actor greedy + V3 grounding ---
        with ThreadPoolExecutor(max_workers=2) as pool:
            actor_future = pool.submit(_call_v2, v2_client, v2_model, actor_messages)

            v3_future = None
            if step.get("thought"):
                grounding_text = GROUNDING_PROMPT.format(instruction=step["thought"])
                v3_msgs = [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": grounding_text},
                ]}]
                v3_future = pool.submit(_call_v3, v3_client, v3_model, v3_msgs)

            actor_response = actor_future.result()
            v3_coord = None
            if v3_future:
                v3_text = v3_future.result()
                v3_coord = parse_coordinate_response(v3_text)
                if v3_coord is not None:
                    v3_coord = transform_coord_to_original(v3_coord, orig_wh, resized_wh)

        pred_function, pred_args, pred_status = parse_action(actor_response)

        # Transform V2 coordinates to original space
        if pred_args and pred_args.get("coordinate"):
            coord = pred_args["coordinate"]
            if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                try:
                    pred_args["coordinate"] = transform_coord_to_original(
                        [float(coord[0]), float(coord[1])], orig_wh, resized_wh)
                except (TypeError, ValueError):
                    pass

        # --- Verifier ---
        action_for_verify = {}
        if pred_function:
            action_for_verify = {"function": pred_function, "args": pred_args or {}, "status": pred_status or "CONTINUE"}
        action_json_str = json.dumps(action_for_verify, ensure_ascii=False)
        verdict, reason = call_verifier(v2_client, v2_model, data_url, step["request"], action_json_str)

        resampled = False
        agreement = None

        # --- If FAIL: resample and majority vote ---
        if verdict == "FAIL":
            resampled = True
            r_func, r_args, r_status, r_raw, r_agreement = resample_and_vote(
                v2_client, v2_model, actor_messages, K=K, temperature=temperature)
            agreement = r_agreement

            if r_func is not None:
                pred_function = r_func
                pred_args = r_args
                pred_status = r_status
                # Transform resampled V2 coordinates
                if pred_args and pred_args.get("coordinate"):
                    coord = pred_args["coordinate"]
                    if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                        try:
                            pred_args["coordinate"] = transform_coord_to_original(
                                [float(coord[0]), float(coord[1])], orig_wh, resized_wh)
                        except (TypeError, ValueError):
                            pass

        # --- Replace with V3 coordinate if available ---
        if v3_coord is not None and pred_args is not None and pred_function in coord_actions:
            eval_args = pred_args.copy()
            eval_args["coordinate"] = v3_coord
        else:
            eval_args = pred_args

        fm, am, sm, cm = compare_actions(
            pred_function, eval_args, pred_status,
            gt_function, gt_args, gt_status, gt_rect)
        success = fm and am and sm

        elapsed = time.time() - t0
        step_results.append({
            "sample_id": sample_id, "step_num": step_num,
            "gt_function": gt_function, "pred_function": pred_function,
            "success": success, "func_match": fm, "args_match": am,
            "coord_match": cm, "status_match": sm,
            "verdict": verdict, "resampled": resampled,
            "agreement": agreement, "v3_coord": v3_coord,
            "execution_time": elapsed,
        })

        # Update action history for subsequent steps
        if pred_function:
            hist_entry = f"Step {step_num}: {pred_function}"
            if pred_args and pred_args.get("coordinate"):
                try:
                    c = pred_args["coordinate"]
                    hist_entry += f"(coordinate=[{float(c[0]):.0f}, {float(c[1]):.0f}])"
                except (TypeError, ValueError, IndexError):
                    pass
            action_history.append(hist_entry)

        sym = "O" if success else "X"
        v_sym = "P" if verdict == "PASS" else "F"
        r_sym = "R" if resampled else ""
        print(f"  S{step_num}:{sym}{v_sym}{r_sym}({pred_function}) ", end="", flush=True)

        if stop_on_error and not success:
            stopped_early = True

    total_steps = len(steps)
    correct = sum(1 for s in step_results if s["success"])
    traj_success = (correct == total_steps)
    progress = correct / total_steps if total_steps > 0 else 0

    return {
        "trajectory_id": traj_id, "domain": domain, "category": trajectory["category"],
        "num_steps": total_steps,
        "trajectory_success": traj_success,
        "progress_rate": progress,
        "step_results": step_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_evaluation(args):
    from openai import OpenAI

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "actor_verifier_results.jsonl"

    completed = set()
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                d = json.loads(line)
                completed.add(d["trajectory_id"])
        print(f"Resuming: {len(completed)} trajectories already complete")

    dataset_root = str(PROJECT_ROOT / "datasets" / "GUI-360" / "test")
    print(f"Loading trajectories from {dataset_root}")
    trajectories = list(load_trajectories(dataset_root, max_trajectories=args.max_trajectories))
    remaining = [t for t in trajectories if t["trajectory_id"] not in completed]
    print(f"Loaded {len(trajectories)} total, {len(remaining)} remaining")

    v2_client = OpenAI(base_url=args.v2_endpoint, api_key="none")
    v3_client = OpenAI(base_url=args.v3_endpoint, api_key="none")
    v2_model = args.v2_model or v2_client.models.list().data[0].id
    v3_model = args.v3_model or v3_client.models.list().data[0].id
    print(f"V2: {v2_model}, V3: {v3_model}")
    print(f"K={args.K}, temperature={args.temperature}")

    n_done = len(completed)
    n_total = len(trajectories)
    t_start = time.time()

    def _eval_one(traj):
        return evaluate_trajectory(
            traj, v2_client, v2_model, v3_client, v3_model,
            K=args.K, temperature=args.temperature,
        )

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_eval_one, t): t for t in remaining}
        for future in as_completed(futures):
            traj = futures[future]
            n_done += 1
            try:
                result = future.result()
                with open(results_path, "a") as f:
                    f.write(json.dumps(result) + "\n")

                elapsed = time.time() - t_start
                rate = (n_done - len(completed)) / elapsed * 3600
                status = "OK" if result["trajectory_success"] else "FAIL"
                print(f"\n[{n_done}/{n_total}] {traj['trajectory_id']} "
                      f"-> {status} ({result['progress_rate']:.0%}) [{rate:.0f}/hr]")
            except Exception as e:
                print(f"\n[{n_done}/{n_total}] {traj['trajectory_id']} ERROR: {e}")

    print(f"\nDone. Results at {results_path}")


def analyze_results(results_path=None, output_dir=None):
    if results_path is None:
        results_path = Path(output_dir) / "actor_verifier_results.jsonl" if output_dir else \
            PROJECT_ROOT / "outputs" / "eval_u7_gui360" / "actor_verifier_results.jsonl"
    results_path = Path(results_path)

    results = []
    with open(results_path) as f:
        for line in f:
            results.append(json.loads(line))

    # Load Eval A results for comparison
    eval_a_path = PROJECT_ROOT / "outputs" / "eval_a" / "trajectory_results.jsonl"
    eval_a = {}
    if eval_a_path.exists():
        with open(eval_a_path) as f:
            for line in f:
                d = json.loads(line)
                eval_a[d["trajectory_id"]] = d

    n = len(results)
    print(f"\n{'='*70}")
    print(f"  Exp U7: Actor-Verifier AR Results ({n} trajectories)")
    print(f"{'='*70}")

    u7_tsr = np.mean([r["trajectory_success"] for r in results])
    u7_prog = np.mean([r["progress_rate"] for r in results])

    # Match with Eval A
    matched = [r for r in results if r["trajectory_id"] in eval_a]
    if matched:
        a_tsr = np.mean([eval_a[r["trajectory_id"]]["b_trajectory_success"] for r in matched])
        a_prog = np.mean([eval_a[r["trajectory_id"]]["b_progress_rate"] for r in matched])
    else:
        a_tsr = a_prog = 0

    print(f"\n  {'Condition':<35s} {'TSR':>7s} {'Avg Prog':>9s}")
    print(f"  {'-'*35} {'-'*7} {'-'*9}")
    print(f"  {'Eval A: V2+V3 (baseline)':.<35s} {a_tsr:>7.1%} {a_prog:>9.3f}")
    print(f"  {'U7: Actor-Verifier':.<35s} {u7_tsr:>7.1%} {u7_prog:>9.3f}")
    if matched:
        print(f"  {'Delta':.<35s} {u7_tsr-a_tsr:>+7.1%} {u7_prog-a_prog:>+9.3f}")

    # --- Verifier statistics ---
    all_steps = [s for r in results for s in r["step_results"]]
    evaluated_steps = [s for s in all_steps if s.get("verdict") is not None]
    total_evaluated = len(evaluated_steps)
    total_pass = sum(1 for s in evaluated_steps if s["verdict"] == "PASS")
    total_fail = sum(1 for s in evaluated_steps if s["verdict"] == "FAIL")
    total_resampled = sum(1 for s in evaluated_steps if s["resampled"])

    pass_correct = sum(1 for s in evaluated_steps if s["verdict"] == "PASS" and s["success"])
    fail_correct = sum(1 for s in evaluated_steps if s["verdict"] == "FAIL" and s["success"])
    pass_acc = pass_correct / total_pass if total_pass > 0 else 0
    fail_acc = fail_correct / total_fail if total_fail > 0 else 0
    overall_step_acc = sum(1 for s in evaluated_steps if s["success"]) / total_evaluated if total_evaluated > 0 else 0

    resample_agreements = [s["agreement"] for s in evaluated_steps if s["resampled"] and s["agreement"] is not None]
    avg_agreement = np.mean(resample_agreements) if resample_agreements else 0

    print(f"\n  Verifier Statistics:")
    print(f"    Total steps evaluated:  {total_evaluated}")
    print(f"    PASS: {total_pass} ({total_pass/total_evaluated:.1%})")
    print(f"    FAIL: {total_fail} ({total_fail/total_evaluated:.1%})")
    print(f"    PASS accuracy:  {pass_acc:.1%} ({pass_correct}/{total_pass})")
    print(f"    FAIL accuracy:  {fail_acc:.1%} ({fail_correct}/{total_fail})")
    print(f"    Overall step acc: {overall_step_acc:.1%}")
    print(f"    Resampled steps: {total_resampled}")
    print(f"    Avg resample agreement: {avg_agreement:.3f}")

    # --- Per-domain ---
    print(f"\n  Per-domain TSR:")
    by_domain = defaultdict(list)
    for r in results:
        by_domain[r["domain"]].append(r)

    print(f"  {'Domain':<10s} {'N':>5s} {'A_TSR':>7s} {'U7_TSR':>7s} {'Delta':>7s}")
    for domain in sorted(by_domain.keys()):
        subset = by_domain[domain]
        u7 = np.mean([r["trajectory_success"] for r in subset])
        m = [r for r in subset if r["trajectory_id"] in eval_a]
        a = np.mean([eval_a[r["trajectory_id"]]["b_trajectory_success"] for r in m]) if m else 0
        print(f"  {domain:<10s} {len(subset):>5d} {a:>7.1%} {u7:>7.1%} {u7-a:>+7.1%}")

    # --- By length bucket ---
    print(f"\n  By trajectory length:")
    for lo, hi, label in [(1, 3, "Short"), (4, 7, "Med"), (8, 15, "Long"), (16, 100, "VLong")]:
        subset = [r for r in results if lo <= r["num_steps"] <= hi]
        if not subset:
            continue
        u7 = np.mean([r["trajectory_success"] for r in subset])
        m = [r for r in subset if r["trajectory_id"] in eval_a]
        a = np.mean([eval_a[r["trajectory_id"]]["b_trajectory_success"] for r in m]) if m else 0
        print(f"  {label:<10s} {len(subset):>5d} {a:>7.1%} {u7:>7.1%} {u7-a:>+7.1%}")

    # --- Head-to-head ---
    if matched:
        both = sum(1 for r in matched if r["trajectory_success"] and eval_a[r["trajectory_id"]]["b_trajectory_success"])
        a_only = sum(1 for r in matched if not r["trajectory_success"] and eval_a[r["trajectory_id"]]["b_trajectory_success"])
        u7_only = sum(1 for r in matched if r["trajectory_success"] and not eval_a[r["trajectory_id"]]["b_trajectory_success"])
        neither = sum(1 for r in matched if not r["trajectory_success"] and not eval_a[r["trajectory_id"]]["b_trajectory_success"])
        print(f"\n  Head-to-head ({len(matched)} matched):")
        print(f"    Both OK:              {both:>5d}")
        print(f"    Eval A only:          {a_only:>5d}")
        print(f"    U7 wins:              {u7_only:>5d}")
        print(f"    Neither:              {neither:>5d}")

    # --- Step-position-weighted accuracy (Analysis A method) ---
    print(f"\n  Step-position analysis:")
    step_pos_data = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        n_steps = r["num_steps"]
        for s in r["step_results"]:
            if s.get("verdict") is None:
                continue
            pos = s["step_num"]
            weight = (n_steps - pos + 1) / n_steps  # early steps weigh more
            step_pos_data[pos]["total"] += 1
            if s["success"]:
                step_pos_data[pos]["correct"] += 1

    for pos in sorted(step_pos_data.keys())[:10]:
        d = step_pos_data[pos]
        acc = d["correct"] / d["total"] if d["total"] > 0 else 0
        print(f"    Step {pos}: {acc:.1%} ({d['correct']}/{d['total']})")

    # --- Save summary ---
    summary = {
        "experiment": "U7_actor_verifier_gui360",
        "n_trajectories": n,
        "u7_tsr": float(u7_tsr),
        "u7_avg_progress": float(u7_prog),
        "eval_a_tsr": float(a_tsr) if matched else None,
        "delta_tsr": float(u7_tsr - a_tsr) if matched else None,
        "total_steps_evaluated": total_evaluated,
        "verifier_pass": total_pass,
        "verifier_fail": total_fail,
        "verifier_fail_rate": total_fail / total_evaluated if total_evaluated > 0 else 0,
        "pass_accuracy": float(pass_acc),
        "fail_accuracy": float(fail_acc),
        "overall_step_accuracy": float(overall_step_acc),
        "avg_resample_agreement": float(avg_agreement),
        "per_domain": {
            domain: {
                "n": len(subset),
                "tsr": float(np.mean([r["trajectory_success"] for r in subset])),
            }
            for domain, subset in by_domain.items()
        },
    }
    summary_path = results_path.parent / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Exp U7: Actor-Verifier AR on GUI-360")
    parser.add_argument("--v2_endpoint", default="http://localhost:19816/v1")
    parser.add_argument("--v3_endpoint", default="http://localhost:19815/v1")
    parser.add_argument("--v2_model", default=None, help="V2 model name (auto-detect if not set)")
    parser.add_argument("--v3_model", default=None, help="V3 model name (auto-detect if not set)")
    parser.add_argument("--K", type=int, default=5, help="Resample count on FAIL")
    parser.add_argument("--temperature", type=float, default=0.6, help="Resample temperature")
    parser.add_argument("--max_trajectories", type=int, default=0, help="0 = all")
    parser.add_argument("--max_workers", type=int, default=8, help="Parallel trajectory workers")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--analyze_only", action="store_true")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(PROJECT_ROOT / "outputs" / "eval_u7_gui360")

    if args.analyze_only:
        analyze_results(output_dir=args.output_dir)
    else:
        run_evaluation(args)
        analyze_results(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
