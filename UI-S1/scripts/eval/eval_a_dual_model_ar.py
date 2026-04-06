#!/usr/bin/env python3
"""
Eval A: V2+V3 Dual-Model Pipeline Trajectory-Level Baseline (Autoregressive)

Question: What is the TSR when using V2 for action prediction + V3 for grounding
in an autoregressive trajectory evaluation?

Architecture:
  For each trajectory step:
    1. V2 predicts action (function, args, status) using predicted_history
    2. If coordinate-based action, V3 grounds the coordinate using V2's thoughts
    3. Replace V2's coordinate with V3's grounded coordinate
    4. Evaluate composite action against GT
    5. Append V2's thoughts to predicted_history for next step

Conditions evaluated:
  A. V2-only baseline (single model, same as existing AR eval)
  B. V2 action + V3 greedy coordinate (dual model)

Usage:
    # Needs two vLLM servers:
    python scripts/eval/eval_a_dual_model_ar.py \
        --v2_endpoint http://localhost:19816/v1 \
        --v3_endpoint http://localhost:19815/v1

    # Analyze existing results:
    python scripts/eval/eval_a_dual_model_ar.py --analyze_only
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
from collections import defaultdict
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
    ACTION_PREDICTION_USER_PROMPT_QWEN as ACTION_PREDICTION_USER_PROMPT,
)
from evaluator.tool_definitions import normalize_tool_args as _normalize_tool_args


# ---------------------------------------------------------------------------
# Data loading: trajectories from GUI-360 test set
# ---------------------------------------------------------------------------
def load_trajectories(dataset_root: str, max_trajectories: int = 0):
    """Load trajectories from GUI-360 test set.

    Each JSONL file = one trajectory. Yields dicts with:
        trajectory_id, request, domain, category, steps: [...]
    """
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

                            # Must have action_prediction tag
                            if "action_prediction" not in data["step"].get("tags", []):
                                continue

                            # Build image path
                            clean_img = os.path.join(
                                dataset_root, "image", domain, category,
                                data["step"]["screenshot_clean"],
                            )
                            if not os.path.exists(clean_img):
                                continue

                            # Status normalization
                            status = data["step"]["status"]
                            if status == "OVERALL_FINISH":
                                status = "FINISH"
                            elif status == "FINISH":
                                status = "CONTINUE"

                            action = data["step"]["action"]

                            # Skip drag and no-rectangle steps (same as official AR eval)
                            if action.get("function", "") == "drag" or not action.get("rectangle", {}):
                                continue

                            sample_id = f"{trajectory_id}_{line_num}"
                            steps.append({
                                "sample_id": sample_id,
                                "line_num": line_num,
                                "request": data["request"],
                                "screenshot_clean": clean_img,
                                "thought": data["step"]["thought"],
                                "action": action,
                                "status": status,
                                "domain": domain,
                                "category": category,
                            })

                    if steps:
                        count += 1
                        yield {
                            "trajectory_id": trajectory_id,
                            "request": steps[0]["request"],
                            "domain": domain,
                            "category": category,
                            "steps": steps,
                        }
                        if max_trajectories > 0 and count >= max_trajectories:
                            return

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue


# ---------------------------------------------------------------------------
# Action parsing (matches Qwen2.5-VL parse_action)
# ---------------------------------------------------------------------------
def parse_action(response: str):
    """Parse action from model response. Returns (function, args, status) or (None, None, None)."""
    if not response:
        return None, None, None

    # Look for <tool_call>...</tool_call>
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)

    if match:
        try:
            obj = json.loads(match.group(1))
            return obj.get("function"), obj.get("args", {}), obj.get("status")
        except json.JSONDecodeError:
            pass

    # Fallback: try to find JSON with function/args/status
    pattern2 = r'\{\s*"function":\s*"([^"]*)",\s*"args":\s*(\{.*?\}),\s*"status":\s*"([^"]+)"\s*\}'
    match2 = re.search(pattern2, response, re.DOTALL)
    if match2:
        try:
            args = json.loads(match2.group(2))
            return match2.group(1), args, match2.group(3)
        except json.JSONDecodeError:
            return match2.group(1), {}, match2.group(3)

    return None, None, None


def is_coord_in_rect(coord, rect):
    """Check if coordinate [x, y] is inside rectangle {left, top, right, bottom}."""
    if not coord or not rect:
        return False
    try:
        x, y = float(coord[0]), float(coord[1])
        return (rect["left"] <= x <= rect["right"] and
                rect["top"] <= y <= rect["bottom"])
    except (TypeError, ValueError, KeyError, IndexError):
        return False


def compare_actions(pred_function, pred_args, pred_status,
                    gt_function, gt_args, gt_status, gt_rect):
    """Compare predicted action with ground truth. Returns (func_match, args_match, status_match, coord_match)."""
    func_match = pred_function == gt_function if pred_function else False
    status_match = pred_status == gt_status if pred_status else False

    args_match = False
    coord_match = False

    if pred_args and gt_args:
        p_norm = _normalize_tool_args(pred_function or "", pred_args)
        g_norm = _normalize_tool_args(gt_function or "", gt_args)

        # Check coordinate
        p_coord = p_norm.get("coordinate")
        if p_coord and isinstance(p_coord, (list, tuple)) and len(p_coord) >= 2:
            coord_match = is_coord_in_rect(p_coord, gt_rect)

        # Check all args
        all_match = True
        for key in p_norm:
            p_val = p_norm.get(key)
            g_val = g_norm.get(key)

            if key == "coordinate":
                if not coord_match:
                    all_match = False
            elif key in ("start_coordinate", "end_coordinate"):
                continue
            else:
                p_str = str(p_val).lower() if p_val is not None else "none"
                g_str = str(g_val).lower() if g_val is not None else "none"
                if g_str and g_str != "none" and p_str != g_str:
                    all_match = False

        args_match = all_match

    return func_match, args_match, status_match, coord_match


# ---------------------------------------------------------------------------
# Dual-model trajectory evaluation
# ---------------------------------------------------------------------------
def evaluate_trajectory(trajectory, v2_client, v2_model, v3_client, v3_model,
                        stop_on_error=True):
    """Evaluate a single trajectory autoregressively with V2+V3 dual model.

    Returns dict with trajectory-level metrics and per-step results.
    """
    from PIL import Image

    traj_id = trajectory["trajectory_id"]
    steps = trajectory["steps"]
    domain = trajectory["domain"]

    predicted_history = []
    step_results = []
    first_error_step = None
    stopped_early = False

    # Select domain-specific actions
    if domain.lower() == "excel":
        actions_str = SUPPORTED_ACTIONS_EXCEL
    elif domain.lower() == "ppt":
        actions_str = SUPPORTED_ACTIONS_PPT
    else:
        actions_str = SUPPORTED_ACTIONS_WORD

    coord_actions = {"click", "right_click", "double_click"}

    for i, step in enumerate(steps):
        step_num = i + 1
        sample_id = step["sample_id"]
        t0 = time.time()

        try:
            clean_img = step["screenshot_clean"]
            action = step["action"].copy()
            action_args = action.get("args", {}).copy()
            action["args"] = action_args

            resolution = Image.open(clean_img).size

            # Normalize GT args
            gt_rect = action.get("rectangle", {})
            action_args.pop("x", None)
            action_args.pop("y", None)
            if "coordinate_x" in action and action["coordinate_x"]:
                action_args["coordinate"] = [action["coordinate_x"], action["coordinate_y"]]

            gt_function = action.get("function", "")
            gt_args = action_args
            gt_status = step["status"]

            # --- Step 1: V2 action prediction ---
            history_str = "\n".join(predicted_history) if predicted_history else ""
            user_prompt = ACTION_PREDICTION_USER_PROMPT.format(
                instruction=step["request"],
                history=history_str,
                actions=actions_str,
            )

            # Preprocess image for V2
            data_url_v2, orig_wh_v2, resized_wh_v2 = preprocess_image(clean_img)
            messages_v2 = [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url_v2}},
                {"type": "text", "text": user_prompt},
            ]}]

            v2_response = ""
            for retry in range(3):
                try:
                    resp = v2_client.chat.completions.create(
                        model=v2_model, messages=messages_v2,
                        temperature=0.0, max_tokens=4096,
                    )
                    v2_response = resp.choices[0].message.content or ""
                    break
                except Exception as e:
                    if retry == 2:
                        print(f"  V2 failed for {sample_id}: {e}")
                    time.sleep(5)

            # Parse V2 action
            pred_function, pred_args, pred_status = parse_action(v2_response)
            v2_thoughts = v2_response  # Full response as thoughts (matches Qwen model)

            # Transform V2 coordinates to original space
            if pred_args and pred_args.get("coordinate"):
                coord = pred_args["coordinate"]
                if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                    try:
                        pred_args["coordinate"] = transform_coord_to_original(
                            [float(coord[0]), float(coord[1])], orig_wh_v2, resized_wh_v2
                        )
                    except (TypeError, ValueError):
                        pass

            # --- Step 2: V3 grounding (only for coordinate-based actions) ---
            v3_coord = None
            if pred_function in coord_actions and step.get("thought"):
                grounding_text = GROUNDING_PROMPT.format(instruction=step["thought"])
                data_url_v3, orig_wh_v3, resized_wh_v3 = preprocess_image(clean_img)
                messages_v3 = [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url_v3}},
                    {"type": "text", "text": grounding_text},
                ]}]

                for retry in range(3):
                    try:
                        resp_v3 = v3_client.chat.completions.create(
                            model=v3_model, messages=messages_v3,
                            temperature=0.0, max_tokens=512,
                        )
                        v3_text = resp_v3.choices[0].message.content or ""
                        v3_coord = parse_coordinate_response(v3_text)
                        if v3_coord is not None:
                            v3_coord = transform_coord_to_original(v3_coord, orig_wh_v3, resized_wh_v3)
                        break
                    except Exception as e:
                        if retry == 2:
                            print(f"  V3 failed for {sample_id}: {e}")
                        time.sleep(5)

            # --- Evaluate condition A: V2 only ---
            fm_a, am_a, sm_a, cm_a = compare_actions(
                pred_function, pred_args, pred_status,
                gt_function, gt_args, gt_status, gt_rect
            )
            success_a = fm_a and am_a and sm_a

            # --- Evaluate condition B: V2 action + V3 coordinate ---
            if v3_coord is not None and pred_args is not None:
                dual_args = pred_args.copy()
                dual_args["coordinate"] = v3_coord
                fm_b, am_b, sm_b, cm_b = compare_actions(
                    pred_function, dual_args, pred_status,
                    gt_function, gt_args, gt_status, gt_rect
                )
            else:
                # Fallback to V2 if V3 didn't produce a coordinate
                fm_b, am_b, sm_b, cm_b = fm_a, am_a, sm_a, cm_a
            success_b = fm_b and am_b and sm_b

            elapsed = time.time() - t0

            step_results.append({
                "sample_id": sample_id,
                "step_num": step_num,
                "gt_function": gt_function,
                "pred_function": pred_function,
                # Condition A (V2 only)
                "a_success": success_a,
                "a_func_match": fm_a,
                "a_args_match": am_a,
                "a_coord_match": cm_a,
                "a_status_match": sm_a,
                # Condition B (V2+V3)
                "b_success": success_b,
                "b_func_match": fm_b,
                "b_args_match": am_b,
                "b_coord_match": cm_b,
                "b_status_match": sm_b,
                # Debug info
                "v3_coord": v3_coord,
                "execution_time": elapsed,
            })

            # History update (using V2's thoughts)
            if v2_thoughts and v2_thoughts.strip():
                # Truncate long responses to keep history manageable
                thoughts_short = v2_thoughts[:500]
                predicted_history.append(f"Step {step_num}: {thoughts_short}")
            else:
                predicted_history.append(
                    f"Step {step_num}: Performed {pred_function or 'unknown'} action"
                )

            # Check stop condition (using condition B for dual-model)
            if not success_b and first_error_step is None:
                first_error_step = step_num

            status_b = "\u2713" if success_b else "\u2717"
            status_a = "\u2713" if success_a else "\u2717"
            print(f"  [{traj_id}] step {step_num}/{len(steps)}: "
                  f"{pred_function} ({status_b}B) "
                  f"a={status_a} "
                  f"elapsed={elapsed:.1f}s")

            if stop_on_error and not success_b:
                stopped_early = True
                break

        except Exception as e:
            print(f"  [{traj_id}] step {step_num} ERROR: {traceback.format_exc()}")
            step_results.append({
                "sample_id": sample_id,
                "step_num": step_num,
                "a_success": False, "b_success": False,
                "error": str(e),
                "execution_time": time.time() - t0,
            })
            if first_error_step is None:
                first_error_step = step_num
            predicted_history.append(f"Step {step_num}: Error occurred")
            if stop_on_error:
                stopped_early = True
                break

    # Trajectory-level metrics
    n_steps = len(steps)
    n_eval = len(step_results)

    # Condition A metrics
    a_correct = sum(1 for s in step_results if s.get("a_success"))
    a_first_err = None
    for s in step_results:
        if not s.get("a_success"):
            a_first_err = s.get("step_num", 1)
            break
    a_tsr = (a_correct == n_steps and n_eval == n_steps)
    a_progress = (a_first_err - 1) / n_steps if a_first_err else (1.0 if a_correct == n_steps else 0.0)
    a_scattered = a_correct / n_steps if n_steps > 0 else 0

    # Condition B metrics
    b_correct = sum(1 for s in step_results if s.get("b_success"))
    b_tsr = (b_correct == n_steps and n_eval == n_steps)
    b_progress = (first_error_step - 1) / n_steps if first_error_step else (1.0 if b_correct == n_steps else 0.0)
    b_scattered = b_correct / n_steps if n_steps > 0 else 0

    return {
        "trajectory_id": traj_id,
        "domain": domain,
        "category": trajectory["category"],
        "num_steps": n_steps,
        "num_evaluated": n_eval,
        "stopped_early": stopped_early,
        # Condition A
        "a_trajectory_success": a_tsr,
        "a_progress_rate": a_progress,
        "a_scattered_progress": a_scattered,
        # Condition B
        "b_trajectory_success": b_tsr,
        "b_progress_rate": b_progress,
        "b_scattered_progress": b_scattered,
        # Per-step
        "step_results": step_results,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def run_evaluation(args):
    """Run dual-model AR trajectory evaluation."""
    from openai import OpenAI

    dataset_root = args.dataset_root
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "trajectory_results.jsonl"

    # Resume support
    completed_ids = set()
    if results_path.exists() and not args.overwrite:
        with open(results_path) as f:
            for line in f:
                completed_ids.add(json.loads(line)["trajectory_id"])
        print(f"Resuming: {len(completed_ids)} trajectories already completed")

    # Load trajectories
    print(f"Loading trajectories from {dataset_root}...")
    trajectories = list(load_trajectories(dataset_root, args.max_trajectories))
    print(f"Loaded {len(trajectories)} trajectories, "
          f"{sum(len(t['steps']) for t in trajectories)} total steps")

    pending = [t for t in trajectories if t["trajectory_id"] not in completed_ids]
    print(f"Pending: {len(pending)} trajectories")

    if not pending:
        print("All trajectories already evaluated. Use --overwrite to re-run.")
        analyze_results(str(output_dir))
        return

    # Create clients
    v2_client = OpenAI(api_key="EMPTY", base_url=args.v2_endpoint, timeout=600)
    v3_client = OpenAI(api_key="EMPTY", base_url=args.v3_endpoint, timeout=600)

    t0 = time.time()
    processed = 0

    # Each trajectory is evaluated sequentially within itself,
    # but multiple trajectories run in parallel
    def process_one(traj):
        return evaluate_trajectory(
            traj, v2_client, args.v2_model, v3_client, args.v3_model,
            stop_on_error=args.stop_on_error,
        )

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_one, t): t for t in pending}

        for future in as_completed(futures):
            traj = futures[future]
            try:
                result = future.result(timeout=1800)  # 30 min per trajectory
            except Exception as e:
                print(f"Trajectory {traj['trajectory_id']} FAILED: {e}")
                result = {
                    "trajectory_id": traj["trajectory_id"],
                    "domain": traj["domain"],
                    "category": traj["category"],
                    "num_steps": len(traj["steps"]),
                    "num_evaluated": 0,
                    "stopped_early": True,
                    "a_trajectory_success": False,
                    "a_progress_rate": 0.0,
                    "a_scattered_progress": 0.0,
                    "b_trajectory_success": False,
                    "b_progress_rate": 0.0,
                    "b_scattered_progress": 0.0,
                    "step_results": [],
                    "error": str(e),
                }

            with open(results_path, "a") as f:
                f.write(json.dumps(result, default=str) + "\n")

            processed += 1
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0

            # Running stats
            all_results = []
            with open(results_path) as f:
                for line in f:
                    all_results.append(json.loads(line))
            n = len(all_results)
            a_tsr = sum(1 for r in all_results if r.get("a_trajectory_success")) / n
            b_tsr = sum(1 for r in all_results if r.get("b_trajectory_success")) / n

            print(f"\n{'='*60}")
            print(f"Progress: {processed}/{len(pending)} "
                  f"({rate:.2f} traj/s, ETA {(len(pending)-processed)/rate/60:.0f}min)")
            print(f"Running TSR: A(V2-only)={a_tsr:.1%}, B(V2+V3)={b_tsr:.1%}  [n={n}]")
            print(f"{'='*60}\n")

    print(f"\nDone! Results saved to {results_path}")
    analyze_results(str(output_dir))


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze_results(output_dir: str):
    """Analyze dual-model AR trajectory results."""
    results_path = Path(output_dir) / "trajectory_results.jsonl"
    if not results_path.exists():
        print(f"No results at {results_path}")
        return

    results = [json.loads(line) for line in open(results_path)]
    n = len(results)

    print("\n" + "=" * 70)
    print("  Eval A: V2+V3 Dual-Model AR Trajectory Evaluation")
    print("=" * 70)
    print(f"\n  Total trajectories: {n}")
    print(f"  Total steps: {sum(r['num_steps'] for r in results)}")

    for cond, label in [("a", "V2 only"), ("b", "V2 + V3 greedy")]:
        tsr = sum(1 for r in results if r.get(f"{cond}_trajectory_success")) / n
        avg_prog = np.mean([r.get(f"{cond}_progress_rate", 0) for r in results])
        avg_scat = np.mean([r.get(f"{cond}_scattered_progress", 0) for r in results])

        print(f"\n  [{label}]")
        print(f"    TSR:                {tsr:.1%} ({sum(1 for r in results if r.get(f'{cond}_trajectory_success'))}/{n})")
        print(f"    Avg Progress Rate:  {avg_prog:.3f}")
        print(f"    Avg Scattered Prog: {avg_scat:.3f}")

    # Domain breakdown
    print(f"\n  --- Domain Breakdown ---")
    domains = defaultdict(list)
    for r in results:
        domains[r["domain"]].append(r)

    print(f"  {'Domain':>8s} {'N':>5s}  {'A_TSR':>7s} {'B_TSR':>7s} {'A_Prog':>7s} {'B_Prog':>7s}")
    for domain in sorted(domains):
        drs = domains[domain]
        dn = len(drs)
        a_tsr = sum(1 for r in drs if r.get("a_trajectory_success")) / dn
        b_tsr = sum(1 for r in drs if r.get("b_trajectory_success")) / dn
        a_prog = np.mean([r.get("a_progress_rate", 0) for r in drs])
        b_prog = np.mean([r.get("b_progress_rate", 0) for r in drs])
        print(f"  {domain:>8s} {dn:>5d}  {a_tsr:>7.1%} {b_tsr:>7.1%} {a_prog:>7.3f} {b_prog:>7.3f}")

    # Length buckets
    print(f"\n  --- Trajectory Length Buckets ---")
    buckets = {"short(1-5)": (1, 5), "medium(6-15)": (6, 15), "long(16+)": (16, 999)}
    print(f"  {'Bucket':>15s} {'N':>5s}  {'A_TSR':>7s} {'B_TSR':>7s} {'B_Prog':>7s}")
    for bname, (lo, hi) in buckets.items():
        brs = [r for r in results if lo <= r["num_steps"] <= hi]
        if not brs:
            continue
        bn = len(brs)
        a_tsr = sum(1 for r in brs if r.get("a_trajectory_success")) / bn
        b_tsr = sum(1 for r in brs if r.get("b_trajectory_success")) / bn
        b_prog = np.mean([r.get("b_progress_rate", 0) for r in brs])
        print(f"  {bname:>15s} {bn:>5d}  {a_tsr:>7.1%} {b_tsr:>7.1%} {b_prog:>7.3f}")

    # Step-level coord_match analysis
    all_steps = []
    for r in results:
        for s in r.get("step_results", []):
            all_steps.append(s)

    if all_steps:
        a_cm = sum(1 for s in all_steps if s.get("a_coord_match")) / len(all_steps)
        b_cm = sum(1 for s in all_steps if s.get("b_coord_match")) / len(all_steps)
        a_fm = sum(1 for s in all_steps if s.get("a_func_match")) / len(all_steps)
        b_fm = sum(1 for s in all_steps if s.get("b_func_match")) / len(all_steps)
        print(f"\n  --- Step-Level Metrics ({len(all_steps)} evaluated steps) ---")
        print(f"    func_match:  A={a_fm:.1%}, B={b_fm:.1%}")
        print(f"    coord_match: A={a_cm:.1%}, B={b_cm:.1%}")
        print(f"    coord lift:  {b_cm - a_cm:+.1%}")

    # Save summary
    summary = {
        "n_trajectories": n,
        "n_steps_total": sum(r["num_steps"] for r in results),
        "a_tsr": sum(1 for r in results if r.get("a_trajectory_success")) / n,
        "b_tsr": sum(1 for r in results if r.get("b_trajectory_success")) / n,
        "a_avg_progress": float(np.mean([r.get("a_progress_rate", 0) for r in results])),
        "b_avg_progress": float(np.mean([r.get("b_progress_rate", 0) for r in results])),
    }
    summary_path = Path(output_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")

    # Decision
    b_tsr_val = summary["b_tsr"]
    print(f"\n  === DECISION ===")
    if b_tsr_val >= 0.25:
        print(f"  B_TSR = {b_tsr_val:.1%} >= 25% → Proceed with Phase 2 RL as planned")
    elif b_tsr_val >= 0.20:
        print(f"  B_TSR = {b_tsr_val:.1%} (20-25%) → Marginal, review pipeline details")
    else:
        print(f"  B_TSR = {b_tsr_val:.1%} < 20% → Unexpected. Diagnose pipeline issues first.")


def main():
    parser = argparse.ArgumentParser(description="Eval A: Dual-Model AR Trajectory Baseline")
    parser.add_argument("--v2_endpoint", default="http://localhost:19816/v1",
                        help="V2 (action) vLLM endpoint")
    parser.add_argument("--v3_endpoint", default="http://localhost:19815/v1",
                        help="V3 (grounding) vLLM endpoint")
    parser.add_argument("--v2_model", default="sft_v2",
                        help="V2 model name in vLLM")
    parser.add_argument("--v3_model", default="sft_v3",
                        help="V3 model name in vLLM")
    parser.add_argument("--dataset_root",
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/GUI-360/test",
                        help="GUI-360 test dataset root")
    parser.add_argument("--output_dir", default="outputs/eval_a",
                        help="Output directory")
    parser.add_argument("--num_workers", type=int, default=3,
                        help="Number of parallel trajectory evaluations")
    parser.add_argument("--max_trajectories", type=int, default=0,
                        help="Max trajectories to evaluate (0=all)")
    parser.add_argument("--stop_on_error", action="store_true", default=True,
                        help="Stop trajectory on first error")
    parser.add_argument("--no_stop_on_error", dest="stop_on_error", action="store_false")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing results")
    parser.add_argument("--analyze_only", action="store_true",
                        help="Only analyze existing results")

    args = parser.parse_args()

    if args.analyze_only:
        analyze_results(args.output_dir)
    else:
        run_evaluation(args)


if __name__ == "__main__":
    main()
