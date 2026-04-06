"""Cognitive Interference Hypothesis — vLLM-based inference (fast).

Uses vLLM OpenAI-compatible API with ThreadPoolExecutor for massive parallelism.
No hidden states needed → vLLM is much faster than HF Transformers.

Phase 1: Condition C — AR greedy evaluation (parallel across trajectories)
Phase 2: Conditions A, B, D — step-level parallel evaluation
"""

import argparse
import base64
import json
import os
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO

from openai import OpenAI
from PIL import Image
from tqdm import tqdm

# Reuse action parsing and comparison from verifier script
from verifier_ar_inference import (
    SUBTASK_ISOLATED_USER_PROMPT,
    SUPPORTED_ACTIONS_WORD,
    SUPPORTED_ACTIONS_EXCEL,
    SUPPORTED_ACTIONS_PPT,
    TOOL_DEFAULTS,
    normalize_tool_args,
    parse_action,
    compare_actions,
    format_action_brief,
    smart_resize,
)


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def encode_image_base64(image_path):
    """Load image, apply smart_resize, return base64 + original dimensions."""
    img = Image.open(image_path)
    orig_w, orig_h = img.size

    resized_h, resized_w = smart_resize(orig_h, orig_w, factor=28)
    if (resized_w, resized_h) != (orig_w, orig_h):
        img = img.resize((resized_w, resized_h), Image.Resampling.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}", orig_w, orig_h, resized_w, resized_h


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

CONDITION_A_PROMPT = """You are looking at a screenshot of {domain}.
Describe the current UI state in detail:
- Which application area is currently active?
- Are there any open menus, dialogs, or selections?
- What UI elements are visible and interactable?

Output your description in 2-3 sentences."""

CONDITION_B_PROMPT = """You are controlling {domain}.

Overall Task: {request}

Current Subtask: {subtask}

Current UI State: {ui_state_description}

The history of actions are:
{history}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

Based on the UI state description above, decide the next action to take.

Output your action within <tool_call></tool_call> tag like:
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


def get_actions_for_domain(domain):
    d = domain.lower()
    if d == "word":
        return SUPPORTED_ACTIONS_WORD
    elif d == "excel":
        return SUPPORTED_ACTIONS_EXCEL
    elif d == "ppt":
        return SUPPORTED_ACTIONS_PPT
    return SUPPORTED_ACTIONS_WORD


# ---------------------------------------------------------------------------
# Phase 1: Condition C — AR greedy (parallel across trajectories)
# ---------------------------------------------------------------------------

def load_trajectories(data_root, trajectory_ids):
    """Load raw trajectory data (same logic as verifier_ar_inference.py)."""
    id_set = set(trajectory_ids)
    data_path = os.path.join(data_root, "data")
    trajectories = {}

    for domain in sorted(os.listdir(data_path)):
        domain_path = os.path.join(data_path, domain)
        if not os.path.isdir(domain_path):
            continue
        for category in sorted(os.listdir(domain_path)):
            success_path = os.path.join(domain_path, category, "success")
            if not os.path.isdir(success_path):
                continue
            for fname in sorted(os.listdir(success_path)):
                if not fname.endswith(".jsonl"):
                    continue
                file_stem = os.path.splitext(fname)[0]
                traj_id = f"{domain}_{category}_{file_stem}"
                if traj_id not in id_set:
                    continue

                fpath = os.path.join(success_path, fname)
                steps = []
                with open(fpath, "r") as f:
                    for line_num, line in enumerate(f, 1):
                        if not line.strip():
                            continue
                        try:
                            d = json.loads(line.strip())
                        except json.JSONDecodeError:
                            continue

                        action = d["step"]["action"]
                        if action.get("function", "") == "drag" or not action.get("rectangle", {}):
                            continue

                        sample_id = f"{traj_id}_{line_num}"
                        clean_img = os.path.join(
                            data_root, "image", domain, category,
                            d["step"]["screenshot_clean"],
                        )

                        status = d["step"]["status"]
                        if status == "OVERALL_FINISH":
                            status = "FINISH"
                        elif status == "FINISH":
                            status = "CONTINUE"

                        steps.append({
                            "sample_id": sample_id,
                            "line_num": line_num,
                            "request": d["request"],
                            "screenshot_clean": clean_img,
                            "thought": d["step"].get("thought", ""),
                            "subtask": d["step"].get("subtask", ""),
                            "action": action,
                            "status": status,
                        })

                trajectories[traj_id] = {
                    "trajectory_id": traj_id,
                    "domain": domain,
                    "category": category,
                    "steps": steps,
                }

    return trajectories


def segment_by_subtask(steps):
    """Group consecutive steps by subtask field."""
    if not steps:
        return []
    segments = []
    current_subtask = steps[0].get("subtask", "")
    current_segment = []
    for step in steps:
        subtask = step.get("subtask", "")
        if subtask != current_subtask:
            if current_segment:
                segments.append((current_subtask, current_segment))
            current_subtask = subtask
            current_segment = [step]
        else:
            current_segment.append(step)
    if current_segment:
        segments.append((current_subtask, current_segment))
    return segments


def prepare_gt(step, orig_w, orig_h):
    """Prepare GT action for comparison."""
    action = step["action"].copy()
    action_args = action.get("args", {}).copy()
    action["args"] = action_args

    gt_rect = action.get("rectangle", {})
    gt_rect_end = None

    if action["function"] == "drag":
        sx, sy = action_args["start_x"], action_args["start_y"]
        ex, ey = action_args["end_x"], action_args["end_y"]
        action_args["start_coordinate"] = [sx, sy]
        action_args["end_coordinate"] = [ex, ey]
        for k in ["start_x", "start_y", "end_x", "end_y"]:
            action_args.pop(k, None)
        gt_rect = {"left": max(0, sx) - 25, "top": max(0, sy) - 25,
                    "right": min(sx + 25, orig_w), "bottom": min(sy + 25, orig_h)}
        gt_rect_end = {"left": max(0, ex) - 25, "top": max(0, ey) - 25,
                       "right": min(ex + 25, orig_w), "bottom": min(ey + 25, orig_h)}
    else:
        action_args.pop("x", None)
        action_args.pop("y", None)
        if "coordinate_x" in action and action["coordinate_x"]:
            action_args["coordinate"] = [action["coordinate_x"], action["coordinate_y"]]

    return action.get("function", ""), action_args, step.get("status", ""), gt_rect, gt_rect_end


def evaluate_trajectory_ar(client, model_name, traj, data_root):
    """Evaluate a single trajectory with AR subtask_isolated (Condition C)."""
    traj_id = traj["trajectory_id"]
    domain = traj["domain"]
    steps = traj["steps"]
    actions_text = get_actions_for_domain(domain)

    segments = segment_by_subtask(steps)
    step_results = []
    step_num = 0
    all_correct = True

    for seg_idx, (subtask_desc, seg_steps) in enumerate(segments):
        history_entries = []

        for local_step_num, step in enumerate(seg_steps):
            step_num += 1
            t0 = time.time()

            # Encode image
            try:
                image_url, orig_w, orig_h, res_w, res_h = encode_image_base64(step["screenshot_clean"])
            except Exception as e:
                step_results.append({
                    "sample_id": step["sample_id"],
                    "step_num": step_num,
                    "subtask_idx": seg_idx,
                    "subtask_description": subtask_desc,
                    "local_step_num": local_step_num + 1,
                    "success": False,
                    "error_message": f"Image error: {e}",
                })
                all_correct = False
                continue

            # Build prompt
            history_text = "\n".join(history_entries) if history_entries else "None"
            user_prompt = SUBTASK_ISOLATED_USER_PROMPT.format(
                instruction=step["request"],
                subtask_description=subtask_desc or step["request"],
                history=history_text,
                actions=actions_text,
            )

            # Call vLLM
            messages = [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": user_prompt},
            ]}]

            try:
                result = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=512,
                    temperature=0,
                )
                response = result.choices[0].message.content or ""
            except Exception as e:
                response = ""

            # Parse and compare
            pred_fn, pred_args, pred_status = parse_action(response, orig_w, orig_h)
            gt_fn, gt_args, gt_status, gt_rect, gt_rect_end = prepare_gt(step, orig_w, orig_h)

            try:
                if gt_fn == "drag":
                    fm, am, sm = compare_actions(pred_fn, pred_args, pred_status,
                                                  gt_fn, gt_args, gt_status, gt_rect, gt_rect_end)
                else:
                    fm, am, sm = compare_actions(pred_fn, pred_args, pred_status,
                                                  gt_fn, gt_args, gt_status, gt_rect)
            except Exception:
                fm, am, sm = False, False, False

            success = fm and am and sm
            if not success:
                all_correct = False

            step_results.append({
                "sample_id": step["sample_id"],
                "step_num": step_num,
                "subtask_idx": seg_idx,
                "subtask_description": subtask_desc,
                "local_step_num": local_step_num + 1,
                "success": success,
                "function_match": fm,
                "args_match": am,
                "status_match": sm,
                "predicted_function": pred_fn,
                "predicted_args": pred_args,
                "predicted_status": pred_status,
                "ground_truth_function": gt_fn,
                "ground_truth_args": gt_args,
                "ground_truth_status": gt_status,
                "ground_truth_rect": gt_rect,
                "raw_model_output": response,
                "execution_time": time.time() - t0,
                "temperature_used": 0,
            })

            # Update history
            brief = format_action_brief(pred_fn, pred_args)
            history_entries.append(f"Step {local_step_num + 1}: {brief}")

    # Trajectory-level metrics
    n_steps = len(step_results)
    first_error = next((sr["step_num"] for sr in step_results if not sr["success"]), n_steps + 1)
    progress = (first_error - 1) / n_steps if n_steps > 0 else 0
    scattered = sum(1 for sr in step_results if sr["success"]) / n_steps if n_steps > 0 else 0

    return {
        "trajectory_id": traj_id,
        "num_steps": n_steps,
        "trajectory_success": all_correct and n_steps > 0,
        "progress_rate": progress,
        "scattered_progress_rate": scattered,
        "first_error_step": first_error,
        "domain": domain,
        "category": traj["category"],
        "step_results": step_results,
    }


def run_phase1_condition_c(client, model_name, trajectories, max_workers=128):
    """Run Condition C: AR greedy on all trajectories in parallel."""
    print(f"\n{'='*60}")
    print(f"Phase 1: Condition C — AR Greedy ({len(trajectories)} trajectories, {max_workers} workers)")
    print(f"{'='*60}")

    results = []
    t0 = time.time()

    with tqdm(total=len(trajectories), desc="Condition C (AR)") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(evaluate_trajectory_ar, client, model_name, traj, ""): traj_id
                for traj_id, traj in trajectories.items()
            }

            for future in as_completed(futures):
                traj_id = futures[future]
                try:
                    result = future.result(timeout=1800)
                    results.append(result)
                except Exception as e:
                    print(f"  ERROR on {traj_id}: {e}")
                    results.append({
                        "trajectory_id": traj_id,
                        "num_steps": 0,
                        "trajectory_success": False,
                        "progress_rate": 0,
                        "scattered_progress_rate": 0,
                        "first_error_step": 0,
                        "domain": "",
                        "category": "",
                        "step_results": [],
                    })
                pbar.update(1)

    elapsed = time.time() - t0
    n_traj = len(results)
    n_steps = sum(r["num_steps"] for r in results)
    tsr = sum(1 for r in results if r["trajectory_success"]) / n_traj if n_traj else 0
    step_acc = (sum(sum(1 for s in r["step_results"] if s.get("success")) for r in results)
                / n_steps if n_steps else 0)
    print(f"  Done in {elapsed:.0f}s | TSR={tsr:.4f} | Step Acc={step_acc:.4f} | {n_steps} steps")

    return results


# ---------------------------------------------------------------------------
# Phase 2: Conditions A, B, D — step-level parallel
# ---------------------------------------------------------------------------

def run_step_abd(client, model_name, step, greedy_by_traj, greedy_by_step):
    """Run Conditions A, B, D for a single step."""
    sample_id = step["sample_id"]
    t0_all = time.time()

    # Encode image for Condition A
    try:
        image_url, orig_w, orig_h, res_w, res_h = encode_image_base64(step["screenshot_clean"])
    except Exception as e:
        return {"sample_id": sample_id, "error": str(e)}

    result = {
        "sample_id": sample_id,
        "trajectory_id": step["trajectory_id"],
        "domain": step["domain"],
        "category": step["category"],
        "step_num": step["step_num"],
        "num_steps": step["num_steps"],
        "position_bucket": step["position_bucket"],
        "greedy_correct": step["greedy_correct"],
        "gt_thought": step["thought"],
        "gt_subtask": step.get("subtask", ""),
        "request": step["request"],
    }

    # Prepare GT
    gt_fn, gt_args, gt_status, gt_rect, gt_rect_end = prepare_gt(step, orig_w, orig_h)
    result["gt_function"] = gt_fn
    result["gt_args"] = gt_args
    result["gt_status"] = gt_status

    # Build history from greedy results (subtask_isolated)
    traj_id = step["trajectory_id"]
    step_num = step["step_num"]
    traj_greedy = greedy_by_traj.get(traj_id, {})
    all_traj_steps = sorted(traj_greedy.get("step_results", []), key=lambda s: s["step_num"])

    current_subtask_idx = None
    for s in all_traj_steps:
        if s["step_num"] == step_num:
            current_subtask_idx = s.get("subtask_idx", 0)
            break

    history_entries = []
    local_idx = 0
    for ps in all_traj_steps:
        if ps["step_num"] >= step_num:
            break
        if current_subtask_idx is not None and ps.get("subtask_idx", 0) != current_subtask_idx:
            continue
        local_idx += 1
        fn = ps.get("predicted_function", "unknown")
        args = ps.get("predicted_args", {})
        history_entries.append(f"Step {local_idx}: {format_action_brief(fn, args)}")
    history_text = "\n".join(history_entries) if history_entries else "None"

    # ---- Condition C (from greedy results) ----
    greedy_step = greedy_by_step.get(sample_id, {})
    result["condition_c"] = {
        "success": greedy_step.get("success", False),
        "function_match": greedy_step.get("function_match", False),
        "args_match": greedy_step.get("args_match", False),
        "status_match": greedy_step.get("status_match", False),
        "predicted_function": greedy_step.get("predicted_function"),
        "predicted_args": greedy_step.get("predicted_args"),
        "predicted_status": greedy_step.get("predicted_status"),
        "source": "phase1_greedy",
    }

    # ---- Condition A: Screenshot → UI description ----
    try:
        domain_name = {"ppt": "PowerPoint", "excel": "Microsoft Excel", "word": "Microsoft Word"}.get(
            step["domain"].lower(), step["domain"].capitalize())
        a_prompt = CONDITION_A_PROMPT.format(domain=domain_name)
        a_messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": a_prompt},
        ]}]
        a_result = client.chat.completions.create(
            model=model_name, messages=a_messages, max_tokens=256, temperature=0)
        a_response = a_result.choices[0].message.content or ""
        result["condition_a"] = {"ui_state_description": a_response}
    except Exception as e:
        a_response = ""
        result["condition_a"] = {"ui_state_description": "", "error": str(e)}

    # ---- Condition B: GT UI state → Action (no screenshot) ----
    try:
        t0 = time.time()
        gt_ui_state = step["thought"]
        b_prompt = CONDITION_B_PROMPT.format(
            domain=step["domain"].capitalize(),
            request=step["request"],
            subtask=step.get("subtask", "") or step["request"],
            ui_state_description=gt_ui_state,
            history=history_text,
            actions=get_actions_for_domain(step["domain"]),
        )
        b_messages = [{"role": "user", "content": [{"type": "text", "text": b_prompt}]}]
        b_result = client.chat.completions.create(
            model=model_name, messages=b_messages, max_tokens=512, temperature=0)
        b_response = b_result.choices[0].message.content or ""

        b_fn, b_args, b_status = parse_action(b_response, orig_w, orig_h)
        try:
            if gt_fn == "drag":
                b_fm, b_am, b_sm = compare_actions(b_fn, b_args, b_status, gt_fn, gt_args, gt_status, gt_rect, gt_rect_end)
            else:
                b_fm, b_am, b_sm = compare_actions(b_fn, b_args, b_status, gt_fn, gt_args, gt_status, gt_rect)
        except Exception:
            b_fm, b_am, b_sm = False, False, False

        result["condition_b"] = {
            "success": b_fm and b_am and b_sm,
            "function_match": b_fm, "args_match": b_am, "status_match": b_sm,
            "predicted_function": b_fn, "predicted_args": b_args, "predicted_status": b_status,
            "raw_response": b_response, "execution_time": time.time() - t0,
        }
    except Exception as e:
        result["condition_b"] = {"success": False, "error": str(e)}

    # ---- Condition D: Chained A→B (no screenshot) ----
    try:
        t0 = time.time()
        model_ui_state = a_response if a_response else "Unable to describe current UI state."
        d_prompt = CONDITION_B_PROMPT.format(
            domain=step["domain"].capitalize(),
            request=step["request"],
            subtask=step.get("subtask", "") or step["request"],
            ui_state_description=model_ui_state,
            history=history_text,
            actions=get_actions_for_domain(step["domain"]),
        )
        d_messages = [{"role": "user", "content": [{"type": "text", "text": d_prompt}]}]
        d_result = client.chat.completions.create(
            model=model_name, messages=d_messages, max_tokens=512, temperature=0)
        d_response = d_result.choices[0].message.content or ""

        d_fn, d_args, d_status = parse_action(d_response, orig_w, orig_h)
        try:
            if gt_fn == "drag":
                d_fm, d_am, d_sm = compare_actions(d_fn, d_args, d_status, gt_fn, gt_args, gt_status, gt_rect, gt_rect_end)
            else:
                d_fm, d_am, d_sm = compare_actions(d_fn, d_args, d_status, gt_fn, gt_args, gt_status, gt_rect)
        except Exception:
            d_fm, d_am, d_sm = False, False, False

        result["condition_d"] = {
            "success": d_fm and d_am and d_sm,
            "function_match": d_fm, "args_match": d_am, "status_match": d_sm,
            "predicted_function": d_fn, "predicted_args": d_args, "predicted_status": d_status,
            "raw_response": d_response, "execution_time": time.time() - t0,
        }
    except Exception as e:
        result["condition_d"] = {"success": False, "error": str(e)}

    result["total_time"] = time.time() - t0_all
    return result


def run_phase2_abd(client, model_name, steps, greedy_by_traj, greedy_by_step, max_workers=128):
    """Run Conditions A, B, D for all steps in parallel."""
    print(f"\n{'='*60}")
    print(f"Phase 2: Conditions A, B, D ({len(steps)} steps, {max_workers} workers)")
    print(f"{'='*60}")

    results = []
    t0 = time.time()

    with tqdm(total=len(steps), desc="Conditions A+B+D") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_step_abd, client, model_name, step, greedy_by_traj, greedy_by_step): step["sample_id"]
                for step in steps
            }

            for future in as_completed(futures):
                sid = futures[future]
                try:
                    result = future.result(timeout=600)
                    results.append(result)
                except Exception as e:
                    print(f"  ERROR on {sid}: {e}")
                    results.append({"sample_id": sid, "error": str(e)})
                pbar.update(1)

    elapsed = time.time() - t0
    b_acc = sum(1 for r in results if r.get("condition_b", {}).get("success")) / len(results) if results else 0
    d_acc = sum(1 for r in results if r.get("condition_d", {}).get("success")) / len(results) if results else 0
    c_acc = sum(1 for r in results if r.get("condition_c", {}).get("success")) / len(results) if results else 0
    print(f"  Done in {elapsed:.0f}s | C={c_acc:.4f} B={b_acc:.4f} D={d_acc:.4f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cognitive Interference — vLLM fast inference")
    parser.add_argument("--api_url", type=str, default="http://localhost:19806/v1")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name as served by vLLM (usually the model path)")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--trajectory_ids", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=128)
    parser.add_argument("--max_trajectories", type=int, default=None)
    args = parser.parse_args()

    client = OpenAI(api_key="0", base_url=args.api_url, timeout=600)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load trajectory IDs
    with open(args.trajectory_ids) as f:
        traj_ids = json.load(f)
    print(f"Pattern B trajectories: {len(traj_ids)}")

    # Load raw trajectory data
    print("Loading trajectories...")
    trajectories = load_trajectories(args.data_root, traj_ids)
    print(f"  Loaded {len(trajectories)} trajectories, "
          f"{sum(len(t['steps']) for t in trajectories.values())} steps")

    if args.max_trajectories:
        keep = list(trajectories.keys())[:args.max_trajectories]
        trajectories = {k: trajectories[k] for k in keep}
        print(f"  Limited to {len(trajectories)} trajectories")

    # ======================================================================
    # Phase 1: Condition C
    # ======================================================================
    greedy_results_path = os.path.join(args.output_dir, "base_always_greedy_results.json")

    if os.path.exists(greedy_results_path):
        print(f"\nPhase 1 results exist, loading from {greedy_results_path}")
        with open(greedy_results_path) as f:
            greedy_data = json.load(f)
        c_results = greedy_data["detailed_results"]
    else:
        c_results = run_phase1_condition_c(client, args.model_name, trajectories, args.max_workers)

        # Save Phase 1 results
        n_steps = sum(r["num_steps"] for r in c_results)
        tsr = sum(1 for r in c_results if r["trajectory_success"]) / len(c_results) if c_results else 0
        step_acc = (sum(sum(1 for s in r["step_results"] if s.get("success")) for r in c_results)
                    / n_steps if n_steps else 0)
        greedy_data = {
            "config": {
                "mode": "always_greedy",
                "model": args.model_name,
                "timestamp": datetime.now().isoformat(),
            },
            "statistics": {
                "num_trajectories": len(c_results),
                "num_steps": n_steps,
                "trajectory_success_rate": tsr,
                "step_success_rate": step_acc,
            },
            "trajectory_results": [
                {k: v for k, v in r.items() if k != "step_results"} for r in c_results
            ],
            "detailed_results": c_results,
        }
        with open(greedy_results_path, "w") as f:
            json.dump(greedy_data, f, indent=2, default=str)
        print(f"  Saved Phase 1 → {greedy_results_path}")

    # Build lookups from Phase 1
    greedy_by_traj = {t["trajectory_id"]: t for t in c_results}
    greedy_by_step = {}
    for traj in c_results:
        for s in traj["step_results"]:
            greedy_by_step[s["sample_id"]] = s

    # ======================================================================
    # Prepare all steps for Phase 2
    # ======================================================================
    from sample_cognitive_interference import assign_position_bucket

    all_steps = []
    for traj in c_results:
        traj_id = traj["trajectory_id"]
        raw_traj = trajectories.get(traj_id)
        if not raw_traj:
            continue
        raw_steps_by_id = {s["sample_id"]: s for s in raw_traj["steps"]}

        for sr in traj["step_results"]:
            raw = raw_steps_by_id.get(sr["sample_id"])
            if not raw:
                continue
            all_steps.append({
                **raw,
                "trajectory_id": traj_id,
                "domain": traj["domain"],
                "category": traj["category"],
                "greedy_correct": sr.get("success", False),
                "step_num": sr["step_num"],
                "num_steps": traj["num_steps"],
                "position_bucket": assign_position_bucket(sr["step_num"], traj["num_steps"]),
            })

    print(f"\nPrepared {len(all_steps)} steps for Phase 2")

    # ======================================================================
    # Phase 2: Conditions A, B, D
    # ======================================================================
    abd_results = run_phase2_abd(client, args.model_name, all_steps, greedy_by_traj, greedy_by_step, args.max_workers)

    # Save final merged results
    merged = {
        "config": {
            "model": args.model_name,
            "conditions": ["A", "B", "C", "D"],
            "n_samples": len(abd_results),
            "max_workers": args.max_workers,
            "timestamp": datetime.now().isoformat(),
        },
        "results": abd_results,
    }
    merged_path = os.path.join(args.output_dir, "cognitive_interference_results.json")
    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2, default=str)
    print(f"\nSaved merged results → {merged_path}")

    # Quick summary
    n = len(abd_results)
    for cond, key in [("C", "condition_c"), ("B", "condition_b"), ("D", "condition_d")]:
        succ = sum(1 for r in abd_results if r.get(key, {}).get("success", False))
        print(f"  {cond}: {succ}/{n} = {succ/n:.4f}")

    a_descs = [r.get("condition_a", {}).get("ui_state_description", "") for r in abd_results]
    a_valid = sum(1 for d in a_descs if d and "<tool_call>" not in d)
    print(f"  A: {a_valid}/{n} valid descriptions (non-tool_call)")


if __name__ == "__main__":
    main()
