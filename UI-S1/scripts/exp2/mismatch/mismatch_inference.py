"""Part B: Oracle History Inference (Exp2e).

Runs SFT v2 / Base model with GT action history instead of predicted history.

3 conditions:
  - oracle:         SFT v2 with GT history (same prompt as C0)
  - oracle_agent_h: Base model History Agent with GT history
  - oracle_f4:      SFT v2 Full Decomposition with GT history + oracle_agent_h + agent_v

Adapts multiagent/inference.py's load_steps() to use ground_truth history
from C0 results instead of predicted history.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Add parent dir for imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP2_DIR = os.path.dirname(SCRIPT_DIR)
if EXP2_DIR not in sys.path:
    sys.path.insert(0, EXP2_DIR)

from verifier_ar_inference import (
    SUBTASK_ISOLATED_USER_PROMPT,
    TOOL_DEFAULTS,
    normalize_tool_args,
    parse_action,
    compare_actions,
    format_action_brief,
    smart_resize,
)
from cognitive_interference_vllm import (
    encode_image_base64,
    get_actions_for_domain,
    segment_by_subtask,
    prepare_gt,
)
from sample_cognitive_interference import assign_position_bucket

# Reuse prompt templates from multiagent inference
from multiagent.inference import (
    AGENT_H_PROMPT,
    DOMAIN_NAMES,
    load_aux_outputs,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_CONDITIONS = ["oracle", "oracle_agent_h", "oracle_f4"]
TEXT_CONDITIONS = {"oracle_agent_h"}
ACTION_CONDITIONS = {"oracle", "oracle_f4"}


# ---------------------------------------------------------------------------
# Data loading — oracle history variant
# ---------------------------------------------------------------------------

def load_trajectories(data_root, trajectory_ids):
    """Load raw trajectory data from GUI-360 test set."""
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
                            "control_test": action.get("control_test", ""),
                        })

                trajectories[traj_id] = {
                    "trajectory_id": traj_id,
                    "domain": domain,
                    "category": category,
                    "steps": steps,
                }

    return trajectories


def load_steps_oracle(data_root, trajectory_ids, c0_results_path):
    """Load all steps with oracle (GT) history.

    Key difference from multiagent/inference.py's load_steps():
    History is built from ground_truth_function / ground_truth_args
    instead of predicted_function / predicted_args.
    """
    trajectories = load_trajectories(data_root, trajectory_ids)
    print(f"  Loaded {len(trajectories)} trajectories, "
          f"{sum(len(t['steps']) for t in trajectories.values())} steps")

    # Load C0 results for GT history extraction
    with open(c0_results_path) as f:
        c0_data = json.load(f)

    c0_by_traj = {}
    for traj in c0_data["detailed_results"]:
        c0_by_traj[traj["trajectory_id"]] = traj

    all_steps = []
    for traj_id, traj in trajectories.items():
        domain = traj["domain"]
        category = traj["category"]
        steps = traj["steps"]
        num_steps = len(steps)

        c0_traj = c0_by_traj.get(traj_id, {})
        c0_step_results = c0_traj.get("step_results", [])

        # Build subtask segments
        segments = segment_by_subtask(steps)

        step_subtask_map = {}
        sample_to_idx = {s["sample_id"]: i for i, s in enumerate(steps)}
        for seg_idx, (subtask_desc, seg_steps_list) in enumerate(segments):
            for local_idx, seg_step in enumerate(seg_steps_list):
                idx = sample_to_idx.get(seg_step["sample_id"])
                if idx is not None:
                    step_subtask_map[idx] = (seg_idx, local_idx, subtask_desc)

        # Map sample_id -> C0 step result (for GT extraction)
        c0_by_sample = {}
        for sr in c0_step_results:
            c0_by_sample[sr["sample_id"]] = sr

        for step_idx, step in enumerate(steps):
            seg_idx, local_idx, subtask_desc = step_subtask_map.get(
                step_idx, (0, 0, step.get("subtask", "") or step["request"]))

            # Build ORACLE history from GT actions within same subtask
            history_entries = []
            local_counter = 0
            for prev_idx in range(step_idx):
                prev_seg_idx = step_subtask_map.get(prev_idx, (None,))[0]
                if prev_seg_idx != seg_idx:
                    continue
                local_counter += 1
                prev_sample_id = steps[prev_idx]["sample_id"]
                prev_c0 = c0_by_sample.get(prev_sample_id, {})
                # KEY CHANGE: use ground_truth instead of predicted
                fn = prev_c0.get("ground_truth_function", "unknown")
                args = prev_c0.get("ground_truth_args", {})
                history_entries.append(f"Step {local_counter}: {format_action_brief(fn, args)}")
            history_text = "\n".join(history_entries) if history_entries else "None"

            # Prepare GT
            try:
                img = Image.open(step["screenshot_clean"])
                orig_w, orig_h = img.size
                img.close()
            except Exception:
                orig_w, orig_h = 1920, 1080

            gt_fn, gt_args, gt_status, gt_rect, gt_rect_end = prepare_gt(step, orig_w, orig_h)

            all_steps.append({
                "sample_id": step["sample_id"],
                "trajectory_id": traj_id,
                "domain": domain,
                "category": category,
                "step_num": step_idx + 1,
                "num_steps": num_steps,
                "position_bucket": assign_position_bucket(step_idx + 1, num_steps),
                "request": step["request"],
                "subtask": subtask_desc or step.get("subtask", "") or step["request"],
                "screenshot_clean": step["screenshot_clean"],
                "history_text": history_text,
                "control_test": step.get("control_test", ""),
                "gt_function": gt_fn,
                "gt_args": gt_args,
                "gt_status": gt_status,
                "gt_rect": gt_rect,
                "gt_rect_end": gt_rect_end,
                "orig_w": orig_w,
                "orig_h": orig_h,
            })

    return all_steps


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt(condition, step, aux_data):
    """Build prompt for a given condition and step.

    Returns (prompt_text, uses_image, temperature).
    """
    domain = step["domain"]
    domain_name = DOMAIN_NAMES.get(domain.lower(), domain.capitalize())
    actions_text = get_actions_for_domain(domain)

    if condition == "oracle_agent_h":
        # History Agent with oracle history (text output, no screenshot)
        text = AGENT_H_PROMPT.format(
            task=step["request"],
            history=step["history_text"],
        )
        return text, False, 0

    elif condition == "oracle":
        # SFT v2 with oracle history — same prompt as C0
        user_prompt = SUBTASK_ISOLATED_USER_PROMPT.format(
            instruction=step["request"],
            subtask_description=step["subtask"],
            history=step["history_text"],
            actions=actions_text,
        )
        return user_prompt, True, 0

    elif condition == "oracle_f4":
        # Full decomposition with oracle components:
        # agent_v (screenshot-only, reused from Exp2d) + oracle_agent_h
        visual_desc = aux_data.get("agent_v", {}).get(step["sample_id"], "")
        progress = aux_data.get("oracle_agent_h", {}).get(step["sample_id"], "")

        user_prompt = SUBTASK_ISOLATED_USER_PROMPT.format(
            instruction=step["request"],
            subtask_description=step["subtask"],
            history=step["history_text"],
            actions=actions_text,
        )

        # Inject Agent H progress after task
        if progress:
            insert_after_task = f"\nTask progress: {progress}\n"
            marker = f"Overall Task:\n{step['request']}\n"
            idx = user_prompt.find(marker)
            if idx >= 0:
                insert_pos = idx + len(marker)
                user_prompt = user_prompt[:insert_pos] + insert_after_task + user_prompt[insert_pos:]

        # Inject Agent V description before actions
        if visual_desc:
            insert_after_history = f"\nCurrent screen elements: {visual_desc}\n"
            marker = "The actions supported are:"
            idx = user_prompt.find(marker)
            if idx >= 0:
                user_prompt = user_prompt[:idx] + insert_after_history + "\n" + user_prompt[idx:]

        return user_prompt, True, 0

    else:
        raise ValueError(f"Unknown condition: {condition}")


# ---------------------------------------------------------------------------
# Single step inference
# ---------------------------------------------------------------------------

def run_step(client, model_name, condition, step, aux_data):
    """Run inference for one step."""
    sample_id = step["sample_id"]
    t0 = time.time()

    try:
        prompt_text, uses_image, temperature = build_prompt(condition, step, aux_data)
    except Exception as e:
        return {"sample_id": sample_id, "error": f"prompt build: {e}"}

    # Encode image if needed
    image_url = None
    if uses_image:
        try:
            image_url, orig_w, orig_h, res_w, res_h = encode_image_base64(
                step["screenshot_clean"])
        except Exception as e:
            return {"sample_id": sample_id, "error": f"image encode: {e}"}
    else:
        orig_w = step["orig_w"]
        orig_h = step["orig_h"]

    # Build messages
    content = []
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]

    # Run inference
    try:
        result = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=512,
            temperature=temperature,
        )
        response = result.choices[0].message.content or ""
    except Exception as e:
        return {"sample_id": sample_id, "error": f"API call: {e}"}

    elapsed = time.time() - t0

    result_dict = {
        "sample_id": sample_id,
        "trajectory_id": step["trajectory_id"],
        "domain": step["domain"],
        "category": step["category"],
        "step_num": step["step_num"],
        "num_steps": step["num_steps"],
        "position_bucket": step["position_bucket"],
        "gt_function": step["gt_function"],
        "gt_args": step["gt_args"],
        "gt_status": step["gt_status"],
        "gt_control_test": step["control_test"],
        "execution_time": elapsed,
    }

    if condition in TEXT_CONDITIONS:
        # Text output — no action parsing
        control_test = step.get("control_test", "")
        hit = False
        if control_test and response:
            hit = control_test.lower() in response.lower()
        result_dict["text_output"] = response
        result_dict["control_test_hit"] = hit
    else:
        # Action conditions: parse and compare
        orig_w_step = step["orig_w"]
        orig_h_step = step["orig_h"]
        pred_fn, pred_args, pred_status = parse_action(response, orig_w_step, orig_h_step)

        gt_fn = step["gt_function"]
        gt_args = step["gt_args"]
        gt_status = step["gt_status"]
        gt_rect = step["gt_rect"]
        gt_rect_end = step["gt_rect_end"]

        try:
            if gt_fn == "drag":
                fm, am, sm = compare_actions(pred_fn, pred_args, pred_status,
                                             gt_fn, gt_args, gt_status, gt_rect, gt_rect_end)
            else:
                fm, am, sm = compare_actions(pred_fn, pred_args, pred_status,
                                             gt_fn, gt_args, gt_status, gt_rect)
        except Exception:
            fm, am, sm = False, False, False

        result_dict.update({
            "predicted_function": pred_fn,
            "predicted_args": pred_args,
            "predicted_status": pred_status,
            "success": fm and am and sm,
            "function_match": fm,
            "args_match": am,
            "status_match": sm,
            "raw_response": response,
        })

    return result_dict


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------

def run_all_steps(client, model_name, condition, steps, aux_data, max_workers):
    """Run all steps in parallel."""
    print(f"\nRunning condition '{condition}' on {len(steps)} steps "
          f"({max_workers} workers)...")
    t0 = time.time()
    results = []

    with tqdm(total=len(steps), desc=condition) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_step, client, model_name, condition, step, aux_data): step["sample_id"]
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
    n_errors = sum(1 for r in results if "error" in r)

    if condition in TEXT_CONDITIONS:
        n_valid = sum(1 for r in results if r.get("text_output"))
        hit_rate = sum(1 for r in results if r.get("control_test_hit")) / len(results) if results else 0
        print(f"  Done in {elapsed:.0f}s | {n_valid} valid outputs | "
              f"thought-hit={hit_rate:.4f} | {n_errors} errors")
    else:
        n_success = sum(1 for r in results if r.get("success"))
        step_acc = n_success / len(results) if results else 0
        fm_rate = sum(1 for r in results if r.get("function_match")) / len(results) if results else 0
        am_rate = sum(1 for r in results if r.get("args_match")) / len(results) if results else 0
        print(f"  Done in {elapsed:.0f}s | step_acc={step_acc:.4f} | "
              f"fm={fm_rate:.4f} | am={am_rate:.4f} | {n_errors} errors")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exp2e Part B: Oracle History Inference")
    parser.add_argument("--condition", type=str, required=True,
                        choices=ALL_CONDITIONS,
                        help="Condition to run")
    parser.add_argument("--api_url", type=str, default="http://localhost:19806/v1",
                        help="vLLM server URL")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model path for vLLM")
    parser.add_argument("--data_root", type=str, required=True,
                        help="GUI-360 test data root")
    parser.add_argument("--trajectory_ids", type=str, required=True,
                        help="Path to pattern_b_ids.json")
    parser.add_argument("--c0_results", type=str, required=True,
                        help="Path to C0 results JSON (for GT extraction)")
    parser.add_argument("--aux_dir", type=str, default=None,
                        help="Directory with prior conditions' outputs")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Results output directory")
    parser.add_argument("--max_workers", type=int, default=128,
                        help="ThreadPoolExecutor workers")
    parser.add_argument("--max_trajectories", type=int, default=None,
                        help="Limit trajectories (for testing)")
    args = parser.parse_args()

    condition = args.condition
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"Exp2e Part B: Oracle History — Condition: {condition}")
    print(f"Model: {args.model_name}")
    print(f"Workers: {args.max_workers}")
    print("=" * 60)

    # Load trajectory IDs
    with open(args.trajectory_ids) as f:
        traj_ids = json.load(f)
    if args.max_trajectories:
        traj_ids = traj_ids[:args.max_trajectories]
    print(f"Trajectories: {len(traj_ids)}")

    # Load steps with oracle history
    print("Loading steps with oracle history...")
    steps = load_steps_oracle(args.data_root, traj_ids, args.c0_results)
    print(f"Total steps: {len(steps)}")

    # Load auxiliary outputs if needed
    aux_data = {}
    deps = {
        "oracle_f4": ["agent_v", "oracle_agent_h"],
    }

    if condition in deps:
        aux_dir = args.aux_dir or args.output_dir
        for dep_name in deps[condition]:
            print(f"Loading auxiliary: {dep_name}...")
            aux_data[dep_name] = load_aux_outputs(aux_dir, dep_name)

    # Create client
    client = OpenAI(api_key="0", base_url=args.api_url, timeout=600)

    # Run inference
    results = run_all_steps(client, args.model_name, condition, steps,
                            aux_data, args.max_workers)

    # Save results as JSONL
    output_path = os.path.join(args.output_dir, f"{condition}.jsonl")
    with open(output_path, "w") as f:
        for r in sorted(results, key=lambda x: x.get("sample_id", "")):
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\nSaved {len(results)} results to {output_path}")

    # Summary
    print(f"\n{'='*40} Summary {'='*40}")
    print(f"Condition: {condition}")
    print(f"Total steps: {len(results)}")
    n_errors = sum(1 for r in results if "error" in r)
    if n_errors:
        print(f"Errors: {n_errors}")
    if condition in TEXT_CONDITIONS:
        n_valid = sum(1 for r in results if r.get("text_output"))
        hit_rate = sum(1 for r in results if r.get("control_test_hit")) / len(results) if results else 0
        print(f"Valid outputs: {n_valid}/{len(results)}")
        print(f"Thought-hit rate: {hit_rate:.4f}")
    else:
        n_success = sum(1 for r in results if r.get("success"))
        step_acc = n_success / len(results) if results else 0
        fm = sum(1 for r in results if r.get("function_match")) / len(results) if results else 0
        am = sum(1 for r in results if r.get("args_match")) / len(results) if results else 0
        sm = sum(1 for r in results if r.get("status_match")) / len(results) if results else 0
        print(f"Step accuracy: {step_acc:.4f} ({n_success}/{len(results)})")
        print(f"Function match: {fm:.4f}")
        print(f"Args match: {am:.4f}")
        print(f"Status match: {sm:.4f}")


if __name__ == "__main__":
    main()
