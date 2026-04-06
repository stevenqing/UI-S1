"""Multi-Agent Decomposition Experiment — Unified Inference (Exp2d).

Handles all 10 conditions via --condition flag:
  Base model (text output): agent_v, agent_h, pass1, f5_pass1
  SFT v2 (action output):   f1, f2, f3, f4, f5, f6

Uses vLLM OpenAI-compatible API with ThreadPoolExecutor for parallelism.
"""

import argparse
import base64
import json
import os
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO

import numpy as np
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Add parent dir to path for imports from sibling scripts
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP2_DIR = os.path.dirname(SCRIPT_DIR)
if EXP2_DIR not in sys.path:
    sys.path.insert(0, EXP2_DIR)

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
from cognitive_interference_vllm import (
    encode_image_base64,
    get_actions_for_domain,
    segment_by_subtask,
    prepare_gt,
)
from sample_cognitive_interference import assign_position_bucket

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_CONDITIONS = [
    "agent_v", "agent_h", "pass1", "f5_pass1",
    "f1", "f2", "f3", "f4", "f5", "f6",
]
TEXT_CONDITIONS = {"agent_v", "agent_h", "pass1", "f5_pass1"}
ACTION_CONDITIONS = {"f1", "f2", "f3", "f4", "f5", "f6"}

DOMAIN_NAMES = {
    "ppt": "PowerPoint",
    "excel": "Microsoft Excel",
    "word": "Microsoft Word",
}

# ---------------------------------------------------------------------------
# Prompt templates for base model conditions
# ---------------------------------------------------------------------------
AGENT_V_PROMPT = """Describe the current state of this {domain} screen. List all visible interactive elements with their names and locations. Note any open dialogs, selected items, or active menus. Be specific and exhaustive."""

AGENT_H_PROMPT = """Given this task: {task}

And these completed actions:
{history}

Analyze the current progress:
1. What has been accomplished?
2. What is the logical next category of action needed?
3. Are there signs the task is going wrong?

Output a concise progress analysis in 3-5 sentences."""

PASS1_PROMPT = """You are a helpful assistant analyzing a screenshot of {domain}.

Overall Task: {task}

Current Subtask: {subtask}

The history of actions are:
{history}

Before taking any action, think step by step:
1. What is the current state of the screen?
2. What has been accomplished so far?
3. What should be the next action and why?

Output your reasoning only. Do not output any action."""

F5_PASS1_PROMPT = """You are a helpful assistant analyzing a screenshot of {domain}.

Overall Task: {task}

Current Subtask: {subtask}

Visual analysis of the current screen:
{visual_description}

Task progress analysis:
{progress_analysis}

The history of actions are:
{history}

Before taking any action, think step by step:
1. Given the visual analysis and progress analysis above, what is the current state?
2. What has been accomplished so far?
3. What should be the next action and why?

Output your reasoning only. Do not output any action."""


# ---------------------------------------------------------------------------
# Data loading
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


def load_steps(data_root, trajectory_ids, c0_results_path):
    """Load all steps with GT data, screenshots, and C0 history.

    Returns list of step dicts ready for inference.
    """
    # Load trajectories
    trajectories = load_trajectories(data_root, trajectory_ids)
    print(f"  Loaded {len(trajectories)} trajectories, "
          f"{sum(len(t['steps']) for t in trajectories.values())} steps")

    # Load C0 results for history building
    with open(c0_results_path) as f:
        c0_data = json.load(f)

    c0_by_traj = {}
    for traj in c0_data["detailed_results"]:
        c0_by_traj[traj["trajectory_id"]] = traj

    # Build step list
    all_steps = []
    for traj_id, traj in trajectories.items():
        domain = traj["domain"]
        category = traj["category"]
        steps = traj["steps"]
        num_steps = len(steps)

        # Get C0 results for this trajectory (for history building)
        c0_traj = c0_by_traj.get(traj_id, {})
        c0_step_results = c0_traj.get("step_results", [])

        # Build subtask segments for history boundaries
        # segment_by_subtask returns (subtask_desc, [step_dicts])
        segments = segment_by_subtask(steps)

        # Map step index (int) → (subtask_idx, local_step_within_subtask, subtask_desc)
        step_subtask_map = {}
        # Build sample_id → step_index lookup
        sample_to_idx = {s["sample_id"]: i for i, s in enumerate(steps)}
        for seg_idx, (subtask_desc, seg_steps_list) in enumerate(segments):
            for local_idx, seg_step in enumerate(seg_steps_list):
                idx = sample_to_idx.get(seg_step["sample_id"])
                if idx is not None:
                    step_subtask_map[idx] = (seg_idx, local_idx, subtask_desc)

        # Map sample_id → C0 predicted action for history
        c0_by_sample = {}
        for sr in c0_step_results:
            c0_by_sample[sr["sample_id"]] = sr

        for step_idx, step in enumerate(steps):
            seg_idx, local_idx, subtask_desc = step_subtask_map.get(
                step_idx, (0, 0, step.get("subtask", "") or step["request"]))

            # Build history from C0's predicted actions within same subtask
            history_entries = []
            local_counter = 0
            for prev_idx in range(step_idx):
                prev_seg_idx = step_subtask_map.get(prev_idx, (None,))[0]
                if prev_seg_idx != seg_idx:
                    continue
                local_counter += 1
                prev_sample_id = steps[prev_idx]["sample_id"]
                prev_c0 = c0_by_sample.get(prev_sample_id, {})
                fn = prev_c0.get("predicted_function", "unknown")
                args = prev_c0.get("predicted_args", {})
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


def load_aux_outputs(aux_dir, condition_name):
    """Load JSONL output from a prior condition.

    Returns dict: sample_id -> text output.
    """
    path = os.path.join(aux_dir, f"{condition_name}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Auxiliary output not found: {path}")

    result = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line.strip())
            sid = d["sample_id"]
            result[sid] = d.get("text_output", "")
    print(f"  Loaded {len(result)} entries from {condition_name}.jsonl")
    return result


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt(condition, step, aux_data):
    """Build the prompt + messages for a given condition and step.

    Returns (messages, uses_image, temperature, n_samples).
    """
    domain = step["domain"]
    domain_name = DOMAIN_NAMES.get(domain.lower(), domain.capitalize())
    actions_text = get_actions_for_domain(domain)

    if condition == "agent_v":
        # Visual Agent: screenshot only -> UI description
        text = AGENT_V_PROMPT.format(domain=domain_name)
        return text, True, 0, 1

    elif condition == "agent_h":
        # History Agent: task + history (no screenshot)
        text = AGENT_H_PROMPT.format(
            task=step["request"],
            history=step["history_text"],
        )
        return text, False, 0, 1

    elif condition == "pass1":
        # Two-pass reasoning: screenshot + task + history -> reasoning
        text = PASS1_PROMPT.format(
            domain=domain_name,
            task=step["request"],
            subtask=step["subtask"],
            history=step["history_text"],
        )
        return text, True, 0, 1

    elif condition == "f5_pass1":
        # Enhanced reasoning with Agent V + Agent H outputs
        visual_desc = aux_data.get("agent_v", {}).get(step["sample_id"], "")
        progress = aux_data.get("agent_h", {}).get(step["sample_id"], "")
        text = F5_PASS1_PROMPT.format(
            domain=domain_name,
            task=step["request"],
            subtask=step["subtask"],
            visual_description=visual_desc or "(not available)",
            progress_analysis=progress or "(not available)",
            history=step["history_text"],
        )
        return text, True, 0, 1

    # --- SFT v2 action conditions ---
    # Start with standard SUBTASK_ISOLATED_USER_PROMPT, inject auxiliary text

    prefix = ""
    insert_after_task = ""
    insert_after_history = ""

    if condition == "f1":
        # Two-pass Serial: prepend pass1 reasoning
        reasoning = aux_data.get("pass1", {}).get(step["sample_id"], "")
        if reasoning:
            prefix = f"Reasoning about current state: {reasoning}\n\n"

    elif condition == "f2":
        # Visual Agent + Decision: append Agent V after history
        visual_desc = aux_data.get("agent_v", {}).get(step["sample_id"], "")
        if visual_desc:
            insert_after_history = f"\nCurrent screen elements: {visual_desc}\n"

    elif condition == "f3":
        # History Agent + Decision: append Agent H after task
        progress = aux_data.get("agent_h", {}).get(step["sample_id"], "")
        if progress:
            insert_after_task = f"\nTask progress: {progress}\n"

    elif condition == "f4":
        # Full Decomposition: Agent V + Agent H
        visual_desc = aux_data.get("agent_v", {}).get(step["sample_id"], "")
        progress = aux_data.get("agent_h", {}).get(step["sample_id"], "")
        if progress:
            insert_after_task = f"\nTask progress: {progress}\n"
        if visual_desc:
            insert_after_history = f"\nCurrent screen elements: {visual_desc}\n"

    elif condition == "f5":
        # Serial + Decomposition: Agent V + Agent H + F5 reasoning
        visual_desc = aux_data.get("agent_v", {}).get(step["sample_id"], "")
        progress = aux_data.get("agent_h", {}).get(step["sample_id"], "")
        reasoning = aux_data.get("f5_pass1", {}).get(step["sample_id"], "")
        if reasoning:
            prefix = f"Reasoning about current state: {reasoning}\n\n"
        if progress:
            insert_after_task = f"\nTask progress: {progress}\n"
        if visual_desc:
            insert_after_history = f"\nCurrent screen elements: {visual_desc}\n"

    elif condition == "f6":
        # Ensemble: standard prompt x3 at T=0.7
        user_prompt = SUBTASK_ISOLATED_USER_PROMPT.format(
            instruction=step["request"],
            subtask_description=step["subtask"],
            history=step["history_text"],
            actions=actions_text,
        )
        return user_prompt, True, 0.7, 3

    # Build the modified SUBTASK_ISOLATED_USER_PROMPT
    user_prompt = SUBTASK_ISOLATED_USER_PROMPT.format(
        instruction=step["request"],
        subtask_description=step["subtask"],
        history=step["history_text"],
        actions=actions_text,
    )

    # Inject auxiliary text at the right positions
    if insert_after_task:
        # Insert after "Overall Task:\n{instruction}\n"
        marker = f"Overall Task:\n{step['request']}\n"
        idx = user_prompt.find(marker)
        if idx >= 0:
            insert_pos = idx + len(marker)
            user_prompt = user_prompt[:insert_pos] + insert_after_task + user_prompt[insert_pos:]

    if insert_after_history:
        # Insert after the history section, before "The actions supported are:"
        marker = "The actions supported are:"
        idx = user_prompt.find(marker)
        if idx >= 0:
            user_prompt = user_prompt[:idx] + insert_after_history + "\n" + user_prompt[idx:]

    if prefix:
        user_prompt = prefix + user_prompt

    return user_prompt, True, 0, 1


# ---------------------------------------------------------------------------
# Single step inference
# ---------------------------------------------------------------------------

def run_step(client, model_name, condition, step, aux_data):
    """Run inference for one step. Returns result dict."""
    sample_id = step["sample_id"]
    t0 = time.time()

    try:
        prompt_text, uses_image, temperature, n_samples = build_prompt(
            condition, step, aux_data)
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

    # Determine max_tokens
    if condition in TEXT_CONDITIONS:
        max_tokens = 512
    else:
        max_tokens = 512

    # Run inference
    responses = []
    try:
        if n_samples > 1:
            # F6: multiple samples
            for _ in range(n_samples):
                result = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                responses.append(result.choices[0].message.content or "")
        else:
            result = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            responses.append(result.choices[0].message.content or "")
    except Exception as e:
        return {"sample_id": sample_id, "error": f"API call: {e}"}

    elapsed = time.time() - t0

    # Build result
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
        text_output = responses[0]
        control_test = step.get("control_test", "")
        hit = False
        if control_test and text_output:
            hit = control_test.lower() in text_output.lower()
        result_dict["text_output"] = text_output
        result_dict["control_test_hit"] = hit

    elif condition == "f6":
        # Ensemble: majority vote
        orig_w_step = step["orig_w"]
        orig_h_step = step["orig_h"]
        parsed = []
        for resp in responses:
            fn, args, st = parse_action(resp, orig_w_step, orig_h_step)
            parsed.append((fn, args, st, resp))

        # Majority vote on function name + status
        vote_keys = []
        for fn, args, st, _ in parsed:
            key = f"{fn}||{st}"
            vote_keys.append(key)

        counter = Counter(vote_keys)
        winner_key = counter.most_common(1)[0][0] if counter else "None||None"

        # Pick first response matching winner
        pred_fn, pred_args, pred_status, raw_resp = None, None, None, ""
        for fn, args, st, resp in parsed:
            if f"{fn}||{st}" == winner_key:
                pred_fn, pred_args, pred_status, raw_resp = fn, args, st, resp
                break

        # Compare with GT
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
            "raw_response": raw_resp,
            "all_responses": responses,
            "vote_distribution": dict(counter),
        })

    else:
        # Action conditions f1-f5: parse action, compare with GT
        response = responses[0]
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
    """Run all steps in parallel using ThreadPoolExecutor."""
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
        description="Multi-Agent Decomposition — Unified Inference")
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
                        help="Path to C0 results JSON (for building history)")
    parser.add_argument("--aux_dir", type=str, default=None,
                        help="Directory with prior conditions' JSONL outputs")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Results output directory")
    parser.add_argument("--max_workers", type=int, default=128,
                        help="ThreadPoolExecutor workers")
    parser.add_argument("--max_trajectories", type=int, default=None,
                        help="Limit number of trajectories (for testing)")
    args = parser.parse_args()

    condition = args.condition
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"Multi-Agent Decomposition — Condition: {condition}")
    print(f"Model: {args.model_name}")
    print(f"Workers: {args.max_workers}")
    print("=" * 60)

    # Load trajectory IDs
    with open(args.trajectory_ids) as f:
        traj_ids = json.load(f)
    if args.max_trajectories:
        traj_ids = traj_ids[:args.max_trajectories]
    print(f"Trajectories: {len(traj_ids)}")

    # Load steps
    print("Loading steps...")
    steps = load_steps(args.data_root, traj_ids, args.c0_results)
    print(f"Total steps: {len(steps)}")

    # Load auxiliary outputs if needed
    aux_data = {}
    deps = {
        "f5_pass1": ["agent_v", "agent_h"],
        "f1": ["pass1"],
        "f2": ["agent_v"],
        "f3": ["agent_h"],
        "f4": ["agent_v", "agent_h"],
        "f5": ["agent_v", "agent_h", "f5_pass1"],
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

    # Quick summary
    print(f"\n{'='*40} Summary {'='*40}")
    print(f"Condition: {condition}")
    print(f"Total steps: {len(results)}")
    n_errors = sum(1 for r in results if "error" in r)
    if n_errors:
        print(f"Errors: {n_errors}")
    if condition in TEXT_CONDITIONS:
        n_valid = sum(1 for r in results if r.get("text_output"))
        hit_rate = sum(1 for r in results if r.get("control_test_hit")) / len(results) if results else 0
        avg_len = np.mean([len(r.get("text_output", "")) for r in results if r.get("text_output")]) if n_valid else 0
        print(f"Valid outputs: {n_valid}/{len(results)}")
        print(f"Thought-hit rate: {hit_rate:.4f}")
        print(f"Avg output length: {avg_len:.0f} chars")
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
