#!/usr/bin/env python3
"""
Exp D1: Zero-Shot Observer AR Evaluation (Optimized)

Adds a structured Observer to V2+V3 pipeline. Compares against
Eval A results (already saved) rather than re-running baseline.

Architecture per step:
  1. Observer (V2) + V3 grounding — run in parallel
  2. V2 action prediction (using state document context)
  3. Replace coordinate with V3 grounding result

Usage:
    python scripts/eval/eval_d1_observer_ar.py \
        --v2_endpoint http://localhost:19816/v1 \
        --v3_endpoint http://localhost:19815/v1
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
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
OBSERVER_PROMPT = """You are an Observer agent in a GUI automation system. Describe the current screen state concisely.

Task: {instruction}
{last_action_info}
Describe: (1) What app/dialog is showing (2) Key UI elements visible (3) What changed from last step (4) Task progress estimate. Be brief."""

ACTION_WITH_STATE_DOC_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction, and a state document maintained by an observer, you need to decide the next action to take.

The instruction is:
{instruction}

The observer's state document:
{state_document}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

First, explain your reasoning process—describe how you analyze the screenshot, understand the current state based on the observer's notes, and determine what action should be taken next.

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

Only **ONE** action should be taken at a time. If the instruction could apply to multiple elements, choose the most relevant one based on the context provided by the screenshot, observer notes, and task progress."""


# ---------------------------------------------------------------------------
# Data loading
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
                except Exception as e:
                    continue


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
    match2 = re.search(r'\{\s*"function":\s*"([^"]*)",\s*"args":\s*(\{.*?\}),\s*"status":\s*"([^"]+)"\s*\}', response, re.DOTALL)
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
                if not coord_match: all_match = False
            elif key in ("start_coordinate", "end_coordinate"):
                continue
            else:
                p_str = str(p_norm.get(key)).lower() if p_norm.get(key) is not None else "none"
                g_str = str(g_norm.get(key)).lower() if g_norm.get(key) is not None else "none"
                if g_str and g_str != "none" and p_str != g_str: all_match = False
        args_match = all_match
    return func_match, args_match, status_match, coord_match


def _call_v2(client, model, messages, max_tokens=4096):
    """Single V2 call with retry."""
    for retry in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.0, max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
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
        except Exception as e:
            if retry == 2:
                return ""
            time.sleep(2)
    return ""


# ---------------------------------------------------------------------------
# Trajectory evaluation — Observer only (no Condition A re-run)
# ---------------------------------------------------------------------------
def evaluate_trajectory(trajectory, v2_client, v2_model, v3_client, v3_model,
                        stop_on_error=True):
    from PIL import Image

    traj_id = trajectory["trajectory_id"]
    steps = trajectory["steps"]
    domain = trajectory["domain"]

    state_document_entries = []
    step_results = []
    stopped_early = False

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
                "v3_coord": None, "execution_time": 0,
            })
            continue

        data_url, orig_wh, resized_wh = preprocess_image(clean_img)

        # --- Parallel: Observer + V3 grounding ---
        with ThreadPoolExecutor(max_workers=2) as pool:
            # Observer call
            last_action_info = ""
            if step_idx > 0:
                last_action_info = "\nThe last action taken was described in the state document."
            obs_prompt = OBSERVER_PROMPT.format(
                instruction=step["request"], last_action_info=last_action_info)
            obs_msgs = [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": obs_prompt},
            ]}]
            obs_future = pool.submit(_call_v2, v2_client, v2_model, obs_msgs, 256)

            # V3 grounding (if coordinate-based action expected)
            v3_future = None
            if step.get("thought"):
                grounding_text = GROUNDING_PROMPT.format(instruction=step["thought"])
                v3_msgs = [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": grounding_text},
                ]}]
                v3_future = pool.submit(_call_v3, v3_client, v3_model, v3_msgs)

            observer_desc = obs_future.result()
            v3_coord = None
            if v3_future:
                v3_text = v3_future.result()
                v3_coord = parse_coordinate_response(v3_text)
                if v3_coord is not None:
                    v3_coord = transform_coord_to_original(v3_coord, orig_wh, resized_wh)

        # --- V2 action prediction with state document ---
        state_doc_str = "\n\n".join(
            f"[Step {i+1}]\n{entry}" for i, entry in enumerate(state_document_entries)
        )
        if observer_desc:
            current_obs = f"\n\n[Current (Step {step_num})]\n{observer_desc}"
            state_doc_str = (state_doc_str + current_obs) if state_doc_str else current_obs.strip()
        if not state_doc_str:
            state_doc_str = "(No prior observations — this is the first step)"

        user_prompt = ACTION_WITH_STATE_DOC_PROMPT.format(
            instruction=step["request"], state_document=state_doc_str, actions=actions_str)
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": user_prompt},
        ]}]

        v2_response = _call_v2(v2_client, v2_model, messages)
        pred_function, pred_args, pred_status = parse_action(v2_response)

        # Transform V2 coordinates
        if pred_args and pred_args.get("coordinate"):
            coord = pred_args["coordinate"]
            if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                try:
                    pred_args["coordinate"] = transform_coord_to_original(
                        [float(coord[0]), float(coord[1])], orig_wh, resized_wh)
                except (TypeError, ValueError):
                    pass

        # Replace with V3 coordinate if available
        if v3_coord is not None and pred_args is not None and pred_function in coord_actions:
            dual_args = pred_args.copy()
            dual_args["coordinate"] = v3_coord
            fm, am, sm, cm = compare_actions(
                pred_function, dual_args, pred_status,
                gt_function, gt_args, gt_status, gt_rect)
        else:
            fm, am, sm, cm = compare_actions(
                pred_function, pred_args, pred_status,
                gt_function, gt_args, gt_status, gt_rect)
        success = fm and am and sm

        elapsed = time.time() - t0
        step_results.append({
            "sample_id": sample_id, "step_num": step_num,
            "gt_function": gt_function, "pred_function": pred_function,
            "success": success, "func_match": fm, "args_match": am,
            "coord_match": cm, "status_match": sm,
            "v3_coord": v3_coord, "execution_time": elapsed,
        })

        # Update state document
        action_desc = ""
        if pred_function:
            action_desc = f"\nAction taken: {pred_function}"
            if pred_args and pred_args.get("coordinate"):
                try:
                    c = pred_args["coordinate"]
                    action_desc += f" at ({float(c[0]):.0f}, {float(c[1]):.0f})"
                except (TypeError, ValueError, IndexError):
                    pass
        entry = (observer_desc[:300] + action_desc) if observer_desc else f"Action: {pred_function or 'unknown'}"
        state_document_entries.append(entry)

        sym = "O" if success else "X"
        print(f"  S{step_num}:{sym}({pred_function}) ", end="", flush=True)

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

    output_dir = PROJECT_ROOT / "outputs" / "eval_d1"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "trajectory_results.jsonl"

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
    v2_model = v2_client.models.list().data[0].id
    v3_model = v3_client.models.list().data[0].id
    print(f"V2: {v2_model}, V3: {v3_model}")

    n_done = len(completed)
    n_total = len(trajectories)
    t_start = time.time()

    def _eval_one(traj):
        return evaluate_trajectory(traj, v2_client, v2_model, v3_client, v3_model)

    # Process trajectories with parallelism
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
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
                      f"→ {status} ({result['progress_rate']:.0%}) [{rate:.0f}/hr]")
            except Exception as e:
                print(f"\n[{n_done}/{n_total}] {traj['trajectory_id']} ERROR: {e}")

    print(f"\nDone. Results at {results_path}")


def analyze_results(results_path=None):
    if results_path is None:
        results_path = PROJECT_ROOT / "outputs" / "eval_d1" / "trajectory_results.jsonl"

    # Load D1 results
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
    print(f"  Exp D1: Observer AR Evaluation Results ({n} trajectories)")
    print(f"{'='*70}")

    d1_tsr = np.mean([r["trajectory_success"] for r in results])
    d1_prog = np.mean([r["progress_rate"] for r in results])

    # Match with Eval A
    matched = [r for r in results if r["trajectory_id"] in eval_a]
    if matched:
        a_tsr = np.mean([eval_a[r["trajectory_id"]]["b_trajectory_success"] for r in matched])
        a_prog = np.mean([eval_a[r["trajectory_id"]]["b_progress_rate"] for r in matched])
    else:
        a_tsr = a_prog = 0

    print(f"\n  {'Condition':<35s} {'TSR':>7s} {'Avg Prog':>9s}")
    print(f"  {'-'*35} {'-'*7} {'-'*9}")
    print(f"  {'Eval A: V2+V3 raw history':.<35s} {a_tsr:>7.1%} {a_prog:>9.3f}")
    print(f"  {'D1: V2+V3 + Observer':.<35s} {d1_tsr:>7.1%} {d1_prog:>9.3f}")
    if matched:
        print(f"  {'Delta':.<35s} {d1_tsr-a_tsr:>+7.1%} {d1_prog-a_prog:>+9.3f}")

    # Per-domain
    print(f"\n  Per-domain TSR:")
    by_domain = defaultdict(list)
    for r in results:
        by_domain[r["domain"]].append(r)

    print(f"  {'Domain':<10s} {'N':>5s} {'A_TSR':>7s} {'D1_TSR':>7s} {'Delta':>7s}")
    for domain in sorted(by_domain.keys()):
        subset = by_domain[domain]
        d1 = np.mean([r["trajectory_success"] for r in subset])
        m = [r for r in subset if r["trajectory_id"] in eval_a]
        a = np.mean([eval_a[r["trajectory_id"]]["b_trajectory_success"] for r in m]) if m else 0
        print(f"  {domain:<10s} {len(subset):>5d} {a:>7.1%} {d1:>7.1%} {d1-a:>+7.1%}")

    # By length
    print(f"\n  By trajectory length:")
    for lo, hi, label in [(1,3,"Short"), (4,7,"Med"), (8,15,"Long"), (16,100,"VLong")]:
        subset = [r for r in results if lo <= r["num_steps"] <= hi]
        if not subset: continue
        d1 = np.mean([r["trajectory_success"] for r in subset])
        m = [r for r in subset if r["trajectory_id"] in eval_a]
        a = np.mean([eval_a[r["trajectory_id"]]["b_trajectory_success"] for r in m]) if m else 0
        print(f"  {label:<10s} {len(subset):>5d} {a:>7.1%} {d1:>7.1%} {d1-a:>+7.1%}")

    # Head-to-head
    if matched:
        both = sum(1 for r in matched if r["trajectory_success"] and eval_a[r["trajectory_id"]]["b_trajectory_success"])
        a_only = sum(1 for r in matched if not r["trajectory_success"] and eval_a[r["trajectory_id"]]["b_trajectory_success"])
        b_only = sum(1 for r in matched if r["trajectory_success"] and not eval_a[r["trajectory_id"]]["b_trajectory_success"])
        neither = sum(1 for r in matched if not r["trajectory_success"] and not eval_a[r["trajectory_id"]]["b_trajectory_success"])
        print(f"\n  Head-to-head ({len(matched)} matched):")
        print(f"    Both OK:              {both:>5d}")
        print(f"    Eval A only:          {a_only:>5d}")
        print(f"    Observer wins:        {b_only:>5d}")
        print(f"    Neither:              {neither:>5d}")

    # Save
    output_dir = PROJECT_ROOT / "outputs" / "eval_d1"
    summary = {
        "n_trajectories": n,
        "d1_tsr": float(d1_tsr),
        "d1_avg_progress": float(d1_prog),
        "eval_a_tsr": float(a_tsr) if matched else None,
        "delta_tsr": float(d1_tsr - a_tsr) if matched else None,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_dir / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2_endpoint", default="http://localhost:19816/v1")
    parser.add_argument("--v3_endpoint", default="http://localhost:19815/v1")
    parser.add_argument("--max_trajectories", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--analyze_only", action="store_true")
    args = parser.parse_args()

    if args.analyze_only:
        analyze_results()
    else:
        run_evaluation(args)
        analyze_results()


if __name__ == "__main__":
    main()
