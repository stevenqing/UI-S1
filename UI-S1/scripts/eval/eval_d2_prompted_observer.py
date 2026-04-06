#!/usr/bin/env python3
"""
Exp D2: Prompted Observer AR Evaluation

Same architecture as D1 but with a much more targeted Observer prompt that:
1. Explicitly asks for task-relevant state changes (not generic screenshot description)
2. Tracks which sub-goals are done vs remaining
3. Detects stuck/loop states
4. Domain-specific focus (e.g., Excel: which cell is selected, which menu is open)

Compares against D1 (generic Observer) and Eval A (no Observer) results.

Usage:
    python scripts/eval/eval_d2_prompted_observer.py \
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
# Prompts — the key difference from D1
# ---------------------------------------------------------------------------

# Domain-specific focus instructions
DOMAIN_FOCUS = {
    "excel": """Excel-specific: Report the active cell address (e.g., A1), which sheet tab is selected, whether any menu/ribbon/dialog is open, and what data is visible in the active area.""",
    "word": """Word-specific: Report what section of the document is visible, whether any dialog/menu is open, cursor position context, and any active selection.""",
    "ppt": """PowerPoint-specific: Report which slide number is being edited, whether in slide view or edit mode, which element (if any) is selected, and whether any menu/dialog is open.""",
}

OBSERVER_PROMPT_V2 = """You are a State Tracker for a GUI automation system. Your job is NOT to describe what the screenshot looks like — the action model can see the screenshot itself. Instead, track task-relevant state that the action model cannot easily infer.

Task: {instruction}

{last_action_section}

{domain_focus}

Report ONLY these items (be very concise, one line each):
1. LOCATION: What app view/dialog/menu is currently showing
2. CHANGE: What specifically changed after the last action (skip if first step)
3. PROGRESS: Which sub-steps of the task are done, which remain
4. WARNING: Any signs of being stuck (repeated actions, unexpected state, wrong menu) — write "none" if everything looks normal

Do NOT describe UI elements the model can see in the screenshot. Focus on WHAT CHANGED and WHERE WE ARE in the task."""

# State document format for action model — more structured than D1
ACTION_WITH_STATE_DOC_PROMPT_V2 = """You are a helpful assistant. Given a screenshot of the current screen, user instruction, and a state tracker's notes, you need to decide the next action to take.

The instruction is:
{instruction}

State tracker notes:
{state_document}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

Use the state tracker's notes to understand:
- What has already been accomplished (don't repeat completed sub-steps)
- What the immediate next sub-step should be
- Whether we're in the right place to perform the next action

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

Only **ONE** action should be taken at a time. If the instruction could apply to multiple elements, choose the most relevant one based on the context provided by the screenshot, state tracker notes, and task progress."""


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


def _call_v2(client, model, messages, max_tokens=4096):
    for retry in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.0, max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            if retry == 2:
                return ""
            time.sleep(2)
    return ""


def _call_v3(client, model, messages):
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
# Trajectory evaluation — Prompted Observer (D2)
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
    last_pred_function = None
    last_pred_args = None
    repeat_count = 0

    actions_str = {"word": SUPPORTED_ACTIONS_WORD, "excel": SUPPORTED_ACTIONS_EXCEL,
                   "ppt": SUPPORTED_ACTIONS_PPT}.get(domain, SUPPORTED_ACTIONS_WORD)
    coord_actions = {"click", "right_click", "double_click"}
    domain_focus = DOMAIN_FOCUS.get(domain, "")

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

        # --- Build last_action_section for Observer ---
        last_action_section = ""
        if step_idx > 0 and last_pred_function:
            action_desc = f"Last action: {last_pred_function}"
            if last_pred_args and last_pred_args.get("coordinate"):
                try:
                    c = last_pred_args["coordinate"]
                    action_desc += f" at ({float(c[0]):.0f}, {float(c[1]):.0f})"
                except (TypeError, ValueError, IndexError):
                    pass
            # Add other args
            for k, v in (last_pred_args or {}).items():
                if k != "coordinate" and v is not None:
                    action_desc += f", {k}={v}"

            # Detect repetition
            if last_pred_function == (step_results[-1]["pred_function"] if len(step_results) >= 2 else None):
                repeat_count += 1
            else:
                repeat_count = 0

            if repeat_count >= 2:
                action_desc += f"\n⚠ WARNING: The same action type has been repeated {repeat_count+1} times. The agent may be stuck."

            last_action_section = f"Previous step result:\n{action_desc}\nThe above action was taken in step {step_num-1}. Look at the screenshot to see what actually happened."
        elif step_idx == 0:
            last_action_section = "This is the first step. No previous actions have been taken."

        # --- Parallel: Observer + V3 grounding ---
        with ThreadPoolExecutor(max_workers=2) as pool:
            obs_prompt = OBSERVER_PROMPT_V2.format(
                instruction=step["request"],
                last_action_section=last_action_section,
                domain_focus=domain_focus,
            )
            obs_msgs = [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": obs_prompt},
            ]}]
            obs_future = pool.submit(_call_v2, v2_client, v2_model, obs_msgs, 256)

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

        # --- Build structured state document ---
        # Format: numbered entries with clear structure
        state_entries_formatted = []
        for i, entry in enumerate(state_document_entries):
            state_entries_formatted.append(f"Step {i+1}: {entry}")

        # Current observation
        if observer_desc:
            state_entries_formatted.append(f"Step {step_num} (current): {observer_desc.strip()}")

        if state_entries_formatted:
            state_doc_str = "\n".join(state_entries_formatted)
        else:
            state_doc_str = "(First step — no prior state)"

        # --- V2 action prediction with state document ---
        user_prompt = ACTION_WITH_STATE_DOC_PROMPT_V2.format(
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

        # Save for next iteration
        last_pred_function = pred_function
        last_pred_args = pred_args

        # Update state document — compact entry
        action_desc = ""
        if pred_function:
            action_desc = f" → {pred_function}"
            if pred_args:
                for k, v in pred_args.items():
                    if k == "coordinate":
                        try:
                            action_desc += f"({float(v[0]):.0f},{float(v[1]):.0f})"
                        except (TypeError, ValueError, IndexError):
                            pass
                    elif v is not None:
                        action_desc += f" {k}={v}"

        # Truncate observer desc for state doc (keep it compact)
        obs_compact = observer_desc.strip()[:200] if observer_desc else "no observation"
        entry = f"{obs_compact}{action_desc}"
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

    output_dir = PROJECT_ROOT / "outputs" / "eval_d2"
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
        results_path = PROJECT_ROOT / "outputs" / "eval_d2" / "trajectory_results.jsonl"

    results = []
    with open(results_path) as f:
        for line in f:
            results.append(json.loads(line))

    # Load Eval A and D1 for comparison
    eval_a_path = PROJECT_ROOT / "outputs" / "eval_a" / "trajectory_results.jsonl"
    eval_a = {}
    if eval_a_path.exists():
        with open(eval_a_path) as f:
            for line in f:
                d = json.loads(line)
                eval_a[d["trajectory_id"]] = d

    eval_d1_path = PROJECT_ROOT / "outputs" / "eval_d1" / "trajectory_results.jsonl"
    eval_d1 = {}
    if eval_d1_path.exists():
        with open(eval_d1_path) as f:
            for line in f:
                d = json.loads(line)
                # Only use new-format D1 results
                if "trajectory_success" in d:
                    eval_d1[d["trajectory_id"]] = d

    n = len(results)
    print(f"\n{'='*70}")
    print(f"  Exp D2: Prompted Observer AR Evaluation ({n} trajectories)")
    print(f"{'='*70}")

    d2_tsr = np.mean([r["trajectory_success"] for r in results])
    d2_prog = np.mean([r["progress_rate"] for r in results])

    # Match with Eval A
    matched_a = [r for r in results if r["trajectory_id"] in eval_a]
    if matched_a:
        a_tsr = np.mean([eval_a[r["trajectory_id"]]["b_trajectory_success"] for r in matched_a])
        a_prog = np.mean([eval_a[r["trajectory_id"]]["b_progress_rate"] for r in matched_a])
    else:
        a_tsr = a_prog = 0

    # Match with D1
    matched_d1 = [r for r in results if r["trajectory_id"] in eval_d1]
    if matched_d1:
        d1_tsr = np.mean([eval_d1[r["trajectory_id"]]["trajectory_success"] for r in matched_d1])
        d1_prog = np.mean([eval_d1[r["trajectory_id"]]["progress_rate"] for r in matched_d1])
    else:
        d1_tsr = d1_prog = 0

    print(f"\n  {'Condition':<40s} {'TSR':>7s} {'Prog':>7s}")
    print(f"  {'-'*40} {'-'*7} {'-'*7}")
    if matched_a:
        print(f"  {'Eval A: V2+V3 raw history':.<40s} {a_tsr:>7.1%} {a_prog:>7.1%}")
    if matched_d1:
        print(f"  {'D1: V2+V3 + generic Observer':.<40s} {d1_tsr:>7.1%} {d1_prog:>7.1%}")
    print(f"  {'D2: V2+V3 + prompted Observer':.<40s} {d2_tsr:>7.1%} {d2_prog:>7.1%}")
    if matched_a:
        print(f"  {'Delta vs Eval A':.<40s} {d2_tsr-a_tsr:>+7.1%} {d2_prog-a_prog:>+7.1%}")
    if matched_d1:
        print(f"  {'Delta vs D1':.<40s} {d2_tsr-d1_tsr:>+7.1%} {d2_prog-d1_prog:>+7.1%}")

    # Per-domain
    print(f"\n  Per-domain TSR:")
    by_domain = defaultdict(list)
    for r in results:
        by_domain[r["domain"]].append(r)

    print(f"  {'Domain':<10s} {'N':>5s} {'A':>7s} {'D1':>7s} {'D2':>7s} {'D2-A':>7s} {'D2-D1':>7s}")
    for domain in sorted(by_domain.keys()):
        subset = by_domain[domain]
        d2 = np.mean([r["trajectory_success"] for r in subset])
        m_a = [r for r in subset if r["trajectory_id"] in eval_a]
        a = np.mean([eval_a[r["trajectory_id"]]["b_trajectory_success"] for r in m_a]) if m_a else 0
        m_d1 = [r for r in subset if r["trajectory_id"] in eval_d1]
        d1 = np.mean([eval_d1[r["trajectory_id"]]["trajectory_success"] for r in m_d1]) if m_d1 else 0
        print(f"  {domain:<10s} {len(subset):>5d} {a:>7.1%} {d1:>7.1%} {d2:>7.1%} {d2-a:>+7.1%} {d2-d1:>+7.1%}")

    # By trajectory length
    print(f"\n  By trajectory length:")
    print(f"  {'Length':<10s} {'N':>5s} {'A':>7s} {'D1':>7s} {'D2':>7s} {'D2-A':>7s}")
    for lo, hi, label in [(1, 3, "Short"), (4, 7, "Med"), (8, 15, "Long"), (16, 100, "VLong")]:
        subset = [r for r in results if lo <= r["num_steps"] <= hi]
        if not subset:
            continue
        d2 = np.mean([r["trajectory_success"] for r in subset])
        m_a = [r for r in subset if r["trajectory_id"] in eval_a]
        a = np.mean([eval_a[r["trajectory_id"]]["b_trajectory_success"] for r in m_a]) if m_a else 0
        m_d1 = [r for r in subset if r["trajectory_id"] in eval_d1]
        d1 = np.mean([eval_d1[r["trajectory_id"]]["trajectory_success"] for r in m_d1]) if m_d1 else 0
        print(f"  {label:<10s} {len(subset):>5d} {a:>7.1%} {d1:>7.1%} {d2:>7.1%} {d2-a:>+7.1%}")

    # Head-to-head vs Eval A
    if matched_a:
        both = sum(1 for r in matched_a if r["trajectory_success"] and eval_a[r["trajectory_id"]]["b_trajectory_success"])
        a_only = sum(1 for r in matched_a if not r["trajectory_success"] and eval_a[r["trajectory_id"]]["b_trajectory_success"])
        d2_only = sum(1 for r in matched_a if r["trajectory_success"] and not eval_a[r["trajectory_id"]]["b_trajectory_success"])
        neither = sum(1 for r in matched_a if not r["trajectory_success"] and not eval_a[r["trajectory_id"]]["b_trajectory_success"])
        print(f"\n  Head-to-head vs Eval A ({len(matched_a)} matched):")
        print(f"    Both OK:              {both:>5d}")
        print(f"    Eval A only:          {a_only:>5d}")
        print(f"    D2 Observer wins:     {d2_only:>5d}")
        print(f"    Neither:              {neither:>5d}")
        if a_only > 0:
            print(f"    Win ratio (D2:A):     {d2_only}:{a_only} = {d2_only/a_only:.1f}:1")

    # Head-to-head vs D1
    if matched_d1:
        both = sum(1 for r in matched_d1 if r["trajectory_success"] and eval_d1[r["trajectory_id"]]["trajectory_success"])
        d1_only = sum(1 for r in matched_d1 if not r["trajectory_success"] and eval_d1[r["trajectory_id"]]["trajectory_success"])
        d2_only = sum(1 for r in matched_d1 if r["trajectory_success"] and not eval_d1[r["trajectory_id"]]["trajectory_success"])
        neither = sum(1 for r in matched_d1 if not r["trajectory_success"] and not eval_d1[r["trajectory_id"]]["trajectory_success"])
        print(f"\n  Head-to-head vs D1 ({len(matched_d1)} matched):")
        print(f"    Both OK:              {both:>5d}")
        print(f"    D1 only:              {d1_only:>5d}")
        print(f"    D2 wins:              {d2_only:>5d}")
        print(f"    Neither:              {neither:>5d}")
        if d1_only > 0:
            print(f"    Win ratio (D2:D1):    {d2_only}:{d1_only} = {d2_only/d1_only:.1f}:1")

    # Save summary
    output_dir = PROJECT_ROOT / "outputs" / "eval_d2"
    summary = {
        "n_trajectories": n,
        "d2_tsr": float(d2_tsr),
        "d2_avg_progress": float(d2_prog),
        "eval_a_tsr": float(a_tsr) if matched_a else None,
        "d1_tsr": float(d1_tsr) if matched_d1 else None,
        "delta_vs_a": float(d2_tsr - a_tsr) if matched_a else None,
        "delta_vs_d1": float(d2_tsr - d1_tsr) if matched_d1 else None,
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
