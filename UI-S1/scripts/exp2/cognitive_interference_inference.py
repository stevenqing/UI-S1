"""Cognitive Interference Hypothesis — Inference for Conditions A, B, D.

Condition A: Isolated UI State (screenshot only → describe UI)
Condition B: Isolated Action Planning (GT UI state + history, no screenshot → predict action)
Condition D: Chained (model-generated UI state from A + history, no screenshot → predict action)
Condition C: Extracted from existing always_greedy results (no inference needed)

Uses HF Transformers for consistency with verifier experiment.
"""

import argparse
import json
import os
import time
import traceback
from datetime import datetime

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# Reuse action parsing and comparison from verifier script
from verifier_ar_inference import (
    SUPPORTED_ACTIONS_WORD,
    SUPPORTED_ACTIONS_EXCEL,
    SUPPORTED_ACTIONS_PPT,
    parse_action,
    compare_actions,
    format_action_brief,
)

# ---------------------------------------------------------------------------
# Prompt Templates
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
# Build history for a step within its trajectory context
# ---------------------------------------------------------------------------

def build_step_history(step, greedy_by_traj):
    """Build AR-style compressed history for a step from greedy results.

    Looks up the same trajectory's prior steps from the greedy results.
    Uses subtask_isolated logic: only include steps within the same subtask segment.
    """
    traj_id = step["trajectory_id"]
    step_num = step["step_num"]

    # Find prior steps in this trajectory from greedy results
    traj_greedy = greedy_by_traj.get(traj_id, {})
    all_steps = sorted(
        traj_greedy.get("step_results", []),
        key=lambda s: s["step_num"],
    )

    # Find current step's subtask_idx for segment boundary
    current_subtask_idx = None
    for s in all_steps:
        if s["step_num"] == step_num:
            current_subtask_idx = s.get("subtask_idx", 0)
            break

    # Collect prior steps in the same subtask segment
    history_entries = []
    local_idx = 0
    for ps in all_steps:
        if ps["step_num"] >= step_num:
            break
        # Only include steps from the same subtask segment
        if current_subtask_idx is not None and ps.get("subtask_idx", 0) != current_subtask_idx:
            continue
        local_idx += 1
        fn = ps.get("predicted_function", "unknown")
        args = ps.get("predicted_args", {})
        brief = format_action_brief(fn, args)
        history_entries.append(f"Step {local_idx}: {brief}")

    return "\n".join(history_entries) if history_entries else "None"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_condition_a(model, processor, step, device):
    """Condition A: Isolated UI state description (screenshot only)."""
    domain_name = step["domain"].capitalize()
    if domain_name == "Ppt":
        domain_name = "PowerPoint"
    elif domain_name == "Excel":
        domain_name = "Microsoft Excel"
    elif domain_name == "Word":
        domain_name = "Microsoft Word"

    prompt = CONDITION_A_PROMPT.format(domain=domain_name)

    messages = [{"role": "user", "content": [
        {"type": "image", "image": step["screenshot_clean"]},
        {"type": "text", "text": prompt},
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    response = processor.decode(generated, skip_special_tokens=True)
    return response


def run_condition_b(model, processor, step, ui_state_desc, history_text, device):
    """Condition B/D: Action planning from UI state description (no screenshot)."""
    actions = get_actions_for_domain(step["domain"])
    subtask = step.get("subtask", "") or step["request"]

    prompt = CONDITION_B_PROMPT.format(
        domain=step["domain"].capitalize(),
        request=step["request"],
        subtask=subtask,
        ui_state_description=ui_state_desc,
        history=history_text,
        actions=actions,
    )

    # Text-only: no image
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    response = processor.decode(generated, skip_special_tokens=True)
    return response


def prepare_gt_data(step):
    """Prepare GT action data for comparison (same as verifier script)."""
    action = step["action"].copy()
    action_args = action.get("args", {}).copy()
    action["args"] = action_args

    img = Image.open(step["screenshot_clean"])
    orig_w, orig_h = img.size

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

    gt_fn = action.get("function", "")
    gt_args = action_args
    gt_status = step.get("status", "")

    return gt_fn, gt_args, gt_status, gt_rect, gt_rect_end, orig_w, orig_h


def main():
    parser = argparse.ArgumentParser(description="Cognitive Interference inference")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--steps_file", type=str, required=True,
                        help="Path to all_steps.json from prepare script")
    parser.add_argument("--greedy_results", type=str, required=True,
                        help="Path to verifier_always_greedy_results.json (for history + Condition C)")
    parser.add_argument("--conditions", type=str, nargs="+",
                        default=["A", "B", "D"],
                        help="Which conditions to run (A=UI state, B=action from GT state, D=chained A→B)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--shard", type=str, default=None,
                        help="Shard spec 'i/N' to split steps across N GPUs (e.g., '0/4')")
    args = parser.parse_args()

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    # Load model
    print(f"Loading model from {args.model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    # With device_map="auto", inputs auto-move to correct device
    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}")

    # Load all steps
    with open(args.steps_file) as f:
        sampled = json.load(f)

    # Shard if requested (for multi-GPU parallelism)
    if args.shard:
        shard_idx, n_shards = map(int, args.shard.split("/"))
        total = len(sampled)
        shard_size = (total + n_shards - 1) // n_shards
        start = shard_idx * shard_size
        end = min(start + shard_size, total)
        sampled = sampled[start:end]
        print(f"Shard {shard_idx}/{n_shards}: steps {start}-{end} ({len(sampled)} steps)")

    if args.max_samples:
        sampled = sampled[:args.max_samples]
    print(f"Total steps to process: {len(sampled)}")

    # Load greedy results for history and Condition C
    with open(args.greedy_results) as f:
        greedy_data = json.load(f)
    greedy_by_traj = {t["trajectory_id"]: t for t in greedy_data["detailed_results"]}
    greedy_by_step = {}
    for traj in greedy_data["detailed_results"]:
        for s in traj["step_results"]:
            greedy_by_step[s["sample_id"]] = s

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    conditions = [c.upper() for c in args.conditions]

    for i, step in enumerate(tqdm(sampled, desc="Cognitive Interference")):
        sample_id = step["sample_id"]
        start_time = time.time()

        result = {
            "sample_id": sample_id,
            "trajectory_id": step["trajectory_id"],
            "domain": step["domain"],
            "category": step["category"],
            "step_num": step["step_num"],
            "num_steps": step["num_steps"],
            "position_bucket": step["position_bucket"],
            "gt_thought": step["thought"],
            "gt_subtask": step.get("subtask", ""),
            "request": step["request"],
        }

        # Prepare GT for action comparison
        gt_fn, gt_args, gt_status, gt_rect, gt_rect_end, orig_w, orig_h = prepare_gt_data(step)
        result["gt_function"] = gt_fn
        result["gt_args"] = gt_args
        result["gt_status"] = gt_status

        # Build history from greedy results
        history_text = build_step_history(step, greedy_by_traj)

        # ---- Condition C: extract from greedy results ----
        greedy_step = greedy_by_step.get(sample_id, {})
        result["condition_c"] = {
            "success": greedy_step.get("success", False),
            "predicted_function": greedy_step.get("predicted_function"),
            "predicted_args": greedy_step.get("predicted_args"),
            "predicted_status": greedy_step.get("predicted_status"),
            "source": "existing_greedy_results",
        }

        # ---- Condition A: Isolated UI State ----
        if "A" in conditions:
            try:
                a_response = run_condition_a(model, processor, step, device)
                result["condition_a"] = {
                    "ui_state_description": a_response,
                    "execution_time": time.time() - start_time,
                }
            except Exception as e:
                print(f"  Condition A error on {sample_id}: {e}")
                result["condition_a"] = {
                    "ui_state_description": "",
                    "error": str(e),
                }

        # ---- Condition B: Isolated Action Planning (GT UI state) ----
        if "B" in conditions:
            try:
                t0 = time.time()
                gt_ui_state = step["thought"]  # GT thought as UI state description
                b_response = run_condition_b(
                    model, processor, step, gt_ui_state, history_text, device,
                )
                b_fn, b_args, b_status = parse_action(b_response, orig_w, orig_h)
                if gt_fn == "drag":
                    b_fm, b_am, b_sm = compare_actions(
                        b_fn, b_args, b_status, gt_fn, gt_args, gt_status,
                        gt_rect, gt_rect_end,
                    )
                else:
                    b_fm, b_am, b_sm = compare_actions(
                        b_fn, b_args, b_status, gt_fn, gt_args, gt_status, gt_rect,
                    )
                b_success = b_fm and b_am and b_sm
                result["condition_b"] = {
                    "success": b_success,
                    "function_match": b_fm,
                    "args_match": b_am,
                    "status_match": b_sm,
                    "predicted_function": b_fn,
                    "predicted_args": b_args,
                    "predicted_status": b_status,
                    "raw_response": b_response,
                    "execution_time": time.time() - t0,
                }
            except Exception as e:
                print(f"  Condition B error on {sample_id}: {e}")
                traceback.print_exc()
                result["condition_b"] = {"success": False, "error": str(e)}

        # ---- Condition D: Chained (A output → B input) ----
        if "D" in conditions and "A" in conditions:
            try:
                t0 = time.time()
                model_ui_state = result.get("condition_a", {}).get("ui_state_description", "")
                if not model_ui_state:
                    model_ui_state = "Unable to describe current UI state."
                d_response = run_condition_b(
                    model, processor, step, model_ui_state, history_text, device,
                )
                d_fn, d_args, d_status = parse_action(d_response, orig_w, orig_h)
                if gt_fn == "drag":
                    d_fm, d_am, d_sm = compare_actions(
                        d_fn, d_args, d_status, gt_fn, gt_args, gt_status,
                        gt_rect, gt_rect_end,
                    )
                else:
                    d_fm, d_am, d_sm = compare_actions(
                        d_fn, d_args, d_status, gt_fn, gt_args, gt_status, gt_rect,
                    )
                d_success = d_fm and d_am and d_sm
                result["condition_d"] = {
                    "success": d_success,
                    "function_match": d_fm,
                    "args_match": d_am,
                    "status_match": d_sm,
                    "predicted_function": d_fn,
                    "predicted_args": d_args,
                    "predicted_status": d_status,
                    "raw_response": d_response,
                    "execution_time": time.time() - t0,
                }
            except Exception as e:
                print(f"  Condition D error on {sample_id}: {e}")
                result["condition_d"] = {"success": False, "error": str(e)}

        results.append(result)

        # Progress log
        c_ok = "Y" if result.get("condition_c", {}).get("success") else "N"
        b_ok = "Y" if result.get("condition_b", {}).get("success") else "N"
        d_ok = "Y" if result.get("condition_d", {}).get("success") else "N"
        a_len = len(result.get("condition_a", {}).get("ui_state_description", ""))
        elapsed = time.time() - start_time
        print(f"  [{i+1}/{len(sampled)}] {sample_id} | C={c_ok} B={b_ok} D={d_ok} | "
              f"A={a_len}chars | {elapsed:.1f}s")

        # Periodic save
        if (i + 1) % 50 == 0:
            shard_suffix = f"_shard{args.shard.split('/')[0]}" if args.shard else ""
            _save_results(args.output_dir, results, conditions, shard_suffix)

    # Final save
    shard_suffix = f"_shard{args.shard.split('/')[0]}" if args.shard else ""
    _save_results(args.output_dir, results, conditions, shard_suffix)
    print(f"\nDone. {len(results)} steps processed.")

    # Quick summary
    for cond in ["condition_a", "condition_b", "condition_c", "condition_d"]:
        if cond == "condition_a":
            n_with = sum(1 for r in results if cond in r and r[cond].get("ui_state_description"))
            print(f"  {cond}: {n_with}/{len(results)} have descriptions")
        else:
            n_success = sum(1 for r in results if r.get(cond, {}).get("success", False))
            n_total = sum(1 for r in results if cond in r)
            acc = n_success / n_total if n_total else 0
            print(f"  {cond}: {n_success}/{n_total} = {acc:.4f}")


def _save_results(output_dir, results, conditions, shard_suffix=""):
    output = {
        "config": {
            "conditions": conditions,
            "n_samples": len(results),
            "timestamp": datetime.now().isoformat(),
        },
        "results": results,
    }
    path = os.path.join(output_dir, f"cognitive_interference_results{shard_suffix}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)


if __name__ == "__main__":
    main()
