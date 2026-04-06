"""Exp2f: Reset Mechanism Inference.

AR inference with L26 correctness probe monitoring. When the probe detects
consecutive failures (N steps with P(correct) < threshold), the action
history is reset — the model continues from an empty history for the
current subtask (Option C: no history dependency after reset).

Uses HF Transformers for hidden state access during generation.
Closely follows verifier_ar_inference.py's evaluate_trajectory() pattern.

Conditions:
  - reset:    SFT v2 AR with probe-monitored history reset
  - baseline: SFT v2 AR greedy (no probe, no reset) — for fair comparison
"""

import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime

import joblib
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP2_DIR = os.path.dirname(SCRIPT_DIR)
if EXP2_DIR not in sys.path:
    sys.path.insert(0, EXP2_DIR)

from verifier_ar_inference import (
    SUBTASK_ISOLATED_USER_PROMPT,
    SUPPORTED_ACTIONS_WORD,
    SUPPORTED_ACTIONS_EXCEL,
    SUPPORTED_ACTIONS_PPT,
    parse_action,
    compare_actions,
    format_action_brief,
    smart_resize,
)


# ──────────────────────────────────────────────────────────────────────
# Data loading (adapted from verifier_ar_inference.py)
# ──────────────────────────────────────────────────────────────────────

def load_trajectories(data_root, trajectory_ids=None):
    """Load trajectories from GUI-360 test dataset. Returns a list."""
    data_path = os.path.join(data_root, "data")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    id_set = set(trajectory_ids) if trajectory_ids else None
    trajectories = []

    for domain in sorted(os.listdir(data_path)):
        domain_path = os.path.join(data_path, domain)
        if not os.path.isdir(domain_path):
            continue
        for category in sorted(os.listdir(domain_path)):
            cat_path = os.path.join(domain_path, category, "success")
            if not os.path.isdir(cat_path):
                continue
            for jsonl_file in sorted(os.listdir(cat_path)):
                if not jsonl_file.endswith(".jsonl"):
                    continue
                file_stem = os.path.splitext(jsonl_file)[0]
                traj_id = f"{domain}_{category}_{file_stem}"

                if id_set and traj_id not in id_set:
                    continue

                file_path = os.path.join(cat_path, jsonl_file)
                try:
                    steps = []
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line_num, line in enumerate(f, 1):
                            if not line.strip():
                                continue
                            try:
                                data = json.loads(line.strip())
                            except json.JSONDecodeError:
                                continue

                            action = data["step"]["action"]
                            if action.get("function", "") == "drag" or not action.get("rectangle", {}):
                                continue

                            clean_img = os.path.join(
                                data_root, "image", domain, category,
                                data["step"]["screenshot_clean"],
                            )
                            if not os.path.exists(clean_img):
                                continue

                            status = data["step"]["status"]
                            if status == "OVERALL_FINISH":
                                status = "FINISH"
                            elif status == "FINISH":
                                status = "CONTINUE"

                            steps.append({
                                "sample_id": f"{traj_id}_{line_num}",
                                "line_num": line_num,
                                "request": data["request"],
                                "screenshot_clean": clean_img,
                                "thought": data["step"]["thought"],
                                "subtask": data["step"].get("subtask", ""),
                                "action": action,
                                "status": status,
                                "domain": domain,
                                "category": category,
                                "step_index": len(steps) + 1,
                            })

                    if steps:
                        trajectories.append({
                            "trajectory_id": traj_id,
                            "request": steps[0]["request"],
                            "domain": domain,
                            "category": category,
                            "steps": steps,
                        })
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue

    print(f"Loaded {len(trajectories)} trajectories, "
          f"{sum(len(t['steps']) for t in trajectories)} total steps")
    return trajectories


def segment_by_subtask(steps):
    """Group consecutive steps by subtask field. Returns (desc, [indices])."""
    if not steps:
        return []
    segments = []
    current_subtask = steps[0].get("subtask", "")
    current_indices = [0]
    for i in range(1, len(steps)):
        st = steps[i].get("subtask", "")
        if st != current_subtask and st:
            segments.append((current_subtask, current_indices))
            current_subtask = st
            current_indices = [i]
        else:
            current_indices.append(i)
    segments.append((current_subtask, current_indices))
    return segments


def get_actions_for_domain(domain):
    """Get supported actions text for a domain."""
    d = domain.lower()
    if d == "word":
        return SUPPORTED_ACTIONS_WORD
    elif d == "excel":
        return SUPPORTED_ACTIONS_EXCEL
    elif d == "ppt":
        return SUPPORTED_ACTIONS_PPT
    return SUPPORTED_ACTIONS_WORD


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


# ──────────────────────────────────────────────────────────────────────
# Threshold calibration
# ──────────────────────────────────────────────────────────────────────

def calibrate_threshold(probe, probe_data_dir):
    """Calibrate probe threshold using non-Pattern-B clean samples."""
    labels_path = os.path.join(probe_data_dir, "labels_v2.json")
    with open(labels_path) as f:
        labels = json.load(f)

    # Get clean non-Pattern-B samples with known correctness
    clean_non_pb = []
    for i, l in enumerate(labels):
        if not l["is_pattern_b"] and l["correctness_clean"] and l["step_correct"] is not None:
            clean_non_pb.append((i, int(l["step_correct"])))

    if not clean_non_pb:
        print("Warning: no clean non-Pattern-B samples for calibration, using default 0.5")
        return 0.5

    indices, y_true = zip(*clean_non_pb)
    indices = np.array(indices)
    y_true = np.array(y_true)

    # Load L26 hidden states
    layer = probe.get("layer", 26)
    X = np.load(os.path.join(probe_data_dir, f"layer_{layer}_last.npy"))
    X_sub = X[indices]

    # Apply probe pipeline
    X_scaled = probe["scaler"].transform(X_sub)
    X_pca = probe["pca"].transform(X_scaled)
    probs = probe["clf"].predict_proba(X_pca)[:, 1]  # P(correct)

    # Find threshold maximizing F1 for wrong-step detection
    best_f1 = 0
    best_threshold = 0.5
    for threshold in np.arange(0.30, 0.71, 0.01):
        pred_wrong = (probs < threshold).astype(int)
        actual_wrong = (1 - y_true)
        tp = ((pred_wrong == 1) & (actual_wrong == 1)).sum()
        fp = ((pred_wrong == 1) & (actual_wrong == 0)).sum()
        fn = ((pred_wrong == 0) & (actual_wrong == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    print(f"Calibrated threshold: {best_threshold:.2f} (F1={best_f1:.4f})")
    print(f"  Data: {len(clean_non_pb)} clean non-PB samples "
          f"({y_true.sum()} correct, {(1-y_true).sum()} wrong)")

    # Report performance at chosen threshold
    pred_wrong = probs < best_threshold
    actual_wrong = y_true == 0
    tp = (pred_wrong & actual_wrong).sum()
    fp = (pred_wrong & ~actual_wrong).sum()
    fn = (~pred_wrong & actual_wrong).sum()
    tn = (~pred_wrong & ~actual_wrong).sum()
    print(f"  At t={best_threshold:.2f}: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"  Wrong detection recall: {recall:.4f}, precision: {precision:.4f}")

    return best_threshold


# ──────────────────────────────────────────────────────────────────────
# Trajectory evaluation with reset mechanism
# ──────────────────────────────────────────────────────────────────────

def evaluate_trajectory(
    model, processor, probe, trajectory, mode, threshold, consecutive_limit,
):
    """Evaluate a single trajectory.

    mode='reset':    probe-monitored history reset
    mode='baseline': standard greedy (no probe, no reset)
    """
    from qwen_vl_utils import process_vision_info

    traj_id = trajectory["trajectory_id"]
    steps = trajectory["steps"]
    domain = trajectory["domain"]

    segments = segment_by_subtask(steps)

    all_step_results = []
    first_error_step = None
    total_resets = 0

    for seg_idx, (subtask_desc, step_indices) in enumerate(segments):
        compressed_history = []
        consecutive_probe_wrong = 0

        if not subtask_desc:
            subtask_desc = steps[step_indices[0]]["request"]

        for local_i, global_i in enumerate(step_indices):
            step = steps[global_i]
            start_time = time.time()
            step_num = global_i + 1

            try:
                clean_img_path = step["screenshot_clean"]
                img = Image.open(clean_img_path)
                original_width, original_height = img.size

                # Prepare GT
                gt_fn, gt_args, gt_status, gt_rect, gt_rect_end = prepare_gt(
                    step, original_width, original_height)

                # Build prompt
                actions_text = get_actions_for_domain(domain)
                history_text = "\n".join(compressed_history) if compressed_history else "None"
                user_prompt = SUBTASK_ISOLATED_USER_PROMPT.format(
                    instruction=step["request"],
                    subtask_description=subtask_desc,
                    history=history_text,
                    actions=actions_text,
                )

                messages = [{"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image", "image": clean_img_path},
                ]}]

                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text], images=image_inputs, videos=video_inputs,
                    padding=True, return_tensors="pt",
                ).to(model.device)

                input_len = inputs["input_ids"].shape[1]

                # ── Generate ──
                probe_prob = None
                reset_triggered = False

                if mode == "reset":
                    # Generate with hidden states for probe
                    with torch.no_grad():
                        gen_out = model.generate(
                            **inputs, max_new_tokens=512, do_sample=False,
                            output_hidden_states=True, return_dict_in_generate=True,
                        )

                    # Extract L26 hidden state (prefill, last token)
                    probe_layer = probe.get("layer", 26)
                    prefill_hs = gen_out.hidden_states[0][probe_layer + 1][0, -1, :].cpu().float().numpy()

                    # Apply probe
                    x = probe["scaler"].transform(prefill_hs.reshape(1, -1))
                    x = probe["pca"].transform(x)
                    probe_prob = float(probe["clf"].predict_proba(x)[0, 1])

                    generated_ids = gen_out.sequences[0, input_len:]
                    response = processor.decode(generated_ids, skip_special_tokens=True)

                    del gen_out
                    torch.cuda.empty_cache()

                    # Track consecutive probe failures
                    if probe_prob < threshold:
                        consecutive_probe_wrong += 1
                    else:
                        consecutive_probe_wrong = 0

                    # Check if reset should trigger (affects NEXT step's history)
                    if consecutive_probe_wrong >= consecutive_limit:
                        reset_triggered = True
                        total_resets += 1

                else:
                    # Baseline: greedy, no probe
                    with torch.no_grad():
                        output_ids = model.generate(
                            **inputs, max_new_tokens=512, do_sample=False,
                        )
                    generated_ids = output_ids[0, input_len:]
                    response = processor.decode(generated_ids, skip_special_tokens=True)

                # Parse action
                pred_function, pred_args, pred_status = parse_action(
                    response, original_width, original_height,
                )

                # Compare
                if gt_fn == "drag":
                    fn_match, args_match, status_match = compare_actions(
                        pred_function, pred_args, pred_status,
                        gt_fn, gt_args, gt_status, gt_rect, gt_rect_end,
                    )
                else:
                    fn_match, args_match, status_match = compare_actions(
                        pred_function, pred_args, pred_status,
                        gt_fn, gt_args, gt_status, gt_rect,
                    )

                success = fn_match and args_match and status_match
                exec_time = time.time() - start_time

                step_result = {
                    "sample_id": step["sample_id"],
                    "step_num": step_num,
                    "subtask_idx": seg_idx,
                    "subtask_description": subtask_desc,
                    "local_step_num": local_i + 1,
                    "success": success,
                    "function_match": fn_match,
                    "args_match": args_match,
                    "status_match": status_match,
                    "predicted_function": pred_function,
                    "predicted_args": pred_args,
                    "predicted_status": pred_status,
                    "ground_truth_function": gt_fn,
                    "ground_truth_args": gt_args,
                    "ground_truth_status": gt_status,
                    "ground_truth_rect": gt_rect,
                    "raw_model_output": response,
                    "execution_time": exec_time,
                    "error_message": None,
                }

                if mode == "reset":
                    step_result["probe_prob_correct"] = probe_prob
                    step_result["consecutive_probe_wrong"] = consecutive_probe_wrong
                    step_result["reset_triggered"] = reset_triggered
                    step_result["history_length_before"] = len(compressed_history)

                if not success and first_error_step is None:
                    first_error_step = step_num

                all_step_results.append(step_result)

                # ── Update history ──
                if reset_triggered:
                    # Reset: wipe history, start fresh from current step
                    compressed_history = []
                    consecutive_probe_wrong = 0
                    brief = format_action_brief(pred_function, pred_args)
                    compressed_history.append(f"Step {local_i + 1}: {brief}")
                else:
                    brief = format_action_brief(pred_function, pred_args)
                    compressed_history.append(f"Step {local_i + 1}: {brief}")

                status_char = "\u2713" if success else "\u2717"
                probe_info = f" P(c)={probe_prob:.2f} cw={consecutive_probe_wrong}" if probe_prob is not None else ""
                reset_info = " RESET!" if reset_triggered else ""
                print(f"  [{traj_id}] seg{seg_idx+1} step {local_i+1}/{len(step_indices)} "
                      f"(g{step_num}): {pred_function} vs {gt_fn} ({status_char})"
                      f"{probe_info}{reset_info}")

            except Exception as e:
                exec_time = time.time() - start_time
                print(f"  ERROR at step {step_num}: {traceback.format_exc()}")
                all_step_results.append({
                    "sample_id": step["sample_id"],
                    "step_num": step_num,
                    "subtask_idx": seg_idx,
                    "success": False,
                    "function_match": False,
                    "args_match": False,
                    "status_match": False,
                    "error_message": str(e),
                    "execution_time": exec_time,
                })
                if first_error_step is None:
                    first_error_step = step_num
                compressed_history.append(f"Step {local_i + 1}: Error occurred")

    # Trajectory-level metrics
    num_correct = sum(1 for s in all_step_results if s.get("success", False))
    n_total = len(steps)
    traj_success = num_correct == n_total and len(all_step_results) == n_total

    if first_error_step is not None:
        seq_progress = (first_error_step - 1) / n_total
    else:
        seq_progress = 1.0

    scattered_progress = num_correct / n_total if n_total > 0 else 0.0

    return {
        "trajectory_id": traj_id,
        "num_steps": n_total,
        "trajectory_success": traj_success,
        "progress_rate": seq_progress,
        "scattered_progress_rate": scattered_progress,
        "first_error_step": first_error_step,
        "domain": trajectory["domain"],
        "category": trajectory["category"],
        "total_resets": total_resets,
        "step_results": all_step_results,
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exp2f: Reset Mechanism Inference")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to SFT v2 HF model")
    parser.add_argument("--data_root", type=str, required=True,
                        help="GUI-360 test data root")
    parser.add_argument("--probe_path", type=str, default=None,
                        help="Path to correctness_probe_L26.pkl (required for reset mode)")
    parser.add_argument("--probe_data_dir", type=str, default=None,
                        help="Dir with .npy files for threshold calibration")
    parser.add_argument("--trajectory_ids", type=str, default=None,
                        help="Path to JSON file with trajectory ID list")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["reset", "baseline"],
                        help="'reset' = probe-monitored reset, 'baseline' = greedy")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Probe threshold (auto-calibrate if not given)")
    parser.add_argument("--consecutive_limit", type=int, default=3,
                        help="Consecutive failures before reset (default: 3)")
    parser.add_argument("--max_trajectories", type=int, default=None,
                        help="Max trajectories to process (for debugging)")
    args = parser.parse_args()

    if args.mode == "reset" and not args.probe_path:
        parser.error("--probe_path is required for reset mode")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"Exp2f: Reset Mechanism Inference — Mode: {args.mode}")
    print(f"Model: {args.model_path}")
    print("=" * 60)

    # Load probe
    probe = None
    threshold = 0.5
    if args.mode == "reset":
        probe = joblib.load(args.probe_path)
        print(f"Probe loaded: layer={probe.get('layer', 26)}, "
              f"train_acc={probe.get('train_accuracy', 'N/A')}")

        # Calibrate or set threshold
        if args.threshold is not None:
            threshold = args.threshold
            print(f"Using specified threshold: {threshold}")
        elif args.probe_data_dir:
            threshold = calibrate_threshold(probe, args.probe_data_dir)
        else:
            print(f"Using default threshold: {threshold}")

        print(f"Consecutive limit: {args.consecutive_limit}")

    # Load trajectories
    traj_ids = None
    if args.trajectory_ids:
        with open(args.trajectory_ids) as f:
            traj_ids = json.load(f)
        print(f"Filtering to {len(traj_ids)} trajectory IDs")

    trajectories = load_trajectories(args.data_root, traj_ids)
    if args.max_trajectories:
        trajectories = trajectories[:args.max_trajectories]
        print(f"Limited to {len(trajectories)} trajectories")

    # Load model
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    print(f"Loading model from {args.model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    print("Model loaded")

    # Run inference
    all_results = []
    total_steps = 0
    total_correct = 0
    total_resets = 0

    for traj_idx, trajectory in enumerate(tqdm(trajectories, desc=f"[{args.mode}] Trajectories")):
        traj_result = evaluate_trajectory(
            model, processor, probe, trajectory, args.mode,
            threshold, args.consecutive_limit,
        )
        all_results.append(traj_result)

        n_steps = traj_result["num_steps"]
        n_correct = sum(1 for s in traj_result["step_results"] if s.get("success", False))
        total_steps += n_steps
        total_correct += n_correct
        total_resets += traj_result.get("total_resets", 0)

        print(f"  [{traj_idx+1}/{len(trajectories)}] {traj_result['trajectory_id']}: "
              f"TSR={'Y' if traj_result['trajectory_success'] else 'N'}, "
              f"progress={traj_result['progress_rate']:.2f}, "
              f"scattered={traj_result['scattered_progress_rate']:.2f}, "
              f"steps={n_correct}/{n_steps}"
              + (f", resets={traj_result['total_resets']}" if args.mode == 'reset' else ''))

    # Compute overall statistics
    n_traj = len(all_results)
    stats = {
        "mode": args.mode,
        "num_trajectories": n_traj,
        "num_steps": total_steps,
        "trajectory_success_rate": sum(1 for t in all_results if t["trajectory_success"]) / n_traj if n_traj else 0,
        "avg_progress_rate": float(np.mean([t["progress_rate"] for t in all_results])) if n_traj else 0,
        "avg_scattered_progress_rate": float(np.mean([t["scattered_progress_rate"] for t in all_results])) if n_traj else 0,
        "step_success_rate": total_correct / total_steps if total_steps else 0,
    }

    # Probe and reset analysis (reset mode only)
    if args.mode == "reset":
        stats["total_resets"] = total_resets
        stats["avg_resets_per_trajectory"] = total_resets / n_traj if n_traj else 0

        all_probe_probs = []
        all_step_success = []
        all_reset_flags = []
        for t in all_results:
            for s in t["step_results"]:
                pp = s.get("probe_prob_correct")
                if pp is not None:
                    all_probe_probs.append(pp)
                    all_step_success.append(s["success"])
                    all_reset_flags.append(s.get("reset_triggered", False))

        if all_probe_probs:
            probs = np.array(all_probe_probs)
            successes = np.array(all_step_success)
            resets = np.array(all_reset_flags)

            pred_wrong = probs < threshold
            actual_wrong = ~successes

            tp = (pred_wrong & actual_wrong).sum()
            fp = (pred_wrong & ~actual_wrong).sum()
            fn = (~pred_wrong & actual_wrong).sum()

            stats["probe_analysis"] = {
                "threshold": float(threshold),
                "consecutive_limit": args.consecutive_limit,
                "avg_prob_correct": float(probs.mean()),
                "avg_prob_when_correct": float(probs[successes].mean()) if successes.any() else None,
                "avg_prob_when_wrong": float(probs[~successes].mean()) if (~successes).any() else None,
                "wrong_detection_recall": float(tp / (tp + fn)) if (tp + fn) > 0 else None,
                "wrong_detection_precision": float(tp / (tp + fp)) if (tp + fp) > 0 else None,
                "total_resets": int(resets.sum()),
            }

            # Accuracy of steps immediately AFTER reset
            after_reset_correct = []
            for t in all_results:
                was_reset = False
                for s in t["step_results"]:
                    if was_reset:
                        after_reset_correct.append(s.get("success", False))
                        was_reset = False
                    if s.get("reset_triggered", False):
                        was_reset = True
            if after_reset_correct:
                stats["probe_analysis"]["steps_after_reset_accuracy"] = float(np.mean(after_reset_correct))
                stats["probe_analysis"]["n_steps_after_reset"] = len(after_reset_correct)

    # Domain breakdown
    domain_stats = defaultdict(lambda: {"n": 0, "correct": 0})
    for t in all_results:
        for s in t["step_results"]:
            d = t["domain"]
            domain_stats[d]["n"] += 1
            if s.get("success"):
                domain_stats[d]["correct"] += 1
    stats["by_domain"] = {
        d: {"n": v["n"], "step_accuracy": v["correct"] / v["n"] if v["n"] > 0 else 0}
        for d, v in domain_stats.items()
    }

    # Save results
    output = {
        "config": {
            "mode": args.mode,
            "model_path": args.model_path,
            "data_root": args.data_root,
            "probe_path": args.probe_path if args.mode == "reset" else None,
            "threshold": float(threshold) if args.mode == "reset" else None,
            "consecutive_limit": args.consecutive_limit if args.mode == "reset" else None,
            "num_trajectories": n_traj,
            "timestamp": datetime.now().isoformat(),
        },
        "statistics": stats,
        "trajectory_results": [
            {
                "trajectory_id": t["trajectory_id"],
                "num_steps": t["num_steps"],
                "trajectory_success": t["trajectory_success"],
                "progress_rate": t["progress_rate"],
                "scattered_progress_rate": t["scattered_progress_rate"],
                "first_error_step": t["first_error_step"],
                "domain": t["domain"],
                "category": t["category"],
                "total_resets": t.get("total_resets", 0),
            }
            for t in all_results
        ],
        "detailed_results": all_results,
    }

    output_file = os.path.join(args.output_dir, f"reset_{args.mode}_results.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")

    # Summary
    print(f"\n{'='*50} Summary {'='*50}")
    print(f"Mode: {args.mode}")
    if args.mode == "reset":
        print(f"Threshold: {threshold:.2f}, Consecutive limit: {args.consecutive_limit}")
    print(f"TSR: {stats['trajectory_success_rate']:.4f}")
    print(f"Step Accuracy: {stats['step_success_rate']:.4f}")
    print(f"Avg Progress: {stats['avg_progress_rate']:.4f}")
    print(f"Avg Scattered Progress: {stats['avg_scattered_progress_rate']:.4f}")
    if args.mode == "reset":
        print(f"Total Resets: {total_resets} ({total_resets/n_traj:.1f} per trajectory)")
        pa = stats.get("probe_analysis", {})
        if pa.get("steps_after_reset_accuracy") is not None:
            print(f"Post-reset step accuracy: {pa['steps_after_reset_accuracy']:.4f} "
                  f"({pa['n_steps_after_reset']} steps)")
    for d in sorted(stats.get("by_domain", {}).keys()):
        ds = stats["by_domain"][d]
        print(f"  {d}: step_acc={ds['step_accuracy']:.4f} (n={ds['n']})")


if __name__ == "__main__":
    main()
