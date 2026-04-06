#!/usr/bin/env python3
"""
Binding Validation for AndroidControl (vLLM-based).

Tests whether fixing cross-modal binding with oracle info improves AC accuracy.

Conditions:
  baseline — standard inference
  redbox   — draw red bbox on GT target in screenshot
  texthint — add step_instruction hint to prompt
  crop     — crop screenshot around GT coordinate (40% area)

Uses vLLM server (OpenAI-compatible) with AC-native message format.
Evaluates with evaluate_android_control_action() (type_match, extract_match).

Usage:
  python eval_binding_validation_ac.py \
      --model_name Qwen2.5-VL-7B \
      --dataset evaluation/dataset/android_control_evaluation_std.jsonl \
      --condition all \
      --n_samples 500 \
      --output_dir evaluation/results/ac_binding_val_base
"""

import argparse
import copy
import json
import os
import re
import sys
import time
import traceback
import numpy as np
from collections import defaultdict
from PIL import Image, ImageDraw

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts', 'eval', 'ac'))

from scripts.eval.ac.ac_utils import (
    load_ac_trajectories,
    init_format,
    safe_parse_response,
    evaluate_android_control_action,
)
from evaluation.qwenvl_utils import (
    call_mobile_agent_vllm,
    find_last_image_ele,
)
from x.qwen.image import make_qwen_image_item, smart_resize


# ═══════════════════════════════════════════════════════════════════════
# Image modification (same as eval_binding_validation.py)
# ═══════════════════════════════════════════════════════════════════════

def draw_red_box(image, bbox):
    """Draw red bounding box. bbox = [x1, y1, x2, y2]."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = bbox
    for offset in range(4):
        draw.rectangle(
            [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
            outline='red')
    return img


def crop_around_target(image, gt_coord, crop_ratio=0.4):
    """Crop image to region around GT target coordinate."""
    w, h = image.size
    cx, cy = int(gt_coord[0]), int(gt_coord[1])

    half_w = int(w * crop_ratio / 2)
    half_h = int(h * crop_ratio / 2)
    half_w = max(half_w, 100)
    half_h = max(half_h, 100)

    x1 = max(0, cx - half_w)
    y1 = max(0, cy - half_h)
    x2 = min(w, cx + half_w)
    y2 = min(h, cy + half_h)

    cropped = image.crop((x1, y1, x2, y2))
    return cropped, (x1, y1, x2, y2)


# ═══════════════════════════════════════════════════════════════════════
# Data preparation
# ═══════════════════════════════════════════════════════════════════════

def extract_click_steps(episodes, image_root):
    """Extract individual click steps with non-empty candidate_bbox."""
    samples = []
    for ep in episodes:
        goal = ep["goal"]
        episode_id = ep.get("episode_id", 0)
        for si, step in enumerate(ep["steps"]):
            ac = step["action_content"]
            if ac["action"] != "click":
                continue
            check = step.get("check_options", {})
            candidate_bboxes = check.get("candidate_bbox", [])
            if not candidate_bboxes:
                continue
            gt_coord = ac.get("coordinate") or check.get("coordinate")
            if not gt_coord or len(gt_coord) != 2:
                continue

            screenshot = step["screenshot"]
            if not os.path.exists(screenshot):
                continue

            samples.append({
                "episode_id": episode_id,
                "step_idx": si,
                "goal": goal,
                "step_instruction": step.get("step_instruction", ""),
                "screenshot": screenshot,
                "action_content": ac,
                "check_options": check,
                "gt_coord": gt_coord,
                "candidate_bboxes": candidate_bboxes,
                "episode_steps": ep["steps"],
            })
    return samples


def find_best_bbox(candidate_bboxes, gt_coord):
    """Find the candidate bbox containing gt_coord (smallest area first)."""
    cx, cy = gt_coord
    matches = []
    for bbox in candidate_bboxes:
        x1, y1, x2, y2 = bbox
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            area = (x2 - x1) * (y2 - y1)
            matches.append((area, bbox))
    if matches:
        matches.sort(key=lambda x: x[0])
        return matches[0][1]
    if candidate_bboxes:
        return candidate_bboxes[0]
    return None


# ═══════════════════════════════════════════════════════════════════════
# Message building (uses JsonFormat to match training format)
# ═══════════════════════════════════════════════════════════════════════

def build_messages_for_step(sample, fm, modified_image=None, extra_hint=None):
    """Build vLLM-compatible messages for a single step.

    Uses the same format as JsonFormat.gen_next_round (system + user + image).
    If modified_image is provided, it will be saved to a temp file.
    """
    # Build a minimal episode-like dict for gen_next_round
    step = {
        "action_content": sample["action_content"],
        "screenshot": sample["screenshot"],
        "check_options": sample["check_options"],
    }
    if "thought" not in step:
        step["thought"] = ""

    # Use gen_next_round to build messages in canonical format
    line = {
        "goal": sample["goal"],
        "steps": [step],
    }

    state = fm.gen_next_round(line, None)
    if state is None:
        return None

    messages = copy.deepcopy(state["messages"])

    # If we have a modified image, replace the image element
    if modified_image is not None:
        tmp_path = f"/tmp/ac_binding_val_{os.getpid()}_{id(modified_image)}.png"
        modified_image.save(tmp_path)
        # Find and replace image in last user message
        for content_item in messages[-1]["content"]:
            if "image" in content_item:
                content_item["image"] = tmp_path
                # Update size info if present
                w, h = modified_image.size
                rh, rw = smart_resize(h, w, max_pixels=12800 * 28 * 28)
                if "width" in content_item:
                    content_item["width"] = w
                    content_item["height"] = h
                    content_item["resized_width"] = rw
                    content_item["resized_height"] = rh
                break

    # Insert hint text before the image if needed
    if extra_hint:
        for i, content_item in enumerate(messages[-1]["content"]):
            if "image" in content_item:
                messages[-1]["content"].insert(i, {"text": extra_hint})
                break

    return messages


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════

def run_condition(samples, condition, model_name, fm, args):
    """Run a single experimental condition on all samples."""
    results = []

    for idx, sample in enumerate(samples):
        t0 = time.time()
        screenshot = sample["screenshot"]
        gt_coord = sample["gt_coord"]
        check_options = sample["check_options"]
        candidate_bboxes = sample["candidate_bboxes"]

        try:
            image = Image.open(screenshot).convert("RGB")
        except Exception:
            continue

        orig_w, orig_h = image.size
        modified_image = None
        extra_hint = None
        coord_offset = (0, 0)

        if condition == "baseline":
            pass

        elif condition == "redbox":
            bbox = find_best_bbox(candidate_bboxes, gt_coord)
            if bbox is None:
                continue
            modified_image = draw_red_box(image, bbox)

        elif condition == "texthint":
            step_instr = sample.get("step_instruction", "")
            if step_instr:
                extra_hint = f"\nHint: {step_instr}\n"
            else:
                # Fallback: positional hint
                bbox = find_best_bbox(candidate_bboxes, gt_coord)
                if bbox:
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    h_pos = "left" if cx < orig_w * 0.33 else ("center" if cx < orig_w * 0.67 else "right")
                    v_pos = "top" if cy < orig_h * 0.33 else ("middle" if cy < orig_h * 0.67 else "bottom")
                    extra_hint = f"\nHint: Focus on the {v_pos}-{h_pos} area of the screen.\n"

        elif condition == "crop":
            if gt_coord and len(gt_coord) == 2:
                modified_image, crop_box = crop_around_target(image, gt_coord, crop_ratio=0.4)
                coord_offset = (crop_box[0], crop_box[1])
            else:
                continue

        # Build messages
        messages = build_messages_for_step(
            sample, fm,
            modified_image=modified_image,
            extra_hint=extra_hint,
        )
        if messages is None:
            continue

        # Call vLLM
        try:
            response = call_mobile_agent_vllm(messages, model_name=model_name)
        except Exception as e:
            print(f"[{idx}] vLLM error: {e}")
            traceback.print_exc()
            continue

        # Parse response
        try:
            parsed = safe_parse_response(fm, response)
            pred_action = parsed["action_content"]
        except Exception:
            pred_action = {"action": "parse_error"}

        # For crop: adjust predicted coordinates back
        if condition == "crop" and pred_action.get("action") in ("click", "long_press"):
            pred_coord = pred_action.get("coordinate")
            if pred_coord and len(pred_coord) == 2:
                pred_action["coordinate"] = [
                    pred_coord[0] + coord_offset[0],
                    pred_coord[1] + coord_offset[1],
                ]

        # Evaluate
        try:
            # Get image dimensions for evaluation
            _, width, height, resized_width, resized_height = find_last_image_ele(messages)
            type_match, extract_match = evaluate_android_control_action(
                pred_action, check_options,
                width, height, resized_width, resized_height,
            )
        except Exception as e:
            print(f"[{idx}] Eval error: {e}")
            type_match, extract_match = False, False

        results.append({
            "sample_idx": idx,
            "episode_id": sample["episode_id"],
            "step_idx": sample["step_idx"],
            "gt_action": "click",
            "pred_action": pred_action.get("action", ""),
            "type_match": bool(type_match),
            "extract_match": bool(extract_match),
            "response": response[:500],
        })

        n_done = len(results)
        elapsed = time.time() - t0
        if n_done % 20 == 0:
            n_type = sum(1 for r in results if r["type_match"])
            n_ext = sum(1 for r in results if r["extract_match"])
            print(f"[{condition}] {n_done}/{len(samples)} | "
                  f"type={n_type}/{n_done} ({100 * n_type / n_done:.1f}%) | "
                  f"extract={n_ext}/{n_done} ({100 * n_ext / n_done:.1f}%) | "
                  f"{elapsed:.1f}s")

    return results


def print_summary(all_results):
    """Print comparison summary."""
    print("\n" + "=" * 90)
    print("AC BINDING VALIDATION RESULTS")
    print("=" * 90)

    baseline_type = 0
    baseline_ext = 0

    print(f"\n{'Condition':>20} | {'N':>5} | {'Type Match':>12} | {'Extract Match':>14} | "
          f"{'Δ Type':>8} | {'Δ Extract':>10}")
    print("-" * 90)

    for cond_name in ["baseline", "redbox", "texthint", "crop"]:
        if cond_name not in all_results:
            continue
        results = all_results[cond_name]
        n = len(results)
        if n == 0:
            continue
        n_type = sum(1 for r in results if r["type_match"])
        n_ext = sum(1 for r in results if r["extract_match"])
        type_rate = n_type / n
        ext_rate = n_ext / n

        if cond_name == "baseline":
            baseline_type = type_rate
            baseline_ext = ext_rate
            delta_type = ""
            delta_ext = ""
        else:
            delta_type = f"{type_rate - baseline_type:+.1%}"
            delta_ext = f"{ext_rate - baseline_ext:+.1%}"

        print(f"{cond_name:>20} | {n:>5} | {type_rate:>11.1%} | {ext_rate:>13.1%} | "
              f"{delta_type:>8} | {delta_ext:>10}")

    print("\n" + "=" * 90)
    print("INTERPRETATION:")
    print("  redbox >> baseline   → visual binding bypass works → binding IS the bottleneck")
    print("  texthint >> baseline → step instruction helps → grounding warmup justified")
    print("  crop >> baseline     → reducing search space helps → visual search is hard")
    print("  none improve         → problem is beyond binding (coord regression, action knowledge)")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="AC Binding Validation (vLLM)")
    parser.add_argument("--model_name", default="Qwen2.5-VL-7B",
                        help="Model name (as served by vLLM)")
    parser.add_argument("--dataset",
                        default="evaluation/dataset/android_control_evaluation_std.jsonl",
                        help="AC JSONL dataset")
    parser.add_argument("--image_root", default=None,
                        help="Image root directory (default: PROJECT_ROOT/datasets)")
    parser.add_argument("--condition", default="all",
                        choices=["baseline", "redbox", "texthint", "crop", "all"])
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vllm_port", type=int, default=8000,
                        help="vLLM server port")
    args = parser.parse_args()

    # Set vLLM endpoint
    import evaluation.qwenvl_utils as qwenvl_mod
    qwenvl_mod.END_POINT = f"http://localhost:{args.vllm_port}/v1"

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Condition: {args.condition}")
    print(f"Samples: {args.n_samples}")
    print(f"Output: {args.output_dir}")
    print(f"vLLM: {qwenvl_mod.END_POINT}")
    print()

    # Resolve paths
    dataset_path = args.dataset
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(PROJECT_ROOT, dataset_path)

    image_root = args.image_root
    if image_root is None:
        image_root = os.path.join(PROJECT_ROOT, "datasets")

    # Load data
    print("Loading AC trajectories...", flush=True)
    episodes = load_ac_trajectories(dataset_path, image_root)
    print(f"Loaded {len(episodes)} episodes", flush=True)

    # Extract click steps
    print("Extracting click steps...", flush=True)
    samples = extract_click_steps(episodes, image_root)
    print(f"Found {len(samples)} click steps with bbox", flush=True)

    # Sample
    if 0 < args.n_samples < len(samples):
        indices = np.random.choice(len(samples), args.n_samples, replace=False)
        samples = [samples[i] for i in sorted(indices)]
    print(f"Processing {len(samples)} samples\n")

    # Init format
    fm = init_format()

    # Run conditions
    if args.condition == "all":
        conditions = ["baseline", "redbox", "texthint", "crop"]
    else:
        conditions = [args.condition]

    all_results = {}
    for cond in conditions:
        print(f"\n{'=' * 60}")
        print(f"Running condition: {cond}")
        print(f"{'=' * 60}\n")

        results = run_condition(samples, cond, args.model_name, fm, args)
        all_results[cond] = results

        # Save per-condition results
        n = len(results)
        n_type = sum(1 for r in results if r["type_match"])
        n_ext = sum(1 for r in results if r["extract_match"])
        print(f"\n{cond}: type={n_type}/{n} ({100 * n_type / n:.1f}%), "
              f"extract={n_ext}/{n} ({100 * n_ext / n:.1f}%)")

        with open(os.path.join(args.output_dir, f"results_{cond}.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # Print comparison
    if len(all_results) > 1:
        print_summary(all_results)

    # Save combined summary
    summary = {}
    for cond, results in all_results.items():
        n = len(results)
        if n == 0:
            continue
        summary[cond] = {
            "n": n,
            "type_match": round(sum(1 for r in results if r["type_match"]) / n, 4),
            "extract_match": round(sum(1 for r in results if r["extract_match"]) / n, 4),
        }
    with open(os.path.join(args.output_dir, "binding_validation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Cleanup temp files
    import glob
    for tmp in glob.glob(f"/tmp/ac_binding_val_{os.getpid()}_*.png"):
        try:
            os.remove(tmp)
        except OSError:
            pass


if __name__ == "__main__":
    main()
