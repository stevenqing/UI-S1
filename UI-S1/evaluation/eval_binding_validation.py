#!/usr/bin/env python3
"""
Zero-Training Binding Validation Experiments

Tests whether fixing cross-modal binding (with oracle info) improves accuracy.

Exp A: Red Box — Draw red bounding box on GT target in screenshot
Exp B: Text Hint — Add GT thought/element description to prompt
Exp C: Region Crop — Crop screenshot to area around GT target

Usage:
  python eval_binding_validation.py \
      --model_path /path/to/model \
      --condition {baseline,redbox,texthint,crop,all} \
      --n_samples 500 --output_dir results/binding_val
"""

import argparse
import gc
import json
import os
import re
import sys
import time
import numpy as np
import torch
import pandas as pd
from collections import defaultdict
from PIL import Image, ImageDraw

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# ═══════════════════════════════════════════════════════════════════════
# Evaluation logic (same as attention_diagnostic.py)
# ═══════════════════════════════════════════════════════════════════════

def parse_tool_call(text):
    if not text:
        return None
    m = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return None


def evaluate_action(pred, gt):
    if pred is None or gt is None:
        return {"function_match": False, "full_match": False}

    pred_func = pred.get("function", "")
    gt_func = gt.get("function", "")
    fn_match = pred_func == gt_func
    if not fn_match:
        return {"function_match": False, "full_match": False}

    pred_args = pred.get("args", {})
    gt_args = gt.get("args", {})
    gt_bbox = gt.get("bbox")
    args_match = False

    if pred_func == "click":
        pred_coord = pred_args.get("coordinate", [])
        gt_coord = gt_args.get("coordinate", [])
        if len(pred_coord) == 2 and pred_coord[0] is not None and pred_coord[1] is not None:
            if gt_bbox:
                left, top = gt_bbox.get("left"), gt_bbox.get("top")
                right, bottom = gt_bbox.get("right"), gt_bbox.get("bottom")
                if all(v is not None for v in [left, top, right, bottom]):
                    args_match = (left <= pred_coord[0] <= right and
                                  top <= pred_coord[1] <= bottom)
            if not args_match and len(gt_coord) == 2:
                dist = ((pred_coord[0] - gt_coord[0])**2 +
                        (pred_coord[1] - gt_coord[1])**2)**0.5
                args_match = dist < 50
    elif pred_func == "type":
        p = pred_args.get("text", "").lower().strip()
        g = gt_args.get("text", "").lower().strip()
        args_match = p == g or p in g or g in p
    elif pred_func in ("scroll", "wheel_mouse_input"):
        args_match = pred_args.get("direction", "") == gt_args.get("direction", "")
    elif pred_func == "summary":
        args_match = True
    else:
        args_match = pred_args == gt_args

    return {"function_match": True, "full_match": fn_match and args_match}


# ═══════════════════════════════════════════════════════════════════════
# Image modification functions
# ═══════════════════════════════════════════════════════════════════════

def draw_red_box(image, gt_bbox):
    """Draw a red bounding box on the image at GT target location."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    left = gt_bbox["left"]
    top = gt_bbox["top"]
    right = gt_bbox["right"]
    bottom = gt_bbox["bottom"]

    # Draw thick red rectangle
    for offset in range(4):
        draw.rectangle(
            [left - offset, top - offset, right + offset, bottom + offset],
            outline='red')

    return img


def crop_around_target(image, gt_coord, crop_ratio=0.3):
    """Crop image to region around GT target coordinate."""
    w, h = image.size
    cx, cy = int(gt_coord[0]), int(gt_coord[1])

    half_w = int(w * crop_ratio / 2)
    half_h = int(h * crop_ratio / 2)

    # Ensure minimum crop size
    half_w = max(half_w, 100)
    half_h = max(half_h, 100)

    x1 = max(0, cx - half_w)
    y1 = max(0, cy - half_h)
    x2 = min(w, cx + half_w)
    y2 = min(h, cy + half_h)

    cropped = image.crop((x1, y1, x2, y2))

    # Return crop info for coordinate adjustment
    return cropped, (x1, y1, x2, y2)


# ═══════════════════════════════════════════════════════════════════════
# Thought lookup
# ═══════════════════════════════════════════════════════════════════════

def build_thought_lookup(jsonl_path):
    """Build screenshot_path → thought lookup from JSONL trajectory data."""
    lookup = {}
    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            for step in data["steps"]:
                lookup[step["screenshot"]] = step["thought"]
    return lookup


# ═══════════════════════════════════════════════════════════════════════
# Main evaluation
# ═══════════════════════════════════════════════════════════════════════

def run_condition(model, processor, tokenizer, df, condition, thought_lookup, args):
    """Run a single experimental condition."""
    base_dir = args.image_base
    results = []

    for idx in range(len(df)):
        t0 = time.time()
        row = df.iloc[idx]
        messages = row["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)

        user_msg = messages[0]
        gt_response = messages[1]["content"]
        gt_action = parse_tool_call(gt_response)

        if gt_action is None:
            continue

        gt_func = gt_action.get("function", "")
        gt_coord = gt_action.get("args", {}).get("coordinate", [])
        gt_bbox = gt_action.get("bbox")

        # Extract text and image path
        text_content = ""
        image_path = None
        for item in user_msg["content"]:
            if isinstance(item, dict):
                if "text" in item:
                    text_content = item["text"]
                if "image" in item:
                    image_path = item["image"]

        if image_path is None:
            continue

        full_path = os.path.join(base_dir, image_path)
        if not os.path.exists(full_path):
            continue

        try:
            image = Image.open(full_path).convert("RGB")
        except Exception:
            continue

        # ── Apply condition modifications ──
        modified_text = text_content
        modified_image = image
        coord_offset = (0, 0)  # for crop coordinate adjustment

        if condition == "baseline":
            pass  # no modifications

        elif condition == "redbox":
            if gt_bbox and all(gt_bbox.get(k) is not None for k in ["left", "top", "right", "bottom"]):
                modified_image = draw_red_box(image, gt_bbox)
            else:
                continue  # skip samples without bbox

        elif condition == "texthint":
            # Get thought from lookup
            thought = thought_lookup.get(full_path, "")
            if thought:
                # Extract first sentence as a concise hint
                first_sent = thought.split('.')[0].strip()
                if len(first_sent) > 150:
                    first_sent = first_sent[:150]
                hint = f"\nHint: {first_sent}."
                # Insert hint before the action space section
                marker = "The actions supported are:"
                if marker in modified_text:
                    modified_text = modified_text.replace(
                        marker, f"{hint}\n\n{marker}")
                else:
                    modified_text = modified_text + hint
            else:
                # Fallback: use bbox description
                if gt_bbox and all(gt_bbox.get(k) is not None for k in ["left", "top", "right", "bottom"]):
                    w, h = image.size
                    cx = (gt_bbox["left"] + gt_bbox["right"]) / 2
                    cy = (gt_bbox["top"] + gt_bbox["bottom"]) / 2
                    # Describe position in relative terms
                    h_pos = "left" if cx < w * 0.33 else ("center" if cx < w * 0.67 else "right")
                    v_pos = "top" if cy < h * 0.33 else ("middle" if cy < h * 0.67 else "bottom")
                    hint = f"\nHint: Focus on the {v_pos}-{h_pos} area of the screen."
                    marker = "The actions supported are:"
                    if marker in modified_text:
                        modified_text = modified_text.replace(
                            marker, f"{hint}\n\n{marker}")
                    else:
                        modified_text = modified_text + hint

        elif condition == "crop":
            if len(gt_coord) == 2 and gt_coord[0] is not None and gt_coord[1] is not None:
                modified_image, crop_box = crop_around_target(image, gt_coord, crop_ratio=0.4)
                coord_offset = (crop_box[0], crop_box[1])
            else:
                continue  # skip samples without coordinate

        # ── Build prompt and run inference ──
        prompt_messages = [{"role": "user", "content": [
            {"type": "text", "text": modified_text},
            {"type": "image", "image": modified_image},
        ]}]
        prompt_text = processor.apply_chat_template(
            prompt_messages, add_generation_prompt=True, tokenize=False)

        try:
            inputs = processor(
                text=[prompt_text], images=[modified_image],
                return_tensors="pt", padding=True)
        except Exception as e:
            print(f"[{idx}] Processor error: {e}")
            continue

        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs, max_new_tokens=256, do_sample=False)

        resp_ids = gen_ids[0, seq_len:]
        response = tokenizer.decode(resp_ids, skip_special_tokens=True)
        pred_action = parse_tool_call(response)

        # For crop condition, adjust predicted coordinates back to original space
        if condition == "crop" and pred_action and pred_action.get("function") == "click":
            pred_coord = pred_action.get("args", {}).get("coordinate", [])
            if len(pred_coord) == 2 and pred_coord[0] is not None:
                pred_action["args"]["coordinate"] = [
                    pred_coord[0] + coord_offset[0],
                    pred_coord[1] + coord_offset[1],
                ]

        eval_result = evaluate_action(pred_action, gt_action)

        results.append({
            "sample_idx": int(df.index[idx]),
            "gt_function": gt_func,
            "pred_function": pred_action.get("function", "") if pred_action else "",
            "function_match": eval_result["function_match"],
            "full_match": eval_result["full_match"],
        })

        del gen_ids
        torch.cuda.empty_cache()

        n_done = len(results)
        elapsed = time.time() - t0
        if n_done % 20 == 0:
            n_correct = sum(1 for r in results if r["full_match"])
            n_fn = sum(1 for r in results if r["function_match"])
            print(f"[{condition}] {n_done}/{len(df)} | "
                  f"full={n_correct}/{n_done} ({100 * n_correct / n_done:.1f}%) | "
                  f"func={n_fn}/{n_done} ({100 * n_fn / n_done:.1f}%) | "
                  f"{elapsed:.1f}s")

    return results


def print_summary(all_results):
    """Print comparison summary."""
    print("\n" + "=" * 80)
    print("BINDING VALIDATION RESULTS")
    print("=" * 80)

    baseline_full = 0
    baseline_func = 0

    print(f"\n{'Condition':>20} | {'N':>5} | {'Full Match':>12} | {'Func Match':>12} | {'Δ Full':>8} | {'Δ Func':>8}")
    print("-" * 80)

    for cond_name in ["baseline", "redbox", "texthint", "crop"]:
        if cond_name not in all_results:
            continue
        results = all_results[cond_name]
        n = len(results)
        if n == 0:
            continue
        n_full = sum(1 for r in results if r["full_match"])
        n_func = sum(1 for r in results if r["function_match"])
        full_rate = n_full / n
        func_rate = n_func / n

        if cond_name == "baseline":
            baseline_full = full_rate
            baseline_func = func_rate
            delta_full = ""
            delta_func = ""
        else:
            delta_full = f"{full_rate - baseline_full:+.1%}"
            delta_func = f"{func_rate - baseline_func:+.1%}"

        print(f"{cond_name:>20} | {n:>5} | {full_rate:>11.1%} | {func_rate:>11.1%} | {delta_full:>8} | {delta_func:>8}")

    # Per-function breakdown for click
    print(f"\n── Click-only breakdown ──")
    print(f"{'Condition':>20} | {'N_click':>8} | {'Click Full':>12} | {'Δ':>8}")
    print("-" * 60)
    baseline_click = 0
    for cond_name in ["baseline", "redbox", "texthint", "crop"]:
        if cond_name not in all_results:
            continue
        results = all_results[cond_name]
        click_results = [r for r in results if r["gt_function"] == "click"]
        n = len(click_results)
        if n == 0:
            continue
        n_full = sum(1 for r in click_results if r["full_match"])
        rate = n_full / n
        if cond_name == "baseline":
            baseline_click = rate
            delta = ""
        else:
            delta = f"{rate - baseline_click:+.1%}"
        print(f"{cond_name:>20} | {n:>8} | {rate:>11.1%} | {delta:>8}")

    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("  redbox >> baseline   → visual binding bypass works → binding IS the bottleneck")
    print("  texthint >> baseline → text grounding hint works → grounding warmup justified")
    print("  crop >> baseline     → reducing search space helps → visual search is hard")
    print("  none improve         → problem is beyond binding (coord regression, action knowledge)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Zero-Training Binding Validation")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--parquet_file", required=True)
    parser.add_argument("--image_base", default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1")
    parser.add_argument("--jsonl_path",
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/GUI-360/rl_data/gui360_test.jsonl")
    parser.add_argument("--condition", default="all",
                        choices=["baseline", "redbox", "texthint", "crop", "all"])
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    print(f"Model: {args.model_path}")
    print(f"Condition: {args.condition}")
    print(f"Samples: {args.n_samples}")
    print(f"Output: {args.output_dir}")
    print()

    # Load model
    print("Loading model...", flush=True)
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    print("Processor loaded", flush=True)

    # Load thought lookup
    print("Building thought lookup...", flush=True)
    thought_lookup = build_thought_lookup(args.jsonl_path)
    print(f"Thought lookup: {len(thought_lookup)} entries", flush=True)

    # Load data
    print(f"Loading data...", flush=True)
    df = pd.read_parquet(args.parquet_file)
    print(f"Data: {len(df)} rows", flush=True)

    # Sample
    if 0 < args.n_samples < len(df):
        indices = np.random.choice(len(df), args.n_samples, replace=False)
        df = df.iloc[indices].reset_index(drop=True)
    print(f"Processing {len(df)} samples\n")

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

        results = run_condition(model, processor, tokenizer, df, cond, thought_lookup, args)
        all_results[cond] = results

        # Save per-condition results
        n_full = sum(1 for r in results if r["full_match"])
        n_func = sum(1 for r in results if r["function_match"])
        n = len(results)
        print(f"\n{cond}: full={n_full}/{n} ({100 * n_full / n:.1f}%), "
              f"func={n_func}/{n} ({100 * n_func / n:.1f}%)")

        with open(os.path.join(args.output_dir, f"results_{cond}.json"), "w") as f:
            json.dump(results, f)

    # Print comparison
    if len(all_results) > 1:
        print_summary(all_results)

    # Save combined results
    summary = {}
    for cond, results in all_results.items():
        n = len(results)
        if n == 0:
            continue
        summary[cond] = {
            "n": n,
            "full_match": round(sum(1 for r in results if r["full_match"]) / n, 4),
            "function_match": round(sum(1 for r in results if r["function_match"]) / n, 4),
        }
    with open(os.path.join(args.output_dir, "binding_validation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
