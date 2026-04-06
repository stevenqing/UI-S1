#!/usr/bin/env python3
"""
Full Error Decomposition for GUI Agent Predictions

For each sample:
  1. Generate prediction (no hidden states needed)
  2. Save complete pred/gt details (function, coordinate, bbox, text, raw response)
  3. Offline decompose errors into taxonomy:
     - Wrong function (→ sub-categorize by confusion matrix)
     - Right function, wrong params:
       * Near miss (coord in bbox neighborhood)
       * Wrong element (coord on a different element)
       * Far miss (coord far from GT)
       * Text/direction error

Usage:
  # Step 1: Collect predictions
  python error_decomposition.py collect \
      --model_path /path/to/model \
      --parquet_file /path/to/test.parquet \
      --n_samples 500 --output_dir results/error_decomp

  # Step 2: Analyze (no GPU needed)
  python error_decomposition.py analyze \
      --predictions results/error_decomp/predictions.json \
      --output_dir results/error_decomp
"""

import argparse
import gc
import json
import math
import os
import re
import sys
import time
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


# ── Constants ────────────────────────────────────────────────────────
NEAR_MISS_THRESHOLD = 50   # pixels: "almost right"
FAR_MISS_THRESHOLD = 200   # pixels: "completely wrong"
BBOX_EXPANSION = 30        # pixels: expand bbox for "neighborhood" check


# ═══════════════════════════════════════════════════════════════════════
# Utilities
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
        pass
    # Fallback: plain text action format
    m = re.search(r'(click|type|swipe|long_press|scroll|open|system_button|wait|summary|wheel_mouse_input)\s*\(', text)
    if m:
        func = m.group(1)
        result = {"function": func, "args": {}}
        coord_m = re.search(r'coordinate\s*=\s*\[?\s*(\d+)\s*,\s*(\d+)\s*\]?', text)
        if coord_m:
            result["args"]["coordinate"] = [int(coord_m.group(1)), int(coord_m.group(2))]
        text_m = re.search(r'text\s*=\s*["\']([^"\']*)["\']', text)
        if text_m:
            result["args"]["text"] = text_m.group(1)
        dir_m = re.search(r'direction\s*=\s*["\']?(\w+)["\']?', text)
        if dir_m:
            result["args"]["direction"] = dir_m.group(1)
        btn_m = re.search(r'button\s*=\s*["\']([^"\']*)["\']', text)
        if btn_m:
            result["args"]["button"] = btn_m.group(1)
        return result
    return None


# ═══════════════════════════════════════════════════════════════════════
# Step 1: Collect predictions
# ═══════════════════════════════════════════════════════════════════════

def collect_predictions(args):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print("Loading model...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="cuda:0", trust_remote_code=True)
    if args.lora_path:
        from peft import PeftModel
        print(f"Loading LoRA from {args.lora_path}...", flush=True)
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()
        print("LoRA merged", flush=True)
    model.eval()
    print("Model loaded", flush=True)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer

    df = pd.read_parquet(args.parquet_file)
    print(f"Data: {len(df)} rows", flush=True)

    if 0 < args.n_samples < len(df):
        np.random.seed(args.seed)
        indices = np.random.choice(len(df), args.n_samples, replace=False)
        df = df.iloc[indices].reset_index(drop=True)
    print(f"Processing {len(df)} samples\n")

    base_dir = args.image_base
    predictions = []
    n_skipped = 0

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
            n_skipped += 1
            continue

        gt_func = gt_action.get("function", "")
        gt_args = gt_action.get("args", {})
        gt_bbox = gt_action.get("bbox")

        # Extract image path and text
        text_content = ""
        image_path = None
        for item in user_msg["content"]:
            if isinstance(item, dict):
                if "text" in item:
                    text_content = item["text"]
                if "image" in item:
                    image_path = item["image"]

        if image_path is None:
            n_skipped += 1
            continue

        full_path = os.path.join(base_dir, image_path)
        if not os.path.exists(full_path):
            n_skipped += 1
            continue

        try:
            image = Image.open(full_path).convert("RGB")
        except Exception:
            n_skipped += 1
            continue

        orig_w, orig_h = image.size

        prompt_messages = [{"role": "user", "content": [
            {"type": "text", "text": text_content},
            {"type": "image", "image": full_path},
        ]}]
        prompt_text = processor.apply_chat_template(
            prompt_messages, add_generation_prompt=True, tokenize=False)

        try:
            inputs = processor(
                text=[prompt_text], images=[image],
                return_tensors="pt", padding=True)
        except Exception as e:
            print(f"[{idx}] Processor error: {e}")
            n_skipped += 1
            continue

        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        seq_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

        resp_ids = gen_ids[0, seq_len:]
        response = tokenizer.decode(resp_ids, skip_special_tokens=True)
        pred_action = parse_tool_call(response)

        predictions.append({
            "sample_idx": int(df.index[idx]),
            "image_path": image_path,
            "image_size": [orig_w, orig_h],
            "gt_function": gt_func,
            "gt_args": gt_args,
            "gt_bbox": gt_bbox,
            "pred_action": pred_action,
            "raw_response": response[:1000],
        })

        del gen_ids
        torch.cuda.empty_cache()

        n_done = len(predictions)
        elapsed = time.time() - t0
        if n_done % 20 == 0:
            print(f"[{n_done}/{len(df)}] {elapsed:.1f}s | skipped={n_skipped}")

        if n_done % 200 == 0:
            with open(os.path.join(args.output_dir, "predictions.json"), "w") as f:
                json.dump(predictions, f, indent=2)

    # Save final
    out_path = os.path.join(args.output_dir, "predictions.json")
    with open(out_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"\nSaved {len(predictions)} predictions to {out_path} (skipped {n_skipped})")
    return predictions


# ═══════════════════════════════════════════════════════════════════════
# Step 2: Analyze predictions
# ═══════════════════════════════════════════════════════════════════════

def coord_distance(c1, c2):
    if not c1 or not c2 or len(c1) != 2 or len(c2) != 2:
        return float('inf')
    if c1[0] is None or c1[1] is None or c2[0] is None or c2[1] is None:
        return float('inf')
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def point_in_bbox(coord, bbox, expansion=0):
    """Check if coord is within bbox (optionally expanded)."""
    if not coord or not bbox or len(coord) != 2:
        return False
    try:
        coord = [int(coord[0]), int(coord[1])]
    except (ValueError, TypeError):
        return False
    if coord[0] is None or coord[1] is None:
        return False
    left = bbox.get("left", 0) - expansion
    top = bbox.get("top", 0) - expansion
    right = bbox.get("right", 0) + expansion
    bottom = bbox.get("bottom", 0) + expansion
    return left <= coord[0] <= right and top <= coord[1] <= bottom


def classify_click_error(pred_args, gt_args, gt_bbox, image_size):
    """Classify a click error into subcategories."""
    pred_coord = pred_args.get("coordinate", [])
    gt_coord = gt_args.get("coordinate", [])

    if not pred_coord or len(pred_coord) != 2 or pred_coord[0] is None:
        return "no_coordinate"

    dist = coord_distance(pred_coord, gt_coord)

    # Check bbox hit (exact)
    if gt_bbox and point_in_bbox(pred_coord, gt_bbox):
        return "bbox_hit_but_eval_failed"  # shouldn't happen often

    # Check expanded bbox (near miss — neighborhood)
    if gt_bbox and point_in_bbox(pred_coord, gt_bbox, expansion=BBOX_EXPANSION):
        return "near_miss_bbox_neighbor"

    # Distance-based classification
    if dist < NEAR_MISS_THRESHOLD:
        return "near_miss_by_distance"
    elif dist < FAR_MISS_THRESHOLD:
        return "moderate_miss"
    else:
        # Normalize by image diagonal
        if image_size and image_size[0] > 0:
            diag = math.sqrt(image_size[0]**2 + image_size[1]**2)
            rel_dist = dist / diag
            if rel_dist > 0.3:
                return "far_miss_random"
            else:
                return "far_miss_wrong_element"
        return "far_miss"

    return "unknown"


def classify_type_error(pred_args, gt_args):
    """Classify a type error."""
    pred_text = pred_args.get("text", "").strip().lower()
    gt_text = gt_args.get("text", "").strip().lower()

    if not pred_text:
        return "empty_text"
    if pred_text == gt_text:
        return "exact_match"  # shouldn't be here
    if pred_text in gt_text or gt_text in pred_text:
        return "partial_match"
    return "wrong_text"


def classify_scroll_error(pred_args, gt_args):
    """Classify a scroll/swipe error."""
    pred_dir = pred_args.get("direction", "")
    gt_dir = gt_args.get("direction", "")
    if pred_dir == gt_dir:
        return "direction_match"  # shouldn't be here
    return "wrong_direction"


def analyze_predictions(predictions, output_dir):
    """Full error decomposition analysis."""
    n = len(predictions)

    lines = []
    lines.append("=" * 80)
    lines.append("FULL ERROR DECOMPOSITION")
    lines.append("=" * 80)
    lines.append(f"Total samples: {n}")

    # ── Level 0: Parse success ──
    n_parse_fail = sum(1 for p in predictions if p["pred_action"] is None)
    n_parsed = n - n_parse_fail
    lines.append(f"Parse success: {n_parsed}/{n} ({100*n_parsed/n:.1f}%)")
    lines.append(f"Parse failure: {n_parse_fail}/{n} ({100*n_parse_fail/n:.1f}%)")
    lines.append("")

    # ── Level 1: Function match ──
    categories = {
        "correct": [],
        "wrong_function": [],
        "right_func_wrong_params": [],
        "parse_failure": [],
    }

    for p in predictions:
        pred = p["pred_action"]
        gt_func = p["gt_function"]
        gt_args = p["gt_args"]
        gt_bbox = p["gt_bbox"]

        if pred is None:
            categories["parse_failure"].append(p)
            continue

        pred_func = pred.get("function", "")
        pred_args = pred.get("args", {})

        if pred_func != gt_func:
            categories["wrong_function"].append(p)
            continue

        # Function matches — check params
        is_correct = False

        if pred_func == "click":
            pred_coord = pred_args.get("coordinate", [])
            gt_coord = gt_args.get("coordinate", [])
            if len(pred_coord) == 2 and pred_coord[0] is not None:
                if gt_bbox and point_in_bbox(pred_coord, gt_bbox):
                    is_correct = True
                elif len(gt_coord) == 2 and coord_distance(pred_coord, gt_coord) < 50:
                    is_correct = True

        elif pred_func == "type":
            pt = pred_args.get("text", "").lower().strip()
            gt = gt_args.get("text", "").lower().strip()
            is_correct = pt == gt or pt in gt or gt in pt

        elif pred_func in ("scroll", "wheel_mouse_input"):
            is_correct = pred_args.get("direction", "") == gt_args.get("direction", "")

        elif pred_func == "summary":
            is_correct = True

        else:
            is_correct = pred_args == gt_args

        if is_correct:
            categories["correct"].append(p)
        else:
            categories["right_func_wrong_params"].append(p)

    n_correct = len(categories["correct"])
    n_wrong_func = len(categories["wrong_function"])
    n_wrong_params = len(categories["right_func_wrong_params"])
    n_parse = len(categories["parse_failure"])

    lines.append("── Level 1: Top-Level Error Decomposition ──")
    lines.append(f"  Correct:                     {n_correct:>4} ({100*n_correct/n:>5.1f}%)")
    lines.append(f"  Wrong function:              {n_wrong_func:>4} ({100*n_wrong_func/n:>5.1f}%)")
    lines.append(f"  Right func, wrong params:    {n_wrong_params:>4} ({100*n_wrong_params/n:>5.1f}%)")
    lines.append(f"  Parse failure:               {n_parse:>4} ({100*n_parse/n:>5.1f}%)")
    lines.append("")

    # ── Level 2: Function confusion matrix ──
    lines.append("── Level 2a: Function Confusion Matrix ──")

    all_funcs = sorted(set(
        [p["gt_function"] for p in predictions] +
        [p["pred_action"].get("function", "NONE") if p["pred_action"] else "PARSE_FAIL"
         for p in predictions]
    ))

    confusion = defaultdict(lambda: defaultdict(int))
    for p in predictions:
        gt_f = p["gt_function"]
        pred_f = p["pred_action"].get("function", "NONE") if p["pred_action"] else "PARSE_FAIL"
        confusion[gt_f][pred_f] += 1

    # GT function distribution
    gt_func_counts = Counter(p["gt_function"] for p in predictions)
    lines.append(f"\n  GT function distribution:")
    for func, cnt in gt_func_counts.most_common():
        lines.append(f"    {func:>20}: {cnt:>4} ({100*cnt/n:.1f}%)")

    # Confusion matrix (compact)
    pred_funcs_seen = sorted(set(
        p["pred_action"].get("function", "NONE") if p["pred_action"] else "PARSE_FAIL"
        for p in predictions
    ))
    lines.append(f"\n  Confusion (rows=GT, cols=pred):")
    header = f"  {'GT':>20} | " + " | ".join(f"{f[:8]:>8}" for f in pred_funcs_seen) + " | total"
    lines.append(header)
    lines.append("  " + "-" * len(header))
    for gt_f in gt_func_counts:
        row = f"  {gt_f:>20} | "
        row += " | ".join(f"{confusion[gt_f][pf]:>8}" for pf in pred_funcs_seen)
        row += f" | {gt_func_counts[gt_f]:>5}"
        lines.append(row)
    lines.append("")

    # ── Level 2b: Wrong function sub-analysis ──
    lines.append("── Level 2b: Wrong Function — What did the model predict instead? ──")
    wrong_func_confusion = defaultdict(lambda: defaultdict(int))
    for p in categories["wrong_function"]:
        gt_f = p["gt_function"]
        pred_f = p["pred_action"].get("function", "NONE") if p["pred_action"] else "NONE"
        wrong_func_confusion[gt_f][pred_f] += 1

    for gt_f, preds in sorted(wrong_func_confusion.items(), key=lambda x: -sum(x[1].values())):
        total = sum(preds.values())
        lines.append(f"  GT={gt_f} ({total} errors):")
        for pred_f, cnt in sorted(preds.items(), key=lambda x: -x[1]):
            lines.append(f"    → predicted {pred_f}: {cnt} ({100*cnt/total:.0f}%)")
    lines.append("")

    # ── Level 3: Click parameter errors ──
    click_wrong_params = [p for p in categories["right_func_wrong_params"]
                          if p["gt_function"] == "click"]
    if click_wrong_params:
        lines.append(f"── Level 3a: Click Coordinate Error Analysis ({len(click_wrong_params)} errors) ──")

        click_subcats = defaultdict(list)
        distances = []

        for p in click_wrong_params:
            pred_args = p["pred_action"]["args"] if p["pred_action"] else {}
            subcat = classify_click_error(pred_args, p["gt_args"], p["gt_bbox"], p["image_size"])
            click_subcats[subcat].append(p)

            pred_coord = pred_args.get("coordinate", [])
            gt_coord = p["gt_args"].get("coordinate", [])
            dist = coord_distance(pred_coord, gt_coord)
            if dist < float('inf'):
                distances.append(dist)

        for subcat, items in sorted(click_subcats.items(), key=lambda x: -len(x[1])):
            lines.append(f"    {subcat:>30}: {len(items):>4} ({100*len(items)/len(click_wrong_params):.1f}%)")

        if distances:
            lines.append(f"\n  Click coordinate distance stats:")
            lines.append(f"    mean:   {np.mean(distances):.1f} px")
            lines.append(f"    median: {np.median(distances):.1f} px")
            lines.append(f"    p25:    {np.percentile(distances, 25):.1f} px")
            lines.append(f"    p75:    {np.percentile(distances, 75):.1f} px")
            lines.append(f"    p90:    {np.percentile(distances, 90):.1f} px")
            lines.append(f"    max:    {np.max(distances):.1f} px")

            # Distance distribution buckets
            buckets = [0, 30, 50, 100, 200, 500, float('inf')]
            lines.append(f"\n  Distance distribution:")
            for i in range(len(buckets) - 1):
                cnt = sum(1 for d in distances if buckets[i] <= d < buckets[i+1])
                label = f"[{buckets[i]},{buckets[i+1]})" if buckets[i+1] < float('inf') else f"[{buckets[i]},∞)"
                lines.append(f"    {label:>15}: {cnt:>4} ({100*cnt/len(distances):.1f}%)")
    lines.append("")

    # ── Level 3b: Type parameter errors ──
    type_wrong_params = [p for p in categories["right_func_wrong_params"]
                         if p["gt_function"] == "type"]
    if type_wrong_params:
        lines.append(f"── Level 3b: Type Text Error Analysis ({len(type_wrong_params)} errors) ──")
        type_subcats = defaultdict(int)
        for p in type_wrong_params:
            pred_args = p["pred_action"]["args"] if p["pred_action"] else {}
            subcat = classify_type_error(pred_args, p["gt_args"])
            type_subcats[subcat] += 1
        for subcat, cnt in sorted(type_subcats.items(), key=lambda x: -x[1]):
            lines.append(f"    {subcat:>20}: {cnt:>4} ({100*cnt/len(type_wrong_params):.1f}%)")
    lines.append("")

    # ── Level 3c: Scroll/swipe errors ──
    scroll_wrong_params = [p for p in categories["right_func_wrong_params"]
                           if p["gt_function"] in ("scroll", "wheel_mouse_input", "swipe")]
    if scroll_wrong_params:
        lines.append(f"── Level 3c: Scroll/Swipe Error Analysis ({len(scroll_wrong_params)} errors) ──")
        scroll_subcats = defaultdict(int)
        for p in scroll_wrong_params:
            pred_args = p["pred_action"]["args"] if p["pred_action"] else {}
            subcat = classify_scroll_error(pred_args, p["gt_args"])
            scroll_subcats[subcat] += 1
        for subcat, cnt in sorted(scroll_subcats.items(), key=lambda x: -x[1]):
            lines.append(f"    {subcat:>20}: {cnt:>4} ({100*cnt/len(scroll_wrong_params):.1f}%)")
    lines.append("")

    # ── Summary taxonomy ──
    lines.append("=" * 80)
    lines.append("ERROR TAXONOMY SUMMARY")
    lines.append("=" * 80)
    lines.append(f"100% total ({n} samples)")
    lines.append(f"├── {100*n_correct/n:.1f}% Correct")
    lines.append(f"├── {100*n_parse/n:.1f}% Parse failure (model didn't produce valid action)")

    # Wrong function breakdown
    lines.append(f"├── {100*n_wrong_func/n:.1f}% Wrong function")
    if wrong_func_confusion:
        for gt_f, preds in sorted(wrong_func_confusion.items(), key=lambda x: -sum(x[1].values())):
            total = sum(preds.values())
            top_pred = max(preds.items(), key=lambda x: x[1])
            lines.append(f"│   ├── GT={gt_f}: {total} errors (top pred: {top_pred[0]})")

    # Wrong params breakdown
    lines.append(f"└── {100*n_wrong_params/n:.1f}% Right function, wrong params")
    if click_wrong_params:
        lines.append(f"    ├── Click errors: {len(click_wrong_params)}")
        for subcat, items in sorted(click_subcats.items(), key=lambda x: -len(x[1])):
            lines.append(f"    │   ├── {subcat}: {len(items)}")
    if type_wrong_params:
        lines.append(f"    ├── Type errors: {len(type_wrong_params)}")
    if scroll_wrong_params:
        lines.append(f"    └── Scroll errors: {len(scroll_wrong_params)}")

    lines.append("=" * 80)

    # ── Actionable insights ──
    lines.append("\nACTIONABLE INSIGHTS:")
    total_errors = n - n_correct

    if n_wrong_func > 0 and n_wrong_func / total_errors > 0.4:
        lines.append(f"  ! Function selection is the DOMINANT error ({100*n_wrong_func/total_errors:.0f}% of errors)")
    if click_wrong_params:
        near_miss_count = sum(len(v) for k, v in click_subcats.items() if 'near_miss' in k)
        far_miss_count = sum(len(v) for k, v in click_subcats.items() if 'far_miss' in k or 'moderate' in k)
        if near_miss_count > far_miss_count:
            lines.append(f"  ! Click errors are mostly NEAR MISSES → coordinate precision problem")
        elif far_miss_count > near_miss_count:
            lines.append(f"  ! Click errors are mostly FAR MISSES → wrong element / binding problem")
    if n_parse / n > 0.1:
        lines.append(f"  ! High parse failure rate ({100*n_parse/n:.0f}%) → output format issue")

    lines.append("")

    report = "\n".join(lines)
    print(report)

    with open(os.path.join(output_dir, "error_decomposition_report.txt"), "w") as f:
        f.write(report)
    print(f"\nReport saved to {output_dir}/error_decomposition_report.txt")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Full Error Decomposition")
    subparsers = parser.add_subparsers(dest="command")

    # Collect subcommand
    p_collect = subparsers.add_parser("collect", help="Collect predictions (needs GPU)")
    p_collect.add_argument("--model_path", required=True)
    p_collect.add_argument("--lora_path", default=None, help="Path to PEFT LoRA adapter")
    p_collect.add_argument("--parquet_file", required=True)
    p_collect.add_argument("--image_base", default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1")
    p_collect.add_argument("--n_samples", type=int, default=500)
    p_collect.add_argument("--output_dir", required=True)
    p_collect.add_argument("--seed", type=int, default=42)

    # Analyze subcommand
    p_analyze = subparsers.add_parser("analyze", help="Analyze predictions (no GPU)")
    p_analyze.add_argument("--predictions", required=True, help="Path to predictions.json")
    p_analyze.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    if args.command == "collect":
        os.makedirs(args.output_dir, exist_ok=True)
        predictions = collect_predictions(args)
        # Also run analysis immediately
        analyze_predictions(predictions, args.output_dir)

    elif args.command == "analyze":
        with open(args.predictions) as f:
            predictions = json.load(f)
        os.makedirs(args.output_dir, exist_ok=True)
        analyze_predictions(predictions, args.output_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
