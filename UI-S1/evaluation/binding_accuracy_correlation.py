#!/usr/bin/env python3
"""
Per-Sample Binding Quality vs Prediction Accuracy Correlation (Phase 1.8)

No GT is given to the model. For each sample:
  1. Forward pass → hidden states → compute binding quality per layer
  2. model.generate → prediction
  3. Evaluate prediction against GT
  4. Correlate binding quality with accuracy

If corr > 0 and significant → binding IS the bottleneck (causal evidence).

Usage:
  python binding_accuracy_correlation.py \
      --model_path /path/to/model \
      --parquet_file /path/to/test.parquet \
      --n_samples 500 --output_dir results/binding_corr
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
from pathlib import Path
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ── Constants ────────────────────────────────────────────────────────
VISION_START_ID = 151652
VISION_END_ID = 151653
IMAGE_PAD_ID = 151655

KEY_LAYERS = [0, 4, 9, 14, 19, 24, 27]
SPATIAL_MERGE_SIZE = 2
PATCH_SIZE = 14
TOKEN_PIXEL_SIZE = SPATIAL_MERGE_SIZE * PATCH_SIZE  # = 28


# ═══════════════════════════════════════════════════════════════════════
# Utilities (self-contained copies from probing_diagnostic.py / eval_binding_validation.py)
# ═══════════════════════════════════════════════════════════════════════

def parse_tool_call(text):
    if not text:
        return None
    # Try <tool_call> JSON format first
    m = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Try raw JSON
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fallback: parse plain text action format like click(coordinate=[540, 2273])
    m = re.search(r'(click|type|swipe|long_press|scroll|open|system_button|wait|summary)\s*\(', text)
    if m:
        func = m.group(1)
        result = {"function": func, "args": {}}
        # Extract coordinate
        coord_m = re.search(r'coordinate\s*=\s*\[?\s*(\d+)\s*,\s*(\d+)\s*\]?', text)
        if coord_m:
            result["args"]["coordinate"] = [int(coord_m.group(1)), int(coord_m.group(2))]
        # Extract text arg
        text_m = re.search(r'text\s*=\s*["\']([^"\']*)["\']', text)
        if text_m:
            result["args"]["text"] = text_m.group(1)
        # Extract direction for scroll
        dir_m = re.search(r'direction\s*=\s*["\']?(\w+)["\']?', text)
        if dir_m:
            result["args"]["direction"] = dir_m.group(1)
        # Extract button for system_button
        btn_m = re.search(r'button\s*=\s*["\']([^"\']*)["\']', text)
        if btn_m:
            result["args"]["button"] = btn_m.group(1)
        return result
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


def get_image_token_positions(image_grid_thw):
    """Map image token indices to pixel bboxes in resized image space."""
    t, h, w = image_grid_thw
    token_h = h // SPATIAL_MERGE_SIZE
    token_w = w // SPATIAL_MERGE_SIZE

    positions = []
    for row in range(token_h):
        for col in range(token_w):
            y1 = row * TOKEN_PIXEL_SIZE
            x1 = col * TOKEN_PIXEL_SIZE
            positions.append((x1, y1, x1 + TOKEN_PIXEL_SIZE, y1 + TOKEN_PIXEL_SIZE))

    return positions, token_h, token_w


def find_overlapping_tokens(positions, gt_bbox, orig_size, resized_size):
    """Find image token indices overlapping with GT bounding box."""
    orig_w, orig_h = orig_size
    resized_w, resized_h = resized_size

    scale_w = resized_w / orig_w
    scale_h = resized_h / orig_h

    bl = gt_bbox["left"] * scale_w
    bt = gt_bbox["top"] * scale_h
    br = gt_bbox["right"] * scale_w
    bb = gt_bbox["bottom"] * scale_h

    overlapping = []
    for i, (x1, y1, x2, y2) in enumerate(positions):
        if x2 > bl and x1 < br and y2 > bt and y1 < bb:
            overlapping.append(i)

    return overlapping


def identify_text_regions(input_ids, tokenizer):
    """Identify task text token indices."""
    ids = input_ids.squeeze().tolist()

    cum_len = 0
    token_char_starts = []
    for tid in ids:
        token_char_starts.append(cum_len)
        cum_len += len(tokenizer.decode([tid]))

    full_text = tokenizer.decode(ids)

    def find_token_pos(marker):
        pos = full_text.rfind(marker)
        if pos == -1:
            pos = full_text.lower().rfind(marker.lower())
        if pos == -1:
            return None
        for i in range(len(token_char_starts) - 1, -1, -1):
            if token_char_starts[i] <= pos:
                return i
        return None

    instr_pos = find_token_pos("instruction is:\n")
    hist_pos = find_token_pos("history of actions are:\n")
    act_pos = find_token_pos("actions supported are:\n")

    task_indices = []
    if instr_pos is not None:
        task_end = hist_pos if hist_pos is not None else (
            act_pos if act_pos is not None else len(ids))
        if hist_pos is not None:
            task_end = max(0, hist_pos - 2)
        task_start = max(0, instr_pos - 2)
        task_indices = list(range(task_start, task_end))

    return task_indices


# ═══════════════════════════════════════════════════════════════════════
# Core: Per-sample binding quality + prediction + evaluation
# ═══════════════════════════════════════════════════════════════════════

def process_samples(model, processor, tokenizer, df, args):
    """Process each sample: extract binding quality, generate prediction, evaluate."""
    base_dir = args.image_base
    per_sample_data = []

    n_processed = 0
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

        gt_bbox = gt_action.get("bbox")
        if not gt_bbox or any(gt_bbox.get(k) is None for k in ["left", "top", "right", "bottom"]):
            n_skipped += 1
            continue

        gt_func = gt_action.get("function", "")
        gt_coord = gt_action.get("args", {}).get("coordinate", [])
        if len(gt_coord) != 2 or gt_coord[0] is None or gt_coord[1] is None:
            n_skipped += 1
            continue

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

        orig_size = image.size

        # Build prompt
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
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]

        # Get image_grid_thw
        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is None or len(image_grid_thw) == 0:
            n_skipped += 1
            continue
        grid_thw = image_grid_thw[0].tolist()

        # Image token positions
        positions, token_h, token_w = get_image_token_positions(grid_thw)
        n_image_tokens = token_h * token_w

        resized_h = grid_thw[1] * PATCH_SIZE
        resized_w = grid_thw[2] * PATCH_SIZE
        resized_size = (resized_w, resized_h)

        # Find image token range in input_ids
        ids_list = input_ids.squeeze().tolist()
        img_start = img_end = None
        for i, t in enumerate(ids_list):
            if t == VISION_START_ID and img_start is None:
                img_start = i
            if t == VISION_END_ID:
                img_end = i

        if img_start is None or img_end is None:
            n_skipped += 1
            continue

        img_token_start = img_start + 1
        img_token_end = img_end
        actual_n_image_tokens = img_token_end - img_token_start

        if actual_n_image_tokens != n_image_tokens:
            n_skipped += 1
            continue

        # Find target/nontarget tokens
        target_token_indices = find_overlapping_tokens(
            positions, gt_bbox, orig_size, resized_size)

        if len(target_token_indices) == 0:
            n_skipped += 1
            continue

        target_seq_positions = [img_token_start + i for i in target_token_indices]
        all_img_seq_positions = list(range(img_token_start, img_token_end))
        nontarget_seq_positions = [p for p in all_img_seq_positions
                                   if p not in set(target_seq_positions)]

        # Identify task text tokens
        task_token_indices = identify_text_regions(input_ids, tokenizer)
        task_length = len(task_token_indices)

        if task_length == 0:
            n_skipped += 1
            continue

        # ── Step 1: Forward pass → hidden states → binding quality ──
        binding_quality = {}
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states

        for layer in KEY_LAYERS:
            if layer + 1 >= len(hidden_states):
                continue
            hs = hidden_states[layer + 1][0].float()  # (seq_len, hidden_dim)

            target_mean = hs[target_seq_positions].mean(0)
            nontarget_mean = hs[nontarget_seq_positions].mean(0) if len(nontarget_seq_positions) > 0 else torch.zeros_like(target_mean)
            task_mean = hs[task_token_indices].mean(0)

            # Cosine similarities
            def cos_sim(a, b):
                a_norm = torch.norm(a)
                b_norm = torch.norm(b)
                if a_norm == 0 or b_norm == 0:
                    return 0.0
                return float(torch.dot(a, b) / (a_norm * b_norm))

            target_task_sim = cos_sim(target_mean, task_mean)
            nontarget_task_sim = cos_sim(nontarget_mean, task_mean)
            binding_quality[layer] = target_task_sim - nontarget_task_sim

        del outputs, hidden_states
        torch.cuda.empty_cache()
        gc.collect()

        # ── Step 2: Generate prediction (separate forward pass) ──
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs, max_new_tokens=512, do_sample=False)

        resp_ids = gen_ids[0, seq_len:]
        response = tokenizer.decode(resp_ids, skip_special_tokens=True)
        pred_action = parse_tool_call(response)

        # ── Step 3: Evaluate ──
        eval_result = evaluate_action(pred_action, gt_action)

        per_sample_data.append({
            "sample_idx": int(df.index[idx]),
            "binding_quality": {str(k): round(v, 6) for k, v in binding_quality.items()},
            "function_match": eval_result["function_match"],
            "full_match": eval_result["full_match"],
            "gt_function": gt_func,
            "pred_function": pred_action.get("function", "") if pred_action else "",
            "task_length": task_length,
            "n_image_tokens": n_image_tokens,
            "n_target_tokens": len(target_token_indices),
            "seq_len": seq_len,
        })

        del gen_ids
        torch.cuda.empty_cache()

        n_processed += 1
        elapsed = time.time() - t0

        if n_processed % 10 == 0:
            n_correct = sum(1 for d in per_sample_data if d["full_match"])
            print(f"[{n_processed}/{len(df)}] {elapsed:.1f}s/sample | "
                  f"full_match={n_correct}/{n_processed} "
                  f"({100 * n_correct / n_processed:.1f}%) | "
                  f"skipped={n_skipped}")

        if n_processed % 100 == 0:
            # Save intermediate
            out_path = os.path.join(args.output_dir, "per_sample_data.json")
            with open(out_path, "w") as f:
                json.dump(per_sample_data, f, indent=2)

    print(f"\nCollection done: {n_processed} processed, {n_skipped} skipped")
    return per_sample_data


# ═══════════════════════════════════════════════════════════════════════
# Correlation Analysis
# ═══════════════════════════════════════════════════════════════════════

def run_correlation_analysis(per_sample_data, args):
    """Compute correlations between binding quality and accuracy."""
    from scipy import stats

    n = len(per_sample_data)
    if n < 30:
        print(f"ERROR: Only {n} valid samples — need at least 30 for analysis.")
        return

    # Extract arrays
    full_match = np.array([d["full_match"] for d in per_sample_data], dtype=float)
    func_match = np.array([d["function_match"] for d in per_sample_data], dtype=float)
    task_lengths = np.array([d["task_length"] for d in per_sample_data], dtype=float)
    n_img_tokens = np.array([d["n_image_tokens"] for d in per_sample_data], dtype=float)

    lines = []
    lines.append("=" * 80)
    lines.append("BINDING QUALITY vs ACCURACY CORRELATION (Phase 1.8)")
    lines.append("=" * 80)
    lines.append(f"N valid samples: {n}")
    lines.append(f"Overall full_match: {full_match.mean():.3f} ({int(full_match.sum())}/{n})")
    lines.append(f"Overall func_match: {func_match.mean():.3f} ({int(func_match.sum())}/{n})")
    lines.append("")

    # ── 1. Point-biserial correlation per layer ──
    lines.append("── 1. Point-Biserial Correlation (binding_quality vs full_match) ──")
    lines.append(f"{'Layer':>6} | {'r_pb':>8} | {'p-value':>10} | {'sig':>5}")
    lines.append("-" * 40)

    for layer in KEY_LAYERS:
        bq = np.array([d["binding_quality"].get(str(layer), 0.0) for d in per_sample_data])
        if np.std(bq) == 0:
            lines.append(f"{layer:>6} | {'N/A':>8} | {'N/A':>10} | {'N/A':>5}")
            continue
        r, p = stats.pointbiserialr(full_match, bq)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        lines.append(f"{layer:>6} | {r:>+8.4f} | {p:>10.4e} | {sig:>5}")

    lines.append("")

    # ── 2. Spearman rank correlation per layer ──
    lines.append("── 2. Spearman Rank Correlation (binding_quality vs full_match) ──")
    lines.append(f"{'Layer':>6} | {'rho':>8} | {'p-value':>10} | {'sig':>5}")
    lines.append("-" * 40)

    for layer in KEY_LAYERS:
        bq = np.array([d["binding_quality"].get(str(layer), 0.0) for d in per_sample_data])
        if np.std(bq) == 0:
            lines.append(f"{layer:>6} | {'N/A':>8} | {'N/A':>10} | {'N/A':>5}")
            continue
        rho, p = stats.spearmanr(bq, full_match)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        lines.append(f"{layer:>6} | {rho:>+8.4f} | {p:>10.4e} | {sig:>5}")

    lines.append("")

    # ── 3. Binned accuracy ──
    lines.append("── 3. Binned Accuracy (binding_quality → accuracy) ──")
    lines.append("Bins: good (gap > -0.05) | medium (-0.15 < gap <= -0.05) | poor (gap <= -0.15)")
    lines.append("")

    for layer in KEY_LAYERS:
        bq = np.array([d["binding_quality"].get(str(layer), 0.0) for d in per_sample_data])

        good_mask = bq > -0.05
        medium_mask = (bq > -0.15) & (bq <= -0.05)
        poor_mask = bq <= -0.15

        def bin_stats(mask, label):
            n_bin = mask.sum()
            if n_bin == 0:
                return f"{label:>8}: n={0:>4}, acc=N/A"
            acc = full_match[mask].mean()
            return f"{label:>8}: n={n_bin:>4}, acc={acc:.3f} ({int(full_match[mask].sum())}/{n_bin})"

        lines.append(f"  Layer {layer}:")
        lines.append(f"    {bin_stats(good_mask, 'good')}")
        lines.append(f"    {bin_stats(medium_mask, 'medium')}")
        lines.append(f"    {bin_stats(poor_mask, 'poor')}")

    lines.append("")

    # ── 4. Logistic regression (controlling confounds) ──
    lines.append("── 4. Logistic Regression: full_match ~ binding_quality + task_length + n_image_tokens ──")
    lines.append("")

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        for layer in KEY_LAYERS:
            bq = np.array([d["binding_quality"].get(str(layer), 0.0)
                           for d in per_sample_data]).reshape(-1, 1)
            X = np.column_stack([bq, task_lengths.reshape(-1, 1),
                                 n_img_tokens.reshape(-1, 1)])

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            clf = LogisticRegression(max_iter=1000, solver='lbfgs')
            clf.fit(X_scaled, full_match)

            coefs = clf.coef_[0]
            lines.append(f"  Layer {layer}: coef_binding={coefs[0]:+.4f}, "
                         f"coef_task_len={coefs[1]:+.4f}, "
                         f"coef_n_img={coefs[2]:+.4f}, "
                         f"intercept={clf.intercept_[0]:+.4f}")
    except ImportError:
        lines.append("  [sklearn not available — skipped]")

    lines.append("")

    # ── 5. Distribution summary ──
    lines.append("── 5. Binding Quality Distribution ──")
    for layer in KEY_LAYERS:
        bq = np.array([d["binding_quality"].get(str(layer), 0.0) for d in per_sample_data])
        lines.append(f"  Layer {layer}: mean={bq.mean():+.4f}, std={bq.std():.4f}, "
                     f"min={bq.min():+.4f}, max={bq.max():+.4f}, "
                     f"median={np.median(bq):+.4f}")

    lines.append("")

    # ── 6. Interpretation ──
    lines.append("=" * 80)
    lines.append("INTERPRETATION")
    lines.append("=" * 80)

    # Use layer 27 as the primary layer for interpretation
    bq_27 = np.array([d["binding_quality"].get("27", 0.0) for d in per_sample_data])
    if np.std(bq_27) > 0:
        rho, p = stats.spearmanr(bq_27, full_match)
        good_acc = full_match[bq_27 > -0.05].mean() if (bq_27 > -0.05).sum() > 0 else 0
        poor_acc = full_match[bq_27 <= -0.15].mean() if (bq_27 <= -0.15).sum() > 0 else 0

        lines.append(f"Layer 27 Spearman rho = {rho:+.4f}, p = {p:.4e}")
        if rho > 0.15 and p < 0.01:
            lines.append(f"VERDICT: Binding IS the bottleneck (rho={rho:.3f} > 0.15, p < 0.01)")
            lines.append(f"  good bin acc ({good_acc:.3f}) >> poor bin acc ({poor_acc:.3f})")
        elif abs(rho) < 0.05 and p > 0.05:
            lines.append(f"VERDICT: Binding NOT the bottleneck (rho~0, p > 0.05)")
            lines.append(f"  Need to re-locate the bottleneck.")
        else:
            lines.append(f"VERDICT: Inconclusive (rho={rho:.3f}, p={p:.4e})")
            lines.append(f"  good bin acc={good_acc:.3f}, poor bin acc={poor_acc:.3f}")

    lines.append("=" * 80)

    # Print and save
    report = "\n".join(lines)
    print(report)

    report_path = os.path.join(args.output_dir, "correlation_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Per-Sample Binding Quality vs Accuracy Correlation (Phase 1.8)")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--parquet_file", required=True)
    parser.add_argument("--image_base", default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    print(f"Model: {args.model_path}")
    print(f"Data: {args.parquet_file}")
    print(f"Samples: {args.n_samples}")
    print(f"Output: {args.output_dir}")
    print(f"Key layers: {KEY_LAYERS}")
    print()

    # Load model
    print("Loading model (bfloat16)...", flush=True)
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

    # Load data
    print(f"Loading data...", flush=True)
    df = pd.read_parquet(args.parquet_file)
    print(f"Data: {len(df)} rows", flush=True)

    # Sample
    if 0 < args.n_samples < len(df):
        indices = np.random.choice(len(df), args.n_samples, replace=False)
        df = df.iloc[indices].reset_index(drop=True)
    print(f"Processing {len(df)} samples\n")

    # Process samples
    per_sample_data = process_samples(model, processor, tokenizer, df, args)

    # Save per-sample data
    out_path = os.path.join(args.output_dir, "per_sample_data.json")
    with open(out_path, "w") as f:
        json.dump(per_sample_data, f, indent=2)
    print(f"\nPer-sample data saved to {out_path} ({len(per_sample_data)} samples)")

    # Run correlation analysis
    run_correlation_analysis(per_sample_data, args)


if __name__ == "__main__":
    main()
