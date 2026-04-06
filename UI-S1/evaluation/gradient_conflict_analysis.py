#!/usr/bin/env python3
"""
Gradient Conflict Analysis: L_bind vs L_act

Measures whether the binding objective (L_bind) and action selection objective
(L_act) produce opposing gradients on shared parameters, providing an existence
proof for representational interference.

Interpretation (from research plan S1.3):
  conflict < -0.1 in late layers (L19+) -> interference confirmed
  conflict ~ 0 everywhere            -> interference rejected (signal absence)
  conflict > 0                        -> cooperative (insufficient signal)

Usage:
  python evaluation/gradient_conflict_analysis.py \
      --model_path <path> \
      --parquet_file train_GUI_360/data/gui360_eval_sft.parquet \
      --image_base /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1 \
      --n_samples 200 --seed 42 \
      --output_dir evaluation/results/gradient_conflict_<name>
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
import torch
import torch.nn.functional as F
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

PATCH_SIZE = 14
SPATIAL_MERGE_SIZE = 2
TOKEN_PIXEL_SIZE = SPATIAL_MERGE_SIZE * PATCH_SIZE  # = 28

TARGET_BBOX_RADIUS = 56  # +/- 2 tokens around GT coordinate
BIND_LAYER = 27          # compute L_bind at final representation
TEMPERATURE = 0.1        # contrastive loss temperature

NUM_LAYERS = 28
LORA_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


# ═══════════════════════════════════════════════════════════════════════
# Utilities (from probing_diagnostic.py)
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
    return None


def get_image_token_positions(image_grid_thw):
    """
    Map image token indices to pixel bboxes in resized image space.

    image_grid_thw: (t, h, w) -- PRE-MERGE patch dimensions
    Returns: list of (x1, y1, x2, y2) in resized pixel coords, one per token
    """
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
    """
    Find image token indices overlapping with GT bounding box.

    gt_bbox: dict with left, top, right, bottom (in original image pixel coords)
    orig_size: (width, height) of original image
    resized_size: (width, height) of resized image
    Returns: list of token indices
    """
    orig_w, orig_h = orig_size
    resized_w, resized_h = resized_size

    scale_w = resized_w / orig_w
    scale_h = resized_h / orig_h

    # Scale bbox to resized coordinates
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
    """Identify task text token indices using text-based matching."""
    ids = input_ids.squeeze().tolist()

    # Build char-to-token mapping
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

    # Task = instruction to history (or action_space)
    task_indices = []
    if instr_pos is not None:
        task_end = hist_pos if hist_pos is not None else (
            act_pos if act_pos is not None else len(ids))
        if hist_pos is not None:
            task_end = max(0, hist_pos - 2)
        task_start = max(0, instr_pos - 2)
        task_indices = list(range(task_start, task_end))

    return task_indices


def coord_to_bbox(coord, radius=TARGET_BBOX_RADIUS):
    """Synthesize a bounding box from a coordinate with given radius."""
    return {
        "left": coord[0] - radius,
        "top": coord[1] - radius,
        "right": coord[0] + radius,
        "bottom": coord[1] + radius,
    }


# ═══════════════════════════════════════════════════════════════════════
# Gradient collection (memory-efficient: CPU storage, incremental cosine)
# ═══════════════════════════════════════════════════════════════════════

def store_gradients_cpu(model):
    """Store per-parameter gradients on CPU, keyed by name with layer id.

    Returns dict: {param_name: (layer_id, grad_flat_cpu)}
    """
    param_grads = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        m = re.search(r'model\.layers\.(\d+)\.', name)
        if m is None:
            continue
        layer_id = int(m.group(1))
        param_grads[name] = (layer_id, param.grad.detach().float().cpu().flatten())
    return param_grads


def compute_conflict_from_stored(bind_grads, act_grads):
    """Compute per-layer cosine similarity incrementally from stored gradients.

    Avoids concatenating full layer gradient vectors. Computes dot products
    and norms per-parameter, then aggregates per layer.

    Returns (conflicts, ratios) dicts keyed by layer_id.
    """
    layer_dot = defaultdict(float)
    layer_norm_b_sq = defaultdict(float)
    layer_norm_a_sq = defaultdict(float)

    for name, (layer_id, gb) in bind_grads.items():
        if name not in act_grads:
            continue
        _, ga = act_grads[name]
        layer_dot[layer_id] += float(torch.dot(gb, ga))
        layer_norm_b_sq[layer_id] += float(torch.dot(gb, gb))
        layer_norm_a_sq[layer_id] += float(torch.dot(ga, ga))

    conflicts = {}
    ratios = {}
    for l in layer_dot:
        nb = math.sqrt(layer_norm_b_sq[l])
        na = math.sqrt(layer_norm_a_sq[l])
        conflicts[l] = layer_dot[l] / (nb * na + 1e-10)
        ratios[l] = nb / (na + 1e-10)
    return conflicts, ratios


# ═══════════════════════════════════════════════════════════════════════
# Data preparation
# ═══════════════════════════════════════════════════════════════════════

def filter_and_sample(df, n_samples, seed):
    """Filter to click actions with valid coordinates, sample n_samples."""
    valid_indices = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        messages = row["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)

        gt_response = messages[1]["content"]
        gt_action = parse_tool_call(gt_response)
        if gt_action is None:
            continue

        gt_func = gt_action.get("function", "")
        if gt_func in ("FINISH", "") or gt_func is None:
            continue

        gt_coord = gt_action.get("args", {}).get("coordinate", [])
        if len(gt_coord) != 2 or gt_coord[0] is None or gt_coord[1] is None:
            continue

        valid_indices.append(idx)

    print(f"Valid samples with click coordinates: {len(valid_indices)}/{len(df)}")

    np.random.seed(seed)
    if 0 < n_samples < len(valid_indices):
        chosen = np.random.choice(valid_indices, n_samples, replace=False)
    else:
        chosen = np.array(valid_indices)

    return df.iloc[chosen].reset_index(drop=True)


def prepare_sample(row, processor, tokenizer, image_base):
    """Prepare a single sample: tokenize full conversation, create labels.

    Returns dict with inputs, labels, metadata, or None on failure.
    """
    messages = row["messages"]
    if isinstance(messages, str):
        messages = json.loads(messages)

    user_msg = messages[0]
    gt_response = messages[1]["content"]
    gt_action = parse_tool_call(gt_response)
    gt_coord = gt_action["args"]["coordinate"]
    gt_func = gt_action.get("function", "")

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
        return None

    full_path = os.path.join(image_base, image_path)
    if not os.path.exists(full_path):
        return None

    try:
        image = Image.open(full_path).convert("RGB")
    except Exception:
        return None

    orig_size = image.size  # (width, height)

    # Build prompt-only messages (for label masking)
    user_content = [
        {"type": "text", "text": text_content},
        {"type": "image", "image": full_path},
    ]
    messages_prompt = [{"role": "user", "content": user_content}]
    messages_full = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": gt_response}]},
    ]

    try:
        prompt_text = processor.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True)
        full_text = processor.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False)

        prompt_inputs = processor(
            text=[prompt_text], images=[image],
            return_tensors="pt", padding=False)
        full_inputs = processor(
            text=[full_text], images=[image],
            return_tensors="pt", padding=False)
    except Exception as e:
        print(f"  Processor error: {e}")
        return None

    input_ids = full_inputs["input_ids"]  # (1, seq_len)
    prompt_len = prompt_inputs["input_ids"].shape[1]

    # Labels: mask prompt tokens with -100
    labels = input_ids.clone()
    labels[0, :prompt_len] = -100

    # Synthesize bbox from coordinate
    gt_bbox = coord_to_bbox(gt_coord)

    return {
        "full_inputs": full_inputs,
        "labels": labels,
        "input_ids": input_ids,
        "image": image,
        "orig_size": orig_size,
        "gt_coord": gt_coord,
        "gt_bbox": gt_bbox,
        "gt_func": gt_func,
        "prompt_len": prompt_len,
    }


# ═══════════════════════════════════════════════════════════════════════
# Core analysis
# ═══════════════════════════════════════════════════════════════════════

def compute_gradient_conflict(model, processor, tokenizer, df, args):
    """Main loop: per-sample gradient conflict measurement."""
    per_sample_results = []
    layer_conflicts_all = defaultdict(list)  # layer -> list of conflict values
    layer_ratios_all = defaultdict(list)     # layer -> list of ratio values

    n_processed = 0
    n_skipped = 0

    for idx in range(len(df)):
        t0 = time.time()
        row = df.iloc[idx]

        sample = prepare_sample(row, processor, tokenizer, args.image_base)
        if sample is None:
            n_skipped += 1
            continue

        full_inputs = sample["full_inputs"]
        labels = sample["labels"]
        input_ids = sample["input_ids"]
        gt_bbox = sample["gt_bbox"]
        orig_size = sample["orig_size"]

        # Move to device
        inputs = {k: v.to(model.device) for k, v in full_inputs.items()}
        labels = labels.to(model.device)

        # Get image grid info
        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is None or len(image_grid_thw) == 0:
            n_skipped += 1
            continue
        grid_thw = image_grid_thw[0].tolist()

        # Token positions in resized image space
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
        img_token_end = img_end  # exclusive
        actual_n_image_tokens = img_token_end - img_token_start

        if actual_n_image_tokens != n_image_tokens:
            n_skipped += 1
            continue

        # Find target tokens (overlapping with synthetic bbox)
        target_token_indices = find_overlapping_tokens(
            positions, gt_bbox, orig_size, resized_size)

        if len(target_token_indices) == 0:
            n_skipped += 1
            continue

        # Map to sequence positions
        target_seq_positions = [img_token_start + i for i in target_token_indices]
        all_img_seq_positions = list(range(img_token_start, img_token_end))
        nontarget_seq_positions = [
            p for p in all_img_seq_positions
            if p not in set(target_seq_positions)
        ]

        if len(nontarget_seq_positions) == 0:
            n_skipped += 1
            continue

        # Identify task text tokens
        task_token_indices = identify_text_regions(input_ids, tokenizer)
        if len(task_token_indices) == 0:
            n_skipped += 1
            continue

        # Sanity mode: shuffle target/nontarget
        if args.sanity:
            all_img = target_seq_positions + nontarget_seq_positions
            np.random.shuffle(all_img)
            n_t = len(target_seq_positions)
            target_seq_positions = all_img[:n_t]
            nontarget_seq_positions = all_img[n_t:]

        # ════════════════════════════════════════════════════════════
        # Forward pass 1: L_bind (with hidden states, no labels)
        # Two separate forwards avoid retain_graph OOM
        # ════════════════════════════════════════════════════════════
        L_bind_val = float('nan')
        L_act_val = float('nan')
        target_sim_val = float('nan')
        nontarget_sim_val = float('nan')

        try:
            # Forward 1: get hidden states for L_bind
            outputs1 = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

            # hidden_states[0] = embedding, hidden_states[l+1] = after layer l
            hs = outputs1.hidden_states[BIND_LAYER + 1][0]  # (seq_len, hidden_dim)

            if not hs.requires_grad and hs.grad_fn is None:
                if n_processed == 0:
                    print("WARNING: hidden states have no grad_fn! "
                          "L_bind backward will fail. Check gradient_checkpointing config.")
                raise RuntimeError("hidden states detached from computation graph")

            target_mean = hs[target_seq_positions].mean(dim=0)
            nontarget_mean = hs[nontarget_seq_positions].mean(dim=0)
            task_mean = hs[task_token_indices].mean(dim=0)

            target_sim = F.cosine_similarity(
                target_mean.unsqueeze(0), task_mean.unsqueeze(0))
            nontarget_sim = F.cosine_similarity(
                nontarget_mean.unsqueeze(0), task_mean.unsqueeze(0))

            # Contrastive L_bind
            tau = TEMPERATURE
            logit_target = target_sim / tau
            logit_nontarget = nontarget_sim / tau
            L_bind = -torch.log(
                torch.exp(logit_target) /
                (torch.exp(logit_target) + torch.exp(logit_nontarget))
            )

            L_bind_val = float(L_bind.item())
            target_sim_val = float(target_sim.item())
            nontarget_sim_val = float(nontarget_sim.item())

            # Backward L_bind, store gradients on CPU
            model.zero_grad()
            L_bind.backward()
            grad_bind = store_gradients_cpu(model)

            # Free forward 1 graph
            del outputs1, hs, L_bind, target_mean, nontarget_mean, task_mean
            del target_sim, nontarget_sim
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [{idx}] L_bind forward/backward error: {e}")
            n_skipped += 1
            model.zero_grad()
            del inputs, labels
            torch.cuda.empty_cache()
            gc.collect()
            continue

        # ════════════════════════════════════════════════════════════
        # Forward pass 2: L_act (with labels, no hidden states)
        # ════════════════════════════════════════════════════════════
        try:
            outputs2 = model(
                **inputs,
                labels=labels,
                return_dict=True,
            )

            L_act = outputs2.loss
            L_act_val = float(L_act.item())

            model.zero_grad()
            L_act.backward()
            grad_act = store_gradients_cpu(model)

            del outputs2, L_act
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [{idx}] L_act forward/backward error: {e}")
            n_skipped += 1
            model.zero_grad()
            del inputs, labels, grad_bind
            torch.cuda.empty_cache()
            gc.collect()
            continue

        # ════════════════════════════════════════════════════════════
        # Compute per-layer conflict on CPU (incremental cosine)
        # ════════════════════════════════════════════════════════════
        sample_conflicts, sample_ratios = compute_conflict_from_stored(
            grad_bind, grad_act)

        for layer_id in sample_conflicts:
            layer_conflicts_all[layer_id].append(sample_conflicts[layer_id])
            layer_ratios_all[layer_id].append(sample_ratios[layer_id])

        per_sample_results.append({
            "sample_idx": int(idx),
            "gt_func": sample["gt_func"],
            "gt_coord": sample["gt_coord"],
            "n_target_tokens": len(target_token_indices),
            "n_nontarget_tokens": len(nontarget_seq_positions),
            "n_task_tokens": len(task_token_indices),
            "L_bind": L_bind_val,
            "L_act": L_act_val,
            "target_sim": target_sim_val,
            "nontarget_sim": nontarget_sim_val,
            "conflicts": {str(k): round(v, 6) for k, v in sample_conflicts.items()},
            "ratios": {str(k): round(v, 6) for k, v in sample_ratios.items()},
        })

        # Cleanup
        del inputs, labels, grad_bind, grad_act
        torch.cuda.empty_cache()
        gc.collect()

        n_processed += 1
        elapsed = time.time() - t0

        if n_processed % 5 == 0:
            # Print progress with latest conflict for a few layers
            late_conflicts = [
                sample_conflicts.get(l, float('nan'))
                for l in [19, 24, 27] if l in sample_conflicts
            ]
            late_str = ", ".join(f"{v:+.4f}" for v in late_conflicts)
            print(f"[{n_processed}/{len(df)}] {elapsed:.1f}s | "
                  f"L_bind={per_sample_results[-1]['L_bind']:.4f} "
                  f"L_act={per_sample_results[-1]['L_act']:.4f} | "
                  f"conflict[19,24,27]=[{late_str}] | "
                  f"tgt_tokens={len(target_token_indices)} | "
                  f"skip={n_skipped}")

        # Intermediate save every 50 samples
        if n_processed % 50 == 0:
            save_results(per_sample_results, layer_conflicts_all,
                         layer_ratios_all, args, partial=True)

    print(f"\nDone: {n_processed} processed, {n_skipped} skipped")
    return per_sample_results, layer_conflicts_all, layer_ratios_all


# ═══════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════

def save_results(per_sample, layer_conflicts, layer_ratios, args, partial=False):
    """Save results to JSON files."""
    tag = "_partial" if partial else ""

    # Per-layer summary
    per_layer = {}
    for layer_id in range(NUM_LAYERS):
        if layer_id not in layer_conflicts or len(layer_conflicts[layer_id]) == 0:
            continue
        conflicts = layer_conflicts[layer_id]
        ratios = layer_ratios[layer_id]
        per_layer[str(layer_id)] = {
            "conflict_mean": round(float(np.mean(conflicts)), 6),
            "conflict_std": round(float(np.std(conflicts)), 6),
            "conflict_median": round(float(np.median(conflicts)), 6),
            "ratio_mean": round(float(np.mean(ratios)), 6),
            "ratio_std": round(float(np.std(ratios)), 6),
            "n_samples": len(conflicts),
        }

    with open(os.path.join(args.output_dir, f"per_layer_conflict{tag}.json"), "w") as f:
        json.dump(per_layer, f, indent=2)

    with open(os.path.join(args.output_dir, f"per_sample_conflict{tag}.json"), "w") as f:
        json.dump(per_sample, f, indent=2)

    if not partial:
        print(f"Results saved to {args.output_dir}")


def plot_results(args):
    """Generate gradient_conflict_summary.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    with open(os.path.join(args.output_dir, "per_layer_conflict.json")) as f:
        per_layer = json.load(f)

    layers = sorted([int(k) for k in per_layer.keys()])
    means = [per_layer[str(l)]["conflict_mean"] for l in layers]
    stds = [per_layer[str(l)]["conflict_std"] for l in layers]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot 1: Conflict (cosine similarity) per layer
    ax1.plot(layers, means, 'b-o', linewidth=2, markersize=5, label="cos(grad_bind, grad_act)")
    ax1.fill_between(layers,
                      [m - s for m, s in zip(means, stds)],
                      [m + s for m, s in zip(means, stds)],
                      alpha=0.2, color='blue')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=-0.1, color='red', linestyle='--', alpha=0.7, label="Interference threshold (-0.1)")
    ax1.axhline(y=0.1, color='green', linestyle='--', alpha=0.7)
    ax1.set_ylabel("Gradient Conflict (cosine similarity)")
    ax1.set_title("Gradient Conflict: L_bind vs L_act per Layer")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Magnitude ratio per layer
    ratio_means = [per_layer[str(l)]["ratio_mean"] for l in layers]
    ratio_stds = [per_layer[str(l)]["ratio_std"] for l in layers]
    ax2.plot(layers, ratio_means, 'r-s', linewidth=2, markersize=5,
             label="||grad_bind|| / ||grad_act||")
    ax2.fill_between(layers,
                      [m - s for m, s in zip(ratio_means, ratio_stds)],
                      [m + s for m, s in zip(ratio_means, ratio_stds)],
                      alpha=0.2, color='red')
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Gradient Magnitude Ratio")
    ax2.set_title("Gradient Magnitude Ratio per Layer")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, "gradient_conflict_summary.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {out_path}")


def print_interpretation(args):
    """Print interpretation guide based on results."""
    with open(os.path.join(args.output_dir, "per_layer_conflict.json")) as f:
        per_layer = json.load(f)

    print("\n" + "=" * 80)
    print("GRADIENT CONFLICT ANALYSIS RESULTS")
    print("=" * 80)

    print(f"\n{'Layer':>6} | {'Conflict (mean)':>16} | {'Conflict (std)':>15} | "
          f"{'Ratio (mean)':>13} | {'N':>5}")
    print("-" * 75)

    for l in range(NUM_LAYERS):
        key = str(l)
        if key not in per_layer:
            continue
        d = per_layer[key]
        flag = ""
        if d["conflict_mean"] < -0.1:
            flag = " <-- INTERFERENCE"
        elif d["conflict_mean"] > 0.1:
            flag = " <-- COOPERATIVE"
        print(f"{l:>6} | {d['conflict_mean']:>+16.4f} | {d['conflict_std']:>15.4f} | "
              f"{d['ratio_mean']:>13.4f} | {d['n_samples']:>5}{flag}")

    # Late layer analysis (L19+)
    late_conflicts = []
    for l in range(19, 28):
        key = str(l)
        if key in per_layer:
            late_conflicts.append(per_layer[key]["conflict_mean"])

    if late_conflicts:
        mean_late = np.mean(late_conflicts)
        print(f"\nLate layers (L19-27) mean conflict: {mean_late:+.4f}")

        if mean_late < -0.1:
            print("RESULT: Interference CONFIRMED in late layers")
            print("  -> Step 2 Branch A: shared backbone + specialized heads")
        elif all(per_layer.get(str(l), {}).get("conflict_mean", 0) < -0.1
                 for l in range(28) if str(l) in per_layer):
            print("RESULT: Global interference detected")
            print("  -> Step 2 Branch B: fully separate models")
        elif abs(mean_late) < 0.1:
            print("RESULT: No significant interference in late layers")
            print("  -> Step 2 Branch C: add binding objective (auxiliary loss)")
        else:
            print("RESULT: Cooperative gradients (conflict > 0)")
            print("  -> Signal insufficient, consider data augmentation")

    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Gradient Conflict Analysis: L_bind vs L_act")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--parquet_file", required=True)
    parser.add_argument("--image_base",
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sanity", action="store_true",
                        help="Sanity check: shuffle target/nontarget -> conflict should be ~0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 80)
    print("Gradient Conflict Analysis: L_bind vs L_act")
    print("=" * 80)
    print(f"Model:       {args.model_path}")
    print(f"Data:        {args.parquet_file}")
    print(f"Samples:     {args.n_samples}")
    print(f"Seed:        {args.seed}")
    print(f"Output:      {args.output_dir}")
    print(f"Bind layer:  {BIND_LAYER}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"BBox radius: {TARGET_BBOX_RADIUS}")
    print(f"Sanity mode: {args.sanity}")
    print()

    # ── Load model (with gradient computation) ──
    print("Loading model (bfloat16, gradient-enabled)...", flush=True)
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    # Enable gradients only for LLM transformer layers (not vision encoder)
    # This reduces GPU memory by ~30% vs requiring grad for all params
    n_grad = 0
    n_frozen = 0
    for name, p in model.named_parameters():
        if 'model.layers.' in name:
            p.requires_grad_(True)
            n_grad += p.numel()
        else:
            p.requires_grad_(False)
            n_frozen += p.numel()
    # use_reentrant=False is critical: with True (default), output_hidden_states
    # tensors are detached by checkpointing, breaking L_bind backward
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)
    print(f"  Grad-enabled params: {n_grad/1e6:.1f}M | Frozen: {n_frozen/1e6:.1f}M")

    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    print("Processor loaded", flush=True)

    # ── Load and filter data ──
    print(f"\nLoading data from {args.parquet_file}...", flush=True)
    df = pd.read_parquet(args.parquet_file)
    print(f"Total rows: {len(df)}")

    df = filter_and_sample(df, args.n_samples, args.seed)
    print(f"Processing {len(df)} samples\n")

    # ── Run analysis ──
    per_sample, layer_conflicts, layer_ratios = compute_gradient_conflict(
        model, processor, tokenizer, df, args)

    # ── Save final results ──
    save_results(per_sample, layer_conflicts, layer_ratios, args, partial=False)

    # ── Plot ──
    plot_results(args)

    # ── Interpretation ──
    print_interpretation(args)

    # Save config for reproducibility
    config = {
        "model_path": args.model_path,
        "parquet_file": args.parquet_file,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "bind_layer": BIND_LAYER,
        "temperature": TEMPERATURE,
        "bbox_radius": TARGET_BBOX_RADIUS,
        "sanity": args.sanity,
        "n_processed": len(per_sample),
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
