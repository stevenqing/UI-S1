#!/usr/bin/env python3
"""
Step 1b: Projection-Level Gradient Conflict Analysis

Refines the layer-level gradient conflict analysis (Step 1) by decomposing
each layer's gradient into per-module components: q/k/v/o_proj + gate/up/down_proj.

Hypothesis: fragmentation may be concentrated in q/k (attention routing) rather than
uniformly distributed across all projections. If so, this localizes the "locus of
fragmentation" and predicts which decomposition strategy is most effective.

Reuses all forward/backward logic from gradient_conflict_analysis.py.
Only changes: aggregation level (per-layer → per-layer-per-module).

Usage:
  python evaluation/gradient_conflict_projection.py \
      --model_path <path> \
      --parquet_file train_GUI_360/data/gui360_eval_sft.parquet \
      --image_base /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1 \
      --n_samples 200 --seed 42 \
      --output_dir evaluation/results/gradient_conflict_projection_<name>
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

TARGET_BBOX_RADIUS = 56
BIND_LAYER = 27
TEMPERATURE = 0.1

NUM_LAYERS = 28
MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj"]

# Functional groups for summary
MODULE_GROUPS = {
    "attn_routing": ["q_proj", "k_proj"],      # who attends to whom
    "attn_content": ["v_proj", "o_proj"],       # what info flows through attention
    "ffn": ["gate_proj", "up_proj", "down_proj"],  # info transformation
}


# ═══════════════════════════════════════════════════════════════════════
# Utilities (identical to gradient_conflict_analysis.py)
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


def coord_to_bbox(coord, radius=TARGET_BBOX_RADIUS):
    return {
        "left": coord[0] - radius, "top": coord[1] - radius,
        "right": coord[0] + radius, "bottom": coord[1] + radius,
    }


# ═══════════════════════════════════════════════════════════════════════
# Gradient collection — projection-level decomposition
# ═══════════════════════════════════════════════════════════════════════

def extract_module_name(param_name):
    """Extract module name (q_proj, k_proj, etc.) from parameter name.

    Example: 'model.layers.5.self_attn.q_proj.weight' → 'q_proj'
    """
    for mod in MODULES:
        if f".{mod}." in param_name:
            return mod
    return None


def store_gradients_cpu(model):
    """Store per-parameter gradients on CPU, keyed by name with layer id + module."""
    param_grads = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        m = re.search(r'model\.layers\.(\d+)\.', name)
        if m is None:
            continue
        layer_id = int(m.group(1))
        module = extract_module_name(name)
        param_grads[name] = (layer_id, module, param.grad.detach().float().cpu().flatten())
    return param_grads


def compute_conflict_per_module(bind_grads, act_grads):
    """Compute cosine similarity at (layer, module) granularity.

    Returns:
      module_conflicts: {(layer_id, module_name): cosine_similarity}
      module_ratios: {(layer_id, module_name): magnitude_ratio}
      layer_conflicts: {layer_id: cosine_similarity}  (for backward compat)
      layer_ratios: {layer_id: magnitude_ratio}
    """
    # Per-(layer, module) accumulators
    mod_dot = defaultdict(float)
    mod_norm_b_sq = defaultdict(float)
    mod_norm_a_sq = defaultdict(float)

    # Per-layer accumulators (backward compat)
    layer_dot = defaultdict(float)
    layer_norm_b_sq = defaultdict(float)
    layer_norm_a_sq = defaultdict(float)

    for name, (layer_id, module, gb) in bind_grads.items():
        if name not in act_grads:
            continue
        _, _, ga = act_grads[name]

        dot_val = float(torch.dot(gb, ga))
        norm_b = float(torch.dot(gb, gb))
        norm_a = float(torch.dot(ga, ga))

        # Per-layer
        layer_dot[layer_id] += dot_val
        layer_norm_b_sq[layer_id] += norm_b
        layer_norm_a_sq[layer_id] += norm_a

        # Per-module (only for known modules)
        if module is not None:
            key = (layer_id, module)
            mod_dot[key] += dot_val
            mod_norm_b_sq[key] += norm_b
            mod_norm_a_sq[key] += norm_a

    # Compute cosines
    module_conflicts = {}
    module_ratios = {}
    for key in mod_dot:
        nb = math.sqrt(mod_norm_b_sq[key])
        na = math.sqrt(mod_norm_a_sq[key])
        module_conflicts[key] = mod_dot[key] / (nb * na + 1e-10)
        module_ratios[key] = nb / (na + 1e-10)

    layer_conflicts = {}
    layer_ratios = {}
    for l in layer_dot:
        nb = math.sqrt(layer_norm_b_sq[l])
        na = math.sqrt(layer_norm_a_sq[l])
        layer_conflicts[l] = layer_dot[l] / (nb * na + 1e-10)
        layer_ratios[l] = nb / (na + 1e-10)

    return module_conflicts, module_ratios, layer_conflicts, layer_ratios


# ═══════════════════════════════════════════════════════════════════════
# Data preparation (identical to original)
# ═══════════════════════════════════════════════════════════════════════

def filter_and_sample(df, n_samples, seed):
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
    messages = row["messages"]
    if isinstance(messages, str):
        messages = json.loads(messages)
    user_msg = messages[0]
    gt_response = messages[1]["content"]
    gt_action = parse_tool_call(gt_response)
    gt_coord = gt_action["args"]["coordinate"]
    gt_func = gt_action.get("function", "")

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

    orig_size = image.size
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

    input_ids = full_inputs["input_ids"]
    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels = input_ids.clone()
    labels[0, :prompt_len] = -100
    gt_bbox = coord_to_bbox(gt_coord)

    return {
        "full_inputs": full_inputs, "labels": labels, "input_ids": input_ids,
        "image": image, "orig_size": orig_size, "gt_coord": gt_coord,
        "gt_bbox": gt_bbox, "gt_func": gt_func, "prompt_len": prompt_len,
    }


# ═══════════════════════════════════════════════════════════════════════
# Core analysis loop
# ═══════════════════════════════════════════════════════════════════════

def compute_gradient_conflict(model, processor, tokenizer, df, args):
    per_sample_results = []

    # Per-layer accumulators
    layer_conflicts_all = defaultdict(list)
    layer_ratios_all = defaultdict(list)

    # Per-(layer, module) accumulators
    module_conflicts_all = defaultdict(list)  # (layer, mod) -> [values]
    module_ratios_all = defaultdict(list)

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

        inputs = {k: v.to(model.device) for k, v in full_inputs.items()}
        labels = labels.to(model.device)

        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is None or len(image_grid_thw) == 0:
            n_skipped += 1
            continue
        grid_thw = image_grid_thw[0].tolist()

        positions, token_h, token_w = get_image_token_positions(grid_thw)
        n_image_tokens = token_h * token_w
        resized_h = grid_thw[1] * PATCH_SIZE
        resized_w = grid_thw[2] * PATCH_SIZE
        resized_size = (resized_w, resized_h)

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
        if (img_token_end - img_token_start) != n_image_tokens:
            n_skipped += 1
            continue

        target_token_indices = find_overlapping_tokens(
            positions, gt_bbox, orig_size, resized_size)
        if len(target_token_indices) == 0:
            n_skipped += 1
            continue

        target_seq_positions = [img_token_start + i for i in target_token_indices]
        all_img_seq_positions = list(range(img_token_start, img_token_end))
        nontarget_seq_positions = [
            p for p in all_img_seq_positions if p not in set(target_seq_positions)]
        if len(nontarget_seq_positions) == 0:
            n_skipped += 1
            continue

        task_token_indices = identify_text_regions(input_ids, tokenizer)
        if len(task_token_indices) == 0:
            n_skipped += 1
            continue

        # ── Forward pass 1: L_bind ──
        try:
            outputs1 = model(**inputs, output_hidden_states=True, return_dict=True)
            hs = outputs1.hidden_states[BIND_LAYER + 1][0]

            if not hs.requires_grad and hs.grad_fn is None:
                raise RuntimeError("hidden states detached")

            target_mean = hs[target_seq_positions].mean(dim=0)
            nontarget_mean = hs[nontarget_seq_positions].mean(dim=0)
            task_mean = hs[task_token_indices].mean(dim=0)

            target_sim = F.cosine_similarity(
                target_mean.unsqueeze(0), task_mean.unsqueeze(0))
            nontarget_sim = F.cosine_similarity(
                nontarget_mean.unsqueeze(0), task_mean.unsqueeze(0))

            logit_target = target_sim / TEMPERATURE
            logit_nontarget = nontarget_sim / TEMPERATURE
            L_bind = -torch.log(
                torch.exp(logit_target) /
                (torch.exp(logit_target) + torch.exp(logit_nontarget)))

            L_bind_val = float(L_bind.item())
            target_sim_val = float(target_sim.item())
            nontarget_sim_val = float(nontarget_sim.item())

            model.zero_grad()
            L_bind.backward()
            grad_bind = store_gradients_cpu(model)

            del outputs1, hs, L_bind, target_mean, nontarget_mean, task_mean
            del target_sim, nontarget_sim
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [{idx}] L_bind error: {e}")
            n_skipped += 1
            model.zero_grad()
            del inputs, labels
            torch.cuda.empty_cache()
            gc.collect()
            continue

        # ── Forward pass 2: L_act ──
        try:
            outputs2 = model(**inputs, labels=labels, return_dict=True)
            L_act = outputs2.loss
            L_act_val = float(L_act.item())

            model.zero_grad()
            L_act.backward()
            grad_act = store_gradients_cpu(model)

            del outputs2, L_act
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [{idx}] L_act error: {e}")
            n_skipped += 1
            model.zero_grad()
            del inputs, labels, grad_bind
            torch.cuda.empty_cache()
            gc.collect()
            continue

        # ── Compute conflict at both granularities ──
        mod_conflicts, mod_ratios, lyr_conflicts, lyr_ratios = \
            compute_conflict_per_module(grad_bind, grad_act)

        for layer_id in lyr_conflicts:
            layer_conflicts_all[layer_id].append(lyr_conflicts[layer_id])
            layer_ratios_all[layer_id].append(lyr_ratios[layer_id])

        for key in mod_conflicts:
            module_conflicts_all[key].append(mod_conflicts[key])
            module_ratios_all[key].append(mod_ratios[key])

        # Per-sample result (includes both layer and module level)
        sample_mod_conflicts = {}
        sample_mod_ratios = {}
        for (l, m), v in mod_conflicts.items():
            sample_mod_conflicts[f"{l}_{m}"] = round(v, 6)
        for (l, m), v in mod_ratios.items():
            sample_mod_ratios[f"{l}_{m}"] = round(v, 6)

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
            "layer_conflicts": {str(k): round(v, 6) for k, v in lyr_conflicts.items()},
            "module_conflicts": sample_mod_conflicts,
        })

        del inputs, labels, grad_bind, grad_act
        torch.cuda.empty_cache()
        gc.collect()

        n_processed += 1
        elapsed = time.time() - t0

        if n_processed % 5 == 0:
            # Print q_proj vs v_proj conflict for late layers
            qk_late = [mod_conflicts.get((l, "q_proj"), 0) for l in range(19, 28)]
            v_late = [mod_conflicts.get((l, "v_proj"), 0) for l in range(19, 28)]
            print(f"[{n_processed}/{len(df)}] {elapsed:.1f}s | "
                  f"L_bind={L_bind_val:.4f} L_act={L_act_val:.4f} | "
                  f"q_proj[19-27]={np.mean(qk_late):+.4f} "
                  f"v_proj[19-27]={np.mean(v_late):+.4f} | "
                  f"skip={n_skipped}")

        if n_processed % 50 == 0:
            save_results(per_sample_results, layer_conflicts_all,
                         module_conflicts_all, module_ratios_all, args, partial=True)

    print(f"\nDone: {n_processed} processed, {n_skipped} skipped")
    return per_sample_results, layer_conflicts_all, module_conflicts_all, module_ratios_all


# ═══════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════

def save_results(per_sample, layer_conflicts, module_conflicts, module_ratios,
                 args, partial=False):
    tag = "_partial" if partial else ""

    # Per-layer summary (backward compat)
    per_layer = {}
    for layer_id in range(NUM_LAYERS):
        if layer_id not in layer_conflicts or len(layer_conflicts[layer_id]) == 0:
            continue
        conflicts = layer_conflicts[layer_id]
        per_layer[str(layer_id)] = {
            "conflict_mean": round(float(np.mean(conflicts)), 6),
            "conflict_std": round(float(np.std(conflicts)), 6),
            "n_samples": len(conflicts),
        }

    with open(os.path.join(args.output_dir, f"per_layer_conflict{tag}.json"), "w") as f:
        json.dump(per_layer, f, indent=2)

    # Per-(layer, module) summary — the new data
    per_module = {}
    for layer_id in range(NUM_LAYERS):
        for mod in MODULES:
            key = (layer_id, mod)
            if key not in module_conflicts or len(module_conflicts[key]) == 0:
                continue
            conflicts = module_conflicts[key]
            ratios = module_ratios[key]
            per_module[f"{layer_id}_{mod}"] = {
                "layer": layer_id,
                "module": mod,
                "conflict_mean": round(float(np.mean(conflicts)), 6),
                "conflict_std": round(float(np.std(conflicts)), 6),
                "ratio_mean": round(float(np.mean(ratios)), 6),
                "n_samples": len(conflicts),
            }

    with open(os.path.join(args.output_dir, f"per_module_conflict{tag}.json"), "w") as f:
        json.dump(per_module, f, indent=2)

    # Per-sample
    with open(os.path.join(args.output_dir, f"per_sample_conflict{tag}.json"), "w") as f:
        json.dump(per_sample, f, indent=2)

    if not partial:
        print(f"Results saved to {args.output_dir}")


def print_summary(args):
    """Print projection-level conflict summary with hypothesis test."""
    with open(os.path.join(args.output_dir, "per_module_conflict.json")) as f:
        per_module = json.load(f)

    print("\n" + "=" * 100)
    print("PROJECTION-LEVEL GRADIENT CONFLICT ANALYSIS")
    print("=" * 100)

    # Table: late layers (L19-27) by module
    print(f"\n{'Module':>12} | ", end="")
    for l in range(19, 28):
        print(f"{'L'+str(l):>8}", end=" ")
    print(f"| {'Mean':>8} | {'Group':>14}")
    print("-" * 120)

    group_means = defaultdict(list)

    for mod in MODULES:
        print(f"{mod:>12} | ", end="")
        mod_vals = []
        for l in range(19, 28):
            key = f"{l}_{mod}"
            val = per_module.get(key, {}).get("conflict_mean", float('nan'))
            mod_vals.append(val)
            print(f"{val:>+8.4f}", end=" ")

        mean_val = np.nanmean(mod_vals)
        # Determine group
        for group_name, group_mods in MODULE_GROUPS.items():
            if mod in group_mods:
                group_means[group_name].extend(mod_vals)
                print(f"| {mean_val:>+8.4f} | {group_name:>14}")
                break

    # Group summary
    print("\n── Group Summary (L19-27) ──")
    for group_name, vals in group_means.items():
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            mean = np.mean(vals)
            std = np.std(vals)
            print(f"  {group_name:>14}: mean={mean:+.4f}, std={std:.4f}, n={len(vals)}")

    # Statistical test: is attn_routing conflict different from attn_content?
    routing_vals = []
    content_vals = []
    for l in range(19, 28):
        for mod in MODULE_GROUPS["attn_routing"]:
            key = f"{l}_{mod}"
            val = per_module.get(key, {}).get("conflict_mean")
            if val is not None:
                routing_vals.append(val)
        for mod in MODULE_GROUPS["attn_content"]:
            key = f"{l}_{mod}"
            val = per_module.get(key, {}).get("conflict_mean")
            if val is not None:
                content_vals.append(val)

    if routing_vals and content_vals:
        print(f"\n── Hypothesis Test: attn_routing vs attn_content (L19-27) ──")
        print(f"  attn_routing (q/k): mean={np.mean(routing_vals):+.4f}, "
              f"n={len(routing_vals)}")
        print(f"  attn_content (v/o): mean={np.mean(content_vals):+.4f}, "
              f"n={len(content_vals)}")
        diff = np.mean(routing_vals) - np.mean(content_vals)
        print(f"  Difference: {diff:+.4f}")

        if abs(np.mean(routing_vals)) > abs(np.mean(content_vals)) * 1.5:
            print("  → HYPOTHESIS SUPPORTED: conflict concentrated in attention routing")
        elif abs(np.mean(routing_vals)) < abs(np.mean(content_vals)) * 0.67:
            print("  → HYPOTHESIS REJECTED: conflict concentrated in attention content")
        else:
            print("  → INCONCLUSIVE: conflict similar across attention projections")

    # Full table: all 28 layers
    print(f"\n── Full Table (All Layers) ──")
    print(f"{'Layer':>6} | ", end="")
    for mod in MODULES:
        print(f"{mod:>10}", end=" ")
    print(f"| {'layer_avg':>10}")
    print("-" * 100)

    for l in range(NUM_LAYERS):
        print(f"{l:>6} | ", end="")
        vals = []
        for mod in MODULES:
            key = f"{l}_{mod}"
            val = per_module.get(key, {}).get("conflict_mean", float('nan'))
            vals.append(val)
            print(f"{val:>+10.4f}", end=" ")
        print(f"| {np.nanmean(vals):>+10.4f}")

    print("=" * 100)


def plot_results(args):
    """Generate projection-level heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    with open(os.path.join(args.output_dir, "per_module_conflict.json")) as f:
        per_module = json.load(f)

    # Build matrix: layers × modules
    matrix = np.full((NUM_LAYERS, len(MODULES)), np.nan)
    for l in range(NUM_LAYERS):
        for j, mod in enumerate(MODULES):
            key = f"{l}_{mod}"
            if key in per_module:
                matrix[l, j] = per_module[key]["conflict_mean"]

    fig, ax = plt.subplots(figsize=(10, 12))
    vmax = max(0.05, np.nanmax(np.abs(matrix)))
    im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(MODULES)))
    ax.set_xticklabels(MODULES, rotation=45, ha='right')
    ax.set_yticks(range(NUM_LAYERS))
    ax.set_yticklabels([f"L{l}" for l in range(NUM_LAYERS)])
    ax.set_xlabel("Projection Module")
    ax.set_ylabel("Layer")
    ax.set_title("Gradient Conflict: cos(∇L_bind, ∇L_act) per (Layer, Module)")

    # Add text annotations
    for l in range(NUM_LAYERS):
        for j in range(len(MODULES)):
            val = matrix[l, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > vmax * 0.6 else 'black'
                ax.text(j, l, f"{val:+.3f}", ha='center', va='center',
                        fontsize=6, color=color)

    # Add group dividers
    ax.axvline(x=1.5, color='black', linewidth=2)  # after k_proj
    ax.axvline(x=3.5, color='black', linewidth=2)  # after o_proj

    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    plt.tight_layout()

    out_path = os.path.join(args.output_dir, "projection_conflict_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {out_path}")

    # Also plot group means across layers
    fig, ax = plt.subplots(figsize=(12, 5))
    for group_name, group_mods in MODULE_GROUPS.items():
        means = []
        for l in range(NUM_LAYERS):
            vals = [per_module.get(f"{l}_{m}", {}).get("conflict_mean", np.nan)
                    for m in group_mods]
            means.append(np.nanmean(vals))
        ax.plot(range(NUM_LAYERS), means, '-o', markersize=4, label=group_name)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=-0.1, color='red', linestyle='--', alpha=0.3, label='threshold')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Gradient Conflict")
    ax.set_title("Gradient Conflict by Functional Group")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path2 = os.path.join(args.output_dir, "projection_conflict_groups.png")
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Group plot saved to {out_path2}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Step 1b: Projection-Level Gradient Conflict Analysis")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--parquet_file", required=True)
    parser.add_argument("--image_base",
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--peft_path", default=None,
                        help="Path to PEFT adapter (LoRA). If set, --model_path is base model.")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 80)
    print("Step 1b: Projection-Level Gradient Conflict Analysis")
    print("=" * 80)
    print(f"Model:       {args.model_path}")
    if args.peft_path:
        print(f"PEFT:        {args.peft_path}")
    print(f"Data:        {args.parquet_file}")
    print(f"Samples:     {args.n_samples}")
    print(f"Output:      {args.output_dir}")
    print(f"Modules:     {MODULES}")
    print(f"Groups:      {MODULE_GROUPS}")
    print()

    # Load model
    print("Loading model (bfloat16, gradient-enabled)...", flush=True)
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )

    # Apply PEFT adapter if provided, then merge into base weights
    if args.peft_path:
        from peft import PeftModel
        print(f"Loading PEFT adapter from {args.peft_path}...", flush=True)
        model = PeftModel.from_pretrained(model, args.peft_path)
        model = model.merge_and_unload()
        print("  LoRA merged into base weights")

    n_grad = 0
    n_frozen = 0
    for name, p in model.named_parameters():
        if 'model.layers.' in name:
            p.requires_grad_(True)
            n_grad += p.numel()
        else:
            p.requires_grad_(False)
            n_frozen += p.numel()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"  Grad-enabled: {n_grad/1e6:.1f}M | Frozen: {n_frozen/1e6:.1f}M")

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer

    # Load data
    print(f"\nLoading data...", flush=True)
    df = pd.read_parquet(args.parquet_file)
    print(f"Total rows: {len(df)}")
    df = filter_and_sample(df, args.n_samples, args.seed)
    print(f"Processing {len(df)} samples\n")

    # Run
    per_sample, layer_conflicts, module_conflicts, module_ratios = \
        compute_gradient_conflict(model, processor, tokenizer, df, args)

    # Save
    save_results(per_sample, layer_conflicts, module_conflicts, module_ratios,
                 args, partial=False)

    # Summary
    print_summary(args)

    # Plot
    plot_results(args)

    # Config
    config = {
        "model_path": args.model_path,
        "parquet_file": args.parquet_file,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "bind_layer": BIND_LAYER,
        "temperature": TEMPERATURE,
        "bbox_radius": TARGET_BBOX_RADIUS,
        "modules": MODULES,
        "module_groups": MODULE_GROUPS,
        "n_processed": len(per_sample),
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
