#!/usr/bin/env python3
"""
Attention Diagnostic for Qwen2.5-VL GUI Agent

D1: Correct vs Incorrect attention pattern comparison
D2: Step position (step 0 vs later) attention analysis
D3: Attention intervention (boost/suppress regions) — accuracy comparison

Usage:
  # D1+D2: observe attention patterns (500 samples, ~1h)
  python attention_diagnostic.py --mode observe \
      --model_path /path/to/model --parquet_file /path/to/test.parquet \
      --n_samples 500 --output_dir results/attn_diag

  # D3: attention intervention (200 samples, ~2h)
  python attention_diagnostic.py --mode intervene \
      --model_path /path/to/model --parquet_file /path/to/test.parquet \
      --n_samples 200 --output_dir results/attn_interv
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

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ── Token IDs (Qwen2.5-VL) ──────────────────────────────────────────
VISION_START_ID = 151652
VISION_END_ID = 151653
IMAGE_PAD_ID = 151655
IM_START_ID = 151644
IM_END_ID = 151645

# ── Text markers ─────────────────────────────────────────────────────
INSTRUCTION_MARKER = "The instruction is:"
HISTORY_MARKER = "The history of actions are:"
ACTION_SPACE_MARKER = "The actions supported are:"


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════

def find_subseq(seq, subseq):
    """Find first occurrence of subseq in seq."""
    n, m = len(seq), len(subseq)
    for i in range(n - m + 1):
        if seq[i:i + m] == subseq:
            return i
    return None


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
    """Evaluate prediction vs ground truth. Returns full_match bool."""
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


def count_history_steps(text):
    return len(re.findall(r'Step \d+:', text))


# ═══════════════════════════════════════════════════════════════════════
# Region Identification
# ═══════════════════════════════════════════════════════════════════════

def _char_to_token_pos(input_ids, tokenizer, marker_text):
    """Find token position of a text marker by decoding tokens and searching text."""
    ids = input_ids.squeeze().tolist()
    # Decode each token individually
    cum_len = 0
    token_char_starts = []
    for tid in ids:
        token_char_starts.append(cum_len)
        decoded = tokenizer.decode([tid])
        cum_len += len(decoded)

    # Build full decoded text
    full_text = tokenizer.decode(ids)

    # Find LAST occurrence of marker (skip matches in system prompt intro)
    pos = full_text.rfind(marker_text)
    if pos == -1:
        # Try case-insensitive
        pos = full_text.lower().rfind(marker_text.lower())
    if pos == -1:
        return None

    # Map char position to token position
    for i in range(len(token_char_starts) - 1, -1, -1):
        if token_char_starts[i] <= pos:
            return i
    return None


def identify_regions(input_ids, tokenizer):
    """
    Identify token regions in the input sequence.

    Returns dict: region_name -> list of token indices
    Regions: image, system, task, history, action_space, other

    Uses text-based matching (decode → find → map back) to handle
    context-dependent BPE tokenization.
    """
    ids = input_ids.squeeze().tolist()
    n = len(ids)

    # ── Image region (token ID based — reliable) ──
    img_start = img_end = None
    for i, t in enumerate(ids):
        if t == VISION_START_ID and img_start is None:
            img_start = i
        if t == VISION_END_ID:
            img_end = i

    image_set = set()
    if img_start is not None and img_end is not None:
        image_set = set(range(img_start, img_end + 1))

    # ── Text markers (text-based matching) ──
    instr_pos = _char_to_token_pos(input_ids, tokenizer, "instruction is:\n")
    hist_pos = _char_to_token_pos(input_ids, tokenizer, "history of actions are:\n")
    act_pos = _char_to_token_pos(input_ids, tokenizer, "actions supported are:\n")

    # ── Assign regions ──
    regions = {"image": sorted(image_set)}

    # Order: system ... task ... history ... [action_space ...] image
    if instr_pos is not None:
        # Go back a few tokens to include "The " before "instruction"
        instr_pos = max(0, instr_pos - 2)

        regions["system"] = [i for i in range(0, instr_pos) if i not in image_set]

        task_end = hist_pos if hist_pos is not None else (
            act_pos if act_pos is not None else (
                img_start if img_start is not None else n))
        if hist_pos is not None:
            hist_pos = max(0, hist_pos - 2)
            task_end = hist_pos
        regions["task"] = [i for i in range(instr_pos, task_end) if i not in image_set]

        if hist_pos is not None:
            hist_end = act_pos if act_pos is not None else (
                img_start if img_start is not None else n)
            if act_pos is not None:
                act_pos_adj = max(0, act_pos - 2)
                hist_end = act_pos_adj
            regions["history"] = [i for i in range(hist_pos, hist_end) if i not in image_set]
        else:
            regions["history"] = []

        if act_pos is not None:
            act_pos_adj = max(0, act_pos - 2)
            as_end = img_start if img_start is not None else n
            regions["action_space"] = [i for i in range(act_pos_adj, as_end) if i not in image_set]
        else:
            regions["action_space"] = []
    else:
        regions["system"] = [i for i in range(0, n) if i not in image_set]
        regions["task"] = []
        regions["history"] = []
        regions["action_space"] = []

    # Other: everything not assigned
    assigned = set()
    for v in regions.values():
        assigned.update(v)
    regions["other"] = [i for i in range(n) if i not in assigned]

    return regions


# ═══════════════════════════════════════════════════════════════════════
# Mode: OBSERVE (D1 + D2)
# ═══════════════════════════════════════════════════════════════════════

def run_observe(model, processor, tokenizer, df, args):
    """Extract attention patterns and generate predictions."""
    results = []
    base_dir = args.image_base

    for idx in range(len(df)):
        t0 = time.time()
        row = df.iloc[idx]
        messages = row["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)

        user_msg = messages[0]
        gt_response = messages[1]["content"]
        gt_action = parse_tool_call(gt_response)

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

        step_position = count_history_steps(text_content)

        # Build prompt for processor
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
            continue

        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]

        # Identify regions
        regions = identify_regions(input_ids, tokenizer)

        # ── Step 1: Generate response (no attention overhead) ──
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs, max_new_tokens=256, do_sample=False)

        resp_ids = gen_ids[0, seq_len:]
        response = tokenizer.decode(resp_ids, skip_special_tokens=True)
        pred_action = parse_tool_call(response)
        eval_result = evaluate_action(pred_action, gt_action)

        del gen_ids
        torch.cuda.empty_cache()

        # ── Step 2: Forward pass for attention ──
        target_layers = args.target_layers
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, return_dict=True)

        # Extract per-region attention from last token
        attn_data = {}
        for li in (target_layers if target_layers else range(len(outputs.attentions))):
            if li >= len(outputs.attentions):
                continue
            attn = outputs.attentions[li]  # (1, heads, seq, seq)
            # Cast to float32 to avoid fp16 NaN issues
            last_attn = attn[0, :, -1, :].float()  # (heads, seq)

            # Check for NaN and handle
            if torch.isnan(last_attn).any():
                # Re-normalize: some positions may have -inf from causal mask leaking
                last_attn = torch.nan_to_num(last_attn, nan=0.0)
                row_sums = last_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                last_attn = last_attn / row_sums

            layer_data = {}
            for rname, rindices in regions.items():
                if len(rindices) > 0:
                    idx_t = torch.tensor(rindices, device=last_attn.device)
                    region_vals = last_attn[:, idx_t]
                    layer_data[rname] = round(float(region_vals.sum(dim=-1).mean()), 6)
                else:
                    layer_data[rname] = 0.0
            attn_data[li] = layer_data

        del outputs
        torch.cuda.empty_cache()
        gc.collect()

        # Store result
        region_sizes = {k: len(v) for k, v in regions.items()}
        result = {
            "sample_idx": int(df.index[idx]),
            "step_position": step_position,
            "correct": eval_result["full_match"],
            "function_match": eval_result["function_match"],
            "pred_function": pred_action.get("function", "") if pred_action else "",
            "gt_function": gt_action.get("function", "") if gt_action else "",
            "seq_len": seq_len,
            "region_sizes": region_sizes,
            "attention": attn_data,
        }
        results.append(result)

        elapsed = time.time() - t0
        if (len(results)) % 10 == 0:
            n_correct = sum(1 for r in results if r["correct"])
            print(f"[{len(results)}/{len(df)}] acc={n_correct}/{len(results)} "
                  f"({100 * n_correct / len(results):.1f}%) | {elapsed:.1f}s/sample")

        # Periodic save
        if len(results) % 50 == 0:
            save_observe_results(results, args)

    save_observe_results(results, args)
    print_observe_summary(results, args)
    return results


def save_observe_results(results, args):
    with open(os.path.join(args.output_dir, "attention_results.json"), "w") as f:
        json.dump(results, f)
    summary = compute_observe_summary(results)
    with open(os.path.join(args.output_dir, "attention_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def compute_observe_summary(results):
    """Compute D1 and D2 summaries."""
    summary = {}
    regions_of_interest = ["image", "task", "history", "system", "action_space", "other"]

    # ── D1: Correct vs Incorrect ──
    correct = [r for r in results if r["correct"]]
    incorrect = [r for r in results if not r["correct"]]

    d1 = {"n_correct": len(correct), "n_incorrect": len(incorrect)}
    for group_name, group in [("correct", correct), ("incorrect", incorrect)]:
        if not group:
            continue
        layer_region = defaultdict(lambda: defaultdict(list))
        for r in group:
            for layer_str, rdata in r["attention"].items():
                layer = int(layer_str)
                for rname in regions_of_interest:
                    layer_region[layer][rname].append(rdata.get(rname, 0.0))

        avg = {}
        for layer in sorted(layer_region.keys()):
            avg[str(layer)] = {}
            for rname in regions_of_interest:
                vals = layer_region[layer][rname]
                if vals:
                    avg[str(layer)][rname] = {
                        "mean": round(float(np.mean(vals)), 6),
                        "std": round(float(np.std(vals)), 6),
                    }
        d1[f"{group_name}_attention"] = avg

    # Compute difference (correct - incorrect) for key layers
    if "correct_attention" in d1 and "incorrect_attention" in d1:
        diff = {}
        for layer_str in d1["correct_attention"]:
            if layer_str in d1["incorrect_attention"]:
                diff[layer_str] = {}
                for rname in regions_of_interest:
                    c = d1["correct_attention"][layer_str].get(rname, {}).get("mean", 0)
                    i = d1["incorrect_attention"][layer_str].get(rname, {}).get("mean", 0)
                    diff[layer_str][rname] = round(c - i, 6)
        d1["attention_diff_correct_minus_incorrect"] = diff

    summary["D1_correct_vs_incorrect"] = d1

    # ── D2: Step Position ──
    step_groups = defaultdict(list)
    for r in results:
        sp = r["step_position"]
        if sp == 0:
            step_groups["step_0"].append(r)
        elif sp <= 3:
            step_groups["step_1-3"].append(r)
        elif sp <= 6:
            step_groups["step_4-6"].append(r)
        else:
            step_groups["step_7+"].append(r)

    d2 = {}
    for gname, group in sorted(step_groups.items()):
        n_correct = sum(1 for r in group if r["correct"])
        gdata = {"n_samples": len(group), "accuracy": round(n_correct / len(group), 4) if group else 0}

        layer_region = defaultdict(lambda: defaultdict(list))
        for r in group:
            for layer_str, rdata in r["attention"].items():
                layer = int(layer_str)
                for rname in regions_of_interest:
                    layer_region[layer][rname].append(rdata.get(rname, 0.0))

        avg = {}
        for layer in sorted(layer_region.keys()):
            avg[str(layer)] = {}
            for rname in regions_of_interest:
                vals = layer_region[layer][rname]
                if vals:
                    avg[str(layer)][rname] = round(float(np.mean(vals)), 6)
        gdata["attention"] = avg
        d2[gname] = gdata

    summary["D2_step_position"] = d2
    return summary


def print_observe_summary(results, args):
    summary = compute_observe_summary(results)
    d1 = summary["D1_correct_vs_incorrect"]
    d2 = summary["D2_step_position"]

    print("\n" + "=" * 80)
    print("ATTENTION DIAGNOSTIC RESULTS")
    print("=" * 80)

    # ── D1 ──
    print(f"\n── D1: Correct ({d1['n_correct']}) vs Incorrect ({d1['n_incorrect']}) ──")
    print(f"\nAttention allocation (fraction of total attention to each region):")

    if "correct_attention" in d1 and "incorrect_attention" in d1:
        layers = sorted(d1["correct_attention"].keys(), key=int)
        # Show first, middle, last layers
        show_layers = []
        if len(layers) >= 3:
            show_layers = [layers[0], layers[len(layers) // 2], layers[-1]]
        else:
            show_layers = layers

        for layer in show_layers:
            print(f"\n  Layer {layer}:")
            print(f"  {'Region':>14} | {'Correct':>10} | {'Incorrect':>10} | {'Diff':>10}")
            print(f"  {'-' * 50}")
            for rname in ["image", "task", "history", "system", "action_space", "other"]:
                c = d1["correct_attention"].get(layer, {}).get(rname, {}).get("mean", 0)
                i = d1["incorrect_attention"].get(layer, {}).get(rname, {}).get("mean", 0)
                d = c - i
                print(f"  {rname:>14} | {c:>10.4f} | {i:>10.4f} | {d:>+10.4f}")

    # ── D2 ──
    print(f"\n── D2: Step Position ──")
    layers = None
    for gname in ["step_0", "step_1-3", "step_4-6", "step_7+"]:
        if gname not in d2:
            continue
        g = d2[gname]
        if layers is None and g.get("attention"):
            layers = sorted(g["attention"].keys(), key=int)
        print(f"\n  {gname} (n={g['n_samples']}, acc={g['accuracy']:.1%}):")
        if layers and g.get("attention"):
            last_layer = layers[-1]
            for rname in ["image", "task", "history", "system"]:
                val = g["attention"].get(last_layer, {}).get(rname, 0)
                print(f"    {rname:>14}: {val:.4f}")

    print("\n" + "=" * 80)


# ═══════════════════════════════════════════════════════════════════════
# Mode: INTERVENE (D3)
# ═══════════════════════════════════════════════════════════════════════

class AttentionBiasHook:
    """Pre-forward hook that adds bias to attention mask for specific regions."""

    def __init__(self, region_indices, bias_values):
        """
        region_indices: dict of region_name -> list of token indices
        bias_values: dict of region_name -> float bias to add
        """
        self.region_indices = region_indices
        self.bias_values = bias_values

    def __call__(self, module, args, kwargs):
        attn_mask = kwargs.get("attention_mask")
        if attn_mask is not None and attn_mask.dim() == 4:
            mask = attn_mask.clone()
            key_len = mask.shape[-1]
            for rname, indices in self.region_indices.items():
                if rname in self.bias_values:
                    bias = self.bias_values[rname]
                    valid_idx = [i for i in indices if i < key_len]
                    if valid_idx:
                        idx_t = torch.tensor(valid_idx, device=mask.device)
                        mask[:, :, :, idx_t] += bias
            kwargs["attention_mask"] = mask
        return args, kwargs


def apply_attention_bias(model, regions, bias_values):
    """Register attention bias hooks on all layers. Returns hook handles."""
    handles = []
    # Qwen2.5-VL: ForConditionalGeneration -> model (Qwen2_5_VLModel) -> language_model (TextModel) -> layers
    lang_model = model.model.language_model
    for layer in lang_model.layers:
        hook = AttentionBiasHook(regions, bias_values)
        h = layer.self_attn.register_forward_pre_hook(hook, with_kwargs=True)
        handles.append(h)
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# D3 intervention conditions
INTERVENTIONS = {
    "baseline": {},  # no bias
    "boost_task": {"task": 2.0},
    "suppress_image": {"image": -2.0},
    "boost_history": {"history": 2.0},
    "boost_task_suppress_image": {"task": 2.0, "image": -2.0},
}


def run_intervene(model, processor, tokenizer, df, args):
    """Run attention intervention experiments."""
    base_dir = args.image_base
    intervention_results = {name: [] for name in INTERVENTIONS}

    for idx in range(len(df)):
        t0 = time.time()
        row = df.iloc[idx]
        messages = row["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)

        user_msg = messages[0]
        gt_response = messages[1]["content"]
        gt_action = parse_tool_call(gt_response)

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

        step_position = count_history_steps(text_content)

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
        except Exception:
            continue

        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]

        regions = identify_regions(input_ids, tokenizer)

        # Run each intervention
        for interv_name, bias_values in INTERVENTIONS.items():
            handles = []
            if bias_values:
                handles = apply_attention_bias(model, regions, bias_values)

            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs, max_new_tokens=256, do_sample=False)

            if handles:
                remove_hooks(handles)

            resp_ids = gen_ids[0, seq_len:]
            response = tokenizer.decode(resp_ids, skip_special_tokens=True)
            pred_action = parse_tool_call(response)
            eval_result = evaluate_action(pred_action, gt_action)

            intervention_results[interv_name].append({
                "sample_idx": int(df.index[idx]),
                "step_position": step_position,
                "correct": eval_result["full_match"],
                "function_match": eval_result["function_match"],
                "pred_function": pred_action.get("function", "") if pred_action else "",
                "gt_function": gt_action.get("function", "") if gt_action else "",
            })

            del gen_ids
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        n_done = len(intervention_results["baseline"])
        if n_done % 10 == 0:
            print(f"[{n_done}/{len(df)}] {elapsed:.1f}s/sample | ", end="")
            for iname in INTERVENTIONS:
                n_c = sum(1 for r in intervention_results[iname] if r["correct"])
                n_t = len(intervention_results[iname])
                print(f"{iname}={100 * n_c / n_t:.1f}%  ", end="")
            print()

        if n_done % 50 == 0:
            save_intervene_results(intervention_results, args)

    save_intervene_results(intervention_results, args)
    print_intervene_summary(intervention_results)
    return intervention_results


def save_intervene_results(results, args):
    with open(os.path.join(args.output_dir, "intervention_results.json"), "w") as f:
        json.dump(results, f)

    summary = compute_intervene_summary(results)
    with open(os.path.join(args.output_dir, "intervention_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def compute_intervene_summary(results):
    summary = {}
    for iname, iresults in results.items():
        if not iresults:
            continue
        n_correct = sum(1 for r in iresults if r["correct"])
        n_fn = sum(1 for r in iresults if r["function_match"])
        n = len(iresults)
        summary[iname] = {
            "n_samples": n,
            "full_accuracy": round(n_correct / n, 4) if n else 0,
            "function_accuracy": round(n_fn / n, 4) if n else 0,
        }

        # Per-step breakdown
        step_acc = defaultdict(lambda: [0, 0])
        for r in iresults:
            sp = r["step_position"]
            key = "step_0" if sp == 0 else ("step_1-3" if sp <= 3 else "step_4+")
            step_acc[key][0] += int(r["correct"])
            step_acc[key][1] += 1
        summary[iname]["per_step"] = {
            k: round(v[0] / v[1], 4) if v[1] else 0
            for k, v in sorted(step_acc.items())
        }

    return summary


def print_intervene_summary(results):
    summary = compute_intervene_summary(results)
    print("\n" + "=" * 80)
    print("D3: ATTENTION INTERVENTION RESULTS")
    print("=" * 80)

    baseline_acc = summary.get("baseline", {}).get("full_accuracy", 0)
    print(f"\n{'Intervention':>30} | {'Accuracy':>10} | {'Delta':>10} | {'Function':>10}")
    print("-" * 70)
    for iname in INTERVENTIONS:
        if iname in summary:
            acc = summary[iname]["full_accuracy"]
            fn = summary[iname]["function_accuracy"]
            delta = acc - baseline_acc
            marker = "**" if abs(delta) > 0.02 else ""
            print(f"{iname:>30} | {acc:>9.1%} | {delta:>+9.1%} | {fn:>9.1%} {marker}")

    print("\nPer-step breakdown:")
    for iname in INTERVENTIONS:
        if iname in summary:
            step_data = summary[iname].get("per_step", {})
            step_str = ", ".join(f"{k}={v:.1%}" for k, v in step_data.items())
            print(f"  {iname:>30}: {step_str}")

    print("\n" + "=" * 80)
    print("Interpretation:")
    print("  boost_task > baseline    → task attention is insufficient")
    print("  suppress_image > baseline → image attention is excessive/distracting")
    print("  boost_history > baseline  → history utilization is insufficient")
    print("  no change                → problem is not in attention allocation")
    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Attention Diagnostic for GUI Agent VLM")
    parser.add_argument("--mode", choices=["observe", "intervene"], required=True,
                        help="observe=D1+D2 (attention extraction), intervene=D3 (bias experiments)")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--parquet_file", required=True)
    parser.add_argument("--image_base", default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layers", default="0,4,9,14,19,24,27",
                        help="Comma-separated layer indices for attention extraction (observe mode)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse target layers
    args.target_layers = [int(x) for x in args.layers.split(",")]

    print(f"Mode: {args.mode}")
    print(f"Model: {args.model_path}")
    print(f"Samples: {args.n_samples}")
    print(f"Output: {args.output_dir}")
    if args.mode == "observe":
        print(f"Target layers: {args.target_layers}")
    print()

    # Load model — use bfloat16 to avoid fp16 NaN in attention scores
    print("Loading model with eager attention (bfloat16)...", flush=True)
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    print(f"Processor loaded", flush=True)

    # Load data
    print(f"Loading data from {args.parquet_file}", flush=True)
    df = pd.read_parquet(args.parquet_file)
    print(f"Data loaded: {len(df)} rows", flush=True)

    # Sample
    np.random.seed(args.seed)
    if 0 < args.n_samples < len(df):
        indices = np.random.choice(len(df), args.n_samples, replace=False)
        df = df.iloc[indices].reset_index(drop=True)
    print(f"Processing {len(df)} samples\n")

    if args.mode == "observe":
        run_observe(model, processor, tokenizer, df, args)
    elif args.mode == "intervene":
        run_intervene(model, processor, tokenizer, df, args)


if __name__ == "__main__":
    main()
