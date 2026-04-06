#!/usr/bin/env python3
"""
Cross-Modal Probing Diagnostic for Qwen2.5-VL GUI Agent

Diagnoses where the model fails in the cross-modal binding pipeline:
  A: Visual Feature Quality — can image tokens distinguish target element?
  B: Coordinate Regression — does the last token encode correct spatial info?
  C: Cross-Modal Alignment — do target image tokens align with task text tokens?

Usage:
  python probing_diagnostic.py \
      --model_path /path/to/model \
      --parquet_file /path/to/test.parquet \
      --n_samples 300 --output_dir results/probing
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
    return None


def get_image_token_positions(image_grid_thw):
    """
    Map image token indices to pixel bboxes in resized image space.

    image_grid_thw: (t, h, w) — PRE-MERGE patch dimensions
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


# ═══════════════════════════════════════════════════════════════════════
# Main Probing
# ═══════════════════════════════════════════════════════════════════════

def run_probing(model, processor, tokenizer, df, args):
    """Collect hidden state features for probing."""
    base_dir = args.image_base

    # Feature storage per layer
    # Probe A: image token features + target labels
    probe_a_features = {l: [] for l in KEY_LAYERS}  # list of (features, labels) per layer
    # Probe B: last token features + GT coordinates
    probe_b_features = {l: [] for l in KEY_LAYERS}
    probe_b_targets = []
    # Probe C: target image mean, non-target image mean, task text mean
    probe_c_data = {l: {"target_img": [], "nontarget_img": [], "task_text": []} for l in KEY_LAYERS}

    # Additional metadata
    sample_meta = []

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

        # Only use samples with click/type that have bbox
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

        # Extract image path
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

        orig_size = image.size  # (width, height)

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
        grid_thw = image_grid_thw[0].tolist()  # (t, h, w)

        # Get image token positions in resized space
        positions, token_h, token_w = get_image_token_positions(grid_thw)
        n_image_tokens = token_h * token_w

        # Resized image size
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

        # Image tokens are between vision_start+1 and vision_end-1
        img_token_start = img_start + 1
        img_token_end = img_end  # exclusive
        actual_n_image_tokens = img_token_end - img_token_start

        if actual_n_image_tokens != n_image_tokens:
            # Mismatch — skip
            n_skipped += 1
            continue

        # Find target tokens (overlapping with GT bbox)
        target_token_indices = find_overlapping_tokens(
            positions, gt_bbox, orig_size, resized_size)

        if len(target_token_indices) == 0:
            n_skipped += 1
            continue

        # Map to sequence positions
        target_seq_positions = [img_token_start + i for i in target_token_indices]
        all_img_seq_positions = list(range(img_token_start, img_token_end))
        nontarget_seq_positions = [p for p in all_img_seq_positions if p not in set(target_seq_positions)]

        # Identify task text tokens
        task_token_indices = identify_text_regions(input_ids, tokenizer)

        # ── Forward pass with hidden states ──
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states  # tuple of (1, seq_len, hidden_dim)

        # Extract features per layer
        for li in KEY_LAYERS:
            if li + 1 >= len(hidden_states):
                continue
            # hidden_states[0] = embedding, hidden_states[1] = after layer 0, etc.
            hs = hidden_states[li + 1][0].float()  # (seq_len, hidden_dim)

            # ── Probe A: image token features ──
            # Subsample to keep memory bounded
            target_feats = hs[target_seq_positions].cpu()  # (n_target, hidden_dim)
            # Subsample non-target to match target count (balanced)
            if len(nontarget_seq_positions) > 0:
                n_sample_nt = min(len(nontarget_seq_positions), max(len(target_token_indices) * 3, 20))
                nt_indices = np.random.choice(
                    nontarget_seq_positions, n_sample_nt, replace=False).tolist()
                nontarget_feats = hs[nt_indices].cpu()
            else:
                nontarget_feats = torch.zeros(0, hs.shape[1])

            # Labels: 1 for target, 0 for non-target
            labels = np.concatenate([
                np.ones(len(target_feats)),
                np.zeros(len(nontarget_feats)),
            ])
            feats = torch.cat([target_feats, nontarget_feats], dim=0).numpy()
            probe_a_features[li].append((feats, labels))

            # ── Probe B: last token ──
            last_feat = hs[-1].cpu().numpy()  # (hidden_dim,)
            probe_b_features[li].append(last_feat)

            # ── Probe C: cross-modal similarity ──
            target_mean = target_feats.mean(dim=0)  # (hidden_dim,)
            if len(nontarget_feats) > 0:
                nontarget_mean = nontarget_feats.mean(dim=0)
            else:
                nontarget_mean = torch.zeros_like(target_mean)

            if len(task_token_indices) > 0:
                task_feats = hs[task_token_indices].cpu()
                task_mean = task_feats.mean(dim=0)
            else:
                task_mean = torch.zeros_like(target_mean)

            probe_c_data[li]["target_img"].append(target_mean.numpy())
            probe_c_data[li]["nontarget_img"].append(nontarget_mean.numpy())
            probe_c_data[li]["task_text"].append(task_mean.numpy())

        # GT coordinates for Probe B
        probe_b_targets.append(gt_coord)

        sample_meta.append({
            "idx": int(df.index[idx]),
            "gt_function": gt_func,
            "gt_coord": gt_coord,
            "gt_bbox": gt_bbox,
            "n_target_tokens": len(target_token_indices),
            "n_image_tokens": n_image_tokens,
            "orig_size": list(orig_size),
            "resized_size": list(resized_size),
        })

        del outputs, hidden_states
        torch.cuda.empty_cache()
        gc.collect()

        n_processed += 1
        elapsed = time.time() - t0

        if n_processed % 10 == 0:
            print(f"[{n_processed}/{len(df)}] {elapsed:.1f}s/sample | "
                  f"target_tokens={len(target_token_indices)} | "
                  f"skipped={n_skipped}")

        if n_processed % 100 == 0:
            # Save intermediate
            save_probe_features(probe_a_features, probe_b_features, probe_b_targets,
                                probe_c_data, sample_meta, args)

    print(f"\nFeature collection done: {n_processed} processed, {n_skipped} skipped")

    # Save all features
    save_probe_features(probe_a_features, probe_b_features, probe_b_targets,
                        probe_c_data, sample_meta, args)

    # ── Train probes and report ──
    train_and_report(probe_a_features, probe_b_features, probe_b_targets,
                     probe_c_data, sample_meta, args)


def save_probe_features(probe_a, probe_b, probe_b_targets, probe_c, meta, args):
    """Save features to disk."""
    out = {
        "probe_b_targets": probe_b_targets,
        "sample_meta": meta,
    }
    with open(os.path.join(args.output_dir, "probe_meta.json"), "w") as f:
        json.dump(out, f)


def train_and_report(probe_a_features, probe_b_features, probe_b_targets,
                     probe_c_data, sample_meta, args):
    """Train linear probes and report results."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.preprocessing import StandardScaler

    print("\n" + "=" * 80)
    print("PROBING DIAGNOSTIC RESULTS")
    print("=" * 80)

    results = {}

    # ══════════════════════════════════════════════════════════════
    # Probe A: Visual Feature — target vs non-target image tokens
    # ══════════════════════════════════════════════════════════════
    print("\n── Probe A: Visual Feature Quality ──")
    print("Can image token representations distinguish target element from distractors?")
    print(f"{'Layer':>6} | {'AUC':>8} | {'Accuracy':>10} | {'n_samples':>10}")
    print("-" * 50)

    probe_a_results = {}
    for li in KEY_LAYERS:
        if not probe_a_features[li]:
            continue

        # Concatenate all features and labels
        all_feats = np.concatenate([f for f, _ in probe_a_features[li]], axis=0)
        all_labels = np.concatenate([l for _, l in probe_a_features[li]], axis=0)

        if len(np.unique(all_labels)) < 2:
            print(f"{li:>6} | {'N/A':>8} | {'N/A':>10} | {len(all_labels):>10}")
            continue

        # Standardize
        scaler = StandardScaler()
        all_feats_scaled = scaler.fit_transform(all_feats)

        # Train/test split
        n = len(all_labels)
        indices = np.random.permutation(n)
        split = int(0.7 * n)
        train_idx, test_idx = indices[:split], indices[split:]

        X_train, y_train = all_feats_scaled[train_idx], all_labels[train_idx]
        X_test, y_test = all_feats_scaled[test_idx], all_labels[test_idx]

        # Logistic Regression
        clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
        clf.fit(X_train, y_train)

        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        auc = roc_auc_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)

        print(f"{li:>6} | {auc:>8.4f} | {acc:>10.4f} | {n:>10}")
        probe_a_results[li] = {"auc": round(auc, 4), "accuracy": round(acc, 4), "n": n}

    results["probe_A_visual_feature"] = probe_a_results

    # ══════════════════════════════════════════════════════════════
    # Probe B: Coordinate Regression — last token → GT coordinates
    # ══════════════════════════════════════════════════════════════
    print("\n── Probe B: Coordinate Regression ──")
    print("Can the last token's representation predict GT click coordinates?")

    gt_coords = np.array(probe_b_targets, dtype=np.float32)  # (n, 2)

    # Normalize coordinates by image size for fair comparison
    orig_sizes = np.array([m["orig_size"] for m in sample_meta], dtype=np.float32)
    gt_coords_norm = gt_coords / orig_sizes  # normalize to [0, 1]

    # Compute random baseline: average distance when guessing screen center
    # Random uniform baseline: E[dist] for uniform random guess on [0,W]x[0,H]
    # Use Monte Carlo for each test sample
    rng = np.random.RandomState(42)

    # Hit thresholds in pixels
    HIT_THRESHOLDS = [50, 100, 200]

    header = (f"{'Layer':>6} | {'MeanDist':>9} | {'MedianDist':>11} | "
              + " | ".join(f"{'Hit@'+str(t)+'px':>9}" for t in HIT_THRESHOLDS)
              + f" | {'BBox_Hit':>9} | {'n':>5}")
    print(header)
    print("-" * len(header))

    probe_b_results = {}
    random_baseline_dists = None  # computed once

    for li in KEY_LAYERS:
        if not probe_b_features[li]:
            continue

        feats = np.stack(probe_b_features[li])  # (n, hidden_dim)
        n = len(feats)

        scaler = StandardScaler()
        feats_scaled = scaler.fit_transform(feats)

        indices = np.random.permutation(n)
        split = int(0.7 * n)
        train_idx, test_idx = indices[:split], indices[split:]

        X_train, y_train = feats_scaled[train_idx], gt_coords_norm[train_idx]
        X_test, y_test = feats_scaled[test_idx], gt_coords_norm[test_idx]

        reg = Ridge(alpha=1.0)
        reg.fit(X_train, y_train)

        y_pred = reg.predict(X_test)

        # Denormalize to pixel coordinates
        test_orig_sizes = orig_sizes[test_idx]
        y_pred_abs = y_pred * test_orig_sizes
        y_test_abs = y_test * test_orig_sizes

        # Euclidean distance in pixels
        dists = np.sqrt(np.sum((y_pred_abs - y_test_abs) ** 2, axis=1))
        mean_dist = float(np.mean(dists))
        median_dist = float(np.median(dists))

        # Hit rate at thresholds
        hit_rates = {}
        for thresh in HIT_THRESHOLDS:
            hit_rates[thresh] = float(np.mean(dists <= thresh))

        # BBox hit
        n_hit = 0
        for j, tidx in enumerate(test_idx):
            meta = sample_meta[tidx]
            bbox = meta["gt_bbox"]
            px, py = y_pred_abs[j]
            if (bbox["left"] <= px <= bbox["right"] and
                    bbox["top"] <= py <= bbox["bottom"]):
                n_hit += 1
        bbox_hit = n_hit / len(test_idx)

        # Compute random baseline once (using test set geometry)
        if random_baseline_dists is None:
            rand_preds = rng.uniform(size=(len(test_idx), 2)) * test_orig_sizes
            random_baseline_dists = np.sqrt(np.sum((rand_preds - y_test_abs) ** 2, axis=1))

        row = (f"{li:>6} | {mean_dist:>8.1f}px | {median_dist:>9.1f}px | "
               + " | ".join(f"{hit_rates[t]:>9.1%}" for t in HIT_THRESHOLDS)
               + f" | {bbox_hit:>9.1%} | {n:>5}")
        print(row)

        probe_b_results[li] = {
            "mean_dist_px": round(mean_dist, 1),
            "median_dist_px": round(median_dist, 1),
            **{f"hit_at_{t}px": round(hit_rates[t], 4) for t in HIT_THRESHOLDS},
            "bbox_hit": round(bbox_hit, 4),
            "n": n,
        }

    # Print random baseline
    if random_baseline_dists is not None:
        rand_mean = float(np.mean(random_baseline_dists))
        rand_median = float(np.median(random_baseline_dists))
        rand_hits = {t: float(np.mean(random_baseline_dists <= t)) for t in HIT_THRESHOLDS}
        row = (f"{'Random':>6} | {rand_mean:>8.1f}px | {rand_median:>9.1f}px | "
               + " | ".join(f"{rand_hits[t]:>9.1%}" for t in HIT_THRESHOLDS)
               + f" | {'—':>9} | {'—':>5}")
        print(row)
        probe_b_results["random_baseline"] = {
            "mean_dist_px": round(rand_mean, 1),
            "median_dist_px": round(rand_median, 1),
            **{f"hit_at_{t}px": round(rand_hits[t], 4) for t in HIT_THRESHOLDS},
        }

    results["probe_B_coordinate_regression"] = probe_b_results

    # ══════════════════════════════════════════════════════════════
    # Probe C: Cross-Modal Alignment
    # ══════════════════════════════════════════════════════════════
    print("\n── Probe C: Cross-Modal Alignment ──")
    print("How well do target image tokens align with task text tokens?")
    print(f"{'Layer':>6} | {'target-task':>12} | {'nontarget-task':>14} | {'gap':>8}")
    print("-" * 55)

    probe_c_results = {}
    for li in KEY_LAYERS:
        target_imgs = probe_c_data[li]["target_img"]
        nontarget_imgs = probe_c_data[li]["nontarget_img"]
        task_texts = probe_c_data[li]["task_text"]

        if not target_imgs:
            continue

        target_sims = []
        nontarget_sims = []
        for i in range(len(target_imgs)):
            t_img = target_imgs[i]
            nt_img = nontarget_imgs[i]
            task = task_texts[i]

            # Cosine similarity
            t_norm = np.linalg.norm(t_img)
            nt_norm = np.linalg.norm(nt_img)
            task_norm = np.linalg.norm(task)

            if t_norm > 0 and task_norm > 0:
                target_sims.append(np.dot(t_img, task) / (t_norm * task_norm))
            if nt_norm > 0 and task_norm > 0:
                nontarget_sims.append(np.dot(nt_img, task) / (nt_norm * task_norm))

        mean_target_sim = np.mean(target_sims) if target_sims else 0
        mean_nontarget_sim = np.mean(nontarget_sims) if nontarget_sims else 0
        gap = mean_target_sim - mean_nontarget_sim

        print(f"{li:>6} | {mean_target_sim:>12.4f} | {mean_nontarget_sim:>14.4f} | {gap:>+8.4f}")
        probe_c_results[li] = {
            "target_task_sim": round(float(mean_target_sim), 4),
            "nontarget_task_sim": round(float(mean_nontarget_sim), 4),
            "gap": round(float(gap), 4),
        }

    results["probe_C_crossmodal_alignment"] = probe_c_results

    # ══════════════════════════════════════════════════════════════
    # Summary & Interpretation
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    # Probe A interpretation
    if probe_a_results:
        last_layer_auc = probe_a_results.get(27, {}).get("auc", 0)
        early_layer_auc = probe_a_results.get(0, {}).get("auc", 0)
        mid_layer_auc = probe_a_results.get(14, {}).get("auc", 0)
        print(f"\nProbe A (Visual Feature Quality):")
        print(f"  Early (L0) AUC={early_layer_auc:.4f}, "
              f"Mid (L14) AUC={mid_layer_auc:.4f}, "
              f"Late (L27) AUC={last_layer_auc:.4f}")
        if last_layer_auc > 0.80:
            print(f"  → Visual features ARE discriminative (AUC={last_layer_auc:.2f} > 0.80)")
            print(f"  → Failure Point A (visual feature) is NOT the bottleneck")
        elif last_layer_auc > 0.65:
            print(f"  → Visual features are MODERATELY discriminative (AUC={last_layer_auc:.2f})")
            print(f"  → Partial Failure Point A — room for improvement")
        else:
            print(f"  → Visual features are POOR (AUC={last_layer_auc:.2f} < 0.65)")
            print(f"  → Failure Point A is likely the bottleneck")

        # Check if early layers are better (forgetting)
        if early_layer_auc > last_layer_auc + 0.05:
            print(f"  ⚠ Visual info DEGRADES from L0 ({early_layer_auc:.2f}) to L27 ({last_layer_auc:.2f})")
            print(f"  → FFN layers may be 'forgetting' visual features")

    # Probe B interpretation
    if probe_b_results:
        last_layer = probe_b_results.get(27, {})
        last_mean = last_layer.get("mean_dist_px", 0)
        last_hit100 = last_layer.get("hit_at_100px", 0)
        last_bbox = last_layer.get("bbox_hit", 0)
        rand = probe_b_results.get("random_baseline", {})
        rand_mean = rand.get("mean_dist_px", 0)
        rand_hit100 = rand.get("hit_at_100px", 0)
        print(f"\nProbe B (Coordinate Regression):")
        print(f"  Late (L27): MeanDist={last_mean:.0f}px, Hit@100px={last_hit100:.1%}, BBox_Hit={last_bbox:.1%}")
        print(f"  Random baseline: MeanDist={rand_mean:.0f}px, Hit@100px={rand_hit100:.1%}")
        if last_hit100 > 0.30:
            print(f"  → Spatial information IS present in last token representation")
            print(f"  → Failure Point C (coordinate regression) is NOT the main bottleneck")
        elif last_hit100 > 0.10:
            print(f"  → Some spatial information, but noisy (better than random by {last_hit100 - rand_hit100:.1%})")
            print(f"  → Partial Failure Point C")
        else:
            print(f"  → Very little spatial info in last token")
            print(f"  → Failure Point C may be a bottleneck")

    # Probe C interpretation
    if probe_c_results:
        last_gap = probe_c_results.get(27, {}).get("gap", 0)
        early_gap = probe_c_results.get(0, {}).get("gap", 0)
        print(f"\nProbe C (Cross-Modal Alignment):")
        print(f"  Early (L0) gap={early_gap:+.4f}, Late (L27) gap={last_gap:+.4f}")
        if last_gap > 0.05:
            print(f"  → Target image tokens DO align with task text (gap={last_gap:+.4f})")
            print(f"  → Failure Point B (binding) is NOT the main bottleneck")
        elif last_gap > 0.01:
            print(f"  → Weak alignment between target image and task text")
            print(f"  → Partial Failure Point B")
        else:
            print(f"  → No meaningful alignment (gap={last_gap:+.4f})")
            print(f"  → Failure Point B (cross-modal binding) is likely a bottleneck")

    print("\n" + "=" * 80)

    # Save results
    def _json_default(obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(os.path.join(args.output_dir, "probing_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=_json_default)

    return results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Cross-Modal Probing Diagnostic")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--lora_path", default=None, help="Path to PEFT LoRA adapter")
    parser.add_argument("--parquet_file", required=True)
    parser.add_argument("--image_base", default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1")
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    print(f"Model: {args.model_path}")
    if args.lora_path:
        print(f"LoRA: {args.lora_path}")
    print(f"Data: {args.parquet_file}")
    print(f"Samples: {args.n_samples}")
    print(f"Output: {args.output_dir}")
    print(f"Key layers: {KEY_LAYERS}")
    print()

    # Load model — need hidden states, no attention needed
    print("Loading model (bfloat16, output_hidden_states)...", flush=True)
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    if args.lora_path:
        from peft import PeftModel
        print(f"Loading LoRA from {args.lora_path}...", flush=True)
        model = PeftModel.from_pretrained(model, args.lora_path)
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

    run_probing(model, processor, tokenizer, df, args)


if __name__ == "__main__":
    main()
