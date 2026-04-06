#!/usr/bin/env python3
"""
Exp E + F: Binding Analysis

Exp E: Analyze WHAT non-target tokens are most similar to task text.
       Map top-K similar image tokens to spatial positions and categorize.

Exp F: Linear binding probe — does concat(image_token, task_text) improve
       target detection over image_token alone?

Usage:
  python binding_analysis.py --model_path /path/to/model --n_samples 300
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
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

VISION_START_ID = 151652
VISION_END_ID = 151653
SPATIAL_MERGE_SIZE = 2
PATCH_SIZE = 14
TOKEN_PIXEL_SIZE = SPATIAL_MERGE_SIZE * PATCH_SIZE  # 28

KEY_LAYERS = [0, 4, 9, 14, 19, 24, 27]
TOP_K = 10


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


def identify_text_regions(input_ids, tokenizer):
    ids = input_ids.squeeze().tolist()
    cum_len = 0
    token_char_starts = []
    for tid in ids:
        token_char_starts.append(cum_len)
        cum_len += len(tokenizer.decode([tid]))
    full_text = tokenizer.decode(ids)

    def find_pos(marker):
        pos = full_text.rfind(marker)
        if pos == -1:
            return None
        for i in range(len(token_char_starts) - 1, -1, -1):
            if token_char_starts[i] <= pos:
                return i
        return None

    instr_pos = find_pos("instruction is:\n")
    hist_pos = find_pos("history of actions are:\n")
    act_pos = find_pos("actions supported are:\n")

    task_indices = []
    if instr_pos is not None:
        task_end = hist_pos if hist_pos is not None else (
            act_pos if act_pos is not None else len(ids))
        if hist_pos is not None:
            task_end = max(0, hist_pos - 2)
        task_start = max(0, instr_pos - 2)
        task_indices = list(range(task_start, task_end))

    return task_indices


def run_analysis(model, processor, tokenizer, df, args):
    base_dir = args.image_base

    # ── Exp E storage ──
    # For each sample: where are top-K similar tokens relative to GT target?
    exp_e_data = {li: [] for li in KEY_LAYERS}

    # ── Exp F storage ──
    # Probe A (image only) vs Probe F (image + task concat)
    exp_f_img_only = {li: [] for li in KEY_LAYERS}    # (feat, label) for image-only
    exp_f_concat = {li: [] for li in KEY_LAYERS}       # (feat, label) for concat

    n_processed = 0

    for idx in range(len(df)):
        t0 = time.time()
        row = df.iloc[idx]
        messages = row["messages"]
        if isinstance(messages, str):
            messages = json.loads(messages)

        gt_action = parse_tool_call(messages[1]["content"])
        if gt_action is None:
            continue
        gt_bbox = gt_action.get("bbox")
        if not gt_bbox or any(gt_bbox.get(k) is None for k in ["left", "top", "right", "bottom"]):
            continue
        gt_coord = gt_action.get("args", {}).get("coordinate", [])
        if len(gt_coord) != 2 or gt_coord[0] is None:
            continue

        text_content = ""
        image_path = None
        for item in messages[0]["content"]:
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
        except Exception:
            continue

        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]

        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is None or len(image_grid_thw) == 0:
            continue
        grid_thw = image_grid_thw[0].tolist()

        positions, token_h, token_w = get_image_token_positions(grid_thw)
        n_image_tokens = token_h * token_w

        resized_h = grid_thw[1] * PATCH_SIZE
        resized_w = grid_thw[2] * PATCH_SIZE
        scale_w = resized_w / orig_w
        scale_h = resized_h / orig_h

        # GT bbox in resized space
        bl = gt_bbox["left"] * scale_w
        bt = gt_bbox["top"] * scale_h
        br = gt_bbox["right"] * scale_w
        bb = gt_bbox["bottom"] * scale_h
        gt_center_x = (bl + br) / 2
        gt_center_y = (bt + bb) / 2

        # Image token range
        ids_list = input_ids.squeeze().tolist()
        img_start = img_end = None
        for i, t in enumerate(ids_list):
            if t == VISION_START_ID and img_start is None:
                img_start = i
            if t == VISION_END_ID:
                img_end = i

        if img_start is None or img_end is None:
            continue

        img_token_start = img_start + 1
        img_token_end = img_end
        actual_n = img_token_end - img_token_start
        if actual_n != n_image_tokens:
            continue

        # Find target token indices (overlapping with GT bbox)
        target_set = set()
        for i, (x1, y1, x2, y2) in enumerate(positions):
            if x2 > bl and x1 < br and y2 > bt and y1 < bb:
                target_set.add(i)

        if len(target_set) == 0:
            continue

        # Task text tokens
        task_indices = identify_text_regions(input_ids, tokenizer)

        # ── Forward pass ──
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        hidden_states = outputs.hidden_states

        for li in KEY_LAYERS:
            if li + 1 >= len(hidden_states):
                continue
            hs = hidden_states[li + 1][0].float()  # (seq_len, hidden_dim)

            # Image token features
            img_feats = hs[img_token_start:img_token_end]  # (n_img, dim)

            # Task text mean
            if len(task_indices) > 0:
                task_mean = hs[task_indices].mean(dim=0)  # (dim,)
            else:
                continue

            # ── Exp E: find top-K image tokens most similar to task text ──
            img_norms = img_feats.norm(dim=1, keepdim=True).clamp(min=1e-8)
            task_norm = task_mean.norm().clamp(min=1e-8)
            sims = (img_feats @ task_mean) / (img_norms.squeeze() * task_norm)  # (n_img,)

            topk_vals, topk_ids = sims.topk(min(TOP_K, len(sims)))

            # Categorize top-K tokens
            n_in_target = 0
            distances_to_gt = []
            for rank, tid in enumerate(topk_ids.tolist()):
                if tid in target_set:
                    n_in_target += 1
                # Distance to GT center
                tx1, ty1, tx2, ty2 = positions[tid]
                tcx = (tx1 + tx2) / 2
                tcy = (ty1 + ty2) / 2
                dist = ((tcx - gt_center_x)**2 + (tcy - gt_center_y)**2)**0.5
                distances_to_gt.append(dist)

            # Also check: what's the rank of the best target token?
            target_sims = sims[list(target_set)]
            best_target_sim = target_sims.max().item() if len(target_sims) > 0 else 0
            # Rank of best target token among all image tokens
            best_target_rank = (sims > best_target_sim).sum().item()

            exp_e_data[li].append({
                "n_in_target_topk": n_in_target,
                "topk_mean_dist_to_gt": float(np.mean(distances_to_gt)),
                "topk_min_dist_to_gt": float(np.min(distances_to_gt)),
                "best_target_rank": best_target_rank,
                "best_target_sim": float(best_target_sim),
                "topk_mean_sim": float(topk_vals.mean()),
                "n_target_tokens": len(target_set),
                "n_image_tokens": n_image_tokens,
            })

            # ── Exp F: collect features for binding probe ──
            # Subsample for memory
            target_list = list(target_set)
            n_nt_sample = min(len(target_list) * 3, n_image_tokens - len(target_set), 30)
            nontarget_list = [i for i in range(n_image_tokens) if i not in target_set]
            if len(nontarget_list) > n_nt_sample:
                nontarget_list = np.random.choice(nontarget_list, n_nt_sample, replace=False).tolist()

            all_indices = target_list + nontarget_list
            labels = np.array([1] * len(target_list) + [0] * len(nontarget_list))

            # Image-only features
            img_only_feats = img_feats[all_indices].cpu().numpy()
            exp_f_img_only[li].append((img_only_feats, labels))

            # Concat features: [image_token; task_text_mean]
            task_expanded = task_mean.unsqueeze(0).expand(len(all_indices), -1)
            concat_feats = torch.cat([img_feats[all_indices], task_expanded], dim=-1).cpu().numpy()
            exp_f_concat[li].append((concat_feats, labels))

        del outputs, hidden_states
        torch.cuda.empty_cache()
        gc.collect()

        n_processed += 1
        if n_processed % 20 == 0:
            elapsed = time.time() - t0
            print(f"[{n_processed}/{len(df)}] {elapsed:.1f}s/sample")

    print(f"\nProcessed {n_processed} samples")

    # ── Report Exp E ──
    report_exp_e(exp_e_data, args)

    # ── Report Exp F ──
    report_exp_f(exp_f_img_only, exp_f_concat, args)


def report_exp_e(exp_e_data, args):
    print("\n" + "=" * 80)
    print("EXP E: WHERE ARE THE TOP-K MOST SIMILAR TOKENS?")
    print("=" * 80)
    print(f"(Top-{TOP_K} image tokens most similar to task text)")

    print(f"\n{'Layer':>6} | {'%inTarget':>10} | {'MeanDist':>10} | {'BestTgtRank':>12} | {'TopK_sim':>9} | {'BestTgt_sim':>12}")
    print("-" * 75)

    results = {}
    for li in KEY_LAYERS:
        data = exp_e_data[li]
        if not data:
            continue
        pct_in_target = np.mean([d["n_in_target_topk"] / TOP_K for d in data])
        mean_dist = np.mean([d["topk_mean_dist_to_gt"] for d in data])
        mean_best_rank = np.mean([d["best_target_rank"] for d in data])
        mean_topk_sim = np.mean([d["topk_mean_sim"] for d in data])
        mean_best_tgt_sim = np.mean([d["best_target_sim"] for d in data])

        print(f"{li:>6} | {pct_in_target:>9.1%} | {mean_dist:>10.1f} | {mean_best_rank:>12.1f} | {mean_topk_sim:>9.4f} | {mean_best_tgt_sim:>12.4f}")

        results[li] = {
            "pct_target_in_topk": round(float(pct_in_target), 4),
            "mean_dist_to_gt": round(float(mean_dist), 1),
            "best_target_rank": round(float(mean_best_rank), 1),
            "topk_mean_sim": round(float(mean_topk_sim), 4),
            "best_target_sim": round(float(mean_best_tgt_sim), 4),
        }

    print(f"\nInterpretation:")
    l14 = results.get(14, {})
    pct = l14.get("pct_target_in_topk", 0)
    rank = l14.get("best_target_rank", 999)
    if pct < 0.05:
        print(f"  < 5% of top-{TOP_K} tokens are in GT target → model NOT binding to correct element")
        print(f"  Best target token rank = {rank:.0f} (out of ~hundreds) → target is buried")
    elif pct < 0.20:
        print(f"  ~{pct:.0%} of top-{TOP_K} overlap with target → weak/partial binding")
    else:
        print(f"  {pct:.0%} of top-{TOP_K} overlap with target → binding partially works")

    with open(os.path.join(args.output_dir, "exp_e_results.json"), "w") as f:
        json.dump(results, f, indent=2)


def report_exp_f(img_only_data, concat_data, args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    print("\n" + "=" * 80)
    print("EXP F: LINEAR BINDING PROBE")
    print("=" * 80)
    print("Probe A (image only) vs Probe F (image + task text concat)")

    print(f"\n{'Layer':>6} | {'AUC_imgOnly':>12} | {'AUC_concat':>12} | {'Δ AUC':>8} | {'n':>8}")
    print("-" * 55)

    results = {}
    for li in KEY_LAYERS:
        if not img_only_data[li] or not concat_data[li]:
            continue

        # Image-only probe
        X_img = np.concatenate([f for f, _ in img_only_data[li]])
        y = np.concatenate([l for _, l in img_only_data[li]])

        if len(np.unique(y)) < 2:
            continue

        n = len(y)
        indices = np.random.permutation(n)
        split = int(0.7 * n)
        train_idx, test_idx = indices[:split], indices[split:]

        scaler_img = StandardScaler()
        X_img_scaled = scaler_img.fit_transform(X_img)

        clf_img = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
        clf_img.fit(X_img_scaled[train_idx], y[train_idx])
        auc_img = roc_auc_score(y[test_idx], clf_img.predict_proba(X_img_scaled[test_idx])[:, 1])

        # Concat probe
        X_cat = np.concatenate([f for f, _ in concat_data[li]])
        scaler_cat = StandardScaler()
        X_cat_scaled = scaler_cat.fit_transform(X_cat)

        clf_cat = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
        clf_cat.fit(X_cat_scaled[train_idx], y[train_idx])
        auc_cat = roc_auc_score(y[test_idx], clf_cat.predict_proba(X_cat_scaled[test_idx])[:, 1])

        delta = auc_cat - auc_img
        print(f"{li:>6} | {auc_img:>12.4f} | {auc_cat:>12.4f} | {delta:>+8.4f} | {n:>8}")

        results[li] = {
            "auc_image_only": round(auc_img, 4),
            "auc_concat": round(auc_cat, 4),
            "delta": round(delta, 4),
            "n": n,
        }

    print(f"\nInterpretation:")
    l14 = results.get(14, {})
    delta = l14.get("delta", 0)
    if delta > 0.03:
        print(f"  Concat AUC >> img-only AUC (Δ={delta:+.4f})")
        print(f"  → Task text DOES help binding → info is available, model just doesn't use it")
        print(f"  → Light binding adapter could work")
    elif delta > -0.01:
        print(f"  Concat ≈ img-only (Δ={delta:+.4f})")
        print(f"  → Task text adds no binding info → cross-modal interaction is missing")
        print(f"  → Need deeper cross-modal training")
    else:
        print(f"  Concat HURTS (Δ={delta:+.4f})")
        print(f"  → Task text INTERFERES with binding → representations are misaligned")
        print(f"  → Need contrastive realignment before binding")

    print("=" * 80)

    with open(os.path.join(args.output_dir, "exp_f_results.json"), "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Binding Analysis (Exp E + F)")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--parquet_file", required=True)
    parser.add_argument("--image_base", default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1")
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    print(f"Model: {args.model_path}")
    print(f"Samples: {args.n_samples}")
    print(f"Output: {args.output_dir}")
    print()

    print("Loading model (bfloat16)...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded", flush=True)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer

    df = pd.read_parquet(args.parquet_file)
    if 0 < args.n_samples < len(df):
        indices = np.random.choice(len(df), args.n_samples, replace=False)
        df = df.iloc[indices].reset_index(drop=True)
    print(f"Processing {len(df)} samples\n")

    run_analysis(model, processor, tokenizer, df, args)


if __name__ == "__main__":
    main()
