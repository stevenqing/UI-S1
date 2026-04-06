#!/usr/bin/env python3
"""
E_NEW1B Stage 2: Coordinate Spread for Grounding Error Detection
=================================================================
In a two-stage router design:
  Stage 1: agreement_rate filters out action errors (low agreement) vs
           high-agreement steps (likely pass_through or grounding).
  Stage 2: Within the high-agreement subset, can coordinate_spread
           distinguish grounding errors from true pass-through steps?

This script tests whether coord_spread (std of predicted coordinates across
K=10 samples) can detect grounding errors among steps where action-type
agreement is high (>= 0.9).

Key question: Is AUROC > 0.6 for coord_spread predicting grounding error
within the high-agreement, coordinate-based action subset?

Data source: multisample_results.jsonl (K=10 samples per step)

Routing labels (from greedy = sample[0]):
  - pass_through: type_match=True AND extract_match=True
  - action:       type_match=False
  - grounding:    type_match=True AND extract_match=False
"""

import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = (
    "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/outputs/eval_c4c7_ac/"
    "Qwen2.5-VL-7B/multisample_results.jsonl"
)
OUTPUT_DIR = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/outputs/eval_e_new1b"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "e_new1b_stage2_coord_spread.json")

# Coordinate-based action types (have x,y coordinates to measure spread)
COORD_ACTION_TYPES = {"click", "long_press"}
# Non-coordinate action types
NON_COORD_ACTION_TYPES = {"swipe", "type", "wait", "open", "system_button"}
# Agreement threshold for "high agreement" (Stage 1 pass)
AGREEMENT_THRESHOLD = 0.9

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def compute_coord_spread(coords_x, coords_y):
    """Compute coordinate spread as mean of x_std and y_std.
    Also returns euclidean std (std of distances from centroid)."""
    if len(coords_x) < 2:
        return {"x_std": float("nan"), "y_std": float("nan"),
                "mean_std": float("nan"), "euclidean_std": float("nan")}
    x_std = float(np.std(coords_x))
    y_std = float(np.std(coords_y))
    mean_std = (x_std + y_std) / 2.0
    # Euclidean: std of distances from centroid
    cx, cy = np.mean(coords_x), np.mean(coords_y)
    dists = np.sqrt((np.array(coords_x) - cx) ** 2 + (np.array(coords_y) - cy) ** 2)
    euclidean_std = float(np.std(dists))
    return {"x_std": x_std, "y_std": y_std,
            "mean_std": mean_std, "euclidean_std": euclidean_std}


def auroc_manual(scores, labels):
    """Compute AUROC manually (no sklearn dependency for this).
    scores: higher = more likely positive.
    labels: 1 = positive, 0 = negative.
    Returns AUROC or NaN if degenerate."""
    pairs = list(zip(scores, labels))
    # Remove NaN scores
    pairs = [(s, l) for s, l in pairs if not math.isnan(s)]
    if not pairs:
        return float("nan")
    positives = sum(1 for _, l in pairs if l == 1)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return float("nan")

    # Sort by score descending
    pairs.sort(key=lambda x: -x[0])
    tp = 0
    fp = 0
    auc = 0.0
    prev_score = None
    prev_tp = 0
    prev_fp = 0
    for score, label in pairs:
        if prev_score is not None and score != prev_score:
            # Trapezoid
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
            prev_tp = tp
            prev_fp = fp
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
    return auc / (positives * negatives)


def precision_recall_at_thresholds(scores, labels, thresholds):
    """Compute precision, recall, F1 at given thresholds.
    Positive prediction = score >= threshold."""
    results = []
    for thr in thresholds:
        preds = [1 if s >= thr else 0 for s in scores]
        tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results.append({
            "threshold": round(thr, 1),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "n_flagged": sum(preds),
            "tp": tp, "fp": fp, "fn": fn,
        })
    return results


# ===================================================================
# 1. Load all steps
# ===================================================================
print("=" * 72)
print("E_NEW1B Stage 2: Coordinate Spread for Grounding Error Detection")
print("=" * 72)
print(f"\nData: {DATA_PATH}")

steps = []
with open(DATA_PATH) as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            ep = json.loads(line)
        except json.JSONDecodeError:
            print(f"  [WARN] Skipping malformed JSON at line {line_num}")
            continue
        episode_id = ep.get("episode_id", f"ep_{line_num}")
        num_steps = ep.get("num_steps", 0)
        for step_data in ep.get("step_samples", []):
            step_data["_episode_id"] = episode_id
            step_data["_num_steps"] = num_steps
            steps.append(step_data)

print(f"Loaded {len(steps)} steps from {line_num} episodes.\n")

# ===================================================================
# 2. Process each step: label, agreement, coord_spread
# ===================================================================
records = []

for step in steps:
    samples = step.get("samples", [])
    if not samples:
        continue

    gt_action_type = step.get("gt_action_type", "unknown")
    step_num = step.get("step_num", 0)
    episode_id = step.get("_episode_id")

    # Greedy = first sample
    greedy = samples[0]
    type_match = greedy.get("type_match", False)
    extract_match = greedy.get("extract_match", False)

    # Routing label
    if type_match and extract_match:
        label = "pass_through"
    elif not type_match:
        label = "action"
    else:
        label = "grounding"

    # ---- Features from K samples ----
    pred_action_types = []
    coords_x = []
    coords_y = []
    for s in samples:
        pa = s.get("pred_action")
        if pa is None or not isinstance(pa, dict):
            pred_action_types.append("_none_")
        else:
            at = pa.get("action", "_none_")
            pred_action_types.append(at)
            coord = pa.get("coordinate")
            if coord and isinstance(coord, (list, tuple)) and len(coord) >= 2:
                try:
                    cx, cy = float(coord[0]), float(coord[1])
                    if abs(cx) < 1e6 and abs(cy) < 1e6:
                        coords_x.append(cx)
                        coords_y.append(cy)
                except (ValueError, TypeError):
                    pass

    type_counter = Counter(pred_action_types)
    K = len(samples)

    # agreement_rate: fraction sharing mode type
    mode_type, mode_count = type_counter.most_common(1)[0]
    agreement_rate = mode_count / K

    # Coordinate spread
    spread = compute_coord_spread(coords_x, coords_y)

    # Is this a coordinate-based action (by ground truth)?
    is_coord_action_gt = gt_action_type in COORD_ACTION_TYPES
    # Is this a coordinate-based action (by predicted mode type)?
    is_coord_action_pred = mode_type in COORD_ACTION_TYPES

    records.append({
        "label": label,
        "gt_action_type": gt_action_type,
        "pred_mode_type": mode_type,
        "agreement_rate": agreement_rate,
        "is_coord_action_gt": is_coord_action_gt,
        "is_coord_action_pred": is_coord_action_pred,
        "coord_spread_mean_std": spread["mean_std"],
        "coord_spread_x_std": spread["x_std"],
        "coord_spread_y_std": spread["y_std"],
        "coord_spread_euclidean": spread["euclidean_std"],
        "n_coord_samples": len(coords_x),
        "episode_id": episode_id,
        "step_num": step_num,
    })

N = len(records)
print(f"Total steps: {N}")

# ===================================================================
# 3. Overall label distribution
# ===================================================================
label_counter = Counter(r["label"] for r in records)
print(f"\nOverall label distribution:")
for lbl in ["pass_through", "action", "grounding"]:
    cnt = label_counter[lbl]
    print(f"  {lbl:15s}: {cnt:5d}  ({100*cnt/N:.1f}%)")

# ===================================================================
# 4. High-agreement subset analysis
# ===================================================================
print("\n" + "=" * 72)
print(f"HIGH-AGREEMENT SUBSET (agreement_rate >= {AGREEMENT_THRESHOLD})")
print("=" * 72)

high_agree = [r for r in records if r["agreement_rate"] >= AGREEMENT_THRESHOLD]
N_ha = len(high_agree)
print(f"\nTotal high-agreement steps: {N_ha} / {N} ({100*N_ha/N:.1f}%)")

ha_label_counter = Counter(r["label"] for r in high_agree)
print(f"\nLabel distribution in high-agreement subset:")
for lbl in ["pass_through", "action", "grounding"]:
    cnt = ha_label_counter[lbl]
    pct = 100 * cnt / N_ha if N_ha > 0 else 0
    print(f"  {lbl:15s}: {cnt:5d}  ({pct:.1f}%)")

# Also show by GT action type
ha_at_counter = Counter(r["gt_action_type"] for r in high_agree)
print(f"\nGT action type distribution in high-agreement subset:")
for at, cnt in ha_at_counter.most_common():
    print(f"  {at:15s}: {cnt:5d}  ({100*cnt/N_ha:.1f}%)")

# ===================================================================
# 5. Stage 2 analysis: coord_spread for grounding detection
#    within high-agreement + coordinate-based steps
# ===================================================================
print("\n" + "=" * 72)
print("STAGE 2: Coord Spread Analysis (High-Agreement + Coordinate Actions)")
print("=" * 72)

# Filter: high agreement + coordinate-based GT action + not "action" label
# (action label means type mismatch -- those are filtered by Stage 1)
ha_coord = [r for r in high_agree
            if r["is_coord_action_gt"] and r["label"] in ("pass_through", "grounding")]
N_hac = len(ha_coord)
hac_label_counter = Counter(r["label"] for r in ha_coord)

print(f"\nHigh-agreement + coord-based + (pass_through|grounding): {N_hac}")
for lbl in ["pass_through", "grounding"]:
    cnt = hac_label_counter[lbl]
    pct = 100 * cnt / N_hac if N_hac > 0 else 0
    print(f"  {lbl:15s}: {cnt:5d}  ({pct:.1f}%)")

# Extract scores and labels
hac_scores = [r["coord_spread_mean_std"] for r in ha_coord]
hac_binary = [1 if r["label"] == "grounding" else 0 for r in ha_coord]

# Filter out NaN
valid_mask = [not math.isnan(s) for s in hac_scores]
hac_scores_valid = [s for s, v in zip(hac_scores, valid_mask) if v]
hac_binary_valid = [b for b, v in zip(hac_binary, valid_mask) if v]
n_valid = len(hac_scores_valid)
n_nan = sum(1 for v in valid_mask if not v)
print(f"\n  Valid (non-NaN coord_spread): {n_valid}  (NaN dropped: {n_nan})")

# Mean coord_spread by label
grounding_spreads = [s for s, b in zip(hac_scores_valid, hac_binary_valid) if b == 1]
passthrough_spreads = [s for s, b in zip(hac_scores_valid, hac_binary_valid) if b == 0]

print(f"\n  Mean coord_spread (mean_std):")
if grounding_spreads:
    print(f"    grounding:    {np.mean(grounding_spreads):.2f}  (median: {np.median(grounding_spreads):.2f}, n={len(grounding_spreads)})")
if passthrough_spreads:
    print(f"    pass_through: {np.mean(passthrough_spreads):.2f}  (median: {np.median(passthrough_spreads):.2f}, n={len(passthrough_spreads)})")

# Also compute for euclidean spread
hac_euc_scores = [r["coord_spread_euclidean"] for r in ha_coord]
hac_euc_valid = [s for s, v in zip(hac_euc_scores, valid_mask) if v]
grounding_euc = [s for s, b in zip(hac_euc_valid, hac_binary_valid) if b == 1]
passthrough_euc = [s for s, b in zip(hac_euc_valid, hac_binary_valid) if b == 0]

print(f"\n  Mean coord_spread (euclidean_std):")
if grounding_euc:
    print(f"    grounding:    {np.mean(grounding_euc):.2f}  (median: {np.median(grounding_euc):.2f})")
if passthrough_euc:
    print(f"    pass_through: {np.mean(passthrough_euc):.2f}  (median: {np.median(passthrough_euc):.2f})")

# AUROC
auroc_mean_std = auroc_manual(hac_scores_valid, hac_binary_valid)
auroc_euclidean = auroc_manual(hac_euc_valid, hac_binary_valid)

# Also compute x_std and y_std separately
hac_x_scores = [r["coord_spread_x_std"] for r in ha_coord]
hac_y_scores = [r["coord_spread_y_std"] for r in ha_coord]
hac_x_valid = [s for s, v in zip(hac_x_scores, valid_mask) if v]
hac_y_valid = [s for s, v in zip(hac_y_scores, valid_mask) if v]
auroc_x = auroc_manual(hac_x_valid, hac_binary_valid)
auroc_y = auroc_manual(hac_y_valid, hac_binary_valid)

print(f"\n  AUROC (grounding=1 vs pass_through=0, high-agreement + coord-based):")
print(f"    coord_spread (mean_std):    {auroc_mean_std:.4f}")
print(f"    coord_spread (euclidean):   {auroc_euclidean:.4f}")
print(f"    coord_spread (x_std only):  {auroc_x:.4f}")
print(f"    coord_spread (y_std only):  {auroc_y:.4f}")

# Correlation (point-biserial = Pearson with binary variable)
if n_valid > 2 and len(set(hac_binary_valid)) > 1:
    corr = float(np.corrcoef(hac_scores_valid, hac_binary_valid)[0, 1])
else:
    corr = float("nan")
print(f"\n  Correlation (coord_spread_mean_std vs grounding_binary): {corr:.4f}")

# Precision-recall at thresholds
thresholds = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
pr_results = precision_recall_at_thresholds(hac_scores_valid, hac_binary_valid, thresholds)
print(f"\n  Precision-Recall at coord_spread thresholds (predicting grounding error):")
print(f"    {'Thresh':>8s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'Flagged':>8s} {'TP':>5s} {'FP':>5s} {'FN':>5s}")
for pr in pr_results:
    print(f"    {pr['threshold']:8.0f} {pr['precision']:8.4f} {pr['recall']:8.4f} "
          f"{pr['f1']:8.4f} {pr['n_flagged']:8d} {pr['tp']:5d} {pr['fp']:5d} {pr['fn']:5d}")

# ===================================================================
# 6. FULL dataset AUROC (for comparison with high-agreement subset)
# ===================================================================
print("\n" + "=" * 72)
print("FULL DATASET: Coord Spread AUROC (for comparison)")
print("=" * 72)

# All coordinate-based steps (not just high-agreement)
all_coord = [r for r in records
             if r["is_coord_action_gt"] and r["label"] in ("pass_through", "grounding")]
N_ac = len(all_coord)
ac_label_counter = Counter(r["label"] for r in all_coord)
print(f"\nAll coord-based + (pass_through|grounding): {N_ac}")
for lbl in ["pass_through", "grounding"]:
    cnt = ac_label_counter[lbl]
    print(f"  {lbl:15s}: {cnt:5d}  ({100*cnt/N_ac:.1f}%)")

ac_scores = [r["coord_spread_mean_std"] for r in all_coord]
ac_binary = [1 if r["label"] == "grounding" else 0 for r in all_coord]
ac_valid_mask = [not math.isnan(s) for s in ac_scores]
ac_scores_valid = [s for s, v in zip(ac_scores, ac_valid_mask) if v]
ac_binary_valid = [b for b, v in zip(ac_binary, ac_valid_mask) if v]

auroc_full_mean = auroc_manual(ac_scores_valid, ac_binary_valid)
auroc_full_euc = auroc_manual(
    [s for s, v in zip([r["coord_spread_euclidean"] for r in all_coord], ac_valid_mask) if v],
    ac_binary_valid
)

grounding_all = [s for s, b in zip(ac_scores_valid, ac_binary_valid) if b == 1]
passthrough_all = [s for s, b in zip(ac_scores_valid, ac_binary_valid) if b == 0]

print(f"\n  Mean coord_spread (mean_std) -- full dataset:")
if grounding_all:
    print(f"    grounding:    {np.mean(grounding_all):.2f}  (median: {np.median(grounding_all):.2f}, n={len(grounding_all)})")
if passthrough_all:
    print(f"    pass_through: {np.mean(passthrough_all):.2f}  (median: {np.median(passthrough_all):.2f}, n={len(passthrough_all)})")

print(f"\n  AUROC (full dataset, coord-based steps only):")
print(f"    coord_spread (mean_std):    {auroc_full_mean:.4f}")
print(f"    coord_spread (euclidean):   {auroc_full_euc:.4f}")

# ===================================================================
# 7. Click-specific analysis (click is the dominant coord action)
# ===================================================================
print("\n" + "=" * 72)
print("CLICK-SPECIFIC ANALYSIS (High-Agreement + Click Only)")
print("=" * 72)

ha_click = [r for r in high_agree
            if r["gt_action_type"] == "click" and r["label"] in ("pass_through", "grounding")]
N_hac_click = len(ha_click)
hac_click_labels = Counter(r["label"] for r in ha_click)

print(f"\nHigh-agreement + click + (pass_through|grounding): {N_hac_click}")
for lbl in ["pass_through", "grounding"]:
    cnt = hac_click_labels[lbl]
    pct = 100 * cnt / N_hac_click if N_hac_click > 0 else 0
    print(f"  {lbl:15s}: {cnt:5d}  ({pct:.1f}%)")

click_scores = [r["coord_spread_mean_std"] for r in ha_click]
click_binary = [1 if r["label"] == "grounding" else 0 for r in ha_click]
click_valid_mask = [not math.isnan(s) for s in click_scores]
click_scores_valid = [s for s, v in zip(click_scores, click_valid_mask) if v]
click_binary_valid = [b for b, v in zip(click_binary, click_valid_mask) if v]

click_grounding = [s for s, b in zip(click_scores_valid, click_binary_valid) if b == 1]
click_passthrough = [s for s, b in zip(click_scores_valid, click_binary_valid) if b == 0]

print(f"\n  Mean coord_spread (mean_std) -- click only:")
if click_grounding:
    print(f"    grounding:    {np.mean(click_grounding):.2f}  (median: {np.median(click_grounding):.2f}, n={len(click_grounding)})")
if click_passthrough:
    print(f"    pass_through: {np.mean(click_passthrough):.2f}  (median: {np.median(click_passthrough):.2f}, n={len(click_passthrough)})")

auroc_click = auroc_manual(click_scores_valid, click_binary_valid)
auroc_click_euc = auroc_manual(
    [s for s, v in zip([r["coord_spread_euclidean"] for r in ha_click], click_valid_mask) if v],
    click_binary_valid
)

print(f"\n  AUROC (click-specific, high-agreement):")
print(f"    coord_spread (mean_std):    {auroc_click:.4f}")
print(f"    coord_spread (euclidean):   {auroc_click_euc:.4f}")

# Correlation
if len(click_scores_valid) > 2 and len(set(click_binary_valid)) > 1:
    corr_click = float(np.corrcoef(click_scores_valid, click_binary_valid)[0, 1])
else:
    corr_click = float("nan")
print(f"\n  Correlation (click): {corr_click:.4f}")

# PR at thresholds
pr_click = precision_recall_at_thresholds(click_scores_valid, click_binary_valid, thresholds)
print(f"\n  Precision-Recall at thresholds (click-specific):")
print(f"    {'Thresh':>8s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'Flagged':>8s} {'TP':>5s} {'FP':>5s} {'FN':>5s}")
for pr in pr_click:
    print(f"    {pr['threshold']:8.0f} {pr['precision']:8.4f} {pr['recall']:8.4f} "
          f"{pr['f1']:8.4f} {pr['n_flagged']:8d} {pr['tp']:5d} {pr['fp']:5d} {pr['fn']:5d}")

# ===================================================================
# 8. Percentile analysis: distribution of coord_spread by label
# ===================================================================
print("\n" + "=" * 72)
print("DISTRIBUTION ANALYSIS (Percentiles)")
print("=" * 72)

percentiles = [10, 25, 50, 75, 90, 95, 99]
print(f"\n  Coord_spread (mean_std) percentiles -- high-agreement + coord-based:")
print(f"    {'Pctl':>6s} {'Grounding':>12s} {'Pass-thru':>12s}")
for p in percentiles:
    g_val = np.percentile(grounding_spreads, p) if grounding_spreads else float("nan")
    pt_val = np.percentile(passthrough_spreads, p) if passthrough_spreads else float("nan")
    print(f"    {p:5d}% {g_val:12.1f} {pt_val:12.1f}")

# ===================================================================
# 9. Oracle TSR improvement analysis
# ===================================================================
print("\n" + "=" * 72)
print("STAGE 2 VALUE: Oracle TSR Improvement")
print("=" * 72)

# Compute: if we could perfectly detect grounding errors in high-agreement steps,
# what fraction of ALL errors would we catch?
total_grounding_errors = sum(1 for r in records if r["label"] == "grounding")
ha_grounding_errors = sum(1 for r in high_agree if r["label"] == "grounding")
ha_coord_grounding_errors = hac_label_counter.get("grounding", 0)

total_action_errors = sum(1 for r in records if r["label"] == "action")
total_errors = total_grounding_errors + total_action_errors
total_correct = sum(1 for r in records if r["label"] == "pass_through")

print(f"\n  Total steps:              {N}")
print(f"  Total correct (pass-thru):  {total_correct} ({100*total_correct/N:.1f}%)")
print(f"  Total errors:               {total_errors} ({100*total_errors/N:.1f}%)")
print(f"    Action errors:            {total_action_errors} ({100*total_action_errors/N:.1f}%)")
print(f"    Grounding errors:         {total_grounding_errors} ({100*total_grounding_errors/N:.1f}%)")
print(f"\n  High-agreement subset:")
print(f"    Grounding errors in HA:   {ha_grounding_errors} / {total_grounding_errors} "
      f"({100*ha_grounding_errors/max(total_grounding_errors,1):.1f}% of all grounding errors)")
print(f"    Coord-based grounding:    {ha_coord_grounding_errors} / {total_grounding_errors} "
      f"({100*ha_coord_grounding_errors/max(total_grounding_errors,1):.1f}% of all grounding errors)")

# Oracle TSR improvement: if all grounding errors in HA+coord are fixed
# Step accuracy improvement
baseline_step_accuracy = total_correct / N
oracle_corrected = total_correct + ha_coord_grounding_errors
oracle_step_accuracy = oracle_corrected / N

print(f"\n  Baseline step accuracy:      {baseline_step_accuracy:.4f}")
print(f"  Oracle (fix HA coord ground): {oracle_step_accuracy:.4f}")
print(f"  Improvement:                  +{oracle_step_accuracy - baseline_step_accuracy:.4f} "
      f"(+{100*(oracle_step_accuracy - baseline_step_accuracy)/baseline_step_accuracy:.2f}% relative)")

# Trajectory-level analysis
# Group by episode, check if fixing grounding errors in HA+coord would flip trajectory success
episode_steps = {}
for r in records:
    eid = r["episode_id"]
    if eid not in episode_steps:
        episode_steps[eid] = []
    episode_steps[eid].append(r)

ha_coord_grounding_episodes = set()
for r in ha_coord:
    if r["label"] == "grounding":
        ha_coord_grounding_episodes.add(r["episode_id"])

# Compute trajectory success rate (TSR) = all steps pass_through
baseline_traj_correct = 0
oracle_traj_correct = 0
total_trajs = len(episode_steps)

for eid, ep_steps in episode_steps.items():
    all_pass = all(s["label"] == "pass_through" for s in ep_steps)
    if all_pass:
        baseline_traj_correct += 1
        oracle_traj_correct += 1
    else:
        # Oracle: fix all grounding errors in HA+coord subset
        oracle_pass = True
        for s in ep_steps:
            if s["label"] == "pass_through":
                continue
            elif (s["label"] == "grounding" and
                  s["agreement_rate"] >= AGREEMENT_THRESHOLD and
                  s["is_coord_action_gt"]):
                # This one gets fixed by oracle
                continue
            else:
                oracle_pass = False
                break
        if oracle_pass:
            oracle_traj_correct += 1

baseline_tsr = baseline_traj_correct / total_trajs
oracle_tsr = oracle_traj_correct / total_trajs

print(f"\n  Trajectory-level analysis:")
print(f"    Total trajectories:        {total_trajs}")
print(f"    Baseline TSR:              {baseline_tsr:.4f} ({baseline_traj_correct}/{total_trajs})")
print(f"    Oracle TSR (fix HA coord): {oracle_tsr:.4f} ({oracle_traj_correct}/{total_trajs})")
print(f"    TSR improvement:           +{oracle_tsr - baseline_tsr:.4f} "
      f"(+{100*(oracle_tsr - baseline_tsr)/max(baseline_tsr, 1e-9):.2f}% relative)")

# ===================================================================
# 10. Multiple agreement thresholds
# ===================================================================
print("\n" + "=" * 72)
print("SENSITIVITY: AUROC at Different Agreement Thresholds")
print("=" * 72)

threshold_aurocs = []
for agree_thr in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    subset = [r for r in records
              if r["agreement_rate"] >= agree_thr
              and r["is_coord_action_gt"]
              and r["label"] in ("pass_through", "grounding")]
    scores = [r["coord_spread_mean_std"] for r in subset]
    binary = [1 if r["label"] == "grounding" else 0 for r in subset]
    vmask = [not math.isnan(s) for s in scores]
    sv = [s for s, v in zip(scores, vmask) if v]
    bv = [b for b, v in zip(binary, vmask) if v]
    n_g = sum(bv)
    n_p = len(bv) - n_g
    auc = auroc_manual(sv, bv)
    threshold_aurocs.append({
        "agreement_threshold": agree_thr,
        "n_total": len(bv),
        "n_grounding": n_g,
        "n_passthrough": n_p,
        "auroc": round(auc, 4) if not math.isnan(auc) else None,
    })
    print(f"  agree >= {agree_thr:.1f}: n={len(bv):5d} (ground={n_g:4d}, pass={n_p:4d})  AUROC={auc:.4f}")

# ===================================================================
# 11. Summary verdict
# ===================================================================
print("\n" + "=" * 72)
print("SUMMARY VERDICT")
print("=" * 72)

validated = auroc_mean_std > 0.6 if not math.isnan(auroc_mean_std) else False
print(f"\n  Primary AUROC (mean_std, HA + coord):  {auroc_mean_std:.4f}")
print(f"  Click-specific AUROC:                  {auroc_click:.4f}")
print(f"  Full-dataset AUROC:                    {auroc_full_mean:.4f}")
print(f"  Correlation (HA + coord):              {corr:.4f}")
print(f"  Oracle TSR improvement:                +{oracle_tsr - baseline_tsr:.4f}")
print(f"\n  Two-stage router VALIDATED: {'YES' if validated else 'NO'} (threshold: AUROC > 0.6)")

if validated:
    print(f"\n  Interpretation: coord_spread can distinguish grounding errors from")
    print(f"  pass-through steps within the high-agreement subset. The two-stage")
    print(f"  router design (Stage 1: agreement -> Stage 2: coord_spread) is viable.")
else:
    print(f"\n  Interpretation: coord_spread does NOT reliably distinguish grounding")
    print(f"  errors from pass-through steps in the high-agreement subset.")
    print(f"  Alternative Stage 2 signals may be needed (e.g., verifier, critic).")

# ===================================================================
# 12. Save results
# ===================================================================
output = {
    "metadata": {
        "data_path": DATA_PATH,
        "n_total_steps": N,
        "n_episodes": len(episode_steps),
        "K_samples": 10,
        "agreement_threshold": AGREEMENT_THRESHOLD,
        "coord_action_types": sorted(COORD_ACTION_TYPES),
    },
    "label_distribution": {
        "total": {k: v for k, v in label_counter.items()},
        "high_agreement": {k: v for k, v in ha_label_counter.items()},
    },
    "stage2_high_agreement_coord": {
        "n_steps": N_hac,
        "n_valid": n_valid,
        "n_grounding": len(grounding_spreads),
        "n_passthrough": len(passthrough_spreads),
        "mean_spread_grounding": round(float(np.mean(grounding_spreads)), 2) if grounding_spreads else None,
        "mean_spread_passthrough": round(float(np.mean(passthrough_spreads)), 2) if passthrough_spreads else None,
        "median_spread_grounding": round(float(np.median(grounding_spreads)), 2) if grounding_spreads else None,
        "median_spread_passthrough": round(float(np.median(passthrough_spreads)), 2) if passthrough_spreads else None,
        "auroc_mean_std": round(auroc_mean_std, 4) if not math.isnan(auroc_mean_std) else None,
        "auroc_euclidean": round(auroc_euclidean, 4) if not math.isnan(auroc_euclidean) else None,
        "auroc_x_std": round(auroc_x, 4) if not math.isnan(auroc_x) else None,
        "auroc_y_std": round(auroc_y, 4) if not math.isnan(auroc_y) else None,
        "correlation": round(corr, 4) if not math.isnan(corr) else None,
        "precision_recall_at_thresholds": pr_results,
    },
    "full_dataset_coord": {
        "n_steps": len(ac_scores_valid),
        "n_grounding": len(grounding_all),
        "n_passthrough": len(passthrough_all),
        "mean_spread_grounding": round(float(np.mean(grounding_all)), 2) if grounding_all else None,
        "mean_spread_passthrough": round(float(np.mean(passthrough_all)), 2) if passthrough_all else None,
        "auroc_mean_std": round(auroc_full_mean, 4) if not math.isnan(auroc_full_mean) else None,
        "auroc_euclidean": round(auroc_full_euc, 4) if not math.isnan(auroc_full_euc) else None,
    },
    "click_specific_high_agreement": {
        "n_steps": len(click_scores_valid),
        "n_grounding": len(click_grounding),
        "n_passthrough": len(click_passthrough),
        "mean_spread_grounding": round(float(np.mean(click_grounding)), 2) if click_grounding else None,
        "mean_spread_passthrough": round(float(np.mean(click_passthrough)), 2) if click_passthrough else None,
        "auroc_mean_std": round(auroc_click, 4) if not math.isnan(auroc_click) else None,
        "auroc_euclidean": round(auroc_click_euc, 4) if not math.isnan(auroc_click_euc) else None,
        "correlation": round(corr_click, 4) if not math.isnan(corr_click) else None,
        "precision_recall_at_thresholds": pr_click,
    },
    "oracle_tsr_analysis": {
        "baseline_step_accuracy": round(baseline_step_accuracy, 4),
        "oracle_step_accuracy": round(oracle_step_accuracy, 4),
        "step_accuracy_improvement": round(oracle_step_accuracy - baseline_step_accuracy, 4),
        "baseline_tsr": round(baseline_tsr, 4),
        "oracle_tsr": round(oracle_tsr, 4),
        "tsr_improvement": round(oracle_tsr - baseline_tsr, 4),
        "total_grounding_errors": total_grounding_errors,
        "ha_grounding_errors": ha_grounding_errors,
        "ha_coord_grounding_errors": ha_coord_grounding_errors,
        "grounding_errors_captured_pct": round(100 * ha_coord_grounding_errors / max(total_grounding_errors, 1), 1),
    },
    "sensitivity_agreement_threshold": threshold_aurocs,
    "verdict": {
        "two_stage_validated": validated,
        "primary_auroc": round(auroc_mean_std, 4) if not math.isnan(auroc_mean_std) else None,
        "click_auroc": round(auroc_click, 4) if not math.isnan(auroc_click) else None,
        "threshold": 0.6,
    },
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to:\n  {OUTPUT_PATH}")
print("\nDone.")
