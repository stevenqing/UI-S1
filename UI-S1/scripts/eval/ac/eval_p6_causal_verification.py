#!/usr/bin/env python3
"""
Experiment P6: Causal Verification of Prompt-Induced Improvements

Uses existing P2+P3 data (4 conditions per step x 1500 steps) to verify that
prompt improvements come from the CORRECT reasoning layer, not just surface-level
prompt differences.

Conditions:
  - baseline:          no additional reasoning prompt
  - prompt_A_action:   action-type-focused reasoning
  - prompt_B_grounding: grounding-focused reasoning
  - prompt_C_combined: combined reasoning prompt

Error types (each N=500):
  - correct:          baseline was already correct
  - action_error:     baseline predicted wrong action type
  - grounding_error:  baseline predicted right type but wrong target

Analyses:
  A. Action error steps: type_correction_rate per prompt, confusion matrices
  B. Grounding error steps: coordinate distance improvements, type-vs-coord fixes
  C. Cross-layer leakage: layer purity of each prompt's improvements
  D. Response analysis: think tag presence, response length, thinking-accuracy correlation

Data sources:
  1. P2+P3 results:  outputs/eval_p2p3/p2p3_results.jsonl  (1500 lines)
  2. C4+C7 multisample: outputs/eval_c4c7_ac/Qwen2.5-VL-7B/multisample_results.jsonl

Output: outputs/eval_p6/p6_causal_verification.json
"""

import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path("/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1")
P2P3_PATH = BASE_DIR / "outputs/eval_p2p3/p2p3_results.jsonl"
C4C7_PATH = BASE_DIR / "outputs/eval_c4c7_ac/Qwen2.5-VL-7B/multisample_results.jsonl"
OUT_DIR = BASE_DIR / "outputs/eval_p6"
OUT_PATH = OUT_DIR / "p6_causal_verification.json"

os.makedirs(OUT_DIR, exist_ok=True)

CONDITIONS = ["baseline", "prompt_A_action", "prompt_B_grounding", "prompt_C_combined"]
COND_SHORT = {
    "baseline": "base",
    "prompt_A_action": "A",
    "prompt_B_grounding": "B",
    "prompt_C_combined": "C",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path):
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def parse_coordinates_from_response(response_text):
    """Parse coordinate values from a model response string.

    Handles formats like:
        "coordinate": [720, 2254]
        "coordinate": [720, 2254]}
        "coordinate2": [532, 0]

    Returns the FIRST coordinate found (the primary action target),
    or None if no coordinate found.
    """
    if not response_text:
        return None
    # Match "coordinate": [x, y] - first occurrence only (primary target)
    m = re.search(r'"coordinate"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]', response_text)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return None


def parse_think_content(response_text):
    """Extract content within <think>...</think> tags.

    Returns (has_think: bool, think_text: str, think_len: int).
    """
    if not response_text:
        return False, "", 0
    m = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
    if m:
        think_text = m.group(1).strip()
        return True, think_text, len(think_text)
    return False, "", 0


def parse_action_type_from_response(response_text):
    """Extract the action type from a response, as a backup to pred_action_type."""
    if not response_text:
        return None
    m = re.search(r'"action"\s*:\s*"(\w+)"', response_text)
    if m:
        return m.group(1)
    return None


def coord_distance(c1, c2):
    """Euclidean distance between two (x,y) coordinate tuples."""
    if c1 is None or c2 is None:
        return None
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def safe_mean(values):
    """Mean of a list, returning None for empty list."""
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return None
    return float(np.mean(values))


def safe_median(values):
    """Median of a list, returning None for empty list."""
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return None
    return float(np.median(values))


def safe_std(values):
    """Std of a list, returning None for empty/singleton."""
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if len(values) < 2:
        return None
    return float(np.std(values, ddof=1))


def cohens_d(a, b):
    """Cohen's d effect size between two groups."""
    a = [x for x in a if x is not None and not (isinstance(x, float) and math.isnan(x))]
    b = [x for x in b if x is not None and not (isinstance(x, float) and math.isnan(x))]
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return None
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("=" * 70)
print("P6: Causal Verification of Prompt-Induced Improvements")
print("=" * 70)

print("\nLoading P2+P3 data...")
p2p3_data = load_jsonl(str(P2P3_PATH))
print(f"  Loaded {len(p2p3_data)} steps")

# Split by error type
by_error = defaultdict(list)
for row in p2p3_data:
    by_error[row["error_type"]].append(row)

for et, rows in sorted(by_error.items()):
    print(f"  {et}: {len(rows)} steps")

# Load C4+C7 for GT coordinates lookup
print("\nLoading C4+C7 multisample data for GT coordinate lookup...")
c4c7_data = load_jsonl(str(C4C7_PATH))
print(f"  Loaded {len(c4c7_data)} episodes")

# Build (episode_id, step_num) -> gt_action mapping
gt_action_map = {}
for ep in c4c7_data:
    eid = ep["episode_id"]
    for step_sample in ep["step_samples"]:
        step_num = step_sample["step_num"]
        gt_action_map[(eid, step_num)] = step_sample["gt_action"]

print(f"  GT action map: {len(gt_action_map)} entries")

# Test lookup coverage
found, missed = 0, 0
for row in p2p3_data:
    key = (row["episode_id"], row["step_id"])
    if key in gt_action_map:
        found += 1
    else:
        missed += 1
print(f"  Lookup coverage: {found} found, {missed} missed")

results = {}

# =========================================================================
# A. ACTION ERROR ANALYSIS (N=500)
# =========================================================================
print("\n" + "=" * 70)
print("A. Action Error Steps Analysis")
print("=" * 70)

action_errors = by_error["action_error"]
print(f"  N = {len(action_errors)}")

# A1: Type correction rate per condition
type_corrections = {cond: 0 for cond in CONDITIONS}
type_correction_details = {cond: [] for cond in CONDITIONS}

for row in action_errors:
    gt_type = row["gt_action_type"]
    for cond in CONDITIONS:
        cdata = row["conditions"].get(cond, {})
        pred_type = cdata.get("pred_action_type") or "None"
        corrected = (pred_type == gt_type)
        if corrected:
            type_corrections[cond] += 1
        type_correction_details[cond].append({
            "episode_id": row["episode_id"],
            "step_id": row["step_id"],
            "gt_type": gt_type,
            "pred_type": pred_type,
            "corrected": corrected,
        })

n_action = len(action_errors)
type_correction_rates = {cond: type_corrections[cond] / n_action for cond in CONDITIONS}

print("\n  Type correction rate (fraction of action_error steps where pred == GT type):")
for cond in CONDITIONS:
    print(f"    {COND_SHORT[cond]:>5}: {type_corrections[cond]:3d}/{n_action} = {type_correction_rates[cond]:.3f}")

# A2: Selectivity comparison - prompt_A should correct more action errors than prompt_B
sel_A_vs_B = type_correction_rates["prompt_A_action"] - type_correction_rates["prompt_B_grounding"]
print(f"\n  Selectivity (A - B): {sel_A_vs_B:+.3f}")
print(f"    If positive: action prompt corrects action types more (expected)")

# A3: Confusion analysis for baseline wrong types
confusion_baseline = Counter()  # (gt_type, pred_type) counts for wrong baseline
confusion_A_fixed = Counter()   # (gt_type, baseline_wrong_type) for steps A fixed
confusion_A_still_wrong = Counter()  # (gt_type, A_pred_type) for steps A didn't fix

for row in action_errors:
    gt_type = row["gt_action_type"]
    base_pred = row["conditions"]["baseline"].get("pred_action_type") or "None"
    a_pred = row["conditions"]["prompt_A_action"].get("pred_action_type") or "None"

    if base_pred != gt_type:
        confusion_baseline[(gt_type, base_pred)] += 1

    if base_pred != gt_type and a_pred == gt_type:
        confusion_A_fixed[(gt_type, base_pred)] += 1
    elif base_pred != gt_type and a_pred != gt_type:
        confusion_A_still_wrong[(gt_type, a_pred)] += 1

print(f"\n  Top baseline confusions (GT -> wrong pred):")
for (gt, pred), count in confusion_baseline.most_common(10):
    gt_s = str(gt) if gt is not None else "None"
    pred_s = str(pred) if pred is not None else "None"
    print(f"    {gt_s:>15} -> {pred_s:<15}: {count}")

print(f"\n  Top confusions fixed by prompt_A (GT -> baseline_wrong_pred):")
for (gt, pred), count in confusion_A_fixed.most_common(10):
    gt_s = str(gt) if gt is not None else "None"
    pred_s = str(pred) if pred is not None else "None"
    print(f"    {gt_s:>15} -> {pred_s:<15}: {count}")

# A4: Steps where A helped but B didn't, and vice versa
a_only_fix = []  # A fixed, B didn't
b_only_fix = []  # B fixed, A didn't
both_fix = []    # Both fixed
neither_fix = [] # Neither fixed

for row in action_errors:
    gt_type = row["gt_action_type"]
    base_pred = row["conditions"]["baseline"].get("pred_action_type") or "None"
    a_pred = row["conditions"]["prompt_A_action"].get("pred_action_type") or "None"
    b_pred = row["conditions"]["prompt_B_grounding"].get("pred_action_type") or "None"

    # Only consider steps where baseline was wrong
    if base_pred == gt_type:
        continue

    a_fixed = (a_pred == gt_type)
    b_fixed = (b_pred == gt_type)

    step_info = {
        "episode_id": row["episode_id"],
        "step_id": row["step_id"],
        "gt_type": gt_type,
        "base_pred": base_pred,
        "a_pred": a_pred,
        "b_pred": b_pred,
    }

    if a_fixed and not b_fixed:
        a_only_fix.append(step_info)
    elif b_fixed and not a_fixed:
        b_only_fix.append(step_info)
    elif a_fixed and b_fixed:
        both_fix.append(step_info)
    else:
        neither_fix.append(step_info)

n_base_wrong = len(a_only_fix) + len(b_only_fix) + len(both_fix) + len(neither_fix)
print(f"\n  Among steps where baseline was wrong (N={n_base_wrong}):")
print(f"    A only fixed:    {len(a_only_fix):3d} ({100*len(a_only_fix)/max(1,n_base_wrong):.1f}%)")
print(f"    B only fixed:    {len(b_only_fix):3d} ({100*len(b_only_fix)/max(1,n_base_wrong):.1f}%)")
print(f"    Both fixed:      {len(both_fix):3d} ({100*len(both_fix)/max(1,n_base_wrong):.1f}%)")
print(f"    Neither fixed:   {len(neither_fix):3d} ({100*len(neither_fix)/max(1,n_base_wrong):.1f}%)")

# Characterize A-only vs B-only fixes
a_only_gt_types = Counter(s["gt_type"] for s in a_only_fix)
b_only_gt_types = Counter(s["gt_type"] for s in b_only_fix)
a_only_step_pos = [s["step_id"] for s in a_only_fix]
b_only_step_pos = [s["step_id"] for s in b_only_fix]

print(f"\n  A-only fixes by GT type: {dict(a_only_gt_types.most_common(8))}")
print(f"  B-only fixes by GT type: {dict(b_only_gt_types.most_common(8))}")
print(f"  A-only mean step position: {safe_mean(a_only_step_pos)}")
print(f"  B-only mean step position: {safe_mean(b_only_step_pos)}")

# Store section A results
results["A_action_error_analysis"] = {
    "n_action_errors": n_action,
    "type_correction_rates": type_correction_rates,
    "selectivity_A_minus_B": sel_A_vs_B,
    "baseline_confusions_top10": [
        {"gt": gt, "pred": pred, "count": c}
        for (gt, pred), c in confusion_baseline.most_common(10)
    ],
    "A_fixed_confusions_top10": [
        {"gt": gt, "pred": pred, "count": c}
        for (gt, pred), c in confusion_A_fixed.most_common(10)
    ],
    "differential_fix_counts": {
        "n_baseline_wrong": n_base_wrong,
        "A_only_fixed": len(a_only_fix),
        "B_only_fixed": len(b_only_fix),
        "both_fixed": len(both_fix),
        "neither_fixed": len(neither_fix),
    },
    "A_only_gt_type_distribution": dict(a_only_gt_types),
    "B_only_gt_type_distribution": dict(b_only_gt_types),
    "A_only_mean_step_position": safe_mean(a_only_step_pos),
    "B_only_mean_step_position": safe_mean(b_only_step_pos),
}


# =========================================================================
# B. GROUNDING ERROR ANALYSIS (N=500)
# =========================================================================
print("\n" + "=" * 70)
print("B. Grounding Error Steps Analysis")
print("=" * 70)

grounding_errors = by_error["grounding_error"]
print(f"  N = {len(grounding_errors)}")

# B1-B2: Parse coordinates and compute distances to GT
coord_distances = {cond: [] for cond in CONDITIONS}
coord_distance_records = []  # per-step records with all distances

n_has_gt_coord = 0
n_missing_gt_coord = 0
n_missing_pred_coord = {cond: 0 for cond in CONDITIONS}

for row in grounding_errors:
    eid = row["episode_id"]
    sid = row["step_id"]
    gt_type = row["gt_action_type"]

    # Look up GT coordinates
    gt_action = gt_action_map.get((eid, sid))
    gt_coord = None
    if gt_action and "coordinate" in gt_action:
        gt_coord = (gt_action["coordinate"][0], gt_action["coordinate"][1])

    if gt_coord is None:
        n_missing_gt_coord += 1
        continue

    n_has_gt_coord += 1
    record = {
        "episode_id": eid,
        "step_id": sid,
        "gt_type": gt_type,
        "gt_coord": gt_coord,
    }

    for cond in CONDITIONS:
        cdata = row["conditions"].get(cond, {})
        resp = cdata.get("response", "")
        pred_coord = parse_coordinates_from_response(resp)

        if pred_coord is None:
            n_missing_pred_coord[cond] += 1
            dist = None
        else:
            dist = coord_distance(gt_coord, pred_coord)

        coord_distances[cond].append(dist)
        record[f"coord_{COND_SHORT[cond]}"] = pred_coord
        record[f"dist_{COND_SHORT[cond]}"] = dist

    coord_distance_records.append(record)

print(f"\n  GT coordinates found: {n_has_gt_coord}")
print(f"  GT coordinates missing: {n_missing_gt_coord}")
for cond in CONDITIONS:
    print(f"  Pred coordinates missing ({COND_SHORT[cond]}): {n_missing_pred_coord[cond]}")

# Compute mean/median coordinate errors per condition
print(f"\n  Coordinate distance to GT (mean / median / std):")
coord_stats = {}
for cond in CONDITIONS:
    valid = [d for d in coord_distances[cond] if d is not None]
    m = safe_mean(valid)
    md = safe_median(valid)
    s = safe_std(valid)
    n_v = len(valid)
    coord_stats[cond] = {"mean": m, "median": md, "std": s, "n_valid": n_v}
    print(f"    {COND_SHORT[cond]:>5}: mean={m:.1f}, median={md:.1f}, std={s:.1f} (n={n_v})")

# B3: Does prompt_B reduce coord error MORE than prompt_A?
# Compute paired differences: baseline_dist - condition_dist (positive = improvement)
paired_improvements = {cond: [] for cond in CONDITIONS if cond != "baseline"}
for record in coord_distance_records:
    base_dist = record.get("dist_base")
    if base_dist is None:
        continue
    for cond in CONDITIONS:
        if cond == "baseline":
            continue
        cond_dist = record.get(f"dist_{COND_SHORT[cond]}")
        if cond_dist is None:
            continue
        improvement = base_dist - cond_dist  # positive = this condition is closer to GT
        paired_improvements[cond].append(improvement)

print(f"\n  Paired coordinate improvement (baseline_dist - condition_dist, positive=better):")
improvement_stats = {}
for cond in ["prompt_A_action", "prompt_B_grounding", "prompt_C_combined"]:
    vals = paired_improvements[cond]
    m = safe_mean(vals)
    md = safe_median(vals)
    n_improved = sum(1 for v in vals if v > 0)
    n_worsened = sum(1 for v in vals if v < 0)
    n_same = sum(1 for v in vals if v == 0)
    improvement_stats[cond] = {
        "mean_improvement": m,
        "median_improvement": md,
        "n_improved": n_improved,
        "n_worsened": n_worsened,
        "n_same": n_same,
        "n_total": len(vals),
        "frac_improved": n_improved / max(1, len(vals)),
    }
    print(f"    {COND_SHORT[cond]:>5}: mean={m:+.1f}, improved={n_improved}, worsened={n_worsened}, same={n_same}")

# Selectivity: does B improve coords more than A?
sel_B_vs_A_coord = (improvement_stats["prompt_B_grounding"]["mean_improvement"] or 0) - \
                   (improvement_stats["prompt_A_action"]["mean_improvement"] or 0)
print(f"\n  Selectivity (B_improvement - A_improvement): {sel_B_vs_A_coord:+.1f}")
print(f"    If positive: grounding prompt improves coordinates more (expected)")

# B4: How many grounding errors are "fixed" by changing type vs better coordinates?
grounding_fix_type = {cond: 0 for cond in CONDITIONS if cond != "baseline"}
grounding_fix_coord = {cond: 0 for cond in CONDITIONS if cond != "baseline"}
grounding_fix_neither = {cond: 0 for cond in CONDITIONS if cond != "baseline"}
grounding_fix_both = {cond: 0 for cond in CONDITIONS if cond != "baseline"}

for row in grounding_errors:
    base_cond = row["conditions"]["baseline"]
    base_match = base_cond["extract_match"]

    for cond in CONDITIONS:
        if cond == "baseline":
            continue
        cond_data = row["conditions"].get(cond, {})
        cond_match = cond_data.get("extract_match", False)

        if cond_match and not base_match:
            # This condition improved from wrong to right
            base_type = base_cond.get("pred_action_type") or "None"
            cond_type = cond_data.get("pred_action_type") or "None"
            type_changed = (base_type != cond_type)
            # Check if coord changed (type stayed same, coord must have improved)
            base_resp = base_cond.get("response", "")
            cond_resp = cond_data.get("response", "")
            base_coord = parse_coordinates_from_response(base_resp)
            cond_coord = parse_coordinates_from_response(cond_resp)
            coord_changed = (base_coord != cond_coord)

            if type_changed and coord_changed:
                grounding_fix_both[cond] += 1
            elif type_changed:
                grounding_fix_type[cond] += 1
            elif coord_changed:
                grounding_fix_coord[cond] += 1
            else:
                grounding_fix_neither[cond] += 1

print(f"\n  Among grounding errors fixed by each condition (baseline wrong -> condition right):")
print(f"  {'Condition':>10} | {'Type change':>12} | {'Coord change':>12} | {'Both':>6} | {'Neither':>8}")
for cond in ["prompt_A_action", "prompt_B_grounding", "prompt_C_combined"]:
    total_fixed = grounding_fix_type[cond] + grounding_fix_coord[cond] + grounding_fix_both[cond] + grounding_fix_neither[cond]
    print(f"  {COND_SHORT[cond]:>10} | {grounding_fix_type[cond]:>12} | {grounding_fix_coord[cond]:>12} | "
          f"{grounding_fix_both[cond]:>6} | {grounding_fix_neither[cond]:>8} (total={total_fixed})")

# Store section B results
results["B_grounding_error_analysis"] = {
    "n_grounding_errors": len(grounding_errors),
    "n_has_gt_coord": n_has_gt_coord,
    "n_missing_gt_coord": n_missing_gt_coord,
    "n_missing_pred_coord": n_missing_pred_coord,
    "coord_distance_stats": coord_stats,
    "paired_improvement_stats": improvement_stats,
    "selectivity_B_minus_A_coord_improvement": sel_B_vs_A_coord,
    "fix_mechanism_breakdown": {
        cond: {
            "fixed_by_type_change": grounding_fix_type[cond],
            "fixed_by_coord_change": grounding_fix_coord[cond],
            "fixed_by_both": grounding_fix_both[cond],
            "fixed_by_neither": grounding_fix_neither[cond],
        }
        for cond in ["prompt_A_action", "prompt_B_grounding", "prompt_C_combined"]
    },
}


# =========================================================================
# C. CROSS-LAYER LEAKAGE ANALYSIS
# =========================================================================
print("\n" + "=" * 70)
print("C. Cross-Layer Leakage Analysis")
print("=" * 70)

# For each condition, among steps where extract_match improved (False -> True):
#   1. Was improvement from type_match changing? (cross-layer leakage)
#   2. Was improvement from better coords while type stayed same? (correct layer)
#   3. Compute "layer purity" = correct_layer_fixes / total_fixes

layer_analysis = {}

for cond in ["prompt_A_action", "prompt_B_grounding", "prompt_C_combined"]:
    correct_layer_fixes = 0
    cross_layer_fixes = 0
    ambiguous_fixes = 0
    total_fixes = 0
    total_regressions = 0  # extract_match went from True to False

    # Track per error type
    fixes_by_error_type = Counter()
    correct_layer_by_error_type = Counter()
    cross_layer_by_error_type = Counter()

    for row in p2p3_data:
        error_type = row["error_type"]
        base_cond = row["conditions"]["baseline"]
        cond_data = row["conditions"].get(cond, {})

        base_em = base_cond.get("extract_match", False)
        cond_em = cond_data.get("extract_match", False)
        base_tm = base_cond.get("type_match", False)
        cond_tm = cond_data.get("type_match", False)

        if cond_em and not base_em:
            # extract_match improved
            total_fixes += 1
            fixes_by_error_type[error_type] += 1

            type_changed = (base_tm != cond_tm)

            if error_type == "action_error":
                # For action error steps: fix should come from type change (correct layer for A)
                if cond == "prompt_A_action":
                    if type_changed:
                        correct_layer_fixes += 1
                        correct_layer_by_error_type[error_type] += 1
                    else:
                        cross_layer_fixes += 1
                        cross_layer_by_error_type[error_type] += 1
                elif cond == "prompt_B_grounding":
                    if type_changed:
                        cross_layer_fixes += 1  # grounding prompt shouldn't fix type
                        cross_layer_by_error_type[error_type] += 1
                    else:
                        correct_layer_fixes += 1
                        correct_layer_by_error_type[error_type] += 1
                else:
                    ambiguous_fixes += 1  # combined prompt - can't assign layer

            elif error_type == "grounding_error":
                # For grounding error steps: fix should come from better coords, not type change
                if cond == "prompt_B_grounding":
                    if not type_changed:
                        correct_layer_fixes += 1
                        correct_layer_by_error_type[error_type] += 1
                    else:
                        cross_layer_fixes += 1
                        cross_layer_by_error_type[error_type] += 1
                elif cond == "prompt_A_action":
                    if not type_changed:
                        cross_layer_fixes += 1  # action prompt shouldn't fix coords
                        cross_layer_by_error_type[error_type] += 1
                    else:
                        correct_layer_fixes += 1
                        correct_layer_by_error_type[error_type] += 1
                else:
                    ambiguous_fixes += 1

            else:
                # "correct" error type - baseline was already correct, fix means
                # condition also got it right (from a different path)
                ambiguous_fixes += 1

        elif base_em and not cond_em:
            total_regressions += 1

    assignable = correct_layer_fixes + cross_layer_fixes
    purity = correct_layer_fixes / max(1, assignable)

    layer_analysis[cond] = {
        "total_fixes": total_fixes,
        "correct_layer_fixes": correct_layer_fixes,
        "cross_layer_fixes": cross_layer_fixes,
        "ambiguous_fixes": ambiguous_fixes,
        "layer_purity": purity,
        "total_regressions": total_regressions,
        "fixes_by_error_type": dict(fixes_by_error_type),
        "correct_layer_by_error_type": dict(correct_layer_by_error_type),
        "cross_layer_by_error_type": dict(cross_layer_by_error_type),
    }

    print(f"\n  {COND_SHORT[cond]:>5} ({cond}):")
    print(f"    Total extract_match improvements: {total_fixes}")
    print(f"    Correct layer fixes: {correct_layer_fixes}")
    print(f"    Cross-layer leakage fixes: {cross_layer_fixes}")
    print(f"    Ambiguous (combined/correct): {ambiguous_fixes}")
    print(f"    Layer purity: {purity:.3f}")
    print(f"    Total regressions (True->False): {total_regressions}")
    print(f"    Fixes by error type: {dict(fixes_by_error_type)}")

# Overall layer purity comparison
print(f"\n  Summary of layer purity:")
for cond in ["prompt_A_action", "prompt_B_grounding", "prompt_C_combined"]:
    la = layer_analysis[cond]
    print(f"    {COND_SHORT[cond]:>5}: purity={la['layer_purity']:.3f} "
          f"(correct={la['correct_layer_fixes']}, cross={la['cross_layer_fixes']})")

results["C_cross_layer_leakage"] = layer_analysis


# =========================================================================
# D. RESPONSE ANALYSIS
# =========================================================================
print("\n" + "=" * 70)
print("D. Response Analysis")
print("=" * 70)

# D1-D2: Think tag presence and response length per condition
think_presence = {cond: 0 for cond in CONDITIONS}
response_lengths = {cond: [] for cond in CONDITIONS}
think_lengths = {cond: [] for cond in CONDITIONS}

# D3: For responses with think, does longer thinking correlate with accuracy?
think_accuracy_data = {cond: {"think_len": [], "extract_match": []} for cond in CONDITIONS}

for row in p2p3_data:
    for cond in CONDITIONS:
        cdata = row["conditions"].get(cond, {})
        resp = cdata.get("response", "")
        if not resp:
            continue

        # Response length
        response_lengths[cond].append(len(resp))

        # Think tag analysis
        has_think, think_text, think_len = parse_think_content(resp)
        if has_think:
            think_presence[cond] += 1
            think_lengths[cond].append(think_len)

            # Correlation between think length and accuracy
            em = 1.0 if cdata.get("extract_match", False) else 0.0
            think_accuracy_data[cond]["think_len"].append(think_len)
            think_accuracy_data[cond]["extract_match"].append(em)

n_total = len(p2p3_data)
print(f"\n  Think tag presence (out of {n_total} steps):")
for cond in CONDITIONS:
    frac = think_presence[cond] / n_total
    print(f"    {COND_SHORT[cond]:>5}: {think_presence[cond]:4d} ({100*frac:.1f}%)")

print(f"\n  Response length (characters, mean / median):")
resp_length_stats = {}
for cond in CONDITIONS:
    m = safe_mean(response_lengths[cond])
    md = safe_median(response_lengths[cond])
    resp_length_stats[cond] = {"mean": m, "median": md}
    print(f"    {COND_SHORT[cond]:>5}: mean={m:.0f}, median={md:.0f}")

print(f"\n  Think content length (characters, mean / median):")
think_length_stats = {}
for cond in CONDITIONS:
    m = safe_mean(think_lengths[cond])
    md = safe_median(think_lengths[cond])
    think_length_stats[cond] = {"mean": m, "median": md, "n": len(think_lengths[cond])}
    if m is not None:
        print(f"    {COND_SHORT[cond]:>5}: mean={m:.0f}, median={md:.0f} (n={len(think_lengths[cond])})")
    else:
        print(f"    {COND_SHORT[cond]:>5}: no think tags")

# D3: Think length vs accuracy correlation
print(f"\n  Think length vs extract_match correlation (point-biserial r):")
think_accuracy_results = {}
for cond in CONDITIONS:
    tl = think_accuracy_data[cond]["think_len"]
    em = think_accuracy_data[cond]["extract_match"]
    if len(tl) < 10:
        print(f"    {COND_SHORT[cond]:>5}: too few samples with think tags ({len(tl)})")
        think_accuracy_results[cond] = {"r": None, "p": None, "n": len(tl)}
        continue

    tl_arr = np.array(tl, dtype=float)
    em_arr = np.array(em, dtype=float)

    if np.std(tl_arr) == 0 or len(np.unique(em_arr)) < 2:
        print(f"    {COND_SHORT[cond]:>5}: degenerate data (n={len(tl)})")
        think_accuracy_results[cond] = {"r": None, "p": None, "n": len(tl)}
        continue

    from scipy import stats
    r, p = stats.pointbiserialr(em_arr, tl_arr)
    think_accuracy_results[cond] = {"r": float(r), "p": float(p), "n": len(tl)}
    print(f"    {COND_SHORT[cond]:>5}: r={r:.3f}, p={p:.4f} (n={len(tl)})")

    # Also report: mean think length for correct vs incorrect
    correct_tl = [t for t, e in zip(tl, em) if e > 0.5]
    incorrect_tl = [t for t, e in zip(tl, em) if e <= 0.5]
    think_accuracy_results[cond]["mean_think_len_correct"] = safe_mean(correct_tl)
    think_accuracy_results[cond]["mean_think_len_incorrect"] = safe_mean(incorrect_tl)
    if correct_tl and incorrect_tl:
        print(f"          correct mean_think_len={safe_mean(correct_tl):.0f}, "
              f"incorrect mean_think_len={safe_mean(incorrect_tl):.0f}")

# Think presence by error type
print(f"\n  Think tag presence by error type:")
think_by_error = {et: {cond: 0 for cond in CONDITIONS} for et in ["correct", "action_error", "grounding_error"]}
count_by_error = {et: 0 for et in ["correct", "action_error", "grounding_error"]}
for row in p2p3_data:
    et = row["error_type"]
    count_by_error[et] += 1
    for cond in CONDITIONS:
        resp = row["conditions"].get(cond, {}).get("response", "")
        if resp and "<think>" in resp:
            think_by_error[et][cond] += 1

for et in ["correct", "action_error", "grounding_error"]:
    print(f"    {et} (N={count_by_error[et]}):")
    for cond in CONDITIONS:
        frac = think_by_error[et][cond] / max(1, count_by_error[et])
        print(f"      {COND_SHORT[cond]:>5}: {think_by_error[et][cond]:3d} ({100*frac:.1f}%)")

# Store section D results
results["D_response_analysis"] = {
    "think_presence": {cond: {"count": think_presence[cond], "fraction": think_presence[cond] / n_total}
                       for cond in CONDITIONS},
    "response_length_stats": resp_length_stats,
    "think_length_stats": think_length_stats,
    "think_accuracy_correlation": think_accuracy_results,
    "think_by_error_type": {
        et: {cond: {"count": think_by_error[et][cond],
                     "fraction": think_by_error[et][cond] / max(1, count_by_error[et])}
             for cond in CONDITIONS}
        for et in ["correct", "action_error", "grounding_error"]
    },
}


# =========================================================================
# OVERALL SUMMARY
# =========================================================================
print("\n" + "=" * 70)
print("OVERALL CAUSAL VERIFICATION SUMMARY")
print("=" * 70)

# Compute overall extract_match rates per condition and error type
overall_em = {cond: {"correct": 0, "action_error": 0, "grounding_error": 0, "total": 0}
              for cond in CONDITIONS}
for row in p2p3_data:
    et = row["error_type"]
    for cond in CONDITIONS:
        cdata = row["conditions"].get(cond, {})
        if cdata.get("extract_match", False):
            overall_em[cond][et] += 1
            overall_em[cond]["total"] += 1

print(f"\n  extract_match rates by condition and error type:")
print(f"  {'Cond':>5} | {'correct':>9} | {'action_err':>10} | {'ground_err':>10} | {'total':>8}")
for cond in CONDITIONS:
    em = overall_em[cond]
    n_c, n_a, n_g = count_by_error["correct"], count_by_error["action_error"], count_by_error["grounding_error"]
    print(f"  {COND_SHORT[cond]:>5} | {em['correct']:3d}/{n_c} ({100*em['correct']/n_c:.1f}%) | "
          f"{em['action_error']:3d}/{n_a} ({100*em['action_error']/n_a:.1f}%) | "
          f"{em['grounding_error']:3d}/{n_g} ({100*em['grounding_error']/n_g:.1f}%) | "
          f"{em['total']:4d}/{n_total} ({100*em['total']/n_total:.1f}%)")

# Type match rates
overall_tm = {cond: {"correct": 0, "action_error": 0, "grounding_error": 0, "total": 0}
              for cond in CONDITIONS}
for row in p2p3_data:
    et = row["error_type"]
    for cond in CONDITIONS:
        cdata = row["conditions"].get(cond, {})
        if cdata.get("type_match", False):
            overall_tm[cond][et] += 1
            overall_tm[cond]["total"] += 1

print(f"\n  type_match rates by condition and error type:")
print(f"  {'Cond':>5} | {'correct':>9} | {'action_err':>10} | {'ground_err':>10} | {'total':>8}")
for cond in CONDITIONS:
    tm = overall_tm[cond]
    n_c, n_a, n_g = count_by_error["correct"], count_by_error["action_error"], count_by_error["grounding_error"]
    print(f"  {COND_SHORT[cond]:>5} | {tm['correct']:3d}/{n_c} ({100*tm['correct']/n_c:.1f}%) | "
          f"{tm['action_error']:3d}/{n_a} ({100*tm['action_error']/n_a:.1f}%) | "
          f"{tm['grounding_error']:3d}/{n_g} ({100*tm['grounding_error']/n_g:.1f}%) | "
          f"{tm['total']:4d}/{n_total} ({100*tm['total']/n_total:.1f}%)")

results["overall_summary"] = {
    "extract_match_rates": {
        cond: {
            et: overall_em[cond][et] / max(1, count_by_error[et])
            for et in ["correct", "action_error", "grounding_error"]
        } | {"total": overall_em[cond]["total"] / n_total}
        for cond in CONDITIONS
    },
    "type_match_rates": {
        cond: {
            et: overall_tm[cond][et] / max(1, count_by_error[et])
            for et in ["correct", "action_error", "grounding_error"]
        } | {"total": overall_tm[cond]["total"] / n_total}
        for cond in CONDITIONS
    },
    "n_steps_per_error_type": dict(count_by_error),
}

# Causal selectivity verdict
print(f"\n  CAUSAL SELECTIVITY VERDICTS:")
print(f"    Action layer selectivity (A type_correction - B type_correction): "
      f"{sel_A_vs_B:+.3f}")
if sel_A_vs_B > 0:
    print(f"      -> prompt_A action reasoning corrects types MORE than prompt_B (EXPECTED)")
else:
    print(f"      -> prompt_B grounding reasoning corrects types MORE than prompt_A (UNEXPECTED)")

print(f"    Grounding layer selectivity (B coord_improvement - A coord_improvement): "
      f"{sel_B_vs_A_coord:+.1f}")
if sel_B_vs_A_coord > 0:
    print(f"      -> prompt_B grounding reasoning improves coords MORE than prompt_A (EXPECTED)")
else:
    print(f"      -> prompt_A action reasoning improves coords MORE than prompt_B (UNEXPECTED)")

# Overall layer purity
for cond in ["prompt_A_action", "prompt_B_grounding"]:
    la = layer_analysis[cond]
    print(f"    Layer purity ({COND_SHORT[cond]}): {la['layer_purity']:.3f}")
    if la['layer_purity'] > 0.5:
        print(f"      -> Improvements come primarily from the CORRECT layer (purity > 0.5)")
    else:
        print(f"      -> WARNING: Improvements come primarily from the WRONG layer (cross-leakage)")

results["causal_verdict"] = {
    "action_layer_selectivity_A_minus_B": sel_A_vs_B,
    "grounding_layer_selectivity_B_minus_A": sel_B_vs_A_coord,
    "prompt_A_layer_purity": layer_analysis["prompt_A_action"]["layer_purity"],
    "prompt_B_layer_purity": layer_analysis["prompt_B_grounding"]["layer_purity"],
    "action_layer_verdict": "EXPECTED" if sel_A_vs_B > 0 else "UNEXPECTED",
    "grounding_layer_verdict": "EXPECTED" if sel_B_vs_A_coord > 0 else "UNEXPECTED",
}


# =========================================================================
# Save results
# =========================================================================
print(f"\nSaving results to {OUT_PATH}")
with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2, default=_json_default)

print(f"Done. Output: {OUT_PATH}")
print(f"Output size: {os.path.getsize(OUT_PATH)} bytes")
