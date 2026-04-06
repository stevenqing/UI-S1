#!/usr/bin/env python3
"""
Experiment P4: Reasoning Quality x Step Position x TSR (SPWA Validation)

Tests whether reasoning quality at early steps is more correlated with
trajectory success than at later steps, validating the Step-Position
Weighted Advantage (SPWA) weighting scheme.

Data sources:
  - C4+C7 multi-sample data (K=10 per step)
  - Eval A trajectory data (trajectory success labels)

Output: outputs/eval_p4/p4_reasoning_spwa.json
"""

import json
import os
import sys
import math
import warnings
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
from scipy import stats

# ───────────────────────────── paths ──────────────────────────────

BASE = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
C4C7_PATH = os.path.join(BASE, "outputs/eval_c4c7_ac/Qwen2.5-VL-7B/multisample_results.jsonl")
EVAL_A_PATH = os.path.join(BASE, "outputs/eval_a_ac/Qwen2.5-VL-7B/trajectory_results.jsonl")
OUT_DIR = os.path.join(BASE, "outputs/eval_p4")
OUT_PATH = os.path.join(OUT_DIR, "p4_reasoning_spwa.json")

os.makedirs(OUT_DIR, exist_ok=True)


# ───────────────────────────── helpers ────────────────────────────

def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def shannon_entropy(probs):
    """Compute Shannon entropy from a probability distribution."""
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log2(p)
    return h


def point_biserial(binary_var, continuous_var):
    """Point-biserial correlation between a binary var and continuous var.
    Returns (r, p_value). If degenerate, returns (nan, nan)."""
    binary_var = np.asarray(binary_var, dtype=float)
    continuous_var = np.asarray(continuous_var, dtype=float)
    if len(np.unique(binary_var)) < 2 or len(binary_var) < 3:
        return float("nan"), float("nan")
    if np.std(continuous_var) == 0:
        return float("nan"), float("nan")
    r, p = stats.pointbiserialr(binary_var, continuous_var)
    return float(r), float(p)


# ───────────────────────────── load data ──────────────────────────

print("Loading data...")

c4c7_data = load_jsonl(C4C7_PATH)
eval_a_data = load_jsonl(EVAL_A_PATH)

# Build episode_id -> task_success mapping from Eval A
tsr_map = {}
for ep in eval_a_data:
    tsr_map[ep["episode_id"]] = ep["task_success"]

print(f"  C4+C7 episodes: {len(c4c7_data)}")
print(f"  Eval A episodes: {len(eval_a_data)}")
n_success = sum(1 for v in tsr_map.values() if v)
print(f"  TSR: {n_success}/{len(tsr_map)} = {n_success/len(tsr_map):.3f}")


# ──────── Step 1: Compute per-step reasoning quality proxies ──────

print("\nComputing per-step reasoning quality metrics...")

# For every (episode, step), store RQ metrics
# Also store them indexed by absolute step position and relative position
all_step_records = []  # list of dicts

for ep in c4c7_data:
    eid = ep["episode_id"]
    n_steps = ep["num_steps"]
    success = tsr_map.get(eid, False)

    for step_info in ep["step_samples"]:
        step_num = step_info["step_num"]
        samples = step_info["samples"]
        K = len(samples)

        # rq_type_correct: fraction with type_match=True
        n_type = sum(1 for s in samples if s["type_match"])
        rq_type = n_type / K

        # rq_full_correct: fraction with type_match AND extract_match
        n_full = sum(1 for s in samples if s["type_match"] and s["extract_match"])
        rq_full = n_full / K

        # rq_agreement: fraction matching majority action type
        type_counts = Counter()
        for s in samples:
            pa = s.get("pred_action")
            atype = pa.get("action", "unknown") if pa else "none"
            type_counts[atype] += 1
        majority_count = type_counts.most_common(1)[0][1]
        rq_agree = majority_count / K

        # rq_entropy: Shannon entropy of action type distribution
        type_probs = [c / K for c in type_counts.values()]
        rq_ent = shannon_entropy(type_probs)

        rec = {
            "episode_id": eid,
            "step_num": step_num,
            "num_steps": n_steps,
            "rel_pos": step_num / max(n_steps - 1, 1),  # 0..1
            "tsr": success,
            "rq_type": rq_type,
            "rq_full": rq_full,
            "rq_agree": rq_agree,
            "rq_entropy": rq_ent,
        }
        all_step_records.append(rec)

print(f"  Total step records: {len(all_step_records)}")


# ──────── Step 2: Bucket by step position ─────────────────────────

def bucket_abs(step_num):
    if step_num <= 2:
        return "early(0-2)"
    elif step_num <= 5:
        return "mid(3-5)"
    else:
        return "late(6+)"


def bucket_rel(rel_pos):
    if rel_pos <= 0.33:
        return "first_third"
    elif rel_pos <= 0.66:
        return "mid_third"
    else:
        return "last_third"


# ──────── Step 3: Correlation(RQ_k, TSR) per bucket ──────────────

print("\n=== Analysis 1: Point-Biserial Correlation (RQ vs TSR) by Step Bucket ===")

results = {}

# --- 3a: By absolute step position ---
abs_buckets = defaultdict(lambda: {"tsr": [], "rq_type": [], "rq_full": [], "rq_agree": [], "rq_entropy": []})
for rec in all_step_records:
    b = bucket_abs(rec["step_num"])
    abs_buckets[b]["tsr"].append(int(rec["tsr"]))
    abs_buckets[b]["rq_type"].append(rec["rq_type"])
    abs_buckets[b]["rq_full"].append(rec["rq_full"])
    abs_buckets[b]["rq_agree"].append(rec["rq_agree"])
    abs_buckets[b]["rq_entropy"].append(rec["rq_entropy"])

corr_by_abs_bucket = {}
for b in ["early(0-2)", "mid(3-5)", "late(6+)"]:
    d = abs_buckets[b]
    n = len(d["tsr"])
    corrs = {}
    for metric in ["rq_type", "rq_full", "rq_agree", "rq_entropy"]:
        r, p = point_biserial(d["tsr"], d[metric])
        corrs[metric] = {"r": round(r, 4), "p": round(p, 6), "n": n}
    corr_by_abs_bucket[b] = corrs
    print(f"\n  Bucket {b} (n={n}):")
    for m, v in corrs.items():
        print(f"    {m}: r={v['r']:.4f}, p={v['p']:.6f}")

results["correlation_by_abs_bucket"] = corr_by_abs_bucket

# --- 3b: By relative step position ---
rel_buckets = defaultdict(lambda: {"tsr": [], "rq_type": [], "rq_full": [], "rq_agree": [], "rq_entropy": []})
for rec in all_step_records:
    b = bucket_rel(rec["rel_pos"])
    rel_buckets[b]["tsr"].append(int(rec["tsr"]))
    rel_buckets[b]["rq_type"].append(rec["rq_type"])
    rel_buckets[b]["rq_full"].append(rec["rq_full"])
    rel_buckets[b]["rq_agree"].append(rec["rq_agree"])
    rel_buckets[b]["rq_entropy"].append(rec["rq_entropy"])

corr_by_rel_bucket = {}
for b in ["first_third", "mid_third", "last_third"]:
    d = rel_buckets[b]
    n = len(d["tsr"])
    corrs = {}
    for metric in ["rq_type", "rq_full", "rq_agree", "rq_entropy"]:
        r, p = point_biserial(d["tsr"], d[metric])
        corrs[metric] = {"r": round(r, 4), "p": round(p, 6), "n": n}
    corr_by_rel_bucket[b] = corrs
    print(f"\n  Relative bucket {b} (n={n}):")
    for m, v in corrs.items():
        print(f"    {m}: r={v['r']:.4f}, p={v['p']:.6f}")

results["correlation_by_rel_bucket"] = corr_by_rel_bucket

# --- 3c: Fine-grained by individual step position ---
print("\n=== Analysis 1b: Per-Step-Position Correlation (rq_full vs TSR) ===")
step_pos_data = defaultdict(lambda: {"tsr": [], "rq_full": []})
for rec in all_step_records:
    step_pos_data[rec["step_num"]]["tsr"].append(int(rec["tsr"]))
    step_pos_data[rec["step_num"]]["rq_full"].append(rec["rq_full"])

per_step_corr = {}
for step_num in sorted(step_pos_data.keys()):
    d = step_pos_data[step_num]
    n = len(d["tsr"])
    if n < 10:
        continue
    r, p = point_biserial(d["tsr"], d["rq_full"])
    per_step_corr[int(step_num)] = {"r": round(r, 4), "p": round(p, 6), "n": n}
    sig = "*" if p < 0.05 else ""
    print(f"  Step {step_num:2d} (n={n:4d}): r={r:+.4f}  p={p:.4f} {sig}")

results["correlation_per_step_position"] = per_step_corr


# ──────── Step 4: SPWA Weight Validation ──────────────────────────

print("\n=== Analysis 2: SPWA Weight Validation ===")

# Compute per-step-position accuracy (greedy = majority vote accuracy is not right;
# use the mean rq_full which is the accuracy proxy at each step)
step_accuracy = {}
for step_num in sorted(step_pos_data.keys()):
    d = step_pos_data[step_num]
    acc = np.mean(d["rq_full"])
    step_accuracy[int(step_num)] = float(acc)

print("\n  Per-step accuracy (mean rq_full):")
for s in sorted(step_accuracy.keys()):
    print(f"    Step {s:2d}: {step_accuracy[s]:.4f}")

# Compute SPWA weight: w_k = product of accuracies of remaining steps k+1..N-1
# Intuition: the value of getting step k right is proportional to the probability
# that all remaining steps ALSO succeed.
# For each episode length N, for step k, w_k ~ prod(acc[k+1], ..., acc[N-1])
# We compute this at the population level using step_accuracy.

# Aggregate across all episodes: for each step position, compute the
# "expected remaining success probability"
max_step = max(step_accuracy.keys())
spwa_weights = {}
for k in sorted(step_accuracy.keys()):
    # product of accuracies for steps k+1 through max_step
    remaining_acc = 1.0
    n_remaining = 0
    for j in range(k + 1, max_step + 1):
        if j in step_accuracy:
            remaining_acc *= step_accuracy[j]
            n_remaining += 1
    spwa_weights[int(k)] = {
        "weight_raw": round(remaining_acc, 6),
        "n_remaining_steps": n_remaining,
        "step_accuracy": round(step_accuracy[k], 4),
    }

# Normalize weights
total_w = sum(v["weight_raw"] for v in spwa_weights.values())
for k in spwa_weights:
    spwa_weights[k]["weight_normalized"] = round(spwa_weights[k]["weight_raw"] / total_w, 6)

print("\n  SPWA weights (product of remaining accuracies):")
for k in sorted(spwa_weights.keys()):
    w = spwa_weights[k]
    print(f"    Step {k:2d}: w_raw={w['weight_raw']:.6f}  w_norm={w['weight_normalized']:.6f}")

# Compare SPWA weights with actual correlation values
print("\n  SPWA weight vs actual correlation (rq_full vs TSR):")
comparison = {}
for k in sorted(per_step_corr.keys()):
    if k in spwa_weights:
        corr_val = per_step_corr[k]["r"]
        w_norm = spwa_weights[k]["weight_normalized"]
        comparison[int(k)] = {"correlation_r": corr_val, "spwa_weight_norm": w_norm}
        print(f"    Step {k:2d}: corr={corr_val:+.4f}  spwa_w={w_norm:.4f}")

# Compute Spearman rank correlation between SPWA weights and actual correlations
if len(comparison) >= 3:
    steps = sorted(comparison.keys())
    corr_vals = [comparison[s]["correlation_r"] for s in steps]
    spwa_vals = [comparison[s]["spwa_weight_norm"] for s in steps]
    # filter nan
    valid = [(c, w) for c, w in zip(corr_vals, spwa_vals) if not (math.isnan(c) or math.isnan(w))]
    if len(valid) >= 3:
        cv, wv = zip(*valid)
        rho, p_rho = stats.spearmanr(cv, wv)
        print(f"\n  Spearman(SPWA_weight, actual_corr): rho={rho:.4f}, p={p_rho:.4f}")
        results["spwa_vs_correlation_spearman"] = {"rho": round(float(rho), 4), "p": round(float(p_rho), 6)}

results["spwa_weights"] = spwa_weights
results["spwa_vs_correlation"] = comparison


# ──────── Step 5: Conditional TSR Analysis ────────────────────────

print("\n=== Analysis 3: Conditional TSR by Step RQ Level ===")

# Group by step position bucket and RQ level
rq_thresholds = {"high": 0.8, "low": 0.3}
cond_results = {}

for bucket_name, bucket_fn in [("abs", bucket_abs), ("rel", bucket_rel)]:
    bucket_cond = {}
    grouped = defaultdict(lambda: {"high_tsr": [], "low_tsr": [], "mid_tsr": []})

    for rec in all_step_records:
        b = bucket_fn(rec["step_num"] if bucket_name == "abs" else rec["rel_pos"])
        rq = rec["rq_full"]
        tsr_val = int(rec["tsr"])

        if rq >= rq_thresholds["high"]:
            grouped[b]["high_tsr"].append(tsr_val)
        elif rq <= rq_thresholds["low"]:
            grouped[b]["low_tsr"].append(tsr_val)
        else:
            grouped[b]["mid_tsr"].append(tsr_val)

    buckets_order = (["early(0-2)", "mid(3-5)", "late(6+)"]
                     if bucket_name == "abs"
                     else ["first_third", "mid_third", "last_third"])

    print(f"\n  [{bucket_name.upper()} buckets] TSR when step RQ is HIGH (>={rq_thresholds['high']}) vs LOW (<={rq_thresholds['low']}):")
    for b in buckets_order:
        g = grouped[b]
        high_tsr = np.mean(g["high_tsr"]) if g["high_tsr"] else float("nan")
        low_tsr = np.mean(g["low_tsr"]) if g["low_tsr"] else float("nan")
        mid_tsr = np.mean(g["mid_tsr"]) if g["mid_tsr"] else float("nan")
        bucket_cond[b] = {
            "high_rq_tsr": round(float(high_tsr), 4),
            "high_rq_n": len(g["high_tsr"]),
            "mid_rq_tsr": round(float(mid_tsr), 4),
            "mid_rq_n": len(g["mid_tsr"]),
            "low_rq_tsr": round(float(low_tsr), 4),
            "low_rq_n": len(g["low_tsr"]),
        }
        gap = high_tsr - low_tsr if not (math.isnan(high_tsr) or math.isnan(low_tsr)) else float("nan")
        print(f"    {b}: HIGH RQ TSR={high_tsr:.3f} (n={len(g['high_tsr'])}), "
              f"LOW RQ TSR={low_tsr:.3f} (n={len(g['low_tsr'])}), "
              f"gap={gap:+.3f}")

    cond_results[bucket_name] = bucket_cond

results["conditional_tsr"] = cond_results


# ──────── Step 6: Oracle Fix Impact by Step Position ──────────────

print("\n=== Analysis 4: Oracle Fix Impact (early vs late) ===")

# For each episode, compute:
# - Current trajectory success = all steps correct (rq_full=1.0 for each step, using majority/greedy)
#   Actually: we approximate. For each step, greedy correctness = 1 if rq_full > 0.5 (majority says correct)
#   Better: use the actual sample-level data. We define "step k correct" if the best sample
#   (or more precisely, a greedy decode would match). Since we have K=10 samples,
#   let's define step k correct = 1 if rq_full >= 0.5 (majority of samples get it right).
#   Actually for oracle analysis, let's use rq_full as the probability of step k being correct.
#
# Oracle fix at step k: set P(step k correct) = 1.0
# Trajectory success ~ prod of all step correctness probs
# TSR improvement from oracle-fixing step k = product with step k forced to 1

# Build per-episode step accuracy lists
episode_step_rq = defaultdict(dict)
episode_tsr = {}
for rec in all_step_records:
    eid = rec["episode_id"]
    episode_step_rq[eid][rec["step_num"]] = rec["rq_full"]
    episode_tsr[eid] = int(rec["tsr"])

# For each step position, compute TSR improvement from oracle fixing that step
print("\n  Expected TSR under oracle fix at each step position:")
print("  (Using prod(rq_full_k) as trajectory success proxy)")

# First compute baseline expected TSR (product model)
baseline_tsrs = []
for eid in episode_step_rq:
    steps = episode_step_rq[eid]
    prob = 1.0
    for k in sorted(steps.keys()):
        prob *= steps[k]
    baseline_tsrs.append(prob)

baseline_etsr = np.mean(baseline_tsrs)
print(f"\n  Baseline expected TSR (product model): {baseline_etsr:.4f}")
print(f"  Actual TSR: {np.mean(list(episode_tsr.values())):.4f}")

oracle_results = {}
max_steps_to_analyze = 15  # limit for clarity

for fix_step in range(max_steps_to_analyze):
    oracle_tsrs = []
    n_applicable = 0
    for eid in episode_step_rq:
        steps = episode_step_rq[eid]
        if fix_step not in steps:
            continue
        n_applicable += 1
        prob = 1.0
        for k in sorted(steps.keys()):
            if k == fix_step:
                prob *= 1.0  # oracle fix
            else:
                prob *= steps[k]
        oracle_tsrs.append(prob)

    if n_applicable < 10:
        continue

    oracle_etsr = np.mean(oracle_tsrs)
    # Also compute baseline for just these episodes
    baseline_sub = []
    for eid in episode_step_rq:
        steps = episode_step_rq[eid]
        if fix_step not in steps:
            continue
        prob = 1.0
        for k in sorted(steps.keys()):
            prob *= steps[k]
        baseline_sub.append(prob)
    baseline_sub_etsr = np.mean(baseline_sub)

    improvement = oracle_etsr - baseline_sub_etsr
    rel_improvement = improvement / max(baseline_sub_etsr, 1e-10)

    oracle_results[int(fix_step)] = {
        "oracle_etsr": round(float(oracle_etsr), 4),
        "baseline_etsr": round(float(baseline_sub_etsr), 4),
        "improvement_abs": round(float(improvement), 4),
        "improvement_rel": round(float(rel_improvement), 4),
        "n_applicable": n_applicable,
    }
    print(f"    Fix step {fix_step:2d}: oracle_TSR={oracle_etsr:.4f}, "
          f"baseline={baseline_sub_etsr:.4f}, "
          f"delta={improvement:+.4f} ({rel_improvement:+.1%}), n={n_applicable}")

results["oracle_fix_impact"] = oracle_results
results["baseline_expected_tsr"] = round(float(baseline_etsr), 4)
results["actual_tsr"] = round(float(np.mean(list(episode_tsr.values()))), 4)


# ──────── Step 7: Reasoning Quality Decay Profile ─────────────────

print("\n=== Analysis 5: Reasoning Quality Decay Profile ===")

decay_profile = {}
for step_num in sorted(step_pos_data.keys()):
    d = step_pos_data[step_num]
    n = len(d["rq_full"])
    if n < 10:
        continue
    mean_rq = np.mean(d["rq_full"])
    std_rq = np.std(d["rq_full"])
    mean_tsr = np.mean(d["tsr"])
    decay_profile[int(step_num)] = {
        "mean_rq_full": round(float(mean_rq), 4),
        "std_rq_full": round(float(std_rq), 4),
        "mean_tsr": round(float(mean_tsr), 4),
        "n": n,
    }
    print(f"  Step {step_num:2d}: mean_rq={mean_rq:.4f} +/- {std_rq:.4f}, "
          f"mean_tsr={mean_tsr:.4f}, n={n}")

results["rq_decay_profile"] = decay_profile

# Test for monotonic decrease: Spearman correlation between step_num and mean_rq
if len(decay_profile) >= 3:
    steps_list = sorted(decay_profile.keys())
    rq_list = [decay_profile[s]["mean_rq_full"] for s in steps_list]
    rho, p_rho = stats.spearmanr(steps_list, rq_list)
    print(f"\n  Spearman(step_position, mean_rq_full): rho={rho:.4f}, p={p_rho:.6f}")
    results["rq_decay_spearman"] = {"rho": round(float(rho), 4), "p": round(float(p_rho), 6)}
    if rho < 0 and p_rho < 0.05:
        print("  => RQ decreases with step depth (significant). SPWA compensates correctly.")
    elif rho < 0:
        print("  => RQ tends to decrease with step depth (not significant).")
    else:
        print("  => RQ does NOT decrease with step depth.")


# ──────── Step 8: Additional analysis — per-episode SPWA score ────

print("\n=== Analysis 6: Per-Episode SPWA-Weighted Score vs TSR ===")

# For each episode, compute:
# 1. Uniform-weighted mean RQ: mean(rq_full_k)
# 2. SPWA-weighted mean RQ: sum(w_k * rq_full_k) / sum(w_k)
#    where w_k = prod(step_accuracy[j] for j in range(k+1, N))

episode_scores = []
for eid in episode_step_rq:
    steps = episode_step_rq[eid]
    n = len(steps)
    sorted_steps = sorted(steps.keys())

    # Uniform
    uniform_score = np.mean(list(steps.values()))

    # SPWA weighted: w_k = prod of population-level accuracies for remaining steps
    spwa_score_num = 0.0
    spwa_score_den = 0.0
    for k in sorted_steps:
        # weight = product of accuracies for steps after k in this episode
        w = 1.0
        for j in sorted_steps:
            if j > k:
                w *= step_accuracy.get(j, 0.5)
        spwa_score_num += w * steps[k]
        spwa_score_den += w

    spwa_score = spwa_score_num / max(spwa_score_den, 1e-10)

    episode_scores.append({
        "episode_id": eid,
        "tsr": episode_tsr[eid],
        "uniform_score": uniform_score,
        "spwa_score": spwa_score,
        "n_steps": n,
    })

tsr_arr = np.array([e["tsr"] for e in episode_scores])
uniform_arr = np.array([e["uniform_score"] for e in episode_scores])
spwa_arr = np.array([e["spwa_score"] for e in episode_scores])

r_uniform, p_uniform = point_biserial(tsr_arr, uniform_arr)
r_spwa, p_spwa = point_biserial(tsr_arr, spwa_arr)

print(f"  Uniform-weighted RQ vs TSR: r={r_uniform:.4f}, p={p_uniform:.6f}")
print(f"  SPWA-weighted RQ vs TSR:    r={r_spwa:.4f}, p={p_spwa:.6f}")
print(f"  Improvement: delta_r = {r_spwa - r_uniform:+.4f}")

if r_spwa > r_uniform:
    print("  => SPWA weighting improves correlation with TSR.")
else:
    print("  => SPWA weighting does NOT improve correlation with TSR.")

results["episode_score_correlation"] = {
    "uniform": {"r": round(r_uniform, 4), "p": round(p_uniform, 6)},
    "spwa": {"r": round(r_spwa, 4), "p": round(p_spwa, 6)},
    "delta_r": round(r_spwa - r_uniform, 4),
}

# AUC-ROC comparison: which score better predicts TSR?
try:
    from sklearn.metrics import roc_auc_score
    auc_uniform = roc_auc_score(tsr_arr, uniform_arr)
    auc_spwa = roc_auc_score(tsr_arr, spwa_arr)
    print(f"\n  AUC-ROC (Uniform): {auc_uniform:.4f}")
    print(f"  AUC-ROC (SPWA):    {auc_spwa:.4f}")
    results["episode_score_auc"] = {
        "uniform": round(auc_uniform, 4),
        "spwa": round(auc_spwa, 4),
    }
except ImportError:
    print("  (sklearn not available for AUC-ROC)")


# ──────── Step 9: Summary Statistics ──────────────────────────────

print("\n=== Summary ===")

# Key finding 1: Does early step RQ correlate more with TSR?
early_corr = corr_by_abs_bucket.get("early(0-2)", {}).get("rq_full", {}).get("r", float("nan"))
mid_corr = corr_by_abs_bucket.get("mid(3-5)", {}).get("rq_full", {}).get("r", float("nan"))
late_corr = corr_by_abs_bucket.get("late(6+)", {}).get("rq_full", {}).get("r", float("nan"))

print(f"\n  1. Correlation(rq_full, TSR) by position:")
print(f"     Early(0-2): r={early_corr:.4f}")
print(f"     Mid(3-5):   r={mid_corr:.4f}")
print(f"     Late(6+):   r={late_corr:.4f}")

spwa_validated = early_corr > late_corr if not (math.isnan(early_corr) or math.isnan(late_corr)) else None
if spwa_validated is True:
    print("     => Early steps have HIGHER correlation => SPWA justified")
elif spwa_validated is False:
    print("     => Early steps do NOT have higher correlation => SPWA may not be justified")
else:
    print("     => Insufficient data to compare")

# Key finding 2: Oracle fix impact ratio
if 0 in oracle_results and len(oracle_results) > 1:
    max_step_oracle = max(oracle_results.keys())
    early_impact = oracle_results[0]["improvement_abs"]
    late_impact = oracle_results[max_step_oracle]["improvement_abs"]
    print(f"\n  2. Oracle fix impact:")
    print(f"     Fix step 0: +{early_impact:.4f} TSR")
    print(f"     Fix step {max_step_oracle}: +{late_impact:.4f} TSR")
    if early_impact > 0 and late_impact > 0:
        print(f"     Ratio: {early_impact/late_impact:.2f}x")

# Key finding 3: SPWA vs uniform
print(f"\n  3. SPWA vs Uniform episode scoring:")
print(f"     Uniform: r={r_uniform:.4f}")
print(f"     SPWA:    r={r_spwa:.4f}")
print(f"     Delta:   {r_spwa - r_uniform:+.4f}")

results["summary"] = {
    "spwa_justified": spwa_validated,
    "early_corr_rq_full": round(early_corr, 4) if not math.isnan(early_corr) else None,
    "mid_corr_rq_full": round(mid_corr, 4) if not math.isnan(mid_corr) else None,
    "late_corr_rq_full": round(late_corr, 4) if not math.isnan(late_corr) else None,
    "spwa_score_better": bool(r_spwa > r_uniform),
    "n_episodes": len(c4c7_data),
    "n_steps_total": len(all_step_records),
}


# ──────── Save ────────────────────────────────────────────────────

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to: {OUT_PATH}")
print("Done.")
