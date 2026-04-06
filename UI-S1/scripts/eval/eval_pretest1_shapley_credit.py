#!/usr/bin/env python3
"""
Pre-test 1: Shapley Credit Function Shape Estimation

Estimates φ_V2(t) and φ_V3(t) — the marginal contribution of each agent
at step t — using existing eval data from three conditions:
  - Condition A (V2 only): Eval A, a_* fields
  - Condition B (V2+V3): Eval A, b_* fields
  - Condition C (V2+V3+Observer): D1

Hypothesis:
  - φ_V3(t) should increase with t (V3 more valuable in later steps)
  - φ_V2(t) should decrease with t (V2 more valuable in early steps)
  - Crossover at step 3-4

Method 1: Per-step success rate comparison across conditions
Method 2: First-error-type contribution by step position
"""

import json
import os
import argparse
from collections import defaultdict
import numpy as np


def load_eval_a(path):
    results = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                results[d["trajectory_id"]] = d
    return results


def load_d1(path):
    results = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                if "trajectory_success" in d:
                    results[d["trajectory_id"]] = d
    return results


def classify_first_error(step_results, prefix="b_"):
    """Classify the type of first error in a trajectory."""
    for sr in step_results:
        success_key = f"{prefix}success" if f"{prefix}success" in sr else "success"
        if not sr.get(success_key, True):
            func_key = f"{prefix}func_match" if f"{prefix}func_match" in sr else "func_match"
            coord_key = f"{prefix}coord_match" if f"{prefix}coord_match" in sr else "coord_match"

            func_match = sr.get(func_key, True)
            coord_match = sr.get(coord_key, True)

            if not func_match:
                return sr["step_num"], "action_error"
            elif not coord_match:
                return sr["step_num"], "grounding_error"
            else:
                return sr["step_num"], "other_error"
    return None, None


def method1_per_step_success(eval_a_data):
    """Method 1: Per-step success rate for V2-only vs V2+V3."""
    print("=" * 80)
    print("METHOD 1: Per-Step Success Rate (V2-only vs V2+V3)")
    print("=" * 80)

    step_stats = defaultdict(lambda: {
        "a_correct": 0, "a_total": 0,  # V2 only
        "b_correct": 0, "b_total": 0,  # V2+V3
    })

    for traj in eval_a_data.values():
        for sr in traj["step_results"]:
            step = sr["step_num"]
            step_stats[step]["a_total"] += 1
            step_stats[step]["b_total"] += 1
            if sr.get("a_success", False):
                step_stats[step]["a_correct"] += 1
            if sr.get("b_success", False):
                step_stats[step]["b_correct"] += 1

    print(f"\n{'Step':>5} {'N':>6} {'V2-only':>8} {'V2+V3':>8} {'φ_V3(t)':>10} {'φ_V3 cumul':>11}")
    print("-" * 55)

    phi_v3 = {}
    for step in sorted(step_stats.keys()):
        s = step_stats[step]
        if s["a_total"] == 0:
            continue
        a_rate = s["a_correct"] / s["a_total"]
        b_rate = s["b_correct"] / s["b_total"]
        phi = b_rate - a_rate
        phi_v3[step] = phi

        print(f"{step:>5} {s['a_total']:>6} {100*a_rate:>7.1f}% {100*b_rate:>7.1f}% {100*phi:>+9.1f}pp")

    return phi_v3


def method2_first_error_contribution(eval_a_data):
    """Method 2: V3's marginal contribution by first-error step position."""
    print(f"\n{'='*80}")
    print("METHOD 2: V3 Marginal Contribution by First-Error Step")
    print("=" * 80)

    # For each step t, find trajectories where first error occurs at step t
    # Compare V2-only vs V2+V3 success for those specific trajectories
    step_data = defaultdict(lambda: {
        "a_grounding_errors": 0,  # V2-only had grounding error at this step
        "b_grounding_errors": 0,  # V2+V3 had grounding error at this step
        "a_action_errors": 0,
        "b_action_errors": 0,
        "a_total_errors": 0,
        "b_total_errors": 0,
    })

    # Also track: for trajectories with first error at step t,
    # what fraction succeed in V2+V3 vs V2-only?
    step_rescue = defaultdict(lambda: {
        "a_fail_b_success": 0,  # V3 rescued
        "a_fail_b_fail": 0,     # V3 couldn't rescue
        "both_fail": 0,
        "total_a_fail": 0,
    })

    for traj in eval_a_data.values():
        # Classify first error in condition A (V2 only)
        a_step, a_type = classify_first_error(traj["step_results"], "a_")
        # Classify first error in condition B (V2+V3)
        b_step, b_type = classify_first_error(traj["step_results"], "b_")

        a_success = traj.get("a_trajectory_success", False)
        b_success = traj.get("b_trajectory_success", False)

        if a_step is not None:
            step_data[a_step]["a_total_errors"] += 1
            if a_type == "grounding_error":
                step_data[a_step]["a_grounding_errors"] += 1
            elif a_type == "action_error":
                step_data[a_step]["a_action_errors"] += 1

            # Rescue analysis: did V3 save this trajectory?
            if not a_success:
                step_rescue[a_step]["total_a_fail"] += 1
                if b_success:
                    step_rescue[a_step]["a_fail_b_success"] += 1
                else:
                    step_rescue[a_step]["a_fail_b_fail"] += 1

        if b_step is not None:
            step_data[b_step]["b_total_errors"] += 1
            if b_type == "grounding_error":
                step_data[b_step]["b_grounding_errors"] += 1
            elif b_type == "action_error":
                step_data[b_step]["b_action_errors"] += 1

    print(f"\n{'Step':>5} {'A errors':>9} {'A ground%':>10} {'B errors':>9} {'B ground%':>10} {'V3 rescue%':>11}")
    print("-" * 60)

    for step in sorted(step_data.keys()):
        d = step_data[step]
        r = step_rescue[step]

        a_ground_pct = 100 * d["a_grounding_errors"] / d["a_total_errors"] if d["a_total_errors"] > 0 else 0
        b_ground_pct = 100 * d["b_grounding_errors"] / d["b_total_errors"] if d["b_total_errors"] > 0 else 0
        rescue_pct = 100 * r["a_fail_b_success"] / r["total_a_fail"] if r["total_a_fail"] > 0 else 0

        print(f"{step:>5} {d['a_total_errors']:>9} {a_ground_pct:>9.1f}% {d['b_total_errors']:>9} {b_ground_pct:>9.1f}% {rescue_pct:>10.1f}%")

    return step_data, step_rescue


def method3_phi_by_trajectory_length(eval_a_data, d1_data):
    """Method 3: φ_V3(t) estimated per trajectory length bucket."""
    print(f"\n{'='*80}")
    print("METHOD 3: φ_V3 and φ_Obs by Trajectory Length Bucket")
    print("=" * 80)

    def bucket(n):
        if n <= 3: return "1-3"
        elif n <= 7: return "4-7"
        elif n <= 15: return "8-15"
        else: return "16+"

    matched_ids = set(eval_a_data.keys()) & set(d1_data.keys())

    # Per-step × per-length-bucket analysis
    length_step_stats = defaultdict(lambda: defaultdict(lambda: {
        "a_correct": 0, "b_correct": 0, "d1_correct": 0, "total": 0
    }))

    for tid in matched_ids:
        a = eval_a_data[tid]
        d = d1_data[tid]
        length_bucket = bucket(a["num_steps"])

        # We can only compare step-level for eval_a (has a/b per step)
        # D1 only has trajectory-level success
        for sr in a["step_results"]:
            step = sr["step_num"]
            s = length_step_stats[length_bucket][step]
            s["total"] += 1
            if sr.get("a_success", False):
                s["a_correct"] += 1
            if sr.get("b_success", False):
                s["b_correct"] += 1

    for lb in ["1-3", "4-7", "8-15", "16+"]:
        print(f"\n  --- Length bucket: {lb} ---")
        print(f"  {'Step':>5} {'N':>6} {'V2-only':>8} {'V2+V3':>8} {'φ_V3':>8}")
        print(f"  {'-'*40}")

        for step in sorted(length_step_stats[lb].keys()):
            s = length_step_stats[lb][step]
            if s["total"] < 10:
                continue
            a_rate = s["a_correct"] / s["total"]
            b_rate = s["b_correct"] / s["total"]
            phi = b_rate - a_rate
            print(f"  {step:>5} {s['total']:>6} {100*a_rate:>7.1f}% {100*b_rate:>7.1f}% {100*phi:>+7.1f}pp")


def method4_phi_trajectory_level(eval_a_data, d1_data):
    """Method 4: φ at trajectory level — decomposed by first-error step."""
    print(f"\n{'='*80}")
    print("METHOD 4: Trajectory-Level φ by First-Error Step")
    print("=" * 80)
    print("For trajectories that fail in V2-only at step t,")
    print("what fraction are rescued by V2+V3 (φ_V3) vs V2+V3+Obs (φ_Obs)?")

    matched_ids = set(eval_a_data.keys()) & set(d1_data.keys())

    step_rescue = defaultdict(lambda: {
        "a_fail": 0,
        "b_rescues": 0,   # V3 rescued (a_fail, b_success)
        "d1_rescues": 0,   # Observer rescued (b_fail, d1_success)
        "a_fail_grounding": 0,
        "a_fail_action": 0,
    })

    for tid in matched_ids:
        a = eval_a_data[tid]
        d = d1_data[tid]

        a_success = a.get("a_trajectory_success", False)
        b_success = a.get("b_trajectory_success", False)
        d1_success = d.get("trajectory_success", False)

        if a_success:
            continue

        # Find first error step in V2-only
        first_step, error_type = classify_first_error(a["step_results"], "a_")
        if first_step is None:
            continue

        r = step_rescue[first_step]
        r["a_fail"] += 1
        if error_type == "grounding_error":
            r["a_fail_grounding"] += 1
        elif error_type == "action_error":
            r["a_fail_action"] += 1

        if b_success:
            r["b_rescues"] += 1
        elif d1_success:
            r["d1_rescues"] += 1

    print(f"\n{'Step':>5} {'A fails':>8} {'V3 rescue':>10} {'Obs rescue':>11} {'φ_V3':>8} {'φ_Obs':>8} {'Ground%':>9}")
    print("-" * 70)

    for step in sorted(step_rescue.keys()):
        r = step_rescue[step]
        if r["a_fail"] < 5:
            continue
        v3_rate = r["b_rescues"] / r["a_fail"]
        obs_rate = r["d1_rescues"] / r["a_fail"]
        ground_pct = r["a_fail_grounding"] / r["a_fail"]

        print(f"{step:>5} {r['a_fail']:>8} {r['b_rescues']:>10} {r['d1_rescues']:>11} {100*v3_rate:>+7.1f}% {100*obs_rate:>+7.1f}% {100*ground_pct:>8.1f}%")


def method5_phi_monotonicity_test(eval_a_data):
    """Method 5: Statistical test for φ_V3(t) monotonicity."""
    print(f"\n{'='*80}")
    print("METHOD 5: φ_V3(t) Monotonicity Test")
    print("=" * 80)

    # Compute φ_V3 at each step
    step_phi = {}
    step_stats = defaultdict(lambda: {"a": [], "b": []})

    for traj in eval_a_data.values():
        for sr in traj["step_results"]:
            step = sr["step_num"]
            step_stats[step]["a"].append(1 if sr.get("a_success", False) else 0)
            step_stats[step]["b"].append(1 if sr.get("b_success", False) else 0)

    steps = []
    phis = []
    for step in sorted(step_stats.keys()):
        if len(step_stats[step]["a"]) < 20:
            continue
        a_mean = np.mean(step_stats[step]["a"])
        b_mean = np.mean(step_stats[step]["b"])
        phi = b_mean - a_mean
        step_phi[step] = phi
        steps.append(step)
        phis.append(phi)

    if len(steps) < 3:
        print("Not enough steps for monotonicity test.")
        return

    # Kendall's tau for monotonicity
    from scipy import stats as scipy_stats
    tau, p_value = scipy_stats.kendalltau(steps, phis)

    print(f"\nKendall's tau (φ_V3 vs step): τ = {tau:.3f}, p = {p_value:.4f}")

    if tau > 0 and p_value < 0.05:
        print("→ φ_V3(t) is significantly INCREASING with step ✓ (supports hypothesis)")
    elif tau < 0 and p_value < 0.05:
        print("→ φ_V3(t) is significantly DECREASING with step ✗ (contradicts hypothesis)")
    else:
        print("→ φ_V3(t) has no significant monotonic trend (inconclusive)")

    # Also test: does grounding error fraction increase with step?
    step_grounding_frac = []
    step_nums = []
    for traj in eval_a_data.values():
        for sr in traj["step_results"]:
            if not sr.get("b_success", True):
                step = sr["step_num"]
                is_grounding = sr.get("b_func_match", True) and not sr.get("b_coord_match", True)
                step_grounding_frac.append((step, 1 if is_grounding else 0))

    # Group by step
    step_ground = defaultdict(list)
    for step, is_ground in step_grounding_frac:
        step_ground[step].append(is_ground)

    gsteps = []
    gfracs = []
    for step in sorted(step_ground.keys()):
        if len(step_ground[step]) < 10:
            continue
        gsteps.append(step)
        gfracs.append(np.mean(step_ground[step]))

    if len(gsteps) >= 3:
        tau2, p2 = scipy_stats.kendalltau(gsteps, gfracs)
        print(f"\nKendall's tau (grounding_error_fraction vs step): τ = {tau2:.3f}, p = {p2:.4f}")
        if tau2 > 0 and p2 < 0.05:
            print("→ Grounding error fraction INCREASES with step ✓ (consistent with D7)")

    return step_phi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_a_dir", default="outputs/eval_a")
    parser.add_argument("--d1_dir", default="outputs/eval_d1")
    parser.add_argument("--output_dir", default="outputs/eval_pretest1")
    args = parser.parse_args()

    print("=" * 80)
    print("PRE-TEST 1: SHAPLEY CREDIT FUNCTION SHAPE ESTIMATION")
    print("=" * 80)

    eval_a = load_eval_a(os.path.join(args.eval_a_dir, "trajectory_results.jsonl"))
    d1 = load_d1(os.path.join(args.d1_dir, "trajectory_results.jsonl"))
    print(f"Eval A: {len(eval_a)} trajectories")
    print(f"D1 (new format): {len(d1)} trajectories")

    # Run all methods
    phi_v3_m1 = method1_per_step_success(eval_a)
    step_data, step_rescue = method2_first_error_contribution(eval_a)
    method3_phi_by_trajectory_length(eval_a, d1)
    method4_phi_trajectory_level(eval_a, d1)

    try:
        step_phi = method5_phi_monotonicity_test(eval_a)
    except ImportError:
        print("\nScipy not available, skipping monotonicity test.")
        step_phi = phi_v3_m1

    # Save summary
    os.makedirs(args.output_dir, exist_ok=True)
    summary = {
        "phi_v3_per_step": {str(k): v for k, v in phi_v3_m1.items()},
        "method": "per_step_success_rate_difference",
        "hypothesis": "phi_V3 should increase with step number",
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print("PRE-TEST 1 COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
