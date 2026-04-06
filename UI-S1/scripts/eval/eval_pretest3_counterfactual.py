#!/usr/bin/env python3
"""
Pre-test 3: Emergent Adaptive Activation — Counterfactual Verification

Tests whether fixing different agents' errors has asymmetric benefits
across short vs long trajectories, which would justify role-conditioned
reward weighting.

Method:
For trajectories that fail in V2+V3, classify the first error type
(action vs grounding). Then compute: if we could oracle-fix that error,
how much would TSR improve?

The oracle fix is approximated as: the trajectory would succeed if
ALL subsequent steps also succeed (conservative lower bound).

Prediction (if framework is correct):
- Short trajectories: fixing action errors >> fixing grounding errors
- Long trajectories: fixing grounding errors >> fixing action errors
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


def length_bucket(n):
    if n <= 3: return "short (1-3)"
    elif n <= 7: return "medium (4-7)"
    elif n <= 15: return "long (8-15)"
    else: return "vlong (16+)"


def classify_first_error(step_results, prefix="b_"):
    """Return (step_num, error_type, remaining_steps_all_correct)."""
    for i, sr in enumerate(step_results):
        success_key = f"{prefix}success" if f"{prefix}success" in sr else "success"
        if not sr.get(success_key, True):
            func_key = f"{prefix}func_match" if f"{prefix}func_match" in sr else "func_match"
            coord_key = f"{prefix}coord_match" if f"{prefix}coord_match" in sr else "coord_match"

            func_match = sr.get(func_key, True)
            coord_match = sr.get(coord_key, True)

            # Check if all remaining steps would succeed (oracle ceiling)
            remaining = step_results[i + 1:]
            remaining_all_correct = all(
                sr2.get(success_key, True) for sr2 in remaining
            )

            if not func_match:
                return sr["step_num"], "action_error", remaining_all_correct
            elif not coord_match:
                return sr["step_num"], "grounding_error", remaining_all_correct
            else:
                return sr["step_num"], "other_error", remaining_all_correct
    return None, None, False


def analyze_oracle_fix(eval_a_data):
    """Main analysis: oracle fix benefits by error type × trajectory length."""
    print("=" * 80)
    print("ORACLE FIX ANALYSIS: Error Type × Trajectory Length")
    print("=" * 80)

    # Structure: bucket → error_type → {count, oracle_fixable}
    stats = defaultdict(lambda: defaultdict(lambda: {
        "count": 0,
        "oracle_fixable": 0,  # would succeed if this error fixed + rest OK
        "remaining_all_ok": 0,  # ALL remaining steps succeed
    }))

    # Also track: trajectory-level TSR improvement potential
    tsr_stats = defaultdict(lambda: {
        "total": 0,
        "b_success": 0,
        "action_errors": 0,
        "grounding_errors": 0,
        "other_errors": 0,
        "action_fixable": 0,
        "grounding_fixable": 0,
    })

    for traj in eval_a_data.values():
        bucket = length_bucket(traj["num_steps"])
        b_success = traj.get("b_trajectory_success", False)

        tsr_stats[bucket]["total"] += 1
        if b_success:
            tsr_stats[bucket]["b_success"] += 1
            continue

        # Failed trajectory — classify first error
        step, error_type, remaining_ok = classify_first_error(
            traj["step_results"], "b_"
        )

        if step is None:
            continue

        stats[bucket][error_type]["count"] += 1
        if remaining_ok:
            stats[bucket][error_type]["remaining_all_ok"] += 1
            stats[bucket][error_type]["oracle_fixable"] += 1

        if error_type == "action_error":
            tsr_stats[bucket]["action_errors"] += 1
            if remaining_ok:
                tsr_stats[bucket]["action_fixable"] += 1
        elif error_type == "grounding_error":
            tsr_stats[bucket]["grounding_errors"] += 1
            if remaining_ok:
                tsr_stats[bucket]["grounding_fixable"] += 1
        else:
            tsr_stats[bucket]["other_errors"] += 1

    # Print per-bucket analysis
    bucket_order = ["short (1-3)", "medium (4-7)", "long (8-15)", "vlong (16+)"]

    print(f"\n{'Bucket':<16} {'Error Type':<18} {'Count':>6} {'Remaining OK':>13} {'Oracle Fix %':>12}")
    print("-" * 70)

    for bucket in bucket_order:
        for etype in ["action_error", "grounding_error", "other_error"]:
            s = stats[bucket][etype]
            if s["count"] == 0:
                continue
            fix_pct = 100 * s["oracle_fixable"] / s["count"]
            print(f"{bucket:<16} {etype:<18} {s['count']:>6} {s['remaining_all_ok']:>13} {fix_pct:>11.1f}%")
        print()

    # TSR improvement potential
    print(f"\n{'='*80}")
    print("TSR IMPROVEMENT POTENTIAL (if oracle-fixing one error type)")
    print("=" * 80)

    print(f"\n{'Bucket':<16} {'N':>5} {'Current TSR':>12} {'Fix Action':>11} {'Fix Ground':>11} {'Δ Action':>9} {'Δ Ground':>9} {'Ratio':>7}")
    print("-" * 85)

    for bucket in bucket_order:
        t = tsr_stats[bucket]
        n = t["total"]
        if n == 0:
            continue

        current_tsr = t["b_success"] / n
        # If we fixed all action errors where remaining steps are OK
        new_tsr_action = (t["b_success"] + t["action_fixable"]) / n
        new_tsr_ground = (t["b_success"] + t["grounding_fixable"]) / n
        delta_action = new_tsr_action - current_tsr
        delta_ground = new_tsr_ground - current_tsr

        ratio = delta_action / delta_ground if delta_ground > 0 else float('inf')

        print(f"{bucket:<16} {n:>5} {100*current_tsr:>11.1f}% {100*new_tsr_action:>10.1f}% {100*new_tsr_ground:>10.1f}% {100*delta_action:>+8.1f}pp {100*delta_ground:>+8.1f}pp {ratio:>6.2f}")

    return stats, tsr_stats


def analyze_v2_only_comparison(eval_a_data):
    """Compare oracle fix benefits using V2-only data for additional signal."""
    print(f"\n{'='*80}")
    print("V2-ONLY vs V2+V3: Oracle Fix Comparison")
    print("=" * 80)

    bucket_order = ["short (1-3)", "medium (4-7)", "long (8-15)", "vlong (16+)"]

    comparison = defaultdict(lambda: {
        "total": 0,
        "a_success": 0, "b_success": 0,
        "a_ground_first": 0, "b_ground_first": 0,
        "a_action_first": 0, "b_action_first": 0,
    })

    for traj in eval_a_data.values():
        bucket = length_bucket(traj["num_steps"])
        c = comparison[bucket]
        c["total"] += 1

        a_success = traj.get("a_trajectory_success", False)
        b_success = traj.get("b_trajectory_success", False)
        if a_success:
            c["a_success"] += 1
        if b_success:
            c["b_success"] += 1

        if not a_success:
            _, etype, _ = classify_first_error(traj["step_results"], "a_")
            if etype == "grounding_error":
                c["a_ground_first"] += 1
            elif etype == "action_error":
                c["a_action_first"] += 1

        if not b_success:
            _, etype, _ = classify_first_error(traj["step_results"], "b_")
            if etype == "grounding_error":
                c["b_ground_first"] += 1
            elif etype == "action_error":
                c["b_action_first"] += 1

    print(f"\n{'Bucket':<16} {'N':>5} {'A TSR':>7} {'B TSR':>7} {'A ground%':>10} {'B ground%':>10} {'A action%':>10} {'B action%':>10}")
    print("-" * 85)

    for bucket in bucket_order:
        c = comparison[bucket]
        n = c["total"]
        if n == 0:
            continue
        a_tsr = 100 * c["a_success"] / n
        b_tsr = 100 * c["b_success"] / n
        a_fail = n - c["a_success"]
        b_fail = n - c["b_success"]
        a_gp = 100 * c["a_ground_first"] / a_fail if a_fail > 0 else 0
        b_gp = 100 * c["b_ground_first"] / b_fail if b_fail > 0 else 0
        a_ap = 100 * c["a_action_first"] / a_fail if a_fail > 0 else 0
        b_ap = 100 * c["b_action_first"] / b_fail if b_fail > 0 else 0

        print(f"{bucket:<16} {n:>5} {a_tsr:>6.1f}% {b_tsr:>6.1f}% {a_gp:>9.1f}% {b_gp:>9.1f}% {a_ap:>9.1f}% {b_ap:>9.1f}%")

    print(f"\nKey: A=V2-only, B=V2+V3. ground%=fraction of fails with grounding first-error")
    print("If V3 fixes grounding: B_ground% should be LOWER than A_ground%")
    print("If V3 doesn't fix action: B_action% should be HIGHER than A_action%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_a_dir", default="outputs/eval_a")
    parser.add_argument("--output_dir", default="outputs/eval_pretest3")
    args = parser.parse_args()

    print("=" * 80)
    print("PRE-TEST 3: COUNTERFACTUAL ORACLE FIX ANALYSIS")
    print("=" * 80)

    eval_a = load_eval_a(os.path.join(args.eval_a_dir, "trajectory_results.jsonl"))
    print(f"Loaded {len(eval_a)} trajectories")

    stats, tsr_stats = analyze_oracle_fix(eval_a)
    analyze_v2_only_comparison(eval_a)

    # Save summary
    os.makedirs(args.output_dir, exist_ok=True)
    summary = {
        "method": "counterfactual_oracle_fix",
        "prediction": "short trajs: action fix > grounding fix; long trajs: grounding fix > action fix",
    }
    for bucket in ["short (1-3)", "medium (4-7)", "long (8-15)", "vlong (16+)"]:
        t = tsr_stats[bucket]
        n = t["total"]
        if n == 0:
            continue
        summary[bucket] = {
            "n": n,
            "current_tsr": t["b_success"] / n,
            "action_fixable": t["action_fixable"],
            "grounding_fixable": t["grounding_fixable"],
            "delta_action": t["action_fixable"] / n,
            "delta_grounding": t["grounding_fixable"] / n,
        }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print("PRE-TEST 3 COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
