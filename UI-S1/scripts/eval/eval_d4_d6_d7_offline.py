#!/usr/bin/env python3
"""
Eval D4 + D6 + D7: Offline analyses on existing Eval A and D1 data.

D4: Planner Oracle Ceiling
    - Find "Observer wins step but trajectory still fails" cases
    - These are "know where I am but don't know where to go" → Planner's target

D6: Failure Type Taxonomy
    - Classify first-error in each failed trajectory:
      Type A: Grounding error (func_match=T, coord_match=F) → V3 RL
      Type B: Action decision error (func_match=F) → V2 improvement
      Type C: Status/completion error (status_match=F) → Verifier
      Type D: Planning error (all match at step level but trajectory fails) → Planner
    - Also: state confusion subset of Type A (func_match=T, coord_match=F)

D7: Length × Failure Cause Cross-Analysis
    - How failure type distribution changes with trajectory length
    - Short (1-3), Med (4-7), Long (8-15), VLong (16+)
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_eval_a():
    """Load Eval A results."""
    results = []
    path = PROJECT_ROOT / "outputs" / "eval_a" / "trajectory_results.jsonl"
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def load_eval_d1():
    """Load Eval D1 results (new format only)."""
    results = {}
    path = PROJECT_ROOT / "outputs" / "eval_d1" / "trajectory_results.jsonl"
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "trajectory_success" in r:
                results[r["trajectory_id"]] = r
    return results


def classify_first_error(step_results, key_prefix="b_"):
    """Classify the type of first error in a trajectory.

    Returns (error_type, step_idx, details) or None if no error.
    """
    for i, sr in enumerate(step_results):
        # Determine success key: Eval A uses "b_success", D1 uses "success"
        success_key = f"{key_prefix}success" if f"{key_prefix}success" in sr else "success"
        func_key = f"{key_prefix}func_match" if f"{key_prefix}func_match" in sr else "func_match"
        coord_key = f"{key_prefix}coord_match" if f"{key_prefix}coord_match" in sr else "coord_match"
        args_key = f"{key_prefix}args_match" if f"{key_prefix}args_match" in sr else "args_match"
        status_key = f"{key_prefix}status_match" if f"{key_prefix}status_match" in sr else "status_match"

        if sr.get(success_key, True):
            continue

        func_match = sr.get(func_key, sr.get("func_match", False))
        coord_match = sr.get(coord_key, sr.get("coord_match", False))
        args_match = sr.get(args_key, sr.get("args_match", False))
        status_match = sr.get(status_key, sr.get("status_match", False))

        step_idx = sr.get("step_idx", sr.get("step_num", i + 1))
        gt_func = sr.get("gt_function", "unknown")
        pred_func = sr.get(f"{key_prefix}pred_function", sr.get("pred_function", "unknown"))

        if not func_match:
            # Type B: Wrong action entirely
            return ("B_action_error", step_idx, {
                "gt": gt_func, "pred": pred_func,
                "func_match": func_match, "coord_match": coord_match
            })
        elif func_match and not coord_match:
            # Type A: Right action, wrong coordinate (state confusion)
            return ("A_grounding_error", step_idx, {
                "gt": gt_func, "pred": pred_func,
                "func_match": True, "coord_match": False
            })
        elif not status_match:
            # Type C: Status error (CONTINUE vs FINISH mismatch)
            return ("C_status_error", step_idx, {
                "gt": gt_func, "pred": pred_func,
                "status_issue": True
            })
        else:
            # Type D: All individual checks pass but step is marked as failed
            # This happens when args match fails on non-coordinate args
            return ("D_other_args_error", step_idx, {
                "gt": gt_func, "pred": pred_func,
                "func_match": func_match, "coord_match": coord_match,
                "args_match": args_match, "status_match": status_match
            })

    return None  # No error found


# =====================================================================
# D6: Failure Type Taxonomy
# =====================================================================
def eval_d6(eval_a_results):
    print("\n" + "=" * 70)
    print("  Eval D6: Failure Type Taxonomy")
    print("=" * 70)

    # Only look at condition B (V2+V3) trajectories that failed
    failed = [r for r in eval_a_results if not r.get("b_trajectory_success", False)]
    success = [r for r in eval_a_results if r.get("b_trajectory_success", False)]

    print(f"\n  Total trajectories: {len(eval_a_results)}")
    print(f"  Successful: {len(success)} ({len(success)/len(eval_a_results)*100:.1f}%)")
    print(f"  Failed: {len(failed)} ({len(failed)/len(eval_a_results)*100:.1f}%)")

    # Classify each failed trajectory's first error
    type_counts = Counter()
    type_by_domain = defaultdict(Counter)
    type_by_step = defaultdict(list)
    type_examples = defaultdict(list)

    for r in failed:
        classification = classify_first_error(r.get("step_results", []), key_prefix="b_")
        if classification is None:
            type_counts["unclassified"] += 1
            continue

        error_type, step_idx, details = classification
        type_counts[error_type] += 1
        type_by_domain[r["domain"]][error_type] += 1
        type_by_step[error_type].append(step_idx)

        if len(type_examples[error_type]) < 5:
            type_examples[error_type].append({
                "trajectory_id": r["trajectory_id"],
                "step": step_idx,
                "gt": details.get("gt"),
                "pred": details.get("pred"),
            })

    # Summary
    print(f"\n  Failure Type Distribution (first error):")
    print(f"  {'Type':<25s} {'Count':>6s} {'%':>7s} {'Description'}")
    print(f"  {'-'*25} {'-'*6} {'-'*7} {'-'*40}")

    type_labels = {
        "A_grounding_error": "V3 RL 可解决 (坐标错误)",
        "B_action_error": "V2 改进可解决 (action 决策错误)",
        "C_status_error": "Verifier 可解决 (状态判断错误)",
        "D_other_args_error": "Planner 可能解决 (其他 args 错误)",
        "unclassified": "无法分类",
    }

    total_classified = sum(type_counts.values())
    for error_type in ["A_grounding_error", "B_action_error", "C_status_error", "D_other_args_error", "unclassified"]:
        count = type_counts[error_type]
        pct = count / total_classified * 100 if total_classified > 0 else 0
        label = type_labels.get(error_type, error_type)
        print(f"  {error_type:<25s} {count:>6d} {pct:>6.1f}% {label}")

    # Per-domain breakdown
    print(f"\n  Per-domain failure type distribution:")
    print(f"  {'Domain':<10s} {'A_ground':>10s} {'B_action':>10s} {'C_status':>10s} {'D_other':>10s}")
    for domain in sorted(type_by_domain.keys()):
        counts = type_by_domain[domain]
        total = sum(counts.values())
        a = counts["A_grounding_error"] / total * 100 if total > 0 else 0
        b = counts["B_action_error"] / total * 100 if total > 0 else 0
        c = counts["C_status_error"] / total * 100 if total > 0 else 0
        d = counts["D_other_args_error"] / total * 100 if total > 0 else 0
        print(f"  {domain:<10s} {a:>9.1f}% {b:>9.1f}% {c:>9.1f}% {d:>9.1f}%")

    # Average first-error step by type
    print(f"\n  Average first-error step by type:")
    for error_type in ["A_grounding_error", "B_action_error", "C_status_error", "D_other_args_error"]:
        steps = type_by_step[error_type]
        if steps:
            print(f"  {error_type:<25s} mean={np.mean(steps):.1f}, median={np.median(steps):.1f}, "
                  f"step1={sum(1 for s in steps if s <= 1)/len(steps)*100:.0f}%")

    # Examples
    print(f"\n  Examples per type:")
    for error_type in ["A_grounding_error", "B_action_error", "C_status_error", "D_other_args_error"]:
        examples = type_examples[error_type][:3]
        if examples:
            print(f"  {error_type}:")
            for ex in examples:
                print(f"    {ex['trajectory_id']} step {ex['step']}: gt={ex['gt']}, pred={ex['pred']}")

    return type_counts, type_by_domain, type_by_step


# =====================================================================
# D7: Length × Failure Type Cross-Analysis
# =====================================================================
def eval_d7(eval_a_results, type_counts_data=None):
    print("\n" + "=" * 70)
    print("  Eval D7: Length × Failure Type Cross-Analysis")
    print("=" * 70)

    length_bins = [
        (1, 3, "Short (1-3)"),
        (4, 7, "Med (4-7)"),
        (8, 15, "Long (8-15)"),
        (16, 100, "VLong (16+)"),
    ]

    failed = [r for r in eval_a_results if not r.get("b_trajectory_success", False)]

    print(f"\n  {'Length':<15s} {'N':>5s} {'A_ground':>10s} {'B_action':>10s} {'C_status':>10s} {'D_other':>10s} {'TSR':>7s}")
    print(f"  {'-'*15} {'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*7}")

    length_type_data = {}

    for lo, hi, label in length_bins:
        bin_all = [r for r in eval_a_results if lo <= r["num_steps"] <= hi]
        bin_failed = [r for r in failed if lo <= r["num_steps"] <= hi]

        if not bin_failed:
            continue

        type_counts = Counter()
        for r in bin_failed:
            classification = classify_first_error(r.get("step_results", []), key_prefix="b_")
            if classification:
                type_counts[classification[0]] += 1
            else:
                type_counts["unclassified"] += 1

        total = sum(type_counts.values())
        a = type_counts["A_grounding_error"] / total * 100
        b = type_counts["B_action_error"] / total * 100
        c = type_counts["C_status_error"] / total * 100
        d = type_counts["D_other_args_error"] / total * 100
        tsr = sum(1 for r in bin_all if r.get("b_trajectory_success", False)) / len(bin_all) * 100

        print(f"  {label:<15s} {len(bin_failed):>5d} {a:>9.1f}% {b:>9.1f}% {c:>9.1f}% {d:>9.1f}% {tsr:>6.1f}%")

        length_type_data[label] = {
            "n_failed": len(bin_failed),
            "n_total": len(bin_all),
            "tsr": tsr,
            "A_grounding": a,
            "B_action": b,
            "C_status": c,
            "D_other": d,
        }

    # Key insight: how does state confusion change with length?
    print(f"\n  Key insight: Grounding error (Type A = state confusion) by length:")
    for lo, hi, label in length_bins:
        bin_failed = [r for r in failed if lo <= r["num_steps"] <= hi]
        if not bin_failed:
            continue
        type_a = sum(1 for r in bin_failed
                     if classify_first_error(r.get("step_results", []), key_prefix="b_")
                     and classify_first_error(r.get("step_results", []), key_prefix="b_")[0] == "A_grounding_error")
        print(f"  {label:<15s}: {type_a}/{len(bin_failed)} = {type_a/len(bin_failed)*100:.1f}%")

    return length_type_data


# =====================================================================
# D4: Planner Oracle Ceiling
# =====================================================================
def eval_d4(eval_a_results, eval_d1_results):
    print("\n" + "=" * 70)
    print("  Eval D4: Planner Oracle Ceiling")
    print("=" * 70)

    # Find trajectories where D1 (Observer) won at step level but trajectory still failed
    # These are cases where Observer helped with state awareness but the agent still
    # made wrong decisions → Planner could help

    # First: match D1 results with Eval A
    matched = []
    for r in eval_a_results:
        tid = r["trajectory_id"]
        if tid in eval_d1_results:
            matched.append((r, eval_d1_results[tid]))

    print(f"\n  Matched trajectories: {len(matched)}")

    # Category 1: D1 succeeded but Eval A failed → Observer already solved
    observer_solved = [(a, d) for a, d in matched
                       if not a["b_trajectory_success"] and d["trajectory_success"]]

    # Category 2: Both failed, but D1 had better progress → Observer helped partially
    observer_partial = [(a, d) for a, d in matched
                        if not a["b_trajectory_success"] and not d["trajectory_success"]
                        and d["progress_rate"] > a["b_progress_rate"] + 0.01]

    # Category 3: Both failed, same progress → Observer didn't help
    observer_noop = [(a, d) for a, d in matched
                     if not a["b_trajectory_success"] and not d["trajectory_success"]
                     and abs(d["progress_rate"] - a["b_progress_rate"]) <= 0.01]

    # Category 4: Both failed, D1 worse → Observer confused things
    observer_hurt = [(a, d) for a, d in matched
                     if not a["b_trajectory_success"] and not d["trajectory_success"]
                     and d["progress_rate"] < a["b_progress_rate"] - 0.01]

    # Category 5: Eval A succeeded
    already_ok = [(a, d) for a, d in matched if a["b_trajectory_success"]]

    # Category 6: D1 failed but Eval A succeeded → Observer hurt (regression)
    observer_regression = [(a, d) for a, d in matched
                           if a["b_trajectory_success"] and not d["trajectory_success"]]

    total_a_failed = len(matched) - len(already_ok)

    print(f"\n  Eval A failed trajectories: {total_a_failed}")
    print(f"  Breakdown:")
    print(f"    Observer solved (D1 OK, A fail):      {len(observer_solved):>5d} ({len(observer_solved)/total_a_failed*100:.1f}%)")
    print(f"    Observer helped (D1 better progress):  {len(observer_partial):>5d} ({len(observer_partial)/total_a_failed*100:.1f}%)")
    print(f"    Observer no effect (same progress):    {len(observer_noop):>5d} ({len(observer_noop)/total_a_failed*100:.1f}%)")
    print(f"    Observer hurt (D1 worse progress):     {len(observer_hurt):>5d} ({len(observer_hurt)/total_a_failed*100:.1f}%)")

    print(f"\n  Observer regression (A OK, D1 fail):    {len(observer_regression):>5d}")

    # Planner's target: Cases where Observer helped partially but trajectory still failed
    # These are "know where I am but don't know where to go"
    planner_target = observer_partial
    planner_noop_target = observer_noop  # Observer didn't help either → harder problem

    print(f"\n  === Planner Value Estimation ===")
    print(f"  Planner primary target (Observer helped but failed): {len(planner_target)}")
    print(f"  Planner secondary target (Observer no effect):       {len(planner_noop_target)}")

    # If Planner could fix all primary targets → additional TSR gain
    planner_primary_gain = len(planner_target) / len(matched) * 100
    planner_total_gain = (len(planner_target) + len(planner_noop_target)) / len(matched) * 100

    current_tsr = len(already_ok) / len(matched) * 100
    d1_tsr = (len(already_ok) + len(observer_solved) - len(observer_regression)) / len(matched) * 100

    print(f"\n  Current Eval A TSR:                     {current_tsr:.1f}%")
    print(f"  D1 Observer TSR:                        {d1_tsr:.1f}%")
    print(f"  + Planner fixes primary targets:        +{planner_primary_gain:.1f}pp → {d1_tsr + planner_primary_gain:.1f}%")
    print(f"  + Planner fixes ALL remaining failures: +{planner_total_gain:.1f}pp → {d1_tsr + planner_total_gain:.1f}%")

    # Analyze planner target characteristics
    if planner_target:
        print(f"\n  Planner target characteristics:")
        domains = Counter(a["domain"] for a, d in planner_target)
        for domain, count in domains.most_common():
            print(f"    {domain}: {count} ({count/len(planner_target)*100:.0f}%)")

        lengths = [a["num_steps"] for a, d in planner_target]
        print(f"    Avg length: {np.mean(lengths):.1f} steps")
        print(f"    Avg progress improvement: {np.mean([d['progress_rate'] - a['b_progress_rate'] for a, d in planner_target])*100:.1f}pp")

    # Also analyze: what type of error caused the remaining failures?
    print(f"\n  First-error type in Planner target trajectories:")
    planner_error_types = Counter()
    for a, d in planner_target:
        classification = classify_first_error(d.get("step_results", []), key_prefix="")
        if classification:
            planner_error_types[classification[0]] += 1
        else:
            planner_error_types["unclassified"] += 1

    for etype, count in planner_error_types.most_common():
        print(f"    {etype}: {count} ({count/len(planner_target)*100:.0f}%)")

    return {
        "observer_solved": len(observer_solved),
        "observer_partial": len(observer_partial),
        "observer_noop": len(observer_noop),
        "observer_hurt": len(observer_hurt),
        "observer_regression": len(observer_regression),
        "planner_primary_gain": planner_primary_gain,
        "planner_total_gain": planner_total_gain,
    }


def main():
    print("Loading data...")
    eval_a = load_eval_a()
    eval_d1 = load_eval_d1()
    print(f"  Eval A: {len(eval_a)} trajectories")
    print(f"  Eval D1: {len(eval_d1)} trajectories")

    # Run all analyses
    d6_types, d6_domain, d6_steps = eval_d6(eval_a)
    d7_data = eval_d7(eval_a)
    d4_data = eval_d4(eval_a, eval_d1)

    # Save combined summary
    output_dir = PROJECT_ROOT / "outputs" / "eval_d4_d6_d7"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "d6_failure_types": dict(d6_types),
        "d7_length_failure": d7_data,
        "d4_planner_ceiling": d4_data,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n\nAll results saved to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
