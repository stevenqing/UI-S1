#!/usr/bin/env python3
"""
Exp D0: Observer Ceiling Analysis

Offline analysis of trajectory failure patterns to estimate the potential
value of adding a structured Observer agent to the pipeline.

Questions:
1. How many failures are "state confusion" (func correct, coord wrong)?
2. What's the pattern of correct→incorrect transitions?
3. Do failures show repeated actions or circular behavior?
4. How does trajectory length affect failure type?
5. What fraction of failures could an Observer detect in principle?

Data source: outputs/eval_a/trajectory_results.jsonl
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_trajectory_results():
    path = PROJECT_ROOT / "outputs" / "eval_a" / "trajectory_results.jsonl"
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def analyze_state_confusion(trajectories):
    """Analyze failures where model knows WHAT to do but not WHERE."""
    print("=" * 70)
    print("  D0.1: State Confusion Analysis (func_match=T, coord_match=F)")
    print("=" * 70)

    # For condition B (V2+V3), categorize each error step
    error_types = Counter()
    error_by_step = defaultdict(Counter)

    for traj in trajectories:
        for step in traj["step_results"]:
            if step["b_success"]:
                continue

            # Categorize the error
            fm = step["b_func_match"]
            cm = step["b_coord_match"]
            sm = step["b_status_match"]

            if fm and not cm:
                etype = "state_confusion"  # right action, wrong location
            elif not fm and cm:
                etype = "wrong_action"     # wrong action type, right place
            elif not fm and not cm:
                etype = "total_mismatch"   # both wrong
            elif fm and cm and not sm:
                etype = "status_error"     # right action and coord, wrong status
            else:
                etype = "other"

            error_types[etype] += 1
            error_by_step[step["step_num"]][etype] += 1

    total_errors = sum(error_types.values())
    print(f"\n  Total error steps: {total_errors}")
    print(f"\n  Error Type Breakdown:")
    for etype in ["state_confusion", "wrong_action", "total_mismatch", "status_error", "other"]:
        cnt = error_types.get(etype, 0)
        desc = {
            "state_confusion": "Right action, wrong location (Observer target)",
            "wrong_action": "Wrong action type, right location",
            "total_mismatch": "Both action and location wrong",
            "status_error": "Right action+coord, wrong status",
            "other": "Other errors",
        }[etype]
        print(f"    {etype:<20s}: {cnt:>5d} ({cnt/total_errors:.1%})  — {desc}")

    # State confusion is the primary Observer target
    sc = error_types.get("state_confusion", 0)
    print(f"\n  → {sc} state confusion errors are the primary Observer target")
    print(f"  → These represent {sc/total_errors:.1%} of all errors")

    # State confusion by step position
    print(f"\n  State confusion rate by step:")
    print(f"    {'Step':>4s} {'Errors':>7s} {'State Conf':>11s} {'Rate':>7s}")
    for step_num in sorted(error_by_step.keys())[:12]:
        total = sum(error_by_step[step_num].values())
        sc_count = error_by_step[step_num].get("state_confusion", 0)
        rate = sc_count / total if total > 0 else 0
        print(f"    {step_num:>4d} {total:>7d} {sc_count:>11d} {rate:>7.1%}")

    return error_types


def analyze_transition_patterns(trajectories):
    """Analyze correct→incorrect transition patterns in multi-step trajs."""
    print(f"\n{'=' * 70}")
    print("  D0.2: Correct→Incorrect Transition Analysis")
    print("=" * 70)

    # For trajectories with >1 evaluated step, analyze transition patterns
    multi_step = [t for t in trajectories
                  if len(t["step_results"]) > 1 and not t["b_trajectory_success"]]

    print(f"\n  Multi-step failed trajectories: {len(multi_step)}")

    # Categorize transition patterns
    patterns = Counter()
    for traj in multi_step:
        steps = traj["step_results"]
        pattern = "".join("1" if s["b_success"] else "0" for s in steps)
        # Simplify long patterns
        if len(pattern) > 5:
            pattern = pattern[:5] + f"...(len={len(pattern)})"
        patterns[pattern] += 1

    print(f"\n  Most common step patterns (1=correct, 0=wrong):")
    for pattern, cnt in patterns.most_common(15):
        print(f"    {pattern:<25s} {cnt:>5d}")

    # Specifically: how many start correct then fail?
    starts_correct = 0
    stays_wrong = 0
    alternating = 0

    for traj in multi_step:
        steps = traj["step_results"]
        first_correct = steps[0]["b_success"]
        second_correct = steps[1]["b_success"] if len(steps) > 1 else None

        if first_correct and second_correct is False:
            starts_correct += 1  # Correct then fail — context loss
        elif not first_correct and second_correct is False:
            stays_wrong += 1    # Wrong from start
        elif first_correct and second_correct:
            alternating += 1    # Multiple correct then fail later

    print(f"\n  Transition classification:")
    print(f"    Starts correct, then fails: {starts_correct} ({starts_correct/len(multi_step):.1%})")
    print(f"    Wrong from the start:       {stays_wrong} ({stays_wrong/len(multi_step):.1%})")
    print(f"    Multiple correct then fail: {alternating} ({alternating/len(multi_step):.1%})")

    # For trajectories that start correct: at which step do they fail?
    fail_step_after_correct = []
    for traj in multi_step:
        steps = traj["step_results"]
        saw_correct = False
        for s in steps:
            if s["b_success"]:
                saw_correct = True
            elif saw_correct:
                fail_step_after_correct.append(s["step_num"])
                break

    if fail_step_after_correct:
        print(f"\n  Among 'correct then fail' trajectories:")
        print(f"    First failure step distribution:")
        counter = Counter(fail_step_after_correct)
        for step in sorted(counter.keys())[:10]:
            cnt = counter[step]
            print(f"      Step {step}: {cnt} ({cnt/len(fail_step_after_correct):.1%})")


def analyze_repeated_actions(trajectories):
    """Detect repeated/circular action patterns suggesting agent is stuck."""
    print(f"\n{'=' * 70}")
    print("  D0.3: Repeated Action Detection (Agent Stuck)")
    print("=" * 70)

    # Look for same function being predicted at consecutive steps
    has_repeat = 0
    total_multi = 0
    repeat_lengths = []

    for traj in trajectories:
        if len(traj["step_results"]) < 2:
            continue
        total_multi += 1

        steps = traj["step_results"]
        max_repeat = 1
        current_repeat = 1
        for i in range(1, len(steps)):
            if steps[i]["pred_function"] == steps[i-1]["pred_function"]:
                current_repeat += 1
                max_repeat = max(max_repeat, current_repeat)
            else:
                current_repeat = 1

        if max_repeat >= 2:
            has_repeat += 1
            repeat_lengths.append(max_repeat)

    print(f"\n  Multi-step trajectories: {total_multi}")
    print(f"  With ≥2 consecutive same function: {has_repeat} ({has_repeat/total_multi:.1%})")

    if repeat_lengths:
        print(f"  Max repeat length: mean={np.mean(repeat_lengths):.1f}, "
              f"max={max(repeat_lengths)}")
        counter = Counter(repeat_lengths)
        for length in sorted(counter.keys()):
            print(f"    {length}× repeat: {counter[length]} trajectories")


def analyze_length_failure_correlation(trajectories):
    """Analyze how trajectory length affects failure type."""
    print(f"\n{'=' * 70}")
    print("  D0.4: Trajectory Length vs Failure Mode")
    print("=" * 70)

    length_buckets = [
        ("1 step", 1, 1),
        ("2-3 steps", 2, 3),
        ("4-5 steps", 4, 5),
        ("6-10 steps", 6, 10),
        ("11-15 steps", 11, 15),
        ("16+ steps", 16, 100),
    ]

    print(f"\n  {'Bucket':<12s} {'N':>5s} {'B_TSR':>7s} {'Avg Prog':>9s} "
          f"{'Step1 Err':>9s} {'Late Err':>9s} {'SC Rate':>8s}")
    print(f"  {'-'*12} {'-'*5} {'-'*7} {'-'*9} {'-'*9} {'-'*9} {'-'*8}")

    for label, lo, hi in length_buckets:
        subset = [t for t in trajectories if lo <= t["num_steps"] <= hi]
        if not subset:
            continue

        n = len(subset)
        tsr = np.mean([t["b_trajectory_success"] for t in subset])
        avg_prog = np.mean([t["b_progress_rate"] for t in subset])

        # Step 1 error rate
        step1_err = 0
        late_err = 0
        state_confusion = 0
        total_errors = 0

        for t in subset:
            if t["b_trajectory_success"]:
                continue
            steps = t["step_results"]
            for s in steps:
                if not s["b_success"]:
                    total_errors += 1
                    if s["step_num"] == 1:
                        step1_err += 1
                    else:
                        late_err += 1
                    if s["b_func_match"] and not s["b_coord_match"]:
                        state_confusion += 1

        s1_rate = step1_err / total_errors if total_errors > 0 else 0
        late_rate = late_err / total_errors if total_errors > 0 else 0
        sc_rate = state_confusion / total_errors if total_errors > 0 else 0

        print(f"  {label:<12s} {n:>5d} {tsr:>7.1%} {avg_prog:>9.2f} "
              f"{s1_rate:>9.1%} {late_rate:>9.1%} {sc_rate:>8.1%}")


def estimate_observer_ceiling(trajectories):
    """Estimate the theoretical ceiling for Observer improvement."""
    print(f"\n{'=' * 70}")
    print("  D0.5: Observer Improvement Ceiling Estimate")
    print("=" * 70)

    n_total = len(trajectories)
    current_tsr = sum(1 for t in trajectories if t["b_trajectory_success"]) / n_total
    failed = [t for t in trajectories if not t["b_trajectory_success"]]

    # Category 1: First error at step 1 — Observer can't help (no prior state)
    step1_failures = 0
    # Category 2: First error at step 2+ with state confusion — Observer can help
    late_state_confusion = 0
    # Category 3: First error at step 2+ but wrong action — Observer might help
    late_wrong_action = 0
    # Category 4: First error at step 2+ but total mismatch — Observer unlikely to help
    late_total_mismatch = 0

    for traj in failed:
        steps = traj["step_results"]
        first_error = None
        for s in steps:
            if not s["b_success"]:
                first_error = s
                break

        if first_error is None:
            continue

        if first_error["step_num"] == 1:
            step1_failures += 1
        else:
            fm = first_error["b_func_match"]
            cm = first_error["b_coord_match"]
            if fm and not cm:
                late_state_confusion += 1
            elif not fm:
                late_wrong_action += 1

    n_failed = len(failed)
    print(f"\n  Current TSR: {current_tsr:.1%} ({n_total - n_failed}/{n_total})")
    print(f"  Failed trajectories: {n_failed}")
    print(f"\n  Failure decomposition:")
    print(f"    Step-1 failures (Observer can't help):     {step1_failures} ({step1_failures/n_failed:.1%})")
    print(f"    Late state confusion (Observer PRIMARY):   {late_state_confusion} ({late_state_confusion/n_failed:.1%})")
    print(f"    Late wrong action (Observer SECONDARY):    {late_wrong_action} ({late_wrong_action/n_failed:.1%})")

    # Optimistic ceiling: Observer prevents ALL late state confusion errors
    # and the trajectory succeeds if no more errors after that
    optimistic = n_total - n_failed + late_state_confusion
    # Conservative ceiling: Observer prevents 50% of late state confusion
    conservative = n_total - n_failed + late_state_confusion // 2

    print(f"\n  Observer ceiling estimates:")
    print(f"    Current TSR:         {current_tsr:.1%}")
    print(f"    Conservative (+50% SC fix): {conservative/n_total:.1%} (+{(conservative/n_total - current_tsr):.1%})")
    print(f"    Optimistic (+100% SC fix):  {optimistic/n_total:.1%} (+{(optimistic/n_total - current_tsr):.1%})")

    # More realistic: Observer + Planner could help with late wrong actions too
    full_late = late_state_confusion + late_wrong_action
    full_ceiling = n_total - n_failed + full_late
    print(f"    Full framework ceiling:     {full_ceiling/n_total:.1%} (+{(full_ceiling/n_total - current_tsr):.1%})")
    print(f"    (assumes Observer+Planner fixes all late errors)")

    # Compare with perfect verification ceiling
    near_miss = sum(1 for t in failed if t["b_progress_rate"] >= 0.5)
    verify_ceiling = (n_total - n_failed + near_miss) / n_total
    print(f"\n    For reference:")
    print(f"    Perfect verify+recovery:    {verify_ceiling:.1%}")
    print(f"    Observer complements verification — they target different failure modes")

    return {
        "current_tsr": current_tsr,
        "step1_failures": step1_failures,
        "late_state_confusion": late_state_confusion,
        "late_wrong_action": late_wrong_action,
        "conservative_ceiling": conservative / n_total,
        "optimistic_ceiling": optimistic / n_total,
        "full_framework_ceiling": full_ceiling / n_total,
    }


def analyze_progress_vs_state(trajectories):
    """Analyze how progress rate relates to state tracking quality."""
    print(f"\n{'=' * 70}")
    print("  D0.6: Progress Rate & State Tracking Quality")
    print("=" * 70)

    # High-progress failures are the most interesting for Observer
    # These trajectories got many steps right but still failed
    failed = [t for t in trajectories if not t["b_trajectory_success"]]

    progress_bins = [
        ("0%", 0.0, 0.01),
        ("1-25%", 0.01, 0.25),
        ("25-50%", 0.25, 0.50),
        ("50-75%", 0.50, 0.75),
        ("75-99%", 0.75, 1.0),
    ]

    print(f"\n  {'Progress':>10s} {'N':>5s} {'Avg Steps':>10s} {'Avg Eval':>9s} "
          f"{'SC at fail':>10s} {'Avg Length':>10s}")
    print(f"  {'-'*10} {'-'*5} {'-'*10} {'-'*9} {'-'*10} {'-'*10}")

    for label, lo, hi in progress_bins:
        subset = [t for t in failed if lo <= t["b_progress_rate"] < hi]
        if not subset:
            continue

        n = len(subset)
        avg_steps = np.mean([t["num_steps"] for t in subset])
        avg_eval = np.mean([t["num_evaluated"] for t in subset])

        # State confusion at first failure point
        sc_count = 0
        for t in subset:
            for s in t["step_results"]:
                if not s["b_success"]:
                    if s["b_func_match"] and not s["b_coord_match"]:
                        sc_count += 1
                    break

        sc_rate = sc_count / n if n > 0 else 0

        print(f"  {label:>10s} {n:>5d} {avg_steps:>10.1f} {avg_eval:>9.1f} "
              f"{sc_rate:>10.1%} {avg_steps:>10.1f}")

    # Near-miss trajectories (progress ≥ 75%) — most valuable for Observer
    near_miss = [t for t in failed if t["b_progress_rate"] >= 0.75]
    print(f"\n  Near-miss (progress ≥ 75%): {len(near_miss)} trajectories")
    print(f"  → These are trajectories that were almost successful")
    print(f"  → Observer could detect the critical failure point and trigger recovery")

    # Show some domain breakdown of near-misses
    nm_by_domain = Counter(t["domain"] for t in near_miss)
    for domain, cnt in nm_by_domain.most_common():
        print(f"    {domain}: {cnt}")


def main():
    print("Loading trajectory results...")
    trajectories = load_trajectory_results()
    print(f"Loaded {len(trajectories)} trajectories\n")

    error_types = analyze_state_confusion(trajectories)
    analyze_transition_patterns(trajectories)
    analyze_repeated_actions(trajectories)
    analyze_length_failure_correlation(trajectories)
    ceiling = estimate_observer_ceiling(trajectories)
    analyze_progress_vs_state(trajectories)

    # Save summary
    output_dir = PROJECT_ROOT / "outputs" / "eval_d0"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_trajectories": len(trajectories),
        "error_types": dict(error_types),
        **ceiling,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
