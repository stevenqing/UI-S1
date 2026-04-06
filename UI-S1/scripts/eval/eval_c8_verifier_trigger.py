#!/usr/bin/env python3
"""
Exp C8: Verifier Trigger Rate & Early-Stop Analysis

Using Eval A trajectory data (AR evaluation with V2-only and V2+V3):
1. Silent failure rate: steps that "succeed" but lead to trajectory failure
2. Ideal early-stop TSR: if we could detect errors and stop, what's the ceiling?
3. Agreement-gated verification: use V3 agreement as proxy for verification trigger

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


def load_k10_agreement():
    """Load agreement rates from K=10 grounding eval."""
    path = PROJECT_ROOT / "outputs" / "exp1_1" / "results_K10.jsonl"
    agreement = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            agreement[d["sample_id"]] = d.get("agreement_rate", 0)
    return agreement


def analyze_silent_failures(trajectories):
    """Analyze steps that appear correct but lead to trajectory failure."""
    print("=" * 70)
    print("  C8.1: Silent Failure Analysis")
    print("=" * 70)

    # For condition B (V2+V3), analyze step-level vs trajectory-level success
    total_steps = 0
    step_correct_traj_fail = 0  # silent failure: step ok but traj fails
    step_wrong_traj_fail = 0    # detected failure: step wrong, traj fails
    step_correct_traj_ok = 0    # true positive: step ok, traj ok
    step_wrong_traj_ok = 0      # recovery: step wrong but traj still ok

    by_step = defaultdict(lambda: {"total": 0, "silent_fail": 0, "detected_fail": 0})

    for traj in trajectories:
        traj_success = traj["b_trajectory_success"]
        for step in traj["step_results"]:
            total_steps += 1
            step_ok = step["b_success"]
            step_num = step["step_num"]

            by_step[step_num]["total"] += 1

            if step_ok and traj_success:
                step_correct_traj_ok += 1
            elif step_ok and not traj_success:
                step_correct_traj_fail += 1
                by_step[step_num]["silent_fail"] += 1
            elif not step_ok and not traj_success:
                step_wrong_traj_fail += 1
                by_step[step_num]["detected_fail"] += 1
            else:  # wrong step, ok traj
                step_wrong_traj_ok += 1

    print(f"\n  Total steps evaluated: {total_steps}")
    print(f"  Step correct + Traj ok:     {step_correct_traj_ok:>6d} ({step_correct_traj_ok/total_steps:.1%})")
    print(f"  Step correct + Traj FAIL:   {step_correct_traj_fail:>6d} ({step_correct_traj_fail/total_steps:.1%})  [silent failure]")
    print(f"  Step WRONG + Traj FAIL:     {step_wrong_traj_fail:>6d} ({step_wrong_traj_fail/total_steps:.1%})  [detectable]")
    print(f"  Step WRONG + Traj ok:       {step_wrong_traj_ok:>6d} ({step_wrong_traj_ok/total_steps:.1%})  [recovery]")

    # Silent failure rate among failed trajectories
    failed_trajs = [t for t in trajectories if not t["b_trajectory_success"]]
    n_failed = len(failed_trajs)
    n_have_silent = 0
    for t in failed_trajs:
        # Does this failed traj have any step marked correct?
        if any(s["b_success"] for s in t["step_results"]):
            n_have_silent += 1

    print(f"\n  Failed trajectories: {n_failed}")
    print(f"  Failed trajs with >= 1 'correct' step: {n_have_silent} ({n_have_silent/n_failed:.1%})")
    print(f"  → These are trajectories where a verifier would NOT trigger")

    # By step position
    print(f"\n  Silent failure rate by step position:")
    print(f"    {'Step':>4s} {'Total':>6s} {'Silent':>7s} {'Rate':>7s}")
    for step_num in sorted(by_step.keys())[:15]:
        d = by_step[step_num]
        rate = d["silent_fail"] / d["total"] if d["total"] > 0 else 0
        print(f"    {step_num:>4d} {d['total']:>6d} {d['silent_fail']:>7d} {rate:>7.1%}")


def analyze_early_stop(trajectories):
    """Simulate ideal early-stop: stop at first wrong step."""
    print(f"\n{'=' * 70}")
    print("  C8.2: Ideal Early-Stop Analysis")
    print("=" * 70)

    # For each trajectory, find the first step where prediction goes wrong
    # In AR mode, subsequent steps are affected by previous errors
    # If we could detect the first error and stop, how many more trajs would "succeed"?

    n_total = len(trajectories)

    for condition in ["a", "b"]:
        label = "V2-only" if condition == "a" else "V2+V3"
        success_key = f"{condition}_trajectory_success"

        n_success = sum(1 for t in trajectories if t[success_key])
        n_fail = n_total - n_success

        # Among failed trajectories, where does the first error occur?
        first_error_steps = []
        could_save = 0  # trajectories where first error is NOT step 1

        for t in trajectories:
            if t[success_key]:
                continue
            success_key_step = f"{condition}_success"
            steps = t["step_results"]
            first_error = None
            for s in steps:
                if not s[success_key_step]:
                    first_error = s["step_num"]
                    break
            if first_error is not None:
                first_error_steps.append(first_error)
                if first_error > 1:
                    could_save += 1

        print(f"\n  --- {label} ---")
        print(f"  Current TSR: {n_success}/{n_total} = {n_success/n_total:.1%}")
        print(f"  Failed trajectories: {n_fail}")

        if first_error_steps:
            print(f"  First error step distribution:")
            counter = Counter(first_error_steps)
            for step in sorted(counter.keys())[:10]:
                cnt = counter[step]
                print(f"    Step {step}: {cnt} ({cnt/n_fail:.1%})")

            print(f"\n  Trajs where first error is after step 1: {could_save} ({could_save/n_fail:.1%})")
            print(f"  → With perfect verifier + retry on those: potential recovery target")

        # Simulate: if we detect error at step N and retry with different sample
        # How many could potentially succeed?
        # Approximation: if first error is at step N, and we have K=10 candidates,
        # the chance of at least one being correct is ~best_of_k rate for that step
        print(f"\n  Step-1 error rate: {counter.get(1, 0)/n_fail:.1%} of failures")
        print(f"  → These are the hardest cases (wrong from the start)")


def analyze_agreement_gated_verification(trajectories, agreement_map):
    """Use V3 agreement rate as proxy for verification trigger."""
    print(f"\n{'=' * 70}")
    print("  C8.3: Agreement-Gated Verification Simulation")
    print("=" * 70)

    # For each step in each trajectory, check if V3 agreement is available
    # Simulate: if agreement < threshold, mark step as "needs verification"

    # Map sample_id from trajectory to K=10 data
    # Trajectory sample_id: "excel_in_app_excel_1_116_1" → need to match to K=10 IDs

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    print(f"\n  {'Threshold':>10s} {'Trigger%':>9s} {'Precision':>10s} {'Recall':>7s} {'F1':>5s}")
    print(f"  {'-'*10} {'-'*9} {'-'*10} {'-'*7} {'-'*5}")

    for thresh in thresholds:
        tp = fp = tn = fn = 0
        matched = 0

        for traj in trajectories:
            for step in traj["step_results"]:
                sid = step["sample_id"]
                agree = agreement_map.get(sid)
                if agree is None:
                    continue

                matched += 1
                step_wrong = not step["b_success"]
                trigger = agree < thresh

                if trigger and step_wrong:
                    tp += 1
                elif trigger and not step_wrong:
                    fp += 1
                elif not trigger and step_wrong:
                    fn += 1
                else:
                    tn += 1

        total = tp + fp + tn + fn
        if total == 0:
            continue

        trigger_rate = (tp + fp) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"  <{thresh:<9.1f} {trigger_rate:>9.1%} {precision:>10.1%} {recall:>7.1%} {f1:>5.2f}")

    print(f"\n  Note: matched {matched} of {sum(len(t['step_results']) for t in trajectories)} steps to K=10 data")
    print(f"  Agreement comes from K=10 grounding eval; in AR mode, only step 1 uses GT screenshot")
    print(f"  → Real system would need to compute agreement at each AR step (more expensive)")


def analyze_verification_value(trajectories):
    """How much would perfect step-level verification help?"""
    print(f"\n{'=' * 70}")
    print("  C8.4: Perfect Verification Value Analysis")
    print("=" * 70)

    n_total = len(trajectories)

    for condition in ["a", "b"]:
        label = "V2-only" if condition == "a" else "V2+V3"
        success_key = f"{condition}_trajectory_success"
        step_key = f"{condition}_success"

        current_tsr = sum(1 for t in trajectories if t[success_key]) / n_total

        # Scenario 1: Perfect verifier + stop (don't execute wrong steps)
        # This doesn't help TSR directly but saves compute

        # Scenario 2: Perfect verifier + oracle retry
        # If we detect wrong step and retry with correct action, all trajs with
        # at least partial correct steps could succeed

        # For now: count how many failed trajs have >50% steps correct
        failed = [t for t in trajectories if not t[success_key]]
        high_partial = 0
        for t in failed:
            correct_steps = sum(1 for s in t["step_results"] if s[step_key])
            total_steps = len(t["step_results"])
            if total_steps > 0 and correct_steps / total_steps >= 0.5:
                high_partial += 1

        # Progress rate distribution for failed trajectories
        progress_key = f"{condition}_progress_rate"
        failed_progress = [t[progress_key] for t in failed]

        print(f"\n  --- {label} ---")
        print(f"  Current TSR: {current_tsr:.1%}")
        print(f"  Failed trajs: {len(failed)}")
        print(f"  Failed with >=50% steps correct: {high_partial} ({high_partial/len(failed):.1%})")
        print(f"  Failed progress distribution:")

        bins = [(0, 0, "0%"), (0.01, 0.25, "1-25%"), (0.25, 0.50, "25-50%"),
                (0.50, 0.75, "50-75%"), (0.75, 0.99, "75-99%"), (0.99, 1.01, "~100%")]
        for lo, hi, label_bin in bins:
            cnt = sum(1 for p in failed_progress if lo <= p < hi)
            print(f"    {label_bin:>7s}: {cnt:>5d} ({cnt/len(failed):.1%})")

        # Near-miss analysis: failed trajs with progress >= 0.5
        near_miss = sum(1 for p in failed_progress if p >= 0.5)
        print(f"\n  Near-miss (progress >= 50%): {near_miss} ({near_miss/len(failed):.1%})")
        print(f"  → Perfect verification + retry could recover up to {near_miss} trajectories")
        print(f"  → Ceiling TSR with recovery: {(sum(1 for t in trajectories if t[success_key]) + near_miss)/n_total:.1%}")


def main():
    print("Loading trajectory results...")
    trajectories = load_trajectory_results()
    print(f"Loaded {len(trajectories)} trajectories\n")

    print("Loading K=10 agreement data...")
    agreement = load_k10_agreement()
    print(f"Loaded agreement for {len(agreement)} samples\n")

    analyze_silent_failures(trajectories)
    analyze_early_stop(trajectories)
    analyze_agreement_gated_verification(trajectories, agreement)
    analyze_verification_value(trajectories)

    # Save summary
    output_dir = PROJECT_ROOT / "outputs" / "eval_c8"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(trajectories)
    summary = {
        "n_trajectories": n_total,
        "a_tsr": sum(1 for t in trajectories if t["a_trajectory_success"]) / n_total,
        "b_tsr": sum(1 for t in trajectories if t["b_trajectory_success"]) / n_total,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
