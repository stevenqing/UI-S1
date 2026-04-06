#!/usr/bin/env python3
"""
Cascade analysis for SFT v2 no-stop evaluation results.
Computes step-0 failure rate, survival probabilities, mean cascade depth,
post-error accuracy, and error type breakdown with coordinate distance distributions.
"""

import json
import math
import sys
from collections import defaultdict

RESULTS_FILE = (
    "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/results/"
    "sft_v2/gui360/nostop_20260320_053216/"
    "ar_evaluation_results_20260320_055609.json"
)

# Base model reference numbers
BASE = {
    "TSR": 1.58,
    "step_acc": 9.26,
    "scattered_progress": 12.02,
    "step0_failure": 78.72,
    "post_error_acc": 5.07,
    "survival": {0: 0.227, 1: 0.172, 2: 0.160, 3: 0.099, 4: 0.099, 5: 0.060},
}


def get_coordinate(args):
    """Extract primary coordinate from predicted/ground_truth args."""
    if not isinstance(args, dict):
        return None
    coord = args.get("coordinate")
    if coord and isinstance(coord, (list, tuple)) and len(coord) >= 2:
        try:
            if coord[0] is not None and coord[1] is not None:
                return (float(coord[0]), float(coord[1]))
        except (ValueError, TypeError):
            pass
    # For drag, use start_coordinate
    coord = args.get("start_coordinate")
    if coord and isinstance(coord, (list, tuple)) and len(coord) >= 2:
        try:
            if coord[0] is not None and coord[1] is not None:
                return (float(coord[0]), float(coord[1]))
        except (ValueError, TypeError):
            pass
    return None


def coord_distance(c1, c2):
    """Euclidean distance between two (x, y) coordinates."""
    if c1 is None or c2 is None:
        return None
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def main():
    print("Loading evaluation results...")
    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    trajectories = data["trajectory_results"]
    stats = data.get("statistics", {})
    n_traj = len(trajectories)

    print(f"Loaded {n_traj} trajectories, {stats.get('total_steps', '?')} total steps")
    print(f"Overall TSR: {stats.get('trajectory_success_rate', 0) * 100:.2f}%")
    print(f"Overall Step Acc: {stats.get('step_success_rate', 0) * 100:.2f}%")
    print(f"Overall Scattered Progress: {stats.get('avg_scattered_progress_rate', 0) * 100:.2f}%")
    print()

    # =========================================================================
    # 1. Step-0 Failure Rate
    # =========================================================================
    step0_fail = 0
    step0_total = 0
    for t in trajectories:
        steps = t.get("step_results", [])
        if steps:
            step0_total += 1
            if not steps[0]["success"]:
                step0_fail += 1

    step0_failure_rate = step0_fail / step0_total * 100 if step0_total else 0
    print("=" * 70)
    print("1. STEP-0 FAILURE RATE")
    print("=" * 70)
    print(f"   Step-0 failures: {step0_fail} / {step0_total}")
    print(f"   Step-0 failure rate: {step0_failure_rate:.2f}%")
    print(f"   [Base model: {BASE['step0_failure']:.2f}%]")
    print(f"   Improvement: {BASE['step0_failure'] - step0_failure_rate:+.2f} pp")
    print()

    # =========================================================================
    # 2. Survival Probability P(step k+1 correct | step k correct)
    # =========================================================================
    # For each k, count trajectories where step k is correct, and among those,
    # how many also have step k+1 correct.
    max_k = 10  # compute up to k=10
    correct_at_k = defaultdict(int)      # count of trajs where step k is correct
    correct_at_k_and_k1 = defaultdict(int)  # count where both step k and k+1 correct

    for t in trajectories:
        steps = t.get("step_results", [])
        for k in range(min(len(steps), max_k + 1)):
            if steps[k]["success"]:
                correct_at_k[k] += 1
                if k + 1 < len(steps) and steps[k + 1]["success"]:
                    correct_at_k_and_k1[k] += 1

    print("=" * 70)
    print("2. SURVIVAL PROBABILITY P(step k+1 correct | step k correct)")
    print("=" * 70)
    print(f"   {'k':>3s}  {'P(k+1|k)':>10s}  {'correct@k':>10s}  {'correct@k+1':>12s}  {'Base P(k)':>10s}  {'Delta':>8s}")
    print(f"   {'---':>3s}  {'--------':>10s}  {'---------':>10s}  {'-----------':>12s}  {'---------':>10s}  {'-----':>8s}")
    for k in range(min(6, max_k + 1)):
        if correct_at_k[k] > 0:
            surv = correct_at_k_and_k1[k] / correct_at_k[k]
        else:
            surv = 0.0
        base_surv = BASE["survival"].get(k, float("nan"))
        delta = surv - base_surv if not math.isnan(base_surv) else float("nan")
        print(
            f"   {k:3d}  {surv:10.4f}  {correct_at_k[k]:10d}  "
            f"{correct_at_k_and_k1[k]:12d}  {base_surv:10.3f}  {delta:+8.3f}"
        )
    print()

    # =========================================================================
    # 3. Mean Cascade Depth
    # =========================================================================
    # For each trajectory:
    #   - "correct_run": number of consecutive correct steps from the start
    #   - "error_run": number of consecutive wrong steps after the first error
    correct_runs = []
    error_runs = []
    traj_lengths = []

    for t in trajectories:
        steps = t.get("step_results", [])
        n = len(steps)
        traj_lengths.append(n)

        # Correct run from start
        cr = 0
        for s in steps:
            if s["success"]:
                cr += 1
            else:
                break
        else:
            # all steps correct
            cr = n
        correct_runs.append(cr)

        # Error run after first error
        first_err_idx = None
        for i, s in enumerate(steps):
            if not s["success"]:
                first_err_idx = i
                break

        if first_err_idx is not None:
            er = 0
            for s in steps[first_err_idx:]:
                if not s["success"]:
                    er += 1
                else:
                    break
            error_runs.append(er)

    mean_correct_run = sum(correct_runs) / len(correct_runs) if correct_runs else 0
    mean_error_run = sum(error_runs) / len(error_runs) if error_runs else 0
    mean_traj_len = sum(traj_lengths) / len(traj_lengths) if traj_lengths else 0

    print("=" * 70)
    print("3. MEAN CASCADE DEPTH")
    print("=" * 70)
    print(f"   Mean trajectory length: {mean_traj_len:.2f} steps")
    print(f"   Mean correct run from start: {mean_correct_run:.3f} steps")
    print(f"   Mean error run after first error: {mean_error_run:.3f} steps")
    print(f"   Trajectories with at least one error: {len(error_runs)} / {n_traj}")

    # Distribution of correct run lengths
    cr_dist = defaultdict(int)
    for cr in correct_runs:
        cr_dist[cr] += 1
    print(f"\n   Distribution of initial correct-run length:")
    for length in sorted(cr_dist.keys()):
        pct = cr_dist[length] / n_traj * 100
        bar = "#" * int(pct / 2)
        print(f"     {length:3d} steps: {cr_dist[length]:5d} ({pct:5.1f}%) {bar}")
        if length >= 15:
            remaining = sum(v for k, v in cr_dist.items() if k > 15)
            if remaining > 0:
                print(f"     >15 steps: {remaining:5d} ({remaining / n_traj * 100:5.1f}%)")
            break
    print()

    # =========================================================================
    # 4. Post-Error Accuracy
    # =========================================================================
    post_error_correct = 0
    post_error_total = 0

    for t in trajectories:
        steps = t.get("step_results", [])
        first_err_idx = None
        for i, s in enumerate(steps):
            if not s["success"]:
                first_err_idx = i
                break
        if first_err_idx is not None and first_err_idx + 1 < len(steps):
            for s in steps[first_err_idx + 1:]:
                post_error_total += 1
                if s["success"]:
                    post_error_correct += 1

    post_error_acc = post_error_correct / post_error_total * 100 if post_error_total else 0
    print("=" * 70)
    print("4. POST-ERROR ACCURACY")
    print("=" * 70)
    print(f"   Post-error steps: {post_error_total}")
    print(f"   Post-error correct: {post_error_correct}")
    print(f"   Post-error accuracy: {post_error_acc:.2f}%")
    print(f"   [Base model: {BASE['post_error_acc']:.2f}%]")
    print(f"   Improvement: {post_error_acc - BASE['post_error_acc']:+.2f} pp")
    print()

    # =========================================================================
    # 5. Error Type Breakdown
    # =========================================================================
    total_failed = 0
    stuck_repeating = 0
    type_mismatch = 0
    coord_error = 0
    near_miss = 0
    other_error = 0

    coord_distances = []  # distances for coord_error cases
    near_miss_distances = []

    # Distance bins
    dist_bins = {"<25": 0, "25-50": 0, "50-100": 0, "100-200": 0, "200-500": 0, ">500": 0}

    for t in trajectories:
        steps = t.get("step_results", [])
        for i, s in enumerate(steps):
            if s["success"]:
                continue

            total_failed += 1
            pred_fn = s.get("predicted_function", "")
            gt_fn = s.get("ground_truth_function", "")
            pred_args = s.get("predicted_args", {})
            gt_args = s.get("ground_truth_args", {})

            # Check stuck/repeating: same coordinate AND same action type as previous step
            is_stuck = False
            if i > 0:
                prev = steps[i - 1]
                prev_fn = prev.get("predicted_function", "")
                prev_args = prev.get("predicted_args", {})
                prev_coord = get_coordinate(prev_args)
                curr_coord = get_coordinate(pred_args)
                if (
                    pred_fn == prev_fn
                    and curr_coord is not None
                    and prev_coord is not None
                    and curr_coord == prev_coord
                ):
                    is_stuck = True
                    stuck_repeating += 1

            if is_stuck:
                continue  # already classified

            # Check type mismatch
            if pred_fn != gt_fn:
                type_mismatch += 1
                continue

            # Type matches - check coordinates
            pred_coord = get_coordinate(pred_args)
            gt_coord = get_coordinate(gt_args)

            if pred_coord is not None and gt_coord is not None:
                dist = coord_distance(pred_coord, gt_coord)
                coord_distances.append(dist)

                if dist < 50:
                    near_miss += 1
                    near_miss_distances.append(dist)
                else:
                    coord_error += 1

                # Bin the distance
                if dist < 25:
                    dist_bins["<25"] += 1
                elif dist < 50:
                    dist_bins["25-50"] += 1
                elif dist < 100:
                    dist_bins["50-100"] += 1
                elif dist < 200:
                    dist_bins["100-200"] += 1
                elif dist < 500:
                    dist_bins["200-500"] += 1
                else:
                    dist_bins[">500"] += 1
            else:
                # No coordinates to compare (e.g., text content mismatch for type action,
                # or complex args mismatch)
                other_error += 1

    print("=" * 70)
    print("5. ERROR TYPE BREAKDOWN")
    print("=" * 70)
    print(f"   Total failed steps: {total_failed}")
    print()
    print(f"   {'Category':<25s}  {'Count':>7s}  {'% of failed':>12s}  {'% of all steps':>15s}")
    print(f"   {'-' * 25}  {'-' * 7}  {'-' * 12}  {'-' * 15}")

    total_steps = stats.get("total_steps", 1)
    categories = [
        ("stuck/repeating", stuck_repeating),
        ("type mismatch", type_mismatch),
        ("coord error (>=50px)", coord_error),
        ("near miss (<50px)", near_miss),
        ("other (content etc.)", other_error),
    ]
    for name, count in categories:
        pct_fail = count / total_failed * 100 if total_failed else 0
        pct_all = count / total_steps * 100 if total_steps else 0
        print(f"   {name:<25s}  {count:7d}  {pct_fail:11.2f}%  {pct_all:14.2f}%")

    verify_sum = stuck_repeating + type_mismatch + coord_error + near_miss + other_error
    print(f"\n   Verification: {verify_sum} classified = {total_failed} total failed? {'YES' if verify_sum == total_failed else 'NO ('+str(total_failed - verify_sum)+' unclassified)'}")

    # Coordinate distance distribution for all coord-comparable errors
    print(f"\n   Coordinate distance distribution (type-matching failed steps with coordinates):")
    print(f"   Total coord-comparable errors: {len(coord_distances)}")
    print(f"   {'Bin':<12s}  {'Count':>7s}  {'%':>8s}  {'Cumulative %':>13s}")
    print(f"   {'-' * 12}  {'-' * 7}  {'-' * 8}  {'-' * 13}")
    cum = 0
    for bin_name in ["<25", "25-50", "50-100", "100-200", "200-500", ">500"]:
        cnt = dist_bins[bin_name]
        pct = cnt / len(coord_distances) * 100 if coord_distances else 0
        cum += pct
        bar = "#" * int(pct / 2)
        print(f"   {bin_name:<12s}  {cnt:7d}  {pct:7.2f}%  {cum:12.2f}%  {bar}")

    if coord_distances:
        coord_distances.sort()
        print(f"\n   Coord distance statistics:")
        print(f"     Mean: {sum(coord_distances) / len(coord_distances):.1f} px")
        print(f"     Median: {coord_distances[len(coord_distances) // 2]:.1f} px")
        print(f"     P25: {coord_distances[len(coord_distances) // 4]:.1f} px")
        print(f"     P75: {coord_distances[3 * len(coord_distances) // 4]:.1f} px")
        print(f"     P90: {coord_distances[int(0.9 * len(coord_distances))]:.1f} px")
        print(f"     P95: {coord_distances[int(0.95 * len(coord_distances))]:.1f} px")
        print(f"     Max: {coord_distances[-1]:.1f} px")
    print()

    # =========================================================================
    # 6. Comparison with Base Model
    # =========================================================================
    sft_tsr = stats.get("trajectory_success_rate", 0) * 100
    sft_step_acc = stats.get("step_success_rate", 0) * 100
    sft_scattered = stats.get("avg_scattered_progress_rate", 0) * 100

    print("=" * 70)
    print("6. COMPARISON WITH BASE MODEL (no-stop)")
    print("=" * 70)
    print(f"\n   {'Metric':<30s}  {'Base':>10s}  {'SFT v2':>10s}  {'Delta':>10s}  {'Ratio':>8s}")
    print(f"   {'-' * 30}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 8}")

    comparisons = [
        ("TSR (%)", BASE["TSR"], sft_tsr),
        ("Step Accuracy (%)", BASE["step_acc"], sft_step_acc),
        ("Scattered Progress (%)", BASE["scattered_progress"], sft_scattered),
        ("Step-0 Failure Rate (%)", BASE["step0_failure"], step0_failure_rate),
        ("Post-Error Accuracy (%)", BASE["post_error_acc"], post_error_acc),
    ]
    for name, base_val, sft_val in comparisons:
        delta = sft_val - base_val
        ratio = sft_val / base_val if base_val != 0 else float("inf")
        print(f"   {name:<30s}  {base_val:10.2f}  {sft_val:10.2f}  {delta:+10.2f}  {ratio:7.2f}x")

    print(f"\n   Survival Probabilities:")
    print(f"   {'k':>3s}  {'Base P(k+1|k)':>14s}  {'SFT v2 P(k+1|k)':>16s}  {'Delta':>8s}")
    print(f"   {'---':>3s}  {'-' * 14}  {'-' * 16}  {'-' * 8}")
    for k in range(6):
        base_surv = BASE["survival"].get(k, 0)
        if correct_at_k[k] > 0:
            sft_surv = correct_at_k_and_k1[k] / correct_at_k[k]
        else:
            sft_surv = 0
        delta = sft_surv - base_surv
        print(f"   {k:3d}  {base_surv:14.4f}  {sft_surv:16.4f}  {delta:+8.4f}")
    print()

    # =========================================================================
    # Additional: Per-domain breakdown
    # =========================================================================
    print("=" * 70)
    print("BONUS: PER-DOMAIN STEP-0 FAILURE & POST-ERROR ACCURACY")
    print("=" * 70)
    domain_stats = defaultdict(lambda: {
        "total": 0, "step0_fail": 0, "post_err_total": 0, "post_err_correct": 0
    })
    for t in trajectories:
        domain = t.get("domain", "unknown")
        steps = t.get("step_results", [])
        ds = domain_stats[domain]
        ds["total"] += 1
        if steps and not steps[0]["success"]:
            ds["step0_fail"] += 1
        first_err_idx = None
        for i, s in enumerate(steps):
            if not s["success"]:
                first_err_idx = i
                break
        if first_err_idx is not None:
            for s in steps[first_err_idx + 1:]:
                ds["post_err_total"] += 1
                if s["success"]:
                    ds["post_err_correct"] += 1

    print(f"\n   {'Domain':<15s}  {'N':>6s}  {'Step0 Fail%':>12s}  {'PostErr Acc%':>13s}")
    print(f"   {'-' * 15}  {'-' * 6}  {'-' * 12}  {'-' * 13}")
    for domain in sorted(domain_stats.keys()):
        ds = domain_stats[domain]
        s0f = ds["step0_fail"] / ds["total"] * 100 if ds["total"] else 0
        pea = ds["post_err_correct"] / ds["post_err_total"] * 100 if ds["post_err_total"] else 0
        print(f"   {domain:<15s}  {ds['total']:6d}  {s0f:11.2f}%  {pea:12.2f}%")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  SFT v2 (no-stop) achieves:")
    print(f"    - TSR: {sft_tsr:.2f}% (vs base {BASE['TSR']:.2f}%, {sft_tsr / BASE['TSR']:.1f}x improvement)")
    print(f"    - Step Acc: {sft_step_acc:.2f}% (vs base {BASE['step_acc']:.2f}%, {sft_step_acc / BASE['step_acc']:.1f}x)")
    print(f"    - Step-0 Failure: {step0_failure_rate:.2f}% (vs base {BASE['step0_failure']:.2f}%, reduced by {BASE['step0_failure'] - step0_failure_rate:.1f} pp)")
    print(f"    - Post-Error Acc: {post_error_acc:.2f}% (vs base {BASE['post_error_acc']:.2f}%, {post_error_acc / BASE['post_error_acc']:.1f}x)")
    print(f"    - Scattered Progress: {sft_scattered:.2f}% (vs base {BASE['scattered_progress']:.2f}%)")
    print(f"  Error breakdown ({total_failed} failed steps):")
    print(f"    - Stuck/repeating: {stuck_repeating} ({stuck_repeating / total_failed * 100:.1f}%)")
    print(f"    - Type mismatch: {type_mismatch} ({type_mismatch / total_failed * 100:.1f}%)")
    print(f"    - Coord error >=50px: {coord_error} ({coord_error / total_failed * 100:.1f}%)")
    print(f"    - Near miss <50px: {near_miss} ({near_miss / total_failed * 100:.1f}%)")
    print(f"    - Other: {other_error} ({other_error / total_failed * 100:.1f}%)")
    if coord_distances:
        print(f"  Coordinate error median distance: {coord_distances[len(coord_distances) // 2]:.1f} px")


if __name__ == "__main__":
    main()
