#!/usr/bin/env python3
"""
Error Decomposition Analysis for Long-Horizon GUI Agent

Analyzes what affects long-horizon task success beyond step accuracy:
1. Step position vs accuracy (where do errors concentrate?)
2. Reward component breakdown (format / type / content)
3. Error fragility analysis (how many predictions are near-threshold?)
4. Per-action-type accuracy
5. Task-level failure patterns
6. Theoretical vs actual TSR gap
"""

import json
import sys
import os
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

PROJECT = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"


def load_data():
    bon_path = f"{PROJECT}/v15_gui_360/outputs/preliminary_tests/bon_metrics/bon_raw_20260523_122948.json"
    train_path = f"{PROJECT}/v12_gui_360/data/gui360_train_2000_balanced.jsonl"

    bon = json.load(open(bon_path))

    train_episodes = {}
    with open(train_path) as f:
        for line in f:
            ep = json.loads(line)
            train_episodes[ep["episode_id"]] = ep

    return bon, train_episodes


def analyze_step_position_accuracy(bon, train_episodes):
    """1. Step position vs accuracy — where do errors concentrate?"""
    print("=" * 70)
    print("1. Step Position vs Accuracy")
    print("=" * 70)

    # Group by step position
    step_data = defaultdict(lambda: {"correct": 0, "total": 0, "rewards": []})

    # Also track by action type at each position
    step_type_data = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    for ep_id, ep in bon.items():
        T = len(ep["steps"])
        for step in ep["steps"]:
            s_idx = step["step_idx"]
            gt_type = step["gt_type"]

            # Best-of-N accuracy
            best = max(step["samples"], key=lambda x: x["reward"])
            is_correct = best["is_correct"]

            step_data[s_idx]["total"] += 1
            step_data[s_idx]["rewards"].append(best["reward"])
            if is_correct:
                step_data[s_idx]["correct"] += 1

            step_type_data[s_idx][gt_type]["total"] += 1
            if is_correct:
                step_type_data[s_idx][gt_type]["correct"] += 1

    # Print step-wise accuracy
    print("\nStep | Total | Accuracy | Mean Reward | Action Type Breakdown")
    print("-" * 80)
    for s in range(min(15, max(step_data.keys()) + 1)):
        if step_data[s]["total"] == 0:
            continue
        d = step_data[s]
        acc = d["correct"] / d["total"]
        mean_r = np.mean(d["rewards"])

        type_parts = []
        for at in ["click", "type", "swipe"]:
            td = step_type_data[s].get(at, {"correct": 0, "total": 0})
            if td["total"] > 0:
                type_parts.append(f"{at}={td['correct']}/{td['total']}({td['correct']/td['total']:.0%})")

        print(f"  {s:2d}  | {d['total']:5d} | {acc:7.1%}  | {mean_r:10.3f}  | {', '.join(type_parts)}")

    # Relative position (normalize by T)
    print("\n--- Relative Position (step_idx / T) ---")
    rel_bins = defaultdict(lambda: {"correct": 0, "total": 0})
    for ep_id, ep in bon.items():
        T = len(ep["steps"])
        for step in ep["steps"]:
            s_idx = step["step_idx"]
            rel_pos = s_idx / max(T - 1, 1)
            bin_idx = min(int(rel_pos * 5), 4)  # 5 bins: 0-20%, 20-40%, etc.

            best = max(step["samples"], key=lambda x: x["reward"])
            rel_bins[bin_idx]["total"] += 1
            if best["is_correct"]:
                rel_bins[bin_idx]["correct"] += 1

    bin_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    print(f"\n  {'Position':>10s} | {'Total':>5s} | {'Accuracy':>8s}")
    print("  " + "-" * 35)
    for b in range(5):
        d = rel_bins[b]
        if d["total"] > 0:
            print(f"  {bin_labels[b]:>10s} | {d['total']:5d} | {d['correct']/d['total']:7.1%}")


def analyze_reward_components(bon):
    """2. Reward component breakdown — format / type / content."""
    print("\n" + "=" * 70)
    print("2. Reward Component Breakdown")
    print("=" * 70)

    # For each sample, we have reward, type_reward, content_reward
    # reward = w_format * format_ok + w_type * type_reward + w_content * content_reward
    # format = 0.1 if has_valid_action else 0
    # type = 0.2 if pred_type == gt_type else 0
    # content = 0.7 * content_match

    format_ok = 0
    type_ok = 0
    content_ok = 0  # reward > 0.5
    total = 0

    # Detailed breakdown
    error_modes = Counter()

    for ep in bon.values():
        for step in ep["steps"]:
            for sample in step["samples"]:
                total += 1
                has_format = sample.get("has_valid_action", False)
                type_match = (sample.get("pred_type", "") == sample.get("gt_type", ""))
                content_r = sample.get("content_reward", 0)
                is_correct = sample["is_correct"]

                if has_format:
                    format_ok += 1
                if type_match:
                    type_ok += 1
                if is_correct:
                    content_ok += 1

                # Error mode classification
                if is_correct:
                    error_modes["correct"] += 1
                elif not has_format:
                    error_modes["format_error"] += 1
                elif not type_match:
                    error_modes["type_error"] += 1
                else:
                    error_modes["content_error (right type, wrong location)"] += 1

    print(f"\nTotal samples: {total}")
    print(f"  Format valid:  {format_ok}/{total} = {format_ok/total:.1%}")
    print(f"  Type correct:  {type_ok}/{total} = {type_ok/total:.1%}")
    print(f"  Full correct:  {content_ok}/{total} = {content_ok/total:.1%}")

    print(f"\nError Mode Decomposition:")
    for mode, count in error_modes.most_common():
        print(f"  {mode:45s}: {count:5d} ({count/total:.1%})")

    # Per action type
    print("\nPer Action Type:")
    for gt_type in ["click", "type", "swipe"]:
        type_total = 0
        type_correct = 0
        type_errors = Counter()
        for ep in bon.values():
            for step in ep["steps"]:
                if step["gt_type"] != gt_type:
                    continue
                for sample in step["samples"]:
                    type_total += 1
                    if sample["is_correct"]:
                        type_correct += 1
                        type_errors["correct"] += 1
                    elif not sample.get("has_valid_action", False):
                        type_errors["format"] += 1
                    elif sample.get("pred_type", "") != gt_type:
                        type_errors["wrong_type"] += 1
                    else:
                        type_errors["wrong_content"] += 1
        if type_total > 0:
            print(f"\n  {gt_type}: {type_correct}/{type_total} = {type_correct/type_total:.1%}")
            for mode, count in type_errors.most_common():
                print(f"    {mode}: {count} ({count/type_total:.1%})")


def analyze_fragility(bon):
    """3. Fragility — how many predictions are near-threshold?"""
    print("\n" + "=" * 70)
    print("3. Prediction Fragility (Reward Distribution)")
    print("=" * 70)

    all_rewards = []
    best_rewards = []
    for ep in bon.values():
        for step in ep["steps"]:
            for sample in step["samples"]:
                all_rewards.append(sample["reward"])
            best = max(step["samples"], key=lambda x: x["reward"])
            best_rewards.append(best["reward"])

    all_rewards = np.array(all_rewards)
    best_rewards = np.array(best_rewards)

    # Histogram of best-of-N rewards
    print("\nBest-of-N Reward Distribution:")
    bins = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    for lo, hi in bins:
        count = ((best_rewards >= lo) & (best_rewards < hi)).sum()
        label = f"[{lo:.1f}, {hi:.1f})"
        bar = "█" * int(count / len(best_rewards) * 50)
        marker = " ← threshold" if lo == 0.5 else ""
        print(f"  {label}: {count:4d} ({count/len(best_rewards):5.1%}) {bar}{marker}")

    # Fragile correct: reward in [0.5, 0.7) — barely correct
    fragile = ((best_rewards >= 0.5) & (best_rewards < 0.7)).sum()
    solid = (best_rewards >= 0.7).sum()
    wrong = (best_rewards < 0.5).sum()
    total = len(best_rewards)

    print(f"\n  Fragile correct (0.5-0.7): {fragile:4d} ({fragile/total:.1%}) — barely over threshold")
    print(f"  Solid correct   (0.7-1.0): {solid:4d} ({solid/total:.1%}) — confidently correct")
    print(f"  Wrong           (< 0.5):   {wrong:4d} ({wrong/total:.1%})")

    # All samples (not just best-of-N)
    print(f"\nAll Samples Reward Stats (N={len(all_rewards)}):")
    print(f"  Mean: {all_rewards.mean():.3f}, Median: {np.median(all_rewards):.3f}")
    print(f"  Std: {all_rewards.std():.3f}")
    near_threshold = ((all_rewards >= 0.4) & (all_rewards < 0.6)).sum()
    print(f"  Near threshold (0.4-0.6): {near_threshold} ({near_threshold/len(all_rewards):.1%})")


def analyze_tsr_gap(bon, train_episodes):
    """4. Theoretical vs actual TSR gap."""
    print("\n" + "=" * 70)
    print("4. Theoretical vs Actual TSR (Step Independence Analysis)")
    print("=" * 70)

    # Group episodes by T
    t_groups = defaultdict(lambda: {"episodes": [], "step_accs": defaultdict(list)})

    for ep_id, ep in bon.items():
        T = len(ep["steps"])
        all_correct = True

        for step in ep["steps"]:
            best = max(step["samples"], key=lambda x: x["reward"])
            is_correct = best["is_correct"]
            t_groups[T]["step_accs"][step["step_idx"]].append(1 if is_correct else 0)
            if not is_correct:
                all_correct = False

        t_groups[T]["episodes"].append({
            "ep_id": ep_id,
            "all_correct": all_correct,
        })

    print(f"\n  {'T':>3s} | {'Episodes':>8s} | {'Actual TSR':>10s} | {'Theo TSR':>10s} | {'Gap':>8s} | {'Mean Step Acc':>13s}")
    print("  " + "-" * 70)

    overall_actual = 0
    overall_total = 0

    for T in sorted(t_groups.keys()):
        if T > 15:
            continue
        g = t_groups[T]
        n_eps = len(g["episodes"])
        n_success = sum(1 for e in g["episodes"] if e["all_correct"])
        actual_tsr = n_success / n_eps if n_eps > 0 else 0

        # Theoretical TSR assuming independence
        step_accs = []
        for s in range(T):
            accs = g["step_accs"].get(s, [])
            if accs:
                step_accs.append(np.mean(accs))
            else:
                step_accs.append(0)

        theo_tsr = np.prod(step_accs) if step_accs else 0
        mean_step_acc = np.mean(step_accs) if step_accs else 0
        gap = actual_tsr - theo_tsr

        overall_actual += n_success
        overall_total += n_eps

        if n_eps >= 3:
            print(f"  {T:3d} | {n_eps:8d} | {actual_tsr:9.1%}  | {theo_tsr:9.1%}  | {gap:+7.1%}  | {mean_step_acc:12.1%}")

    if overall_total > 0:
        print(f"\n  Overall actual TSR (best-of-N, step-independent): {overall_actual}/{overall_total} = {overall_actual/overall_total:.1%}")
        print(f"  Note: This is step-independent (GT screenshot per step), NOT on-policy.")
        print(f"  Real on-policy TSR would be lower due to error cascading.")


def analyze_task_patterns(bon, train_episodes):
    """5. Task-level failure patterns."""
    print("\n" + "=" * 70)
    print("5. Task-Level Failure Patterns")
    print("=" * 70)

    # Find which steps fail most within each episode
    first_error_dist = Counter()
    task_difficulty = []

    for ep_id, ep in bon.items():
        T = len(ep["steps"])
        ep_idx = int(ep_id)
        goal = train_episodes.get(ep_idx, {}).get("goal", "Unknown")

        first_error = None
        step_results = []
        for step in ep["steps"]:
            best = max(step["samples"], key=lambda x: x["reward"])
            step_results.append(best["is_correct"])
            if not best["is_correct"] and first_error is None:
                first_error = step["step_idx"]

        if first_error is not None:
            first_error_dist[first_error] += 1

        n_correct = sum(step_results)
        task_difficulty.append({
            "ep_id": ep_id,
            "T": T,
            "n_correct": n_correct,
            "step_acc": n_correct / T if T > 0 else 0,
            "first_error": first_error,
            "goal": goal[:60],
        })

    # First error position distribution
    print("\nFirst Error Position Distribution:")
    print(f"  No errors (all correct): {sum(1 for t in task_difficulty if t['first_error'] is None)}")
    for pos in range(min(10, max(first_error_dist.keys()) + 1)):
        if pos in first_error_dist:
            bar = "█" * (first_error_dist[pos] // 2)
            print(f"  Step {pos:2d}: {first_error_dist[pos]:3d} {bar}")

    # Hardest tasks (lowest step accuracy)
    task_difficulty.sort(key=lambda x: x["step_acc"])
    print(f"\nTop 10 Hardest Tasks (lowest step accuracy):")
    for t in task_difficulty[:10]:
        print(f"  ep {t['ep_id']:>3s}: {t['n_correct']}/{t['T']} ({t['step_acc']:.0%}) | {t['goal']}")

    # Easiest tasks
    task_difficulty.sort(key=lambda x: -x["step_acc"])
    print(f"\nTop 10 Easiest Tasks (highest step accuracy, T>=4):")
    for t in [x for x in task_difficulty if x["T"] >= 4][:10]:
        print(f"  ep {t['ep_id']:>3s}: {t['n_correct']}/{t['T']} ({t['step_acc']:.0%}) | {t['goal']}")

    # App type analysis (Word/Excel/PowerPoint)
    app_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for t in task_difficulty:
        goal = t["goal"].lower()
        if "word" in goal:
            app = "Word"
        elif "excel" in goal or "spreadsheet" in goal:
            app = "Excel"
        elif "powerpoint" in goal or "slide" in goal:
            app = "PowerPoint"
        else:
            app = "Other"
        app_stats[app]["total"] += t["T"]
        app_stats[app]["correct"] += t["n_correct"]

    print(f"\nPer-App Step Accuracy:")
    for app in ["Word", "Excel", "PowerPoint", "Other"]:
        d = app_stats[app]
        if d["total"] > 0:
            print(f"  {app:12s}: {d['correct']}/{d['total']} = {d['correct']/d['total']:.1%}")


def analyze_click_coordinate_errors(bon):
    """6. Click coordinate error analysis — how far off are wrong clicks?"""
    print("\n" + "=" * 70)
    print("6. Click Coordinate Error Analysis")
    print("=" * 70)

    # For click steps, analyze distance between predicted and GT coordinates
    # We don't have GT coordinates directly, but we can analyze from reward/content_reward
    # and from the coordinate agreement and predicted coords

    click_rewards = []
    type_correct_click_rewards = []

    for ep in bon.values():
        for step in ep["steps"]:
            if step["gt_type"] != "click":
                continue
            for sample in step["samples"]:
                if not sample.get("has_valid_action", False):
                    continue
                click_rewards.append(sample["reward"])
                if sample.get("pred_type") == "click":
                    type_correct_click_rewards.append(sample["content_reward"])

    click_rewards = np.array(click_rewards)
    type_correct_click_rewards = np.array(type_correct_click_rewards)

    print(f"\nClick predictions (type=click, valid format): {len(type_correct_click_rewards)}")
    print(f"  Content reward distribution:")
    bins = [(0, 0.01), (0.01, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    for lo, hi in bins:
        count = ((type_correct_click_rewards >= lo) & (type_correct_click_rewards < hi)).sum()
        pct = count / len(type_correct_click_rewards) if len(type_correct_click_rewards) > 0 else 0
        bar = "█" * int(pct * 50)
        print(f"    [{lo:.2f}, {hi:.2f}): {count:4d} ({pct:5.1%}) {bar}")

    # Analyze coordinate agreement (self-consistency of predictions)
    coord_agreements = []
    for ep in bon.values():
        for step in ep["steps"]:
            if step["gt_type"] != "click":
                continue
            coords = []
            for sample in step["samples"]:
                if sample.get("pred_type") == "click" and sample.get("coord_x") is not None:
                    coords.append((sample["coord_x"], sample["coord_y"]))
            if len(coords) >= 2:
                # Compute pairwise distances
                dists = []
                for i in range(len(coords)):
                    for j in range(i + 1, len(coords)):
                        d = np.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)
                        dists.append(d)
                coord_agreements.append({
                    "mean_dist": np.mean(dists),
                    "max_dist": np.max(dists),
                    "n_coords": len(coords),
                    "best_reward": max(s["reward"] for s in step["samples"]),
                })

    if coord_agreements:
        mean_dists = [c["mean_dist"] for c in coord_agreements]
        print(f"\nClick Self-Consistency (pairwise distance between N=5 predictions):")
        print(f"  Mean pairwise distance: {np.mean(mean_dists):.1f}px")
        print(f"  Median: {np.median(mean_dists):.1f}px")

        # Correlation: high consistency → higher reward?
        consistent = [c for c in coord_agreements if c["mean_dist"] < 50]
        inconsistent = [c for c in coord_agreements if c["mean_dist"] > 200]
        if consistent and inconsistent:
            print(f"\n  Consistent (<50px spread): {len(consistent)} steps, "
                  f"mean best_reward={np.mean([c['best_reward'] for c in consistent]):.3f}")
            print(f"  Inconsistent (>200px spread): {len(inconsistent)} steps, "
                  f"mean best_reward={np.mean([c['best_reward'] for c in inconsistent]):.3f}")


def analyze_long_horizon_bottleneck(bon, train_episodes):
    """7. Long-horizon bottleneck — what limits multi-step success?"""
    print("\n" + "=" * 70)
    print("7. Long-Horizon Bottleneck Analysis")
    print("=" * 70)

    # For each episode, compute:
    # - How many consecutive correct steps from the start (simulated on-policy)
    # - Where the first error happens
    # - What type of error it is

    bottleneck_data = []

    for ep_id, ep in bon.items():
        T = len(ep["steps"])
        ep_idx = int(ep_id)
        goal = train_episodes.get(ep_idx, {}).get("goal", "Unknown")

        consecutive = 0
        first_error_type = None
        first_error_gt_type = None

        for step in ep["steps"]:
            best = max(step["samples"], key=lambda x: x["reward"])
            if best["is_correct"]:
                consecutive += 1
            else:
                # Classify error
                if not best.get("has_valid_action", False):
                    first_error_type = "format"
                elif best.get("pred_type") != step["gt_type"]:
                    first_error_type = "wrong_action_type"
                else:
                    first_error_type = "wrong_content"
                first_error_gt_type = step["gt_type"]
                break

        bottleneck_data.append({
            "T": T,
            "consecutive": consecutive,
            "progress": consecutive / T if T > 0 else 0,
            "first_error_type": first_error_type,
            "first_error_gt_type": first_error_gt_type,
            "goal": goal[:60],
        })

    # Progress distribution
    print("\nSimulated On-Policy Progress (consecutive correct from start):")
    progress_bins = defaultdict(int)
    for b in bottleneck_data:
        bin_key = f"{int(b['progress']*100)//20*20}-{int(b['progress']*100)//20*20+20}%"
        progress_bins[bin_key] += 1

    for label in ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]:
        count = progress_bins.get(label, 0)
        bar = "█" * (count // 2)
        print(f"  {label:>8s}: {count:3d} {bar}")

    complete = sum(1 for b in bottleneck_data if b["progress"] == 1.0)
    print(f"\n  Fully complete: {complete}/{len(bottleneck_data)} = {complete/len(bottleneck_data):.1%}")
    print(f"  Mean progress: {np.mean([b['progress'] for b in bottleneck_data]):.1%}")

    # First error type breakdown
    error_types = Counter()
    error_gt_types = Counter()
    for b in bottleneck_data:
        if b["first_error_type"]:
            error_types[b["first_error_type"]] += 1
            error_gt_types[b["first_error_gt_type"]] += 1

    print(f"\nFirst Error Type (what kills the trajectory):")
    for et, count in error_types.most_common():
        print(f"  {et:25s}: {count:3d} ({count/len(bottleneck_data):.1%})")

    print(f"\nFirst Error Action Type (which action type fails):")
    for gt, count in error_gt_types.most_common():
        print(f"  {gt:25s}: {count:3d} ({count/len(bottleneck_data):.1%})")

    # By T
    print(f"\nProgress by Task Length:")
    t_progress = defaultdict(list)
    for b in bottleneck_data:
        t_progress[b["T"]].append(b["progress"])

    for T in sorted(t_progress.keys()):
        if len(t_progress[T]) >= 3 and T <= 15:
            progs = t_progress[T]
            print(f"  T={T:2d}: mean={np.mean(progs):.1%}, "
                  f"complete={sum(1 for p in progs if p==1.0)}/{len(progs)}, "
                  f"n={len(progs)}")


def main():
    print("Loading data...")
    bon, train_episodes = load_data()
    print(f"Loaded {len(bon)} episodes, {len(train_episodes)} train episodes\n")

    analyze_step_position_accuracy(bon, train_episodes)
    analyze_reward_components(bon)
    analyze_fragility(bon)
    analyze_tsr_gap(bon, train_episodes)
    analyze_task_patterns(bon, train_episodes)
    analyze_click_coordinate_errors(bon)
    analyze_long_horizon_bottleneck(bon, train_episodes)

    print("\n" + "=" * 70)
    print("Summary: Key Insights for Long-Horizon Improvement")
    print("=" * 70)
    print("""
Based on the analysis above, the key factors limiting long-horizon
performance beyond step accuracy are:

1. ERROR CONCENTRATION: Where errors cluster (early vs late steps,
   specific action types) determines the highest-leverage improvements.

2. ERROR TYPE: Format vs type vs content errors require different
   solutions (prompt engineering vs action space vs visual grounding).

3. FRAGILITY: "Barely correct" predictions (reward 0.5-0.7) are
   unreliable in multi-step chains — improving these to solid correct
   (>0.7) matters more than converting wrong→correct.

4. TASK STRUCTURE: Some task categories are systematically harder.
   Targeted improvement on those categories gives disproportionate TSR gains.

5. CASCADING: Simulated on-policy progress shows how quickly errors
   compound. Even small per-step improvements have multiplicative effects
   on long-horizon success.
""")


if __name__ == "__main__":
    main()
