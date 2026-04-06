#!/usr/bin/env python3
"""
Eval D10: V2 Action Failure Mode Diagnosis (Offline)

Analyzes V2's 41.4% action errors in detail:
1. Action confusion matrix (gt_function → pred_function)
2. Step position analysis (early vs late errors)
3. Domain-specific confusion patterns
4. Near-miss vs complete-miss classification
5. Error recoverability analysis
"""

import json
import os
import argparse
from collections import defaultdict, Counter
import numpy as np


NEAR_MISS_PAIRS = {
    ("click", "double_click"), ("double_click", "click"),
    ("click", "right_click"), ("right_click", "click"),
    ("click", "set_focus"), ("set_focus", "click"),
    ("select_text", "select_paragraph"), ("select_paragraph", "select_text"),
    ("select_table", "select_table_range"), ("select_table_range", "select_table"),
    ("type", "click"),  # common: should type but clicks instead
    ("click", "type"),  # common: should click but types instead
}


def load_results(eval_a_dir):
    results = []
    path = os.path.join(eval_a_dir, "trajectory_results.jsonl")
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def analyze_confusion_matrix(results):
    """Build gt_function → pred_function confusion matrix."""
    confusion = defaultdict(Counter)
    total_steps = 0
    total_func_errors = 0

    for traj in results:
        for sr in traj["step_results"]:
            total_steps += 1
            if not sr["b_func_match"]:
                total_func_errors += 1
                gt = sr["gt_function"]
                pred = sr["pred_function"]
                confusion[gt][pred] += 1

    print(f"\n{'='*70}")
    print(f"1. ACTION CONFUSION MATRIX")
    print(f"{'='*70}")
    print(f"Total steps: {total_steps}")
    print(f"Total func errors (all steps): {total_func_errors} ({100*total_func_errors/total_steps:.1f}%)")

    # Top confusions
    all_confusions = []
    for gt, preds in confusion.items():
        for pred, count in preds.items():
            all_confusions.append((gt, pred, count))
    all_confusions.sort(key=lambda x: -x[2])

    print(f"\nTop 20 confusions (gt → pred):")
    print(f"{'GT Function':<25} {'Pred Function':<25} {'Count':>6} {'% of errors':>10}")
    print("-" * 70)
    for gt, pred, count in all_confusions[:20]:
        pct = 100 * count / total_func_errors if total_func_errors > 0 else 0
        near = " [near-miss]" if (gt, pred) in NEAR_MISS_PAIRS else ""
        print(f"{gt:<25} {pred:<25} {count:>6} {pct:>9.1f}%{near}")

    return confusion, total_func_errors


def analyze_first_error_confusion(results):
    """For failed trajectories, analyze the FIRST func_match error."""
    first_error_confusion = defaultdict(Counter)
    first_error_step_positions = []
    n_failed = 0
    n_func_first_error = 0

    for traj in results:
        if traj["b_trajectory_success"]:
            continue
        n_failed += 1

        for sr in traj["step_results"]:
            if not sr["b_success"]:
                # This is the first error
                if not sr["b_func_match"]:
                    n_func_first_error += 1
                    gt = sr["gt_function"]
                    pred = sr["pred_function"]
                    first_error_confusion[gt][pred] += 1
                    rel_pos = sr["step_num"] / traj["num_steps"] if traj["num_steps"] > 0 else 0
                    first_error_step_positions.append({
                        "step_num": sr["step_num"],
                        "total_steps": traj["num_steps"],
                        "rel_position": rel_pos,
                        "domain": traj["domain"],
                        "gt": gt,
                        "pred": pred,
                    })
                break

    print(f"\n{'='*70}")
    print(f"2. FIRST-ERROR CONFUSION (failed trajectories only)")
    print(f"{'='*70}")
    print(f"Failed trajectories: {n_failed}")
    print(f"First error is func_match=F: {n_func_first_error} ({100*n_func_first_error/n_failed:.1f}%)")

    all_confusions = []
    for gt, preds in first_error_confusion.items():
        for pred, count in preds.items():
            all_confusions.append((gt, pred, count))
    all_confusions.sort(key=lambda x: -x[2])

    print(f"\nTop 15 first-error confusions:")
    print(f"{'GT Function':<25} {'Pred Function':<25} {'Count':>6} {'% of func errors':>15}")
    print("-" * 75)
    for gt, pred, count in all_confusions[:15]:
        pct = 100 * count / n_func_first_error if n_func_first_error > 0 else 0
        near = " [near-miss]" if (gt, pred) in NEAR_MISS_PAIRS else ""
        print(f"{gt:<25} {pred:<25} {count:>6} {pct:>14.1f}%{near}")

    return first_error_confusion, first_error_step_positions


def analyze_step_position(first_error_positions):
    """Analyze WHERE in the trajectory func errors occur."""
    print(f"\n{'='*70}")
    print(f"3. STEP POSITION ANALYSIS (first func error)")
    print(f"{'='*70}")

    if not first_error_positions:
        print("No func errors found.")
        return

    # Absolute step position
    step_nums = [p["step_num"] for p in first_error_positions]
    step_counter = Counter(step_nums)

    print(f"\nAbsolute step position of first func error:")
    print(f"{'Step':<8} {'Count':>6} {'%':>8} {'Cumulative %':>12}")
    print("-" * 36)
    cum = 0
    total = len(first_error_positions)
    for step in sorted(step_counter.keys())[:15]:
        count = step_counter[step]
        pct = 100 * count / total
        cum += pct
        print(f"Step {step:<4} {count:>6} {pct:>7.1f}% {cum:>11.1f}%")

    # Relative position buckets
    rel_positions = [p["rel_position"] for p in first_error_positions]
    buckets = [(0, 0.25, "Early (0-25%)"), (0.25, 0.5, "Mid-early (25-50%)"),
               (0.5, 0.75, "Mid-late (50-75%)"), (0.75, 1.01, "Late (75-100%)")]

    print(f"\nRelative position of first func error:")
    for lo, hi, label in buckets:
        count = sum(1 for r in rel_positions if lo <= r < hi)
        pct = 100 * count / total
        print(f"  {label:<25} {count:>6} ({pct:.1f}%)")

    # Step 1 vs later: instruction parsing vs state tracking
    step1_errors = [p for p in first_error_positions if p["step_num"] == 1]
    later_errors = [p for p in first_error_positions if p["step_num"] > 1]
    print(f"\nStep 1 (instruction parsing): {len(step1_errors)} ({100*len(step1_errors)/total:.1f}%)")
    print(f"Step 2+ (state tracking):     {len(later_errors)} ({100*len(later_errors)/total:.1f}%)")

    # Step 1 confusion breakdown
    if step1_errors:
        step1_confusion = Counter((p["gt"], p["pred"]) for p in step1_errors)
        print(f"\n  Step 1 top confusions:")
        for (gt, pred), count in step1_confusion.most_common(10):
            print(f"    {gt} → {pred}: {count}")

    # Step 2+ confusion breakdown
    if later_errors:
        later_confusion = Counter((p["gt"], p["pred"]) for p in later_errors)
        print(f"\n  Step 2+ top confusions:")
        for (gt, pred), count in later_confusion.most_common(10):
            print(f"    {gt} → {pred}: {count}")


def analyze_near_miss_vs_complete_miss(results):
    """Classify func errors as near-miss or complete-miss."""
    print(f"\n{'='*70}")
    print(f"4. NEAR-MISS vs COMPLETE-MISS CLASSIFICATION")
    print(f"{'='*70}")

    near_miss = 0
    complete_miss = 0
    empty_pred = 0

    for traj in results:
        if traj["b_trajectory_success"]:
            continue
        for sr in traj["step_results"]:
            if not sr["b_success"]:
                if not sr["b_func_match"]:
                    gt = sr["gt_function"]
                    pred = sr["pred_function"]
                    if pred == "" or pred is None:
                        empty_pred += 1
                    elif (gt, pred) in NEAR_MISS_PAIRS:
                        near_miss += 1
                    else:
                        complete_miss += 1
                break

    total = near_miss + complete_miss + empty_pred
    if total == 0:
        print("No func errors found.")
        return

    print(f"Near-miss (related action):   {near_miss} ({100*near_miss/total:.1f}%)")
    print(f"Complete-miss (wrong action):  {complete_miss} ({100*complete_miss/total:.1f}%)")
    print(f"Empty prediction:             {empty_pred} ({100*empty_pred/total:.1f}%)")

    return {"near_miss": near_miss, "complete_miss": complete_miss, "empty_pred": empty_pred}


def analyze_domain_patterns(results):
    """Per-domain func error analysis."""
    print(f"\n{'='*70}")
    print(f"5. DOMAIN-SPECIFIC PATTERNS")
    print(f"{'='*70}")

    domain_stats = defaultdict(lambda: {
        "total_steps": 0, "func_errors": 0,
        "confusion": defaultdict(Counter),
        "failed_trajs": 0, "total_trajs": 0,
        "first_error_func": 0, "first_error_total": 0,
    })

    for traj in results:
        domain = traj["domain"]
        domain_stats[domain]["total_trajs"] += 1
        if not traj["b_trajectory_success"]:
            domain_stats[domain]["failed_trajs"] += 1

        first_error_found = False
        for sr in traj["step_results"]:
            domain_stats[domain]["total_steps"] += 1
            if not sr["b_func_match"]:
                domain_stats[domain]["func_errors"] += 1
                domain_stats[domain]["confusion"][sr["gt_function"]][sr["pred_function"]] += 1

            if not first_error_found and not sr["b_success"]:
                first_error_found = True
                domain_stats[domain]["first_error_total"] += 1
                if not sr["b_func_match"]:
                    domain_stats[domain]["first_error_func"] += 1

    for domain in sorted(domain_stats.keys()):
        stats = domain_stats[domain]
        print(f"\n--- {domain.upper()} ---")
        print(f"Trajectories: {stats['total_trajs']} (failed: {stats['failed_trajs']})")
        print(f"Steps: {stats['total_steps']}, func errors: {stats['func_errors']} ({100*stats['func_errors']/stats['total_steps']:.1f}%)")
        if stats["first_error_total"] > 0:
            pct = 100 * stats["first_error_func"] / stats["first_error_total"]
            print(f"First-error is func: {stats['first_error_func']}/{stats['first_error_total']} ({pct:.1f}%)")

        # Top confusions for this domain
        all_conf = []
        for gt, preds in stats["confusion"].items():
            for pred, count in preds.items():
                all_conf.append((gt, pred, count))
        all_conf.sort(key=lambda x: -x[2])

        print(f"Top confusions:")
        for gt, pred, count in all_conf[:8]:
            print(f"  {gt} → {pred}: {count}")


def analyze_v2_only_vs_v2v3(results):
    """Compare func_match between condition A (V2 only) and B (V2+V3).
    Since V3 only provides coordinates, func_match should be identical."""
    print(f"\n{'='*70}")
    print(f"6. V2-ONLY vs V2+V3 FUNC MATCH COMPARISON")
    print(f"{'='*70}")

    same = 0
    a_only = 0  # a_func_match=T, b_func_match=F (shouldn't happen)
    b_only = 0  # a_func_match=F, b_func_match=T (shouldn't happen)
    both_wrong = 0

    for traj in results:
        for sr in traj["step_results"]:
            a = sr["a_func_match"]
            b = sr["b_func_match"]
            if a == b:
                if a:
                    same += 1
                else:
                    both_wrong += 1
            elif a and not b:
                a_only += 1
            else:
                b_only += 1

    total = same + a_only + b_only + both_wrong
    print(f"Both correct:  {same} ({100*same/total:.1f}%)")
    print(f"Both wrong:    {both_wrong} ({100*both_wrong/total:.1f}%)")
    print(f"A correct only: {a_only}")
    print(f"B correct only: {b_only}")
    if a_only + b_only > 0:
        print("NOTE: Discrepancy detected — func_match should be identical between A and B!")


def analyze_error_cascading(results):
    """How many steps after the first func error are also func errors?"""
    print(f"\n{'='*70}")
    print(f"7. ERROR CASCADING ANALYSIS")
    print(f"{'='*70}")

    cascade_lengths = []
    post_error_func_match_rates = []

    for traj in results:
        if traj["b_trajectory_success"]:
            continue

        first_func_error_idx = None
        for i, sr in enumerate(traj["step_results"]):
            if not sr["b_func_match"]:
                first_func_error_idx = i
                break

        if first_func_error_idx is None:
            continue

        remaining = traj["step_results"][first_func_error_idx + 1:]
        if not remaining:
            continue

        n_subsequent_func_errors = sum(1 for sr in remaining if not sr["b_func_match"])
        cascade_lengths.append(n_subsequent_func_errors)
        func_match_rate = sum(1 for sr in remaining if sr["b_func_match"]) / len(remaining)
        post_error_func_match_rates.append(func_match_rate)

    if not cascade_lengths:
        print("No cascading data.")
        return

    print(f"Trajectories with func error + remaining steps: {len(cascade_lengths)}")
    print(f"Avg subsequent func errors: {np.mean(cascade_lengths):.2f}")
    print(f"Avg post-error func_match rate: {100*np.mean(post_error_func_match_rates):.1f}%")

    # Distribution
    print(f"\nSubsequent func errors distribution:")
    cascade_counter = Counter(cascade_lengths)
    for n in sorted(cascade_counter.keys())[:10]:
        count = cascade_counter[n]
        print(f"  {n} more func errors: {count} ({100*count/len(cascade_lengths):.1f}%)")


def analyze_action_type_accuracy(results):
    """Per-action-type accuracy."""
    print(f"\n{'='*70}")
    print(f"8. PER-ACTION-TYPE ACCURACY")
    print(f"{'='*70}")

    action_stats = defaultdict(lambda: {"correct": 0, "total": 0, "confused_with": Counter()})

    for traj in results:
        for sr in traj["step_results"]:
            gt = sr["gt_function"]
            pred = sr["pred_function"]
            action_stats[gt]["total"] += 1
            if sr["b_func_match"]:
                action_stats[gt]["correct"] += 1
            else:
                action_stats[gt]["confused_with"][pred] += 1

    print(f"\n{'Action':<25} {'Total':>6} {'Correct':>8} {'Accuracy':>10} {'Top Confusion':>30}")
    print("-" * 85)
    for action in sorted(action_stats.keys(), key=lambda a: -action_stats[a]["total"]):
        stats = action_stats[action]
        acc = 100 * stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        top_conf = stats["confused_with"].most_common(1)
        top_conf_str = f"→ {top_conf[0][0]} ({top_conf[0][1]})" if top_conf else ""
        print(f"{action:<25} {stats['total']:>6} {stats['correct']:>8} {acc:>9.1f}% {top_conf_str:>30}")


def generate_summary(results, output_dir):
    """Generate machine-readable summary."""
    summary = {
        "total_trajectories": len(results),
        "failed_trajectories": sum(1 for r in results if not r["b_trajectory_success"]),
        "total_steps": sum(len(r["step_results"]) for r in results),
        "per_action_accuracy": {},
        "confusion_matrix": {},
        "first_error_analysis": {},
    }

    # Per-action accuracy
    action_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for traj in results:
        for sr in traj["step_results"]:
            gt = sr["gt_function"]
            action_stats[gt]["total"] += 1
            if sr["b_func_match"]:
                action_stats[gt]["correct"] += 1
    for action, stats in action_stats.items():
        summary["per_action_accuracy"][action] = {
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
        }

    # Confusion matrix (top pairs only)
    confusion = defaultdict(Counter)
    for traj in results:
        for sr in traj["step_results"]:
            if not sr["b_func_match"]:
                confusion[sr["gt_function"]][sr["pred_function"]] += 1
    top_confusions = []
    for gt, preds in confusion.items():
        for pred, count in preds.items():
            top_confusions.append({"gt": gt, "pred": pred, "count": count})
    top_confusions.sort(key=lambda x: -x["count"])
    summary["confusion_matrix"] = top_confusions[:30]

    # First error analysis
    first_error_step1 = 0
    first_error_later = 0
    first_error_near_miss = 0
    first_error_complete_miss = 0
    total_first_errors = 0

    for traj in results:
        if traj["b_trajectory_success"]:
            continue
        for sr in traj["step_results"]:
            if not sr["b_success"]:
                if not sr["b_func_match"]:
                    total_first_errors += 1
                    if sr["step_num"] == 1:
                        first_error_step1 += 1
                    else:
                        first_error_later += 1
                    gt = sr["gt_function"]
                    pred = sr["pred_function"]
                    if (gt, pred) in NEAR_MISS_PAIRS:
                        first_error_near_miss += 1
                    else:
                        first_error_complete_miss += 1
                break

    summary["first_error_analysis"] = {
        "total_func_first_errors": total_first_errors,
        "step1_errors": first_error_step1,
        "later_step_errors": first_error_later,
        "near_miss": first_error_near_miss,
        "complete_miss": first_error_complete_miss,
        "step1_pct": first_error_step1 / total_first_errors if total_first_errors > 0 else 0,
        "near_miss_pct": first_error_near_miss / total_first_errors if total_first_errors > 0 else 0,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "d10_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_dir}/d10_summary.json")

    return summary


def main():
    parser = argparse.ArgumentParser(description="D10: V2 Action Failure Mode Diagnosis")
    parser.add_argument("--eval_a_dir", default="outputs/eval_a",
                        help="Directory containing eval_a trajectory_results.jsonl")
    parser.add_argument("--output_dir", default="outputs/eval_d10",
                        help="Output directory for results")
    args = parser.parse_args()

    print("=" * 70)
    print("EVAL D10: V2 ACTION FAILURE MODE DIAGNOSIS")
    print("=" * 70)

    results = load_results(args.eval_a_dir)
    print(f"Loaded {len(results)} trajectories")

    # Run all analyses
    confusion, total_func_errors = analyze_confusion_matrix(results)
    first_error_confusion, first_error_positions = analyze_first_error_confusion(results)
    analyze_step_position(first_error_positions)
    analyze_near_miss_vs_complete_miss(results)
    analyze_domain_patterns(results)
    analyze_v2_only_vs_v2v3(results)
    analyze_error_cascading(results)
    analyze_action_type_accuracy(results)
    summary = generate_summary(results, args.output_dir)

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
