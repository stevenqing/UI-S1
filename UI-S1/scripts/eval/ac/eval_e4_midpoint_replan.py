"""E4: Midpoint Replanning — Baseline vs Adaptive Context Reset.

For trajectories with 8+ steps, compares:
  Condition A (baseline): No replanning, growing context throughout
  Condition B (adaptive): At step T//2, planner generates structured message,
                          executor resets context to O(1)

Both conditions use GT screenshots (teacher-forced) but agent-predicted actions.
Key metric: StepAcc for steps[T//2:] under both conditions, stratified by length.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from replan_utils import (
    get_adapter, generate_planner_message, format_action_text,
    compute_replan_metrics, compute_second_half_metrics,
    save_json, append_jsonl, _json_default, length_bucket,
)

result_lock = Lock()


def process_trajectory(trajectory, adapter, args):
    """Process one trajectory: run baseline and adaptive conditions.

    First half (steps 0..midpoint-1) is shared between conditions.
    Second half forks into baseline (continue) vs adaptive (replan+reset).
    """
    midpoint = trajectory.num_steps // 2

    # === First half: shared between both conditions ===
    shared_history = []
    shared_results = []
    for step_id in range(midpoint):
        try:
            result = adapter.predict_step(
                trajectory, step_id, trajectory.goal, shared_history, args
            )
            type_match, extract_match = adapter.evaluate_step(
                result['pred'], trajectory, step_id, result['dims']
            )
            shared_history.append(result['action_text'])
            shared_results.append({
                'step_num': step_id,
                'type_match': type_match,
                'extract_match': extract_match,
                'is_boundary': step_id == 0,
                'action_text': result['action_text'],
                'gt_action_type': trajectory.steps[step_id].gt_action_type,
            })
        except Exception as e:
            shared_history.append('error')
            shared_results.append({
                'step_num': step_id,
                'type_match': False,
                'extract_match': False,
                'is_boundary': step_id == 0,
                'action_text': 'error',
                'gt_action_type': trajectory.steps[step_id].gt_action_type,
                'error': str(e),
            })

    # === Condition A: baseline (continue without replanning) ===
    history_a = shared_history.copy()
    results_a = [dict(s, condition='baseline') for s in shared_results]
    for step_id in range(midpoint, trajectory.num_steps):
        try:
            result = adapter.predict_step(
                trajectory, step_id, trajectory.goal, history_a, args
            )
            type_match, extract_match = adapter.evaluate_step(
                result['pred'], trajectory, step_id, result['dims']
            )
            history_a.append(result['action_text'])
            results_a.append({
                'step_num': step_id,
                'type_match': type_match,
                'extract_match': extract_match,
                'is_boundary': False,
                'condition': 'baseline',
                'action_text': result['action_text'],
                'gt_action_type': trajectory.steps[step_id].gt_action_type,
            })
        except Exception as e:
            history_a.append('error')
            results_a.append({
                'step_num': step_id,
                'type_match': False,
                'extract_match': False,
                'is_boundary': False,
                'condition': 'baseline',
                'action_text': 'error',
                'gt_action_type': trajectory.steps[step_id].gt_action_type,
                'error': str(e),
            })

    # === Condition B: adaptive (replan at midpoint) ===
    planner_msg = generate_planner_message(trajectory, midpoint, 'structured')
    history_b = []
    results_b = [dict(s, condition='adaptive') for s in shared_results]
    for step_id in range(midpoint, trajectory.num_steps):
        try:
            result = adapter.predict_step(
                trajectory, step_id, planner_msg, history_b, args
            )
            type_match, extract_match = adapter.evaluate_step(
                result['pred'], trajectory, step_id, result['dims']
            )
            history_b.append(result['action_text'])
            results_b.append({
                'step_num': step_id,
                'type_match': type_match,
                'extract_match': extract_match,
                'is_boundary': step_id == midpoint,
                'condition': 'adaptive',
                'action_text': result['action_text'],
                'gt_action_type': trajectory.steps[step_id].gt_action_type,
                'instruction_given': planner_msg[:200],
            })
        except Exception as e:
            history_b.append('error')
            results_b.append({
                'step_num': step_id,
                'type_match': False,
                'extract_match': False,
                'is_boundary': step_id == midpoint,
                'condition': 'adaptive',
                'action_text': 'error',
                'gt_action_type': trajectory.steps[step_id].gt_action_type,
                'error': str(e),
            })

    output = {
        'trajectory_id': trajectory.trajectory_id,
        'goal': trajectory.goal,
        'num_steps': trajectory.num_steps,
        'midpoint': midpoint,
        'length_bucket': length_bucket(trajectory.num_steps),
        'planner_msg': planner_msg[:300],
        'baseline': {
            'step_results': results_a,
            'total_correct': sum(1 for s in results_a if s['extract_match']),
            'step_accuracy': sum(1 for s in results_a if s['extract_match']) / len(results_a) if results_a else 0,
        },
        'adaptive': {
            'step_results': results_b,
            'total_correct': sum(1 for s in results_b if s['extract_match']),
            'step_accuracy': sum(1 for s in results_b if s['extract_match']) / len(results_b) if results_b else 0,
        },
    }

    # Write to results file
    append_jsonl(output, os.path.join(args.output_dir, 'trajectory_results.jsonl'), result_lock)

    return output


def analyze_results(results, output_dir):
    """Compute and print comparative analysis."""
    print("\n" + "=" * 70)
    print("E4: Midpoint Replanning Results")
    print("=" * 70)

    for cond in ['baseline', 'adaptive']:
        # Overall
        all_steps = [s for r in results for s in r[cond]['step_results']]
        total = len(all_steps)
        correct = sum(1 for s in all_steps if s['extract_match'])
        type_c = sum(1 for s in all_steps if s['type_match'])
        print(f"\n  {cond.upper()} (all steps):")
        print(f"    StepAcc: {correct}/{total} = {correct/total*100:.1f}%")
        print(f"    TypeAcc: {type_c}/{total} = {type_c/total*100:.1f}%")

        # Second half only
        second_half = [
            s for r in results
            for s in r[cond]['step_results']
            if s['step_num'] >= r['midpoint']
        ]
        sh_total = len(second_half)
        sh_correct = sum(1 for s in second_half if s['extract_match'])
        sh_type = sum(1 for s in second_half if s['type_match'])
        print(f"    SecondHalf StepAcc: {sh_correct}/{sh_total} = {sh_correct/sh_total*100:.1f}%")
        print(f"    SecondHalf TypeAcc: {sh_type}/{sh_total} = {sh_type/sh_total*100:.1f}%")

    # Per-length-bucket breakdown
    print("\n--- Second Half StepAcc by Length Bucket ---")
    by_bucket = defaultdict(list)
    for r in results:
        by_bucket[r['length_bucket']].append(r)

    for bucket in sorted(by_bucket.keys()):
        group = by_bucket[bucket]
        print(f"\n  {bucket} (n={len(group)}):")
        for cond in ['baseline', 'adaptive']:
            second_half = [
                s for r in group
                for s in r[cond]['step_results']
                if s['step_num'] >= r['midpoint']
            ]
            total = len(second_half)
            correct = sum(1 for s in second_half if s['extract_match'])
            print(f"    {cond:10s}: {correct}/{total} = {correct/total*100:.1f}%" if total > 0 else f"    {cond:10s}: N/A")

    # Paired comparison: per-trajectory second-half improvement
    print("\n--- Paired Comparison (second half) ---")
    baseline_better = 0
    adaptive_better = 0
    tied = 0
    for r in results:
        mid = r['midpoint']
        b_acc = sum(1 for s in r['baseline']['step_results'] if s['step_num'] >= mid and s['extract_match'])
        a_acc = sum(1 for s in r['adaptive']['step_results'] if s['step_num'] >= mid and s['extract_match'])
        if b_acc > a_acc:
            baseline_better += 1
        elif a_acc > b_acc:
            adaptive_better += 1
        else:
            tied += 1
    print(f"  Baseline better: {baseline_better}")
    print(f"  Adaptive better: {adaptive_better}")
    print(f"  Tied:            {tied}")

    # Build summary
    summary = {
        'n_trajectories': len(results),
        'conditions': {},
    }
    for cond in ['baseline', 'adaptive']:
        all_steps = [s for r in results for s in r[cond]['step_results']]
        second_half = [
            s for r in results for s in r[cond]['step_results']
            if s['step_num'] >= r['midpoint']
        ]
        summary['conditions'][cond] = {
            'step_accuracy': sum(1 for s in all_steps if s['extract_match']) / len(all_steps) if all_steps else 0,
            'type_accuracy': sum(1 for s in all_steps if s['type_match']) / len(all_steps) if all_steps else 0,
            'second_half_accuracy': sum(1 for s in second_half if s['extract_match']) / len(second_half) if second_half else 0,
            'second_half_type_accuracy': sum(1 for s in second_half if s['type_match']) / len(second_half) if second_half else 0,
            'total_steps': len(all_steps),
            'second_half_steps': len(second_half),
        }

    # Per-bucket
    summary['length_bucket_stats'] = {}
    for bucket, group in by_bucket.items():
        summary['length_bucket_stats'][bucket] = {'n': len(group), 'conditions': {}}
        for cond in ['baseline', 'adaptive']:
            second_half = [
                s for r in group for s in r[cond]['step_results']
                if s['step_num'] >= r['midpoint']
            ]
            total = len(second_half)
            correct = sum(1 for s in second_half if s['extract_match'])
            summary['length_bucket_stats'][bucket]['conditions'][cond] = {
                'second_half_accuracy': correct / total if total > 0 else 0,
                'second_half_steps': total,
            }

    summary['paired'] = {
        'baseline_better': baseline_better,
        'adaptive_better': adaptive_better,
        'tied': tied,
    }

    save_json(summary, os.path.join(output_dir, 'summary.json'))
    print(f"\nSummary saved to {os.path.join(output_dir, 'summary.json')}")


def main():
    parser = argparse.ArgumentParser(description="E4: Midpoint Replanning Evaluation")
    parser.add_argument("--dataset", type=str, required=True, choices=['ac', 'gui360'])
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=32)
    # AC-specific
    parser.add_argument("--jsonl_file", type=str, default=None)
    parser.add_argument("--max_episodes", type=int, default=None)
    # GUI-360-specific
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--api_url", type=str, default="http://localhost:19815/v1")
    parser.add_argument("--max_samples", type=int, default=None)
    # Minimum trajectory length
    parser.add_argument("--min_steps", type=int, default=8, help="Minimum trajectory length to include")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'trajectory_results.jsonl')
    if os.path.exists(out_path):
        os.remove(out_path)

    adapter = get_adapter(args.dataset)
    trajectories = adapter.load_trajectories(args)

    # Filter to long trajectories
    long_trajs = [t for t in trajectories if t.num_steps >= args.min_steps]
    print(f"Loaded {len(trajectories)} trajectories, {len(long_trajs)} have {args.min_steps}+ steps.")
    print(f"Dataset: {args.dataset} | Workers: {args.max_workers}")

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_trajectory, t, adapter, args): t
            for t in long_trajs
        }
        for future in as_completed(futures):
            try:
                r = future.result()
                results.append(r)
                if len(results) % 20 == 0:
                    # Quick progress update
                    b_acc = sum(r['baseline']['total_correct'] for r in results) / max(sum(len(r['baseline']['step_results']) for r in results), 1)
                    a_acc = sum(r['adaptive']['total_correct'] for r in results) / max(sum(len(r['adaptive']['step_results']) for r in results), 1)
                    print(f"Progress: {len(results)}/{len(long_trajs)} | baseline={b_acc:.3f} adaptive={a_acc:.3f}")
            except Exception as e:
                print(f"Exception processing trajectory: {e}")

    if results:
        analyze_results(results, args.output_dir)
    else:
        print("No results collected. Check data and parameters.")


if __name__ == "__main__":
    main()
