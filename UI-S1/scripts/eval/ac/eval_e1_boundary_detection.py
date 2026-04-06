"""E1: Boundary Detection Methods — When should the planner reactivate?

Compares 4 boundary detection methods, all using structured protocol:
  A. Oracle:     GT action type transitions (ceiling)
  B. Agreement:  Low-agreement sampling (threshold hyperparameter)
  C. Fixed:      Every K steps (K hyperparameter)
  D. No boundary: step_0 only (= standard AR baseline)

Teacher-forced evaluation: GT screenshots, predicted actions for history.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from replan_utils import (
    get_adapter, detect_oracle_boundaries, detect_fixed_interval_boundaries,
    run_replan_trajectory, compute_replan_metrics,
    save_json, append_jsonl, _json_default, length_bucket,
)

PROTOCOL = 'structured'
result_lock = Lock()


def process_trajectory_oracle(trajectory, adapter, args):
    """Condition A: Oracle boundaries."""
    boundaries = detect_oracle_boundaries(trajectory)
    result = run_replan_trajectory(trajectory, adapter, args, boundaries, PROTOCOL)
    result['method'] = 'oracle'
    result['length_bucket'] = length_bucket(trajectory.num_steps)
    return result


def process_trajectory_agreement(trajectory, adapter, args):
    """Condition B: Agreement-based boundaries."""
    threshold = args.agreement_threshold
    K = args.agreement_k
    boundaries = [0]  # Always include step 0

    for step_id in range(1, trajectory.num_steps):
        is_boundary = adapter.detect_agreement_boundary(
            trajectory, step_id, args, threshold, K
        )
        if is_boundary:
            boundaries.append(step_id)

    result = run_replan_trajectory(trajectory, adapter, args, boundaries, PROTOCOL)
    result['method'] = 'agreement'
    result['agreement_threshold'] = threshold
    result['length_bucket'] = length_bucket(trajectory.num_steps)
    return result


def process_trajectory_fixed(trajectory, adapter, args):
    """Condition C: Fixed interval boundaries."""
    interval = args.fixed_interval
    boundaries = detect_fixed_interval_boundaries(trajectory, interval)
    result = run_replan_trajectory(trajectory, adapter, args, boundaries, PROTOCOL)
    result['method'] = f'fixed_{interval}'
    result['length_bucket'] = length_bucket(trajectory.num_steps)
    return result


def process_trajectory_no_boundary(trajectory, adapter, args):
    """Condition D: No boundary (step_0 only)."""
    boundaries = [0]
    result = run_replan_trajectory(trajectory, adapter, args, boundaries, PROTOCOL)
    result['method'] = 'no_boundary'
    result['length_bucket'] = length_bucket(trajectory.num_steps)
    return result


CONDITION_MAP = {
    'oracle': process_trajectory_oracle,
    'agreement': process_trajectory_agreement,
    'fixed': process_trajectory_fixed,
    'no_boundary': process_trajectory_no_boundary,
}


def run_condition(condition, trajectories, adapter, args):
    """Run all trajectories for a single condition."""
    process_fn = CONDITION_MAP[condition]

    out_path = os.path.join(args.output_dir, f'trajectory_results_{condition}.jsonl')
    if os.path.exists(out_path):
        os.remove(out_path)

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_fn, t, adapter, args): t
            for t in trajectories
        }
        for future in as_completed(futures):
            try:
                r = future.result()
                results.append(r)
                append_jsonl(r, out_path, result_lock)
                if len(results) % 50 == 0:
                    acc = sum(r['total_correct'] for r in results) / max(sum(len(r['step_results']) for r in results), 1)
                    print(f"  [{condition}] {len(results)}/{len(trajectories)} | StepAcc={acc:.3f}")
            except Exception as e:
                print(f"  [{condition}] Exception: {e}")
    return results


def analyze_results(all_results, output_dir):
    """Compare boundary detection methods."""
    print("\n" + "=" * 70)
    print("E1: Boundary Detection Results (Structured Protocol)")
    print("=" * 70)

    summary = {'conditions': {}}

    for condition in all_results:
        results = all_results[condition]
        metrics = compute_replan_metrics(results)
        summary['conditions'][condition] = metrics

        print(f"\n  {condition}:")
        print(f"    StepAcc:     {metrics['step_accuracy']:.4f} ({metrics['total_correct']}/{metrics['total_steps']})")
        print(f"    TypeAcc:     {metrics['type_accuracy']:.4f}")
        print(f"    TSR:         {metrics['tsr']:.4f}")
        print(f"    BoundaryAcc: {metrics['boundary_accuracy']:.4f} (n={metrics['boundary_total']})")
        print(f"    WithinAcc:   {metrics['within_phase_accuracy']:.4f} (n={metrics['within_phase_total']})")
        print(f"    AvgPlannerCalls: {metrics['avg_planner_calls_per_traj']:.1f}")

    # Per-length breakdown
    print("\n--- StepAcc by Length Bucket ---")
    for bucket in ['short(1-3)', 'medium(4-7)', 'long(8-15)', 'vlong(16+)']:
        row = f"  {bucket:15s}"
        for condition in all_results:
            metrics = summary['conditions'][condition]
            bm = metrics.get('length_bucket_stats', {}).get(bucket, {})
            if bm:
                row += f"  {condition}={bm['step_accuracy']:.3f}(n={bm['n']})"
        print(row)

    # Efficiency: StepAcc vs planner calls
    print("\n--- Efficiency: StepAcc per Planner Call ---")
    for condition in all_results:
        metrics = summary['conditions'][condition]
        avg_calls = metrics['avg_planner_calls_per_traj']
        step_acc = metrics['step_accuracy']
        print(f"  {condition:15s}: StepAcc={step_acc:.4f} AvgCalls={avg_calls:.1f} Ratio={step_acc/avg_calls:.4f}" if avg_calls > 0 else f"  {condition:15s}: StepAcc={step_acc:.4f} AvgCalls=0")

    save_json(summary, os.path.join(output_dir, 'summary.json'))
    print(f"\nSummary saved to {os.path.join(output_dir, 'summary.json')}")


def main():
    parser = argparse.ArgumentParser(description="E1: Boundary Detection Evaluation")
    parser.add_argument("--dataset", type=str, required=True, choices=['ac', 'gui360'])
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--conditions", type=str, nargs='+',
                        default=['oracle', 'fixed', 'no_boundary'],
                        choices=['oracle', 'agreement', 'fixed', 'no_boundary'],
                        help="Which conditions to evaluate (agreement is slow)")
    # Boundary detection hyperparameters
    parser.add_argument("--agreement_threshold", type=float, default=0.6,
                        help="Agreement threshold for boundary detection")
    parser.add_argument("--agreement_k", type=int, default=5,
                        help="Number of samples for agreement check")
    parser.add_argument("--fixed_interval", type=int, default=3,
                        help="Fixed interval for boundary detection")
    # AC-specific
    parser.add_argument("--jsonl_file", type=str, default=None)
    parser.add_argument("--max_episodes", type=int, default=None)
    # GUI-360-specific
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--api_url", type=str, default="http://localhost:19815/v1")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    adapter = get_adapter(args.dataset)
    trajectories = adapter.load_trajectories(args)
    print(f"Loaded {len(trajectories)} trajectories. Dataset: {args.dataset}")

    all_results = {}
    for condition in args.conditions:
        print(f"\n{'='*40}")
        print(f"Running condition: {condition}")
        if condition == 'agreement':
            print(f"  threshold={args.agreement_threshold}, K={args.agreement_k}")
        elif condition == 'fixed':
            print(f"  interval={args.fixed_interval}")
        print(f"{'='*40}")
        all_results[condition] = run_condition(condition, trajectories, adapter, args)

    analyze_results(all_results, args.output_dir)


if __name__ == "__main__":
    main()
