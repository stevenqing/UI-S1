"""E3: Replanning Frequency — How often should the planner fire?

For long trajectories (8+ steps), sweeps:
  Fixed interval: every 1, 2, 3, 5 steps
  Adaptive:       agreement threshold 0.2, 0.4, 0.6, 0.8

All conditions use structured protocol.
Pareto analysis: StepAcc vs planner call count.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from replan_utils import (
    get_adapter, detect_fixed_interval_boundaries, detect_oracle_boundaries,
    run_replan_trajectory, compute_replan_metrics,
    save_json, append_jsonl, _json_default, length_bucket,
)

PROTOCOL = 'structured'
result_lock = Lock()

# Sweep configurations
FIXED_INTERVALS = [1, 2, 3, 5]
AGREEMENT_THRESHOLDS = [0.2, 0.4, 0.6, 0.8]


def process_trajectory_fixed(trajectory, adapter, args, interval):
    """Run one trajectory with fixed-interval boundaries."""
    boundaries = detect_fixed_interval_boundaries(trajectory, interval)
    result = run_replan_trajectory(trajectory, adapter, args, boundaries, PROTOCOL)
    result['method'] = f'fixed_{interval}'
    result['interval'] = interval
    result['length_bucket'] = length_bucket(trajectory.num_steps)
    return result


def process_trajectory_agreement(trajectory, adapter, args, threshold, K=5):
    """Run one trajectory with agreement-based boundaries."""
    boundaries = [0]
    agreement_scores = {}

    for step_id in range(1, trajectory.num_steps):
        is_boundary = adapter.detect_agreement_boundary(
            trajectory, step_id, args, threshold, K
        )
        if is_boundary:
            boundaries.append(step_id)

    result = run_replan_trajectory(trajectory, adapter, args, boundaries, PROTOCOL)
    result['method'] = f'agreement_{threshold}'
    result['threshold'] = threshold
    result['length_bucket'] = length_bucket(trajectory.num_steps)
    return result


def run_fixed_sweep(trajectories, adapter, args):
    """Run all fixed-interval conditions."""
    all_results = {}

    for interval in args.fixed_intervals:
        label = f'fixed_{interval}'
        out_path = os.path.join(args.output_dir, f'trajectory_results_{label}.jsonl')
        if os.path.exists(out_path):
            os.remove(out_path)

        print(f"\n  Running fixed interval={interval}...")
        results = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(process_trajectory_fixed, t, adapter, args, interval): t
                for t in trajectories
            }
            for future in as_completed(futures):
                try:
                    r = future.result()
                    results.append(r)
                    append_jsonl(r, out_path, result_lock)
                    if len(results) % 20 == 0:
                        acc = sum(r['total_correct'] for r in results) / max(sum(len(r['step_results']) for r in results), 1)
                        print(f"    [{label}] {len(results)}/{len(trajectories)} | StepAcc={acc:.3f}")
                except Exception as e:
                    print(f"    [{label}] Exception: {e}")

        all_results[label] = results
    return all_results


def run_agreement_sweep(trajectories, adapter, args):
    """Run all agreement-threshold conditions."""
    all_results = {}

    for threshold in args.agreement_thresholds:
        label = f'agreement_{threshold}'
        out_path = os.path.join(args.output_dir, f'trajectory_results_{label}.jsonl')
        if os.path.exists(out_path):
            os.remove(out_path)

        print(f"\n  Running agreement threshold={threshold}...")
        # Reduce workers for agreement (it generates K samples per step)
        agreement_workers = max(1, args.max_workers // 4)
        results = []
        with ThreadPoolExecutor(max_workers=agreement_workers) as executor:
            futures = {
                executor.submit(process_trajectory_agreement, t, adapter, args, threshold, args.agreement_k): t
                for t in trajectories
            }
            for future in as_completed(futures):
                try:
                    r = future.result()
                    results.append(r)
                    append_jsonl(r, out_path, result_lock)
                    if len(results) % 10 == 0:
                        acc = sum(r['total_correct'] for r in results) / max(sum(len(r['step_results']) for r in results), 1)
                        print(f"    [{label}] {len(results)}/{len(trajectories)} | StepAcc={acc:.3f}")
                except Exception as e:
                    print(f"    [{label}] Exception: {e}")

        all_results[label] = results
    return all_results


def analyze_results(all_results, output_dir):
    """Pareto analysis: StepAcc vs planner call count."""
    print("\n" + "=" * 70)
    print("E3: Replanning Frequency Results")
    print("=" * 70)

    summary = {'conditions': {}}

    # Compute metrics for each condition
    for label in sorted(all_results.keys()):
        results = all_results[label]
        metrics = compute_replan_metrics(results)
        summary['conditions'][label] = metrics

    # Print Pareto table
    print(f"\n  {'Condition':25s} {'StepAcc':>8s} {'TypeAcc':>8s} {'TSR':>6s} {'AvgCalls':>9s} {'Efficiency':>11s}")
    print("  " + "-" * 75)

    pareto_points = []
    for label in sorted(summary['conditions'].keys()):
        m = summary['conditions'][label]
        avg_calls = m['avg_planner_calls_per_traj']
        efficiency = m['step_accuracy'] / avg_calls if avg_calls > 0 else 0
        print(f"  {label:25s} {m['step_accuracy']:8.4f} {m['type_accuracy']:8.4f} {m['tsr']:6.4f} {avg_calls:9.1f} {efficiency:11.4f}")
        pareto_points.append({
            'label': label,
            'step_accuracy': m['step_accuracy'],
            'avg_planner_calls': avg_calls,
            'efficiency': efficiency,
        })

    # Identify Pareto-optimal points
    print("\n--- Pareto Frontier ---")
    pareto_points.sort(key=lambda x: x['avg_planner_calls'])
    frontier = []
    best_acc = -1
    for p in pareto_points:
        if p['step_accuracy'] > best_acc:
            frontier.append(p)
            best_acc = p['step_accuracy']
    for p in frontier:
        print(f"  {p['label']:25s} StepAcc={p['step_accuracy']:.4f} Calls={p['avg_planner_calls']:.1f}")
    summary['pareto_frontier'] = frontier

    # Per-length breakdown
    print("\n--- StepAcc by Length Bucket ---")
    for bucket in ['long(8-15)', 'vlong(16+)']:
        row = f"  {bucket:15s}"
        for label in sorted(summary['conditions'].keys()):
            bm = summary['conditions'][label].get('length_bucket_stats', {}).get(bucket, {})
            if bm:
                row += f"  {label}={bm['step_accuracy']:.3f}"
        print(row)

    save_json(summary, os.path.join(output_dir, 'summary.json'))
    print(f"\nSummary saved to {os.path.join(output_dir, 'summary.json')}")


def main():
    parser = argparse.ArgumentParser(description="E3: Replanning Frequency Sweep")
    parser.add_argument("--dataset", type=str, required=True, choices=['ac', 'gui360'])
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--min_steps", type=int, default=8, help="Minimum trajectory length")
    parser.add_argument("--run_fixed", action='store_true', default=True, help="Run fixed interval sweep")
    parser.add_argument("--run_agreement", action='store_true', default=False,
                        help="Run agreement threshold sweep (slow: K samples per step)")
    parser.add_argument("--agreement_k", type=int, default=5, help="K for agreement sampling")
    parser.add_argument("--fixed_intervals", type=int, nargs='+', default=FIXED_INTERVALS)
    parser.add_argument("--agreement_thresholds", type=float, nargs='+', default=AGREEMENT_THRESHOLDS)
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

    # Filter to long trajectories
    long_trajs = [t for t in trajectories if t.num_steps >= args.min_steps]
    print(f"Loaded {len(trajectories)} trajectories, {len(long_trajs)} have {args.min_steps}+ steps.")
    print(f"Dataset: {args.dataset} | Workers: {args.max_workers}")

    all_results = {}

    # Also run oracle as reference
    print("\n=== Running oracle boundary (reference) ===")
    oracle_out = os.path.join(args.output_dir, 'trajectory_results_oracle.jsonl')
    if os.path.exists(oracle_out):
        os.remove(oracle_out)
    oracle_results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                lambda t: (lambda r: (append_jsonl(r, oracle_out, result_lock), r)[-1])(
                    run_replan_trajectory(t, adapter, args, detect_oracle_boundaries(t), PROTOCOL)
                ), t
            ): t for t in long_trajs
        }
        for future in as_completed(futures):
            try:
                r = future.result()
                r['method'] = 'oracle'
                r['length_bucket'] = length_bucket(r['num_steps'])
                oracle_results.append(r)
            except Exception as e:
                print(f"  [oracle] Exception: {e}")
    all_results['oracle'] = oracle_results

    if args.run_fixed:
        print("\n=== Fixed Interval Sweep ===")
        fixed_results = run_fixed_sweep(long_trajs, adapter, args)
        all_results.update(fixed_results)

    if args.run_agreement:
        print("\n=== Agreement Threshold Sweep ===")
        agreement_results = run_agreement_sweep(long_trajs, adapter, args)
        all_results.update(agreement_results)

    if all_results:
        analyze_results(all_results, args.output_dir)
    else:
        print("No results collected.")


if __name__ == "__main__":
    main()
