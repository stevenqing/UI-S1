"""E2: Communication Protocol Ablation — What info should the planner send?

All conditions use oracle boundaries (GT action type transitions).
Tests 5 planner communication protocols at each boundary:
  1. none:                context reset only, goal as instruction
  2. nl_instruction:      full GT step instruction (oracle upper bound)
  3. structured:          "{action_type}: {target}"
  4. type_only:           "Perform a {action_type} action."
  5. structured_progress: structured + completed/remaining summary

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
    get_adapter, detect_oracle_boundaries, run_replan_trajectory,
    compute_replan_metrics, save_json, append_jsonl, _json_default,
    length_bucket,
)

PROTOCOLS = ['none', 'nl_instruction', 'structured', 'type_only', 'structured_progress']

result_lock = Lock()


def process_trajectory_for_protocol(trajectory, adapter, args, protocol):
    """Run one trajectory with oracle boundaries and the given protocol."""
    boundaries = detect_oracle_boundaries(trajectory)
    result = run_replan_trajectory(trajectory, adapter, args, boundaries, protocol)
    result['protocol'] = protocol
    result['length_bucket'] = length_bucket(trajectory.num_steps)
    result['n_oracle_boundaries'] = len(boundaries)

    append_jsonl(
        result,
        os.path.join(args.output_dir, f'trajectory_results_{protocol}.jsonl'),
        result_lock,
    )
    return result


def run_protocol(protocol, trajectories, adapter, args):
    """Run all trajectories for a single protocol."""
    out_path = os.path.join(args.output_dir, f'trajectory_results_{protocol}.jsonl')
    if os.path.exists(out_path):
        os.remove(out_path)

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_trajectory_for_protocol, t, adapter, args, protocol): t
            for t in trajectories
        }
        for future in as_completed(futures):
            try:
                r = future.result()
                results.append(r)
                if len(results) % 50 == 0:
                    acc = sum(r['total_correct'] for r in results) / max(sum(len(r['step_results']) for r in results), 1)
                    print(f"  [{protocol}] {len(results)}/{len(trajectories)} | StepAcc={acc:.3f}")
            except Exception as e:
                print(f"  [{protocol}] Exception: {e}")
    return results


def analyze_results(all_results, output_dir):
    """Compare protocols."""
    print("\n" + "=" * 70)
    print("E2: Communication Protocol Results (Oracle Boundaries)")
    print("=" * 70)

    summary = {'protocols': {}}

    for protocol in PROTOCOLS:
        results = all_results[protocol]
        metrics = compute_replan_metrics(results)
        summary['protocols'][protocol] = metrics

        print(f"\n  {protocol}:")
        print(f"    StepAcc:     {metrics['step_accuracy']:.4f} ({metrics['total_correct']}/{metrics['total_steps']})")
        print(f"    TypeAcc:     {metrics['type_accuracy']:.4f}")
        print(f"    TSR:         {metrics['tsr']:.4f}")
        print(f"    BoundaryAcc: {metrics['boundary_accuracy']:.4f} (n={metrics['boundary_total']})")
        print(f"    WithinAcc:   {metrics['within_phase_accuracy']:.4f} (n={metrics['within_phase_total']})")
        print(f"    AvgPlannerCalls: {metrics['avg_planner_calls_per_traj']:.1f}")

    # Per-length breakdown for each protocol
    print("\n--- StepAcc by Length Bucket ---")
    for bucket in ['short(1-3)', 'medium(4-7)', 'long(8-15)', 'vlong(16+)']:
        row = f"  {bucket:15s}"
        for protocol in PROTOCOLS:
            metrics = summary['protocols'][protocol]
            bm = metrics.get('length_bucket_stats', {}).get(bucket, {})
            if bm:
                row += f"  {protocol}={bm['step_accuracy']:.3f}(n={bm['n']})"
        print(row)

    # Pairwise comparison: structured vs others
    print("\n--- Pairwise: Structured vs Others ---")
    for protocol in PROTOCOLS:
        if protocol == 'structured':
            continue
        both = set(r['trajectory_id'] for r in all_results['structured']) & \
               set(r['trajectory_id'] for r in all_results[protocol])
        struct_map = {r['trajectory_id']: r for r in all_results['structured']}
        other_map = {r['trajectory_id']: r for r in all_results[protocol]}
        struct_better = 0
        other_better = 0
        tied = 0
        for tid in both:
            s_acc = struct_map[tid]['step_accuracy']
            o_acc = other_map[tid]['step_accuracy']
            if s_acc > o_acc:
                struct_better += 1
            elif o_acc > s_acc:
                other_better += 1
            else:
                tied += 1
        print(f"  vs {protocol}: structured_better={struct_better} other_better={other_better} tied={tied}")

    save_json(summary, os.path.join(output_dir, 'summary.json'))
    print(f"\nSummary saved to {os.path.join(output_dir, 'summary.json')}")


def main():
    parser = argparse.ArgumentParser(description="E2: Communication Protocol Ablation")
    parser.add_argument("--dataset", type=str, required=True, choices=['ac', 'gui360'])
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--protocols", type=str, nargs='+', default=PROTOCOLS,
                        choices=PROTOCOLS, help="Which protocols to evaluate")
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
    for protocol in args.protocols:
        print(f"\n{'='*40}")
        print(f"Running protocol: {protocol}")
        print(f"{'='*40}")
        all_results[protocol] = run_protocol(protocol, trajectories, adapter, args)

    analyze_results(all_results, args.output_dir)


if __name__ == "__main__":
    main()
