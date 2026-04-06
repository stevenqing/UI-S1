#!/usr/bin/env python3
"""
T0.4: PAMARL Validation Evaluation Script

Evaluates PAMARL validation checkpoints on GUI-360 test set with:
1. Per-length-bucket TSR (short/medium/long/vlong)
2. func_match, coord_match, per-action-type accuracy
3. Near-miss error analysis (click<->type confusion count)
4. Comparison table across conditions A/B/C

Usage:
    # Evaluate a single checkpoint
    python scripts/eval/eval_pamarl_validation.py \
        --checkpoint checkpoints/pamarl_validation/baseline_A_12345/global_step_50 \
        --condition A

    # Compare all conditions
    python scripts/eval/eval_pamarl_validation.py \
        --compare \
        --checkpoint_a checkpoints/pamarl_validation/baseline_A_12345/global_step_50 \
        --checkpoint_b checkpoints/pamarl_validation/nearmiss_B_12346/global_step_50 \
        --checkpoint_c checkpoints/pamarl_validation/pamarl_C_12347/global_step_50
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Length bucket definitions (same as scientific_plan)
LENGTH_BUCKETS = {
    'short': (1, 3),
    'medium': (4, 7),
    'long': (8, 15),
    'vlong': (16, 999),
}


def load_eval_results(result_path):
    """Load evaluation results from a JSONL file."""
    results = []
    with open(result_path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def get_length_bucket(num_steps):
    """Get length bucket for a trajectory."""
    for bucket, (lo, hi) in LENGTH_BUCKETS.items():
        if lo <= num_steps <= hi:
            return bucket
    return 'vlong'


def compute_per_step_metrics(results):
    """Compute per-step metrics from evaluation results.

    Expects each result to have:
    - trajectory_id
    - num_steps
    - steps: list of {pred_action, gt_action, type_match, extract_match, coord_score, ...}
    """
    # Per-trajectory metrics
    traj_metrics = defaultdict(lambda: {
        'num_steps': 0,
        'type_matches': 0,
        'extract_matches': 0,
        'steps': [],
    })

    for r in results:
        tid = r.get('trajectory_id', r.get('traj_uid', ''))
        num_steps = r.get('num_steps', len(r.get('steps', [])))
        steps = r.get('steps', [])

        traj_metrics[tid]['num_steps'] = num_steps

        for step in steps:
            traj_metrics[tid]['steps'].append(step)
            if step.get('type_match', False):
                traj_metrics[tid]['type_matches'] += 1
            if step.get('extract_match', False):
                traj_metrics[tid]['extract_matches'] += 1

    return traj_metrics


def compute_tsr(traj_metrics):
    """Compute Trajectory Success Rate (TSR) per length bucket.

    TSR = fraction of trajectories where ALL steps have extract_match=True.
    """
    bucket_stats = {b: {'total': 0, 'success': 0} for b in LENGTH_BUCKETS}
    overall = {'total': 0, 'success': 0}

    for tid, metrics in traj_metrics.items():
        num_steps = metrics['num_steps']
        bucket = get_length_bucket(num_steps)
        all_match = metrics['extract_matches'] == num_steps and num_steps > 0

        bucket_stats[bucket]['total'] += 1
        bucket_stats[bucket]['success'] += int(all_match)
        overall['total'] += 1
        overall['success'] += int(all_match)

    results = {}
    for bucket, stats in bucket_stats.items():
        results[bucket] = {
            'tsr': stats['success'] / max(stats['total'], 1) * 100,
            'total': stats['total'],
            'success': stats['success'],
        }
    results['overall'] = {
        'tsr': overall['success'] / max(overall['total'], 1) * 100,
        'total': overall['total'],
        'success': overall['success'],
    }

    return results


def compute_action_accuracy(traj_metrics):
    """Compute per-action-type accuracy."""
    action_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

    for tid, metrics in traj_metrics.items():
        for step in metrics['steps']:
            gt_action = step.get('gt_action_type', step.get('gt_action', {}).get('action', ''))
            type_match = step.get('type_match', False)
            action_stats[gt_action]['total'] += 1
            action_stats[gt_action]['correct'] += int(type_match)

    results = {}
    for action, stats in sorted(action_stats.items(), key=lambda x: -x[1]['total']):
        results[action] = {
            'accuracy': stats['correct'] / max(stats['total'], 1) * 100,
            'total': stats['total'],
            'correct': stats['correct'],
        }

    return results


def count_near_miss_errors(traj_metrics):
    """Count click<->type confusion errors."""
    confusion = defaultdict(int)

    for tid, metrics in traj_metrics.items():
        for step in metrics['steps']:
            gt_action = step.get('gt_action_type', step.get('gt_action', {}).get('action', ''))
            pred_action = step.get('pred_action_type', step.get('pred_action', {}).get('action', ''))
            if gt_action != pred_action and gt_action and pred_action:
                confusion[(gt_action, pred_action)] += 1

    return confusion


def compute_aggregate_metrics(traj_metrics):
    """Compute aggregate metrics: func_match, coord_match."""
    total_steps = 0
    func_matches = 0
    coord_sum = 0.0

    for tid, metrics in traj_metrics.items():
        for step in metrics['steps']:
            total_steps += 1
            if step.get('type_match', False):
                func_matches += 1
            coord_sum += step.get('coord_score', 0.0)

    return {
        'func_match': func_matches / max(total_steps, 1) * 100,
        'coord_match': coord_sum / max(total_steps, 1) * 100,
        'total_steps': total_steps,
    }


def print_report(condition, tsr, action_acc, confusion, agg_metrics):
    """Print a formatted evaluation report."""
    print(f"\n{'='*60}")
    print(f"  PAMARL Validation — Condition {condition}")
    print(f"{'='*60}")

    print(f"\n--- Aggregate Metrics ---")
    print(f"  func_match:  {agg_metrics['func_match']:.2f}%")
    print(f"  coord_match: {agg_metrics['coord_match']:.2f}%")
    print(f"  total_steps: {agg_metrics['total_steps']}")

    print(f"\n--- TSR by Length Bucket ---")
    print(f"  {'Bucket':<10} {'TSR':>8} {'Success':>8} {'Total':>8}")
    print(f"  {'-'*34}")
    for bucket in list(LENGTH_BUCKETS.keys()) + ['overall']:
        stats = tsr[bucket]
        print(f"  {bucket:<10} {stats['tsr']:>7.2f}% {stats['success']:>8} {stats['total']:>8}")

    print(f"\n--- Per-Action-Type Accuracy (top 10) ---")
    print(f"  {'Action':<20} {'Accuracy':>10} {'Correct':>8} {'Total':>8}")
    print(f"  {'-'*46}")
    for action, stats in list(action_acc.items())[:10]:
        print(f"  {action:<20} {stats['accuracy']:>9.2f}% {stats['correct']:>8} {stats['total']:>8}")

    print(f"\n--- Top Near-Miss Errors ---")
    sorted_conf = sorted(confusion.items(), key=lambda x: -x[1])[:10]
    print(f"  {'GT -> Pred':<30} {'Count':>8}")
    print(f"  {'-'*38}")
    for (gt, pred), count in sorted_conf:
        print(f"  {gt} -> {pred:<20} {count:>8}")

    # Highlight click<->type specifically
    ct_errors = confusion.get(('click', 'type'), 0) + confusion.get(('type', 'click'), 0)
    print(f"\n  click<->type errors: {ct_errors}")


def compare_conditions(results_dict):
    """Print comparison table across conditions A/B/C."""
    print(f"\n{'='*70}")
    print(f"  PAMARL Validation — Condition Comparison (A vs B vs C)")
    print(f"{'='*70}")

    # TSR comparison
    print(f"\n--- TSR by Length Bucket ---")
    header = f"  {'Bucket':<10}"
    for cond in sorted(results_dict.keys()):
        header += f" {cond:>10}"
    print(header)
    print(f"  {'-'*40}")

    for bucket in list(LENGTH_BUCKETS.keys()) + ['overall']:
        row = f"  {bucket:<10}"
        for cond in sorted(results_dict.keys()):
            tsr_val = results_dict[cond]['tsr'][bucket]['tsr']
            row += f" {tsr_val:>9.2f}%"
        print(row)

    # Delta table (vs baseline A)
    if 'A' in results_dict:
        print(f"\n--- TSR Delta (vs Baseline A) ---")
        header = f"  {'Bucket':<10}"
        for cond in sorted(results_dict.keys()):
            if cond != 'A':
                header += f" {cond}-A:>10"
        print(header)
        print(f"  {'-'*30}")

        for bucket in list(LENGTH_BUCKETS.keys()) + ['overall']:
            row = f"  {bucket:<10}"
            base_tsr = results_dict['A']['tsr'][bucket]['tsr']
            for cond in sorted(results_dict.keys()):
                if cond != 'A':
                    delta = results_dict[cond]['tsr'][bucket]['tsr'] - base_tsr
                    row += f" {delta:>+9.2f}%"
            print(row)

    # Core prediction: Long lift / Short lift
    if 'A' in results_dict and 'C' in results_dict:
        base_short = results_dict['A']['tsr']['short']['tsr']
        base_long = results_dict['A']['tsr']['long']['tsr']
        c_short = results_dict['C']['tsr']['short']['tsr']
        c_long = results_dict['C']['tsr']['long']['tsr']

        short_lift = c_short - base_short
        long_lift = c_long - base_long

        print(f"\n--- Core PAMARL Prediction ---")
        print(f"  Short TSR lift (C-A): {short_lift:+.2f}pp")
        print(f"  Long TSR lift (C-A):  {long_lift:+.2f}pp")
        if abs(short_lift) > 0.01:
            ratio = long_lift / short_lift
            print(f"  Long lift / Short lift: {ratio:.2f}")
            print(f"  Prediction (>2.0): {'PASS' if ratio > 2.0 else 'FAIL'}")
        else:
            print(f"  Short lift ~0 -> ratio undefined")

    # Aggregate comparison
    print(f"\n--- Aggregate Metrics ---")
    header = f"  {'Metric':<15}"
    for cond in sorted(results_dict.keys()):
        header += f" {cond:>10}"
    print(header)
    print(f"  {'-'*45}")

    for metric in ['func_match', 'coord_match']:
        row = f"  {metric:<15}"
        for cond in sorted(results_dict.keys()):
            val = results_dict[cond]['agg'][metric]
            row += f" {val:>9.2f}%"
        print(row)

    # click<->type error comparison
    row = f"  {'click<->type':<15}"
    for cond in sorted(results_dict.keys()):
        ct = results_dict[cond]['confusion'].get(('click', 'type'), 0)
        tc = results_dict[cond]['confusion'].get(('type', 'click'), 0)
        row += f" {ct + tc:>10}"
    print(row)


def evaluate_checkpoint(result_path, condition='A'):
    """Evaluate a single checkpoint's results."""
    results = load_eval_results(result_path)
    traj_metrics = compute_per_step_metrics(results)
    tsr = compute_tsr(traj_metrics)
    action_acc = compute_action_accuracy(traj_metrics)
    confusion = count_near_miss_errors(traj_metrics)
    agg = compute_aggregate_metrics(traj_metrics)

    print_report(condition, tsr, action_acc, confusion, agg)

    return {
        'tsr': tsr,
        'action_acc': action_acc,
        'confusion': confusion,
        'agg': agg,
    }


def main():
    parser = argparse.ArgumentParser(description='PAMARL Validation Evaluation')
    parser.add_argument('--result_file', type=str, help='Path to evaluation result JSONL')
    parser.add_argument('--condition', type=str, default='A', help='Condition label (A/B/C)')

    parser.add_argument('--compare', action='store_true', help='Compare multiple conditions')
    parser.add_argument('--result_a', type=str, help='Condition A result JSONL')
    parser.add_argument('--result_b', type=str, help='Condition B result JSONL')
    parser.add_argument('--result_c', type=str, help='Condition C result JSONL')

    parser.add_argument('--output', type=str, help='Output JSON path for structured results')

    args = parser.parse_args()

    if args.compare:
        results_dict = {}
        for cond, path in [('A', args.result_a), ('B', args.result_b), ('C', args.result_c)]:
            if path and os.path.exists(path):
                results = load_eval_results(path)
                traj_metrics = compute_per_step_metrics(results)
                results_dict[cond] = {
                    'tsr': compute_tsr(traj_metrics),
                    'action_acc': compute_action_accuracy(traj_metrics),
                    'confusion': count_near_miss_errors(traj_metrics),
                    'agg': compute_aggregate_metrics(traj_metrics),
                }

        compare_conditions(results_dict)

        if args.output:
            # Serialize for JSON (convert tuple keys)
            serializable = {}
            for cond, data in results_dict.items():
                serializable[cond] = {
                    'tsr': data['tsr'],
                    'agg': data['agg'],
                    'confusion': {f"{k[0]}->{k[1]}": v for k, v in data['confusion'].items()},
                }
            with open(args.output, 'w') as f:
                json.dump(serializable, f, indent=2)
            print(f"\nResults saved to {args.output}")

    elif args.result_file:
        result = evaluate_checkpoint(args.result_file, args.condition)

        if args.output:
            serializable = {
                'tsr': result['tsr'],
                'agg': result['agg'],
                'confusion': {f"{k[0]}->{k[1]}": v for k, v in result['confusion'].items()},
            }
            with open(args.output, 'w') as f:
                json.dump(serializable, f, indent=2)
            print(f"\nResults saved to {args.output}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
