"""Q4: Action Inertia Mechanism Analysis

Why does 88.5% boundary detection not prevent 22.5% action inertia?

Analyses:
  1. Inertia by run length: longer GT action runs → higher inertia at boundary?
  2. Inertia by transition pair: (prev_type → gt_type) error rates
  3. Inertia by trajectory position: does inertia increase later?
  4. Baseline vs oracle comparison: does step_instruction reduce inertia?
  5. Non-boundary baseline: error rate at continuation steps for comparison

Data sources:
  - baseline: outputs/eval_a_ac/.../trajectory_results.jsonl (2905 steps)
  - oracle:   outputs/eval_context_subtask/.../trajectory_results.jsonl (4106 steps)
  - dataset:  datasets/android_control_evaluation_std.jsonl (1543 episodes, full GT)
"""

import argparse
import json
import os
import sys
from collections import defaultdict, Counter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

ALL_ACTION_TYPES = ['click', 'long_press', 'swipe', 'type', 'open', 'system_button', 'wait']


def save_json(data, path):
    """Save dict to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import numpy as np
    def _default(obj):
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_default)


def load_trajectory_results(path):
    """Load trajectory results and index by episode_id."""
    results = {}
    with open(path) as f:
        for line in f:
            ep = json.loads(line.strip())
            results[ep['episode_id']] = ep
    return results


def load_dataset_gt(path):
    """Load dataset and build GT action type sequences per episode."""
    gt_sequences = {}
    with open(path) as f:
        for line in f:
            ep = json.loads(line.strip())
            eid = ep['episode_id']
            types = [s['action_content']['action'] for s in ep['steps']]
            gt_sequences[eid] = types
    return gt_sequences


def compute_run_length_before_boundary(gt_types, boundary_idx):
    """Count consecutive same-type GT actions before this boundary."""
    if boundary_idx == 0:
        return 0
    prev_type = gt_types[boundary_idx - 1]
    run = 1
    for i in range(boundary_idx - 2, -1, -1):
        if gt_types[i] == prev_type:
            run += 1
        else:
            break
    return run


def analyze_condition(results_by_ep, gt_sequences, condition_name):
    """Analyze inertia patterns for one condition (baseline or oracle)."""
    # Collect boundary and non-boundary steps
    boundary_steps = []
    nonboundary_steps = []
    all_steps = []

    for eid, ep in results_by_ep.items():
        gt_types = gt_sequences.get(eid, [])
        if not gt_types:
            continue

        for sr in ep.get('step_results', []):
            step_num = sr['step_num']
            gt_type = sr['gt_action_type']
            pred_action = sr.get('pred_action', {})
            pred_type = pred_action.get('action', 'unknown') if pred_action else 'unknown'
            type_match = sr.get('type_match', False)

            # Determine if boundary
            is_boundary = False
            prev_type = None
            if step_num > 0 and step_num < len(gt_types):
                prev_type = gt_types[step_num - 1]
                is_boundary = (prev_type != gt_type)

            run_length = 0
            if is_boundary:
                run_length = compute_run_length_before_boundary(gt_types, step_num)

            step_info = {
                'episode_id': eid,
                'step_num': step_num,
                'gt_type': gt_type,
                'pred_type': pred_type,
                'prev_gt_type': prev_type,
                'type_match': type_match,
                'is_boundary': is_boundary,
                'run_length': run_length,
                'num_steps': len(gt_types),
                'position_frac': step_num / max(len(gt_types) - 1, 1),
            }
            all_steps.append(step_info)
            if is_boundary:
                boundary_steps.append(step_info)
            elif step_num > 0:  # non-boundary continuation (skip step 0)
                nonboundary_steps.append(step_info)

    results = {
        'condition': condition_name,
        'total_steps': len(all_steps),
        'boundary_steps': len(boundary_steps),
        'nonboundary_steps': len(nonboundary_steps),
    }

    # --- Analysis 1: Inertia by run length ---
    run_bins = defaultdict(lambda: {'total': 0, 'inertia': 0, 'type_errors': 0})
    for s in boundary_steps:
        rl = s['run_length']
        bin_key = min(rl, 5)  # bin: 1, 2, 3, 4, 5+
        run_bins[bin_key]['total'] += 1
        if not s['type_match']:
            run_bins[bin_key]['type_errors'] += 1
            # Check if prediction matches previous type (true inertia)
            if s['pred_type'] == s['prev_gt_type']:
                run_bins[bin_key]['inertia'] += 1

    inertia_by_run = {}
    for rl in sorted(run_bins.keys()):
        d = run_bins[rl]
        label = f"{rl}" if rl < 5 else "5+"
        inertia_by_run[label] = {
            'total': d['total'],
            'type_errors': d['type_errors'],
            'inertia_count': d['inertia'],
            'type_error_rate': d['type_errors'] / d['total'] if d['total'] else 0,
            'inertia_rate': d['inertia'] / d['total'] if d['total'] else 0,
        }
    results['inertia_by_run_length'] = inertia_by_run

    # --- Analysis 2: Inertia by transition pair ---
    pair_stats = defaultdict(lambda: {'total': 0, 'type_errors': 0, 'inertia': 0})
    for s in boundary_steps:
        pair = f"{s['prev_gt_type']}→{s['gt_type']}"
        pair_stats[pair]['total'] += 1
        if not s['type_match']:
            pair_stats[pair]['type_errors'] += 1
            if s['pred_type'] == s['prev_gt_type']:
                pair_stats[pair]['inertia'] += 1

    transition_table = {}
    for pair in sorted(pair_stats.keys(), key=lambda x: pair_stats[x]['total'], reverse=True):
        d = pair_stats[pair]
        transition_table[pair] = {
            'total': d['total'],
            'type_errors': d['type_errors'],
            'inertia_count': d['inertia'],
            'type_error_rate': d['type_errors'] / d['total'] if d['total'] else 0,
            'inertia_rate': d['inertia'] / d['total'] if d['total'] else 0,
        }
    results['inertia_by_transition_pair'] = transition_table

    # --- Analysis 3: Inertia by trajectory position ---
    pos_bins = defaultdict(lambda: {'total': 0, 'type_errors': 0, 'inertia': 0})
    for s in boundary_steps:
        # Bin position into quintiles
        pf = s['position_frac']
        if pf < 0.2:
            pbin = '0-20%'
        elif pf < 0.4:
            pbin = '20-40%'
        elif pf < 0.6:
            pbin = '40-60%'
        elif pf < 0.8:
            pbin = '60-80%'
        else:
            pbin = '80-100%'

        pos_bins[pbin]['total'] += 1
        if not s['type_match']:
            pos_bins[pbin]['type_errors'] += 1
            if s['pred_type'] == s['prev_gt_type']:
                pos_bins[pbin]['inertia'] += 1

    inertia_by_position = {}
    for pbin in ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']:
        d = pos_bins[pbin]
        inertia_by_position[pbin] = {
            'total': d['total'],
            'type_errors': d['type_errors'],
            'inertia_count': d['inertia'],
            'type_error_rate': d['type_errors'] / d['total'] if d['total'] else 0,
            'inertia_rate': d['inertia'] / d['total'] if d['total'] else 0,
        }
    results['inertia_by_position'] = inertia_by_position

    # --- Analysis 5: Non-boundary error rates ---
    nb_type_errors = sum(1 for s in nonboundary_steps if not s['type_match'])
    results['nonboundary_error_rate'] = nb_type_errors / len(nonboundary_steps) if nonboundary_steps else 0
    results['boundary_error_rate'] = (
        sum(1 for s in boundary_steps if not s['type_match']) / len(boundary_steps)
        if boundary_steps else 0
    )

    # Summary: which transitions are most problematic
    top_inertia_pairs = sorted(
        transition_table.items(),
        key=lambda x: x[1]['inertia_rate'],
        reverse=True
    )[:10]
    results['top_inertia_pairs'] = {k: v for k, v in top_inertia_pairs}

    return results


def main():
    parser = argparse.ArgumentParser(description="Q4: Action Inertia Mechanism Analysis")
    parser.add_argument("--baseline_results", type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'eval_a_ac', 'Qwen2.5-VL-7B', 'trajectory_results.jsonl'))
    parser.add_argument("--oracle_results", type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'eval_context_subtask', 'Qwen2.5-VL-7B', 'trajectory_results.jsonl'))
    parser.add_argument("--dataset", type=str,
                        default=os.path.join(PROJECT_ROOT, 'datasets', 'android_control_evaluation_std.jsonl'))
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'analysis_q4_inertia'))
    args = parser.parse_args()

    print("Loading data...")
    baseline_results = load_trajectory_results(args.baseline_results)
    oracle_results = load_trajectory_results(args.oracle_results)
    gt_sequences = load_dataset_gt(args.dataset)

    print(f"Baseline: {len(baseline_results)} episodes, Oracle: {len(oracle_results)} episodes, GT: {len(gt_sequences)} episodes")

    # Analyze each condition
    print("\n===== Baseline Condition =====")
    baseline_analysis = analyze_condition(baseline_results, gt_sequences, 'baseline')

    print("\n===== Oracle Condition =====")
    oracle_analysis = analyze_condition(oracle_results, gt_sequences, 'oracle')

    # --- Analysis 4: Baseline vs oracle comparison ---
    comparison = {
        'baseline_boundary_error_rate': baseline_analysis['boundary_error_rate'],
        'oracle_boundary_error_rate': oracle_analysis['boundary_error_rate'],
        'baseline_nonboundary_error_rate': baseline_analysis['nonboundary_error_rate'],
        'oracle_nonboundary_error_rate': oracle_analysis['nonboundary_error_rate'],
        'boundary_error_reduction': baseline_analysis['boundary_error_rate'] - oracle_analysis['boundary_error_rate'],
        'nonboundary_error_reduction': baseline_analysis['nonboundary_error_rate'] - oracle_analysis['nonboundary_error_rate'],
    }

    # Per-transition comparison
    per_transition_comparison = {}
    all_pairs = set(list(baseline_analysis['inertia_by_transition_pair'].keys()) +
                    list(oracle_analysis['inertia_by_transition_pair'].keys()))
    for pair in sorted(all_pairs):
        bl = baseline_analysis['inertia_by_transition_pair'].get(pair, {})
        oc = oracle_analysis['inertia_by_transition_pair'].get(pair, {})
        per_transition_comparison[pair] = {
            'baseline_inertia_rate': bl.get('inertia_rate', 0),
            'oracle_inertia_rate': oc.get('inertia_rate', 0),
            'baseline_n': bl.get('total', 0),
            'oracle_n': oc.get('total', 0),
            'inertia_reduction': bl.get('inertia_rate', 0) - oc.get('inertia_rate', 0),
        }
    comparison['per_transition'] = per_transition_comparison

    # Final output
    output = {
        'baseline': baseline_analysis,
        'oracle': oracle_analysis,
        'comparison': comparison,
        'summary': {
            'key_finding': (
                f"Boundary type error rate: baseline={baseline_analysis['boundary_error_rate']:.3f}, "
                f"oracle={oracle_analysis['boundary_error_rate']:.3f}. "
                f"Non-boundary type error rate: baseline={baseline_analysis['nonboundary_error_rate']:.3f}, "
                f"oracle={oracle_analysis['nonboundary_error_rate']:.3f}."
            ),
        }
    }

    # Print summary
    print("\n" + "=" * 60)
    print("Q4 SUMMARY")
    print("=" * 60)
    print(f"Baseline boundary error rate: {baseline_analysis['boundary_error_rate']:.3f}")
    print(f"Oracle boundary error rate:   {oracle_analysis['boundary_error_rate']:.3f}")
    print(f"Baseline non-boundary error:  {baseline_analysis['nonboundary_error_rate']:.3f}")
    print(f"Oracle non-boundary error:    {oracle_analysis['nonboundary_error_rate']:.3f}")
    print(f"\nInertia by run length (baseline):")
    for rl, d in sorted(baseline_analysis['inertia_by_run_length'].items()):
        print(f"  Run={rl}: inertia_rate={d['inertia_rate']:.3f} (n={d['total']})")
    print(f"\nTop inertia transition pairs (baseline):")
    for pair, d in list(baseline_analysis['top_inertia_pairs'].items())[:5]:
        print(f"  {pair}: inertia={d['inertia_rate']:.3f}, type_err={d['type_error_rate']:.3f} (n={d['total']})")

    save_json(output, os.path.join(args.output_dir, 'q4_results.json'))
    print(f"\nResults saved to {os.path.join(args.output_dir, 'q4_results.json')}")


if __name__ == "__main__":
    main()
