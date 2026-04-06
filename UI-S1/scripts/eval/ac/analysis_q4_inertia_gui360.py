"""Q4: Action Inertia Mechanism Analysis — GUI-360 Version

Adapts AC Q4 analysis to GUI-360 dataset.
Key differences from AC:
  - Data: flat step list (baseline.json/sft.json) not trajectory JSONL
  - GT sequences from gui360_test.jsonl (keyed by execution_id)
  - ~16 action types (click, type, drag, select_text, select_table_range, etc.) vs AC 7
  - Empty gt_type ("") → labeled 'unknown' and filtered from boundary analysis
  - SFT vs baseline comparison replaces AC's oracle vs baseline

Data sources:
  - baseline: outputs/gui360_eval_results/baseline.json (2201 steps, flat)
  - sft:      outputs/gui360_eval_results/sft.json (2201 steps, flat)
  - dataset:  datasets/GUI-360/rl_data/gui360_test.jsonl (GT action sequences)
"""

import argparse
import json
import os
import sys
from collections import defaultdict, Counter

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    def _default(obj):
        if hasattr(obj, 'item'): return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_default)


def load_eval_results(path):
    """Load GUI-360 eval results (flat step list inside JSON wrapper)."""
    with open(path) as f:
        data = json.load(f)
    # Group by episode_id, sort by step_num
    episodes = defaultdict(list)
    for r in data['results']:
        episodes[r['episode_id']].append(r)
    for eid in episodes:
        episodes[eid].sort(key=lambda x: x['step_num'])
    return dict(episodes), data


def load_dataset_gt(path):
    """Load gui360_test.jsonl and build GT action type sequences keyed by execution_id."""
    gt_sequences = {}
    with open(path) as f:
        for line in f:
            ep = json.loads(line.strip())
            eid = ep['execution_id']
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


def analyze_condition(episodes_by_ep, gt_sequences, condition_name):
    """Analyze inertia patterns for one condition (baseline or SFT)."""
    boundary_steps = []
    nonboundary_steps = []
    all_steps = []
    skipped_unknown = 0

    for eid, steps in episodes_by_ep.items():
        gt_types = gt_sequences.get(eid, [])
        if not gt_types:
            continue

        for sr in steps:
            step_num = sr['step_num']
            gt_type = sr['gt_type']
            pred_type = sr['pred_type']
            type_match = sr.get('type_match', False)

            # Filter empty gt_type
            if not gt_type or gt_type.strip() == '':
                skipped_unknown += 1
                continue

            # Determine boundary using GT sequence
            is_boundary = False
            prev_type = None
            if step_num > 0 and step_num < len(gt_types):
                prev_type = gt_types[step_num - 1]
                if prev_type and prev_type.strip():
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
            elif step_num > 0:
                nonboundary_steps.append(step_info)

    results = {
        'condition': condition_name,
        'total_steps': len(all_steps),
        'boundary_steps': len(boundary_steps),
        'nonboundary_steps': len(nonboundary_steps),
        'skipped_unknown_gt': skipped_unknown,
    }

    # --- Analysis 1: Inertia by run length ---
    run_bins = defaultdict(lambda: {'total': 0, 'inertia': 0, 'type_errors': 0})
    for s in boundary_steps:
        rl = s['run_length']
        bin_key = min(rl, 5)
        run_bins[bin_key]['total'] += 1
        if not s['type_match']:
            run_bins[bin_key]['type_errors'] += 1
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
        if s['prev_gt_type']:
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

    # Top inertia pairs
    top_inertia_pairs = sorted(
        transition_table.items(),
        key=lambda x: x[1]['inertia_rate'],
        reverse=True
    )[:10]
    results['top_inertia_pairs'] = {k: v for k, v in top_inertia_pairs}

    return results


def main():
    parser = argparse.ArgumentParser(description="Q4: Action Inertia — GUI-360")
    parser.add_argument("--baseline_results", type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'gui360_eval_results', 'baseline.json'))
    parser.add_argument("--sft_results", type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'gui360_eval_results', 'sft.json'))
    parser.add_argument("--dataset", type=str,
                        default=os.path.join(PROJECT_ROOT, 'datasets', 'GUI-360', 'rl_data', 'gui360_train.jsonl'))
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'analysis_q4_inertia_gui360'))
    args = parser.parse_args()

    print("Loading data...")
    baseline_eps, baseline_meta = load_eval_results(args.baseline_results)
    sft_eps, sft_meta = load_eval_results(args.sft_results)
    gt_sequences = load_dataset_gt(args.dataset)

    print(f"Baseline: {len(baseline_eps)} episodes ({baseline_meta['n']} steps, type_acc={baseline_meta['type_acc']:.3f})")
    print(f"SFT: {len(sft_eps)} episodes ({sft_meta['n']} steps, type_acc={sft_meta['type_acc']:.3f})")
    print(f"GT sequences: {len(gt_sequences)} episodes")

    # Action type distribution
    all_gt_types = Counter()
    for steps in baseline_eps.values():
        for s in steps:
            gt = s['gt_type']
            all_gt_types[gt if gt else 'unknown'] += 1
    print(f"\nGT type distribution: {dict(all_gt_types.most_common())}")

    # Analyze each condition
    print("\n===== Baseline Condition =====")
    baseline_analysis = analyze_condition(baseline_eps, gt_sequences, 'baseline')

    print("\n===== SFT Condition =====")
    sft_analysis = analyze_condition(sft_eps, gt_sequences, 'sft')

    # --- Analysis 4: Baseline vs SFT comparison ---
    comparison = {
        'baseline_boundary_error_rate': baseline_analysis['boundary_error_rate'],
        'sft_boundary_error_rate': sft_analysis['boundary_error_rate'],
        'baseline_nonboundary_error_rate': baseline_analysis['nonboundary_error_rate'],
        'sft_nonboundary_error_rate': sft_analysis['nonboundary_error_rate'],
        'boundary_error_reduction': baseline_analysis['boundary_error_rate'] - sft_analysis['boundary_error_rate'],
        'nonboundary_error_reduction': baseline_analysis['nonboundary_error_rate'] - sft_analysis['nonboundary_error_rate'],
    }

    # Per-transition comparison
    per_transition_comparison = {}
    all_pairs = set(list(baseline_analysis['inertia_by_transition_pair'].keys()) +
                    list(sft_analysis['inertia_by_transition_pair'].keys()))
    for pair in sorted(all_pairs):
        bl = baseline_analysis['inertia_by_transition_pair'].get(pair, {})
        sf = sft_analysis['inertia_by_transition_pair'].get(pair, {})
        per_transition_comparison[pair] = {
            'baseline_inertia_rate': bl.get('inertia_rate', 0),
            'sft_inertia_rate': sf.get('inertia_rate', 0),
            'baseline_n': bl.get('total', 0),
            'sft_n': sf.get('total', 0),
            'inertia_reduction': bl.get('inertia_rate', 0) - sf.get('inertia_rate', 0),
        }
    comparison['per_transition'] = per_transition_comparison

    # Final output
    output = {
        'baseline': baseline_analysis,
        'sft': sft_analysis,
        'comparison': comparison,
        'gt_type_distribution': dict(all_gt_types.most_common()),
        'summary': {
            'key_finding': (
                f"Boundary type error rate: baseline={baseline_analysis['boundary_error_rate']:.3f}, "
                f"sft={sft_analysis['boundary_error_rate']:.3f}. "
                f"Non-boundary type error rate: baseline={baseline_analysis['nonboundary_error_rate']:.3f}, "
                f"sft={sft_analysis['nonboundary_error_rate']:.3f}."
            ),
            'boundary_vs_nonboundary_ratio_baseline': (
                baseline_analysis['boundary_error_rate'] / baseline_analysis['nonboundary_error_rate']
                if baseline_analysis['nonboundary_error_rate'] > 0 else float('inf')
            ),
            'boundary_vs_nonboundary_ratio_sft': (
                sft_analysis['boundary_error_rate'] / sft_analysis['nonboundary_error_rate']
                if sft_analysis['nonboundary_error_rate'] > 0 else float('inf')
            ),
        }
    }

    # Print summary
    print("\n" + "=" * 60)
    print("Q4 SUMMARY — GUI-360")
    print("=" * 60)
    print(f"Baseline boundary error rate: {baseline_analysis['boundary_error_rate']:.3f}")
    print(f"SFT boundary error rate:      {sft_analysis['boundary_error_rate']:.3f}")
    print(f"Baseline non-boundary error:  {baseline_analysis['nonboundary_error_rate']:.3f}")
    print(f"SFT non-boundary error:       {sft_analysis['nonboundary_error_rate']:.3f}")
    print(f"Boundary/non-boundary ratio (baseline): {output['summary']['boundary_vs_nonboundary_ratio_baseline']:.2f}x")
    print(f"Boundary/non-boundary ratio (SFT):      {output['summary']['boundary_vs_nonboundary_ratio_sft']:.2f}x")
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
