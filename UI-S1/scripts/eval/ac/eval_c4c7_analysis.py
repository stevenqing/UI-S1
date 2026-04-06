"""Eval C4+C7: Multi-Sample Analysis (Offline).

Analyzes agreement, oracle accuracy, DBSCAN clustering, and adaptive K strategies.
"""

import argparse
import os
from collections import defaultdict, Counter

import numpy as np

from ac_utils import load_jsonl, save_json, categorize_action, ALL_ACTION_TYPES


def main(args):
    results = load_jsonl(args.input_file)
    print(f"Loaded {len(results)} episodes.")

    total_steps = 0
    agreement_counts = []
    oracle_correct = 0
    greedy_correct = 0
    all_correct = 0

    # Per action type
    action_agreement = defaultdict(list)
    action_oracle = defaultdict(lambda: {'correct': 0, 'total': 0})

    # Coordinate clustering stats
    coord_clusters = []

    for r in results:
        for step_data in r['step_samples']:
            total_steps += 1
            samples = step_data['samples']
            gt_at = step_data['gt_action_type']
            K = len(samples)

            # Agreement: how many samples match the most common prediction
            pred_actions = [s['pred_action']['action'] for s in samples if s['pred_action'] is not None]
            if pred_actions:
                most_common_action, most_common_count = Counter(pred_actions).most_common(1)[0]
                agreement_rate = most_common_count / len(pred_actions)
            else:
                agreement_rate = 0
            agreement_counts.append(agreement_rate)
            action_agreement[gt_at].append(agreement_rate)

            # Oracle: best-of-K
            any_correct = any(s['extract_match'] for s in samples)
            oracle_correct += int(any_correct)
            action_oracle[gt_at]['total'] += 1
            action_oracle[gt_at]['correct'] += int(any_correct)

            # Greedy (first sample)
            if samples and samples[0]['extract_match']:
                greedy_correct += 1

            # All correct
            if all(s['extract_match'] for s in samples):
                all_correct += 1

            # Coordinate clustering for coord-based actions
            if gt_at in ('click', 'long_press'):
                coords = []
                for s in samples:
                    if s['pred_action'] is not None and 'coordinate' in s['pred_action']:
                        coords.append(s['pred_action']['coordinate'])
                if len(coords) >= 2:
                    coords_arr = np.array(coords, dtype=np.float64)
                    spread = np.std(coords_arr, axis=0).mean()
                    coord_clusters.append({
                        'gt_action_type': gt_at,
                        'n_coords': len(coords),
                        'spread': float(spread),
                        'any_correct': any_correct,
                    })

    # Adaptive K analysis
    # Strategy: if agreement >= threshold, use K=1 (greedy), else use K=5
    adaptive_results = {}
    for threshold in [0.6, 0.7, 0.8, 0.9]:
        high_agree_correct = 0
        low_agree_correct = 0
        high_agree_total = 0
        low_agree_total = 0
        idx = 0
        for r in results:
            for step_data in r['step_samples']:
                samples = step_data['samples']
                agree = agreement_counts[idx]
                if agree >= threshold:
                    high_agree_total += 1
                    if samples and samples[0]['extract_match']:
                        high_agree_correct += 1
                else:
                    low_agree_total += 1
                    if any(s['extract_match'] for s in samples[:5]):
                        low_agree_correct += 1
                idx += 1

        total = high_agree_total + low_agree_total
        correct = high_agree_correct + low_agree_correct
        adaptive_results[str(threshold)] = {
            'accuracy': correct / total if total > 0 else 0,
            'high_agree_frac': high_agree_total / total if total > 0 else 0,
            'high_agree_acc': high_agree_correct / high_agree_total if high_agree_total > 0 else 0,
            'low_agree_acc': low_agree_correct / low_agree_total if low_agree_total > 0 else 0,
        }

    # Action-type oracle rates
    action_oracle_rates = {}
    for at in action_oracle:
        t = action_oracle[at]['total']
        action_oracle_rates[at] = {
            **action_oracle[at],
            'oracle_rate': action_oracle[at]['correct'] / t if t > 0 else 0,
        }

    summary = {
        'total_steps': total_steps,
        'greedy_accuracy': greedy_correct / total_steps if total_steps > 0 else 0,
        'oracle_accuracy': oracle_correct / total_steps if total_steps > 0 else 0,
        'all_correct_rate': all_correct / total_steps if total_steps > 0 else 0,
        'oracle_gain': (oracle_correct - greedy_correct) / total_steps if total_steps > 0 else 0,
        'mean_agreement': float(np.mean(agreement_counts)) if agreement_counts else 0,
        'action_type_oracle': action_oracle_rates,
        'action_type_mean_agreement': {at: float(np.mean(v)) for at, v in action_agreement.items()},
        'adaptive_k_strategies': adaptive_results,
        'coord_spread_stats': {
            'n_steps': len(coord_clusters),
            'mean_spread': float(np.mean([c['spread'] for c in coord_clusters])) if coord_clusters else 0,
            'oracle_rate_coord': sum(c['any_correct'] for c in coord_clusters) / len(coord_clusters) if coord_clusters else 0,
        },
    }

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(summary, os.path.join(args.output_dir, 'eval_c4c7_analysis.json'))

    print(f"Greedy accuracy: {summary['greedy_accuracy']:.3f}")
    print(f"Oracle (best-of-K): {summary['oracle_accuracy']:.3f}")
    print(f"Oracle gain: +{summary['oracle_gain']:.3f}")
    print(f"Mean agreement: {summary['mean_agreement']:.3f}")
    print(f"\nAdaptive K strategies:")
    for th, data in adaptive_results.items():
        print(f"  threshold={th}: acc={data['accuracy']:.3f}, high_agree={data['high_agree_frac']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval C4+C7: Multi-Sample Analysis")
    parser.add_argument("--input_file", type=str, required=True, help="multisample_results.jsonl from GPU phase")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_c4c7_ac")
    args = parser.parse_args()
    main(args)
