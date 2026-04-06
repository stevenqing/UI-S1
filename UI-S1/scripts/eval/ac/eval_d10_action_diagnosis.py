"""Eval D10: Action Confusion Matrix (Offline).

Full 7x7 action type confusion matrix and error distribution analysis.
"""

import argparse
import os
from collections import defaultdict

from ac_utils import load_jsonl, save_json, ALL_ACTION_TYPES


def main(args):
    results = load_jsonl(args.input_file)
    print(f"Loaded {len(results)} episodes.")

    # Full confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))
    # Per-action accuracy
    action_acc = defaultdict(lambda: {'correct': 0, 'total': 0})
    # Near-miss vs complete-miss
    near_miss = 0  # type_match but not extract_match
    complete_miss = 0  # not type_match
    total_errors = 0
    # Error by step position
    error_by_step = defaultdict(lambda: {'near_miss': 0, 'complete_miss': 0, 'total': 0})

    for r in results:
        for s in r['step_results']:
            gt_at = s['gt_action_type']
            pred_at = s['pred_action'].get('action', 'unknown')
            pos = str(s['step_num']) if s['step_num'] < 10 else '10+'

            confusion[gt_at][pred_at] += 1
            action_acc[gt_at]['total'] += 1
            error_by_step[pos]['total'] += 1

            if s['extract_match']:
                action_acc[gt_at]['correct'] += 1
            else:
                total_errors += 1
                if s['type_match']:
                    near_miss += 1
                    error_by_step[pos]['near_miss'] += 1
                else:
                    complete_miss += 1
                    error_by_step[pos]['complete_miss'] += 1

    # Build confusion matrix as 2D array
    all_types = sorted(set(ALL_ACTION_TYPES) | set(confusion.keys()))
    pred_types = set()
    for gt in confusion:
        pred_types.update(confusion[gt].keys())
    all_pred_types = sorted(set(all_types) | pred_types)

    matrix = []
    for gt in all_types:
        row = [confusion[gt][pred] for pred in all_pred_types]
        matrix.append(row)

    # Top confusion pairs
    pairs = []
    for gt in confusion:
        for pred, cnt in confusion[gt].items():
            if gt != pred:
                pairs.append({'gt': gt, 'pred': pred, 'count': cnt})
    pairs.sort(key=lambda x: -x['count'])

    # Per-action accuracy
    action_rates = {}
    for at in action_acc:
        t = action_acc[at]['total']
        action_rates[at] = {
            **action_acc[at],
            'accuracy': action_acc[at]['correct'] / t if t > 0 else 0,
        }

    # Error by step
    error_step_rates = {}
    for pos in error_by_step:
        t = error_by_step[pos]['total']
        error_step_rates[pos] = {
            **error_by_step[pos],
            'near_miss_rate': error_by_step[pos]['near_miss'] / t if t > 0 else 0,
            'complete_miss_rate': error_by_step[pos]['complete_miss'] / t if t > 0 else 0,
        }

    summary = {
        'confusion_matrix': {
            'gt_labels': all_types,
            'pred_labels': all_pred_types,
            'matrix': matrix,
        },
        'top_confusion_pairs': pairs[:15],
        'per_action_accuracy': action_rates,
        'near_miss_total': near_miss,
        'complete_miss_total': complete_miss,
        'total_errors': total_errors,
        'near_miss_fraction': near_miss / total_errors if total_errors > 0 else 0,
        'error_by_step_position': error_step_rates,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(summary, os.path.join(args.output_dir, 'eval_d10_diagnosis.json'))

    print(f"\n=== Per-Action Accuracy ===")
    for at in ALL_ACTION_TYPES:
        if at in action_rates:
            r = action_rates[at]
            print(f"  {at:15s}: {r['accuracy']:.3f} ({r['correct']}/{r['total']})")

    print(f"\n=== Error Classification ===")
    print(f"  Near-miss (right type, wrong target): {near_miss} ({summary['near_miss_fraction']:.3f})")
    print(f"  Complete-miss (wrong type): {complete_miss} ({1-summary['near_miss_fraction']:.3f})")

    print(f"\n=== Top Confusion Pairs ===")
    for p in pairs[:10]:
        print(f"  GT={p['gt']:15s} -> Pred={p['pred']:15s}: {p['count']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval D10: Action Confusion Matrix")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_d10_ac")
    args = parser.parse_args()
    main(args)
