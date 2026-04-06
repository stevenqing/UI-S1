"""Eval C2: Per-Action-Type Analysis (Offline).

Analyzes accuracy broken down by action type, step position, and trajectory length.
"""

import argparse
import os
from collections import defaultdict

from ac_utils import load_jsonl, save_json, length_bucket, ALL_ACTION_TYPES


def main(args):
    results = load_jsonl(args.input_file)
    print(f"Loaded {len(results)} episodes.")

    # Per action type stats
    action_stats = defaultdict(lambda: {'total': 0, 'type_match': 0, 'extract_match': 0})

    # Confusion matrix: gt_action -> pred_action -> count
    confusion = defaultdict(lambda: defaultdict(int))

    # Cross: action_type x step_position
    action_step = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'extract_match': 0}))

    # Cross: action_type x trajectory_length
    action_length = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'extract_match': 0}))

    for r in results:
        bucket = length_bucket(r['num_steps'])
        for s in r['step_results']:
            gt_at = s['gt_action_type']
            pred_at = s['pred_action'].get('action', 'unknown')
            pos = str(s['step_num']) if s['step_num'] < 10 else '10+'

            action_stats[gt_at]['total'] += 1
            action_stats[gt_at]['type_match'] += int(s['type_match'])
            action_stats[gt_at]['extract_match'] += int(s['extract_match'])

            confusion[gt_at][pred_at] += 1

            action_step[gt_at][pos]['total'] += 1
            action_step[gt_at][pos]['extract_match'] += int(s['extract_match'])

            action_length[gt_at][bucket]['total'] += 1
            action_length[gt_at][bucket]['extract_match'] += int(s['extract_match'])

    # Compute rates
    for at in action_stats:
        t = action_stats[at]['total']
        action_stats[at]['type_match_rate'] = action_stats[at]['type_match'] / t if t > 0 else 0
        action_stats[at]['extract_match_rate'] = action_stats[at]['extract_match'] / t if t > 0 else 0

    def add_rates(d):
        out = {}
        for k1, v1 in d.items():
            out[k1] = {}
            for k2, v2 in v1.items():
                out[k1][k2] = {**v2, 'rate': v2['extract_match'] / v2['total'] if v2['total'] > 0 else 0}
        return out

    summary = {
        'action_type_stats': dict(action_stats),
        'confusion_matrix': {k: dict(v) for k, v in confusion.items()},
        'action_type_x_step_position': add_rates(action_step),
        'action_type_x_trajectory_length': add_rates(action_length),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(summary, os.path.join(args.output_dir, 'eval_c2_action_type.json'))

    print("\n=== Per-Action-Type Accuracy ===")
    for at in ALL_ACTION_TYPES:
        if at in action_stats:
            s = action_stats[at]
            print(f"  {at:15s}: type_match={s['type_match_rate']:.3f}  extract_match={s['extract_match_rate']:.3f}  (n={s['total']})")

    print("\n=== Confusion Matrix (top pairs) ===")
    pairs = []
    for gt, preds in confusion.items():
        for pred, cnt in preds.items():
            if gt != pred:
                pairs.append((gt, pred, cnt))
    pairs.sort(key=lambda x: -x[2])
    for gt, pred, cnt in pairs[:10]:
        print(f"  GT={gt:15s} -> Pred={pred:15s}: {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval C2: Per-Action-Type Analysis")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_c2_ac")
    args = parser.parse_args()
    main(args)
