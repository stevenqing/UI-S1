"""Eval B: Step-Position Accuracy Analysis (Offline).

Analyzes per-step-position accuracy from Eval A results.
"""

import argparse
import json
import os
from collections import defaultdict

from ac_utils import load_jsonl, save_json, length_bucket


def main(args):
    results = load_jsonl(args.input_file)
    print(f"Loaded {len(results)} episodes.")

    # Per step position
    step_type_match = defaultdict(lambda: {'correct': 0, 'total': 0})
    step_extract_match = defaultdict(lambda: {'correct': 0, 'total': 0})

    # Length-conditioned step accuracy
    length_step_extract = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))

    for r in results:
        bucket = length_bucket(r['num_steps'])
        for s in r['step_results']:
            pos = s['step_num']
            pos_key = str(pos) if pos < 10 else '10+'

            step_type_match[pos_key]['total'] += 1
            step_type_match[pos_key]['correct'] += int(s['type_match'])

            step_extract_match[pos_key]['total'] += 1
            step_extract_match[pos_key]['correct'] += int(s['extract_match'])

            length_step_extract[bucket][pos_key]['total'] += 1
            length_step_extract[bucket][pos_key]['correct'] += int(s['extract_match'])

    # Compute rates
    def compute_rates(d):
        return {k: {**v, 'rate': v['correct'] / v['total'] if v['total'] > 0 else 0}
                for k, v in sorted(d.items(), key=lambda x: (x[0] if x[0] != '10+' else '99'))}

    type_match_rates = compute_rates(step_type_match)
    extract_match_rates = compute_rates(step_extract_match)

    length_conditioned = {}
    for bucket, step_data in length_step_extract.items():
        length_conditioned[bucket] = compute_rates(step_data)

    summary = {
        'type_match_by_step': type_match_rates,
        'extract_match_by_step': extract_match_rates,
        'length_conditioned_extract_match': length_conditioned,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(summary, os.path.join(args.output_dir, 'eval_b_step_analysis.json'))

    print("\n=== Extract Match by Step Position ===")
    for pos, data in extract_match_rates.items():
        print(f"  Step {pos}: {data['rate']:.3f} ({data['correct']}/{data['total']})")

    print("\n=== Type Match by Step Position ===")
    for pos, data in type_match_rates.items():
        print(f"  Step {pos}: {data['rate']:.3f} ({data['correct']}/{data['total']})")

    print("\n=== Length-Conditioned Extract Match ===")
    for bucket in ['short(1-3)', 'medium(4-7)', 'long(8-15)', 'vlong(16+)']:
        if bucket in length_conditioned:
            print(f"  {bucket}:")
            for pos, data in length_conditioned[bucket].items():
                print(f"    Step {pos}: {data['rate']:.3f} ({data['correct']}/{data['total']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval B: Step-Position Accuracy Analysis")
    parser.add_argument("--input_file", type=str, required=True, help="Path to Eval A trajectory_results.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_b_ac")
    args = parser.parse_args()
    main(args)
