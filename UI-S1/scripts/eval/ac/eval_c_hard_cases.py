"""Eval C: Hard Cases Identification (Offline).

Identifies consistently failing steps from Eval A results.
"""

import argparse
import os
from collections import defaultdict

from ac_utils import load_jsonl, save_json, length_bucket


def main(args):
    results = load_jsonl(args.input_file)
    print(f"Loaded {len(results)} episodes.")

    # Identify failing steps
    failed_steps = []
    total_steps = 0
    for r in results:
        for s in r['step_results']:
            total_steps += 1
            if not s['extract_match']:
                failed_steps.append({
                    'episode_id': r.get('episode_id'),
                    'goal': r['goal'],
                    'step_num': s['step_num'],
                    'gt_action': s['gt_action'],
                    'pred_action': s['pred_action'],
                    'gt_action_type': s['gt_action_type'],
                    'num_steps': r['num_steps'],
                })

    # Hard case rate by step position
    step_fail = defaultdict(lambda: {'fail': 0, 'total': 0})
    for r in results:
        for s in r['step_results']:
            pos = str(s['step_num']) if s['step_num'] < 10 else '10+'
            step_fail[pos]['total'] += 1
            if not s['extract_match']:
                step_fail[pos]['fail'] += 1

    step_fail_rates = {k: {**v, 'rate': v['fail'] / v['total'] if v['total'] > 0 else 0}
                       for k, v in sorted(step_fail.items())}

    # Hard case rate by action type
    action_fail = defaultdict(lambda: {'fail': 0, 'total': 0})
    for r in results:
        for s in r['step_results']:
            at = s['gt_action_type']
            action_fail[at]['total'] += 1
            if not s['extract_match']:
                action_fail[at]['fail'] += 1

    action_fail_rates = {k: {**v, 'rate': v['fail'] / v['total'] if v['total'] > 0 else 0}
                         for k, v in sorted(action_fail.items())}

    # Distribution stats
    fail_by_episode = defaultdict(int)
    for f in failed_steps:
        fail_by_episode[f['episode_id']] += 1

    summary = {
        'total_steps': total_steps,
        'total_failed': len(failed_steps),
        'overall_fail_rate': len(failed_steps) / total_steps if total_steps > 0 else 0,
        'fail_rate_by_step_position': step_fail_rates,
        'fail_rate_by_action_type': action_fail_rates,
        'episodes_with_failures': len(fail_by_episode),
        'episodes_total': len(results),
        'avg_failures_per_failed_episode': sum(fail_by_episode.values()) / len(fail_by_episode) if fail_by_episode else 0,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(summary, os.path.join(args.output_dir, 'eval_c_hard_cases.json'))

    print(f"\nTotal steps: {total_steps}, Failed: {len(failed_steps)} ({summary['overall_fail_rate']:.3f})")
    print(f"\n=== Fail Rate by Step Position ===")
    for pos, data in step_fail_rates.items():
        print(f"  Step {pos}: {data['rate']:.3f} ({data['fail']}/{data['total']})")
    print(f"\n=== Fail Rate by Action Type ===")
    for at, data in action_fail_rates.items():
        print(f"  {at}: {data['rate']:.3f} ({data['fail']}/{data['total']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval C: Hard Cases Identification")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_c_ac")
    args = parser.parse_args()
    main(args)
