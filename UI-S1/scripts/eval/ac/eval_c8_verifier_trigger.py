"""Eval C8: Verifier Trigger Analysis (Offline).

Analyzes silent failures, first error distribution, and verification ceilings.
"""

import argparse
import os
from collections import defaultdict

from ac_utils import load_jsonl, save_json, length_bucket


def main(args):
    results = load_jsonl(args.input_file)
    print(f"Loaded {len(results)} episodes.")

    total_episodes = len(results)
    failed_episodes = [r for r in results if not r['task_success']]
    successful_episodes = [r for r in results if r['task_success']]

    # First error step distribution
    first_error_dist = defaultdict(int)
    for r in failed_episodes:
        for s in r['step_results']:
            if not s['extract_match']:
                pos = str(s['step_num']) if s['step_num'] < 10 else '10+'
                first_error_dist[pos] += 1
                break

    # Silent failure rate: steps that passed but trajectory still failed
    # (all correct steps in failed trajectories except the last wrong one)
    silent_fail_steps = 0
    total_correct_steps = 0
    for r in results:
        for s in r['step_results']:
            if s['extract_match']:
                total_correct_steps += 1
                if not r['task_success']:
                    silent_fail_steps += 1

    # Near-miss analysis: failed trajectories with progress >= 50%
    near_misses = []
    for r in failed_episodes:
        progress = r['final_step_id'] / r['num_steps'] if r['num_steps'] > 0 else 0
        if progress >= 0.5:
            near_misses.append({
                'episode_id': r.get('episode_id'),
                'progress': progress,
                'num_steps': r['num_steps'],
                'final_step_id': r['final_step_id'],
            })

    # Perfect verification ceiling: if we could perfectly detect and recover from errors
    # Assume recovery = skip the wrong step and continue
    # Upper bound: all trajectories succeed
    perfect_verify_tsr = 1.0

    # Conservative ceiling: detect first error, retry once with oracle
    # If the error is at step k, we oracle-fix it and continue
    # Approximation: each failed trajectory gets +1 step correct
    conservative_extra_success = 0
    for r in failed_episodes:
        # If the first error was the last step, fixing it completes the task
        if r['final_step_id'] == r['num_steps'] - 1:
            conservative_extra_success += 1

    conservative_tsr = (len(successful_episodes) + conservative_extra_success) / total_episodes if total_episodes > 0 else 0

    # By length bucket
    bucket_stats = defaultdict(lambda: {'total': 0, 'success': 0, 'failed': 0, 'near_miss': 0})
    for r in results:
        b = length_bucket(r['num_steps'])
        bucket_stats[b]['total'] += 1
        if r['task_success']:
            bucket_stats[b]['success'] += 1
        else:
            bucket_stats[b]['failed'] += 1
            progress = r['final_step_id'] / r['num_steps'] if r['num_steps'] > 0 else 0
            if progress >= 0.5:
                bucket_stats[b]['near_miss'] += 1

    summary = {
        'total_episodes': total_episodes,
        'success': len(successful_episodes),
        'failed': len(failed_episodes),
        'tsr': len(successful_episodes) / total_episodes if total_episodes > 0 else 0,
        'silent_fail_steps': silent_fail_steps,
        'total_correct_steps': total_correct_steps,
        'silent_fail_rate': silent_fail_steps / total_correct_steps if total_correct_steps > 0 else 0,
        'first_error_step_distribution': dict(first_error_dist),
        'near_miss_count': len(near_misses),
        'near_miss_rate': len(near_misses) / len(failed_episodes) if failed_episodes else 0,
        'perfect_verify_ceiling_tsr': perfect_verify_tsr,
        'conservative_verify_ceiling_tsr': conservative_tsr,
        'bucket_stats': dict(bucket_stats),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(summary, os.path.join(args.output_dir, 'eval_c8_verifier.json'))

    print(f"\nTSR: {summary['tsr']:.3f}")
    print(f"Silent fail rate: {summary['silent_fail_rate']:.3f}")
    print(f"Near-miss count: {len(near_misses)} ({summary['near_miss_rate']:.3f} of failures)")
    print(f"Conservative verify ceiling TSR: {conservative_tsr:.3f}")
    print(f"\nFirst error step distribution:")
    for pos in sorted(first_error_dist.keys()):
        print(f"  Step {pos}: {first_error_dist[pos]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval C8: Verifier Trigger Analysis")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_c8_ac")
    args = parser.parse_args()
    main(args)
