"""Eval E2: Observer Value by Length (Offline).

Analyzes observer contribution (TSR delta, progress delta) by trajectory length bucket.
Requires both Eval A (baseline) and D1 (observer) results.
"""

import argparse
import os
from collections import defaultdict

from ac_utils import load_jsonl, save_json, length_bucket, compute_trajectory_metrics


def main(args):
    baseline = load_jsonl(args.baseline_file)
    observer = load_jsonl(args.observer_file)
    print(f"Loaded baseline: {len(baseline)}, observer: {len(observer)}")

    baseline_by_id = {r.get('episode_id'): r for r in baseline}
    observer_by_id = {r.get('episode_id'): r for r in observer}
    common_ids = set(baseline_by_id.keys()) & set(observer_by_id.keys())
    print(f"Common episodes: {len(common_ids)}")

    # Per-length bucket comparison
    bucket_baseline = defaultdict(list)
    bucket_observer = defaultdict(list)

    for eid in common_ids:
        b = baseline_by_id[eid]
        o = observer_by_id[eid]
        bucket = length_bucket(b['num_steps'])
        bucket_baseline[bucket].append(b)
        bucket_observer[bucket].append(o)

    summary = {}
    for bucket in ['short(1-3)', 'medium(4-7)', 'long(8-15)', 'vlong(16+)']:
        if bucket in bucket_baseline:
            bm = compute_trajectory_metrics(bucket_baseline[bucket])
            om = compute_trajectory_metrics(bucket_observer[bucket])
            summary[bucket] = {
                'n': bm['n'],
                'baseline_tsr': bm['tsr'],
                'observer_tsr': om['tsr'],
                'tsr_delta': om['tsr'] - bm['tsr'],
                'baseline_progress': bm['avg_progress'],
                'observer_progress': om['avg_progress'],
                'progress_delta': om['avg_progress'] - bm['avg_progress'],
            }

    # Overall
    overall_bm = compute_trajectory_metrics([baseline_by_id[eid] for eid in common_ids])
    overall_om = compute_trajectory_metrics([observer_by_id[eid] for eid in common_ids])
    summary['overall'] = {
        'n': overall_bm['n'],
        'baseline_tsr': overall_bm['tsr'],
        'observer_tsr': overall_om['tsr'],
        'tsr_delta': overall_om['tsr'] - overall_bm['tsr'],
        'baseline_progress': overall_bm['avg_progress'],
        'observer_progress': overall_om['avg_progress'],
        'progress_delta': overall_om['avg_progress'] - overall_bm['avg_progress'],
    }

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(summary, os.path.join(args.output_dir, 'eval_e2_observer_length.json'))

    print(f"\n=== Observer Value by Length ===")
    for bucket in ['short(1-3)', 'medium(4-7)', 'long(8-15)', 'vlong(16+)', 'overall']:
        if bucket in summary:
            s = summary[bucket]
            delta_sign = '+' if s['tsr_delta'] >= 0 else ''
            print(f"  {bucket:15s}: n={s['n']:4d} | Baseline TSR={s['baseline_tsr']:.3f} | Observer TSR={s['observer_tsr']:.3f} | Delta={delta_sign}{s['tsr_delta']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval E2: Observer Value by Length")
    parser.add_argument("--baseline_file", type=str, required=True, help="Eval A trajectory_results.jsonl")
    parser.add_argument("--observer_file", type=str, required=True, help="D1 observer_results.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_e2_ac")
    args = parser.parse_args()
    main(args)
