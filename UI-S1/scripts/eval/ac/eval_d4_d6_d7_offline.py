"""Eval D4/D6/D7: Planner Ceiling + Failure Types + Length Analysis (Offline).

D4: Analyze observer (D1) wins vs losses compared to baseline (Eval A).
D6: Classify all failed trajectories by error type.
D7: Cross-tabulate failure types with trajectory length.
"""

import argparse
import os
from collections import defaultdict

from ac_utils import load_jsonl, save_json, length_bucket


def classify_trajectory_failure(step_results):
    """Classify trajectory failure type based on first error.

    A: Grounding error — type_match=True, extract_match=False
    B: Action error — type_match=False
    C: No steps attempted (empty)
    D: Unknown
    """
    for s in step_results:
        if not s['extract_match']:
            if s['type_match']:
                return 'A_grounding'
            else:
                return 'B_action'
    return 'D_unknown'


def main(args):
    # Load baseline (Eval A)
    baseline = load_jsonl(args.baseline_file)
    baseline_by_id = {r.get('episode_id'): r for r in baseline}
    print(f"Loaded baseline: {len(baseline)} episodes")

    # Load observer (D1) if available
    observer = None
    observer_by_id = {}
    if args.observer_file and os.path.exists(args.observer_file):
        observer = load_jsonl(args.observer_file)
        observer_by_id = {r.get('episode_id'): r for r in observer}
        print(f"Loaded observer: {len(observer)} episodes")

    # === D4: Planner ceiling (observer vs baseline) ===
    d4_results = None
    if observer:
        wins = 0  # observer succeeds, baseline fails
        losses = 0  # observer fails, baseline succeeds
        both_success = 0
        both_fail = 0
        common_ids = set(baseline_by_id.keys()) & set(observer_by_id.keys())

        for eid in common_ids:
            b = baseline_by_id[eid]['task_success']
            o = observer_by_id[eid]['task_success']
            if o and not b:
                wins += 1
            elif b and not o:
                losses += 1
            elif b and o:
                both_success += 1
            else:
                both_fail += 1

        total = len(common_ids)
        d4_results = {
            'common_episodes': total,
            'observer_wins': wins,
            'observer_losses': losses,
            'both_success': both_success,
            'both_fail': both_fail,
            'baseline_tsr': (both_success + losses) / total if total > 0 else 0,
            'observer_tsr': (both_success + wins) / total if total > 0 else 0,
            'net_gain': (wins - losses) / total if total > 0 else 0,
        }

    # === D6: Failure type classification ===
    failure_types = defaultdict(int)
    failed_episodes = [r for r in baseline if not r['task_success']]

    for r in failed_episodes:
        ftype = classify_trajectory_failure(r['step_results'])
        failure_types[ftype] += 1

    total_failed = len(failed_episodes)
    failure_rates = {k: v / total_failed if total_failed > 0 else 0 for k, v in failure_types.items()}

    # === D7: Length x failure type ===
    length_failure = defaultdict(lambda: defaultdict(int))
    length_total = defaultdict(int)

    for r in failed_episodes:
        bucket = length_bucket(r['num_steps'])
        ftype = classify_trajectory_failure(r['step_results'])
        length_failure[bucket][ftype] += 1
        length_total[bucket] += 1

    length_failure_rates = {}
    for bucket in length_failure:
        total = length_total[bucket]
        length_failure_rates[bucket] = {
            'total_failed': total,
            **{k: {'count': v, 'rate': v / total if total > 0 else 0}
               for k, v in length_failure[bucket].items()}
        }

    summary = {
        'd4_planner_ceiling': d4_results,
        'd6_failure_types': {
            'total_failed': total_failed,
            'counts': dict(failure_types),
            'rates': failure_rates,
        },
        'd7_length_x_failure': length_failure_rates,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(summary, os.path.join(args.output_dir, 'eval_d4_d6_d7.json'))

    print(f"\n=== D6: Failure Types ===")
    print(f"Total failed: {total_failed}")
    for ft, cnt in sorted(failure_types.items()):
        print(f"  {ft}: {cnt} ({failure_rates[ft]:.3f})")

    if d4_results:
        print(f"\n=== D4: Observer vs Baseline ===")
        print(f"Baseline TSR: {d4_results['baseline_tsr']:.3f}")
        print(f"Observer TSR: {d4_results['observer_tsr']:.3f}")
        print(f"Wins: {d4_results['observer_wins']}, Losses: {d4_results['observer_losses']}")

    print(f"\n=== D7: Length x Failure ===")
    for bucket in ['short(1-3)', 'medium(4-7)', 'long(8-15)', 'vlong(16+)']:
        if bucket in length_failure_rates:
            data = length_failure_rates[bucket]
            print(f"  {bucket} (n={data['total_failed']}):")
            for ft in ['A_grounding', 'B_action']:
                if ft in data:
                    print(f"    {ft}: {data[ft]['count']} ({data[ft]['rate']:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval D4/D6/D7: Offline Failure Analysis")
    parser.add_argument("--baseline_file", type=str, required=True, help="Eval A trajectory_results.jsonl")
    parser.add_argument("--observer_file", type=str, default=None, help="D1 observer_results.jsonl (optional)")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_d4d6d7_ac")
    args = parser.parse_args()
    main(args)
