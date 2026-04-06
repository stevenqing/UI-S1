"""Pre-test 3: Counterfactual Oracle Fix (Offline).

If we oracle-fix grounding errors vs action errors, how much TSR improvement per length bucket?
Tests whether the error-type crossover holds for AndroidControl.
"""

import argparse
import os
from collections import defaultdict

from ac_utils import load_jsonl, save_json, length_bucket, compute_trajectory_metrics


def simulate_oracle_fix(results, fix_type):
    """Simulate oracle fix for a specific error type.

    fix_type: 'grounding' (type_match=True, extract_match=False) or 'action' (type_match=False)

    Returns modified results with oracle-fixed steps.
    """
    modified = []
    for r in results:
        new_steps = []
        still_running = True
        for s in r['step_results']:
            new_s = dict(s)
            if still_running and not s['extract_match']:
                if fix_type == 'grounding' and s['type_match'] and not s['extract_match']:
                    new_s['extract_match'] = True
                elif fix_type == 'action' and not s['type_match']:
                    new_s['extract_match'] = True
                    new_s['type_match'] = True
                else:
                    still_running = False
            elif not still_running:
                pass  # AR stops after first real error
            new_steps.append(new_s)

        # Recompute success
        correct = 0
        for s in new_steps:
            if s['extract_match']:
                correct += 1
            else:
                break

        modified.append({
            **r,
            'step_results': new_steps,
            'task_success': correct == r['num_steps'],
            'final_step_id': correct,
        })
    return modified


def main(args):
    results = load_jsonl(args.input_file)
    print(f"Loaded {len(results)} episodes.")

    # Baseline metrics
    baseline_metrics = compute_trajectory_metrics(results)

    # Oracle-fix grounding errors
    fixed_grounding = simulate_oracle_fix(results, 'grounding')
    grounding_metrics = compute_trajectory_metrics(fixed_grounding)

    # Oracle-fix action errors
    fixed_action = simulate_oracle_fix(results, 'action')
    action_metrics = compute_trajectory_metrics(fixed_action)

    # Per-length bucket analysis
    bucket_results = defaultdict(list)
    bucket_grounding = defaultdict(list)
    bucket_action = defaultdict(list)

    for r, rg, ra in zip(results, fixed_grounding, fixed_action):
        bucket = length_bucket(r['num_steps'])
        bucket_results[bucket].append(r)
        bucket_grounding[bucket].append(rg)
        bucket_action[bucket].append(ra)

    bucket_analysis = {}
    for bucket in ['short(1-3)', 'medium(4-7)', 'long(8-15)', 'vlong(16+)']:
        if bucket in bucket_results:
            bm = compute_trajectory_metrics(bucket_results[bucket])
            gm = compute_trajectory_metrics(bucket_grounding[bucket])
            am = compute_trajectory_metrics(bucket_action[bucket])

            # Count error types in this bucket
            grounding_errors = 0
            action_errors = 0
            for r in bucket_results[bucket]:
                for s in r['step_results']:
                    if not s['extract_match']:
                        if s['type_match']:
                            grounding_errors += 1
                        else:
                            action_errors += 1
                        break  # Only first error matters for AR

            bucket_analysis[bucket] = {
                'n': bm['n'],
                'baseline_tsr': bm['tsr'],
                'fix_grounding_tsr': gm['tsr'],
                'fix_action_tsr': am['tsr'],
                'grounding_gain': gm['tsr'] - bm['tsr'],
                'action_gain': am['tsr'] - bm['tsr'],
                'grounding_first_errors': grounding_errors,
                'action_first_errors': action_errors,
                'dominant_error': 'grounding' if grounding_errors > action_errors else 'action',
            }

    summary = {
        'baseline': baseline_metrics,
        'fix_grounding': {
            **grounding_metrics,
            'tsr_gain': grounding_metrics['tsr'] - baseline_metrics['tsr'],
        },
        'fix_action': {
            **action_metrics,
            'tsr_gain': action_metrics['tsr'] - baseline_metrics['tsr'],
        },
        'per_length_bucket': bucket_analysis,
        'crossover_test': {
            'description': 'Does error type dominance change with trajectory length?',
            'results': {b: v['dominant_error'] for b, v in bucket_analysis.items()},
        },
    }

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(summary, os.path.join(args.output_dir, 'eval_pretest3_counterfactual.json'))

    print(f"\n=== Counterfactual Oracle Fix ===")
    print(f"Baseline TSR: {baseline_metrics['tsr']:.3f}")
    print(f"Fix grounding: {grounding_metrics['tsr']:.3f} (+{grounding_metrics['tsr']-baseline_metrics['tsr']:.3f})")
    print(f"Fix action: {action_metrics['tsr']:.3f} (+{action_metrics['tsr']-baseline_metrics['tsr']:.3f})")

    print(f"\n=== Per-Length Crossover ===")
    for bucket in ['short(1-3)', 'medium(4-7)', 'long(8-15)', 'vlong(16+)']:
        if bucket in bucket_analysis:
            b = bucket_analysis[bucket]
            print(f"  {bucket}: grounding_gain=+{b['grounding_gain']:.3f}, action_gain=+{b['action_gain']:.3f}, dominant={b['dominant_error']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-test 3: Counterfactual Oracle Fix")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_pretest3_ac")
    args = parser.parse_args()
    main(args)
