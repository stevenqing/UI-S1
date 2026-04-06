"""Eval D0: Error Type Ceiling Analysis (Offline).

Classifies errors into grounding/action/content types and estimates observer ceiling.
"""

import argparse
import os
from collections import defaultdict

from ac_utils import load_jsonl, save_json, length_bucket, compute_trajectory_metrics


def classify_error(step_result):
    """Classify a step error into type A/B/C.

    A: Grounding error — type_match=True but extract_match=False
    B: Action error — type_match=False
    C: Content error — type_match=True, action type matches but text/args wrong
       (subset of A, but we keep A as the general category)
    """
    if step_result['type_match'] and not step_result['extract_match']:
        return 'A_grounding'
    elif not step_result['type_match']:
        return 'B_action'
    return None  # No error


def main(args):
    results = load_jsonl(args.input_file)
    print(f"Loaded {len(results)} episodes.")

    # Error classification
    error_counts = defaultdict(int)
    total_errors = 0
    total_steps = 0

    # By step position
    error_by_step = defaultdict(lambda: defaultdict(int))
    step_totals = defaultdict(int)

    # Repeated action detection
    repeated_actions = 0
    total_consecutive_pairs = 0

    # Per-length error distribution
    error_by_length = defaultdict(lambda: defaultdict(int))
    length_totals = defaultdict(int)

    for r in results:
        bucket = length_bucket(r['num_steps'])
        prev_action = None
        for s in r['step_results']:
            total_steps += 1
            pos = str(s['step_num']) if s['step_num'] < 10 else '10+'
            step_totals[pos] += 1
            length_totals[bucket] += 1

            if not s['extract_match']:
                total_errors += 1
                err_type = classify_error(s)
                if err_type:
                    error_counts[err_type] += 1
                    error_by_step[pos][err_type] += 1
                    error_by_length[bucket][err_type] += 1

            # Repeated action check
            current_action = s['pred_action']
            if prev_action is not None:
                total_consecutive_pairs += 1
                if (current_action.get('action') == prev_action.get('action') and
                    current_action.get('coordinate') == prev_action.get('coordinate') and
                    current_action.get('text') == prev_action.get('text')):
                    repeated_actions += 1
            prev_action = current_action

    # Error type rates
    error_rates = {k: v / total_errors if total_errors > 0 else 0 for k, v in error_counts.items()}

    # Observer ceiling estimation
    # Grounding errors might be fixable with better observation
    grounding_errors = error_counts.get('A_grounding', 0)
    current_metrics = compute_trajectory_metrics(results)

    # Conservative: fix 50% of grounding errors
    conservative_fix = grounding_errors * 0.5
    # Optimistic: fix 100% of grounding errors
    optimistic_fix = grounding_errors * 1.0

    # Estimate TSR improvement (rough: assume errors are independent)
    # Each fixed error might unlock subsequent steps
    current_step_accuracy = (total_steps - total_errors) / total_steps if total_steps > 0 else 0
    conservative_step_acc = (total_steps - total_errors + conservative_fix) / total_steps if total_steps > 0 else 0
    optimistic_step_acc = (total_steps - total_errors + optimistic_fix) / total_steps if total_steps > 0 else 0

    summary = {
        'total_steps': total_steps,
        'total_errors': total_errors,
        'step_accuracy': current_step_accuracy,
        'error_counts': dict(error_counts),
        'error_type_rates': error_rates,
        'error_by_step_position': {k: dict(v) for k, v in error_by_step.items()},
        'error_by_length_bucket': {k: dict(v) for k, v in error_by_length.items()},
        'repeated_action_rate': repeated_actions / total_consecutive_pairs if total_consecutive_pairs > 0 else 0,
        'repeated_actions': repeated_actions,
        'observer_ceiling': {
            'current_tsr': current_metrics['tsr'],
            'current_step_accuracy': current_step_accuracy,
            'conservative_step_accuracy': conservative_step_acc,
            'optimistic_step_accuracy': optimistic_step_acc,
            'grounding_errors_fixable': grounding_errors,
        },
    }

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(summary, os.path.join(args.output_dir, 'eval_d0_error_ceiling.json'))

    print(f"\nTotal errors: {total_errors}/{total_steps} ({1-current_step_accuracy:.3f})")
    print(f"\nError type breakdown:")
    for et, cnt in sorted(error_counts.items()):
        print(f"  {et}: {cnt} ({error_rates[et]:.3f})")
    print(f"\nRepeated actions: {repeated_actions}/{total_consecutive_pairs} ({summary['repeated_action_rate']:.3f})")
    print(f"\nObserver ceiling (step accuracy):")
    print(f"  Current: {current_step_accuracy:.3f}")
    print(f"  Conservative (+50% grounding fix): {conservative_step_acc:.3f}")
    print(f"  Optimistic (+100% grounding fix): {optimistic_step_acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval D0: Error Type Ceiling Analysis")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_d0_ac")
    args = parser.parse_args()
    main(args)
