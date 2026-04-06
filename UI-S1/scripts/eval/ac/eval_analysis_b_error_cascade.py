"""Analysis B: Error Cascade Markov Structure.

Tests whether error cascades in GUI agent trajectories have Markov structure:
does the current step's error probability depend only on the current state
(agreement) or also on error history (was previous step correct)?

If Markov order-0: P(correct | agreement) is sufficient
If Markov order-1: P(correct | agreement, prev_correct) needed

Uses C4+C7 multi-sample data (for agreement) crossed with Eval A trajectory
results (for actual AR correctness sequences).

All offline — 0 GPU required.
"""

import argparse
import json
import os
import numpy as np
from collections import defaultdict


def load_trajectory_results(jsonl_path):
    """Load Eval A trajectory results indexed by episode_id."""
    results = {}
    with open(jsonl_path) as f:
        for line in f:
            ep = json.loads(line)
            eid = ep.get('episode_id')
            if eid is not None:
                results[eid] = ep
    return results


def load_multisample_agreements(jsonl_path):
    """Load per-step agreement from C4+C7 multi-sample data."""
    agreements = {}  # (episode_id, step_num) -> agreement
    with open(jsonl_path) as f:
        for line in f:
            ep = json.loads(line)
            eid = ep.get('episode_id')
            for step in ep.get('step_samples', []):
                samples = step.get('samples', [])
                if not samples:
                    continue
                K = len(samples)
                from collections import Counter
                action_types = []
                for s in samples:
                    pa = s.get('pred_action')
                    if pa and isinstance(pa, dict):
                        action_types.append(pa.get('action', 'unknown'))
                    else:
                        action_types.append('parse_fail')
                if action_types:
                    type_counter = Counter(action_types)
                    agreement = type_counter.most_common(1)[0][1] / K
                    agreements[(eid, step.get('step_num', 0))] = agreement
    return agreements


def build_markov_transitions_from_multisample(multisample_path):
    """Build Markov transition matrix using C4+C7 per-step data.

    Unlike AR trajectory data (stop-on-error), C4+C7 evaluates ALL steps
    independently against GT screenshots, so we can observe both
    correct→correct and incorrect→correct transitions.

    State: (agreement_bin, prev_step_greedy_correct)
    Transition: -> correct / incorrect at current step
    """
    agree_bins = [(0, 0.5, 'low'), (0.5, 0.7, 'med'), (0.7, 0.9, 'high'), (0.9, 1.01, 'vhigh')]

    def get_bin(a):
        for lo, hi, name in agree_bins:
            if lo <= a < hi:
                return name
        return 'vhigh'

    transitions = defaultdict(lambda: {'correct': 0, 'incorrect': 0, 'total': 0})
    order0_transitions = defaultdict(lambda: {'correct': 0, 'incorrect': 0, 'total': 0})

    matched_episodes = 0
    total_transitions = 0

    with open(multisample_path) as f:
        for line in f:
            ep = json.loads(line)
            step_data = []

            for step in ep.get('step_samples', []):
                samples = step.get('samples', [])
                if not samples:
                    continue
                K = len(samples)
                from collections import Counter
                action_types = []
                for s in samples:
                    pa = s.get('pred_action')
                    if pa and isinstance(pa, dict):
                        action_types.append(pa.get('action', 'unknown'))
                    else:
                        action_types.append('parse_fail')
                if not action_types:
                    continue

                type_counter = Counter(action_types)
                agreement = type_counter.most_common(1)[0][1] / K
                greedy_correct = samples[0].get('extract_match', False)

                step_data.append({
                    'step_num': step.get('step_num', 0),
                    'agreement': agreement,
                    'greedy_correct': greedy_correct,
                })

            if len(step_data) < 2:
                continue

            matched_episodes += 1

            # Sort by step_num
            step_data.sort(key=lambda x: x['step_num'])

            for i, sd in enumerate(step_data):
                agree_bin = get_bin(sd['agreement'])
                is_correct = sd['greedy_correct']

                # Order-0
                order0_transitions[agree_bin]['correct' if is_correct else 'incorrect'] += 1
                order0_transitions[agree_bin]['total'] += 1

                # Order-1
                if i > 0:
                    prev_correct = step_data[i-1]['greedy_correct']
                    state = (agree_bin, prev_correct)
                    transitions[state]['correct' if is_correct else 'incorrect'] += 1
                    transitions[state]['total'] += 1
                    total_transitions += 1

    return transitions, order0_transitions, matched_episodes, total_transitions


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading trajectory results...")
    traj_results = load_trajectory_results(args.trajectory_jsonl)
    print(f"  Loaded {len(traj_results)} trajectories")

    # Build transitions from C4+C7 multisample data
    # (C4+C7 evaluates ALL steps independently against GT screenshots,
    #  so we can observe both correct→correct and incorrect→correct transitions,
    #  unlike AR trajectory data which stops on first error)
    print("\nBuilding Markov transition matrix from C4+C7 multisample data...")
    transitions, order0, n_matched, n_transitions = build_markov_transitions_from_multisample(args.multisample_jsonl)
    print(f"  Matched episodes: {n_matched}")
    print(f"  Total order-1 transitions: {n_transitions}")

    # --- Order-0: P(correct | agreement_bin) ---
    print(f"\n{'='*70}")
    print("ORDER-0: P(correct | agreement)")
    print(f"{'='*70}")
    print(f"  {'Agree Bin':>10} | {'N':>6} | {'P(correct)':>11}")
    print(f"  {'-'*10} | {'-'*6} | {'-'*11}")
    for abin in ['low', 'med', 'high', 'vhigh']:
        d = order0[abin]
        if d['total'] == 0:
            continue
        p_correct = d['correct'] / d['total']
        print(f"  {abin:>10} | {d['total']:6d} | {p_correct:11.4f}")

    # --- Order-1: P(correct | agreement_bin, prev_correct) ---
    print(f"\n{'='*70}")
    print("ORDER-1: P(correct | agreement, prev_correct)")
    print(f"{'='*70}")
    print(f"  {'Agree Bin':>10} | {'Prev':>6} | {'N':>6} | {'P(correct)':>11} | {'Delta vs order-0':>17}")
    print(f"  {'-'*10} | {'-'*6} | {'-'*6} | {'-'*11} | {'-'*17}")

    for abin in ['low', 'med', 'high', 'vhigh']:
        order0_p = order0[abin]['correct'] / order0[abin]['total'] if order0[abin]['total'] > 0 else 0
        for prev in [True, False]:
            state = (abin, prev)
            d = transitions[state]
            if d['total'] == 0:
                continue
            p_correct = d['correct'] / d['total']
            delta = p_correct - order0_p
            prev_label = 'PASS' if prev else 'FAIL'
            print(f"  {abin:>10} | {prev_label:>6} | {d['total']:6d} | {p_correct:11.4f} | {delta:+17.4f}")

    # --- Markov order test ---
    print(f"\n{'='*70}")
    print("MARKOV ORDER TEST")
    print(f"{'='*70}")

    deltas = []
    for abin in ['low', 'med', 'high', 'vhigh']:
        d_pass = transitions[(abin, True)]
        d_fail = transitions[(abin, False)]
        if d_pass['total'] > 10 and d_fail['total'] > 10:
            p_after_pass = d_pass['correct'] / d_pass['total']
            p_after_fail = d_fail['correct'] / d_fail['total']
            delta = p_after_pass - p_after_fail
            deltas.append((abin, delta, d_pass['total'], d_fail['total']))
            print(f"  {abin:>10}: P(correct|prev=PASS) - P(correct|prev=FAIL) = {delta:+.4f}")
            print(f"             ({p_after_pass:.4f} vs {p_after_fail:.4f}, N={d_pass['total']}+{d_fail['total']})")

    if deltas:
        avg_delta = np.mean([d[1] for d in deltas])
        print(f"\n  Average history effect: {avg_delta:+.4f}")

        if abs(avg_delta) < 0.05:
            print(f"\n  ★ History effect is SMALL ({avg_delta:+.4f})")
            print(f"    → Error cascade is approximately Markov order-0")
            print(f"    → Agreement alone is sufficient for predicting step correctness")
            print(f"    → Verifier/Selector can use agreement without tracking error history")
        else:
            print(f"\n  ★ History effect is SIGNIFICANT ({avg_delta:+.4f})")
            print(f"    → Error cascade has Markov order-1 structure")
            print(f"    → Previous step outcome affects current step probability")
            print(f"    → Verifier should consider error history for better predictions")

    # --- TSR prediction from Markov model ---
    print(f"\n{'='*70}")
    print("THEORETICAL TSR PREDICTION (from Markov model)")
    print(f"{'='*70}")

    # Get overall step accuracy distribution
    total_correct = sum(d['correct'] for d in order0.values())
    total_all = sum(d['total'] for d in order0.values())
    overall_acc = total_correct / total_all if total_all > 0 else 0

    print(f"  Overall step accuracy: {overall_acc:.4f}")

    # Predict TSR for different trajectory lengths assuming i.i.d. steps
    for length in [1, 2, 3, 5, 7, 10]:
        iid_tsr = overall_acc ** length
        print(f"  Length {length:2d}: i.i.d. TSR = {iid_tsr:.4f} ({iid_tsr*100:.1f}%)")

    # --- Cross-dataset comparison ---
    print(f"\n{'='*70}")
    print("CROSS-DATASET ERROR CASCADE COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Metric':>25} | {'GUI-360':>10} | {'AC':>10}")
    print(f"  {'-'*25} | {'-'*10} | {'-'*10}")
    print(f"  {'Silent fail rate':>25} | {'36.1%':>10} | {'53.5%':>10}")
    print(f"  {'Step-level accuracy':>25} | {'81.6%':>10} | {overall_acc*100:.1f}%{'':>4}")
    print(f"  {'Avg consecutive correct':>25} | {'2.4':>10} | {'1.05':>10}")

    # Compute average consecutive correct from data
    run_lengths = []
    for eid, traj in traj_results.items():
        steps = traj.get('step_results', [])
        current_run = 0
        for step in steps:
            if step.get('extract_match', False):
                current_run += 1
            else:
                if current_run > 0:
                    run_lengths.append(current_run)
                current_run = 0
        if current_run > 0:
            run_lengths.append(current_run)

    if run_lengths:
        avg_run = np.mean(run_lengths)
        print(f"\n  Computed avg consecutive correct: {avg_run:.2f}")
        print(f"  P(run >= 3): {sum(1 for r in run_lengths if r >= 3) / len(run_lengths):.4f}")

    # --- Save results ---
    results = {
        'n_matched_episodes': n_matched,
        'n_transitions': n_transitions,
        'order0': {
            abin: {
                'total': order0[abin]['total'],
                'p_correct': order0[abin]['correct'] / order0[abin]['total'] if order0[abin]['total'] > 0 else 0,
            }
            for abin in ['low', 'med', 'high', 'vhigh']
        },
        'order1': {
            f"{abin}_{prev}": {
                'total': transitions[(abin, prev)]['total'],
                'p_correct': transitions[(abin, prev)]['correct'] / transitions[(abin, prev)]['total']
                if transitions[(abin, prev)]['total'] > 0 else 0,
            }
            for abin in ['low', 'med', 'high', 'vhigh']
            for prev in [True, False]
        },
        'history_effect': float(avg_delta) if deltas else None,
        'overall_step_accuracy': overall_acc,
        'avg_consecutive_correct': float(avg_run) if run_lengths else None,
    }

    out_path = os.path.join(args.output_dir, 'analysis_b_error_cascade.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analysis B: Error Cascade Markov Structure")
    parser.add_argument("--trajectory_jsonl", type=str, required=True,
                        help="Path to Eval A trajectory_results.jsonl")
    parser.add_argument("--multisample_jsonl", type=str, required=True,
                        help="Path to C4+C7 multisample_results.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_analysis_b",
                        help="Output directory")
    args = parser.parse_args()
    main(args)
