"""Pre-test 1: Credit Analysis (Offline).

Per-step accuracy curves. If D1 results available, compute observer credit by step position.
"""

import argparse
import os
from collections import defaultdict

from ac_utils import load_jsonl, save_json, length_bucket


def main(args):
    baseline = load_jsonl(args.baseline_file)
    print(f"Loaded baseline: {len(baseline)} episodes.")

    # Per-step accuracy curve
    step_acc = defaultdict(lambda: {'correct': 0, 'total': 0})

    for r in baseline:
        for s in r['step_results']:
            pos = s['step_num']
            step_acc[pos]['total'] += 1
            step_acc[pos]['correct'] += int(s['extract_match'])

    step_curve = {}
    for pos in sorted(step_acc.keys()):
        t = step_acc[pos]['total']
        step_curve[str(pos)] = {
            **step_acc[pos],
            'accuracy': step_acc[pos]['correct'] / t if t > 0 else 0,
        }

    # Cumulative success (probability of reaching step k)
    # P(reach step k) = prod(P(correct at step i) for i in 0..k-1)
    cumulative = {}
    prob = 1.0
    for pos in sorted(step_acc.keys()):
        t = step_acc[pos]['total']
        acc = step_acc[pos]['correct'] / t if t > 0 else 0
        prob *= acc
        cumulative[str(pos)] = prob

    # Observer credit by step position (if D1 available)
    observer_credit = None
    if args.observer_file and os.path.exists(args.observer_file):
        observer = load_jsonl(args.observer_file)
        observer_by_id = {r.get('episode_id'): r for r in observer}
        baseline_by_id = {r.get('episode_id'): r for r in baseline}

        # Per-step comparison
        step_observer_credit = defaultdict(lambda: {'baseline_correct': 0, 'observer_correct': 0, 'total': 0})

        for eid in set(baseline_by_id.keys()) & set(observer_by_id.keys()):
            b = baseline_by_id[eid]
            o = observer_by_id[eid]
            # Compare step by step
            max_steps = min(len(b['step_results']), len(o['step_results']))
            for i in range(max_steps):
                bs = b['step_results'][i]
                os_r = o['step_results'][i]
                if bs['step_num'] == os_r['step_num']:
                    pos = bs['step_num']
                    step_observer_credit[pos]['total'] += 1
                    step_observer_credit[pos]['baseline_correct'] += int(bs['extract_match'])
                    step_observer_credit[pos]['observer_correct'] += int(os_r['extract_match'])

        observer_credit = {}
        for pos in sorted(step_observer_credit.keys()):
            data = step_observer_credit[pos]
            t = data['total']
            b_acc = data['baseline_correct'] / t if t > 0 else 0
            o_acc = data['observer_correct'] / t if t > 0 else 0
            observer_credit[str(pos)] = {
                **data,
                'baseline_accuracy': b_acc,
                'observer_accuracy': o_acc,
                'credit': o_acc - b_acc,
            }

    summary = {
        'step_accuracy_curve': step_curve,
        'cumulative_success_probability': cumulative,
        'observer_credit_by_step': observer_credit,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(summary, os.path.join(args.output_dir, 'eval_pretest1_shapley.json'))

    print(f"\n=== Step Accuracy Curve ===")
    for pos, data in step_curve.items():
        print(f"  Step {pos}: {data['accuracy']:.3f} ({data['correct']}/{data['total']})")

    print(f"\n=== Cumulative Success ===")
    for pos, prob in cumulative.items():
        print(f"  Reach step {pos}: {prob:.4f}")

    if observer_credit:
        print(f"\n=== Observer Credit by Step ===")
        for pos, data in observer_credit.items():
            sign = '+' if data['credit'] >= 0 else ''
            print(f"  Step {pos}: {sign}{data['credit']:.3f} (base={data['baseline_accuracy']:.3f}, obs={data['observer_accuracy']:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-test 1: Credit Analysis")
    parser.add_argument("--baseline_file", type=str, required=True)
    parser.add_argument("--observer_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_pretest1_ac")
    args = parser.parse_args()
    main(args)
