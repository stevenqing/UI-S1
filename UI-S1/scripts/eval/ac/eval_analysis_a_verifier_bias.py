"""Analysis A: Verifier Step-Position Bias Measurement.

Explains the U7 paradox: per-step accuracy 57.4% < baseline 62%,
but TSR +0.59pp. Tests whether Verifier PASS/FAIL distribution
is asymmetric across step positions.

Hypothesis: Verifier PASS rate increases with step position,
making the step-position-weighted accuracy higher for U7 than baseline
even though the uniform-average is lower.

All offline — uses existing U7 results.
"""

import argparse
import json
import os
import numpy as np
from collections import defaultdict


def load_u7_results(jsonl_path):
    """Load U7 actor-verifier trajectory results."""
    results = []
    with open(jsonl_path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def load_baseline_results(jsonl_path):
    """Load baseline (Eval A) trajectory results."""
    results = []
    with open(jsonl_path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def analyze_verifier_by_step(u7_results):
    """Analyze Verifier PASS/FAIL behavior by step position."""
    by_step = defaultdict(lambda: {
        'total': 0, 'pass': 0, 'fail': 0,
        'pass_correct': 0, 'fail_correct': 0,
        'correct': 0,
    })

    for traj in u7_results:
        for step in traj.get('step_results', []):
            sn = step['step_num']
            s = by_step[sn]
            s['total'] += 1
            is_correct = step.get('extract_match', False)
            if is_correct:
                s['correct'] += 1

            verdict = step.get('verified', 'PASS')
            if verdict == 'PASS':
                s['pass'] += 1
                if is_correct:
                    s['pass_correct'] += 1
            else:
                s['fail'] += 1
                if is_correct:
                    s['fail_correct'] += 1

    return by_step


def analyze_baseline_by_step(baseline_results):
    """Get baseline per-step accuracy."""
    by_step = defaultdict(lambda: {'total': 0, 'correct': 0})

    for traj in baseline_results:
        for step in traj.get('step_results', []):
            sn = step['step_num']
            by_step[sn]['total'] += 1
            if step.get('extract_match', False):
                by_step[sn]['correct'] += 1

    return by_step


def compute_weighted_accuracy(by_step_data, max_step=20):
    """Compute step-position-weighted accuracy.

    Weight w_k = P(reaching step k) ≈ product of accuracies at steps 0..k-1.
    This reflects the TSR contribution of each step position.
    """
    # First compute per-step accuracy
    step_accs = {}
    for sn in sorted(by_step_data.keys()):
        if sn > max_step:
            break
        s = by_step_data[sn]
        if s['total'] > 0:
            step_accs[sn] = s['correct'] / s['total']

    if not step_accs:
        return 0.0, {}

    # Compute weights: w_k = product of p_0 ... p_{k-1}
    weights = {}
    for sn in sorted(step_accs.keys()):
        if sn == 0:
            weights[sn] = 1.0  # always start at step 0
        else:
            prev_acc = step_accs.get(sn - 1, 0.5)
            weights[sn] = weights.get(sn - 1, 1.0) * prev_acc

    # Weighted accuracy
    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0, weights

    weighted_acc = sum(weights[sn] * step_accs[sn] for sn in step_accs) / total_weight
    return weighted_acc, weights


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading U7 results...")
    u7_results = load_u7_results(args.u7_jsonl)
    print(f"  Loaded {len(u7_results)} U7 trajectories")

    baseline_results = None
    if args.baseline_jsonl and os.path.exists(args.baseline_jsonl):
        print("Loading baseline results...")
        baseline_results = load_baseline_results(args.baseline_jsonl)
        print(f"  Loaded {len(baseline_results)} baseline trajectories")

    # --- U7 Verifier behavior by step position ---
    print("\n" + "=" * 80)
    print("ANALYSIS A: VERIFIER STEP-POSITION BIAS")
    print("=" * 80)

    u7_by_step = analyze_verifier_by_step(u7_results)

    print(f"\n{'Step':>5} | {'N':>5} | {'PASS%':>6} | {'PASS Acc':>9} | {'FAIL Acc':>9} | {'Overall':>8} | {'Baseline':>9}")
    print(f"{'-'*5} | {'-'*5} | {'-'*6} | {'-'*9} | {'-'*9} | {'-'*8} | {'-'*9}")

    baseline_by_step = analyze_baseline_by_step(baseline_results) if baseline_results else {}

    for sn in sorted(u7_by_step.keys()):
        if sn > 15:
            break
        s = u7_by_step[sn]
        if s['total'] == 0:
            continue

        pass_rate = s['pass'] / s['total'] * 100
        pass_acc = s['pass_correct'] / s['pass'] if s['pass'] > 0 else 0
        fail_acc = s['fail_correct'] / s['fail'] if s['fail'] > 0 else 0
        overall = s['correct'] / s['total']

        bl = baseline_by_step.get(sn, {})
        bl_acc = bl['correct'] / bl['total'] if bl.get('total', 0) > 0 else 0

        print(f"{sn:5d} | {s['total']:5d} | {pass_rate:5.1f}% | {pass_acc:9.4f} | {fail_acc:9.4f} | {overall:8.4f} | {bl_acc:9.4f}")

    # --- PASS rate trend ---
    print("\n--- PASS Rate Trend ---")
    step_nums = sorted(sn for sn in u7_by_step.keys() if sn <= 15 and u7_by_step[sn]['total'] >= 10)
    if len(step_nums) >= 3:
        early_steps = [sn for sn in step_nums if sn <= 2]
        late_steps = [sn for sn in step_nums if sn >= 4]

        early_pass_rate = sum(u7_by_step[sn]['pass'] for sn in early_steps) / sum(u7_by_step[sn]['total'] for sn in early_steps) if early_steps else 0
        late_pass_rate = sum(u7_by_step[sn]['pass'] for sn in late_steps) / sum(u7_by_step[sn]['total'] for sn in late_steps) if late_steps else 0

        print(f"  Early steps (0-2) PASS rate: {early_pass_rate:.4f}")
        print(f"  Late steps (4+) PASS rate:   {late_pass_rate:.4f}")
        print(f"  Delta: {late_pass_rate - early_pass_rate:+.4f}")

        if late_pass_rate > early_pass_rate:
            print(f"  ✓ PASS rate increases with step position — hypothesis SUPPORTED")
        else:
            print(f"  ✗ PASS rate does NOT increase with step position — hypothesis REJECTED")

    # --- Step-position-weighted accuracy ---
    print("\n--- Step-Position-Weighted Accuracy ---")

    # U7
    u7_weighted, u7_weights = compute_weighted_accuracy(u7_by_step)
    u7_uniform = np.mean([u7_by_step[sn]['correct'] / u7_by_step[sn]['total']
                          for sn in u7_by_step if u7_by_step[sn]['total'] > 0])

    print(f"  U7 uniform-average accuracy:    {u7_uniform:.4f}")
    print(f"  U7 weighted accuracy:           {u7_weighted:.4f}")
    print(f"  U7 weighted - uniform delta:    {u7_weighted - u7_uniform:+.4f}")

    # Baseline
    if baseline_by_step:
        bl_weighted, bl_weights = compute_weighted_accuracy(baseline_by_step)
        bl_uniform = np.mean([baseline_by_step[sn]['correct'] / baseline_by_step[sn]['total']
                              for sn in baseline_by_step if baseline_by_step[sn]['total'] > 0])

        print(f"\n  Baseline uniform-average accuracy: {bl_uniform:.4f}")
        print(f"  Baseline weighted accuracy:        {bl_weighted:.4f}")
        print(f"  Baseline weighted - uniform delta: {bl_weighted - bl_uniform:+.4f}")

        print(f"\n  U7 - Baseline (uniform):   {u7_uniform - bl_uniform:+.4f}")
        print(f"  U7 - Baseline (weighted):  {u7_weighted - bl_weighted:+.4f}")

        if u7_weighted > bl_weighted and u7_uniform < bl_uniform:
            print(f"\n  ★ PARADOX EXPLAINED: U7 weighted > baseline weighted,")
            print(f"    but U7 uniform < baseline uniform.")
            print(f"    → Verifier improves accuracy at high-weight steps (late steps)")
            print(f"       while degrading accuracy at low-weight steps (early steps)")
        elif u7_weighted > bl_weighted:
            print(f"\n  ★ U7 weighted > baseline weighted — consistent with TSR improvement")
        else:
            print(f"\n  ✗ U7 weighted ≤ baseline weighted — paradox not fully explained by weighting")

    # --- Per-step weight visualization ---
    print("\n--- Step Weights (TSR contribution) ---")
    print(f"  {'Step':>5} | {'Weight':>8} | {'U7 Acc':>8} | {'BL Acc':>8} | {'U7 contribution':>16} | {'BL contribution':>16}")
    for sn in sorted(u7_by_step.keys()):
        if sn > 10:
            break
        s = u7_by_step[sn]
        if s['total'] == 0:
            continue
        u7_acc = s['correct'] / s['total']
        w = u7_weights.get(sn, 0)
        bl = baseline_by_step.get(sn, {})
        bl_acc = bl['correct'] / bl['total'] if bl.get('total', 0) > 0 else 0
        bw = bl_weights.get(sn, 0) if baseline_by_step else 0
        print(f"  {sn:5d} | {w:8.4f} | {u7_acc:8.4f} | {bl_acc:8.4f} | {w * u7_acc:16.4f} | {bw * bl_acc:16.4f}")

    # --- Resample win/loss analysis ---
    print("\n--- Resample Win/Loss by Step Position ---")
    for sn in sorted(u7_by_step.keys()):
        if sn > 10:
            break
        s = u7_by_step[sn]
        if s['fail'] == 0:
            continue
        fail_acc = s['fail_correct'] / s['fail']
        bl = baseline_by_step.get(sn, {})
        bl_acc = bl['correct'] / bl['total'] if bl.get('total', 0) > 0 else 0
        delta = fail_acc - bl_acc
        status = "WIN" if delta > 0 else "LOSS"
        print(f"  Step {sn}: resample_acc={fail_acc:.3f}, baseline_acc={bl_acc:.3f}, delta={delta:+.3f} ({status})")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    # Compute overall PASS/FAIL stats
    total_pass = sum(s['pass'] for s in u7_by_step.values())
    total_fail = sum(s['fail'] for s in u7_by_step.values())
    total_all = total_pass + total_fail
    pass_correct = sum(s['pass_correct'] for s in u7_by_step.values())
    fail_correct = sum(s['fail_correct'] for s in u7_by_step.values())

    print(f"  Total steps: {total_all}")
    print(f"  PASS: {total_pass} ({total_pass/total_all*100:.1f}%), accuracy: {pass_correct/total_pass:.4f}")
    print(f"  FAIL: {total_fail} ({total_fail/total_all*100:.1f}%), accuracy: {fail_correct/total_fail:.4f}")
    print(f"  Overall accuracy: {(pass_correct+fail_correct)/total_all:.4f}")

    # Save results
    results = {
        'u7_uniform_accuracy': float(u7_uniform),
        'u7_weighted_accuracy': float(u7_weighted),
        'per_step': {
            str(sn): {
                'total': s['total'],
                'pass_rate': s['pass'] / s['total'] if s['total'] > 0 else 0,
                'pass_accuracy': s['pass_correct'] / s['pass'] if s['pass'] > 0 else 0,
                'fail_accuracy': s['fail_correct'] / s['fail'] if s['fail'] > 0 else 0,
                'overall_accuracy': s['correct'] / s['total'] if s['total'] > 0 else 0,
            }
            for sn, s in sorted(u7_by_step.items()) if s['total'] > 0
        },
    }
    if baseline_by_step:
        results['baseline_uniform_accuracy'] = float(bl_uniform)
        results['baseline_weighted_accuracy'] = float(bl_weighted)

    out_path = os.path.join(args.output_dir, 'analysis_a_verifier_bias.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analysis A: Verifier Step-Position Bias")
    parser.add_argument("--u7_jsonl", type=str, required=True,
                        help="Path to U7 actor_verifier_results.jsonl")
    parser.add_argument("--baseline_jsonl", type=str, default=None,
                        help="Path to Eval A trajectory_results.jsonl (optional)")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_analysis_a",
                        help="Output directory")
    args = parser.parse_args()
    main(args)
