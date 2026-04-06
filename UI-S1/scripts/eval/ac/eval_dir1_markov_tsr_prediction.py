"""Direction 1: Markov Model TSR Prediction.

Uses Analysis B's transition matrix to predict TSR for:
1. Baseline (Eval A) trajectories — compare i.i.d. vs Markov order-1 predictions
2. U7 trajectories — test if Markov model can predict intervention effects

If Markov model error < i.i.d. model error, error cascade is structural, not noise.
If Markov model predicts U7 TSR, we can predict unseen interventions' TSR.

All offline — 0 GPU required.
"""

import argparse
import json
import os
import numpy as np
from collections import defaultdict, Counter


def load_jsonl(path):
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def load_analysis_b(path):
    with open(path) as f:
        return json.load(f)


def get_agreement_bin(agreement):
    if agreement < 0.5:
        return 'low'
    elif agreement < 0.7:
        return 'med'
    elif agreement < 0.9:
        return 'high'
    else:
        return 'vhigh'


def build_transition_matrix_from_multisample(multisample_path):
    """Build full Markov order-1 transition matrix from C4+C7 data.

    Returns:
        order0: {agree_bin: p_correct}
        order1: {(agree_bin, prev_correct): p_correct}
        initial: {agree_bin: p_correct}  (for step 0, no prev)
        agree_dist: {agree_bin: fraction}  (marginal distribution of agreement bins)
        agree_dist_by_prev: {prev_correct: {agree_bin: fraction}}
    """
    order0_counts = defaultdict(lambda: {'correct': 0, 'total': 0})
    order1_counts = defaultdict(lambda: {'correct': 0, 'total': 0})
    initial_counts = defaultdict(lambda: {'correct': 0, 'total': 0})

    with open(multisample_path) as f:
        for line in f:
            ep = json.loads(line)
            step_data = []

            for step in ep.get('step_samples', []):
                samples = step.get('samples', [])
                if not samples:
                    continue
                K = len(samples)
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

            if not step_data:
                continue

            step_data.sort(key=lambda x: x['step_num'])

            for i, sd in enumerate(step_data):
                abin = get_agreement_bin(sd['agreement'])
                correct = sd['greedy_correct']

                order0_counts[abin]['correct' if correct else 'x'] = \
                    order0_counts[abin].get('correct' if correct else 'x', 0)
                order0_counts[abin]['total'] += 1
                if correct:
                    order0_counts[abin]['correct'] += 1

                if i == 0:
                    initial_counts[abin]['total'] += 1
                    if correct:
                        initial_counts[abin]['correct'] += 1
                else:
                    prev_correct = step_data[i-1]['greedy_correct']
                    key = (abin, prev_correct)
                    order1_counts[key]['total'] += 1
                    if correct:
                        order1_counts[key]['correct'] += 1

    # Convert to probabilities
    order0 = {}
    for abin, d in order0_counts.items():
        order0[abin] = d['correct'] / d['total'] if d['total'] > 0 else 0.5

    order1 = {}
    for key, d in order1_counts.items():
        order1[key] = d['correct'] / d['total'] if d['total'] > 0 else 0.5

    initial = {}
    for abin, d in initial_counts.items():
        initial[abin] = d['correct'] / d['total'] if d['total'] > 0 else 0.5

    # Agreement bin distribution
    total_all = sum(d['total'] for d in order0_counts.values())
    agree_dist = {abin: order0_counts[abin]['total'] / total_all for abin in order0_counts}

    return order0, order1, initial, agree_dist


def predict_tsr_iid(per_step_accuracy, length_distribution):
    """Predict TSR under i.i.d. assumption: TSR = sum over lengths of p^N * P(length=N)."""
    total_tsr = 0
    total_weight = 0
    for length, count in length_distribution.items():
        tsr = per_step_accuracy ** length
        total_tsr += tsr * count
        total_weight += count
    return total_tsr / total_weight if total_weight > 0 else 0


def predict_tsr_iid_per_step(per_step_accuracies, trajectories):
    """Predict TSR using per-step-position accuracy (still i.i.d. across trajectories).

    TSR for trajectory of length N = product(p_0, p_1, ..., p_{N-1})
    """
    total_tsr = 0
    for traj in trajectories:
        n_steps = traj['num_steps']
        tsr = 1.0
        for k in range(n_steps):
            p_k = per_step_accuracies.get(k, per_step_accuracies.get(max(per_step_accuracies.keys()), 0.5))
            tsr *= p_k
        total_tsr += tsr
    return total_tsr / len(trajectories) if trajectories else 0


def predict_tsr_markov(order0, order1, initial, multisample_path, trajectories):
    """Predict TSR using Markov order-1 model.

    For each trajectory of length N, simulate:
      P(all correct) = P(correct at step 0) * prod_{k=1}^{N-1} P(correct at k | correct at k-1)

    Since we need agreement at each step, we use the marginal agreement distribution
    from the multisample data. For a more accurate prediction, we compute the expected
    TSR by marginalizing over agreement bins.
    """
    # We need per-step agreement data. Load from multisample.
    # Build per-episode, per-step agreement from C4+C7
    episode_agreements = {}
    with open(multisample_path) as f:
        for line in f:
            ep = json.loads(line)
            eid = ep.get('episode_id')
            step_agrees = {}
            for step in ep.get('step_samples', []):
                samples = step.get('samples', [])
                if not samples:
                    continue
                K = len(samples)
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
                    step_agrees[step.get('step_num', 0)] = agreement
            if step_agrees:
                episode_agreements[eid] = step_agrees

    total_tsr = 0
    matched = 0

    for traj in trajectories:
        eid = traj['episode_id']
        n_steps = traj['num_steps']

        if eid not in episode_agreements:
            # Fall back to marginal agreement distribution
            continue

        step_agrees = episode_agreements[eid]

        # Compute P(all correct) using Markov chain
        # P(all correct) = P(c_0) * P(c_1|c_0) * P(c_2|c_1) * ...
        # where P(c_k|c_{k-1}) = P(correct | agree_bin_k, prev_correct=True)
        #       P(c_0) = P(correct | agree_bin_0) from initial distribution

        prob = 1.0
        for k in range(n_steps):
            agree = step_agrees.get(k, 0.8)  # default
            abin = get_agreement_bin(agree)

            if k == 0:
                p_correct = initial.get(abin, order0.get(abin, 0.5))
            else:
                # For TSR, all previous steps must be correct, so prev_correct=True
                p_correct = order1.get((abin, True), order0.get(abin, 0.5))

            prob *= p_correct

        total_tsr += prob
        matched += 1

    return total_tsr / matched if matched > 0 else 0, matched


def predict_tsr_markov_full(order0, order1, initial, multisample_path, trajectories):
    """Markov order-1 with full state tracking (not just the all-correct path).

    Computes P(all steps correct) by tracking the full distribution over
    (correct, incorrect) at each step, conditioned on agreement and prev state.

    This is more accurate than the simplified version because it accounts for
    the fact that agreement bins vary across steps.
    """
    episode_agreements = {}
    with open(multisample_path) as f:
        for line in f:
            ep = json.loads(line)
            eid = ep.get('episode_id')
            step_agrees = {}
            for step in ep.get('step_samples', []):
                samples = step.get('samples', [])
                if not samples:
                    continue
                K = len(samples)
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
                    step_agrees[step.get('step_num', 0)] = agreement
            if step_agrees:
                episode_agreements[eid] = step_agrees

    total_tsr = 0
    matched = 0

    for traj in trajectories:
        eid = traj['episode_id']
        n_steps = traj['num_steps']

        if eid not in episode_agreements:
            continue

        step_agrees = episode_agreements[eid]

        # For stop-on-error TSR, we only care about P(all correct)
        # In a Markov chain:
        # P(all correct) = P(c_0) * prod_{k=1}^{N-1} P(c_k | c_{k-1}=True)
        # This is valid because TSR requires ALL steps correct,
        # so at each step k>0, we condition on prev being correct.

        prob = 1.0
        for k in range(n_steps):
            agree = step_agrees.get(k, 0.8)
            abin = get_agreement_bin(agree)

            if k == 0:
                p_correct = initial.get(abin, order0.get(abin, 0.5))
            else:
                p_correct = order1.get((abin, True), order0.get(abin, 0.5))

            prob *= p_correct

        total_tsr += prob
        matched += 1

    return total_tsr / matched if matched > 0 else 0, matched


def compute_actual_tsr(trajectories):
    """Compute actual TSR from trajectory results."""
    n_success = sum(1 for t in trajectories if t.get('task_success', False))
    return n_success / len(trajectories) if trajectories else 0


def compute_per_step_accuracy(trajectories):
    """Compute per-step-position accuracy."""
    by_step = defaultdict(lambda: {'correct': 0, 'total': 0})
    for traj in trajectories:
        for step in traj.get('step_results', []):
            sn = step['step_num']
            by_step[sn]['total'] += 1
            if step.get('extract_match', False):
                by_step[sn]['correct'] += 1
    return {sn: d['correct'] / d['total'] for sn, d in by_step.items() if d['total'] > 0}


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading baseline (Eval A) trajectories...")
    baseline = load_jsonl(args.baseline_jsonl)
    print(f"  Loaded {len(baseline)} trajectories")

    u7_data = None
    if args.u7_jsonl and os.path.exists(args.u7_jsonl):
        print("Loading U7 trajectories...")
        u7_data = load_jsonl(args.u7_jsonl)
        print(f"  Loaded {len(u7_data)} trajectories")

    # Build transition matrix from C4+C7
    print("\nBuilding Markov transition matrix from C4+C7 data...")
    order0, order1, initial, agree_dist = build_transition_matrix_from_multisample(args.multisample_jsonl)

    print("\n  Order-0 (agreement → accuracy):")
    for abin in ['low', 'med', 'high', 'vhigh']:
        print(f"    {abin:>6}: P(correct) = {order0.get(abin, 0):.4f}")

    print("\n  Initial (step 0):")
    for abin in ['low', 'med', 'high', 'vhigh']:
        print(f"    {abin:>6}: P(correct) = {initial.get(abin, 0):.4f}")

    print("\n  Order-1 (agreement × prev_correct → accuracy):")
    for abin in ['low', 'med', 'high', 'vhigh']:
        for prev in [True, False]:
            p = order1.get((abin, prev), 0)
            label = 'PASS' if prev else 'FAIL'
            print(f"    {abin:>6} × {label}: P(correct) = {p:.4f}")

    print(f"\n  Agreement distribution: {agree_dist}")

    # ===================================================================
    # PREDICTION 1: Baseline TSR
    # ===================================================================
    print(f"\n{'='*70}")
    print("PREDICTION 1: BASELINE TSR")
    print(f"{'='*70}")

    actual_tsr = compute_actual_tsr(baseline)
    per_step_acc = compute_per_step_accuracy(baseline)
    overall_acc = sum(d.get('extract_match', False) for t in baseline for d in t.get('step_results', [])) / \
                  sum(len(t.get('step_results', [])) for t in baseline)

    print(f"\n  Actual baseline TSR: {actual_tsr:.4f} ({actual_tsr*100:.2f}%)")
    print(f"  Overall step accuracy: {overall_acc:.4f}")
    print(f"  Per-step accuracies: {dict(sorted(per_step_acc.items())[:10])}")

    # Model 1: Simple i.i.d. with uniform step accuracy
    length_dist = defaultdict(int)
    for t in baseline:
        length_dist[t['num_steps']] += 1

    iid_uniform_tsr = predict_tsr_iid(overall_acc, length_dist)
    print(f"\n  Model 1 (i.i.d. uniform): TSR = {iid_uniform_tsr:.4f} ({iid_uniform_tsr*100:.2f}%)")
    print(f"    Error: {abs(iid_uniform_tsr - actual_tsr):.4f} ({abs(iid_uniform_tsr - actual_tsr)/actual_tsr*100:.1f}%)")

    # Model 2: i.i.d. per-step-position
    iid_perstep_tsr = predict_tsr_iid_per_step(per_step_acc, baseline)
    print(f"\n  Model 2 (i.i.d. per-step-position): TSR = {iid_perstep_tsr:.4f} ({iid_perstep_tsr*100:.2f}%)")
    print(f"    Error: {abs(iid_perstep_tsr - actual_tsr):.4f} ({abs(iid_perstep_tsr - actual_tsr)/actual_tsr*100:.1f}%)")

    # Model 3: Markov order-1
    markov_tsr, markov_matched = predict_tsr_markov(
        order0, order1, initial, args.multisample_jsonl, baseline)
    print(f"\n  Model 3 (Markov order-1): TSR = {markov_tsr:.4f} ({markov_tsr*100:.2f}%)")
    print(f"    Matched episodes: {markov_matched}/{len(baseline)}")
    print(f"    Error: {abs(markov_tsr - actual_tsr):.4f} ({abs(markov_tsr - actual_tsr)/actual_tsr*100:.1f}%)")

    # Summary comparison
    print(f"\n  {'Model':<30} | {'Predicted':>10} | {'Actual':>10} | {'Error':>10} | {'Rel Error':>10}")
    print(f"  {'-'*30} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")
    for name, pred in [('i.i.d. uniform', iid_uniform_tsr),
                        ('i.i.d. per-step', iid_perstep_tsr),
                        ('Markov order-1', markov_tsr)]:
        err = abs(pred - actual_tsr)
        rel = err / actual_tsr * 100 if actual_tsr > 0 else 0
        print(f"  {name:<30} | {pred*100:9.2f}% | {actual_tsr*100:9.2f}% | {err*100:9.2f}% | {rel:9.1f}%")

    # ===================================================================
    # PREDICTION 1b: Baseline TSR by length bucket
    # ===================================================================
    print(f"\n{'='*70}")
    print("PREDICTION 1b: BASELINE TSR BY LENGTH BUCKET")
    print(f"{'='*70}")

    bucket_order = ['short(1-3)', 'medium(4-7)', 'long(8-15)', 'vlong(16+)']
    by_bucket = defaultdict(list)
    for t in baseline:
        by_bucket[t.get('length_bucket', 'unknown')].append(t)

    print(f"\n  {'Bucket':<15} | {'N':>5} | {'Actual':>8} | {'i.i.d.':>8} | {'Markov':>8} | {'Best':>8}")
    print(f"  {'-'*15} | {'-'*5} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")

    bucket_results = {}
    for bucket in bucket_order:
        trajs = by_bucket.get(bucket, [])
        if not trajs:
            continue
        actual = compute_actual_tsr(trajs)
        iid_pred = predict_tsr_iid_per_step(per_step_acc, trajs)
        markov_pred, _ = predict_tsr_markov(order0, order1, initial, args.multisample_jsonl, trajs)
        iid_err = abs(iid_pred - actual)
        mk_err = abs(markov_pred - actual)
        best = 'Markov' if mk_err < iid_err else 'i.i.d.'
        print(f"  {bucket:<15} | {len(trajs):5d} | {actual*100:7.2f}% | {iid_pred*100:7.2f}% | {markov_pred*100:7.2f}% | {best:>8}")
        bucket_results[bucket] = {
            'n': len(trajs),
            'actual_tsr': actual,
            'iid_tsr': iid_pred,
            'markov_tsr': markov_pred,
        }

    # ===================================================================
    # PREDICTION 2: U7 TSR
    # ===================================================================
    if u7_data:
        print(f"\n{'='*70}")
        print("PREDICTION 2: U7 TSR")
        print(f"{'='*70}")

        u7_actual_tsr = compute_actual_tsr(u7_data)
        u7_per_step = compute_per_step_accuracy(u7_data)
        u7_overall_acc = sum(d.get('extract_match', False) for t in u7_data for d in t.get('step_results', [])) / \
                         sum(len(t.get('step_results', [])) for t in u7_data)

        print(f"\n  Actual U7 TSR: {u7_actual_tsr:.4f} ({u7_actual_tsr*100:.2f}%)")
        print(f"  U7 overall step accuracy: {u7_overall_acc:.4f}")

        # U7 uses the same model but with verifier, so Markov parameters from
        # baseline C4+C7 may not perfectly apply. But if they do, it means
        # the verifier doesn't fundamentally change the error cascade structure.

        u7_iid_uniform = predict_tsr_iid(u7_overall_acc, length_dist)
        u7_iid_perstep = predict_tsr_iid_per_step(u7_per_step, u7_data)
        u7_markov, u7_matched = predict_tsr_markov(
            order0, order1, initial, args.multisample_jsonl, u7_data)

        print(f"\n  Model 1 (i.i.d. uniform): TSR = {u7_iid_uniform:.4f}")
        print(f"  Model 2 (i.i.d. per-step): TSR = {u7_iid_perstep:.4f}")
        print(f"  Model 3 (Markov, baseline params): TSR = {u7_markov:.4f} (matched: {u7_matched})")

        print(f"\n  {'Model':<30} | {'Predicted':>10} | {'Actual':>10} | {'Error':>10}")
        print(f"  {'-'*30} | {'-'*10} | {'-'*10} | {'-'*10}")
        for name, pred in [('i.i.d. uniform', u7_iid_uniform),
                            ('i.i.d. per-step', u7_iid_perstep),
                            ('Markov (baseline params)', u7_markov)]:
            err = abs(pred - u7_actual_tsr)
            print(f"  {name:<30} | {pred*100:9.2f}% | {u7_actual_tsr*100:9.2f}% | {err*100:9.2f}%")

        # Key question: does Markov model predict the TSR *improvement*?
        baseline_markov_pred = markov_tsr
        delta_pred = u7_markov - baseline_markov_pred
        delta_actual = u7_actual_tsr - actual_tsr
        print(f"\n  Predicted TSR delta (U7 - baseline): {delta_pred*100:+.2f}pp")
        print(f"  Actual TSR delta (U7 - baseline):    {delta_actual*100:+.2f}pp")

        if abs(delta_pred - delta_actual) < 0.02:
            print(f"  ★ Markov model SUCCESSFULLY predicts intervention effect!")
        else:
            print(f"  ✗ Markov model cannot predict intervention effect — U7 changes cascade structure")

    # ===================================================================
    # STRUCTURAL ANALYSIS: Does the Markov model capture compounding?
    # ===================================================================
    print(f"\n{'='*70}")
    print("STRUCTURAL ANALYSIS: COMPOUNDING IN MARKOV MODEL")
    print(f"{'='*70}")

    # Compare predicted vs actual compounding curves
    # For steps 0-7, compare:
    #   Actual: fraction of trajectories that are correct at step k given they reached k
    #   Markov: P(correct at k | all previous correct)

    print(f"\n  {'Step':>5} | {'Actual P(c|reached)':>20} | {'Markov P(c|all prev c)':>23} | {'Delta':>8}")
    print(f"  {'-'*5} | {'-'*20} | {'-'*23} | {'-'*8}")

    for k in range(min(8, max(per_step_acc.keys()) + 1)):
        if k not in per_step_acc:
            continue
        actual_p = per_step_acc[k]

        # For Markov: what's the marginal P(correct at step k) given all previous correct?
        # This depends on the agreement distribution at step k.
        # We approximate using the overall agreement distribution.
        if k == 0:
            markov_p = sum(initial.get(abin, 0) * agree_dist.get(abin, 0)
                          for abin in agree_dist)
        else:
            # Conditioned on prev=correct
            markov_p = sum(order1.get((abin, True), order0.get(abin, 0)) * agree_dist.get(abin, 0)
                          for abin in agree_dist)

        delta = markov_p - actual_p
        print(f"  {k:5d} | {actual_p:20.4f} | {markov_p:23.4f} | {delta:+8.4f}")

    # ===================================================================
    # Save results
    # ===================================================================
    results = {
        'baseline': {
            'actual_tsr': actual_tsr,
            'iid_uniform_tsr': iid_uniform_tsr,
            'iid_perstep_tsr': iid_perstep_tsr,
            'markov_tsr': markov_tsr,
            'markov_matched': markov_matched,
            'overall_step_accuracy': overall_acc,
            'per_bucket': bucket_results,
        },
        'model_parameters': {
            'order0': {k: float(v) for k, v in order0.items()},
            'order1': {f"{k[0]}_{k[1]}": float(v) for k, v in order1.items()},
            'initial': {k: float(v) for k, v in initial.items()},
            'agree_dist': {k: float(v) for k, v in agree_dist.items()},
        },
    }

    if u7_data:
        results['u7'] = {
            'actual_tsr': u7_actual_tsr,
            'iid_uniform_tsr': u7_iid_uniform,
            'iid_perstep_tsr': u7_iid_perstep,
            'markov_tsr': u7_markov,
            'markov_matched': u7_matched,
            'overall_step_accuracy': u7_overall_acc,
            'predicted_delta': float(u7_markov - markov_tsr),
            'actual_delta': float(u7_actual_tsr - actual_tsr),
        }

    out_path = os.path.join(args.output_dir, 'dir1_markov_tsr_prediction.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direction 1: Markov Model TSR Prediction")
    parser.add_argument("--baseline_jsonl", type=str, required=True,
                        help="Path to Eval A trajectory_results.jsonl")
    parser.add_argument("--u7_jsonl", type=str, default=None,
                        help="Path to U7 actor_verifier_results.jsonl")
    parser.add_argument("--multisample_jsonl", type=str, required=True,
                        help="Path to C4+C7 multisample_results.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_dir1",
                        help="Output directory")
    args = parser.parse_args()
    main(args)
