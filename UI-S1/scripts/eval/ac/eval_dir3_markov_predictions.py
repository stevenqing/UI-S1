"""Direction 3: Pre-Registered Markov Predictions for U10 and U11.

Uses Analysis B's transition matrix to predict TSR for hypothetical interventions:
- U10: History-Aware Verifier (2D policy: agreement × prev_correct)
- U11: Action-History Observer (action disambiguation instead of state description)

These predictions are made BEFORE running the actual experiments,
serving as a pre-registered test of the Markov predictive framework.

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


def get_agreement_bin(agreement):
    if agreement < 0.5:
        return 'low'
    elif agreement < 0.7:
        return 'med'
    elif agreement < 0.9:
        return 'high'
    else:
        return 'vhigh'


def build_per_episode_agreements(multisample_path):
    """Build per-episode, per-step agreement and correctness from C4+C7."""
    episode_data = {}
    with open(multisample_path) as f:
        for line in f:
            ep = json.loads(line)
            eid = ep.get('episode_id')
            steps = {}
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

                # Action entropy
                probs = np.array([c / K for c in type_counter.values()])
                entropy = -np.sum(probs * np.log2(probs + 1e-10))

                greedy_correct = samples[0].get('extract_match', False)
                greedy_type_match = samples[0].get('type_match', False)

                steps[step.get('step_num', 0)] = {
                    'agreement': agreement,
                    'greedy_correct': greedy_correct,
                    'type_match': greedy_type_match,
                    'action_entropy': entropy,
                    'n_unique_types': len(type_counter),
                }
            if steps:
                episode_data[eid] = steps
    return episode_data


def build_transition_params(multisample_path):
    """Build Markov order-1 transition parameters from C4+C7."""
    order0 = defaultdict(lambda: {'correct': 0, 'total': 0})
    order1 = defaultdict(lambda: {'correct': 0, 'total': 0})
    initial = defaultdict(lambda: {'correct': 0, 'total': 0})

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
                order0[abin]['total'] += 1
                if correct:
                    order0[abin]['correct'] += 1
                if i == 0:
                    initial[abin]['total'] += 1
                    if correct:
                        initial[abin]['correct'] += 1
                else:
                    prev_correct = step_data[i-1]['greedy_correct']
                    key = (abin, prev_correct)
                    order1[key]['total'] += 1
                    if correct:
                        order1[key]['correct'] += 1

    # Convert to probabilities
    o0 = {k: d['correct'] / d['total'] if d['total'] > 0 else 0.5
          for k, d in order0.items()}
    o1 = {k: d['correct'] / d['total'] if d['total'] > 0 else 0.5
          for k, d in order1.items()}
    ini = {k: d['correct'] / d['total'] if d['total'] > 0 else 0.5
           for k, d in initial.items()}

    return o0, o1, ini


def simulate_u10_tsr(trajectories, episode_agreements, order0, order1, initial):
    """Simulate U10 (History-Aware Verifier) TSR.

    U10 policy:
    - If agreement >= 0.9 AND prev_correct: PASS → use greedy (accuracy from order1)
    - If agreement >= 0.9 AND prev_wrong: CONDITIONAL_PASS → use greedy but with ~5% correction
    - If agreement in [0.5, 0.7] AND prev_wrong: FAIL → resample with K=5, get ~+10pp improvement
    - If agreement in [0.5, 0.7] AND prev_correct: PASS → use greedy
    - If agreement in [0.7, 0.9]: similar to vhigh with slight discount
    - If agreement < 0.5: FAIL → resample (but history doesn't help, marginal improvement)

    For each trajectory, compute P(all steps correct) under this policy.
    """
    # U10 per-step accuracy modifications over baseline
    # Based on Analysis B data + U7 resample data
    u10_accuracy = {}

    # Step 0: no prev_correct available → use agreement-only policy
    # PASS if agreement >= 0.7, FAIL otherwise
    # PASS accuracy ≈ initial accuracy (greedy)
    # FAIL → resample: assume ~5pp improvement over greedy for low/med agreement
    for abin in ['low', 'med', 'high', 'vhigh']:
        u10_accuracy[('step0', abin)] = initial.get(abin, 0.5)
        if abin in ['low', 'med']:
            # Resample at step 0 for low/med agreement: modest improvement
            u10_accuracy[('step0', abin)] = min(initial.get(abin, 0.5) + 0.05, 0.95)

    # Steps 1+: use 2D policy (agreement × prev_correct)
    # Baseline accuracy from order1
    for abin in ['low', 'med', 'high', 'vhigh']:
        for prev in [True, False]:
            base = order1.get((abin, prev), order0.get(abin, 0.5))
            key = (abin, prev)

            if abin == 'vhigh' and prev:
                # PASS: use greedy, no change
                u10_accuracy[key] = base
            elif abin == 'vhigh' and not prev:
                # CONDITIONAL PASS: greedy but more cautious
                # Small improvement from being "warned" about prev error
                u10_accuracy[key] = base + 0.02  # conservative
            elif abin in ['high'] and prev:
                # PASS: use greedy
                u10_accuracy[key] = base
            elif abin in ['high'] and not prev:
                # CONDITIONAL: slight improvement from awareness
                u10_accuracy[key] = base + 0.03
            elif abin == 'med' and prev:
                # PASS: greedy (52%)
                u10_accuracy[key] = base
            elif abin == 'med' and not prev:
                # FAIL → resample with K=5
                # U7 data shows resample accuracy varies, but for med-agree + prev_wrong
                # this is the highest ROI zone. Assume +10pp over baseline 39.5%
                u10_accuracy[key] = min(base + 0.10, 0.95)  # 39.5% → ~49.5%
            elif abin == 'low':
                # FAIL → resample, but history doesn't help much
                u10_accuracy[key] = min(base + 0.03, 0.95)

    # Simulate TSR
    total_tsr = 0
    matched = 0
    per_step_decisions = defaultdict(lambda: {'pass': 0, 'fail': 0, 'conditional': 0})

    for traj in trajectories:
        eid = traj['episode_id']
        n_steps = traj['num_steps']
        if eid not in episode_agreements:
            continue

        step_data = episode_agreements[eid]
        prob = 1.0
        prev_correct_prob = 1.0  # For TSR, we condition on all-correct path

        for k in range(n_steps):
            sd = step_data.get(k)
            if sd is None:
                prob *= 0.5  # fallback
                continue

            abin = get_agreement_bin(sd['agreement'])

            if k == 0:
                p_correct = u10_accuracy.get(('step0', abin), initial.get(abin, 0.5))
                if abin in ['low', 'med']:
                    per_step_decisions[k]['fail'] += 1
                else:
                    per_step_decisions[k]['pass'] += 1
            else:
                # On the all-correct path, prev_correct = True
                p_correct = u10_accuracy.get((abin, True), order1.get((abin, True), 0.5))
                if abin in ['vhigh', 'high']:
                    per_step_decisions[k]['pass'] += 1
                elif abin == 'med':
                    per_step_decisions[k]['pass'] += 1  # prev=True → PASS
                else:
                    per_step_decisions[k]['fail'] += 1

            prob *= p_correct

        total_tsr += prob
        matched += 1

    tsr = total_tsr / matched if matched > 0 else 0

    # Compute PASS/FAIL rates
    total_decisions = sum(sum(d.values()) for d in per_step_decisions.values())
    total_pass = sum(d['pass'] for d in per_step_decisions.values())
    total_fail = sum(d['fail'] for d in per_step_decisions.values())
    total_cond = sum(d['conditional'] for d in per_step_decisions.values())

    return tsr, matched, {
        'pass_rate': total_pass / total_decisions if total_decisions > 0 else 0,
        'fail_rate': total_fail / total_decisions if total_decisions > 0 else 0,
        'conditional_rate': total_cond / total_decisions if total_decisions > 0 else 0,
    }


def simulate_u11_tsr(trajectories, episode_agreements, order0, order1, initial):
    """Simulate U11 (Action-History Observer) TSR.

    U11 idea: Observer outputs action disambiguation signal instead of state description.
    Key mechanism: for steps where prev was wrong AND action error was the cause,
    Observer tells actor "avoid [wrong action type]" → reduces action confusion.

    Modeling assumptions (conservative):
    - Observer correctly identifies action errors: 50%
    - When identified, actor corrects: 30%
    - Only applies to action errors (type_match=False), not grounding errors
    - Effect is strongest in med-agreement zone (where action entropy is high)
    """
    total_tsr = 0
    matched = 0
    corrections = 0
    total_steps = 0

    for traj in trajectories:
        eid = traj['episode_id']
        n_steps = traj['num_steps']
        if eid not in episode_agreements:
            continue

        step_data = episode_agreements[eid]
        prob = 1.0

        for k in range(n_steps):
            sd = step_data.get(k)
            if sd is None:
                prob *= 0.5
                continue

            abin = get_agreement_bin(sd['agreement'])

            if k == 0:
                p_correct = initial.get(abin, 0.5)
            else:
                # On all-correct path, prev=True
                p_correct = order1.get((abin, True), order0.get(abin, 0.5))

                # U11 correction: if this step has action confusion (high entropy)
                # and is in the medium-agreement zone, Observer can help
                if sd.get('action_entropy', 0) > 0.5 and abin in ['med', 'high']:
                    # Action error probability at this step
                    p_action_error = 1.0 - p_correct
                    # Fraction that are action type errors (vs grounding)
                    type_match_rate = 0.3 if abin == 'med' else 0.5  # estimate
                    action_error_frac = 1.0 - type_match_rate  # fraction of errors that are action errors
                    # Observer identifies 50% of action errors
                    p_identified = 0.5
                    # Actor corrects 30% of identified errors
                    p_corrected = 0.3

                    correction = p_action_error * action_error_frac * p_identified * p_corrected
                    p_correct += correction
                    p_correct = min(p_correct, 0.99)
                    if correction > 0.001:
                        corrections += 1

            prob *= p_correct
            total_steps += 1

        total_tsr += prob
        matched += 1

    tsr = total_tsr / matched if matched > 0 else 0
    return tsr, matched, corrections, total_steps


def simulate_baseline_tsr(trajectories, episode_agreements, order0, order1, initial):
    """Baseline TSR prediction using Markov model (for comparison)."""
    total_tsr = 0
    matched = 0

    for traj in trajectories:
        eid = traj['episode_id']
        n_steps = traj['num_steps']
        if eid not in episode_agreements:
            continue

        step_data = episode_agreements[eid]
        prob = 1.0

        for k in range(n_steps):
            sd = step_data.get(k)
            if sd is None:
                prob *= 0.5
                continue

            abin = get_agreement_bin(sd['agreement'])

            if k == 0:
                p_correct = initial.get(abin, 0.5)
            else:
                # On all-correct path, prev=True
                p_correct = order1.get((abin, True), order0.get(abin, 0.5))

            prob *= p_correct

        total_tsr += prob
        matched += 1

    return total_tsr / matched if matched > 0 else 0, matched


def compute_actual_tsr(trajectories):
    n_success = sum(1 for t in trajectories if t.get('task_success', False))
    return n_success / len(trajectories) if trajectories else 0


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    baseline = load_jsonl(args.baseline_jsonl)
    print(f"  Loaded {len(baseline)} baseline trajectories")

    u7_data = None
    if args.u7_jsonl and os.path.exists(args.u7_jsonl):
        u7_data = load_jsonl(args.u7_jsonl)
        print(f"  Loaded {len(u7_data)} U7 trajectories")

    print("Building Markov parameters from C4+C7...")
    order0, order1, initial = build_transition_params(args.multisample_jsonl)
    episode_agreements = build_per_episode_agreements(args.multisample_jsonl)
    print(f"  Built agreement data for {len(episode_agreements)} episodes")

    # Print Markov parameters
    print(f"\n  Order-0:")
    for abin in ['low', 'med', 'high', 'vhigh']:
        print(f"    {abin:>6}: {order0.get(abin, 0):.4f}")
    print(f"\n  Initial:")
    for abin in ['low', 'med', 'high', 'vhigh']:
        print(f"    {abin:>6}: {initial.get(abin, 0):.4f}")
    print(f"\n  Order-1:")
    for abin in ['low', 'med', 'high', 'vhigh']:
        for prev in [True, False]:
            label = 'PASS' if prev else 'FAIL'
            print(f"    {abin:>6}×{label}: {order1.get((abin, prev), 0):.4f}")

    # ===================================================================
    # BASELINE CALIBRATION
    # ===================================================================
    print(f"\n{'='*70}")
    print("BASELINE CALIBRATION")
    print(f"{'='*70}")

    actual_bl_tsr = compute_actual_tsr(baseline)
    pred_bl_tsr, bl_matched = simulate_baseline_tsr(
        baseline, episode_agreements, order0, order1, initial)

    print(f"  Actual baseline TSR: {actual_bl_tsr:.4f} ({actual_bl_tsr*100:.2f}%)")
    print(f"  Markov predicted:    {pred_bl_tsr:.4f} ({pred_bl_tsr*100:.2f}%)")
    print(f"  Error:               {abs(pred_bl_tsr - actual_bl_tsr)*100:.2f}pp")
    print(f"  Matched episodes:    {bl_matched}")

    # Calibration factor: actual / predicted
    cal_factor = actual_bl_tsr / pred_bl_tsr if pred_bl_tsr > 0 else 1.0
    print(f"  Calibration factor:  {cal_factor:.4f}")
    print(f"  (Multiplied to raw Markov predictions to account for systematic underprediction)")

    # U7 calibration check
    if u7_data:
        actual_u7_tsr = compute_actual_tsr(u7_data)
        pred_u7_tsr, u7_matched = simulate_baseline_tsr(
            u7_data, episode_agreements, order0, order1, initial)
        calibrated_u7 = pred_u7_tsr * cal_factor

        print(f"\n  U7 actual TSR:       {actual_u7_tsr:.4f}")
        print(f"  U7 raw predicted:    {pred_u7_tsr:.4f}")
        print(f"  U7 calibrated:       {calibrated_u7:.4f}")
        print(f"  Calibrated error:    {abs(calibrated_u7 - actual_u7_tsr)*100:.2f}pp")

    # ===================================================================
    # U10 PREDICTION: HISTORY-AWARE VERIFIER
    # ===================================================================
    print(f"\n{'='*70}")
    print("PRE-REGISTERED PREDICTION: U10 (History-Aware Verifier)")
    print(f"{'='*70}")

    u10_tsr, u10_matched, u10_decisions = simulate_u10_tsr(
        baseline, episode_agreements, order0, order1, initial)
    calibrated_u10 = u10_tsr * cal_factor

    print(f"\n  U10 Policy:")
    print(f"    High-agree + prev_correct → PASS (greedy)")
    print(f"    High-agree + prev_wrong   → CONDITIONAL PASS (+2-3pp correction)")
    print(f"    Med-agree + prev_correct  → PASS (greedy)")
    print(f"    Med-agree + prev_wrong    → FAIL+RESAMPLE (+10pp, highest ROI)")
    print(f"    Low-agree                 → FAIL+RESAMPLE (+3pp, minimal history help)")
    print(f"\n  Decision distribution:")
    print(f"    PASS rate:        {u10_decisions['pass_rate']:.4f}")
    print(f"    FAIL rate:        {u10_decisions['fail_rate']:.4f}")
    print(f"    Conditional rate: {u10_decisions['conditional_rate']:.4f}")

    print(f"\n  ★ PREDICTION: U10 TSR")
    print(f"    Raw Markov:     {u10_tsr:.4f} ({u10_tsr*100:.2f}%)")
    print(f"    Calibrated:     {calibrated_u10:.4f} ({calibrated_u10*100:.2f}%)")
    print(f"    Predicted delta: {(calibrated_u10 - actual_bl_tsr)*100:+.2f}pp over baseline")
    if u7_data:
        print(f"    Predicted delta: {(calibrated_u10 - actual_u7_tsr)*100:+.2f}pp over U7")
    print(f"    Matched: {u10_matched}")

    # Sensitivity analysis: vary the med-agree + prev_wrong improvement
    print(f"\n  --- Sensitivity Analysis (med-agree + prev_wrong improvement) ---")
    print(f"  {'Improvement':>12} | {'Raw TSR':>10} | {'Calibrated':>12} | {'Delta vs BL':>12}")
    print(f"  {'-'*12} | {'-'*10} | {'-'*12} | {'-'*12}")

    for improvement in [0.0, 0.05, 0.10, 0.15, 0.20]:
        # Re-simulate with different improvement levels
        # Quick approximation: scale the U10 effect proportionally
        # The main effect comes from med-agree + prev_wrong zone
        # Default improvement was +10pp
        scale = improvement / 0.10  # relative to default
        adjusted_tsr = pred_bl_tsr + (u10_tsr - pred_bl_tsr) * scale
        cal_adjusted = adjusted_tsr * cal_factor
        delta = (cal_adjusted - actual_bl_tsr) * 100
        print(f"  {improvement*100:>10.0f}pp | {adjusted_tsr*100:9.2f}% | {cal_adjusted*100:11.2f}% | {delta:+11.2f}pp")

    # ===================================================================
    # U11 PREDICTION: ACTION-HISTORY OBSERVER
    # ===================================================================
    print(f"\n{'='*70}")
    print("PRE-REGISTERED PREDICTION: U11 (Action-History Observer)")
    print(f"{'='*70}")

    u11_tsr, u11_matched, u11_corrections, u11_total_steps = simulate_u11_tsr(
        baseline, episode_agreements, order0, order1, initial)
    calibrated_u11 = u11_tsr * cal_factor

    print(f"\n  U11 Mechanism:")
    print(f"    Observer outputs: action disambiguation signal (not state description)")
    print(f"    Targets: steps with high action entropy AND medium agreement")
    print(f"    Correction: 'avoid [wrong action type]' → reduce action confusion")

    print(f"\n  U11 Stats:")
    print(f"    Total steps analyzed: {u11_total_steps}")
    print(f"    Steps with corrections: {u11_corrections}")
    print(f"    Correction rate: {u11_corrections/u11_total_steps*100:.2f}% of steps")

    print(f"\n  ★ PREDICTION: U11 TSR")
    print(f"    Raw Markov:     {u11_tsr:.4f} ({u11_tsr*100:.2f}%)")
    print(f"    Calibrated:     {calibrated_u11:.4f} ({calibrated_u11*100:.2f}%)")
    print(f"    Predicted delta: {(calibrated_u11 - actual_bl_tsr)*100:+.2f}pp over baseline")
    if u7_data:
        print(f"    Predicted delta: {(calibrated_u11 - actual_u7_tsr)*100:+.2f}pp over U7")
    print(f"    Matched: {u11_matched}")

    # Sensitivity: Observer identification × Actor correction rates
    print(f"\n  --- Sensitivity (Observer ID rate × Actor correction rate) ---")
    print(f"  {'ID rate':>8} | {'Correct rate':>12} | {'Calibrated TSR':>15} | {'Delta':>8}")
    print(f"  {'-'*8} | {'-'*12} | {'-'*15} | {'-'*8}")

    for id_rate in [0.3, 0.5, 0.7]:
        for corr_rate in [0.2, 0.3, 0.5]:
            # Scale relative to default (0.5 × 0.3 = 0.15)
            scale = (id_rate * corr_rate) / (0.5 * 0.3)
            adj_tsr = pred_bl_tsr + (u11_tsr - pred_bl_tsr) * scale
            cal_adj = adj_tsr * cal_factor
            delta = (cal_adj - actual_bl_tsr) * 100
            print(f"  {id_rate*100:>6.0f}% | {corr_rate*100:>10.0f}% | {cal_adj*100:14.2f}% | {delta:+7.2f}pp")

    # ===================================================================
    # COMBINED: U10 + U11
    # ===================================================================
    print(f"\n{'='*70}")
    print("COMBINED PREDICTION: U10 + U11 (if effects are additive)")
    print(f"{'='*70}")

    u10_delta = u10_tsr - pred_bl_tsr
    u11_delta = u11_tsr - pred_bl_tsr
    combined_tsr = pred_bl_tsr + u10_delta + u11_delta
    cal_combined = combined_tsr * cal_factor

    print(f"  U10 raw delta:     {u10_delta*100:+.2f}pp")
    print(f"  U11 raw delta:     {u11_delta*100:+.2f}pp")
    print(f"  Combined raw TSR:  {combined_tsr*100:.2f}%")
    print(f"  Calibrated:        {cal_combined*100:.2f}%")
    print(f"  Total delta vs BL: {(cal_combined - actual_bl_tsr)*100:+.2f}pp")
    if u7_data:
        print(f"  Total delta vs U7: {(cal_combined - actual_u7_tsr)*100:+.2f}pp")

    print(f"\n  ★ If additive effects hold, U10+U11 should achieve ~{cal_combined*100:.1f}% TSR")
    print(f"    This is a {(cal_combined - actual_bl_tsr)*100:+.1f}pp improvement over baseline")

    # ===================================================================
    # SUMMARY TABLE
    # ===================================================================
    print(f"\n{'='*70}")
    print("SUMMARY: PRE-REGISTERED PREDICTIONS")
    print(f"{'='*70}")

    print(f"\n  {'Method':<30} | {'Calibrated TSR':>15} | {'Delta vs BL':>12} | {'Status':>10}")
    print(f"  {'-'*30} | {'-'*15} | {'-'*12} | {'-'*10}")
    print(f"  {'Baseline (Eval A)':<30} | {actual_bl_tsr*100:>14.2f}% | {'—':>12} | {'Actual':>10}")
    if u7_data:
        print(f"  {'U7 (Actor-Verifier)':<30} | {actual_u7_tsr*100:>14.2f}% | {(actual_u7_tsr-actual_bl_tsr)*100:>+11.2f}pp | {'Actual':>10}")
    print(f"  {'U10 (History-Aware Verifier)':<30} | {calibrated_u10*100:>14.2f}% | {(calibrated_u10-actual_bl_tsr)*100:>+11.2f}pp | {'Predicted':>10}")
    print(f"  {'U11 (Action-History Obs.)':<30} | {calibrated_u11*100:>14.2f}% | {(calibrated_u11-actual_bl_tsr)*100:>+11.2f}pp | {'Predicted':>10}")
    print(f"  {'U10+U11 Combined':<30} | {cal_combined*100:>14.2f}% | {(cal_combined-actual_bl_tsr)*100:>+11.2f}pp | {'Predicted':>10}")

    # ===================================================================
    # Save
    # ===================================================================
    results = {
        'calibration': {
            'actual_baseline_tsr': actual_bl_tsr,
            'markov_baseline_tsr': pred_bl_tsr,
            'calibration_factor': cal_factor,
        },
        'u10_prediction': {
            'raw_tsr': u10_tsr,
            'calibrated_tsr': calibrated_u10,
            'delta_vs_baseline': calibrated_u10 - actual_bl_tsr,
            'decisions': u10_decisions,
            'matched': u10_matched,
        },
        'u11_prediction': {
            'raw_tsr': u11_tsr,
            'calibrated_tsr': calibrated_u11,
            'delta_vs_baseline': calibrated_u11 - actual_bl_tsr,
            'corrections': u11_corrections,
            'total_steps': u11_total_steps,
            'matched': u11_matched,
        },
        'combined_prediction': {
            'calibrated_tsr': cal_combined,
            'delta_vs_baseline': cal_combined - actual_bl_tsr,
        },
    }

    if u7_data:
        results['u7_calibration'] = {
            'actual_tsr': actual_u7_tsr,
            'raw_predicted': pred_u7_tsr,
            'calibrated': pred_u7_tsr * cal_factor,
        }
        results['u10_prediction']['delta_vs_u7'] = calibrated_u10 - actual_u7_tsr
        results['u11_prediction']['delta_vs_u7'] = calibrated_u11 - actual_u7_tsr
        results['combined_prediction']['delta_vs_u7'] = cal_combined - actual_u7_tsr

    out_path = os.path.join(args.output_dir, 'dir3_markov_predictions.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direction 3: Pre-Registered Markov Predictions")
    parser.add_argument("--baseline_jsonl", type=str, required=True)
    parser.add_argument("--u7_jsonl", type=str, default=None)
    parser.add_argument("--multisample_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_dir3")
    args = parser.parse_args()
    main(args)
