"""Analysis D: Verifier PASS Precision Ceiling.

Computes the theoretical upper bound of the U7 verifier framework.
If we had a perfect verifier that only PASSes truly correct greedy actions,
what's the maximum TSR improvement? This tells us the ROI ceiling of the
Verifier direction.

0 GPU needed — pure data analysis.

Data files:
  1. AR baseline: trajectory_results.jsonl (1543 episodes, per-step extract_match)
  2. C4+C7 multisample: multisample_results.jsonl (per-step K=10 samples)
  3. U7 results: actor_verifier_results.jsonl (per-step verifier PASS/FAIL)

Key analyses:
  1. Current U7 decomposition (PASS vs FAIL path contributions)
  2. Agreement threshold sweep (optimal threshold verifier)
  3. Perfect oracle verifier ceiling
  4. Step-position-weighted TSR prediction for all scenarios
"""

import argparse
import json
import os
import sys
import numpy as np
from collections import defaultdict, Counter

# Add parent directory to path for ac_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from ac_utils import save_json
except (ImportError, ModuleNotFoundError):
    def save_json(data, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_jsonl(path):
    """Load JSONL file to list of dicts."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ---------------------------------------------------------------------------
# 1. Current U7 Decomposition
# ---------------------------------------------------------------------------

def decompose_u7(u7_results):
    """Decompose U7 into PASS and FAIL paths with per-step-position detail."""
    # Global counts
    pass_total = 0
    pass_correct = 0
    fail_total = 0
    fail_correct = 0

    # Per step-position
    by_step = defaultdict(lambda: {
        'pass_total': 0, 'pass_correct': 0,
        'fail_total': 0, 'fail_correct': 0,
    })

    for traj in u7_results:
        for step in traj.get('step_results', []):
            sn = step['step_num']
            correct = step.get('extract_match', False)
            verdict = step.get('verified', 'PASS')

            if verdict == 'PASS':
                pass_total += 1
                by_step[sn]['pass_total'] += 1
                if correct:
                    pass_correct += 1
                    by_step[sn]['pass_correct'] += 1
            else:
                fail_total += 1
                by_step[sn]['fail_total'] += 1
                if correct:
                    fail_correct += 1
                    by_step[sn]['fail_correct'] += 1

    total = pass_total + fail_total
    pass_rate = pass_total / total if total > 0 else 0
    pass_acc = pass_correct / pass_total if pass_total > 0 else 0
    fail_rate = fail_total / total if total > 0 else 0
    fail_acc = fail_correct / fail_total if fail_total > 0 else 0
    overall_acc = (pass_correct + fail_correct) / total if total > 0 else 0

    return {
        'total_steps': total,
        'pass_rate': pass_rate,
        'pass_accuracy': pass_acc,
        'pass_correct': pass_correct,
        'pass_total': pass_total,
        'fail_rate': fail_rate,
        'fail_accuracy': fail_acc,
        'fail_correct': fail_correct,
        'fail_total': fail_total,
        'overall_accuracy': overall_acc,
        'by_step': dict(by_step),
    }


def compute_u7_tsr(u7_results):
    """Compute actual U7 TSR from trajectory results."""
    n = len(u7_results)
    if n == 0:
        return 0.0
    success = sum(1 for r in u7_results if r.get('task_success', False))
    return success / n


# ---------------------------------------------------------------------------
# 2. Baseline decomposition
# ---------------------------------------------------------------------------

def decompose_baseline(ar_results):
    """Extract per-step accuracy from AR baseline results."""
    by_step = defaultdict(lambda: {'total': 0, 'correct': 0})
    total_steps = 0
    total_correct = 0

    # Also track per-trajectory for TSR computation
    n_traj = len(ar_results)
    n_success = sum(1 for r in ar_results if r.get('task_success', False))
    tsr = n_success / n_traj if n_traj > 0 else 0

    # Track trajectory length distribution for TSR prediction
    length_dist = defaultdict(int)

    for traj in ar_results:
        num_steps = traj.get('num_steps', 0)
        length_dist[num_steps] += 1
        for step in traj.get('step_results', []):
            sn = step['step_num']
            by_step[sn]['total'] += 1
            total_steps += 1
            if step.get('extract_match', False):
                by_step[sn]['correct'] += 1
                total_correct += 1

    overall_acc = total_correct / total_steps if total_steps > 0 else 0

    return {
        'total_trajectories': n_traj,
        'total_steps': total_steps,
        'total_correct': total_correct,
        'overall_accuracy': overall_acc,
        'tsr': tsr,
        'n_success': n_success,
        'by_step': dict(by_step),
        'length_dist': dict(length_dist),
    }


# ---------------------------------------------------------------------------
# 3. C4+C7 Agreement Analysis
# ---------------------------------------------------------------------------

def analyze_multisample(multisample_results):
    """Analyze C4+C7 data for agreement vs correctness calibration.

    For each step:
      - agreement = fraction of K samples matching the most-common action type
      - greedy_correct = whether sample[0] (greedy) is correct
      - oracle_correct = whether any of K samples is correct
      - majority_vote_correct = whether most-common action type sample is correct

    Returns per-step data and aggregated statistics.
    """
    step_data = []

    for ep in multisample_results:
        eid = ep.get('episode_id')
        num_steps = ep.get('num_steps', 0)

        for step in ep.get('step_samples', []):
            sn = step.get('step_num', 0)
            samples = step.get('samples', [])
            K = len(samples)
            if K == 0:
                continue

            # Agreement: fraction matching most common action type
            action_types = []
            for s in samples:
                pa = s.get('pred_action')
                if pa and isinstance(pa, dict):
                    action_types.append(pa.get('action', 'unknown'))
                else:
                    action_types.append('parse_fail')

            if action_types:
                type_counter = Counter(action_types)
                most_common_type, most_common_count = type_counter.most_common(1)[0]
                agreement = most_common_count / K
            else:
                agreement = 0.0
                most_common_type = 'unknown'

            # Greedy correct (sample 0)
            greedy_correct = samples[0].get('extract_match', False)

            # Oracle correct (any sample correct)
            oracle_correct = any(s.get('extract_match', False) for s in samples)

            # Majority vote correct: pick first sample with voted action type, check it
            mv_correct = False
            for s in samples:
                pa = s.get('pred_action')
                if pa and isinstance(pa, dict) and pa.get('action') == most_common_type:
                    mv_correct = s.get('extract_match', False)
                    break

            # Count how many of K samples are correct
            n_correct = sum(1 for s in samples if s.get('extract_match', False))

            step_data.append({
                'episode_id': eid,
                'step_num': sn,
                'num_steps': num_steps,
                'agreement': agreement,
                'greedy_correct': bool(greedy_correct),
                'oracle_correct': bool(oracle_correct),
                'mv_correct': bool(mv_correct),
                'n_correct_of_K': n_correct,
                'K': K,
            })

    return step_data


def compute_agreement_calibration(step_data, n_bins=20):
    """Compute P(greedy correct | agreement) calibration curve.

    Bins agreement into n_bins and computes conditional probabilities.
    """
    bins = np.linspace(0, 1.0 + 1e-9, n_bins + 1)
    calibration = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = [s for s in step_data if lo <= s['agreement'] < hi]
        if not in_bin:
            continue

        n = len(in_bin)
        greedy_acc = sum(1 for s in in_bin if s['greedy_correct']) / n
        oracle_acc = sum(1 for s in in_bin if s['oracle_correct']) / n
        mv_acc = sum(1 for s in in_bin if s['mv_correct']) / n
        avg_n_correct = np.mean([s['n_correct_of_K'] for s in in_bin])

        calibration.append({
            'bin_lo': float(lo),
            'bin_hi': float(hi),
            'bin_mid': float((lo + hi) / 2),
            'n': n,
            'greedy_accuracy': greedy_acc,
            'oracle_accuracy': oracle_acc,
            'mv_accuracy': mv_acc,
            'avg_n_correct_of_K': float(avg_n_correct),
        })

    return calibration


# ---------------------------------------------------------------------------
# 4. Agreement Threshold Sweep
# ---------------------------------------------------------------------------

def threshold_sweep(step_data, baseline_greedy_acc, thresholds=None):
    """Sweep agreement thresholds to find optimal verifier configuration.

    For each threshold t:
      - PASS if agreement >= t (keep greedy action)
      - FAIL if agreement < t (resample)

    Compute:
      - PASS rate, PASS accuracy
      - FAIL rate, need to estimate FAIL fallback accuracy

    Fallback accuracies on FAIL:
      - resample_mv: majority vote from K samples (use mv_correct)
      - resample_random: random sample accuracy (n_correct_of_K / K)
      - oracle: best of K (oracle_correct)
    """
    if thresholds is None:
        thresholds = np.arange(0.0, 1.05, 0.05).tolist()

    results = []
    for t in thresholds:
        pass_steps = [s for s in step_data if s['agreement'] >= t]
        fail_steps = [s for s in step_data if s['agreement'] < t]

        n_total = len(step_data)
        n_pass = len(pass_steps)
        n_fail = len(fail_steps)

        pass_rate = n_pass / n_total if n_total > 0 else 0
        fail_rate = n_fail / n_total if n_total > 0 else 0

        # PASS path: keep greedy
        pass_acc = (sum(1 for s in pass_steps if s['greedy_correct']) / n_pass
                    if n_pass > 0 else 0)

        # FAIL path options:
        # (a) majority vote from K samples
        fail_mv_acc = (sum(1 for s in fail_steps if s['mv_correct']) / n_fail
                       if n_fail > 0 else 0)

        # (b) random sample accuracy (average of n_correct_of_K / K)
        fail_random_acc = (np.mean([s['n_correct_of_K'] / s['K'] for s in fail_steps])
                           if n_fail > 0 else 0)

        # (c) oracle (any correct in K)
        fail_oracle_acc = (sum(1 for s in fail_steps if s['oracle_correct']) / n_fail
                           if n_fail > 0 else 0)

        # Greedy accuracy on FAIL steps (what would have happened without verifier)
        fail_greedy_acc = (sum(1 for s in fail_steps if s['greedy_correct']) / n_fail
                           if n_fail > 0 else 0)

        # Combined step accuracy for each fallback strategy
        # Strategy: resample_mv
        combined_mv = pass_rate * pass_acc + fail_rate * fail_mv_acc
        # Strategy: oracle
        combined_oracle = pass_rate * pass_acc + fail_rate * fail_oracle_acc
        # Strategy: no fallback (just greedy everywhere)
        combined_greedy = pass_rate * pass_acc + fail_rate * fail_greedy_acc

        # Net gain over baseline greedy
        gain_mv = combined_mv - baseline_greedy_acc
        gain_oracle = combined_oracle - baseline_greedy_acc

        results.append({
            'threshold': float(t),
            'pass_rate': pass_rate,
            'fail_rate': fail_rate,
            'pass_accuracy': pass_acc,
            'fail_mv_accuracy': fail_mv_acc,
            'fail_random_accuracy': float(fail_random_acc),
            'fail_oracle_accuracy': fail_oracle_acc,
            'fail_greedy_accuracy': fail_greedy_acc,
            'combined_accuracy_mv': combined_mv,
            'combined_accuracy_oracle': combined_oracle,
            'combined_accuracy_greedy_only': combined_greedy,
            'gain_over_baseline_mv': gain_mv,
            'gain_over_baseline_oracle': gain_oracle,
            'n_pass': n_pass,
            'n_fail': n_fail,
        })

    return results


# ---------------------------------------------------------------------------
# 5. TSR Prediction from Step Accuracy
# ---------------------------------------------------------------------------

def predict_tsr_from_step_accuracy(step_acc, length_dist):
    """Predict TSR using the Markov assumption: TSR = sum_L P(L) * p^L.

    Args:
        step_acc: uniform step accuracy (assumes same accuracy at each step)
        length_dist: dict {length: count}

    Returns:
        Predicted TSR.
    """
    total = sum(length_dist.values())
    if total == 0:
        return 0.0

    tsr = 0.0
    for length, count in length_dist.items():
        p_length = count / total
        tsr += p_length * (step_acc ** length)

    return tsr


def predict_tsr_position_weighted(per_step_acc, length_dist, max_step=30):
    """Predict TSR with per-step-position accuracy.

    TSR = sum_L P(L) * product_{k=0}^{L-1} p_k

    Args:
        per_step_acc: dict {step_num: accuracy}
        length_dist: dict {length: count}
        max_step: maximum step to consider

    Returns:
        Predicted TSR.
    """
    total = sum(length_dist.values())
    if total == 0:
        return 0.0

    # Fallback accuracy for steps beyond what we have data for
    known_accs = [v for k, v in per_step_acc.items() if v is not None]
    fallback_acc = np.mean(known_accs) if known_accs else 0.5

    tsr = 0.0
    for length, count in length_dist.items():
        p_length = count / total
        product = 1.0
        for k in range(length):
            acc_k = per_step_acc.get(k, fallback_acc)
            product *= acc_k
        tsr += p_length * product

    return tsr


# ---------------------------------------------------------------------------
# 6. Perfect Oracle Verifier Scenarios
# ---------------------------------------------------------------------------

def compute_perfect_verifier_scenarios(step_data, baseline_info, u7_decomp):
    """Compute TSR for various verifier ceiling scenarios.

    Scenario B: Perfect verifier + resample (current U7 resample accuracy)
      - PASS only if greedy is correct -> PASS rate = baseline greedy acc
      - FAIL -> resample with U7's observed resample accuracy

    Scenario C: Perfect verifier + majority vote
      - PASS only if greedy is correct
      - FAIL -> use majority vote from K samples

    Scenario D: Perfect verifier + oracle resample
      - PASS only if greedy is correct
      - FAIL -> best of K samples (upper bound of resampling)
    """
    length_dist = baseline_info['length_dist']
    baseline_greedy_acc = baseline_info['overall_accuracy']
    u7_fail_acc = u7_decomp['fail_accuracy']

    # Per-step-position computations from C4+C7
    by_step_c4c7 = defaultdict(lambda: {
        'total': 0, 'greedy_correct': 0,
        'mv_correct_given_greedy_wrong': 0, 'greedy_wrong': 0,
        'oracle_correct_given_greedy_wrong': 0,
    })

    for s in step_data:
        sn = s['step_num']
        by_step_c4c7[sn]['total'] += 1
        if s['greedy_correct']:
            by_step_c4c7[sn]['greedy_correct'] += 1
        else:
            by_step_c4c7[sn]['greedy_wrong'] += 1
            if s['mv_correct']:
                by_step_c4c7[sn]['mv_correct_given_greedy_wrong'] += 1
            if s['oracle_correct']:
                by_step_c4c7[sn]['oracle_correct_given_greedy_wrong'] += 1

    scenarios = {}

    # --- Baseline ---
    per_step_baseline = {}
    for sn, d in baseline_info['by_step'].items():
        if d['total'] > 0:
            per_step_baseline[sn] = d['correct'] / d['total']

    tsr_baseline = predict_tsr_position_weighted(per_step_baseline, length_dist)
    scenarios['baseline_greedy'] = {
        'description': 'Baseline greedy (Eval A)',
        'step_accuracy': baseline_greedy_acc,
        'predicted_tsr': tsr_baseline,
        'actual_tsr': baseline_info['tsr'],
    }

    # --- Current U7 ---
    # Combined accuracy: pass_rate * pass_acc + fail_rate * fail_acc
    u7_combined = u7_decomp['pass_rate'] * u7_decomp['pass_accuracy'] + \
                  u7_decomp['fail_rate'] * u7_decomp['fail_accuracy']
    tsr_u7_uniform = predict_tsr_from_step_accuracy(u7_combined, length_dist)
    scenarios['current_u7'] = {
        'description': 'Current U7 (PASS=37.1%, PASS_acc=76.2%)',
        'pass_rate': u7_decomp['pass_rate'],
        'pass_accuracy': u7_decomp['pass_accuracy'],
        'fail_accuracy': u7_decomp['fail_accuracy'],
        'step_accuracy': u7_combined,
        'predicted_tsr': tsr_u7_uniform,
    }

    # --- Scenario B: Perfect verifier + U7 resample accuracy ---
    # Perfect verifier: PASS iff greedy is correct
    # PASS rate = greedy accuracy per step
    # PASS accuracy = 1.0 (only correct greedy passes)
    # FAIL rate = 1 - greedy accuracy
    # FAIL accuracy = U7's observed resample accuracy
    per_step_B = {}
    for sn, d in by_step_c4c7.items():
        if d['total'] > 0:
            greedy_acc = d['greedy_correct'] / d['total']
            # perfect verifier: passes all correct, fails all wrong
            # step acc = greedy_acc * 1.0 + (1 - greedy_acc) * u7_fail_acc
            per_step_B[sn] = greedy_acc * 1.0 + (1 - greedy_acc) * u7_fail_acc

    combined_B = np.mean(list(per_step_B.values())) if per_step_B else 0
    tsr_B = predict_tsr_position_weighted(per_step_B, length_dist)
    scenarios['perfect_verifier_u7_resample'] = {
        'description': 'Perfect verifier + U7 resample (FAIL acc={:.3f})'.format(u7_fail_acc),
        'pass_rate': baseline_greedy_acc,
        'pass_accuracy': 1.0,
        'fail_accuracy': u7_fail_acc,
        'step_accuracy': combined_B,
        'predicted_tsr': tsr_B,
    }

    # --- Scenario C: Perfect verifier + majority vote ---
    per_step_C = {}
    for sn, d in by_step_c4c7.items():
        if d['total'] > 0:
            greedy_acc = d['greedy_correct'] / d['total']
            # On FAIL (greedy wrong), use MV: P(MV correct | greedy wrong)
            mv_given_wrong = (d['mv_correct_given_greedy_wrong'] / d['greedy_wrong']
                              if d['greedy_wrong'] > 0 else 0)
            per_step_C[sn] = greedy_acc * 1.0 + (1 - greedy_acc) * mv_given_wrong

    combined_C = np.mean(list(per_step_C.values())) if per_step_C else 0
    # Overall MV accuracy given greedy wrong
    total_greedy_wrong = sum(d['greedy_wrong'] for d in by_step_c4c7.values())
    total_mv_correct_given_wrong = sum(d['mv_correct_given_greedy_wrong']
                                       for d in by_step_c4c7.values())
    mv_acc_given_wrong = (total_mv_correct_given_wrong / total_greedy_wrong
                          if total_greedy_wrong > 0 else 0)

    tsr_C = predict_tsr_position_weighted(per_step_C, length_dist)
    scenarios['perfect_verifier_mv'] = {
        'description': 'Perfect verifier + majority vote (FAIL->MV acc={:.3f})'.format(mv_acc_given_wrong),
        'pass_rate': baseline_greedy_acc,
        'pass_accuracy': 1.0,
        'fail_accuracy_mv_given_wrong': mv_acc_given_wrong,
        'step_accuracy': combined_C,
        'predicted_tsr': tsr_C,
    }

    # --- Scenario D: Perfect verifier + oracle resample ---
    per_step_D = {}
    for sn, d in by_step_c4c7.items():
        if d['total'] > 0:
            greedy_acc = d['greedy_correct'] / d['total']
            oracle_given_wrong = (d['oracle_correct_given_greedy_wrong'] / d['greedy_wrong']
                                  if d['greedy_wrong'] > 0 else 0)
            per_step_D[sn] = greedy_acc * 1.0 + (1 - greedy_acc) * oracle_given_wrong

    combined_D = np.mean(list(per_step_D.values())) if per_step_D else 0
    total_oracle_correct_given_wrong = sum(d['oracle_correct_given_greedy_wrong']
                                           for d in by_step_c4c7.values())
    oracle_acc_given_wrong = (total_oracle_correct_given_wrong / total_greedy_wrong
                              if total_greedy_wrong > 0 else 0)

    tsr_D = predict_tsr_position_weighted(per_step_D, length_dist)
    scenarios['perfect_verifier_oracle'] = {
        'description': 'Perfect verifier + oracle best-of-K (FAIL->oracle acc={:.3f})'.format(oracle_acc_given_wrong),
        'pass_rate': baseline_greedy_acc,
        'pass_accuracy': 1.0,
        'fail_accuracy_oracle_given_wrong': oracle_acc_given_wrong,
        'step_accuracy': combined_D,
        'predicted_tsr': tsr_D,
    }

    # --- Scenario E: Absolute ceiling (perfect verifier + perfect resample) ---
    # Every step is correct
    tsr_E = predict_tsr_from_step_accuracy(1.0, length_dist)
    scenarios['perfect_everything'] = {
        'description': 'Perfect verifier + perfect resample (100% step acc)',
        'step_accuracy': 1.0,
        'predicted_tsr': tsr_E,
    }

    return scenarios


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # ===================================================================
    # Load data
    # ===================================================================
    print("Loading AR baseline results...")
    ar_results = load_jsonl(args.ar_results)
    print(f"  Loaded {len(ar_results)} AR trajectories")

    print("Loading C4+C7 multisample results...")
    ms_results = load_jsonl(args.multisample_results)
    print(f"  Loaded {len(ms_results)} multisample episodes")

    print("Loading U7 actor-verifier results...")
    u7_results = load_jsonl(args.u7_results)
    print(f"  Loaded {len(u7_results)} U7 trajectories")

    # ===================================================================
    # 1. Current U7 Decomposition
    # ===================================================================
    print("\n" + "=" * 80)
    print("1. CURRENT U7 DECOMPOSITION")
    print("=" * 80)

    u7_decomp = decompose_u7(u7_results)
    u7_actual_tsr = compute_u7_tsr(u7_results)

    print(f"\n  Total steps evaluated: {u7_decomp['total_steps']}")
    print(f"\n  PASS path:")
    print(f"    Rate:     {u7_decomp['pass_rate']*100:.1f}% ({u7_decomp['pass_total']} steps)")
    print(f"    Accuracy: {u7_decomp['pass_accuracy']*100:.1f}% ({u7_decomp['pass_correct']}/{u7_decomp['pass_total']})")
    print(f"\n  FAIL->resample path:")
    print(f"    Rate:     {u7_decomp['fail_rate']*100:.1f}% ({u7_decomp['fail_total']} steps)")
    print(f"    Accuracy: {u7_decomp['fail_accuracy']*100:.1f}% ({u7_decomp['fail_correct']}/{u7_decomp['fail_total']})")
    print(f"\n  Combined accuracy: {u7_decomp['overall_accuracy']*100:.1f}%")
    print(f"  Actual U7 TSR: {u7_actual_tsr*100:.2f}%")

    # ===================================================================
    # 2. Baseline Analysis
    # ===================================================================
    print("\n" + "=" * 80)
    print("2. BASELINE ANALYSIS")
    print("=" * 80)

    baseline_info = decompose_baseline(ar_results)

    print(f"\n  Trajectories: {baseline_info['total_trajectories']}")
    print(f"  Total steps: {baseline_info['total_steps']}")
    print(f"  Step accuracy (conditional): {baseline_info['overall_accuracy']*100:.1f}%")
    print(f"  Actual TSR: {baseline_info['tsr']*100:.2f}% ({baseline_info['n_success']}/{baseline_info['total_trajectories']})")

    # Compare FAIL->resample accuracy to baseline greedy
    print(f"\n  FAIL->resample acc vs baseline greedy:")
    print(f"    FAIL resample accuracy: {u7_decomp['fail_accuracy']*100:.1f}%")
    print(f"    Baseline greedy accuracy: {baseline_info['overall_accuracy']*100:.1f}%")
    delta = u7_decomp['fail_accuracy'] - baseline_info['overall_accuracy']
    if delta < 0:
        print(f"    Delta: {delta*100:+.1f}pp -- FAIL resample is WORSE than greedy (net loss on FAIL path)")
    else:
        print(f"    Delta: {delta*100:+.1f}pp -- FAIL resample is better than greedy")

    # ===================================================================
    # 3. C4+C7 Agreement Analysis
    # ===================================================================
    print("\n" + "=" * 80)
    print("3. AGREEMENT vs CORRECTNESS CALIBRATION (C4+C7)")
    print("=" * 80)

    step_data = analyze_multisample(ms_results)
    print(f"\n  Total step-level data points: {len(step_data)}")

    overall_greedy_acc = (sum(1 for s in step_data if s['greedy_correct'])
                          / len(step_data) if step_data else 0)
    overall_oracle_acc = (sum(1 for s in step_data if s['oracle_correct'])
                          / len(step_data) if step_data else 0)
    overall_mv_acc = (sum(1 for s in step_data if s['mv_correct'])
                      / len(step_data) if step_data else 0)

    print(f"  Greedy accuracy (C4+C7, all steps): {overall_greedy_acc*100:.1f}%")
    print(f"  Majority vote accuracy: {overall_mv_acc*100:.1f}%")
    print(f"  Oracle best-of-K accuracy: {overall_oracle_acc*100:.1f}%")

    calibration = compute_agreement_calibration(step_data)
    print(f"\n  Agreement Calibration (P(greedy correct | agreement)):")
    print(f"  {'Agreement':>12} | {'N':>6} | {'Greedy Acc':>10} | {'MV Acc':>8} | {'Oracle':>8}")
    print(f"  {'-'*12} | {'-'*6} | {'-'*10} | {'-'*8} | {'-'*8}")
    for c in calibration:
        print(f"  {c['bin_lo']:.2f}-{c['bin_hi']:.2f} | {c['n']:6d} | "
              f"{c['greedy_accuracy']*100:9.1f}% | {c['mv_accuracy']*100:7.1f}% | "
              f"{c['oracle_accuracy']*100:7.1f}%")

    # ===================================================================
    # 4. Agreement Threshold Sweep
    # ===================================================================
    print("\n" + "=" * 80)
    print("4. AGREEMENT THRESHOLD SWEEP")
    print("=" * 80)

    sweep = threshold_sweep(step_data, baseline_info['overall_accuracy'])

    print(f"\n  {'Thresh':>6} | {'PASS%':>6} | {'PASS Acc':>8} | {'FAIL MV':>8} | "
          f"{'Comb MV':>8} | {'Gain MV':>8} | {'Comb Orc':>8} | {'Gain Orc':>8}")
    print(f"  {'-'*6} | {'-'*6} | {'-'*8} | {'-'*8} | "
          f"{'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")

    best_gain_mv = -999
    best_threshold_mv = 0
    best_gain_oracle = -999
    best_threshold_oracle = 0

    for r in sweep:
        t = r['threshold']
        print(f"  {t:6.2f} | {r['pass_rate']*100:5.1f}% | {r['pass_accuracy']*100:7.1f}% | "
              f"{r['fail_mv_accuracy']*100:7.1f}% | {r['combined_accuracy_mv']*100:7.1f}% | "
              f"{r['gain_over_baseline_mv']*100:+7.2f}% | {r['combined_accuracy_oracle']*100:7.1f}% | "
              f"{r['gain_over_baseline_oracle']*100:+7.2f}%")

        if r['gain_over_baseline_mv'] > best_gain_mv:
            best_gain_mv = r['gain_over_baseline_mv']
            best_threshold_mv = t
        if r['gain_over_baseline_oracle'] > best_gain_oracle:
            best_gain_oracle = r['gain_over_baseline_oracle']
            best_threshold_oracle = t

    print(f"\n  Best threshold (MV fallback): {best_threshold_mv:.2f} "
          f"-> step acc gain: {best_gain_mv*100:+.2f}pp")
    print(f"  Best threshold (Oracle fallback): {best_threshold_oracle:.2f} "
          f"-> step acc gain: {best_gain_oracle*100:+.2f}pp")

    # Get best sweep entries for TSR prediction
    best_mv_entry = [r for r in sweep if r['threshold'] == best_threshold_mv][0]
    best_oracle_entry = [r for r in sweep if r['threshold'] == best_threshold_oracle][0]

    # ===================================================================
    # 5. TSR Predictions for All Scenarios
    # ===================================================================
    print("\n" + "=" * 80)
    print("5. TSR PREDICTIONS (Step-Position-Weighted)")
    print("=" * 80)

    scenarios = compute_perfect_verifier_scenarios(step_data, baseline_info, u7_decomp)

    # Add optimal threshold verifier scenarios
    opt_mv_step_acc = best_mv_entry['combined_accuracy_mv']
    opt_mv_tsr = predict_tsr_from_step_accuracy(opt_mv_step_acc, baseline_info['length_dist'])
    scenarios['optimal_threshold_mv'] = {
        'description': 'Optimal threshold verifier (t={:.2f}) + MV fallback'.format(best_threshold_mv),
        'threshold': best_threshold_mv,
        'pass_rate': best_mv_entry['pass_rate'],
        'pass_accuracy': best_mv_entry['pass_accuracy'],
        'fail_mv_accuracy': best_mv_entry['fail_mv_accuracy'],
        'step_accuracy': opt_mv_step_acc,
        'predicted_tsr': opt_mv_tsr,
    }

    opt_oracle_step_acc = best_oracle_entry['combined_accuracy_oracle']
    opt_oracle_tsr = predict_tsr_from_step_accuracy(opt_oracle_step_acc,
                                                     baseline_info['length_dist'])
    scenarios['optimal_threshold_oracle'] = {
        'description': 'Optimal threshold verifier (t={:.2f}) + oracle fallback'.format(best_threshold_oracle),
        'threshold': best_threshold_oracle,
        'pass_rate': best_oracle_entry['pass_rate'],
        'pass_accuracy': best_oracle_entry['pass_accuracy'],
        'fail_oracle_accuracy': best_oracle_entry['fail_oracle_accuracy'],
        'step_accuracy': opt_oracle_step_acc,
        'predicted_tsr': opt_oracle_tsr,
    }

    # Display order
    display_order = [
        'baseline_greedy',
        'current_u7',
        'optimal_threshold_mv',
        'perfect_verifier_u7_resample',
        'perfect_verifier_mv',
        'optimal_threshold_oracle',
        'perfect_verifier_oracle',
        'perfect_everything',
    ]

    baseline_tsr = scenarios['baseline_greedy']['predicted_tsr']
    actual_baseline_tsr = baseline_info['tsr']

    print(f"\n  {'Scenario':<55} | {'Step Acc':>8} | {'Pred TSR':>9} | {'Delta TSR':>10}")
    print(f"  {'-'*55} | {'-'*8} | {'-'*9} | {'-'*10}")

    for key in display_order:
        if key not in scenarios:
            continue
        sc = scenarios[key]
        step_acc = sc.get('step_accuracy', 0)
        pred_tsr = sc.get('predicted_tsr', 0)
        delta = pred_tsr - baseline_tsr
        desc = sc['description']
        if len(desc) > 55:
            desc = desc[:52] + '...'
        print(f"  {desc:<55} | {step_acc*100:7.1f}% | {pred_tsr*100:8.2f}% | {delta*100:+9.2f}pp")

    # ===================================================================
    # 6. Summary and ROI Assessment
    # ===================================================================
    print("\n" + "=" * 80)
    print("6. VERIFIER DIRECTION ROI SUMMARY")
    print("=" * 80)

    current_u7_delta_actual = u7_actual_tsr - actual_baseline_tsr
    perfect_mv_delta = scenarios['perfect_verifier_mv']['predicted_tsr'] - baseline_tsr
    perfect_oracle_delta = scenarios['perfect_verifier_oracle']['predicted_tsr'] - baseline_tsr
    optimal_mv_delta = scenarios['optimal_threshold_mv']['predicted_tsr'] - baseline_tsr
    perfect_resample_delta = scenarios['perfect_verifier_u7_resample']['predicted_tsr'] - baseline_tsr

    print(f"\n  Actual baseline TSR:      {actual_baseline_tsr*100:.2f}%")
    print(f"  Predicted baseline TSR:   {baseline_tsr*100:.2f}%")
    print(f"  Actual U7 TSR:            {u7_actual_tsr*100:.2f}%")
    print(f"\n  --- Current vs Ceiling ---")
    print(f"  Current U7 improvement:                {current_u7_delta_actual*100:+.2f}pp (actual)")
    print(f"  Optimal threshold verifier + MV:       {optimal_mv_delta*100:+.2f}pp (predicted)")
    print(f"  Perfect verifier + U7 resample:        {perfect_resample_delta*100:+.2f}pp (predicted)")
    print(f"  Perfect verifier + majority vote:      {perfect_mv_delta*100:+.2f}pp (predicted)")
    print(f"  Perfect verifier + oracle best-of-K:   {perfect_oracle_delta*100:+.2f}pp (predicted)")

    # Utilization
    if perfect_oracle_delta > 0:
        utilization = current_u7_delta_actual / perfect_oracle_delta * 100
    else:
        utilization = 0
    print(f"\n  Current U7 utilization of ceiling:  {utilization:.1f}%")
    print(f"  (current gain / perfect verifier+oracle ceiling)")

    # Key insight: decompose where the loss comes from
    print(f"\n  --- Loss Decomposition ---")
    print(f"  U7 PASS accuracy: {u7_decomp['pass_accuracy']*100:.1f}% (perfect=100%)")
    print(f"    -> PASS precision loss: {(1-u7_decomp['pass_accuracy'])*100:.1f}pp")
    print(f"       (FALSEly passing {(1-u7_decomp['pass_accuracy'])*u7_decomp['pass_total']:.0f} wrong actions)")
    print(f"  U7 PASS rate: {u7_decomp['pass_rate']*100:.1f}% vs perfect PASS rate: {baseline_info['overall_accuracy']*100:.1f}%")
    false_fail_rate = 0
    if u7_decomp['fail_total'] > 0:
        # False fails: steps where verifier said FAIL but greedy was actually correct
        # We can compute this from U7 data: PASS_correct / PASS_total vs what should be
        # Actually: among FAIL steps, how many had greedy correct?
        # In U7 AR, after FAIL we resample, so extract_match tells us about the resample.
        # We can't directly see if the original greedy was correct from U7 data alone.
        # But we can infer: PASS accuracy reflects true positive rate of verifier
        pass
    print(f"  U7 FAIL resample accuracy: {u7_decomp['fail_accuracy']*100:.1f}% "
          f"(MV fallback would give: ~{overall_mv_acc*100:.1f}%)")

    print(f"\n  --- Bottom Line ---")
    if perfect_oracle_delta > 0.01:
        print(f"  The Verifier direction has a ceiling of {perfect_oracle_delta*100:+.2f}pp TSR")
        print(f"  with perfect verification + oracle resampling.")
        print(f"  The more realistic MV ceiling is {perfect_mv_delta*100:+.2f}pp TSR.")
        if perfect_mv_delta > 0.02:
            print(f"  -> Meaningful room for improvement. Focus on PASS precision.")
        elif perfect_mv_delta > 0.005:
            print(f"  -> Modest room for improvement. Consider cost/benefit.")
        else:
            print(f"  -> Limited ceiling. Verifier ROI may not justify complexity.")
    else:
        print(f"  The Verifier ceiling is very low ({perfect_oracle_delta*100:+.2f}pp).")
        print(f"  -> Limited ROI for the Verifier direction.")

    # ===================================================================
    # Save results
    # ===================================================================
    output = {
        'u7_decomposition': {
            'total_steps': u7_decomp['total_steps'],
            'pass_rate': u7_decomp['pass_rate'],
            'pass_accuracy': u7_decomp['pass_accuracy'],
            'fail_rate': u7_decomp['fail_rate'],
            'fail_accuracy': u7_decomp['fail_accuracy'],
            'overall_accuracy': u7_decomp['overall_accuracy'],
            'actual_tsr': u7_actual_tsr,
        },
        'baseline': {
            'total_trajectories': baseline_info['total_trajectories'],
            'step_accuracy': baseline_info['overall_accuracy'],
            'actual_tsr': baseline_info['tsr'],
            'predicted_tsr': baseline_tsr,
        },
        'c4c7_summary': {
            'total_steps': len(step_data),
            'greedy_accuracy': overall_greedy_acc,
            'mv_accuracy': overall_mv_acc,
            'oracle_accuracy': overall_oracle_acc,
        },
        'agreement_calibration': calibration,
        'threshold_sweep': sweep,
        'best_threshold_mv': {
            'threshold': best_threshold_mv,
            'step_accuracy_gain': best_gain_mv,
        },
        'best_threshold_oracle': {
            'threshold': best_threshold_oracle,
            'step_accuracy_gain': best_gain_oracle,
        },
        'scenarios': {
            k: {kk: vv for kk, vv in v.items()}
            for k, v in scenarios.items()
        },
        'roi_summary': {
            'current_u7_tsr_delta': current_u7_delta_actual,
            'optimal_threshold_mv_tsr_delta': optimal_mv_delta,
            'perfect_verifier_u7_resample_tsr_delta': perfect_resample_delta,
            'perfect_verifier_mv_tsr_delta': perfect_mv_delta,
            'perfect_verifier_oracle_tsr_delta': perfect_oracle_delta,
            'utilization_pct': utilization,
        },
    }

    out_path = os.path.join(args.output_dir, 'analysis_d_pass_precision_ceiling.json')
    save_json(output, out_path)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    PROJECT_ROOT = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
    parser = argparse.ArgumentParser(
        description="Analysis D: Verifier PASS Precision Ceiling"
    )
    parser.add_argument(
        "--ar_results", type=str,
        default=f"{PROJECT_ROOT}/outputs/eval_a_ac/Qwen2.5-VL-7B/trajectory_results.jsonl",
        help="Path to AR baseline trajectory_results.jsonl",
    )
    parser.add_argument(
        "--multisample_results", type=str,
        default=f"{PROJECT_ROOT}/outputs/eval_c4c7_ac/Qwen2.5-VL-7B/multisample_results.jsonl",
        help="Path to C4+C7 multisample_results.jsonl",
    )
    parser.add_argument(
        "--u7_results", type=str,
        default=f"{PROJECT_ROOT}/outputs/eval_u7_ac/Qwen2.5-VL-7B/actor_verifier_results.jsonl",
        help="Path to U7 actor_verifier_results.jsonl",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=f"{PROJECT_ROOT}/outputs/eval_analysis_d",
        help="Output directory",
    )
    args = parser.parse_args()
    main(args)
