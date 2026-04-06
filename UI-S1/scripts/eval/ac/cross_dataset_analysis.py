"""Cross-dataset commonality analysis: AndroidControl vs GUI-360.

Analyzes universal problems shared across both datasets:
1. Compounding decomposition: intrinsic step difficulty vs recovery rate
2. Majority vote simulation (using C4+C7 multi-sample data)
3. Confidence calibration: agreement as uncertainty proxy
4. Action error pattern universality
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict, Counter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def analysis_1_compounding_decomposition(trajectory_results):
    """Decompose compounding: intrinsic step difficulty vs error propagation.
    
    For each step position k, compute:
    - P(correct at k | all steps 0..k-1 correct) = intrinsic difficulty
    - P(correct at k | at least one prior error) = recovery rate (always 0 in AR, since we stop on error)
    
    In AR evaluation, we stop on first error, so recovery rate is N/A.
    Instead, compute: P(reaching step k) and P(correct at k | reached k).
    """
    print("=" * 60)
    print("Analysis 1: Compounding Decomposition")
    print("=" * 60)
    
    max_step = 20
    reached = defaultdict(int)  # number of episodes that reached step k
    correct_at = defaultdict(int)  # number correct at step k (given reached)
    total_at = defaultdict(int)  # total episodes with at least k+1 steps
    
    for r in trajectory_results:
        n = r['num_steps']
        correct = r['final_step_id']  # number of consecutive correct steps
        for k in range(min(n, max_step)):
            total_at[k] += 1
            if k < correct:  # reached and was correct
                reached[k+1] += 1  # can reach k+1
                correct_at[k] += 1
            elif k == correct and k < n:  # reached but failed
                correct_at[k] += 0  # failed at this step
                # didn't reach k+1
            # if k > correct: didn't reach this step (AR stops on error)
    
    # Reached step 0 = all episodes
    reached[0] = len(trajectory_results)
    
    print(f"\n{'Step':>5} {'Reached':>10} {'Correct|Reached':>16} {'P(correct|reached)':>20} {'P(reached)':>12} {'Cumulative P':>14}")
    print("-" * 80)
    
    cum_p = 1.0
    step_data = []
    for k in range(max_step):
        if reached[k] == 0:
            break
        n_reached = reached[k]
        n_correct = correct_at[k]
        n_total = total_at[k]
        
        p_correct_given_reached = n_correct / n_reached if n_reached > 0 else 0
        p_reached = n_reached / len(trajectory_results)
        cum_p = p_reached * p_correct_given_reached if k == 0 else (n_correct / len(trajectory_results) if n_total > 0 else 0)
        
        # Actually, cumulative P = P(all steps 0..k correct) = reached[k+1] / total
        cum_correct = reached.get(k+1, 0) if k < max_step - 1 else n_correct
        cum_p = cum_correct / len(trajectory_results)
        
        step_data.append({
            'step': k,
            'reached': n_reached,
            'correct': n_correct,
            'total_with_step': n_total,
            'p_correct_given_reached': p_correct_given_reached,
            'p_reached': p_reached,
            'cum_p': cum_p,
        })
        
        print(f"{k:>5} {n_reached:>10} {n_correct:>8}/{n_reached:<7} {p_correct_given_reached:>18.3f} {p_reached:>12.3f} {cum_p:>14.4f}")
    
    # Key insight: is per-step accuracy CONSTANT or does it change?
    accuracies = [d['p_correct_given_reached'] for d in step_data if d['reached'] >= 50]
    if len(accuracies) >= 3:
        early = np.mean(accuracies[:3])
        late = np.mean(accuracies[3:min(8, len(accuracies))])
        print(f"\nEarly steps (0-2) avg accuracy: {early:.3f}")
        print(f"Later steps (3-7) avg accuracy: {late:.3f}")
        print(f"Trend: {'improving' if late > early + 0.02 else 'declining' if late < early - 0.02 else 'stable'}")
    
    return step_data


def analysis_2_majority_vote(multisample_results):
    """Simulate majority vote using C4+C7 multi-sample data.
    
    For each step, take K=10 samples and:
    - Greedy: first sample's correctness
    - Majority vote (action type): most common action type, then most common full action
    - Oracle: any sample correct
    """
    print("\n" + "=" * 60)
    print("Analysis 2: Majority Vote Simulation (C4+C7 data)")
    print("=" * 60)
    
    total_steps = 0
    greedy_correct = 0
    majority_type_correct = 0
    majority_full_correct = 0
    oracle_correct = 0
    
    # Per action type
    type_stats = defaultdict(lambda: {'total': 0, 'greedy': 0, 'majority': 0, 'oracle': 0})
    
    def parse_bool(v):
        """Parse string or bool to bool."""
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() == 'true'

    def parse_pred_action(v):
        """Parse pred_action which may be string repr of dict."""
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            try:
                return eval(v)  # safe here - it's our own data
            except:
                return {}
        return {}

    for episode in multisample_results:
        for step in episode.get('step_samples', []):
            samples = step.get('samples', [])
            if not samples:
                continue

            gt_action = step.get('gt_action', {})
            gt_type = step.get('gt_action_type', gt_action.get('action', 'unknown'))
            total_steps += 1

            # Greedy: first sample
            if parse_bool(samples[0].get('extract_match', False)):
                greedy_correct += 1
                type_stats[gt_type]['greedy'] += 1

            # Oracle: any sample
            any_correct = any(parse_bool(s.get('extract_match', False)) for s in samples)
            if any_correct:
                oracle_correct += 1
                type_stats[gt_type]['oracle'] += 1

            # Majority vote on action type
            action_types = []
            for s in samples:
                pred = parse_pred_action(s.get('pred_action', {}))
                if pred:
                    action_types.append(pred.get('action', 'unknown'))

            if action_types:
                type_counter = Counter(action_types)
                voted_type = type_counter.most_common(1)[0][0]

                # Among samples with the voted type, check if any is correct
                voted_samples = [s for i, s in enumerate(samples)
                                if i < len(action_types) and action_types[i] == voted_type]

                voted_correct = any(parse_bool(s.get('extract_match', False)) for s in voted_samples)
                if voted_correct:
                    majority_full_correct += 1
                    type_stats[gt_type]['majority'] += 1

                # Check if voted type matches GT type
                if voted_type == gt_type:
                    majority_type_correct += 1

            type_stats[gt_type]['total'] += 1
    
    print(f"\nTotal steps: {total_steps}")
    print(f"\n{'Method':<25} {'Accuracy':>10} {'Correct':>10}")
    print("-" * 50)
    print(f"{'Greedy (K=1)':<25} {greedy_correct/total_steps:>10.3f} {greedy_correct:>10}")
    print(f"{'Majority Vote Type':<25} {majority_type_correct/total_steps:>10.3f} {majority_type_correct:>10}")
    print(f"{'Majority Vote (oracle in voted type)':<25} {majority_full_correct/total_steps:>10.3f} {majority_full_correct:>10}")
    print(f"{'Oracle (best-of-K)':<25} {oracle_correct/total_steps:>10.3f} {oracle_correct:>10}")
    
    print(f"\n{'Action Type':<15} {'Total':>6} {'Greedy':>8} {'Majority':>10} {'Oracle':>8}")
    print("-" * 50)
    for at in sorted(type_stats.keys()):
        s = type_stats[at]
        t = s['total']
        print(f"{at:<15} {t:>6} {s['greedy']/t:>8.3f} {s['majority']/t:>10.3f} {s['oracle']/t:>8.3f}")
    
    return {
        'total_steps': total_steps,
        'greedy': greedy_correct / total_steps,
        'majority_type_acc': majority_type_correct / total_steps,
        'majority_full': majority_full_correct / total_steps,
        'oracle': oracle_correct / total_steps,
        'per_type': {k: {kk: vv for kk, vv in v.items()} for k, v in type_stats.items()},
    }


def analysis_3_confidence_calibration(multisample_results):
    """Analyze whether multi-sample agreement is a good confidence signal.
    
    Bin steps by agreement level, compute accuracy in each bin.
    Good calibration = high agreement → high accuracy.
    """
    print("\n" + "=" * 60)
    print("Analysis 3: Confidence Calibration (Agreement as Uncertainty)")
    print("=" * 60)
    
    bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    bin_data = {f"{lo:.1f}-{hi:.1f}": {'total': 0, 'correct': 0} for lo, hi in bins}
    
    all_agreements = []
    all_correct = []
    
    def parse_bool(v):
        if isinstance(v, bool): return v
        return str(v).strip().lower() == 'true'
    def parse_pred(v):
        if isinstance(v, dict): return v
        if isinstance(v, str):
            try: return eval(v)
            except: return {}
        return {}

    for episode in multisample_results:
        for step in episode.get('step_samples', []):
            samples = step.get('samples', [])
            if not samples:
                continue

            # Agreement = fraction of samples that match the most common action type
            action_types = []
            for s in samples:
                pred = parse_pred(s.get('pred_action', {}))
                if pred:
                    action_types.append(pred.get('action', 'unknown'))
            if not action_types:
                continue

            type_counter = Counter(action_types)
            agreement = type_counter.most_common(1)[0][1] / len(action_types)

            greedy_correct = 1 if parse_bool(samples[0].get('extract_match', False)) else 0

            all_agreements.append(agreement)
            all_correct.append(greedy_correct)

            for lo, hi in bins:
                if lo <= agreement < hi:
                    key = f"{lo:.1f}-{hi:.1f}"
                    bin_data[key]['total'] += 1
                    bin_data[key]['correct'] += greedy_correct
                    break
    
    print(f"\n{'Agreement Bin':<15} {'Count':>8} {'Accuracy':>10} {'Fraction':>10}")
    print("-" * 45)
    total = sum(b['total'] for b in bin_data.values())
    for key, b in bin_data.items():
        if b['total'] > 0:
            acc = b['correct'] / b['total']
            frac = b['total'] / total
            print(f"{key:<15} {b['total']:>8} {acc:>10.3f} {frac:>10.3f}")
    
    # Correlation
    if all_agreements:
        corr = np.corrcoef(all_agreements, all_correct)[0, 1]
        print(f"\nPearson correlation (agreement vs correct): {corr:.3f}")
        print(f"Interpretation: {'Strong' if abs(corr) > 0.3 else 'Moderate' if abs(corr) > 0.15 else 'Weak'} signal")
    
    return bin_data


def analysis_4_error_pattern_universality(trajectory_results):
    """Analyze action error patterns for cross-dataset comparison.
    
    Focus on universal patterns (not dataset-specific like 'open'):
    - What fraction of errors are "near-miss" (right type, wrong target)?
    - What fraction are "complete miss" (wrong type)?
    - Is there a universal "hardest step" pattern?
    - Error clustering: do errors come in bursts or are uniformly distributed?
    """
    print("\n" + "=" * 60)
    print("Analysis 4: Universal Error Patterns")
    print("=" * 60)
    
    total_errors = 0
    near_miss = 0  # type_match=True, extract_match=False
    complete_miss = 0  # type_match=False
    
    # Error by relative step position (normalized to trajectory length)
    position_bins = {f"{i*10}-{(i+1)*10}%": {'total': 0, 'errors': 0} for i in range(10)}
    
    # First-error position distribution
    first_error_positions = []
    
    # Consecutive correct run lengths
    run_lengths = []
    
    for r in trajectory_results:
        n = r['num_steps']
        correct_run = r['final_step_id']
        run_lengths.append(correct_run)
        
        if correct_run < n:
            # There was an error
            first_error_positions.append(correct_run / n)  # normalized position
        
        for s in r.get('step_results', []):
            if not s.get('extract_match', False):
                total_errors += 1
                if s.get('type_match', False):
                    near_miss += 1
                else:
                    complete_miss += 1
            
            # Bin by relative position
            rel_pos = s['step_num'] / n
            bin_idx = min(int(rel_pos * 10), 9)
            bin_key = f"{bin_idx*10}-{(bin_idx+1)*10}%"
            position_bins[bin_key]['total'] += 1
            if not s.get('extract_match', False):
                position_bins[bin_key]['errors'] += 1
    
    print(f"\nTotal errors: {total_errors}")
    print(f"Near-miss (right type, wrong target): {near_miss} ({near_miss/total_errors*100:.1f}%)")
    print(f"Complete-miss (wrong type): {complete_miss} ({complete_miss/total_errors*100:.1f}%)")
    print(f"\nNear-miss / Complete-miss ratio: {near_miss/complete_miss:.2f}")
    
    print(f"\nError rate by relative step position:")
    print(f"{'Position':<12} {'Total':>8} {'Errors':>8} {'Error Rate':>12}")
    print("-" * 42)
    for key, b in position_bins.items():
        if b['total'] > 0:
            print(f"{key:<12} {b['total']:>8} {b['errors']:>8} {b['errors']/b['total']:>12.3f}")
    
    # First error position stats
    if first_error_positions:
        print(f"\nFirst error position (normalized):")
        print(f"  Mean: {np.mean(first_error_positions):.3f}")
        print(f"  Median: {np.median(first_error_positions):.3f}")
        print(f"  Std: {np.std(first_error_positions):.3f}")
        pct_first_quarter = sum(1 for p in first_error_positions if p < 0.25) / len(first_error_positions)
        print(f"  Errors in first 25% of trajectory: {pct_first_quarter:.1%}")
    
    # Run length stats
    print(f"\nConsecutive correct run lengths:")
    print(f"  Mean: {np.mean(run_lengths):.2f}")
    print(f"  Median: {np.median(run_lengths):.1f}")
    print(f"  Max: {max(run_lengths)}")
    for k in [0, 1, 2, 3, 5, 10]:
        pct = sum(1 for r in run_lengths if r >= k) / len(run_lengths)
        print(f"  P(run >= {k}): {pct:.3f}")
    
    return {
        'total_errors': total_errors,
        'near_miss_rate': near_miss / total_errors,
        'complete_miss_rate': complete_miss / total_errors,
        'mean_first_error_pos': float(np.mean(first_error_positions)) if first_error_positions else None,
        'mean_run_length': float(np.mean(run_lengths)),
    }


def main():
    output_dir = os.path.join(PROJECT_ROOT, 'outputs', 'cross_dataset_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load AC trajectory results
    ac_traj_path = os.path.join(PROJECT_ROOT, 'outputs', 'eval_a_ac', 'Qwen2.5-VL-7B', 'trajectory_results.jsonl')
    ac_traj = load_jsonl(ac_traj_path)
    print(f"Loaded {len(ac_traj)} AC trajectory results")
    
    # Load C4+C7 multi-sample results
    c4c7_path = os.path.join(PROJECT_ROOT, 'outputs', 'eval_c4c7_ac', 'Qwen2.5-VL-7B', 'multisample_results.jsonl')
    c4c7 = load_jsonl(c4c7_path)
    print(f"Loaded {len(c4c7)} C4+C7 episodes")
    
    # Run analyses
    compounding = analysis_1_compounding_decomposition(ac_traj)
    majority = analysis_2_majority_vote(c4c7)
    calibration = analysis_3_confidence_calibration(c4c7)
    errors = analysis_4_error_pattern_universality(ac_traj)
    
    # Save results
    results = {
        'compounding': compounding,
        'majority_vote': majority,
        'confidence_calibration': {k: v for k, v in calibration.items()},
        'error_patterns': errors,
    }
    
    out_path = os.path.join(output_dir, 'cross_dataset_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else None)
    
    print(f"\n\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
