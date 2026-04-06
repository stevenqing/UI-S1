#!/usr/bin/env python3
"""
GUI-360 Cross-Dataset Verification: Replicate P1, P4, Stage1 from AC on GUI-360.

Uses existing GUI-360 multi-sample data (19,046 steps) to verify:
- Step 2 (P4): SPWA weight validation — early steps correlate more with TSR
- Step 3 (P1): Feature distributions differ across error types
- Step 5 (Stage1 + Oracle Gap): Routing signals detectable + routing potential
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score, f1_score)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'eval_gui360_verification'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = PROJECT_ROOT / 'outputs' / 'eval_gui360_multisample' / 'multisample_results.jsonl'


def load_data():
    """Load GUI-360 multi-sample data."""
    data = []
    with open(DATA_PATH) as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Loaded {len(data)} steps from {len(set(d['trajectory_id'] for d in data))} trajectories")
    return data


def classify_error_type(step):
    """Classify step into error type."""
    if step['greedy_function_match'] and step['greedy_args_match']:
        return 'correct'
    elif not step['greedy_function_match']:
        return 'action_error'
    else:
        return 'grounding_error'


def compute_coord_spread(step):
    """Compute coordinate spread from K=10 predictions."""
    coords = []
    for pred in step.get('K_predictions', []):
        c = pred.get('coordinates')
        if c and isinstance(c, (list, tuple)) and len(c) >= 2:
            try:
                coords.append([float(c[0]), float(c[1])])
            except (ValueError, TypeError):
                pass
    if len(coords) < 2:
        return None
    coords = np.array(coords)
    return float(np.mean(np.std(coords, axis=0)))


# ===========================================================================
# STEP 2: P4 — SPWA Weight Validation
# ===========================================================================
def run_p4_step_correlation(data):
    """Replicate P4: Do early steps correlate more with TSR?"""
    print("\n" + "=" * 70)
    print("  STEP 2 (P4): SPWA Weight Validation on GUI-360")
    print("=" * 70)

    # Group by trajectory
    trajs = defaultdict(list)
    for step in data:
        trajs[step['trajectory_id']].append(step)

    # Sort steps within each trajectory
    for tid in trajs:
        trajs[tid].sort(key=lambda s: s['step_index'])

    # Per-step-position statistics
    step_stats = defaultdict(lambda: {'greedy_acc': [], 'agreement': [],
                                       'temp_acc': [], 'oracle_acc': []})
    for step in data:
        k = step['step_index']
        step_stats[k]['greedy_acc'].append(1 if step['greedy_function_match'] else 0)
        step_stats[k]['agreement'].append(step['agreement_rate'])
        step_stats[k]['temp_acc'].append(step['temp_function_match_rate'])
        step_stats[k]['oracle_acc'].append(1 if step['oracle_function_match'] else 0)

    print("\nPer-step-position accuracy:")
    print(f"{'Step':>5} {'N':>6} {'Greedy':>8} {'Temp':>8} {'Oracle':>8} {'Agreement':>10}")
    positions = sorted(step_stats.keys())
    step_p = {}  # greedy accuracy per position
    for k in positions:
        s = step_stats[k]
        n = len(s['greedy_acc'])
        ga = np.mean(s['greedy_acc'])
        ta = np.mean(s['temp_acc'])
        oa = np.mean(s['oracle_acc'])
        ag = np.mean(s['agreement'])
        step_p[k] = ga
        print(f"{k:>5} {n:>6} {ga:>8.4f} {ta:>8.4f} {oa:>8.4f} {ag:>10.4f}")

    # SPWA weights
    print("\nSPWA weights (w_k = prod_{j<k} p_j):")
    spwa_weights = {}
    for k in positions:
        w = 1.0
        for j in positions:
            if j < k:
                w *= step_p.get(j, 0.5)
        spwa_weights[k] = w
        print(f"  w_{k} = {w:.6f}")

    # Episode-level TSR
    traj_tsr = {}
    for tid, steps in trajs.items():
        tsr = all(s['greedy_function_match'] and s['greedy_args_match'] for s in steps)
        traj_tsr[tid] = 1 if tsr else 0
    overall_tsr = np.mean(list(traj_tsr.values()))
    print(f"\nOverall greedy TSR: {overall_tsr:.4f} ({sum(traj_tsr.values())}/{len(traj_tsr)})")

    # Per-step correlation with episode TSR
    print("\nPer-step reasoning quality correlation with episode TSR:")
    step_corr = {}
    for k in positions:
        # For each trajectory, get reasoning quality at step k
        rq_list = []
        tsr_list = []
        for tid, steps in trajs.items():
            steps_at_k = [s for s in steps if s['step_index'] == k]
            if steps_at_k:
                s = steps_at_k[0]
                rq = s['temp_function_match_rate']  # K=10 correct rate as proxy
                rq_list.append(rq)
                tsr_list.append(traj_tsr[tid])

        if len(set(rq_list)) > 1 and len(set(tsr_list)) > 1:
            r, p = pearsonr(rq_list, tsr_list)
            rho, p_s = spearmanr(rq_list, tsr_list)
        else:
            r, p, rho, p_s = 0, 1, 0, 1
        step_corr[k] = {'r': r, 'p': p, 'rho': rho, 'p_spearman': p_s,
                         'n': len(rq_list)}
        print(f"  Step {k}: r={r:.4f} (p={p:.4f}), rho={rho:.4f}, N={len(rq_list)}")

    # Early vs Late
    early_r = [step_corr[k]['r'] for k in positions if k <= 2 and k in step_corr]
    late_r = [step_corr[k]['r'] for k in positions if k > 2 and k in step_corr]
    print(f"\n  Early (k<=2) mean r: {np.mean(early_r):.4f}")
    print(f"  Late  (k>2)  mean r: {np.mean(late_r):.4f}")
    print(f"  Ratio: {np.mean(early_r) / max(np.mean(late_r), 0.001):.2f}x")

    # SPWA weight ratio
    early_w = [spwa_weights[k] for k in positions if k <= 2]
    late_w = [spwa_weights[k] for k in positions if k > 2]
    print(f"\n  SPWA weight early mean: {np.mean(early_w):.6f}")
    print(f"  SPWA weight late mean:  {np.mean(late_w):.6f}")
    print(f"  Weight ratio: {np.mean(early_w) / max(np.mean(late_w), 1e-10):.1f}x")

    # Compare with AC
    print("\n  [Compare] AC: early r=0.30, late r=0.10, ratio=3.0x")
    print(f"  [Compare] GUI-360: early r={np.mean(early_r):.4f}, late r={np.mean(late_r):.4f}")

    results = {
        'step_accuracy': {str(k): {'greedy': np.mean(step_stats[k]['greedy_acc']),
                                     'temp': np.mean(step_stats[k]['temp_acc']),
                                     'oracle': np.mean(step_stats[k]['oracle_acc']),
                                     'agreement': np.mean(step_stats[k]['agreement']),
                                     'n': len(step_stats[k]['greedy_acc'])}
                          for k in positions},
        'spwa_weights': {str(k): v for k, v in spwa_weights.items()},
        'overall_tsr': overall_tsr,
        'step_correlations': {str(k): v for k, v in step_corr.items()},
        'early_mean_r': float(np.mean(early_r)),
        'late_mean_r': float(np.mean(late_r)),
        'early_late_ratio': float(np.mean(early_r) / max(np.mean(late_r), 0.001)),
    }
    return results


# ===========================================================================
# STEP 3: P1 — Feature Distribution Analysis
# ===========================================================================
def run_p1_feature_analysis(data):
    """Replicate P1: Do features differ across error types?"""
    print("\n" + "=" * 70)
    print("  STEP 3 (P1): Feature Distribution Analysis on GUI-360")
    print("=" * 70)

    # Classify steps
    by_type = defaultdict(list)
    for step in data:
        et = classify_error_type(step)
        step['error_type'] = et
        by_type[et].append(step)

    print("\nError type distribution:")
    for et in ['correct', 'action_error', 'grounding_error']:
        n = len(by_type[et])
        print(f"  {et}: {n} ({n/len(data)*100:.2f}%)")

    # Feature statistics per error type
    print("\nFeature statistics by error type:")
    print(f"{'Type':<20} {'N':>6} {'Agreement':>10} {'Entropy':>10} {'CoordSpd':>10}")

    feat_by_type = {}
    for et in ['correct', 'action_error', 'grounding_error']:
        steps = by_type[et]
        agreements = [s['agreement_rate'] for s in steps]
        entropies = [s['action_entropy'] for s in steps]
        coord_spreads = [compute_coord_spread(s) for s in steps]
        coord_spreads = [c for c in coord_spreads if c is not None]

        feat_by_type[et] = {
            'agreement': agreements,
            'entropy': entropies,
            'coord_spread': coord_spreads,
        }

        cs_mean = np.mean(coord_spreads) if coord_spreads else float('nan')
        print(f"  {et:<20} {len(steps):>6} {np.mean(agreements):>10.4f} "
              f"{np.mean(entropies):>10.4f} {cs_mean:>10.4f}")

    # Statistical tests
    print("\nStatistical tests (Mann-Whitney U):")
    tests = []

    # H1: action_error has lower agreement than correct
    if feat_by_type['correct']['agreement'] and feat_by_type['action_error']['agreement']:
        u, p = mannwhitneyu(feat_by_type['action_error']['agreement'],
                           feat_by_type['correct']['agreement'],
                           alternative='less')
        d = (np.mean(feat_by_type['correct']['agreement']) -
             np.mean(feat_by_type['action_error']['agreement']))
        pooled_std = np.sqrt((np.std(feat_by_type['correct']['agreement'])**2 +
                             np.std(feat_by_type['action_error']['agreement'])**2) / 2)
        cohens_d = d / pooled_std if pooled_std > 0 else 0
        print(f"  H1 (action < correct agreement): U={u:.0f}, p={p:.2e}, Cohen's d={cohens_d:.3f}")
        tests.append({'name': 'H1_action_lower_agreement', 'U': u, 'p': p, 'd': cohens_d})

    # H2: action_error has higher entropy than correct
    if feat_by_type['correct']['entropy'] and feat_by_type['action_error']['entropy']:
        u, p = mannwhitneyu(feat_by_type['action_error']['entropy'],
                           feat_by_type['correct']['entropy'],
                           alternative='greater')
        d = (np.mean(feat_by_type['action_error']['entropy']) -
             np.mean(feat_by_type['correct']['entropy']))
        pooled_std = np.sqrt((np.std(feat_by_type['correct']['entropy'])**2 +
                             np.std(feat_by_type['action_error']['entropy'])**2) / 2)
        cohens_d = d / pooled_std if pooled_std > 0 else 0
        print(f"  H2 (action > correct entropy):   U={u:.0f}, p={p:.2e}, Cohen's d={cohens_d:.3f}")
        tests.append({'name': 'H2_action_higher_entropy', 'U': u, 'p': p, 'd': cohens_d})

    # H3: grounding has higher coord_spread than action
    if (feat_by_type['grounding_error']['coord_spread'] and
        feat_by_type['action_error']['coord_spread']):
        u, p = mannwhitneyu(feat_by_type['grounding_error']['coord_spread'],
                           feat_by_type['action_error']['coord_spread'],
                           alternative='greater')
        d = (np.mean(feat_by_type['grounding_error']['coord_spread']) -
             np.mean(feat_by_type['action_error']['coord_spread']))
        pooled_std = np.sqrt((np.std(feat_by_type['grounding_error']['coord_spread'])**2 +
                             np.std(feat_by_type['action_error']['coord_spread'])**2) / 2)
        cohens_d = d / pooled_std if pooled_std > 0 else 0
        print(f"  H3 (grounding > action coord):   U={u:.0f}, p={p:.2e}, Cohen's d={cohens_d:.3f}")
        tests.append({'name': 'H3_grounding_higher_coord', 'U': u, 'p': p, 'd': cohens_d})

    # H4: correct has higher agreement than all errors
    error_agreements = (feat_by_type['action_error']['agreement'] +
                       feat_by_type['grounding_error']['agreement'])
    if feat_by_type['correct']['agreement'] and error_agreements:
        u, p = mannwhitneyu(feat_by_type['correct']['agreement'],
                           error_agreements, alternative='greater')
        d = np.mean(feat_by_type['correct']['agreement']) - np.mean(error_agreements)
        pooled_std = np.sqrt((np.std(feat_by_type['correct']['agreement'])**2 +
                             np.std(error_agreements)**2) / 2)
        cohens_d = d / pooled_std if pooled_std > 0 else 0
        print(f"  H4 (correct > error agreement):  U={u:.0f}, p={p:.2e}, Cohen's d={cohens_d:.3f}")
        tests.append({'name': 'H4_correct_higher_agreement', 'U': u, 'p': p, 'd': cohens_d})

    # 4-layer classification
    print("\n4-Layer Reasoning Classification:")
    median_entropy = np.median([s['action_entropy'] for s in data])
    layers = defaultdict(int)
    for step in data:
        et = step['error_type']
        a = step['agreement_rate']
        e = step['action_entropy']
        if et == 'correct' and a > 0.7:
            layers['correct_confident'] += 1
        elif et == 'correct':
            layers['correct_uncertain'] += 1
        elif et == 'grounding_error':
            layers['grounding_uncertainty'] += 1
        elif a < 0.5 and e > median_entropy:
            layers['action_confusion'] += 1
        elif a < 0.5 and e <= median_entropy:
            layers['goal_confusion'] += 1
        else:
            layers['action_other'] += 1

    for layer, count in sorted(layers.items(), key=lambda x: -x[1]):
        print(f"  {layer}: {count} ({count/len(data)*100:.2f}%)")

    # Compare with AC
    print("\n  [Compare] AC P1:")
    print("    correct: agreement=0.78, entropy=0.24")
    print("    action_error: agreement=0.39, entropy=1.63")
    print(f"  [Compare] GUI-360:")
    print(f"    correct: agreement={np.mean(feat_by_type['correct']['agreement']):.4f}, "
          f"entropy={np.mean(feat_by_type['correct']['entropy']):.4f}")
    print(f"    action_error: agreement={np.mean(feat_by_type['action_error']['agreement']):.4f}, "
          f"entropy={np.mean(feat_by_type['action_error']['entropy']):.4f}")

    results = {
        'error_distribution': {et: len(by_type[et]) for et in by_type},
        'feature_stats': {
            et: {
                'mean_agreement': float(np.mean(feat_by_type[et]['agreement'])),
                'std_agreement': float(np.std(feat_by_type[et]['agreement'])),
                'mean_entropy': float(np.mean(feat_by_type[et]['entropy'])),
                'std_entropy': float(np.std(feat_by_type[et]['entropy'])),
                'mean_coord_spread': float(np.mean(feat_by_type[et]['coord_spread']))
                    if feat_by_type[et]['coord_spread'] else None,
                'n_with_coords': len(feat_by_type[et]['coord_spread']),
            }
            for et in feat_by_type
        },
        'statistical_tests': [{k: float(v) if isinstance(v, (int, float, np.floating))
                                else v for k, v in t.items()} for t in tests],
        'layer_distribution': dict(layers),
    }
    return results


# ===========================================================================
# STEP 5: Stage 1 Classification + Oracle Gap
# ===========================================================================
def run_stage1_and_oracle(data):
    """Stage 1 error classification + oracle gap routing potential."""
    print("\n" + "=" * 70)
    print("  STEP 5: Stage 1 Classification + Oracle Gap on GUI-360")
    print("=" * 70)

    # --- Part A: Stage 1 Classification ---
    print("\n--- Part A: Error Classification ---")

    features = []
    labels = []
    binary_labels = []  # error vs correct
    for step in data:
        et = classify_error_type(step)
        f = [step['agreement_rate'], step['action_entropy'],
             step['step_index'] / max(step['total_steps'], 1)]
        cs = compute_coord_spread(step)
        f.append(cs if cs is not None else 0.0)
        features.append(f)
        labels.append(et)
        binary_labels.append(0 if et == 'correct' else 1)

    X = np.array(features)
    y = np.array(labels)
    y_bin = np.array(binary_labels)

    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    _, _, y_bin_train, y_bin_test = train_test_split(
        X, y_bin, test_size=0.2, random_state=42, stratify=y)

    # 3-class classification
    clf3 = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    clf3.fit(X_train, y_train)
    y_pred3 = clf3.predict(X_test)
    acc3 = accuracy_score(y_test, y_pred3)
    print(f"\n3-class accuracy: {acc3:.4f}")
    print(classification_report(y_test, y_pred3))

    # Binary classification (error vs correct)
    clf_bin = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    clf_bin.fit(X_train, y_bin_train)
    y_pred_bin_proba = clf_bin.predict_proba(X_test)[:, 1]
    auroc_bin = roc_auc_score(y_bin_test, y_pred_bin_proba)
    print(f"Binary AUROC (error vs correct): {auroc_bin:.4f}")
    print(f"  [Compare] AC Stage 1 AUROC: 0.6477")

    # Feature importance
    print("\nFeature importance (3-class coefficients):")
    feature_names = ['agreement', 'entropy', 'step_ratio', 'coord_spread']
    for i, name in enumerate(feature_names):
        coeffs = clf3.coef_[:, i]
        print(f"  {name}: {coeffs}")

    # --- Part B: Oracle Gap ---
    print("\n--- Part B: Oracle Gap (Routing Potential) ---")

    # Group by trajectory
    trajs = defaultdict(list)
    for step in data:
        trajs[step['trajectory_id']].append(step)

    # Episode-level TSR with different strategies
    greedy_tsrs = []
    oracle_tsrs = []
    routing_tsrs = []  # perfect routing: oracle on errors, greedy on correct

    for tid, steps in trajs.items():
        # Greedy TSR
        greedy_ok = all(s['greedy_function_match'] and s['greedy_args_match'] for s in steps)
        greedy_tsrs.append(1 if greedy_ok else 0)

        # Oracle TSR (best of K=10)
        oracle_ok = all(s['oracle_function_match'] and s['oracle_args_match'] for s in steps)
        oracle_tsrs.append(1 if oracle_ok else 0)

        # Perfect routing: for each step, use oracle if error, greedy if correct
        routing_ok = True
        for s in steps:
            if s['greedy_function_match'] and s['greedy_args_match']:
                # Correct step — keep greedy
                pass
            else:
                # Error step — use oracle
                if not (s['oracle_function_match'] and s['oracle_args_match']):
                    routing_ok = False
                    break
        routing_tsrs.append(1 if routing_ok else 0)

    print(f"\nEpisode-level TSR ({len(trajs)} trajectories):")
    print(f"  Greedy TSR:         {np.mean(greedy_tsrs)*100:.2f}%")
    print(f"  Oracle TSR:         {np.mean(oracle_tsrs)*100:.2f}%")
    print(f"  Perfect Routing:    {np.mean(routing_tsrs)*100:.2f}%")
    print(f"  Oracle Gap:         +{(np.mean(oracle_tsrs) - np.mean(greedy_tsrs))*100:.2f}pp")
    print(f"  Routing Gap:        +{(np.mean(routing_tsrs) - np.mean(greedy_tsrs))*100:.2f}pp")

    # Step-level improvement potential
    print("\nStep-level accuracy:")
    greedy_fn = np.mean([1 if s['greedy_function_match'] else 0 for s in data])
    oracle_fn = np.mean([1 if s['oracle_function_match'] else 0 for s in data])
    greedy_both = np.mean([1 if s['greedy_function_match'] and s['greedy_args_match'] else 0
                           for s in data])
    oracle_both = np.mean([1 if s['oracle_function_match'] and s['oracle_args_match'] else 0
                           for s in data])
    print(f"  Greedy function match: {greedy_fn*100:.2f}%")
    print(f"  Oracle function match: {oracle_fn*100:.2f}%")
    print(f"  Greedy full match:     {greedy_both*100:.2f}%")
    print(f"  Oracle full match:     {oracle_both*100:.2f}%")
    print(f"  Function oracle gap:   +{(oracle_fn-greedy_fn)*100:.2f}pp")
    print(f"  Full oracle gap:       +{(oracle_both-greedy_both)*100:.2f}pp")

    # Compare with AC
    print(f"\n  [Compare] AC: oracle gap = +19pp (greedy 62% → oracle 81%)")
    print(f"  [Compare] GUI-360: oracle gap = +{(oracle_fn-greedy_fn)*100:.2f}pp "
          f"(greedy {greedy_fn*100:.2f}% → oracle {oracle_fn*100:.2f}%)")

    results = {
        'classification': {
            'accuracy_3class': float(acc3),
            'auroc_binary': float(auroc_bin),
            'report': classification_report(y_test, y_pred3, output_dict=True),
        },
        'oracle_gap': {
            'greedy_tsr': float(np.mean(greedy_tsrs)),
            'oracle_tsr': float(np.mean(oracle_tsrs)),
            'routing_tsr': float(np.mean(routing_tsrs)),
            'oracle_gap_pp': float((np.mean(oracle_tsrs) - np.mean(greedy_tsrs)) * 100),
            'routing_gap_pp': float((np.mean(routing_tsrs) - np.mean(greedy_tsrs)) * 100),
            'step_greedy_fn': float(greedy_fn),
            'step_oracle_fn': float(oracle_fn),
            'step_greedy_both': float(greedy_both),
            'step_oracle_both': float(oracle_both),
        },
    }
    return results


def main():
    data = load_data()

    # Run all three analyses
    p4_results = run_p4_step_correlation(data)
    p1_results = run_p1_feature_analysis(data)
    stage1_results = run_stage1_and_oracle(data)

    # Combine and save
    all_results = {
        'step2_p4_spwa': p4_results,
        'step3_p1_features': p1_results,
        'step5_stage1_oracle': stage1_results,
    }

    out_path = OUTPUT_DIR / 'gui360_cross_dataset_verification.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nAll results saved to {out_path}")


if __name__ == '__main__':
    main()
