"""X1: Oracle Gap Regression — What factors predict multi-sampling value?

Builds logistic regression predicting oracle_gain (greedy wrong, some sample correct)
from step-level features using C4+C7 multi-sample data.

Analyzes feature importance to understand what drives the oracle gap.
All offline — 0 GPU required.
"""

import argparse
import json
import os
import numpy as np
from collections import Counter, defaultdict


def load_multisample_steps(jsonl_path):
    """Load per-step multi-sample data from C4+C7 results."""
    steps = []
    with open(jsonl_path) as f:
        for line in f:
            ep = json.loads(line)
            num_ep_steps = ep.get('num_steps', 0)
            for step in ep.get('step_samples', []):
                samples = step.get('samples', [])
                if not samples:
                    continue

                K = len(samples)

                # Extract action types from all samples
                action_types = []
                coords = []
                for s in samples:
                    pa = s.get('pred_action')
                    if pa and isinstance(pa, dict):
                        action_types.append(pa.get('action', 'unknown'))
                        # Extract coordinates if available
                        if 'coordinate' in pa:
                            coord = pa['coordinate']
                            if isinstance(coord, (list, tuple)) and len(coord) == 2:
                                coords.append(coord)
                    else:
                        action_types.append('parse_fail')

                if not action_types:
                    continue

                type_counter = Counter(action_types)
                agreement = type_counter.most_common(1)[0][1] / K
                n_unique_types = len(type_counter)

                # Action type entropy
                type_probs = [c / K for c in type_counter.values()]
                entropy = -sum(p * np.log2(p + 1e-10) for p in type_probs)

                # Coordinate spread (std of coords)
                if len(coords) >= 2:
                    try:
                        coords_arr = np.array(coords, dtype=np.float64)
                        std_vals = np.std(coords_arr, axis=0)
                        coord_spread = float(np.nanmean(std_vals))
                        if np.isnan(coord_spread) or np.isinf(coord_spread):
                            coord_spread = 0.0
                    except (ValueError, TypeError):
                        coord_spread = 0.0
                else:
                    coord_spread = 0.0

                # Greedy accuracy
                greedy_correct = int(samples[0].get('extract_match', False))

                # Oracle accuracy
                oracle_correct = int(any(s.get('extract_match', False) for s in samples))

                # Oracle gain: greedy wrong but some sample correct
                oracle_gain = int(not greedy_correct and oracle_correct)

                # GT action type
                gt_type = step.get('gt_action_type', 'unknown')

                # Is coordinate-based action?
                is_coord_action = int(gt_type in ['click', 'long_press'])

                steps.append({
                    'step_num': step.get('step_num', 0),
                    'trajectory_length': num_ep_steps,
                    'gt_action_type': gt_type,
                    'agreement': agreement,
                    'n_unique_types': n_unique_types,
                    'action_entropy': entropy,
                    'coord_spread': coord_spread,
                    'is_coord_action': is_coord_action,
                    'greedy_correct': greedy_correct,
                    'oracle_correct': oracle_correct,
                    'oracle_gain': oracle_gain,
                    'K': K,
                })

    return steps


def logistic_regression_manual(X, y, lr=0.01, n_iter=1000):
    """Simple logistic regression with gradient descent.

    Returns weights and bias.
    """
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for _ in range(n_iter):
        z = X @ w + b
        z = np.clip(z, -20, 20)  # numerical stability
        pred = 1 / (1 + np.exp(-z))

        # Gradient
        error = pred - y
        dw = (X.T @ error) / n
        db = np.mean(error)

        w -= lr * dw
        b -= lr * db

    return w, b


def compute_auroc(y_true, y_scores):
    """Compute AUROC manually."""
    # Sort by score descending
    indices = np.argsort(-y_scores)
    y_sorted = y_true[indices]

    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp = 0
    fp = 0
    auc = 0.0

    for y in y_sorted:
        if y == 1:
            tp += 1
        else:
            fp += 1
            auc += tp

    return auc / (n_pos * n_neg)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading multi-sample data...")
    steps = load_multisample_steps(args.multisample_jsonl)
    print(f"  Loaded {len(steps)} step-level samples")

    # --- Basic statistics ---
    total = len(steps)
    greedy_correct = sum(s['greedy_correct'] for s in steps)
    oracle_correct = sum(s['oracle_correct'] for s in steps)
    oracle_gains = sum(s['oracle_gain'] for s in steps)

    print(f"\n=== Basic Statistics ===")
    print(f"  Greedy accuracy: {greedy_correct/total:.4f} ({greedy_correct}/{total})")
    print(f"  Oracle accuracy: {oracle_correct/total:.4f} ({oracle_correct}/{total})")
    print(f"  Oracle gap: {(oracle_correct-greedy_correct)/total:.4f} ({oracle_correct-greedy_correct} steps)")
    print(f"  Oracle gain rate: {oracle_gains/total:.4f} ({oracle_gains} steps where greedy wrong but oracle right)")

    # --- Oracle gain by feature ---
    print(f"\n=== Oracle Gain by Action Type ===")
    by_type = defaultdict(list)
    for s in steps:
        by_type[s['gt_action_type']].append(s)

    print(f"  {'Type':>15} | {'N':>6} | {'Greedy':>8} | {'Oracle':>8} | {'OG Rate':>8} | {'Entropy':>8}")
    for atype, items in sorted(by_type.items(), key=lambda x: -len(x[1])):
        n = len(items)
        gr = sum(s['greedy_correct'] for s in items) / n
        oracle = sum(s['oracle_correct'] for s in items) / n
        og = sum(s['oracle_gain'] for s in items) / n
        ent = np.mean([s['action_entropy'] for s in items])
        print(f"  {atype:>15} | {n:6d} | {gr:8.3f} | {oracle:8.3f} | {og:8.3f} | {ent:8.3f}")

    print(f"\n=== Oracle Gain by Agreement Bin ===")
    agree_bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    for lo, hi in agree_bins:
        in_bin = [s for s in steps if lo <= s['agreement'] < hi]
        if not in_bin:
            continue
        n = len(in_bin)
        og = sum(s['oracle_gain'] for s in in_bin) / n
        gr = sum(s['greedy_correct'] for s in in_bin) / n
        print(f"  [{lo:.1f}, {hi:.1f}): N={n:5d}, greedy={gr:.3f}, OG_rate={og:.3f}")

    print(f"\n=== Oracle Gain by Step Position ===")
    by_step = defaultdict(list)
    for s in steps:
        by_step[s['step_num']].append(s)
    for step_num in sorted(by_step.keys())[:12]:
        items = by_step[step_num]
        n = len(items)
        og = sum(s['oracle_gain'] for s in items) / n
        gr = sum(s['greedy_correct'] for s in items) / n
        print(f"  Step {step_num:2d}: N={n:5d}, greedy={gr:.3f}, OG_rate={og:.3f}")

    # --- Logistic Regression ---
    print(f"\n=== Logistic Regression: P(oracle_gain) ~ features ===")

    feature_names = ['agreement', 'action_entropy', 'coord_spread',
                     'step_num', 'trajectory_length', 'is_coord_action', 'n_unique_types']

    X = np.array([[s[f] for f in feature_names] for s in steps], dtype=float)
    y = np.array([s['oracle_gain'] for s in steps], dtype=float)

    # Replace NaN/inf in features
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-10
    X_norm = (X - X_mean) / X_std

    # Fit logistic regression
    w, b = logistic_regression_manual(X_norm, y, lr=0.1, n_iter=2000)

    # Predictions
    z = X_norm @ w + b
    z = np.clip(z, -20, 20)
    pred = 1 / (1 + np.exp(-z))

    auroc = compute_auroc(y, pred)

    print(f"\n  AUROC: {auroc:.4f}")
    print(f"\n  Feature importance (standardized coefficients):")
    sorted_features = sorted(zip(feature_names, w), key=lambda x: -abs(x[1]))
    for fname, coeff in sorted_features:
        direction = "↑ more OG" if coeff > 0 else "↓ less OG"
        print(f"    {fname:>20}: {coeff:+8.4f}  ({direction})")

    # --- Key Insights ---
    print(f"\n{'='*60}")
    print(f"KEY INSIGHTS")
    print(f"{'='*60}")

    # Most important feature
    top_feature = sorted_features[0][0]
    print(f"\n  Most predictive feature: {top_feature}")
    print(f"    → This is the primary driver of oracle gain")

    # Agreement direction
    agree_coeff = dict(zip(feature_names, w))['agreement']
    if agree_coeff < 0:
        print(f"\n  Agreement has NEGATIVE coefficient ({agree_coeff:.4f})")
        print(f"    → Low agreement → more oracle gain (expected: model uncertainty = opportunity)")
    else:
        print(f"\n  Agreement has POSITIVE coefficient ({agree_coeff:.4f})")
        print(f"    → Unexpected: high agreement correlates with oracle gain")

    # Coord spread direction
    coord_coeff = dict(zip(feature_names, w))['coord_spread']
    if coord_coeff > 0:
        print(f"\n  Coord spread has POSITIVE coefficient ({coord_coeff:.4f})")
        print(f"    → Wider coordinate distribution → more oracle gain")
        print(f"    → Supports: bimodal distributions contain correct candidates")

    # Step position direction
    step_coeff = dict(zip(feature_names, w))['step_num']
    if step_coeff > 0:
        print(f"\n  Step position has POSITIVE coefficient ({step_coeff:.4f})")
        print(f"    → Later steps have more oracle gain (more uncertainty at late steps)")
    else:
        print(f"\n  Step position has NEGATIVE coefficient ({step_coeff:.4f})")
        print(f"    → Earlier steps have more oracle gain")

    # --- Save results ---
    results = {
        'total_steps': total,
        'greedy_accuracy': greedy_correct / total,
        'oracle_accuracy': oracle_correct / total,
        'oracle_gap': (oracle_correct - greedy_correct) / total,
        'oracle_gain_rate': oracle_gains / total,
        'regression': {
            'features': feature_names,
            'coefficients': w.tolist(),
            'bias': float(b),
            'auroc': auroc,
            'feature_importance': {
                fname: float(coeff) for fname, coeff in sorted_features
            },
        },
        'by_action_type': {
            atype: {
                'n': len(items),
                'greedy_acc': sum(s['greedy_correct'] for s in items) / len(items),
                'oracle_acc': sum(s['oracle_correct'] for s in items) / len(items),
                'og_rate': sum(s['oracle_gain'] for s in items) / len(items),
            }
            for atype, items in by_type.items()
        },
        'by_agreement_bin': {
            f"[{lo:.1f},{hi:.1f})": {
                'n': len([s for s in steps if lo <= s['agreement'] < hi]),
                'og_rate': sum(s['oracle_gain'] for s in steps if lo <= s['agreement'] < hi)
                          / max(1, len([s for s in steps if lo <= s['agreement'] < hi])),
            }
            for lo, hi in agree_bins
        },
    }

    out_path = os.path.join(args.output_dir, 'x1_oracle_gap_regression.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X1: Oracle Gap Regression Analysis")
    parser.add_argument("--multisample_jsonl", type=str, required=True,
                        help="Path to C4+C7 multisample_results.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_x1",
                        help="Output directory")
    args = parser.parse_args()
    main(args)
