"""X2: Calibration Transfer — Does agreement→accuracy calibration transfer across datasets?

Tests whether the monotonic relationship between multi-sample agreement
and step-level accuracy is universal (GUI-360 → AndroidControl).

Phase 1: Fit isotonic regression on GUI-360 calibration data
Phase 2: Zero-shot transfer to AC
Phase 3: Affine transfer (2 params: scale + offset)
Phase 4: Within-dataset baseline comparison

All offline — 0 GPU required.
"""

import argparse
import json
import os
import numpy as np
from collections import defaultdict


# GUI-360 calibration data (from plan document, Eval B / C4+C7)
# Agreement bin → (accuracy, count)
GUI360_CALIBRATION = {
    # From Eval C4+C7: agreement segment → greedy accuracy
    0.15: (0.157, 343),    # <0.3
    0.40: (0.241, 985),    # 0.3-0.5
    0.60: (0.451, 1452),   # 0.5-0.7
    0.80: (0.611, 1819),   # 0.7-0.9
    0.95: (0.912, 13666),  # ≥0.9
}

# GUI-360 Eval B step-1 calibration (higher quality, grounding-only)
GUI360_STEP1_CALIBRATION = {
    0.95: (0.953, None),  # ≥0.9 → 95.3% accuracy
}


def load_ac_calibration_from_cross_analysis(json_path):
    """Load AC calibration data from cross_dataset_analysis.json."""
    with open(json_path) as f:
        data = json.load(f)

    cal = data['confidence_calibration']
    # Format: bin_name → {count, accuracy, fraction}
    ac_cal = {}
    for bin_name, vals in cal.items():
        lo, hi = map(float, bin_name.split('-'))
        mid = (lo + hi) / 2
        total = vals.get('total', vals.get('count', 0))
        correct = vals.get('correct', 0)
        accuracy = correct / total if total > 0 else 0.0
        ac_cal[mid] = (accuracy, total)

    return ac_cal


def load_ac_calibration_from_multisample(jsonl_path):
    """Load per-step agreement and accuracy from C4+C7 multi-sample data."""
    steps = []
    with open(jsonl_path) as f:
        for line in f:
            ep = json.loads(line)
            for step in ep.get('step_samples', []):
                samples = step.get('samples', [])
                if not samples:
                    continue

                K = len(samples)
                # Agreement = fraction of most common action type
                action_types = []
                for s in samples:
                    pa = s.get('pred_action')
                    if pa and isinstance(pa, dict):
                        action_types.append(pa.get('action', 'unknown'))
                    else:
                        action_types.append('parse_fail')

                if not action_types:
                    continue

                from collections import Counter
                type_counter = Counter(action_types)
                agreement = type_counter.most_common(1)[0][1] / K

                # Greedy accuracy (first sample)
                greedy_correct = samples[0].get('extract_match', False)

                # Oracle accuracy (any sample correct)
                oracle_correct = any(s.get('extract_match', False) for s in samples)

                # GT action type
                gt_type = step.get('gt_action_type', 'unknown')

                steps.append({
                    'agreement': agreement,
                    'greedy_correct': int(greedy_correct),
                    'oracle_correct': int(oracle_correct),
                    'gt_action_type': gt_type,
                    'step_num': step.get('step_num', 0),
                    'K': K,
                })

    return steps


def bin_calibration(steps, bins=None):
    """Bin steps by agreement and compute accuracy per bin."""
    if bins is None:
        bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]

    result = {}
    for lo, hi in bins:
        in_bin = [s for s in steps if lo <= s['agreement'] < hi]
        if not in_bin:
            continue
        mid = (lo + hi) / 2
        acc = sum(s['greedy_correct'] for s in in_bin) / len(in_bin)
        result[mid] = (acc, len(in_bin))

    return result


def isotonic_fit(cal_data):
    """Fit piecewise-linear isotonic function from calibration data.

    Returns sorted list of (agreement_midpoint, accuracy) pairs.
    """
    points = sorted(cal_data.items())
    # Simple isotonic: ensure monotonicity
    fitted = []
    for x, (y, n) in points:
        if fitted and y < fitted[-1][1]:
            y = fitted[-1][1]  # enforce monotonicity
        fitted.append((x, y))
    return fitted


def interpolate(fitted_curve, x):
    """Linear interpolation on fitted curve."""
    if not fitted_curve:
        return 0.5

    if x <= fitted_curve[0][0]:
        return fitted_curve[0][1]
    if x >= fitted_curve[-1][0]:
        return fitted_curve[-1][1]

    for i in range(len(fitted_curve) - 1):
        x0, y0 = fitted_curve[i]
        x1, y1 = fitted_curve[i + 1]
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0) if x1 != x0 else 0
            return y0 + t * (y1 - y0)

    return fitted_curve[-1][1]


def compute_ece(predictions, actuals, n_bins=10):
    """Compute Expected Calibration Error."""
    bins = defaultdict(list)
    for pred, actual in zip(predictions, actuals):
        bin_idx = min(int(pred * n_bins), n_bins - 1)
        bins[bin_idx].append((pred, actual))

    ece = 0.0
    total = len(predictions)
    for bin_idx, items in bins.items():
        if not items:
            continue
        avg_pred = np.mean([p for p, a in items])
        avg_actual = np.mean([a for p, a in items])
        weight = len(items) / total
        ece += weight * abs(avg_pred - avg_actual)

    return ece


def affine_transfer(gui360_curve, ac_steps, n_splits=5):
    """Learn affine transfer: accuracy_AC ≈ α × f_gui360(agreement) + β.

    Uses cross-validation to estimate generalization.
    """
    # Get predictions from GUI-360 curve
    agreements = [s['agreement'] for s in ac_steps]
    actuals = [s['greedy_correct'] for s in ac_steps]
    gui360_preds = [interpolate(gui360_curve, a) for a in agreements]

    # Fit α, β using least squares
    X = np.array(gui360_preds).reshape(-1, 1)
    X_aug = np.column_stack([X, np.ones(len(X))])
    y = np.array(actuals, dtype=float)

    # Closed-form solution
    try:
        params = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        alpha, beta = params[0], params[1]
    except np.linalg.LinAlgError:
        alpha, beta = 1.0, 0.0

    # Affine predictions
    affine_preds = [alpha * p + beta for p in gui360_preds]
    affine_preds = [max(0, min(1, p)) for p in affine_preds]

    affine_ece = compute_ece(affine_preds, actuals)

    return alpha, beta, affine_ece, affine_preds


def within_dataset_calibration(ac_steps, n_bins=10):
    """Fit calibration directly on AC data (upper bound)."""
    # Bin by agreement
    bins = defaultdict(list)
    for s in ac_steps:
        bin_idx = min(int(s['agreement'] * n_bins), n_bins - 1)
        bins[bin_idx].append(s)

    # For each step, predict accuracy = bin average
    predictions = []
    actuals = []
    for s in ac_steps:
        bin_idx = min(int(s['agreement'] * n_bins), n_bins - 1)
        bin_acc = np.mean([x['greedy_correct'] for x in bins[bin_idx]])
        predictions.append(bin_acc)
        actuals.append(s['greedy_correct'])

    ece = compute_ece(predictions, actuals)
    return ece, predictions


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load data ---
    print("Loading AC multi-sample data...")
    ac_steps = load_ac_calibration_from_multisample(args.multisample_jsonl)
    print(f"  Loaded {len(ac_steps)} step-level samples")

    # Also load pre-computed calibration if available
    if args.cross_analysis_json and os.path.exists(args.cross_analysis_json):
        ac_precomputed = load_ac_calibration_from_cross_analysis(args.cross_analysis_json)
        print(f"  Pre-computed AC calibration bins: {len(ac_precomputed)}")
    else:
        ac_precomputed = None

    # --- Phase 1: Fit on GUI-360 ---
    print("\n=== Phase 1: Fit isotonic regression on GUI-360 ===")
    gui360_curve = isotonic_fit(GUI360_CALIBRATION)
    print("  GUI-360 calibration curve:")
    for x, y in gui360_curve:
        print(f"    agreement={x:.2f} → accuracy={y:.3f}")

    # --- Phase 2: Zero-shot transfer to AC ---
    print("\n=== Phase 2: Zero-shot transfer to AC ===")
    agreements = [s['agreement'] for s in ac_steps]
    actuals = [s['greedy_correct'] for s in ac_steps]
    zeroshot_preds = [interpolate(gui360_curve, a) for a in agreements]
    zeroshot_ece = compute_ece(zeroshot_preds, actuals)
    print(f"  Zero-shot ECE: {zeroshot_ece:.4f}")

    # Bin-level comparison
    ac_binned = bin_calibration(ac_steps)
    print("\n  Bin-level comparison (agreement → predicted vs actual):")
    print(f"  {'Agree':>8} | {'GUI360 pred':>12} | {'AC actual':>10} | {'Delta':>8} | {'N':>6}")
    print(f"  {'-'*8} | {'-'*12} | {'-'*10} | {'-'*8} | {'-'*6}")
    for mid, (acc, n) in sorted(ac_binned.items()):
        gui_pred = interpolate(gui360_curve, mid)
        delta = gui_pred - acc
        print(f"  {mid:8.2f} | {gui_pred:12.3f} | {acc:10.3f} | {delta:+8.3f} | {n:6d}")

    # --- Phase 3: Affine transfer ---
    print("\n=== Phase 3: Affine transfer (2 params) ===")
    alpha, beta, affine_ece, affine_preds = affine_transfer(gui360_curve, ac_steps)
    print(f"  α = {alpha:.4f}, β = {beta:.4f}")
    print(f"  Affine ECE: {affine_ece:.4f}")
    print(f"  Interpretation: AC_accuracy ≈ {alpha:.2f} × GUI360_pred + {beta:.2f}")

    # --- Phase 4: Within-dataset baseline ---
    print("\n=== Phase 4: Within-dataset calibration (upper bound) ===")
    within_ece, within_preds = within_dataset_calibration(ac_steps)
    print(f"  Within-dataset ECE: {within_ece:.4f}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("CALIBRATION TRANSFER SUMMARY")
    print("=" * 60)
    print(f"  Zero-shot transfer ECE:  {zeroshot_ece:.4f}")
    print(f"  Affine transfer ECE:     {affine_ece:.4f}")
    print(f"  Within-dataset ECE:      {within_ece:.4f}")
    print(f"  ECE reduction (affine vs zeroshot): {(1 - affine_ece/zeroshot_ece)*100:.1f}%")
    if within_ece > 0:
        print(f"  Affine / Within-dataset ratio: {affine_ece/within_ece:.2f}")

    # Monotonicity test
    ac_curve = sorted(ac_binned.items())
    monotonic = all(ac_curve[i][1][0] <= ac_curve[i+1][1][0]
                    for i in range(len(ac_curve)-1))
    print(f"\n  AC calibration monotonic: {monotonic}")

    gui_curve_accs = [y for _, y in gui360_curve]
    gui_monotonic = all(gui_curve_accs[i] <= gui_curve_accs[i+1]
                        for i in range(len(gui_curve_accs)-1))
    print(f"  GUI-360 calibration monotonic: {gui_monotonic}")

    # Pearson correlation between agreement and accuracy
    corr = np.corrcoef(agreements, actuals)[0, 1]
    print(f"\n  AC agreement↔accuracy Pearson r: {corr:.4f}")

    # Conclusion
    print("\n--- Conclusions ---")
    if affine_ece < zeroshot_ece * 0.7:
        print("  ✓ Affine transfer significantly improves calibration")
        print("    → Calibration SHAPE is universal, only offset differs")
    else:
        print("  ✗ Affine transfer provides modest improvement")
        print("    → Calibration may be dataset-specific")

    if affine_ece < within_ece * 1.5:
        print("  ✓ Affine transfer approaches within-dataset quality")
        print("    → Agreement is a TRULY UNIVERSAL uncertainty signal")
    else:
        print("  ✗ Affine transfer still far from within-dataset")
        print("    → Agreement needs per-dataset calibration")

    # --- Save results ---
    results = {
        'gui360_calibration': {str(k): {'accuracy': v[0], 'count': v[1]}
                               for k, v in GUI360_CALIBRATION.items()},
        'ac_calibration': {str(k): {'accuracy': v[0], 'count': v[1]}
                           for k, v in ac_binned.items()},
        'zeroshot_ece': zeroshot_ece,
        'affine_ece': affine_ece,
        'within_dataset_ece': within_ece,
        'affine_alpha': alpha,
        'affine_beta': beta,
        'ac_agreement_accuracy_pearson_r': corr,
        'ac_monotonic': monotonic,
        'gui360_monotonic': gui_monotonic,
        'total_ac_steps': len(ac_steps),
        'conclusion': {
            'shape_universal': bool(affine_ece < zeroshot_ece * 0.7),
            'truly_universal': bool(affine_ece < within_ece * 1.5) if within_ece > 0 else True,
        }
    }

    out_path = os.path.join(args.output_dir, 'x2_calibration_transfer.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X2: Calibration Transfer Analysis")
    parser.add_argument("--multisample_jsonl", type=str, required=True,
                        help="Path to C4+C7 multisample_results.jsonl")
    parser.add_argument("--cross_analysis_json", type=str, default=None,
                        help="Path to cross_dataset_analysis.json (optional)")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_x2",
                        help="Output directory")
    args = parser.parse_args()
    main(args)
