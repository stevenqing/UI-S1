"""Q3: Signal Degradation Analysis — GUI-360 Version

Adapts AC Q3 analysis to GUI-360 dataset.
Key differences from AC:
  - Oracle trajectory results from baseline.json (matched by episode_id + step_num)
  - Probe data from probe_gui360_extract.py output
  - ~16 action types vs AC 7
  - Baseline output accuracy: ~46.8% (vs AC ~57%)

Two phases:
  Phase A: Oracle failure probe analysis (match probe_labels.json to baseline.json)
  Phase B: Gap decomposition table (oracle vs baseline vs cross-apply)

Usage:
  python analysis_q3_signal_degradation_gui360.py --phase A    # immediate
  python analysis_q3_signal_degradation_gui360.py --phase B    # after SLURM
  python analysis_q3_signal_degradation_gui360.py --phase AB   # both
"""

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    def _default(obj):
        if hasattr(obj, 'item'): return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_default)


def load_eval_results_indexed(path):
    """Load GUI-360 eval results indexed by (episode_id, step_num)."""
    with open(path) as f:
        data = json.load(f)
    results = {}
    for r in data['results']:
        key = (r['episode_id'], r['step_num'])
        results[key] = r
    return results, data


def phase_a(args):
    """Phase A: Oracle failure probe analysis using probe data + eval results."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import cross_val_predict

    oracle_dir = args.oracle_dir
    eval_path = args.eval_results

    print("=== Phase A: Oracle Failure Probe Analysis (GUI-360) ===")

    # Load probe labels (from oracle extraction)
    labels_path = os.path.join(oracle_dir, 'probe_labels.json')
    with open(labels_path) as f:
        probe_labels = json.load(f)
    print(f"Probe labels: {len(probe_labels)} samples")

    # Load eval results (baseline.json as the output reference)
    eval_steps, eval_meta = load_eval_results_indexed(eval_path)
    print(f"Eval steps: {len(eval_steps)} (type_acc={eval_meta['type_acc']:.3f})")

    # Match probe samples to eval results
    matched = []
    unmatched = 0
    for i, pl in enumerate(probe_labels):
        key = (pl['episode_id'], pl['step_num'])
        if key in eval_steps:
            sr = eval_steps[key]
            matched.append({
                'idx': i,
                'episode_id': pl['episode_id'],
                'step_num': pl['step_num'],
                'gt_action_type': pl['action_type'],
                'is_boundary': pl['is_boundary'],
                'type_match': sr.get('type_match', False),
                'extract_match': sr.get('extract_match', False),
                'pred_type': sr.get('pred_type', 'unknown'),
            })
        else:
            unmatched += 1

    print(f"Matched: {len(matched)}, Unmatched: {unmatched}")

    # Use cross-validated predictions
    probe_layers = [0, 7, 14, 21, 27]
    action_types = [pl['action_type'] for pl in probe_labels]
    le = LabelEncoder()
    le.fit(action_types)
    y_true = le.transform(action_types)

    results_by_layer = {}

    for layer_idx in probe_layers:
        layer_key = f'layer_{layer_idx}'
        fpath = os.path.join(oracle_dir, f'{layer_key}.npy')
        if not os.path.exists(fpath):
            print(f"Skipping {layer_key}")
            continue

        X = np.load(fpath)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lr = LogisticRegression(max_iter=1000, C=1.0)
        y_pred = cross_val_predict(lr, X_scaled, y_true, cv=5)

        # Classify failure types for matched samples
        decoding_failures = 0
        representation_failures = 0
        output_correct = 0
        both_correct = 0
        probe_wrong_output_correct = 0

        boundary_decoding = 0
        boundary_representation = 0
        boundary_total_failures = 0

        for m in matched:
            idx = m['idx']
            probe_correct = (y_pred[idx] == y_true[idx])
            output_type_correct = m['type_match']

            if output_type_correct:
                output_correct += 1
                if probe_correct:
                    both_correct += 1
                else:
                    probe_wrong_output_correct += 1
            else:
                if probe_correct:
                    decoding_failures += 1
                    if m['is_boundary']:
                        boundary_decoding += 1
                else:
                    representation_failures += 1
                    if m['is_boundary']:
                        boundary_representation += 1
                if m['is_boundary']:
                    boundary_total_failures += 1

        total_failures = decoding_failures + representation_failures
        total = len(matched)

        results_by_layer[layer_key] = {
            'total_matched': total,
            'output_correct': output_correct,
            'output_failure': total_failures,
            'decoding_failures': decoding_failures,
            'representation_failures': representation_failures,
            'both_correct': both_correct,
            'probe_wrong_output_correct': probe_wrong_output_correct,
            'output_failure_rate': total_failures / total if total else 0,
            'decoding_failure_frac': decoding_failures / total_failures if total_failures else 0,
            'representation_failure_frac': representation_failures / total_failures if total_failures else 0,
            'probe_cv_acc': float(np.mean(y_pred == y_true)),
            'boundary_failures': {
                'total': boundary_total_failures,
                'decoding': boundary_decoding,
                'representation': boundary_representation,
                'decoding_frac': boundary_decoding / boundary_total_failures if boundary_total_failures else 0,
            },
        }

        print(f"\n{layer_key}:")
        print(f"  Probe CV acc: {np.mean(y_pred == y_true):.4f}")
        print(f"  Output failures: {total_failures}/{total} ({total_failures/total*100:.1f}%)")
        if total_failures:
            print(f"  Decoding failures (probe✓ output✗): {decoding_failures} ({decoding_failures/total_failures*100:.1f}%)")
            print(f"  Representation failures (probe✗ output✗): {representation_failures} ({representation_failures/total_failures*100:.1f}%)")

    return {'phase_a': results_by_layer}


def phase_b(args):
    """Phase B: Gap decomposition + boundary signal analysis."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    oracle_dir = args.oracle_dir
    baseline_dir = args.baseline_dir

    print("=== Phase B: Gap Decomposition (GUI-360) ===")

    # Check if baseline data exists
    baseline_labels_path = os.path.join(baseline_dir, 'baseline_labels.json')
    if not os.path.exists(baseline_labels_path):
        print(f"ERROR: Baseline data not found at {baseline_labels_path}")
        print("Run probe_gui360_extract.slurm first")
        return {}

    # Load probe results
    oracle_probe_path = os.path.join(oracle_dir, 'probe_results.json')
    baseline_probe_path = os.path.join(baseline_dir, 'probe_results.json')
    cross_apply_path = os.path.join(baseline_dir, 'cross_apply_results.json')

    with open(oracle_probe_path) as f:
        oracle_probe = json.load(f)
    with open(baseline_probe_path) as f:
        baseline_probe = json.load(f)

    cross_apply = {}
    if os.path.exists(cross_apply_path):
        with open(cross_apply_path) as f:
            cross_apply = json.load(f)

    # Gap decomposition table
    # GUI-360 baseline output accuracy = 46.8%
    baseline_output_acc = 0.468

    print("\nGap Decomposition Table:")
    print(f"{'Layer':>8} | {'Oracle Probe (A)':>16} | {'Baseline Probe (B)':>18} | {'Cross-Apply (C)':>15} | {'Repr Gap (A-B)':>14} | {'Decoding Gap':>12}")
    print("-" * 100)

    gap_table = {}
    probe_layers = [0, 7, 14, 21, 27]
    for layer_idx in probe_layers:
        layer_key = f'layer_{layer_idx}'
        a = oracle_probe.get(layer_key, {}).get('action_type_acc', 0)
        b = baseline_probe.get(layer_key, {}).get('action_type_acc', 0)
        c = cross_apply.get(layer_key, {}).get('action_type_cross_apply', 0)
        repr_gap = a - b
        decoding_gap = b - baseline_output_acc

        gap_table[layer_key] = {
            'oracle_probe_acc': float(a),
            'baseline_probe_acc': float(b),
            'cross_apply_acc': float(c),
            'repr_gap': float(repr_gap),
            'decoding_gap': float(decoding_gap),
            'baseline_output_acc': baseline_output_acc,
        }

        print(f"{layer_idx:>8} | {a:>16.4f} | {b:>18.4f} | {c:>15.4f} | {repr_gap:>14.4f} | {decoding_gap:>12.4f}")

    # --- Boundary signal analysis ---
    print("\n=== Boundary Signal Analysis ===")

    with open(os.path.join(baseline_dir, 'baseline_labels.json')) as f:
        baseline_labels = json.load(f)

    boundary_indices = [i for i, l in enumerate(baseline_labels) if l['is_boundary']]
    print(f"Boundary samples in baseline: {len(boundary_indices)}/{len(baseline_labels)}")

    action_types_all = [l['action_type'] for l in baseline_labels]
    le = LabelEncoder()
    le.fit(action_types_all)
    y_true_all = le.transform(action_types_all)

    boundary_signal = {}
    for layer_idx in probe_layers:
        layer_key = f'layer_{layer_idx}'
        fpath = os.path.join(baseline_dir, f'{layer_key}.npy')
        if not os.path.exists(fpath):
            continue

        X = np.load(fpath)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(X_scaled, y_true_all)
        y_pred_all = lr.predict(X_scaled)

        predicts_new = 0
        predicts_old = 0
        predicts_other = 0
        total_boundary = 0

        for idx in boundary_indices:
            label = baseline_labels[idx]
            prev_idx = idx - 1
            if prev_idx < 0 or baseline_labels[prev_idx]['episode_id'] != label['episode_id']:
                continue

            prev_type = baseline_labels[prev_idx]['action_type']
            gt_type = label['action_type']
            pred_label = le.inverse_transform([y_pred_all[idx]])[0]

            total_boundary += 1
            if pred_label == gt_type:
                predicts_new += 1
            elif pred_label == prev_type:
                predicts_old += 1
            else:
                predicts_other += 1

        boundary_signal[layer_key] = {
            'total_boundary': total_boundary,
            'predicts_new_type': predicts_new,
            'predicts_old_type': predicts_old,
            'predicts_other': predicts_other,
            'new_rate': predicts_new / total_boundary if total_boundary else 0,
            'old_rate': predicts_old / total_boundary if total_boundary else 0,
            'other_rate': predicts_other / total_boundary if total_boundary else 0,
        }

        print(f"  {layer_key}: new={predicts_new} ({predicts_new/total_boundary*100:.1f}%), "
              f"old={predicts_old} ({predicts_old/total_boundary*100:.1f}%), "
              f"other={predicts_other} ({predicts_other/total_boundary*100:.1f}%), "
              f"n={total_boundary}")

    return {
        'phase_b': {
            'gap_decomposition': gap_table,
            'boundary_signal': boundary_signal,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Q3: Signal Degradation — GUI-360")
    parser.add_argument("--phase", type=str, required=True, choices=['A', 'B', 'AB'],
                        help="Phase A (immediate), B (after SLURM), or AB (both)")
    parser.add_argument("--oracle_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'probe_hidden_states_gui360', 'Qwen2.5-VL-7B'))
    parser.add_argument("--baseline_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'probe_hidden_states_gui360_baseline', 'Qwen2.5-VL-7B'))
    parser.add_argument("--eval_results", type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'gui360_eval_results', 'baseline.json'))
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, 'outputs', 'analysis_q3_signal_degradation_gui360'))
    args = parser.parse_args()

    results = {}
    output_path = os.path.join(args.output_dir, 'q3_results.json')

    # Load existing results if running phase B after A
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)

    if 'A' in args.phase:
        phase_a_results = phase_a(args)
        results.update(phase_a_results)

    if 'B' in args.phase:
        phase_b_results = phase_b(args)
        results.update(phase_b_results)

    # Summary
    if 'phase_a' in results:
        layer_21 = results['phase_a'].get('layer_21', {})
        if layer_21:
            results['summary_phase_a'] = (
                f"GUI-360 Layer 21: {layer_21.get('decoding_failure_frac', 0)*100:.1f}% of oracle failures "
                f"have correct internal signal (decoding failure), "
                f"{layer_21.get('representation_failure_frac', 0)*100:.1f}% lack it (representation failure). "
                f"Probe CV acc: {layer_21.get('probe_cv_acc', 0)*100:.1f}%."
            )

    if 'phase_b' in results:
        gap = results['phase_b'].get('gap_decomposition', {})
        l21 = gap.get('layer_21', {})
        if l21:
            results['summary_phase_b'] = (
                f"GUI-360 Layer 21: oracle_probe={l21['oracle_probe_acc']:.3f}, "
                f"baseline_probe={l21['baseline_probe_acc']:.3f}, "
                f"repr_gap={l21['repr_gap']:.3f}, "
                f"decoding_gap={l21['decoding_gap']:.3f}."
            )

    save_json(results, output_path)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
