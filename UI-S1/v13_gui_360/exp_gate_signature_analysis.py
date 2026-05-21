"""
Gate Signature Analysis — Offline (no GPU needed).

Uses existing gate_reasoning_path data (968 episodes) to compare
gate trajectories of correct vs incorrect episodes.

Questions:
  1. Do correct episodes have distinct gate signatures?
  2. Can planning-phase gates predict success?
  3. Which layers' gates correlate most with coordinate accuracy?
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict


def load_gate_data(data_dir):
    """Load gate reasoning path data from shards."""
    episodes = []
    for shard_id in range(4):
        path = os.path.join(data_dir, f"shard_{shard_id}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                episodes.append(json.loads(line))
    return episodes


def load_eval_results(eval_path):
    """Load eval results to get correct/incorrect labels."""
    with open(eval_path) as f:
        data = json.load(f)
    # Map episode_id -> first step success
    result_map = {}
    for ep in data['episodes']:
        eid = ep['episode_id']
        if ep['step_results']:
            result_map[eid] = ep['step_results'][0]['success']
        else:
            result_map[eid] = False
    return result_map


def main():
    gate_data_dir = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/outputs/gate_reasoning_path"
    eval_results_path = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/outputs/epoch-3/eval_results_20260425_170836.json"
    output_dir = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/outputs/gate_signature"

    os.makedirs(output_dir, exist_ok=True)

    print("Loading gate data...")
    gate_episodes = load_gate_data(gate_data_dir)
    print(f"  {len(gate_episodes)} episodes with gate data")

    print("Loading eval results...")
    eval_map = load_eval_results(eval_results_path)
    print(f"  {len(eval_map)} episodes with eval results")

    # Match gate data with eval results
    # Data format: {'eid': int, 'goal': str, 'runs': [{'phases': {phase: {layer: {mean, std, n}}}, 'trajectory': {layer: [values]}}]}
    correct_gates = []
    incorrect_gates = []
    layers = ['L10', 'L18', 'L27']

    for ep in gate_episodes:
        eid = ep.get('eid')
        if eid not in eval_map:
            continue

        runs = ep.get('runs', [])
        if not runs:
            continue
        run0 = runs[0]
        phases_data = run0.get('phases', {})

        # Extract phase-averaged gates: {phase: {layer: mean_value}}
        phase_gates = {}
        for phase in ['planning', 'action_type', 'coordinate']:
            phase_gates[phase] = {}
            phase_info = phases_data.get(phase, {})
            for layer in layers:
                layer_info = phase_info.get(layer, {})
                phase_gates[phase][layer] = layer_info.get('mean', 0.5)

        # Also extract trajectory for variance analysis
        trajectory = run0.get('trajectory', {})

        entry = {
            'episode_id': eid,
            'correct': eval_map[eid],
            'phase_gates': phase_gates,
            'trajectory': trajectory,
        }

        if eval_map[eid]:
            correct_gates.append(entry)
        else:
            incorrect_gates.append(entry)

    print(f"\nMatched: {len(correct_gates)} correct, {len(incorrect_gates)} incorrect")

    # ═══════════════════════════════════════════
    # Analysis 1: Correct vs Incorrect gate patterns
    # ═══════════════════════════════════════════
    print("\n" + "=" * 80)
    print("CORRECT vs INCORRECT — Gate Signatures")
    print("=" * 80)

    phases = ['planning', 'action_type', 'coordinate']
    print(f"\n{'Phase':<14} | {'Layer':<5} | {'Correct':>8} | {'Incorrect':>10} | {'Diff':>8} | {'p-value':>8}")
    print("-" * 70)

    significant_findings = []
    for phase in phases:
        for layer in layers:
            correct_vals = [e['phase_gates'].get(phase, {}).get(layer, 0.5) for e in correct_gates]
            incorrect_vals = [e['phase_gates'].get(phase, {}).get(layer, 0.5) for e in incorrect_gates]

            if not correct_vals or not incorrect_vals:
                continue

            mean_c = np.mean(correct_vals)
            mean_i = np.mean(incorrect_vals)
            diff = mean_c - mean_i

            # Simple t-test (normal approximation for large N)
            import math
            std_c = np.std(correct_vals) + 1e-8
            std_i = np.std(incorrect_vals) + 1e-8
            n_c = len(correct_vals)
            n_i = len(incorrect_vals)
            se = np.sqrt(std_c**2/n_c + std_i**2/n_i)
            t_stat = diff / se
            # p-value via error function (no scipy needed)
            p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))

            sig = "*" if p_value < 0.05 else " "
            print(f"  {phase:<12} | {layer:<5} | {mean_c:>7.4f} | {mean_i:>9.4f} | {diff:>+7.4f} | {p_value:>7.4f} {sig}")

            if p_value < 0.05:
                significant_findings.append({
                    'phase': phase, 'layer': layer,
                    'diff': diff, 'p_value': p_value,
                    'mean_correct': mean_c, 'mean_incorrect': mean_i,
                })

    # ═══════════════════════════════════════════
    # Analysis 2: Can planning gates predict success?
    # ═══════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PREDICTIVE POWER — Planning gates as success predictor")
    print("=" * 80)

    all_entries = correct_gates + incorrect_gates
    for layer in layers:
        planning_vals = [e['phase_gates'].get('planning', {}).get(layer, 0.5) for e in all_entries]
        labels = [1 if e['correct'] else 0 for e in all_entries]

        if not planning_vals:
            continue

        # Split by median gate value
        median_gate = np.median(planning_vals)
        high_gate = [labels[i] for i, v in enumerate(planning_vals) if v >= median_gate]
        low_gate = [labels[i] for i, v in enumerate(planning_vals) if v < median_gate]

        acc_high = np.mean(high_gate) * 100 if high_gate else 0
        acc_low = np.mean(low_gate) * 100 if low_gate else 0
        print(f"  {layer} planning gate (median={median_gate:.4f}):")
        print(f"    High gate (>= median): {acc_high:.1f}% correct ({len(high_gate)} eps)")
        print(f"    Low gate  (<  median): {acc_low:.1f}% correct ({len(low_gate)} eps)")
        print(f"    Difference: {acc_high - acc_low:+.1f}%")

    # ═══════════════════════════════════════════
    # Analysis 3: Gate variance and success
    # ═══════════════════════════════════════════
    print("\n" + "=" * 80)
    print("GATE VARIANCE — Within-generation variation vs success")
    print("=" * 80)

    # Use trajectory data for variance analysis
    correct_vars = defaultdict(list)
    incorrect_vars = defaultdict(list)

    all_entries = correct_gates + incorrect_gates
    for entry in all_entries:
        traj = entry.get('trajectory', {})
        for layer in layers:
            vals = traj.get(layer, [])
            if len(vals) < 5:
                continue
            var = np.std(vals)
            if entry['correct']:
                correct_vars[layer].append(var)
            else:
                incorrect_vars[layer].append(var)

    for layer in layers:
        if correct_vars[layer] and incorrect_vars[layer]:
            mean_cv = np.mean(correct_vars[layer])
            mean_iv = np.mean(incorrect_vars[layer])
            print(f"  {layer}: correct_std={mean_cv:.4f}, incorrect_std={mean_iv:.4f}, diff={mean_cv-mean_iv:+.4f}")

    # ═══════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if significant_findings:
        print(f"\n  {len(significant_findings)} significant differences (p<0.05):")
        for f in significant_findings:
            direction = "correct HIGHER" if f['diff'] > 0 else "correct LOWER"
            print(f"    {f['phase']}/{f['layer']}: {direction} by {abs(f['diff']):.4f} (p={f['p_value']:.4f})")
    else:
        print("\n  No significant gate differences between correct and incorrect episodes.")
        print("  Gates are NOT predictive of success → Direction A (Phase-Aware Reward) less promising.")

    # Save
    summary = {
        'n_correct': len(correct_gates),
        'n_incorrect': len(incorrect_gates),
        'significant_findings': significant_findings,
    }
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSaved to {output_dir}/summary.json")


if __name__ == "__main__":
    main()
