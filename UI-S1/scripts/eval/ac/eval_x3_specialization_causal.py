"""X3: Role Specialization Causal Verification.

Computes oracle-fix ceilings for action errors vs grounding errors
from Eval A trajectory results, completing the 2×2 causal matrix.

Key question: If we could oracle-fix one error type at a time,
which gives more TSR improvement? This validates the Error Type
Dominance hypothesis.

All offline — 0 GPU required.
"""

import argparse
import json
import os
from collections import defaultdict


def load_trajectory_results(jsonl_path):
    """Load Eval A trajectory results."""
    results = []
    with open(jsonl_path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def classify_step_error(step):
    """Classify a step's error type.

    Returns one of:
    - 'correct': both type and extract match
    - 'action_error': type_match=False (wrong action type)
    - 'grounding_error': type_match=True but extract_match=False (right type, wrong target)
    """
    if step.get('extract_match', False):
        return 'correct'
    elif not step.get('type_match', False):
        return 'action_error'
    else:
        return 'grounding_error'


def simulate_oracle_fix(results, fix_type):
    """Simulate oracle-fixing one error type and recompute TSR.

    fix_type: 'action' or 'grounding'

    For each trajectory:
    - Walk through steps in order
    - If a step has the target error type, oracle-fix it (treat as correct)
    - If a step has a different error type, still fail
    - Stop on first un-fixed failure
    - Recompute task_success and progress
    """
    fixed_results = []

    for traj in results:
        steps = traj.get('step_results', [])
        num_steps = traj['num_steps']

        correct_steps = 0
        for step in steps:
            error_type = classify_step_error(step)

            if error_type == 'correct':
                correct_steps += 1
            elif fix_type == 'action' and error_type == 'action_error':
                correct_steps += 1  # oracle fix
            elif fix_type == 'grounding' and error_type == 'grounding_error':
                correct_steps += 1  # oracle fix
            else:
                break  # still fails

        # Check if we also need to account for steps beyond what was evaluated
        # In AR, we only evaluate up to first failure, so oracle-fixing may
        # "unlock" additional steps that were never evaluated
        # Conservative: only count steps that were actually evaluated
        # Optimistic: assume remaining steps would be correct if unlocked

        task_success = (correct_steps >= num_steps)

        fixed_results.append({
            'episode_id': traj.get('episode_id'),
            'num_steps': num_steps,
            'task_success': task_success,
            'correct_steps': correct_steps,
            'progress': correct_steps / num_steps if num_steps > 0 else 0,
            'length_bucket': traj.get('length_bucket', 'unknown'),
        })

    return fixed_results


def compute_metrics(results):
    """Compute TSR and average progress."""
    n = len(results)
    if n == 0:
        return {'tsr': 0, 'avg_progress': 0, 'n': 0}

    success = sum(1 for r in results if r['task_success'])
    avg_progress = sum(r['progress'] for r in results) / n

    return {
        'tsr': success / n,
        'success_count': success,
        'avg_progress': avg_progress,
        'n': n,
    }


def compute_metrics_by_bucket(results):
    """Compute metrics grouped by length bucket."""
    buckets = defaultdict(list)
    for r in results:
        buckets[r['length_bucket']].append(r)

    metrics = {}
    for bucket, items in sorted(buckets.items()):
        m = compute_metrics(items)
        metrics[bucket] = m

    return metrics


def analyze_error_distribution(results):
    """Analyze the distribution of error types across all steps."""
    error_counts = defaultdict(int)
    first_error_types = defaultdict(int)
    total_steps = 0

    for traj in results:
        steps = traj.get('step_results', [])
        first_error_found = False
        for step in steps:
            total_steps += 1
            error_type = classify_step_error(step)
            error_counts[error_type] += 1

            if not first_error_found and error_type != 'correct':
                first_error_types[error_type] += 1
                first_error_found = True

    return {
        'total_steps': total_steps,
        'error_counts': dict(error_counts),
        'first_error_types': dict(first_error_types),
    }


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading Eval A trajectory results...")
    results = load_trajectory_results(args.trajectory_jsonl)
    print(f"  Loaded {len(results)} trajectories")

    # --- Baseline metrics ---
    print("\n=== Baseline (No Oracle Fix) ===")
    baseline_metrics = compute_metrics([{
        'task_success': r['task_success'],
        'progress': r.get('final_step_id', 0) / r['num_steps'] if r['num_steps'] > 0 else 0,
        'num_steps': r['num_steps'],
        'length_bucket': r.get('length_bucket', 'unknown'),
    } for r in results])
    print(f"  TSR: {baseline_metrics['tsr']:.4f} ({baseline_metrics['success_count']}/{baseline_metrics['n']})")
    print(f"  Avg Progress: {baseline_metrics['avg_progress']:.4f}")

    # --- Error distribution ---
    print("\n=== Error Distribution ===")
    error_dist = analyze_error_distribution(results)
    total = error_dist['total_steps']
    for etype, count in sorted(error_dist['error_counts'].items()):
        print(f"  {etype}: {count} ({count/total*100:.1f}%)")

    print("\n  First error type distribution:")
    total_first = sum(error_dist['first_error_types'].values())
    for etype, count in sorted(error_dist['first_error_types'].items()):
        print(f"  {etype}: {count} ({count/total_first*100:.1f}%)")

    # --- Oracle fix: Action errors ---
    print("\n=== Oracle Fix: Action Errors ===")
    action_fixed = simulate_oracle_fix(results, 'action')
    action_metrics = compute_metrics(action_fixed)
    action_delta = action_metrics['tsr'] - baseline_metrics['tsr']
    print(f"  TSR: {action_metrics['tsr']:.4f} ({action_metrics['success_count']}/{action_metrics['n']})")
    print(f"  Delta: {action_delta:+.4f} ({action_delta*100:+.2f}pp)")
    print(f"  Avg Progress: {action_metrics['avg_progress']:.4f}")

    # --- Oracle fix: Grounding errors ---
    print("\n=== Oracle Fix: Grounding Errors ===")
    grounding_fixed = simulate_oracle_fix(results, 'grounding')
    grounding_metrics = compute_metrics(grounding_fixed)
    grounding_delta = grounding_metrics['tsr'] - baseline_metrics['tsr']
    print(f"  TSR: {grounding_metrics['tsr']:.4f} ({grounding_metrics['success_count']}/{grounding_metrics['n']})")
    print(f"  Delta: {grounding_delta:+.4f} ({grounding_delta*100:+.2f}pp)")
    print(f"  Avg Progress: {grounding_metrics['avg_progress']:.4f}")

    # --- Oracle fix: Both errors ---
    print("\n=== Oracle Fix: Both Errors (ceiling) ===")
    # If we fix both, every step in the evaluated trajectory becomes correct
    # But AR means we only evaluated up to first failure
    both_fixed_results = []
    for traj in results:
        steps = traj.get('step_results', [])
        num_steps = traj['num_steps']
        # All evaluated steps become correct
        correct_steps = len(steps)
        # But there may be unevaluated steps beyond (conservative: assume they'd be correct)
        task_success = (correct_steps >= num_steps)
        both_fixed_results.append({
            'task_success': task_success,
            'progress': correct_steps / num_steps if num_steps > 0 else 0,
            'num_steps': num_steps,
            'length_bucket': traj.get('length_bucket', 'unknown'),
        })
    both_metrics = compute_metrics(both_fixed_results)
    both_delta = both_metrics['tsr'] - baseline_metrics['tsr']
    print(f"  TSR: {both_metrics['tsr']:.4f} ({both_metrics['success_count']}/{both_metrics['n']})")
    print(f"  Delta: {both_delta:+.4f} ({both_delta*100:+.2f}pp)")

    # --- Per-length-bucket analysis ---
    print("\n=== Per-Length-Bucket Analysis ===")
    bucket_order = ['short(1-3)', 'medium(4-7)', 'long(8-15)', 'vlong(16+)']

    baseline_buckets = compute_metrics_by_bucket([{
        'task_success': r['task_success'],
        'progress': r.get('final_step_id', 0) / r['num_steps'] if r['num_steps'] > 0 else 0,
        'num_steps': r['num_steps'],
        'length_bucket': r.get('length_bucket', 'unknown'),
    } for r in results])

    action_buckets = compute_metrics_by_bucket(action_fixed)
    grounding_buckets = compute_metrics_by_bucket(grounding_fixed)

    print(f"  {'Bucket':>8} | {'Baseline':>10} | {'Fix Action':>12} | {'Fix Grounding':>14} | {'Act Delta':>10} | {'Grd Delta':>10}")
    print(f"  {'-'*8} | {'-'*10} | {'-'*12} | {'-'*14} | {'-'*10} | {'-'*10}")
    for bucket in bucket_order:
        if bucket not in baseline_buckets:
            continue
        b = baseline_buckets[bucket]
        a = action_buckets.get(bucket, {'tsr': 0})
        g = grounding_buckets.get(bucket, {'tsr': 0})
        ad = a['tsr'] - b['tsr']
        gd = g['tsr'] - b['tsr']
        print(f"  {bucket:>8} | {b['tsr']:10.4f} | {a['tsr']:12.4f} | {g['tsr']:14.4f} | {ad:+10.4f} | {gd:+10.4f}")

    # --- Crossover analysis ---
    print("\n=== Error Type Crossover Analysis ===")
    print("  Testing if action errors dominate in short trajs and grounding in long trajs")
    for bucket in bucket_order:
        if bucket not in baseline_buckets:
            continue
        a_delta = action_buckets.get(bucket, {'tsr': 0})['tsr'] - baseline_buckets[bucket]['tsr']
        g_delta = grounding_buckets.get(bucket, {'tsr': 0})['tsr'] - baseline_buckets[bucket]['tsr']
        dominant = 'ACTION' if a_delta > g_delta else 'GROUNDING'
        ratio = a_delta / g_delta if g_delta > 0 else float('inf')
        print(f"  {bucket:>8}: action_ceiling={a_delta*100:+.2f}pp, grounding_ceiling={g_delta*100:+.2f}pp → {dominant} dominant (ratio={ratio:.2f})")

    # --- 2×2 Matrix ---
    print("\n" + "=" * 60)
    print("ROLE SPECIALIZATION 2×2 CAUSAL MATRIX")
    print("=" * 60)
    print(f"""
  AndroidControl Error Structure:
    Action errors (type_match=F):     {error_dist['first_error_types'].get('action_error', 0)} first-errors ({error_dist['first_error_types'].get('action_error', 0)/total_first*100:.1f}%)
    Grounding errors (type_match=T):  {error_dist['first_error_types'].get('grounding_error', 0)} first-errors ({error_dist['first_error_types'].get('grounding_error', 0)/total_first*100:.1f}%)

  Oracle Fix Ceilings:
    Fix action errors → TSR {action_metrics['tsr']:.4f} ({action_delta*100:+.2f}pp)
    Fix grounding errors → TSR {grounding_metrics['tsr']:.4f} ({grounding_delta*100:+.2f}pp)
    Fix both → TSR {both_metrics['tsr']:.4f} ({both_delta*100:+.2f}pp)

  Dominant bottleneck: {'ACTION' if action_delta > grounding_delta else 'GROUNDING'}
  Action/Grounding ceiling ratio: {action_delta/grounding_delta:.2f}x

  ┌──────────────────┬─────────────────────┬─────────────────────┐
  │                  │ Aligned Direction   │ Wrong Direction     │
  ├──────────────────┼─────────────────────┼─────────────────────┤
  │ GUI-360          │ V2+V3 grounding:    │ Critic (D9):        │
  │ (grounding dom.) │ +13.0pp ✓           │ zero-shot useless ✗ │
  │                  │ Observer: +1.34pp ✓ │                     │
  ├──────────────────┼─────────────────────┼─────────────────────┤
  │ AndroidControl   │ M3 router: +2.5pp ✓ │ Observer: -4.0pp ✗  │
  │ (action dom.)    │ U7 Verifier:        │ U1 MV: -3.56pp ✗   │
  │                  │ +0.59pp ✓           │                     │
  ├──────────────────┼─────────────────────┼─────────────────────┤
  │ AC Oracle Fix    │ Action: {action_delta*100:+.2f}pp       │ Grounding: {grounding_delta*100:+.2f}pp   │
  │ (this experiment)│                     │                     │
  └──────────────────┴─────────────────────┴─────────────────────┘
""")

    # --- Save results ---
    summary = {
        'baseline': baseline_metrics,
        'oracle_fix_action': {
            **action_metrics,
            'delta_tsr': action_delta,
        },
        'oracle_fix_grounding': {
            **grounding_metrics,
            'delta_tsr': grounding_delta,
        },
        'oracle_fix_both': {
            **both_metrics,
            'delta_tsr': both_delta,
        },
        'error_distribution': error_dist,
        'per_bucket': {
            'baseline': {k: v for k, v in baseline_buckets.items()},
            'fix_action': {k: v for k, v in action_buckets.items()},
            'fix_grounding': {k: v for k, v in grounding_buckets.items()},
        },
        'dominant_bottleneck': 'action' if action_delta > grounding_delta else 'grounding',
        'action_grounding_ratio': action_delta / grounding_delta if grounding_delta > 0 else float('inf'),
    }

    out_path = os.path.join(args.output_dir, 'x3_specialization_causal.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X3: Role Specialization Causal Verification")
    parser.add_argument("--trajectory_jsonl", type=str, required=True,
                        help="Path to Eval A trajectory_results.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_x3",
                        help="Output directory")
    args = parser.parse_args()
    main(args)
