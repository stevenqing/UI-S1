"""X4: Bottleneck Step Identification.

Identifies trajectory "bottleneck steps" where agreement drops sharply,
and analyzes their correlation with actual errors.

Connects to hierarchical RL theory: bottleneck states correspond to
graph Laplacian spectral gaps / covering option boundaries.

All offline — 0 GPU required.
"""

import argparse
import json
import os
import numpy as np
from collections import Counter, defaultdict


def load_multisample_trajectories(jsonl_path):
    """Load per-episode multi-sample data, computing per-step agreement."""
    trajectories = []
    with open(jsonl_path) as f:
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

                type_counter = Counter(action_types)
                agreement = type_counter.most_common(1)[0][1] / K

                greedy_correct = int(samples[0].get('extract_match', False))
                oracle_correct = int(any(s.get('extract_match', False) for s in samples))

                step_data.append({
                    'step_num': step.get('step_num', 0),
                    'agreement': agreement,
                    'greedy_correct': greedy_correct,
                    'oracle_correct': oracle_correct,
                    'gt_action_type': step.get('gt_action_type', 'unknown'),
                    'n_samples': K,
                })

            if len(step_data) >= 2:  # need at least 2 steps for gradient
                trajectories.append({
                    'episode_id': ep.get('episode_id'),
                    'num_steps': ep.get('num_steps', len(step_data)),
                    'steps': step_data,
                })

    return trajectories


def identify_bottlenecks(trajectory, threshold=-0.15):
    """Identify bottleneck steps where agreement drops sharply.

    Returns list of (step_num, agreement_drop) for bottleneck steps.
    """
    steps = trajectory['steps']
    bottlenecks = []

    for i in range(1, len(steps)):
        delta = steps[i]['agreement'] - steps[i-1]['agreement']
        if delta < threshold:
            bottlenecks.append({
                'step_num': steps[i]['step_num'],
                'delta_agreement': delta,
                'agreement_before': steps[i-1]['agreement'],
                'agreement_after': steps[i]['agreement'],
                'greedy_correct': steps[i]['greedy_correct'],
                'gt_action_type': steps[i]['gt_action_type'],
            })

    return bottlenecks


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading multi-sample trajectory data...")
    trajs = load_multisample_trajectories(args.multisample_jsonl)
    print(f"  Loaded {len(trajs)} trajectories with ≥2 steps")

    total_steps = sum(len(t['steps']) for t in trajs)
    print(f"  Total steps: {total_steps}")

    # --- Per-step agreement profile ---
    print(f"\n=== Per-Step Agreement Profile ===")
    by_step = defaultdict(list)
    for t in trajs:
        for s in t['steps']:
            by_step[s['step_num']].append(s)

    print(f"  {'Step':>5} | {'N':>5} | {'Mean Agree':>11} | {'Greedy Acc':>11} | {'Oracle Acc':>11}")
    for step_num in sorted(by_step.keys())[:15]:
        items = by_step[step_num]
        mean_agree = np.mean([s['agreement'] for s in items])
        greedy_acc = np.mean([s['greedy_correct'] for s in items])
        oracle_acc = np.mean([s['oracle_correct'] for s in items])
        print(f"  {step_num:5d} | {len(items):5d} | {mean_agree:11.4f} | {greedy_acc:11.4f} | {oracle_acc:11.4f}")

    # --- Identify bottlenecks ---
    print(f"\n=== Bottleneck Identification (threshold={args.threshold}) ===")
    all_bottlenecks = []
    trajs_with_bottlenecks = 0

    for t in trajs:
        bns = identify_bottlenecks(t, threshold=args.threshold)
        all_bottlenecks.extend(bns)
        if bns:
            trajs_with_bottlenecks += 1

    print(f"  Total bottleneck steps: {len(all_bottlenecks)}")
    print(f"  Trajectories with bottlenecks: {trajs_with_bottlenecks}/{len(trajs)} ({trajs_with_bottlenecks/len(trajs)*100:.1f}%)")
    if all_bottlenecks:
        avg_drop = np.mean([b['delta_agreement'] for b in all_bottlenecks])
        print(f"  Average agreement drop at bottleneck: {avg_drop:.4f}")

    # --- Bottleneck position distribution ---
    print(f"\n=== Bottleneck Step Position Distribution ===")
    bn_step_counts = Counter(b['step_num'] for b in all_bottlenecks)
    for step_num in sorted(bn_step_counts.keys())[:15]:
        total_at_step = len(by_step.get(step_num, []))
        bn_count = bn_step_counts[step_num]
        rate = bn_count / total_at_step if total_at_step > 0 else 0
        print(f"  Step {step_num:2d}: {bn_count:4d} bottlenecks / {total_at_step:5d} total = {rate:.3f}")

    # --- Bottleneck vs Error correlation ---
    print(f"\n=== Bottleneck ↔ Error Correlation ===")
    bn_correct = sum(1 for b in all_bottlenecks if b['greedy_correct'])
    bn_wrong = len(all_bottlenecks) - bn_correct
    if all_bottlenecks:
        bn_error_rate = bn_wrong / len(all_bottlenecks)
    else:
        bn_error_rate = 0

    # Non-bottleneck steps
    non_bn_steps = []
    bn_step_set = set()
    for t in trajs:
        bns = identify_bottlenecks(t, threshold=args.threshold)
        bn_step_nums = set(b['step_num'] for b in bns)
        for s in t['steps']:
            if s['step_num'] not in bn_step_nums:
                non_bn_steps.append(s)

    non_bn_error_rate = 1.0 - np.mean([s['greedy_correct'] for s in non_bn_steps]) if non_bn_steps else 0

    print(f"  Bottleneck steps: error rate = {bn_error_rate:.4f} ({bn_wrong}/{len(all_bottlenecks)})")
    print(f"  Non-bottleneck steps: error rate = {non_bn_error_rate:.4f}")
    print(f"  Error rate ratio: {bn_error_rate/non_bn_error_rate:.2f}x" if non_bn_error_rate > 0 else "")

    # --- Bottleneck action type ---
    print(f"\n=== Bottleneck Action Type Distribution ===")
    bn_types = Counter(b['gt_action_type'] for b in all_bottlenecks)
    all_types = Counter(s['gt_action_type'] for t in trajs for s in t['steps'])
    print(f"  {'Type':>15} | {'BN Count':>9} | {'BN%':>6} | {'Overall%':>9} | {'Ratio':>6}")
    for atype, count in bn_types.most_common(10):
        bn_pct = count / len(all_bottlenecks) * 100 if all_bottlenecks else 0
        overall_pct = all_types[atype] / total_steps * 100
        ratio = (bn_pct / overall_pct) if overall_pct > 0 else 0
        print(f"  {atype:>15} | {count:9d} | {bn_pct:5.1f}% | {overall_pct:8.1f}% | {ratio:5.2f}x")

    # --- Recovery after bottleneck ---
    print(f"\n=== Recovery After Bottleneck ===")
    recovery_stats = {'recovered': 0, 'failed': 0, 'no_next': 0}

    for t in trajs:
        bns = identify_bottlenecks(t, threshold=args.threshold)
        for bn in bns:
            bn_step = bn['step_num']
            # Find next step after bottleneck
            next_step = None
            for s in t['steps']:
                if s['step_num'] == bn_step + 1:
                    next_step = s
                    break

            if next_step is None:
                recovery_stats['no_next'] += 1
            elif next_step['greedy_correct']:
                recovery_stats['recovered'] += 1
            else:
                recovery_stats['failed'] += 1

    total_with_next = recovery_stats['recovered'] + recovery_stats['failed']
    if total_with_next > 0:
        recovery_rate = recovery_stats['recovered'] / total_with_next
        print(f"  Recovery rate (correct after bottleneck): {recovery_rate:.4f}")
        print(f"    Recovered: {recovery_stats['recovered']}")
        print(f"    Failed: {recovery_stats['failed']}")
        print(f"    No next step: {recovery_stats['no_next']}")
    else:
        print(f"  No bottleneck→next step pairs found")

    # --- Agreement gradient statistics ---
    print(f"\n=== Agreement Gradient Distribution ===")
    all_gradients = []
    for t in trajs:
        for i in range(1, len(t['steps'])):
            delta = t['steps'][i]['agreement'] - t['steps'][i-1]['agreement']
            all_gradients.append(delta)

    if all_gradients:
        gradients = np.array(all_gradients)
        print(f"  Mean gradient: {gradients.mean():.4f}")
        print(f"  Std gradient: {gradients.std():.4f}")
        print(f"  Min gradient: {gradients.min():.4f}")
        print(f"  Max gradient: {gradients.max():.4f}")
        pctiles = [5, 10, 25, 50, 75, 90, 95]
        for p in pctiles:
            print(f"  P{p}: {np.percentile(gradients, p):.4f}")

    # --- Multi-threshold analysis ---
    print(f"\n=== Multi-Threshold Bottleneck Analysis ===")
    print(f"  {'Threshold':>10} | {'N BN':>6} | {'BN Error%':>10} | {'NonBN Error%':>13} | {'Ratio':>6}")
    for thresh in [-0.05, -0.10, -0.15, -0.20, -0.25, -0.30]:
        thresh_bns = []
        for t in trajs:
            thresh_bns.extend(identify_bottlenecks(t, threshold=thresh))

        if not thresh_bns:
            continue

        bn_err = 1.0 - np.mean([b['greedy_correct'] for b in thresh_bns])

        # Non-bottleneck for this threshold
        thresh_bn_set = set()
        for t in trajs:
            for b in identify_bottlenecks(t, threshold=thresh):
                thresh_bn_set.add((t['episode_id'], b['step_num']))

        non_bn = [s for t in trajs for s in t['steps']
                  if (t['episode_id'], s['step_num']) not in thresh_bn_set]
        non_bn_err = 1.0 - np.mean([s['greedy_correct'] for s in non_bn]) if non_bn else 0

        ratio = bn_err / non_bn_err if non_bn_err > 0 else float('inf')
        print(f"  {thresh:10.2f} | {len(thresh_bns):6d} | {bn_err*100:9.1f}% | {non_bn_err*100:12.1f}% | {ratio:5.2f}x")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"BOTTLENECK IDENTIFICATION SUMMARY")
    print(f"{'='*60}")
    if all_bottlenecks:
        print(f"  Bottleneck rate: {len(all_bottlenecks)/total_steps*100:.1f}% of steps (threshold={args.threshold})")
        print(f"  Bottleneck error rate: {bn_error_rate*100:.1f}% vs non-bottleneck: {non_bn_error_rate*100:.1f}%")
        print(f"  Error concentration: {bn_error_rate/non_bn_error_rate:.2f}x" if non_bn_error_rate > 0 else "")
        if total_with_next > 0:
            print(f"  Recovery rate after bottleneck: {recovery_rate*100:.1f}%")
        print(f"\n  Implication for compute allocation:")
        print(f"    → Increase K at bottleneck steps (agreement drop > {args.threshold})")
        print(f"    → Keep K=1 at non-bottleneck steps (save compute)")

    # --- Save results ---
    results = {
        'total_trajectories': len(trajs),
        'total_steps': total_steps,
        'threshold': args.threshold,
        'total_bottlenecks': len(all_bottlenecks),
        'trajs_with_bottlenecks': trajs_with_bottlenecks,
        'bottleneck_error_rate': bn_error_rate,
        'non_bottleneck_error_rate': non_bn_error_rate,
        'error_concentration_ratio': bn_error_rate / non_bn_error_rate if non_bn_error_rate > 0 else None,
        'recovery_stats': recovery_stats,
        'recovery_rate': recovery_stats['recovered'] / total_with_next if total_with_next > 0 else None,
        'agreement_gradient': {
            'mean': float(np.mean(all_gradients)) if all_gradients else 0,
            'std': float(np.std(all_gradients)) if all_gradients else 0,
        },
        'bottleneck_action_types': dict(bn_types),
        'bottleneck_by_step': dict(bn_step_counts),
    }

    out_path = os.path.join(args.output_dir, 'x4_bottleneck_identification.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X4: Bottleneck Step Identification")
    parser.add_argument("--multisample_jsonl", type=str, required=True,
                        help="Path to C4+C7 multisample_results.jsonl")
    parser.add_argument("--threshold", type=float, default=-0.15,
                        help="Agreement drop threshold for bottleneck detection")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_x4",
                        help="Output directory")
    args = parser.parse_args()
    main(args)
