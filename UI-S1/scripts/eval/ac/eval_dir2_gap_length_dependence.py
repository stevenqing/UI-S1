"""Direction 2: Simulation-Reality Gap Length Dependence.

Tests the prediction that majority vote's effectiveness depends on trajectory length:
- Short trajectories (N<=3): MV may have positive effect (no compounding)
- Long trajectories (N>=5): MV harmful (compounding dominates)
- Crossover point: around N≈3

Uses existing U1 (majority vote) and Eval A (baseline) results.
Also analyzes by action type to find if short-trajectory action types drive gains.

All offline — 0 GPU required.
"""

import argparse
import json
import os
import numpy as np
from collections import defaultdict


def load_jsonl(path):
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def compute_tsr(trajectories):
    if not trajectories:
        return 0, 0
    n_success = sum(1 for t in trajectories if t.get('task_success', False))
    return n_success / len(trajectories), len(trajectories)


def compute_progress(trajectories):
    """Average progress (fraction of correct steps from start)."""
    if not trajectories:
        return 0
    progresses = []
    for t in trajectories:
        steps = t.get('step_results', [])
        if not steps:
            progresses.append(0)
            continue
        correct = 0
        for s in steps:
            if s.get('extract_match', False):
                correct += 1
            else:
                break
        progresses.append(correct / len(steps))
    return np.mean(progresses)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading baseline (Eval A) results...")
    baseline = load_jsonl(args.baseline_jsonl)
    print(f"  Loaded {len(baseline)} baseline trajectories")

    print("Loading U1 (majority vote) results...")
    u1_data = load_jsonl(args.u1_jsonl)
    print(f"  Loaded {len(u1_data)} U1 trajectories")

    # ===================================================================
    # ANALYSIS 1: TSR by exact trajectory length
    # ===================================================================
    print(f"\n{'='*70}")
    print("ANALYSIS 1: TSR BY EXACT TRAJECTORY LENGTH")
    print(f"{'='*70}")

    # Group by num_steps
    bl_by_len = defaultdict(list)
    u1_by_len = defaultdict(list)
    for t in baseline:
        bl_by_len[t['num_steps']].append(t)
    for t in u1_data:
        u1_by_len[t['num_steps']].append(t)

    print(f"\n  {'Length':>6} | {'N':>5} | {'BL TSR':>8} | {'U1 TSR':>8} | {'Delta':>8} | {'Rel Delta':>10} | {'Signal':>8}")
    print(f"  {'-'*6} | {'-'*5} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*10} | {'-'*8}")

    length_results = {}
    crossover_lengths = []

    for length in sorted(set(list(bl_by_len.keys()) + list(u1_by_len.keys()))):
        if length > 15:
            break
        bl_trajs = bl_by_len.get(length, [])
        u1_trajs = u1_by_len.get(length, [])

        if not bl_trajs or not u1_trajs:
            continue

        bl_tsr, bl_n = compute_tsr(bl_trajs)
        u1_tsr, u1_n = compute_tsr(u1_trajs)
        delta = u1_tsr - bl_tsr
        rel_delta = delta / bl_tsr * 100 if bl_tsr > 0 else float('inf') if delta > 0 else float('-inf') if delta < 0 else 0

        signal = "MV+" if delta > 0 else "MV-" if delta < 0 else "EVEN"
        if delta > 0:
            crossover_lengths.append(length)

        length_results[length] = {
            'n_baseline': bl_n,
            'n_u1': u1_n,
            'baseline_tsr': bl_tsr,
            'u1_tsr': u1_tsr,
            'delta': delta,
            'rel_delta': rel_delta if abs(rel_delta) < 1000 else None,
        }

        rel_str = f"{rel_delta:+9.1f}%" if abs(rel_delta) < 1000 else "    N/A"
        print(f"  {length:6d} | {bl_n:5d} | {bl_tsr*100:7.2f}% | {u1_tsr*100:7.2f}% | {delta*100:+7.2f}% | {rel_str} | {signal:>8}")

    # ===================================================================
    # ANALYSIS 2: TSR by length bucket
    # ===================================================================
    print(f"\n{'='*70}")
    print("ANALYSIS 2: TSR BY LENGTH BUCKET")
    print(f"{'='*70}")

    bucket_order = ['short(1-3)', 'medium(4-7)', 'long(8-15)', 'vlong(16+)']

    bl_by_bucket = defaultdict(list)
    u1_by_bucket = defaultdict(list)
    for t in baseline:
        bl_by_bucket[t.get('length_bucket', 'unknown')].append(t)
    for t in u1_data:
        u1_by_bucket[t.get('length_bucket', 'unknown')].append(t)

    print(f"\n  {'Bucket':<15} | {'N':>5} | {'BL TSR':>8} | {'U1 TSR':>8} | {'Delta':>8} | {'BL Prog':>8} | {'U1 Prog':>8}")
    print(f"  {'-'*15} | {'-'*5} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")

    bucket_results = {}
    for bucket in bucket_order:
        bl_trajs = bl_by_bucket.get(bucket, [])
        u1_trajs = u1_by_bucket.get(bucket, [])
        if not bl_trajs or not u1_trajs:
            continue

        bl_tsr, bl_n = compute_tsr(bl_trajs)
        u1_tsr, u1_n = compute_tsr(u1_trajs)
        bl_prog = compute_progress(bl_trajs)
        u1_prog = compute_progress(u1_trajs)
        delta = u1_tsr - bl_tsr

        bucket_results[bucket] = {
            'n': bl_n,
            'baseline_tsr': bl_tsr,
            'u1_tsr': u1_tsr,
            'delta': delta,
            'baseline_progress': float(bl_prog),
            'u1_progress': float(u1_prog),
        }

        print(f"  {bucket:<15} | {bl_n:5d} | {bl_tsr*100:7.2f}% | {u1_tsr*100:7.2f}% | {delta*100:+7.2f}% | {bl_prog*100:7.2f}% | {u1_prog*100:7.2f}%")

    # ===================================================================
    # ANALYSIS 3: Theoretical gap decomposition
    # ===================================================================
    print(f"\n{'='*70}")
    print("ANALYSIS 3: THEORETICAL GAP DECOMPOSITION")
    print(f"{'='*70}")

    # Compute per-step accuracy for baseline and U1
    bl_step_acc = defaultdict(lambda: {'correct': 0, 'total': 0})
    u1_step_acc = defaultdict(lambda: {'correct': 0, 'total': 0})

    for t in baseline:
        for s in t.get('step_results', []):
            sn = s['step_num']
            bl_step_acc[sn]['total'] += 1
            if s.get('extract_match', False):
                bl_step_acc[sn]['correct'] += 1

    for t in u1_data:
        for s in t.get('step_results', []):
            sn = s['step_num']
            u1_step_acc[sn]['total'] += 1
            if s.get('extract_match', False):
                u1_step_acc[sn]['correct'] += 1

    bl_overall = sum(d['correct'] for d in bl_step_acc.values()) / sum(d['total'] for d in bl_step_acc.values())
    u1_overall = sum(d['correct'] for d in u1_step_acc.values()) / sum(d['total'] for d in u1_step_acc.values())

    print(f"\n  Baseline overall step accuracy: {bl_overall:.4f}")
    print(f"  U1 overall step accuracy:      {u1_overall:.4f}")
    print(f"  Delta:                         {u1_overall - bl_overall:+.4f}")

    # Theoretical: for each length N, compute the gap
    print(f"\n  Theoretical compounding gap (p_greedy={bl_overall:.3f}, p_temp={u1_overall:.3f}):")
    print(f"  {'Length':>6} | {'Greedy TSR':>10} | {'Temp TSR':>10} | {'Gap (abs)':>10} | {'Gap (rel)':>10}")
    print(f"  {'-'*6} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")

    for N in [1, 2, 3, 4, 5, 7, 10]:
        greedy_tsr = bl_overall ** N
        temp_tsr = u1_overall ** N
        gap = greedy_tsr - temp_tsr
        rel = greedy_tsr / temp_tsr if temp_tsr > 0 else float('inf')
        print(f"  {N:6d} | {greedy_tsr*100:9.2f}% | {temp_tsr*100:9.2f}% | {gap*100:+9.2f}% | {rel:10.2f}x")

    # ===================================================================
    # ANALYSIS 4: Length=1 deep dive
    # ===================================================================
    print(f"\n{'='*70}")
    print("ANALYSIS 4: LENGTH=1 DEEP DIVE (Single-step tasks)")
    print(f"{'='*70}")

    bl_len1 = bl_by_len.get(1, [])
    u1_len1 = u1_by_len.get(1, [])

    if bl_len1 and u1_len1:
        bl_tsr1, _ = compute_tsr(bl_len1)
        u1_tsr1, _ = compute_tsr(u1_len1)

        print(f"\n  N(baseline)={len(bl_len1)}, N(U1)={len(u1_len1)}")
        print(f"  Baseline TSR (length=1): {bl_tsr1:.4f} ({bl_tsr1*100:.2f}%)")
        print(f"  U1 TSR (length=1):       {u1_tsr1:.4f} ({u1_tsr1*100:.2f}%)")
        print(f"  Delta:                   {(u1_tsr1-bl_tsr1)*100:+.2f}pp")

        # For length=1, TSR = step accuracy. No compounding.
        # Any difference is purely from voting benefit vs temperature degradation.
        if u1_tsr1 > bl_tsr1:
            print(f"\n  ★ MV WINS at length=1!")
            print(f"    → Voting benefit > temperature degradation for single-step tasks")
            print(f"    → Confirms: compounding is the reason MV fails at longer lengths")
        elif u1_tsr1 < bl_tsr1:
            print(f"\n  ✗ MV LOSES even at length=1")
            print(f"    → Temperature degradation dominates even without compounding")
            print(f"    → MV has no length range where it's beneficial")
        else:
            print(f"\n  = MV ties at length=1 — effects cancel out")

        # Per-action-type analysis for length=1
        print(f"\n  --- Per-Action-Type at Length=1 ---")
        bl_by_type = defaultdict(lambda: {'correct': 0, 'total': 0})
        u1_by_type = defaultdict(lambda: {'correct': 0, 'total': 0})

        for t in bl_len1:
            for s in t.get('step_results', []):
                atype = s.get('gt_action_type', 'unknown')
                bl_by_type[atype]['total'] += 1
                if s.get('extract_match', False):
                    bl_by_type[atype]['correct'] += 1

        for t in u1_len1:
            for s in t.get('step_results', []):
                atype = s.get('gt_action_type', 'unknown')
                u1_by_type[atype]['total'] += 1
                if s.get('extract_match', False):
                    u1_by_type[atype]['correct'] += 1

        print(f"  {'Action Type':<15} | {'BL N':>5} | {'BL Acc':>8} | {'U1 N':>5} | {'U1 Acc':>8} | {'Delta':>8}")
        print(f"  {'-'*15} | {'-'*5} | {'-'*8} | {'-'*5} | {'-'*8} | {'-'*8}")

        for atype in sorted(set(list(bl_by_type.keys()) + list(u1_by_type.keys()))):
            bl = bl_by_type[atype]
            u1 = u1_by_type[atype]
            if bl['total'] == 0 and u1['total'] == 0:
                continue
            bl_acc = bl['correct'] / bl['total'] if bl['total'] > 0 else 0
            u1_acc = u1['correct'] / u1['total'] if u1['total'] > 0 else 0
            delta = u1_acc - bl_acc
            print(f"  {atype:<15} | {bl['total']:5d} | {bl_acc*100:7.2f}% | {u1['total']:5d} | {u1_acc*100:7.2f}% | {delta*100:+7.2f}%")
    else:
        print("\n  No length=1 trajectories found.")

    # ===================================================================
    # ANALYSIS 5: Crossover point detection
    # ===================================================================
    print(f"\n{'='*70}")
    print("ANALYSIS 5: CROSSOVER POINT DETECTION")
    print(f"{'='*70}")

    # Find the exact length where MV goes from positive to negative (or vice versa)
    positive_lengths = []
    negative_lengths = []

    for length in sorted(length_results.keys()):
        r = length_results[length]
        if r['delta'] > 0:
            positive_lengths.append(length)
        elif r['delta'] < 0:
            negative_lengths.append(length)

    if positive_lengths:
        print(f"\n  Lengths where MV helps: {positive_lengths}")
    else:
        print(f"\n  MV helps at NO trajectory length")

    if negative_lengths:
        print(f"  Lengths where MV hurts: {negative_lengths}")

    if positive_lengths and negative_lengths:
        crossover = max(positive_lengths)
        print(f"\n  ★ Crossover point: N ≈ {crossover}")
        print(f"    MV beneficial for lengths ≤ {crossover}, harmful for longer")
    elif not positive_lengths:
        print(f"\n  ★ No crossover — MV harmful at all lengths")
        print(f"    Temperature degradation dominates even without compounding")

    # ===================================================================
    # ANALYSIS 6: Relative loss scaling verification
    # ===================================================================
    print(f"\n{'='*70}")
    print("ANALYSIS 6: RELATIVE LOSS SCALING")
    print(f"{'='*70}")

    print(f"\n  Theory predicts: relative loss ∝ (p_greedy/p_temp)^N = {bl_overall/u1_overall:.3f}^N")
    print(f"\n  {'Bucket':<15} | {'BL TSR':>8} | {'U1 TSR':>8} | {'Abs Loss':>9} | {'Rel Loss':>9} | {'Theory Rel':>10}")
    print(f"  {'-'*15} | {'-'*8} | {'-'*8} | {'-'*9} | {'-'*9} | {'-'*10}")

    for bucket in bucket_order:
        r = bucket_results.get(bucket)
        if not r:
            continue
        abs_loss = r['baseline_tsr'] - r['u1_tsr']
        rel_loss = abs_loss / r['baseline_tsr'] * 100 if r['baseline_tsr'] > 0 else 0

        # Average length for this bucket
        trajs = bl_by_bucket.get(bucket, [])
        avg_len = np.mean([t['num_steps'] for t in trajs]) if trajs else 0
        theory_rel = (1 - (u1_overall / bl_overall) ** avg_len) * 100

        print(f"  {bucket:<15} | {r['baseline_tsr']*100:7.2f}% | {r['u1_tsr']*100:7.2f}% | {abs_loss*100:+8.2f}% | {rel_loss:8.1f}% | {theory_rel:9.1f}%")

    # ===================================================================
    # ANALYSIS 7: Per-step accuracy by step position (BL vs U1)
    # ===================================================================
    print(f"\n{'='*70}")
    print("ANALYSIS 7: PER-STEP ACCURACY COMPARISON (BL vs U1)")
    print(f"{'='*70}")

    print(f"\n  {'Step':>5} | {'BL N':>6} | {'BL Acc':>8} | {'U1 N':>6} | {'U1 Acc':>8} | {'Delta':>8}")
    print(f"  {'-'*5} | {'-'*6} | {'-'*8} | {'-'*6} | {'-'*8} | {'-'*8}")

    for sn in sorted(set(list(bl_step_acc.keys()) + list(u1_step_acc.keys()))):
        if sn > 10:
            break
        bl = bl_step_acc[sn]
        u1 = u1_step_acc[sn]
        if bl['total'] == 0 and u1['total'] == 0:
            continue
        bl_acc = bl['correct'] / bl['total'] if bl['total'] > 0 else 0
        u1_acc = u1['correct'] / u1['total'] if u1['total'] > 0 else 0
        delta = u1_acc - bl_acc
        print(f"  {sn:5d} | {bl['total']:6d} | {bl_acc*100:7.2f}% | {u1['total']:6d} | {u1_acc*100:7.2f}% | {delta*100:+7.2f}%")

    # ===================================================================
    # Save results
    # ===================================================================
    results = {
        'by_length': {str(k): v for k, v in length_results.items()},
        'by_bucket': bucket_results,
        'baseline_overall_step_acc': bl_overall,
        'u1_overall_step_acc': u1_overall,
        'positive_lengths': positive_lengths,
        'negative_lengths': negative_lengths,
        'crossover': max(positive_lengths) if positive_lengths else None,
    }

    out_path = os.path.join(args.output_dir, 'dir2_gap_length_dependence.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direction 2: Simulation-Reality Gap Length Dependence")
    parser.add_argument("--baseline_jsonl", type=str, required=True,
                        help="Path to Eval A trajectory_results.jsonl")
    parser.add_argument("--u1_jsonl", type=str, required=True,
                        help="Path to U1 majority_vote_results.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_dir2",
                        help="Output directory")
    args = parser.parse_args()
    main(args)
