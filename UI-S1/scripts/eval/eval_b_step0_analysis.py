#!/usr/bin/env python3
"""
Eval B: Step-0 Performance 专项分析

Question: 82% of divergences happen at step 0.
What is V3's grounding accuracy specifically at step 0 vs later steps?
What is the agreement rate distribution at step 0?

Data sources:
  - Exp 1.1 results (V3 multi-sample, K=1/5/10): 18,265 grounding samples
  - Exp 1.3 results (dual-model): 14,467 action prediction samples
  - Raw data: step_index encoded in sample_id (last number, 1-indexed)

Output: outputs/eval_b/
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def extract_step_index(sample_id: str) -> int:
    """Extract step index (1-indexed) from sample_id.

    Format: {domain}_{category}_{execution_id}_{step_index}
    e.g., 'excel_in_app_excel_1_2_3' -> step 3
    """
    parts = sample_id.rsplit("_", 1)
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return -1


def extract_trajectory_id(sample_id: str) -> str:
    """Extract trajectory ID (everything before last _step).

    e.g., 'excel_in_app_excel_1_2_3' -> 'excel_in_app_excel_1_2'
    """
    parts = sample_id.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else sample_id


def get_total_steps(sample_id: str, traj_steps: dict) -> int:
    """Get total steps in this trajectory."""
    traj_id = extract_trajectory_id(sample_id)
    return traj_steps.get(traj_id, -1)


def analyze_exp1_1():
    """Analyze V3 grounding accuracy by step position from Exp 1.1 data."""
    print("=" * 70)
    print("  Eval B.1: V3 Grounding Accuracy by Step Position (Exp 1.1)")
    print("=" * 70)

    # Build trajectory step counts
    traj_steps = defaultdict(int)

    for K in [1, 5, 10]:
        results_path = PROJECT_ROOT / "outputs" / "exp1_1" / f"results_K{K}.jsonl"
        if not results_path.exists():
            print(f"\n  [K={K}] File not found: {results_path}")
            continue

        results = []
        with open(results_path) as f:
            for line in f:
                results.append(json.loads(line))

        if K == 1:
            # Count trajectory lengths
            for r in results:
                traj_id = extract_trajectory_id(r["sample_id"])
                step_idx = extract_step_index(r["sample_id"])
                traj_steps[traj_id] = max(traj_steps[traj_id], step_idx)

        print(f"\n  --- K={K} ({len(results)} samples) ---")

        # Group by step position
        by_step = defaultdict(list)
        by_step_relative = defaultdict(list)  # normalized position

        for r in results:
            step_idx = extract_step_index(r["sample_id"])
            if step_idx < 1:
                continue
            by_step[step_idx].append(r)

            total = get_total_steps(r["sample_id"], traj_steps)
            if total > 0:
                rel_pos = (step_idx - 1) / max(total - 1, 1)  # 0.0 to 1.0
                bucket = round(rel_pos, 1)
                by_step_relative[bucket].append(r)

        # Step 0 (step_idx=1) vs others
        step1 = by_step.get(1, [])
        later = [r for s, rs in by_step.items() if s > 1 for r in rs]

        step1_acc = np.mean([r["greedy_correct"] for r in step1]) if step1 else 0
        later_acc = np.mean([r["greedy_correct"] for r in later]) if later else 0

        print(f"\n  Step 1 (first step): {len(step1)} samples, greedy accuracy = {step1_acc:.1%}")
        print(f"  Steps 2+:            {len(later)} samples, greedy accuracy = {later_acc:.1%}")
        print(f"  Delta: {step1_acc - later_acc:+.1%}")

        if K >= 5:
            step1_cluster = np.mean([r.get("cluster_center_correct", False) for r in step1]) if step1 else 0
            later_cluster = np.mean([r.get("cluster_center_correct", False) for r in later]) if later else 0
            step1_best = np.mean([r.get("best_of_k_correct", False) for r in step1]) if step1 else 0
            later_best = np.mean([r.get("best_of_k_correct", False) for r in later]) if later else 0

            print(f"\n  Cluster accuracy: Step 1 = {step1_cluster:.1%}, Steps 2+ = {later_cluster:.1%}")
            print(f"  Best-of-K accuracy: Step 1 = {step1_best:.1%}, Steps 2+ = {later_best:.1%}")
            print(f"  Oracle headroom at Step 1: {step1_best - step1_acc:+.1%}")
            print(f"  Oracle headroom at Steps 2+: {later_best - later_acc:+.1%}")

        # Agreement rate distribution by step
        if K >= 5:
            step1_agreements = [r.get("agreement_rate", 0) for r in step1]
            later_agreements = [r.get("agreement_rate", 0) for r in later]

            print(f"\n  Agreement rate distribution:")
            print(f"    Step 1: mean={np.mean(step1_agreements):.3f}, "
                  f"≥0.9={sum(1 for a in step1_agreements if a >= 0.9)/len(step1_agreements):.1%}, "
                  f"<0.5={sum(1 for a in step1_agreements if a < 0.5)/len(step1_agreements):.1%}")
            print(f"    Steps 2+: mean={np.mean(later_agreements):.3f}, "
                  f"≥0.9={sum(1 for a in later_agreements if a >= 0.9)/len(later_agreements):.1%}, "
                  f"<0.5={sum(1 for a in later_agreements if a < 0.5)/len(later_agreements):.1%}")

            # Calibration: agreement vs accuracy at step 1
            print(f"\n  Agreement calibration at Step 1 (K={K}):")
            for lo, hi, label in [(0.9, 1.01, "≥0.9"), (0.5, 0.9, "0.5-0.9"), (0.0, 0.5, "<0.5")]:
                subset = [r for r in step1 if lo <= r.get("agreement_rate", 0) < hi]
                if subset:
                    acc = np.mean([r["greedy_correct"] for r in subset])
                    print(f"      {label}: {len(subset)} samples ({len(subset)/len(step1):.1%}), accuracy = {acc:.1%}")

        # Accuracy by absolute step position
        print(f"\n  Accuracy by step position:")
        for step_idx in sorted(by_step.keys()):
            if step_idx > 15:
                break
            step_results = by_step[step_idx]
            acc = np.mean([r["greedy_correct"] for r in step_results])
            n = len(step_results)
            print(f"    Step {step_idx:>2d}: {n:>5d} samples, accuracy = {acc:.1%}")

    return traj_steps


def analyze_exp1_3():
    """Analyze dual-model coord_match by step position from Exp 1.3 data."""
    print("\n" + "=" * 70)
    print("  Eval B.2: Dual-Model coord_match by Step Position (Exp 1.3)")
    print("=" * 70)

    results_path = PROJECT_ROOT / "outputs" / "exp1_3" / "results.jsonl"
    if not results_path.exists():
        print(f"  File not found: {results_path}")
        return

    results = []
    with open(results_path) as f:
        for line in f:
            results.append(json.loads(line))

    print(f"\n  Total samples: {len(results)}")

    by_step = defaultdict(list)
    for r in results:
        step_idx = extract_step_index(r["sample_id"])
        if step_idx >= 1:
            by_step[step_idx].append(r)

    step1 = by_step.get(1, [])
    later = [r for s, rs in by_step.items() if s > 1 for r in rs]

    for cond, label in [("a", "V2 baseline"), ("b", "V2+V3 coord"), ("c", "Oracle coord")]:
        step1_coord = np.mean([r.get(f"{cond}_coord_match", False) for r in step1]) if step1 else 0
        later_coord = np.mean([r.get(f"{cond}_coord_match", False) for r in later]) if later else 0
        step1_func = np.mean([r.get(f"{cond}_function_match", False) for r in step1]) if step1 else 0
        later_func = np.mean([r.get(f"{cond}_function_match", False) for r in later]) if later else 0

        print(f"\n  [{label}]")
        print(f"    Step 1: func={step1_func:.1%}, coord={step1_coord:.1%} ({len(step1)} samples)")
        print(f"    Steps 2+: func={later_func:.1%}, coord={later_coord:.1%} ({len(later)} samples)")
        print(f"    Delta coord: {step1_coord - later_coord:+.1%}")

    # coord_match by step position for condition B (V2+V3)
    print(f"\n  V2+V3 coord_match by step:")
    for step_idx in sorted(by_step.keys()):
        if step_idx > 12:
            break
        step_results = by_step[step_idx]
        b_coord = np.mean([r.get("b_coord_match", False) for r in step_results])
        c_coord = np.mean([r.get("c_coord_match", False) for r in step_results])
        print(f"    Step {step_idx:>2d}: {len(step_results):>5d} samples, "
              f"V2+V3={b_coord:.1%}, Oracle={c_coord:.1%}")


def analyze_exp1_5():
    """Analyze dual-model eval by step position from Exp 1.5 data."""
    print("\n" + "=" * 70)
    print("  Eval B.3: Exp 1.5 Dual-Model by Step Position")
    print("=" * 70)

    results_path = PROJECT_ROOT / "outputs" / "exp1_5" / "results.jsonl"
    if not results_path.exists():
        print(f"  File not found: {results_path}")
        return

    results = []
    with open(results_path) as f:
        for line in f:
            results.append(json.loads(line))

    by_step = defaultdict(list)
    for r in results:
        step_idx = extract_step_index(r["sample_id"])
        if step_idx >= 1:
            by_step[step_idx].append(r)

    step1 = by_step.get(1, [])
    later = [r for s, rs in by_step.items() if s > 1 for r in rs]

    # V3 agreement rate at step 1 vs later
    step1_agree = [r.get("v3_agreement_rate", 0) for r in step1 if r.get("v3_agreement_rate") is not None]
    later_agree = [r.get("v3_agreement_rate", 0) for r in later if r.get("v3_agreement_rate") is not None]

    print(f"\n  Total: {len(results)} samples")
    print(f"  Step 1: {len(step1)}, Steps 2+: {len(later)}")

    if step1_agree:
        print(f"\n  V3 agreement rate:")
        print(f"    Step 1: mean={np.mean(step1_agree):.3f}, "
              f"≥0.9={sum(1 for a in step1_agree if a >= 0.9)/len(step1_agree):.1%}")
        print(f"    Steps 2+: mean={np.mean(later_agree):.3f}, "
              f"≥0.9={sum(1 for a in later_agree if a >= 0.9)/len(later_agree):.1%}")

    # Coord match by step for all conditions
    for cond, label in [("a", "V2 only"), ("c", "V2+V3 greedy"), ("d", "V2+V3 K=5 cluster")]:
        step1_coord = np.mean([r.get(f"{cond}_coord_match", False) for r in step1]) if step1 else 0
        later_coord = np.mean([r.get(f"{cond}_coord_match", False) for r in later]) if later else 0
        print(f"\n  [{label}] coord_match: Step 1 = {step1_coord:.1%}, Steps 2+ = {later_coord:.1%}, "
              f"delta = {step1_coord - later_coord:+.1%}")


def save_summary(traj_steps):
    """Save summary to JSON."""
    output_dir = PROJECT_ROOT / "outputs" / "eval_b"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count trajectory length distribution
    lengths = list(traj_steps.values())
    length_dist = Counter(lengths)

    summary = {
        "total_trajectories": len(traj_steps),
        "total_steps_in_grounding_eval": sum(lengths),
        "step1_count": len([l for l in lengths if l >= 1]),
        "trajectory_length_distribution": dict(sorted(length_dist.items())),
        "mean_trajectory_length": np.mean(lengths) if lengths else 0,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_dir / 'summary.json'}")


def main():
    traj_steps = analyze_exp1_1()
    analyze_exp1_3()
    analyze_exp1_5()

    if traj_steps:
        save_summary(traj_steps)

    print("\n" + "=" * 70)
    print("  Eval B: CONCLUSIONS")
    print("=" * 70)
    print("""
  Key questions answered:
  1. Is step-0 grounding accuracy lower than other steps?
  2. Does agreement rate differ at step 0?
  3. Is the step_amplifier=3× justified by accuracy differences?

  Decision criteria:
  - step-0 accuracy ≤ 70% → step_amplifier to 5×, step-0 data augmentation
  - step-0 accuracy ≥ 75% → step_amplifier 3× sufficient
    """)


if __name__ == "__main__":
    main()
