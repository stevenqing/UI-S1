#!/usr/bin/env python3
"""
Experiment 0.4: Per-Domain FCV (Future Crossing Value) Curves

Compute FCV curves separately for excel, ppt, word domains to determine
if bottleneck patterns are consistent across domains or domain-specific.

FCV at step i of a success trajectory = probability that a fail trajectory
diverges at step i (given they share the same task/request).

This is a pure data analysis experiment (no model inference needed).

Success criteria: Determine if unified FCV or per-domain FCV is needed

Usage:
    python scripts/exp0/exp0_4_domain_fcv.py --max_pairs 5000
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.exp0.data_utils import (
    FAIL_DIR,
    SUCCESS_TEST_DIR,
    SUCCESS_TRAIN_DIR,
    find_divergence_step,
    load_paired_trajectories,
    load_trajectory,
    normalize_action,
    scan_trajectory_files,
)


def compute_fcv_curves(pairs: list[dict], max_steps: int = 30) -> dict:
    """
    Compute FCV curves from paired trajectories.

    FCV(step_i) = fraction of pairs where divergence occurs at step i
                  among pairs that haven't diverged before step i.

    This is essentially the hazard function of divergence.

    Returns:
        {
            'all': {step_idx: fcv_value},
            'excel': {step_idx: fcv_value},
            'word': {step_idx: fcv_value},
            'ppt': {step_idx: fcv_value},
            'raw_counts': {domain: {step_idx: (diverged, at_risk)}},
        }
    """
    # Group by domain
    domain_pairs = defaultdict(list)
    for pair in pairs:
        domain_pairs[pair["domain"]].append(pair)
        domain_pairs["all"].append(pair)

    result = {}
    raw_counts = {}

    for domain, dpairs in domain_pairs.items():
        # Compute divergence step for each pair
        div_steps = []
        for pair in dpairs:
            div = find_divergence_step(pair["success_steps"], pair["fail_steps"])
            div_steps.append(div)

        # Compute FCV as hazard function
        # At each step i, FCV(i) = n_diverged_at_i / n_at_risk_at_i
        fcv = {}
        counts = {}
        total = len(div_steps)

        for step in range(max_steps):
            at_risk = sum(1 for d in div_steps if d >= step)
            diverged_here = sum(1 for d in div_steps if d == step)

            if at_risk > 0:
                fcv[step] = diverged_here / at_risk
            else:
                fcv[step] = 0.0

            counts[step] = (diverged_here, at_risk)

        result[domain] = fcv
        raw_counts[domain] = counts

    result["raw_counts"] = raw_counts
    return result


def compute_cumulative_fcv(pairs: list[dict], max_steps: int = 30) -> dict:
    """
    Compute cumulative FCV: what fraction of divergences have happened by step i.

    CDF(i) = P(divergence ≤ i)
    """
    domain_pairs = defaultdict(list)
    for pair in pairs:
        domain_pairs[pair["domain"]].append(pair)
        domain_pairs["all"].append(pair)

    result = {}
    for domain, dpairs in domain_pairs.items():
        div_steps = []
        for pair in dpairs:
            div = find_divergence_step(pair["success_steps"], pair["fail_steps"])
            div_steps.append(div)

        total = len(div_steps)
        cdf = {}
        for step in range(max_steps):
            cdf[step] = sum(1 for d in div_steps if d <= step) / max(total, 1)
        result[domain] = cdf

    return result


def analyze_bottleneck_steps(pairs: list[dict]) -> dict:
    """
    Identify specific bottleneck action types at divergence points.
    """
    domain_analysis = defaultdict(lambda: defaultdict(int))

    for pair in pairs:
        div = find_divergence_step(pair["success_steps"], pair["fail_steps"])
        domain = pair["domain"]

        if div < len(pair["success_steps"]):
            s_action = normalize_action(pair["success_steps"][div])
            action_type = s_action["action_type"]
            domain_analysis[domain][action_type] += 1
            domain_analysis["all"][action_type] += 1

    return dict(domain_analysis)


def main():
    parser = argparse.ArgumentParser(description="Experiment 0.4: Per-Domain FCV Curves")
    parser.add_argument("--max_pairs", type=int, default=5000)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "exp0_4"))

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load paired trajectories
    print(f"Loading up to {args.max_pairs} paired trajectories...")
    pairs = load_paired_trajectories(max_pairs=args.max_pairs)
    print(f"Loaded {len(pairs)} pairs")

    if not pairs:
        print("ERROR: No paired trajectories found!")
        return

    # Count per domain
    domain_counts = defaultdict(int)
    for p in pairs:
        domain_counts[p["domain"]] += 1
    print(f"Per-domain: {dict(domain_counts)}")

    # Compute FCV curves
    print("\nComputing FCV curves...")
    fcv_curves = compute_fcv_curves(pairs, args.max_steps)
    cumulative_fcv = compute_cumulative_fcv(pairs, args.max_steps)
    bottleneck_actions = analyze_bottleneck_steps(pairs)

    # Save raw data
    with open(output_dir / "fcv_curves.json", "w") as f:
        # Convert int keys to str for JSON
        serializable = {}
        for domain, curve in fcv_curves.items():
            if domain == "raw_counts":
                serializable["raw_counts"] = {
                    d: {str(k): v for k, v in counts.items()}
                    for d, counts in curve.items()
                }
            else:
                serializable[domain] = {str(k): v for k, v in curve.items()}
        json.dump(serializable, f, indent=2)

    with open(output_dir / "cumulative_fcv.json", "w") as f:
        serializable = {
            domain: {str(k): v for k, v in curve.items()}
            for domain, curve in cumulative_fcv.items()
        }
        json.dump(serializable, f, indent=2)

    with open(output_dir / "bottleneck_actions.json", "w") as f:
        json.dump(dict(bottleneck_actions), f, indent=2)

    # === Print Analysis ===
    n = len(pairs)
    print("\n" + "=" * 60)
    print(f"  Experiment 0.4: Per-Domain FCV Curves (N={n})")
    print("=" * 60)

    # 1. FCV peaks per domain
    print(f"\n  FCV Peak Analysis (highest divergence probability steps):")
    domains = ["all", "excel", "word", "ppt"]
    for domain in domains:
        if domain not in fcv_curves or domain == "raw_counts":
            continue
        curve = fcv_curves[domain]
        if not curve:
            continue
        # Find top-3 FCV steps
        sorted_steps = sorted(curve.items(), key=lambda x: -x[1])[:3]
        print(f"\n    {domain.upper()} (n={domain_counts.get(domain, n)}):")
        for step, val in sorted_steps:
            raw = fcv_curves.get("raw_counts", {}).get(domain, {}).get(step, (0, 0))
            print(f"      Step {step}: FCV={val:.3f}  (diverged={raw[0]}, at_risk={raw[1]})")

    # 2. Cumulative divergence
    print(f"\n  Cumulative Divergence (% of pairs diverged by step):")
    print(f"    {'Step':>6}", end="")
    for domain in domains:
        if domain not in cumulative_fcv:
            continue
        print(f"  {domain:>8}", end="")
    print()

    for step in [0, 1, 2, 3, 5, 10, 15, 20]:
        if step >= args.max_steps:
            break
        print(f"    {step:>6}", end="")
        for domain in domains:
            if domain not in cumulative_fcv:
                continue
            val = cumulative_fcv[domain].get(step, 0)
            print(f"  {val:>7.1%}", end="")
        print()

    # 3. Bottleneck action types
    print(f"\n  Action Types at Divergence Point:")
    for domain in domains:
        if domain not in bottleneck_actions:
            continue
        actions = bottleneck_actions[domain]
        total = sum(actions.values())
        print(f"\n    {domain.upper()}:")
        for action_type, count in sorted(actions.items(), key=lambda x: -x[1])[:5]:
            print(f"      {action_type}: {count}/{total} ({count / total:.1%})")

    # 4. Cross-domain consistency
    print(f"\n  Cross-Domain Consistency Analysis:")
    domain_fcv_vectors = {}
    for domain in ["excel", "word", "ppt"]:
        if domain in fcv_curves:
            vec = [fcv_curves[domain].get(i, 0) for i in range(min(15, args.max_steps))]
            domain_fcv_vectors[domain] = vec

    if len(domain_fcv_vectors) >= 2:
        domain_list = list(domain_fcv_vectors.keys())
        print(f"    Correlation between domain FCV curves (steps 0-14):")
        for i in range(len(domain_list)):
            for j in range(i + 1, len(domain_list)):
                d1, d2 = domain_list[i], domain_list[j]
                v1 = np.array(domain_fcv_vectors[d1])
                v2 = np.array(domain_fcv_vectors[d2])
                if np.std(v1) > 0 and np.std(v2) > 0:
                    corr = np.corrcoef(v1, v2)[0, 1]
                    print(f"      {d1} vs {d2}: r={corr:.3f}")

    # 5. Determine if unified or per-domain FCV is needed
    print(f"\n  Recommendation:")
    # Check if FCV peaks occur at similar steps across domains
    peak_steps = {}
    for domain in ["excel", "word", "ppt"]:
        if domain in fcv_curves:
            curve = fcv_curves[domain]
            if curve:
                peak = max(curve.items(), key=lambda x: x[1])
                peak_steps[domain] = peak

    if peak_steps:
        peak_step_values = [p[0] for p in peak_steps.values()]
        peak_spread = max(peak_step_values) - min(peak_step_values) if peak_step_values else 0
        print(f"    Peak steps: {', '.join(f'{d}=step_{p[0]}' for d, p in peak_steps.items())}")
        print(f"    Peak spread: {peak_spread} steps")

        if peak_spread <= 2:
            print(f"    → Unified FCV recommended (peaks are consistent across domains)")
        else:
            print(f"    → Per-domain FCV recommended (peaks differ by {peak_spread} steps)")

    # 6. Suggested FCV thresholds for adaptive K
    print(f"\n  Suggested FCV Thresholds for Adaptive K:")
    all_fcv = fcv_curves.get("all", {})
    if all_fcv:
        fcv_values = sorted(all_fcv.values(), reverse=True)
        p90 = np.percentile(fcv_values, 90) if fcv_values else 0
        p50 = np.percentile(fcv_values, 50) if fcv_values else 0
        print(f"    FCV P90: {p90:.3f}  (top 10% bottleneck steps)")
        print(f"    FCV P50: {p50:.3f}  (median)")
        print(f"    Suggested thresholds:")
        print(f"      High (K=5): FCV > {p90:.3f}")
        print(f"      Medium (K=3): FCV > {p50:.3f}")
        print(f"      Low (K=1): FCV ≤ {p50:.3f}")

    print("\n" + "=" * 60)

    # Save summary
    summary = {
        "n_pairs": n,
        "domain_counts": dict(domain_counts),
        "peak_steps": {d: {"step": int(p[0]), "fcv": p[1]} for d, p in peak_steps.items()} if peak_steps else {},
        "cumulative_20pct": {
            domain: next((s for s in range(args.max_steps) if cumulative_fcv.get(domain, {}).get(s, 0) >= 0.2), -1)
            for domain in domains if domain in cumulative_fcv
        },
        "cumulative_50pct": {
            domain: next((s for s in range(args.max_steps) if cumulative_fcv.get(domain, {}).get(s, 0) >= 0.5), -1)
            for domain in domains if domain in cumulative_fcv
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {output_dir / 'summary.json'}")

    # Generate matplotlib plot if available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: FCV curves
        ax = axes[0]
        for domain in ["all", "excel", "word", "ppt"]:
            if domain not in fcv_curves or domain == "raw_counts":
                continue
            curve = fcv_curves[domain]
            steps = sorted(int(k) for k in curve.keys())
            values = [curve[s] for s in steps]
            style = "--" if domain == "all" else "-"
            ax.plot(steps, values, style, label=domain, linewidth=2 if domain == "all" else 1.5)

        ax.set_xlabel("Step Index")
        ax.set_ylabel("FCV (Divergence Hazard)")
        ax.set_title("FCV Curves by Domain")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Cumulative divergence
        ax = axes[1]
        for domain in ["all", "excel", "word", "ppt"]:
            if domain not in cumulative_fcv:
                continue
            curve = cumulative_fcv[domain]
            steps = sorted(int(k) for k in curve.keys())
            values = [curve[s] for s in steps]
            style = "--" if domain == "all" else "-"
            ax.plot(steps, values, style, label=domain, linewidth=2 if domain == "all" else 1.5)

        ax.set_xlabel("Step Index")
        ax.set_ylabel("Cumulative Divergence Rate")
        ax.set_title("Cumulative Divergence by Domain")
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="50%")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "fcv_curves.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved to: {plot_path}")
        plt.close()

    except ImportError:
        print("matplotlib not available, skipping plot generation")


if __name__ == "__main__":
    main()
