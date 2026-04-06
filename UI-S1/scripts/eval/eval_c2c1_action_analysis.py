#!/usr/bin/env python3
"""
Exp C2 + C1: Action Type Analysis & Description Quality

C2: Per-action-type breakdown of V3 grounding accuracy.
    - Which action types have the worst grounding?
    - Does action type predict grounding difficulty?

C1: GT thought/description quality vs V3 accuracy.
    - Do samples with spatial cues in the GT description ground better?
    - What linguistic features correlate with grounding success?

Data sources:
- outputs/exp1_1/results_K10.jsonl (K=10 grounding results)
- train_GUI_360/data/gui360_test_a11y_eval.parquet (for GT details)
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_k10_results():
    path = PROJECT_ROOT / "outputs" / "exp1_1" / "results_K10.jsonl"
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def analyze_c2_action_types(results):
    """C2: Per-action-type breakdown of grounding accuracy."""
    print("=" * 70)
    print("  C2: Per-Action-Type Grounding Analysis")
    print("=" * 70)

    # Group by gt_function
    by_func = defaultdict(list)
    for r in results:
        func = r.get("gt_function", "unknown")
        by_func[func].append(r)

    print(f"\n  {'Action Type':<25s} {'N':>6s} {'Greedy':>8s} {'Cluster':>8s} "
          f"{'Oracle':>8s} {'Headroom':>9s} {'Agree':>7s}")
    print(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*7}")

    rows = []
    for func in sorted(by_func.keys(), key=lambda f: -len(by_func[f])):
        subset = by_func[func]
        n = len(subset)
        if n < 10:
            continue

        greedy = np.mean([r["greedy_correct"] for r in subset])
        cluster = np.mean([r["cluster_center_correct"] for r in subset])
        oracle = np.mean([r["best_of_k_correct"] for r in subset])
        agree = np.mean([r.get("agreement_rate", 0) for r in subset])
        headroom = oracle - greedy

        print(f"  {func:<25s} {n:>6d} {greedy:>8.1%} {cluster:>8.1%} "
              f"{oracle:>8.1%} {headroom:>+9.1%} {agree:>7.2f}")
        rows.append({"func": func, "n": n, "greedy": greedy, "cluster": cluster,
                      "oracle": oracle, "headroom": headroom, "agree": agree})

    # Which action types benefit most from multi-sampling?
    print(f"\n  --- Action types with highest oracle headroom ---")
    rows.sort(key=lambda r: -r["headroom"])
    for r in rows[:5]:
        print(f"    {r['func']:<25s} headroom={r['headroom']:+.1%} "
              f"(greedy={r['greedy']:.1%} → oracle={r['oracle']:.1%})")

    # Which action types are already well-handled?
    print(f"\n  --- Action types with highest greedy accuracy ---")
    rows.sort(key=lambda r: -r["greedy"])
    for r in rows[:5]:
        print(f"    {r['func']:<25s} greedy={r['greedy']:.1%} "
              f"(n={r['n']})")

    return by_func


def analyze_c1_description_quality(results):
    """C1: Analyze if GT description/response quality affects grounding."""
    print(f"\n{'=' * 70}")
    print("  C1: Description Quality vs Grounding Accuracy")
    print("=" * 70)

    # Use greedy_response text to analyze what the model outputs
    # Check if spatial/positional words in the prompt or response correlate with accuracy

    # Spatial cue keywords
    spatial_keywords = [
        "top", "bottom", "left", "right", "center", "middle",
        "upper", "lower", "corner", "edge", "side",
        "above", "below", "next to", "beside", "near",
        "toolbar", "menu", "ribbon", "tab", "panel",
        "first", "second", "third", "last",
    ]

    has_spatial = {"correct": 0, "total": 0}
    no_spatial = {"correct": 0, "total": 0}

    response_length_correct = []
    response_length_wrong = []

    spatial_keyword_counts = Counter()

    for r in results:
        response = r.get("greedy_response", "")
        correct = r["greedy_correct"]

        # Check for spatial keywords
        response_lower = response.lower()
        found_spatial = False
        for kw in spatial_keywords:
            if kw in response_lower:
                found_spatial = True
                spatial_keyword_counts[kw] += 1

        if found_spatial:
            has_spatial["total"] += 1
            if correct:
                has_spatial["correct"] += 1
        else:
            no_spatial["total"] += 1
            if correct:
                no_spatial["correct"] += 1

        # Response length analysis
        if correct:
            response_length_correct.append(len(response))
        else:
            response_length_wrong.append(len(response))

    # Results
    sp_acc = has_spatial["correct"] / has_spatial["total"] if has_spatial["total"] > 0 else 0
    nsp_acc = no_spatial["correct"] / no_spatial["total"] if no_spatial["total"] > 0 else 0

    print(f"\n  --- Spatial Cue Analysis ---")
    print(f"  With spatial cues:    {has_spatial['total']:>6d} samples, accuracy={sp_acc:.1%}")
    print(f"  Without spatial cues: {no_spatial['total']:>6d} samples, accuracy={nsp_acc:.1%}")
    print(f"  Difference: {sp_acc - nsp_acc:+.1%}")

    print(f"\n  Most common spatial keywords in responses:")
    for kw, cnt in spatial_keyword_counts.most_common(10):
        print(f"    '{kw}': {cnt}")

    print(f"\n  --- Response Length Analysis ---")
    print(f"  Correct predictions: mean={np.mean(response_length_correct):.0f} chars, "
          f"median={np.median(response_length_correct):.0f}")
    print(f"  Wrong predictions:   mean={np.mean(response_length_wrong):.0f} chars, "
          f"median={np.median(response_length_wrong):.0f}")


def analyze_domain_x_action(results):
    """Cross-tabulate domain x action type accuracy."""
    print(f"\n{'=' * 70}")
    print("  C2.2: Domain x Action Type Cross-Analysis")
    print("=" * 70)

    # Build domain x action matrix
    matrix = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    for r in results:
        domain = r.get("domain", "unknown")
        func = r.get("gt_function", "unknown")
        matrix[domain][func]["total"] += 1
        if r["greedy_correct"]:
            matrix[domain][func]["correct"] += 1

    # Get top action types
    all_funcs = Counter()
    for r in results:
        all_funcs[r.get("gt_function", "unknown")] += 1
    top_funcs = [f for f, _ in all_funcs.most_common(8)]

    # Print matrix
    header = f"  {'Domain':<10s}" + "".join(f" {f:>12s}" for f in top_funcs)
    print(f"\n{header}")
    print(f"  {'-'*10}" + "".join(f" {'-'*12}" for _ in top_funcs))

    for domain in sorted(matrix.keys()):
        row = f"  {domain:<10s}"
        for func in top_funcs:
            d = matrix[domain][func]
            if d["total"] > 0:
                acc = d["correct"] / d["total"]
                row += f" {acc:>5.0%} ({d['total']:>4d})"
            else:
                row += f" {'n/a':>12s}"
        print(row)


def analyze_agreement_by_action(results):
    """Agreement rate calibration per action type."""
    print(f"\n{'=' * 70}")
    print("  C2.3: Agreement Rate Calibration by Action Type")
    print("=" * 70)

    by_func = defaultdict(list)
    for r in results:
        func = r.get("gt_function", "unknown")
        by_func[func].append(r)

    print(f"\n  {'Action Type':<20s} {'High(≥0.9)':>12s} {'Mid(0.5-0.9)':>14s} {'Low(<0.5)':>12s}")
    print(f"  {'-'*20} {'-'*12} {'-'*14} {'-'*12}")

    for func in sorted(by_func.keys(), key=lambda f: -len(by_func[f])):
        subset = by_func[func]
        if len(subset) < 50:
            continue

        high = [r for r in subset if r.get("agreement_rate", 0) >= 0.9]
        mid = [r for r in subset if 0.5 <= r.get("agreement_rate", 0) < 0.9]
        low = [r for r in subset if r.get("agreement_rate", 0) < 0.5]

        high_acc = np.mean([r["greedy_correct"] for r in high]) if high else 0
        mid_acc = np.mean([r["greedy_correct"] for r in mid]) if mid else 0
        low_acc = np.mean([r["greedy_correct"] for r in low]) if low else 0

        high_str = f"{high_acc:.0%} ({len(high):>4d})"
        mid_str = f"{mid_acc:.0%} ({len(mid):>4d})"
        low_str = f"{low_acc:.0%} ({len(low):>4d})"

        print(f"  {func:<20s} {high_str:>12s} {mid_str:>14s} {low_str:>12s}")


def main():
    print("Loading K=10 results...")
    results = load_k10_results()
    print(f"Loaded {len(results)} samples\n")

    by_func = analyze_c2_action_types(results)
    analyze_c1_description_quality(results)
    analyze_domain_x_action(results)
    analyze_agreement_by_action(results)

    # Save summary
    output_dir = PROJECT_ROOT / "outputs" / "eval_c2c1"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-action summary
    func_summary = {}
    for func, subset in by_func.items():
        if len(subset) < 10:
            continue
        func_summary[func] = {
            "n": len(subset),
            "greedy_acc": float(np.mean([r["greedy_correct"] for r in subset])),
            "cluster_acc": float(np.mean([r["cluster_center_correct"] for r in subset])),
            "oracle_acc": float(np.mean([r["best_of_k_correct"] for r in subset])),
            "avg_agreement": float(np.mean([r.get("agreement_rate", 0) for r in subset])),
        }

    with open(output_dir / "summary.json", "w") as f:
        json.dump({"per_action_type": func_summary}, f, indent=2)
    print(f"\nSummary saved to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
