#!/usr/bin/env python3
"""
Exp C4 + C7: Selector Ceiling & Adaptive K Cost-Benefit Analysis

C4: Spatial distribution analysis of V3 candidates when agreement is low.
    - How many natural clusters? Bimodal vs random?
    - In bimodal cases, does one peak match GT? → Selector ceiling

C7: Adaptive K simulation.
    - Compare fixed K=1/5/10 vs adaptive strategies
    - Draw accuracy vs avg_K pareto frontier

Data source: Exp 1.1 K=10 results (18,265 samples with all_coords)
"""

import json
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


def is_in_rect(coord, rect):
    if not coord or not rect:
        return False
    try:
        x, y = float(coord[0]), float(coord[1])
        return (rect["left"] <= x <= rect["right"] and
                rect["top"] <= y <= rect["bottom"])
    except (TypeError, ValueError, KeyError, IndexError):
        return False


def cluster_coords_simple(coords, eps=30.0):
    """Simple DBSCAN-like clustering for small N."""
    if not coords:
        return []
    from sklearn.cluster import DBSCAN
    X = np.array(coords)
    labels = DBSCAN(eps=eps, min_samples=1).fit_predict(X)
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(coords[i])
    return [{"center": np.mean(c, axis=0).tolist(), "coords": c, "size": len(c)}
            for c in clusters.values()]


def analyze_c4(results):
    """C4: Spatial distribution analysis of uncertain candidates."""
    print("=" * 70)
    print("  C4: V3 Candidate Spatial Distribution Analysis")
    print("=" * 70)

    # Group by agreement rate
    bins = [
        ("≥0.9", 0.9, 1.01),
        ("0.7-0.9", 0.7, 0.9),
        ("0.5-0.7", 0.5, 0.7),
        ("0.3-0.5", 0.3, 0.5),
        ("<0.3", 0.0, 0.3),
    ]

    for label, lo, hi in bins:
        subset = [r for r in results if lo <= r.get("agreement_rate", 0) < hi]
        if not subset:
            continue

        n_clusters_dist = Counter()
        bimodal_one_correct = 0
        bimodal_total = 0
        avg_spread = []

        for r in subset:
            coords = r.get("all_coords", [])
            rect = r.get("gt_rectangle", {})
            if len(coords) < 2:
                continue

            # Cluster the K candidates
            clusters = cluster_coords_simple(coords, eps=30.0)
            n_clusters_dist[len(clusters)] += 1

            # Spread: max distance between any two candidates
            X = np.array(coords)
            dists = np.sqrt(((X[:, None] - X[None, :]) ** 2).sum(axis=2))
            avg_spread.append(np.max(dists))

            # Bimodal analysis
            if len(clusters) == 2:
                bimodal_total += 1
                # Check if either cluster center matches GT
                for cl in clusters:
                    if is_in_rect(cl["center"], rect):
                        bimodal_one_correct += 1
                        break

        n = len(subset)
        greedy_acc = np.mean([r["greedy_correct"] for r in subset])
        best_of_k = np.mean([r["best_of_k_correct"] for r in subset])

        print(f"\n  --- Agreement {label}: {n} samples ---")
        print(f"    Greedy accuracy: {greedy_acc:.1%}")
        print(f"    Best-of-K=10:    {best_of_k:.1%}")
        print(f"    Oracle headroom: {best_of_k - greedy_acc:+.1%}")

        if avg_spread:
            print(f"    Avg max spread:  {np.mean(avg_spread):.1f}px "
                  f"(median={np.median(avg_spread):.1f}px)")

        # Cluster distribution
        print(f"    Cluster distribution:")
        for nc in sorted(n_clusters_dist.keys()):
            cnt = n_clusters_dist[nc]
            total_clustered = sum(n_clusters_dist.values())
            print(f"      {nc} clusters: {cnt} ({cnt/total_clustered:.1%})")

        if bimodal_total > 0:
            print(f"    Bimodal (2 clusters): {bimodal_total} samples")
            print(f"      One cluster matches GT: {bimodal_one_correct} "
                  f"({bimodal_one_correct/bimodal_total:.1%})")
            print(f"      → Selector ceiling for bimodal: {bimodal_one_correct/bimodal_total:.1%}")

    # Overall Selector ceiling
    print(f"\n  --- Overall Selector Ceiling ---")
    total = len(results)
    # For each sample, check if ANY of K=10 coords matches GT
    any_correct = sum(1 for r in results if r.get("best_of_k_correct"))
    # For greedy
    greedy_correct = sum(1 for r in results if r.get("greedy_correct"))
    # For cluster center
    cluster_correct = sum(1 for r in results if r.get("cluster_center_correct"))

    print(f"    Greedy (K=1):        {greedy_correct/total:.1%}")
    print(f"    DBSCAN cluster:      {cluster_correct/total:.1%}")
    print(f"    Oracle best-of-K=10: {any_correct/total:.1%}")
    print(f"    → Perfect Selector ceiling: {any_correct/total:.1%}")
    print(f"    → Selector must recover: {(any_correct - greedy_correct)/total:.1%} "
          f"({any_correct - greedy_correct} samples)")


def analyze_c7(results):
    """C7: Adaptive K cost-benefit analysis."""
    print("\n" + "=" * 70)
    print("  C7: Adaptive K Cost-Benefit Analysis")
    print("=" * 70)

    # For each sample, we have K=10 candidates and know which are correct
    # We simulate different K strategies using the first K candidates

    def eval_strategy(results, k_func, use_oracle_selector=True):
        """Evaluate a K strategy. Returns (accuracy, avg_K)."""
        total_k = 0
        correct = 0
        n = 0

        for r in results:
            coords = r.get("all_coords", [])
            rect = r.get("gt_rectangle", {})
            agree = r.get("agreement_rate", 0)

            k = k_func(agree)
            k = min(k, len(coords))
            total_k += k
            n += 1

            if k == 0:
                continue

            used_coords = coords[:k]

            if use_oracle_selector:
                # Oracle: pick the correct one if any
                any_hit = any(is_in_rect(c, rect) for c in used_coords)
                if any_hit:
                    correct += 1
            else:
                # Greedy: use first candidate
                if is_in_rect(used_coords[0], rect):
                    correct += 1

        return correct / n if n > 0 else 0, total_k / n if n > 0 else 0

    strategies = {
        "K=1 (greedy)":           lambda a: 1,
        "K=3 (fixed)":            lambda a: 3,
        "K=5 (fixed)":            lambda a: 5,
        "K=10 (fixed)":           lambda a: 10,
        "Adaptive v1 (1/5)":      lambda a: 1 if a >= 0.9 else 5,
        "Adaptive v2 (1/5/10)":   lambda a: 1 if a >= 0.9 else (5 if a >= 0.5 else 10),
        "Adaptive v3 (1/3/10)":   lambda a: 1 if a >= 0.7 else (3 if a >= 0.4 else 10),
        "Adaptive v4 (1/3/7)":    lambda a: 1 if a >= 0.8 else (3 if a >= 0.5 else 7),
    }

    # Note: agreement_rate is computed from K=10, so for strategies using it,
    # there's a "chicken and egg" problem. In practice, the first K=1 greedy
    # wouldn't have agreement. We use K=10 agreement as an oracle signal.
    # A real system would need a separate confidence estimator.

    print(f"\n  {'Strategy':<28s} {'Oracle Acc':>10s} {'Greedy Acc':>10s} {'Avg K':>7s}")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*7}")

    pareto_points = []

    for name, k_func in strategies.items():
        oracle_acc, avg_k = eval_strategy(results, k_func, use_oracle_selector=True)
        greedy_acc, _ = eval_strategy(results, k_func, use_oracle_selector=False)
        print(f"  {name:<28s} {oracle_acc:>10.1%} {greedy_acc:>10.1%} {avg_k:>7.1f}")
        pareto_points.append((avg_k, oracle_acc, name))

    # Pareto frontier analysis
    print(f"\n  --- Pareto Frontier (Oracle Selector) ---")
    pareto_points.sort(key=lambda x: x[0])
    frontier = []
    best_acc = 0
    for k, acc, name in pareto_points:
        if acc > best_acc:
            frontier.append((k, acc, name))
            best_acc = acc

    print(f"  {'Strategy':<28s} {'Avg K':>7s} {'Oracle Acc':>10s} {'Marginal':>10s}")
    prev_acc = 0
    for k, acc, name in frontier:
        marginal = f"+{acc-prev_acc:.1%}/K" if prev_acc > 0 else ""
        print(f"  {name:<28s} {k:>7.1f} {acc:>10.1%} {marginal:>10s}")
        prev_acc = acc

    # Cost-benefit: how much does each extra K buy?
    print(f"\n  --- Efficiency Analysis ---")
    k1_acc = sum(1 for r in results if r["greedy_correct"]) / len(results)
    k10_oracle = sum(1 for r in results if r["best_of_k_correct"]) / len(results)
    print(f"    K=1 greedy:   {k1_acc:.1%}")
    print(f"    K=10 oracle:  {k10_oracle:.1%}")
    print(f"    Gap:          {k10_oracle - k1_acc:.1%} ({k10_oracle - k1_acc:.1%} over 9 extra samples)")
    print(f"    Per-sample:   {(k10_oracle - k1_acc)/9:.2%} per extra K")

    # Where does the gain come from?
    gains_by_agreement = defaultdict(lambda: {"total": 0, "k1_correct": 0, "k10_correct": 0})
    for r in results:
        agree = r.get("agreement_rate", 0)
        bucket = "≥0.9" if agree >= 0.9 else ("0.5-0.9" if agree >= 0.5 else "<0.5")
        gains_by_agreement[bucket]["total"] += 1
        gains_by_agreement[bucket]["k1_correct"] += r["greedy_correct"]
        gains_by_agreement[bucket]["k10_correct"] += r["best_of_k_correct"]

    print(f"\n  Oracle headroom by agreement bucket:")
    print(f"    {'Bucket':>8s} {'N':>6s} {'K=1':>7s} {'K=10':>7s} {'Headroom':>9s}")
    for bucket in ["≥0.9", "0.5-0.9", "<0.5"]:
        d = gains_by_agreement[bucket]
        n = d["total"]
        k1 = d["k1_correct"] / n if n > 0 else 0
        k10 = d["k10_correct"] / n if n > 0 else 0
        print(f"    {bucket:>8s} {n:>6d} {k1:>7.1%} {k10:>7.1%} {k10-k1:>+9.1%}")


def main():
    print("Loading K=10 results...")
    results = load_k10_results()
    print(f"Loaded {len(results)} samples\n")

    analyze_c4(results)
    analyze_c7(results)

    # Save results
    output_dir = PROJECT_ROOT / "outputs" / "eval_c4c7"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "total_samples": len(results),
        "greedy_accuracy": sum(1 for r in results if r["greedy_correct"]) / len(results),
        "best_of_k10_accuracy": sum(1 for r in results if r["best_of_k_correct"]) / len(results),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
