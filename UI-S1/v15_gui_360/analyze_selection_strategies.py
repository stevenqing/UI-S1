#!/usr/bin/env python3
"""
Selection Strategy Analysis for GUI Agent

Compare different sample selection strategies using existing BoN data:
1. Oracle (max reward) — current upper bound
2. Random (baseline)
3. Max logprob (model confidence)
4. Centroid selection (self-consistency)
5. Type majority vote + centroid
6. Adaptive: simulate different N allocations

All evaluated on step accuracy and simulated on-policy TSR.
"""

import json
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"


def load_data():
    bon_path = f"{PROJECT}/v15_gui_360/outputs/preliminary_tests/bon_metrics/bon_raw_20260523_122948.json"
    bon = json.load(open(bon_path))

    train_path = f"{PROJECT}/v12_gui_360/data/gui360_train_2000_balanced.jsonl"
    train = {}
    with open(train_path) as f:
        for line in f:
            ep = json.loads(line)
            train[ep["episode_id"]] = ep

    return bon, train


# ============================================================
# Selection strategies
# ============================================================

def select_oracle(samples):
    """Best possible: select sample with highest reward."""
    return max(samples, key=lambda s: s["reward"])


def select_random(samples, rng):
    """Random baseline."""
    return samples[rng.integers(len(samples))]


def select_max_logprob(samples):
    """Select most confident (highest mean logprob)."""
    return max(samples, key=lambda s: s.get("mean_logprob", -999))


def select_centroid(samples, gt_type):
    """Select sample closest to centroid of all predictions (self-consistency)."""
    if gt_type == "click":
        # For click: compute centroid of coordinates, pick closest
        coords = []
        valid_indices = []
        for i, s in enumerate(samples):
            if (s.get("pred_type") == "click" and
                s.get("coord_x") is not None and
                s.get("has_valid_action", False)):
                coords.append((s["coord_x"], s["coord_y"]))
                valid_indices.append(i)

        if len(coords) < 2:
            return select_max_logprob(samples)

        cx = np.mean([c[0] for c in coords])
        cy = np.mean([c[1] for c in coords])

        best_idx = None
        best_dist = float("inf")
        for i, (x, y) in zip(valid_indices, coords):
            d = np.sqrt((x - cx)**2 + (y - cy)**2)
            if d < best_dist:
                best_dist = d
                best_idx = i

        return samples[best_idx]

    elif gt_type == "type":
        # For type: majority vote on predicted text (use reward as proxy)
        return select_max_logprob(samples)

    else:
        # For swipe/other: majority vote on action type
        return select_max_logprob(samples)


def select_type_vote_centroid(samples, gt_type):
    """First majority-vote on action type, then centroid within that type."""
    # Count predicted action types
    type_counts = Counter()
    for s in samples:
        if s.get("has_valid_action", False):
            type_counts[s.get("pred_type", "unknown")] += 1

    if not type_counts:
        return select_max_logprob(samples)

    majority_type = type_counts.most_common(1)[0][0]

    # Filter to majority type
    filtered = [s for s in samples
                if s.get("pred_type") == majority_type and s.get("has_valid_action", False)]

    if not filtered:
        return select_max_logprob(samples)

    if majority_type == "click" and len(filtered) >= 2:
        return select_centroid(filtered, "click")
    else:
        return select_max_logprob(filtered)


def select_consistency_weighted(samples, gt_type):
    """Weight each sample by inverse distance to centroid, then pick highest weighted logprob."""
    if gt_type != "click":
        return select_type_vote_centroid(samples, gt_type)

    coords = []
    valid_indices = []
    for i, s in enumerate(samples):
        if (s.get("pred_type") == "click" and
            s.get("coord_x") is not None and
            s.get("has_valid_action", False)):
            coords.append((s["coord_x"], s["coord_y"]))
            valid_indices.append(i)

    if len(coords) < 2:
        return select_max_logprob(samples)

    cx = np.mean([c[0] for c in coords])
    cy = np.mean([c[1] for c in coords])

    # Score = logprob_rank + consistency_rank (both normalized)
    dists = [np.sqrt((x - cx)**2 + (y - cy)**2) for x, y in coords]
    logprobs = [samples[i].get("mean_logprob", -999) for i in valid_indices]

    # Rank-based scoring (lower distance = higher rank, higher logprob = higher rank)
    n = len(valid_indices)
    dist_ranks = np.argsort(np.argsort(dists))  # 0 = smallest dist = best
    lp_ranks = np.argsort(np.argsort([-lp for lp in logprobs]))  # 0 = highest logprob = best

    combined = dist_ranks + lp_ranks  # Lower is better
    best = np.argmin(combined)

    return samples[valid_indices[best]]


# ============================================================
# Evaluation
# ============================================================

def evaluate_strategy(bon, strategy_fn, name, n_trials=50):
    """Evaluate a selection strategy on step accuracy and simulated TSR."""
    step_correct = 0
    step_total = 0

    # Simulated on-policy TSR (consecutive correct from step 0)
    episode_results = {}

    is_random = "random" in name.lower()

    for ep_id, ep in bon.items():
        all_steps_correct = True
        for step in ep["steps"]:
            gt_type = step["gt_type"]

            if is_random:
                # Average over trials for random
                correct_count = 0
                for trial in range(n_trials):
                    rng = np.random.default_rng(trial * 10000 + int(ep_id) * 100 + step["step_idx"])
                    selected = strategy_fn(step["samples"], rng)
                    if selected["is_correct"]:
                        correct_count += 1
                is_correct = correct_count / n_trials > 0.5
                step_correct += correct_count / n_trials
            else:
                selected = strategy_fn(step["samples"], gt_type) if "gt_type" in strategy_fn.__code__.co_varnames else strategy_fn(step["samples"])
                is_correct = selected["is_correct"]
                step_correct += 1 if is_correct else 0

            step_total += 1
            if not is_correct:
                all_steps_correct = False

        episode_results[ep_id] = all_steps_correct

    step_acc = step_correct / step_total
    tsr = sum(episode_results.values()) / len(episode_results)

    return step_acc, tsr, episode_results


def evaluate_with_subsets(bon, strategy_fn, name, N_values=[1, 2, 3, 5]):
    """Evaluate strategy with different numbers of samples (simulate different N)."""
    results = {}
    is_random = "random" in name.lower()

    for N in N_values:
        step_correct = 0
        step_total = 0
        ep_success = 0
        ep_total = 0

        for ep_id, ep in bon.items():
            all_correct = True
            for step in ep["steps"]:
                gt_type = step["gt_type"]
                samples = step["samples"][:N]  # Take first N

                if len(samples) == 0:
                    step_total += 1
                    all_correct = False
                    continue

                if N == 1:
                    selected = samples[0]
                elif is_random:
                    rng = np.random.default_rng(int(ep_id) * 100 + step["step_idx"])
                    selected = strategy_fn(samples, rng)
                else:
                    selected = strategy_fn(samples, gt_type)

                is_correct = selected["is_correct"]
                step_correct += 1 if is_correct else 0
                step_total += 1
                if not is_correct:
                    all_correct = False

            ep_total += 1
            if all_correct:
                ep_success += 1

        results[N] = {
            "step_acc": step_correct / step_total if step_total > 0 else 0,
            "tsr": ep_success / ep_total if ep_total > 0 else 0,
        }

    return results


def analyze_consistency_correlation(bon):
    """Analyze how well consistency predicts correctness."""
    print("=" * 70)
    print("Consistency as Correctness Predictor")
    print("=" * 70)

    # For click steps, compute spread and check if correct
    data_points = []
    for ep in bon.values():
        for step in ep["steps"]:
            if step["gt_type"] != "click":
                continue

            coords = []
            for s in step["samples"]:
                if s.get("pred_type") == "click" and s.get("coord_x") is not None:
                    coords.append((s["coord_x"], s["coord_y"]))

            if len(coords) < 2:
                continue

            # Compute spread (mean pairwise distance)
            dists = []
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    d = np.sqrt((coords[i][0] - coords[j][0])**2 +
                                (coords[i][1] - coords[j][1])**2)
                    dists.append(d)

            spread = np.mean(dists)
            best_reward = max(s["reward"] for s in step["samples"])
            best_correct = max(s["is_correct"] for s in step["samples"])

            # Centroid selection result
            cx = np.mean([c[0] for c in coords])
            cy = np.mean([c[1] for c in coords])
            centroid_idx = np.argmin([np.sqrt((c[0]-cx)**2 + (c[1]-cy)**2) for c in coords])

            valid_click_samples = [s for s in step["samples"]
                                   if s.get("pred_type") == "click" and s.get("coord_x") is not None]
            if centroid_idx < len(valid_click_samples):
                centroid_correct = valid_click_samples[centroid_idx]["is_correct"]
                centroid_reward = valid_click_samples[centroid_idx]["reward"]
            else:
                centroid_correct = False
                centroid_reward = 0

            data_points.append({
                "spread": spread,
                "best_correct": best_correct,
                "centroid_correct": centroid_correct,
                "best_reward": best_reward,
                "centroid_reward": centroid_reward,
            })

    # Bin by spread
    bins = [(0, 30), (30, 80), (80, 150), (150, 300), (300, 9999)]
    print(f"\n  {'Spread (px)':>15s} | {'N':>4s} | {'Oracle Acc':>10s} | {'Centroid Acc':>12s} | {'Centroid vs Oracle':>18s}")
    print("  " + "-" * 75)

    for lo, hi in bins:
        pts = [p for p in data_points if lo <= p["spread"] < hi]
        if not pts:
            continue
        oracle_acc = np.mean([p["best_correct"] for p in pts])
        centroid_acc = np.mean([p["centroid_correct"] for p in pts])
        label = f"[{lo}, {hi})" if hi < 9999 else f"[{lo}, ∞)"
        gap = centroid_acc - oracle_acc
        print(f"  {label:>15s} | {len(pts):4d} | {oracle_acc:9.1%}  | {centroid_acc:11.1%}  | {gap:+17.1%}")

    # Can we use spread as a filter?
    print(f"\n  Using spread < 80px as confidence filter:")
    confident = [p for p in data_points if p["spread"] < 80]
    unconfident = [p for p in data_points if p["spread"] >= 80]
    if confident:
        print(f"    Confident:   {len(confident)} steps, "
              f"oracle={np.mean([p['best_correct'] for p in confident]):.1%}, "
              f"centroid={np.mean([p['centroid_correct'] for p in confident]):.1%}")
    if unconfident:
        print(f"    Unconfident: {len(unconfident)} steps, "
              f"oracle={np.mean([p['best_correct'] for p in unconfident]):.1%}, "
              f"centroid={np.mean([p['centroid_correct'] for p in unconfident]):.1%}")


def main():
    print("Loading data...")
    bon, train = load_data()
    print(f"Loaded {len(bon)} episodes\n")

    # ============================================================
    # 1. Compare selection strategies
    # ============================================================
    print("=" * 70)
    print("Selection Strategy Comparison (N=5)")
    print("=" * 70)

    strategies = [
        ("Oracle (max reward)", lambda samples, gt_type: select_oracle(samples)),
        ("Max logprob", lambda samples, gt_type: select_max_logprob(samples)),
        ("Centroid (consistency)", select_centroid),
        ("Type vote + centroid", select_type_vote_centroid),
        ("Consistency-weighted", select_consistency_weighted),
    ]

    print(f"\n  {'Strategy':>30s} | {'Step Acc':>8s} | {'Sim TSR':>8s}")
    print("  " + "-" * 55)

    strategy_results = {}
    for name, fn in strategies:
        step_acc, tsr, ep_results = evaluate_strategy(bon, fn, name)
        strategy_results[name] = (step_acc, tsr, ep_results)
        print(f"  {name:>30s} | {step_acc:7.1%}  | {tsr:7.1%}")

    # Random baseline (averaged)
    rng_fn = lambda samples, rng: select_random(samples, rng)
    step_acc_r, tsr_r, _ = evaluate_strategy(bon, rng_fn, "Random", n_trials=100)
    print(f"  {'Random (avg N=100)':>30s} | {step_acc_r:7.1%}  | {tsr_r:7.1%}")

    # ============================================================
    # 2. Per action type breakdown
    # ============================================================
    print(f"\n{'=' * 70}")
    print("Per Action Type: Strategy Comparison")
    print("=" * 70)

    for gt_type in ["click", "type", "swipe"]:
        print(f"\n  --- {gt_type} ---")
        for name, fn in strategies:
            correct = 0
            total = 0
            for ep in bon.values():
                for step in ep["steps"]:
                    if step["gt_type"] != gt_type:
                        continue
                    selected = fn(step["samples"], gt_type)
                    if selected["is_correct"]:
                        correct += 1
                    total += 1
            if total > 0:
                print(f"  {name:>30s}: {correct}/{total} = {correct/total:.1%}")

    # ============================================================
    # 3. Consistency as predictor
    # ============================================================
    print()
    analyze_consistency_correlation(bon)

    # ============================================================
    # 4. N scaling analysis
    # ============================================================
    print(f"\n{'=' * 70}")
    print("N Scaling: How many samples are needed?")
    print("=" * 70)

    for name, fn in [("Oracle", lambda s, gt: select_oracle(s)),
                     ("Centroid", select_centroid),
                     ("Type vote + centroid", select_type_vote_centroid)]:
        results = evaluate_with_subsets(bon, fn, name, N_values=[1, 2, 3, 5])
        print(f"\n  {name}:")
        for N, r in sorted(results.items()):
            print(f"    N={N}: step_acc={r['step_acc']:.1%}, TSR={r['tsr']:.1%}")

    # ============================================================
    # 5. Where does centroid beat oracle?
    # ============================================================
    print(f"\n{'=' * 70}")
    print("Episode-level: Where centroid beats/loses to oracle")
    print("=" * 70)

    oracle_results = strategy_results["Oracle (max reward)"][2]
    centroid_results = strategy_results["Type vote + centroid"][2]

    centroid_wins = []
    centroid_loses = []
    for ep_id in oracle_results:
        if centroid_results[ep_id] and not oracle_results[ep_id]:
            centroid_wins.append(ep_id)
        elif oracle_results[ep_id] and not centroid_results[ep_id]:
            centroid_loses.append(ep_id)

    print(f"\n  Centroid wins (centroid correct, oracle wrong): {len(centroid_wins)}")
    print(f"  Centroid loses (oracle correct, centroid wrong): {len(centroid_loses)}")
    print(f"  Both correct: {sum(1 for e in oracle_results if oracle_results[e] and centroid_results[e])}")
    print(f"  Both wrong: {sum(1 for e in oracle_results if not oracle_results[e] and not centroid_results[e])}")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'=' * 70}")
    print("Key Takeaways")
    print("=" * 70)
    print("""
1. ORACLE GAP: The gap between oracle (max reward) and achievable
   selection strategies shows how much room there is for better
   selection at inference time.

2. CONSISTENCY VALUE: If centroid ≈ oracle, then self-consistency
   is a reliable signal — we don't need ground truth to select well.

3. N SCALING: Diminishing returns after N=3 suggests compute is
   better spent on step-0 focused scaling or other improvements.

4. TYPE VOTING: If type vote helps, it suggests action type confusion
   is a significant and addressable error source.
""")


if __name__ == "__main__":
    main()
