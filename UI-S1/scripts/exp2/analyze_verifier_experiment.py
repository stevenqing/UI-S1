"""Analyze verifier experiment results across all 3 conditions.

Produces:
  Table 1: Overall comparison (TSR, progress, step accuracy)
  Table 2: Probe calibration (confusion matrix, precision/recall)
  Table 3: Verifier decision breakdown (greedy vs resample accuracy)
  Figure 1: Per-trajectory TSR comparison (paired, verifier vs greedy)
"""

import argparse
import json
import os
from collections import defaultdict
from glob import glob

import numpy as np


def load_results(results_dir, timestamp=None):
    """Load results from all conditions."""
    conditions = {}

    # Find result files
    if timestamp:
        pattern = os.path.join(results_dir, f"verifier_*_{timestamp}", "verifier_*_results.json")
    else:
        pattern = os.path.join(results_dir, "verifier_*", "verifier_*_results.json")

    files = sorted(glob(pattern))
    if not files:
        # Try without subdirectory timestamp
        pattern = os.path.join(results_dir, "verifier_*_results.json")
        files = sorted(glob(pattern))

    for f in files:
        with open(f, "r") as fp:
            data = json.load(fp)
        mode = data["config"]["mode"]
        conditions[mode] = data
        print(f"Loaded {mode}: {f}")

    return conditions


def print_table1(conditions):
    """Table 1: Overall Comparison."""
    print("\n" + "=" * 80)
    print("Table 1: Overall Comparison")
    print("=" * 80)

    header = f"{'Condition':<25} {'TSR':>8} {'Seq Prog':>10} {'Scat Prog':>10} {'Step Acc':>10} {'ΔTSR vs greedy':>15}"
    print(header)
    print("-" * 80)

    greedy_tsr = None
    rows = []

    for mode in ["always_greedy", "verifier", "always_temp1"]:
        if mode not in conditions:
            continue
        stats = conditions[mode]["statistics"]
        tsr = stats["trajectory_success_rate"]
        seq = stats["avg_progress_rate"]
        scat = stats["avg_scattered_progress_rate"]
        step = stats["step_success_rate"]

        if mode == "always_greedy":
            greedy_tsr = tsr
            delta = "—"
        else:
            delta = f"{tsr - greedy_tsr:+.4f}" if greedy_tsr is not None else "N/A"

        label = {
            "always_greedy": "always_greedy (HF base)",
            "verifier": "verifier (probe-guided)",
            "always_temp1": "always_temp1 (stochastic)",
        }[mode]

        print(f"{label:<25} {tsr:>8.4f} {seq:>10.4f} {scat:>10.4f} {step:>10.4f} {delta:>15}")


def print_table2(conditions):
    """Table 2: Probe Calibration."""
    if "verifier" not in conditions:
        print("\nTable 2: Skipped (no verifier results)")
        return

    print("\n" + "=" * 80)
    print("Table 2: Probe Calibration")
    print("=" * 80)

    # Collect per-step data from verifier condition
    probs = []
    actuals = []

    for traj in conditions["verifier"]["detailed_results"]:
        for step in traj["step_results"]:
            if "probe_prob_correct" in step:
                probs.append(step["probe_prob_correct"])
                actuals.append(step["success"])

    probs = np.array(probs)
    actuals = np.array(actuals)
    predictions = probs > 0.5  # True = predict correct

    # Confusion matrix
    tp = int(((predictions == True) & (actuals == True)).sum())
    fp = int(((predictions == True) & (actuals == False)).sum())
    fn = int(((predictions == False) & (actuals == True)).sum())
    tn = int(((predictions == False) & (actuals == False)).sum())

    print(f"\nConfusion Matrix (probe prediction vs actual correctness):")
    print(f"                    Actually Correct   Actually Wrong")
    print(f"Predict Correct     {tp:>10}          {fp:>10}")
    print(f"Predict Wrong       {fn:>10}          {tn:>10}")

    # Precision/recall for "wrong" prediction
    wrong_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    wrong_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    wrong_f1 = 2 * wrong_precision * wrong_recall / (wrong_precision + wrong_recall) if (wrong_precision + wrong_recall) > 0 else 0

    print(f"\n'Wrong' Prediction Metrics:")
    print(f"  Precision (of steps predicted wrong, how many truly wrong): {wrong_precision:.4f}")
    print(f"  Recall (of truly wrong steps, how many predicted wrong):    {wrong_recall:.4f}")
    print(f"  F1:                                                         {wrong_f1:.4f}")

    # Correct prediction metrics
    correct_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    correct_recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"\n'Correct' Prediction Metrics:")
    print(f"  Precision: {correct_precision:.4f}")
    print(f"  Recall:    {correct_recall:.4f}")

    # Average P(correct) by actual outcome
    if actuals.any():
        print(f"\nAvg P(correct) for actually-correct steps: {probs[actuals].mean():.4f} (n={actuals.sum()})")
    if (~actuals).any():
        print(f"Avg P(correct) for actually-wrong steps:   {probs[~actuals].mean():.4f} (n={(~actuals).sum()})")

    print(f"\nOverall accuracy: {((predictions == actuals).mean()):.4f}")
    print(f"Majority baseline: {max(actuals.mean(), 1 - actuals.mean()):.4f}")


def print_table3(conditions):
    """Table 3: Verifier Decisions Breakdown."""
    if "verifier" not in conditions:
        print("\nTable 3: Skipped (no verifier results)")
        return

    print("\n" + "=" * 80)
    print("Table 3: Verifier Decisions Breakdown")
    print("=" * 80)

    greedy_steps = []
    resample_steps = []

    for traj in conditions["verifier"]["detailed_results"]:
        for step in traj["step_results"]:
            if step.get("probe_decision") == "greedy":
                greedy_steps.append(step)
            elif step.get("probe_decision") == "resample":
                resample_steps.append(step)

    n_greedy = len(greedy_steps)
    n_resample = len(resample_steps)
    n_total = n_greedy + n_resample

    greedy_acc = sum(1 for s in greedy_steps if s["success"]) / n_greedy if n_greedy else 0
    resample_acc = sum(1 for s in resample_steps if s["success"]) / n_resample if n_resample else 0

    print(f"\nRouting Statistics:")
    print(f"  Total steps:     {n_total}")
    print(f"  → Greedy (T=0):  {n_greedy} ({n_greedy/n_total*100:.1f}%)")
    print(f"  → Resample (T=1):{n_resample} ({n_resample/n_total*100:.1f}%)")

    print(f"\nAccuracy by Route:")
    print(f"  Greedy-routed steps:   {greedy_acc:.4f} ({sum(1 for s in greedy_steps if s['success'])}/{n_greedy})")
    print(f"  Resample-routed steps: {resample_acc:.4f} ({sum(1 for s in resample_steps if s['success'])}/{n_resample})")

    # Compare: for resample-routed steps, what's the baseline accuracy in always_greedy?
    if "always_greedy" in conditions:
        # Match steps by sample_id
        greedy_baseline = {}
        for traj in conditions["always_greedy"]["detailed_results"]:
            for step in traj["step_results"]:
                greedy_baseline[step["sample_id"]] = step.get("success", False)

        resample_ids = [s["sample_id"] for s in resample_steps]
        baseline_matches = [greedy_baseline.get(sid, False) for sid in resample_ids if sid in greedy_baseline]

        if baseline_matches:
            baseline_acc = sum(baseline_matches) / len(baseline_matches)
            print(f"\n  For probe-flagged steps (sent to resample):")
            print(f"    Accuracy with always_greedy baseline: {baseline_acc:.4f}")
            print(f"    Accuracy with verifier resampling:    {resample_acc:.4f}")
            print(f"    Delta:                                {resample_acc - baseline_acc:+.4f}")


def print_figure1_text(conditions):
    """Figure 1: Per-trajectory TSR comparison (text-based)."""
    if "always_greedy" not in conditions or "verifier" not in conditions:
        print("\nFigure 1: Skipped (need both always_greedy and verifier)")
        return

    print("\n" + "=" * 80)
    print("Figure 1: Per-Trajectory Comparison (verifier vs greedy)")
    print("=" * 80)

    # Build per-trajectory lookup
    greedy_traj = {t["trajectory_id"]: t for t in conditions["always_greedy"]["trajectory_results"]}
    verifier_traj = {t["trajectory_id"]: t for t in conditions["verifier"]["trajectory_results"]}

    common_ids = sorted(set(greedy_traj.keys()) & set(verifier_traj.keys()))

    improved = 0
    degraded = 0
    same = 0
    deltas = []

    for tid in common_ids:
        g = greedy_traj[tid]
        v = verifier_traj[tid]
        g_scat = g["scattered_progress_rate"]
        v_scat = v["scattered_progress_rate"]
        delta = v_scat - g_scat
        deltas.append(delta)

        if delta > 0.001:
            improved += 1
        elif delta < -0.001:
            degraded += 1
        else:
            same += 1

    deltas = np.array(deltas)

    print(f"\nPaired comparison ({len(common_ids)} trajectories):")
    print(f"  Improved (verifier > greedy): {improved} ({improved/len(common_ids)*100:.1f}%)")
    print(f"  Same:                         {same} ({same/len(common_ids)*100:.1f}%)")
    print(f"  Degraded (verifier < greedy): {degraded} ({degraded/len(common_ids)*100:.1f}%)")
    print(f"\n  Mean Δ scattered progress: {deltas.mean():+.4f}")
    print(f"  Median Δ scattered progress: {np.median(deltas):+.4f}")
    print(f"  Std Δ scattered progress:    {deltas.std():.4f}")

    # TSR comparison
    g_successes = sum(1 for tid in common_ids if greedy_traj[tid]["trajectory_success"])
    v_successes = sum(1 for tid in common_ids if verifier_traj[tid]["trajectory_success"])
    print(f"\n  Greedy TSR:   {g_successes}/{len(common_ids)} = {g_successes/len(common_ids):.4f}")
    print(f"  Verifier TSR: {v_successes}/{len(common_ids)} = {v_successes/len(common_ids):.4f}")

    # Save plot data
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Scatter: greedy vs verifier scattered progress
        g_scats = [greedy_traj[tid]["scattered_progress_rate"] for tid in common_ids]
        v_scats = [verifier_traj[tid]["scattered_progress_rate"] for tid in common_ids]

        axes[0].scatter(g_scats, v_scats, alpha=0.5, s=20)
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
        axes[0].set_xlabel("Greedy Scattered Progress")
        axes[0].set_ylabel("Verifier Scattered Progress")
        axes[0].set_title("Per-Trajectory Scattered Progress")
        axes[0].set_xlim(-0.05, 1.05)
        axes[0].set_ylim(-0.05, 1.05)

        # Histogram of deltas
        axes[1].hist(deltas, bins=30, edgecolor="black", alpha=0.7)
        axes[1].axvline(0, color="red", linestyle="--", alpha=0.5)
        axes[1].set_xlabel("Δ Scattered Progress (verifier - greedy)")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"Distribution of Improvements (mean={deltas.mean():.3f})")

        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(next(iter(conditions.values()))["config"].get("model_path", "")), "verifier_comparison.png")
        # Save next to the results
        for mode, data in conditions.items():
            if "detailed_results" in data:
                out_dir = os.path.dirname(data["config"].get("probe_path", "")) or "."
                break
        else:
            out_dir = "."

        plot_path = os.path.join(args.results_dir, "verifier_comparison.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\n  Plot saved to {plot_path}")
        plt.close()
    except ImportError:
        print("\n  (matplotlib not available, skipping plot)")
    except Exception as e:
        print(f"\n  (Error saving plot: {e})")


def print_domain_breakdown(conditions):
    """Per-domain breakdown."""
    print("\n" + "=" * 80)
    print("Domain Breakdown")
    print("=" * 80)

    for mode in ["always_greedy", "verifier", "always_temp1"]:
        if mode not in conditions:
            continue

        domain_stats = defaultdict(lambda: {"n": 0, "success": 0, "steps": 0, "steps_correct": 0})
        for traj in conditions[mode]["trajectory_results"]:
            d = traj["domain"]
            domain_stats[d]["n"] += 1
            domain_stats[d]["success"] += int(traj["trajectory_success"])

        for traj in conditions[mode]["detailed_results"]:
            d = traj["domain"]
            domain_stats[d]["steps"] += len(traj["step_results"])
            domain_stats[d]["steps_correct"] += sum(1 for s in traj["step_results"] if s.get("success", False))

        print(f"\n  {mode}:")
        print(f"    {'Domain':<10} {'TSR':>8} {'Step Acc':>10} {'N traj':>8}")
        print(f"    {'-'*40}")
        for d in sorted(domain_stats.keys()):
            ds = domain_stats[d]
            tsr = ds["success"] / ds["n"] if ds["n"] else 0
            sacc = ds["steps_correct"] / ds["steps"] if ds["steps"] else 0
            print(f"    {d:<10} {tsr:>8.4f} {sacc:>10.4f} {ds['n']:>8}")


def main():
    global args
    parser = argparse.ArgumentParser(description="Analyze verifier experiment results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing verifier_*/ result subdirectories")
    parser.add_argument("--timestamp", type=str, default=None,
                        help="Timestamp to filter results (e.g., 20260320_150000)")
    parser.add_argument("--baseline_path", type=str, default=None,
                        help="Optional path to existing subtask_isolated vLLM baseline results")
    args = parser.parse_args()

    conditions = load_results(args.results_dir, args.timestamp)

    if not conditions:
        print("ERROR: No result files found!")
        print(f"Searched in: {args.results_dir}")
        return

    print(f"\nLoaded {len(conditions)} conditions: {list(conditions.keys())}")

    # Load optional baseline
    if args.baseline_path and os.path.exists(args.baseline_path):
        with open(args.baseline_path, "r") as f:
            baseline = json.load(f)
        conditions["subtask_isolated_vllm"] = baseline
        print(f"Loaded vLLM baseline from {args.baseline_path}")

    print_table1(conditions)
    print_table2(conditions)
    print_table3(conditions)
    print_figure1_text(conditions)
    print_domain_breakdown(conditions)


if __name__ == "__main__":
    main()
