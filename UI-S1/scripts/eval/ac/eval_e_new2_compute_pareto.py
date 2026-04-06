#!/usr/bin/env python3
"""
E_NEW2 Analysis: Compute-Efficiency Pareto Curve
=================================================

Shows how Pass-through routing can save inference compute without losing TSR.

For each step, we can either:
  - Run K=10 samples (specialist mode): use oracle selection (best-of-K)
  - Use the greedy prediction (pass-through mode): use sample[0], costs 1 call

The agreement rate measures model confidence on each step. When agreement is
high, the model is confident and greedy is likely correct -- so we can skip
the expensive K-sample specialist and just pass through.

For each agreement threshold tau:
  - Steps with agreement >= tau: pass-through (greedy, cost=1)
  - Steps with agreement < tau:  specialist  (oracle K=10, cost=10)

We sweep tau from 0.0 to 1.0 and compute the resulting TSR and average cost,
producing a Pareto curve of TSR vs compute.

Agreement rate definition:
  For each step, we canonicalize each of K predictions into a hashable action
  signature. The agreement rate = (count of the most common signature) / K.
"""

import json
import os
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
MULTISAMPLE_PATH = os.path.join(
    BASE, "outputs/eval_c4c7_ac/Qwen2.5-VL-7B/multisample_results.jsonl"
)
BASELINE_PATH = os.path.join(
    BASE, "outputs/eval_a_ac/Qwen2.5-VL-7B/trajectory_results.jsonl"
)
OUTPUT_PATH = os.path.join(BASE, "outputs/eval_e_new2/e_new2_compute_pareto.json")

K = 10  # number of samples per step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def canonicalize_action(pred_action):
    """
    Convert a predicted action dict into a hashable canonical form for
    agreement computation. We group predictions that are "the same action"
    even if coordinates differ slightly.

    For coordinate-based actions (click, swipe), we bin coordinates to a
    coarse grid (50px buckets) so that minor jitter doesn't break agreement.
    For text-based actions (type, open), we normalize case.
    For other actions, we use the action type + key fields.
    """
    if pred_action is None:
        return ("__NONE__",)

    action_type = pred_action.get("action", "__UNKNOWN__")

    if action_type == "click":
        coord = pred_action.get("coordinate", [0, 0])
        # Bin to 100px grid for agreement (coarse enough to group similar clicks)
        bx = coord[0] // 100
        by = coord[1] // 100
        return ("click", bx, by)

    elif action_type == "swipe":
        c1 = pred_action.get("coordinate", [0, 0])
        c2 = pred_action.get("coordinate2", [0, 0])
        bx1, by1 = c1[0] // 200, c1[1] // 200
        bx2, by2 = c2[0] // 200, c2[1] // 200
        return ("swipe", bx1, by1, bx2, by2)

    elif action_type == "type":
        text = pred_action.get("text", "").strip().lower()
        return ("type", text)

    elif action_type == "open":
        text = pred_action.get("text", "").strip().lower()
        return ("open", text)

    elif action_type == "terminate":
        status = pred_action.get("status", "")
        return ("terminate", status)

    elif action_type == "system_button":
        button = pred_action.get("button", "")
        return ("system_button", button)

    elif action_type == "wait":
        return ("wait",)

    elif action_type == "scroll":
        direction = pred_action.get("direction", "")
        return ("scroll", direction)

    else:
        # Fallback: just use action type
        return (action_type,)


def compute_agreement_rate(samples):
    """
    Compute the agreement rate for a list of K sample predictions.
    Agreement rate = (count of most common canonical action) / K.
    """
    canonical = [canonicalize_action(s.get("pred_action")) for s in samples]
    counter = Counter(canonical)
    most_common_count = counter.most_common(1)[0][1]
    return most_common_count / len(samples)


def load_multisample_data(path):
    """
    Load multisample results. Returns a list of episode dicts, each with:
      - episode_id
      - steps: list of step dicts with:
          - step_num
          - agreement_rate (computed)
          - greedy_correct (sample[0].extract_match)
          - oracle_correct (any sample has extract_match == True)
          - gt_action_type
    """
    episodes = []
    with open(path) as f:
        for line in f:
            ep = json.loads(line)
            steps = []
            for step_data in ep["step_samples"]:
                samples = step_data["samples"]

                agreement_rate = compute_agreement_rate(samples)
                greedy_correct = samples[0].get("extract_match", False)
                oracle_correct = any(s.get("extract_match", False) for s in samples)

                steps.append({
                    "step_num": step_data["step_num"],
                    "agreement_rate": agreement_rate,
                    "greedy_correct": greedy_correct,
                    "oracle_correct": oracle_correct,
                    "gt_action_type": step_data.get("gt_action_type", "unknown"),
                })

            episodes.append({
                "episode_id": ep["episode_id"],
                "num_steps": ep["num_steps"],
                "steps": steps,
            })
    return episodes


def load_baseline_data(path):
    """Load baseline trajectory results. Returns dict of episode_id -> episode."""
    baseline = {}
    with open(path) as f:
        for line in f:
            ep = json.loads(line)
            baseline[ep["episode_id"]] = ep
    return baseline


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_pareto_analysis(episodes, baseline):
    """
    Sweep agreement thresholds and compute TSR + cost for each.

    For a given threshold tau:
      - Step with agreement >= tau: pass-through (use greedy, cost=1)
      - Step with agreement <  tau: specialist   (use oracle K, cost=K)

    Episode is successful iff ALL steps are correct:
      - Pass-through step correct iff greedy_correct
      - Specialist step correct iff oracle_correct
    """
    n_episodes = len(episodes)
    total_steps = sum(ep["num_steps"] for ep in episodes)

    # Compute baseline greedy TSR (from baseline file)
    baseline_tsr_count = sum(
        1 for ep in baseline.values() if ep.get("task_success", False)
    )
    baseline_tsr = baseline_tsr_count / len(baseline) * 100

    # Compute full oracle TSR (all steps use oracle K=10)
    oracle_tsr_count = sum(
        1 for ep in episodes
        if all(s["oracle_correct"] for s in ep["steps"])
    )
    oracle_tsr = oracle_tsr_count / n_episodes * 100

    # Compute greedy TSR from multisample data (should match baseline ~closely)
    greedy_tsr_count = sum(
        1 for ep in episodes
        if all(s["greedy_correct"] for s in ep["steps"])
    )
    greedy_tsr = greedy_tsr_count / n_episodes * 100

    print("=" * 80)
    print("E_NEW2: Compute-Efficiency Pareto Analysis")
    print("=" * 80)
    print(f"\nDataset: {n_episodes} episodes, {total_steps} total steps")
    print(f"K = {K} samples per step")
    print(f"\nReference points:")
    print(f"  Greedy TSR (baseline file):     {baseline_tsr:.2f}%  (cost = 1.0 per step)")
    print(f"  Greedy TSR (multisample s[0]):  {greedy_tsr:.2f}%  (cost = 1.0 per step)")
    print(f"  Oracle K={K} TSR:                {oracle_tsr:.2f}%  (cost = {K:.1f} per step)")
    print()

    # Also compute per-step statistics
    all_agreements = [s["agreement_rate"] for ep in episodes for s in ep["steps"]]
    print(f"Agreement rate statistics across {len(all_agreements)} steps:")
    all_agreements_sorted = sorted(all_agreements)
    print(f"  Min:    {min(all_agreements):.3f}")
    print(f"  P25:    {all_agreements_sorted[len(all_agreements_sorted)//4]:.3f}")
    print(f"  Median: {all_agreements_sorted[len(all_agreements_sorted)//2]:.3f}")
    print(f"  P75:    {all_agreements_sorted[3*len(all_agreements_sorted)//4]:.3f}")
    print(f"  Max:    {max(all_agreements):.3f}")
    print(f"  Mean:   {sum(all_agreements)/len(all_agreements):.3f}")
    print()

    # Sweep thresholds
    thresholds = [round(t * 0.05, 2) for t in range(0, 21)]  # 0.00 to 1.00

    results = []
    for tau in thresholds:
        n_passthrough = 0
        n_specialist = 0
        tsr_count = 0

        for ep in episodes:
            all_steps_correct = True
            for step in ep["steps"]:
                if step["agreement_rate"] >= tau:
                    # Pass-through: use greedy
                    n_passthrough += 1
                    if not step["greedy_correct"]:
                        all_steps_correct = False
                else:
                    # Specialist: use oracle K
                    n_specialist += 1
                    if not step["oracle_correct"]:
                        all_steps_correct = False

            if all_steps_correct:
                tsr_count += 1

        tsr = tsr_count / n_episodes * 100
        avg_cost = (n_passthrough * 1 + n_specialist * K) / total_steps
        compute_savings = (1 - avg_cost / K) * 100
        passthrough_pct = n_passthrough / total_steps * 100

        results.append({
            "tau": tau,
            "n_passthrough": n_passthrough,
            "n_specialist": n_specialist,
            "passthrough_pct": round(passthrough_pct, 2),
            "tsr": round(tsr, 2),
            "tsr_count": tsr_count,
            "avg_cost": round(avg_cost, 3),
            "compute_savings_pct": round(compute_savings, 2),
        })

    return results, {
        "baseline_tsr": round(baseline_tsr, 2),
        "greedy_tsr_multisample": round(greedy_tsr, 2),
        "oracle_tsr": round(oracle_tsr, 2),
        "n_episodes": n_episodes,
        "total_steps": total_steps,
        "K": K,
    }


def find_pareto_optimal(results):
    """
    Find Pareto-optimal points: no other point has both higher TSR and lower cost.
    We want to maximize TSR and minimize cost.
    """
    pareto = []
    for r in results:
        dominated = False
        for other in results:
            if other["tsr"] >= r["tsr"] and other["avg_cost"] < r["avg_cost"]:
                if other["tsr"] > r["tsr"] or other["avg_cost"] < r["avg_cost"]:
                    dominated = True
                    break
        if not dominated:
            pareto.append(r["tau"])
    return pareto


def print_results_table(results, pareto_taus, ref):
    """Print a formatted table of results."""
    print("-" * 105)
    print(f"{'tau':>5} | {'Pass-thru':>9} | {'Specialist':>10} | {'PT %':>7} | "
          f"{'TSR':>7} | {'TSR_n':>5} | {'Avg Cost':>8} | {'Savings':>8} | {'Pareto':>6}")
    print("-" * 105)

    for r in results:
        is_pareto = "*" if r["tau"] in pareto_taus else ""
        print(f"{r['tau']:5.2f} | {r['n_passthrough']:9d} | {r['n_specialist']:10d} | "
              f"{r['passthrough_pct']:6.1f}% | {r['tsr']:6.2f}% | {r['tsr_count']:5d} | "
              f"{r['avg_cost']:8.3f} | {r['compute_savings_pct']:7.1f}% | {is_pareto:>6}")

    print("-" * 105)
    print(f"\nReference: Baseline greedy TSR = {ref['baseline_tsr']:.2f}%, "
          f"Oracle K={ref['K']} TSR = {ref['oracle_tsr']:.2f}%")
    print(f"Note: tau=0.00 means ALL steps use specialist (oracle K); "
          f"tau=1.00+ means ALL steps use pass-through (greedy)")


def find_key_operating_points(results, ref):
    """Find interesting operating points on the Pareto curve."""
    print("\n" + "=" * 80)
    print("Key Operating Points")
    print("=" * 80)

    # 1. Point where TSR matches or exceeds baseline greedy, with max savings
    baseline_tsr = ref["baseline_tsr"]
    matching_baseline = [r for r in results if r["tsr"] >= baseline_tsr]
    if matching_baseline:
        best_savings = max(matching_baseline, key=lambda r: r["compute_savings_pct"])
        print(f"\n1. Max savings while matching baseline TSR ({baseline_tsr:.2f}%):")
        print(f"   tau = {best_savings['tau']:.2f}, TSR = {best_savings['tsr']:.2f}%, "
              f"Cost = {best_savings['avg_cost']:.3f}, "
              f"Savings = {best_savings['compute_savings_pct']:.1f}%, "
              f"Pass-through = {best_savings['passthrough_pct']:.1f}%")

    # 2. Point with highest TSR (should be tau=0, full oracle)
    best_tsr = max(results, key=lambda r: r["tsr"])
    print(f"\n2. Maximum TSR:")
    print(f"   tau = {best_tsr['tau']:.2f}, TSR = {best_tsr['tsr']:.2f}%, "
          f"Cost = {best_tsr['avg_cost']:.3f}, "
          f"Savings = {best_tsr['compute_savings_pct']:.1f}%")

    # 3. Best TSR/cost ratio (efficiency)
    best_efficiency = max(results, key=lambda r: r["tsr"] / max(r["avg_cost"], 0.01))
    print(f"\n3. Best TSR/Cost efficiency:")
    print(f"   tau = {best_efficiency['tau']:.2f}, TSR = {best_efficiency['tsr']:.2f}%, "
          f"Cost = {best_efficiency['avg_cost']:.3f}, "
          f"Ratio = {best_efficiency['tsr']/max(best_efficiency['avg_cost'],0.01):.2f}, "
          f"Savings = {best_efficiency['compute_savings_pct']:.1f}%")

    # 4. Point where we lose < 1% TSR vs oracle but save most compute
    oracle_tsr = ref["oracle_tsr"]
    within_1pct = [r for r in results if r["tsr"] >= oracle_tsr - 1.0]
    if within_1pct:
        best_save = max(within_1pct, key=lambda r: r["compute_savings_pct"])
        print(f"\n4. Max savings within 1% of oracle TSR ({oracle_tsr:.2f}%):")
        print(f"   tau = {best_save['tau']:.2f}, TSR = {best_save['tsr']:.2f}%, "
              f"Cost = {best_save['avg_cost']:.3f}, "
              f"Savings = {best_save['compute_savings_pct']:.1f}%, "
              f"TSR drop = {oracle_tsr - best_save['tsr']:.2f}%")

    # 5. 50% compute savings point
    half_cost = [r for r in results if r["compute_savings_pct"] >= 50.0]
    if half_cost:
        best_tsr_at_half = max(half_cost, key=lambda r: r["tsr"])
        print(f"\n5. Best TSR with >= 50% compute savings:")
        print(f"   tau = {best_tsr_at_half['tau']:.2f}, TSR = {best_tsr_at_half['tsr']:.2f}%, "
              f"Cost = {best_tsr_at_half['avg_cost']:.3f}, "
              f"Savings = {best_tsr_at_half['compute_savings_pct']:.1f}%")


def analyze_agreement_vs_accuracy(episodes):
    """Analyze the relationship between agreement rate and greedy accuracy."""
    print("\n" + "=" * 80)
    print("Agreement Rate vs Greedy Accuracy (step-level)")
    print("=" * 80)

    # Bin steps by agreement rate
    bins = {}
    for ep in episodes:
        for step in ep["steps"]:
            ar = step["agreement_rate"]
            # Bin to nearest 0.1
            bin_val = round(round(ar * 10) / 10, 1)
            if bin_val not in bins:
                bins[bin_val] = {"greedy_correct": 0, "oracle_correct": 0, "total": 0}
            bins[bin_val]["total"] += 1
            if step["greedy_correct"]:
                bins[bin_val]["greedy_correct"] += 1
            if step["oracle_correct"]:
                bins[bin_val]["oracle_correct"] += 1

    print(f"\n{'AR bin':>7} | {'N steps':>7} | {'Greedy Acc':>10} | {'Oracle Acc':>10} | {'Oracle lift':>11}")
    print("-" * 62)
    for b in sorted(bins.keys()):
        d = bins[b]
        g_acc = d["greedy_correct"] / d["total"] * 100
        o_acc = d["oracle_correct"] / d["total"] * 100
        lift = o_acc - g_acc
        print(f"  {b:5.1f} | {d['total']:7d} | {g_acc:9.1f}% | {o_acc:9.1f}% | {lift:+10.1f}%")


def main():
    print("Loading multisample data...")
    episodes = load_multisample_data(MULTISAMPLE_PATH)
    print(f"  Loaded {len(episodes)} episodes")

    print("Loading baseline data...")
    baseline = load_baseline_data(BASELINE_PATH)
    print(f"  Loaded {len(baseline)} episodes")
    print()

    # Run Pareto analysis
    results, ref = run_pareto_analysis(episodes, baseline)

    # Find Pareto-optimal points
    pareto_taus = find_pareto_optimal(results)

    # Print table
    print("\nPareto Curve: Agreement Threshold vs TSR vs Compute Cost")
    print("(* = Pareto-optimal point)\n")
    print_results_table(results, pareto_taus, ref)

    # Find key operating points
    find_key_operating_points(results, ref)

    # Agreement vs accuracy analysis
    analyze_agreement_vs_accuracy(episodes)

    # Save results to JSON
    output = {
        "description": "E_NEW2: Compute-Efficiency Pareto Curve via Agreement-based Pass-through Routing",
        "method": (
            "For each step, if agreement_rate >= tau, use greedy (pass-through, cost=1); "
            "otherwise use oracle best-of-K (specialist, cost=K). "
            "Agreement rate = fraction of K samples with the most common canonical action."
        ),
        "reference": ref,
        "pareto_curve": results,
        "pareto_optimal_taus": pareto_taus,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n\nResults saved to: {OUTPUT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
