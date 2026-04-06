#!/usr/bin/env python3
"""
Experiment 0.3: Oracle Recovery Upper Bound

For paired success/fail trajectories, simulate oracle recovery:
- At the divergence step, replace the fail action with the success action
- Measure how many fail trajectories could be "saved" by perfect recovery

Three recovery modes:
1. 1-step oracle: Replace exactly 1 action (at divergence point)
2. 2-step oracle: Replace up to 2 actions
3. Perfect oracle: Replace all actions from divergence onward

This is a pure data analysis experiment (no model inference needed).

Success criteria: 1-step oracle recovery saves >20% of fail trajectories

Usage:
    python scripts/exp0/exp0_3_oracle_recovery.py --max_pairs 500
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
    coordinate_distance,
    find_divergence_step,
    load_paired_trajectories,
    normalize_action,
)


def analyze_divergence(pair: dict) -> dict:
    """
    Analyze a single paired trajectory for oracle recovery potential.

    Returns dict with divergence analysis:
        - divergence_step: int
        - total_success_steps: int
        - total_fail_steps: int
        - divergence_type: str ('coordinate', 'action_type', 'text', 'end_of_trajectory')
        - success_action_at_div: dict (normalized)
        - fail_action_at_div: dict (normalized)
        - coordinate_distance_at_div: float
        - steps_wasted_after_div: int (fail steps after divergence)
        - sub_scores_at_div: dict (if available)
    """
    s_steps = pair["success_steps"]
    f_steps = pair["fail_steps"]
    div_step = find_divergence_step(s_steps, f_steps)

    result = {
        "execution_id": pair["execution_id"],
        "domain": pair["domain"],
        "request": pair["request"][:200],
        "divergence_step": div_step,
        "total_success_steps": len(s_steps),
        "total_fail_steps": len(f_steps),
        "steps_wasted_after_div": max(0, len(f_steps) - div_step - 1),
    }

    # Analyze divergence type
    if div_step >= len(s_steps) or div_step >= len(f_steps):
        result["divergence_type"] = "end_of_trajectory"
        return result

    s_action = normalize_action(s_steps[div_step])
    f_action = normalize_action(f_steps[div_step])

    result["success_action_at_div"] = {
        "type": s_action["action_type"],
        "coordinate": s_action["coordinate"],
    }
    result["fail_action_at_div"] = {
        "type": f_action["action_type"],
        "coordinate": f_action["coordinate"],
    }

    if s_action["action_type"] != f_action["action_type"]:
        result["divergence_type"] = "action_type"
    elif s_action["action_type"] == "type":
        result["divergence_type"] = "text"
    else:
        result["divergence_type"] = "coordinate"

    result["coordinate_distance_at_div"] = coordinate_distance(
        s_action["coordinate"], f_action["coordinate"]
    )

    # Get sub_scores if available
    eval_info = f_steps[div_step].get("evaluation", {})
    result["sub_scores_at_div"] = eval_info.get("sub_scores", {})

    return result


def simulate_oracle_recovery(pair: dict, div_analysis: dict) -> dict:
    """
    Simulate oracle recovery at different levels.

    For each recovery level, determine if replacing N actions at the divergence
    point with success actions would allow the trajectory to "re-align".

    Re-alignment is determined by:
    - After replacement, do the subsequent fail steps match success steps?
    - We measure by checking if the next non-replaced fail step matches
      the corresponding success step.
    """
    s_steps = pair["success_steps"]
    f_steps = pair["fail_steps"]
    div_step = div_analysis["divergence_step"]

    result = {
        "execution_id": pair["execution_id"],
        "divergence_step": div_step,
    }

    # Helper: check if step i in fail trajectory matches step i in success
    def steps_match(f_idx, s_idx, threshold=50.0):
        if f_idx >= len(f_steps) or s_idx >= len(s_steps):
            return s_idx >= len(s_steps)  # Both finished → match
        f_action = normalize_action(f_steps[f_idx])
        s_action = normalize_action(s_steps[s_idx])

        if f_action["action_type"] != s_action["action_type"]:
            return False
        dist = coordinate_distance(f_action["coordinate"], s_action["coordinate"])
        if dist < threshold:
            return True
        if f_action["action_type"] == "type":
            f_text = (f_action.get("text") or "").lower().strip()
            s_text = (s_action.get("text") or "").lower().strip()
            return f_text == s_text
        return False

    # 1-step oracle: replace action at div_step only
    # Check if the step AFTER div_step in fail matches div_step+1 in success
    if div_step + 1 < len(f_steps) and div_step + 1 < len(s_steps):
        result["1step_realigns"] = steps_match(div_step + 1, div_step + 1)
    elif div_step + 1 >= len(s_steps):
        # Success was about to end → replacing last fail step with success action = done
        result["1step_realigns"] = True
    else:
        result["1step_realigns"] = False

    # 2-step oracle: replace actions at div_step and div_step+1
    if div_step + 2 < len(f_steps) and div_step + 2 < len(s_steps):
        result["2step_realigns"] = steps_match(div_step + 2, div_step + 2)
    elif div_step + 2 >= len(s_steps):
        result["2step_realigns"] = True
    else:
        result["2step_realigns"] = False

    # Perfect oracle: replace ALL actions from div_step onward
    # This always succeeds if we have the success trajectory
    result["perfect_realigns"] = True

    # Compute "recovery potential" = steps saved / total fail steps
    total_fail = len(f_steps)
    steps_after_div = total_fail - div_step - 1
    result["steps_saved_1step"] = steps_after_div - 1 if result["1step_realigns"] else 0
    result["steps_saved_2step"] = steps_after_div - 2 if result["2step_realigns"] else 0
    result["steps_saved_perfect"] = steps_after_div
    result["wasted_steps"] = steps_after_div

    # Check if divergence is in early portion (first 20% of success trajectory)
    result["early_divergence"] = div_step < max(1, len(s_steps) * 0.2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Experiment 0.3: Oracle Recovery Upper Bound")
    parser.add_argument("--max_pairs", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "exp0_3"))

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load paired trajectories
    print(f"Loading up to {args.max_pairs} paired trajectories...")
    pairs = load_paired_trajectories(max_pairs=args.max_pairs)
    print(f"Loaded {len(pairs)} pairs")

    if not pairs:
        print("ERROR: No paired trajectories found!")
        print("Make sure both success and fail trajectory data exist in datasets/GUI-360/")
        return

    # Analyze each pair
    div_analyses = []
    recovery_results = []

    for pair in pairs:
        div = analyze_divergence(pair)
        div_analyses.append(div)

        recovery = simulate_oracle_recovery(pair, div)
        recovery_results.append(recovery)

    # Save raw results
    with open(output_dir / "divergence_analysis.jsonl", "w") as f:
        for d in div_analyses:
            f.write(json.dumps(d, default=str) + "\n")

    with open(output_dir / "recovery_results.jsonl", "w") as f:
        for r in recovery_results:
            f.write(json.dumps(r, default=str) + "\n")

    # === Analysis ===
    n = len(pairs)
    print("\n" + "=" * 60)
    print(f"  Experiment 0.3: Oracle Recovery Upper Bound (N={n})")
    print("=" * 60)

    # 1. Divergence statistics
    div_steps = [d["divergence_step"] for d in div_analyses]
    s_lengths = [d["total_success_steps"] for d in div_analyses]
    f_lengths = [d["total_fail_steps"] for d in div_analyses]
    wasted = [d["steps_wasted_after_div"] for d in div_analyses]

    print(f"\n  Trajectory Statistics:")
    print(f"    Mean success length:   {np.mean(s_lengths):.1f} steps")
    print(f"    Mean fail length:      {np.mean(f_lengths):.1f} steps")
    print(f"    Mean divergence step:  {np.mean(div_steps):.1f}")
    print(f"    Mean wasted steps:     {np.mean(wasted):.1f}")
    print(f"    Fail/Success ratio:    {np.mean(f_lengths) / np.mean(s_lengths):.2f}x")

    # 2. Divergence timing distribution
    rel_div = [d["divergence_step"] / max(d["total_success_steps"], 1) for d in div_analyses]
    early = sum(1 for r in rel_div if r < 0.2)
    mid = sum(1 for r in rel_div if 0.2 <= r < 0.5)
    late = sum(1 for r in rel_div if r >= 0.5)

    print(f"\n  Divergence Timing (relative to success length):")
    print(f"    Early (<20%):  {early}/{n} ({early / n:.1%})")
    print(f"    Mid (20-50%):  {mid}/{n} ({mid / n:.1%})")
    print(f"    Late (≥50%):   {late}/{n} ({late / n:.1%})")

    # 3. Divergence type
    div_types = [d["divergence_type"] for d in div_analyses]
    print(f"\n  Divergence Types:")
    for dtype, count in sorted(set((t, div_types.count(t)) for t in set(div_types)),
                                key=lambda x: -x[1]):
        print(f"    {dtype}: {count}/{n} ({count / n:.1%})")

    # 4. Oracle recovery rates
    r1 = sum(1 for r in recovery_results if r["1step_realigns"])
    r2 = sum(1 for r in recovery_results if r["2step_realigns"])
    rp = sum(1 for r in recovery_results if r["perfect_realigns"])

    print(f"\n  Oracle Recovery Rates:")
    print(f"    1-step oracle: {r1}/{n} ({r1 / n:.1%})")
    print(f"    2-step oracle: {r2}/{n} ({r2 / n:.1%})")
    print(f"    Perfect oracle: {rp}/{n} ({rp / n:.1%})")

    # 5. Steps saved
    total_wasted = sum(r["wasted_steps"] for r in recovery_results)
    saved_1 = sum(r["steps_saved_1step"] for r in recovery_results)
    saved_2 = sum(r["steps_saved_2step"] for r in recovery_results)

    print(f"\n  Steps Saved:")
    print(f"    Total wasted steps:    {total_wasted}")
    print(f"    1-step oracle saves:   {saved_1} ({saved_1 / max(total_wasted, 1):.1%} of wasted)")
    print(f"    2-step oracle saves:   {saved_2} ({saved_2 / max(total_wasted, 1):.1%} of wasted)")

    # 6. Coordinate distance at divergence
    coord_dists = [d["coordinate_distance_at_div"]
                   for d in div_analyses
                   if d.get("coordinate_distance_at_div") is not None
                   and d["coordinate_distance_at_div"] < float("inf")]
    if coord_dists:
        print(f"\n  Coordinate Distance at Divergence:")
        print(f"    Mean: {np.mean(coord_dists):.1f}px  Median: {np.median(coord_dists):.1f}px")
        print(f"    <50px:  {sum(1 for d in coord_dists if d < 50)}")
        print(f"    50-100: {sum(1 for d in coord_dists if 50 <= d < 100)}")
        print(f"    100-200: {sum(1 for d in coord_dists if 100 <= d < 200)}")
        print(f"    >200px: {sum(1 for d in coord_dists if d >= 200)}")

    # 7. Per-domain breakdown
    print(f"\n  Per-Domain Recovery (1-step oracle):")
    domains = set(d["domain"] for d in div_analyses)
    for domain in sorted(domains):
        domain_indices = [i for i, d in enumerate(div_analyses) if d["domain"] == domain]
        if domain_indices:
            domain_recovered = sum(1 for i in domain_indices if recovery_results[i]["1step_realigns"])
            print(f"    {domain}: {domain_recovered}/{len(domain_indices)} "
                  f"({domain_recovered / len(domain_indices):.1%})")

    # 8. Early divergence analysis
    early_indices = [i for i, r in enumerate(recovery_results) if r["early_divergence"]]
    if early_indices:
        early_recovered = sum(1 for i in early_indices if recovery_results[i]["1step_realigns"])
        print(f"\n  Early Divergence (<20% of trajectory):")
        print(f"    Count: {len(early_indices)}/{n} ({len(early_indices) / n:.1%})")
        print(f"    1-step recovery: {early_recovered}/{len(early_indices)} "
              f"({early_recovered / len(early_indices):.1%})")

    # Verdict
    print(f"\n  VERDICT: {'PASS' if r1 / n > 0.2 else 'BELOW THRESHOLD'}")
    if r1 / n > 0.2:
        print(f"    1-step oracle recovery rate {r1 / n:.1%} > 20% threshold!")
        print(f"    Recovery mechanism has significant potential.")
    else:
        print(f"    1-step oracle recovery rate {r1 / n:.1%} ≤ 20%.")
        print(f"    Consider multi-step recovery or different recovery strategies.")

    print("=" * 60)

    # Save summary
    summary = {
        "n_pairs": n,
        "mean_divergence_step": float(np.mean(div_steps)),
        "mean_success_length": float(np.mean(s_lengths)),
        "mean_fail_length": float(np.mean(f_lengths)),
        "fail_success_ratio": float(np.mean(f_lengths) / np.mean(s_lengths)),
        "early_divergence_rate": early / n,
        "recovery_rate_1step": r1 / n,
        "recovery_rate_2step": r2 / n,
        "recovery_rate_perfect": rp / n,
        "mean_wasted_steps": float(np.mean(wasted)),
        "coordinate_divergence_types": {t: div_types.count(t) for t in set(div_types)},
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
