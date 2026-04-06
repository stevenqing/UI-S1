#!/usr/bin/env python3
"""
Exp E2-offline: Observer Value × Trajectory Length Analysis

Core hypothesis: Observer value should be concentrated in longer trajectories,
because state tracking problems dominate in long sequences (D7/D8 findings).

Compares:
- Eval A condition B (V2+V3, no Observer): baseline
- D1 (V2+V3+Observer): observer-enhanced
- Per-length-bucket difference = Observer value

Also validates:
- Shapley value shift: V2 dominates short, V3+Observer dominates long
"""

import json
import os
import argparse
from collections import defaultdict
import numpy as np


def load_eval_a(path):
    """Load eval_a results, return dict by trajectory_id."""
    results = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                results[d["trajectory_id"]] = d
    return results


def load_d1(path):
    """Load D1 results (new format only), return dict by trajectory_id."""
    results = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                # Only use new format (has trajectory_success)
                if "trajectory_success" in d:
                    results[d["trajectory_id"]] = d
    return results


def length_bucket(num_steps):
    if num_steps <= 3:
        return "short (1-3)"
    elif num_steps <= 7:
        return "medium (4-7)"
    elif num_steps <= 15:
        return "long (8-15)"
    else:
        return "vlong (16+)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_a_dir", default="outputs/eval_a")
    parser.add_argument("--d1_dir", default="outputs/eval_d1")
    parser.add_argument("--output_dir", default="outputs/eval_e2_offline")
    args = parser.parse_args()

    eval_a = load_eval_a(os.path.join(args.eval_a_dir, "trajectory_results.jsonl"))
    d1 = load_d1(os.path.join(args.d1_dir, "trajectory_results.jsonl"))

    print(f"Eval A trajectories: {len(eval_a)}")
    print(f"D1 trajectories (new format): {len(d1)}")

    # Match trajectories
    matched_ids = set(eval_a.keys()) & set(d1.keys())
    print(f"Matched: {len(matched_ids)}")

    # Per-length analysis
    buckets = defaultdict(lambda: {
        "n": 0,
        "eval_a_v2only_success": 0,  # condition A: V2 only
        "eval_a_v2v3_success": 0,     # condition B: V2+V3
        "d1_observer_success": 0,      # D1: V2+V3+Observer
        "eval_a_v2v3_progress": [],
        "d1_observer_progress": [],
        # Per-trajectory comparison
        "observer_wins": 0,   # D1 success, eval_a fail
        "observer_loses": 0,  # D1 fail, eval_a success
        "both_success": 0,
        "both_fail": 0,
    })

    for tid in sorted(matched_ids):
        a = eval_a[tid]
        d = d1[tid]

        num_steps = a["num_steps"]
        bucket = length_bucket(num_steps)
        b = buckets[bucket]
        b["n"] += 1

        a_v2only = a["a_trajectory_success"]
        a_v2v3 = a["b_trajectory_success"]
        d1_obs = d["trajectory_success"]

        if a_v2only:
            b["eval_a_v2only_success"] += 1
        if a_v2v3:
            b["eval_a_v2v3_success"] += 1
        if d1_obs:
            b["d1_observer_success"] += 1

        b["eval_a_v2v3_progress"].append(a["b_progress_rate"])
        b["d1_observer_progress"].append(d["progress_rate"])

        if d1_obs and not a_v2v3:
            b["observer_wins"] += 1
        elif not d1_obs and a_v2v3:
            b["observer_loses"] += 1
        elif d1_obs and a_v2v3:
            b["both_success"] += 1
        else:
            b["both_fail"] += 1

    # Print results
    print(f"\n{'='*90}")
    print(f"OBSERVER VALUE × TRAJECTORY LENGTH")
    print(f"{'='*90}")

    bucket_order = ["short (1-3)", "medium (4-7)", "long (8-15)", "vlong (16+)"]

    print(f"\n{'Bucket':<16} {'N':>5} {'V2-only':>8} {'V2+V3':>8} {'Observer':>8} {'Δ(Obs-V2V3)':>12} {'Obs win':>8} {'Obs lose':>9}")
    print("-" * 82)

    overall = {"n": 0, "v2only": 0, "v2v3": 0, "obs": 0, "wins": 0, "loses": 0}

    for bucket_name in bucket_order:
        b = buckets[bucket_name]
        n = b["n"]
        if n == 0:
            continue

        v2only_tsr = 100 * b["eval_a_v2only_success"] / n
        v2v3_tsr = 100 * b["eval_a_v2v3_success"] / n
        obs_tsr = 100 * b["d1_observer_success"] / n
        delta = obs_tsr - v2v3_tsr
        win_ratio = f"{b['observer_wins']}:{b['observer_loses']}"

        print(f"{bucket_name:<16} {n:>5} {v2only_tsr:>7.1f}% {v2v3_tsr:>7.1f}% {obs_tsr:>7.1f}% {delta:>+10.1f}pp {win_ratio:>8}")

        overall["n"] += n
        overall["v2only"] += b["eval_a_v2only_success"]
        overall["v2v3"] += b["eval_a_v2v3_success"]
        overall["obs"] += b["d1_observer_success"]
        overall["wins"] += b["observer_wins"]
        overall["loses"] += b["observer_loses"]

    n = overall["n"]
    print("-" * 82)
    print(f"{'Overall':<16} {n:>5} {100*overall['v2only']/n:>7.1f}% {100*overall['v2v3']/n:>7.1f}% {100*overall['obs']/n:>7.1f}% {100*(overall['obs']-overall['v2v3'])/n:>+10.1f}pp {overall['wins']}:{overall['loses']:>6}")

    # Progress rate analysis
    print(f"\n{'='*90}")
    print(f"PROGRESS RATE ANALYSIS (average progress rate per bucket)")
    print(f"{'='*90}")

    print(f"\n{'Bucket':<16} {'N':>5} {'V2+V3 prog':>12} {'Observer prog':>13} {'Δ prog':>8}")
    print("-" * 58)

    for bucket_name in bucket_order:
        b = buckets[bucket_name]
        n = b["n"]
        if n == 0:
            continue

        v2v3_prog = np.mean(b["eval_a_v2v3_progress"])
        obs_prog = np.mean(b["d1_observer_progress"])
        delta_prog = obs_prog - v2v3_prog

        print(f"{bucket_name:<16} {n:>5} {100*v2v3_prog:>11.1f}% {100*obs_prog:>12.1f}% {100*delta_prog:>+7.1f}pp")

    # Shapley-style contribution analysis
    print(f"\n{'='*90}")
    print(f"SHAPLEY-STYLE CONTRIBUTION ANALYSIS")
    print(f"{'='*90}")
    print("Marginal contributions of each component:")

    print(f"\n{'Bucket':<16} {'V3 marginal':>12} {'Obs marginal':>13} {'V3 share':>10} {'Obs share':>10}")
    print("-" * 65)

    for bucket_name in bucket_order:
        b = buckets[bucket_name]
        n = b["n"]
        if n == 0:
            continue

        v2only_tsr = b["eval_a_v2only_success"] / n
        v2v3_tsr = b["eval_a_v2v3_success"] / n
        obs_tsr = b["d1_observer_success"] / n

        v3_marginal = v2v3_tsr - v2only_tsr  # V3's contribution on top of V2
        obs_marginal = obs_tsr - v2v3_tsr     # Observer's contribution on top of V2+V3

        total_improvement = obs_tsr - v2only_tsr
        if total_improvement > 0:
            v3_share = v3_marginal / total_improvement
            obs_share = obs_marginal / total_improvement
        else:
            v3_share = obs_share = 0

        print(f"{bucket_name:<16} {100*v3_marginal:>+11.1f}pp {100*obs_marginal:>+12.1f}pp {100*v3_share:>9.0f}% {100*obs_share:>9.0f}%")

    # Domain × Length breakdown
    print(f"\n{'='*90}")
    print(f"DOMAIN × LENGTH: OBSERVER DELTA")
    print(f"{'='*90}")

    domain_length = defaultdict(lambda: {"n": 0, "v2v3_success": 0, "obs_success": 0})
    for tid in matched_ids:
        a = eval_a[tid]
        d = d1[tid]
        key = (a["domain"], length_bucket(a["num_steps"]))
        domain_length[key]["n"] += 1
        if a["b_trajectory_success"]:
            domain_length[key]["v2v3_success"] += 1
        if d["trajectory_success"]:
            domain_length[key]["obs_success"] += 1

    for domain in ["excel", "ppt", "word"]:
        print(f"\n  {domain.upper()}:")
        for bucket_name in bucket_order:
            key = (domain, bucket_name)
            stats = domain_length[key]
            n = stats["n"]
            if n == 0:
                continue
            v2v3 = 100 * stats["v2v3_success"] / n
            obs = 100 * stats["obs_success"] / n
            delta = obs - v2v3
            print(f"    {bucket_name:<16} n={n:>4}  V2V3={v2v3:.1f}%  Obs={obs:.1f}%  Δ={delta:+.1f}pp")

    # Save summary
    os.makedirs(args.output_dir, exist_ok=True)
    summary = {}
    for bucket_name in bucket_order:
        b = buckets[bucket_name]
        n = b["n"]
        if n == 0:
            continue
        summary[bucket_name] = {
            "n": n,
            "v2only_tsr": b["eval_a_v2only_success"] / n,
            "v2v3_tsr": b["eval_a_v2v3_success"] / n,
            "observer_tsr": b["d1_observer_success"] / n,
            "observer_delta": (b["d1_observer_success"] - b["eval_a_v2v3_success"]) / n,
            "observer_wins": b["observer_wins"],
            "observer_loses": b["observer_loses"],
            "v2v3_avg_progress": float(np.mean(b["eval_a_v2v3_progress"])),
            "observer_avg_progress": float(np.mean(b["d1_observer_progress"])),
        }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
