"""Prepare ALL steps for Cognitive Interference experiment.

Loads all Pattern B trajectory steps (~1916), merges with greedy AR results
for correctness labels and position info. Outputs a JSON file with full step
metadata + GT info needed for all conditions (A, B, C, D).

No sampling — uses all matched steps.
"""

import argparse
import json
import os
import random
from collections import defaultdict


def load_greedy_results(results_path):
    """Load always_greedy results and build per-step lookup."""
    with open(results_path, "r") as f:
        data = json.load(f)

    step_lookup = {}
    for traj in data["detailed_results"]:
        for step in traj["step_results"]:
            step_lookup[step["sample_id"]] = {
                "success": step.get("success", False),
                "trajectory_id": traj["trajectory_id"],
                "domain": traj["domain"],
                "category": traj["category"],
                "step_num": step["step_num"],
                "num_steps": traj["num_steps"],
            }
    return step_lookup


def load_trajectory_data(data_root, trajectory_ids):
    """Load raw trajectory data for GT thought/subtask extraction."""
    id_set = set(trajectory_ids)
    data_path = os.path.join(data_root, "data")
    all_steps = {}

    for domain in sorted(os.listdir(data_path)):
        domain_path = os.path.join(data_path, domain)
        if not os.path.isdir(domain_path):
            continue
        for category in sorted(os.listdir(domain_path)):
            success_path = os.path.join(domain_path, category, "success")
            if not os.path.isdir(success_path):
                continue
            for fname in sorted(os.listdir(success_path)):
                if not fname.endswith(".jsonl"):
                    continue
                file_stem = os.path.splitext(fname)[0]
                traj_id = f"{domain}_{category}_{file_stem}"
                if traj_id not in id_set:
                    continue

                fpath = os.path.join(success_path, fname)
                with open(fpath, "r") as f:
                    for line_num, line in enumerate(f, 1):
                        if not line.strip():
                            continue
                        try:
                            d = json.loads(line.strip())
                        except json.JSONDecodeError:
                            continue

                        action = d["step"]["action"]
                        if action.get("function", "") == "drag" or not action.get("rectangle", {}):
                            continue

                        sample_id = f"{traj_id}_{line_num}"

                        # Build image path
                        clean_img = os.path.join(
                            data_root, "image", domain, category,
                            d["step"]["screenshot_clean"],
                        )

                        status = d["step"]["status"]
                        if status == "OVERALL_FINISH":
                            status = "FINISH"
                        elif status == "FINISH":
                            status = "CONTINUE"

                        all_steps[sample_id] = {
                            "sample_id": sample_id,
                            "trajectory_id": traj_id,
                            "line_num": line_num,
                            "domain": domain,
                            "category": category,
                            "request": d["request"],
                            "screenshot_clean": clean_img,
                            "thought": d["step"].get("thought", ""),
                            "subtask": d["step"].get("subtask", ""),
                            "action": action,
                            "status": status,
                        }

    return all_steps


def assign_position_bucket(step_num, num_steps):
    """Assign early/mid/late based on relative position."""
    if num_steps <= 1:
        return "early"
    ratio = (step_num - 1) / (num_steps - 1)
    if ratio < 0.33:
        return "early"
    elif ratio < 0.67:
        return "mid"
    else:
        return "late"


def stratified_sample(candidates, n_per_domain=100, seed=42):
    """Stratified sampling: domain × position × correctness."""
    random.seed(seed)

    # Group by (domain, position, correct)
    groups = defaultdict(list)
    for c in candidates:
        key = (c["domain"], c["position_bucket"], c["greedy_correct"])
        groups[key].append(c)

    sampled = []
    for domain in ["word", "excel", "ppt"]:
        domain_pool = [c for c in candidates if c["domain"] == domain]
        if not domain_pool:
            print(f"  WARNING: No candidates for domain {domain}")
            continue

        # Target: ~33 per position, ~50/50 correct/wrong
        target_per_cell = n_per_domain // 6  # 6 cells = 3 positions × 2 correctness
        remainder = n_per_domain - target_per_cell * 6

        domain_sampled = []
        for pos in ["early", "mid", "late"]:
            for correct in [True, False]:
                key = (domain, pos, correct)
                pool = groups.get(key, [])
                n_take = min(target_per_cell, len(pool))
                domain_sampled.extend(random.sample(pool, n_take))

        # Fill remainder from largest remaining pools
        used_ids = {s["sample_id"] for s in domain_sampled}
        remaining = [c for c in domain_pool if c["sample_id"] not in used_ids]
        n_fill = min(n_per_domain - len(domain_sampled), len(remaining))
        if n_fill > 0:
            domain_sampled.extend(random.sample(remaining, n_fill))

        sampled.extend(domain_sampled[:n_per_domain])

        # Stats
        n_correct = sum(1 for s in domain_sampled[:n_per_domain] if s["greedy_correct"])
        pos_counts = defaultdict(int)
        for s in domain_sampled[:n_per_domain]:
            pos_counts[s["position_bucket"]] += 1
        print(f"  {domain}: {len(domain_sampled[:n_per_domain])} samples, "
              f"{n_correct} correct, positions: {dict(pos_counts)}")

    return sampled


def main():
    parser = argparse.ArgumentParser(description="Prepare all steps for cognitive interference experiment")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--greedy_results", type=str, required=True,
                        help="Path to verifier_always_greedy_results.json")
    parser.add_argument("--trajectory_ids", type=str, required=True,
                        help="Path to pattern_b_ids.json")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path for all steps")
    args = parser.parse_args()

    # Load trajectory IDs
    with open(args.trajectory_ids) as f:
        traj_ids = json.load(f)
    print(f"Pattern B trajectories: {len(traj_ids)}")

    # Load greedy results
    print("Loading greedy AR results...")
    greedy_lookup = load_greedy_results(args.greedy_results)
    print(f"  {len(greedy_lookup)} step results")

    # Load raw trajectory data
    print("Loading raw trajectory data...")
    raw_steps = load_trajectory_data(args.data_root, traj_ids)
    print(f"  {len(raw_steps)} raw steps")

    # Merge: only keep steps that appear in both
    candidates = []
    for sample_id, raw in raw_steps.items():
        if sample_id not in greedy_lookup:
            continue
        greedy = greedy_lookup[sample_id]
        candidates.append({
            **raw,
            "greedy_correct": greedy["success"],
            "step_num": greedy["step_num"],
            "num_steps": greedy["num_steps"],
            "position_bucket": assign_position_bucket(greedy["step_num"], greedy["num_steps"]),
        })

    # Sort by trajectory_id, then step_num for deterministic ordering
    candidates.sort(key=lambda c: (c["trajectory_id"], c["step_num"]))

    print(f"Total matched steps: {len(candidates)}")

    # Domain distribution
    domain_counts = defaultdict(int)
    for c in candidates:
        domain_counts[c["domain"]] += 1
    print(f"By domain: {dict(domain_counts)}")

    # Position distribution
    pos_counts = defaultdict(int)
    for c in candidates:
        pos_counts[c["position_bucket"]] += 1
    print(f"By position: {dict(pos_counts)}")

    # Correctness
    n_correct = sum(1 for c in candidates if c["greedy_correct"])
    print(f"Correct/Wrong: {n_correct}/{len(candidates) - n_correct}")

    # Save ALL steps (no sampling)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(candidates, f, indent=2, default=str)
    print(f"Saved {len(candidates)} steps to {args.output}")


if __name__ == "__main__":
    main()
