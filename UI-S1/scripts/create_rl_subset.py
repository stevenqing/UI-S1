#!/usr/bin/env python3
"""
Create a stratified subset of GUI-360 RL training data for fast validation.

Strategy:
1. Stratify by domain (excel/ppt/word) × length bucket (short/medium/long/vlong)
2. Oversample trajectories containing rare actions (summary, save_as, insert_table, etc.)
3. Target ~2000 trajectories (~15% of 13,750)
4. Output JSONL references same image paths — no image copying needed

Usage:
    python scripts/create_rl_subset.py --target_size 2000
    python scripts/create_rl_subset.py --target_size 3000 --seed 42
"""

import argparse
import json
import os
import random
from collections import defaultdict, Counter
import numpy as np


RARE_ACTIONS = {
    "summary", "save_as", "insert_table", "insert_excel_table",
    "select_table", "set_background_color", "set_focus",
    "select_paragraph", "set_font", "table2markdown",
    "reorder_columns", "wheel_mouse_input", "drag",
}

# Minimum 3x oversample for trajectories with rare actions
RARE_OVERSAMPLE = 3.0


def get_domain(traj):
    eid = traj.get("execution_id", "")
    screenshot = traj["steps"][0]["screenshot"] if traj["steps"] else ""
    for dom in ["excel", "word", "ppt"]:
        if dom in eid.lower() or dom in screenshot.lower():
            return dom
    return "unknown"


def get_length_bucket(n_steps):
    if n_steps <= 3:
        return "short"
    elif n_steps <= 7:
        return "medium"
    elif n_steps <= 15:
        return "long"
    else:
        return "vlong"


def has_rare_action(traj):
    for step in traj["steps"]:
        ac = step.get("action_content", {})
        if ac.get("action", "") in RARE_ACTIONS:
            return True
    return False


def get_action_set(traj):
    actions = set()
    for step in traj["steps"]:
        ac = step.get("action_content", {})
        actions.add(ac.get("action", ""))
    return actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="datasets/GUI-360/rl_data/gui360_train.jsonl")
    parser.add_argument("--output_dir", default="datasets/GUI-360/rl_data")
    parser.add_argument("--target_size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load all trajectories
    print(f"Loading {args.input}...")
    trajectories = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))
    print(f"Loaded {len(trajectories)} trajectories")

    # Categorize
    buckets = defaultdict(list)  # (domain, length_bucket) → [idx, ...]
    rare_indices = set()

    for i, traj in enumerate(trajectories):
        domain = get_domain(traj)
        n_steps = len(traj["steps"])
        lb = get_length_bucket(n_steps)
        buckets[(domain, lb)].append(i)

        if has_rare_action(traj):
            rare_indices.add(i)

    # Print current distribution
    print(f"\nCurrent distribution:")
    print(f"{'Domain':<8} {'short':>7} {'medium':>7} {'long':>7} {'vlong':>7} {'total':>7}")
    print("-" * 47)
    for domain in ["excel", "ppt", "word"]:
        counts = [len(buckets[(domain, lb)]) for lb in ["short", "medium", "long", "vlong"]]
        print(f"{domain:<8} {counts[0]:>7} {counts[1]:>7} {counts[2]:>7} {counts[3]:>7} {sum(counts):>7}")

    print(f"\nTrajectories with rare actions: {len(rare_indices)} ({100*len(rare_indices)/len(trajectories):.1f}%)")

    # Compute sampling weights
    # Base: proportional to current distribution
    # Rare action trajectories get RARE_OVERSAMPLE weight
    total = len(trajectories)
    target = args.target_size
    base_rate = target / total  # ~0.145 for 2000/13750

    # First, guarantee all rare action trajectories
    selected = set()
    # Always include rare action trajectories (up to target/3)
    rare_list = list(rare_indices)
    random.shuffle(rare_list)
    max_rare = min(len(rare_list), target // 3)
    for idx in rare_list[:max_rare]:
        selected.add(idx)
    print(f"Pre-selected {len(selected)} rare-action trajectories")

    # Remaining budget: stratified sampling
    remaining = target - len(selected)

    # Compute per-bucket allocation (proportional to current size)
    bucket_sizes = {k: len(v) for k, v in buckets.items()}
    total_non_selected = sum(
        sum(1 for idx in v if idx not in selected) for v in buckets.values()
    )

    for key in sorted(buckets.keys()):
        available = [idx for idx in buckets[key] if idx not in selected]
        if not available:
            continue

        # Proportional allocation
        alloc = int(remaining * len(available) / total_non_selected)
        alloc = max(alloc, 1)  # at least 1 per bucket
        alloc = min(alloc, len(available))

        sampled = random.sample(available, alloc)
        for idx in sampled:
            selected.add(idx)

    # If still under target, random fill
    while len(selected) < target:
        idx = random.randint(0, len(trajectories) - 1)
        if idx not in selected:
            selected.add(idx)

    # If over target, trim random non-rare
    while len(selected) > target:
        non_rare_selected = [idx for idx in selected if idx not in rare_indices]
        if non_rare_selected:
            remove = random.choice(non_rare_selected)
            selected.remove(remove)
        else:
            break

    # Collect and write
    selected_trajs = [trajectories[i] for i in sorted(selected)]

    # Analyze subset
    subset_stats = defaultdict(lambda: {"count": 0, "steps": 0, "actions": Counter()})
    for traj in selected_trajs:
        domain = get_domain(traj)
        n_steps = len(traj["steps"])
        lb = get_length_bucket(n_steps)
        key = (domain, lb)
        subset_stats[domain]["count"] += 1
        subset_stats[domain]["steps"] += n_steps
        for step in traj["steps"]:
            ac = step.get("action_content", {})
            subset_stats[domain]["actions"][ac.get("action", "")] += 1

    print(f"\nSubset distribution ({len(selected_trajs)} trajectories):")
    print(f"{'Domain':<8} {'Trajs':>6} {'Steps':>7} {'Avg len':>8}")
    print("-" * 32)
    total_steps = 0
    for domain in ["excel", "ppt", "word"]:
        s = subset_stats[domain]
        total_steps += s["steps"]
        avg = s["steps"] / s["count"] if s["count"] > 0 else 0
        print(f"{domain:<8} {s['count']:>6} {s['steps']:>7} {avg:>8.1f}")
    print(f"{'TOTAL':<8} {len(selected_trajs):>6} {total_steps:>7}")

    # Rare action coverage
    all_subset_actions = Counter()
    for s in subset_stats.values():
        all_subset_actions.update(s["actions"])

    all_full_actions = Counter()
    for traj in trajectories:
        for step in traj["steps"]:
            ac = step.get("action_content", {})
            all_full_actions[ac.get("action", "")] += 1

    print(f"\nRare action coverage:")
    print(f"{'Action':<25} {'Full':>6} {'Subset':>7} {'Coverage':>9}")
    print("-" * 50)
    for action in sorted(RARE_ACTIONS):
        full = all_full_actions.get(action, 0)
        sub = all_subset_actions.get(action, 0)
        cov = 100 * sub / full if full > 0 else 0
        if full > 0:
            print(f"{action:<25} {full:>6} {sub:>7} {cov:>8.1f}%")

    # Length bucket coverage
    print(f"\nLength bucket distribution:")
    full_lb = Counter()
    sub_lb = Counter()
    for traj in trajectories:
        full_lb[get_length_bucket(len(traj["steps"]))] += 1
    for traj in selected_trajs:
        sub_lb[get_length_bucket(len(traj["steps"]))] += 1

    print(f"{'Bucket':<10} {'Full':>6} {'Subset':>7} {'Full %':>7} {'Sub %':>7}")
    print("-" * 40)
    for lb in ["short", "medium", "long", "vlong"]:
        full_pct = 100 * full_lb[lb] / len(trajectories)
        sub_pct = 100 * sub_lb[lb] / len(selected_trajs)
        print(f"{lb:<10} {full_lb[lb]:>6} {sub_lb[lb]:>7} {full_pct:>6.1f}% {sub_pct:>6.1f}%")

    # Write output
    output_file = os.path.join(args.output_dir, f"gui360_train_subset_{args.target_size}.jsonl")
    with open(output_file, "w") as f:
        for traj in selected_trajs:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")

    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\nWritten to: {output_file}")
    print(f"File size: {file_size:.1f} MB")
    print(f"Compression ratio: {len(selected_trajs)}/{len(trajectories)} = {100*len(selected_trajs)/len(trajectories):.1f}%")


if __name__ == "__main__":
    main()
