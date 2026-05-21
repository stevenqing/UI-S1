#!/usr/bin/env python3
"""Prepare AndroidControl episode-grouped data for v11 trajectory GRPO.

Reads the episode-level source data (ui_s1_train_with_desc.jsonl) and outputs
episode-grouped JSONL where each line is one episode with ordered steps.

Source format (per line):
    {"goal": "...", "is_successful": true, "steps": [
        {"action_content": {...}, "screenshot": "path.png", "desc_t1": "..."},
        ...
    ]}

Output format (per line):
    {"episode_id": 0, "goal": "...", "num_steps": 4, "steps": [
        {"step_idx": 0, "action": {...}, "screenshot": "path.png",
         "image_w": 1080, "image_h": 2400},
        ...
    ]}

Usage:
    python v11/prepare_episode_data.py [--input ...] [--train_output ...] [--val_output ...]
"""

import argparse
import json
import os
import random
import sys

from PIL import Image
from tqdm import tqdm


def clean_action(action_content):
    """Remove null fields from action dict."""
    if not isinstance(action_content, dict):
        return action_content
    return {k: v for k, v in action_content.items() if v is not None}


def process_episode(episode, episode_id):
    """Convert one raw episode to the v11 training format.

    Returns None if the episode has missing screenshots or no steps.
    """
    goal = episode.get("goal", "")
    steps_raw = episode.get("steps", [])
    if not steps_raw or not goal:
        return None

    steps = []
    for si, step in enumerate(steps_raw):
        screenshot = step.get("screenshot", "")
        if not screenshot or not os.path.exists(screenshot):
            return None  # skip episodes with any missing screenshot

        action_content = clean_action(step.get("action_content", {}))
        if not action_content:
            return None

        # Get image dimensions (needed for coordinate normalization in reward)
        try:
            with Image.open(screenshot) as img:
                image_w, image_h = img.size
        except Exception:
            return None

        steps.append({
            "step_idx": si,
            "action": action_content,
            "screenshot": screenshot,
            "image_w": image_w,
            "image_h": image_h,
        })

    return {
        "episode_id": episode_id,
        "goal": goal,
        "num_steps": len(steps),
        "steps": steps,
    }


def write_jsonl(path, samples, desc="Writing"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for s in tqdm(samples, desc=desc):
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def print_stats(samples, label):
    if not samples:
        print(f"\n=== {label}: 0 episodes ===")
        return
    n = len(samples)
    n_steps = [s["num_steps"] for s in samples]
    print(f"\n=== {label} ===")
    print(f"  episodes:       {n}")
    print(f"  total steps:    {sum(n_steps)}")
    print(f"  avg steps/ep:   {sum(n_steps)/n:.1f}")
    print(f"  min/max steps:  {min(n_steps)}/{max(n_steps)}")

    # Action type distribution
    type_counts = {}
    for s in samples:
        for step in s["steps"]:
            atype = step["action"].get("action", "unknown")
            type_counts[atype] = type_counts.get(atype, 0) + 1
    print(f"  action types:   {dict(sorted(type_counts.items(), key=lambda x: -x[1]))}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    parser = argparse.ArgumentParser(
        description="Prepare AC episode data for v11 trajectory GRPO")
    parser.add_argument(
        "--input",
        default=os.path.join(project_dir,
                             "datasets/ui_s1_dataset/ui_s1_train_with_desc.jsonl"),
    )
    parser.add_argument(
        "--train_output",
        default=os.path.join(script_dir, "data/ac_train_episodes.jsonl"),
    )
    parser.add_argument(
        "--val_output",
        default=os.path.join(script_dir, "data/ac_val_episodes.jsonl"),
    )
    parser.add_argument("--val_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only_successful", action="store_true",
                        help="Only include episodes marked as successful")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    raw_episodes = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                raw_episodes.append(json.loads(line))
    print(f"Loaded {len(raw_episodes)} raw episodes")

    if args.only_successful:
        raw_episodes = [ep for ep in raw_episodes if ep.get("is_successful")]
        print(f"  After filtering successful: {len(raw_episodes)}")

    # Train/val split (episode-level, same seed as other scripts)
    random.seed(args.seed)
    indices = list(range(len(raw_episodes)))
    random.shuffle(indices)
    val_indices = set(indices[:args.val_episodes])

    train_raw = [ep for i, ep in enumerate(raw_episodes) if i not in val_indices]
    val_raw = [ep for i, ep in enumerate(raw_episodes) if i in val_indices]
    print(f"Split: train={len(train_raw)} raw, val={len(val_raw)} raw")

    # Process episodes
    train_samples, val_samples = [], []
    train_skipped = val_skipped = 0

    eid = 0
    for ep in tqdm(train_raw, desc="Processing train"):
        result = process_episode(ep, eid)
        if result:
            train_samples.append(result)
            eid += 1
        else:
            train_skipped += 1

    for ep in tqdm(val_raw, desc="Processing val"):
        result = process_episode(ep, eid)
        if result:
            val_samples.append(result)
            eid += 1
        else:
            val_skipped += 1

    print_stats(train_samples, f"TRAIN (skipped {train_skipped})")
    print_stats(val_samples, f"VAL (skipped {val_skipped})")

    # Write output
    write_jsonl(args.train_output, train_samples, "Writing train")
    write_jsonl(args.val_output, val_samples, "Writing val")

    # Show example
    if train_samples:
        ex = train_samples[0]
        print(f"\n=== Example episode (id={ex['episode_id']}, {ex['num_steps']} steps) ===")
        print(f"  goal: {ex['goal'][:100]}")
        for step in ex["steps"][:3]:
            print(f"  step {step['step_idx']}: {json.dumps(step['action'])} "
                  f"({step['image_w']}x{step['image_h']})")
        if ex["num_steps"] > 3:
            print(f"  ... ({ex['num_steps'] - 3} more steps)")

    print(f"\nTrain: {args.train_output}")
    print(f"Val:   {args.val_output}")
    print("Done.")


if __name__ == "__main__":
    main()
