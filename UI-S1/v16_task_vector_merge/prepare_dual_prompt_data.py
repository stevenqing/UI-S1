#!/usr/bin/env python3
"""Prepare dual-prompt dataset for V16 cooperative RL.

Augments each step in the balanced GUI-360 dataset with grounder ground truth:
  - has_coordinate: whether the step has a coordinate target for the grounder
  - grounder_gt: the ground-truth coordinate in <coordinate>[x, y]</coordinate> format

Steps WITH coordinates get a two-pass treatment during RL:
  1. Grounder pass (greedy, no grad) → <coordinate> prediction
  2. Actor pass (sampled, PPO grad) → <tool_call> action with grounding

Steps WITHOUT coordinates get a single actor pass (standard prompt, no grounding).

Input:  v12_gui_360/data/gui360_train_2000_balanced.jsonl  (V12 episode format)
Output: v16_task_vector_merge/data/gui360_dual_prompt_train.jsonl
        v16_task_vector_merge/data/gui360_dual_prompt_val.jsonl

Usage:
    python v16_task_vector_merge/prepare_dual_prompt_data.py \
        --train_input v12_gui_360/data/gui360_train_2000_balanced.jsonl \
        --val_input v12_gui_360/data/gui360_val_50.jsonl \
        --output_dir v16_task_vector_merge/data
"""

import argparse
import json
import os
from collections import defaultdict


def augment_episode(episode: dict) -> dict:
    """Add grounder_gt and has_coordinate fields to each step."""
    for step in episode["steps"]:
        action = step["action"]
        atype = action.get("action", "")
        coord = action.get("coordinate")

        if atype in ("click", "long_press") and coord:
            step["grounder_gt"] = f"<coordinate> [{coord[0]}, {coord[1]}] </coordinate>"
            step["has_coordinate"] = True
        elif atype == "type" and coord:
            step["grounder_gt"] = f"<coordinate> [{coord[0]}, {coord[1]}] </coordinate>"
            step["has_coordinate"] = True
        elif atype in ("swipe", "drag"):
            start = coord or action.get("startCoordinate")
            if start:
                step["grounder_gt"] = f"<coordinate> [{start[0]}, {start[1]}] </coordinate>"
                step["has_coordinate"] = True
            else:
                step["has_coordinate"] = False
                step["grounder_gt"] = ""
        else:
            step["has_coordinate"] = False
            step["grounder_gt"] = ""

    return episode


def process_file(input_path: str, output_path: str, label: str):
    """Process one JSONL file: augment and write."""
    episodes = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episodes.append(json.loads(line))

    print(f"\n{label}: {len(episodes)} episodes from {input_path}")

    # Augment
    stats = defaultdict(int)
    for ep in episodes:
        augment_episode(ep)
        for step in ep["steps"]:
            stats["total"] += 1
            if step["has_coordinate"]:
                stats["has_coord"] += 1
            else:
                stats["no_coord"] += 1
            atype = step["action"]["action"]
            stats[f"type_{atype}"] += 1
            if step["has_coordinate"]:
                stats[f"coord_{atype}"] += 1

    # Write
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    # Report
    total = stats["total"]
    has = stats["has_coord"]
    no = stats["no_coord"]
    print(f"  Total steps: {total}")
    print(f"  Has coordinate: {has} ({has/total*100:.1f}%)")
    print(f"  No coordinate:  {no} ({no/total*100:.1f}%)")
    print(f"  Action type breakdown:")
    for atype in sorted(set(k.replace("type_", "") for k in stats if k.startswith("type_"))):
        t = stats[f"type_{atype}"]
        c = stats.get(f"coord_{atype}", 0)
        print(f"    {atype}: {t} total, {c} with coord ({c/t*100:.1f}%)")
    print(f"  Written to: {output_path}")

    # Verify a sample
    with open(output_path) as f:
        sample = json.loads(f.readline())
    s0 = sample["steps"][0]
    print(f"\n  Sample episode: {sample['goal'][:80]}...")
    print(f"  Step 0: action={s0['action']}")
    print(f"          has_coordinate={s0['has_coordinate']}")
    print(f"          grounder_gt={s0.get('grounder_gt', '')[:60]}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dual-prompt dataset for V16 cooperative RL")
    parser.add_argument("--train_input", type=str, required=True,
                        help="Path to balanced training JSONL")
    parser.add_argument("--val_input", type=str, default="",
                        help="Path to validation JSONL")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    args = parser.parse_args()

    print("=" * 60)
    print("  V16 Dual-Prompt Data Preparation")
    print("=" * 60)

    process_file(
        args.train_input,
        os.path.join(args.output_dir, "gui360_dual_prompt_train.jsonl"),
        "Train",
    )

    if args.val_input and os.path.exists(args.val_input):
        process_file(
            args.val_input,
            os.path.join(args.output_dir, "gui360_dual_prompt_val.jsonl"),
            "Val",
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
