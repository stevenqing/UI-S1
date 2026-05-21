#!/usr/bin/env python3
"""
Prepare GUI-360 data with thought-augmented labels for cooperative LoRA training.

Recovers (screenshot, step_description) pairs from trajectory history alignment:
  - Step N's screenshot paired with step N's description (found in step N+1's history)
  - Last steps and parse failures keep original label without <thought>

Input:  train_GUI_360/llamafactory/data/gui360_train.json  (101,700 samples)
Output: datasets/cooperative_thought/gui360_train_thought.json

Usage:
    python datasets/cooperative_thought/prepare_gui360_thought.py
    python datasets/cooperative_thought/prepare_gui360_thought.py --val  # also process val
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))


def group_into_trajectories(data):
    """Group flat sample list into trajectories by instruction text.

    Returns dict: instruction -> [(n_history_steps, original_index, sample), ...]
    sorted by n_history_steps (= step order within trajectory).
    """
    traj = defaultdict(list)
    for idx, item in enumerate(tqdm(data, desc="Grouping into trajectories")):
        human = item["conversations"][0]["value"]
        inst_m = re.search(r"instruction is:\n(.+?)\n", human)
        inst = inst_m.group(1) if inst_m else f"__no_inst_{idx}"
        traj[inst].append((idx, item))

    # Determine step order by counting "Step N:" entries in history
    ordered = {}
    for inst, items in tqdm(traj.items(), desc="Ordering steps"):
        steps = []
        for idx, item in items:
            human = item["conversations"][0]["value"]
            hist = re.search(
                r"history of actions are:\n(.+?)(?:\nThe actions supported)",
                human,
                re.DOTALL,
            )
            n_steps = hist.group(1).strip().count("Step ") if hist else 0
            steps.append((n_steps, idx, item))
        steps.sort(key=lambda x: x[0])
        ordered[inst] = steps

    return ordered


def recover_thought(next_item):
    """Extract the last step description from the next sample's history.

    Returns the thought string or None if not recoverable.
    """
    human = next_item["conversations"][0]["value"]
    hist = re.search(
        r"history of actions are:\n(.+?)(?:\nThe actions supported)",
        human,
        re.DOTALL,
    )
    if not hist:
        return None
    step_descs = re.findall(
        r"Step \d+: (.+?)(?=Step \d+:|$)", hist.group(1), re.DOTALL
    )
    if not step_descs:
        return None
    return step_descs[-1].strip()


def extract_gt_coord(assistant_text):
    """Extract ground-truth coordinate from tool_call response.

    Returns [x, y] or None.
    """
    tc = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", assistant_text, re.DOTALL)
    if not tc:
        return None
    try:
        call = json.loads(tc.group(1))
    except json.JSONDecodeError:
        return None
    coord = call.get("args", {}).get("coordinate", [None, None])
    if coord and len(coord) == 2 and coord[0] is not None and coord[1] is not None:
        return [int(coord[0]), int(coord[1])]
    return None


def build_thought_dataset(data):
    """Build thought-augmented dataset from GUI-360 flat samples.

    Returns (output_samples, stats).
    """
    trajectories = group_into_trajectories(data)

    stats = {
        "total": len(data),
        "trajectories": len(trajectories),
        "with_thought": 0,
        "without_thought_last_step": 0,
        "without_thought_parse_fail": 0,
        "with_gt_coord": 0,
    }

    # Build index: original_idx -> thought
    idx_to_thought = {}
    for inst, steps in tqdm(trajectories.items(), desc="Recovering thoughts"):
        for i, (n_hist, idx, item) in enumerate(steps):
            if i + 1 < len(steps):
                thought = recover_thought(steps[i + 1][2])
                if thought:
                    idx_to_thought[idx] = thought
                    stats["with_thought"] += 1
                else:
                    stats["without_thought_parse_fail"] += 1
            else:
                stats["without_thought_last_step"] += 1

    # Build output samples (preserve original order)
    output = []
    for idx, item in enumerate(tqdm(data, desc="Building output")):
        conversations = item["conversations"]
        human_msg = conversations[0]["value"]
        assistant_msg = conversations[1]["value"]

        thought = idx_to_thought.get(idx)

        # Build new assistant message
        if thought:
            new_assistant = f"<thought>{thought}</thought>\n{assistant_msg}"
        else:
            new_assistant = assistant_msg

        # Extract gt_coord
        gt_coord = extract_gt_coord(assistant_msg)
        if gt_coord:
            stats["with_gt_coord"] += 1

        sample = {
            "conversations": [
                {"from": "human", "value": human_msg},
                {"from": "assistant", "value": new_assistant},
            ],
            "images": item.get("images", []),
            "has_thought": thought is not None,
        }
        if gt_coord:
            sample["gt_coords"] = gt_coord
        if item.get("images"):
            # Store original image size info if needed later
            sample["orig_image_path"] = item["images"][0]

        output.append(sample)

    return output, stats


def write_jsonl(path, samples, desc="Writing"):
    """Write samples as JSONL with batched I/O for Lustre."""
    CHUNK = 5000
    with open(path, "w", buffering=8 * 1024 * 1024) as f:
        for i in tqdm(range(0, len(samples), CHUNK), desc=desc):
            chunk = samples[i : i + CHUNK]
            lines = [json.dumps(s, ensure_ascii=False) for s in chunk]
            f.write("\n".join(lines) + "\n")


def print_stats(stats, label="Stats"):
    print(f"\n=== {label} ===")
    print(f"  Total samples:       {stats['total']}")
    print(f"  Trajectories:        {stats['trajectories']}")
    print(f"  With thought:        {stats['with_thought']} ({stats['with_thought']/stats['total']*100:.1f}%)")
    print(f"  No thought (last):   {stats['without_thought_last_step']}")
    print(f"  No thought (parse):  {stats['without_thought_parse_fail']}")
    print(f"  With GT coordinate:  {stats['with_gt_coord']} ({stats['with_gt_coord']/stats['total']*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare GUI-360 thought-augmented data"
    )
    parser.add_argument(
        "--input",
        default=os.path.join(
            PROJECT_DIR,
            "train_GUI_360/llamafactory/data/gui360_train.json",
        ),
        help="Input GUI-360 SFT training data",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(SCRIPT_DIR, "gui360_train_thought.jsonl"),
        help="Output train path (JSONL format)",
    )
    parser.add_argument(
        "--val_output",
        default=os.path.join(SCRIPT_DIR, "gui360_val_thought.jsonl"),
    )
    parser.add_argument(
        "--val_trajectories",
        type=int,
        default=500,
        help="Number of trajectories to hold out for val",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    # Load all data
    print(f"Loading {args.input}...")
    with open(args.input) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")

    # Group into trajectories for splitting
    print("Grouping into trajectories...")
    traj = defaultdict(list)
    for idx, item in enumerate(data):
        human = item["conversations"][0]["value"]
        inst_m = re.search(r"instruction is:\n(.+?)\n", human)
        inst = inst_m.group(1) if inst_m else f"__no_inst_{idx}"
        traj[inst].append(idx)

    # Split trajectories into train/val
    random.seed(args.seed)
    traj_keys = list(traj.keys())
    random.shuffle(traj_keys)
    val_keys = set(traj_keys[: args.val_trajectories])
    train_keys = set(traj_keys[args.val_trajectories :])

    train_indices = set()
    val_indices = set()
    for k in train_keys:
        train_indices.update(traj[k])
    for k in val_keys:
        val_indices.update(traj[k])

    train_data = [data[i] for i in sorted(train_indices)]
    val_data = [data[i] for i in sorted(val_indices)]
    print(f"Split: {len(train_keys)} train traj ({len(train_data)} samples), "
          f"{len(val_keys)} val traj ({len(val_data)} samples)")

    # Build thought-augmented datasets
    print("\n--- Train ---")
    train_output, train_stats = build_thought_dataset(train_data)
    print_stats(train_stats, "Train Stats")

    print("\n--- Val ---")
    val_output, val_stats = build_thought_dataset(val_data)
    print_stats(val_stats, "Val Stats")

    # Write outputs
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"\nWriting train: {args.output} ({len(train_output)} samples)")
    write_jsonl(args.output, train_output, "Writing train JSONL")

    print(f"Writing val: {args.val_output} ({len(val_output)} samples)")
    write_jsonl(args.val_output, val_output, "Writing val JSONL")

    # Show examples
    print("\n=== Examples (train) ===")
    count = 0
    for s in train_output:
        if s["has_thought"]:
            assistant = s["conversations"][1]["value"]
            thought_m = re.search(r"<thought>(.*?)</thought>", assistant, re.DOTALL)
            tc_m = re.search(r"<tool_call>\s*\{.*?\"function\"\s*:\s*\"(\w*)\"", assistant, re.DOTALL)
            if thought_m and tc_m:
                print(f'  [{count}] action={tc_m.group(1)}')
                print(f'      thought: "{thought_m.group(1)[:120]}"')
                print(f'      gt_coord: {s.get("gt_coords")}')
                print()
                count += 1
                if count >= 3:
                    break

    print("Done.")


if __name__ == "__main__":
    main()
