#!/usr/bin/env python3
"""Convert GUI-360 RL data to V12-compatible episode format.

Input format (GUI-360):
  {
    "goal": "...",
    "steps": [{
      "action_content": {"action": "click", "coordinate": [39, 71], "text": null, ...},
      "screenshot": "/path/to/image.png",
      "thought": "..."
    }]
  }

Output format (V12-compatible):
  {
    "episode_id": 0,
    "goal": "...",
    "num_steps": N,
    "steps": [{
      "step_idx": 0,
      "action": {"action": "click", "coordinate": [39, 71]},
      "screenshot": "/abs/path/to/image.png",
      "image_w": 1040,
      "image_h": 736
    }]
  }

Usage:
  python v12_gui_360/prepare_gui360_data.py \
      --input datasets/GUI-360/rl_data/gui360_train.jsonl \
      --output_dir v12_gui_360/data \
      --n_train 1000 --n_val 50 --seed 42
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image


# Action types we keep (desktop GUI actions)
VALID_ACTION_TYPES = {"click", "type", "drag", "wheel_mouse_input", ""}

# Normalization map for action types in output
ACTION_TYPE_MAP = {
    "wheel_mouse_input": "swipe",
    "drag": "swipe",
}


def normalize_action(action_content: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert action_content to clean action dict, stripping null fields.

    Returns None for empty actions (should be skipped).
    """
    raw_type = action_content.get("action", "")
    if raw_type == "":
        return None  # skip empty action steps

    mapped_type = ACTION_TYPE_MAP.get(raw_type, raw_type)

    action = {"action": mapped_type}

    # Copy non-null fields
    if action_content.get("coordinate") is not None:
        action["coordinate"] = action_content["coordinate"]
    if action_content.get("text") is not None:
        action["text"] = action_content["text"]
    if action_content.get("coordinate2") is not None:
        # For drag: coordinate2 is the end coordinate
        action["endCoordinate"] = action_content["coordinate2"]
    if action_content.get("status") is not None:
        action["status"] = action_content["status"]

    # For swipe (drag/wheel), ensure we have start and end coordinates
    if mapped_type == "swipe":
        if "coordinate" in action and "endCoordinate" not in action:
            # wheel_mouse_input may not have coordinate2; use coordinate as both
            action["endCoordinate"] = action["coordinate"]

    return action


def get_image_dimensions(path: str) -> tuple:
    """Get actual image dimensions via PIL."""
    try:
        with Image.open(path) as img:
            return img.size  # (width, height)
    except Exception:
        return (1040, 736)  # default desktop dimensions


def is_valid_episode(episode: Dict) -> bool:
    """Check if all steps have valid GUI action types."""
    for step in episode.get("steps", []):
        ac = step.get("action_content", {})
        atype = ac.get("action", "")
        if atype not in VALID_ACTION_TYPES:
            return False
    return True


def convert_episode(episode: Dict, episode_id: int) -> Optional[Dict]:
    """Convert a GUI-360 episode to V12 format."""
    goal = episode.get("goal", "")
    raw_steps = episode.get("steps", [])

    if not raw_steps:
        return None

    converted_steps = []
    step_idx = 0
    for step in raw_steps:
        ac = step.get("action_content", {})
        action = normalize_action(ac)
        if action is None:
            continue  # skip empty action steps

        screenshot = step.get("screenshot", "")
        if not screenshot:
            continue

        # Make path absolute
        if not os.path.isabs(screenshot):
            screenshot = os.path.abspath(screenshot)

        if not os.path.exists(screenshot):
            return None  # skip episode if any screenshot missing

        w, h = get_image_dimensions(screenshot)

        converted_steps.append({
            "step_idx": step_idx,
            "action": action,
            "screenshot": screenshot,
            "image_w": w,
            "image_h": h,
        })
        step_idx += 1

    if not converted_steps:
        return None

    return {
        "episode_id": episode_id,
        "goal": goal,
        "num_steps": len(converted_steps),
        "steps": converted_steps,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert GUI-360 data to V12 format")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to gui360_train.jsonl")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for converted data")
    parser.add_argument("--n_train", type=int, default=1000,
                        help="Number of training episodes to sample")
    parser.add_argument("--n_val", type=int, default=50,
                        help="Number of validation episodes to sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load and filter episodes
    print(f"Loading episodes from {args.input}...")
    valid_episodes = []
    total = 0
    skipped_action = 0
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            ep = json.loads(line)
            if is_valid_episode(ep):
                valid_episodes.append(ep)
            else:
                skipped_action += 1

    print(f"Total episodes: {total}")
    print(f"Valid (GUI-only actions): {len(valid_episodes)}")
    print(f"Skipped (non-GUI actions): {skipped_action}")

    # Sample train + val
    rng = np.random.RandomState(args.seed)
    n_needed = args.n_train + args.n_val
    if n_needed > len(valid_episodes):
        print(f"WARNING: Requested {n_needed} but only {len(valid_episodes)} available. "
              f"Using all.")
        n_needed = len(valid_episodes)

    indices = rng.choice(len(valid_episodes), n_needed, replace=False)
    rng.shuffle(indices)

    train_indices = sorted(indices[:args.n_train])
    val_indices = sorted(indices[args.n_train:args.n_train + args.n_val])

    # Convert and write
    for split, idxs, filename in [
        ("train", train_indices, f"gui360_train_{args.n_train}.jsonl"),
        ("val", val_indices, f"gui360_val_{args.n_val}.jsonl"),
    ]:
        outpath = os.path.join(args.output_dir, filename)
        n_written = 0
        n_failed = 0
        with open(outpath, "w") as f:
            for i, idx in enumerate(idxs):
                ep = valid_episodes[idx]
                converted = convert_episode(ep, episode_id=i)
                if converted is not None:
                    f.write(json.dumps(converted, ensure_ascii=False) + "\n")
                    n_written += 1
                else:
                    n_failed += 1

        print(f"\n{split}: {n_written} episodes written to {outpath}")
        if n_failed > 0:
            print(f"  {n_failed} episodes failed (missing screenshots)")

    # Print sample for verification
    sample_path = os.path.join(args.output_dir, f"gui360_train_{args.n_train}.jsonl")
    with open(sample_path) as f:
        first = json.loads(f.readline())
    print(f"\nSample episode:")
    print(f"  episode_id: {first['episode_id']}")
    print(f"  goal: {first['goal'][:100]}...")
    print(f"  num_steps: {first['num_steps']}")
    if first['steps']:
        s = first['steps'][0]
        print(f"  step 0: action={s['action']}, image={s['image_w']}x{s['image_h']}")
        print(f"          screenshot={s['screenshot']}")


if __name__ == "__main__":
    main()
