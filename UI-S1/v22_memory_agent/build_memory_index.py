"""V22: Build offline memory indices from balanced 2000 training episodes.

Source: v12_gui_360/data/gui360_train_2000_balanced.jsonl (2,000 episodes, ~17K steps)
Thoughts: looked up from datasets/GUI-360/rl_data/gui360_train.jsonl by screenshot path.

Outputs (in indices/):
  - step_records.json       : per-step records with goal, thought, action info
  - episode_metadata.json   : per-episode metadata (goal, app, action_type sequence, num_steps)
  - goal_vectorizer.pkl     : fitted TF-IDF vectorizer on goal texts
  - goal_tfidf_matrix.pkl   : TF-IDF matrix (n_episodes x features)
  - type_index.json         : dict mapping (app, action_type) -> list of step indices
"""

import argparse
import json
import os
import pickle
import re
import sys
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_app_from_screenshot(screenshot_path: str) -> str:
    """Extract app name from screenshot path.

    Path format: .../datasets/GUI-360/train/image/{app}/{category}/{result}/{folder}/action_stepN.png
    """
    parts = screenshot_path.replace("\\", "/").split("/")
    try:
        img_idx = parts.index("image")
        return parts[img_idx + 1]
    except (ValueError, IndexError):
        return "unknown"


def _valid_coord(c):
    """Check if coordinate is a valid [x, y] pair."""
    return (c and isinstance(c, (list, tuple)) and len(c) >= 2
            and c[0] is not None and c[1] is not None)


def format_action_summary(action: dict) -> str:
    """Format action dict into a brief string summary."""
    atype = action.get("action", "unknown")
    coord = action.get("coordinate")
    text = action.get("text", "")
    end_coord = action.get("endCoordinate")

    if atype == "click" and _valid_coord(coord):
        return f"click([{int(float(coord[0]))}, {int(float(coord[1]))}])"
    elif atype == "type" and text:
        t = text[:40] + "..." if len(text) > 40 else text
        if _valid_coord(coord):
            return f"type([{int(float(coord[0]))}, {int(float(coord[1]))}], '{t}')"
        return f"type('{t}')"
    elif atype in ("swipe", "drag") and _valid_coord(coord) and _valid_coord(end_coord):
        return (f"drag([{int(float(coord[0]))}, {int(float(coord[1]))}] -> "
                f"[{int(float(end_coord[0]))}, {int(float(end_coord[1]))}])")
    else:
        return f"{atype}()"


def normalize_action_type(action: dict) -> str:
    """Normalize action type to canonical form."""
    atype = action.get("action", "unknown").lower().strip()
    aliases = {
        "left_click": "click", "tap": "click", "double_click": "click",
        "input": "type", "scroll": "swipe", "drag": "swipe",
        "wheel_mouse_input": "swipe",
    }
    return aliases.get(atype, atype)


def build_thought_lookup(rl_data_path: str) -> dict:
    """Build screenshot_path -> thought lookup from RL training data.

    Returns dict mapping screenshot path (normalized) to thought string.
    """
    lookup = {}
    with open(rl_data_path, "r") as f:
        for line in f:
            episode = json.loads(line.strip())
            for step in episode.get("steps", []):
                screenshot = step.get("screenshot", "")
                thought = step.get("thought", "")
                if screenshot and thought:
                    # Normalize path for matching
                    key = screenshot.rstrip("/")
                    lookup[key] = thought
    return lookup


def main():
    parser = argparse.ArgumentParser(description="Build V22 memory indices")
    parser.add_argument(
        "--balanced_data",
        default="v12_gui_360/data/gui360_train_2000_balanced.jsonl",
        help="Path to balanced 2000 training episodes",
    )
    parser.add_argument(
        "--rl_data",
        default="datasets/GUI-360/rl_data/gui360_train.jsonl",
        help="Path to RL training data (for thought lookup)",
    )
    parser.add_argument(
        "--output_dir",
        default="v22_memory_agent/indices",
        help="Directory to write index files",
    )
    parser.add_argument(
        "--max_tfidf_features",
        type=int,
        default=5000,
        help="Max features for TF-IDF vectorizer",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Build thought lookup from RL data
    print(f"Building thought lookup from {args.rl_data} ...")
    thought_lookup = build_thought_lookup(args.rl_data)
    print(f"  Thought lookup: {len(thought_lookup)} entries")

    # Step 2: Load balanced episodes and build records
    print(f"Loading balanced episodes from {args.balanced_data} ...")
    episodes = []
    with open(args.balanced_data, "r") as f:
        for line in f:
            episodes.append(json.loads(line.strip()))
    print(f"  Loaded {len(episodes)} episodes")

    step_records = []       # flat list of per-step records
    episode_metadata = []   # per-episode metadata
    goal_texts = []         # for TF-IDF
    type_index = defaultdict(list)  # (app, action_type) -> [step_record_idx]

    thought_matches = 0
    thought_misses = 0

    for ep in episodes:
        episode_id = ep["episode_id"]
        goal = ep["goal"]
        steps = ep["steps"]
        num_steps = ep["num_steps"]

        # Extract app from first step's screenshot
        app = "unknown"
        if steps:
            app = extract_app_from_screenshot(steps[0]["screenshot"])

        action_type_sequence = []
        history_actions = []

        for step in steps:
            step_idx = step["step_idx"]
            action = step["action"]
            screenshot = step["screenshot"]

            # Look up thought
            thought_key = screenshot.rstrip("/")
            thought = thought_lookup.get(thought_key, "")
            if thought:
                thought_matches += 1
            else:
                thought_misses += 1

            atype = normalize_action_type(action)
            action_summary = format_action_summary(action)
            action_type_sequence.append(atype)

            record_idx = len(step_records)
            step_records.append({
                "episode_id": episode_id,
                "step_idx": step_idx,
                "goal": goal,
                "app": app,
                "thought": thought,
                "action_type": atype,
                "action_summary": action_summary,
                "history_actions": list(history_actions),  # copy
                "history_length": len(history_actions),
            })

            # Update type index
            type_key = f"{app}|{atype}"
            type_index[type_key].append(record_idx)

            history_actions.append(action_summary)

        # Episode metadata
        episode_metadata.append({
            "episode_id": episode_id,
            "goal": goal,
            "app": app,
            "num_steps": num_steps,
            "action_type_sequence": action_type_sequence,
        })
        goal_texts.append(goal)

    total_steps = len(step_records)
    print(f"  Total steps: {total_steps}")
    print(f"  Thought matches: {thought_matches}, misses: {thought_misses} "
          f"({100*thought_matches/(thought_matches+thought_misses):.1f}% match rate)")

    # Step 3: Build TF-IDF index on goal texts
    print("Building TF-IDF index on goal texts ...")
    vectorizer = TfidfVectorizer(max_features=args.max_tfidf_features)
    tfidf_matrix = vectorizer.fit_transform(goal_texts)
    print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Step 4: Save all indices
    print(f"Saving indices to {args.output_dir} ...")

    # Step records
    step_records_path = os.path.join(args.output_dir, "step_records.json")
    with open(step_records_path, "w") as f:
        json.dump(step_records, f)
    print(f"  step_records.json: {len(step_records)} records")

    # Episode metadata
    meta_path = os.path.join(args.output_dir, "episode_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(episode_metadata, f)
    print(f"  episode_metadata.json: {len(episode_metadata)} episodes")

    # TF-IDF vectorizer
    vec_path = os.path.join(args.output_dir, "goal_vectorizer.pkl")
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"  goal_vectorizer.pkl saved")

    # TF-IDF matrix
    mat_path = os.path.join(args.output_dir, "goal_tfidf_matrix.pkl")
    with open(mat_path, "wb") as f:
        pickle.dump(tfidf_matrix, f)
    print(f"  goal_tfidf_matrix.pkl saved")

    # Type index
    type_path = os.path.join(args.output_dir, "type_index.json")
    with open(type_path, "w") as f:
        json.dump(dict(type_index), f)
    print(f"  type_index.json: {len(type_index)} keys")

    # Summary
    print("\n=== Index Build Summary ===")
    print(f"Episodes: {len(episodes)}")
    print(f"Total steps: {total_steps}")
    print(f"Unique apps: {len(set(m['app'] for m in episode_metadata))}")
    print(f"Type index keys: {len(type_index)}")
    print(f"TF-IDF features: {tfidf_matrix.shape[1]}")
    print("Done!")


if __name__ == "__main__":
    main()
