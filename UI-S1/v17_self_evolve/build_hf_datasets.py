#!/usr/bin/env python3
"""
Build HuggingFace datasets for GUI-360 balanced data (episode-level).

Merges balanced episode structure with processed data fields (bbox, reward,
conversation, status). Each row = one episode.

Creates:
1. gui360_balanced       - non-a11y (integer coordinates), 2000 train + 1000 test
2. gui360_balanced_a11y  - a11y (float coordinates), 2000 train
"""

import argparse
import json
import os
from collections import defaultdict

from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage


def load_processed_index(json_path: str) -> dict:
    """Load processed data indexed by (folder, step_num)."""
    print(f"Loading {json_path}...")
    with open(json_path) as f:
        data = json.load(f)
    index = {}
    for d in data:
        img = d["images"][0]
        parts = img.split("/")
        folder = parts[1]
        step_num = int(parts[2].replace("action_step", "").replace(".png", ""))
        index[(folder, step_num)] = d
    del data
    print(f"  Indexed {len(index)} entries")
    return index


def load_test_raw_index(test_data_dir: str) -> dict:
    """Load raw test data from per-episode jsonl files.

    Returns index: (folder, step_num) -> {bbox, status, reward, action_details}
    """
    import glob
    print(f"Loading raw test data from {test_data_dir}...")
    index = {}
    jsonl_files = glob.glob(os.path.join(test_data_dir, "**/*.jsonl"), recursive=True)
    for fpath in jsonl_files:
        with open(fpath) as f:
            for line in f:
                d = json.loads(line)
                folder = d["execution_id"]
                step_num = d["step_id"]
                step = d["step"]
                action = step.get("action", {})

                # Extract bbox from rectangle
                rect = action.get("rectangle")
                bbox = None
                if rect:
                    bbox = [rect["left"], rect["top"], rect["right"], rect["bottom"]]

                index[(folder, step_num)] = {
                    "bbox": bbox,
                    "status": step.get("status", "CONTINUE"),
                    "reward": 1 if d.get("evaluation", {}).get("reason") else None,
                    "action_raw": action,
                }
    print(f"  Indexed {len(index)} test entries from {len(jsonl_files)} files")
    return index


def load_episodes(jsonl_path: str) -> list:
    episodes = []
    with open(jsonl_path) as f:
        for line in f:
            episodes.append(json.loads(line))
    print(f"Loaded {len(episodes)} episodes from {jsonl_path}")
    return episodes


def enrich_test_episodes(episodes, test_index, label=""):
    """Enrich test episodes with raw test data (bbox, status)."""
    enriched = []
    skipped_eps = 0

    for ep in episodes:
        folder = ep["steps"][0]["screenshot"].split("/")[-2]
        new_steps = []
        all_ok = True

        for step in ep["steps"]:
            key = (folder, step["step_idx"] + 1)
            raw = test_index.get(key)
            if raw is None:
                all_ok = False
                break

            new_step = {
                "step_idx": step["step_idx"],
                "action": step["action"],
                "screenshot": step["screenshot"],
                "image_w": step.get("image_w"),
                "image_h": step.get("image_h"),
                "bbox": raw["bbox"],
                "reward": raw.get("reward"),
                "status": raw["status"],
                "conversation_human": None,  # No processed conversation for test
                "conversation_gpt": None,
            }
            new_steps.append(new_step)

        if all_ok:
            enriched.append({
                "episode_id": ep["episode_id"],
                "goal": ep["goal"],
                "num_steps": ep["num_steps"],
                "steps": new_steps,
            })
        else:
            skipped_eps += 1

    print(f"  {label}: {len(enriched)} enriched, {skipped_eps} skipped")
    return enriched


def enrich_episodes(episodes, proc_index, label=""):
    """Enrich balanced episodes with processed data fields (bbox, reward, conversation, status)."""
    enriched = []
    skipped_eps = 0

    for ep in episodes:
        folder = ep["steps"][0]["screenshot"].split("/")[-2]
        new_steps = []
        all_ok = True

        for step in ep["steps"]:
            key = (folder, step["step_idx"] + 1)
            proc = proc_index.get(key)
            if proc is None:
                all_ok = False
                break

            # Extract status from GPT response
            gpt_resp = proc["conversation"][1]["value"]
            status = "CONTINUE"
            if '"status": "FINISH"' in gpt_resp or '"status":"FINISH"' in gpt_resp:
                status = "FINISH"

            new_step = {
                "step_idx": step["step_idx"],
                "action": step["action"],
                "screenshot": step["screenshot"],
                "image_w": step.get("image_w"),
                "image_h": step.get("image_h"),
                # Fields from processed data
                "bbox": proc.get("bbox"),
                "reward": proc.get("reward", 1),
                "status": status,
                "conversation_human": proc["conversation"][0]["value"],
                "conversation_gpt": gpt_resp,
            }
            new_steps.append(new_step)

        if all_ok:
            enriched_ep = {
                "episode_id": ep["episode_id"],
                "goal": ep["goal"],
                "num_steps": ep["num_steps"],
                "steps": new_steps,
            }
            enriched.append(enriched_ep)
        else:
            skipped_eps += 1

    print(f"  {label}: {len(enriched)} enriched, {skipped_eps} skipped")
    return enriched


def make_generator(episodes):
    """Generator yielding one row per episode with lazy image loading."""
    def gen():
        for ep in episodes:
            images = []
            for step in ep["steps"]:
                path = step["screenshot"]
                try:
                    img = PILImage.open(path)
                    img.load()
                    images.append(img)
                except Exception:
                    images.append(None)

            yield {
                "episode_id": ep["episode_id"],
                "goal": ep["goal"],
                "num_steps": ep["num_steps"],
                "steps": json.dumps(ep["steps"]),
                "screenshots": images,
            }
    return gen


FEATURES = Features({
    "episode_id": Value("int32"),
    "goal": Value("string"),
    "num_steps": Value("int32"),
    "steps": Value("string"),  # JSON string: [{step_idx, action, bbox, reward, status, conversation_human, conversation_gpt, ...}]
    "screenshots": Sequence(Image()),
})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced_train", default="v12_gui_360/data/gui360_train_2000_balanced.jsonl")
    parser.add_argument("--balanced_test", default="v12_gui_360/data/gui360_test_1000_balanced.jsonl")
    parser.add_argument("--processed_dir", default="datasets/GUI-360/processed_data")
    parser.add_argument("--test_data_dir", default="datasets/GUI-360/test/data")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_org", default="")
    args = parser.parse_args()

    noa11y_path = os.path.join(args.processed_dir, "action_prediction_train_resize/training_data.json")
    a11y_path = os.path.join(args.processed_dir, "action_prediction_train_resize_a11y/training_data.json")

    noa11y_index = load_processed_index(noa11y_path)

    train_eps = load_episodes(args.balanced_train)
    test_eps = load_episodes(args.balanced_test)

    # === 1. Non-a11y balanced (train + test) ===
    print("\n=== Enriching non-a11y train ===")
    train_enriched = enrich_episodes(train_eps, noa11y_index, "train")

    # Test set: enrich from raw test data (per-episode jsonl files with bbox, status)
    print("\n=== Enriching test from raw test data ===")
    test_raw_index = load_test_raw_index(args.test_data_dir)
    test_enriched = enrich_test_episodes(test_eps, test_raw_index, "test")
    del test_raw_index

    print(f"\n=== Creating non-a11y HF datasets ===")
    print(f"  Train: {len(train_enriched)} episodes")
    print(f"  Test:  {len(test_enriched)} episodes")

    train_ds = Dataset.from_generator(make_generator(train_enriched), features=FEATURES)
    print(f"  Train dataset: {len(train_ds)} rows")

    if args.push_to_hub and args.hub_org:
        repo = f"{args.hub_org}/gui360-balanced"
        print(f"  Pushing train to {repo}...")
        train_ds.push_to_hub(repo, split="train")
        print(f"  Train pushed!")
    del train_ds

    test_ds = Dataset.from_generator(make_generator(test_enriched), features=FEATURES)
    print(f"  Test dataset: {len(test_ds)} rows")

    if args.push_to_hub and args.hub_org:
        print(f"  Pushing test to {repo}...")
        test_ds.push_to_hub(repo, split="test")
        print(f"  Test pushed!")
    del test_ds, test_enriched

    # === 2. A11y balanced (train only) ===
    print("\n=== Enriching a11y train ===")
    a11y_index = load_processed_index(a11y_path)
    a11y_enriched = enrich_episodes(train_eps, a11y_index, "a11y-train")
    del a11y_index, noa11y_index, train_eps, test_eps

    print(f"\n=== Creating a11y HF dataset ===")
    print(f"  A11y train: {len(a11y_enriched)} episodes")

    a11y_ds = Dataset.from_generator(make_generator(a11y_enriched), features=FEATURES)
    print(f"  A11y dataset: {len(a11y_ds)} rows")

    if args.push_to_hub and args.hub_org:
        repo = f"{args.hub_org}/gui360-balanced-a11y"
        print(f"  Pushing a11y to {repo}...")
        a11y_ds.push_to_hub(repo, split="train")
        print(f"  A11y pushed!")
    del a11y_ds

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
