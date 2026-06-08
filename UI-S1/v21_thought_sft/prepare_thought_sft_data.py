#!/usr/bin/env python3
"""Prepare thought-augmented SFT training data.

Takes the existing ShareGPT training data (gui360_balanced_train.jsonl) and
prepends GT thought to each GPT response, so the model learns to plan before acting.

Current format:
    GPT: <tool_call>{"function": "click", ...}</tool_call>

Thought-augmented format:
    GPT: To navigate to the File tab, I need to click on it to access the Info section.

         <tool_call>{"function": "click", ...}</tool_call>

The GT thought is looked up from the GUI-360 source data via screenshot path matching.
"""

import json
import os
import sys
from collections import Counter

# Paths
PROJECT = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
SOURCE_TRAIN = f"{PROJECT}/datasets/GUI-360/rl_data/gui360_train.jsonl"
SOURCE_TEST = f"{PROJECT}/datasets/GUI-360/rl_data/gui360_test.jsonl"
SHAREGPT_TRAIN = f"{PROJECT}/v15_gui_360/data/gui360_balanced_train.jsonl"
SHAREGPT_VAL = f"{PROJECT}/v12_gui_360/data/gui360_val_50.jsonl"  # check if exists

OUTPUT_TRAIN = f"{PROJECT}/v21_thought_sft/data/gui360_thought_train.jsonl"
OUTPUT_VAL = f"{PROJECT}/v21_thought_sft/data/gui360_thought_val.jsonl"


def build_screenshot_to_thought_index(*source_files):
    """Build index: screenshot path -> GT thought from GUI-360 source data."""
    index = {}
    for source_file in source_files:
        if not os.path.exists(source_file):
            print(f"  [SKIP] {source_file} not found")
            continue
        count = 0
        with open(source_file) as f:
            for line in f:
                ep = json.loads(line.strip())
                for step in ep.get("steps", []):
                    ss = step.get("screenshot", "")
                    thought = step.get("thought", "")
                    if ss and thought:
                        # Normalize to absolute path
                        if not ss.startswith("/"):
                            ss = os.path.join(PROJECT, ss)
                        index[ss] = thought
                        count += 1
        print(f"  Indexed {count} thoughts from {os.path.basename(source_file)}")
    return index


def augment_sharegpt(input_path, output_path, thought_index):
    """Add GT thought to GPT responses in ShareGPT format data."""
    if not os.path.exists(input_path):
        print(f"  [SKIP] {input_path} not found")
        return 0, 0

    augmented = []
    total = 0
    matched = 0
    thought_lengths = []

    with open(input_path) as f:
        for line in f:
            d = json.loads(line.strip())
            total += 1

            # Get screenshot path from images field
            images = d.get("images", [])
            thought = ""
            if images:
                ss = images[0]
                if not ss.startswith("/"):
                    ss = os.path.join(PROJECT, ss)
                thought = thought_index.get(ss, "")

            if thought:
                matched += 1
                thought_lengths.append(len(thought))

                # Prepend thought to GPT response
                for conv in d["conversations"]:
                    if conv["from"] == "gpt":
                        conv["value"] = thought + "\n\n" + conv["value"]
                        break

            augmented.append(d)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for d in augmented:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    import statistics
    avg_len = statistics.mean(thought_lengths) if thought_lengths else 0
    print(f"  {os.path.basename(input_path)}: {matched}/{total} matched "
          f"({matched/total*100:.1f}%), avg thought len={avg_len:.0f} chars")
    return total, matched


def main():
    print("Building screenshot -> thought index...")
    thought_index = build_screenshot_to_thought_index(SOURCE_TRAIN, SOURCE_TEST)
    print(f"Total index size: {len(thought_index)}")

    print("\nAugmenting training data...")
    total, matched = augment_sharegpt(SHAREGPT_TRAIN, OUTPUT_TRAIN, thought_index)
    print(f"  Output: {OUTPUT_TRAIN}")

    # Also check if val data exists and augment it
    val_candidates = [
        f"{PROJECT}/v12_gui_360/data/gui360_val_50.jsonl",
        f"{PROJECT}/v15_gui_360/data/gui360_balanced_val.jsonl",
    ]
    for val_path in val_candidates:
        if os.path.exists(val_path):
            # Val data might be in v12 format (not ShareGPT), check
            with open(val_path) as f:
                first = json.loads(f.readline())
            if "conversations" in first:
                print(f"\nAugmenting val data from {val_path}...")
                augment_sharegpt(val_path, OUTPUT_VAL, thought_index)
                print(f"  Output: {OUTPUT_VAL}")
            else:
                print(f"\n  [SKIP] {val_path} is not ShareGPT format")
            break

    # Show example
    print("\n=== Example augmented response ===")
    with open(OUTPUT_TRAIN) as f:
        d = json.loads(f.readline())
    for conv in d["conversations"]:
        if conv["from"] == "gpt":
            print(conv["value"][:500])
            break


if __name__ == "__main__":
    main()
