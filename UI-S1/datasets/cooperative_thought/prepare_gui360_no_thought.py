#!/usr/bin/env python3
"""
Prepare GUI-360 data WITHOUT thought labels for cooperative LoRA v6 training.

Reads thought-augmented JSONL and strips <thought>...</thought> from assistant
responses, keeping only <tool_call>...</tool_call>.

Input:  gui360_train_thought.jsonl / gui360_val_thought.jsonl
Output: gui360_train_nothought.jsonl / gui360_val_nothought.jsonl

Usage:
    python datasets/cooperative_thought/prepare_gui360_no_thought.py
"""

import json
import os
import re
import sys

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def strip_thought(text):
    """Remove <thought>...</thought> and trailing newline from assistant text."""
    return re.sub(r"<thought>.*?</thought>\n?", "", text, flags=re.DOTALL)


def process_file(input_path, output_path):
    """Read thought JSONL, strip thoughts, write no-thought JSONL."""
    samples = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # Strip thought from assistant response
            convs = item["conversations"]
            convs[1]["value"] = strip_thought(convs[1]["value"])
            item["has_thought"] = False
            samples.append(item)

    with open(output_path, "w", buffering=8 * 1024 * 1024) as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    return len(samples)


def main():
    for split in ("train", "val"):
        input_path = os.path.join(SCRIPT_DIR, f"gui360_{split}_thought.jsonl")
        output_path = os.path.join(SCRIPT_DIR, f"gui360_{split}_nothought.jsonl")

        if not os.path.exists(input_path):
            print(f"Skipping {split}: {input_path} not found")
            continue

        n = process_file(input_path, output_path)
        print(f"{split}: {n} samples -> {output_path}")

    # Verify a sample
    sample_path = os.path.join(SCRIPT_DIR, "gui360_train_nothought.jsonl")
    if os.path.exists(sample_path):
        with open(sample_path) as f:
            sample = json.loads(f.readline())
        assistant = sample["conversations"][1]["value"]
        has_thought = "<thought>" in assistant
        has_tool = "<tool_call>" in assistant
        print(f"\nVerification: has_thought={has_thought}, has_tool_call={has_tool}")
        print(f"  Assistant (first 200 chars): {assistant[:200]}")


if __name__ == "__main__":
    main()
