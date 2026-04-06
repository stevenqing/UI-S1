#!/usr/bin/env python3
"""
Add thinking/reasoning to gui360_train.json for LLaMA Factory SFT.

Takes the existing gui360_train.json (ShareGPT format) and adds
"Reasoning: {thought}" prefix to assistant responses using thought
from the raw GUI-360 trajectory JSONL files.

Output: gui360_train_with_thinking.json (same ShareGPT format, for LLaMA Factory)

Usage:
    python train_GUI_360/data_preparation/prepare_gui360_sft_with_thinking_llamafactory.py \
        --input train_GUI_360/llamafactory/data/gui360_train.json \
        --raw-data-dir datasets/GUI-360/train/data \
        --output train_GUI_360/llamafactory/data/gui360_train_with_thinking.json
"""

import argparse
import json
import os
import re
import sys
import glob
from collections import defaultdict


def build_file_index(raw_data_dir):
    """Build execution_id -> jsonl file path index."""
    index = {}
    for f in glob.glob(os.path.join(raw_data_dir, '**', '*.jsonl'), recursive=True):
        eid = os.path.basename(f).replace('.jsonl', '')
        index[eid] = f
    return index


def load_trajectory(jsonl_path):
    """Load and sort steps from a trajectory JSONL file."""
    steps = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                steps.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    steps.sort(key=lambda x: x.get('step_id', 0))
    return steps


def extract_from_image_path(image_path):
    """Extract (execution_id, step_num) from image path."""
    # Pattern 1: .../images/excel_4s_1/action_step4.png (processed data)
    m = re.search(r'/images/([^/]+)/action_step(\d+)', image_path)
    if m:
        return m.group(1), int(m.group(2))
    # Pattern 2: .../image/excel/in_app/.../excel_1_81/action_step1.png (raw data)
    m = re.search(r'/([^/]+)/action_step(\d+)', image_path)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def process(input_file, raw_data_dir, output_file):
    """Add reasoning to assistant responses."""
    # Build file index
    print(f"Indexing raw trajectory files...", file=sys.stderr)
    file_index = build_file_index(raw_data_dir)
    print(f"Indexed {len(file_index)} trajectory files", file=sys.stderr)

    # Load input
    print(f"Loading: {input_file}", file=sys.stderr)
    with open(input_file) as f:
        data = json.load(f)
    print(f"Total samples: {len(data)}", file=sys.stderr)

    # Cache loaded trajectories to avoid re-reading
    traj_cache = {}

    matched = 0
    no_thought = 0
    not_found = 0

    for i, sample in enumerate(data):
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{len(data)} (matched={matched})...", file=sys.stderr)

        images = sample.get('images', [])
        if not images:
            not_found += 1
            continue

        eid, step_num = extract_from_image_path(images[0])
        if not eid or step_num is None:
            not_found += 1
            continue

        jsonl_path = file_index.get(eid)
        if not jsonl_path:
            not_found += 1
            continue

        # Load trajectory (with caching)
        if eid not in traj_cache:
            traj_cache[eid] = load_trajectory(jsonl_path)
        steps = traj_cache[eid]

        # Match step: step_num in image path is 1-based
        idx = step_num - 1
        thought = None
        if 0 <= idx < len(steps):
            thought = steps[idx].get('step', {}).get('thought', '')

        if thought:
            for turn in sample['conversations']:
                if turn['from'] == 'gpt':
                    turn['value'] = f"Reasoning: {thought}\n\n{turn['value']}"
                    break
            matched += 1
        else:
            no_thought += 1

    print(f"\nResults:", file=sys.stderr)
    print(f"  Matched with thought: {matched} ({matched/len(data)*100:.1f}%)", file=sys.stderr)
    print(f"  No thought available: {no_thought}", file=sys.stderr)
    print(f"  Trajectory not found: {not_found}", file=sys.stderr)

    # Save
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

    size_mb = os.path.getsize(output_file) / 1024 / 1024
    print(f"\nSaved: {output_file} ({size_mb:.1f} MB)", file=sys.stderr)

    # Show sample
    for sample in data:
        for turn in sample['conversations']:
            if turn['from'] == 'gpt' and turn['value'].startswith('Reasoning:'):
                print(f"\nSample response:", file=sys.stderr)
                print(turn['value'][:400], file=sys.stderr)
                return
    return


def main():
    parser = argparse.ArgumentParser(
        description="Add thinking/reasoning to GUI-360 SFT data for LLaMA Factory"
    )
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--raw-data-dir", "-r", required=True)
    parser.add_argument("--output", "-o", required=True)
    args = parser.parse_args()
    process(args.input, args.raw_data_dir, args.output)


if __name__ == "__main__":
    main()
