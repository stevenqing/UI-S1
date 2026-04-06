#!/usr/bin/env python3
"""
Create micro grounding SFT dataset (200 samples) for Exp D.

Format: screenshot + task instruction → target element coordinates
This trains the model to explicitly locate the target element before acting.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

JSONL_PATH = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/GUI-360/rl_data/gui360_test.jsonl"
OUTPUT_PATH = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation/dataset/gui360_grounding_200.json"
N_SAMPLES = 200


def main():
    np.random.seed(42)

    # Load trajectories and extract click steps with coordinates
    samples = []
    with open(JSONL_PATH) as f:
        for line in f:
            traj = json.loads(line)
            goal = traj["goal"]
            for step in traj["steps"]:
                act = step["action_content"]
                coord = act.get("coordinate")
                if act["action"] == "click" and coord and all(c is not None for c in coord):
                    screenshot = step["screenshot"]
                    if os.path.exists(screenshot):
                        thought = step.get("thought", "")
                        samples.append({
                            "goal": goal,
                            "thought": thought,
                            "screenshot": screenshot,
                            "coord": [int(c) for c in coord],
                            "action": act["action"],
                        })

    print(f"Found {len(samples)} click samples with coordinates")

    # Sample N
    indices = np.random.choice(len(samples), min(N_SAMPLES, len(samples)), replace=False)
    selected = [samples[i] for i in indices]

    # Build SFT dataset in LLaMA-Factory sharegpt format
    sft_data = []
    for s in selected:
        x, y = s["coord"]

        # Extract element description from thought (first clause)
        thought = s["thought"]
        first_sent = thought.split('.')[0].strip() if thought else ""
        if len(first_sent) > 200:
            first_sent = first_sent[:200]

        user_text = (
            f"You are a GUI grounding assistant. Given a screenshot and a task instruction, "
            f"locate the UI element that should be interacted with next.\n\n"
            f"Task: {s['goal']}\n\n"
            f"Context: {first_sent}\n\n"
            f"Output the coordinates of the target element."
        )

        assistant_text = (
            f"The target element is located at coordinates ({x}, {y}) in the screenshot."
        )

        sft_data.append({
            "conversations": [
                {"from": "human", "value": f"<image>\n{user_text}"},
                {"from": "gpt", "value": assistant_text},
            ],
            "images": [s["screenshot"]],
        })

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(sft_data, f, indent=2)

    print(f"Saved {len(sft_data)} grounding samples to {OUTPUT_PATH}")

    # Also save as simple format for custom training
    simple_path = OUTPUT_PATH.replace(".json", "_simple.jsonl")
    with open(simple_path, "w") as f:
        for s in selected:
            f.write(json.dumps({
                "screenshot": s["screenshot"],
                "goal": s["goal"],
                "thought_first": s["thought"].split('.')[0].strip() if s["thought"] else "",
                "coord": s["coord"],
            }) + "\n")
    print(f"Saved simple format to {simple_path}")


if __name__ == "__main__":
    main()
