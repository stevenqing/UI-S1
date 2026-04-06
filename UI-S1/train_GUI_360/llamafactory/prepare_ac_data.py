"""Prepare AndroidControl (UI-S1) data for LLaMA-Factory SFT training.

Follows GUI-360 format: each step is an independent single-turn sample with
1 image (current screenshot) and action history as text.

- Train: all steps from 1000 trajectories in ui_s1_train.jsonl
- Val: 100 step-samples from eval parquet
"""

import json
import random
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from x.data.agent.json import MOBILE_USE, OUTPUT_FORMAT, generate_prompt
from x.data.agent.space.std_space import RAW_SPACE

DATA_DIR = Path(__file__).parent / "data"
TRAIN_FILE = PROJECT_ROOT / "datasets" / "ui_s1_dataset" / "ui_s1_train.jsonl"
EVAL_FILE = PROJECT_ROOT / "datasets" / "ui_s1_dataset" / "ui_s1_eval_sft.parquet"

# Build system prompt (only_action format, no thought)
SYSTEM_PROMPT = MOBILE_USE.format(
    OUTPUT_FORMAT["only_action"], generate_prompt(RAW_SPACE, add_thought=False)
)


def format_action_text(action_content):
    """Format action content as JSON string (no coordinate resizing)."""
    return json.dumps(action_content, ensure_ascii=False)


def trajectory_to_step_samples(line):
    """Split a trajectory into per-step single-turn samples.

    Each sample has 1 image and text history of previous actions.
    """
    samples = []
    for si, step in enumerate(line["steps"]):
        screenshot = step["screenshot"]
        if not Path(screenshot).is_absolute():
            screenshot = str(PROJECT_ROOT / screenshot)

        # Build history of previous actions
        history_parts = []
        for prev_si in range(si):
            prev_action = format_action_text(line["steps"][prev_si]["action_content"])
            history_parts.append(
                f"Step {prev_si + 1}: <action>\n{prev_action}\n</action>"
            )
        history_text = "\n".join(history_parts)

        # Build user prompt
        user_text = SYSTEM_PROMPT + "\n"
        user_text += f"User Instruction: {line['goal']}\n"
        if history_text:
            user_text += f"\nPrevious Actions:\n{history_text}\n"
        user_text += "\n<image>"

        # Build assistant response
        action_str = format_action_text(step["action_content"])
        response = f"<action>\n{action_str}\n</action>"

        samples.append(
            {
                "conversations": [
                    {"from": "human", "value": user_text},
                    {"from": "gpt", "value": response},
                ],
                "images": [screenshot],
            }
        )
    return samples


def parquet_row_to_trajectory(messages_raw):
    """Reconstruct trajectory dict from eval parquet messages format."""
    if isinstance(messages_raw, str):
        messages = json.loads(messages_raw)
    else:
        messages = messages_raw

    goal = None
    steps = []

    for i in range(0, len(messages), 2):
        user_msg = messages[i]
        asst_msg = messages[i + 1] if i + 1 < len(messages) else None
        if asst_msg is None:
            break

        screenshot = None
        for item in user_msg["content"]:
            if isinstance(item, dict) and item.get("type") == "image":
                screenshot = item["image"]
            elif isinstance(item, dict) and item.get("type") == "text":
                text = item["text"]
                if goal is None and "Task:" in text:
                    goal = text.split("Task: ", 1)[1].split("\n")[0].strip()

        action_content = json.loads(asst_msg["content"])
        steps.append({"action_content": action_content, "screenshot": screenshot})

    return {"goal": goal or "", "steps": steps}


def main():
    # === Train: all steps from 1000 trajectories ===
    print(f"Loading train trajectories from {TRAIN_FILE}...")
    train_lines = []
    with open(TRAIN_FILE) as f:
        for line_str in f:
            train_lines.append(json.loads(line_str))
    print(f"  Loaded {len(train_lines)} trajectories")

    train_samples = []
    for i, line in enumerate(train_lines):
        if i % 200 == 0:
            print(f"  Processing trajectory {i}/{len(train_lines)}...")
        train_samples.extend(trajectory_to_step_samples(line))
    print(f"  Train: {len(train_samples)} step-samples from {len(train_lines)} trajectories")

    # === Val: sample 100 step-samples from eval trajectories ===
    print(f"\nLoading eval data from {EVAL_FILE}...")
    df = pd.read_parquet(EVAL_FILE)
    print(f"  Loaded {len(df)} eval trajectories")

    # Convert all eval trajectories, then sample 100 step-samples
    val_all = []
    for idx in range(len(df)):
        traj = parquet_row_to_trajectory(df.iloc[idx]["messages"])
        val_all.extend(trajectory_to_step_samples(traj))
    print(f"  Total eval step-samples: {len(val_all)}")

    random.seed(42)
    val_samples = random.sample(val_all, min(100, len(val_all)))
    print(f"  Val: {len(val_samples)} step-samples")

    # === Save ===
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    random.shuffle(train_samples)

    out_train = DATA_DIR / "ac_train.json"
    with open(out_train, "w") as f:
        json.dump(train_samples, f, ensure_ascii=False)
    print(f"\nSaved train to {out_train}")

    out_val = DATA_DIR / "ac_val.json"
    with open(out_val, "w") as f:
        json.dump(val_samples, f, ensure_ascii=False)
    print(f"Saved val to {out_val}")

    # === Verify ===
    print("\n=== Train Sample 0 ===")
    s = train_samples[0]
    print(f"  Turns: {len(s['conversations'])}, Images: {len(s['images'])}")
    print(f"  Image: {s['images'][0]}")
    for c in s["conversations"]:
        preview = c["value"][:200].replace("\n", "\\n")
        print(f"  [{c['from']}] {preview}...")

    print("\nDone!")


if __name__ == "__main__":
    main()
