"""Cooperative LoRA dataset for AndroidControl.

Converts the v10 JSONL format into verl's RLHFDataset-compatible parquet,
and provides a custom Dataset class that produces grounder prompts for
the first-pass rollout. The trainer will construct actor prompts on the fly
after receiving grounder outputs.

Parquet schema:
  prompt        : list[dict]   — grounder messages (system + user with image)
  reward_model  : dict         — {"ground_truth": gt_action_json}
  extra_info    : dict         — {"image_w", "image_h", "goal", "history"}
  images        : list[str]    — [image_path]
  data_source   : str          — "android_control"
"""

import json
import os
import re
from typing import Optional

import pandas as pd
from PIL import Image

# ── Prompt templates (same as v10/train_grpo.py) ────────────────────────────

GROUNDER_SYSTEM = (
    "You are a GUI grounding agent. Given a screenshot and an instruction, "
    "describe the target UI element that should be interacted with for the "
    "next action. Describe its appearance, text content, and approximate "
    "location on screen."
)

ACTOR_SYSTEM = (
    "You are a GUI agent. Given a screenshot, instruction, and a grounding "
    "description, perform the next action.\n"
    'Output format: <action>{"action": "...", ...}</action>'
)


def format_grounder_text(goal: str, history: str) -> str:
    parts = [f"Instruction: {goal}"]
    if history:
        parts.append(f"\nPrevious actions:\n{history}")
    parts.append("\nDescribe the target UI element for the next action.")
    return "\n".join(parts)


def format_actor_text(goal: str, history: str, grounding: str) -> str:
    parts = [f"Instruction: {goal}"]
    if history:
        parts.append(f"\nPrevious actions:\n{history}")
    parts.append(f"\nGrounding: {grounding}")
    parts.append("\nOutput the next action.")
    return "\n".join(parts)


def build_grounder_messages(image_path: str, goal: str, history: str):
    """Build the grounder's chat messages (system + user with image).

    Uses <image> placeholder format compatible with verl's RLHFDataset._build_messages,
    which splits on <image>/<video> tags and reconstructs structured content.
    """
    user_text = format_grounder_text(goal, history)
    return [
        {"role": "system", "content": GROUNDER_SYSTEM},
        {"role": "user", "content": f"<image>\n{user_text}"},
    ]


def build_actor_messages(image_path: str, goal: str, history: str, grounding: str):
    """Build the actor's chat messages (system + user with image + grounding).

    Uses <image> placeholder format compatible with verl's RLHFDataset._build_messages.
    """
    user_text = format_actor_text(goal, history, grounding)
    return [
        {"role": "system", "content": ACTOR_SYSTEM},
        {"role": "user", "content": f"<image>\n{user_text}"},
    ]


def build_grounder_messages_structured(image_path: str, goal: str, history: str):
    """Build grounder messages in Qwen VL structured format (for direct processor use)."""
    user_text = format_grounder_text(goal, history)
    return [
        {"role": "system", "content": GROUNDER_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def build_actor_messages_structured(image_path: str, goal: str, history: str, grounding: str):
    """Build actor messages in Qwen VL structured format (for direct processor use)."""
    user_text = format_actor_text(goal, history, grounding)
    return [
        {"role": "system", "content": ACTOR_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text},
            ],
        },
    ]


# ── JSONL parsing (from v10/train_grpo.py V10Dataset) ──────────────────────

_GOAL_RE = re.compile(r"## User Instruction\n(.+?)(?:\n\n## |\n## |\Z)", re.DOTALL)
_HIST_RE = re.compile(r"## History of previous actions\n(.+?)(?:\n\n## |\n## |\Z)", re.DOTALL)
_ACTION_TAG_RE = re.compile(r"<action>\s*(\{.*?\})\s*</action>", re.DOTALL)
_ACTION_RAW_RE = re.compile(r'\{[^{}]*"action"[^{}]*\}')


def _parse_action(text: str):
    m = _ACTION_TAG_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = _ACTION_RAW_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _parse_sample(sample: dict):
    """Parse one JSONL sample → (goal, history, gt_action, image_path) or None."""
    convs = sample.get("conversations", [])
    if len(convs) < 2:
        return None
    human_msg = convs[0].get("value", "")
    assistant_msg = convs[1].get("value", "")

    m = _GOAL_RE.search(human_msg)
    goal = m.group(1).strip() if m else ""
    if not goal:
        return None

    m = _HIST_RE.search(human_msg)
    history = m.group(1).strip() if m else ""

    gt_action = _parse_action(assistant_msg)
    if gt_action is None:
        return None

    images = sample.get("images", [])
    image_path = images[0] if images else None
    if image_path is None or not os.path.exists(image_path):
        return None

    return {
        "goal": goal,
        "history": history,
        "gt_action": gt_action,
        "image_path": image_path,
    }


# ── Parquet conversion ──────────────────────────────────────────────────────

def jsonl_to_parquet(jsonl_path: str, output_path: str, max_samples: int = 0):
    """Convert v10 cooperative-thought JSONL to verl-compatible parquet.

    The parquet stores grounder prompts. The trainer constructs actor prompts
    on the fly after receiving grounder outputs.
    """
    records = []
    skipped = 0

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            parsed = _parse_sample(sample)
            if parsed is None:
                skipped += 1
                continue

            goal = parsed["goal"]
            history = parsed["history"]
            image_path = parsed["image_path"]
            gt_action = parsed["gt_action"]

            # Get image dimensions
            try:
                img = Image.open(image_path)
                image_w, image_h = img.size
                img.close()
            except Exception:
                image_w, image_h = 1080, 2400

            # Build grounder prompt messages
            messages = build_grounder_messages(image_path, goal, history)

            records.append({
                "prompt": messages,
                "images": [image_path],
                "data_source": "android_control",
                "reward_model": {
                    "ground_truth": gt_action,
                },
                "extra_info": {
                    "image_w": image_w,
                    "image_h": image_h,
                    "goal": goal,
                    "history": history,
                    "image_path": image_path,
                },
            })

    if 0 < max_samples < len(records):
        import numpy as np
        rng = np.random.RandomState(42)
        idx = rng.choice(len(records), max_samples, replace=False)
        records = [records[i] for i in sorted(idx)]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    print(f"Converted {len(records)} samples to {output_path} (skipped {skipped})")
    return output_path


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert AC JSONL to verl parquet")
    parser.add_argument("--input", required=True, help="Path to ac_train_thought.jsonl")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()
    jsonl_to_parquet(args.input, args.output, args.max_samples)
