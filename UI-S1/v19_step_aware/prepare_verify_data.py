#!/usr/bin/env python3
"""Construct verification training data from balanced train episodes.

For each consecutive step pair (step_i, step_{i+1}):
  - screenshot_before = step_i.screenshot
  - screenshot_after  = step_{i+1}.screenshot  (result of GT action at step_i)
  - action            = step_i.action

Positive sample (YES): correct action + correct screen change
Negative sample (NO):  perturbed action + same screen change (action doesn't match)

Negative sampling strategies:
  - coord_shift (40%):  Same type, shifted coordinates by 50-200px
  - type_swap (30%):    Wrong action type (click↔type, click↔swipe)
  - temporal (20%):     Action from a different step in the same episode
  - cross_episode (10%): Action from a different episode

Output: JSONL in LLaMA-Factory conversation format with 2 images.

Usage:
    python v19_step_aware/prepare_verify_data.py \
        --train_data v12_gui_360/data/gui360_train_2000_balanced.jsonl \
        --output v19_step_aware/data/verify_train.jsonl \
        --neg_ratio 1.0
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════
# Verification prompt (same as eval, for consistency)
# ═══════════════════════════════════════════════════════════════════════

VERIFY_PROMPT = """You are verifying whether a GUI action was executed correctly.

You will see two screenshots:
- Screenshot 1 (BEFORE): The screen before the action
- Screenshot 2 (AFTER): The screen after the action was executed

The action that was performed:
{action_description}

The task instruction: {instruction}

Carefully compare the two screenshots. Does Screenshot 2 show the expected result of performing the described action on Screenshot 1?

Consider:
- If it was a click action: Did the clicked element respond? (e.g., button pressed, menu opened, field focused)
- If it was a type action: Does the typed text appear in the correct field?
- If it was a scroll/drag: Did the content move in the expected direction?

Answer ONLY with one word: YES or NO"""


def _safe_coord(coord) -> Optional[List[int]]:
    """Safely convert coordinate to [int, int], or None if invalid."""
    if not coord or not isinstance(coord, (list, tuple)) or len(coord) < 2:
        return None
    if coord[0] is None or coord[1] is None:
        return None
    return [int(float(coord[0])), int(float(coord[1]))]


def describe_action(action: Dict) -> str:
    """Convert action dict to natural language description."""
    atype = action.get("action", "unknown")
    coord = _safe_coord(action.get("coordinate"))

    if atype == "click":
        if coord:
            return f"Click at coordinates [{coord[0]}, {coord[1]}]"
        return "Click (no coordinates)"
    elif atype == "type":
        text = action.get("text", "")
        if coord and text:
            return f"Type '{text[:50]}' at coordinates [{coord[0]}, {coord[1]}]"
        elif text:
            return f"Type '{text[:50]}'"
        elif coord:
            return f"Type at coordinates [{coord[0]}, {coord[1]}]"
        return "Type (no details)"
    elif atype in ("swipe", "drag"):
        start = _safe_coord(action.get("coordinate"))
        end = _safe_coord(action.get("endCoordinate"))
        if start and end:
            return f"Drag from [{start[0]}, {start[1]}] to [{end[0]}, {end[1]}]"
        return "Drag/scroll"
    else:
        return f"{atype} action"


# ═══════════════════════════════════════════════════════════════════════
# Negative sampling strategies
# ═══════════════════════════════════════════════════════════════════════

def _shift_coord(coord, image_w: int = 1040, image_h: int = 736,
                 rng: random.Random = None) -> List:
    """Shift coordinates by 50-200px in random direction."""
    sc = _safe_coord(coord)
    if sc is None:
        sc = [rng.randint(100, image_w - 100), rng.randint(100, image_h - 100)]
    dx = rng.randint(50, 200) * rng.choice([-1, 1])
    dy = rng.randint(50, 200) * rng.choice([-1, 1])
    x = max(0, min(image_w, sc[0] + dx))
    y = max(0, min(image_h, sc[1] + dy))
    return [x, y]


def neg_coord_shift(action: Dict, rng: random.Random,
                    image_w: int = 1040, image_h: int = 736) -> Dict:
    """Same action type, shifted coordinates."""
    neg = dict(action)
    atype = action.get("action", "click")

    if atype in ("click", "type"):
        neg["coordinate"] = _shift_coord(action.get("coordinate"), image_w, image_h, rng)
    elif atype in ("swipe", "drag"):
        neg["coordinate"] = _shift_coord(action.get("coordinate"), image_w, image_h, rng)
        neg["endCoordinate"] = _shift_coord(action.get("endCoordinate"), image_w, image_h, rng)
    return neg


def neg_type_swap(action: Dict, rng: random.Random,
                  all_texts: List[str] = None) -> Dict:
    """Swap action type: click↔type, click↔swipe."""
    neg = dict(action)
    atype = action.get("action", "click")
    coord = _safe_coord(action.get("coordinate")) or [rng.randint(100, 900), rng.randint(100, 600)]

    if atype == "click":
        if rng.random() < 0.6:
            neg["action"] = "type"
            neg["text"] = rng.choice(all_texts) if all_texts else "example text"
            neg["coordinate"] = coord
        else:
            neg["action"] = "swipe"
            neg["coordinate"] = coord
            neg["endCoordinate"] = [
                max(0, coord[0] + rng.randint(-200, 200)),
                max(0, coord[1] + rng.randint(-200, 200)),
            ]
    elif atype == "type":
        neg["action"] = "click"
        neg["coordinate"] = coord
        neg.pop("text", None)
    elif atype in ("swipe", "drag"):
        neg["action"] = "click"
        neg["coordinate"] = coord
        neg.pop("endCoordinate", None)
    return neg


def neg_temporal(episode_steps: List[Dict], current_idx: int,
                 rng: random.Random) -> Optional[Dict]:
    """Pick action from a different step in the same episode."""
    candidates = [j for j in range(len(episode_steps)) if j != current_idx]
    if not candidates:
        return None
    j = rng.choice(candidates)
    return dict(episode_steps[j]["action"])


def neg_cross_episode(all_episodes: List[Dict], current_ep_id: int,
                      rng: random.Random) -> Optional[Dict]:
    """Pick action from a random step in a different episode."""
    attempts = 0
    while attempts < 10:
        ep = rng.choice(all_episodes)
        if ep["episode_id"] != current_ep_id and ep["steps"]:
            step = rng.choice(ep["steps"])
            return dict(step["action"])
        attempts += 1
    return None


# ═══════════════════════════════════════════════════════════════════════
# Build samples
# ═══════════════════════════════════════════════════════════════════════

def build_sample(screenshot_before: str, screenshot_after: str,
                 action: Dict, instruction: str, label: str,
                 neg_strategy: str = "positive") -> Dict:
    """Build one training sample in LLaMA-Factory conversation format."""
    action_desc = describe_action(action)
    prompt = VERIFY_PROMPT.format(
        action_description=action_desc,
        instruction=instruction,
    )

    # Two images: before and after
    human_text = (
        "Screenshot 1 (BEFORE the action):\n<image>\n\n"
        "Screenshot 2 (AFTER the action):\n<image>\n\n"
        + prompt
    )

    return {
        "conversations": [
            {"from": "human", "value": human_text},
            {"from": "gpt", "value": label},
        ],
        "images": [screenshot_before, screenshot_after],
        "metadata": {
            "label": label,
            "neg_strategy": neg_strategy,
            "action_type": action.get("action", "unknown"),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Build verification training data")
    parser.add_argument("--train_data", required=True,
                        help="Path to balanced train JSONL")
    parser.add_argument("--output", required=True,
                        help="Output JSONL path")
    parser.add_argument("--neg_ratio", type=float, default=1.0,
                        help="Negative:positive ratio (default 1.0 = balanced)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check_images", action="store_true", default=False,
                        help="Verify screenshot files exist (slow)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load episodes
    episodes = []
    with open(args.train_data) as f:
        for line in f:
            episodes.append(json.loads(line))
    print(f"Loaded {len(episodes)} episodes")

    # Collect all type texts for negative sampling
    all_texts = []
    for ep in episodes:
        for step in ep["steps"]:
            text = step["action"].get("text", "")
            if text:
                all_texts.append(text)
    print(f"Collected {len(all_texts)} text snippets for negatives")

    # Negative strategy weights
    NEG_STRATEGIES = [
        ("coord_shift", 0.40),
        ("type_swap", 0.30),
        ("temporal", 0.20),
        ("cross_episode", 0.10),
    ]
    strategy_names = [s[0] for s in NEG_STRATEGIES]
    strategy_weights = [s[1] for s in NEG_STRATEGIES]

    # Build samples
    samples = []
    skipped = 0
    neg_counts = defaultdict(int)

    for ep in episodes:
        steps = ep["steps"]
        goal = ep["goal"]
        ep_id = ep["episode_id"]

        for i in range(len(steps) - 1):
            screenshot_before = steps[i]["screenshot"]
            screenshot_after = steps[i + 1]["screenshot"]
            gt_action = steps[i]["action"]
            image_w = steps[i].get("image_w", 1040)
            image_h = steps[i].get("image_h", 736)

            # Check images exist (optional)
            if args.check_images:
                if not os.path.exists(screenshot_before) or not os.path.exists(screenshot_after):
                    skipped += 1
                    continue

            # ── Positive sample ──
            samples.append(build_sample(
                screenshot_before, screenshot_after,
                gt_action, goal, "YES", "positive",
            ))

            # ── Negative sample(s) ──
            n_neg = max(1, int(args.neg_ratio))
            for _ in range(n_neg):
                strategy = rng.choices(strategy_names, weights=strategy_weights, k=1)[0]

                neg_action = None
                if strategy == "coord_shift":
                    neg_action = neg_coord_shift(gt_action, rng, image_w, image_h)
                elif strategy == "type_swap":
                    neg_action = neg_type_swap(gt_action, rng, all_texts)
                elif strategy == "temporal":
                    neg_action = neg_temporal(steps, i, rng)
                elif strategy == "cross_episode":
                    neg_action = neg_cross_episode(episodes, ep_id, rng)

                if neg_action is None:
                    # Fallback to coord_shift
                    neg_action = neg_coord_shift(gt_action, rng, image_w, image_h)
                    strategy = "coord_shift_fallback"

                samples.append(build_sample(
                    screenshot_before, screenshot_after,
                    neg_action, goal, "NO", strategy,
                ))
                neg_counts[strategy] += 1

    # Shuffle
    rng.shuffle(samples)

    # Write output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Stats
    n_pos = sum(1 for s in samples if s["metadata"]["label"] == "YES")
    n_neg = sum(1 for s in samples if s["metadata"]["label"] == "NO")

    print(f"\n{'='*50}")
    print(f"Verification Training Data")
    print(f"{'='*50}")
    print(f"  Total samples:   {len(samples)}")
    print(f"  Positive (YES):  {n_pos}")
    print(f"  Negative (NO):   {n_neg}")
    print(f"  Ratio (neg/pos): {n_neg/n_pos:.2f}")
    print(f"  Skipped (missing images): {skipped}")
    print(f"\n  Negative strategy breakdown:")
    for strategy, count in sorted(neg_counts.items(), key=lambda x: -x[1]):
        print(f"    {strategy:>20s}: {count} ({count/n_neg*100:.1f}%)")
    print(f"\n  Output: {args.output}")
    print(f"{'='*50}")

    # Also create a small validation split from the data
    val_size = min(500, len(samples) // 10)
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]

    val_path = args.output.replace(".jsonl", "_val.jsonl")
    train_path = args.output.replace(".jsonl", "_train.jsonl")

    with open(train_path, "w") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(val_path, "w") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"  Train split: {len(train_samples)} → {train_path}")
    print(f"  Val split:   {len(val_samples)} → {val_path}")


if __name__ == "__main__":
    main()
