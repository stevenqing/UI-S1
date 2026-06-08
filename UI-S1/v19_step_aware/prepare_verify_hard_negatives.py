#!/usr/bin/env python3
"""Build verification data using model's real errors as hard negatives.

Reads eval results (from running base_sft on training episodes) and constructs:
  - YES: correct predictions + GT actions (model got it right, screen matches)
  - NO (hard): model's wrong predictions (real errors the verifier must detect)
  - NO (synthetic): coord-shift / type-swap for diversity

Usage:
    python v19_step_aware/prepare_verify_hard_negatives.py \
        --eval_results v19_step_aware/outputs/train_predictions/eval_results_*.json \
        --train_data v12_gui_360/data/gui360_train_2000_balanced.jsonl \
        --output v19_step_aware/data/verify_hard.jsonl
"""

import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

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
    if not coord or not isinstance(coord, (list, tuple)) or len(coord) < 2:
        return None
    if coord[0] is None or coord[1] is None:
        return None
    return [int(float(coord[0])), int(float(coord[1]))]


def describe_action(action: Dict) -> str:
    if action is None:
        return "(no action predicted)"
    atype = action.get("action", "unknown")
    coord = _safe_coord(action.get("coordinate"))
    if atype == "click":
        return f"Click at coordinates [{coord[0]}, {coord[1]}]" if coord else "Click (no coordinates)"
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
    return f"{atype} action"


def build_sample(img_before, img_after, action_desc, instruction, label, strategy, atype="unknown"):
    prompt = VERIFY_PROMPT.format(action_description=action_desc, instruction=instruction)
    human_text = (
        "Screenshot 1 (BEFORE the action):\n<image>\n\n"
        "Screenshot 2 (AFTER the action):\n<image>\n\n" + prompt
    )
    return {
        "conversations": [
            {"from": "human", "value": human_text},
            {"from": "gpt", "value": label},
        ],
        "images": [img_before, img_after],
        "metadata": {"label": label, "neg_strategy": strategy, "action_type": atype},
    }


def _shift_coord(coord, rng, image_w=1040, image_h=736):
    sc = _safe_coord(coord) or [rng.randint(100, image_w-100), rng.randint(100, image_h-100)]
    dx = rng.randint(50, 200) * rng.choice([-1, 1])
    dy = rng.randint(50, 200) * rng.choice([-1, 1])
    return [max(0, min(image_w, sc[0]+dx)), max(0, min(image_h, sc[1]+dy))]


def make_synthetic_neg(gt_action, rng, all_texts, image_w=1040, image_h=736):
    neg = dict(gt_action)
    atype = gt_action.get("action", "click")
    coord = _safe_coord(gt_action.get("coordinate")) or [rng.randint(100, 900), rng.randint(100, 600)]
    if rng.random() < 0.5:  # coord_shift
        if atype in ("click", "type"):
            neg["coordinate"] = _shift_coord(gt_action.get("coordinate"), rng, image_w, image_h)
        elif atype in ("swipe", "drag"):
            neg["coordinate"] = _shift_coord(gt_action.get("coordinate"), rng, image_w, image_h)
            neg["endCoordinate"] = _shift_coord(gt_action.get("endCoordinate"), rng, image_w, image_h)
    else:  # type_swap
        if atype == "click":
            neg["action"] = "type"
            neg["text"] = rng.choice(all_texts) if all_texts else "example"
            neg["coordinate"] = coord
        elif atype == "type":
            neg["action"] = "click"
            neg["coordinate"] = coord
            neg.pop("text", None)
        elif atype in ("swipe", "drag"):
            neg["action"] = "click"
            neg["coordinate"] = coord
            neg.pop("endCoordinate", None)
    return neg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_results", required=True, help="Glob pattern for eval result JSONs")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--synthetic_ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load eval results
    result_files = sorted(glob.glob(args.eval_results))
    print(f"Found {len(result_files)} result files")
    eval_results = {}
    for rf in result_files:
        with open(rf) as f:
            eval_results.update(json.load(f))
    print(f"Loaded predictions for {len(eval_results)} episodes")

    # Load training episodes
    episodes = {}
    with open(args.train_data) as f:
        for line in f:
            ep = json.loads(line)
            episodes[str(ep["episode_id"])] = ep
    print(f"Loaded {len(episodes)} training episodes")

    all_texts = []
    for ep in episodes.values():
        for step in ep["steps"]:
            t = step["action"].get("text", "")
            if t:
                all_texts.append(t)

    samples = []
    stats = defaultdict(int)

    for eid, result in eval_results.items():
        ep = episodes.get(str(eid))
        if ep is None:
            stats["missing_episode"] += 1
            continue
        goal = ep["goal"]
        steps = ep["steps"]

        for sr in result.get("steps", []):
            idx = sr["step_idx"]
            if idx + 1 >= len(steps):
                continue
            img_before = steps[idx]["screenshot"]
            img_after = steps[idx + 1]["screenshot"]
            pred_action = sr.get("pred_action")
            gt_action = steps[idx]["action"]
            is_correct = sr.get("success", False)
            gt_type = sr.get("gt_type", gt_action.get("action", "unknown"))
            image_w = steps[idx].get("image_w", 1040)
            image_h = steps[idx].get("image_h", 736)

            # YES from GT action (always correct)
            samples.append(build_sample(
                img_before, img_after, describe_action(gt_action),
                goal, "YES", "gt_positive", gt_type))
            stats["gt_positive"] += 1

            if pred_action is None:
                stats["no_pred"] += 1
                continue

            if is_correct:
                # YES from model's correct prediction
                samples.append(build_sample(
                    img_before, img_after, describe_action(pred_action),
                    goal, "YES", "model_correct", gt_type))
                stats["model_correct"] += 1
            else:
                # NO from model's wrong prediction (HARD NEGATIVE)
                samples.append(build_sample(
                    img_before, img_after, describe_action(pred_action),
                    goal, "NO", "model_error", gt_type))
                stats["hard_negative"] += 1

            # Synthetic negative for diversity
            if rng.random() < args.synthetic_ratio:
                syn = make_synthetic_neg(gt_action, rng, all_texts, image_w, image_h)
                samples.append(build_sample(
                    img_before, img_after, describe_action(syn),
                    goal, "NO", "synthetic", gt_type))
                stats["synthetic"] += 1

    # Balance YES/NO
    yes_s = [s for s in samples if s["conversations"][1]["value"] == "YES"]
    no_s = [s for s in samples if s["conversations"][1]["value"] == "NO"]
    print(f"\nBefore balancing: YES={len(yes_s)}, NO={len(no_s)}")

    if len(yes_s) > len(no_s) * 1.2:
        rng.shuffle(yes_s)
        yes_s = yes_s[:int(len(no_s) * 1.1)]
    elif len(no_s) > len(yes_s) * 1.2:
        rng.shuffle(no_s)
        no_s = no_s[:int(len(yes_s) * 1.1)]

    samples = yes_s + no_s
    rng.shuffle(samples)

    # Split
    val_size = min(500, len(samples) // 10)
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]

    # Write JSONL
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Write LLaMA-Factory JSON (strip metadata)
    train_json_path = "train_GUI_360/llamafactory/data/verify_hard_train.json"
    val_json_path = "train_GUI_360/llamafactory/data/verify_hard_val.json"
    train_clean = [{k: v for k, v in s.items() if k != "metadata"} for s in train_samples]
    val_clean = [{k: v for k, v in s.items() if k != "metadata"} for s in val_samples]
    with open(train_json_path, "w") as f:
        json.dump(train_clean, f, ensure_ascii=False)
    with open(val_json_path, "w") as f:
        json.dump(val_clean, f, ensure_ascii=False)

    n_yes = sum(1 for s in samples if s["conversations"][1]["value"] == "YES")
    n_no = len(samples) - n_yes
    print(f"\n{'='*50}")
    print(f"Hard Negative Verification Data")
    print(f"{'='*50}")
    print(f"  Total: {len(samples)}  (YES={n_yes}, NO={n_no})")
    print(f"  Train: {len(train_samples)} → {train_json_path}")
    print(f"  Val:   {len(val_samples)} → {val_json_path}")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {k:>20s}: {v}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
