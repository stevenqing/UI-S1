#!/usr/bin/env python3
"""
Experiment 0.2: Zero-Shot Verification Test

Test if the current SFT model can judge "did my action succeed?" without
any verification-specific training.

For each divergence step from paired trajectories:
1. Construct a verification prompt with before/after screenshots + sub-goal
2. Ask the model: "Did this action achieve the sub-goal? YES or NO"
3. Measure classification accuracy

We construct both positive examples (from success trajectories, where actions
DO succeed) and negative examples (from fail trajectories at divergence points,
where actions FAIL).

Success criteria: Accuracy > 70% → model already has verification capability

Usage:
    python scripts/exp0/exp0_2_zeroshot_verification.py \
        --model_name gui360_lora_v4_ckpt354 \
        --num_examples 100

    # Analysis only:
    python scripts/exp0/exp0_2_zeroshot_verification.py --analyze_only
"""

import argparse
import base64
import json
import os
import re
import sys
from io import BytesIO
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.exp0.data_utils import (
    SUCCESS_TEST_DIR,
    SUCCESS_TRAIN_DIR,
    get_screenshot_path,
    load_trajectory,
    normalize_action,
    scan_trajectory_files,
)

VERIFICATION_PROMPT_TEMPLATE = """You are evaluating whether a GUI action achieved its intended goal.

Task: {task}
Sub-goal for this step: {sub_goal}
Action taken: {action_description}

You are given two screenshots:
1. BEFORE the action (first image)
2. AFTER the action (second image)

Based on the visual difference between the two screenshots, did the action successfully achieve the sub-goal?

Answer with exactly one word: YES or NO
Then briefly explain your reasoning in one sentence."""


def image_to_data_url(image_path: str) -> str:
    """Convert image file to base64 data URL."""
    from PIL import Image

    image = Image.open(image_path)
    buf = BytesIO()
    image.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64_str}"


def describe_action(action: dict) -> str:
    """Create human-readable description of an action."""
    action_type = action.get("action_type", "unknown")
    coord = action.get("coordinate")
    text = action.get("text")

    if action_type == "click" and coord:
        return f"Click at ({coord[0]:.0f}, {coord[1]:.0f})"
    elif action_type == "type" and text:
        coord_str = f" at ({coord[0]:.0f}, {coord[1]:.0f})" if coord else ""
        return f'Type "{text}"{coord_str}'
    elif action_type == "drag" and coord:
        coord2 = action.get("coordinate2")
        if coord2:
            return f"Drag from ({coord[0]:.0f}, {coord[1]:.0f}) to ({coord2[0]:.0f}, {coord2[1]:.0f})"
        return f"Drag from ({coord[0]:.0f}, {coord[1]:.0f})"
    elif action_type in ("scroll", "wheel_mouse_input"):
        direction = action.get("direction", "unknown")
        return f"Scroll {direction}"
    else:
        return f"{action_type}" + (f" at ({coord[0]:.0f}, {coord[1]:.0f})" if coord else "")


def get_sub_goal(step: dict) -> str:
    """Extract sub-goal from step's evaluation sub_scores or subtask field."""
    # Try subtask field (from fail data)
    step_data = step.get("step", {})
    subtask = step_data.get("subtask", "")
    if subtask:
        return subtask

    # Try evaluation sub_scores
    evaluation = step.get("evaluation", {})
    sub_scores = evaluation.get("sub_scores", {})
    if sub_scores:
        # Return the first incomplete sub-score as the current goal
        for goal, status in sub_scores.items():
            if status.lower() in ("no", "partial"):
                return goal
        # All complete: return last one
        return list(sub_scores.keys())[-1] if sub_scores else ""

    # Try thought field
    thought = step_data.get("thought", step.get("thought", ""))
    if thought:
        return thought[:200]

    return "Complete the current step"


def build_verification_examples(max_examples: int = 100) -> list[dict]:
    """
    Build verification examples from success trajectories.

    Since fail screenshots are not extracted (stored as tar.gz archives),
    we construct both positive and negative examples from success trajectories:
    - Positive: (before_i, action_i, after_i) from consecutive steps → YES
    - Negative: (before_i, action_i, after_j where j≠i+1) scrambled pairs → NO

    Returns list of dicts with label, screenshots, action description, etc.
    """
    import random
    random.seed(42)

    half = max_examples // 2  # half positive, half negative

    # Collect valid step pairs from success trajectories
    print("Scanning success trajectories for screenshot pairs...")
    valid_steps = []  # list of (task, sub_goal, action_desc, before_path, after_path, eid, step_idx)

    # Use both test and train success data
    for data_dir, img_source in [(SUCCESS_TEST_DIR, "test"), (SUCCESS_TRAIN_DIR, "train")]:
        if not data_dir.exists():
            continue
        files = scan_trajectory_files(data_dir)
        for eid, fpath in list(files.items())[:500]:
            try:
                steps = load_trajectory(fpath)
                for i in range(len(steps) - 1):
                    action = normalize_action(steps[i])
                    if not action["action_type"]:
                        continue
                    before_path = get_screenshot_path(steps[i], source=img_source)
                    after_path = get_screenshot_path(steps[i + 1], source=img_source)
                    if not before_path or not after_path:
                        continue
                    if not os.path.exists(before_path) or not os.path.exists(after_path):
                        continue
                    task = steps[0].get("request", "")
                    valid_steps.append({
                        "task": task,
                        "sub_goal": get_sub_goal(steps[i]),
                        "action_description": describe_action(action),
                        "before_screenshot": before_path,
                        "after_screenshot": after_path,
                        "execution_id": eid,
                        "step_idx": i,
                    })
            except Exception:
                continue
        if len(valid_steps) >= max_examples * 3:
            break

    print(f"Found {len(valid_steps)} valid step pairs with screenshots")

    if len(valid_steps) < 10:
        print("ERROR: Not enough screenshot pairs found!")
        return []

    random.shuffle(valid_steps)
    examples = []

    # Positive examples: correct (before, action, after) triples
    positive_count = 0
    for step_info in valid_steps:
        if positive_count >= half:
            break
        examples.append({
            "label": "YES",
            "source": "success_positive",
            **step_info,
        })
        positive_count += 1

    # Negative examples: scrambled — use correct before + action but WRONG after screenshot
    # Pick after_screenshot from a different step/trajectory
    negative_count = 0
    after_pool = [s["after_screenshot"] for s in valid_steps]
    for step_info in valid_steps[half:]:
        if negative_count >= half:
            break
        # Find a different after screenshot (from a different execution or far-away step)
        for _ in range(20):
            wrong_after = random.choice(after_pool)
            if wrong_after != step_info["after_screenshot"]:
                break
        else:
            continue

        examples.append({
            "label": "NO",
            "task": step_info["task"],
            "sub_goal": step_info["sub_goal"],
            "action_description": step_info["action_description"],
            "before_screenshot": step_info["before_screenshot"],
            "after_screenshot": wrong_after,
            "source": "scrambled_negative",
            "execution_id": step_info["execution_id"],
            "step_idx": step_info["step_idx"],
        })
        negative_count += 1

    random.shuffle(examples)
    print(f"Built {len(examples)} verification examples ({positive_count} positive, {negative_count} negative)")
    return examples






def run_verification(args):
    """Run zero-shot verification test."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build examples
    examples = build_verification_examples(args.num_examples)
    if not examples:
        print("ERROR: Could not build any verification examples!")
        return

    # Save examples metadata
    examples_path = output_dir / "verification_examples.json"
    with open(examples_path, "w") as f:
        json.dump(examples, f, indent=2)

    from openai import OpenAI

    client = OpenAI(api_key="EMPTY", base_url=args.endpoint, timeout=300)

    results_path = output_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()

    for i, ex in enumerate(examples):
        # Build verification prompt
        sub_goal = ex["sub_goal"]
        prompt_text = VERIFICATION_PROMPT_TEMPLATE.format(
            task=ex["task"],
            sub_goal=sub_goal,
            action_description=ex["action_description"],
        )

        # Build message with before/after images
        try:
            before_url = image_to_data_url(ex["before_screenshot"])
            after_url = image_to_data_url(ex["after_screenshot"])
        except Exception as e:
            print(f"Error loading images for example {i}: {e}")
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": before_url}},
                    {"type": "image_url", "image_url": {"url": after_url}},
                ],
            }
        ]

        # Call model
        try:
            response = client.chat.completions.create(
                model=args.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=200,
            )
            model_response = response.choices[0].message.content
        except Exception as e:
            print(f"API error for example {i}: {e}")
            model_response = ""

        # Parse YES/NO
        response_upper = model_response.upper().strip()
        if response_upper.startswith("YES"):
            prediction = "YES"
        elif response_upper.startswith("NO"):
            prediction = "NO"
        else:
            # Look for YES/NO anywhere in first 20 chars
            first_part = response_upper[:20]
            if "YES" in first_part:
                prediction = "YES"
            elif "NO" in first_part:
                prediction = "NO"
            else:
                prediction = "UNKNOWN"

        result = {
            "idx": i,
            "label": ex["label"],
            "prediction": prediction,
            "correct": prediction == ex["label"],
            "source": ex["source"],
            "model_response": model_response[:500],
            "execution_id": ex["execution_id"],
            "step_idx": ex["step_idx"],
        }

        with open(results_path, "a") as f:
            f.write(json.dumps(result) + "\n")

        if (i + 1) % 10 == 0:
            correct_so_far = sum(
                1 for line in open(results_path)
                if json.loads(line).get("correct", False)
            )
            print(f"Progress: {i + 1}/{len(examples)}  Running accuracy: {correct_so_far}/{i + 1}")

    print(f"Results saved to: {results_path}")
    return results_path


def analyze_results(results_path: str):
    """Analyze verification results."""
    results = []
    with open(results_path) as f:
        for line in f:
            results.append(json.loads(line))

    if not results:
        print("No results to analyze!")
        return

    n = len(results)
    print("\n" + "=" * 60)
    print(f"  Experiment 0.2: Zero-Shot Verification Results (N={n})")
    print("=" * 60)

    # Overall accuracy
    correct = sum(1 for r in results if r["correct"])
    unknown = sum(1 for r in results if r["prediction"] == "UNKNOWN")
    print(f"\n  Overall accuracy: {correct}/{n} ({correct / n:.1%})")
    print(f"  Unknown/unparseable: {unknown}/{n} ({unknown / n:.1%})")

    # Per-label accuracy
    for label in ["YES", "NO"]:
        label_results = [r for r in results if r["label"] == label]
        if label_results:
            label_correct = sum(1 for r in label_results if r["correct"])
            print(f"  {label} accuracy: {label_correct}/{len(label_results)} ({label_correct / len(label_results):.1%})")

    # Per-source breakdown
    print("\n  Per-source breakdown:")
    sources = set(r["source"] for r in results)
    for source in sorted(sources):
        src_results = [r for r in results if r["source"] == source]
        src_correct = sum(1 for r in src_results if r["correct"])
        print(f"    {source}: {src_correct}/{len(src_results)} ({src_correct / len(src_results):.1%})")

    # Confusion matrix
    tp = sum(1 for r in results if r["label"] == "YES" and r["prediction"] == "YES")
    tn = sum(1 for r in results if r["label"] == "NO" and r["prediction"] == "NO")
    fp = sum(1 for r in results if r["label"] == "NO" and r["prediction"] == "YES")
    fn = sum(1 for r in results if r["label"] == "YES" and r["prediction"] == "NO")

    print(f"\n  Confusion Matrix:")
    print(f"                  Predicted YES  Predicted NO  Unknown")
    print(f"    Actual YES:   {tp:>10}     {fn:>10}     {sum(1 for r in results if r['label']=='YES' and r['prediction']=='UNKNOWN'):>7}")
    print(f"    Actual NO:    {fp:>10}     {tn:>10}     {sum(1 for r in results if r['label']=='NO' and r['prediction']=='UNKNOWN'):>7}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n  Precision: {precision:.1%}  Recall: {recall:.1%}  F1: {f1:.1%}")

    # Verdict
    overall_acc = correct / n if n > 0 else 0
    print(f"\n  VERDICT: {'PASS' if overall_acc > 0.7 else 'NEEDS TRAINING'}")
    if overall_acc > 0.7:
        print("    Model already has reasonable verification ability!")
    else:
        print(f"    Accuracy {overall_acc:.1%} < 70% threshold. Verification SFT needed.")

    print("=" * 60)

    # Save summary
    summary = {
        "n_examples": n,
        "overall_accuracy": correct / n if n > 0 else 0,
        "yes_accuracy": (tp / (tp + fn)) if (tp + fn) > 0 else 0,
        "no_accuracy": (tn / (tn + fp)) if (tn + fp) > 0 else 0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "unknown_rate": unknown / n if n > 0 else 0,
    }
    summary_path = Path(results_path).parent / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Experiment 0.2: Zero-Shot Verification Test")
    parser.add_argument("--model_name", type=str, default="gui360_lora_v4_ckpt354")
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--endpoint", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "exp0_2"))
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--results_path", type=str, default="")

    args = parser.parse_args()

    if args.analyze_only:
        path = args.results_path or str(Path(args.output_dir) / "results.jsonl")
        analyze_results(path)
    else:
        results_path = run_verification(args)
        if results_path:
            analyze_results(str(results_path))


if __name__ == "__main__":
    main()
