#!/usr/bin/env python3
"""
Experiment 1.4: SFT v3 Grounding Under Action Prediction Prompt

Question: Is SFT v3's 80% grounding accuracy specific to the grounding prompt,
or is it a general capability that transfers to action prediction format?

Method:
  1. Call SFT v3 with the action prediction prompt (tool_call format)
  2. Extract coordinates from the tool_call output
  3. Evaluate using grounding eval criteria (coord in bbox)

Success criteria:
  - >70% → universal capability, can use directly in action pipeline
  - <50% → prompt-specific, need dedicated grounding prompt as bridge

Usage:
    python scripts/exp1/exp1_4_sft_v3_action_prompt.py \
        --endpoint http://localhost:19815/v1

    python scripts/exp1/exp1_4_sft_v3_action_prompt.py \
        --analyze_only --results_dir outputs/exp1_4
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.exp0.exp0_1_uncertainty_analysis import (
    _json_default,
    extract_coordinate,
    image_to_data_url,
    parse_tool_call,
)
from scripts.exp0.data_utils import DATASET_ROOT, load_trajectory, is_coord_in_bbox
from scripts.exp1.grounding_utils import (
    evaluate_grounding,
    preprocess_image,
    parse_coordinate_response,
    transform_coord_to_original,
    GROUNDING_PROMPT,
)


def load_grounding_samples(dataset_root: str, max_samples: int = 0):
    """Load grounding samples from GUI-360 test set."""
    test_root = Path(dataset_root) / "test"
    data_path = test_root / "data"
    image_path = test_root / "image"

    samples = []
    for domain in sorted(data_path.iterdir()):
        if not domain.is_dir():
            continue
        for category in sorted(domain.iterdir()):
            if not category.is_dir():
                continue
            success_dir = category / "success"
            if not success_dir.exists():
                continue

            for jsonl_file in sorted(success_dir.glob("*.jsonl")):
                steps = load_trajectory(jsonl_file)
                for i, step_data in enumerate(steps):
                    step = step_data.get("step", {})
                    action = step.get("action", {})
                    tags = step.get("tags", [])

                    if "grounding" not in tags:
                        continue

                    if not action.get("rectangle"):
                        continue

                    # Only coordinate-based actions
                    if action.get("function") not in {"click", "right_click", "double_click"}:
                        continue

                    screenshot = image_path / domain.name / category.name / step.get("screenshot_clean", "")
                    if not screenshot.exists():
                        continue

                    # Build previous actions for action prompt context
                    prev_actions = []
                    for j in range(i):
                        prev_step = steps[j].get("step", {})
                        prev_actions.append(f"Step {j+1}: {prev_step.get('thought', '')}")

                    sample = {
                        "sample_id": f"{domain.name}_{category.name}_{jsonl_file.stem}_{i+1}",
                        "request": step_data.get("request", ""),
                        "thought": step.get("thought", ""),
                        "screenshot": str(screenshot),
                        "domain": domain.name,
                        "category": category.name,
                        "previous_actions": prev_actions,
                        "gt_function": action.get("function", ""),
                        "gt_args": action.get("args", {}),
                        "gt_rectangle": action.get("rectangle", {}),
                    }
                    samples.append(sample)

                    if max_samples > 0 and len(samples) >= max_samples:
                        return samples

    return samples


def call_with_action_prompt(client, model_name, sample):
    """Call SFT v3 with the action prediction prompt format.
    Uses smart_resize for consistent coordinate space."""
    data_url, orig_wh, resized_wh = preprocess_image(sample["screenshot"])
    prev_text = "\n".join(sample["previous_actions"]) if sample["previous_actions"] else "None"

    user_text = (
        f"You are a helpful assistant that helps users interact with their computer.\n\n"
        f"Task: {sample['request']}\n\n"
        f"Previous actions:\n{prev_text}\n\n"
        f"Screenshot resolution: {resized_wh}\n\n"
        f"Based on the screenshot and task, determine the next action to take. "
        f"Output your response as a tool_call in the format:\n"
        f"<tool_call>\n"
        f'{{"function": "action_name", "args": {{"key": "value"}}, "status": "CONTINUE|FINISH"}}\n'
        f"</tool_call>"
    )

    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": user_text},
        ]}
    ]

    try:
        response = client.chat.completions.create(
            model=model_name, messages=messages, temperature=0.0, max_tokens=2048,
        )
        return response.choices[0].message.content or "", orig_wh, resized_wh
    except Exception as e:
        print(f"API call failed: {e}")
        return "", orig_wh, resized_wh, orig_wh, resized_wh


def call_with_grounding_prompt(client, model_name, sample):
    """Call SFT v3 with the standard grounding prompt (baseline).
    Uses smart_resize for consistent coordinate space."""
    data_url, orig_wh, resized_wh = preprocess_image(sample["screenshot"])
    user_text = GROUNDING_PROMPT.format(instruction=sample["thought"])

    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": user_text},
        ]}
    ]

    try:
        response = client.chat.completions.create(
            model=model_name, messages=messages, temperature=0.0, max_tokens=512,
        )
        return response.choices[0].message.content or "", orig_wh, resized_wh
    except Exception as e:
        print(f"API call failed: {e}")
        return "", orig_wh, resized_wh


def parse_coord_from_response(text: str) -> list | None:
    """Extract coordinate from either grounding or action prediction response."""
    # Try tool_call format
    action = parse_tool_call(text)
    coord = extract_coordinate(action)
    if coord:
        return coord

    # Try <coordinate> tag
    match = re.search(r"<coordinate>\s*\[?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]?\s*</coordinate>", text)
    if match:
        return [float(match.group(1)), float(match.group(2))]

    # Try JSON coordinates
    match = re.search(r'"coordinates"\s*:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', text)
    if match:
        return [float(match.group(1)), float(match.group(2))]

    # Try bare coordinate
    match = re.search(r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', text)
    if match:
        return [float(match.group(1)), float(match.group(2))]

    return None


def run_experiment(args):
    """Run grounding generalization test."""
    from openai import OpenAI

    print("Loading grounding samples...")
    samples = load_grounding_samples(str(DATASET_ROOT), max_samples=args.num_samples)
    print(f"Loaded {len(samples)} grounding samples")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"

    completed_ids = set()
    if results_path.exists() and not args.overwrite:
        with open(results_path) as f:
            for line in f:
                completed_ids.add(json.loads(line)["sample_id"])
        print(f"Resuming: {len(completed_ids)} already completed")

    client = OpenAI(api_key="EMPTY", base_url=args.endpoint, timeout=300)
    total = len(samples)
    t0 = time.time()

    for i, sample in enumerate(samples):
        if sample["sample_id"] in completed_ids:
            continue

        # Call with grounding prompt (baseline)
        grounding_resp, g_orig_wh, g_resized_wh = call_with_grounding_prompt(client, args.model_name, sample)
        grounding_coord = parse_coordinate_response(grounding_resp)
        if grounding_coord is not None:
            grounding_coord = transform_coord_to_original(grounding_coord, g_orig_wh, g_resized_wh)

        # Call with action prediction prompt
        action_resp, a_orig_wh, a_resized_wh = call_with_action_prompt(client, args.model_name, sample)
        action_coord = parse_coord_from_response(action_resp)
        if action_coord is not None:
            action_coord = transform_coord_to_original(action_coord, a_orig_wh, a_resized_wh)

        # Evaluate both
        gt_rect = sample["gt_rectangle"]

        grounding_eval = evaluate_grounding(grounding_coord, gt_rect)
        action_eval = evaluate_grounding(action_coord, gt_rect)

        result = {
            "sample_id": sample["sample_id"],
            "domain": sample["domain"],
            "gt_function": sample["gt_function"],
            "grounding_coord": grounding_coord,
            "action_coord": action_coord,
            "grounding_correct": grounding_eval["correct"],
            "grounding_distance": grounding_eval["distance"],
            "action_correct": action_eval["correct"],
            "action_distance": action_eval["distance"],
        }

        with open(results_path, "a") as f:
            f.write(json.dumps(result, default=_json_default) + "\n")

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (total - i - 1)
            print(f"Progress: {i + 1}/{total}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    print(f"Results saved to: {results_path}")


def analyze_results(results_dir: str):
    """Analyze prompt generalization results."""
    results_path = Path(results_dir) / "results.jsonl"
    if not results_path.exists():
        print(f"No results at {results_path}")
        return

    results = [json.loads(line) for line in open(results_path)]
    n = len(results)

    print("\n" + "=" * 70)
    print("  Experiment 1.4: SFT v3 Grounding Under Action Prompt")
    print("=" * 70)

    gnd_acc = sum(1 for r in results if r["grounding_correct"]) / n
    act_acc = sum(1 for r in results if r["action_correct"]) / n

    print(f"\n  Grounding prompt (baseline): {gnd_acc:.1%} ({sum(1 for r in results if r['grounding_correct'])}/{n})")
    print(f"  Action prompt:               {act_acc:.1%} ({sum(1 for r in results if r['action_correct'])}/{n})")
    print(f"  Delta:                        {act_acc - gnd_acc:+.1%}")

    # Confusion matrix
    both_correct = sum(1 for r in results if r["grounding_correct"] and r["action_correct"])
    gnd_only = sum(1 for r in results if r["grounding_correct"] and not r["action_correct"])
    act_only = sum(1 for r in results if not r["grounding_correct"] and r["action_correct"])
    both_wrong = sum(1 for r in results if not r["grounding_correct"] and not r["action_correct"])

    print(f"\n  Confusion matrix:")
    print(f"    Both correct:    {both_correct:>5d} ({both_correct/n:.1%})")
    print(f"    Grounding only:  {gnd_only:>5d} ({gnd_only/n:.1%})")
    print(f"    Action only:     {act_only:>5d} ({act_only/n:.1%})")
    print(f"    Both wrong:      {both_wrong:>5d} ({both_wrong/n:.1%})")

    # Go/No-Go
    print(f"\n  GO/NO-GO CHECK:")
    if act_acc > 0.70:
        print(f"    ✓ {act_acc:.1%} > 70% → Universal capability, use directly in action pipeline")
    elif act_acc > 0.50:
        print(f"    ~ {act_acc:.1%} → Partial transfer (50-70%), may need prompt bridge")
    else:
        print(f"    ✗ {act_acc:.1%} < 50% → Prompt-specific, need dedicated grounding prompt as bridge")

    # Per-domain
    domains = sorted(set(r["domain"] for r in results))
    if len(domains) > 1:
        print(f"\n  Per-Domain:")
        for domain in domains:
            dr = [r for r in results if r["domain"] == domain]
            nd = len(dr)
            g = sum(1 for r in dr if r["grounding_correct"]) / nd
            a = sum(1 for r in dr if r["action_correct"]) / nd
            print(f"    {domain:>6s} (N={nd}): grounding={g:.1%}  action={a:.1%}  Δ={a-g:+.1%}")

    summary = {
        "n_samples": n,
        "grounding_prompt_accuracy": gnd_acc,
        "action_prompt_accuracy": act_acc,
        "delta": act_acc - gnd_acc,
        "verdict": "universal" if act_acc > 0.70 else "partial" if act_acc > 0.50 else "prompt_specific",
    }
    summary_path = Path(results_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to: {summary_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Exp 1.4: SFT v3 Action Prompt Generalization")
    parser.add_argument("--model_name", type=str,
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/llamafactory/output/gui360_full_sft_v3_grounding")
    parser.add_argument("--endpoint", type=str, default="http://localhost:19815/v1")
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "outputs" / "exp1_4"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--results_dir", type=str, default="")

    args = parser.parse_args()

    if args.analyze_only:
        analyze_results(args.results_dir or args.output_dir)
    else:
        run_experiment(args)
        analyze_results(args.output_dir)


if __name__ == "__main__":
    main()
