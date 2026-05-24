#!/usr/bin/env python3
"""Evaluate grounder + SFT model combination.

Phase 1: Deploy gui_grounding model → predict coordinates for all steps
Phase 2: Deploy SFT model → predict action type/text for all steps
Phase 3: Combine: SFT action type + grounder coordinates → evaluate

Also measures grounder standalone coordinate accuracy.

Usage:
    # Phase 1: Generate grounder predictions
    python v15_gui_360/eval_grounder_combined.py phase1 \
        --test_data v12_gui_360/data/gui360_test_1000_balanced.jsonl \
        --api_url http://localhost:8000/v1 \
        --model_name gui_grounding \
        --output_file v15_gui_360/outputs/preliminary_tests/grounder_predictions.json \
        --threads 64

    # Phase 2: Generate SFT predictions (reuse existing eval results)

    # Phase 3: Combine and evaluate
    python v15_gui_360/eval_grounder_combined.py phase3 \
        --grounder_preds v15_gui_360/outputs/preliminary_tests/grounder_predictions.json \
        --sft_results v15_gui_360/outputs/history_ablation/gt_history_full/eval_results_*.json \
        --test_data v12_gui_360/data/gui360_test_1000_balanced.jsonl \
        --output_dir v15_gui_360/outputs/preliminary_tests/grounder_combined
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v13_gui_360.eval_gui360_template import _format_action_for_history, _encode_screenshot
from v13_gui_360.reward import compute_step_reward, _normalize_action_type

# ═══════════════════════════════════════════════════════════════════════
# Grounder prompt (matches V16 format)
# ═══════════════════════════════════════════════════════════════════════

GROUNDER_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen and user instruction, you need to output the position of the element you will operate.

The instruction is:
{instruction}

The history of actions are:
{history}

Output the coordinate of the element you will operate within <coordinate></coordinate> tag:
<coordinate> [x, y] </coordinate>"""


def parse_grounder_output(text: str) -> Optional[List[float]]:
    """Parse <coordinate> [x, y] </coordinate> from grounder output."""
    m = re.search(r'<coordinate>\s*\[?\s*([\d.]+)\s*,\s*([\d.]+)\s*\]?\s*</coordinate>', text)
    if m:
        return [float(m.group(1)), float(m.group(2))]
    # Fallback: try to find any coordinate pattern
    m = re.search(r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', text)
    if m:
        return [float(m.group(1)), float(m.group(2))]
    return None


def generate_grounder_predictions(client, model_name, episode):
    """Generate grounder predictions for all steps of an episode (GT history)."""
    episode_id = episode["episode_id"]
    goal = episode["goal"]
    steps = episode["steps"]

    history = []
    predictions = []

    for i, step in enumerate(steps):
        screenshot = step["screenshot"]
        gt_action = step["action"]

        history_text = "\n".join(history) if history else "None"
        prompt_text = GROUNDER_PROMPT.format(
            instruction=goal,
            history=history_text,
        )

        b64 = _encode_screenshot(screenshot)

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt_text},
                ]}],
                max_tokens=128,
                temperature=0.0,
            )
            pred_text = response.choices[0].message.content or ""
        except Exception as e:
            print(f"  [ep {episode_id}] step {i} error: {e}")
            pred_text = ""

        coord = parse_grounder_output(pred_text)
        predictions.append({
            "step_idx": i,
            "pred_text": pred_text[:200],
            "pred_coordinate": coord,
        })

        # GT history (teacher-forced)
        history.append(_format_action_for_history(gt_action, i + 1))

    return {
        "episode_id": episode_id,
        "predictions": predictions,
    }


def run_phase1(args):
    """Phase 1: Generate grounder predictions."""
    episodes = []
    with open(args.test_data) as f:
        for line in f:
            episodes.append(json.loads(line))
    print(f"Loaded {len(episodes)} episodes")

    client = OpenAI(base_url=args.api_url, api_key="dummy")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    results = {}
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(generate_grounder_predictions, client, args.model_name, ep): ep["episode_id"]
            for ep in episodes
        }
        pbar = tqdm(total=len(episodes), desc="Grounder predictions")
        for future in as_completed(futures):
            result = future.result()
            results[result["episode_id"]] = result
            pbar.update(1)
        pbar.close()

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved grounder predictions to {args.output_file}")


def run_phase3(args):
    """Phase 3: Combine grounder coords + SFT action type, evaluate."""
    import glob

    # Load grounder predictions
    with open(args.grounder_preds) as f:
        grounder = json.load(f)

    # Load SFT eval results
    sft_files = sorted(glob.glob(args.sft_results))
    with open(sft_files[-1]) as f:
        sft = json.load(f)

    # Load test data for GT
    episodes = {}
    with open(args.test_data) as f:
        for line in f:
            ep = json.loads(line)
            episodes[str(ep["episode_id"])] = ep

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Analysis 1: Grounder standalone coordinate accuracy ──
    print("=" * 70)
    print("Grounder Standalone Coordinate Accuracy")
    print("=" * 70)

    grounder_correct = 0
    grounder_total = 0
    grounder_dists = []

    for eid, ep in episodes.items():
        gpreds = grounder.get(eid, {}).get("predictions", [])
        for step in ep["steps"]:
            idx = step["step_idx"]
            gt_action = step["action"]
            gt_type = gt_action.get("action", "")
            gt_coord = gt_action.get("coordinate")

            if gt_type != "click" or not gt_coord or gt_coord == [None, None]:
                continue

            if idx >= len(gpreds):
                continue

            pred_coord = gpreds[idx].get("pred_coordinate")
            if not pred_coord:
                grounder_total += 1
                continue

            grounder_total += 1
            image_w = step.get("image_w", 1040)
            image_h = step.get("image_h", 736)
            dx = pred_coord[0] - gt_coord[0]
            dy = pred_coord[1] - gt_coord[1]
            dist = (dx**2 + dy**2) ** 0.5
            grounder_dists.append(dist)

            # Use same threshold as reward function (5% of image diagonal)
            diag = (image_w**2 + image_h**2) ** 0.5
            if dist < diag * 0.05:
                grounder_correct += 1

    print(f"Click steps with GT coords: {grounder_total}")
    print(f"Grounder coord accuracy (5% threshold): {grounder_correct}/{grounder_total} = "
          f"{grounder_correct/grounder_total*100:.1f}%")
    if grounder_dists:
        arr = np.array(grounder_dists)
        print(f"Mean distance: {arr.mean():.1f}px, Median: {np.median(arr):.1f}px")
        print(f"  <25px:  {(arr < 25).sum()} ({(arr < 25).sum()/len(arr)*100:.1f}%)")
        print(f"  <50px:  {(arr < 50).sum()} ({(arr < 50).sum()/len(arr)*100:.1f}%)")
        print(f"  >100px: {(arr >= 100).sum()} ({(arr >= 100).sum()/len(arr)*100:.1f}%)")

    # ── Analysis 2: SFT standalone type accuracy ──
    print()
    print("=" * 70)
    print("SFT Action Type Accuracy (from existing eval)")
    print("=" * 70)

    sft_type_correct = 0
    sft_type_total = 0
    for eid, ep in sft.items():
        for s in ep["steps"]:
            sft_type_total += 1
            gt_type = _normalize_action_type(s.get("gt_type") or "")
            pred_type = _normalize_action_type(s.get("pred_type") or "")
            if gt_type == pred_type:
                sft_type_correct += 1

    print(f"SFT type accuracy: {sft_type_correct}/{sft_type_total} = "
          f"{sft_type_correct/sft_type_total*100:.1f}%")

    # ── Analysis 3: Combined — SFT action type + grounder coords ──
    print()
    print("=" * 70)
    print("Combined: SFT Action Type + Grounder Coordinates")
    print("=" * 70)

    combined_correct = 0
    combined_total = 0
    sft_only_correct = 0
    grounder_helped = 0
    grounder_hurt = 0

    for eid, ep_data in episodes.items():
        sft_ep = sft.get(eid, {})
        gpreds = grounder.get(eid, {}).get("predictions", [])
        sft_steps = sft_ep.get("steps", [])

        for step in ep_data["steps"]:
            idx = step["step_idx"]
            gt_action = step["action"]
            image_w = step.get("image_w", 1040)
            image_h = step.get("image_h", 736)

            if idx >= len(sft_steps) or idx >= len(gpreds):
                continue

            sft_step = sft_steps[idx]
            sft_pred = sft_step.get("pred_action") or {}
            sft_success = sft_step.get("success", False)

            # Build combined action: SFT type/text + grounder coordinates
            combined_action = dict(sft_pred) if sft_pred else {}
            grounder_coord = gpreds[idx].get("pred_coordinate")

            gt_type = gt_action.get("action", "")
            if grounder_coord and gt_type in ("click", "long_press"):
                # For click actions, replace SFT coordinates with grounder's
                combined_action["coordinate"] = grounder_coord

            # Evaluate combined action
            if combined_action:
                fake_text = f"<action>{json.dumps(combined_action)}</action>"
            else:
                fake_text = ""

            reward, info = compute_step_reward(fake_text, gt_action, image_w, image_h)
            combined_success = reward >= 0.5

            combined_total += 1
            if combined_success:
                combined_correct += 1
            if sft_success:
                sft_only_correct += 1

            if combined_success and not sft_success:
                grounder_helped += 1
            elif not combined_success and sft_success:
                grounder_hurt += 1

    print(f"Total steps: {combined_total}")
    print(f"SFT only:    {sft_only_correct}/{combined_total} = {sft_only_correct/combined_total*100:.1f}%")
    print(f"Combined:    {combined_correct}/{combined_total} = {combined_correct/combined_total*100:.1f}%")
    print(f"Grounder helped (SFT wrong → combined right): {grounder_helped}")
    print(f"Grounder hurt  (SFT right → combined wrong):  {grounder_hurt}")
    print(f"Net improvement: {grounder_helped - grounder_hurt} steps")

    # ── Analysis 4: By action type ──
    print()
    print("=" * 70)
    print("Combined vs SFT by GT Action Type")
    print("=" * 70)

    type_stats = defaultdict(lambda: {"sft": 0, "combined": 0, "total": 0})
    for eid, ep_data in episodes.items():
        sft_ep = sft.get(eid, {})
        gpreds = grounder.get(eid, {}).get("predictions", [])
        sft_steps = sft_ep.get("steps", [])

        for step in ep_data["steps"]:
            idx = step["step_idx"]
            gt_action = step["action"]
            gt_type = gt_action.get("action", "")
            image_w = step.get("image_w", 1040)
            image_h = step.get("image_h", 736)

            if idx >= len(sft_steps) or idx >= len(gpreds):
                continue

            sft_step = sft_steps[idx]
            sft_pred = sft_step.get("pred_action") or {}

            combined_action = dict(sft_pred) if sft_pred else {}
            grounder_coord = gpreds[idx].get("pred_coordinate")
            if grounder_coord and gt_type in ("click", "long_press"):
                combined_action["coordinate"] = grounder_coord

            if combined_action:
                reward, _ = compute_step_reward(
                    f"<action>{json.dumps(combined_action)}</action>",
                    gt_action, image_w, image_h
                )
            else:
                reward = 0

            type_stats[gt_type]["total"] += 1
            if sft_step.get("success"):
                type_stats[gt_type]["sft"] += 1
            if reward >= 0.5:
                type_stats[gt_type]["combined"] += 1

    for gt_type in ["click", "type", "swipe"]:
        s = type_stats.get(gt_type, {"sft": 0, "combined": 0, "total": 0})
        if s["total"] > 0:
            print(f"  {gt_type:<8}: SFT={s['sft']}/{s['total']}={s['sft']/s['total']*100:.1f}%  "
                  f"Combined={s['combined']}/{s['total']}={s['combined']/s['total']*100:.1f}%  "
                  f"delta={s['combined']-s['sft']:+d}")

    # Save results
    summary = {
        "grounder_coord_accuracy": grounder_correct / grounder_total if grounder_total > 0 else 0,
        "sft_type_accuracy": sft_type_correct / sft_type_total if sft_type_total > 0 else 0,
        "sft_step_sr": sft_only_correct / combined_total if combined_total > 0 else 0,
        "combined_step_sr": combined_correct / combined_total if combined_total > 0 else 0,
        "grounder_helped": grounder_helped,
        "grounder_hurt": grounder_hurt,
        "net_improvement": grounder_helped - grounder_hurt,
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.output_dir, f"combined_summary_{ts}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="phase")

    # Phase 1: grounder predictions
    p1 = subparsers.add_parser("phase1")
    p1.add_argument("--test_data", required=True)
    p1.add_argument("--api_url", default="http://localhost:8000/v1")
    p1.add_argument("--model_name", required=True)
    p1.add_argument("--output_file", required=True)
    p1.add_argument("--threads", type=int, default=64)

    # Phase 3: combine and evaluate
    p3 = subparsers.add_parser("phase3")
    p3.add_argument("--grounder_preds", required=True)
    p3.add_argument("--sft_results", required=True)
    p3.add_argument("--test_data", required=True)
    p3.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    if args.phase == "phase1":
        run_phase1(args)
    elif args.phase == "phase3":
        run_phase3(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
