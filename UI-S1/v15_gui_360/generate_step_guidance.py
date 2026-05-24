#!/usr/bin/env python3
"""Generate per-step guidance from a base model for all episodes.

Phase 1 of the guided eval experiment: use a general-purpose VLM (not SFT'd)
to analyze each screenshot + goal + GT history and describe what action
should be taken next. Save guidance to JSON for Phase 2.

Usage:
    python v15_gui_360/generate_step_guidance.py \
        --test_data v12_gui_360/data/gui360_test_1000_balanced.jsonl \
        --api_url http://localhost:8000/v1 \
        --model_name Qwen2.5-VL-7B-Instruct \
        --output_file v15_gui_360/outputs/preliminary_tests/step_guidance.json \
        --threads 64
"""

import argparse
import base64
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

from openai import OpenAI
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GUIDANCE_PROMPT = """You are an expert GUI automation assistant. Given a screenshot of the current screen, a user instruction, and the history of actions already taken, analyze the current state and describe what specific action should be taken next.

The instruction is:
{instruction}

The history of actions are:
{history}

Based on the screenshot and the context above, describe in 1-2 sentences:
1. What is the current state of the screen?
2. What specific action should be taken next? (e.g., "Click the 'Insert' tab at the top menu bar", "Type 'hello' in the search box", "Scroll down to find the settings option")

Be specific about the target element and action type (click, type, drag, or scroll). Focus on WHAT to do, not HOW to format it."""


def _format_gt_action_for_history(action, step_id):
    """Format GT action for history (same as eval script)."""
    if action is None:
        return f"Step {step_id}: (no action)"

    atype = action.get("action", "")
    coord = action.get("coordinate")

    def _valid_coord(c):
        return c and isinstance(c, (list, tuple)) and len(c) >= 2 and c[0] is not None

    if atype == "click":
        if _valid_coord(coord):
            return f"Step {step_id}: click(coordinate=[{int(float(coord[0]))}, {int(float(coord[1]))}])"
        return f"Step {step_id}: click()"
    elif atype == "type":
        text = action.get("text", "")
        if _valid_coord(coord) and text:
            t = text[:30] + "..." if len(text) > 30 else text
            return f"Step {step_id}: type(coordinate=[{int(float(coord[0]))}, {int(float(coord[1]))}], keys='{t}')"
        elif text:
            t = text[:30] + "..." if len(text) > 30 else text
            return f"Step {step_id}: type(keys='{t}')"
        return f"Step {step_id}: type()"
    elif atype in ("swipe", "drag"):
        start = action.get("coordinate")
        end = action.get("endCoordinate")
        if _valid_coord(start) and _valid_coord(end):
            return (f"Step {step_id}: drag(start_coordinate=[{int(float(start[0]))}, {int(float(start[1]))}], "
                    f"end_coordinate=[{int(float(end[0]))}, {int(float(end[1]))}])")
        return f"Step {step_id}: drag()"
    else:
        return f"Step {step_id}: {atype}()"


def generate_episode_guidance(client, model_name, episode):
    """Generate guidance for all steps of an episode."""
    episode_id = episode["episode_id"]
    goal = episode["goal"]
    steps = episode["steps"]

    history = []
    guidances = []

    for i, step in enumerate(steps):
        screenshot = step["screenshot"]
        gt_action = step["action"]

        history_text = "\n".join(history) if history else "None"
        prompt_text = GUIDANCE_PROMPT.format(
            instruction=goal,
            history=history_text,
        )

        # Build multimodal message
        user_content = []
        try:
            img = Image.open(screenshot).convert("RGB")
            buf = BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })
        except Exception as e:
            print(f"  [ep {episode_id}] step {i} image error: {e}")
            guidances.append("")
            history.append(_format_gt_action_for_history(gt_action, i + 1))
            continue

        user_content.append({"type": "text", "text": prompt_text})

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": user_content}],
                max_tokens=256,
                temperature=0.0,
            )
            guidance = response.choices[0].message.content or ""
        except Exception as e:
            print(f"  [ep {episode_id}] step {i} API error: {e}")
            guidance = ""

        guidances.append(guidance)
        # Use GT history (teacher-forced)
        history.append(_format_gt_action_for_history(gt_action, i + 1))

    return {
        "episode_id": episode_id,
        "goal": goal,
        "num_steps": len(steps),
        "guidances": guidances,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--threads", type=int, default=64)
    args = parser.parse_args()

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
            executor.submit(generate_episode_guidance, client, args.model_name, ep): ep["episode_id"]
            for ep in episodes
        }

        pbar = tqdm(total=len(episodes), desc="Generating guidance")
        for future in as_completed(futures):
            result = future.result()
            results[result["episode_id"]] = result
            pbar.update(1)
        pbar.close()

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    total_guidances = sum(len(r["guidances"]) for r in results.values())
    non_empty = sum(1 for r in results.values() for g in r["guidances"] if g)
    print(f"Generated {total_guidances} guidances ({non_empty} non-empty) for {len(results)} episodes")
    print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    main()
