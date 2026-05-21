#!/usr/bin/env python3
"""Evaluate cooperative model on gui360_test_968.jsonl.

Uses the same prompt format and action parsing as RL training.
Reads episodes from JSONL, sends to cooperative model server, compares with GT.

Usage:
    python v13_gui_360/eval_cooperative.py \
        --test_data v13_gui_360/data/gui360_test_968.jsonl \
        --api_url http://localhost:8000/v1 \
        --model_name v12_gui360_rl_epoch-0 \
        --output_dir v12_gui_360/outputs/epoch-0 \
        --threads 4
"""

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v13_gui_360.reward import parse_action_from_text, compute_step_reward, _normalize_action_type

# Same system prompt as training
SYSTEM_PROMPT = (
    "You are a GUI agent. You are given a screenshot and a task instruction. "
    "Perform the next action to complete the task.\n\n"
    "Action space:\n"
    '  click: {"action": "click", "coordinate": [x, y]}\n'
    '  type: {"action": "type", "text": "content"}\n'
    '  drag: {"action": "drag", "coordinate": [x1, y1], "endCoordinate": [x2, y2]}\n'
    '  terminate: {"action": "terminate", "status": "success|failure"}\n\n'
    "Output format: <action>{JSON action}</action>"
)


def build_step_prompt(goal: str, screenshot_path: str, step_idx: int, history: List[str]) -> List[dict]:
    """Build messages for a single step prediction."""
    history_text = ""
    if history:
        history_text = "\nPrevious actions:\n" + "\n".join(history) + "\n"

    user_content = []

    # Add screenshot as base64
    img = Image.open(screenshot_path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    user_content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64}"}
    })

    # Add text prompt
    text = f"Task: {goal}\n{history_text}\nStep {step_idx + 1}: What action should be taken?"
    user_content.append({"type": "text", "text": text})

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def evaluate_episode(
    client: OpenAI,
    model_name: str,
    episode: Dict,
    stop_on_error: bool = True,
    match_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate a single episode autoregressively."""
    episode_id = episode["episode_id"]
    goal = episode["goal"]
    steps = episode["steps"]
    num_steps = len(steps)

    history = []
    step_results = []
    first_error_step = None
    correct_steps = 0

    for i, step in enumerate(steps):
        gt_action = step["action"]
        screenshot = step["screenshot"]
        image_w = step.get("image_w", 1040)
        image_h = step.get("image_h", 736)

        # Build prompt
        messages = build_step_prompt(goal, screenshot, i, history)

        # Get prediction
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=512,
                temperature=0.0,
            )
            pred_text = response.choices[0].message.content or ""
        except Exception as e:
            pred_text = ""
            print(f"  [ep {episode_id}] step {i+1} API error: {e}")

        # Compute reward using training reward function
        reward, info = compute_step_reward(
            pred_text, gt_action, image_w, image_h
        )

        success = reward >= match_threshold

        step_results.append({
            "step_idx": i,
            "success": success,
            "reward": reward,
            "pred_text": pred_text[:200],
            "pred_action": info.get("pred_action"),
            "pred_type": info.get("pred_type"),
            "gt_type": info.get("gt_type"),
            "format_reward": info.get("format_reward", 0),
            "type_reward": info.get("type_reward", 0),
            "content_reward": info.get("content_reward", 0),
        })

        # Update history with model's output
        pred_action = parse_action_from_text(pred_text)
        if pred_action:
            history.append(f"Step {i+1}: {json.dumps(pred_action)}")
        else:
            history.append(f"Step {i+1}: (no valid action)")

        if success:
            correct_steps += 1
        else:
            if first_error_step is None:
                first_error_step = i + 1
            if stop_on_error:
                break

    # Compute metrics
    progress = (first_error_step - 1) / num_steps if first_error_step else 1.0
    tsr = 1.0 if correct_steps == num_steps else 0.0
    step_sr = correct_steps / max(len(step_results), 1)

    return {
        "episode_id": episode_id,
        "goal": goal,
        "num_steps": num_steps,
        "evaluated_steps": len(step_results),
        "correct_steps": correct_steps,
        "tsr": tsr,
        "progress": progress,
        "step_sr": step_sr,
        "first_error_step": first_error_step,
        "step_results": step_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", required=True, help="Path to gui360_test_968.jsonl")
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--model_name", default="cooperative")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--stop_on_error", action="store_true", default=True)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    args = parser.parse_args()

    # Load episodes
    episodes = []
    with open(args.test_data) as f:
        for line in f:
            episodes.append(json.loads(line.strip()))
    print(f"Loaded {len(episodes)} episodes")

    os.makedirs(args.output_dir, exist_ok=True)

    client = OpenAI(api_key="0", base_url=args.api_url, timeout=600)

    # Evaluate
    results = []
    total_tsr = 0
    total_progress = 0
    total_step_correct = 0
    total_step_eval = 0

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(
                evaluate_episode, client, args.model_name, ep,
                args.stop_on_error, args.match_threshold
            ): ep["episode_id"]
            for ep in episodes
        }

        pbar = tqdm(total=len(episodes), desc="Evaluating")
        for future in as_completed(futures):
            ep_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                total_tsr += result["tsr"]
                total_progress += result["progress"]
                total_step_correct += result["correct_steps"]
                total_step_eval += result["evaluated_steps"]

                n = len(results)
                pbar.set_postfix({
                    "TSR": f"{total_tsr/n:.3f}",
                    "Progress": f"{total_progress/n:.3f}",
                    "StepSR": f"{total_step_correct/max(total_step_eval,1):.3f}",
                })
            except Exception as e:
                print(f"Episode {ep_id} failed: {e}")
            pbar.update(1)
        pbar.close()

    # Summary
    n = len(results)
    summary = {
        "num_episodes": n,
        "tsr": total_tsr / n if n else 0,
        "avg_progress": total_progress / n if n else 0,
        "step_sr": total_step_correct / max(total_step_eval, 1),
        "total_steps_evaluated": total_step_eval,
        "total_steps_correct": total_step_correct,
        "match_threshold": args.match_threshold,
        "stop_on_error": args.stop_on_error,
    }

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Episodes:     {n}")
    print(f"TSR:          {summary['tsr']:.4f} ({int(total_tsr)}/{n})")
    print(f"Avg Progress: {summary['avg_progress']:.4f}")
    print(f"Step SR:      {summary['step_sr']:.4f} ({total_step_correct}/{total_step_eval})")
    print("=" * 60)

    # Save results
    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.output_dir, f"eval_results_{ts}.json"), "w") as f:
        json.dump({"summary": summary, "episodes": results}, f, indent=2)
    with open(os.path.join(args.output_dir, f"eval_summary_{ts}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
