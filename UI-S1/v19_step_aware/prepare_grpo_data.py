#!/usr/bin/env python3
"""Generate N candidate actions per step on training data for GRPO training.

For each episode, for each step:
  1. Build prompt with GT history (clean starting state)
  2. Generate N candidates: 1 greedy + (N-1) sampled at temperature T
  3. Compute GT reward for each candidate
  4. Output: JSONL where each line = one step with N candidates + rewards

Output format (per line):
{
  "episode_id": 42,
  "step_idx": 0,
  "goal": "...",
  "screenshot": "/path/to/img.png",
  "history": "None",
  "image_w": 1040, "image_h": 736,
  "gt_action": {...},
  "candidates": [
    {"response": "<tool_call>...</tool_call>", "reward": 1.0, "type_reward": 1.0, "content_reward": 1.0, "pred_action": {...}},
    ...
  ]
}

Usage:
    python v19_step_aware/prepare_grpo_data.py \
        --train_data v12_gui_360/data/gui360_train_2000_balanced.jsonl \
        --api_url http://localhost:8000/v1 --model_name base_sft \
        --output v19_step_aware/data/grpo_candidates_n5.jsonl \
        --n_candidates 5 --sample_temperature 0.7 --threads 48
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v13_gui_360.eval_gui360_template import (
    parse_tool_call, _format_action_for_history, SUPPORTED_ACTIONS,
    _encode_screenshot,
)
from v13_gui_360.reward import compute_step_reward


PREDICT_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

Before acting, determine:
1. What type of interaction is needed next? (click to select/navigate, type to enter text, drag to move, scroll to view more)
2. Which UI element should receive this interaction?
3. Output the action with precise coordinates.

After your reasoning, output your action within <tool_call></tool_call> tag:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time."""


def _parse_prediction(pred_text: str) -> Optional[Dict]:
    pred_action = parse_tool_call(pred_text)
    if pred_action is None:
        m = re.search(r'<action>\s*(\{.*?\})\s*</action>', pred_text, re.DOTALL)
        if m:
            try:
                pred_action = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    return pred_action


def generate_candidates_for_episode(
    client: OpenAI,
    model_name: str,
    episode: Dict,
    n_candidates: int = 5,
    sample_temperature: float = 0.7,
    match_threshold: float = 0.5,
    image_max_pixels: Optional[int] = None,
) -> List[Dict]:
    """Generate N candidates per step for one episode (using GT history)."""
    episode_id = episode["episode_id"]
    goal = episode["goal"]
    steps = episode["steps"]
    results = []

    history = []

    for i, step in enumerate(steps):
        gt_action = step["action"]
        screenshot = step["screenshot"]
        image_w = step.get("image_w", 1040)
        image_h = step.get("image_h", 736)

        history_text = "\n".join(history) if history else "None"

        # Build prompt
        prompt_text = PREDICT_PROMPT.format(
            instruction=goal, history=history_text, actions=SUPPORTED_ACTIONS)

        b64 = _encode_screenshot(screenshot, image_max_pixels)
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt_text},
        ]}]

        # Generate N candidates
        candidates = []

        for k in range(n_candidates):
            temp = 0.0 if k == 0 else sample_temperature
            try:
                response = client.chat.completions.create(
                    model=model_name, messages=messages,
                    max_tokens=1024, temperature=temp)
                pred_text = response.choices[0].message.content or ""
            except Exception as e:
                pred_text = f"ERROR: {e}"

            pred_action = _parse_prediction(pred_text)

            # Compute reward
            if pred_action:
                fake_text = f"<action>{json.dumps(pred_action)}</action>"
            else:
                fake_text = pred_text
            reward, info = compute_step_reward(fake_text, gt_action, image_w, image_h)

            candidates.append({
                "response": pred_text,
                "reward": reward,
                "type_reward": info.get("type_reward", 0),
                "content_reward": info.get("content_reward", 0),
                "is_correct": reward >= match_threshold,
                "pred_type": info.get("pred_type"),
                "pred_action": info.get("pred_action"),
                "temperature": temp,
            })

        results.append({
            "episode_id": episode_id,
            "step_idx": i,
            "goal": goal,
            "screenshot": screenshot,
            "history": history_text,
            "image_w": image_w,
            "image_h": image_h,
            "gt_action": gt_action,
            "candidates": candidates,
        })

        # Always use GT history for clean training signal
        history.append(_format_action_for_history(gt_action, i + 1))

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate GRPO training data")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output", required=True,
                        help="Output JSONL path")
    parser.add_argument("--n_candidates", type=int, default=5)
    parser.add_argument("--sample_temperature", type=float, default=0.7)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--image_max_pixels", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    episodes = []
    with open(args.train_data) as f:
        for line in f:
            episodes.append(json.loads(line))
    episodes = episodes[args.start:args.end]
    print(f"Loaded {len(episodes)} episodes")
    print(f"API: {args.api_url} / {args.model_name}")
    print(f"N={args.n_candidates}, temp={args.sample_temperature}")

    client = OpenAI(base_url=args.api_url, api_key="dummy")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    all_steps = []
    total_correct = 0
    total_candidates = 0
    groups_with_both = 0  # groups with at least 1 correct AND 1 wrong

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(
                generate_candidates_for_episode,
                client, args.model_name, ep,
                args.n_candidates, args.sample_temperature,
                args.match_threshold, args.image_max_pixels,
            ): ep["episode_id"]
            for ep in episodes
        }
        pbar = tqdm(total=len(episodes), desc="GenCandidates")
        for future in as_completed(futures):
            step_results = future.result()
            for sr in step_results:
                all_steps.append(sr)
                n_correct = sum(1 for c in sr["candidates"] if c["is_correct"])
                n_wrong = len(sr["candidates"]) - n_correct
                total_correct += n_correct
                total_candidates += len(sr["candidates"])
                if n_correct > 0 and n_wrong > 0:
                    groups_with_both += 1
            pbar.update(1)
        pbar.close()

    # Write output
    with open(args.output, "w") as f:
        for sr in all_steps:
            f.write(json.dumps(sr, ensure_ascii=False) + "\n")

    # Statistics
    n_steps = len(all_steps)
    print(f"\n{'='*60}")
    print(f"GRPO Data Generation Summary")
    print(f"{'='*60}")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Total steps: {n_steps}")
    print(f"  Total candidates: {total_candidates}")
    print(f"  Correct candidates: {total_correct} ({total_correct/total_candidates*100:.1f}%)")
    print(f"  Groups with both correct+wrong: {groups_with_both} ({groups_with_both/n_steps*100:.1f}%)")
    print(f"  Groups all correct: {n_steps - groups_with_both - sum(1 for s in all_steps if all(not c['is_correct'] for c in s['candidates']))}")
    all_wrong = sum(1 for s in all_steps if all(not c["is_correct"] for c in s["candidates"]))
    print(f"  Groups all wrong: {all_wrong} ({all_wrong/n_steps*100:.1f}%)")
    print(f"  Output: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
