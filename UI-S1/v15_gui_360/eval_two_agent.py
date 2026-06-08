#!/usr/bin/env python3
"""Two-agent evaluation: Reasoning Agent + Action Agent.

Agent 1 (Reasoner): CoT SFT model — analyzes the screen, identifies target element
Agent 2 (Actor):    Base SFT model — takes reasoning as guidance, predicts precise action

The reasoner's CoT output (CURRENT STATE, SUB-GOAL, LOCATE) is passed as
"advisor guidance" to the actor via USER_PROMPT_TEMPLATE_GUIDED.

Usage:
    python v15_gui_360/eval_two_agent.py \
        --test_data v12_gui_360/data/gui360_test_1000_balanced.jsonl \
        --reasoner_url http://localhost:8000/v1 \
        --reasoner_model cot_sft \
        --actor_url http://localhost:8001/v1 \
        --actor_model base_sft \
        --output_dir v15_gui_360/outputs/eval_two_agent
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v13_gui_360.reward import compute_step_reward
from v13_gui_360.eval_gui360_template import (
    USER_PROMPT_TEMPLATE,
    USER_PROMPT_TEMPLATE_GUIDED,
    SUPPORTED_ACTIONS,
    parse_tool_call,
    _format_action_for_history,
    _encode_screenshot,
)


# Reasoner prompt: same as training data format — outputs CoT reasoning
REASONER_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

First, explain your reasoning process—describe how you analyze the screenshot, understand the current state, and determine what action should be taken next based on the instruction and previous actions.

Then output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished, you can output status as "FINISH" and take no action. Like:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time. If the instruction could apply to multiple elements, choose the most relevant one based on the context provided by the screenshot and previous actions.
"""


def extract_reasoning(cot_text: str, strip_coords: bool = False) -> str:
    """Extract reasoning portion from CoT output (everything before <tool_call>).

    If strip_coords=True, remove all [x, y] coordinate patterns from the
    reasoning. This forces the actor to do its own visual grounding instead
    of blindly copying the reasoner's coordinates.
    """
    # Find <tool_call> and take everything before it
    idx = cot_text.find("<tool_call>")
    if idx > 0:
        reasoning = cot_text[:idx].strip()
    else:
        reasoning = cot_text.strip()

    if strip_coords:
        # Remove coordinate patterns like [204, 682], [449, 70], etc.
        # Keep semantic info: element name, type, region description
        reasoning = re.sub(
            r',?\s*at approximately \[\d+,\s*\d+\]', '', reasoning
        )
        # Also remove standalone coordinate patterns
        reasoning = re.sub(r'\[\d+,\s*\d+\]', '', reasoning)

    return reasoning


def build_reasoner_prompt(goal, screenshot_path, history, image_max_pixels=None):
    """Build prompt for the reasoning agent."""
    history_text = "\n".join(history) if history else "None"
    prompt_text = REASONER_PROMPT.format(
        instruction=goal,
        history=history_text,
        actions=SUPPORTED_ACTIONS,
    )
    b64 = _encode_screenshot(screenshot_path, image_max_pixels)
    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": prompt_text},
    ]
    return [{"role": "user", "content": user_content}]


def build_actor_prompt(goal, screenshot_path, history, guidance, image_max_pixels=None):
    """Build prompt for the action agent with reasoning guidance."""
    history_text = "\n".join(history) if history else "None"
    prompt_text = USER_PROMPT_TEMPLATE_GUIDED.format(
        instruction=goal,
        history=history_text,
        guidance=guidance,
        actions=SUPPORTED_ACTIONS,
    )
    b64 = _encode_screenshot(screenshot_path, image_max_pixels)
    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": prompt_text},
    ]
    return [{"role": "user", "content": user_content}]


def evaluate_episode(
    reasoner_client: OpenAI,
    reasoner_model: str,
    actor_client: OpenAI,
    actor_model: str,
    episode: Dict,
    match_threshold: float = 0.5,
    gt_history: bool = False,
    image_max_pixels: Optional[int] = None,
    strip_coords: bool = False,
) -> Dict[str, Any]:
    """Evaluate a single episode with two-agent pipeline."""
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

        # --- Agent 1: Reasoner ---
        try:
            reasoner_msgs = build_reasoner_prompt(
                goal, screenshot, history, image_max_pixels
            )
            reasoner_resp = reasoner_client.chat.completions.create(
                model=reasoner_model,
                messages=reasoner_msgs,
                max_tokens=1024,
                temperature=0.0,
            )
            cot_text = reasoner_resp.choices[0].message.content or ""
        except Exception as e:
            cot_text = ""
            print(f"  [ep {episode_id}] step {i+1} reasoner error: {e}")

        reasoning = extract_reasoning(cot_text, strip_coords=strip_coords)

        # --- Agent 2: Actor (with reasoning as guidance) ---
        try:
            actor_msgs = build_actor_prompt(
                goal, screenshot, history, reasoning, image_max_pixels
            )
            actor_resp = actor_client.chat.completions.create(
                model=actor_model,
                messages=actor_msgs,
                max_tokens=512,
                temperature=0.0,
            )
            pred_text = actor_resp.choices[0].message.content or ""
        except Exception as e:
            pred_text = ""
            print(f"  [ep {episode_id}] step {i+1} actor error: {e}")

        # Parse action from actor output
        pred_action = parse_tool_call(pred_text)
        if pred_action is None:
            m = re.search(r'<action>\s*(\{.*?\})\s*</action>', pred_text, re.DOTALL)
            if m:
                try:
                    pred_action = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass

        if pred_action:
            fake_text = f"<action>{json.dumps(pred_action)}</action>"
        else:
            fake_text = pred_text

        reward, info = compute_step_reward(fake_text, gt_action, image_w, image_h)
        success = reward >= match_threshold

        step_results.append({
            "step_idx": i,
            "success": success,
            "reward": reward,
            "reasoning": reasoning[:500],
            "pred_text": pred_text[:300],
            "pred_action": info.get("pred_action"),
            "pred_type": info.get("pred_type"),
            "gt_type": info.get("gt_type"),
            "format_reward": info.get("format_reward", 0),
            "type_reward": info.get("type_reward", 0),
            "content_reward": info.get("content_reward", 0),
        })

        # Update history
        if gt_history:
            history.append(_format_action_for_history(gt_action, i + 1))
        else:
            history.append(_format_action_for_history(pred_action, i + 1))

        if success:
            correct_steps += 1
        else:
            if first_error_step is None:
                first_error_step = i + 1

    progress = (first_error_step - 1) / num_steps if first_error_step else 1.0
    task_success = first_error_step is None and len(step_results) == num_steps

    return {
        "episode_id": episode_id,
        "goal": goal,
        "num_steps": num_steps,
        "steps_evaluated": len(step_results),
        "correct_steps": correct_steps,
        "task_success": task_success,
        "progress": progress,
        "first_error_step": first_error_step,
        "steps": step_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--reasoner_url", required=True, help="vLLM URL for reasoning agent")
    parser.add_argument("--reasoner_model", required=True)
    parser.add_argument("--actor_url", required=True, help="vLLM URL for action agent")
    parser.add_argument("--actor_model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--gt_history", action="store_true", default=False)
    parser.add_argument("--strip_coords", action="store_true", default=False,
                        help="Strip coordinates from reasoning to force actor's own grounding")
    parser.add_argument("--image_max_pixels", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    # Load test data
    with open(args.test_data) as f:
        episodes = [json.loads(line) for line in f]
    if args.end:
        episodes = episodes[args.start:args.end]
    else:
        episodes = episodes[args.start:]
    print(f"Loaded {len(episodes)} episodes")

    os.makedirs(args.output_dir, exist_ok=True)

    # Create clients
    reasoner_client = OpenAI(base_url=args.reasoner_url, api_key="dummy")
    actor_client = OpenAI(base_url=args.actor_url, api_key="dummy")

    # Evaluate
    results = {}
    total_correct = 0
    total_steps = 0
    total_success = 0

    def eval_one(ep):
        return evaluate_episode(
            reasoner_client, args.reasoner_model,
            actor_client, args.actor_model,
            ep,
            match_threshold=args.match_threshold,
            gt_history=args.gt_history,
            image_max_pixels=args.image_max_pixels,
            strip_coords=args.strip_coords,
        )

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(eval_one, ep): ep for ep in episodes}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            try:
                result = future.result()
                eid = result["episode_id"]
                results[eid] = result
                total_correct += result["correct_steps"]
                total_steps += result["steps_evaluated"]
                if result["task_success"]:
                    total_success += 1
            except Exception as e:
                ep = futures[future]
                print(f"Error evaluating episode {ep.get('episode_id', '?')}: {e}")

    # Summary
    num_episodes = len(results)
    tsr = total_success / num_episodes if num_episodes > 0 else 0
    step_sr = total_correct / total_steps if total_steps > 0 else 0
    avg_progress = sum(r["progress"] for r in results.values()) / num_episodes if num_episodes > 0 else 0
    mean_reward = sum(
        s["reward"] for r in results.values() for s in r["steps"]
    ) / total_steps if total_steps > 0 else 0

    summary = {
        "num_episodes": num_episodes,
        "tsr": tsr,
        "avg_progress": avg_progress,
        "step_sr": step_sr,
        "mean_reward": mean_reward,
        "total_steps_evaluated": total_steps,
        "total_steps_correct": total_correct,
        "match_threshold": args.match_threshold,
        "gt_history": args.gt_history,
        "strip_coords": args.strip_coords,
        "reasoner_model": args.reasoner_model,
        "actor_model": args.actor_model,
        "prompt_format": "two_agent",
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.output_dir, f"eval_results_{ts}.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.output_dir, f"eval_summary_{ts}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Two-Agent Eval Results")
    print(f"{'='*50}")
    print(f"Reasoner: {args.reasoner_model}")
    print(f"Actor:    {args.actor_model}")
    print(f"Episodes: {num_episodes}")
    print(f"TSR:      {tsr:.3f}")
    print(f"Step SR:  {step_sr:.3f}")
    print(f"Progress: {avg_progress:.3f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
