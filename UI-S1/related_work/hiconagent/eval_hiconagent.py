"""HiconAgent evaluation on GUI-360 balanced benchmark.

Standalone eval script for 2 HiconAgent experiments:
  - hiconagent_method:           Multi-image visual history (τ=2) + <think>/<action>
  - hiconagent_method_no_visual: Text-only history (ablation, same format but no images)

Reference: HiconAgent (CVPR 2026) — "History Context-aware Policy Optimization
for Long-horizon GUI Agents"

Key differences from HAR:
  - Visual history: previous screenshots included as multi-image input
  - No Act2Sum: no second model call per step (but more tokens from images)
  - Output format: <think>/<action> instead of <think>/<answer>
  - History window: τ=2 (2 previous screenshots) instead of k=4 text summaries
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openai import OpenAI
from tqdm import tqdm

# Reuse from v13
from v13_gui_360.eval_gui360_template import (
    parse_tool_call,
    _format_action_for_history,
    _encode_screenshot,
    SUPPORTED_ACTIONS,
)
from v13_gui_360.reward import compute_step_reward

# HiconAgent modules
from related_work.hiconagent.prompts_hiconagent import (
    GUI360_HICONAGENT_ACTION_SPACE,
    format_history_section_visual,
    format_history_section_text,
    format_hiconagent_prompt,
)


# ═══════════════════════════════════════════════════════════════════════
# Experiment configurations
# ═══════════════════════════════════════════════════════════════════════

EXPERIMENT_CONFIGS = {
    "hiconagent_method": {
        # Multi-image visual history (τ=2) + <think>/<action> with <tool_call>
        "use_visual_history": True,
        "tau_history": 2,
    },
    "hiconagent_method_no_visual": {
        # Text-only history (ablation: same prompt format but no screenshot history)
        "use_visual_history": False,
        "tau_history": 2,
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Output parsing
# ═══════════════════════════════════════════════════════════════════════

def parse_hiconagent_output(text: str) -> Tuple[str, str, Optional[Dict]]:
    """Parse HiconAgent output with <think>/<action> tags.

    Returns (think_text, action_text, parsed_action_dict).
    """
    # Extract <think> content
    think_m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    think_text = think_m.group(1).strip() if think_m else ""

    # Extract <action> content
    action_m = re.search(r'<action>(.*?)</action>', text, re.DOTALL)
    action_text = action_m.group(1).strip() if action_m else ""

    # Try to parse <tool_call> from <action> block
    parsed = None
    search_text = action_text if action_text else text
    tc_m = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', search_text, re.DOTALL)
    if tc_m:
        try:
            tc = json.loads(tc_m.group(1))
            status = tc.get("status", "CONTINUE").upper()
            if status != "FINISH":
                func = tc.get("function", "")
                args = tc.get("args", {})
                if func:
                    parsed = {"function": func, "args": args, "status": status}
        except json.JSONDecodeError:
            pass

    # Fallback: try parse_tool_call (handles raw JSON in text)
    if parsed is None:
        parsed = parse_tool_call(text)

    return think_text, action_text, parsed


# ═══════════════════════════════════════════════════════════════════════
# Multi-image model call
# ═══════════════════════════════════════════════════════════════════════

def call_agent_multi_image(
    client: OpenAI,
    model_name: str,
    history_images: List[str],
    history_actions: List[str],
    current_image: str,
    prompt_text: str,
    temperature: float = 0.0,
) -> str:
    """Call model with multi-image input (visual history + current).

    Args:
        history_images: base64 encoded previous screenshots (τ most recent)
        history_actions: text descriptions of previous actions
        current_image: base64 encoded current screenshot
        prompt_text: the formatted prompt text
        temperature: sampling temperature
    Returns:
        Raw text response from the model
    """
    content = []

    # History images interleaved with action descriptions
    for img_b64, action_desc in zip(history_images, history_actions):
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
        })
        content.append({
            "type": "text",
            "text": f"Previous action: {action_desc}",
        })

    # Current screenshot
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{current_image}"},
    })

    # Prompt text
    content.append({"type": "text", "text": prompt_text})

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]

    pred_text = ""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1024,
            temperature=temperature,
        )
        pred_text = response.choices[0].message.content or ""
    except Exception as e:
        print(f"  API error: {e}")

    return pred_text


def call_agent_single_image(
    client: OpenAI,
    model_name: str,
    b64_image: str,
    prompt_text: str,
    temperature: float = 0.0,
) -> str:
    """Call model with single image (no visual history)."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
            {"type": "text", "text": prompt_text},
        ]},
    ]

    pred_text = ""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1024,
            temperature=temperature,
        )
        pred_text = response.choices[0].message.content or ""
    except Exception as e:
        print(f"  API error: {e}")

    return pred_text


# ═══════════════════════════════════════════════════════════════════════
# Episode evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate_episode(
    client: OpenAI,
    model_name: str,
    episode: Dict,
    use_visual_history: bool = True,
    tau_history: int = 2,
    gt_history: bool = False,
    match_threshold: float = 0.5,
    image_max_pixels: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate one episode with HiconAgent-style inference."""
    episode_id = episode["episode_id"]
    goal = episode["goal"]
    steps = episode["steps"]
    num_steps = len(steps)

    action_space = GUI360_HICONAGENT_ACTION_SPACE

    action_history_text = []  # text descriptions of past actions
    step_results = []
    first_error_step = None
    correct_steps = 0
    total_inferences = 0

    for i, step in enumerate(steps):
        gt_action = step["action"]
        screenshot = step["screenshot"]
        image_w = step.get("image_w", 1040)
        image_h = step.get("image_h", 736)

        b64 = _encode_screenshot(screenshot, image_max_pixels)

        # Build visual history (τ most recent screenshots)
        if use_visual_history and i > 0:
            history_start = max(0, i - tau_history)
            history_images = []
            history_actions_subset = []
            for j in range(history_start, i):
                prev_b64 = _encode_screenshot(steps[j]["screenshot"], image_max_pixels)
                history_images.append(prev_b64)
                history_actions_subset.append(action_history_text[j])
            history_section = format_history_section_visual(history_actions_subset)
        else:
            history_images = []
            history_actions_subset = []
            if not use_visual_history and action_history_text:
                # Text-only history: use τ most recent action descriptions
                recent = action_history_text[-tau_history:]
                history_section = format_history_section_text(recent)
            else:
                history_section = "This is the task's initial state. The current screenshot is shown below."

        prompt = format_hiconagent_prompt(goal, history_section, action_space)

        # Call model
        if history_images:
            pred_text = call_agent_multi_image(
                client, model_name, history_images, history_actions_subset,
                b64, prompt, 0.0,
            )
        else:
            pred_text = call_agent_single_image(
                client, model_name, b64, prompt, 0.0,
            )
        total_inferences += 1

        # Parse output
        think, action_text, parsed = parse_hiconagent_output(pred_text)

        # Check for FINISH signal
        is_finish = False
        if parsed is None:
            finish_m = re.search(r'"status"\s*:\s*"FINISH"', pred_text, re.IGNORECASE)
            if finish_m:
                is_finish = True

        # Update action history for next step
        if gt_history:
            action_history_text.append(_format_action_for_history(gt_action, i + 1))
        else:
            action_history_text.append(_format_action_for_history(parsed or gt_action, i + 1))

        # Score against GT
        if parsed:
            fake_text = f"<action>{json.dumps(parsed)}</action>"
        else:
            fake_text = ""
        reward, info = compute_step_reward(fake_text, gt_action, image_w, image_h)
        success = reward >= match_threshold

        step_results.append({
            "step_idx": i,
            "success": success,
            "reward": reward,
            "pred_text": pred_text[:300],
            "pred_action": parsed,
            "pred_type": info.get("pred_type"),
            "gt_type": info.get("gt_type"),
            "format_reward": info.get("format_reward", 0),
            "type_reward": info.get("type_reward", 0),
            "content_reward": info.get("content_reward", 0),
            "think": think[:200] if think else "",
            "action_text": action_text[:200] if action_text else "",
            "is_finish": is_finish,
            "num_history_images": len(history_images),
        })

        if success:
            correct_steps += 1
        elif first_error_step is None:
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
        "total_inferences": total_inferences,
        "steps": step_results,
    }


# ═══════════════════════════════════════════════════════════════════════
# Breakdown analysis (same as HAR / agentprog)
# ═══════════════════════════════════════════════════════════════════════

def compute_breakdown(results: Dict[str, Dict]) -> Dict[str, Any]:
    """Compute per-step-position and per-task-length metrics."""
    step_pos_correct = defaultdict(int)
    step_pos_total = defaultdict(int)
    step_pos_type_correct = defaultdict(int)

    length_buckets = {"1": (1, 1), "2-3": (2, 3), "4-5": (4, 5), "6+": (6, 999)}
    length_success = defaultdict(int)
    length_total = defaultdict(int)
    length_progress = defaultdict(float)
    length_steps_correct = defaultdict(int)
    length_steps_total = defaultdict(int)

    for eid, result in results.items():
        num_steps = result["num_steps"]
        bucket = "6+"
        for bname, (lo, hi) in length_buckets.items():
            if lo <= num_steps <= hi:
                bucket = bname
                break

        length_total[bucket] += 1
        if result["task_success"]:
            length_success[bucket] += 1
        length_progress[bucket] += result["progress"]

        for step in result["steps"]:
            idx = step["step_idx"]
            step_pos_total[idx] += 1
            if step["success"]:
                step_pos_correct[idx] += 1
            if step.get("type_reward", 0) >= 1.0:
                step_pos_type_correct[idx] += 1
            length_steps_total[bucket] += 1
            if step["success"]:
                length_steps_correct[bucket] += 1

    max_step = max(step_pos_total.keys()) if step_pos_total else 0
    step_position_acc = {}
    for idx in range(min(max_step + 1, 15)):
        total = step_pos_total.get(idx, 0)
        correct = step_pos_correct.get(idx, 0)
        type_correct = step_pos_type_correct.get(idx, 0)
        if total > 0:
            step_position_acc[f"step_{idx}"] = {
                "accuracy": correct / total,
                "type_accuracy": type_correct / total,
                "total": total,
                "correct": correct,
            }

    task_length_metrics = {}
    for bname in length_buckets:
        total = length_total.get(bname, 0)
        if total > 0:
            s_total = length_steps_total.get(bname, 0)
            task_length_metrics[bname] = {
                "tsr": length_success[bname] / total,
                "avg_progress": length_progress[bname] / total,
                "step_sr": length_steps_correct[bname] / s_total if s_total > 0 else 0,
                "num_episodes": total,
                "num_success": length_success[bname],
            }

    return {
        "step_position_accuracy": step_position_acc,
        "task_length_metrics": task_length_metrics,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="HiconAgent Evaluation on GUI-360 Balanced Benchmark")
    parser.add_argument("--experiment", required=True,
                        choices=list(EXPERIMENT_CONFIGS.keys()),
                        help="Experiment configuration to run")
    parser.add_argument("--test_data", required=True,
                        help="Path to test JSONL file")
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", default="related_work/hiconagent/outputs")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--gt_history", action="store_true", default=False)
    parser.add_argument("--image_max_pixels", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    cfg = EXPERIMENT_CONFIGS[args.experiment]
    use_visual_history = cfg["use_visual_history"]
    tau_history = cfg["tau_history"]

    print(f"HiconAgent Experiment: {args.experiment}")
    print(f"  Visual history: {use_visual_history}")
    print(f"  τ (history):    {tau_history}")
    print(f"  GT history:     {args.gt_history}")
    print(f"  Model:          {args.model_name}")

    # Load test data
    episodes = []
    with open(args.test_data) as f:
        for line in f:
            episodes.append(json.loads(line))
    total_loaded = len(episodes)
    episodes = episodes[args.start:args.end]
    print(f"Loaded {len(episodes)} episodes (shard [{args.start}:{args.end}] of {total_loaded})")

    # Setup output
    exp_output_dir = os.path.join(args.output_dir, args.experiment)
    os.makedirs(exp_output_dir, exist_ok=True)

    client = OpenAI(base_url=args.api_url, api_key="dummy")

    results = {}
    total_success = 0
    total_progress = 0.0
    total_steps = 0
    total_correct = 0
    total_inferences = 0

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(
                evaluate_episode, client, args.model_name, ep,
                use_visual_history, tau_history,
                args.gt_history, args.match_threshold,
                args.image_max_pixels,
            ): ep["episode_id"]
            for ep in episodes
        }

        pbar = tqdm(total=len(episodes), desc=f"HiconAgent {args.experiment}")
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                eid = futures[future]
                print(f"\nError in episode {eid}: {e}")
                continue

            eid = result["episode_id"]
            results[str(eid)] = result

            if result["task_success"]:
                total_success += 1
            total_progress += result["progress"]
            total_steps += result["steps_evaluated"]
            total_correct += result["correct_steps"]
            total_inferences += result["total_inferences"]

            n = len(results)
            pbar.update(1)
            pbar.set_postfix({
                "TSR": f"{total_success/n:.3f}",
                "StepSR": f"{total_correct/total_steps:.3f}" if total_steps > 0 else "0",
            })
        pbar.close()

    # Compute breakdowns
    breakdown = compute_breakdown(results)

    n = len(results)
    summary = {
        "experiment": args.experiment,
        "num_episodes": n,
        "tsr": total_success / n if n > 0 else 0,
        "avg_progress": total_progress / n if n > 0 else 0,
        "step_sr": total_correct / total_steps if total_steps > 0 else 0,
        "total_steps_evaluated": total_steps,
        "total_steps_correct": total_correct,
        "total_inference_calls": total_inferences,
        "inferences_per_step": total_inferences / total_steps if total_steps > 0 else 0,
        "use_visual_history": use_visual_history,
        "tau_history": tau_history,
        "gt_history": args.gt_history,
        "match_threshold": args.match_threshold,
        **breakdown,
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(exp_output_dir, f"summary_{ts}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(exp_output_dir, f"results_{ts}.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Print results
    print(f"\n{'='*65}")
    print(f"HiconAgent Results: {args.experiment}")
    print(f"{'='*65}")
    print(f"  TSR:              {summary['tsr']*100:.1f}%")
    print(f"  Step SR:          {summary['step_sr']*100:.1f}%")
    print(f"  Avg Progress:     {summary['avg_progress']*100:.1f}%")
    print(f"  Total inferences: {summary['total_inference_calls']}")
    print(f"  Infer/step:       {summary['inferences_per_step']:.1f}x")
    hist_mode = "GT history" if args.gt_history else "Pred history"
    vis_mode = "visual" if use_visual_history else "text-only"
    print(f"  History mode:     {hist_mode} ({vis_mode}, τ={tau_history})")
    print(f"  (Baselines: standard=21.7%, working_subtask=22.4%)")

    print(f"\n  --- By Task Length ---")
    for bname, metrics in sorted(summary.get("task_length_metrics", {}).items()):
        print(f"  {bname:>5s} steps: TSR={metrics['tsr']*100:5.1f}%  "
              f"StepSR={metrics['step_sr']*100:5.1f}%  "
              f"Progress={metrics['avg_progress']*100:5.1f}%  "
              f"(n={metrics['num_episodes']})")

    print(f"\n  --- By Step Position ---")
    for sname, metrics in sorted(summary.get("step_position_accuracy", {}).items(),
                                  key=lambda x: int(x[0].split("_")[1])):
        print(f"  {sname:>8s}: acc={metrics['accuracy']*100:5.1f}%  "
              f"type_acc={metrics['type_accuracy']*100:5.1f}%  "
              f"(n={metrics['total']})")

    print(f"\n  Output: {exp_output_dir}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
