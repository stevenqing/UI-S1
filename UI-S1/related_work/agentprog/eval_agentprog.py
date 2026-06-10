"""AgentProg-inspired evaluation (V22c).

Standalone eval script for 3 AgentProg experiments:
  - agentprog_stp:    Semantic Task Program (workflow + program counter)
  - agentprog_belief: Belief State (observation + accumulated beliefs)
  - agentprog_full:   STP + Belief State combined

Adapted from AgentProg (MobiSys 2026). Uses the same eval loop as
v22_memory_agent/eval_memory_reasoning.py but with AgentProg-specific
prompt templates, state tracking, and post-step updates.
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
from v22_memory_agent.build_memory_index import extract_app_from_screenshot, normalize_action_type

# AgentProg modules
from related_work.agentprog.prompts import (
    AGENTPROG_TEMPLATES,
    AGENTPROG_STEP0_TEMPLATES,
    AGENTPROG_FULL_PROMPT,
)
from related_work.agentprog.helpers import (
    extract_belief_state,
    update_belief_state,
    advance_program_counter,
    format_workflow_with_pc,
    format_agentprog_working_memory,
)


# ═══════════════════════════════════════════════════════════════════════
# Experiment configurations
# ═══════════════════════════════════════════════════════════════════════

EXPERIMENT_CONFIGS = {
    "agentprog_stp":    {"working_memory": "agentprog_stp",    "temperature": 0.0},
    "agentprog_belief": {"working_memory": "agentprog_belief", "temperature": 0.0},
    "agentprog_full":   {"working_memory": "agentprog_full",   "temperature": 0.0},
}


# ═══════════════════════════════════════════════════════════════════════
# Helpers (reused from v22)
# ═══════════════════════════════════════════════════════════════════════

def extract_reasoning(pred_text: str, max_len: int = 200) -> str:
    """Extract reasoning text before <tool_call> tag."""
    if not pred_text:
        return ""
    m = re.search(r'<tool_call>', pred_text)
    if m:
        reasoning = pred_text[:m.start()].strip()
    else:
        reasoning = pred_text.strip()
    if len(reasoning) > max_len:
        reasoning = reasoning[:max_len].rsplit(' ', 1)[0] + "..."
    return reasoning


def parse_subtask_list(reasoning: str) -> List[str]:
    """Parse numbered list (1. xxx  2. xxx) from reasoning text."""
    if not reasoning:
        return []
    pattern = r'^\s*(\d+)[.)]\s*(.+)$'
    subtasks = []
    for line in reasoning.split('\n'):
        m = re.match(pattern, line.strip())
        if m:
            subtasks.append(m.group(2).strip())
    return subtasks


# ═══════════════════════════════════════════════════════════════════════
# Prompt building
# ═══════════════════════════════════════════════════════════════════════

def build_prompt(
    wm_type: str,
    instruction: str,
    history: str,
    actions: str,
    working_memory: str,
    step_idx: int,
    workflow_text: str = "",
) -> str:
    """Build the prompt string for an AgentProg experiment."""
    # Step 0: use decomposition prompt if available
    if step_idx == 0 and wm_type in AGENTPROG_STEP0_TEMPLATES:
        return AGENTPROG_STEP0_TEMPLATES[wm_type].format(
            instruction=instruction,
            history=history,
            actions=actions,
        )

    # agentprog_full steps 1+: needs both workflow and belief
    if wm_type == "agentprog_full":
        return AGENTPROG_FULL_PROMPT.format(
            instruction=instruction,
            history=history,
            workflow=workflow_text,
            working_memory=working_memory,
            actions=actions,
        )

    # agentprog_stp / agentprog_belief steps 1+
    template = AGENTPROG_TEMPLATES.get(wm_type)
    if template:
        return template.format(
            instruction=instruction,
            history=history,
            working_memory=working_memory,
            actions=actions,
        )

    raise ValueError(f"Unknown wm_type: {wm_type}")


# ═══════════════════════════════════════════════════════════════════════
# Model call
# ═══════════════════════════════════════════════════════════════════════

def call_agent(
    client: OpenAI,
    model_name: str,
    b64_image: str,
    prompt_text: str,
    temperature: float = 0.0,
) -> Tuple[Optional[Dict], str]:
    """Call the model with a prompt and return (parsed_action, raw_text)."""
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
        {"type": "text", "text": prompt_text},
    ]}]

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

    pred_action = parse_tool_call(pred_text)
    if pred_action is None:
        m = re.search(r'<action>\s*(\{.*?\})\s*</action>', pred_text, re.DOTALL)
        if m:
            try:
                pred_action = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

    return pred_action, pred_text


# ═══════════════════════════════════════════════════════════════════════
# Episode evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate_episode(
    client: OpenAI,
    model_name: str,
    episode: Dict,
    wm_type: str,
    temperature: float = 0.0,
    gt_history: bool = False,
    match_threshold: float = 0.5,
    image_max_pixels: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate one episode with AgentProg-inspired prompting."""
    episode_id = episode["episode_id"]
    goal = episode["goal"]
    steps = episode["steps"]
    num_steps = len(steps)

    history = []
    action_type_sequence = []
    step_results = []
    first_error_step = None
    correct_steps = 0

    # AgentProg state
    workflow = []          # from step 0 decomposition
    program_counter = 0    # advances across steps
    belief_state = ""      # accumulated belief

    for i, step in enumerate(steps):
        gt_action = step["action"]
        screenshot = step["screenshot"]
        image_w = step.get("image_w", 1040)
        image_h = step.get("image_h", 736)

        history_text = "\n".join(history) if history else "None"
        b64 = _encode_screenshot(screenshot, image_max_pixels)

        # Build working memory
        wm_text = format_agentprog_working_memory(
            wm_type, workflow, program_counter, belief_state,
        )

        # Build workflow text for agentprog_full
        workflow_text = ""
        if wm_type == "agentprog_full" and workflow:
            workflow_text = format_workflow_with_pc(workflow, program_counter)

        # Build prompt
        prompt_text = build_prompt(
            wm_type=wm_type,
            instruction=goal,
            history=history_text,
            actions=SUPPORTED_ACTIONS,
            working_memory=wm_text,
            step_idx=i,
            workflow_text=workflow_text,
        )

        # Call model
        pred_action, pred_text = call_agent(
            client, model_name, b64, prompt_text, temperature,
        )

        # Post-step: extract reasoning and update AgentProg state
        reasoning = extract_reasoning(pred_text)

        # STP: parse workflow at step 0, advance PC at steps 1+
        if wm_type in ("agentprog_stp", "agentprog_full") and i == 0:
            workflow = parse_subtask_list(reasoning)
            program_counter = 0

        if wm_type in ("agentprog_stp", "agentprog_full") and i > 0:
            program_counter = advance_program_counter(
                reasoning, program_counter, len(workflow),
            )

        # Belief: extract and accumulate
        if wm_type in ("agentprog_belief", "agentprog_full"):
            new_bs = extract_belief_state(pred_text)
            belief_state = update_belief_state(belief_state, new_bs)

        # Score against GT
        if pred_action:
            fake_text = f"<action>{json.dumps(pred_action)}</action>"
        else:
            fake_text = ""
        reward, info = compute_step_reward(fake_text, gt_action, image_w, image_h)
        success = reward >= match_threshold

        step_results.append({
            "step_idx": i,
            "success": success,
            "reward": reward,
            "pred_action": pred_action,
            "pred_type": info.get("pred_type"),
            "gt_type": info.get("gt_type"),
            "type_reward": info.get("type_reward", 0),
            "content_reward": info.get("content_reward", 0),
            "program_counter": program_counter if wm_type in ("agentprog_stp", "agentprog_full") else None,
            "belief_state_len": len(belief_state) if wm_type in ("agentprog_belief", "agentprog_full") else None,
        })

        # Update history (pred history mode)
        if gt_history:
            history.append(_format_action_for_history(gt_action, i + 1))
        else:
            history.append(_format_action_for_history(pred_action, i + 1))

        action_type_sequence.append(normalize_action_type(
            pred_action if pred_action else gt_action
        ))

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
        "workflow_len": len(workflow),
        "steps": step_results,
    }


# ═══════════════════════════════════════════════════════════════════════
# Breakdown analysis
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
    parser = argparse.ArgumentParser(description="AgentProg-Inspired Evaluation (V22c)")
    parser.add_argument("--experiment", required=True,
                        choices=list(EXPERIMENT_CONFIGS.keys()),
                        help="Experiment configuration to run")
    parser.add_argument("--test_data", required=True,
                        help="Path to test JSONL file")
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", default="related_work/agentprog/outputs")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--gt_history", action="store_true", default=False)
    parser.add_argument("--image_max_pixels", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    cfg = EXPERIMENT_CONFIGS[args.experiment]
    wm_type = cfg["working_memory"]
    temperature = cfg["temperature"]

    print(f"AgentProg Experiment: {args.experiment}")
    print(f"  Working memory: {wm_type}")
    print(f"  Temperature: {temperature}")
    print(f"  GT history: {args.gt_history}")

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

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(
                evaluate_episode, client, args.model_name, ep,
                wm_type, temperature,
                args.gt_history, args.match_threshold,
                args.image_max_pixels,
            ): ep["episode_id"]
            for ep in episodes
        }

        pbar = tqdm(total=len(episodes), desc=f"AgentProg {args.experiment}")
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
        "total_inference_calls": total_steps,
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
    print(f"AgentProg Results: {args.experiment}")
    print(f"{'='*65}")
    print(f"  TSR:              {summary['tsr']*100:.1f}%")
    print(f"  Step SR:          {summary['step_sr']*100:.1f}%")
    print(f"  Avg Progress:     {summary['avg_progress']*100:.1f}%")
    print(f"  Total inferences: {summary['total_inference_calls']}")
    hist_mode = "GT history" if args.gt_history else "Pred history"
    print(f"  History mode:     {hist_mode}")
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
