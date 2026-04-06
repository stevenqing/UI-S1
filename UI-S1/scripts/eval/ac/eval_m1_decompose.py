"""Eval M1: Task Decomposition Multi-Agent Evaluation.

Architecture:
1. Decomposer agent breaks goal into atomic sub-steps (once, at step 0)
2. Executor agent follows sub-goals one by one with injected context
"""

import argparse
import copy
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from ac_utils import (
    load_ac_trajectories, fix_line, init_format, save_jsonl, save_json,
    compute_trajectory_metrics, length_bucket,
    evaluate_android_control_action, call_mobile_agent_vllm,
    find_last_image_ele, slim_messages, safe_parse_response, _json_default,
)

result_lock = Lock()
fm = None

DECOMPOSE_PROMPT = """You are a mobile task planner. Given a high-level goal and the current screenshot, decompose the goal into a sequence of atomic sub-steps.
Each sub-step should correspond to exactly one UI action (tap, type, swipe, open app, press button, etc.).

Rules:
- If the goal requires opening a specific app, the first sub-step should be opening that app.
- Each sub-step must be concrete and actionable (not vague like "navigate to settings").
- Output a numbered list of sub-steps, one per line.
- Do NOT include any other text besides the numbered list.

Goal: {goal}

Decompose into atomic sub-steps:"""


def call_decomposer(screenshot_path, goal, model_name):
    """Call decomposer to break goal into sub-steps."""
    from x.qwen.image import make_qwen_image_item

    messages = [
        {'role': 'user', 'content': [
            {'text': DECOMPOSE_PROMPT.format(goal=goal)},
            make_qwen_image_item(screenshot_path),
        ]}
    ]

    response = call_mobile_agent_vllm(messages=messages, model_name=model_name)
    return response.strip()


def parse_substeps(response):
    """Parse numbered list of sub-steps from decomposer response."""
    lines = response.strip().split('\n')
    substeps = []
    for line in lines:
        line = line.strip()
        match = re.match(r'^\d+[\.\)]\s*(.+)$', line)
        if match:
            substeps.append(match.group(1).strip())
    return substeps


def process_episode(episode, args):
    """Process episode with task decomposition."""
    global fm

    fixed = fix_line(copy.deepcopy(episode))
    num_steps = len(fixed['steps'])
    state = None
    model_response = None
    step_results = []
    decompose_response = None
    substeps = []

    try:
        # Step 0: Decompose the goal
        first_screenshot = fixed['steps'][0]['screenshot']
        decompose_response = call_decomposer(first_screenshot, episode['goal'], args.model_name)
        substeps = parse_substeps(decompose_response)

        for step_id in range(num_steps):
            current_check = fixed['steps'][step_id]['check_options']
            gt_action = fixed['steps'][step_id]['action_content']

            state = fm.gen_next_round(fixed, state, previous_model_response=model_response)
            if state is None:
                break

            messages = state['messages']

            # Inject current sub-goal into the latest user message
            if step_id < len(substeps):
                subgoal_context = f"\n\n## Current Sub-goal (step {step_id + 1}/{len(substeps)})\n{substeps[step_id]}\nFocus on completing this specific sub-step.\n"
            else:
                subgoal_context = f"\n\n## Sub-goal\nContinue toward the overall goal: {episode['goal']}\n"

            last_user_msg = messages[-1]
            last_user_msg['content'].insert(0, {'text': subgoal_context})

            messages = slim_messages(messages=messages, num_image_limit=args.n_history_image_limit)
            _, width, height, resized_width, resized_height = find_last_image_ele(messages)

            model_response = call_mobile_agent_vllm(messages=messages, model_name=args.model_name)

            pred = safe_parse_response(fm, model_response)
            pred_action = pred['action_content']

            type_match, extract_match = evaluate_android_control_action(
                pred_action, current_check,
                width, height, resized_width, resized_height
            )

            step_results.append({
                'step_num': step_id,
                'type_match': type_match,
                'extract_match': extract_match,
                'pred_action': pred_action,
                'gt_action': gt_action,
                'gt_action_type': gt_action['action'],
                'injected_subgoal': substeps[step_id] if step_id < len(substeps) else None,
            })

            if not extract_match:
                break

    except Exception as e:
        print(f"Error episode {episode.get('episode_id', '?')}: {e}")

    correct_steps = sum(1 for s in step_results if s['extract_match'])
    task_success = (correct_steps == num_steps)

    result = {
        'episode_id': episode.get('episode_id'),
        'goal': episode['goal'],
        'num_steps': num_steps,
        'task_success': task_success,
        'final_step_id': correct_steps,
        'step_results': step_results,
        'length_bucket': length_bucket(num_steps),
        'decompose_response': decompose_response,
        'num_substeps': len(substeps),
    }

    with result_lock:
        out_path = os.path.join(args.output_dir, 'decompose_results.jsonl')
        with open(out_path, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False, default=_json_default) + '\n')

    return result


def main(args):
    global fm
    fm = init_format()
    os.makedirs(args.output_dir, exist_ok=True)

    out_path = os.path.join(args.output_dir, 'decompose_results.jsonl')
    if os.path.exists(out_path):
        os.remove(out_path)

    data = load_ac_trajectories(jsonl_path=args.jsonl_file, max_episodes=args.max_episodes)
    print(f"Loaded {len(data)} episodes. Running M1 Task Decomposition evaluation...")

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_episode, ep, args): ep for ep in data}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                if len(results) % 50 == 0:
                    metrics = compute_trajectory_metrics(results)
                    print(f"Progress: {len(results)}/{len(data)} | TSR: {metrics['tsr']:.3f} | AvgProg: {metrics['avg_progress']:.3f}")
            except Exception as e:
                print(f"Exception: {e}")

    metrics = compute_trajectory_metrics(results)

    # Substep alignment stats
    alignment_stats = {
        'avg_substeps': sum(r['num_substeps'] for r in results) / len(results) if results else 0,
        'avg_gt_steps': sum(r['num_steps'] for r in results) / len(results) if results else 0,
        'exact_match_count': sum(1 for r in results if r['num_substeps'] == r['num_steps']),
    }

    summary = {
        'model': args.model_name,
        'experiment': 'M1_task_decomposition',
        'total_episodes': len(results),
        **metrics,
        'alignment_stats': alignment_stats,
    }
    save_json(summary, os.path.join(args.output_dir, 'summary.json'))

    print(f"\nM1 Task Decomposition completed.")
    print(f"TSR: {metrics['tsr']:.4f} ({metrics['success_count']}/{metrics['n']})")
    print(f"Avg Progress: {metrics['avg_progress']:.4f}")
    print(f"Avg substeps: {alignment_stats['avg_substeps']:.1f} vs GT steps: {alignment_stats['avg_gt_steps']:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval M1: Task Decomposition")
    parser.add_argument("--jsonl_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_m1_ac")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--n_history_image_limit", type=int, default=2)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()
    main(args)
