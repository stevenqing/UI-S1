"""Eval M2: Planner-Executor Multi-Agent Evaluation.

Architecture:
1. Planner agent decides action TYPE + target each step
2. Executor agent generates concrete action JSON guided by the plan
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

PLANNER_PROMPT = """You are a mobile action planner. Look at the screenshot and decide what action to take next.

Task goal: {goal}

{history_context}

Choose the correct action TYPE and TARGET for this step.
Important guidelines:
- If the goal requires using a specific app and that app is NOT currently open on screen, use action_type "open" with the app name.
- If you need to tap a button or element, use "click" with a description of the element.
- If you need to enter text, use "type" with the text to enter.
- If you need to scroll, use "swipe" with the direction.
- If you need to press Back/Home/Recent, use "system_button".
- If you need to wait for loading, use "wait".

Output ONLY a JSON object: {{"action_type": "...", "target": "..."}}"""


def call_planner(screenshot_path, goal, action_history, model_name):
    """Call planner to decide action type and target."""
    from x.qwen.image import make_qwen_image_item

    history_str = ""
    if action_history:
        history_str = "Previous actions taken:\n"
        for i, a in enumerate(action_history):
            history_str += f"  Step {i}: {a}\n"

    messages = [
        {'role': 'user', 'content': [
            {'text': PLANNER_PROMPT.format(goal=goal, history_context=history_str)},
            make_qwen_image_item(screenshot_path),
        ]}
    ]

    response = call_mobile_agent_vllm(messages=messages, model_name=model_name)
    return response.strip()


def parse_plan(response):
    """Parse planner JSON output."""
    match = re.search(r'\{[^{}]*\}', response)
    if match:
        try:
            plan = json.loads(match.group())
            return {
                'action_type': plan.get('action_type', 'unknown'),
                'target': plan.get('target', ''),
            }
        except json.JSONDecodeError:
            pass
    return {'action_type': 'unknown', 'target': response}


def process_episode(episode, args):
    """Process episode with planner-executor architecture."""
    global fm

    fixed = fix_line(copy.deepcopy(episode))
    num_steps = len(fixed['steps'])
    state = None
    model_response = None
    step_results = []
    action_history = []

    try:
        for step_id in range(num_steps):
            current_check = fixed['steps'][step_id]['check_options']
            gt_action = fixed['steps'][step_id]['action_content']
            screenshot = fixed['steps'][step_id]['screenshot']

            # Step 1: Planner decides action type + target
            plan_response = call_planner(
                screenshot, episode['goal'], action_history, args.model_name
            )
            plan = parse_plan(plan_response)

            # Step 2: Build executor messages with plan context
            state = fm.gen_next_round(fixed, state, previous_model_response=model_response)
            if state is None:
                break

            messages = state['messages']

            # Inject plan into the latest user message
            plan_context = (
                f"\n\n## Action Plan\n"
                f"Action type: {plan['action_type']}\n"
                f"Target: {plan['target']}\n"
                f"Execute this specific action. Output the action JSON.\n"
            )
            last_user_msg = messages[-1]
            last_user_msg['content'].insert(0, {'text': plan_context})

            messages = slim_messages(messages=messages, num_image_limit=args.n_history_image_limit)
            _, width, height, resized_width, resized_height = find_last_image_ele(messages)

            # Step 3: Executor generates concrete action
            model_response = call_mobile_agent_vllm(messages=messages, model_name=args.model_name)

            pred = safe_parse_response(fm, model_response)
            pred_action = pred['action_content']

            type_match, extract_match = evaluate_android_control_action(
                pred_action, current_check,
                width, height, resized_width, resized_height
            )

            action_history.append(f"{pred_action.get('action', '?')}: {pred_action.get('text', pred_action.get('coordinate', ''))}")

            step_results.append({
                'step_num': step_id,
                'type_match': type_match,
                'extract_match': extract_match,
                'pred_action': pred_action,
                'gt_action': gt_action,
                'gt_action_type': gt_action['action'],
                'plan': plan,
                'plan_response': plan_response,
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
    }

    with result_lock:
        out_path = os.path.join(args.output_dir, 'planner_executor_results.jsonl')
        with open(out_path, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False, default=_json_default) + '\n')

    return result


def main(args):
    global fm
    fm = init_format()
    os.makedirs(args.output_dir, exist_ok=True)

    out_path = os.path.join(args.output_dir, 'planner_executor_results.jsonl')
    if os.path.exists(out_path):
        os.remove(out_path)

    data = load_ac_trajectories(jsonl_path=args.jsonl_file, max_episodes=args.max_episodes)
    print(f"Loaded {len(data)} episodes. Running M2 Planner-Executor evaluation...")

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

    # Planner accuracy analysis
    planner_stats = {'total': 0, 'type_correct': 0}
    for r in results:
        for s in r['step_results']:
            planner_stats['total'] += 1
            if s['plan']['action_type'] == s['gt_action_type']:
                planner_stats['type_correct'] += 1
    planner_stats['type_accuracy'] = (
        planner_stats['type_correct'] / planner_stats['total']
        if planner_stats['total'] > 0 else 0
    )

    summary = {
        'model': args.model_name,
        'experiment': 'M2_planner_executor',
        'total_episodes': len(results),
        **metrics,
        'planner_stats': planner_stats,
    }
    save_json(summary, os.path.join(args.output_dir, 'summary.json'))

    print(f"\nM2 Planner-Executor completed.")
    print(f"TSR: {metrics['tsr']:.4f} ({metrics['success_count']}/{metrics['n']})")
    print(f"Avg Progress: {metrics['avg_progress']:.4f}")
    print(f"Planner type accuracy: {planner_stats['type_accuracy']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval M2: Planner-Executor")
    parser.add_argument("--jsonl_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_m2_ac")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--n_history_image_limit", type=int, default=2)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()
    main(args)
