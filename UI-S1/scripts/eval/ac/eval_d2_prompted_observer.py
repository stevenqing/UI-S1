"""Eval D2: Prompted Observer AR Evaluation (GPU).

Uses a structured observer prompt with mobile-specific focus.
"""

import argparse
import copy
import json
import os
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

PROMPTED_OBSERVER = """You are a mobile screen analyst. Analyze the current screenshot and provide a structured observation.

Output your analysis in this exact format:

APP: [Name of the currently open app]
SCREEN: [Current screen/page description]
UI_ELEMENTS: [List of key interactive elements visible]
STATE_CHANGE: [What changed from the previous step, or "Initial state" if first step]
PROGRESS: [Assessment of progress toward the task goal]

Be precise and factual. Focus on actionable information."""


def call_prompted_observer(screenshot_path, goal, prev_observation, model_name):
    """Call observer with structured prompt."""
    from x.qwen.image import make_qwen_image_item

    context = f"Task goal: {goal}\n"
    if prev_observation:
        context += f"\nPrevious observation:\n{prev_observation}\n"
    context += "\nAnalyze the current screen:"

    messages = [
        {'role': 'system', 'content': [{'text': PROMPTED_OBSERVER}]},
        {'role': 'user', 'content': [
            {'text': context},
            make_qwen_image_item(screenshot_path),
        ]}
    ]

    response = call_mobile_agent_vllm(messages=messages, model_name=model_name)
    return response.strip()


def process_episode(episode, args):
    """Process episode with prompted observer."""
    global fm

    fixed = fix_line(copy.deepcopy(episode))
    num_steps = len(fixed['steps'])
    state = None
    model_response = None
    step_results = []
    prev_observation = None

    try:
        for step_id in range(num_steps):
            current_check = fixed['steps'][step_id]['check_options']
            gt_action = fixed['steps'][step_id]['action_content']
            screenshot = fixed['steps'][step_id]['screenshot']

            # Prompted observer
            observation = call_prompted_observer(
                screenshot, episode['goal'], prev_observation, args.model_name
            )
            prev_observation = observation

            # Build executor messages
            state = fm.gen_next_round(fixed, state, previous_model_response=model_response)
            if state is None:
                break

            messages = state['messages']

            # Inject structured observation
            obs_context = f"\n\n## Screen Analysis\n{observation}\n"
            messages[-1]['content'].insert(0, {'text': obs_context})

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
                'observation': observation,
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
        out_path = os.path.join(args.output_dir, 'prompted_observer_results.jsonl')
        with open(out_path, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False, default=_json_default) + '\n')

    return result


def main(args):
    global fm
    fm = init_format()
    os.makedirs(args.output_dir, exist_ok=True)

    out_path = os.path.join(args.output_dir, 'prompted_observer_results.jsonl')
    if os.path.exists(out_path):
        os.remove(out_path)

    data = load_ac_trajectories(jsonl_path=args.jsonl_file, max_episodes=args.max_episodes)
    print(f"Loaded {len(data)} episodes. Running prompted observer AR evaluation...")

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_episode, ep, args): ep for ep in data}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                if len(results) % 50 == 0:
                    metrics = compute_trajectory_metrics(results)
                    print(f"Progress: {len(results)}/{len(data)} | TSR: {metrics['tsr']:.3f}")
            except Exception as e:
                print(f"Exception: {e}")

    metrics = compute_trajectory_metrics(results)
    save_json({'model': args.model_name, **metrics}, os.path.join(args.output_dir, 'summary.json'))

    print(f"\nPrompted Observer AR completed.")
    print(f"TSR: {metrics['tsr']:.4f} | Avg Progress: {metrics['avg_progress']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval D2: Prompted Observer AR")
    parser.add_argument("--jsonl_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_d2_ac")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--n_history_image_limit", type=int, default=2)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()
    main(args)
