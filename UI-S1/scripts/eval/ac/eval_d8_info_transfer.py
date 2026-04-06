"""Eval D8: Observer Info Transfer Ablation (GPU).

3 conditions:
- C: No observer (baseline = Eval A)
- B: Current-step observer only (no history)
- A: Full state document (= D1)
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

OBSERVER_PROMPT = """You are a mobile screen observer. Look at the current screenshot and describe what you see.
Focus on: the current app, screen layout, key UI elements, and any visible state changes.
Be concise (2-4 sentences)."""


def call_observer(screenshot_path, goal, model_name):
    """Single-step observer (no history)."""
    from x.qwen.image import make_qwen_image_item
    messages = [
        {'role': 'system', 'content': [{'text': OBSERVER_PROMPT}]},
        {'role': 'user', 'content': [
            {'text': f"Task goal: {goal}\nDescribe the current screen:"},
            make_qwen_image_item(screenshot_path),
        ]}
    ]
    return call_mobile_agent_vllm(messages=messages, model_name=model_name).strip()


def run_condition(episode, condition, args):
    """Run a single condition for an episode."""
    global fm

    fixed = fix_line(copy.deepcopy(episode))
    num_steps = len(fixed['steps'])
    state = None
    model_response = None
    step_results = []
    observations = []

    try:
        for step_id in range(num_steps):
            current_check = fixed['steps'][step_id]['check_options']
            gt_action = fixed['steps'][step_id]['action_content']
            screenshot = fixed['steps'][step_id]['screenshot']

            state = fm.gen_next_round(fixed, state, previous_model_response=model_response)
            if state is None:
                break

            messages = state['messages']

            if condition == 'B':
                # Current-step observer only
                obs = call_observer(screenshot, episode['goal'], args.model_name)
                messages[-1]['content'].insert(0, {'text': f"\n\n## Screen Observation\n{obs}\n"})
            elif condition == 'A':
                # Full state document
                obs = call_observer(screenshot, episode['goal'], args.model_name)
                observations.append(f"Step {step_id}: {obs}")
                state_doc = "\n".join(observations[-5:])
                messages[-1]['content'].insert(0, {'text': f"\n\n## Observer Context\n{state_doc}\n"})
            # condition C: no observer, messages unchanged

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
            })

            if not extract_match:
                break

    except Exception as e:
        print(f"Error episode {episode.get('episode_id', '?')} cond={condition}: {e}")

    correct_steps = sum(1 for s in step_results if s['extract_match'])
    task_success = (correct_steps == num_steps)

    return {
        'episode_id': episode.get('episode_id'),
        'goal': episode['goal'],
        'num_steps': num_steps,
        'task_success': task_success,
        'final_step_id': correct_steps,
        'step_results': step_results,
        'condition': condition,
        'length_bucket': length_bucket(num_steps),
    }


def process_episode(episode, args):
    """Run all 3 conditions for an episode."""
    results = {}
    for cond in ['C', 'B', 'A']:
        results[cond] = run_condition(episode, cond, args)

    with result_lock:
        out_path = os.path.join(args.output_dir, 'info_transfer_results.jsonl')
        with open(out_path, 'a') as f:
            for cond, r in results.items():
                f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + '\n')

    return results


def main(args):
    global fm
    fm = init_format()
    os.makedirs(args.output_dir, exist_ok=True)

    out_path = os.path.join(args.output_dir, 'info_transfer_results.jsonl')
    if os.path.exists(out_path):
        os.remove(out_path)

    data = load_ac_trajectories(jsonl_path=args.jsonl_file, max_episodes=args.max_episodes)
    print(f"Loaded {len(data)} episodes. Running 3-condition ablation...")

    all_results = {'C': [], 'B': [], 'A': []}
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_episode, ep, args): ep for ep in data}
        for future in as_completed(futures):
            try:
                result = future.result()
                for cond in ['C', 'B', 'A']:
                    all_results[cond].append(result[cond])
                done = len(all_results['C'])
                if done % 20 == 0:
                    print(f"Progress: {done}/{len(data)}")
            except Exception as e:
                print(f"Exception: {e}")

    summary = {}
    for cond in ['C', 'B', 'A']:
        metrics = compute_trajectory_metrics(all_results[cond])
        summary[cond] = metrics
        label = {'C': 'No observer', 'B': 'Current-step only', 'A': 'Full state doc'}[cond]
        print(f"\nCondition {cond} ({label}): TSR={metrics['tsr']:.4f}, AvgProg={metrics['avg_progress']:.4f}")

    save_json(summary, os.path.join(args.output_dir, 'summary.json'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval D8: Observer Info Transfer Ablation")
    parser.add_argument("--jsonl_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_d8_ac")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--n_history_image_limit", type=int, default=2)
    parser.add_argument("--max_workers", type=int, default=2)
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()
    main(args)
