"""Eval D9: Critic Zero-shot (GPU).

After each step, asks the model to evaluate whether the step was correct.
Measures precision/recall/F1 for failure detection.
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
    find_last_image_ele, slim_messages, ALL_ACTION_TYPES, safe_parse_response,
    _json_default,
)

result_lock = Lock()
fm = None

CRITIC_PROMPT = """You are a mobile UI action critic. You will be shown a screenshot and an action that was taken on it. Evaluate whether the action is appropriate for the given task.

Respond with exactly one word: PASS or FAIL

PASS: The action correctly advances toward the task goal.
FAIL: The action is incorrect, irrelevant, or counterproductive."""


def call_critic(screenshot_path, action_desc, goal, model_name):
    """Ask model to evaluate an action."""
    from x.qwen.image import make_qwen_image_item

    messages = [
        {'role': 'system', 'content': [{'text': CRITIC_PROMPT}]},
        {'role': 'user', 'content': [
            {'text': f"Task goal: {goal}\n\nAction taken: {json.dumps(action_desc)}\n\nIs this action correct?"},
            make_qwen_image_item(screenshot_path),
        ]}
    ]

    response = call_mobile_agent_vllm(messages=messages, model_name=model_name).strip()
    # Parse PASS/FAIL
    response_upper = response.upper()
    if 'FAIL' in response_upper:
        return 'FAIL'
    return 'PASS'


def process_episode(episode, args):
    """Run AR eval with critic at each step."""
    global fm

    fixed = fix_line(copy.deepcopy(episode))
    num_steps = len(fixed['steps'])
    state = None
    model_response = None
    step_results = []

    try:
        for step_id in range(num_steps):
            current_check = fixed['steps'][step_id]['check_options']
            gt_action = fixed['steps'][step_id]['action_content']
            screenshot = fixed['steps'][step_id]['screenshot']

            state = fm.gen_next_round(fixed, state, previous_model_response=model_response)
            if state is None:
                break

            messages = slim_messages(
                messages=state['messages'],
                num_image_limit=args.n_history_image_limit
            )
            _, width, height, resized_width, resized_height = find_last_image_ele(messages)

            model_response = call_mobile_agent_vllm(messages=messages, model_name=args.model_name)

            pred = safe_parse_response(fm, model_response)
            pred_action = pred['action_content']

            type_match, extract_match = evaluate_android_control_action(
                pred_action, current_check,
                width, height, resized_width, resized_height
            )

            # Critic evaluation
            critic_verdict = call_critic(screenshot, pred_action, episode['goal'], args.model_name)

            step_results.append({
                'step_num': step_id,
                'type_match': type_match,
                'extract_match': extract_match,
                'pred_action': pred_action,
                'gt_action': gt_action,
                'gt_action_type': gt_action['action'],
                'critic_verdict': critic_verdict,
                'actually_correct': extract_match,
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
        out_path = os.path.join(args.output_dir, 'critic_results.jsonl')
        with open(out_path, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False, default=_json_default) + '\n')

    return result


def main(args):
    global fm
    fm = init_format()
    os.makedirs(args.output_dir, exist_ok=True)

    out_path = os.path.join(args.output_dir, 'critic_results.jsonl')
    if os.path.exists(out_path):
        os.remove(out_path)

    data = load_ac_trajectories(jsonl_path=args.jsonl_file, max_episodes=args.max_episodes)
    print(f"Loaded {len(data)} episodes. Running critic zero-shot...")

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_episode, ep, args): ep for ep in data}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                if len(results) % 50 == 0:
                    print(f"Progress: {len(results)}/{len(data)}")
            except Exception as e:
                print(f"Exception: {e}")

    # Compute critic metrics
    tp = fp = fn = tn = 0
    action_critic = {}
    pass_bias = 0
    total_verdicts = 0

    for r in results:
        for s in r['step_results']:
            actual_fail = not s['actually_correct']
            critic_fail = s['critic_verdict'] == 'FAIL'
            total_verdicts += 1
            if s['critic_verdict'] == 'PASS':
                pass_bias += 1

            if actual_fail and critic_fail:
                tp += 1
            elif not actual_fail and critic_fail:
                fp += 1
            elif actual_fail and not critic_fail:
                fn += 1
            else:
                tn += 1

            at = s['gt_action_type']
            if at not in action_critic:
                action_critic[at] = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
            if actual_fail and critic_fail:
                action_critic[at]['tp'] += 1
            elif not actual_fail and critic_fail:
                action_critic[at]['fp'] += 1
            elif actual_fail and not critic_fail:
                action_critic[at]['fn'] += 1
            else:
                action_critic[at]['tn'] += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    action_critic_metrics = {}
    for at, counts in action_critic.items():
        p = counts['tp'] / (counts['tp'] + counts['fp']) if (counts['tp'] + counts['fp']) > 0 else 0
        r = counts['tp'] / (counts['tp'] + counts['fn']) if (counts['tp'] + counts['fn']) > 0 else 0
        action_critic_metrics[at] = {
            **counts,
            'precision': p,
            'recall': r,
            'f1': 2 * p * r / (p + r) if (p + r) > 0 else 0,
        }

    summary = {
        'model': args.model_name,
        'total_steps_evaluated': total_verdicts,
        'critic_for_fail_detection': {
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        },
        'pass_bias': pass_bias / total_verdicts if total_verdicts > 0 else 0,
        'per_action_type_critic': action_critic_metrics,
        **compute_trajectory_metrics(results),
    }

    save_json(summary, os.path.join(args.output_dir, 'summary.json'))

    print(f"\nCritic Zero-shot Results:")
    print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    print(f"  PASS bias: {summary['pass_bias']:.3f}")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval D9: Critic Zero-shot")
    parser.add_argument("--jsonl_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_d9_ac")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--n_history_image_limit", type=int, default=2)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()
    main(args)
