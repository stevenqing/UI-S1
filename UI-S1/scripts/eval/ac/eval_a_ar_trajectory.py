"""Eval A: AR Trajectory Evaluation for AndroidControl.

Runs autoregressive trajectory evaluation using a single model via vLLM.
Saves detailed per-step results for downstream offline analyses.
"""

import argparse
import copy
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from ac_utils import (
    load_ac_trajectories, fix_line, categorize_action, length_bucket,
    init_format, save_jsonl, save_json, compute_trajectory_metrics,
    evaluate_android_control_action, call_mobile_agent_vllm,
    find_last_image_ele, slim_messages, ALL_ACTION_TYPES,
    safe_parse_response, _json_default,
)

result_lock = Lock()
fm = None


def process_episode(episode, args):
    """Process a single episode with AR evaluation."""
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

            state = fm.gen_next_round(fixed, state, previous_model_response=model_response)
            if state is None:
                break

            messages = slim_messages(
                messages=state['messages'],
                num_image_limit=args.n_history_image_limit
            )

            _, width, height, resized_width, resized_height = find_last_image_ele(messages)

            model_response = call_mobile_agent_vllm(
                messages=messages,
                model_name=args.model_name
            )

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
        print(f"Error episode {episode.get('episode_id', '?')}: {e}")

    final_step = len(step_results)
    correct_steps = sum(1 for s in step_results if s['extract_match'])
    task_success = (correct_steps == num_steps and final_step == num_steps)

    result = {
        'episode_id': episode.get('episode_id', None),
        'goal': episode['goal'],
        'num_steps': num_steps,
        'task_success': task_success,
        'final_step_id': correct_steps,
        'step_results': step_results,
        'length_bucket': length_bucket(num_steps),
    }

    with result_lock:
        out_path = os.path.join(args.output_dir, 'trajectory_results.jsonl')
        with open(out_path, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False, default=_json_default) + '\n')

    return result


def main(args):
    global fm
    fm = init_format()

    os.makedirs(args.output_dir, exist_ok=True)

    # Clear previous results
    out_path = os.path.join(args.output_dir, 'trajectory_results.jsonl')
    if os.path.exists(out_path):
        os.remove(out_path)

    data = load_ac_trajectories(
        jsonl_path=args.jsonl_file,
        max_episodes=args.max_episodes
    )
    print(f"Loaded {len(data)} episodes. Starting evaluation...")

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

    # Compute summary
    metrics = compute_trajectory_metrics(results)

    # Per-action-type breakdown
    action_stats = {}
    for r in results:
        for s in r['step_results']:
            at = s['gt_action_type']
            if at not in action_stats:
                action_stats[at] = {'total': 0, 'type_match': 0, 'extract_match': 0}
            action_stats[at]['total'] += 1
            action_stats[at]['type_match'] += int(s['type_match'])
            action_stats[at]['extract_match'] += int(s['extract_match'])

    for at in action_stats:
        t = action_stats[at]['total']
        action_stats[at]['type_match_rate'] = action_stats[at]['type_match'] / t if t > 0 else 0
        action_stats[at]['extract_match_rate'] = action_stats[at]['extract_match'] / t if t > 0 else 0

    # Per-length breakdown
    length_stats = {}
    for r in results:
        b = r['length_bucket']
        if b not in length_stats:
            length_stats[b] = []
        length_stats[b].append(r)
    length_metrics = {b: compute_trajectory_metrics(v) for b, v in length_stats.items()}

    summary = {
        'model': args.model_name,
        'total_episodes': len(results),
        **metrics,
        'action_type_stats': action_stats,
        'length_bucket_stats': length_metrics,
    }

    save_json(summary, os.path.join(args.output_dir, 'summary.json'))

    print(f"\nEvaluation completed.")
    print(f"TSR: {metrics['tsr']:.4f} ({metrics['success_count']}/{metrics['n']})")
    print(f"Avg Progress: {metrics['avg_progress']:.4f}")
    print(f"Scattered Progress: {metrics['scattered_progress']:.4f}")
    print(f"\nPer-action-type extract_match rates:")
    for at in sorted(action_stats.keys()):
        s = action_stats[at]
        print(f"  {at}: {s['extract_match_rate']:.3f} ({s['extract_match']}/{s['total']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval A: AR Trajectory Evaluation for AndroidControl")
    parser.add_argument("--jsonl_file", type=str, default=None, help="Path to evaluation JSONL")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_a_ac", help="Output directory")
    parser.add_argument("--model_name", type=str, required=True, help="Model name for vLLM")
    parser.add_argument("--n_history_image_limit", type=int, default=2)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--max_episodes", type=int, default=None, help="Limit episodes for testing")
    args = parser.parse_args()
    main(args)
