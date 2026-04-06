"""Eval U3: Self-Consistency AR Trajectory Evaluation.

Architecture: K samples per step, majority vote on action type,
then aggregate action content (average coords, vote text).
Universal method with content-level aggregation.
"""

import argparse
import copy
import json
import os
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from ac_utils import (
    load_ac_trajectories, fix_line, init_format, save_jsonl, save_json,
    compute_trajectory_metrics, length_bucket,
    evaluate_android_control_action,
    find_last_image_ele, slim_messages, safe_parse_response, _json_default,
    generate_k_samples_fast,
)

result_lock = Lock()
fm = None


def self_consistency_aggregate(samples):
    """Aggregate samples: majority vote on type, then merge content."""
    valid = [s for s in samples if s['pred_action'] and s['parse_ok']]
    if not valid:
        return samples[0] if samples else None, {}, samples[0]['response'] if samples else ''

    # Majority vote on action type
    action_types = [s['pred_action'].get('action', '?') for s in valid]
    type_counter = Counter(action_types)
    voted_type = type_counter.most_common(1)[0][0]
    agreement = type_counter.most_common(1)[0][1] / len(action_types)

    # Filter to voted type
    typed_samples = [s for s in valid if s['pred_action'].get('action') == voted_type]

    # Aggregate content based on action type
    merged_action = {'action': voted_type}

    if voted_type in ('click', 'long_press'):
        # Average coordinates
        coords = []
        for s in typed_samples:
            c = s['pred_action'].get('coordinate')
            if c and isinstance(c, (list, tuple)) and len(c) == 2:
                coords.append(c)
        if coords:
            avg_coord = [int(round(np.mean([c[0] for c in coords]))),
                         int(round(np.mean([c[1] for c in coords])))]
            merged_action['coordinate'] = avg_coord
        elif typed_samples:
            merged_action['coordinate'] = typed_samples[0]['pred_action'].get('coordinate', [0, 0])

        if voted_type == 'long_press':
            times = [s['pred_action'].get('time', 2) for s in typed_samples if 'time' in s['pred_action']]
            merged_action['time'] = int(round(np.mean(times))) if times else 2

    elif voted_type == 'swipe':
        # Average both coordinate pairs
        coords1, coords2 = [], []
        for s in typed_samples:
            c1 = s['pred_action'].get('coordinate')
            c2 = s['pred_action'].get('coordinate2')
            if c1 and c2:
                coords1.append(c1)
                coords2.append(c2)
        if coords1 and coords2:
            merged_action['coordinate'] = [int(round(np.mean([c[0] for c in coords1]))),
                                            int(round(np.mean([c[1] for c in coords1])))]
            merged_action['coordinate2'] = [int(round(np.mean([c[0] for c in coords2]))),
                                             int(round(np.mean([c[1] for c in coords2])))]
        elif typed_samples:
            merged_action['coordinate'] = typed_samples[0]['pred_action'].get('coordinate', [0, 0])
            merged_action['coordinate2'] = typed_samples[0]['pred_action'].get('coordinate2', [0, 0])

    elif voted_type in ('type', 'open'):
        # Vote on text
        texts = [s['pred_action'].get('text', '') for s in typed_samples if s['pred_action'].get('text')]
        if texts:
            merged_action['text'] = Counter(texts).most_common(1)[0][0]
        elif typed_samples:
            merged_action['text'] = typed_samples[0]['pred_action'].get('text', '')

    elif voted_type == 'system_button':
        buttons = [s['pred_action'].get('button', '') for s in typed_samples if s['pred_action'].get('button')]
        if buttons:
            merged_action['button'] = Counter(buttons).most_common(1)[0][0]
        elif typed_samples:
            merged_action['button'] = typed_samples[0]['pred_action'].get('button', 'Back')

    elif voted_type == 'wait':
        times = [s['pred_action'].get('time', 2) for s in typed_samples if 'time' in s['pred_action']]
        merged_action['time'] = int(round(np.mean(times))) if times else 2

    elif voted_type == 'key':
        texts = [s['pred_action'].get('text', '') for s in typed_samples if s['pred_action'].get('text')]
        if texts:
            merged_action['text'] = Counter(texts).most_common(1)[0][0]

    vote_info = {
        'agreement': agreement,
        'voted_type': voted_type,
        'n_typed': len(typed_samples),
    }

    # Use the first typed sample's response for AR continuation
    best_response = typed_samples[0]['response'] if typed_samples else valid[0]['response']

    return {'pred_action': merged_action, 'parse_ok': True}, vote_info, best_response


def process_episode(episode, args):
    """Process episode with self-consistency AR evaluation."""
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

            # Generate K samples (fast: n=K or parallel, pre-encoded images)
            samples = generate_k_samples_fast(messages, args.model_name, args.K, args.temperature, fm)

            # Self-consistency aggregation
            selected, vote_info, best_response = self_consistency_aggregate(samples)

            if selected and selected['pred_action']:
                pred_action = selected['pred_action']
                model_response = best_response
            else:
                break

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
                'agreement': vote_info.get('agreement', 0),
                'voted_type': vote_info.get('voted_type', ''),
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
        out_path = os.path.join(args.output_dir, 'selfconsistency_results.jsonl')
        with open(out_path, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False, default=_json_default) + '\n')

    return result


def main(args):
    global fm
    fm = init_format()
    os.makedirs(args.output_dir, exist_ok=True)

    out_path = os.path.join(args.output_dir, 'selfconsistency_results.jsonl')
    if os.path.exists(out_path):
        os.remove(out_path)

    data = load_ac_trajectories(jsonl_path=args.jsonl_file, max_episodes=args.max_episodes)
    print(f"Loaded {len(data)} episodes. Running U3 Self-Consistency AR (K={args.K})...")

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

    summary = {
        'model': args.model_name,
        'experiment': 'U3_self_consistency_AR',
        'K': args.K,
        'temperature': args.temperature,
        'total_episodes': len(results),
        **metrics,
    }
    save_json(summary, os.path.join(args.output_dir, 'summary.json'))

    print(f"\nU3 Self-Consistency AR completed.")
    print(f"TSR: {metrics['tsr']:.4f} ({metrics['success_count']}/{metrics['n']})")
    print(f"Avg Progress: {metrics['avg_progress']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval U3: Self-Consistency AR")
    parser.add_argument("--jsonl_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_u3_ac")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n_history_image_limit", type=int, default=2)
    parser.add_argument("--max_workers", type=int, default=2)
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()
    main(args)
