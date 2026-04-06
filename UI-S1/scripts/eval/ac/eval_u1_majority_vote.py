"""Eval U1: Majority Vote AR Trajectory Evaluation.

Architecture: At each step, generate K samples with temperature sampling,
majority vote on action type, pick the first matching sample, continue AR.
Universal method - no dataset-specific knowledge needed.
"""

import argparse
import copy
import json
import os
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


def majority_vote_select(samples):
    """Select action by majority vote on action type, then pick first match."""
    action_types = []
    for s in samples:
        if s['pred_action'] and s['parse_ok']:
            action_types.append(s['pred_action'].get('action', 'unknown'))

    if not action_types:
        # All failed - return first sample's action (or None)
        return samples[0] if samples else None, 'all_failed', {}

    type_counter = Counter(action_types)
    voted_type = type_counter.most_common(1)[0][0]
    agreement = type_counter.most_common(1)[0][1] / len(action_types)

    # Pick first sample with voted type
    for s in samples:
        if s['pred_action'] and s['pred_action'].get('action') == voted_type:
            return s, voted_type, {
                'agreement': agreement,
                'type_counts': dict(type_counter),
                'voted_type': voted_type,
            }

    return samples[0], voted_type, {'agreement': agreement}


def process_episode(episode, args):
    """Process episode with majority vote AR evaluation."""
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

            # Majority vote
            selected, voted_type, vote_info = majority_vote_select(samples)

            if selected and selected['pred_action']:
                pred_action = selected['pred_action']
                model_response = selected['response']
            else:
                # Fallback: use greedy (first sample)
                pred_action = samples[0]['pred_action'] if samples else None
                model_response = samples[0]['response'] if samples else ''
                if pred_action is None:
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
                'voted_type': voted_type,
                'agreement': vote_info.get('agreement', 0),
                'k_samples': len(samples),
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
        out_path = os.path.join(args.output_dir, 'majority_vote_results.jsonl')
        with open(out_path, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False, default=_json_default) + '\n')

    return result


def main(args):
    global fm
    fm = init_format()
    os.makedirs(args.output_dir, exist_ok=True)

    out_path = os.path.join(args.output_dir, 'majority_vote_results.jsonl')
    if os.path.exists(out_path):
        os.remove(out_path)

    data = load_ac_trajectories(jsonl_path=args.jsonl_file, max_episodes=args.max_episodes)
    print(f"Loaded {len(data)} episodes. Running U1 Majority Vote AR (K={args.K})...")

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

    # Agreement stats
    all_agreements = [s['agreement'] for r in results for s in r['step_results']]
    avg_agreement = sum(all_agreements) / len(all_agreements) if all_agreements else 0

    summary = {
        'model': args.model_name,
        'experiment': 'U1_majority_vote_AR',
        'K': args.K,
        'temperature': args.temperature,
        'total_episodes': len(results),
        **metrics,
        'avg_agreement': avg_agreement,
    }
    save_json(summary, os.path.join(args.output_dir, 'summary.json'))

    print(f"\nU1 Majority Vote AR completed.")
    print(f"TSR: {metrics['tsr']:.4f} ({metrics['success_count']}/{metrics['n']})")
    print(f"Avg Progress: {metrics['avg_progress']:.4f}")
    print(f"Avg Agreement: {avg_agreement:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval U1: Majority Vote AR")
    parser.add_argument("--jsonl_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_u1_ac")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n_history_image_limit", type=int, default=2)
    parser.add_argument("--max_workers", type=int, default=2)
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()
    main(args)
