"""Eval U2: Confidence-Guided Selective Execution AR Trajectory.

Architecture: Adaptive K - start with K_init=3, if agreement >= threshold,
use that. Otherwise generate more samples up to K_max=10.
Universal method using agreement as confidence signal.
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
    generate_k_samples_adaptive,
)

result_lock = Lock()
fm = None


def select_from_samples(samples):
    """Majority vote selection from samples."""
    action_types = [s['pred_action'].get('action', '?')
                    for s in samples if s['pred_action'] and s['parse_ok']]
    if not action_types:
        return samples[0] if samples else None
    voted_type = Counter(action_types).most_common(1)[0][0]
    for s in samples:
        if s['pred_action'] and s['pred_action'].get('action') == voted_type:
            return s
    return samples[0]


def process_episode(episode, args):
    """Process episode with confidence-guided adaptive K."""
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

            # Adaptive K: K_init first, expand to K_max if low agreement (fast: n= or parallel)
            samples, agreement, k_used, expanded = generate_k_samples_adaptive(
                messages, args.model_name,
                args.K_init, args.K_max, args.threshold,
                args.temperature, fm
            )

            # Select action
            selected = select_from_samples(samples)
            if selected and selected['pred_action']:
                pred_action = selected['pred_action']
                model_response = selected['response']
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
                'agreement': agreement,
                'k_used': k_used,
                'expanded': expanded,
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
        out_path = os.path.join(args.output_dir, 'confidence_results.jsonl')
        with open(out_path, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False, default=_json_default) + '\n')

    return result


def main(args):
    global fm
    fm = init_format()
    os.makedirs(args.output_dir, exist_ok=True)

    out_path = os.path.join(args.output_dir, 'confidence_results.jsonl')
    if os.path.exists(out_path):
        os.remove(out_path)

    data = load_ac_trajectories(jsonl_path=args.jsonl_file, max_episodes=args.max_episodes)
    print(f"Loaded {len(data)} episodes. Running U2 Confidence-Guided (K_init={args.K_init}, K_max={args.K_max}, threshold={args.threshold})...")

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

    # Compute adaptive stats
    all_steps = [s for r in results for s in r['step_results']]
    expanded_count = sum(1 for s in all_steps if s['expanded'])
    avg_k = sum(s['k_used'] for s in all_steps) / len(all_steps) if all_steps else 0

    summary = {
        'model': args.model_name,
        'experiment': 'U2_confidence_guided',
        'K_init': args.K_init,
        'K_max': args.K_max,
        'threshold': args.threshold,
        'total_episodes': len(results),
        **metrics,
        'expanded_fraction': expanded_count / len(all_steps) if all_steps else 0,
        'avg_k_used': avg_k,
    }
    save_json(summary, os.path.join(args.output_dir, 'summary.json'))

    print(f"\nU2 Confidence-Guided completed.")
    print(f"TSR: {metrics['tsr']:.4f} ({metrics['success_count']}/{metrics['n']})")
    print(f"Avg K used: {avg_k:.1f} | Expanded fraction: {expanded_count}/{len(all_steps)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval U2: Confidence-Guided")
    parser.add_argument("--jsonl_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_u2_ac")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--K_init", type=int, default=3)
    parser.add_argument("--K_max", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n_history_image_limit", type=int, default=2)
    parser.add_argument("--max_workers", type=int, default=2)
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()
    main(args)
