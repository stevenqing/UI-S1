"""AR Trajectory Evaluation for GUI-Odyssey.

Runs autoregressive trajectory evaluation using a single model via vLLM.
Closely follows scripts/eval/ac/eval_a_ar_trajectory.py, adapted for
GUI-Odyssey's coordinate system and metrics.
"""

import argparse
import copy
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'evaluation'))
sys.path.insert(0, os.path.dirname(__file__))

from x.data.agent.json import JsonFormat
from x.data.agent.space.std_space import RAW_SPACE
from x.qwen.data_format import slim_messages
from evaluation.qwenvl_utils import (
    call_mobile_agent_vllm,
    find_last_image_ele,
)
from odyssey_action_matching import evaluate_odyssey_action, pred_coord_to_1k

# ── Utilities (mirrored from ac_utils.py) ──────────────────────────────

result_lock = Lock()
fm = None


def _json_default(obj):
    """Handle numpy types for JSON serialization."""
    import numpy as np
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def init_format():
    return JsonFormat(RAW_SPACE, add_thought=True, force_add_thought=True)


def safe_parse_response(fm_obj, model_response):
    """Parse model response with JSON error tolerance."""
    import re
    try:
        return fm_obj.parse_response(model_response)
    except json.JSONDecodeError:
        fixed = re.sub(r'\}\}(\s*</action>)', r'}\1', model_response)
        if fixed != model_response:
            try:
                return fm_obj.parse_response(fixed)
            except Exception:
                pass
        match = re.search(r'<action>\s*(\{.*?\})\s*</action>', model_response, re.DOTALL)
        if match:
            action_str = match.group(1)
            while action_str.endswith('}}'):
                action_str = action_str[:-1]
            try:
                action_content = json.loads(action_str)
                think_match = re.search(r'<think>(.*?)</think>', model_response, re.DOTALL)
                return {
                    'think': think_match.group(1).strip() if think_match else '',
                    'action': action_str,
                    'action_content': action_content,
                }
            except json.JSONDecodeError:
                pass
        raise


def length_bucket(n):
    if n <= 3:
        return 'short(1-3)'
    elif n <= 7:
        return 'medium(4-7)'
    elif n <= 15:
        return 'long(8-15)'
    else:
        return 'vlong(16+)'


def load_odyssey_trajectories(jsonl_path, max_episodes=None):
    """Load GUI-Odyssey episodes from JSONL (output of convert_to_eval_format.py)."""
    data = []
    with open(jsonl_path) as f:
        for line in f:
            episode = json.loads(line.strip())
            data.append(episode)
            if max_episodes and len(data) >= max_episodes:
                break
    return data


def compute_trajectory_metrics(results):
    """Compute TSR, avg_progress, scattered_progress."""
    if not results:
        return {'tsr': 0, 'avg_progress': 0, 'scattered_progress': 0, 'n': 0}
    n = len(results)
    success_count = sum(1 for r in results if r['task_success'])
    tsr = success_count / n
    progresses = [r['final_step_id'] / r['num_steps'] for r in results]
    avg_progress = sum(progresses) / n
    total_steps = sum(r['num_steps'] for r in results)
    total_correct = sum(r['final_step_id'] for r in results)
    scattered_progress = total_correct / total_steps if total_steps > 0 else 0
    return {
        'tsr': tsr,
        'avg_progress': avg_progress,
        'scattered_progress': scattered_progress,
        'n': n,
        'success_count': success_count,
    }


# ── Core evaluation ────────────────────────────────────────────────────

def process_episode(episode, args):
    """Process a single episode with AR evaluation."""
    global fm

    ep = copy.deepcopy(episode)
    num_steps = len(ep['steps'])
    state = None
    model_response = None
    step_results = []

    try:
        for step_id in range(num_steps):
            current_check = ep['steps'][step_id]['check_options']
            gt_action = ep['steps'][step_id]['action_content']

            state = fm.gen_next_round(ep, state, previous_model_response=model_response)
            if state is None:
                break

            messages = slim_messages(
                messages=state['messages'],
                num_image_limit=args.n_history_image_limit,
            )

            _, width, height, resized_width, resized_height = find_last_image_ele(messages)

            model_response = call_mobile_agent_vllm(
                messages=messages,
                model_name=args.model_name,
            )

            pred = safe_parse_response(fm, model_response)
            pred_action = pred['action_content']

            type_match, extract_match = evaluate_odyssey_action(
                pred_action, current_check,
                resized_width, resized_height,
            )
            type_match = bool(type_match)
            extract_match = bool(extract_match)

            # Compute pred coordinate in [0,1000] space for downstream analysis
            p_coord_1k = None
            pred_coord_raw = pred_action.get('coordinate')
            if pred_coord_raw and isinstance(pred_coord_raw, (list, tuple)) and len(pred_coord_raw) >= 2:
                try:
                    p_coord_1k = pred_coord_to_1k(
                        [float(pred_coord_raw[0]), float(pred_coord_raw[1])],
                        resized_width, resized_height,
                    )
                except (ValueError, TypeError):
                    pass

            gt_coord_1k = current_check.get('coordinate')  # already in [0,1000]

            step_results.append({
                'step_num': step_id,
                'type_match': type_match,
                'extract_match': extract_match,
                'pred_action': pred_action,
                'gt_action': gt_action,
                'gt_action_type': gt_action['action'],
                'resized_width': resized_width,
                'resized_height': resized_height,
                'pred_coord_1k': p_coord_1k,
                'gt_coord_1k': gt_coord_1k,
            })

            if not extract_match and not args.no_stop:
                break

    except Exception as e:
        print(f"Error episode {episode.get('episode_id', '?')}: {e}")
        import traceback
        traceback.print_exc()

    final_step = len(step_results)
    correct_steps = sum(1 for s in step_results if s['extract_match'])
    task_success = (correct_steps == num_steps and final_step == num_steps)

    result = {
        'episode_id': episode.get('episode_id', None),
        'goal': episode['goal'],
        'category': episode.get('category', ''),
        'device_name': episode.get('device_name', ''),
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

    data = load_odyssey_trajectories(
        jsonl_path=args.jsonl_file,
        max_episodes=args.max_episodes,
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
                    print(f"Progress: {len(results)}/{len(data)} | "
                          f"TSR: {metrics['tsr']:.3f} | AvgProg: {metrics['avg_progress']:.3f}")
            except Exception as e:
                print(f"Exception: {e}")

    # ── Compute summary ────────────────────────────────────────────────
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

    # Per-length-bucket breakdown
    length_stats = {}
    for r in results:
        b = r['length_bucket']
        if b not in length_stats:
            length_stats[b] = []
        length_stats[b].append(r)
    length_metrics = {b: compute_trajectory_metrics(v) for b, v in length_stats.items()}

    # Per-category breakdown
    category_stats = {}
    for r in results:
        cat = r.get('category', 'unknown')
        if cat not in category_stats:
            category_stats[cat] = []
        category_stats[cat].append(r)
    category_metrics = {c: compute_trajectory_metrics(v) for c, v in category_stats.items()}

    # Per-device breakdown
    device_stats = {}
    for r in results:
        dev = r.get('device_name', 'unknown')
        if dev not in device_stats:
            device_stats[dev] = []
        device_stats[dev].append(r)
    device_metrics = {d: compute_trajectory_metrics(v) for d, v in device_stats.items()}

    summary = {
        'model': args.model_name,
        'split': args.split_name,
        'total_episodes': len(results),
        **metrics,
        'action_type_stats': action_stats,
        'length_bucket_stats': length_metrics,
        'category_stats': category_metrics,
        'device_stats': device_metrics,
    }

    summary_path = os.path.join(args.output_dir, 'summary.json')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=_json_default)

    # ── Print results ──────────────────────────────────────────────────
    print(f"\nEvaluation completed.")
    print(f"TSR: {metrics['tsr']:.4f} ({metrics['success_count']}/{metrics['n']})")
    print(f"Avg Progress: {metrics['avg_progress']:.4f}")
    print(f"Scattered Progress: {metrics['scattered_progress']:.4f}")

    print(f"\nPer-action-type extract_match rates:")
    for at in sorted(action_stats.keys()):
        s = action_stats[at]
        print(f"  {at}: {s['extract_match_rate']:.3f} ({s['extract_match']}/{s['total']})")

    print(f"\nPer-category TSR:")
    for cat in sorted(category_metrics.keys()):
        m = category_metrics[cat]
        print(f"  {cat}: TSR={m['tsr']:.3f} AvgProg={m['avg_progress']:.3f} (n={m['n']})")

    print(f"\nPer-length-bucket TSR:")
    for b in sorted(length_metrics.keys()):
        m = length_metrics[b]
        print(f"  {b}: TSR={m['tsr']:.3f} AvgProg={m['avg_progress']:.3f} (n={m['n']})")

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AR Trajectory Evaluation for GUI-Odyssey")
    parser.add_argument('--jsonl_file', type=str, required=True,
                        help='Path to evaluation JSONL (from convert_to_eval_format.py)')
    parser.add_argument('--output_dir', type=str, default='outputs/eval_odyssey',
                        help='Output directory')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name served by vLLM')
    parser.add_argument('--split_name', type=str, default='random_split',
                        help='Split name (for summary metadata)')
    parser.add_argument('--n_history_image_limit', type=int, default=2,
                        help='Max number of history images to keep')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--max_episodes', type=int, default=None,
                        help='Limit episodes for testing')
    parser.add_argument('--no_stop', action='store_true',
                        help='Continue evaluating all steps after errors (no stop-on-error)')
    args = parser.parse_args()
    main(args)
