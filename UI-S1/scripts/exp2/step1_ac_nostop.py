"""
Step 1: AndroidControl evaluation WITHOUT stopping on error.

Two modes:
  - natural_cascade: continue with model's (possibly wrong) responses
  - oracle_rescue:   on error, use GT response for next step's context

Records per-step results for cascade analysis.
"""

import argparse
import copy
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'evaluation')))

import evaluation.qwenvl_utils as qwenvl_utils
from evaluation.qwenvl_utils import (
    call_mobile_agent_vllm, evaluate_android_control_action, find_last_image_ele
)
from x.data.agent.json import JsonFormat
from x.qwen.data_format import slim_messages


def _set_endpoint(url):
    """Override the hardcoded END_POINT in qwenvl_utils."""
    qwenvl_utils.END_POINT = url

result_lock = Lock()
completed_count = 0
total_count = 0


def fix_line(line):
    for step in line['steps']:
        if 'check_options' in step:
            continue
        check_options = copy.deepcopy(step['action_content'])
        if 'bbox' in step:
            check_options['candidate_bbox'] = step['bbox']
        else:
            check_options['candidate_bbox'] = []
        step['check_options'] = check_options
    return line


def process_line(line, args, fm):
    global completed_count
    num_steps = len(line['steps'])
    fixed_line = fix_line(line)

    step_results = []
    state = None
    model_response = None  # None means use GT on first call

    for step_id in range(num_steps):
        current_check_pam = fixed_line['steps'][step_id]['check_options']

        try:
            state = fm.gen_next_round(
                fixed_line, state,
                previous_model_response=model_response
            )
        except Exception as e:
            # If state generation fails, record remaining steps as failures
            for remaining_id in range(step_id, num_steps):
                step_results.append({
                    'step_id': remaining_id,
                    'success': False,
                    'type_match': False,
                    'grounding_match': False,
                    'pred_action': {'error': f'state_gen_failed: {str(e)[:100]}'},
                    'gt_action_type': current_check_pam.get('action', 'unknown'),
                    'step_instruction': fixed_line['steps'][remaining_id].get('step_instruction', ''),
                })
            break

        if state is None:
            break

        messages = state['messages']
        messages = slim_messages(messages=messages, num_image_limit=args.n_history_image_limit)

        try:
            _, width, height, resized_width, resized_height = find_last_image_ele(messages)
        except Exception:
            step_results.append({
                'step_id': step_id,
                'success': False,
                'type_match': False,
                'grounding_match': False,
                'pred_action': {'error': 'image_load_failed'},
                'gt_action_type': current_check_pam.get('action', 'unknown'),
                'step_instruction': fixed_line['steps'][step_id].get('step_instruction', ''),
            })
            if args.mode == 'oracle_rescue':
                model_response = None
            continue

        raw_response = call_mobile_agent_vllm(
            messages=messages,
            model_name=args.model_name
        )

        try:
            pred_action = fm.parse_response(raw_response)
            type_match, extract_match = evaluate_android_control_action(
                pred_action['action_content'], current_check_pam,
                width, height, resized_width, resized_height
            )
            pred_action_content = pred_action['action_content']
        except Exception:
            type_match, extract_match = False, False
            pred_action_content = {'error': 'parse_failed', 'raw': raw_response[:200]}

        step_results.append({
            'step_id': step_id,
            'success': extract_match,
            'type_match': type_match,
            'grounding_match': type_match and extract_match,
            'pred_action': pred_action_content,
            'gt_action_type': current_check_pam.get('action', 'unknown'),
            'step_instruction': fixed_line['steps'][step_id].get('step_instruction', ''),
        })

        # Decide context for next step
        if args.mode == 'oracle_rescue' and not extract_match:
            model_response = None  # Use GT for next step
        else:
            model_response = raw_response  # Use model's response

    # Compute trajectory metrics
    first_error = None
    for i, sr in enumerate(step_results):
        if not sr['success']:
            first_error = i
            break

    scattered_correct = sum(1 for sr in step_results if sr['success'])

    result = {
        'dataset': 'androidcontrol',
        'trajectory_id': str(line.get('episode_id', '')),
        'goal': line['goal'],
        'num_steps': num_steps,
        'num_evaluated': len(step_results),
        'mode': args.mode,
        'steps': step_results,
        'trajectory_success': all(sr['success'] for sr in step_results) and len(step_results) == num_steps,
        'first_error_step': first_error,
        'progress_rate': (first_error / num_steps) if first_error is not None else 1.0,
        'scattered_progress_rate': scattered_correct / num_steps if num_steps > 0 else 0.0,
    }

    # Thread-safe write
    with result_lock:
        completed_count += 1
        output_path = os.path.join(args.output_dir, f"ac_nostop_{args.mode}.jsonl")
        with open(output_path, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        if completed_count % 50 == 0:
            print(f"Progress: {completed_count}/{total_count} trajectories completed")

    return result


def main(args):
    global total_count

    from x.data.agent.space.std_space import RAW_SPACE
    fm = JsonFormat(RAW_SPACE, add_thought=True, force_add_thought=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # Clear previous results
    output_path = os.path.join(args.output_dir, f"ac_nostop_{args.mode}.jsonl")
    if os.path.exists(output_path):
        os.remove(output_path)

    # Load data
    std_data = []
    with open(args.jsonl_file, 'r') as f:
        for line in f:
            std_data.append(json.loads(line))

    total_count = len(std_data)
    print(f"Loaded {total_count} trajectories. Mode: {args.mode}")
    print(f"Output: {output_path}")

    # Run evaluation
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_line, line, args, fm): line for line in std_data}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Task exception: {e}")

    # Compute aggregate statistics
    success_count = sum(1 for r in results if r['trajectory_success'])
    success_rate = success_count / len(results) * 100 if results else 0

    avg_progress = sum(r['progress_rate'] for r in results) / len(results) if results else 0
    avg_scattered = sum(r['scattered_progress_rate'] for r in results) / len(results) if results else 0

    # Step-0 failure analysis
    step0_fail = sum(1 for r in results if r['steps'] and not r['steps'][0]['success'])
    step0_total = sum(1 for r in results if r['steps'])
    step0_fail_rate = step0_fail / step0_total * 100 if step0_total > 0 else 0

    # Cascade depth: if first error is fixed, how far can we go?
    cascade_depths = []
    for r in results:
        if r['first_error_step'] is not None and r['first_error_step'] < len(r['steps']) - 1:
            # Count consecutive correct steps after first error
            depth = 0
            for s in r['steps'][r['first_error_step'] + 1:]:
                if s['success']:
                    depth += 1
                else:
                    break
            cascade_depths.append(depth)

    # Error type distribution
    error_types = {'planning': 0, 'grounding': 0, 'total_errors': 0}
    for r in results:
        for s in r['steps']:
            if not s['success']:
                error_types['total_errors'] += 1
                if not s['type_match']:
                    error_types['planning'] += 1  # Wrong action type = planning error
                else:
                    error_types['grounding'] += 1  # Right type, wrong target = grounding error

    summary = {
        'mode': args.mode,
        'total_trajectories': len(results),
        'trajectory_success_rate': success_rate,
        'avg_progress_rate': avg_progress,
        'avg_scattered_progress_rate': avg_scattered,
        'step0_failure_rate': step0_fail_rate,
        'step0_fail_count': step0_fail,
        'step0_total': step0_total,
        'mean_cascade_depth_after_first_error': sum(cascade_depths) / len(cascade_depths) if cascade_depths else 0,
        'error_type_distribution': error_types,
    }

    summary_path = os.path.join(args.output_dir, f"ac_nostop_{args.mode}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"AC No-Stop Evaluation Summary (Mode: {args.mode})")
    print("=" * 60)
    print(f"Trajectories: {len(results)}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Avg Progress Rate: {avg_progress:.4f}")
    print(f"Avg Scattered Progress: {avg_scattered:.4f}")
    print(f"Step-0 Failure Rate: {step0_fail_rate:.2f}%")
    print(f"Mean Cascade Depth: {summary['mean_cascade_depth_after_first_error']:.2f}")
    print(f"Error Types: Planning={error_types['planning']}, Grounding={error_types['grounding']}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_file", type=str,
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation/dataset/android_control_evaluation_std.jsonl")
    parser.add_argument("--output_dir", type=str,
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/results/ac")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=['natural_cascade', 'oracle_rescue'], default='natural_cascade')
    parser.add_argument("--n_history_image_limit", type=int, default=2)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--api_url", type=str, default="http://localhost:19806/v1",
                        help="vLLM API endpoint URL")
    args = parser.parse_args()
    _set_endpoint(args.api_url)
    main(args)
