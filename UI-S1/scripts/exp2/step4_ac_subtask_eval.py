"""
Step 4: AndroidControl evaluation with Subtask Decomposition.

Segments trajectories into subtasks using step_instruction boundaries,
then evaluates each subtask independently with local context only.

Tests whether cross-subtask context accumulation is the main source of
long-horizon failure.
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
from x.data.agent.json import JsonFormat, generate_prompt, MOBILE_USE, OUTPUT_FORMAT
from x.qwen.data_format import slim_messages
from x.qwen.image import make_qwen_image_item


def _set_endpoint(url):
    qwenvl_utils.END_POINT = url

result_lock = Lock()
completed_count = 0


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


def segment_subtasks(steps):
    """
    Segment steps into subtasks based on step_instruction changes.

    Heuristic: when step_instruction changes significantly or when there's
    a natural break (e.g., action type changes from navigation to input),
    create a new subtask boundary.
    """
    if not steps:
        return []

    subtasks = []
    current_subtask = [0]  # Step indices

    for i in range(1, len(steps)):
        prev_instruction = steps[i - 1].get('step_instruction', '')
        curr_instruction = steps[i].get('step_instruction', '')

        # Detect subtask boundary
        is_boundary = False

        # Method 1: Step instruction changes
        if prev_instruction and curr_instruction and prev_instruction != curr_instruction:
            # Check if instructions are sufficiently different
            prev_words = set(prev_instruction.lower().split())
            curr_words = set(curr_instruction.lower().split())
            overlap = len(prev_words & curr_words) / max(len(prev_words | curr_words), 1)
            if overlap < 0.5:
                is_boundary = True

        # Method 2: Action type change (navigation → input)
        prev_type = steps[i - 1]['action_content'].get('action', '')
        curr_type = steps[i]['action_content'].get('action', '')
        nav_actions = {'swipe', 'scroll', 'open'}
        input_actions = {'type', 'click'}
        if prev_type in nav_actions and curr_type in input_actions:
            is_boundary = True

        if is_boundary:
            subtasks.append(current_subtask)
            current_subtask = [i]
        else:
            current_subtask.append(i)

    subtasks.append(current_subtask)
    return subtasks


def evaluate_subtask_isolated(line, subtask_indices, fm, args):
    """
    Evaluate a subtask with fresh (isolated) context.
    Only uses the subtask's goal description + local history.
    """
    steps = line['steps']
    subtask_steps = [steps[i] for i in subtask_indices]

    # Build subtask description from step instructions
    subtask_instructions = [s.get('step_instruction', '') for s in subtask_steps if s.get('step_instruction', '')]
    subtask_desc = subtask_instructions[0] if subtask_instructions else line['goal']

    state = None
    model_response = None
    step_results = []

    # Create a fake "line" for just this subtask
    subtask_line = copy.deepcopy(line)
    subtask_line['steps'] = subtask_steps
    if subtask_desc != line['goal']:
        subtask_line['goal'] = f"{line['goal']} (Current subtask: {subtask_desc})"

    fixed_subtask = fix_line(subtask_line)

    for local_idx in range(len(subtask_steps)):
        global_idx = subtask_indices[local_idx]
        current_check_pam = fixed_subtask['steps'][local_idx]['check_options']

        try:
            state = fm.gen_next_round(
                fixed_subtask, state,
                previous_model_response=model_response
            )
            if state is None:
                break

            messages = state['messages']
            messages = slim_messages(messages=messages, num_image_limit=args.n_history_image_limit)
            _, width, height, resized_width, resized_height = find_last_image_ele(messages)

            raw_response = call_mobile_agent_vllm(
                messages=messages,
                model_name=args.model_name
            )

            pred_action = fm.parse_response(raw_response)
            type_match, extract_match = evaluate_android_control_action(
                pred_action['action_content'], current_check_pam,
                width, height, resized_width, resized_height
            )
            model_response = raw_response
        except Exception:
            type_match, extract_match = False, False
            model_response = None

        step_results.append({
            'step_id': global_idx,
            'local_step_id': local_idx,
            'success': extract_match,
            'type_match': type_match,
        })

        if args.stop_on_error and not extract_match:
            break

    return step_results


def process_line(line, args, fm):
    global completed_count
    num_steps = len(line['steps'])
    fixed_line = fix_line(line)

    # Segment into subtasks
    subtasks = segment_subtasks(fixed_line['steps'])

    # Evaluate each subtask independently
    all_step_results = []
    subtask_successes = []

    for subtask_idx, subtask_indices in enumerate(subtasks):
        subtask_results = evaluate_subtask_isolated(
            fixed_line, subtask_indices, fm, args
        )
        all_step_results.extend(subtask_results)

        # Check if subtask was fully completed
        subtask_success = (
            len(subtask_results) == len(subtask_indices) and
            all(sr['success'] for sr in subtask_results)
        )
        subtask_successes.append({
            'subtask_idx': subtask_idx,
            'step_indices': subtask_indices,
            'num_steps': len(subtask_indices),
            'success': subtask_success,
            'steps_correct': sum(1 for sr in subtask_results if sr['success']),
        })

    # Compute metrics
    total_correct = sum(1 for sr in all_step_results if sr['success'])
    subtask_completion_rate = sum(1 for s in subtask_successes if s['success']) / len(subtask_successes) if subtask_successes else 0

    # Progress rate (sequential): steps before first error / total
    first_error_step = None
    for sr in all_step_results:
        if not sr['success']:
            first_error_step = sr['step_id']
            break
    if first_error_step is not None:
        progress_rate = first_error_step / num_steps
    else:
        progress_rate = 1.0

    # Scattered progress rate: num_correct / total
    scattered_progress_rate = total_correct / num_steps if num_steps > 0 else 0.0

    result = {
        'dataset': 'androidcontrol',
        'trajectory_id': str(line.get('episode_id', '')),
        'goal': line['goal'],
        'num_steps': num_steps,
        'num_subtasks': len(subtasks),
        'subtask_sizes': [len(s) for s in subtasks],
        'step_results': all_step_results,
        'subtask_results': subtask_successes,
        'trajectory_success': total_correct == num_steps,
        'step_accuracy': total_correct / num_steps if num_steps > 0 else 0,
        'progress_rate': progress_rate,
        'scattered_progress_rate': scattered_progress_rate,
        'subtask_completion_rate': subtask_completion_rate,
    }

    with result_lock:
        completed_count += 1
        output_path = os.path.join(args.output_dir, "ac_subtask_eval.jsonl")
        with open(output_path, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        if completed_count % 50 == 0:
            print(f"Progress: {completed_count} trajectories")

    return result


def main(args):
    from x.data.agent.space.std_space import RAW_SPACE
    fm = JsonFormat(RAW_SPACE, add_thought=True, force_add_thought=True)

    os.makedirs(args.output_dir, exist_ok=True)

    output_path = os.path.join(args.output_dir, "ac_subtask_eval.jsonl")
    if os.path.exists(output_path):
        os.remove(output_path)

    std_data = []
    with open(args.jsonl_file, 'r') as f:
        for line in f:
            std_data.append(json.loads(line))

    print(f"Loaded {len(std_data)} trajectories for subtask evaluation")

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_line, line, args, fm): line for line in std_data}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error: {e}")

    # Aggregate
    success_count = sum(1 for r in results if r['trajectory_success'])
    avg_step_acc = sum(r['step_accuracy'] for r in results) / len(results) if results else 0
    avg_subtask_completion = sum(r['subtask_completion_rate'] for r in results) / len(results) if results else 0
    avg_num_subtasks = sum(r['num_subtasks'] for r in results) / len(results) if results else 0

    avg_progress_rate = sum(r['progress_rate'] for r in results) / len(results) if results else 0
    avg_scattered_progress_rate = sum(r['scattered_progress_rate'] for r in results) / len(results) if results else 0

    summary = {
        'total_trajectories': len(results),
        'trajectory_success_rate': success_count / len(results) * 100 if results else 0,
        'avg_step_accuracy': avg_step_acc,
        'avg_progress_rate': avg_progress_rate,
        'avg_scattered_progress_rate': avg_scattered_progress_rate,
        'avg_subtask_completion_rate': avg_subtask_completion,
        'avg_num_subtasks': avg_num_subtasks,
    }

    summary_path = os.path.join(args.output_dir, "ac_subtask_eval_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSubtask Evaluation Results:")
    print(f"  Trajectory Success Rate: {summary['trajectory_success_rate']:.2f}%")
    print(f"  Avg Step Accuracy: {avg_step_acc:.4f}")
    print(f"  Avg Progress Rate (sequential): {avg_progress_rate:.4f}")
    print(f"  Avg Scattered Progress Rate: {avg_scattered_progress_rate:.4f}")
    print(f"  Avg Subtask Completion Rate: {avg_subtask_completion:.4f}")
    print(f"  Avg # Subtasks per Trajectory: {avg_num_subtasks:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_file", type=str,
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation/dataset/android_control_evaluation_std.jsonl")
    parser.add_argument("--output_dir", type=str,
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/results/ac")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--n_history_image_limit", type=int, default=2)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--stop_on_error", action='store_true', default=True)
    parser.add_argument("--api_url", type=str, default="http://localhost:19806/v1")
    args = parser.parse_args()
    _set_endpoint(args.api_url)
    main(args)
