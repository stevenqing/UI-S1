"""
Step 3: AndroidControl evaluation with Summary Context variants.

Three context formats:
  A. action_level:   "Step 1: click at [234, 567]"
  B. semantic_level:  "Step 1: opened Settings" (uses step_instruction)
  C. progress_level:  "Completed: X, Y. Remaining: Z"

Tests whether richer summaries help the model maintain goal focus.
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
    call_mobile_agent_vllm, evaluate_android_control_action, find_last_image_ele,
    image_to_data_url
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


def format_action_summary(step, step_id):
    """Format A: Action-level summary."""
    action = step['action_content']
    action_type = action.get('action', 'unknown')
    if 'coordinate' in action:
        coord = action['coordinate']
        return f"Step {step_id + 1}: {action_type} at [{coord[0]:.0f}, {coord[1]:.0f}]"
    elif 'text' in action:
        return f"Step {step_id + 1}: {action_type} \"{action['text']}\""
    elif 'button' in action:
        return f"Step {step_id + 1}: {action_type} {action['button']}"
    return f"Step {step_id + 1}: {action_type}"


def format_semantic_summary(step, step_id):
    """Format B: Semantic-level summary using step_instruction."""
    instruction = step.get('step_instruction', '') or step.get('thought', '')
    if instruction:
        return f"Step {step_id + 1}: {instruction}"
    return format_action_summary(step, step_id)


def format_progress_summary(steps_so_far, goal, current_step_id):
    """Format C: Progress-level summary."""
    if not steps_so_far:
        return f"Goal: {goal}\nNo steps completed yet."

    completed_actions = []
    for s in steps_so_far:
        inst = s.get('step_instruction', '') or s.get('thought', '')
        if inst:
            completed_actions.append(inst)
        else:
            action = s['action_content']
            completed_actions.append(f"{action.get('action', 'unknown')}")

    completed_str = "; ".join(completed_actions[-5:])  # Last 5 actions
    return f"Goal: {goal}\nCompleted ({len(steps_so_far)} steps): {completed_str}\nContinue with the next action."


def build_summary_messages(line, step_id, summary_format, fm):
    """Build messages with summary context instead of full multi-turn history."""
    from x.data.agent.space.std_space import RAW_SPACE

    line_can_thought = fm.can_thought(line)
    _format = 'thought_action' if line_can_thought else 'only_action'
    system_prompt = MOBILE_USE.format(OUTPUT_FORMAT[_format], generate_prompt(RAW_SPACE))

    messages = [{'role': 'system', 'content': [{'text': system_prompt}]}]

    # Add summary of previous steps
    if step_id > 0 and summary_format != 'none':
        if summary_format == 'action_level':
            summaries = [format_action_summary(line['steps'][i], i) for i in range(step_id)]
            summary_text = "Previous actions:\n" + "\n".join(summaries)
        elif summary_format == 'semantic_level':
            summaries = [format_semantic_summary(line['steps'][i], i) for i in range(step_id)]
            summary_text = "Previous actions:\n" + "\n".join(summaries)
        elif summary_format == 'progress_level':
            summary_text = format_progress_summary(
                line['steps'][:step_id], line['goal'], step_id
            )
        else:
            summary_text = ""

        if summary_text:
            messages.append({
                'role': 'user',
                'content': [{'text': summary_text}]
            })
            messages.append({
                'role': 'assistant',
                'content': [{'text': 'Understood. I will continue from where we left off.'}]
            })

    # Add current step
    step = line['steps'][step_id]
    format_instruct = f"Output Format: {OUTPUT_FORMAT[_format]}"
    user_content = []

    if step_id == 0:
        user_content.append({'text': f"User Instruction: {line['goal']}\n{format_instruct}"})
    else:
        user_content.append({'text': f"User Instruction: {line['goal']}\n{format_instruct}"})

    image_ele = make_qwen_image_item(step['screenshot'], image=step.get('screenshot_pil', None))
    user_content.append(image_ele)

    messages.append({'role': 'user', 'content': user_content})

    return messages


def process_line(line, args, fm):
    global completed_count
    num_steps = len(line['steps'])
    fixed_line = fix_line(line)

    step_results = []

    for step_id in range(num_steps):
        current_check_pam = fixed_line['steps'][step_id]['check_options']

        try:
            messages = build_summary_messages(
                fixed_line, step_id, args.summary_format, fm
            )
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
        except Exception as e:
            type_match, extract_match = False, False

        step_results.append({
            'step_id': step_id,
            'success': extract_match,
            'type_match': type_match,
        })

        # Stop on error for fair comparison with baseline
        if args.stop_on_error and not extract_match:
            break

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
        'summary_format': args.summary_format,
        'steps': step_results,
        'trajectory_success': all(sr['success'] for sr in step_results) and len(step_results) == num_steps,
        'first_error_step': first_error,
        'progress_rate': first_error / num_steps if first_error is not None else 1.0,
        'scattered_progress_rate': scattered_correct / num_steps if num_steps > 0 else 0.0,
    }

    with result_lock:
        completed_count += 1
        output_path = os.path.join(args.output_dir, f"ac_summary_{args.summary_format}.jsonl")
        with open(output_path, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        if completed_count % 50 == 0:
            print(f"Progress: {completed_count} trajectories completed")

    return result


def main(args):
    from x.data.agent.space.std_space import RAW_SPACE
    fm = JsonFormat(RAW_SPACE, add_thought=True, force_add_thought=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # Clear previous results
    output_path = os.path.join(args.output_dir, f"ac_summary_{args.summary_format}.jsonl")
    if os.path.exists(output_path):
        os.remove(output_path)

    std_data = []
    with open(args.jsonl_file, 'r') as f:
        for line in f:
            std_data.append(json.loads(line))

    print(f"Loaded {len(std_data)} trajectories. Summary format: {args.summary_format}")

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_line, line, args, fm): line for line in std_data}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error: {e}")

    success_count = sum(1 for r in results if r['trajectory_success'])
    success_rate = success_count / len(results) * 100 if results else 0
    avg_progress = sum(r['progress_rate'] for r in results) / len(results) if results else 0

    summary = {
        'summary_format': args.summary_format,
        'total': len(results),
        'success_rate': success_rate,
        'avg_progress': avg_progress,
    }

    summary_path = os.path.join(args.output_dir, f"ac_summary_{args.summary_format}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults for format '{args.summary_format}':")
    print(f"  Success Rate: {success_rate:.2f}%")
    print(f"  Avg Progress: {avg_progress:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_file", type=str,
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation/dataset/android_control_evaluation_std.jsonl")
    parser.add_argument("--output_dir", type=str,
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/results/ac")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--summary_format", type=str,
                        choices=['action_level', 'semantic_level', 'progress_level'],
                        default='semantic_level')
    parser.add_argument("--n_history_image_limit", type=int, default=2)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--stop_on_error", action='store_true', default=True)
    parser.add_argument("--no_stop_on_error", action='store_true')
    parser.add_argument("--api_url", type=str, default="http://localhost:19806/v1")
    args = parser.parse_args()
    if args.no_stop_on_error:
        args.stop_on_error = False
    _set_endpoint(args.api_url)
    main(args)
