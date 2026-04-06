"""Eval: Oracle Information Ablation — Which components of oracle instruction drive the 18pp gap?

Tests what specific information in oracle step_instruction is responsible for gains:
  - type_only:    "Perform a click action." (GT action type only)
  - target_only:  "Focus on: the Settings icon" (target from instruction, no action verb)
  - type_target:  "click: the Settings icon" (action type + target, reconstructed)
  - full:         Full oracle step_instruction (existing subtask eval)

At each step:
  System: standard system prompt
  User: goal + ablated instruction + compressed history + format + screenshot
"""

import argparse
import copy
import json
import os
import re
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
from x.data.agent.json import MOBILE_USE, OUTPUT_FORMAT, generate_prompt
from x.qwen.image import make_qwen_image_item

result_lock = Lock()
fm = None


# ─── Instruction Parsing ───

ACTION_VERB_PREFIXES = [
    'click on the ', 'click on ', 'click the ', 'click ',
    'tap on the ', 'tap on ', 'tap the ', 'tap ',
    'long press on the ', 'long press on ', 'long press the ', 'long press ',
    'press and hold the ', 'press and hold on ', 'press and hold ',
    'swipe ', 'scroll ',
    'type in ', 'type ', 'enter the ', 'enter ', 'input the ', 'input ',
    'search for ', 'search ',
    'open the ', 'open ', 'launch the ', 'launch ',
    'press the ', 'press ',
    'go back to ', 'go back', 'navigate back',
    'wait for ', 'wait ',
]


def extract_target(step_instruction):
    """Extract the target noun phrase from a step instruction by stripping the action verb."""
    if not step_instruction:
        return ''
    instr = step_instruction.strip()
    instr_lower = instr.lower()

    for prefix in ACTION_VERB_PREFIXES:
        if instr_lower.startswith(prefix):
            target = instr[len(prefix):].strip()
            # Remove trailing period
            if target.endswith('.'):
                target = target[:-1].strip()
            return target if target else instr

    # Fallback: return the whole instruction (couldn't parse)
    return instr


def ablate_instruction(step_instruction, gt_action, mode):
    """Produce the ablated instruction string.

    Args:
        step_instruction: original oracle instruction (e.g. "Click on the Settings icon")
        gt_action: ground truth action dict (e.g. {'action': 'click', 'coordinate': [...]})
        mode: one of 'type_only', 'target_only', 'type_target', 'full'

    Returns:
        ablated instruction string
    """
    gt_type = gt_action.get('action', 'unknown')

    if mode == 'full':
        return step_instruction

    if mode == 'type_only':
        return f"Perform a {gt_type} action."

    target = extract_target(step_instruction)

    if mode == 'target_only':
        if target:
            return f"Focus on: {target}"
        else:
            return ''

    if mode == 'type_target':
        if target:
            return f"{gt_type}: {target}"
        else:
            return f"Perform a {gt_type} action."

    raise ValueError(f"Unknown ablation mode: {mode}")


# ─── Message Building ───

def format_action_text(pred_action):
    """Create a brief text description of an action for the summary."""
    if pred_action is None:
        return "unknown action"
    action_type = pred_action.get('action', 'unknown')
    if action_type == 'click':
        coord = pred_action.get('coordinate', [])
        return f"click at {coord}"
    elif action_type == 'long_press':
        coord = pred_action.get('coordinate', [])
        return f"long_press at {coord}"
    elif action_type == 'type':
        text = pred_action.get('text', '')
        return f"type \"{text}\""
    elif action_type == 'swipe':
        coord = pred_action.get('coordinate', [])
        direction = pred_action.get('direction', '')
        return f"swipe {direction} at {coord}"
    elif action_type == 'open':
        app = pred_action.get('app', '')
        return f"open \"{app}\""
    elif action_type == 'wait':
        return "wait"
    elif action_type == 'system_button':
        button = pred_action.get('button', '')
        return f"press system button \"{button}\""
    else:
        return f"{action_type}({json.dumps(pred_action, ensure_ascii=False)})"


def build_ablation_messages(fixed, step_id, action_history, fm_obj, ablation_mode):
    """Build single-turn messages with ablated oracle instruction + compressed history."""
    line_can_thought = fm_obj.can_thought(fixed)
    _format = 'thought_action' if line_can_thought else 'only_action'
    system_prompt = MOBILE_USE.format(OUTPUT_FORMAT[_format], generate_prompt(fm_obj.space))

    messages = [{
        'role': 'system',
        'content': [{'text': system_prompt}]
    }]

    step = fixed['steps'][step_id]
    step_instruction_raw = step.get('step_instruction', '')
    gt_action = step['action_content']

    # Ablate the instruction
    instruction = ablate_instruction(step_instruction_raw, gt_action, ablation_mode)

    user_content = []
    text_parts = []

    text_parts.append(f"Overall Task: {fixed['goal']}")

    if instruction:
        text_parts.append(f"\nCurrent Step Instruction: {instruction}")

    if action_history:
        text_parts.append(f"\nCompleted actions ({len(action_history)} step(s)):")
        for i, (action_text, thought_text) in enumerate(action_history):
            text_parts.append(f"  Step {i+1}: {action_text}")
        text_parts.append(f"\nPlease perform step {len(action_history)+1} as instructed above.")
    else:
        text_parts.append(f"\nThis is the first step. Please begin the task.")

    format_instruct = f"Output Format: {OUTPUT_FORMAT[_format]}"
    text_parts.append(f"\n{format_instruct}")

    user_content.append({'text': '\n'.join(text_parts)})

    if step_id == 0:
        user_content.append({
            'text': "If the query asks a question, please answer the question through the answer action before terminating the process.\n"
        })

    image_ele = make_qwen_image_item(
        step['screenshot'],
        image=step.get('screenshot_pil', None)
    )
    user_content.append(image_ele)

    messages.append({'role': 'user', 'content': user_content})

    return messages


# ─── Episode Processing ───

def process_episode(episode, args):
    """Process a single episode with ablated oracle instruction."""
    global fm

    fixed = fix_line(copy.deepcopy(episode))
    num_steps = len(fixed['steps'])
    action_history = []
    step_results = []

    try:
        for step_id in range(num_steps):
            current_check = fixed['steps'][step_id]['check_options']
            gt_action = fixed['steps'][step_id]['action_content']
            step_instruction = fixed['steps'][step_id].get('step_instruction', '')

            messages = build_ablation_messages(
                fixed, step_id, action_history, fm, args.ablation_mode
            )

            messages = slim_messages(
                messages=messages,
                num_image_limit=args.n_history_image_limit
            )

            _, width, height, resized_width, resized_height = find_last_image_ele(messages)

            model_response = call_mobile_agent_vllm(
                messages=messages,
                model_name=args.model_name
            )

            pred = safe_parse_response(fm, model_response)
            pred_action = pred['action_content']
            thought_text = pred.get('think', '')

            type_match, extract_match = evaluate_android_control_action(
                pred_action, current_check,
                width, height, resized_width, resized_height
            )

            action_text = format_action_text(pred_action)
            brief_thought = thought_text[:100] + '...' if len(thought_text) > 100 else thought_text
            action_history.append((action_text, brief_thought))

            # Record what instruction was actually given
            ablated_instr = ablate_instruction(step_instruction, gt_action, args.ablation_mode)

            step_results.append({
                'step_num': step_id,
                'type_match': type_match,
                'extract_match': extract_match,
                'pred_action': pred_action,
                'gt_action': gt_action,
                'gt_action_type': gt_action['action'],
                'ablation_mode': args.ablation_mode,
                'instruction_given': ablated_instr,
                'instruction_raw': step_instruction,
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
        'ablation_mode': args.ablation_mode,
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

    out_path = os.path.join(args.output_dir, 'trajectory_results.jsonl')
    if os.path.exists(out_path):
        os.remove(out_path)

    data = load_ac_trajectories(
        jsonl_path=args.jsonl_file,
        max_episodes=args.max_episodes
    )
    print(f"Loaded {len(data)} episodes. Starting ABLATION mode={args.ablation_mode} evaluation...")

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_episode, ep, args): ep for ep in data}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                if len(results) % 50 == 0:
                    metrics = compute_trajectory_metrics(results)
                    print(f"[{args.ablation_mode}] Progress: {len(results)}/{len(data)} | TSR: {metrics['tsr']:.3f} | AvgProg: {metrics['avg_progress']:.3f}")
            except Exception as e:
                print(f"Exception: {e}")

    metrics = compute_trajectory_metrics(results)

    # Step accuracy
    total_steps_eval = sum(len(r['step_results']) for r in results)
    total_correct = sum(sum(1 for s in r['step_results'] if s['extract_match']) for r in results)
    metrics['step_accuracy'] = total_correct / total_steps_eval if total_steps_eval > 0 else 0
    metrics['total_steps_evaluated'] = total_steps_eval
    metrics['total_steps_correct'] = total_correct

    # Per-length breakdown
    length_stats = {}
    for r in results:
        b = r['length_bucket']
        if b not in length_stats:
            length_stats[b] = []
        length_stats[b].append(r)

    length_metrics = {}
    for b, v in length_stats.items():
        length_metrics[b] = compute_trajectory_metrics(v)
        ts = sum(len(r['step_results']) for r in v)
        tc = sum(sum(1 for s in r['step_results'] if s['extract_match']) for r in v)
        length_metrics[b]['step_accuracy'] = tc / ts if ts > 0 else 0

    summary = {
        'model': args.model_name,
        'ablation_mode': args.ablation_mode,
        'total_episodes': len(results),
        **metrics,
        'length_bucket_stats': length_metrics,
    }

    save_json(summary, os.path.join(args.output_dir, 'summary.json'))

    print(f"\nABLATION mode={args.ablation_mode} evaluation completed.")
    print(f"TSR: {metrics['tsr']:.4f} ({metrics['success_count']}/{metrics['n']})")
    print(f"Avg Progress: {metrics['avg_progress']:.4f}")
    print(f"Step Accuracy: {metrics['step_accuracy']:.4f}")
    print(f"\nPer-length metrics:")
    for b in sorted(length_metrics.keys()):
        m = length_metrics[b]
        print(f"  {b}: TSR={m['tsr']:.3f} AvgProg={m['avg_progress']:.3f} StepAcc={m.get('step_accuracy', 0):.3f} (n={m['n']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval: Oracle Information Ablation for AndroidControl")
    parser.add_argument("--jsonl_file", type=str, default=None, help="Path to evaluation JSONL")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_oracle_ablation", help="Output directory")
    parser.add_argument("--model_name", type=str, required=True, help="Model name for vLLM")
    parser.add_argument("--ablation_mode", type=str, required=True,
                        choices=['type_only', 'target_only', 'type_target', 'full'],
                        help="Which oracle information to provide")
    parser.add_argument("--n_history_image_limit", type=int, default=2)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--max_episodes", type=int, default=None, help="Limit episodes for testing")
    args = parser.parse_args()
    main(args)
