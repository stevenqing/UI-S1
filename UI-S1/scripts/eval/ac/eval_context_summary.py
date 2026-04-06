"""Eval: Context Summary — Replace full multi-turn history with compressed text summary.

Tests whether compressed action summaries can replace growing multi-turn context,
reducing error propagation while retaining useful history information.

At each step:
  System: standard system prompt
  User: goal + compressed text summary of previous actions + format instruction + screenshot

No multi-turn conversation — bounded context growth.
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

# Import prompt templates from the format module
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
from x.data.agent.json import MOBILE_USE, OUTPUT_FORMAT, generate_prompt
from x.qwen.image import make_qwen_image_item

result_lock = Lock()
fm = None


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


def build_summary_messages(fixed, step_id, action_history, fm_obj):
    """Build single-turn messages with compressed summary context.

    Args:
        fixed: episode dict with steps
        step_id: current step index
        action_history: list of (action_text, thought_text) for previous steps
        fm_obj: JsonFormat instance

    Returns:
        messages list in standard format
    """
    line_can_thought = fm_obj.can_thought(fixed)
    _format = 'thought_action' if line_can_thought else 'only_action'
    system_prompt = MOBILE_USE.format(OUTPUT_FORMAT[_format], generate_prompt(fm_obj.space))

    messages = [{
        'role': 'system',
        'content': [{'text': system_prompt}]
    }]

    # Build user message
    user_content = []

    # Goal
    text_parts = [f"User Instruction: {fixed['goal']}"]

    # Compressed action history
    if action_history:
        text_parts.append("\nCompleted actions:")
        for i, (action_text, thought_text) in enumerate(action_history):
            if thought_text:
                text_parts.append(f"  Step {i+1}: [Thought: {thought_text}] {action_text}")
            else:
                text_parts.append(f"  Step {i+1}: {action_text}")
        text_parts.append(f"\nYou have completed {len(action_history)} step(s). Please perform the next action.")

    # Format instruction
    format_instruct = f"Output Format: {OUTPUT_FORMAT[_format]}"
    text_parts.append(f"\n{format_instruct}")

    user_content.append({'text': '\n'.join(text_parts)})

    if step_id == 0:
        user_content.append({
            'text': "If the query asks a question, please answer the question through the answer action before terminating the process.\n"
        })

    # Current screenshot
    step = fixed['steps'][step_id]
    image_ele = make_qwen_image_item(
        step['screenshot'],
        image=step.get('screenshot_pil', None)
    )
    user_content.append(image_ele)

    messages.append({'role': 'user', 'content': user_content})

    return messages


def process_episode(episode, args):
    """Process a single episode with summary-based context management."""
    global fm

    fixed = fix_line(copy.deepcopy(episode))
    num_steps = len(fixed['steps'])
    action_history = []  # List of (action_text, thought_text)
    step_results = []

    try:
        for step_id in range(num_steps):
            current_check = fixed['steps'][step_id]['check_options']
            gt_action = fixed['steps'][step_id]['action_content']

            # Build fresh single-turn messages with summary
            messages = build_summary_messages(fixed, step_id, action_history, fm)

            # Slim for image limit (though we only have 1 image per message)
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

            # Record action for future summaries
            action_text = format_action_text(pred_action)
            # Truncate thought to keep summary compact
            brief_thought = thought_text[:100] + '...' if len(thought_text) > 100 else thought_text
            action_history.append((action_text, brief_thought))

            step_results.append({
                'step_num': step_id,
                'type_match': type_match,
                'extract_match': extract_match,
                'pred_action': pred_action,
                'gt_action': gt_action,
                'gt_action_type': gt_action['action'],
                'context_mode': 'summary',
                'summary_length': len(action_history) - 1,
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
        'context_mode': 'summary',
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
    print(f"Loaded {len(data)} episodes. Starting SUMMARY context evaluation...")

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

    # Compute summary metrics
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
        'context_mode': 'summary',
        'total_episodes': len(results),
        **metrics,
        'action_type_stats': action_stats,
        'length_bucket_stats': length_metrics,
    }

    save_json(summary, os.path.join(args.output_dir, 'summary.json'))

    print(f"\nSUMMARY context evaluation completed.")
    print(f"TSR: {metrics['tsr']:.4f} ({metrics['success_count']}/{metrics['n']})")
    print(f"Avg Progress: {metrics['avg_progress']:.4f}")
    print(f"Scattered Progress: {metrics['scattered_progress']:.4f}")
    print(f"\nPer-length metrics:")
    for b in sorted(length_metrics.keys()):
        m = length_metrics[b]
        print(f"  {b}: TSR={m['tsr']:.3f} AvgProg={m['avg_progress']:.3f} (n={m['n']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval: Summary Context Management for AndroidControl")
    parser.add_argument("--jsonl_file", type=str, default=None, help="Path to evaluation JSONL")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_context_summary", help="Output directory")
    parser.add_argument("--model_name", type=str, required=True, help="Model name for vLLM")
    parser.add_argument("--n_history_image_limit", type=int, default=2)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--max_episodes", type=int, default=None, help="Limit episodes for testing")
    args = parser.parse_args()
    main(args)
