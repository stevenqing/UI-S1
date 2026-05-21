#!/usr/bin/env python3
"""V12 AC trajectory evaluation via vLLM API.

Supports both stop-on-error and no-stop modes. Reports TSR, AvgProgress,
and scattered progress (no-stop only).

Usage:
    python v12_general/eval_ac_vllm.py \
        --model_name v12_sft_ep4 \
        --jsonl_file datasets/android_control_evaluation_std.jsonl \
        --output_dir outputs/v12_sft_ep4_ac \
        --max_workers 128 \
        --no_stop
"""

import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'evaluation')))

from x.data.agent.json import JsonFormat
from x.data.agent.space.std_space import RAW_SPACE
from x.qwen.data_format import slim_messages
from qwenvl_utils import (
    call_mobile_agent_vllm,
    evaluate_android_control_action,
    find_last_image_ele,
)

fm = None
result_lock = Lock()
progress_counter = [0]  # mutable for thread access


def fix_line(line):
    for step in line['steps']:
        if 'check_options' in step:
            continue
        check_options = copy.deepcopy(step['action_content'])
        if 'bbox' in step:
            check_options['candidate_bbox'] = step['bbox']
        elif 'candidate_bbox' not in check_options:
            check_options['candidate_bbox'] = []
        step['check_options'] = check_options
    return line


def process_episode(line, args):
    global fm

    num_steps = len(line['steps'])
    state = None
    model_response = None
    step_results = []
    fixed_line = fix_line(copy.deepcopy(line))

    try:
        for step_id in range(num_steps):
            current_check = fixed_line['steps'][step_id]['check_options']
            gt_action = fixed_line['steps'][step_id]['action_content']

            state = fm.gen_next_round(fixed_line, state,
                                       previous_model_response=model_response)
            if state is None:
                break

            messages = slim_messages(
                messages=state['messages'],
                num_image_limit=args.n_history_image_limit,
            )

            _, width, height, resized_width, resized_height = \
                find_last_image_ele(messages)

            try:
                model_response = call_mobile_agent_vllm(
                    messages=messages,
                    model_name=args.model_name,
                )
            except Exception as e:
                model_response = ''

            try:
                pred = fm.parse_response(model_response)
                pred_action = pred['action_content']
            except Exception:
                pred_action = {'action': 'wait', 'time': 0.1}

            type_match, extract_match = evaluate_android_control_action(
                pred_action, current_check,
                width, height, resized_width, resized_height,
                ignore_actions=[],
            )

            step_results.append({
                'step_num': step_id,
                'type_match': bool(type_match),
                'extract_match': bool(extract_match),
                'gt_action_type': gt_action.get('action', 'unknown'),
            })

            if not extract_match and not args.no_stop:
                break

    except Exception as e:
        pass

    # Compute metrics
    correct_steps = sum(1 for s in step_results if s['extract_match'])

    if args.no_stop:
        # Scattered progress: correct_steps / total
        progress = correct_steps / num_steps if num_steps > 0 else 0
        # Sequential progress: consecutive correct from start
        seq_correct = 0
        for s in step_results:
            if s['extract_match']:
                seq_correct += 1
            else:
                break
    else:
        seq_correct = 0
        for s in step_results:
            if s['extract_match']:
                seq_correct += 1
            else:
                break
        progress = seq_correct / num_steps if num_steps > 0 else 0

    task_success = (correct_steps == num_steps and len(step_results) == num_steps)

    result = {
        'goal': line['goal'],
        'num_steps': num_steps,
        'task_success': task_success,
        'final_step_id': seq_correct,
        'correct_steps': correct_steps,
        'evaluated_steps': len(step_results),
        'progress': progress,
        'step_results': step_results,
    }

    # Thread-safe write
    with result_lock:
        result_path = os.path.join(args.output_dir,
                                    f"{args.model_name}_{args.mode_tag}.jsonl")
        with open(result_path, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        progress_counter[0] += 1
        n = progress_counter[0]
        if n % 100 == 0 or n == args._total:
            print(f"  [{args.mode_tag}] {n}/{args._total} episodes processed")

    return result


def main(args):
    global fm
    fm = JsonFormat(RAW_SPACE, add_thought=True, force_add_thought=True)

    args.mode_tag = "nostop" if args.no_stop else "stop"
    os.makedirs(args.output_dir, exist_ok=True)

    # Clear output file
    result_path = os.path.join(args.output_dir,
                                f"{args.model_name}_{args.mode_tag}.jsonl")
    if os.path.exists(result_path):
        os.remove(result_path)

    # Load data
    data = []
    with open(args.jsonl_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    args._total = len(data)
    progress_counter[0] = 0

    print(f"[{args.mode_tag}] {len(data)} episodes, "
          f"model={args.model_name}, workers={args.max_workers}")

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_episode, line, args): line
                   for line in data}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Exception: {e}")

    elapsed = time.time() - t0
    n = len(results)
    success = sum(1 for r in results if r['task_success'])
    tsr = success / n if n > 0 else 0
    avg_progress = sum(r['progress'] for r in results) / n if n > 0 else 0
    seq_progress = sum(r['final_step_id'] / r['num_steps']
                       for r in results) / n if n > 0 else 0

    if args.no_stop:
        scattered = sum(r['correct_steps'] / r['num_steps']
                        for r in results) / n if n > 0 else 0
    else:
        scattered = seq_progress

    # Per-action stats
    action_stats = {}
    for r in results:
        for s in r['step_results']:
            at = s['gt_action_type']
            action_stats.setdefault(at, {'total': 0, 'type_match': 0, 'extract_match': 0})
            action_stats[at]['total'] += 1
            action_stats[at]['type_match'] += int(s['type_match'])
            action_stats[at]['extract_match'] += int(s['extract_match'])

    summary = {
        'model_name': args.model_name,
        'mode': args.mode_tag,
        'total_episodes': n,
        'tsr': tsr,
        'success_count': success,
        'avg_sequential_progress': seq_progress,
        'avg_scattered_progress': scattered,
        'action_type_stats': {
            k: {**v, 'type_acc': v['type_match']/v['total'] if v['total'] else 0,
                 'extract_acc': v['extract_match']/v['total'] if v['total'] else 0}
            for k, v in action_stats.items()
        },
        'elapsed_seconds': elapsed,
    }

    summary_path = os.path.join(args.output_dir,
                                 f"{args.model_name}_{args.mode_tag}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"[{args.mode_tag}] DONE in {elapsed:.0f}s")
    print(f"  TSR:              {tsr:.4f} ({success}/{n})")
    print(f"  SeqProgress:      {seq_progress:.4f}")
    print(f"  ScatteredProgress: {scattered:.4f}")
    for at, st in sorted(action_stats.items(), key=lambda x: -x[1]['total']):
        ea = st['extract_match'] / st['total'] if st['total'] else 0
        print(f"  {at:15s}: extract_acc={ea:.3f} (n={st['total']})")
    print(f"  Results: {result_path}")
    print(f"  Summary: {summary_path}")
    print(f"{'='*50}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--jsonl_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--n_history_image_limit', type=int, default=2)
    parser.add_argument('--max_workers', type=int, default=128)
    parser.add_argument('--no_stop', action='store_true',
                        help='Continue after errors (evaluate all steps)')
    args = parser.parse_args()
    main(args)
