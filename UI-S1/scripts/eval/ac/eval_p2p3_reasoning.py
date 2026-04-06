"""P2+P3: Reasoning Injection Experiments.

P2: Oracle Reasoning Injection - does giving correct reasoning improve action accuracy?
P3: Targeted Reasoning Prompts - do action-error steps and grounding-error steps
    benefit from different reasoning prompts?

Uses existing C4+C7 data for step selection, AC model for inference.
"""

import argparse
import copy
import json
import math
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

PROJECT_ROOT = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts/eval/ac"))

from ac_utils import (
    load_ac_trajectories, fix_line, init_format, save_jsonl,
    evaluate_android_control_action, find_last_image_ele, slim_messages,
    call_mobile_agent_vllm, safe_parse_response, _json_default,
)
from evaluation.qwenvl_utils import END_POINT, image_to_data_url, message_translate
from openai import OpenAI
from PIL import Image

result_lock = Lock()
fm = None

# ── Reasoning Prompt Templates ──

# P2: Oracle reasoning (inject GT action justification)
ORACLE_REASONING_TEMPLATE = """Before acting, consider:
- The correct action type for this step is: {gt_action_type}
- The target element is at approximately: {gt_target_description}
Now output your action based on this guidance."""

# P3: Action-focused reasoning prompt
PROMPT_A_ACTION = """Before acting, think step by step:
1. Look at the current screenshot carefully.
2. What are ALL possible action types you could take? (click, type, swipe, wait, open, system_button, long_press)
3. For EACH possible action type, explain why it might or might not be appropriate.
4. Choose the action type with the strongest justification.
Then output your action."""

# P3: Grounding-focused reasoning prompt
PROMPT_B_GROUNDING = """Before acting, think step by step:
1. Look at the current screenshot carefully.
2. Identify the SPECIFIC UI element you need to interact with.
3. Describe its visual appearance: what text does it show? What color/shape is it?
4. Describe its EXACT position: which part of the screen? Near which other elements?
5. Make sure you are targeting the correct element, not a similar nearby element.
Then output your action."""

# P3: Combined reasoning prompt (both action + grounding)
PROMPT_C_COMBINED = """Before acting, think step by step:
1. What TYPE of action is needed? Consider all options carefully.
2. Which SPECIFIC element should you target? Describe it precisely.
Then output your action."""


def call_vllm_greedy(messages, model_name):
    """Call vLLM with greedy decoding."""
    messages_oai, screenshot_list = message_translate(messages, to_format='openai')
    screenshot_ptr = 0
    for msg in messages_oai:
        for content in msg['content']:
            if 'image_url' in content:
                url = image_to_data_url(Image.open(screenshot_list[screenshot_ptr]))
                content['image_url']['url'] = url
                screenshot_ptr += 1
    assert screenshot_ptr == len(screenshot_list)

    for attempt in range(3):
        try:
            bot = OpenAI(api_key="EMPTY", base_url=END_POINT, timeout=600)
            resp = bot.chat.completions.create(
                model=model_name,
                messages=messages_oai,
                temperature=0.0,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < 2:
                time.sleep(5)
            else:
                traceback.print_exc()
    return ""


def inject_reasoning_into_messages(messages, reasoning_text):
    """Inject reasoning instruction into the last user message."""
    messages = copy.deepcopy(messages)
    # Find the last user message and append reasoning instruction
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]['role'] == 'user':
            messages[i]['content'].append({"text": reasoning_text})
            break
    return messages


def get_gt_target_description(step, check_options):
    """Create a human-readable description of the GT target."""
    gt_action = step.get('action_content', {})
    action_type = gt_action.get('action', 'unknown')

    if action_type in ('click', 'long_press'):
        check = check_options
        if 'coordinate_check' in check:
            cc = check['coordinate_check']
            if 'click_point' in cc:
                return f"coordinates [{cc['click_point'][0]}, {cc['click_point'][1]}]"
            if 'bounding_box' in cc:
                bb = cc['bounding_box']
                cx = (bb[0] + bb[2]) / 2
                cy = (bb[1] + bb[3]) / 2
                return f"approximately [{cx:.0f}, {cy:.0f}]"
        return "a specific UI element"
    elif action_type == 'type':
        text = gt_action.get('text', '')
        return f"type text: '{text[:50]}'"
    elif action_type == 'swipe':
        direction = gt_action.get('direction', 'unknown')
        return f"swipe {direction}"
    else:
        return action_type


def process_step(episode, step_id, args, conditions):
    """Process a single step under multiple reasoning conditions."""
    global fm

    fixed = fix_line(copy.deepcopy(episode))
    if step_id >= len(fixed['steps']):
        return None

    step = fixed['steps'][step_id]
    check_options = step['check_options']
    gt_action = step['action_content']

    # Build messages up to this step using GT actions
    state = None
    for si in range(step_id + 1):
        state = fm.gen_next_round(fixed, state, previous_model_response=None)
        if state is None:
            return None

    messages = slim_messages(
        messages=state['messages'],
        num_image_limit=args.n_history_image_limit
    )
    _, width, height, resized_width, resized_height = find_last_image_ele(messages)

    results = {}

    for cond_name, reasoning_text in conditions.items():
        if reasoning_text is None:
            # Baseline: no injection
            msgs = messages
        else:
            msgs = inject_reasoning_into_messages(messages, reasoning_text)

        try:
            response = call_vllm_greedy(msgs, args.model_name)
            pred = safe_parse_response(fm, response)
            pred_action = pred['action_content']
            type_match, extract_match = evaluate_android_control_action(
                pred_action, check_options,
                width, height, resized_width, resized_height
            )
            results[cond_name] = {
                'pred_action_type': pred_action.get('action', 'unknown'),
                'type_match': type_match,
                'extract_match': extract_match,
                'response': response[:500],
            }
        except Exception as e:
            results[cond_name] = {
                'pred_action_type': None,
                'type_match': False,
                'extract_match': False,
                'error': str(e),
            }

    return results


def process_step_wrapper(task, args, conditions):
    """Wrapper for ThreadPoolExecutor."""
    episode, step_id, step_meta = task
    results = process_step(episode, step_id, args, conditions)
    if results is None:
        return None

    return {
        'episode_id': episode.get('episode_id'),
        'step_id': step_id,
        'gt_action_type': step_meta['gt_action_type'],
        'error_type': step_meta['error_type'],
        'conditions': results,
    }


def main():
    global fm

    parser = argparse.ArgumentParser(description="P2+P3: Reasoning Injection Experiments")
    parser.add_argument("--jsonl_file", type=str,
                        default=os.path.join(PROJECT_ROOT, "datasets/android_control_evaluation_std.jsonl"))
    parser.add_argument("--c4c7_file", type=str,
                        default=os.path.join(PROJECT_ROOT, "outputs/eval_c4c7_ac/Qwen2.5-VL-7B/multisample_results.jsonl"))
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "outputs/eval_p2p3"))
    parser.add_argument("--model_name", type=str, default="Qwen2.5-VL-7B")
    parser.add_argument("--n_history_image_limit", type=int, default=2)
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max steps per error type to evaluate (for speed)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "p2p3_results.jsonl")

    # Initialize format manager
    fm = init_format()

    # Load C4+C7 results to identify error types per step
    print("Loading C4+C7 results for error type classification...")
    c4c7_step_info = {}  # (episode_id, step_num) -> {error_type, gt_action_type}
    with open(args.c4c7_file) as f:
        for line in f:
            try:
                ep = json.loads(line)
                episode_id = ep['episode_id']
                for step_data in ep.get('step_samples', []):
                    step_num = step_data['step_num']
                    samples = step_data.get('samples', [])
                    if not samples:
                        continue
                    greedy = samples[0]
                    if greedy.get('type_match') and greedy.get('extract_match'):
                        error_type = 'correct'
                    elif not greedy.get('type_match'):
                        error_type = 'action_error'
                    else:
                        error_type = 'grounding_error'

                    c4c7_step_info[(episode_id, step_num)] = {
                        'error_type': error_type,
                        'gt_action_type': step_data.get('gt_action_type', 'unknown'),
                    }
            except Exception:
                continue

    print(f"Loaded {len(c4c7_step_info)} step classifications")
    error_counts = Counter(v['error_type'] for v in c4c7_step_info.values())
    print(f"Error distribution: {dict(error_counts)}")

    # Load episodes
    print("Loading episodes...")
    episodes = load_ac_trajectories(jsonl_path=args.jsonl_file)
    ep_map = {ep.get('episode_id'): ep for ep in episodes}
    print(f"Loaded {len(episodes)} episodes")

    # Select steps: stratified sample from each error type
    tasks = []
    by_error = defaultdict(list)
    for (ep_id, step_num), meta in c4c7_step_info.items():
        if ep_id in ep_map:
            by_error[meta['error_type']].append((ep_id, step_num, meta))

    for error_type, step_list in by_error.items():
        if args.max_steps:
            step_list = step_list[:args.max_steps]
        for ep_id, step_num, meta in step_list:
            tasks.append((ep_map[ep_id], step_num, meta))

    print(f"Total steps to evaluate: {len(tasks)}")
    for et, sl in by_error.items():
        n = min(len(sl), args.max_steps or len(sl))
        print(f"  {et}: {n} steps")

    # Resume support
    completed = set()
    if args.resume and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    completed.add((r['episode_id'], r['step_id']))
                except:
                    pass
        print(f"Resuming: {len(completed)} steps already done")
        tasks = [t for t in tasks if (t[0].get('episode_id'), t[1]) not in completed]
        print(f"Remaining: {len(tasks)} steps")
    elif os.path.exists(out_path):
        os.remove(out_path)

    if not tasks:
        print("No steps to process!")
        return

    # Define conditions
    conditions = {
        'baseline': None,  # No injection
        'prompt_A_action': PROMPT_A_ACTION,
        'prompt_B_grounding': PROMPT_B_GROUNDING,
        'prompt_C_combined': PROMPT_C_COMBINED,
    }

    # For oracle (P2), we need per-step oracle text - handle in wrapper
    # For now, skip oracle since it needs per-step GT info that's complex to pass
    # TODO: Add oracle condition with per-step GT target description

    print(f"Running {len(conditions)} conditions per step...")
    print(f"Total API calls: {len(tasks) * len(conditions)}")

    done = len(completed)
    total = done + len(tasks)
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_step_wrapper, task, args, conditions): task
            for task in tasks
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                if result is None:
                    continue
                done += 1

                with result_lock:
                    with open(out_path, 'a') as f:
                        f.write(json.dumps(result, ensure_ascii=False, default=str) + '\n')

                if done % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = (done - len(completed)) / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    print(f"Progress: {done}/{total} ({rate:.1f}/s, ETA: {eta/60:.0f}m)")
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()

    # ── Compute summary ──
    print("\nComputing summary...")
    all_results = []
    with open(out_path) as f:
        for line in f:
            try:
                all_results.append(json.loads(line))
            except:
                pass

    summary = {
        'total_steps': len(all_results),
        'conditions': list(conditions.keys()),
        'by_error_type': {},
        'by_condition': {},
        'cross_effect': {},  # P3: condition × error_type interaction
    }

    # Per condition overall
    for cond in conditions:
        n = sum(1 for r in all_results if cond in r.get('conditions', {}))
        func_match = sum(1 for r in all_results
                         if r.get('conditions', {}).get(cond, {}).get('type_match', False))
        args_match = sum(1 for r in all_results
                         if r.get('conditions', {}).get(cond, {}).get('extract_match', False))
        summary['by_condition'][cond] = {
            'n': n,
            'function_match': func_match / n if n > 0 else 0,
            'args_match': args_match / n if n > 0 else 0,
        }

    # Per error_type × condition (P3 cross-effect)
    for error_type in ['correct', 'action_error', 'grounding_error']:
        subset = [r for r in all_results if r.get('error_type') == error_type]
        n = len(subset)
        summary['by_error_type'][error_type] = {'n': n}

        for cond in conditions:
            func_match = sum(1 for r in subset
                             if r.get('conditions', {}).get(cond, {}).get('type_match', False))
            full_match = sum(1 for r in subset
                             if r.get('conditions', {}).get(cond, {}).get('type_match', False)
                             and r.get('conditions', {}).get(cond, {}).get('extract_match', False))
            summary['cross_effect'][f'{error_type}__{cond}'] = {
                'n': n,
                'function_match': func_match / n if n > 0 else 0,
                'full_match': full_match / n if n > 0 else 0,
            }

    summary_path = os.path.join(args.output_dir, "p2p3_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*80}")
    print("P2+P3 RESULTS")
    print(f"{'='*80}")
    print(f"Total steps: {len(all_results)}")

    print(f"\n--- Overall by condition ---")
    for cond, stats in summary['by_condition'].items():
        print(f"  {cond:25s}: func={stats['function_match']:.4f}  args={stats['args_match']:.4f}  (N={stats['n']})")

    print(f"\n--- P3: Cross-effect (error_type × condition) ---")
    print(f"{'':25s} {'baseline':>12s} {'A_action':>12s} {'B_grounding':>12s} {'C_combined':>12s}")
    for error_type in ['action_error', 'grounding_error', 'correct']:
        row = f"  {error_type:23s}"
        for cond in ['baseline', 'prompt_A_action', 'prompt_B_grounding', 'prompt_C_combined']:
            key = f'{error_type}__{cond}'
            if key in summary['cross_effect']:
                val = summary['cross_effect'][key]['function_match']
                row += f" {val:>11.4f}"
            else:
                row += f" {'N/A':>11s}"
        print(row)

    print(f"\n--- P3 Key Test: Does prompt_A help action_error more than prompt_B? ---")
    ae_A = summary['cross_effect'].get('action_error__prompt_A_action', {}).get('function_match', 0)
    ae_B = summary['cross_effect'].get('action_error__prompt_B_grounding', {}).get('function_match', 0)
    ge_A = summary['cross_effect'].get('grounding_error__prompt_A_action', {}).get('full_match', 0)
    ge_B = summary['cross_effect'].get('grounding_error__prompt_B_grounding', {}).get('full_match', 0)
    print(f"  Action errors:    prompt_A={ae_A:.4f}  prompt_B={ae_B:.4f}  Δ={ae_A-ae_B:+.4f}")
    print(f"  Grounding errors: prompt_A={ge_A:.4f}  prompt_B={ge_B:.4f}  Δ={ge_B-ge_A:+.4f}")

    if ae_A > ae_B and ge_B > ge_A:
        print("  → CROSS-EFFECT CONFIRMED: Different error types benefit from different reasoning!")
    elif ae_A > ae_B:
        print("  → Partial: Action prompt helps action errors, but grounding prompt doesn't help grounding errors")
    elif ge_B > ge_A:
        print("  → Partial: Grounding prompt helps grounding errors, but action prompt doesn't help action errors")
    else:
        print("  → No cross-effect detected")

    print(f"\n{'='*80}")
    print(f"Results: {out_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
