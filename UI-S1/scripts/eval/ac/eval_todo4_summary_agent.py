"""Eval TODO 4: Summary-Augmented Agent Evaluation.

Tests whether injecting an explicit text summary of progress into the agent's
prompt improves per-step accuracy, especially for long-horizon tasks.

Conditions:
  A) baseline: Standard prompt (goal + screenshots + GT action history)
  B) oracle_summary: Standard prompt + GT-action-based progress summary injected
  C) vlm_summary: Standard prompt + VLM-generated state summary per step

Each step is evaluated independently with GT screenshots (C4+C7 style).
"""

import argparse
import copy
import json
import os
import re
import sys
import traceback
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from ac_utils import (
    load_ac_trajectories, fix_line, init_format, save_jsonl, save_json,
    evaluate_android_control_action, find_last_image_ele, slim_messages,
    call_mobile_agent_vllm, safe_parse_response, _json_default,
)
from evaluation.qwenvl_utils import END_POINT, image_to_data_url, message_translate
from openai import OpenAI
from PIL import Image

result_lock = Lock()
fm = None


# ─────────────────────────────────────────────
# Oracle summary generation (from GT actions)
# ─────────────────────────────────────────────

def _describe_gt_action(action_content):
    """Convert a GT action dict into a human-readable description."""
    action_type = action_content['action']
    if action_type == 'click':
        coord = action_content.get('coordinate', [0, 0])
        return f"Clicked at position ({coord[0]}, {coord[1]})"
    elif action_type == 'long_press':
        coord = action_content.get('coordinate', [0, 0])
        t = action_content.get('time', 2)
        return f"Long pressed at ({coord[0]}, {coord[1]}) for {t}s"
    elif action_type == 'swipe':
        c1 = action_content.get('coordinate', [0, 0])
        c2 = action_content.get('coordinate2', [0, 0])
        return f"Swiped from ({c1[0]}, {c1[1]}) to ({c2[0]}, {c2[1]})"
    elif action_type == 'type':
        text = action_content.get('text', '')
        return f'Typed "{text}"'
    elif action_type == 'open':
        text = action_content.get('text', '')
        return f'Opened app "{text}"'
    elif action_type == 'system_button':
        button = action_content.get('button', '')
        return f"Pressed system button: {button}"
    elif action_type == 'wait':
        t = action_content.get('time', 2)
        return f"Waited for {t} seconds"
    elif action_type == 'terminate':
        status = action_content.get('status', '')
        return f"Terminated with status: {status}"
    else:
        return f"Performed {action_type}"


def build_oracle_summary(episode, step_id):
    """Build an oracle progress summary from GT actions up to step_id.

    Returns a text summary string describing what has been done so far.
    """
    if step_id == 0:
        return ""  # No history at step 0

    num_steps = len(episode['steps'])
    lines = []
    lines.append(f"Progress Summary (step {step_id + 1} of {num_steps}):")
    lines.append("Actions completed so far:")

    for i in range(step_id):
        action = episode['steps'][i]['action_content']
        desc = _describe_gt_action(action)
        lines.append(f"  Step {i + 1}: [{action['action']}] {desc}")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# VLM summary generation
# ─────────────────────────────────────────────

VLM_SUMMARY_PROMPT = """You are observing a mobile phone screen. The user's task goal is: {goal}

So far, {n_done} of {n_total} steps have been completed. Look at the current screenshot and summarize:
1. What has been accomplished so far (based on the visible screen state)
2. What remains to be done to complete the goal

Be concise (2-3 sentences max). Focus on what's visible on screen and what the next logical action should be."""


def generate_vlm_summary(screenshot_path, goal, step_id, num_steps, model_name):
    """Use VLM to generate a state summary from the current screenshot.

    Returns a text summary string.
    """
    prompt = VLM_SUMMARY_PROMPT.format(
        goal=goal,
        n_done=step_id,
        n_total=num_steps,
    )

    img = Image.open(screenshot_path)
    data_url = image_to_data_url(img)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": prompt},
        ]
    }]

    for attempt in range(3):
        try:
            bot = OpenAI(api_key="EMPTY", base_url=END_POINT, timeout=120)
            resp = bot.chat.completions.create(
                model=model_name,
                messages=messages,
                extra_body={"top_k": 1},
                max_tokens=256,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
    return ""


# ─────────────────────────────────────────────
# Inject summary into messages
# ─────────────────────────────────────────────

def inject_summary_into_messages(messages, summary_text):
    """Inject a summary text block into the last user message of the conversation.

    Adds the summary BEFORE the screenshot in the last user message.
    """
    if not summary_text:
        return messages

    messages = copy.deepcopy(messages)

    # Find the last user message
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]['role'] == 'user':
            # Insert summary text at the beginning of user content
            content = messages[i]['content']
            if isinstance(content, list):
                # Insert before the first element
                summary_block = {"text": f"\n{summary_text}\n"}
                content.insert(0, summary_block)
            break

    return messages


# ─────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────

def process_episode(episode, args):
    """Process one episode under all conditions."""
    global fm

    fixed = fix_line(copy.deepcopy(episode))
    num_steps = len(fixed['steps'])
    goal = episode['goal']

    # ─── Condition A: baseline ───
    results_baseline = []
    state_a = None
    for step_id in range(num_steps):
        current_check = fixed['steps'][step_id]['check_options']
        gt_action = fixed['steps'][step_id]['action_content']

        state_a = fm.gen_next_round(fixed, state_a, previous_model_response=None)
        if state_a is None:
            break

        messages = slim_messages(
            messages=state_a['messages'],
            num_image_limit=args.n_history_image_limit,
        )
        _, width, height, resized_width, resized_height = find_last_image_ele(messages)

        try:
            response = call_mobile_agent_vllm(messages=messages, model_name=args.model_name)
            pred = safe_parse_response(fm, response)
            pred_action = pred['action_content']
            type_match, extract_match = evaluate_android_control_action(
                pred_action, current_check, width, height, resized_width, resized_height,
            )
        except Exception:
            pred_action = None
            type_match = False
            extract_match = False

        results_baseline.append({
            'step_num': step_id,
            'type_match': type_match,
            'extract_match': extract_match,
            'pred_action': pred_action,
            'gt_action': gt_action,
            'gt_action_type': gt_action['action'],
        })

    # ─── Condition B: oracle_summary ───
    results_oracle = []
    state_b = None
    for step_id in range(num_steps):
        current_check = fixed['steps'][step_id]['check_options']
        gt_action = fixed['steps'][step_id]['action_content']

        state_b = fm.gen_next_round(fixed, state_b, previous_model_response=None)
        if state_b is None:
            break

        messages = slim_messages(
            messages=state_b['messages'],
            num_image_limit=args.n_history_image_limit,
        )
        _, width, height, resized_width, resized_height = find_last_image_ele(messages)

        # Inject oracle summary
        summary = build_oracle_summary(fixed, step_id)
        if summary:
            messages = inject_summary_into_messages(messages, summary)

        try:
            response = call_mobile_agent_vllm(messages=messages, model_name=args.model_name)
            pred = safe_parse_response(fm, response)
            pred_action = pred['action_content']
            type_match, extract_match = evaluate_android_control_action(
                pred_action, current_check, width, height, resized_width, resized_height,
            )
        except Exception:
            pred_action = None
            type_match = False
            extract_match = False

        results_oracle.append({
            'step_num': step_id,
            'type_match': type_match,
            'extract_match': extract_match,
            'pred_action': pred_action,
            'gt_action': gt_action,
            'gt_action_type': gt_action['action'],
            'summary': summary[:300],
        })

    # ─── Condition C: vlm_summary ───
    results_vlm = []
    state_c = None
    for step_id in range(num_steps):
        current_check = fixed['steps'][step_id]['check_options']
        gt_action = fixed['steps'][step_id]['action_content']

        state_c = fm.gen_next_round(fixed, state_c, previous_model_response=None)
        if state_c is None:
            break

        messages = slim_messages(
            messages=state_c['messages'],
            num_image_limit=args.n_history_image_limit,
        )
        screenshot_path, width, height, resized_width, resized_height = find_last_image_ele(messages)

        # Generate VLM summary (skip step 0 — no history to summarize)
        vlm_summary = ""
        if step_id > 0:
            vlm_summary = generate_vlm_summary(
                screenshot_path, goal, step_id, num_steps, args.model_name,
            )
        if vlm_summary:
            messages = inject_summary_into_messages(messages, f"State Summary: {vlm_summary}")

        try:
            response = call_mobile_agent_vllm(messages=messages, model_name=args.model_name)
            pred = safe_parse_response(fm, response)
            pred_action = pred['action_content']
            type_match, extract_match = evaluate_android_control_action(
                pred_action, current_check, width, height, resized_width, resized_height,
            )
        except Exception:
            pred_action = None
            type_match = False
            extract_match = False

        results_vlm.append({
            'step_num': step_id,
            'type_match': type_match,
            'extract_match': extract_match,
            'pred_action': pred_action,
            'gt_action': gt_action,
            'gt_action_type': gt_action['action'],
            'vlm_summary': vlm_summary[:300],
        })

    # ─── Output ───
    output = {
        'episode_id': episode.get('episode_id'),
        'goal': goal,
        'num_steps': num_steps,
        'results_baseline': results_baseline,
        'results_oracle_summary': results_oracle,
        'results_vlm_summary': results_vlm,
    }

    with result_lock:
        out_path = os.path.join(args.output_dir, 'summary_eval_results.jsonl')
        with open(out_path, 'a') as f:
            f.write(json.dumps(output, ensure_ascii=False, default=_json_default) + '\n')

    return output


# ─────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────

def analyze_results(results, output_dir):
    """Analyze and compare the three conditions."""
    conditions = ['baseline', 'oracle_summary', 'vlm_summary']
    cond_keys = {
        'baseline': 'results_baseline',
        'oracle_summary': 'results_oracle_summary',
        'vlm_summary': 'results_vlm_summary',
    }

    print("\n" + "=" * 80)
    print("RESULTS: Summary-Augmented Agent Evaluation (TODO 4)")
    print("=" * 80)

    # 1. Overall accuracy
    print("\n--- Overall Per-Step Accuracy ---")
    for cond in conditions:
        key = cond_keys[cond]
        type_c = sum(s['type_match'] for r in results for s in r[key])
        ext_c = sum(s['extract_match'] for r in results for s in r[key])
        total = sum(len(r[key]) for r in results)
        print(f"  {cond:20s}: type_match={type_c}/{total} ({type_c / total * 100:.1f}%)  "
              f"extract_match={ext_c}/{total} ({ext_c / total * 100:.1f}%)")

    # 2. By step position (does summary help more for later steps?)
    print("\n--- Accuracy by Step Position ---")
    for cond in conditions:
        key = cond_keys[cond]
        by_step = defaultdict(lambda: {'correct': 0, 'total': 0})
        for r in results:
            for s in r[key]:
                bucket = 'step_0' if s['step_num'] == 0 else ('step_1-3' if s['step_num'] <= 3 else 'step_4+')
                by_step[bucket]['total'] += 1
                if s['extract_match']:
                    by_step[bucket]['correct'] += 1
        print(f"\n  {cond}:")
        for b in ['step_0', 'step_1-3', 'step_4+']:
            if by_step[b]['total'] > 0:
                acc = by_step[b]['correct'] / by_step[b]['total'] * 100
                print(f"    {b:>10s}: {acc:.1f}% ({by_step[b]['correct']}/{by_step[b]['total']})")

    # 3. By trajectory length (does summary help more for longer tasks?)
    print("\n--- Accuracy by Trajectory Length ---")
    for cond in conditions:
        key = cond_keys[cond]
        by_len = defaultdict(lambda: {'correct': 0, 'total': 0})
        for r in results:
            n = r['num_steps']
            bucket = 'short(1-3)' if n <= 3 else ('medium(4-7)' if n <= 7 else 'long(8+)')
            for s in r[key]:
                by_len[bucket]['total'] += 1
                if s['extract_match']:
                    by_len[bucket]['correct'] += 1
        print(f"\n  {cond}:")
        for b in ['short(1-3)', 'medium(4-7)', 'long(8+)']:
            if by_len[b]['total'] > 0:
                acc = by_len[b]['correct'] / by_len[b]['total'] * 100
                print(f"    {b:>15s}: {acc:.1f}% ({by_len[b]['correct']}/{by_len[b]['total']})")

    # 4. Per-action-type accuracy
    print("\n--- Per-Action-Type Accuracy ---")
    for cond in conditions:
        key = cond_keys[cond]
        by_type = defaultdict(lambda: {'correct': 0, 'total': 0})
        for r in results:
            for s in r[key]:
                at = s['gt_action_type']
                by_type[at]['total'] += 1
                if s['extract_match']:
                    by_type[at]['correct'] += 1
        print(f"\n  {cond}:")
        for at in sorted(by_type.keys()):
            d = by_type[at]
            acc = d['correct'] / d['total'] * 100 if d['total'] > 0 else 0
            print(f"    {at:>15s}: {acc:.1f}% ({d['correct']}/{d['total']})")

    # 5. Simulated TSR
    print("\n--- Simulated TSR ---")
    for cond in conditions:
        key = cond_keys[cond]
        sc = sum(
            1 for r in results
            if all(s['extract_match'] for s in r[key]) and len(r[key]) == r['num_steps']
        )
        tsr = sc / len(results) * 100 if results else 0
        print(f"  {cond:20s}: TSR = {sc}/{len(results)} ({tsr:.1f}%)")

    # 6. Paired comparison
    print("\n--- Paired Comparison: baseline vs oracle_summary ---")
    better_base = 0
    better_oracle = 0
    tied = 0
    for r in results:
        base_acc = sum(s['extract_match'] for s in r['results_baseline'])
        oracle_acc = sum(s['extract_match'] for s in r['results_oracle_summary'])
        if base_acc > oracle_acc:
            better_base += 1
        elif oracle_acc > base_acc:
            better_oracle += 1
        else:
            tied += 1
    print(f"  baseline better:       {better_base}")
    print(f"  oracle_summary better: {better_oracle}")
    print(f"  tied:                  {tied}")

    # Save summary
    summary = {'total_episodes': len(results), 'conditions': {}}
    for cond in conditions:
        key = cond_keys[cond]
        ext_c = sum(s['extract_match'] for r in results for s in r[key])
        total = sum(len(r[key]) for r in results)
        sc = sum(
            1 for r in results
            if all(s['extract_match'] for s in r[key]) and len(r[key]) == r['num_steps']
        )
        summary['conditions'][cond] = {
            'extract_match': ext_c / total if total > 0 else 0,
            'total_steps': total,
            'tsr': sc / len(results) if results else 0,
        }
    save_json(summary, os.path.join(output_dir, 'summary_eval_summary.json'))
    print(f"\nSummary saved to {output_dir}/summary_eval_summary.json")


def main(args):
    global fm
    fm = init_format()
    os.makedirs(args.output_dir, exist_ok=True)

    out_path = os.path.join(args.output_dir, 'summary_eval_results.jsonl')
    if os.path.exists(out_path):
        os.remove(out_path)

    data = load_ac_trajectories(jsonl_path=args.jsonl_file, max_episodes=args.max_episodes)
    print(f"Loaded {len(data)} episodes.")
    print(f"Running Summary-Augmented evaluation (3 conditions × {len(data)} episodes)...")
    print(f"  Model: {args.model_name}")
    print(f"  Output: {args.output_dir}")

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_episode, ep, args): ep for ep in data}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                if len(results) % 20 == 0:
                    base_acc = sum(
                        s['extract_match'] for r in results for s in r['results_baseline']
                    ) / max(1, sum(len(r['results_baseline']) for r in results))
                    oracle_acc = sum(
                        s['extract_match'] for r in results for s in r['results_oracle_summary']
                    ) / max(1, sum(len(r['results_oracle_summary']) for r in results))
                    vlm_acc = sum(
                        s['extract_match'] for r in results for s in r['results_vlm_summary']
                    ) / max(1, sum(len(r['results_vlm_summary']) for r in results))
                    print(f"Progress: {len(results)}/{len(data)} | "
                          f"base: {base_acc:.3f} | oracle: {oracle_acc:.3f} | vlm: {vlm_acc:.3f}")
            except Exception as e:
                print(f"Exception: {e}")
                traceback.print_exc()

    analyze_results(results, args.output_dir)
    print(f"\nDone. {len(results)} episodes processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval TODO 4: Summary-Augmented Agent")
    parser.add_argument("--jsonl_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_todo4_summary")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--n_history_image_limit", type=int, default=2)
    parser.add_argument("--max_workers", type=int, default=2)
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()
    main(args)
