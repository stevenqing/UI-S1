"""Eval U11: Action-History Observer Multi-Agent AR Trajectory Evaluation.

Key insight: Traditional observers (D1/D2) transmit STATE DESCRIPTION, but AC errors
are mainly ACTION TYPE errors (79.7%). State description doesn't help action disambiguation.

U11 Observer outputs action disambiguation signal:
  {"last_correct": true/false, "avoid_type": "...", "confidence": "low/med/high"}

This signal is injected into the actor's prompt to help avoid repeating wrong action types.

Architecture per step:
1. Observer reads screenshot + prev action → judges if prev action was appropriate
2. Observer outputs action disambiguation signal (not state description)
3. Actor reads screenshot + disambiguation signal → predicts action
4. Continue AR (stop on first extract_match=False)

Pre-registered prediction (from Markov model): TSR ~16.81% (+0.74pp over baseline).
"""

import argparse
import copy
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from openai import OpenAI
from PIL import Image

from ac_utils import (
    load_ac_trajectories, fix_line, init_format, save_jsonl, save_json,
    compute_trajectory_metrics, length_bucket,
    evaluate_android_control_action,
    find_last_image_ele, slim_messages, safe_parse_response, _json_default,
    call_mobile_agent_vllm,
)
from evaluation.qwenvl_utils import image_to_data_url

result_lock = Lock()
fm = None

ACTION_OBSERVER_PROMPT = """You are an action quality observer for a mobile GUI automation task.

The user's goal is: {goal}

The previous action taken was: {prev_action}

Look at the CURRENT screenshot (the result after the previous action) and evaluate:

1. Did the previous action make appropriate progress toward the goal?
2. Was the ACTION TYPE (click/swipe/type/open/system_button/wait/long_press) appropriate for the situation?
3. If the action type was wrong, what type should have been used instead?

Output a JSON with this exact format:
{{"last_correct": true or false, "likely_wrong_type": "action_type_to_avoid_or_null", "confidence": "low" or "medium" or "high", "suggestion": "brief suggestion for next action type"}}"""

ACTOR_HINT_TEMPLATE = """Action Quality Feedback from Observer:
- Previous action assessment: {assessment}
- Suggestion: {suggestion}
{avoid_hint}
Please consider this feedback when choosing your next action."""


def call_action_observer(screenshot_path, goal, prev_action_str, model_name):
    """Call the action observer to evaluate the previous action.

    Returns:
        dict with keys: last_correct, likely_wrong_type, confidence, suggestion
    """
    prompt = ACTION_OBSERVER_PROMPT.format(
        goal=goal,
        prev_action=prev_action_str,
    )

    img = Image.open(screenshot_path)
    data_url = image_to_data_url(img)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        bot = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1", timeout=600)
        resp = bot.chat.completions.create(
            model=model_name,
            messages=messages,
            extra_body={"top_k": 1},
        )
        response_text = resp.choices[0].message.content
    except Exception as e:
        print(f"Observer call failed: {e}")
        return {'last_correct': True, 'likely_wrong_type': None, 'confidence': 'low', 'suggestion': ''}

    return parse_observer_response(response_text)


def parse_observer_response(response_text):
    """Parse the observer's action assessment response."""
    default = {'last_correct': True, 'likely_wrong_type': None, 'confidence': 'low', 'suggestion': ''}

    # Try JSON parse
    try:
        parsed = json.loads(response_text)
        return {
            'last_correct': parsed.get('last_correct', True),
            'likely_wrong_type': parsed.get('likely_wrong_type'),
            'confidence': parsed.get('confidence', 'low'),
            'suggestion': parsed.get('suggestion', ''),
        }
    except (json.JSONDecodeError, AttributeError):
        pass

    # Try extracting JSON from response
    json_match = re.search(r'\{[^{}]*"last_correct"\s*:[^{}]*\}', response_text)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            return {
                'last_correct': parsed.get('last_correct', True),
                'likely_wrong_type': parsed.get('likely_wrong_type'),
                'confidence': parsed.get('confidence', 'low'),
                'suggestion': parsed.get('suggestion', ''),
            }
        except (json.JSONDecodeError, AttributeError):
            pass

    # Fallback: look for keywords
    if re.search(r'"last_correct"\s*:\s*false', response_text, re.IGNORECASE):
        default['last_correct'] = False
    if re.search(r'\bfalse\b', response_text, re.IGNORECASE) and 'last_correct' in response_text:
        default['last_correct'] = False

    return default


def build_actor_hint(observer_result):
    """Build a hint string to inject into the actor's prompt."""
    if observer_result is None:
        return ""

    assessment = "CORRECT" if observer_result['last_correct'] else "POTENTIALLY WRONG"
    suggestion = observer_result.get('suggestion', '')

    avoid_hint = ""
    if not observer_result['last_correct'] and observer_result.get('likely_wrong_type'):
        avoid_type = observer_result['likely_wrong_type']
        confidence = observer_result.get('confidence', 'low')
        if confidence in ('medium', 'high'):
            avoid_hint = f"- Consider avoiding action type '{avoid_type}' unless clearly needed\n"

    hint = ACTOR_HINT_TEMPLATE.format(
        assessment=assessment,
        suggestion=suggestion,
        avoid_hint=avoid_hint,
    )
    return hint


def inject_hint_into_messages(messages, hint):
    """Inject the observer hint into the actor's message history.

    Adds the hint as a system-level context before the last user message.
    """
    if not hint or not messages:
        return messages

    # Find the last user message and prepend the hint
    modified = list(messages)
    for i in range(len(modified) - 1, -1, -1):
        msg = modified[i]
        if msg.get('role') == 'user':
            content = msg.get('content', [])
            if isinstance(content, list):
                # Add hint as first text element
                hint_element = {'type': 'text', 'text': hint}
                new_content = [hint_element] + list(content)
                modified[i] = {**msg, 'content': new_content}
            elif isinstance(content, str):
                modified[i] = {**msg, 'content': hint + '\n\n' + content}
            break

    return modified


def process_episode(episode, args):
    """Process episode with Action-History Observer."""
    global fm

    fixed = fix_line(copy.deepcopy(episode))
    num_steps = len(fixed['steps'])
    goal = episode['goal']
    state = None
    model_response = None
    step_results = []
    prev_action = None
    observer_result = None

    try:
        for step_id in range(num_steps):
            current_check = fixed['steps'][step_id]['check_options']
            gt_action = fixed['steps'][step_id]['action_content']

            state = fm.gen_next_round(fixed, state, previous_model_response=model_response)
            if state is None:
                break

            messages = slim_messages(
                messages=state['messages'],
                num_image_limit=args.n_history_image_limit,
            )

            image_ele_result = find_last_image_ele(messages)
            screenshot_path = image_ele_result[0]
            width = image_ele_result[1]
            height = image_ele_result[2]
            resized_width = image_ele_result[3]
            resized_height = image_ele_result[4]

            # --- Step 1: Observer evaluates previous action ---
            if step_id > 0 and prev_action is not None:
                prev_action_str = json.dumps(prev_action, ensure_ascii=False)
                observer_result = call_action_observer(
                    screenshot_path=screenshot_path,
                    goal=goal,
                    prev_action_str=prev_action_str,
                    model_name=args.model_name,
                )
            else:
                observer_result = None

            # --- Step 2: Build actor hint from observer ---
            hint = build_actor_hint(observer_result)

            # --- Step 3: Actor prediction with hint ---
            if hint:
                modified_messages = inject_hint_into_messages(messages, hint)
            else:
                modified_messages = messages

            actor_response = call_mobile_agent_vllm(
                messages=modified_messages,
                model_name=args.model_name,
            )
            actor_pred = safe_parse_response(fm, actor_response)
            pred_action = actor_pred['action_content']
            model_response = actor_response

            # --- Evaluate ---
            type_match, extract_match = evaluate_android_control_action(
                pred_action, current_check,
                width, height, resized_width, resized_height,
            )

            step_result = {
                'step_num': step_id,
                'type_match': type_match,
                'extract_match': extract_match,
                'pred_action': pred_action,
                'gt_action': gt_action,
                'gt_action_type': gt_action['action'],
                'observer_result': observer_result,
                'hint_injected': bool(hint),
            }
            step_results.append(step_result)

            prev_action = pred_action

            # Stop on first failure
            if not extract_match:
                break

    except Exception as e:
        print(f"Error episode {episode.get('episode_id', '?')}: {e}")

    correct_steps = sum(1 for s in step_results if s['extract_match'])
    task_success = (correct_steps == num_steps and len(step_results) == num_steps)

    result = {
        'episode_id': episode.get('episode_id'),
        'goal': goal,
        'num_steps': num_steps,
        'task_success': task_success,
        'final_step_id': correct_steps,
        'step_results': step_results,
        'length_bucket': length_bucket(num_steps),
    }

    with result_lock:
        out_path = os.path.join(args.output_dir, 'action_history_observer_results.jsonl')
        with open(out_path, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False, default=_json_default) + '\n')

    return result


def main(args):
    global fm
    fm = init_format()
    os.makedirs(args.output_dir, exist_ok=True)

    out_path = os.path.join(args.output_dir, 'action_history_observer_results.jsonl')

    # Resume support: load already-completed episode IDs
    completed_ids = set()
    if args.resume and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    ep = json.loads(line)
                    completed_ids.add(ep.get('episode_id'))
                except json.JSONDecodeError:
                    pass
        print(f"Resuming: {len(completed_ids)} episodes already completed")
    elif os.path.exists(out_path):
        os.remove(out_path)

    data = load_ac_trajectories(jsonl_path=args.jsonl_file, max_episodes=args.max_episodes)
    # Filter out already completed episodes
    if completed_ids:
        data = [ep for ep in data if ep.get('episode_id') not in completed_ids]
    print(f"Loaded {len(data)} remaining episodes. Running U11 Action-History Observer AR...")

    results = []
    total_episodes = len(data) + len(completed_ids)
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_episode, ep, args): ep for ep in data}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                done = len(results) + len(completed_ids)
                if len(results) % 50 == 0:
                    metrics = compute_trajectory_metrics(results)
                    n_hints = sum(1 for r in results for s in r['step_results'] if s['hint_injected'])
                    n_total = sum(len(r['step_results']) for r in results)
                    print(f"Progress: {done}/{total_episodes} | TSR: {metrics['tsr']:.3f} | "
                          f"Hints: {n_hints}/{n_total}")
            except Exception as e:
                print(f"Exception: {e}")

    # Reload all results (including previously completed) for final summary
    all_results = []
    with open(out_path) as f:
        for line in f:
            try:
                all_results.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    results = all_results

    metrics = compute_trajectory_metrics(results)

    # Observer stats
    total_steps = sum(len(r['step_results']) for r in results)
    hints_injected = sum(1 for r in results for s in r['step_results'] if s['hint_injected'])

    # Observer accuracy (when it said last was wrong, was it actually wrong?)
    observer_calls = []
    for r in results:
        for i, s in enumerate(r['step_results']):
            if s['observer_result'] is not None:
                # Check if observer's assessment matches reality
                # We need to know if the PREVIOUS step was actually correct
                if i > 0:
                    prev_actually_correct = r['step_results'][i-1]['extract_match']
                    obs_said_correct = s['observer_result'].get('last_correct', True)
                    observer_calls.append({
                        'obs_correct': obs_said_correct,
                        'actual_correct': prev_actually_correct,
                    })

    if observer_calls:
        # Observer precision/recall for FAIL detection
        tp = sum(1 for o in observer_calls if not o['obs_correct'] and not o['actual_correct'])
        fp = sum(1 for o in observer_calls if not o['obs_correct'] and o['actual_correct'])
        fn = sum(1 for o in observer_calls if o['obs_correct'] and not o['actual_correct'])
        tn = sum(1 for o in observer_calls if o['obs_correct'] and o['actual_correct'])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    else:
        precision = recall = f1 = 0
        tp = fp = fn = tn = 0

    summary = {
        'model': args.model_name,
        'experiment': 'U11_action_history_observer_AR',
        'total_episodes': len(results),
        **metrics,
        'total_steps_evaluated': total_steps,
        'hints_injected': hints_injected,
        'hint_rate': hints_injected / total_steps if total_steps > 0 else 0,
        'observer_fail_detection': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        },
        'pre_registered_prediction': {
            'predicted_tsr': 0.1681,
            'predicted_delta_vs_baseline': 0.0074,
        },
    }
    save_json(summary, os.path.join(args.output_dir, 'summary.json'))

    print(f"\nU11 Action-History Observer AR completed.")
    print(f"TSR: {metrics['tsr']:.4f} ({metrics['success_count']}/{metrics['n']})")
    print(f"Avg Progress: {metrics['avg_progress']:.4f}")
    print(f"Hints injected: {hints_injected}/{total_steps} ({hints_injected/total_steps*100:.1f}%)")
    if observer_calls:
        print(f"Observer FAIL detection: P={precision:.3f} R={recall:.3f} F1={f1:.3f}")
    print(f"\nPre-registered prediction: TSR=16.81%, delta=+0.74pp")
    print(f"Actual TSR: {metrics['tsr']*100:.2f}%")
    print(f"Prediction error: {abs(metrics['tsr'] - 0.1681)*100:.2f}pp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval U11: Action-History Observer Multi-Agent AR")
    parser.add_argument("--jsonl_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_u11_ac")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--n_history_image_limit", type=int, default=2)
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from existing results")
    args = parser.parse_args()
    main(args)
