"""Eval M3: Step-0 Router Multi-Agent Evaluation.

Architecture:
- Step 0: Router checks if needed app is open; if not, directly generates open action
- Steps 1+: Standard Eval A pipeline (zero overhead)

Targets the dominant error pattern: 69.3% of errors at step 0, mostly open->click confusion.
"""

import argparse
import copy
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from ac_utils import (
    load_ac_trajectories, fix_line, init_format, save_jsonl, save_json,
    compute_trajectory_metrics, length_bucket,
    evaluate_android_control_action, call_mobile_agent_vllm,
    find_last_image_ele, slim_messages, safe_parse_response, _json_default,
)

result_lock = Lock()
fm = None

ROUTER_PROMPT = """Look at the current screenshot. You need to accomplish this task: {goal}

Is the app needed for this task already open and visible on the screen?

Answer with ONLY a JSON object:
- If the needed app IS already open: {{"app_open": true}}
- If the needed app is NOT open (you see home screen, app drawer, or a different app): {{"app_open": false, "app_name": "NAME_OF_APP_TO_OPEN"}}"""


def call_router(screenshot_path, goal, model_name):
    """Call router to check if needed app is open."""
    from x.qwen.image import make_qwen_image_item

    messages = [
        {'role': 'user', 'content': [
            {'text': ROUTER_PROMPT.format(goal=goal)},
            make_qwen_image_item(screenshot_path),
        ]}
    ]

    response = call_mobile_agent_vllm(messages=messages, model_name=model_name)
    return response.strip()


def parse_router_response(response):
    """Parse router JSON output."""
    match = re.search(r'\{[^{}]*\}', response)
    if match:
        try:
            result = json.loads(match.group())
            return {
                'app_open': result.get('app_open', True),
                'app_name': result.get('app_name', ''),
            }
        except json.JSONDecodeError:
            pass
    # Fallback: check for keywords
    lower = response.lower()
    if 'false' in lower or 'not open' in lower or 'no' in lower:
        name_match = re.search(r'"app_name"\s*:\s*"([^"]+)"', response)
        app_name = name_match.group(1) if name_match else ''
        return {'app_open': False, 'app_name': app_name}
    return {'app_open': True, 'app_name': ''}


def process_episode(episode, args):
    """Process episode with step-0 router."""
    global fm

    fixed = fix_line(copy.deepcopy(episode))
    num_steps = len(fixed['steps'])
    state = None
    model_response = None
    step_results = []
    router_used = False
    router_decision = None

    try:
        for step_id in range(num_steps):
            current_check = fixed['steps'][step_id]['check_options']
            gt_action = fixed['steps'][step_id]['action_content']

            if step_id == 0:
                # Step 0: Router decides if app needs opening
                screenshot = fixed['steps'][0]['screenshot']
                router_response = call_router(screenshot, episode['goal'], args.model_name)
                router_decision = parse_router_response(router_response)

                if not router_decision['app_open'] and router_decision['app_name']:
                    # Router says app not open -> generate open action directly
                    router_used = True
                    pred_action = {'action': 'open', 'text': router_decision['app_name']}

                    type_match, extract_match = evaluate_android_control_action(
                        pred_action, current_check,
                        0, 0, 0, 0  # coordinates not needed for open action
                    )

                    # Build model_response string for gen_next_round continuity
                    action_json = json.dumps(pred_action, ensure_ascii=False)
                    model_response = (
                        f"<think>\nOpening {router_decision['app_name']} as directed by router.\n</think>\n"
                        f"<action>\n{action_json}\n</action>"
                    )

                    # Call gen_next_round to initialize state (will be consumed)
                    state = fm.gen_next_round(fixed, state, previous_model_response=None)

                    step_results.append({
                        'step_num': step_id,
                        'type_match': type_match,
                        'extract_match': extract_match,
                        'pred_action': pred_action,
                        'gt_action': gt_action,
                        'gt_action_type': gt_action['action'],
                        'router_used': True,
                        'router_decision': router_decision,
                        'router_response': router_response,
                    })

                    if not extract_match:
                        break
                    continue

            # Standard Eval A pipeline for step 0 (app already open) and steps 1+
            state = fm.gen_next_round(fixed, state, previous_model_response=model_response)
            if state is None:
                break

            messages = slim_messages(
                messages=state['messages'],
                num_image_limit=args.n_history_image_limit
            )
            _, width, height, resized_width, resized_height = find_last_image_ele(messages)

            model_response = call_mobile_agent_vllm(messages=messages, model_name=args.model_name)

            pred = safe_parse_response(fm, model_response)
            pred_action = pred['action_content']

            type_match, extract_match = evaluate_android_control_action(
                pred_action, current_check,
                width, height, resized_width, resized_height
            )

            step_results.append({
                'step_num': step_id,
                'type_match': type_match,
                'extract_match': extract_match,
                'pred_action': pred_action,
                'gt_action': gt_action,
                'gt_action_type': gt_action['action'],
                'router_used': False,
            })

            if not extract_match:
                break

    except Exception as e:
        print(f"Error episode {episode.get('episode_id', '?')}: {e}")

    correct_steps = sum(1 for s in step_results if s['extract_match'])
    task_success = (correct_steps == num_steps)

    result = {
        'episode_id': episode.get('episode_id'),
        'goal': episode['goal'],
        'num_steps': num_steps,
        'task_success': task_success,
        'final_step_id': correct_steps,
        'step_results': step_results,
        'length_bucket': length_bucket(num_steps),
        'router_used': router_used,
        'router_decision': router_decision,
    }

    with result_lock:
        out_path = os.path.join(args.output_dir, 'router_results.jsonl')
        with open(out_path, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False, default=_json_default) + '\n')

    return result


def main(args):
    global fm
    fm = init_format()
    os.makedirs(args.output_dir, exist_ok=True)

    out_path = os.path.join(args.output_dir, 'router_results.jsonl')
    if os.path.exists(out_path):
        os.remove(out_path)

    data = load_ac_trajectories(jsonl_path=args.jsonl_file, max_episodes=args.max_episodes)
    print(f"Loaded {len(data)} episodes. Running M3 Step-0 Router evaluation...")

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

    metrics = compute_trajectory_metrics(results)

    # Router usage stats
    router_stats = {
        'total_episodes': len(results),
        'router_triggered': sum(1 for r in results if r['router_used']),
        'router_skipped': sum(1 for r in results if not r['router_used']),
    }
    step0_with_router = [r for r in results if r['router_used'] and r['step_results']]
    step0_without_router = [r for r in results if not r['router_used'] and r['step_results']]
    router_stats['step0_acc_with_router'] = (
        sum(1 for r in step0_with_router if r['step_results'][0]['extract_match']) / len(step0_with_router)
        if step0_with_router else 0
    )
    router_stats['step0_acc_without_router'] = (
        sum(1 for r in step0_without_router if r['step_results'][0]['extract_match']) / len(step0_without_router)
        if step0_without_router else 0
    )

    summary = {
        'model': args.model_name,
        'experiment': 'M3_step0_router',
        'total_episodes': len(results),
        **metrics,
        'router_stats': router_stats,
    }
    save_json(summary, os.path.join(args.output_dir, 'summary.json'))

    print(f"\nM3 Step-0 Router completed.")
    print(f"TSR: {metrics['tsr']:.4f} ({metrics['success_count']}/{metrics['n']})")
    print(f"Avg Progress: {metrics['avg_progress']:.4f}")
    print(f"Router triggered: {router_stats['router_triggered']}/{router_stats['total_episodes']}")
    print(f"Step-0 acc (with router): {router_stats['step0_acc_with_router']:.3f}")
    print(f"Step-0 acc (without router): {router_stats['step0_acc_without_router']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval M3: Step-0 Router")
    parser.add_argument("--jsonl_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_m3_ac")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--n_history_image_limit", type=int, default=2)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()
    main(args)
