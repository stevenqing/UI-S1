"""Eval U8: Actor-Critic Best-of-K Multi-Agent AR Trajectory.

Architecture per step:
1. Actor Agent: Generate 1 greedy sample + (K-1) temperature samples -> K candidates.
2. Critic Agent: Score each candidate (1-5) using current screenshot + goal + action.
3. Pick highest-scored candidate (ties prefer greedy / sample 0).
4. Continue AR (stop on first extract_match=False).
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
    evaluate_android_control_action,
    call_mobile_agent_vllm, find_last_image_ele,
    slim_messages, safe_parse_response, _json_default,
    generate_k_samples_fast,
)
from evaluation.qwenvl_utils import message_translate, image_to_data_url
from PIL import Image
from openai import OpenAI

result_lock = Lock()
fm = None


CRITIC_PROMPT_TEMPLATE = """You are a critic agent evaluating a candidate action for a mobile GUI task.
The user's goal is: {goal}
Candidate action: {action_json}

Looking at the current screenshot, rate this action on a scale of 1-5:
5 = Clearly correct action and target for this screen
4 = Likely correct
3 = Uncertain
2 = Likely wrong
1 = Clearly wrong action or target

Output a JSON: {{"score": N, "reasoning": "brief explanation"}}"""


def parse_critic_score(response_text):
    """Parse critic response to extract score 1-5.

    Strategy: try JSON parse first, then regex fallback, default to 3.
    """
    # Try JSON parse
    try:
        data = json.loads(response_text)
        score = int(data.get("score", 3))
        if 1 <= score <= 5:
            return score
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Try extracting JSON from within the response text
    json_match = re.search(r'\{[^{}]*"score"\s*:\s*(\d)[^{}]*\}', response_text)
    if json_match:
        score = int(json_match.group(1))
        if 1 <= score <= 5:
            return score

    # Regex fallback: find any digit 1-5
    digit_match = re.search(r'\b([1-5])\b', response_text)
    if digit_match:
        return int(digit_match.group(1))

    return 3  # default


def build_critic_message(screenshot_path, goal, action_json_str):
    """Build OpenAI-format messages for the critic with base64 image."""
    img = Image.open(screenshot_path)
    data_url = image_to_data_url(img)

    prompt_text = CRITIC_PROMPT_TEMPLATE.format(
        goal=goal,
        action_json=action_json_str,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                },
                {
                    "type": "text",
                    "text": prompt_text,
                },
            ],
        }
    ]
    return messages


def call_critic(messages, model_name):
    """Call the critic model and return the raw response text."""
    try:
        bot = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1", timeout=600)
        resp = bot.chat.completions.create(
            model=model_name,
            messages=messages,
            extra_body={"top_k": 1},
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"Critic call failed: {e}")
        return ""


def score_candidates_batch(candidates, screenshot_path, goal, model_name):
    """Score all K candidates in parallel using the critic.

    Returns list of integer scores (1-5), one per candidate.
    """
    # Build critic messages for each candidate
    critic_tasks = []
    for sample in candidates:
        if sample['pred_action'] and sample['parse_ok']:
            action_json_str = json.dumps(sample['pred_action'], ensure_ascii=False)
        else:
            action_json_str = json.dumps({"action": "unknown"})
        critic_msgs = build_critic_message(screenshot_path, goal, action_json_str)
        critic_tasks.append(critic_msgs)

    # Send all critic calls in parallel
    scores = [3] * len(candidates)  # default
    with ThreadPoolExecutor(max_workers=len(candidates)) as executor:
        future_to_idx = {
            executor.submit(call_critic, critic_tasks[i], model_name): i
            for i in range(len(candidates))
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                resp_text = future.result()
                scores[idx] = parse_critic_score(resp_text)
            except Exception:
                scores[idx] = 3

    return scores


def select_best_candidate(candidates, scores):
    """Select candidate with highest score. Ties prefer sample 0 (greedy).

    Returns (selected_sample, selected_idx).
    """
    if not candidates:
        return None, -1

    best_idx = 0
    best_score = scores[0]
    for i in range(1, len(candidates)):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i

    return candidates[best_idx], best_idx


def process_episode(episode, args):
    """Process a single episode with actor-critic AR evaluation."""
    global fm

    fixed = fix_line(copy.deepcopy(episode))
    num_steps = len(fixed['steps'])
    goal = episode['goal']
    state = None
    model_response = None
    step_results = []

    try:
        for step_id in range(num_steps):
            current_check = fixed['steps'][step_id]['check_options']
            gt_action = fixed['steps'][step_id]['action_content']

            state = fm.gen_next_round(fixed, state, previous_model_response=model_response)
            if state is None:
                break

            messages = slim_messages(
                messages=state['messages'],
                num_image_limit=args.n_history_image_limit
            )

            last_image_path, width, height, resized_width, resized_height = find_last_image_ele(messages)

            # --- Actor: Generate K candidates ---
            # Sample 0: greedy
            greedy_response = call_mobile_agent_vllm(
                messages=messages,
                model_name=args.model_name,
            )
            try:
                greedy_pred = safe_parse_response(fm, greedy_response)
                greedy_sample = {
                    'response': greedy_response,
                    'pred_action': greedy_pred['action_content'],
                    'parse_ok': True,
                }
            except Exception:
                greedy_sample = {
                    'response': greedy_response,
                    'pred_action': None,
                    'parse_ok': False,
                }

            # Samples 1..K-1: temperature sampling
            if args.K > 1:
                temp_samples = generate_k_samples_fast(
                    messages, args.model_name, args.K - 1, args.temperature, fm
                )
            else:
                temp_samples = []

            # Combine: greedy first, then temperature samples
            candidates = [greedy_sample] + temp_samples

            # --- Critic: Score all K candidates ---
            scores = score_candidates_batch(
                candidates, last_image_path, goal, args.model_name
            )

            # --- Select best candidate ---
            selected, selected_idx = select_best_candidate(candidates, scores)

            if selected and selected['pred_action']:
                pred_action = selected['pred_action']
                model_response = selected['response']
            else:
                # Fallback: use greedy
                pred_action = greedy_sample['pred_action']
                model_response = greedy_sample['response']
                selected_idx = 0
                if pred_action is None:
                    break

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
                'selected_idx': selected_idx,
                'greedy_score': scores[0],
                'max_score': max(scores),
                'scores': scores,
            })

            if not extract_match:
                break

    except Exception as e:
        print(f"Error episode {episode.get('episode_id', '?')}: {e}")

    correct_steps = sum(1 for s in step_results if s['extract_match'])
    task_success = (correct_steps == num_steps)

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
        out_path = os.path.join(args.output_dir, 'actor_critic_results.jsonl')
        with open(out_path, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False, default=_json_default) + '\n')

    return result


def main(args):
    global fm
    fm = init_format()
    os.makedirs(args.output_dir, exist_ok=True)

    out_path = os.path.join(args.output_dir, 'actor_critic_results.jsonl')
    if os.path.exists(out_path):
        os.remove(out_path)

    data = load_ac_trajectories(jsonl_path=args.jsonl_file, max_episodes=args.max_episodes)
    print(f"Loaded {len(data)} episodes. Running U8 Actor-Critic AR (K={args.K}, temp={args.temperature})...")

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

    # Critic selection stats
    all_selected_idxs = [s['selected_idx'] for r in results for s in r['step_results']]
    all_greedy_scores = [s['greedy_score'] for r in results for s in r['step_results']]
    all_max_scores = [s['max_score'] for r in results for s in r['step_results']]

    greedy_selected_count = sum(1 for idx in all_selected_idxs if idx == 0)
    total_steps_evaluated = len(all_selected_idxs)
    greedy_selected_rate = greedy_selected_count / total_steps_evaluated if total_steps_evaluated > 0 else 0
    avg_greedy_score = sum(all_greedy_scores) / len(all_greedy_scores) if all_greedy_scores else 0
    avg_max_score = sum(all_max_scores) / len(all_max_scores) if all_max_scores else 0

    summary = {
        'model': args.model_name,
        'experiment': 'U8_actor_critic_AR',
        'K': args.K,
        'temperature': args.temperature,
        'total_episodes': len(results),
        **metrics,
        'greedy_selected_rate': greedy_selected_rate,
        'avg_greedy_score': avg_greedy_score,
        'avg_max_score': avg_max_score,
        'total_steps_evaluated': total_steps_evaluated,
    }
    save_json(summary, os.path.join(args.output_dir, 'summary.json'))

    print(f"\nU8 Actor-Critic AR completed.")
    print(f"TSR: {metrics['tsr']:.4f} ({metrics['success_count']}/{metrics['n']})")
    print(f"Avg Progress: {metrics['avg_progress']:.4f}")
    print(f"Greedy selected: {greedy_selected_rate:.3f} ({greedy_selected_count}/{total_steps_evaluated})")
    print(f"Avg greedy score: {avg_greedy_score:.2f} | Avg max score: {avg_max_score:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval U8: Actor-Critic Best-of-K AR Trajectory")
    parser.add_argument("--jsonl_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_u8_ac")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--n_history_image_limit", type=int, default=2)
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()
    main(args)
