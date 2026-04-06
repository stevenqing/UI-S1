"""Eval U7: Actor-Verifier Multi-Agent AR Trajectory Evaluation.

Architecture per step:
1. Actor Agent: Standard greedy prediction (same as baseline Eval A).
2. Verifier Agent: Takes (screenshot, goal, predicted action) and outputs
   {"verdict": "PASS"/"FAIL", "reason": "..."}.
3. If PASS -> use greedy action (preserving baseline quality).
4. If FAIL -> re-generate K samples with temperature=0.6, majority vote on
   action type, pick first matching -> use voted action.
5. Continue AR (stop on first extract_match=False).
"""

import argparse
import copy
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from openai import OpenAI
from PIL import Image

from ac_utils import (
    load_ac_trajectories, fix_line, init_format, save_jsonl, save_json,
    compute_trajectory_metrics, length_bucket,
    evaluate_android_control_action,
    find_last_image_ele, slim_messages, safe_parse_response, _json_default,
    call_mobile_agent_vllm, generate_k_samples_fast,
)
from evaluation.qwenvl_utils import image_to_data_url

result_lock = Lock()
fm = None

VERIFIER_PROMPT_TEMPLATE = """You are a verification agent for a mobile GUI task.
The user's goal is: {goal}
The actor agent looked at the current screenshot and predicted this action: {action_json}

Evaluate whether this action is correct:
1. Is the action TYPE appropriate for the current screen state?
2. Is the TARGET (coordinate/text/button) reasonable given what's visible?
3. Does this action make progress toward the goal?

Output a JSON: {{"verdict": "PASS" or "FAIL", "reason": "brief explanation"}}"""


def call_verifier(screenshot_path, goal, action_json_str, model_name):
    """Call the verifier agent with the current screenshot and predicted action.

    Builds OpenAI-format messages with a base64-encoded screenshot image,
    sends to the vLLM server, and parses the verdict.

    Args:
        screenshot_path: Path to the current screenshot image file.
        goal: The user's task goal string.
        action_json_str: JSON string of the predicted action.
        model_name: Model name for the vLLM endpoint.

    Returns:
        Tuple of (verdict, reason) where verdict is "PASS" or "FAIL".
        Defaults to "PASS" on parse failure (conservative: trust the actor).
    """
    prompt = VERIFIER_PROMPT_TEMPLATE.format(
        goal=goal,
        action_json=action_json_str,
    )

    # Build OpenAI-format messages with base64 image
    img = Image.open(screenshot_path)
    data_url = image_to_data_url(img)

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
                    "text": prompt,
                },
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
        print(f"Verifier call failed: {e}")
        return "PASS", f"verifier_error: {e}"

    return parse_verifier_response(response_text)


def parse_verifier_response(response_text):
    """Parse the verifier's response to extract verdict and reason.

    Tries JSON parsing first, then falls back to regex for PASS/FAIL.
    Defaults to PASS on failure (conservative: trust the actor).

    Args:
        response_text: Raw text response from the verifier.

    Returns:
        Tuple of (verdict, reason).
    """
    # Try JSON parse
    try:
        parsed = json.loads(response_text)
        verdict = parsed.get("verdict", "PASS").upper().strip()
        reason = parsed.get("reason", "")
        if verdict in ("PASS", "FAIL"):
            return verdict, reason
    except (json.JSONDecodeError, AttributeError):
        pass

    # Try extracting JSON from within the response
    json_match = re.search(r'\{[^{}]*"verdict"\s*:\s*"(PASS|FAIL)"[^{}]*\}', response_text, re.IGNORECASE)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            verdict = parsed.get("verdict", "PASS").upper().strip()
            reason = parsed.get("reason", "")
            if verdict in ("PASS", "FAIL"):
                return verdict, reason
        except (json.JSONDecodeError, AttributeError):
            pass

    # Fallback: regex search for PASS or FAIL keywords
    if re.search(r'\bFAIL\b', response_text, re.IGNORECASE):
        return "FAIL", response_text.strip()[:200]
    if re.search(r'\bPASS\b', response_text, re.IGNORECASE):
        return "PASS", response_text.strip()[:200]

    # Default: trust the actor
    return "PASS", "verifier_parse_fallback"


def majority_vote_select(samples):
    """Select action by majority vote on action type, then pick first match.

    Args:
        samples: List of sample dicts from generate_k_samples_fast.

    Returns:
        Tuple of (selected_sample, voted_type, vote_info_dict).
    """
    action_types = []
    for s in samples:
        if s['pred_action'] and s['parse_ok']:
            action_types.append(s['pred_action'].get('action', 'unknown'))

    if not action_types:
        return samples[0] if samples else None, 'all_failed', {}

    type_counter = Counter(action_types)
    voted_type = type_counter.most_common(1)[0][0]
    agreement = type_counter.most_common(1)[0][1] / len(action_types)

    for s in samples:
        if s['pred_action'] and s['pred_action'].get('action') == voted_type:
            return s, voted_type, {
                'agreement': agreement,
                'type_counts': dict(type_counter),
                'voted_type': voted_type,
            }

    return samples[0], voted_type, {'agreement': agreement}


def process_episode(episode, args):
    """Process a single episode with Actor-Verifier AR evaluation.

    For each step:
      1. Actor predicts greedy action.
      2. Verifier judges PASS/FAIL.
      3. On FAIL, re-sample K times and majority-vote.
      4. Evaluate chosen action; stop on first extract_match=False.
    """
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
                num_image_limit=args.n_history_image_limit,
            )

            image_ele_result = find_last_image_ele(messages)
            screenshot_path = image_ele_result[0]
            width = image_ele_result[1]
            height = image_ele_result[2]
            resized_width = image_ele_result[3]
            resized_height = image_ele_result[4]

            # --- Step 1: Actor greedy prediction ---
            actor_response = call_mobile_agent_vllm(
                messages=messages,
                model_name=args.model_name,
            )
            actor_pred = safe_parse_response(fm, actor_response)
            actor_action = actor_pred['action_content']

            # --- Step 2: Verifier ---
            action_json_str = json.dumps(actor_action, ensure_ascii=False)
            verdict, reason = call_verifier(
                screenshot_path=screenshot_path,
                goal=goal,
                action_json_str=action_json_str,
                model_name=args.model_name,
            )

            resampled = False
            agreement = 0.0
            pred_action = actor_action
            model_response = actor_response

            # --- Step 3/4: If FAIL, resample and majority vote ---
            if verdict == "FAIL":
                resampled = True
                samples = generate_k_samples_fast(
                    messages, args.model_name, args.K, args.temperature, fm,
                )
                selected, voted_type, vote_info = majority_vote_select(samples)
                agreement = vote_info.get('agreement', 0.0)

                if selected and selected['pred_action']:
                    pred_action = selected['pred_action']
                    model_response = selected['response']
                # else: keep actor_action and actor_response as fallback

            # --- Evaluate ---
            type_match, extract_match = evaluate_android_control_action(
                pred_action, current_check,
                width, height, resized_width, resized_height,
            )

            step_results.append({
                'step_num': step_id,
                'type_match': type_match,
                'extract_match': extract_match,
                'pred_action': pred_action,
                'gt_action': gt_action,
                'gt_action_type': gt_action['action'],
                'verified': verdict,
                'resampled': resampled,
                'agreement': agreement,
            })

            # --- Step 5: Stop on first failure ---
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
        out_path = os.path.join(args.output_dir, 'actor_verifier_results.jsonl')
        with open(out_path, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False, default=_json_default) + '\n')

    return result


def main(args):
    global fm
    fm = init_format()
    os.makedirs(args.output_dir, exist_ok=True)

    out_path = os.path.join(args.output_dir, 'actor_verifier_results.jsonl')
    if os.path.exists(out_path):
        os.remove(out_path)

    data = load_ac_trajectories(jsonl_path=args.jsonl_file, max_episodes=args.max_episodes)
    print(f"Loaded {len(data)} episodes. Running U7 Actor-Verifier AR (K={args.K}, temp={args.temperature})...")

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

    # Verifier stats
    total_steps = sum(len(r['step_results']) for r in results)
    total_pass = sum(1 for r in results for s in r['step_results'] if s['verified'] == 'PASS')
    total_fail = sum(1 for r in results for s in r['step_results'] if s['verified'] == 'FAIL')
    total_resampled = sum(1 for r in results for s in r['step_results'] if s['resampled'])

    # Agreement stats (only for resampled steps)
    resample_agreements = [s['agreement'] for r in results for s in r['step_results'] if s['resampled']]
    avg_resample_agreement = sum(resample_agreements) / len(resample_agreements) if resample_agreements else 0.0

    summary = {
        'model': args.model_name,
        'experiment': 'U7_actor_verifier_AR',
        'K': args.K,
        'temperature': args.temperature,
        'total_episodes': len(results),
        **metrics,
        'total_steps_evaluated': total_steps,
        'verifier_pass': total_pass,
        'verifier_fail': total_fail,
        'verifier_fail_rate': total_fail / total_steps if total_steps > 0 else 0.0,
        'total_resampled': total_resampled,
        'avg_resample_agreement': avg_resample_agreement,
    }
    save_json(summary, os.path.join(args.output_dir, 'summary.json'))

    print(f"\nU7 Actor-Verifier AR completed.")
    print(f"TSR: {metrics['tsr']:.4f} ({metrics['success_count']}/{metrics['n']})")
    print(f"Avg Progress: {metrics['avg_progress']:.4f}")
    print(f"Verifier PASS: {total_pass}/{total_steps} | FAIL: {total_fail}/{total_steps}")
    print(f"Resampled steps: {total_resampled}/{total_steps}")
    if resample_agreements:
        print(f"Avg resample agreement: {avg_resample_agreement:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval U7: Actor-Verifier Multi-Agent AR")
    parser.add_argument("--jsonl_file", type=str, default=None, help="Path to evaluation JSONL")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_u7_ac", help="Output directory")
    parser.add_argument("--model_name", type=str, required=True, help="Model name for vLLM")
    parser.add_argument("--K", type=int, default=5, help="Number of resample candidates on FAIL")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for resampling")
    parser.add_argument("--n_history_image_limit", type=int, default=2, help="Max history images")
    parser.add_argument("--max_workers", type=int, default=16, help="Thread pool workers for episodes")
    parser.add_argument("--max_episodes", type=int, default=None, help="Limit episodes for testing")
    args = parser.parse_args()
    main(args)
