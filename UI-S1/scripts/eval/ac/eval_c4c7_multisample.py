"""Eval C4+C7: Multi-Sample Grounding (GPU phase).

Collects K samples per step with temperature sampling for agreement and oracle analysis.
"""

import argparse
import copy
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from ac_utils import (
    load_ac_trajectories, fix_line, init_format, save_jsonl,
    evaluate_android_control_action, find_last_image_ele, slim_messages,
    call_mobile_agent_vllm, safe_parse_response, _json_default,
)

# Patch for temperature sampling
from evaluation.qwenvl_utils import END_POINT, image_to_data_url, message_translate
from openai import OpenAI
from PIL import Image
import traceback
import time

result_lock = Lock()
fm = None


def call_vllm_with_temperature(messages, model_name, temperature=1.0, top_k=50):
    """Call vLLM with temperature sampling instead of greedy."""
    messages, screenshot_list = message_translate(messages, to_format='openai')
    screenshot_ptr = 0
    for msg in messages:
        for content in msg['content']:
            if 'image_url' in content:
                url = image_to_data_url(Image.open(screenshot_list[screenshot_ptr]))
                content['image_url']['url'] = url
                screenshot_ptr += 1
    assert screenshot_ptr == len(screenshot_list)

    for attempt in range(5):
        try:
            bot = OpenAI(api_key="EMPTY", base_url=END_POINT, timeout=600)
            resp = bot.chat.completions.create(
                model=model_name,
                messages=messages,
                extra_body={"top_k": top_k},
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except Exception as e:
            traceback.print_exc()
            if attempt < 4:
                time.sleep(5)
    return ""


def process_episode(episode, args):
    """Collect K samples for each step in the episode."""
    global fm

    fixed = fix_line(copy.deepcopy(episode))
    num_steps = len(fixed['steps'])
    step_samples = []

    # Build messages up to each step and collect K samples
    state = None
    for step_id in range(num_steps):
        current_check = fixed['steps'][step_id]['check_options']
        gt_action = fixed['steps'][step_id]['action_content']

        # Use GT actions for previous steps (static eval)
        state = fm.gen_next_round(fixed, state, previous_model_response=None)
        if state is None:
            break

        messages = slim_messages(
            messages=state['messages'],
            num_image_limit=args.n_history_image_limit
        )
        _, width, height, resized_width, resized_height = find_last_image_ele(messages)

        samples = []
        for k in range(args.K):
            try:
                if k == 0:
                    # First sample: greedy
                    response = call_mobile_agent_vllm(messages=messages, model_name=args.model_name)
                else:
                    response = call_vllm_with_temperature(
                        messages, args.model_name,
                        temperature=args.temperature, top_k=50
                    )
                pred = safe_parse_response(fm, response)
                pred_action = pred['action_content']
                type_match, extract_match = evaluate_android_control_action(
                    pred_action, current_check,
                    width, height, resized_width, resized_height
                )
                samples.append({
                    'pred_action': pred_action,
                    'type_match': type_match,
                    'extract_match': extract_match,
                })
            except Exception as e:
                samples.append({
                    'pred_action': None,
                    'type_match': False,
                    'extract_match': False,
                    'error': str(e),
                })

        step_samples.append({
            'step_num': step_id,
            'gt_action': gt_action,
            'gt_action_type': gt_action['action'],
            'samples': samples,
        })

    result = {
        'episode_id': episode.get('episode_id'),
        'goal': episode['goal'],
        'num_steps': num_steps,
        'step_samples': step_samples,
    }

    with result_lock:
        out_path = os.path.join(args.output_dir, 'multisample_results.jsonl')
        with open(out_path, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False, default=_json_default) + '\n')

    return result


def main(args):
    global fm
    fm = init_format()
    os.makedirs(args.output_dir, exist_ok=True)

    out_path = os.path.join(args.output_dir, 'multisample_results.jsonl')
    if os.path.exists(out_path):
        os.remove(out_path)

    data = load_ac_trajectories(jsonl_path=args.jsonl_file, max_episodes=args.max_episodes)
    print(f"Loaded {len(data)} episodes. Collecting K={args.K} samples per step...")

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_episode, ep, args): ep for ep in data}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                if len(results) % 20 == 0:
                    print(f"Progress: {len(results)}/{len(data)}")
            except Exception as e:
                print(f"Exception: {e}")

    print(f"Done. {len(results)} episodes processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval C4+C7: Multi-Sample Grounding (GPU)")
    parser.add_argument("--jsonl_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_c4c7_ac")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--K", type=int, default=10, help="Number of samples per step")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n_history_image_limit", type=int, default=2)
    parser.add_argument("--max_workers", type=int, default=2)
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()
    main(args)
