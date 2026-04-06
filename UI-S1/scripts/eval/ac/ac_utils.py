"""Shared utilities for AndroidControl evaluation experiments."""

import copy
import json
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'evaluation'))

from evaluation.qwenvl_utils import (
    evaluate_android_control_action as _evaluate_android_control_action,
    call_mobile_agent_vllm,
    find_last_image_ele,
)


def evaluate_android_control_action(*args, **kwargs):
    """Wrapper that ensures return values are native Python bools (not numpy.bool_)."""
    type_match, extract_match = _evaluate_android_control_action(*args, **kwargs)
    return bool(type_match), bool(extract_match)
from x.data.agent.json import JsonFormat
from x.data.agent.space.std_space import RAW_SPACE
from x.qwen.data_format import slim_messages
from x.qwen.image import smart_resize

# Default paths
DEFAULT_DATASET = os.path.join(PROJECT_ROOT, 'datasets', 'android_control_evaluation_std.jsonl')
DEFAULT_IMAGE_ROOT = os.path.join(PROJECT_ROOT, 'datasets')
DEFAULT_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'outputs')

# Action categories
COORD_ACTIONS = {'click', 'long_press'}
NON_COORD_ACTIONS = {'type', 'open', 'system_button', 'wait', 'swipe'}
ALL_ACTION_TYPES = ['click', 'long_press', 'swipe', 'type', 'open', 'system_button', 'wait']


def load_ac_trajectories(jsonl_path=None, image_root=None, max_episodes=None):
    """Load AndroidControl episodes and fix image paths.

    Args:
        jsonl_path: Path to the evaluation JSONL file.
        image_root: Root directory for images (replaces /datasets/ prefix).
        max_episodes: If set, only load this many episodes.

    Returns:
        List of episode dicts with fixed screenshot paths.
    """
    if jsonl_path is None:
        jsonl_path = DEFAULT_DATASET
    if image_root is None:
        image_root = DEFAULT_IMAGE_ROOT

    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            episode = json.loads(line.strip())
            # Fix image paths: /datasets/... -> absolute path
            for step in episode['steps']:
                if step['screenshot'].startswith('/datasets/'):
                    step['screenshot'] = step['screenshot'].replace(
                        '/datasets/', image_root + '/', 1
                    )
                # Ensure check_options exists
                if 'check_options' not in step:
                    check_options = copy.deepcopy(step['action_content'])
                    if 'candidate_bbox' not in check_options and 'bbox' in step:
                        check_options['candidate_bbox'] = step['bbox']
                    elif 'candidate_bbox' not in check_options:
                        check_options['candidate_bbox'] = []
                    step['check_options'] = check_options
            data.append(episode)
            if max_episodes and len(data) >= max_episodes:
                break

    return data


def fix_line(line):
    """Ensure each step has check_options (compat with eval_qwenvl.py)."""
    for step in line['steps']:
        if 'check_options' in step:
            continue
        check_options = copy.deepcopy(step['action_content'])
        if 'candidate_bbox' in step:
            check_options['candidate_bbox'] = step['candidate_bbox']
        elif 'bbox' in step:
            check_options['candidate_bbox'] = step['bbox']
        elif 'candidate_bbox' not in check_options:
            check_options['candidate_bbox'] = []
        step['check_options'] = check_options
    return line


def categorize_action(action_type):
    """Categorize action into coord-based vs non-coord-based."""
    if action_type in COORD_ACTIONS:
        return 'coord'
    return 'non_coord'


def length_bucket(n):
    """Assign trajectory length to bucket."""
    if n <= 3:
        return 'short(1-3)'
    elif n <= 7:
        return 'medium(4-7)'
    elif n <= 15:
        return 'long(8-15)'
    else:
        return 'vlong(16+)'


def init_format():
    """Initialize JsonFormat with standard action space."""
    return JsonFormat(RAW_SPACE, add_thought=True, force_add_thought=True)


def safe_parse_response(fm, model_response):
    """Parse model response with JSON error tolerance.

    Handles common issues like double closing braces '}}' in action JSON.
    """
    import re
    try:
        return fm.parse_response(model_response)
    except json.JSONDecodeError:
        # Try fixing double closing brace
        fixed = re.sub(r'\}\}(\s*</action>)', r'}\1', model_response)
        if fixed != model_response:
            try:
                return fm.parse_response(fixed)
            except Exception:
                pass
        # Try extracting JSON from action tags manually
        match = re.search(r'<action>\s*(\{.*?\})\s*</action>', model_response, re.DOTALL)
        if match:
            action_str = match.group(1)
            # Remove trailing extra braces
            while action_str.endswith('}}'):
                action_str = action_str[:-1]
            try:
                action_content = json.loads(action_str)
                think_match = re.search(r'<think>(.*?)</think>', model_response, re.DOTALL)
                return {
                    'think': think_match.group(1).strip() if think_match else '',
                    'action': action_str,
                    'action_content': action_content,
                }
            except json.JSONDecodeError:
                pass
        raise


def save_jsonl(data, path):
    """Save list of dicts to JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False, default=_json_default) + '\n')


def load_jsonl(path):
    """Load JSONL file to list of dicts."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def _json_default(obj):
    """Handle numpy types for JSON serialization."""
    import numpy as np
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(data, path):
    """Save dict to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)


def _pre_encode_messages(messages):
    """Pre-translate messages and base64-encode images ONCE.

    Returns OpenAI-format messages with base64 data URLs ready for API calls.
    """
    from evaluation.qwenvl_utils import message_translate, image_to_data_url
    from PIL import Image

    msgs, screenshot_list = message_translate(messages, to_format='openai')
    screenshot_ptr = 0
    for msg in msgs:
        for content in msg['content']:
            if 'image_url' in content:
                url = image_to_data_url(Image.open(screenshot_list[screenshot_ptr]))
                content['image_url']['url'] = url
                screenshot_ptr += 1
    return msgs


def generate_k_samples_fast(messages, model_name, K, temperature, fm_obj):
    """Generate K samples efficiently.

    Strategy 1: vLLM n=K parameter (1 API call, prompt processed once).
    Strategy 2 (fallback): Parallel thread calls with pre-encoded messages.
    Both avoid re-encoding images K times.
    """
    from openai import OpenAI

    pre_encoded = _pre_encode_messages(messages)

    # Try n=K in a single call
    try:
        bot = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1", timeout=600)
        resp = bot.chat.completions.create(
            model=model_name,
            messages=pre_encoded,
            n=K,
            temperature=temperature,
            extra_body={"top_k": 50},
        )
        samples = []
        for choice in resp.choices:
            response = choice.message.content
            try:
                pred = safe_parse_response(fm_obj, response)
                samples.append({
                    'response': response,
                    'pred_action': pred['action_content'],
                    'parse_ok': True,
                })
            except Exception:
                samples.append({'response': response, 'pred_action': None, 'parse_ok': False})
        return samples
    except Exception as e:
        # Fallback to parallel calls
        return _generate_k_parallel(pre_encoded, model_name, K, temperature, fm_obj)


def _generate_k_parallel(pre_encoded_msgs, model_name, K, temperature, fm_obj):
    """Fallback: K parallel API calls reusing pre-encoded messages."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI

    def _call(k):
        try:
            bot = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1", timeout=600)
            resp = bot.chat.completions.create(
                model=model_name,
                messages=pre_encoded_msgs,
                temperature=temperature,
                extra_body={"top_k": 50},
            )
            response = resp.choices[0].message.content
            pred = safe_parse_response(fm_obj, response)
            return {
                'response': response,
                'pred_action': pred['action_content'],
                'parse_ok': True,
            }
        except Exception:
            return {'response': '', 'pred_action': None, 'parse_ok': False}

    with ThreadPoolExecutor(max_workers=K) as executor:
        futures = {executor.submit(_call, k): k for k in range(K)}
        results = [None] * K
        for future in as_completed(futures):
            k = futures[future]
            results[k] = future.result()
    return results


def generate_k_samples_adaptive(messages, model_name, K_init, K_max, threshold, temperature, fm_obj):
    """Adaptive K: generate K_init first, expand to K_max if low agreement.

    Returns (samples, agreement, k_used, expanded).
    """
    from openai import OpenAI
    from collections import Counter

    pre_encoded = _pre_encode_messages(messages)

    def _batch_call(n):
        try:
            bot = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1", timeout=600)
            resp = bot.chat.completions.create(
                model=model_name,
                messages=pre_encoded,
                n=n,
                temperature=temperature,
                extra_body={"top_k": 50},
            )
            results = []
            for choice in resp.choices:
                response = choice.message.content
                try:
                    pred = safe_parse_response(fm_obj, response)
                    results.append({
                        'response': response,
                        'pred_action': pred['action_content'],
                        'parse_ok': True,
                    })
                except Exception:
                    results.append({'response': response, 'pred_action': None, 'parse_ok': False})
            return results
        except Exception:
            return _generate_k_parallel(pre_encoded, model_name, n, temperature, fm_obj)

    # Phase 1
    samples = _batch_call(K_init)
    action_types = [s['pred_action'].get('action', '?') for s in samples if s['pred_action'] and s['parse_ok']]
    if action_types:
        counter = Counter(action_types)
        agreement = counter.most_common(1)[0][1] / len(action_types)
    else:
        agreement = 0.0

    if agreement >= threshold:
        return samples, agreement, K_init, False

    # Phase 2: expand
    extra = _batch_call(K_max - K_init)
    samples.extend(extra)
    action_types = [s['pred_action'].get('action', '?') for s in samples if s['pred_action'] and s['parse_ok']]
    if action_types:
        counter = Counter(action_types)
        agreement = counter.most_common(1)[0][1] / len(action_types)
    else:
        agreement = 0.0

    return samples, agreement, K_max, True


def compute_trajectory_metrics(results):
    """Compute standard trajectory metrics from results list.

    Each result should have: task_success, final_step_id, num_steps.

    Returns dict with TSR, avg_progress, scattered_progress.
    """
    if not results:
        return {'tsr': 0, 'avg_progress': 0, 'scattered_progress': 0, 'n': 0}

    n = len(results)
    success_count = sum(1 for r in results if r['task_success'])
    tsr = success_count / n

    progresses = [r['final_step_id'] / r['num_steps'] for r in results]
    avg_progress = sum(progresses) / n

    # Scattered progress: fraction of steps correct across all trajectories
    total_steps = sum(r['num_steps'] for r in results)
    total_correct = sum(r['final_step_id'] for r in results)
    scattered_progress = total_correct / total_steps if total_steps > 0 else 0

    return {
        'tsr': tsr,
        'avg_progress': avg_progress,
        'scattered_progress': scattered_progress,
        'n': n,
        'success_count': success_count,
    }
