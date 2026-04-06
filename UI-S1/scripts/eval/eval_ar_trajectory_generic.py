"""Generic AR Trajectory Evaluator for Baseline Models on AndroidControl.

Supports: os-atlas, ui-tars, os-genesis
Both stop-on-error and no-stop modes.
Saves detailed step_results for downstream analysis (cascade, error types,
grounding vs planning).

Usage:
    python eval_ar_trajectory_generic.py \
        --model_type os-atlas \
        --model_name /path/to/OS-Atlas-Pro-7B \
        --jsonl_file /path/to/android_control_evaluation_std.jsonl \
        --output_dir /path/to/output \
        --max_workers 32 \
        --no_stop
"""

import argparse
import copy
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from PIL import Image

# Add project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'evaluation'))

from agentcpm_utils import map_action_space2qwenvl
from qwenvl_utils import evaluate_android_control_action

# Model-specific imports (lazy, see get_adapter)
_os_atlas_utils = None
_ui_tars_utils = None
_os_genesis_utils = None

result_lock = Lock()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ac_trajectories(jsonl_path, max_episodes=None):
    """Load AndroidControl episodes, fix image paths and check_options."""
    image_root = os.path.join(PROJECT_ROOT, 'datasets')
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            episode = json.loads(line.strip())
            for step in episode['steps']:
                # Fix image paths
                if step['screenshot'].startswith('/datasets/'):
                    step['screenshot'] = step['screenshot'].replace(
                        '/datasets/', image_root + '/', 1
                    )
                # Ensure check_options exists
                if 'check_options' not in step:
                    check_options = copy.deepcopy(step['action_content'])
                    if 'bbox' in step:
                        check_options['candidate_bbox'] = step['bbox']
                    elif 'candidate_bbox' not in check_options:
                        check_options['candidate_bbox'] = []
                    step['check_options'] = check_options
            data.append(episode)
            if max_episodes and len(data) >= max_episodes:
                break
    return data


def length_bucket(n):
    if n <= 3:
        return 'short(1-3)'
    elif n <= 7:
        return 'medium(4-7)'
    elif n <= 15:
        return 'long(8-15)'
    else:
        return 'vlong(16+)'


# ---------------------------------------------------------------------------
# Model adapters
# ---------------------------------------------------------------------------

def _get_os_atlas_utils():
    global _os_atlas_utils
    if _os_atlas_utils is None:
        from os_atlas_utils import (
            predict, os_atlas_2minicpm, build_history_actions_str
        )
        _os_atlas_utils = {
            'predict': predict,
            'parse': os_atlas_2minicpm,
            'build_history': build_history_actions_str,
        }
    return _os_atlas_utils


def _get_ui_tars_utils():
    global _ui_tars_utils
    if _ui_tars_utils is None:
        from ui_tars_utils import (
            predict, uitars2minicpm, extract_thought_action
        )
        _ui_tars_utils = {
            'predict': predict,
            'parse': uitars2minicpm,
            'extract_thought_action': extract_thought_action,
        }
    return _ui_tars_utils


def _get_os_genesis_utils():
    global _os_genesis_utils
    if _os_genesis_utils is None:
        from os_genesis_utils import (
            predict, os_gensis_2minicpm
        )
        _os_genesis_utils = {
            'predict': predict,
            'parse': os_gensis_2minicpm,
        }
    return _os_genesis_utils


class ModelAdapter:
    """Base class for model-specific adapters."""

    def __init__(self, model_name):
        self.model_name = model_name

    def init_history(self):
        """Return initial history state."""
        raise NotImplementedError

    def predict(self, instruction, step, history_state, image):
        """Call model, return raw response string."""
        raise NotImplementedError

    def parse_response(self, response, width, height):
        """Parse raw response -> (pred_action_dict, raw_action_str).
        pred_action_dict is in RAW_SPACE format (output of map_action_space2qwenvl).
        """
        raise NotImplementedError

    def update_history(self, history_state, step, response, pred_action_str):
        """Update history state after a step. Mutates in-place or returns new."""
        raise NotImplementedError


class OSAtlasAdapter(ModelAdapter):
    """Adapter for OS-Atlas-Pro-7B / OS-Atlas-Base-7B."""

    @property
    def coord_format(self):
        return 'relative_1000'

    def init_history(self):
        return []  # list of step instruction strings

    def predict(self, instruction, step, history_state, image):
        utils = _get_os_atlas_utils()
        history_str = utils['build_history'](history_state)
        return utils['predict'](
            model_name=self.model_name,
            instruction=instruction,
            low_instruction='',
            history=history_str,
            image=image,
        )

    def parse_response(self, response, width, height):
        utils = _get_os_atlas_utils()
        action_minicpm, action_str = utils['parse'](response)
        pred_action = map_action_space2qwenvl(
            action_minicpm, [width, height],
            coordinate_format='relative_1000',
        )
        return pred_action, action_str

    def update_history(self, history_state, step, response, pred_action_str):
        low = step.get('step_instruction', '')
        history_state.append(low)
        return history_state


class UITarsAdapter(ModelAdapter):
    """Adapter for UI-TARS-7B-DPO."""

    @property
    def coord_format(self):
        return 'relative_1000'  # uitars2minicpm extracts from box format

    def init_history(self):
        return []  # list of dicts {image_path, low_instruction, action}

    def predict(self, instruction, step, history_state, image):
        utils = _get_ui_tars_utils()
        return utils['predict'](
            model_name=self.model_name,
            instruction=instruction,
            low_instruction='',
            history_list=history_state,
            image=image,
        )

    def parse_response(self, response, width, height):
        utils = _get_ui_tars_utils()
        action_minicpm, action_str = utils['parse'](response)
        pred_action = map_action_space2qwenvl(
            action_minicpm, [width, height],
        )
        return pred_action, action_str

    def update_history(self, history_state, step, response, pred_action_str):
        utils = _get_ui_tars_utils()
        _, action = utils['extract_thought_action'](response)
        history_state.append({
            'image_path': step['screenshot'],
            'low_instruction': step.get('step_instruction', ''),
            'action': action,
        })
        return history_state


class OSGenesisAdapter(ModelAdapter):
    """Adapter for OS-Genesis-7B-AC."""

    @property
    def coord_format(self):
        return 'absolute'

    def init_history(self):
        return []  # list of strings (not effectively used by OS-Genesis)

    def predict(self, instruction, step, history_state, image):
        utils = _get_os_genesis_utils()
        return utils['predict'](
            model_name=self.model_name,
            instruction=instruction,
            low_instruction='',
            history='',  # OS-Genesis doesn't use history effectively
            image=image,
        )

    def parse_response(self, response, width, height):
        utils = _get_os_genesis_utils()
        action_minicpm = utils['parse'](response)
        # os_gensis_2minicpm returns just dict, not tuple
        pred_action = map_action_space2qwenvl(
            action_minicpm, [width, height],
            coordinate_format='absolute',
        )
        return pred_action, str(action_minicpm)

    def update_history(self, history_state, step, response, pred_action_str):
        low = step.get('step_instruction', '')
        history_state.append(low)
        return history_state


ADAPTERS = {
    'os-atlas': OSAtlasAdapter,
    'ui-tars': UITarsAdapter,
    'os-genesis': OSGenesisAdapter,
}


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------

def _json_default(obj):
    """Handle numpy types for JSON serialization."""
    try:
        import numpy as np
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def process_episode(episode, adapter, args):
    """Process a single episode with AR evaluation."""
    num_steps = len(episode['steps'])
    history_state = adapter.init_history()
    step_results = []

    try:
        for step_id in range(num_steps):
            step = episode['steps'][step_id]
            current_check = step['check_options']
            gt_action = step['action_content']

            # Load image
            image = Image.open(step['screenshot'])
            width, height = image.size

            # Predict
            response = adapter.predict(
                instruction=episode['goal'],
                step=step,
                history_state=history_state,
                image=image,
            )

            if not response:
                # Model returned empty (max retries exhausted)
                step_results.append({
                    'step_num': step_id,
                    'type_match': False,
                    'extract_match': False,
                    'pred_action': {'action': 'wait', 'time': 0.1},
                    'gt_action': gt_action,
                    'gt_action_type': gt_action.get('action', 'unknown'),
                })
                if not args.no_stop:
                    break
                continue

            # Parse response
            try:
                pred_action, action_str = adapter.parse_response(response, width, height)
            except Exception as e:
                print(f"  Parse error: {e}")
                pred_action = {'action': 'wait', 'time': 0.1}
                action_str = ''

            # Evaluate
            type_match, extract_match = evaluate_android_control_action(
                pred_action, current_check,
                width, height,
                width, height,
                ignore_actions=[],
            )
            type_match = bool(type_match)
            extract_match = bool(extract_match)

            step_results.append({
                'step_num': step_id,
                'type_match': type_match,
                'extract_match': extract_match,
                'pred_action': pred_action,
                'gt_action': gt_action,
                'gt_action_type': gt_action.get('action', 'unknown'),
            })

            # Update history
            adapter.update_history(history_state, step, response, action_str)

            # Stop on error (unless no_stop)
            if not extract_match and not args.no_stop:
                break

    except Exception as e:
        print(f"Error episode {episode.get('episode_id', '?')}: {e}")
        traceback.print_exc()

    # Compute metrics
    correct_steps = sum(1 for s in step_results if s['extract_match'])
    # For stop mode: final_step_id = number of consecutive correct from start
    if args.no_stop:
        final_step_id = correct_steps
    else:
        final_step_id = 0
        for s in step_results:
            if s['extract_match']:
                final_step_id += 1
            else:
                break

    task_success = (correct_steps == num_steps and len(step_results) == num_steps)

    result = {
        'episode_id': episode.get('episode_id', None),
        'goal': episode['goal'],
        'num_steps': num_steps,
        'task_success': task_success,
        'final_step_id': final_step_id,
        'correct_steps': correct_steps,
        'evaluated_steps': len(step_results),
        'step_results': step_results,
        'length_bucket': length_bucket(num_steps),
    }

    with result_lock:
        out_path = os.path.join(args.output_dir, 'trajectory_results.jsonl')
        with open(out_path, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False, default=_json_default) + '\n')

    return result


def compute_trajectory_metrics(results):
    """Compute standard trajectory metrics."""
    if not results:
        return {'tsr': 0, 'avg_progress': 0, 'scattered_progress': 0, 'n': 0, 'success_count': 0}

    n = len(results)
    success_count = sum(1 for r in results if r['task_success'])
    tsr = success_count / n

    progresses = [r['final_step_id'] / r['num_steps'] for r in results]
    avg_progress = sum(progresses) / n

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


def compute_nostop_metrics(results):
    """Compute no-stop specific metrics (step accuracy, step-0 failure, post-error acc)."""
    total_steps = 0
    total_correct = 0
    step0_failures = 0
    step0_total = 0
    post_error_correct = 0
    post_error_total = 0

    for r in results:
        sr = r['step_results']
        if not sr:
            continue

        for i, s in enumerate(sr):
            total_steps += 1
            if s['extract_match']:
                total_correct += 1
            if i == 0:
                step0_total += 1
                if not s['extract_match']:
                    step0_failures += 1

        # Post-error: steps after first error
        first_error = None
        for i, s in enumerate(sr):
            if not s['extract_match']:
                first_error = i
                break
        if first_error is not None:
            for s in sr[first_error + 1:]:
                post_error_total += 1
                if s['extract_match']:
                    post_error_correct += 1

    step_accuracy = total_correct / total_steps if total_steps > 0 else 0
    step0_fail_rate = step0_failures / step0_total if step0_total > 0 else 0
    post_error_acc = post_error_correct / post_error_total if post_error_total > 0 else 0

    # Scattered progress (per-trajectory average of correct/total)
    scattered_list = []
    for r in results:
        sr = r['step_results']
        if sr:
            c = sum(1 for s in sr if s['extract_match'])
            scattered_list.append(c / len(sr))
    scattered_progress = sum(scattered_list) / len(scattered_list) if scattered_list else 0

    return {
        'step_accuracy': step_accuracy,
        'total_steps': total_steps,
        'total_correct': total_correct,
        'step0_failure_rate': step0_fail_rate,
        'step0_failures': step0_failures,
        'step0_total': step0_total,
        'post_error_accuracy': post_error_acc,
        'post_error_correct': post_error_correct,
        'post_error_total': post_error_total,
        'scattered_progress': scattered_progress,
    }


def main(args):
    # Create adapter
    adapter_cls = ADAPTERS.get(args.model_type)
    if adapter_cls is None:
        print(f"ERROR: Unknown model_type '{args.model_type}'. Available: {list(ADAPTERS.keys())}")
        sys.exit(1)
    adapter = adapter_cls(args.model_name)

    os.makedirs(args.output_dir, exist_ok=True)

    # Clear previous results
    out_path = os.path.join(args.output_dir, 'trajectory_results.jsonl')
    if os.path.exists(out_path):
        os.remove(out_path)

    # Load data
    data = load_ac_trajectories(args.jsonl_file, max_episodes=args.max_episodes)
    print(f"Loaded {len(data)} episodes. Model: {args.model_type} ({args.model_name})")
    print(f"Mode: {'no_stop' if args.no_stop else 'stop_on_error'}")
    print(f"Workers: {args.max_workers}")

    # Evaluate
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_episode, ep, adapter, args): ep for ep in data}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                if len(results) % 50 == 0:
                    metrics = compute_trajectory_metrics(results)
                    print(
                        f"Progress: {len(results)}/{len(data)} | "
                        f"TSR: {metrics['tsr']:.4f} | "
                        f"AvgProg: {metrics['avg_progress']:.4f}"
                    )
            except Exception as e:
                print(f"Exception: {e}")

    # Compute summary
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
        'model_type': args.model_type,
        'mode': 'no_stop' if args.no_stop else 'stop_on_error',
        'total_episodes': len(results),
        **metrics,
        'action_type_stats': action_stats,
        'length_bucket_stats': length_metrics,
    }

    # Add no-stop metrics if applicable
    if args.no_stop:
        nostop_metrics = compute_nostop_metrics(results)
        summary['nostop_metrics'] = nostop_metrics

    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=_json_default)

    print(f"\nEvaluation completed.")
    print(f"TSR: {metrics['tsr']:.4f} ({metrics['success_count']}/{metrics['n']})")
    print(f"Avg Progress: {metrics['avg_progress']:.4f}")
    print(f"Scattered Progress: {metrics['scattered_progress']:.4f}")

    if args.no_stop and 'nostop_metrics' in summary:
        ns = summary['nostop_metrics']
        print(f"Step Accuracy: {ns['step_accuracy']:.4f} ({ns['total_correct']}/{ns['total_steps']})")
        print(f"Step-0 Failure Rate: {ns['step0_failure_rate']:.4f}")
        print(f"Post-Error Accuracy: {ns['post_error_accuracy']:.4f}")

    print(f"\nPer-action-type extract_match rates:")
    for at in sorted(action_stats.keys()):
        s = action_stats[at]
        print(f"  {at}: {s['extract_match_rate']:.3f} ({s['extract_match']}/{s['total']})")

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generic AR Trajectory Evaluator for Baseline Models on AndroidControl"
    )
    parser.add_argument("--model_type", type=str, required=True,
                        choices=list(ADAPTERS.keys()),
                        help="Model type: os-atlas, ui-tars, os-genesis")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name/path for vLLM")
    parser.add_argument("--jsonl_file", type=str,
                        default=os.path.join(PROJECT_ROOT,
                                             'datasets',
                                             'android_control_evaluation_std.jsonl'),
                        help="Path to evaluation JSONL")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--max_workers", type=int, default=32,
                        help="Number of parallel threads")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Limit episodes for testing")
    parser.add_argument("--no_stop", action='store_true',
                        help="Continue evaluating after errors (no-stop mode)")
    args = parser.parse_args()
    main(args)
