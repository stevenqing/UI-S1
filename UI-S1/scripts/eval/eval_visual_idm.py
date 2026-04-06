"""
Visual Inverse Dynamics Model (IDM) Evaluation for GUI Agents.

Evaluates whether VLMs can understand visual state transitions:
- Standard: model(s_t) → predict action (normal eval, step-level)
- Hindsight: model(s_t, s_{t+1}) → predict action (with future screenshot)
- Pure IDM: model(s_t, s_{t+1}) → predict action (no task context/history)

Usage:
    python eval_visual_idm.py \
        --model_name <model> \
        --jsonl_file <dataset> \
        --mode standard|hindsight|pure_idm \
        --max_samples 500
"""

import argparse
import copy
import json
import os
import random
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from PIL import Image, ImageFilter
import tempfile
import shutil
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../evaluation')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from qwenvl_utils import (call_mobile_agent_vllm,
                           evaluate_android_control_action, find_last_image_ele)
from x.data.agent.json import JsonFormat, STEP_PREDICTION, generate_prompt
from x.qwen.data_format import slim_messages
from x.qwen.image import make_qwen_image_item

result_lock = Lock()

# Module-level state for causal probing ablation
_temp_dir = None
_all_screenshot_paths = None


def precompute_screenshot_pool(lines):
    """Collect all screenshot paths from dataset for random ablation."""
    global _all_screenshot_paths
    paths = []
    for line in lines:
        for step in line['steps']:
            if 'screenshot' in step and step['screenshot']:
                paths.append(step['screenshot'])
    _all_screenshot_paths = paths
    print(f"Precomputed screenshot pool: {len(paths)} images")


def apply_ablation(ablation, st_screenshot, st1_screenshot, episode_steps):
    """
    Apply causal probing ablation to s_{t+1} screenshot.

    Conditions:
      blur         — Gaussian blur σ=20, preserves layout but loses text/icon detail
      lowres       — 32×32 downsampled → resized back, only color distribution preserved
      random       — Random screenshot from different episode (wrong s_{t+1})
      copy_st      — s_t as s_{t+1} (zero transition information)
      same_ep_rand — Random screenshot from same episode, different step

    Returns: path to (possibly ablated) s_{t+1} screenshot.
    """
    if ablation in ('none', 'full'):
        return st1_screenshot

    if ablation == 'copy_st':
        return st_screenshot

    if ablation == 'blur':
        img = Image.open(st1_screenshot).convert('RGB')
        ablated = img.filter(ImageFilter.GaussianBlur(radius=20))
        out_path = os.path.join(_temp_dir, f"{uuid.uuid4().hex}.png")
        ablated.save(out_path)
        img.close()
        return out_path

    if ablation == 'lowres':
        img = Image.open(st1_screenshot).convert('RGB')
        orig_size = img.size
        ablated = img.resize((32, 32), Image.BILINEAR).resize(orig_size, Image.BILINEAR)
        out_path = os.path.join(_temp_dir, f"{uuid.uuid4().hex}.png")
        ablated.save(out_path)
        img.close()
        return out_path

    if ablation == 'random':
        episode_set = set(s['screenshot'] for s in episode_steps if 'screenshot' in s)
        pool = [p for p in _all_screenshot_paths if p not in episode_set]
        if not pool:
            pool = _all_screenshot_paths
        rng = random.Random(hash(st1_screenshot) + 42)
        return rng.choice(pool)

    if ablation == 'same_ep_rand':
        pool = [s['screenshot'] for s in episode_steps
                if 'screenshot' in s
                and s['screenshot'] != st_screenshot
                and s['screenshot'] != st1_screenshot]
        if not pool:
            # Fallback to cross-episode random if episode too short
            rng = random.Random(hash(st1_screenshot) + 42)
            episode_set = set(s['screenshot'] for s in episode_steps if 'screenshot' in s)
            cross_pool = [p for p in _all_screenshot_paths if p not in episode_set]
            return rng.choice(cross_pool) if cross_pool else st1_screenshot
        rng = random.Random(hash(st1_screenshot) + 42)
        return rng.choice(pool)

    raise ValueError(f"Unknown ablation type: {ablation}")


def fix_line(line):
    for step in line['steps']:
        check_options = copy.deepcopy(step['action_content'])
        if 'candidate_bbox' not in step:
            if 'bbox' in step:
                check_options['candidate_bbox'] = step['bbox']
            else:
                check_options['candidate_bbox'] = []
        else:
            check_options['candidate_bbox'] = step['candidate_bbox']
        step['check_options'] = check_options
    return line


def construct_pure_idm_messages(step, next_step, action_space_prompt):
    """Construct pure IDM prompt: only s_t, s_{t+1}, action space. No task or history."""
    system_prompt = STEP_PREDICTION.format(action_space_prompt)

    messages = [
        {
            'role': 'system',
            'content': [{'text': system_prompt}]
        },
        {
            'role': 'user',
            'content': [
                {'text': 'Pre-operation screenshot:\n'},
                make_qwen_image_item(step['screenshot']),
                {'text': '\nPost-operation screenshot:\n'},
                make_qwen_image_item(next_step['screenshot']),
            ]
        }
    ]
    return messages


def process_step(task_info, args, fm, action_space_prompt):
    """Process a single (episode, step) pair."""
    line = task_info['line']
    step_id = task_info['step_id']
    fixed_line = fix_line(copy.deepcopy(line))

    step = fixed_line['steps'][step_id]
    gt_action = step['check_options']
    has_next = step_id + 1 < len(fixed_line['steps'])

    result = {
        'goal': line['goal'],
        'step_id': step_id,
        'num_steps': len(line['steps']),
        'gt_action_type': gt_action['action'],
        'has_next_screenshot': has_next,
        'mode': args.mode,
    }

    # Skip if mode needs s_{t+1} but it's not available
    if args.mode in ('hindsight', 'pure_idm') and not has_next:
        result['skipped'] = True
        result['skip_reason'] = 'no_next_screenshot'
        return result

    # Apply causal probing ablation to s_{t+1} if requested
    ablation = getattr(args, 'ablation', 'none')
    result['ablation'] = ablation

    try:
        if ablation not in ('none', 'full') and has_next and args.mode in ('hindsight', 'pure_idm'):
            ablated_path = apply_ablation(
                ablation=ablation,
                st_screenshot=step['screenshot'],
                st1_screenshot=fixed_line['steps'][step_id + 1]['screenshot'],
                episode_steps=fixed_line['steps'],
            )
            fixed_line['steps'][step_id + 1]['screenshot'] = ablated_path
        if args.mode == 'pure_idm':
            # Pure IDM: only s_t and s_{t+1}, no task context
            next_step = fixed_line['steps'][step_id + 1]
            messages = construct_pure_idm_messages(step, next_step, action_space_prompt)
        else:
            # Standard or Hindsight: use gen_next_round
            # Build state up to step_id using GT responses for previous steps
            state = None
            for si in range(step_id + 1):
                if si < step_id:
                    # Use GT response for previous steps
                    state = fm.gen_next_round(
                        fixed_line, state,
                        previous_model_response=None,
                        hindsight=False
                    )
                else:
                    # Current step: apply mode
                    use_hindsight = (args.mode == 'hindsight')
                    state = fm.gen_next_round(
                        fixed_line, state,
                        previous_model_response=None,
                        hindsight=use_hindsight
                    )

            if state is None:
                result['skipped'] = True
                result['skip_reason'] = 'gen_next_round_failed'
                return result

            messages = state['messages']
            # Limit images: standard=2, hindsight=3 (need room for s_{t+1})
            n_img_limit = 3 if args.mode == 'hindsight' else 2
            messages = slim_messages(messages=messages, num_image_limit=n_img_limit)

        # Find image dimensions for coordinate normalization
        current_image_ele, width, height, resized_width, resized_height = find_last_image_ele(
            # For hindsight/pure_idm, the last image is s_{t+1}, we need s_t dimensions
            # But find_last_image_ele returns the last image which is s_{t+1}
            # For coordinate normalization we need the current step's image
            messages if args.mode == 'standard' else messages
        )

        # For hindsight/pure_idm modes, we need s_t's dimensions, not s_{t+1}
        if args.mode in ('hindsight', 'pure_idm'):
            img = Image.open(step['screenshot'])
            width, height = img.size
            from x.qwen.image import smart_resize
            resized_height, resized_width = smart_resize(height, width, max_pixels=12800*28*28)

        # Call model
        model_response = call_mobile_agent_vllm(
            messages=messages,
            model_name=args.model_name
        )
        result['model_response'] = model_response

        # Parse response
        pred_action = fm.parse_response(model_response)
        result['pred_action'] = pred_action.get('action_content', {})

        # Evaluate
        type_match, extract_match = evaluate_android_control_action(
            pred_action['action_content'],
            gt_action,
            width, height,
            resized_width, resized_height
        )
        result['type_match'] = type_match
        result['extract_match'] = extract_match
        result['skipped'] = False

    except Exception as e:
        result['skipped'] = True
        result['skip_reason'] = f'error: {str(e)}'
        traceback.print_exc()

    return result


def sample_steps(lines, max_samples, seed=42):
    """Sample steps proportionally by action type, ensuring min 30 per type."""
    random.seed(seed)

    # Collect all eligible steps (non-terminal with s_{t+1})
    all_steps = []
    by_type = defaultdict(list)
    for line in lines:
        fixed_line = fix_line(copy.deepcopy(line))
        for si, step in enumerate(fixed_line['steps']):
            if si + 1 < len(fixed_line['steps']):  # has s_{t+1}
                action_type = step['action_content']['action']
                entry = {'line': line, 'step_id': si, 'action_type': action_type}
                all_steps.append(entry)
                by_type[action_type].append(entry)

    print(f"Total eligible steps: {len(all_steps)}")
    for at, items in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"  {at:15s}: {len(items)}")

    if max_samples >= len(all_steps):
        return all_steps

    # Ensure min 30 per type, then proportional fill
    sampled = []
    remaining_budget = max_samples
    min_per_type = 30

    for at, items in by_type.items():
        n_take = min(min_per_type, len(items))
        sampled.extend(random.sample(items, n_take))
        remaining_budget -= n_take

    # Fill remaining budget proportionally
    already_sampled_set = set(id(s) for s in sampled)
    remaining_pool = [s for s in all_steps if id(s) not in already_sampled_set]
    if remaining_budget > 0 and remaining_pool:
        extra = random.sample(remaining_pool, min(remaining_budget, len(remaining_pool)))
        sampled.extend(extra)

    random.shuffle(sampled)
    print(f"Sampled {len(sampled)} steps")
    return sampled


def main(args):
    global _temp_dir

    from x.data.agent.space.std_space import RAW_SPACE
    fm = JsonFormat(RAW_SPACE, add_thought=True, force_add_thought=True)
    action_space_prompt = generate_prompt(RAW_SPACE)

    # Load data
    lines = []
    with open(args.jsonl_file, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    print(f"Loaded {len(lines)} episodes")

    # Setup causal probing ablation
    ablation = getattr(args, 'ablation', 'none')
    if ablation in ('blur', 'lowres'):
        _temp_dir = tempfile.mkdtemp(prefix='vidm_ablation_')
        print(f"Ablation temp dir: {_temp_dir}")
    if ablation in ('random', 'same_ep_rand'):
        precompute_screenshot_pool(lines)

    # Sample steps
    if args.mode == 'standard':
        # For standard mode, also include last steps (no s_{t+1} needed)
        all_steps = []
        for line in lines:
            fixed_line = fix_line(copy.deepcopy(line))
            for si in range(len(fixed_line['steps'])):
                all_steps.append({'line': line, 'step_id': si, 'action_type': fixed_line['steps'][si]['action_content']['action']})
        if args.max_samples < len(all_steps):
            random.seed(42)
            steps = random.sample(all_steps, args.max_samples)
        else:
            steps = all_steps
    else:
        steps = sample_steps(lines, args.max_samples)

    print(f"\n{'='*60}")
    print(f"Visual IDM Evaluation: mode={args.mode}, model={args.model_name}")
    print(f"Steps to evaluate: {len(steps)}")
    print(f"{'='*60}\n")

    # Output setup
    os.makedirs(args.output_dir, exist_ok=True)
    abl_suffix = f"_{ablation}" if ablation not in ('none', 'full') else ""
    output_file = os.path.join(args.output_dir, f"vidm_{args.mode}_{args.model_name}{abl_suffix}.jsonl")
    # Clear output file
    with open(output_file, 'w') as f:
        pass

    # Process steps
    results = []

    def process_and_save(task_info):
        result = process_step(task_info, args, fm, action_space_prompt)
        with result_lock:
            with open(output_file, 'a') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        return result

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_and_save, s): s for s in steps}
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
                if (i + 1) % 50 == 0:
                    n_eval = sum(1 for r in results if not r.get('skipped', True))
                    n_tm = sum(1 for r in results if r.get('type_match', False))
                    n_em = sum(1 for r in results if r.get('extract_match', False))
                    print(f"Progress: {i+1}/{len(steps)} | "
                          f"type_match={n_tm}/{n_eval} ({n_tm/max(n_eval,1):.1%}) | "
                          f"extract_match={n_em}/{n_eval} ({n_em/max(n_eval,1):.1%})")
            except Exception as e:
                print(f"Exception: {e}")

    # Compute statistics
    print(f"\n{'='*60}")
    print(f"RESULTS: mode={args.mode}, model={args.model_name}")
    print(f"{'='*60}")

    evaluated = [r for r in results if not r.get('skipped', True)]
    skipped = [r for r in results if r.get('skipped', True)]
    print(f"Total: {len(results)}, Evaluated: {len(evaluated)}, Skipped: {len(skipped)}")

    if not evaluated:
        print("No results to report.")
        return

    # Overall
    n_type_match = sum(1 for r in evaluated if r['type_match'])
    n_extract_match = sum(1 for r in evaluated if r['extract_match'])
    print(f"\nOverall:")
    print(f"  type_match:    {n_type_match}/{len(evaluated)} = {n_type_match/len(evaluated):.1%}")
    print(f"  extract_match: {n_extract_match}/{len(evaluated)} = {n_extract_match/len(evaluated):.1%}")

    # Per action type
    by_type = defaultdict(list)
    for r in evaluated:
        by_type[r['gt_action_type']].append(r)

    print(f"\nPer action type:")
    print(f"  {'type':15s} {'count':>6s} {'type_match':>12s} {'extract_match':>14s}")
    print(f"  {'-'*15} {'-'*6} {'-'*12} {'-'*14}")
    for at in sorted(by_type.keys(), key=lambda x: -len(by_type[x])):
        items = by_type[at]
        tm = sum(1 for r in items if r['type_match'])
        em = sum(1 for r in items if r['extract_match'])
        print(f"  {at:15s} {len(items):6d} {tm:5d} ({tm/len(items):5.1%}) {em:5d} ({em/len(items):5.1%})")

    # Save summary
    summary = {
        'mode': args.mode,
        'model_name': args.model_name,
        'ablation': ablation,
        'n_total': len(results),
        'n_evaluated': len(evaluated),
        'n_skipped': len(skipped),
        'overall_type_match': n_type_match / len(evaluated),
        'overall_extract_match': n_extract_match / len(evaluated),
        'per_type': {
            at: {
                'count': len(items),
                'type_match': sum(1 for r in items if r['type_match']) / len(items),
                'extract_match': sum(1 for r in items if r['extract_match']) / len(items),
            }
            for at, items in by_type.items()
        }
    }
    summary_file = os.path.join(args.output_dir, f"vidm_{args.mode}_{args.model_name}{abl_suffix}_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")
    print(f"Results saved to: {output_file}")

    # Cleanup temp directory for ablation
    if _temp_dir and os.path.exists(_temp_dir):
        shutil.rmtree(_temp_dir)
        print(f"Cleaned up temp dir: {_temp_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual IDM evaluation for GUI agents")
    parser.add_argument("--jsonl_file", type=str, required=True, help="Dataset JSONL file")
    parser.add_argument("--output_dir", type=str, default="evaluation/results/visual_idm", help="Output directory")
    parser.add_argument("--model_name", type=str, required=True, help="vLLM model name")
    parser.add_argument("--mode", type=str, required=True, choices=['standard', 'hindsight', 'pure_idm'],
                        help="Evaluation mode: standard (s_t only), hindsight (s_t + s_{t+1}), pure_idm (no task context)")
    parser.add_argument("--max_samples", type=int, default=500, help="Max steps to evaluate")
    parser.add_argument("--max_workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--n_history_image_limit", type=int, default=2, help="Max history images for standard mode")
    parser.add_argument("--ablation", type=str, default="none",
                        choices=['none', 'full', 'blur', 'lowres', 'random', 'copy_st', 'same_ep_rand'],
                        help="Causal probing ablation for s_{t+1}: "
                             "blur=Gaussian σ=20, lowres=32x32→resize, "
                             "random=wrong episode, copy_st=duplicate s_t, "
                             "same_ep_rand=wrong step same episode")

    args = parser.parse_args()
    main(args)
