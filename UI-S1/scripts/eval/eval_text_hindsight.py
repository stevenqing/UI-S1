"""
Text Hindsight Evaluation: Replace s_{t+1} screenshot with pi_V text description.

Compares three conditions using the same evaluation steps:
  Standard:         model(s_t) -> predict action          (V8: 46.5%)
  Visual hindsight: model(s_t, image(s_{t+1})) -> action  (V8: 66.0%)
  Text hindsight:   model(s_t, desc(s_{t+1})) -> action   (V8: ?)

If text_hindsight >= visual_hindsight -> textual OPD is viable for training.

Pipeline:
  Phase A: pi_V(s_{t+1}) -> desc_t1  (image in, text out)
  Phase B: model(standard_prompt + desc_t1) -> predict action  (no s_{t+1} image)
"""

import argparse
import base64
import copy
import json
import os
import random
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from threading import Lock

from openai import OpenAI
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../evaluation')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from qwenvl_utils import (call_mobile_agent_vllm,
                           evaluate_android_control_action)
from x.data.agent.json import JsonFormat, generate_prompt
from x.qwen.data_format import slim_messages
from x.qwen.image import smart_resize

result_lock = Lock()

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DESCRIBE_PROMPT = """Describe the current state of this mobile app screen in detail.
1. What app is this? What screen/page is shown?
2. List the main UI elements visible (buttons, text fields, lists, icons, images).
3. Note any text content displayed (titles, labels, input field values, list items).
4. Describe any active states: open keyboards, dialogs, menus, loading indicators, selected items.
Be specific and concise."""

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def encode_screenshot(path):
    """Encode screenshot as base64 data URL for OpenAI API."""
    img = Image.open(path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def call_vllm_describe(client, model_name, image_path, max_tokens=512):
    """Call vLLM to generate pi_V description of a screenshot."""
    data_url = encode_screenshot(image_path)
    content = [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": DESCRIBE_PROMPT},
    ]
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                extra_body={"top_k": 1},
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < 2:
                import time; time.sleep(3)
            else:
                raise


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


def sample_steps(lines, max_samples, seed=42):
    """Sample steps with s_{t+1}, same logic as eval_visual_idm.py."""
    random.seed(seed)
    all_steps = []
    by_type = defaultdict(list)
    for line in lines:
        fixed_line = fix_line(copy.deepcopy(line))
        for si, step in enumerate(fixed_line['steps']):
            if si + 1 < len(fixed_line['steps']):
                action_type = step['action_content']['action']
                entry = {'line': line, 'step_id': si, 'action_type': action_type}
                all_steps.append(entry)
                by_type[action_type].append(entry)

    print(f"Total eligible steps: {len(all_steps)}")
    for at, items in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"  {at:15s}: {len(items)}")

    if max_samples >= len(all_steps):
        return all_steps

    sampled = []
    remaining_budget = max_samples
    min_per_type = 30

    for at, items in by_type.items():
        n_take = min(min_per_type, len(items))
        sampled.extend(random.sample(items, n_take))
        remaining_budget -= n_take

    already_sampled_set = set(id(s) for s in sampled)
    remaining_pool = [s for s in all_steps if id(s) not in already_sampled_set]
    if remaining_budget > 0 and remaining_pool:
        extra = random.sample(remaining_pool, min(remaining_budget, len(remaining_pool)))
        sampled.extend(extra)

    random.shuffle(sampled)
    print(f"Sampled {len(sampled)} steps")
    return sampled


# ---------------------------------------------------------------------------
# Phase A: Generate pi_V descriptions
# ---------------------------------------------------------------------------

def describe_step(task_info, client, model_name):
    """Generate pi_V description of s_{t+1}."""
    line = task_info['line']
    step_id = task_info['step_id']
    fixed = fix_line(copy.deepcopy(line))
    next_step = fixed['steps'][step_id + 1]

    result = {
        'goal': line['goal'],
        'step_id': step_id,
        'gt_action_type': task_info['action_type'],
    }
    try:
        desc_t1 = call_vllm_describe(client, model_name,
                                      image_path=next_step['screenshot'])
        result['desc_t1'] = desc_t1
        result['error'] = False
    except Exception as e:
        result['desc_t1'] = ''
        result['error'] = True
        result['error_msg'] = str(e)
        traceback.print_exc()
    return result


# ---------------------------------------------------------------------------
# Phase B: Text hindsight prediction
# ---------------------------------------------------------------------------

def process_step_text_hindsight(task_info, desc_t1, args, fm):
    """Process a single step with text description (and optionally screenshot) of s_{t+1}."""
    line = task_info['line']
    step_id = task_info['step_id']
    fixed_line = fix_line(copy.deepcopy(line))
    step = fixed_line['steps'][step_id]
    gt_action = step['check_options']

    hybrid = getattr(args, 'hybrid', False)
    mode_name = 'hybrid_hindsight' if hybrid else 'text_hindsight'

    result = {
        'goal': line['goal'],
        'step_id': step_id,
        'num_steps': len(line['steps']),
        'gt_action_type': gt_action['action'],
        'mode': mode_name,
    }

    try:
        # Build messages: hybrid includes visual s_{t+1}, text-only does not
        state = None
        for si in range(step_id + 1):
            state = fm.gen_next_round(
                fixed_line, state,
                previous_model_response=None,
                hindsight=hybrid,  # True for hybrid, False for text-only
            )

        if state is None:
            result['skipped'] = True
            result['skip_reason'] = 'gen_next_round_failed'
            return result

        messages = state['messages']
        n_img_limit = 3 if hybrid else 2
        messages = slim_messages(messages=messages, num_image_limit=n_img_limit)

        # Inject text description of s_{t+1}
        messages[-1]['content'].append({
            "text": f"\nDescription of the screen after correct action:\n{desc_t1}\n"
        })

        # Get s_t dimensions for coordinate normalization
        img = Image.open(step['screenshot'])
        width, height = img.size
        resized_height, resized_width = smart_resize(
            height, width, max_pixels=12800 * 28 * 28)

        # Call model for action prediction
        model_response = call_mobile_agent_vllm(
            messages=messages,
            model_name=args.model_name,
        )
        result['model_response'] = model_response

        # Parse and evaluate
        pred_action = fm.parse_response(model_response)
        result['pred_action'] = pred_action.get('action_content', {})

        type_match, extract_match = evaluate_android_control_action(
            pred_action['action_content'],
            gt_action,
            width, height,
            resized_width, resized_height,
        )
        result['type_match'] = type_match
        result['extract_match'] = extract_match
        result['skipped'] = False

    except Exception as e:
        result['skipped'] = True
        result['skip_reason'] = f'error: {str(e)}'
        traceback.print_exc()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    from x.data.agent.space.std_space import RAW_SPACE
    fm = JsonFormat(RAW_SPACE, add_thought=True, force_add_thought=True)

    # Load data
    lines = []
    with open(args.jsonl_file, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    print(f"Loaded {len(lines)} episodes")

    steps = sample_steps(lines, args.max_samples)

    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1", timeout=600)

    os.makedirs(args.output_dir, exist_ok=True)
    mode_prefix = "hybrid_hs" if args.hybrid else "text_hs"
    desc_file = os.path.join(args.output_dir,
                              f"{mode_prefix}_descriptions_{args.model_name}.jsonl")
    result_file = os.path.join(args.output_dir,
                                f"{mode_prefix}_results_{args.model_name}.jsonl")

    # ---- Phase A: Generate pi_V descriptions of s_{t+1} ----
    print(f"\n{'='*60}")
    print(f"Phase A: Generating pi_V descriptions ({len(steps)} steps)")
    print(f"{'='*60}\n")

    descriptions = {}  # keyed by (goal, step_id)
    with open(desc_file, 'w') as f:
        pass  # clear

    def describe_and_save(task_info):
        result = describe_step(task_info, client, args.model_name)
        key = (result['goal'], result['step_id'])
        with result_lock:
            with open(desc_file, 'a') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        return key, result

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(describe_and_save, s): s for s in steps}
        for i, future in enumerate(as_completed(futures)):
            try:
                key, desc = future.result()
                descriptions[key] = desc
                if (i + 1) % 50 == 0:
                    n_ok = sum(1 for d in descriptions.values()
                               if not d.get('error'))
                    print(f"Phase A progress: {i+1}/{len(steps)}, "
                          f"success: {n_ok}")
            except Exception as e:
                print(f"Phase A exception: {e}")

    n_success = sum(1 for d in descriptions.values() if not d.get('error'))
    print(f"\nPhase A done: {n_success}/{len(descriptions)} descriptions")

    # ---- Phase B: Text Hindsight Prediction ----
    print(f"\n{'='*60}")
    print(f"Phase B: Text Hindsight Prediction ({n_success} steps)")
    print(f"{'='*60}\n")

    results = []
    with open(result_file, 'w') as f:
        pass  # clear

    def predict_and_save(task_info):
        key = (task_info['line']['goal'], task_info['step_id'])
        desc = descriptions.get(key, {})
        if desc.get('error') or not desc.get('desc_t1'):
            return {
                'skipped': True,
                'skip_reason': 'no_description',
                'gt_action_type': task_info['action_type'],
                'goal': task_info['line']['goal'],
                'step_id': task_info['step_id'],
            }
        result = process_step_text_hindsight(
            task_info, desc['desc_t1'], args, fm)
        with result_lock:
            with open(result_file, 'a') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        return result

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(predict_and_save, s): s for s in steps}
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
                if (i + 1) % 50 == 0:
                    n_eval = sum(1 for r in results
                                 if not r.get('skipped', True))
                    n_tm = sum(1 for r in results
                               if r.get('type_match', False))
                    n_em = sum(1 for r in results
                               if r.get('extract_match', False))
                    print(f"Phase B progress: {i+1}/{len(steps)} | "
                          f"type_match={n_tm}/{n_eval} "
                          f"({n_tm/max(n_eval,1):.1%}) | "
                          f"extract_match={n_em}/{n_eval} "
                          f"({n_em/max(n_eval,1):.1%})")
            except Exception as e:
                print(f"Phase B exception: {e}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"RESULTS: {mode_prefix} (model={args.model_name})")
    print(f"{'='*60}")

    evaluated = [r for r in results if not r.get('skipped', True)]
    skipped = [r for r in results if r.get('skipped', True)]
    print(f"Total: {len(results)}, Evaluated: {len(evaluated)}, "
          f"Skipped: {len(skipped)}")

    if not evaluated:
        print("No results to report.")
        return

    n_type_match = sum(1 for r in evaluated if r['type_match'])
    n_extract_match = sum(1 for r in evaluated if r['extract_match'])
    print(f"\nOverall:")
    print(f"  type_match:    {n_type_match}/{len(evaluated)} = "
          f"{n_type_match/len(evaluated):.1%}")
    print(f"  extract_match: {n_extract_match}/{len(evaluated)} = "
          f"{n_extract_match/len(evaluated):.1%}")

    # Per action type + comparison with baselines
    by_type = defaultdict(list)
    for r in evaluated:
        by_type[r['gt_action_type']].append(r)

    # V8 step70 baselines (from existing eval results)
    vis_hs_tm = {
        'type': 91.1, 'open': 85.5, 'swipe': 71.6,
        'system_button': 66.7, 'wait': 58.1, 'click': 0.0, 'long_press': 0.0,
    }
    std_tm = {
        'type': 93.5, 'open': 80.8, 'swipe': 50.0,
        'system_button': 52.2, 'wait': 41.0, 'click': 0.0, 'long_press': 0.0,
    }

    print(f"\nPer action type (Text HS vs Visual HS vs Standard):")
    print(f"  {'type':15s} {'n':>5s} {'text_hs':>10s} {'vis_hs':>10s} "
          f"{'standard':>10s} {'d(t-v)':>8s}")
    print(f"  {'-'*15} {'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    for at in sorted(by_type.keys(), key=lambda x: -len(by_type[x])):
        items = by_type[at]
        tm = sum(1 for r in items if r['type_match'])
        text_pct = tm / len(items) * 100
        vis_pct = vis_hs_tm.get(at, 0)
        std_pct = std_tm.get(at, 0)
        delta = text_pct - vis_pct
        print(f"  {at:15s} {len(items):5d} {text_pct:9.1f}% "
              f"{vis_pct:9.1f}% {std_pct:9.1f}% {delta:+7.1f}%")

    # Click confusion matrix (critical metric)
    click_items = by_type.get('click', [])
    if click_items:
        click_eval = [r for r in click_items if not r.get('skipped', True)]
        if click_eval:
            print(f"\nClick confusion (n={len(click_eval)}):")
            pred_counts = defaultdict(int)
            for r in click_eval:
                pred = r.get('pred_action', {}).get('action', 'unknown')
                pred_counts[pred] += 1
            for pred, cnt in sorted(pred_counts.items(), key=lambda x: -x[1]):
                print(f"  -> {pred:15s}: {cnt} "
                      f"({cnt/len(click_eval):.1%})")

    # Save summary
    summary = {
        'mode': mode_prefix,
        'model_name': args.model_name,
        'n_total': len(results),
        'n_evaluated': len(evaluated),
        'n_skipped': len(skipped),
        'overall_type_match': n_type_match / len(evaluated),
        'overall_extract_match': n_extract_match / len(evaluated),
        'per_type': {
            at: {
                'count': len(items),
                'type_match': sum(1 for r in items
                                  if r['type_match']) / len(items),
                'extract_match': sum(1 for r in items
                                     if r['extract_match']) / len(items),
            }
            for at, items in by_type.items()
        },
    }
    summary_file = os.path.join(args.output_dir,
                                 f"{mode_prefix}_summary_{args.model_name}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDescriptions: {desc_file}")
    print(f"Results: {result_file}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Text Hindsight Eval: pi_V(s_{t+1}) desc replaces screenshot")
    parser.add_argument("--jsonl_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default="evaluation/results/text_hindsight")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--max_workers", type=int, default=64)
    parser.add_argument("--hybrid", action="store_true",
                        help="Hybrid mode: visual s_{t+1} screenshot + text description")
    args = parser.parse_args()
    main(args)
