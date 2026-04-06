"""
Phase 0: Text-Level Verification Quality Assessment

Tests whether π_V descriptions of s_t and s_{t+1} can produce text diffs
that recover the GT action type — the go/no-go gate for text-level verification.

Pipeline:
  Phase A: π_V(s_t) → desc_t, π_V(s_{t+1}) → desc_t1  (with images)
  Phase B: text_idm(desc_t, desc_t1) → predicted action   (text only, no images)

Compare text-level IDM vs visual IDM (click=0% in visual).
"""

import argparse
import base64
import copy
import json
import os
import random
import re
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from threading import Lock

from openai import OpenAI
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

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

TEXT_IDM_PROMPT = """You are given descriptions of two consecutive mobile app screenshots.
Determine what single action was most likely performed between them.

BEFORE the action:
{desc_t}

AFTER the action:
{desc_t1}

Available actions: click, swipe, type, open, system_button, wait, long_press, terminate

Respond ONLY with JSON (no other text):
{{"action": "<type>", "reasoning": "<what changed between the two states>"}}"""


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


def call_vllm(client, model_name, prompt, image_path=None, max_tokens=512):
    """Call vLLM model via OpenAI API."""
    content = []
    if image_path:
        data_url = encode_screenshot(image_path)
        content.append({"type": "image_url", "image_url": {"url": data_url}})
    content.append({"type": "text", "text": prompt})

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


def parse_text_idm_response(response):
    """Extract action type from text IDM JSON response."""
    if not response:
        return 'unknown', ''
    try:
        json_match = re.search(r'\{[^}]*"action"\s*:\s*"([^"]+)"[^}]*\}', response)
        if json_match:
            full_match = re.search(r'\{[^}]+\}', response)
            data = json.loads(full_match.group())
            return data.get('action', 'unknown').lower().strip(), data.get('reasoning', '')
    except:
        pass
    # Fallback: find action keyword
    response_lower = response.lower()
    for action in ['long_press', 'system_button', 'click', 'swipe', 'type', 'open', 'wait', 'terminate']:
        if action in response_lower:
            return action, response
    return 'unknown', response


def fix_line(line):
    for step in line['steps']:
        check_options = copy.deepcopy(step['action_content'])
        if 'candidate_bbox' not in step:
            check_options['candidate_bbox'] = step.get('bbox', [])
        else:
            check_options['candidate_bbox'] = step['candidate_bbox']
        step['check_options'] = check_options
    return line


def sample_steps(lines, max_samples, seed=42):
    """Sample steps with s_{t+1}, min 30 per action type."""
    random.seed(seed)
    all_steps = []
    by_type = defaultdict(list)
    for line in lines:
        fixed = fix_line(copy.deepcopy(line))
        for si, step in enumerate(fixed['steps']):
            if si + 1 < len(fixed['steps']):
                at = step['action_content']['action']
                entry = {'line': line, 'step_id': si, 'action_type': at}
                all_steps.append(entry)
                by_type[at].append(entry)

    print(f"Total eligible steps: {len(all_steps)}")
    for at, items in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"  {at:15s}: {len(items)}")

    if max_samples >= len(all_steps):
        return all_steps

    sampled = []
    remaining = max_samples
    for at, items in by_type.items():
        n = min(30, len(items))
        sampled.extend(random.sample(items, n))
        remaining -= n

    already = set(id(s) for s in sampled)
    pool = [s for s in all_steps if id(s) not in already]
    if remaining > 0 and pool:
        sampled.extend(random.sample(pool, min(remaining, len(pool))))

    random.shuffle(sampled)
    print(f"Sampled {len(sampled)} steps")
    return sampled


# ---------------------------------------------------------------------------
# Phase A: Generate descriptions
# ---------------------------------------------------------------------------

def describe_step(task_info, client, model_name):
    """Generate π_V descriptions for s_t and s_{t+1}."""
    line = task_info['line']
    step_id = task_info['step_id']
    fixed = fix_line(copy.deepcopy(line))
    step = fixed['steps'][step_id]
    next_step = fixed['steps'][step_id + 1]

    result = {
        'goal': line['goal'],
        'step_id': step_id,
        'gt_action_type': step['action_content']['action'],
        'gt_action': step['action_content'],
    }

    try:
        desc_t = call_vllm(client, model_name, DESCRIBE_PROMPT,
                           image_path=step['screenshot'], max_tokens=512)
        desc_t1 = call_vllm(client, model_name, DESCRIBE_PROMPT,
                            image_path=next_step['screenshot'], max_tokens=512)
        result['desc_t'] = desc_t
        result['desc_t1'] = desc_t1
        result['error'] = False
    except Exception as e:
        result['desc_t'] = ''
        result['desc_t1'] = ''
        result['error'] = True
        result['error_msg'] = str(e)
        traceback.print_exc()

    return result


# ---------------------------------------------------------------------------
# Phase B: Text-level IDM
# ---------------------------------------------------------------------------

def text_idm_step(desc_result, client, model_name):
    """Predict action from text descriptions (no images)."""
    if desc_result.get('error') or not desc_result.get('desc_t') or not desc_result.get('desc_t1'):
        desc_result['text_idm_pred'] = 'unknown'
        desc_result['text_idm_reasoning'] = 'skipped (no descriptions)'
        desc_result['text_idm_type_match'] = False
        return desc_result

    prompt = TEXT_IDM_PROMPT.format(
        desc_t=desc_result['desc_t'],
        desc_t1=desc_result['desc_t1'],
    )

    try:
        response = call_vllm(client, model_name, prompt,
                             image_path=None, max_tokens=256)
        pred_action, reasoning = parse_text_idm_response(response)
        desc_result['text_idm_response'] = response
        desc_result['text_idm_pred'] = pred_action
        desc_result['text_idm_reasoning'] = reasoning
        desc_result['text_idm_type_match'] = (
            pred_action.lower() == desc_result['gt_action_type'].lower()
        )
    except Exception as e:
        desc_result['text_idm_pred'] = 'unknown'
        desc_result['text_idm_reasoning'] = f'error: {e}'
        desc_result['text_idm_type_match'] = False
        traceback.print_exc()

    return desc_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # Load data
    lines = []
    with open(args.jsonl_file, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    print(f"Loaded {len(lines)} episodes")

    steps = sample_steps(lines, args.max_samples)

    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1", timeout=600)

    os.makedirs(args.output_dir, exist_ok=True)
    desc_file = os.path.join(args.output_dir, f"phase0_descriptions_{args.model_name}.jsonl")
    result_file = os.path.join(args.output_dir, f"phase0_text_idm_{args.model_name}.jsonl")

    # ---- Phase A: Generate descriptions ----
    print(f"\n{'='*60}")
    print(f"Phase A: Generating π_V descriptions ({len(steps)} steps)")
    print(f"{'='*60}\n")

    descriptions = []
    with open(desc_file, 'w') as f:
        pass  # clear

    def describe_and_save(task_info):
        result = describe_step(task_info, client, args.model_name)
        with result_lock:
            with open(desc_file, 'a') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        return result

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(describe_and_save, s): s for s in steps}
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                descriptions.append(result)
                if (i + 1) % 50 == 0:
                    n_ok = sum(1 for d in descriptions if not d.get('error'))
                    print(f"Phase A progress: {i+1}/{len(steps)}, success: {n_ok}")
            except Exception as e:
                print(f"Phase A exception: {e}")

    n_success = sum(1 for d in descriptions if not d.get('error'))
    print(f"\nPhase A done: {n_success}/{len(descriptions)} descriptions generated")

    # ---- Phase B: Text-level IDM ----
    print(f"\n{'='*60}")
    print(f"Phase B: Text-level IDM ({n_success} steps)")
    print(f"{'='*60}\n")

    results = []
    with open(result_file, 'w') as f:
        pass  # clear

    def idm_and_save(desc_result):
        result = text_idm_step(desc_result, client, args.model_name)
        with result_lock:
            with open(result_file, 'a') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        return result

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(idm_and_save, d): d for d in descriptions}
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
                if (i + 1) % 50 == 0:
                    n_match = sum(1 for r in results if r.get('text_idm_type_match'))
                    n_eval = sum(1 for r in results if r.get('text_idm_pred') != 'unknown')
                    print(f"Phase B progress: {i+1}/{len(descriptions)}, "
                          f"type_match={n_match}/{n_eval} ({n_match/max(n_eval,1):.1%})")
            except Exception as e:
                print(f"Phase B exception: {e}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"RESULTS: Text-Level IDM (Phase 0)")
    print(f"{'='*60}")

    evaluated = [r for r in results if r.get('text_idm_pred') != 'unknown']
    print(f"Total: {len(results)}, Evaluated: {len(evaluated)}")

    if not evaluated:
        print("No results.")
        return

    n_match = sum(1 for r in evaluated if r['text_idm_type_match'])
    print(f"\nOverall type_match: {n_match}/{len(evaluated)} = {n_match/len(evaluated):.1%}")

    # Per action type
    by_type = defaultdict(list)
    for r in evaluated:
        by_type[r['gt_action_type']].append(r)

    print(f"\nPer action type (Text IDM vs Visual IDM baseline):")
    print(f"  {'type':15s} {'n':>5s} {'text_idm':>10s} {'visual_idm':>12s} {'delta':>8s}")
    print(f"  {'-'*15} {'-'*5} {'-'*10} {'-'*12} {'-'*8}")

    # Visual IDM baselines (pure_idm, base model)
    visual_idm = {
        'click': 0.0, 'swipe': 86.2, 'type': 94.4,
        'open': 19.6, 'system_button': 40.0, 'wait': 2.5, 'long_press': 0.0,
    }

    for at in sorted(by_type.keys(), key=lambda x: -len(by_type[x])):
        items = by_type[at]
        tm = sum(1 for r in items if r['text_idm_type_match'])
        text_pct = tm / len(items) * 100
        vis_pct = visual_idm.get(at, 0)
        delta = text_pct - vis_pct
        print(f"  {at:15s} {len(items):5d} {text_pct:9.1f}% {vis_pct:11.1f}% {delta:+7.1f}%")

    # Confusion matrix for click
    click_items = by_type.get('click', [])
    if click_items:
        print(f"\nClick confusion (n={len(click_items)}):")
        pred_counts = defaultdict(int)
        for r in click_items:
            pred_counts[r['text_idm_pred']] += 1
        for pred, cnt in sorted(pred_counts.items(), key=lambda x: -x[1]):
            print(f"  → {pred:15s}: {cnt} ({cnt/len(click_items):.1%})")

    # Save summary
    summary = {
        'model_name': args.model_name,
        'n_total': len(results),
        'n_evaluated': len(evaluated),
        'overall_type_match': n_match / len(evaluated),
        'per_type': {
            at: {
                'count': len(items),
                'type_match': sum(1 for r in items if r['text_idm_type_match']) / len(items),
            }
            for at, items in by_type.items()
        }
    }
    summary_file = os.path.join(args.output_dir, f"phase0_summary_{args.model_name}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDescriptions: {desc_file}")
    print(f"Results: {result_file}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 0: Text-level verification quality")
    parser.add_argument("--jsonl_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="evaluation/results/text_verify")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--max_workers", type=int, default=64)
    args = parser.parse_args()
    main(args)
