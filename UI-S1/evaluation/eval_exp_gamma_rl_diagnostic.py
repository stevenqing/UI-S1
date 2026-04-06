#!/usr/bin/env python3
"""
Exp γ: RL Diagnostic Battery on Android Control

Step-level evaluation of RL models (V6, UI-S1-GRPO) vs Base on AC.
Tests two hypotheses:
  1. Does RL fix catastrophic narrowing? (per-function accuracy)
  2. Does RL improve grounding? (click error decomposition)

Uses vLLM OpenAI API for fast inference.
Constructs prompts in V6's JsonFormat style for consistency.

Usage:
  # Start vLLM server first, then:
  python eval_exp_gamma_rl_diagnostic.py \
      --model_name <vllm_model_name> \
      --jsonl_file datasets/android_control_evaluation_std.jsonl \
      --n_samples 1000 \
      --output_dir evaluation/results/exp_gamma_v6
"""

import argparse
import base64
import copy
import json
import math
import os
import re
import sys
import time
import traceback
import numpy as np
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from PIL import Image

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'evaluation'))

from x.qwen.image import smart_resize
from qwenvl_utils import evaluate_android_control_action

# ── Action space prompt (from std_space.py) ─────────────────────────

SYSTEM_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.
## Output Format

```
<action> ... </action>
```

## Action Space

You can perform the following actions:
- click: Click the point on the screen with specified (x, y) coordinates.
- long_press: Press the point on the screen with specified (x, y) coordinates for a specified number of seconds.
- swipe: Swipe from starting point with specified (x, y) coordinates to endpoint with specified (x2, y2) coordinates.
- type: Input the specified text into the activated input box.
- system_button: Press the specified system button: Back, Home, Menu, or Enter.
- open: Open an application on the device specified by text.
- wait: Wait for a specified number of seconds for changes to occur.
- terminate: Terminate the current task and report its completion status: success or failure.

The arguments you can use are:
- coordinate: (x, y): The x and y pixels coordinates from the left and top edges.
- coordinate2: (x, y): The x and y pixels coordinates from the left and top edges for the endpoint of a swipe.
- text: Text input required by actions like `type` and `open`.
- time: The time in seconds required by actions like `long_press` and `wait`.
- button: System buttons available for pressing: Back, Home, Menu, or Enter.
- status: The completion status of a terminated task. Possible values: success, failure.

Format your output as a JSON object with the selected action and its arguments at the same level.

Example outputs:
<action>
{"action": "click", "coordinate": [540, 960]}
</action>
<action>
{"action": "type", "text": "hello world"}
</action>

## Note

- Write your action in the `action` part according to the action space.
"""


# ── Utilities ────────────────────────────────────────────────────────

def image_to_data_url(image):
    buf = BytesIO()
    image.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{b64}"


def parse_action_response(text):
    """Parse model response — handles <action>, <tool_call>, and raw JSON."""
    if not text:
        return None

    # Try <action>...</action>
    m = re.search(r'<action>\s*(.*?)\s*</action>', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # Try <tool_call>...</tool_call>
    m = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL)
    if m:
        try:
            tc = json.loads(m.group(1))
            # Normalize tool_call format to action format
            if 'function' in tc:
                result = {'action': tc['function']}
                result.update(tc.get('args', {}))
                return result
            if 'name' in tc and 'arguments' in tc:
                return tc['arguments']
            return tc
        except Exception:
            pass

    # Try raw JSON
    try:
        return json.loads(text.strip())
    except Exception:
        pass

    # Fallback: find JSON in text
    m = re.search(r'\{[^{}]*"action"\s*:\s*"[^"]+?"[^{}]*\}', text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return None


def normalize_pred_for_eval(pred):
    """Normalize parsed prediction to evaluation format (same keys as check_options)."""
    if pred is None:
        return {'action': 'PARSE_FAIL', 'coordinate': None}
    result = copy.deepcopy(pred)
    # Ensure required fields
    if 'action' not in result:
        return {'action': 'PARSE_FAIL', 'coordinate': None}
    for key in ['coordinate', 'coordinate2', 'text', 'button', 'time', 'status']:
        if key not in result:
            result[key] = None
    # candidate_bbox is for GT only
    if 'candidate_bbox' not in result:
        result['candidate_bbox'] = []
    return result


# ── Data loading ─────────────────────────────────────────────────────

def load_ac_steps(jsonl_path, n_samples=0, seed=42):
    """Load individual steps from AC trajectory JSONL."""
    image_root = os.path.join(PROJECT_ROOT, 'datasets')
    steps = []

    with open(jsonl_path) as f:
        for line in f:
            episode = json.loads(line.strip())
            goal = episode['goal']
            for si, step in enumerate(episode['steps']):
                screenshot = step['screenshot']
                # Fix paths
                if screenshot.startswith('/datasets/'):
                    screenshot = screenshot.replace('/datasets/', image_root + '/', 1)
                elif not os.path.isabs(screenshot):
                    screenshot = os.path.join(image_root, screenshot)

                ac = step.get('action_content', {})
                check = step.get('check_options', copy.deepcopy(ac))
                if 'candidate_bbox' not in check:
                    check['candidate_bbox'] = []

                steps.append({
                    'goal': goal,
                    'step_idx': si,
                    'screenshot': screenshot,
                    'action_content': ac,
                    'check_options': check,
                    'gt_action': ac.get('action', ''),
                    'step_instruction': step.get('step_instruction', ''),
                })

    if 0 < n_samples < len(steps):
        rng = np.random.RandomState(seed)
        # Stratified sampling by action type
        by_action = defaultdict(list)
        for i, s in enumerate(steps):
            by_action[s['gt_action']].append(i)

        selected = []
        # Ensure at least min(20, count) per action type
        for action, indices in by_action.items():
            k = min(20, len(indices))
            selected.extend(rng.choice(indices, k, replace=False).tolist())

        remaining_budget = n_samples - len(selected)
        if remaining_budget > 0:
            all_indices = set(range(len(steps)))
            remaining = list(all_indices - set(selected))
            extra = rng.choice(remaining, min(remaining_budget, len(remaining)), replace=False)
            selected.extend(extra.tolist())

        selected = sorted(set(selected))[:n_samples]
        steps = [steps[i] for i in selected]

    print(f"Loaded {len(steps)} steps from {jsonl_path}")
    action_dist = Counter(s['gt_action'] for s in steps)
    for a, c in action_dist.most_common():
        print(f"  {a}: {c} ({100*c/len(steps):.1f}%)")
    return steps


# ── Model calling ────────────────────────────────────────────────────

def call_model(messages, model_name, port=8000):
    """Call vLLM via OpenAI API."""
    from openai import OpenAI
    client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1", timeout=120)

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                extra_body={"top_k": 1},
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
            else:
                print(f"  API error after 3 attempts: {e}")
                return ""


def build_messages(step, image):
    """Build OpenAI-format messages for a single step."""
    data_url = image_to_data_url(image)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": f"User Instruction: {step['goal']}"},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}
    ]
    return messages


# ── Main collection ──────────────────────────────────────────────────

result_lock = Lock()


def process_step(step, model_name, port):
    """Process a single step: load image, call model, parse, evaluate."""
    screenshot = step['screenshot']
    if not os.path.exists(screenshot):
        return None

    try:
        image = Image.open(screenshot).convert("RGB")
    except Exception:
        return None

    orig_w, orig_h = image.size
    resized_h, resized_w = smart_resize(orig_h, orig_w, max_pixels=12800 * 28 * 28)

    messages = build_messages(step, image)
    raw_response = call_model(messages, model_name, port)
    pred = parse_action_response(raw_response)
    pred_norm = normalize_pred_for_eval(pred)

    # Evaluate
    check = copy.deepcopy(step['check_options'])
    gt_action = step['gt_action']

    try:
        type_match, extract_match = evaluate_android_control_action(
            pred_norm, check, orig_w, orig_h, resized_w, resized_h
        )
    except Exception as e:
        type_match, extract_match = False, False

    return {
        'goal': step['goal'],
        'step_idx': step['step_idx'],
        'gt_action': gt_action,
        'gt_args': step['action_content'],
        'pred_action': pred,
        'pred_norm': pred_norm,
        'raw_response': raw_response[:1000] if raw_response else '',
        'type_match': type_match,
        'extract_match': extract_match,
        'image_size': [orig_w, orig_h],
        'resized_size': [resized_w, resized_h],
    }


def collect_predictions(steps, model_name, port, max_workers, output_dir):
    """Collect predictions with concurrent workers."""
    results = []
    n_done = 0
    n_skipped = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(process_step, step, model_name, port): i
            for i, step in enumerate(steps)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                if result is None:
                    n_skipped += 1
                else:
                    with result_lock:
                        results.append(result)
                        n_done += 1
            except Exception as e:
                n_skipped += 1
                print(f"  Error on step {idx}: {e}")

            total = n_done + n_skipped
            if total % 50 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                print(f"  [{n_done}/{len(steps)}] {rate:.1f} steps/s, skipped={n_skipped}")

    # Sort by original order
    results.sort(key=lambda r: (r['goal'], r['step_idx']))

    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, "predictions.json")
    with open(pred_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)} predictions to {pred_path} (skipped {n_skipped})")
    return results


# ── Analysis ─────────────────────────────────────────────────────────

def analyze_results(results, output_dir):
    """Comprehensive per-function analysis and error decomposition."""
    os.makedirs(output_dir, exist_ok=True)
    lines = []

    def pr(s=""):
        lines.append(s)
        print(s)

    N = len(results)
    n_type = sum(1 for r in results if r['type_match'])
    n_extract = sum(1 for r in results if r['extract_match'])

    pr("=" * 60)
    pr("Exp γ: RL Diagnostic — AC Step-Level Results")
    pr("=" * 60)
    pr(f"Total steps: {N}")
    pr(f"Type match (action correct): {n_type} ({100*n_type/N:.1f}%)")
    pr(f"Extract match (action+args correct): {n_extract} ({100*n_extract/N:.1f}%)")
    pr()

    # ── Per-function accuracy ──
    pr("=" * 60)
    pr("Per-Function Accuracy (narrowing diagnostic)")
    pr("=" * 60)

    by_func = defaultdict(list)
    for r in results:
        by_func[r['gt_action']].append(r)

    func_stats = []
    for func in sorted(by_func.keys(), key=lambda f: -len(by_func[f])):
        rs = by_func[func]
        n = len(rs)
        tm = sum(1 for r in rs if r['type_match'])
        em = sum(1 for r in rs if r['extract_match'])
        func_stats.append({
            'function': func,
            'n': n,
            'pct': 100 * n / N,
            'type_match': 100 * tm / n,
            'extract_match': 100 * em / n,
        })

    pr(f"{'Function':<16} {'n':>5} {'%data':>6} {'TypeM':>7} {'ExtractM':>9}")
    pr("-" * 50)
    for fs in func_stats:
        pr(f"{fs['function']:<16} {fs['n']:>5} {fs['pct']:>5.1f}% {fs['type_match']:>6.1f}% {fs['extract_match']:>8.1f}%")
    pr()

    # ── Confusion matrix ──
    pr("=" * 60)
    pr("Action Confusion Matrix (GT → Pred)")
    pr("=" * 60)

    confusion = defaultdict(Counter)
    for r in results:
        gt = r['gt_action']
        pred = r['pred_action']
        if pred is None:
            confusion[gt]['PARSE_FAIL'] += 1
        else:
            confusion[gt][pred.get('action', 'UNKNOWN')] += 1

    all_pred_actions = set()
    for gt_counts in confusion.values():
        all_pred_actions.update(gt_counts.keys())
    all_pred_actions = sorted(all_pred_actions)

    header_row = f"{'GT / Pred':<16}" + "".join(f" {pa[:8]:>8}" for pa in all_pred_actions)
    pr(header_row)
    pr("-" * (16 + 9 * len(all_pred_actions)))
    for gt_func in sorted(confusion.keys(), key=lambda f: -sum(confusion[f].values())):
        total = sum(confusion[gt_func].values())
        row = f"{gt_func:<16}"
        for pa in all_pred_actions:
            c = confusion[gt_func].get(pa, 0)
            if c > 0:
                row += f" {100*c/total:>7.1f}%"
            else:
                row += f" {'':>8}"
        row += f"  (n={total})"
        pr(row)
    pr()

    # ── Parse failure analysis ──
    parse_fails = [r for r in results if r['pred_action'] is None]
    pr(f"Parse failures: {len(parse_fails)} ({100*len(parse_fails)/N:.1f}%)")
    if parse_fails:
        pf_by_func = Counter(r['gt_action'] for r in parse_fails)
        for f, c in pf_by_func.most_common():
            pr(f"  {f}: {c}")
    pr()

    # ── Click error decomposition ──
    click_results = [r for r in results if r['gt_action'] == 'click']
    if click_results:
        pr("=" * 60)
        pr("Click Error Decomposition")
        pr("=" * 60)

        click_correct = [r for r in click_results if r['extract_match']]
        click_wrong = [r for r in click_results if not r['extract_match']]
        pr(f"Click total: {len(click_results)}, correct: {len(click_correct)} ({100*len(click_correct)/len(click_results):.1f}%)")

        # Categorize click errors
        distances = []
        error_cats = Counter()
        for r in click_wrong:
            pred = r['pred_action']
            gt_coord = r['gt_args'].get('coordinate')
            if pred is None:
                error_cats['parse_fail'] += 1
                continue
            if pred.get('action') != 'click':
                error_cats['wrong_action'] += 1
                continue
            pred_coord = pred.get('coordinate')
            if not pred_coord or not gt_coord:
                error_cats['no_coordinate'] += 1
                continue
            try:
                dist = math.sqrt((pred_coord[0] - gt_coord[0])**2 + (pred_coord[1] - gt_coord[1])**2)
                distances.append(dist)
                if dist < 50:
                    error_cats['near_miss'] += 1
                elif dist < 200:
                    error_cats['moderate_miss'] += 1
                else:
                    diag = math.sqrt(r['image_size'][0]**2 + r['image_size'][1]**2)
                    if dist > 0.3 * diag:
                        error_cats['far_miss_random'] += 1
                    else:
                        error_cats['far_miss_wrong_element'] += 1
            except (TypeError, ValueError):
                error_cats['invalid_coordinate'] += 1

        pr(f"\nClick error breakdown ({len(click_wrong)} errors):")
        for cat, c in error_cats.most_common():
            pr(f"  {cat}: {c} ({100*c/len(click_wrong):.1f}%)")

        if distances:
            distances = np.array(distances)
            pr(f"\nClick distance stats (n={len(distances)}):")
            pr(f"  median: {np.median(distances):.0f}px")
            pr(f"  mean: {np.mean(distances):.0f}px")
            pr(f"  p25: {np.percentile(distances, 25):.0f}px")
            pr(f"  p75: {np.percentile(distances, 75):.0f}px")
    pr()

    # ── Narrowing score ──
    pr("=" * 60)
    pr("Narrowing Diagnostic")
    pr("=" * 60)

    n_active_functions = sum(1 for fs in func_stats if fs['type_match'] > 0)
    n_total_functions = len(func_stats)
    pr(f"Active functions (type_match > 0): {n_active_functions}/{n_total_functions}")

    rare_funcs = [fs for fs in func_stats if fs['function'] not in ('click',)]
    if rare_funcs:
        rare_type_avg = np.mean([fs['type_match'] for fs in rare_funcs])
        rare_extract_avg = np.mean([fs['extract_match'] for fs in rare_funcs])
        click_fs = next((fs for fs in func_stats if fs['function'] == 'click'), None)
        if click_fs:
            pr(f"Click type_match: {click_fs['type_match']:.1f}%")
            pr(f"Non-click avg type_match: {rare_type_avg:.1f}%")
            pr(f"Gap (click - non_click): {click_fs['type_match'] - rare_type_avg:+.1f}pp")
            pr(f"  → Large positive gap = narrowing (model only does click)")
            pr(f"  → Small/negative gap = balanced (no narrowing)")
    pr()

    # ── Summary JSON ──
    summary = {
        'total_steps': N,
        'type_match_rate': n_type / N,
        'extract_match_rate': n_extract / N,
        'parse_fail_rate': len(parse_fails) / N,
        'n_active_functions': n_active_functions,
        'n_total_functions': n_total_functions,
        'per_function': func_stats,
    }

    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
        f.write('\n'.join(lines))

    pr(f"\nResults saved to {output_dir}")
    return summary


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exp γ: RL Diagnostic on AC")
    sub = parser.add_subparsers(dest='command')

    # Collect + analyze
    p_run = sub.add_parser('run', help='Collect predictions and analyze')
    p_run.add_argument('--model_name', required=True, help='vLLM model name/path')
    p_run.add_argument('--jsonl_file', required=True, help='AC trajectory JSONL')
    p_run.add_argument('--output_dir', required=True)
    p_run.add_argument('--n_samples', type=int, default=1000)
    p_run.add_argument('--max_workers', type=int, default=16)
    p_run.add_argument('--port', type=int, default=8000)
    p_run.add_argument('--seed', type=int, default=42)

    # Analyze only
    p_analyze = sub.add_parser('analyze', help='Analyze existing predictions')
    p_analyze.add_argument('--predictions', required=True)
    p_analyze.add_argument('--output_dir', required=True)

    args = parser.parse_args()

    if args.command == 'run':
        steps = load_ac_steps(args.jsonl_file, args.n_samples, args.seed)
        results = collect_predictions(
            steps, args.model_name, args.port, args.max_workers, args.output_dir)
        analyze_results(results, args.output_dir)

    elif args.command == 'analyze':
        with open(args.predictions) as f:
            results = json.load(f)
        analyze_results(results, args.output_dir)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
