"""Eval TODO 6: LLM-Based Error Classification.

Uses the VLM to classify each failure step into fine-grained error categories.
Takes existing eval_a trajectory results, and for each failure step:
  - Shows VLM the screenshot + GT action + predicted action
  - Asks VLM to classify the error type with reasoning

Error categories:
  1. PLANNING_WRONG_TYPE: Model chose entirely wrong action type (e.g., clicked when should swipe)
  2. PLANNING_WRONG_TARGET: Right action type but completely wrong target/element
  3. GROUNDING_NEAR_MISS: Right action type, target nearby but outside tolerance
  4. TIMING_ERROR: Action premature or delayed (e.g., should wait, or waited too long)
  5. GOAL_MISUNDERSTANDING: Model misunderstood the task goal
  6. STATE_CONFUSION: Model confused about current screen state
  7. APP_NAVIGATION_ERROR: Model failed to find/open the right app or screen
  8. OTHER: Doesn't fit above categories
"""

import argparse
import copy
import json
import os
import re
import sys
import traceback
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'evaluation'))

from evaluation.qwenvl_utils import image_to_data_url, END_POINT
from openai import OpenAI
from PIL import Image

result_lock = Lock()

ERROR_CATEGORIES = [
    "PLANNING_WRONG_TYPE",
    "PLANNING_WRONG_TARGET",
    "GROUNDING_NEAR_MISS",
    "TIMING_ERROR",
    "GOAL_MISUNDERSTANDING",
    "STATE_CONFUSION",
    "APP_NAVIGATION_ERROR",
    "OTHER",
]

CLASSIFY_PROMPT = """You are an expert evaluator for a mobile GUI automation agent. The agent was given a task but made an error at this step. Analyze the error by comparing the predicted action with the ground truth action.

Task Goal: {goal}
Step Number: {step_num} of {num_steps}

Ground Truth Action: {gt_action}
Predicted Action: {pred_action}

Previous actions taken (GT):
{action_history}

Look at the current screenshot and classify this error into ONE of these categories:

1. PLANNING_WRONG_TYPE: Model chose entirely wrong action type (e.g., clicked when should have swiped, typed when should have clicked)
2. PLANNING_WRONG_TARGET: Right action type, but interacted with completely wrong element/area on screen
3. GROUNDING_NEAR_MISS: Right action type, right general area, but coordinates slightly off (clicked near but not on target)
4. TIMING_ERROR: Model did wrong temporal action (e.g., should wait but didn't, or waited when should act)
5. GOAL_MISUNDERSTANDING: Model seems to misunderstand what the task goal requires
6. STATE_CONFUSION: Model confused about current screen state (e.g., thinks it's on a different screen)
7. APP_NAVIGATION_ERROR: Model failed to navigate to or open the correct app/screen
8. OTHER: Doesn't fit any above category

Output ONLY a JSON object:
{{"category": "<one of the 8 categories>", "reasoning": "<brief explanation of why this category>"}}"""


def _describe_action(action):
    """Convert action dict to human-readable string."""
    if action is None:
        return "None (parse failure)"
    atype = action.get('action', 'unknown')
    parts = [f"action={atype}"]
    if 'coordinate' in action:
        parts.append(f"coordinate={action['coordinate']}")
    if 'coordinate2' in action:
        parts.append(f"coordinate2={action['coordinate2']}")
    if 'text' in action:
        parts.append(f'text="{action["text"]}"')
    if 'button' in action:
        parts.append(f"button={action['button']}")
    if 'time' in action:
        parts.append(f"time={action['time']}")
    if 'status' in action:
        parts.append(f"status={action['status']}")
    return ", ".join(parts)


def classify_error(screenshot_path, goal, step_num, num_steps,
                   gt_action, pred_action, action_history, model_name):
    """Use VLM to classify a single error step.

    Returns (category, reasoning) tuple.
    """
    gt_desc = _describe_action(gt_action)
    pred_desc = _describe_action(pred_action)

    history_lines = []
    for i, ah in enumerate(action_history):
        history_lines.append(f"  Step {i}: [{ah['action']}] {_describe_action(ah)}")
    history_text = "\n".join(history_lines) if history_lines else "  (no previous actions)"

    prompt = CLASSIFY_PROMPT.format(
        goal=goal,
        step_num=step_num + 1,
        num_steps=num_steps,
        gt_action=gt_desc,
        pred_action=pred_desc,
        action_history=history_text,
    )

    # Build message with screenshot
    img = Image.open(screenshot_path)
    data_url = image_to_data_url(img)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": prompt},
        ]
    }]

    for attempt in range(3):
        try:
            bot = OpenAI(api_key="EMPTY", base_url=END_POINT, timeout=120)
            resp = bot.chat.completions.create(
                model=model_name,
                messages=messages,
                extra_body={"top_k": 1},
                max_tokens=512,
            )
            response_text = resp.choices[0].message.content
            return _parse_classification(response_text)
        except Exception as e:
            if attempt < 2:
                time.sleep(2)

    return "OTHER", "classification_failed"


def _parse_classification(response_text):
    """Parse VLM classification response."""
    # Try JSON parse
    try:
        parsed = json.loads(response_text)
        cat = parsed.get('category', 'OTHER').upper().strip()
        reason = parsed.get('reasoning', '')
        if cat in ERROR_CATEGORIES:
            return cat, reason
    except (json.JSONDecodeError, AttributeError):
        pass

    # Try extracting JSON
    match = re.search(r'\{[^{}]*"category"\s*:\s*"([^"]+)"[^{}]*\}', response_text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            cat = parsed.get('category', 'OTHER').upper().strip()
            reason = parsed.get('reasoning', '')
            if cat in ERROR_CATEGORIES:
                return cat, reason
        except (json.JSONDecodeError, AttributeError):
            pass

    # Fallback: keyword search
    for cat in ERROR_CATEGORIES:
        if cat in response_text.upper():
            return cat, response_text[:200]

    return "OTHER", response_text[:200]


def process_episode(episode_result, args):
    """Classify errors for a single episode's failure steps."""
    episode_id = episode_result.get('episode_id')
    goal = episode_result['goal']
    num_steps = episode_result['num_steps']
    step_results = episode_result.get('step_results', [])

    if episode_result.get('task_success', False):
        return None  # Skip successful episodes

    # Find the failure step (last step in step_results)
    if not step_results:
        return None

    # Get the image root for screenshots
    image_root = os.path.join(PROJECT_ROOT, 'datasets')

    # Load the original episode to get screenshot paths
    classifications = []

    for step in step_results:
        if step.get('extract_match', False):
            continue  # Skip correct steps

        step_num = step['step_num']
        gt_action = step.get('gt_action', {})
        pred_action = step.get('pred_action')

        # Build action history (GT actions before this step)
        action_history = []
        for prev in step_results:
            if prev['step_num'] < step_num:
                action_history.append(prev.get('gt_action', {}))

        # Find screenshot path
        screenshot_path = os.path.join(
            image_root,
            'AndroidControl', 'images',
            f"{episode_id}_{step_num}.png"
        )
        if not os.path.exists(screenshot_path):
            # Try alternative naming
            screenshot_path = os.path.join(
                image_root,
                'AndroidControl', 'test', 'images',
                f"{episode_id}_{step_num}.png"
            )
        if not os.path.exists(screenshot_path):
            classifications.append({
                'step_num': step_num,
                'category': 'OTHER',
                'reasoning': f'screenshot_not_found: {screenshot_path}',
                'gt_action': gt_action,
                'pred_action': pred_action,
            })
            continue

        category, reasoning = classify_error(
            screenshot_path=screenshot_path,
            goal=goal,
            step_num=step_num,
            num_steps=num_steps,
            gt_action=gt_action,
            pred_action=pred_action,
            action_history=action_history,
            model_name=args.model_name,
        )

        classifications.append({
            'step_num': step_num,
            'category': category,
            'reasoning': reasoning[:300],
            'gt_action_type': gt_action.get('action', 'unknown'),
            'pred_action_type': pred_action.get('action', 'unknown') if pred_action else 'parse_fail',
            'gt_action': gt_action,
            'pred_action': pred_action,
        })

    if not classifications:
        return None

    output = {
        'episode_id': episode_id,
        'goal': goal,
        'num_steps': num_steps,
        'task_success': False,
        'error_classifications': classifications,
    }

    with result_lock:
        out_path = os.path.join(args.output_dir, 'error_classifications.jsonl')
        with open(out_path, 'a') as f:
            f.write(json.dumps(output, ensure_ascii=False) + '\n')

    return output


def analyze_results(results, output_dir):
    """Analyze error classification results."""
    print("\n" + "=" * 80)
    print("RESULTS: LLM-Based Error Classification (TODO 6)")
    print("=" * 80)

    # Collect all classifications
    all_errors = []
    for r in results:
        if r is None:
            continue
        for c in r['error_classifications']:
            all_errors.append(c)

    print(f"\nTotal failure steps classified: {len(all_errors)}")

    # 1. Overall error distribution
    print("\n--- Error Category Distribution ---")
    cat_counts = Counter(e['category'] for e in all_errors)
    for cat, count in cat_counts.most_common():
        pct = count / len(all_errors) * 100
        print(f"  {cat:30s}: {count:5d} ({pct:5.1f}%)")

    # 2. Error category by GT action type
    print("\n--- Error Category by GT Action Type ---")
    by_gt_type = defaultdict(lambda: Counter())
    for e in all_errors:
        gt_type = e.get('gt_action_type', 'unknown')
        by_gt_type[gt_type][e['category']] += 1

    for gt_type in sorted(by_gt_type.keys()):
        total = sum(by_gt_type[gt_type].values())
        print(f"\n  GT action type: {gt_type} (N={total})")
        for cat, count in by_gt_type[gt_type].most_common(5):
            pct = count / total * 100
            print(f"    {cat:30s}: {count:5d} ({pct:5.1f}%)")

    # 3. Error category by step position
    print("\n--- Error Category by Step Position ---")
    by_pos = defaultdict(lambda: Counter())
    for e in all_errors:
        sn = e['step_num']
        pos = 'step_0' if sn == 0 else ('step_1-3' if sn <= 3 else 'step_4+')
        by_pos[pos][e['category']] += 1

    for pos in ['step_0', 'step_1-3', 'step_4+']:
        total = sum(by_pos[pos].values())
        if total == 0:
            continue
        print(f"\n  Position: {pos} (N={total})")
        for cat, count in by_pos[pos].most_common(5):
            pct = count / total * 100
            print(f"    {cat:30s}: {count:5d} ({pct:5.1f}%)")

    # 4. Planning vs Execution breakdown
    print("\n--- Planning vs Execution (Aggregated) ---")
    planning = sum(1 for e in all_errors if e['category'] in [
        'PLANNING_WRONG_TYPE', 'GOAL_MISUNDERSTANDING', 'APP_NAVIGATION_ERROR'
    ])
    execution = sum(1 for e in all_errors if e['category'] in [
        'PLANNING_WRONG_TARGET', 'GROUNDING_NEAR_MISS'
    ])
    contextual = sum(1 for e in all_errors if e['category'] in [
        'TIMING_ERROR', 'STATE_CONFUSION'
    ])
    other = len(all_errors) - planning - execution - contextual
    print(f"  Planning errors (wrong type/goal/navigation): {planning} ({planning / len(all_errors) * 100:.1f}%)")
    print(f"  Execution errors (wrong target/grounding):    {execution} ({execution / len(all_errors) * 100:.1f}%)")
    print(f"  Contextual errors (timing/state confusion):   {contextual} ({contextual / len(all_errors) * 100:.1f}%)")
    print(f"  Other:                                        {other} ({other / len(all_errors) * 100:.1f}%)")

    # 5. Sample error cases per category
    print("\n--- Sample Errors per Category ---")
    by_cat = defaultdict(list)
    for e in all_errors:
        by_cat[e['category']].append(e)
    for cat in ERROR_CATEGORIES:
        if cat in by_cat and by_cat[cat]:
            print(f"\n  {cat} (showing 2 examples):")
            for ex in by_cat[cat][:2]:
                print(f"    GT: {_describe_action_short(ex.get('gt_action', {}))}")
                print(f"    Pred: {_describe_action_short(ex.get('pred_action'))}")
                print(f"    Reason: {ex.get('reasoning', '')[:100]}")

    # Save summary
    summary = {
        'total_errors': len(all_errors),
        'category_distribution': dict(cat_counts),
        'planning_errors': planning,
        'execution_errors': execution,
        'contextual_errors': contextual,
    }
    with open(os.path.join(output_dir, 'error_classification_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {output_dir}/error_classification_summary.json")


def _describe_action_short(action):
    if action is None:
        return "None"
    atype = action.get('action', '?')
    if 'text' in action:
        return f'{atype}("{action["text"][:30]}")'
    if 'coordinate' in action:
        return f'{atype}({action["coordinate"]})'
    if 'button' in action:
        return f'{atype}({action["button"]})'
    return atype


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    out_path = os.path.join(args.output_dir, 'error_classifications.jsonl')
    if os.path.exists(out_path):
        os.remove(out_path)

    # Load existing trajectory results
    traj_path = args.trajectory_results
    if not os.path.exists(traj_path):
        print(f"ERROR: trajectory results not found: {traj_path}")
        sys.exit(1)

    episodes = []
    with open(traj_path, 'r') as f:
        for line in f:
            ep = json.loads(line.strip())
            if not ep.get('task_success', False):
                episodes.append(ep)

    if args.max_episodes:
        episodes = episodes[:args.max_episodes]

    print(f"Loaded {len(episodes)} failed episodes for error classification.")
    print(f"  Model: {args.model_name}")
    print(f"  Output: {args.output_dir}")

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_episode, ep, args): ep for ep in episodes}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                if len(results) % 50 == 0:
                    print(f"Progress: {len(results)} classified...")
            except Exception as e:
                print(f"Exception: {e}")
                traceback.print_exc()

    analyze_results(results, args.output_dir)
    print(f"\nDone. {len(results)} episodes classified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval TODO 6: LLM Error Classification")
    parser.add_argument("--trajectory_results", type=str,
                        default="outputs/eval_a_ac/Qwen2.5-VL-7B/trajectory_results.jsonl",
                        help="Path to existing trajectory results with failures")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_todo6_error_classify")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()
    main(args)
