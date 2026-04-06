"""
Step 2: LLM-based Error Classification

Classifies each failed step into:
  A. Planning - model doesn't know what action to take
  B. Grounding - correct action type, wrong target/coordinate
  C. Vocabulary - model's action space missing this action type
  D. Context - error caused by accumulated wrong context

Works on results from Step 1 (both AC and GUI-360).
Uses the same vLLM server for classification.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from openai import OpenAI

result_lock = Lock()
completed_count = 0


CLASSIFICATION_PROMPT = """You are an expert at analyzing GUI agent failures. Given a failed action prediction, classify the error type.

## Task Context
- Task Goal: {goal}
- Step {step_id} of {num_steps}
- Oracle (correct) action: {gt_action}
- Step instruction: {step_instruction}

## Model's Prediction
{pred_action}

## Error Classification
Classify this error into ONE of these categories:

A. PLANNING: The model chose the wrong action TYPE (e.g., predicted "click" when should "type", or "swipe" when should "click"). The model doesn't know WHAT to do.

B. GROUNDING: The model chose the correct action type but wrong TARGET (e.g., clicked wrong coordinates, typed wrong text). The model knows what to do but not WHERE/HOW.

C. VOCABULARY: The model's output format is wrong or uses an action not in the action space (e.g., malformed JSON, unknown action type).

D. CONTEXT: The error appears to be caused by incorrect understanding of the current state, likely from accumulated errors in previous steps.

Respond with ONLY the letter (A, B, C, or D) followed by a brief explanation (one sentence).

Format: X: explanation
"""


def classify_single_error(client, model_name, error_info, goal, num_steps):
    """Classify a single error using the LLM."""
    prompt = CLASSIFICATION_PROMPT.format(
        goal=goal,
        step_id=error_info['step_id'],
        num_steps=num_steps,
        gt_action=json.dumps(error_info.get('gt_action_type', 'unknown')),
        step_instruction=error_info.get('step_instruction', 'N/A'),
        pred_action=json.dumps(error_info.get('pred_action', {})),
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.0,
            extra_body={"top_k": 1},
        )
        answer = response.choices[0].message.content.strip()

        # Parse the classification
        category = 'UNKNOWN'
        explanation = answer
        if answer and answer[0].upper() in 'ABCD':
            letter = answer[0].upper()
            category = {'A': 'PLANNING', 'B': 'GROUNDING', 'C': 'VOCABULARY', 'D': 'CONTEXT'}[letter]
            if ':' in answer:
                explanation = answer.split(':', 1)[1].strip()

        return {
            'category': category,
            'explanation': explanation,
            'raw_response': answer,
        }
    except Exception as e:
        return {
            'category': 'ERROR',
            'explanation': str(e),
            'raw_response': '',
        }


def classify_trajectory_errors(client, model_name, trajectory, args):
    """Classify all errors in a single trajectory."""
    global completed_count

    goal = trajectory['goal']
    num_steps = trajectory['num_steps']
    steps = trajectory['steps']

    classified_steps = []
    for step in steps:
        if not step.get('success', False):
            # This step has an error - classify it
            classification = classify_single_error(
                client, model_name, step, goal, num_steps
            )
            classified_steps.append({
                **step,
                'classification': classification,
            })
        else:
            classified_steps.append({
                **step,
                'classification': {'category': 'CORRECT', 'explanation': '', 'raw_response': ''},
            })

    result = {
        'dataset': trajectory.get('dataset', 'unknown'),
        'trajectory_id': trajectory.get('trajectory_id', ''),
        'goal': goal,
        'num_steps': num_steps,
        'steps': classified_steps,
        'trajectory_success': trajectory.get('trajectory_success', False),
    }

    with result_lock:
        completed_count += 1
        output_path = os.path.join(
            args.output_dir,
            f"classified_{trajectory.get('dataset', 'unknown')}.jsonl"
        )
        with open(output_path, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        if completed_count % 20 == 0:
            print(f"Classified {completed_count} trajectories...")

    return result


def aggregate_classifications(results, dataset_name):
    """Aggregate classification results."""
    category_counts = defaultdict(int)
    category_by_position = defaultdict(lambda: defaultdict(int))
    total_errors = 0

    for r in results:
        for step in r['steps']:
            cat = step.get('classification', {}).get('category', 'UNKNOWN')
            if cat != 'CORRECT':
                category_counts[cat] += 1
                total_errors += 1
                # Bucket by relative position
                rel_pos = step['step_id'] / r['num_steps'] if r['num_steps'] > 0 else 0
                bucket = 'early' if rel_pos < 0.33 else ('middle' if rel_pos < 0.67 else 'late')
                category_by_position[bucket][cat] += 1

    summary = {
        'dataset': dataset_name,
        'total_errors': total_errors,
        'category_counts': dict(category_counts),
        'category_percentages': {
            k: v / total_errors * 100 for k, v in category_counts.items()
        } if total_errors > 0 else {},
        'category_by_position': {
            k: dict(v) for k, v in category_by_position.items()
        },
    }
    return summary


def main(args):
    global completed_count

    client = OpenAI(api_key="EMPTY", base_url=args.api_url, timeout=120)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load results from Step 1
    all_results = []
    for dataset_dir in [args.ac_results_dir, args.gui360_results_dir]:
        if not os.path.isdir(dataset_dir):
            print(f"Skipping {dataset_dir} (not found)")
            continue
        for fname in os.listdir(dataset_dir):
            if fname.startswith('ac_nostop_') and fname.endswith('.jsonl'):
                path = os.path.join(dataset_dir, fname)
                with open(path) as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            # Only classify trajectories with errors
                            if not data.get('trajectory_success', False):
                                all_results.append(data)
            elif fname.endswith('_nostop_results.jsonl') or fname.startswith('gui360_nostop'):
                path = os.path.join(dataset_dir, fname)
                with open(path) as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            if not data.get('trajectory_success', False):
                                all_results.append(data)

    if not all_results:
        print("No failed trajectories found. Check Step 1 results.")
        return

    print(f"Found {len(all_results)} failed trajectories to classify")

    # Limit for efficiency (classification is expensive)
    if args.max_trajectories and len(all_results) > args.max_trajectories:
        import random
        random.seed(42)
        all_results = random.sample(all_results, args.max_trajectories)
        print(f"Sampled {args.max_trajectories} trajectories for classification")

    # Clear previous outputs
    for fname in os.listdir(args.output_dir):
        if fname.startswith('classified_') and fname.endswith('.jsonl'):
            os.remove(os.path.join(args.output_dir, fname))

    # Classify errors
    classified_results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                classify_trajectory_errors, client, args.model_name, traj, args
            ): traj for traj in all_results
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                classified_results.append(result)
            except Exception as e:
                print(f"Classification error: {e}")

    # Aggregate by dataset
    ac_results = [r for r in classified_results if r['dataset'] == 'androidcontrol']
    gui360_results = [r for r in classified_results if r['dataset'] != 'androidcontrol']

    print("\n" + "=" * 60)
    print("Error Classification Summary")
    print("=" * 60)

    summaries = {}
    if ac_results:
        ac_summary = aggregate_classifications(ac_results, 'AndroidControl')
        summaries['ac'] = ac_summary
        print(f"\nAndroidControl ({ac_summary['total_errors']} errors):")
        for cat, pct in sorted(ac_summary['category_percentages'].items()):
            print(f"  {cat}: {pct:.1f}% ({ac_summary['category_counts'][cat]})")

    if gui360_results:
        g360_summary = aggregate_classifications(gui360_results, 'GUI-360')
        summaries['gui360'] = g360_summary
        print(f"\nGUI-360 ({g360_summary['total_errors']} errors):")
        for cat, pct in sorted(g360_summary['category_percentages'].items()):
            print(f"  {cat}: {pct:.1f}% ({g360_summary['category_counts'][cat]})")

    # Save aggregate summary
    summary_path = os.path.join(args.output_dir, 'classification_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ac_results_dir", type=str,
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/results/ac")
    parser.add_argument("--gui360_results_dir", type=str,
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/results/gui360")
    parser.add_argument("--output_dir", type=str,
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/results/classification")
    parser.add_argument("--api_url", type=str, default="http://localhost:19806/v1")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--max_trajectories", type=int, default=500,
                        help="Max trajectories to classify (for efficiency)")
    args = parser.parse_args()
    main(args)
