"""
Step 2 补跑: GUI-360 Error Classification + Merge with existing AC results.

1. Loads GUI-360 unified JSONL (from convert_gui360_to_unified.py)
2. Classifies GUI-360 errors with LLM
3. Merges with existing AC classification
4. Generates combined summary
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from openai import OpenAI

result_lock = Lock()
completed_count = 0

CLASSIFICATION_PROMPT = """You are an expert at analyzing GUI agent failures. Given a failed action prediction, classify the error type.

## Task Context
- Trajectory: {trajectory_id}
- Step {step_id} of {num_steps}
- Oracle (correct) action type: {gt_action}
- Step thought/instruction: {step_instruction}

## Model's Prediction
{pred_action}

## Error Classification
Classify this error into ONE of these categories:

A. PLANNING: The model chose the wrong action TYPE (e.g., predicted "click" when should "type"). The model doesn't know WHAT to do.

B. GROUNDING: The model chose the correct action type but wrong TARGET (e.g., clicked wrong coordinates, typed wrong text). The model knows what to do but not WHERE/HOW.

C. VOCABULARY: The model's output format is wrong or uses an action not in the action space (e.g., malformed output, unknown action type, empty prediction).

D. CONTEXT: The error appears to be caused by incorrect understanding of the current state, likely from accumulated errors in previous steps.

Respond with ONLY the letter (A, B, C, or D) followed by a brief explanation (one sentence).
Format: X: explanation
"""


def classify_single_error(client, model_name, error_info, trajectory_id, num_steps):
    prompt = CLASSIFICATION_PROMPT.format(
        trajectory_id=trajectory_id,
        step_id=error_info['step_id'],
        num_steps=num_steps,
        gt_action=error_info.get('gt_action_type', 'unknown'),
        step_instruction=error_info.get('step_instruction', 'N/A')[:200],
        pred_action=json.dumps(error_info.get('pred_action', {}))[:300],
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
        category = 'UNKNOWN'
        explanation = answer
        if answer and answer[0].upper() in 'ABCD':
            letter = answer[0].upper()
            category = {'A': 'PLANNING', 'B': 'GROUNDING', 'C': 'VOCABULARY', 'D': 'CONTEXT'}[letter]
            if ':' in answer:
                explanation = answer.split(':', 1)[1].strip()
        return {'category': category, 'explanation': explanation}
    except Exception as e:
        return {'category': 'ERROR', 'explanation': str(e)}


def classify_trajectory(client, model_name, trajectory, output_dir):
    global completed_count
    classified_steps = []
    for step in trajectory['steps']:
        if not step.get('success', False):
            cls = classify_single_error(
                client, model_name, step,
                trajectory.get('trajectory_id', ''),
                trajectory['num_steps']
            )
            classified_steps.append({**step, 'classification': cls})
        else:
            classified_steps.append({
                **step,
                'classification': {'category': 'CORRECT', 'explanation': ''}
            })

    result = {
        'dataset': trajectory.get('dataset', 'gui360'),
        'trajectory_id': trajectory.get('trajectory_id', ''),
        'goal': trajectory.get('goal', ''),
        'num_steps': trajectory['num_steps'],
        'steps': classified_steps,
        'trajectory_success': trajectory.get('trajectory_success', False),
        'domain': trajectory.get('domain', ''),
    }

    with result_lock:
        completed_count += 1
        path = os.path.join(output_dir, 'classified_gui360.jsonl')
        with open(path, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        if completed_count % 20 == 0:
            print(f"  Classified {completed_count} GUI-360 trajectories...")

    return result


def aggregate(results, dataset_name):
    cat_counts = defaultdict(int)
    cat_by_pos = defaultdict(lambda: defaultdict(int))
    cat_by_domain = defaultdict(lambda: defaultdict(int))
    total_errors = 0

    for r in results:
        domain = r.get('domain', 'unknown')
        for step in r['steps']:
            cat = step.get('classification', {}).get('category', 'UNKNOWN')
            if cat not in ('CORRECT', 'ERROR'):
                cat_counts[cat] += 1
                total_errors += 1
                rel = step['step_id'] / r['num_steps'] if r['num_steps'] > 0 else 0
                bucket = 'early' if rel < 0.33 else ('middle' if rel < 0.67 else 'late')
                cat_by_pos[bucket][cat] += 1
                cat_by_domain[domain][cat] += 1

    return {
        'dataset': dataset_name,
        'total_errors': total_errors,
        'category_counts': dict(cat_counts),
        'category_percentages': {
            k: v / total_errors * 100 for k, v in cat_counts.items()
        } if total_errors > 0 else {},
        'category_by_position': {k: dict(v) for k, v in cat_by_pos.items()},
        'category_by_domain': {k: dict(v) for k, v in cat_by_domain.items()},
    }


def main(args):
    global completed_count

    client = OpenAI(api_key="EMPTY", base_url=args.api_url, timeout=120)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load GUI-360 unified JSONL
    gui360_file = os.path.join(args.gui360_results_dir, 'gui360_nostop_results.jsonl')
    if not os.path.exists(gui360_file):
        print(f"ERROR: {gui360_file} not found. Run convert_gui360_to_unified.py first.")
        return

    all_failed = []
    with open(gui360_file) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if not data.get('trajectory_success', False):
                    all_failed.append(data)

    print(f"Found {len(all_failed)} failed GUI-360 trajectories")

    # Sample for efficiency
    if args.max_trajectories and len(all_failed) > args.max_trajectories:
        import random
        random.seed(42)
        all_failed = random.sample(all_failed, args.max_trajectories)
        print(f"Sampled {args.max_trajectories} for classification")

    # Clear previous GUI-360 classification
    cls_path = os.path.join(args.output_dir, 'classified_gui360.jsonl')
    if os.path.exists(cls_path):
        os.remove(cls_path)

    # Classify
    print("Starting GUI-360 classification...")
    classified = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(classify_trajectory, client, args.model_name, t, args.output_dir): t
            for t in all_failed
        }
        for future in as_completed(futures):
            try:
                classified.append(future.result())
            except Exception as e:
                print(f"Error: {e}")

    g360_summary = aggregate(classified, 'GUI-360')

    # Load existing AC classification
    ac_summary = None
    existing_summary_path = os.path.join(args.output_dir, 'classification_summary.json')
    if os.path.exists(existing_summary_path):
        with open(existing_summary_path) as f:
            existing = json.load(f)
        ac_summary = existing.get('ac')

    # Merge and save
    combined = {}
    if ac_summary:
        combined['ac'] = ac_summary
    combined['gui360'] = g360_summary

    with open(existing_summary_path, 'w') as f:
        json.dump(combined, f, indent=2)

    # Print results
    print("\n" + "=" * 60)
    print("GUI-360 Error Classification Results")
    print("=" * 60)
    print(f"Total errors classified: {g360_summary['total_errors']}")
    for cat, pct in sorted(g360_summary['category_percentages'].items(), key=lambda x: -x[1]):
        print(f"  {cat}: {pct:.1f}% ({g360_summary['category_counts'][cat]})")

    if g360_summary.get('category_by_domain'):
        print("\nBy Domain:")
        for domain, cats in g360_summary['category_by_domain'].items():
            total = sum(cats.values())
            print(f"  {domain} ({total} errors):")
            for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
                print(f"    {cat}: {cnt} ({cnt/total*100:.1f}%)")

    if ac_summary:
        print("\n" + "=" * 60)
        print("Cross-Dataset Classification Comparison")
        print("=" * 60)
        print(f"{'Category':<15} {'AC':>10} {'GUI-360':>10}")
        print("-" * 35)
        all_cats = set(list(ac_summary.get('category_percentages', {}).keys()) +
                       list(g360_summary.get('category_percentages', {}).keys()))
        for cat in sorted(all_cats):
            ac_pct = ac_summary.get('category_percentages', {}).get(cat, 0)
            g360_pct = g360_summary.get('category_percentages', {}).get(cat, 0)
            print(f"{cat:<15} {ac_pct:>9.1f}% {g360_pct:>9.1f}%")

    print(f"\nSaved to {existing_summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui360_results_dir", type=str,
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/results/gui360")
    parser.add_argument("--output_dir", type=str,
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/results/classification")
    parser.add_argument("--api_url", type=str, default="http://localhost:19806/v1")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--max_trajectories", type=int, default=500)
    args = parser.parse_args()
    main(args)
