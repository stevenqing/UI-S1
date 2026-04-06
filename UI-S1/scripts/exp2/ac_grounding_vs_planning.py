"""
AndroidControl Grounding vs Planning classifier.

AC results already have type_match and grounding_match fields:
- type_match=True, grounding_match=False → GROUNDING (right action type, wrong coordinate)
- type_match=False → PLANNING (wrong action type)
- Also detect CASCADE (same action as previous step)
"""

import json
import math
from collections import Counter, defaultdict

RESULT_FILE = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/results/ac/ac_nostop_natural_cascade.jsonl"


def get_coord(action):
    """Extract coordinate from pred_action."""
    if isinstance(action, dict):
        c = action.get('coordinate')
        if isinstance(c, list) and len(c) >= 2 and c[0] is not None and c[1] is not None:
            return (float(c[0]), float(c[1]))
    return None


def main():
    trajectories = []
    with open(RESULT_FILE, 'r') as f:
        for line in f:
            trajectories.append(json.loads(line))

    print(f"Loaded {len(trajectories)} trajectories")

    total_steps = 0
    total_success = 0
    classifications = []

    # Per action type breakdown
    action_type_errors = defaultdict(lambda: Counter())

    for traj in trajectories:
        prev_coord = None

        for step in traj['steps']:
            total_steps += 1

            if step['success']:
                total_success += 1
                coord = get_coord(step['pred_action'])
                if coord:
                    prev_coord = coord
                continue

            # Failed step
            pred_action = step['pred_action']
            pred_coord = get_coord(pred_action)
            gt_type = step['gt_action_type']
            pred_type = pred_action.get('action', '')
            type_match = step['type_match']
            grounding_match = step['grounding_match']

            # Cascade detection
            is_cascade = False
            if prev_coord and pred_coord:
                if abs(prev_coord[0] - pred_coord[0]) < 5 and abs(prev_coord[1] - pred_coord[1]) < 5:
                    is_cascade = True

            if is_cascade:
                category = 'CASCADE'
            elif type_match and not grounding_match:
                category = 'GROUNDING'
            elif not type_match:
                category = 'PLANNING'
            else:
                category = 'OTHER'

            classifications.append({
                'category': category,
                'pred_type': pred_type,
                'gt_type': gt_type,
                'type_match': type_match,
                'step_instruction': step.get('step_instruction', ''),
                'traj_id': traj['trajectory_id'],
            })

            action_type_errors[gt_type][category] += 1

            if pred_coord:
                prev_coord = pred_coord

    print(f"Total steps: {total_steps}")
    print(f"Success: {total_success} ({total_success/total_steps*100:.1f}%)")
    print(f"Failed: {total_steps - total_success} ({(total_steps-total_success)/total_steps*100:.1f}%)")
    print(f"Classified errors: {len(classifications)}")

    # Overall distribution
    cat_counts = Counter(c['category'] for c in classifications)
    print(f"\n{'='*50}")
    print(f"OVERALL ERROR DISTRIBUTION")
    print(f"{'='*50}")
    for cat in ['CASCADE', 'GROUNDING', 'PLANNING', 'OTHER']:
        n = cat_counts.get(cat, 0)
        pct = n / len(classifications) * 100 if classifications else 0
        print(f"  {cat:12s}: {n:5d} ({pct:5.1f}%)")

    # Independent (non-cascade)
    independent = [c for c in classifications if c['category'] != 'CASCADE']
    ind_counts = Counter(c['category'] for c in independent)
    print(f"\n--- Independent Errors (excl cascade) ---")
    print(f"  Total: {len(independent)}")
    for cat in ['GROUNDING', 'PLANNING', 'OTHER']:
        n = ind_counts.get(cat, 0)
        pct = n / len(independent) * 100 if independent else 0
        print(f"  {cat:12s}: {n:5d} ({pct:5.1f}%)")

    # By GT action type
    print(f"\n--- By GT Action Type ---")
    for gt_type in sorted(action_type_errors.keys()):
        cats = action_type_errors[gt_type]
        total = sum(cats.values())
        print(f"\n  {gt_type} (N={total}):")
        for cat in ['CASCADE', 'GROUNDING', 'PLANNING']:
            n = cats.get(cat, 0)
            pct = n / total * 100 if total else 0
            print(f"    {cat:12s}: {n:5d} ({pct:5.1f}%)")

    # Planning error sub-analysis: what did model predict vs GT
    print(f"\n--- Planning Error: Predicted vs GT Action Type ---")
    planning_pairs = Counter()
    for c in classifications:
        if c['category'] == 'PLANNING':
            planning_pairs[(c['gt_type'], c['pred_type'])] += 1

    for (gt, pred), n in planning_pairs.most_common(15):
        print(f"  GT={gt:12s} → Pred={pred:15s}: {n:5d}")

    # Cascade analysis
    cascade_items = [c for c in classifications if c['category'] == 'CASCADE']
    if cascade_items:
        cascade_types = Counter(c['gt_type'] for c in cascade_items)
        print(f"\n--- Cascade by GT Type ---")
        for t, n in cascade_types.most_common():
            print(f"  {t:12s}: {n:5d}")

    # Terminate/finish analysis (common planning pattern)
    terminate_count = sum(1 for c in classifications if c['pred_type'] in ('terminate', 'finish', 'FINISH'))
    print(f"\n--- Special ---")
    print(f"  Model predicted terminate/finish: {terminate_count} ({terminate_count/len(classifications)*100:.1f}%)")


if __name__ == '__main__':
    main()
