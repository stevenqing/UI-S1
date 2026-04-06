"""
Pure Hit-Test Grounding vs Planning classifier.

Unlike the thought-based classifier, this does NOT depend on model's CoT text.
Instead it uses only:
1. Model's predicted coordinate → hit-test against GT UI element list → find clicked element
2. GT target element (control_test)
3. Compare: did the model click the GT target, a related element, or a completely different one?

Classification:
- CASCADE: same coord as previous step (stuck)
- NEAR_MISS: coord distance < 50px
- GROUNDING: model clicked an element with the same control_text as GT target
              OR clicked within expanded GT rect (2x tolerance) but outside strict rect
- PLANNING: model clicked a different element than GT target
- NO_HIT: model clicked empty space (no UI element at predicted coord)
"""

import json
import math
import os
import sys
from collections import Counter, defaultdict
import random
import argparse

random.seed(42)

GT_DATA_ROOT = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/GUI-360/test/data"
OUTPUT_DIR = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/results/analysis"


def load_gt_data():
    """Load all GT data indexed by (execution_id, step_id)."""
    gt_index = {}
    for domain in ['ppt', 'excel', 'word']:
        for category in os.listdir(os.path.join(GT_DATA_ROOT, domain)):
            cat_dir = os.path.join(GT_DATA_ROOT, domain, category)
            if not os.path.isdir(cat_dir):
                continue
            for status_dir in os.listdir(cat_dir):
                full_dir = os.path.join(cat_dir, status_dir)
                if not os.path.isdir(full_dir):
                    continue
                for fname in os.listdir(full_dir):
                    if not fname.endswith('.jsonl'):
                        continue
                    fpath = os.path.join(full_dir, fname)
                    with open(fpath, 'r') as f:
                        for line in f:
                            step_data = json.loads(line)
                            eid = step_data['execution_id']
                            sid = step_data['step_id']
                            gt_index[(eid, sid)] = step_data
    return gt_index


def parse_sample_id(sample_id):
    """Parse sample_id into (execution_id, step_num)."""
    parts = sample_id.rsplit('_', 1)
    if len(parts) != 2:
        return None, None
    step_num = int(parts[1])
    prefix = parts[0]
    for domain in ['ppt', 'excel', 'word']:
        if prefix.startswith(domain + '_'):
            rest = prefix[len(domain) + 1:]
            for cat in ['in_app', 'online', 'search']:
                if rest.startswith(cat + '_'):
                    exec_id = rest[len(cat) + 1:]
                    return exec_id, step_num
    return None, None


def hit_test(x, y, controls):
    """Find the UI control at (x, y). Returns list of (control_text, control_type, area)
    sorted by area (smallest first, for most specific element)."""
    hits = []
    for c in controls:
        r = c['control_rect']  # [left, top, right, bottom]
        if r[0] <= x <= r[2] and r[1] <= y <= r[3]:
            area = (r[2] - r[0]) * (r[3] - r[1])
            hits.append((area, c['control_text'], c['control_type'], c.get('label', -1)))

    if hits:
        hits.sort(key=lambda t: t[0])  # smallest area = most specific
        return hits

    return []


def find_nearest_control(x, y, controls, max_dist=50):
    """Find nearest control within max_dist pixels."""
    best = None
    best_dist = float('inf')
    for c in controls:
        r = c['control_rect']
        # Distance from point to rect
        dx = max(r[0] - x, 0, x - r[2])
        dy = max(r[1] - y, 0, y - r[3])
        d = math.sqrt(dx**2 + dy**2)
        if d < best_dist:
            best_dist = d
            best = c
    if best and best_dist <= max_dist:
        return best['control_text'], best['control_type'], best_dist
    return None, None, None


def normalize(text):
    if not text:
        return ""
    return text.lower().strip()


def is_same_element(clicked_text, gt_text):
    """Check if clicked element is the same as GT target.
    Handles exact match and substring containment."""
    if not clicked_text or not gt_text:
        return False
    ct = normalize(clicked_text)
    gt = normalize(gt_text)
    if ct == gt:
        return True
    # Substring: GT target name contained in clicked element name or vice versa
    # But only if the shorter one is >= 3 chars to avoid false positives
    if len(gt) >= 3 and gt in ct:
        return True
    if len(ct) >= 3 and ct in gt:
        return True
    return False


def classify_step(pred_coord, gt_coord, gt_rect, gt_control_text,
                  controls, prev_pred_coord):
    """Classify a single failed click step using pure hit-test.

    Returns: (category, details)
    """
    # 1. Cascade check
    if prev_pred_coord is not None:
        px, py = prev_pred_coord
        cx, cy = pred_coord
        if px is not None and py is not None and cx is not None and cy is not None:
            if abs(px - cx) < 2 and abs(py - cy) < 2:
                return 'CASCADE', {}

    # 2. Near miss check
    dist = math.sqrt((pred_coord[0] - gt_coord[0])**2 + (pred_coord[1] - gt_coord[1])**2)
    if dist < 50:
        return 'NEAR_MISS', {'distance': dist}

    # 3. Hit test: what element did model click?
    hits = hit_test(pred_coord[0], pred_coord[1], controls)

    details = {
        'distance': dist,
        'gt_target': gt_control_text or '',
    }

    if hits:
        # Take the most specific (smallest area) element
        _, clicked_text, clicked_type, clicked_label = hits[0]
        details['clicked_element'] = clicked_text
        details['clicked_type'] = clicked_type

        # Compare with GT target
        if is_same_element(clicked_text, gt_control_text):
            # Model clicked the right element but coordinates still counted as wrong
            # This is a grounding near-miss (element boundary issue)
            return 'GROUNDING', details
        else:
            # Model clicked a different element
            # Check if any of the hit elements match GT
            for _, ht, _, _ in hits:
                if is_same_element(ht, gt_control_text):
                    return 'GROUNDING', details

            return 'PLANNING', details
    else:
        # No UI element at predicted coordinate
        # Check if near GT rect (expanded tolerance)
        if gt_rect:
            r = gt_rect
            # Check 30px expanded rect
            if (r['left'] - 30 <= pred_coord[0] <= r['right'] + 30 and
                r['top'] - 30 <= pred_coord[1] <= r['bottom'] + 30):
                details['reason'] = 'near_gt_rect'
                return 'GROUNDING', details

        # Find nearest element
        near_text, near_type, near_dist = find_nearest_control(
            pred_coord[0], pred_coord[1], controls)
        if near_text:
            details['nearest_element'] = near_text
            details['nearest_dist'] = near_dist
            if is_same_element(near_text, gt_control_text):
                return 'GROUNDING', details

        details['reason'] = 'empty_space'
        return 'NO_HIT', details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['base', 'sftv2'], required=True)
    args = parser.parse_args()

    if args.model == 'base':
        eval_path = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/results/gui360/latest_nostop/ar_evaluation_results_20260319_182012.json"
        output_name = "hittest_classification_base.json"
        label = "Base Model"
    else:
        eval_path = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/results/sft_v2/gui360/nostop_20260320_053216/ar_evaluation_results_20260320_055609.json"
        output_name = "hittest_classification_sftv2.json"
        label = "SFT v2"

    print(f"=== Hit-Test Classifier: {label} ===")
    print(f"Loading GT data...")
    gt_index = load_gt_data()
    print(f"  Loaded {len(gt_index)} GT steps")

    print(f"Loading eval results from {eval_path}...")
    with open(eval_path, 'r') as f:
        eval_data = json.load(f)

    trajectories = eval_data['trajectory_results']
    detailed_by_id = {dr['sample_id']: dr for dr in eval_data['detailed_results']}
    print(f"  Loaded {len(trajectories)} trajectories")

    classifications = []
    match_failures = 0
    total_click_errors = 0

    for traj in trajectories:
        prev_pred_coord = None

        for step_result in traj['step_results']:
            sample_id = step_result['sample_id']
            dr = detailed_by_id.get(sample_id, step_result)

            pred_func = dr.get('predicted_function', '')
            gt_func = dr.get('ground_truth_function', '')

            if dr.get('success', False):
                pa = dr.get('predicted_args', {})
                if isinstance(pa, dict) and 'coordinate' in pa:
                    c = pa['coordinate']
                    if isinstance(c, list) and len(c) >= 2 and c[0] is not None and c[1] is not None:
                        prev_pred_coord = (float(c[0]), float(c[1]))
                continue

            if pred_func != 'click' or gt_func != 'click':
                pa = dr.get('predicted_args', {})
                if isinstance(pa, dict) and 'coordinate' in pa:
                    c = pa['coordinate']
                    if isinstance(c, list) and len(c) >= 2 and c[0] is not None and c[1] is not None:
                        prev_pred_coord = (float(c[0]), float(c[1]))
                continue

            total_click_errors += 1

            pa = dr.get('predicted_args', {})
            if not isinstance(pa, dict) or 'coordinate' not in pa:
                prev_pred_coord = None
                continue
            pred_c = pa['coordinate']
            if not isinstance(pred_c, list) or len(pred_c) < 2:
                prev_pred_coord = None
                continue
            if pred_c[0] is None or pred_c[1] is None:
                prev_pred_coord = None
                continue
            pred_coord = (float(pred_c[0]), float(pred_c[1]))

            ga = dr.get('ground_truth_args', {})
            gt_c = ga.get('coordinate', [0, 0])
            gt_coord = (float(gt_c[0]), float(gt_c[1]))
            gt_rect = dr.get('ground_truth_rect', None)

            exec_id, step_num = parse_sample_id(sample_id)

            gt_control_text = None
            controls = []

            if exec_id and (exec_id, step_num) in gt_index:
                gt_step = gt_index[(exec_id, step_num)]
                gt_action = gt_step['step'].get('action', {})
                gt_control_text = gt_action.get('control_test', '')
                control_infos = gt_step['step'].get('control_infos', {})
                controls = control_infos.get('uia_controls_info', [])
            else:
                match_failures += 1

            category, details = classify_step(
                pred_coord, gt_coord, gt_rect, gt_control_text,
                controls, prev_pred_coord
            )

            details['sample_id'] = sample_id
            details['domain'] = traj.get('domain', '')
            details['pred_coord'] = pred_coord
            details['gt_coord'] = gt_coord

            classifications.append((category, details))
            prev_pred_coord = pred_coord

    # === Print Results ===
    print(f"\n{'='*60}")
    print(f"HIT-TEST CLASSIFICATION: {label}")
    print(f"{'='*60}")
    print(f"Total click errors: {total_click_errors}")
    print(f"Classified: {len(classifications)}")
    print(f"GT match failures: {match_failures}")

    cat_counts = Counter(c for c, _ in classifications)
    print(f"\n--- Overall Distribution ---")
    for cat in ['CASCADE', 'NEAR_MISS', 'GROUNDING', 'PLANNING', 'NO_HIT']:
        n = cat_counts.get(cat, 0)
        pct = n / len(classifications) * 100 if classifications else 0
        print(f"  {cat:12s}: {n:6d} ({pct:5.1f}%)")

    # Independent errors
    independent = [(c, d) for c, d in classifications if c not in ('CASCADE', 'NEAR_MISS')]
    ind_counts = Counter(c for c, _ in independent)
    print(f"\n--- Independent Errors (excl cascade & near-miss) ---")
    print(f"  Total: {len(independent)}")
    for cat in ['GROUNDING', 'PLANNING', 'NO_HIT']:
        n = ind_counts.get(cat, 0)
        pct = n / len(independent) * 100 if independent else 0
        print(f"  {cat:12s}: {n:6d} ({pct:5.1f}%)")

    # By domain
    print(f"\n--- By Domain (independent errors) ---")
    domain_cats = defaultdict(lambda: Counter())
    for c, d in independent:
        domain_cats[d['domain']][c] += 1

    for domain in ['ppt', 'excel', 'word']:
        dc = domain_cats[domain]
        total = sum(dc.values())
        if total == 0:
            continue
        print(f"\n  {domain.upper()} (N={total}):")
        for cat in ['GROUNDING', 'PLANNING', 'NO_HIT']:
            n = dc.get(cat, 0)
            pct = n / total * 100 if total else 0
            print(f"    {cat:12s}: {n:5d} ({pct:5.1f}%)")

    # Distance distribution
    print(f"\n--- Coordinate Distance (independent errors) ---")
    for cat in ['GROUNDING', 'PLANNING', 'NO_HIT']:
        dists = [d['distance'] for c, d in independent if c == cat and 'distance' in d]
        if dists:
            dists.sort()
            n = len(dists)
            print(f"\n  {cat} (N={n}):")
            print(f"    Mean: {sum(dists)/n:.1f}px")
            print(f"    Median: {dists[n//2]:.1f}px")
            print(f"    P25: {dists[n//4]:.1f}px")
            print(f"    P75: {dists[3*n//4]:.1f}px")

    # Examples
    print(f"\n{'='*60}")
    print(f"EXAMPLES")
    print(f"{'='*60}")

    for cat in ['GROUNDING', 'PLANNING', 'NO_HIT']:
        cat_items = [(c, d) for c, d in independent if c == cat]
        if not cat_items:
            continue
        sample = random.sample(cat_items, min(5, len(cat_items)))
        print(f"\n--- {cat} Examples (N={len(cat_items)}) ---")
        for i, (c, d) in enumerate(sample):
            print(f"\n  [{i+1}] {d['sample_id']}")
            print(f"      Pred: ({d['pred_coord'][0]:.0f}, {d['pred_coord'][1]:.0f})  GT: ({d['gt_coord'][0]:.0f}, {d['gt_coord'][1]:.0f})  Dist: {d['distance']:.0f}px")
            print(f"      GT target: \"{d.get('gt_target', '')}\"")
            if 'clicked_element' in d:
                print(f"      Clicked: \"{d['clicked_element']}\" ({d.get('clicked_type', '')})")
            if 'nearest_element' in d:
                print(f"      Nearest: \"{d['nearest_element']}\" (dist={d.get('nearest_dist', 0):.0f}px)")
            if 'reason' in d:
                print(f"      Reason: {d['reason']}")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, output_name)
    save_data = {
        'model': label,
        'total_click_errors': total_click_errors,
        'classified': len(classifications),
        'distribution': {cat: cat_counts.get(cat, 0) for cat in ['CASCADE', 'NEAR_MISS', 'GROUNDING', 'PLANNING', 'NO_HIT']},
        'independent_distribution': {cat: ind_counts.get(cat, 0) for cat in ['GROUNDING', 'PLANNING', 'NO_HIT']},
        'by_domain': {domain: dict(domain_cats[domain]) for domain in ['ppt', 'excel', 'word']},
    }
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
