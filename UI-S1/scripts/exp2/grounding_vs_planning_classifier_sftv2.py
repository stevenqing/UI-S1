"""
Systematic Grounding vs Planning error classifier for GUI-360.

Uses three signals:
1. GT step.thought vs Model thoughts → intent matching
2. GT step.action.control_test → GT target element name
3. GT step.control_infos → hit-test: what UI element did the model actually click?

Classification:
- GROUNDING: model's intent matches GT intent, but coords are wrong
- PLANNING: model's intent differs from GT intent (wrong action/element)
- CASCADE: stuck/repeating same coord as previous step
- NEAR_MISS: type matches, coord distance < 50px
"""

import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
import random

random.seed(42)

# === Paths ===
EVAL_RESULTS = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/results/sft_v2/gui360/nostop_20260320_053216/ar_evaluation_results_20260320_055609.json"
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
    """Parse sample_id like 'ppt_in_app_ppt_1_4_3' into (execution_id, step_id).

    Format: {domain}_{category}_{execution_id}_{step_num}
    execution_id can contain underscores, so we need to be careful.
    The last number is step_num, and before that is the execution_id.
    """
    # sample_id format: domain_category_execid_stepnum
    # e.g., ppt_in_app_ppt_1_4_3 → domain=ppt, category=in_app, exec=ppt_1_4, step=3
    # e.g., excel_online_excel_3_85_2 → domain=excel, category=online, exec=excel_3_85, step=2
    parts = sample_id.rsplit('_', 1)
    if len(parts) != 2:
        return None, None
    step_num = int(parts[1])
    # The prefix contains domain_category_execution_id
    # execution_id is domain_N_M format
    prefix = parts[0]
    # Find domain
    for domain in ['ppt', 'excel', 'word']:
        if prefix.startswith(domain + '_'):
            rest = prefix[len(domain) + 1:]
            # rest = category_execution_id
            # categories: in_app, online, search
            for cat in ['in_app', 'online', 'search']:
                if rest.startswith(cat + '_'):
                    exec_id = rest[len(cat) + 1:]
                    return exec_id, step_num
    return None, None


def hit_test(x, y, controls):
    """Find which UI control the coordinate (x, y) falls on.
    Returns (control_text, control_type, distance) or None.
    """
    hits = []
    for i, c in enumerate(controls):
        r = c['control_rect']  # [left, top, right, bottom]
        if r[0] <= x <= r[2] and r[1] <= y <= r[3]:
            cx = (r[0] + r[2]) / 2
            cy = (r[1] + r[3]) / 2
            d = math.sqrt((x - cx)**2 + (y - cy)**2)
            hits.append((d, i, c))

    if hits:
        hits.sort(key=lambda t: t[0])
        best = hits[0][2]
        return best['control_text'], best['control_type'], hits[0][0]

    # No exact hit — find nearest
    nearest = []
    for i, c in enumerate(controls):
        r = c['control_rect']
        cx = (r[0] + r[2]) / 2
        cy = (r[1] + r[3]) / 2
        d = math.sqrt((x - cx)**2 + (y - cy)**2)
        nearest.append((d, i, c))

    if nearest:
        nearest.sort(key=lambda t: t[0])
        best = nearest[0][2]
        return best['control_text'], best['control_type'], nearest[0][0]

    return None, None, None


def normalize_text(text):
    """Normalize text for comparison."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def extract_intent_keywords(thought):
    """Extract key action intent words from a thought string."""
    if not thought:
        return set()

    thought = thought.lower()

    # Remove tool_call JSON blocks
    thought = re.sub(r'<tool_call>.*?</tool_call>', '', thought, flags=re.DOTALL)
    thought = re.sub(r'\{[^}]+\}', '', thought)

    # Extract quoted strings (often UI element names)
    quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", thought)
    keywords = set()
    for q in quoted:
        for part in q:
            if part:
                keywords.add(normalize_text(part))

    # Also extract key action verbs and targets
    words = normalize_text(thought).split()
    keywords.update(words)

    return keywords


def intent_match_score(model_thought, gt_thought, gt_control_text):
    """Compute a matching score between model intent and GT intent.

    Returns: (score, match_type)
    - score: 0.0 to 1.0
    - match_type: 'exact_control', 'high_overlap', 'partial', 'no_match'
    """
    if not model_thought or not gt_thought:
        return 0.0, 'no_thought'

    model_lower = model_thought.lower()
    gt_lower = gt_thought.lower()

    # Method 1: Check if GT target control name appears in model thought
    if gt_control_text:
        gt_ctrl_lower = gt_control_text.lower().strip()
        if gt_ctrl_lower and len(gt_ctrl_lower) > 2:
            if gt_ctrl_lower in model_lower:
                return 1.0, 'exact_control'

    # Method 2: Extract quoted element names from both and compare
    model_quoted = set()
    for m in re.findall(r"'([^']+)'|\"([^\"]+)\"", model_lower):
        for part in m:
            if part and len(part) > 1:
                model_quoted.add(part.strip())

    gt_quoted = set()
    for m in re.findall(r"'([^']+)'|\"([^\"]+)\"", gt_lower):
        for part in m:
            if part and len(part) > 1:
                gt_quoted.add(part.strip())

    if model_quoted and gt_quoted:
        overlap = model_quoted & gt_quoted
        if overlap:
            score = len(overlap) / max(len(model_quoted), len(gt_quoted))
            if score >= 0.5:
                return score, 'quoted_match'

    # Method 3: Word overlap (excluding stop words)
    stop_words = {'the', 'a', 'an', 'to', 'in', 'on', 'of', 'for', 'is', 'it', 'i',
                  'need', 'will', 'should', 'can', 'this', 'that', 'with', 'from',
                  'click', 'tap', 'press', 'select', 'choose',  # action verbs are shared
                  'next', 'step', 'first', 'then', 'now', 'current', 'screenshot'}

    model_words = set(normalize_text(model_thought).split()) - stop_words
    gt_words = set(normalize_text(gt_thought).split()) - stop_words

    # Focus on nouns/UI element names (words that are likely element identifiers)
    # Filter to words > 3 chars (skip 'tab', 'the', etc.)
    model_content = {w for w in model_words if len(w) > 3}
    gt_content = {w for w in gt_words if len(w) > 3}

    if model_content and gt_content:
        overlap = model_content & gt_content
        union = model_content | gt_content
        jaccard = len(overlap) / len(union) if union else 0
        if jaccard >= 0.3:
            return jaccard, 'word_overlap'

    return 0.0, 'no_match'


def classify_error(model_thought, gt_thought, gt_control_text,
                   pred_coord, gt_coord, gt_rect, controls,
                   prev_pred_coord):
    """Classify a single click error.

    Returns: (category, details_dict)
    Categories: CASCADE, NEAR_MISS, GROUNDING, PLANNING, AMBIGUOUS
    """
    # 1. Check cascade (stuck)
    if prev_pred_coord is not None:
        px, py = prev_pred_coord
        cx, cy = pred_coord
        if px is not None and py is not None and cx is not None and cy is not None:
            if abs(px - cx) < 2 and abs(py - cy) < 2:
                return 'CASCADE', {'reason': 'same coord as previous step'}

    # 2. Check near miss
    if pred_coord[0] is None or pred_coord[1] is None or gt_coord[0] is None or gt_coord[1] is None:
        return 'AMBIGUOUS', {'reason': 'null coordinates'}
    dist = math.sqrt((pred_coord[0] - gt_coord[0])**2 + (pred_coord[1] - gt_coord[1])**2)
    if dist < 50:
        return 'NEAR_MISS', {'distance': dist}

    # 3. Hit test: what did the model actually click?
    clicked_text, clicked_type, clicked_dist = hit_test(pred_coord[0], pred_coord[1], controls)

    # 4. Intent matching
    score, match_type = intent_match_score(model_thought, gt_thought, gt_control_text)

    # 5. Check if the model clicked the GT element (within rect)
    gt_r = gt_rect
    if gt_r:
        in_gt_rect = (gt_r['left'] <= pred_coord[0] <= gt_r['right'] and
                      gt_r['top'] <= pred_coord[1] <= gt_r['bottom'])
    else:
        in_gt_rect = False

    details = {
        'distance': dist,
        'intent_score': score,
        'intent_match_type': match_type,
        'clicked_element': clicked_text,
        'clicked_type': clicked_type,
        'gt_target': gt_control_text,
        'model_intent_summary': model_thought[:200] if model_thought else '',
        'gt_intent_summary': gt_thought[:200] if gt_thought else '',
    }

    # Classification logic
    if score >= 0.5:
        # Model's intent matches GT → grounding error (can't locate the element)
        return 'GROUNDING', details
    elif score > 0 and match_type == 'word_overlap':
        # Partial match — could be either
        # Additional check: did model click on a different meaningful element?
        if clicked_text and gt_control_text:
            if normalize_text(clicked_text) == normalize_text(gt_control_text):
                return 'GROUNDING', details  # Clicked right element but coords outside rect
            else:
                return 'PLANNING', details  # Different element
        return 'AMBIGUOUS', details
    elif match_type == 'no_thought':
        # No thought available — use hit test only
        if clicked_text and gt_control_text:
            # Check if clicked element is semantically similar to GT target
            if normalize_text(clicked_text) == normalize_text(gt_control_text):
                return 'GROUNDING', details
        return 'AMBIGUOUS', details
    else:
        # Intent doesn't match → planning error
        return 'PLANNING', details


def main():
    print("Loading GT data...")
    gt_index = load_gt_data()
    print(f"  Loaded {len(gt_index)} GT steps")

    print("Loading eval results...")
    with open(EVAL_RESULTS, 'r') as f:
        eval_data = json.load(f)

    trajectories = eval_data['trajectory_results']
    print(f"  Loaded {len(trajectories)} trajectories")

    # Build detailed_results index by sample_id for quick lookup
    detailed_by_id = {}
    for dr in eval_data['detailed_results']:
        detailed_by_id[dr['sample_id']] = dr

    # Process all click errors
    classifications = []
    match_failures = 0
    total_click_errors = 0

    for traj in trajectories:
        prev_pred_coord = None

        for step_result in traj['step_results']:
            sample_id = step_result['sample_id']

            # Get detailed result
            dr = detailed_by_id.get(sample_id, step_result)

            pred_func = dr.get('predicted_function', '')
            gt_func = dr.get('ground_truth_function', '')

            # Only analyze click-type errors where function matches
            if dr.get('success', False):
                # Track coord for cascade detection
                pa = dr.get('predicted_args', {})
                if isinstance(pa, dict) and 'coordinate' in pa:
                    c = pa['coordinate']
                    if isinstance(c, list) and len(c) >= 2:
                        prev_pred_coord = (c[0], c[1])
                continue

            # Only look at click errors (both predicted and GT are click)
            if pred_func != 'click' or gt_func != 'click':
                pa = dr.get('predicted_args', {})
                if isinstance(pa, dict) and 'coordinate' in pa:
                    c = pa['coordinate']
                    if isinstance(c, list) and len(c) >= 2:
                        prev_pred_coord = (c[0], c[1])
                continue

            total_click_errors += 1

            # Get predicted coordinate
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

            # Get GT coordinate
            ga = dr.get('ground_truth_args', {})
            gt_c = ga.get('coordinate', [0, 0])
            gt_coord = (gt_c[0], gt_c[1])

            gt_rect = dr.get('ground_truth_rect', None)

            # Parse sample_id to find GT data
            exec_id, step_num = parse_sample_id(sample_id)

            gt_thought = None
            gt_control_text = None
            controls = []

            if exec_id and (exec_id, step_num) in gt_index:
                gt_step = gt_index[(exec_id, step_num)]
                gt_thought = gt_step['step'].get('thought', '')
                gt_action = gt_step['step'].get('action', {})
                gt_control_text = gt_action.get('control_test', '')
                control_infos = gt_step['step'].get('control_infos', {})
                controls = control_infos.get('uia_controls_info', [])
            else:
                match_failures += 1

            model_thought = dr.get('thoughts', '')

            category, details = classify_error(
                model_thought, gt_thought, gt_control_text,
                pred_coord, gt_coord, gt_rect, controls,
                prev_pred_coord
            )

            details['sample_id'] = sample_id
            details['domain'] = traj.get('domain', '')
            details['category'] = traj.get('category', '')
            details['pred_coord'] = pred_coord
            details['gt_coord'] = gt_coord

            classifications.append((category, details))
            prev_pred_coord = pred_coord

    print(f"\n{'='*60}")
    print(f"GROUNDING vs PLANNING CLASSIFICATION RESULTS")
    print(f"{'='*60}")
    print(f"Total click errors analyzed: {total_click_errors}")
    print(f"Classified: {len(classifications)}")
    print(f"GT match failures: {match_failures}")

    # Aggregate
    cat_counts = Counter(c for c, _ in classifications)
    print(f"\n--- Overall Distribution ---")
    for cat in ['CASCADE', 'NEAR_MISS', 'GROUNDING', 'PLANNING', 'AMBIGUOUS']:
        n = cat_counts.get(cat, 0)
        pct = n / len(classifications) * 100 if classifications else 0
        print(f"  {cat:12s}: {n:6d} ({pct:5.1f}%)")

    # Non-cascade, non-near-miss breakdown
    independent = [(c, d) for c, d in classifications if c not in ('CASCADE', 'NEAR_MISS')]
    ind_counts = Counter(c for c, _ in independent)
    print(f"\n--- Independent Errors (excl cascade & near-miss) ---")
    print(f"  Total: {len(independent)}")
    for cat in ['GROUNDING', 'PLANNING', 'AMBIGUOUS']:
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
        print(f"\n  {domain.upper()} (N={total}):")
        for cat in ['GROUNDING', 'PLANNING', 'AMBIGUOUS']:
            n = dc.get(cat, 0)
            pct = n / total * 100 if total else 0
            print(f"    {cat:12s}: {n:5d} ({pct:5.1f}%)")

    # Intent match type breakdown
    print(f"\n--- Intent Match Types (for GROUNDING & PLANNING) ---")
    match_types = Counter()
    for c, d in independent:
        if c in ('GROUNDING', 'PLANNING'):
            match_types[d.get('intent_match_type', 'unknown')] += 1
    for mt, n in match_types.most_common():
        print(f"  {mt:20s}: {n}")

    # Distance distribution for GROUNDING vs PLANNING
    print(f"\n--- Coordinate Distance (independent errors) ---")
    for cat in ['GROUNDING', 'PLANNING']:
        dists = [d['distance'] for c, d in independent if c == cat and 'distance' in d]
        if dists:
            dists.sort()
            print(f"\n  {cat}:")
            print(f"    Mean: {sum(dists)/len(dists):.1f}px")
            print(f"    Median: {dists[len(dists)//2]:.1f}px")
            print(f"    P25: {dists[len(dists)//4]:.1f}px")
            print(f"    P75: {dists[3*len(dists)//4]:.1f}px")

    # Hit test analysis: what did the model actually click?
    print(f"\n--- Hit Test Analysis (independent errors) ---")
    for cat in ['GROUNDING', 'PLANNING']:
        items = [(c, d) for c, d in independent if c == cat]
        clicked_same = sum(1 for _, d in items
                         if d.get('clicked_element') and d.get('gt_target')
                         and normalize_text(d['clicked_element']) == normalize_text(d['gt_target']))
        clicked_diff = sum(1 for _, d in items
                         if d.get('clicked_element') and d.get('gt_target')
                         and normalize_text(d['clicked_element']) != normalize_text(d['gt_target']))
        clicked_none = sum(1 for _, d in items
                         if not d.get('clicked_element') or not d.get('gt_target'))
        print(f"\n  {cat} (N={len(items)}):")
        print(f"    Clicked GT element: {clicked_same} ({clicked_same/len(items)*100:.1f}%)")
        print(f"    Clicked different element: {clicked_diff} ({clicked_diff/len(items)*100:.1f}%)")
        print(f"    No hit/no GT target: {clicked_none} ({clicked_none/len(items)*100:.1f}%)")

    # Print examples
    print(f"\n{'='*60}")
    print(f"EXAMPLES")
    print(f"{'='*60}")

    for cat in ['GROUNDING', 'PLANNING']:
        cat_items = [(c, d) for c, d in independent if c == cat]
        sample = random.sample(cat_items, min(10, len(cat_items)))
        print(f"\n--- {cat} Examples ---")
        for i, (c, d) in enumerate(sample):
            print(f"\n  [{i+1}] {d['sample_id']}")
            print(f"      Pred coord: ({d['pred_coord'][0]:.0f}, {d['pred_coord'][1]:.0f})")
            print(f"      GT coord:   ({d['gt_coord'][0]:.0f}, {d['gt_coord'][1]:.0f})")
            print(f"      Distance:   {d['distance']:.0f}px")
            print(f"      GT target:  \"{d.get('gt_target', '')}\"")
            print(f"      Clicked:    \"{d.get('clicked_element', '')}\" ({d.get('clicked_type', '')})")
            print(f"      Intent score: {d['intent_score']:.2f} ({d['intent_match_type']})")
            gt_summary = d.get('gt_intent_summary', '')[:150]
            model_summary = d.get('model_intent_summary', '')[:150]
            print(f"      GT thought: {gt_summary}")
            print(f"      Model thought: {model_summary}")

    # Save full results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "grounding_vs_planning_classification_sftv2.json")
    save_data = {
        'summary': {
            'total_click_errors': total_click_errors,
            'classified': len(classifications),
            'gt_match_failures': match_failures,
            'distribution': {cat: cat_counts.get(cat, 0) for cat in ['CASCADE', 'NEAR_MISS', 'GROUNDING', 'PLANNING', 'AMBIGUOUS']},
            'independent_distribution': {cat: ind_counts.get(cat, 0) for cat in ['GROUNDING', 'PLANNING', 'AMBIGUOUS']},
        },
        'by_domain': {
            domain: dict(domain_cats[domain]) for domain in ['ppt', 'excel', 'word']
        },
    }
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
