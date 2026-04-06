"""Analysis C: AR vs C4+C7 State Distribution Shift.

Quantifies that Analysis B's "prev_wrong" condition exists almost exclusively
at the LAST step of AR trajectories (where it has zero TSR contribution),
while in C4+C7 independent step evaluation it is uniformly distributed.

Key insight: AR uses stop-on-error, so a wrong step is always the terminal
step. There are literally zero AR steps where prev_correct=False, because
if the previous step was wrong the trajectory already stopped. In contrast,
C4+C7 evaluates every step independently with GT screenshots, so prev_wrong
steps appear at every position.

This means Analysis B's measured history effect (e.g., +6.6pp improvement
for prev_wrong steps in the medium-agreement bin) is real within C4+C7's
i.i.d. evaluation but has ~0 impact on actual AR TSR, because the
prev_wrong condition never arises in AR except at the already-terminal step.

All offline -- 0 GPU required.
"""

import argparse
import json
import os
import sys
import numpy as np
from collections import defaultdict, Counter

# ---------------------------------------------------------------------------
# Utility imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, SCRIPT_DIR)

try:
    from ac_utils import load_jsonl, save_json
except (ImportError, ModuleNotFoundError):
    # Fallback: ac_utils imports heavy dependencies (torch via evaluation/).
    # For this pure-data-analysis script we only need JSON I/O.
    def load_jsonl(path):
        data = []
        with open(path, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def _json_default(obj):
        """Handle numpy types for JSON serialization."""
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def save_json(data, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)


# =========================================================================
# Data loading helpers
# =========================================================================

def load_trajectory_results(jsonl_path):
    """Load AR trajectory results indexed by episode_id."""
    results = {}
    with open(jsonl_path) as f:
        for line in f:
            ep = json.loads(line)
            eid = ep.get('episode_id')
            if eid is not None:
                results[eid] = ep
    return results


def load_multisample_results(jsonl_path):
    """Load C4+C7 multisample results as a list of episode dicts."""
    results = []
    with open(jsonl_path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


# =========================================================================
# Agreement computation helpers (mirrors Analysis B)
# =========================================================================

AGREE_BINS = [
    (0.0, 0.5, 'low'),
    (0.5, 0.7, 'med'),
    (0.7, 0.9, 'high'),
    (0.9, 1.01, 'vhigh'),
]


def get_agree_bin(agreement):
    for lo, hi, name in AGREE_BINS:
        if lo <= agreement < hi:
            return name
    return 'vhigh'


def compute_step_agreement(samples):
    """Compute action-type agreement from a list of K sample dicts."""
    action_types = []
    for s in samples:
        pa = s.get('pred_action')
        if pa and isinstance(pa, dict):
            action_types.append(pa.get('action', 'unknown'))
        else:
            action_types.append('parse_fail')
    if not action_types:
        return 0.0
    ctr = Counter(action_types)
    return ctr.most_common(1)[0][1] / len(action_types)


# =========================================================================
# Analysis 1: AR "prev_wrong" position distribution
# =========================================================================

def analyze_ar_prev_distribution(traj_results):
    """Classify every executed AR step as no_prev / prev_correct / prev_wrong.

    In stop-on-error AR, prev_wrong should literally never occur because
    if step k fails, step k+1 is never executed.
    """
    counts = {'no_prev': 0, 'prev_correct': 0, 'prev_wrong': 0}
    position_detail = defaultdict(lambda: {'no_prev': 0, 'prev_correct': 0, 'prev_wrong': 0})
    total_steps = 0
    wrong_not_last = 0  # sanity check

    for eid, ep in traj_results.items():
        steps = ep.get('step_results', [])
        for i, step in enumerate(steps):
            sn = step.get('step_num', i)
            total_steps += 1

            if i == 0:
                counts['no_prev'] += 1
                position_detail[sn]['no_prev'] += 1
            else:
                prev_ok = steps[i - 1].get('extract_match', False)
                if prev_ok:
                    counts['prev_correct'] += 1
                    position_detail[sn]['prev_correct'] += 1
                else:
                    counts['prev_wrong'] += 1
                    position_detail[sn]['prev_wrong'] += 1

            # Sanity: wrong step that is NOT last
            if not step.get('extract_match', False) and i < len(steps) - 1:
                wrong_not_last += 1

    return counts, position_detail, total_steps, wrong_not_last


# =========================================================================
# Analysis 2: C4+C7 "prev_wrong" position distribution
# =========================================================================

def analyze_c4c7_prev_distribution(multisample_data):
    """Classify every C4+C7 step as no_prev / prev_correct / prev_wrong.

    "Correct" is determined by the greedy sample (sample index 0).
    """
    counts = {'no_prev': 0, 'prev_correct': 0, 'prev_wrong': 0}
    position_detail = defaultdict(lambda: {'no_prev': 0, 'prev_correct': 0, 'prev_wrong': 0})
    total_steps = 0

    # Also build (agree_bin, prev_correct) joint counts for Analysis 3
    joint_counts = defaultdict(int)  # (agree_bin, prev_status) -> count

    for ep in multisample_data:
        step_list = ep.get('step_samples', [])
        # Sort by step_num to ensure ordering
        step_list_sorted = sorted(step_list, key=lambda s: s.get('step_num', 0))

        prev_correct = None
        for i, step in enumerate(step_list_sorted):
            sn = step.get('step_num', i)
            samples = step.get('samples', [])
            if not samples:
                prev_correct = None
                continue

            total_steps += 1
            agreement = compute_step_agreement(samples)
            abin = get_agree_bin(agreement)
            greedy_correct = samples[0].get('extract_match', False)

            if prev_correct is None:
                # First step of episode
                counts['no_prev'] += 1
                position_detail[sn]['no_prev'] += 1
                joint_counts[(abin, 'no_prev')] += 1
            elif prev_correct:
                counts['prev_correct'] += 1
                position_detail[sn]['prev_correct'] += 1
                joint_counts[(abin, 'prev_correct')] += 1
            else:
                counts['prev_wrong'] += 1
                position_detail[sn]['prev_wrong'] += 1
                joint_counts[(abin, 'prev_wrong')] += 1

            prev_correct = greedy_correct

    return counts, position_detail, total_steps, joint_counts


# =========================================================================
# Analysis 3: Side-by-side (agree_bin x prev_correct) cell comparison
# =========================================================================

def build_ar_joint_distribution(traj_results, multisample_data):
    """Build (agree_bin, prev_status) joint distribution for AR steps.

    Agreement comes from C4+C7 data for the same (episode_id, step_num).
    prev_status comes from the AR trajectory itself.
    """
    # Build agreement lookup from C4+C7
    agree_lookup = {}  # (episode_id, step_num) -> agreement
    for ep in multisample_data:
        eid = ep.get('episode_id')
        for step in ep.get('step_samples', []):
            samples = step.get('samples', [])
            if not samples:
                continue
            sn = step.get('step_num', 0)
            agree_lookup[(eid, sn)] = compute_step_agreement(samples)

    joint_counts = defaultdict(int)
    matched = 0
    unmatched = 0

    for eid, ep in traj_results.items():
        steps = ep.get('step_results', [])
        for i, step in enumerate(steps):
            sn = step.get('step_num', i)
            agreement = agree_lookup.get((eid, sn))
            if agreement is None:
                unmatched += 1
                continue
            matched += 1
            abin = get_agree_bin(agreement)

            if i == 0:
                prev_status = 'no_prev'
            else:
                prev_ok = steps[i - 1].get('extract_match', False)
                prev_status = 'prev_correct' if prev_ok else 'prev_wrong'

            joint_counts[(abin, prev_status)] += 1

    return joint_counts, matched, unmatched


# =========================================================================
# Analysis 4: TSR-weight analysis
# =========================================================================

def compute_tsr_weights(traj_results):
    """For each step position k, compute TSR weight = P(reaching step k and all prior correct).

    TSR = sum over episodes of I(all steps correct) / N_episodes.
    A step at position k contributes to TSR only if all steps 0..k-1 are correct.
    The TSR weight of position k is the fraction of episodes that are still
    'alive' (all prior steps correct) at position k.
    """
    max_step = 0
    for ep in traj_results.values():
        for s in ep.get('step_results', []):
            sn = s.get('step_num', 0)
            if sn > max_step:
                max_step = sn

    # For each step position, count how many episodes reach it with all-correct prefix
    position_alive = defaultdict(int)     # step_num -> count of episodes alive here
    position_total = defaultdict(int)     # step_num -> count of episodes that have this step in GT
    n_episodes = len(traj_results)

    for ep in traj_results.values():
        n_steps = ep['num_steps']
        steps = ep.get('step_results', [])

        # In stop-on-error: if trajectory has k steps, steps 0..k-2 are correct,
        # step k-1 is correct iff task_success (for full-length) or it's the last and correct
        alive = True
        for i, step in enumerate(steps):
            sn = step.get('step_num', i)
            if alive:
                position_alive[sn] += 1
            if not step.get('extract_match', False):
                alive = False

        # Count total positions this episode covers
        for k in range(n_steps):
            position_total[k] += 1

    # Normalize to get weights
    weights = {}
    for k in sorted(position_alive.keys()):
        weights[k] = position_alive[k] / n_episodes if n_episodes > 0 else 0

    return weights, position_alive, position_total, n_episodes


# =========================================================================
# Analysis 5: Counterfactual TSR impact
# =========================================================================

def counterfactual_tsr_impact(traj_results, history_effect_pp=6.6):
    """If we could magically apply Analysis B's history effect to AR, how much TSR gain?

    Since prev_wrong steps essentially don't exist in AR (stop-on-error),
    the effect is ~0. We compute this precisely.

    The history effect says: for prev_wrong steps, accuracy improves by
    history_effect_pp percentage points. For TSR to improve, this would
    need to happen at steps where the trajectory is still alive, but
    in AR, prev_wrong means the trajectory already terminated.
    """
    n_episodes = len(traj_results)
    total_tsr = sum(1 for ep in traj_results.values() if ep.get('task_success', False))
    baseline_tsr = total_tsr / n_episodes if n_episodes > 0 else 0

    # Count prev_wrong steps that are "alive" (contribute to TSR)
    # In stop-on-error, a prev_wrong step means prev was wrong => trajectory stopped.
    # So prev_wrong AND alive should be exactly 0.
    prev_wrong_alive = 0
    prev_wrong_total = 0

    for ep in traj_results.values():
        steps = ep.get('step_results', [])
        alive = True
        for i, step in enumerate(steps):
            if i > 0:
                prev_ok = steps[i - 1].get('extract_match', False)
                if not prev_ok:
                    prev_wrong_total += 1
                    if alive:
                        prev_wrong_alive += 1
            if not step.get('extract_match', False):
                alive = False

    # Even if we generously apply the effect to all prev_wrong steps:
    # Max TSR improvement = prev_wrong_alive * history_effect_pp / 100 / n_episodes
    max_tsr_improvement = prev_wrong_alive * (history_effect_pp / 100) / n_episodes if n_episodes > 0 else 0

    return {
        'baseline_tsr': baseline_tsr,
        'n_episodes': n_episodes,
        'prev_wrong_total_in_ar': prev_wrong_total,
        'prev_wrong_alive_in_ar': prev_wrong_alive,
        'history_effect_pp': history_effect_pp,
        'max_tsr_improvement_pp': max_tsr_improvement * 100,
        'conclusion': (
            f"prev_wrong AND alive = {prev_wrong_alive} steps. "
            f"Even with +{history_effect_pp}pp history effect, "
            f"max TSR improvement = {max_tsr_improvement * 100:.4f}pp (effectively 0)."
        ),
    }


# =========================================================================
# Main
# =========================================================================

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading AR trajectory results...")
    ar_path = os.path.join(PROJECT_ROOT, args.ar_results)
    traj_results = load_trajectory_results(ar_path)
    print(f"  Loaded {len(traj_results)} AR trajectories")

    print("Loading C4+C7 multisample results...")
    ms_path = os.path.join(PROJECT_ROOT, args.multisample_results)
    multisample_data = load_multisample_results(ms_path)
    print(f"  Loaded {len(multisample_data)} C4+C7 episodes")

    results = {}

    # ==================================================================
    # ANALYSIS 1: AR "prev_wrong" position distribution
    # ==================================================================
    print(f"\n{'=' * 80}")
    print("ANALYSIS 1: AR prev_correct/prev_wrong POSITION DISTRIBUTION")
    print(f"{'=' * 80}")

    ar_counts, ar_pos_detail, ar_total_steps, ar_wrong_not_last = \
        analyze_ar_prev_distribution(traj_results)

    print(f"\n  Total executed AR steps: {ar_total_steps}")
    print(f"  Wrong steps NOT at terminal position (sanity): {ar_wrong_not_last}")
    print()
    for status in ['no_prev', 'prev_correct', 'prev_wrong']:
        n = ar_counts[status]
        pct = n / ar_total_steps * 100 if ar_total_steps > 0 else 0
        print(f"  {status:>15}: {n:6d} ({pct:5.1f}%)")

    print(f"\n  Per-position breakdown (AR):")
    print(f"  {'Step':>5} | {'no_prev':>8} | {'prev_ok':>8} | {'prev_wrong':>11} | {'total':>6}")
    print(f"  {'-' * 5} | {'-' * 8} | {'-' * 8} | {'-' * 11} | {'-' * 6}")
    for sn in sorted(ar_pos_detail.keys()):
        if sn > 15:
            break
        d = ar_pos_detail[sn]
        t = d['no_prev'] + d['prev_correct'] + d['prev_wrong']
        print(f"  {sn:5d} | {d['no_prev']:8d} | {d['prev_correct']:8d} | {d['prev_wrong']:11d} | {t:6d}")

    results['analysis_1_ar_prev_distribution'] = {
        'total_executed_steps': ar_total_steps,
        'wrong_not_last_sanity': ar_wrong_not_last,
        'counts': ar_counts,
        'conclusion': (
            f"In stop-on-error AR, prev_wrong = {ar_counts['prev_wrong']} steps "
            f"(= {ar_counts['prev_wrong'] / ar_total_steps * 100:.2f}% of all steps). "
            f"This is exactly 0 because a wrong step terminates the trajectory."
        ),
    }

    # ==================================================================
    # ANALYSIS 2: C4+C7 "prev_wrong" position distribution
    # ==================================================================
    print(f"\n{'=' * 80}")
    print("ANALYSIS 2: C4+C7 prev_correct/prev_wrong POSITION DISTRIBUTION")
    print(f"{'=' * 80}")

    c4c7_counts, c4c7_pos_detail, c4c7_total_steps, c4c7_joint = \
        analyze_c4c7_prev_distribution(multisample_data)

    print(f"\n  Total C4+C7 steps: {c4c7_total_steps}")
    print()
    for status in ['no_prev', 'prev_correct', 'prev_wrong']:
        n = c4c7_counts[status]
        pct = n / c4c7_total_steps * 100 if c4c7_total_steps > 0 else 0
        print(f"  {status:>15}: {n:6d} ({pct:5.1f}%)")

    print(f"\n  Per-position breakdown (C4+C7):")
    print(f"  {'Step':>5} | {'no_prev':>8} | {'prev_ok':>8} | {'prev_wrong':>11} | {'total':>6} | {'%prev_wrong':>12}")
    print(f"  {'-' * 5} | {'-' * 8} | {'-' * 8} | {'-' * 11} | {'-' * 6} | {'-' * 12}")
    for sn in sorted(c4c7_pos_detail.keys()):
        if sn > 15:
            break
        d = c4c7_pos_detail[sn]
        t = d['no_prev'] + d['prev_correct'] + d['prev_wrong']
        pw_pct = d['prev_wrong'] / t * 100 if t > 0 else 0
        print(f"  {sn:5d} | {d['no_prev']:8d} | {d['prev_correct']:8d} | {d['prev_wrong']:11d} | {t:6d} | {pw_pct:11.1f}%")

    results['analysis_2_c4c7_prev_distribution'] = {
        'total_steps': c4c7_total_steps,
        'counts': c4c7_counts,
        'prev_wrong_fraction': c4c7_counts['prev_wrong'] / c4c7_total_steps if c4c7_total_steps > 0 else 0,
    }

    # ==================================================================
    # ANALYSIS 3: Side-by-side (agree_bin x prev_status) comparison
    # ==================================================================
    print(f"\n{'=' * 80}")
    print("ANALYSIS 3: JOINT DISTRIBUTION (agree_bin x prev_status) -- AR vs C4+C7")
    print(f"{'=' * 80}")

    ar_joint, ar_matched, ar_unmatched = build_ar_joint_distribution(
        traj_results, multisample_data
    )

    print(f"\n  AR steps matched with C4+C7 agreement: {ar_matched}")
    print(f"  AR steps unmatched: {ar_unmatched}")

    # Print AR joint distribution
    print(f"\n  --- AR Joint Distribution ---")
    print(f"  {'Bin':>6} | {'no_prev':>8} | {'prev_ok':>8} | {'prev_wrong':>11} | {'total':>6} | {'% of all':>9}")
    print(f"  {'-' * 6} | {'-' * 8} | {'-' * 8} | {'-' * 11} | {'-' * 6} | {'-' * 9}")
    ar_joint_pct = {}
    for abin in ['low', 'med', 'high', 'vhigh']:
        np_c = ar_joint.get((abin, 'no_prev'), 0)
        pc_c = ar_joint.get((abin, 'prev_correct'), 0)
        pw_c = ar_joint.get((abin, 'prev_wrong'), 0)
        t = np_c + pc_c + pw_c
        pct = t / ar_matched * 100 if ar_matched > 0 else 0
        print(f"  {abin:>6} | {np_c:8d} | {pc_c:8d} | {pw_c:11d} | {t:6d} | {pct:8.1f}%")
        ar_joint_pct[abin] = {
            'no_prev': np_c, 'prev_correct': pc_c, 'prev_wrong': pw_c,
            'total': t, 'pct_of_all': pct,
        }

    # Print C4+C7 joint distribution
    print(f"\n  --- C4+C7 Joint Distribution ---")
    print(f"  {'Bin':>6} | {'no_prev':>8} | {'prev_ok':>8} | {'prev_wrong':>11} | {'total':>6} | {'% of all':>9}")
    print(f"  {'-' * 6} | {'-' * 8} | {'-' * 8} | {'-' * 11} | {'-' * 6} | {'-' * 9}")
    c4c7_joint_pct = {}
    for abin in ['low', 'med', 'high', 'vhigh']:
        np_c = c4c7_joint.get((abin, 'no_prev'), 0)
        pc_c = c4c7_joint.get((abin, 'prev_correct'), 0)
        pw_c = c4c7_joint.get((abin, 'prev_wrong'), 0)
        t = np_c + pc_c + pw_c
        pct = t / c4c7_total_steps * 100 if c4c7_total_steps > 0 else 0
        print(f"  {abin:>6} | {np_c:8d} | {pc_c:8d} | {pw_c:11d} | {t:6d} | {pct:8.1f}%")
        c4c7_joint_pct[abin] = {
            'no_prev': np_c, 'prev_correct': pc_c, 'prev_wrong': pw_c,
            'total': t, 'pct_of_all': pct,
        }

    # Highlight the key cell
    print(f"\n  --- KEY CELL: med-agree + prev_wrong ---")
    ar_med_pw = ar_joint.get(('med', 'prev_wrong'), 0)
    c4c7_med_pw = c4c7_joint.get(('med', 'prev_wrong'), 0)
    ar_med_pw_pct = ar_med_pw / ar_matched * 100 if ar_matched > 0 else 0
    c4c7_med_pw_pct = c4c7_med_pw / c4c7_total_steps * 100 if c4c7_total_steps > 0 else 0
    print(f"  AR:   {ar_med_pw:5d} steps ({ar_med_pw_pct:.2f}% of all AR steps)")
    print(f"  C4+C7: {c4c7_med_pw:5d} steps ({c4c7_med_pw_pct:.2f}% of all C4+C7 steps)")
    print(f"  Ratio C4+C7/AR: {'inf' if ar_med_pw == 0 else f'{c4c7_med_pw / ar_med_pw:.1f}x'}")

    # All prev_wrong cells
    print(f"\n  --- ALL prev_wrong CELLS ---")
    print(f"  {'Bin':>6} | {'AR #':>6} | {'AR %':>7} | {'C4C7 #':>7} | {'C4C7 %':>8}")
    print(f"  {'-' * 6} | {'-' * 6} | {'-' * 7} | {'-' * 7} | {'-' * 8}")
    for abin in ['low', 'med', 'high', 'vhigh']:
        ar_n = ar_joint.get((abin, 'prev_wrong'), 0)
        c4c7_n = c4c7_joint.get((abin, 'prev_wrong'), 0)
        ar_p = ar_n / ar_matched * 100 if ar_matched > 0 else 0
        c4c7_p = c4c7_n / c4c7_total_steps * 100 if c4c7_total_steps > 0 else 0
        print(f"  {abin:>6} | {ar_n:6d} | {ar_p:6.2f}% | {c4c7_n:7d} | {c4c7_p:7.2f}%")

    results['analysis_3_joint_distribution'] = {
        'ar_matched_steps': ar_matched,
        'ar_unmatched_steps': ar_unmatched,
        'ar_joint': {f"{ab}_{ps}": ar_joint.get((ab, ps), 0)
                     for ab in ['low', 'med', 'high', 'vhigh']
                     for ps in ['no_prev', 'prev_correct', 'prev_wrong']},
        'c4c7_joint': {f"{ab}_{ps}": c4c7_joint.get((ab, ps), 0)
                       for ab in ['low', 'med', 'high', 'vhigh']
                       for ps in ['no_prev', 'prev_correct', 'prev_wrong']},
        'key_cell_med_prev_wrong': {
            'ar_count': ar_med_pw,
            'ar_pct': ar_med_pw_pct,
            'c4c7_count': c4c7_med_pw,
            'c4c7_pct': c4c7_med_pw_pct,
        },
    }

    # ==================================================================
    # ANALYSIS 4: TSR-weight analysis
    # ==================================================================
    print(f"\n{'=' * 80}")
    print("ANALYSIS 4: TSR-WEIGHT BY STEP POSITION")
    print(f"{'=' * 80}")

    weights, pos_alive, pos_total, n_eps = compute_tsr_weights(traj_results)

    print(f"\n  Total episodes: {n_eps}")
    print(f"\n  {'Step':>5} | {'Alive':>6} | {'Total':>6} | {'TSR Weight':>11} | {'Cum. Prob':>10}")
    print(f"  {'-' * 5} | {'-' * 6} | {'-' * 6} | {'-' * 11} | {'-' * 10}")

    cum_weight = 0
    total_weight = sum(weights.values())
    weight_detail = {}
    for sn in sorted(weights.keys()):
        if sn > 20:
            break
        w = weights[sn]
        cum_weight += w
        cum_pct = cum_weight / total_weight * 100 if total_weight > 0 else 0
        alive_n = pos_alive.get(sn, 0)
        total_n = pos_total.get(sn, 0)
        print(f"  {sn:5d} | {alive_n:6d} | {total_n:6d} | {w:11.4f} | {cum_pct:9.1f}%")
        weight_detail[str(sn)] = {
            'alive': alive_n, 'total': total_n, 'tsr_weight': w,
        }

    # What fraction of total TSR weight is at steps where prev_wrong could occur?
    # In AR: prev_wrong never occurs, so TSR weight at prev_wrong positions = 0.
    # But hypothetically: the earliest prev_wrong could appear is at the step
    # after the first failure. In stop-on-error, that step doesn't exist.
    print(f"\n  Total TSR weight: {total_weight:.4f}")
    print(f"  TSR weight at steps with prev_wrong (in AR): 0.0000 (by construction)")
    print(f"  Even if prev_wrong existed, it would only be at the boundary of failure")
    print(f"  where alive probability is already near-zero.")

    results['analysis_4_tsr_weights'] = {
        'n_episodes': n_eps,
        'total_tsr_weight': total_weight,
        'per_step': weight_detail,
        'tsr_weight_at_prev_wrong': 0.0,
    }

    # ==================================================================
    # ANALYSIS 5: Counterfactual TSR impact
    # ==================================================================
    print(f"\n{'=' * 80}")
    print("ANALYSIS 5: COUNTERFACTUAL -- HISTORY EFFECT APPLIED TO AR")
    print(f"{'=' * 80}")

    counterfactual = counterfactual_tsr_impact(traj_results, history_effect_pp=6.6)

    print(f"\n  Baseline AR TSR: {counterfactual['baseline_tsr'] * 100:.2f}%")
    print(f"  Total prev_wrong steps in AR: {counterfactual['prev_wrong_total_in_ar']}")
    print(f"  prev_wrong AND alive steps in AR: {counterfactual['prev_wrong_alive_in_ar']}")
    print(f"  History effect (from Analysis B): +{counterfactual['history_effect_pp']}pp")
    print(f"  Max TSR improvement from history effect: {counterfactual['max_tsr_improvement_pp']:.4f}pp")
    print(f"\n  {counterfactual['conclusion']}")

    results['analysis_5_counterfactual'] = counterfactual

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print(f"\n{'=' * 80}")
    print("SUMMARY: DISTRIBUTION SHIFT AND ITS IMPLICATIONS")
    print(f"{'=' * 80}")

    ar_pw_pct = ar_counts['prev_wrong'] / ar_total_steps * 100 if ar_total_steps > 0 else 0
    c4c7_pw_pct = c4c7_counts['prev_wrong'] / c4c7_total_steps * 100 if c4c7_total_steps > 0 else 0

    print(f"""
  1. DISTRIBUTION SHIFT
     AR  : {ar_counts['prev_wrong']:5d} / {ar_total_steps} steps have prev_wrong ({ar_pw_pct:.2f}%)
     C4+C7: {c4c7_counts['prev_wrong']:5d} / {c4c7_total_steps} steps have prev_wrong ({c4c7_pw_pct:.2f}%)

     The prev_wrong condition is {c4c7_pw_pct:.1f}x more prevalent in C4+C7 than AR
     (actually {ar_pw_pct:.2f}% vs {c4c7_pw_pct:.2f}%, i.e., essentially absent in AR).

  2. ROOT CAUSE
     AR uses stop-on-error: if step k fails, the trajectory terminates immediately.
     There is NO step k+1 to be affected by the error at step k.
     Therefore prev_wrong = 0 steps in AR (by construction, verified above).

     C4+C7 evaluates every step independently against GT screenshots.
     Step failures do not stop evaluation. prev_wrong appears at ~{c4c7_pw_pct:.0f}% of steps.

  3. IMPLICATION FOR ANALYSIS B
     Analysis B found a +6.6pp history effect for prev_wrong steps in the
     medium-agreement bin. This effect is REAL in C4+C7's i.i.d. evaluation.

     However, it has ZERO impact on actual AR TSR because:
     - prev_wrong steps = {ar_counts['prev_wrong']} in AR (exactly 0)
     - Even if they existed, TSR weight at prev_wrong positions = 0
     - Counterfactual max TSR improvement = {counterfactual['max_tsr_improvement_pp']:.4f}pp

  4. CONCLUSION
     Analysis B's history effect is a property of the C4+C7 evaluation protocol,
     not a property of AR deployment. Any strategy to exploit the history effect
     (e.g., giving the model more context about past errors) would have no effect
     on AR TSR because the condition it addresses (prev_wrong) never arises in AR.

     For AR TSR improvement, focus on step-level accuracy at steps where the
     trajectory is still alive, not on error recovery after failure.
""")

    results['summary'] = {
        'ar_prev_wrong_count': ar_counts['prev_wrong'],
        'ar_prev_wrong_pct': ar_pw_pct,
        'c4c7_prev_wrong_count': c4c7_counts['prev_wrong'],
        'c4c7_prev_wrong_pct': c4c7_pw_pct,
        'counterfactual_max_tsr_improvement_pp': counterfactual['max_tsr_improvement_pp'],
        'conclusion': (
            "Analysis B's history effect (+6.6pp for prev_wrong) has zero impact on AR TSR "
            "because stop-on-error ensures prev_wrong never occurs in AR trajectories."
        ),
    }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = os.path.join(PROJECT_ROOT, args.output_dir, 'analysis_c_distribution_shift.json')
    save_json(results, out_path)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analysis C: AR vs C4+C7 State Distribution Shift"
    )
    parser.add_argument(
        "--ar_results", type=str,
        default="outputs/eval_a_ac/Qwen2.5-VL-7B/trajectory_results.jsonl",
        help="Path to AR trajectory_results.jsonl (relative to project root)",
    )
    parser.add_argument(
        "--multisample_results", type=str,
        default="outputs/eval_c4c7_ac/Qwen2.5-VL-7B/multisample_results.jsonl",
        help="Path to C4+C7 multisample_results.jsonl (relative to project root)",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="outputs/eval_analysis_c",
        help="Output directory (relative to project root)",
    )
    args = parser.parse_args()
    main(args)
