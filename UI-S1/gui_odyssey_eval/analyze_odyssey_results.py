#!/usr/bin/env python3
"""
GUI-Odyssey Failure Analysis — matching FULL_REPORT.md pipeline.

Reads both stop-on-error and no-stop AR evaluation results and produces
GUI_ODYSSEY_ANALYSIS.md with:
  A. AR Stop-on-Error Summary
  B. No-Stop Overall Metrics
  C. Error Cascade Analysis
  D. Error Type Breakdown
  E. Grounding vs Planning Classification
  F. Per-Category / Per-Device / Per-Length Breakdowns
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════

def load_jsonl(path):
    results = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def coord_distance(c1, c2):
    """Euclidean distance between two [x, y] coordinates (in [0,1000] space)."""
    if c1 is None or c2 is None:
        return None
    try:
        return math.sqrt((float(c1[0]) - float(c2[0])) ** 2 +
                         (float(c1[1]) - float(c2[1])) ** 2)
    except (TypeError, ValueError, IndexError):
        return None


def get_pred_coord_1k(step):
    """Get predicted coordinate in [0,1000] space from a step result."""
    c = step.get('pred_coord_1k')
    if c and isinstance(c, (list, tuple)) and len(c) >= 2:
        try:
            return (float(c[0]), float(c[1]))
        except (ValueError, TypeError):
            pass
    return None


def get_gt_coord_1k(step):
    """Get GT coordinate in [0,1000] space from a step result."""
    c = step.get('gt_coord_1k')
    if c and isinstance(c, (list, tuple)) and len(c) >= 2:
        try:
            return (float(c[0]), float(c[1]))
        except (ValueError, TypeError):
            pass
    return None


def pct(n, d):
    return n / d * 100 if d > 0 else 0.0


def fmt_pct(n, d):
    return f"{pct(n, d):.2f}%"


def percentile(sorted_list, p):
    if not sorted_list:
        return 0
    idx = int(p * len(sorted_list))
    idx = min(idx, len(sorted_list) - 1)
    return sorted_list[idx]


# ═══════════════════════════════════════════════════════════════════════
# Section A: AR Stop-on-Error Summary
# ═══════════════════════════════════════════════════════════════════════

def section_a_stop_summary(summary):
    """Generate Section A from stop-on-error summary.json."""
    lines = []
    lines.append("## 1. AR Stop-on-Error Summary")
    lines.append("")

    lines.append("### 1.1 Overall")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|:---:|")
    lines.append(f"| TSR | **{summary['tsr']*100:.2f}%** ({summary['success_count']}/{summary['n']}) |")
    lines.append(f"| Avg Progress | {summary['avg_progress']*100:.2f}% |")
    lines.append(f"| Scattered Progress | {summary['scattered_progress']*100:.2f}% |")
    lines.append(f"| Total Episodes | {summary['n']} |")
    lines.append("")

    # Per-action-type
    action_stats = summary.get('action_type_stats', {})
    if action_stats:
        lines.append("### 1.2 Per Action Type (Stop-on-Error)")
        lines.append("")
        lines.append("| Action Type | Total | Type Match | Extract Match |")
        lines.append("|-------------|:---:|:---:|:---:|")
        for at in sorted(action_stats.keys()):
            s = action_stats[at]
            lines.append(
                f"| {at} | {s['total']} | "
                f"{s.get('type_match_rate', 0)*100:.1f}% ({s['type_match']}) | "
                f"{s.get('extract_match_rate', 0)*100:.1f}% ({s['extract_match']}) |"
            )
        lines.append("")

    # Per-category
    cat_stats = summary.get('category_stats', {})
    if cat_stats:
        lines.append("### 1.3 Per Category (Stop-on-Error)")
        lines.append("")
        lines.append("| Category | N | TSR | Avg Progress | Scattered Progress |")
        lines.append("|----------|:---:|:---:|:---:|:---:|")
        for cat in sorted(cat_stats.keys()):
            m = cat_stats[cat]
            lines.append(
                f"| {cat} | {m['n']} | "
                f"{m['tsr']*100:.2f}% | {m['avg_progress']*100:.2f}% | "
                f"{m['scattered_progress']*100:.2f}% |"
            )
        lines.append("")

    # Per-device
    dev_stats = summary.get('device_stats', {})
    if dev_stats:
        lines.append("### 1.4 Per Device (Stop-on-Error)")
        lines.append("")
        lines.append("| Device | N | TSR | Avg Progress |")
        lines.append("|--------|:---:|:---:|:---:|")
        for dev in sorted(dev_stats.keys()):
            m = dev_stats[dev]
            lines.append(
                f"| {dev} | {m['n']} | "
                f"{m['tsr']*100:.2f}% | {m['avg_progress']*100:.2f}% |"
            )
        lines.append("")

    # Per-length
    len_stats = summary.get('length_bucket_stats', {})
    if len_stats:
        lines.append("### 1.5 Per Trajectory Length (Stop-on-Error)")
        lines.append("")
        lines.append("| Bucket | N | TSR | Avg Progress | Scattered Progress |")
        lines.append("|--------|:---:|:---:|:---:|:---:|")
        for b in sorted(len_stats.keys()):
            m = len_stats[b]
            lines.append(
                f"| {b} | {m['n']} | "
                f"{m['tsr']*100:.2f}% | {m['avg_progress']*100:.2f}% | "
                f"{m['scattered_progress']*100:.2f}% |"
            )
        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Section B: No-Stop Overall Metrics
# ═══════════════════════════════════════════════════════════════════════

def section_b_nostop_overall(trajectories):
    """Generate Section B from no-stop trajectory results."""
    lines = []
    lines.append("## 2. No-Stop Overall Metrics")
    lines.append("")

    n_traj = len(trajectories)
    total_steps = sum(r['num_steps'] for r in trajectories)
    total_correct = 0
    tsr_count = 0
    scattered_progs = []
    step0_fail = 0
    step0_total = 0
    post_error_correct = 0
    post_error_total = 0

    for r in trajectories:
        steps = r.get('step_results', [])
        n = r['num_steps']

        correct = sum(1 for s in steps if s['extract_match'])
        total_correct += correct

        if correct == n and len(steps) == n:
            tsr_count += 1

        scattered_progs.append(correct / n if n > 0 else 0)

        # Step-0
        if steps:
            step0_total += 1
            if not steps[0]['extract_match']:
                step0_fail += 1

        # Post-error accuracy
        first_err_idx = None
        for i, s in enumerate(steps):
            if not s['extract_match']:
                first_err_idx = i
                break
        if first_err_idx is not None and first_err_idx + 1 < len(steps):
            for s in steps[first_err_idx + 1:]:
                post_error_total += 1
                if s['extract_match']:
                    post_error_correct += 1

    step_acc = pct(total_correct, total_steps)
    tsr = pct(tsr_count, n_traj)
    scattered = sum(scattered_progs) / n_traj * 100 if n_traj > 0 else 0
    step0_failure = pct(step0_fail, step0_total)
    post_err_acc = pct(post_error_correct, post_error_total)

    lines.append("| Metric | Value |")
    lines.append("|--------|:---:|")
    lines.append(f"| TSR | **{tsr:.2f}%** ({tsr_count}/{n_traj}) |")
    lines.append(f"| Step Accuracy | {step_acc:.2f}% ({total_correct}/{total_steps}) |")
    lines.append(f"| Scattered Progress | {scattered:.2f}% |")
    lines.append(f"| Step-0 Failure Rate | {step0_failure:.2f}% ({step0_fail}/{step0_total}) |")
    lines.append(f"| Post-Error Accuracy | {post_err_acc:.2f}% ({post_error_correct}/{post_error_total}) |")
    lines.append(f"| Evaluated Steps | {total_steps} |")
    lines.append("")

    return "\n".join(lines), {
        'tsr': tsr,
        'step_acc': step_acc,
        'scattered': scattered,
        'step0_failure': step0_failure,
        'post_err_acc': post_err_acc,
        'total_steps': total_steps,
        'total_correct': total_correct,
        'n_traj': n_traj,
    }


# ═══════════════════════════════════════════════════════════════════════
# Section C: Error Cascade Analysis
# ═══════════════════════════════════════════════════════════════════════

def section_c_cascade(trajectories):
    """Generate Section C: cascade metrics from no-stop results."""
    lines = []
    lines.append("## 3. Error Cascade Analysis")
    lines.append("")

    n_traj = len(trajectories)

    # Survival probability P(k+1 correct | k correct)
    max_k = 10
    correct_at_k = defaultdict(int)
    correct_at_k_and_k1 = defaultdict(int)

    # Correct run from start, error run after first error
    correct_runs = []
    error_runs = []
    first_error_positions = []  # normalized

    for r in trajectories:
        steps = r.get('step_results', [])
        n = r['num_steps']

        # Survival
        for k in range(min(len(steps), max_k + 1)):
            if steps[k]['extract_match']:
                correct_at_k[k] += 1
                if k + 1 < len(steps) and steps[k + 1]['extract_match']:
                    correct_at_k_and_k1[k] += 1

        # Correct run from start
        cr = 0
        for s in steps:
            if s['extract_match']:
                cr += 1
            else:
                break
        else:
            cr = len(steps)
        correct_runs.append(cr)

        # First error position and error run
        first_err_idx = None
        for i, s in enumerate(steps):
            if not s['extract_match']:
                first_err_idx = i
                break
        if first_err_idx is not None:
            first_error_positions.append(first_err_idx / n if n > 0 else 0)
            er = 0
            for s in steps[first_err_idx:]:
                if not s['extract_match']:
                    er += 1
                else:
                    break
            error_runs.append(er)
        else:
            first_error_positions.append(1.0)

    mean_correct_run = sum(correct_runs) / len(correct_runs) if correct_runs else 0
    mean_error_run = sum(error_runs) / len(error_runs) if error_runs else 0
    mean_first_error = sum(first_error_positions) / len(first_error_positions) if first_error_positions else 0

    lines.append("### 3.1 Key Cascade Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|:---:|")
    lines.append(f"| Mean First Error Position (norm) | {mean_first_error:.4f} |")
    lines.append(f"| Mean Correct Run from Start | {mean_correct_run:.3f} steps |")
    lines.append(f"| Mean Error Run after First Error | {mean_error_run:.3f} steps |")
    lines.append(f"| Trajectories with Errors | {len(error_runs)}/{n_traj} |")
    lines.append("")

    # Survival probability table
    lines.append("### 3.2 Survival Probability P(step k+1 correct | step k correct)")
    lines.append("")
    lines.append("| Step k | P(k+1\\|k) | Correct@k | Correct@k+1 |")
    lines.append("|:---:|:---:|:---:|:---:|")
    for k in range(min(11, max_k + 1)):
        if correct_at_k[k] > 0:
            surv = correct_at_k_and_k1[k] / correct_at_k[k]
            lines.append(
                f"| {k} | {surv:.4f} | {correct_at_k[k]} | {correct_at_k_and_k1[k]} |"
            )
        elif k < 6:
            lines.append(f"| {k} | N/A | {correct_at_k[k]} | {correct_at_k_and_k1[k]} |")
    lines.append("")

    # Distribution of initial correct-run lengths
    lines.append("### 3.3 Distribution of Initial Correct-Run Length")
    lines.append("")
    lines.append("| Length | Count | % |")
    lines.append("|:---:|:---:|:---:|")
    cr_dist = Counter(correct_runs)
    for length in sorted(cr_dist.keys()):
        if length > 15:
            break
        lines.append(f"| {length} | {cr_dist[length]} | {pct(cr_dist[length], n_traj):.1f}% |")
    remaining = sum(v for k, v in cr_dist.items() if k > 15)
    if remaining > 0:
        lines.append(f"| >15 | {remaining} | {pct(remaining, n_traj):.1f}% |")
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Section D: Error Type Breakdown
# ═══════════════════════════════════════════════════════════════════════

def section_d_error_types(trajectories):
    """Generate Section D: error type breakdown from no-stop results."""
    lines = []
    lines.append("## 4. Error Type Breakdown")
    lines.append("")

    total_steps = 0
    total_failed = 0
    stuck_repeating = 0
    type_mismatch = 0
    coord_error = 0
    near_miss = 0
    other_error = 0

    coord_distances = []
    near_miss_distances = []
    dist_bins = {"<25": 0, "25-50": 0, "50-100": 0, "100-200": 0, "200-500": 0, ">500": 0}

    for r in trajectories:
        steps = r.get('step_results', [])
        total_steps += len(steps)

        for i, s in enumerate(steps):
            if s['extract_match']:
                continue

            total_failed += 1

            pred_coord = get_pred_coord_1k(s)
            gt_coord = get_gt_coord_1k(s)

            # Check stuck/repeating: same action type + same pred coord as previous step
            is_stuck = False
            if i > 0:
                prev = steps[i - 1]
                prev_pred_coord = get_pred_coord_1k(prev)
                if (s['pred_action'].get('action') == prev['pred_action'].get('action')
                        and pred_coord is not None
                        and prev_pred_coord is not None
                        and pred_coord == prev_pred_coord):
                    is_stuck = True
                    stuck_repeating += 1

            if is_stuck:
                continue

            # Type mismatch
            if not s['type_match']:
                type_mismatch += 1
                continue

            # Type matches — check coordinates
            if pred_coord is not None and gt_coord is not None:
                dist = coord_distance(pred_coord, gt_coord)
                if dist is not None:
                    coord_distances.append(dist)

                    if dist < 50:
                        near_miss += 1
                        near_miss_distances.append(dist)
                    else:
                        coord_error += 1

                    # Bin
                    if dist < 25:
                        dist_bins["<25"] += 1
                    elif dist < 50:
                        dist_bins["25-50"] += 1
                    elif dist < 100:
                        dist_bins["50-100"] += 1
                    elif dist < 200:
                        dist_bins["100-200"] += 1
                    elif dist < 500:
                        dist_bins["200-500"] += 1
                    else:
                        dist_bins[">500"] += 1
                else:
                    other_error += 1
            else:
                other_error += 1

    lines.append("### 4.1 Error Categories")
    lines.append("")
    lines.append(f"Total failed steps: **{total_failed}** / {total_steps}")
    lines.append("")
    lines.append("| Category | Count | % of Failed | % of All Steps |")
    lines.append("|----------|:---:|:---:|:---:|")
    categories = [
        ("Stuck/repeating", stuck_repeating),
        ("Type mismatch", type_mismatch),
        ("Coord error (>=50)", coord_error),
        ("Near miss (<50)", near_miss),
        ("Other (content etc.)", other_error),
    ]
    for name, count in categories:
        lines.append(
            f"| {name} | {count} | {pct(count, total_failed):.1f}% | {pct(count, total_steps):.1f}% |"
        )

    verify = stuck_repeating + type_mismatch + coord_error + near_miss + other_error
    ok = "YES" if verify == total_failed else f"NO ({total_failed - verify} unclassified)"
    lines.append(f"\nVerification: {verify} classified = {total_failed} total? **{ok}**")
    lines.append("")

    # Coordinate distance distribution
    if coord_distances:
        lines.append("### 4.2 Coordinate Distance Distribution (type-matched errors with coords)")
        lines.append("")
        lines.append(f"Total coord-comparable errors: {len(coord_distances)}")
        lines.append("")
        lines.append("| Distance | Count | % | Cumulative % |")
        lines.append("|----------|:---:|:---:|:---:|")
        cum = 0
        for bin_name in ["<25", "25-50", "50-100", "100-200", "200-500", ">500"]:
            cnt = dist_bins[bin_name]
            p = pct(cnt, len(coord_distances))
            cum += p
            lines.append(f"| {bin_name} | {cnt} | {p:.1f}% | {cum:.1f}% |")
        lines.append("")

        coord_distances.sort()
        lines.append("### 4.3 Coordinate Distance Statistics")
        lines.append("")
        lines.append("| Statistic | Value |")
        lines.append("|-----------|:---:|")
        lines.append(f"| Mean | {sum(coord_distances)/len(coord_distances):.1f} |")
        lines.append(f"| Median | {percentile(coord_distances, 0.5):.1f} |")
        lines.append(f"| P25 | {percentile(coord_distances, 0.25):.1f} |")
        lines.append(f"| P75 | {percentile(coord_distances, 0.75):.1f} |")
        lines.append(f"| P90 | {percentile(coord_distances, 0.90):.1f} |")
        lines.append(f"| P95 | {percentile(coord_distances, 0.95):.1f} |")
        lines.append(f"| Max | {coord_distances[-1]:.1f} |")
        lines.append("")

    return "\n".join(lines), {
        'total_failed': total_failed,
        'total_steps': total_steps,
        'stuck_repeating': stuck_repeating,
        'type_mismatch': type_mismatch,
        'coord_error': coord_error,
        'near_miss': near_miss,
        'other_error': other_error,
    }


# ═══════════════════════════════════════════════════════════════════════
# Section E: Grounding vs Planning
# ═══════════════════════════════════════════════════════════════════════

def section_e_grounding_planning(trajectories):
    """Generate Section E: grounding vs planning classification."""
    lines = []
    lines.append("## 5. Grounding vs Planning Classification")
    lines.append("")

    total_steps = 0
    total_success = 0
    classifications = []
    action_type_errors = defaultdict(lambda: Counter())

    CASCADE_TOL = 5  # tolerance in [0,1000] space

    for r in trajectories:
        steps = r.get('step_results', [])
        prev_coord = None

        for s in steps:
            total_steps += 1

            pred_coord = get_pred_coord_1k(s)

            if s['extract_match']:
                total_success += 1
                if pred_coord:
                    prev_coord = pred_coord
                continue

            # Failed step
            gt_type = s['gt_action_type']
            pred_type = s['pred_action'].get('action', '')
            type_match = s['type_match']

            # Cascade detection: same coord as previous step within tolerance
            is_cascade = False
            if prev_coord and pred_coord:
                if (abs(prev_coord[0] - pred_coord[0]) < CASCADE_TOL
                        and abs(prev_coord[1] - pred_coord[1]) < CASCADE_TOL):
                    is_cascade = True

            if is_cascade:
                category = 'CASCADE'
            elif type_match and not s['extract_match']:
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
                'episode_id': r.get('episode_id'),
            })

            action_type_errors[gt_type][category] += 1

            if pred_coord:
                prev_coord = pred_coord

    total_failed = total_steps - total_success
    cat_counts = Counter(c['category'] for c in classifications)

    lines.append("### 5.1 Overall Error Distribution")
    lines.append("")
    lines.append(f"Total steps: {total_steps} | Success: {total_success} ({pct(total_success, total_steps):.1f}%) | Failed: {total_failed}")
    lines.append("")
    lines.append("| Category | Count | % of All Errors |")
    lines.append("|----------|:---:|:---:|")
    for cat in ['CASCADE', 'GROUNDING', 'PLANNING', 'OTHER']:
        n = cat_counts.get(cat, 0)
        lines.append(f"| {cat} | {n} | {pct(n, len(classifications)):.1f}% |")
    lines.append("")

    # Independent (non-cascade)
    independent = [c for c in classifications if c['category'] != 'CASCADE']
    ind_counts = Counter(c['category'] for c in independent)
    lines.append("### 5.2 Independent Errors (excl CASCADE)")
    lines.append("")
    lines.append(f"Total: {len(independent)}")
    lines.append("")
    lines.append("| Category | Count | % |")
    lines.append("|----------|:---:|:---:|")
    for cat in ['GROUNDING', 'PLANNING', 'OTHER']:
        n = ind_counts.get(cat, 0)
        lines.append(f"| {cat} | {n} | {pct(n, len(independent)):.1f}% |")
    lines.append("")

    # By GT action type
    lines.append("### 5.3 By GT Action Type")
    lines.append("")
    lines.append("| GT Action | N | CASCADE | GROUNDING | PLANNING |")
    lines.append("|-----------|:---:|:---:|:---:|:---:|")
    for gt_type in sorted(action_type_errors.keys()):
        cats = action_type_errors[gt_type]
        total = sum(cats.values())
        cascade_p = pct(cats.get('CASCADE', 0), total)
        grounding_p = pct(cats.get('GROUNDING', 0), total)
        planning_p = pct(cats.get('PLANNING', 0), total)
        lines.append(
            f"| {gt_type} | {total} | {cascade_p:.1f}% | {grounding_p:.1f}% | {planning_p:.1f}% |"
        )
    lines.append("")

    # Planning error sub-analysis
    planning_pairs = Counter()
    for c in classifications:
        if c['category'] == 'PLANNING':
            planning_pairs[(c['gt_type'], c['pred_type'])] += 1

    if planning_pairs:
        lines.append("### 5.4 Planning Error: GT -> Predicted Action Type")
        lines.append("")
        lines.append("| GT | Predicted | Count |")
        lines.append("|-----|-----------|:---:|")
        for (gt, pred), n in planning_pairs.most_common(15):
            lines.append(f"| {gt} | {pred} | {n} |")
        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Section F: Per-Category / Per-Device / Per-Length Breakdowns (No-Stop)
# ═══════════════════════════════════════════════════════════════════════

def _nostop_breakdown_metrics(trajectories):
    """Compute no-stop metrics for a group of trajectories."""
    n = len(trajectories)
    if n == 0:
        return None
    total_steps = sum(r['num_steps'] for r in trajectories)
    total_correct = 0
    tsr_count = 0
    scattered_sum = 0
    step0_fail = 0
    step0_total = 0
    post_err_c = 0
    post_err_t = 0

    for r in trajectories:
        steps = r.get('step_results', [])
        ns = r['num_steps']
        correct = sum(1 for s in steps if s['extract_match'])
        total_correct += correct
        if correct == ns and len(steps) == ns:
            tsr_count += 1
        scattered_sum += correct / ns if ns > 0 else 0
        if steps:
            step0_total += 1
            if not steps[0]['extract_match']:
                step0_fail += 1
        first_err = None
        for i, s in enumerate(steps):
            if not s['extract_match']:
                first_err = i
                break
        if first_err is not None:
            for s in steps[first_err + 1:]:
                post_err_t += 1
                if s['extract_match']:
                    post_err_c += 1

    return {
        'n': n,
        'tsr': pct(tsr_count, n),
        'step_acc': pct(total_correct, total_steps),
        'scattered': scattered_sum / n * 100 if n > 0 else 0,
        'step0_fail': pct(step0_fail, step0_total),
        'post_err_acc': pct(post_err_c, post_err_t),
    }


def section_f_breakdowns(trajectories):
    """Generate Section F: per-category, per-device, per-length breakdowns."""
    lines = []
    lines.append("## 6. No-Stop Breakdowns")
    lines.append("")

    # Group by category
    by_cat = defaultdict(list)
    by_dev = defaultdict(list)
    by_len = defaultdict(list)
    for r in trajectories:
        by_cat[r.get('category', 'unknown')].append(r)
        by_dev[r.get('device_name', 'unknown')].append(r)
        by_len[r.get('length_bucket', 'unknown')].append(r)

    # Per-category
    lines.append("### 6.1 Per Category (No-Stop)")
    lines.append("")
    lines.append("| Category | N | TSR | Step Acc | Scattered | Step-0 Fail | Post-Err Acc |")
    lines.append("|----------|:---:|:---:|:---:|:---:|:---:|:---:|")
    for cat in sorted(by_cat.keys()):
        m = _nostop_breakdown_metrics(by_cat[cat])
        if m:
            lines.append(
                f"| {cat} | {m['n']} | {m['tsr']:.2f}% | {m['step_acc']:.2f}% | "
                f"{m['scattered']:.2f}% | {m['step0_fail']:.2f}% | {m['post_err_acc']:.2f}% |"
            )
    lines.append("")

    # Per-device
    lines.append("### 6.2 Per Device (No-Stop)")
    lines.append("")
    lines.append("| Device | N | TSR | Step Acc | Scattered | Step-0 Fail |")
    lines.append("|--------|:---:|:---:|:---:|:---:|:---:|")
    for dev in sorted(by_dev.keys()):
        m = _nostop_breakdown_metrics(by_dev[dev])
        if m:
            lines.append(
                f"| {dev} | {m['n']} | {m['tsr']:.2f}% | {m['step_acc']:.2f}% | "
                f"{m['scattered']:.2f}% | {m['step0_fail']:.2f}% |"
            )
    lines.append("")

    # Per-length
    lines.append("### 6.3 Per Trajectory Length (No-Stop)")
    lines.append("")
    lines.append("| Bucket | N | TSR | Step Acc | Scattered | Step-0 Fail | Post-Err Acc |")
    lines.append("|--------|:---:|:---:|:---:|:---:|:---:|:---:|")
    for b in sorted(by_len.keys()):
        m = _nostop_breakdown_metrics(by_len[b])
        if m:
            lines.append(
                f"| {b} | {m['n']} | {m['tsr']:.2f}% | {m['step_acc']:.2f}% | "
                f"{m['scattered']:.2f}% | {m['step0_fail']:.2f}% | {m['post_err_acc']:.2f}% |"
            )
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Verification
# ═══════════════════════════════════════════════════════════════════════

def section_verification(trajectories, error_counts):
    """Generate verification section."""
    lines = []
    lines.append("## 7. Verification")
    lines.append("")

    issues = []

    # Check 1: every no-stop episode should have num_steps step_results
    incomplete = 0
    for r in trajectories:
        if len(r.get('step_results', [])) != r['num_steps']:
            incomplete += 1
    if incomplete > 0:
        issues.append(f"WARNING: {incomplete} episodes have fewer step_results than num_steps")
    else:
        lines.append(f"- All {len(trajectories)} episodes have exactly num_steps step_results")

    # Check 2: error counts sum to total failed
    total_failed = error_counts['total_failed']
    classified = (error_counts['stuck_repeating'] + error_counts['type_mismatch']
                  + error_counts['coord_error'] + error_counts['near_miss']
                  + error_counts['other_error'])
    if classified != total_failed:
        issues.append(f"WARNING: Classified {classified} != total failed {total_failed}")
    else:
        lines.append(f"- Error counts sum correctly: {classified} = {total_failed} total failed")

    # Check 3: coord distances in [0,1000] range is implicit from the space

    for issue in issues:
        lines.append(f"- {issue}")

    lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main(args):
    report_lines = []
    report_lines.append("# GUI-Odyssey Failure Analysis")
    report_lines.append("")
    report_lines.append(f"Model: {args.model_name or '(see summary.json)'}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # ── Section A: Stop-on-Error Summary ──
    stop_summary_path = os.path.join(args.stop_dir, 'summary.json')
    if os.path.exists(stop_summary_path):
        with open(stop_summary_path) as f:
            stop_summary = json.load(f)
        report_lines.append(section_a_stop_summary(stop_summary))
        report_lines.append("---")
        report_lines.append("")
        print(f"Section A: Loaded stop-on-error summary (TSR={stop_summary['tsr']*100:.2f}%)")
    else:
        report_lines.append("## 1. AR Stop-on-Error Summary\n\n*No stop-on-error results found.*\n")
        print(f"WARNING: Stop summary not found at {stop_summary_path}")

    # ── Load no-stop trajectories ──
    nostop_jsonl = os.path.join(args.nostop_dir, 'trajectory_results.jsonl')
    if not os.path.exists(nostop_jsonl):
        print(f"ERROR: No-stop results not found at {nostop_jsonl}")
        return

    trajectories = load_jsonl(nostop_jsonl)
    print(f"Loaded {len(trajectories)} no-stop trajectories from {nostop_jsonl}")

    # ── Section B: No-Stop Overall ──
    section_b_text, nostop_metrics = section_b_nostop_overall(trajectories)
    report_lines.append(section_b_text)
    report_lines.append("---")
    report_lines.append("")
    print(f"Section B: No-stop TSR={nostop_metrics['tsr']:.2f}%, "
          f"StepAcc={nostop_metrics['step_acc']:.2f}%, "
          f"Scattered={nostop_metrics['scattered']:.2f}%")

    # ── Section C: Cascade Analysis ──
    report_lines.append(section_c_cascade(trajectories))
    report_lines.append("---")
    report_lines.append("")
    print("Section C: Cascade analysis done")

    # ── Section D: Error Type Breakdown ──
    section_d_text, error_counts = section_d_error_types(trajectories)
    report_lines.append(section_d_text)
    report_lines.append("---")
    report_lines.append("")
    print(f"Section D: Error types — stuck={error_counts['stuck_repeating']}, "
          f"type_mismatch={error_counts['type_mismatch']}, "
          f"coord_error={error_counts['coord_error']}, "
          f"near_miss={error_counts['near_miss']}, "
          f"other={error_counts['other_error']}")

    # ── Section E: Grounding vs Planning ──
    report_lines.append(section_e_grounding_planning(trajectories))
    report_lines.append("---")
    report_lines.append("")
    print("Section E: Grounding vs Planning done")

    # ── Section F: Breakdowns ──
    report_lines.append(section_f_breakdowns(trajectories))
    report_lines.append("---")
    report_lines.append("")
    print("Section F: Breakdowns done")

    # ── Verification ──
    report_lines.append(section_verification(trajectories, error_counts))

    # ── Write report ──
    report_text = "\n".join(report_lines)
    output_path = os.path.join(args.output_dir, 'GUI_ODYSSEY_ANALYSIS.md')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report_text)

    print(f"\nReport saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="GUI-Odyssey failure analysis (matching FULL_REPORT.md pipeline)")
    parser.add_argument('--stop_dir', type=str, required=True,
                        help='Directory with stop-on-error results (summary.json)')
    parser.add_argument('--nostop_dir', type=str, required=True,
                        help='Directory with no-stop results (trajectory_results.jsonl)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for GUI_ODYSSEY_ANALYSIS.md')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name (for report header)')
    args = parser.parse_args()
    main(args)
