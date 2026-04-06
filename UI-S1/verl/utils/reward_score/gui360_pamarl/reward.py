"""
PAMARL reward function for GUI-360.

Extends gui360 reward with:
1. Near-miss partial credit (click↔type confusion gets gradient instead of 0)
2. φ(t) position-aware weighting (step_num dependent reward scaling)

Both features are config-controlled and can be independently enabled/disabled.
"""

import copy
import math
import os
import traceback

import numpy as np

# Reuse all helpers from base gui360 reward
from verl.utils.reward_score.gui360.reward import (
    soft_coordinate_score,
    _lerp,
    _norm_coord,
    _check_text,
    _predict_direction,
    fm,
)


# ============================================================================
# Near-miss credit table (D10: click↔type = 55% of action errors)
# ============================================================================

NEAR_MISS_CREDIT = {
    # click ↔ type: 55% of all action errors
    ("click", "type"): 0.15,
    ("type", "click"): 0.15,
    # click variants
    ("click", "double_click"): 0.10,
    ("double_click", "click"): 0.10,
    ("click", "right_click"): 0.10,
    ("right_click", "click"): 0.10,
    ("click", "long_press"): 0.10,
    ("long_press", "click"): 0.10,
    # selection variants
    ("select_text", "select_paragraph"): 0.10,
    ("select_paragraph", "select_text"): 0.10,
    ("select_table", "select_table_range"): 0.10,
    ("select_table_range", "select_table"): 0.10,
    # click ↔ set_focus (functionally similar)
    ("click", "set_focus"): 0.08,
    ("set_focus", "click"): 0.08,
}


# ============================================================================
# φ(t) position-aware weighting functions
# ============================================================================

def phi_v2(t):
    """V2 (action agent) credit: decreases with step number."""
    return max(0.2, 1.0 / (1.0 + 0.3 * t))


def phi_v3(t):
    """V3 (grounding agent) credit: increases with step number."""
    return max(0.3, 1.0 - math.exp(-0.2 * t))


def phi_combined(t):
    """Combined φ for single-model training: weighted average.

    In single-model training (no V2/V3 separation), we use a blended φ
    that slightly upweights later steps (where both action and grounding matter).
    """
    v2 = phi_v2(t)
    v3 = phi_v3(t)
    # Normalize so φ(1) ≈ 1.0
    total = v2 + v3
    return total / (phi_v2(1) + phi_v3(1))  # normalized relative to step 1


# ============================================================================
# Per-action scoring with near-miss credit
# ============================================================================

def _score_action_pamarl(pred, gt, enable_near_miss=True):
    """
    Score predicted action against ground truth.

    Same as gui360._score_action but with near-miss partial credit:
    when action types don't match, check NEAR_MISS_CREDIT table
    instead of returning flat 0.0.

    Returns (type_match: bool, action_score: float, near_miss_used: bool).
    """
    gt_action = gt.get('action', '')
    pred_action = pred.get('action', '')

    # ---- click / long_press ----
    if gt_action in ('click', 'long_press'):
        if pred_action == gt_action:
            score = soft_coordinate_score(pred.get('coordinate'), gt.get('coordinate'))
            return True, score, False
        # Near-miss check
        if enable_near_miss:
            credit = NEAR_MISS_CREDIT.get((gt_action, pred_action), 0.0)
            if credit > 0:
                return False, credit, True
        return False, 0.0, False

    # ---- type / answer / key ----
    if gt_action in ('type', 'answer', 'key'):
        if pred_action == 'type':
            pred_txt = pred.get('text')
            gt_txt = gt.get('text')
            if pred_txt is None or gt_txt is None:
                return True, 0.0, False
            text_score = 1.0 if _check_text(pred_txt, gt_txt) else 0.0
            coord_score = soft_coordinate_score(pred.get('coordinate'), gt.get('coordinate'))
            return True, text_score * 0.8 + coord_score * 0.2, False
        # Near-miss check
        if enable_near_miss:
            credit = NEAR_MISS_CREDIT.get((gt_action, pred_action), 0.0)
            if credit > 0:
                return False, credit, True
        return False, 0.0, False

    # ---- empty action / FINISH / terminate / wait ----
    if gt_action in ('', 'terminate', 'wait'):
        if pred_action in ('', 'terminate', 'wait'):
            return True, 1.0, False
        return False, 0.0, False

    # ---- swipe ----
    if gt_action == 'swipe':
        if pred_action != 'swipe':
            return False, 0.0, False
        if 'direction' in gt and gt['direction'] is not None:
            gt_dir = gt['direction']
        elif gt.get('coordinate') is not None and gt.get('coordinate2') is not None:
            gt_dir = _predict_direction(gt['coordinate'], gt['coordinate2'])
        else:
            return True, 0.0, False
        if pred.get('coordinate') is None or pred.get('coordinate2') is None:
            return True, 0.0, False
        pred_dir = _predict_direction(pred['coordinate'], pred['coordinate2'])
        if gt_dir == 'down':
            gt_dir = 'up'
        elif gt_dir == 'up':
            gt_dir = 'down'
        return True, 1.0 if pred_dir == gt_dir else 0.0, False

    # ---- drag ----
    if gt_action == 'drag':
        if pred_action != 'drag':
            if enable_near_miss:
                credit = NEAR_MISS_CREDIT.get((gt_action, pred_action), 0.0)
                if credit > 0:
                    return False, credit, True
            return False, 0.0, False
        start_score = soft_coordinate_score(pred.get('coordinate'), gt.get('coordinate'))
        end_score = soft_coordinate_score(pred.get('coordinate2'), gt.get('coordinate2'))
        return True, (start_score + end_score) / 2.0, False

    # ---- select_text ----
    if gt_action == 'select_text':
        if pred_action != 'select_text':
            if enable_near_miss:
                credit = NEAR_MISS_CREDIT.get((gt_action, pred_action), 0.0)
                if credit > 0:
                    return False, credit, True
            return False, 0.0, False
        start_score = soft_coordinate_score(pred.get('coordinate'), gt.get('coordinate'))
        end_score = soft_coordinate_score(pred.get('coordinate2'), gt.get('coordinate2'))
        return True, (start_score + end_score) / 2.0, False

    # ---- wheel_mouse_input ----
    if gt_action == 'wheel_mouse_input':
        if pred_action != 'wheel_mouse_input':
            return False, 0.0, False
        if 'direction' in gt and gt['direction'] is not None:
            gt_dir = gt['direction']
        elif gt.get('coordinate') is not None and gt.get('coordinate2') is not None:
            gt_dir = _predict_direction(gt['coordinate'], gt['coordinate2'])
        else:
            return True, 0.0, False
        if 'direction' in pred and pred['direction'] is not None:
            pred_dir = pred['direction']
        elif pred.get('coordinate') is not None and pred.get('coordinate2') is not None:
            pred_dir = _predict_direction(pred['coordinate'], pred['coordinate2'])
        else:
            return True, 0.0, False
        return True, 1.0 if pred_dir == gt_dir else 0.0, False

    # ---- system_button ----
    if gt_action == 'system_button':
        if pred_action != 'system_button':
            return False, 0.0, False
        pred_btn = pred.get('button', '')
        gt_btn = gt.get('button', '')
        if not pred_btn or not gt_btn:
            return True, 0.0, False
        return True, 1.0 if pred_btn.lower().strip() == gt_btn.lower().strip() else 0.0, False

    # ---- open ----
    if gt_action == 'open':
        if pred_action != 'open':
            return False, 0.0, False
        pred_txt = pred.get('text')
        gt_txt = gt.get('text')
        if pred_txt is None or gt_txt is None:
            return True, 0.0, False
        return True, 1.0 if _check_text(pred_txt, gt_txt) else 0.0, False

    # ---- Rare action types: summary, select_table_range, set_font, etc. ----
    if pred_action == gt_action:
        all_match = True
        for k, v in gt.items():
            if k == 'action':
                continue
            if k not in pred or pred[k] != v:
                all_match = False
                break
        return True, 1.0 if all_match else 0.5, False

    # Near-miss fallback for rare actions
    if enable_near_miss:
        credit = NEAR_MISS_CREDIT.get((gt_action, pred_action), 0.0)
        if credit > 0:
            return False, credit, True

    return False, 0.0, False


# ============================================================================
# Main entry point
# ============================================================================

def gui360_pamarl_compute_score(solution_str, ground_truth, extra_info=None):
    """
    Compute PAMARL reward for a single GUI-360 step.

    Args:
        solution_str: Model's raw response string.
        ground_truth: Dict with 'check_options' (action dict) and 'num_steps'.
        extra_info: Dict with width, height, resized_width, resized_height,
                    and optionally:
                    - step_num (int): current step index (1-based)
                    - enable_near_miss (bool): enable near-miss credit (default True)
                    - enable_phi_t (bool): enable φ(t) weighting (default False)
    Returns:
        Dict with score, format_score, type_match, extract_match, step_reward,
        plus near_miss_used, phi_weight.
    """
    check_options = ground_truth['check_options']
    num_steps = ground_truth['num_steps']

    if extra_info is None:
        extra_info = {}
    width = extra_info.get('width', 1024)
    height = extra_info.get('height', 720)
    resized_width = extra_info.get('resized_width', 1024)
    resized_height = extra_info.get('resized_height', 720)

    # PAMARL config — check extra_info first, then env var fallback
    enable_near_miss = extra_info.get('enable_near_miss', True)
    enable_phi_t = extra_info.get('enable_phi_t',
                                  os.environ.get('PAMARL_ENABLE_PHI_T', '0') == '1')
    step_num = extra_info.get('step_num', 1)

    format_score = 0.0
    action_score = 0.0
    type_match = False
    extract_match = False
    near_miss_used = False

    try:
        result = fm.parse_response(solution_str)
        think_str = result.get('think')
        if think_str is not None and think_str.strip():
            format_score = 1.0

        if 'action_content' in result:
            pred_action = _norm_coord(result['action_content'], resized_width, resized_height)
            gt_action = _norm_coord(check_options, width, height)

            type_match, action_score, near_miss_used = _score_action_pamarl(
                pred_action, gt_action, enable_near_miss=enable_near_miss
            )

            annotation = check_options.get('annotation', 'GOOD')
            if annotation == 'HARMFUL':
                action_score = 1.0 - action_score
            elif annotation == 'NEUTRAL':
                action_score = 0.5
        else:
            action_score = 0.0
    except Exception:
        traceback.print_exc()
        print("Error Response:")
        print(solution_str)

    # Base score (same formula as gui360)
    base_score = format_score * 0.1 + action_score * 0.9

    # φ(t) position weighting
    phi_weight = 1.0
    if enable_phi_t and step_num >= 1:
        phi_weight = phi_combined(step_num)

    score = base_score * phi_weight

    extract_match = type_match and action_score >= 0.8
    step_reward = (1.0 / num_steps if extract_match else 0.0) * phi_weight

    return {
        "score": score,
        "format_score": format_score,
        "type_match": float(type_match),
        "extract_match": float(extract_match),
        "step_reward": step_reward,
        "near_miss_used": float(near_miss_used),
        "phi_weight": phi_weight,
        "action_score": action_score,
    }
