"""
GUI-360 dedicated reward function with soft coordinate scoring.

Key differences from gui_traj.py (UI-S1):
- Soft (continuous) coordinate reward instead of binary hit/miss
- Handles GUI-360 specific action types: drag, select_text, wheel_mouse_input,
  summary, select_table_range, set_font, etc.
- No bbox support needed (GUI-360 data has no bboxes)
"""

import copy
import traceback

import numpy as np
from x.data.agent.sftv2 import SFTv2Format
from x.data.agent.space.std_space import RAW_SPACE

fm = SFTv2Format(RAW_SPACE)

# ============================================================================
# Soft coordinate reward
# ============================================================================

# Thresholds in normalized [0,1] coordinate space (~1024x720 pixels)
_EXACT_THRESH = 0.02   # ~20px: full score
_NEAR_THRESH = 0.05    # ~50px: interpolate 1.0 → 0.3
_MEDIUM_THRESH = 0.10  # ~100px: interpolate 0.3 → 0.1
_FAR_THRESH = 0.15     # ~150px: interpolate 0.1 → 0.0


def soft_coordinate_score(pred_coord, gt_coord):
    """Compute a continuous [0,1] reward based on Euclidean distance in normalized space."""
    if pred_coord is None or gt_coord is None:
        return 0.0
    dist = np.linalg.norm([pred_coord[0] - gt_coord[0], pred_coord[1] - gt_coord[1]])
    if dist <= _EXACT_THRESH:
        return 1.0
    elif dist <= _NEAR_THRESH:
        return _lerp(1.0, 0.3, (dist - _EXACT_THRESH) / (_NEAR_THRESH - _EXACT_THRESH))
    elif dist <= _MEDIUM_THRESH:
        return _lerp(0.3, 0.1, (dist - _NEAR_THRESH) / (_MEDIUM_THRESH - _NEAR_THRESH))
    elif dist <= _FAR_THRESH:
        return _lerp(0.1, 0.0, (dist - _MEDIUM_THRESH) / (_FAR_THRESH - _MEDIUM_THRESH))
    else:
        return 0.0


def _lerp(a, b, t):
    """Linear interpolation from a to b by factor t in [0,1]."""
    return a + (b - a) * t


# ============================================================================
# Coordinate normalization (same convention as gui_traj.py)
# ============================================================================

def _norm_coord(action, width, height):
    """Normalize absolute pixel coordinates to [0,1] range."""
    action = copy.deepcopy(action)
    if 'coordinate' in action and action['coordinate'] is not None:
        c = action['coordinate']
        if isinstance(c, (list, tuple)) and len(c) >= 2 and c[0] is not None and c[1] is not None:
            action['coordinate'] = [c[0] / width, c[1] / height]
        else:
            action['coordinate'] = None
    if 'coordinate2' in action and action['coordinate2'] is not None:
        c = action['coordinate2']
        if isinstance(c, (list, tuple)) and len(c) >= 2 and c[0] is not None and c[1] is not None:
            action['coordinate2'] = [c[0] / width, c[1] / height]
        else:
            action['coordinate2'] = None
    return action


# ============================================================================
# Text matching
# ============================================================================

def _check_text(pred_text, gt_text):
    """Substring match (case-insensitive), same as gui_traj.py."""
    pred_text = pred_text.lower().strip()
    gt_text = gt_text.lower().strip()
    return (pred_text in gt_text) or (gt_text in pred_text)


# ============================================================================
# Direction prediction (for swipe/wheel/drag)
# ============================================================================

def _predict_direction(start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if abs(dx) > abs(dy):
        return 'right' if dx > 0 else 'left'
    else:
        return 'down' if dy > 0 else 'up'


# ============================================================================
# Per-action scoring
# ============================================================================

def _score_action(pred, gt):
    """
    Score a single predicted action against ground truth.
    Returns (type_match: bool, action_score: float in [0,1]).
    """
    gt_action = gt.get('action', '')
    pred_action = pred.get('action', '')

    # ---- click / long_press ----
    if gt_action in ('click', 'long_press'):
        if pred_action != gt_action:
            return False, 0.0
        score = soft_coordinate_score(pred.get('coordinate'), gt.get('coordinate'))
        return True, score

    # ---- type / answer / key ----
    if gt_action in ('type', 'answer', 'key'):
        if pred_action != 'type':
            return False, 0.0
        pred_txt = pred.get('text')
        gt_txt = gt.get('text')
        if pred_txt is None or gt_txt is None:
            return True, 0.0
        text_score = 1.0 if _check_text(pred_txt, gt_txt) else 0.0
        # Optional soft coordinate bonus (weighted low)
        coord_score = soft_coordinate_score(pred.get('coordinate'), gt.get('coordinate'))
        return True, text_score * 0.8 + coord_score * 0.2

    # ---- empty action / FINISH / terminate / wait ----
    if gt_action in ('', 'terminate', 'wait'):
        if pred_action in ('', 'terminate', 'wait'):
            return True, 1.0
        return False, 0.0

    # ---- swipe ----
    if gt_action == 'swipe':
        if pred_action != 'swipe':
            return False, 0.0
        # Direction matching
        if 'direction' in gt and gt['direction'] is not None:
            gt_dir = gt['direction']
        elif gt.get('coordinate') is not None and gt.get('coordinate2') is not None:
            gt_dir = _predict_direction(gt['coordinate'], gt['coordinate2'])
        else:
            return True, 0.0
        if pred.get('coordinate') is None or pred.get('coordinate2') is None:
            return True, 0.0
        pred_dir = _predict_direction(pred['coordinate'], pred['coordinate2'])
        # Match swipe direction convention (invert up/down like gui_traj.py)
        if gt_dir == 'down':
            gt_dir = 'up'
        elif gt_dir == 'up':
            gt_dir = 'down'
        return True, 1.0 if pred_dir == gt_dir else 0.0

    # ---- drag ----
    if gt_action == 'drag':
        if pred_action != 'drag':
            return False, 0.0
        start_score = soft_coordinate_score(pred.get('coordinate'), gt.get('coordinate'))
        end_score = soft_coordinate_score(pred.get('coordinate2'), gt.get('coordinate2'))
        return True, (start_score + end_score) / 2.0

    # ---- select_text ----
    if gt_action == 'select_text':
        if pred_action != 'select_text':
            return False, 0.0
        start_score = soft_coordinate_score(pred.get('coordinate'), gt.get('coordinate'))
        end_score = soft_coordinate_score(pred.get('coordinate2'), gt.get('coordinate2'))
        return True, (start_score + end_score) / 2.0

    # ---- wheel_mouse_input (direction-only, like swipe) ----
    if gt_action == 'wheel_mouse_input':
        if pred_action != 'wheel_mouse_input':
            return False, 0.0
        if 'direction' in gt and gt['direction'] is not None:
            gt_dir = gt['direction']
        elif gt.get('coordinate') is not None and gt.get('coordinate2') is not None:
            gt_dir = _predict_direction(gt['coordinate'], gt['coordinate2'])
        else:
            return True, 0.0
        if 'direction' in pred and pred['direction'] is not None:
            pred_dir = pred['direction']
        elif pred.get('coordinate') is not None and pred.get('coordinate2') is not None:
            pred_dir = _predict_direction(pred['coordinate'], pred['coordinate2'])
        else:
            return True, 0.0
        return True, 1.0 if pred_dir == gt_dir else 0.0

    # ---- system_button ----
    if gt_action == 'system_button':
        if pred_action != 'system_button':
            return False, 0.0
        pred_btn = pred.get('button', '')
        gt_btn = gt.get('button', '')
        if not pred_btn or not gt_btn:
            return True, 0.0
        return True, 1.0 if pred_btn.lower().strip() == gt_btn.lower().strip() else 0.0

    # ---- open ----
    if gt_action == 'open':
        if pred_action != 'open':
            return False, 0.0
        pred_txt = pred.get('text')
        gt_txt = gt.get('text')
        if pred_txt is None or gt_txt is None:
            return True, 0.0
        return True, 1.0 if _check_text(pred_txt, gt_txt) else 0.0

    # ---- Rare action types: summary, select_table_range, set_font, etc. ----
    # Reward action type match (0.5) and full match if all fields equal (1.0)
    if pred_action == gt_action:
        # Check if all non-action fields match
        all_match = True
        for k, v in gt.items():
            if k == 'action':
                continue
            if k not in pred or pred[k] != v:
                all_match = False
                break
        return True, 1.0 if all_match else 0.5

    return False, 0.0


# ============================================================================
# Main entry point
# ============================================================================

def gui360_compute_score(solution_str, ground_truth, extra_info=None):
    """
    Compute reward for a single GUI-360 step.

    Args:
        solution_str: Model's raw response string.
        ground_truth: Dict with 'check_options' (action dict) and 'num_steps'.
        extra_info: Dict with width, height, resized_width, resized_height.

    Returns:
        Dict with score, format_score, type_match, extract_match, step_reward.
    """
    check_options = ground_truth['check_options']
    num_steps = ground_truth['num_steps']

    if extra_info is None:
        extra_info = {}
    width = extra_info.get('width', 1024)
    height = extra_info.get('height', 720)
    resized_width = extra_info.get('resized_width', 1024)
    resized_height = extra_info.get('resized_height', 720)

    format_score = 0.0
    action_score = 0.0
    type_match = False
    extract_match = False

    try:
        result = fm.parse_response(solution_str)
        think_str = result.get('think')
        if think_str is not None and think_str.strip():
            format_score = 1.0

        if result.get('action_content') is not None:
            pred_action = _norm_coord(result['action_content'], resized_width, resized_height)
            gt_action = _norm_coord(check_options, width, height)

            type_match, action_score = _score_action(pred_action, gt_action)

            # Handle HARMFUL / NEUTRAL annotations (same as gui_traj.py)
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

    # extract_match: consider "matched" if action_score >= 0.8 (near-exact)
    extract_match = type_match and action_score >= 0.8
    step_reward = 1.0 / num_steps if extract_match else 0.0

    return {
        "score": format_score * 0.1 + action_score * 0.9,
        "format_score": format_score,
        "type_match": type_match,
        "extract_match": extract_match,
        "step_reward": step_reward,
    }
