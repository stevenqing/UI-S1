"""
GUI-360 Binary Reward (Exp 2: Vanilla GRPO)

Same parsing as gui360/reward.py but returns BINARY reward:
  score = 1.0 if action type matches AND action_score >= 0.8
  score = 0.0 otherwise

This isolates the effect of reward sparsity: Exp 2 (binary) vs Exp 3 (dense/default gui360).
"""

import copy
import traceback

from verl.utils.reward_score.gui360.reward import (
    fm, _norm_coord, _score_action
)


def gui360_binary_compute_score(solution_str, ground_truth, extra_info=None):
    """Binary reward: 1.0 if near-exact match, 0.0 otherwise."""
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

            annotation = check_options.get('annotation', 'GOOD')
            if annotation == 'HARMFUL':
                action_score = 1.0 - action_score
            elif annotation == 'NEUTRAL':
                action_score = 0.5
        else:
            action_score = 0.0
    except Exception:
        traceback.print_exc()

    extract_match = type_match and action_score >= 0.8

    # Binary reward: 1.0 if match, 0.0 otherwise
    # Small format bonus (0.1) to encourage well-formatted output
    binary_score = 1.0 if extract_match else 0.0
    score = format_score * 0.1 + binary_score * 0.9

    return {
        "score": score,
        "format_score": format_score,
        "type_match": type_match,
        "action_score": action_score,  # keep continuous for logging
        "extract_match": extract_match,
        "step_reward": 1.0 / num_steps if extract_match else 0.0,
    }
