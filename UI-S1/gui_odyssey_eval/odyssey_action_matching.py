"""Action matching for GUI-Odyssey evaluation.

Key differences from AC's evaluate_android_control_action:
- GT coordinates are in [0,1000] normalized space (not pixel space).
- Predicted coordinates are in resized pixel space (model output) — must convert to [0,1000].
- Click matching uses sam2_bbox (in [0,1000]) OR Euclidean distance <= 140 (in [0,1000] space).
- Scroll matching compares derived directions.
- Text matching uses ANLS >= 0.5 (Levenshtein).
"""

import copy
import numpy as np

# Reuse text matching from existing GUI-Odyssey implementation
from GUIOdyssey_action_matching import (
    levenshtein_distance,
    text_matching,
    CLICK_COORD_THRESHOLD,
    TEXT_ANLS_THRESHOLD,
)

# Distance threshold in [0,1000] space: 0.14 * 1000 = 140
CLICK_DISTANCE_THRESHOLD_1K = CLICK_COORD_THRESHOLD * 1000  # 140
BBOX_ENLARGE_FACTOR = 1.2


def get_scroll_direction(coord1, coord2):
    """Derive scroll direction from two [0,1000] coordinates.

    Returns: 'up', 'down', 'left', or 'right'
    """
    dx = coord2[0] - coord1[0]
    dy = coord2[1] - coord1[1]
    if abs(dx) > abs(dy):
        return 'right' if dx > 0 else 'left'
    else:
        return 'down' if dy > 0 else 'up'


def pred_coord_to_1k(coord, resized_width, resized_height):
    """Convert predicted coordinate from resized pixel space to [0,1000] space.

    Args:
        coord: [x, y] in resized pixel space.
        resized_width: Width of the resized image.
        resized_height: Height of the resized image.

    Returns:
        [x, y] in [0,1000] space.
    """
    return [
        coord[0] / resized_width * 1000,
        coord[1] / resized_height * 1000,
    ]


def enlarge_bbox_1k(bbox, scale_factor=BBOX_ENLARGE_FACTOR):
    """Enlarge a single [x1,y1,x2,y2] bbox in [0,1000] space.

    Returns: [x1,y1,x2,y2] enlarged bbox.
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1) * scale_factor
    h = (y2 - y1) * scale_factor
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def check_click_1k(pred_coord_1k, candidate_bbox, gt_coord_1k):
    """Check if predicted click matches ground truth in [0,1000] space.

    Args:
        pred_coord_1k: [x,y] predicted coordinate in [0,1000].
        candidate_bbox: List of [x1,y1,x2,y2] bboxes in [0,1000].
        gt_coord_1k: [x,y] ground truth coordinate in [0,1000], or None.

    Returns:
        True if click is correct.
    """
    # Check bbox containment first
    if candidate_bbox:
        for bbox in candidate_bbox:
            enlarged = enlarge_bbox_1k(bbox)
            if (enlarged[0] <= pred_coord_1k[0] <= enlarged[2] and
                    enlarged[1] <= pred_coord_1k[1] <= enlarged[3]):
                return True

    # Fallback to Euclidean distance
    if gt_coord_1k is not None:
        dist = np.linalg.norm([
            pred_coord_1k[0] - gt_coord_1k[0],
            pred_coord_1k[1] - gt_coord_1k[1],
        ])
        return dist <= CLICK_DISTANCE_THRESHOLD_1K

    return False


def evaluate_odyssey_action(pred_action, gt_check, resized_width, resized_height):
    """Evaluate a predicted action against GUI-Odyssey ground truth.

    Args:
        pred_action: Dict from model output, e.g. {"action": "click", "coordinate": [x,y]}.
                     Coordinates are in resized pixel space.
        gt_check: Dict with GT action + check_options, e.g.
                  {"action": "click", "coordinate": [x,y], "candidate_bbox": [[x1,y1,x2,y2]]}.
                  Coordinates are in [0,1000] space.
        resized_width: Width of the resized image (for normalizing pred coords).
        resized_height: Height of the resized image.

    Returns:
        (type_match, extract_match): Tuple of booleans.
    """
    pred_action = copy.deepcopy(pred_action)
    gt_check = copy.deepcopy(gt_check)

    pred_type = pred_action.get('action', '')
    gt_type = gt_check.get('action', '')

    # Terminate actions
    if gt_type == 'terminate':
        if pred_type == 'terminate':
            return True, pred_action.get('status', '') == gt_check.get('status', '')
        return False, False

    # System button
    if gt_type == 'system_button':
        if pred_type == 'system_button':
            return True, pred_action.get('button', '').lower().strip() == gt_check.get('button', '').lower().strip()
        return False, False

    # Wait
    if gt_type == 'wait':
        if pred_type == 'wait':
            return True, True
        return False, False

    # Type
    if gt_type == 'type':
        if pred_type == 'type':
            return True, text_matching(gt_check.get('text', ''), pred_action.get('text', ''))
        return False, False

    # Swipe (scroll)
    if gt_type == 'swipe':
        if pred_type == 'swipe':
            # Convert pred coords to [0,1000] and compare directions
            pred_c1 = pred_coord_to_1k(pred_action.get('coordinate', [0, 0]), resized_width, resized_height)
            pred_c2 = pred_coord_to_1k(pred_action.get('coordinate2', [0, 0]), resized_width, resized_height)
            gt_c1 = gt_check.get('coordinate', [0, 0])
            gt_c2 = gt_check.get('coordinate2', [0, 0])
            pred_dir = get_scroll_direction(pred_c1, pred_c2)
            gt_dir = get_scroll_direction(gt_c1, gt_c2)
            return True, pred_dir == gt_dir
        return False, False

    # Click / Long press
    if gt_type in ('click', 'long_press'):
        if pred_type == gt_type:
            pred_coord_1k = pred_coord_to_1k(
                pred_action.get('coordinate', [0, 0]),
                resized_width, resized_height,
            )
            gt_coord_1k = gt_check.get('coordinate', None)
            candidate_bbox = gt_check.get('candidate_bbox', [])
            return True, check_click_1k(pred_coord_1k, candidate_bbox, gt_coord_1k)
        return False, False

    # Open app
    if gt_type == 'open':
        if pred_type == 'open':
            return True, text_matching(gt_check.get('text', ''), pred_action.get('text', ''))
        return False, False

    # Fallback: unknown action type
    if pred_type == gt_type:
        return True, True
    return False, False
