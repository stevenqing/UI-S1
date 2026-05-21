"""v9 credit assignment for cooperative LoRA.

Given (pred_action, gt_action), compute (coord_correct, action_correct, has_coord)
INDEPENDENTLY (not via the standard matcher's coupled type/extract output), then
map to (r_g, r_a) via the v9 credit table.

Credit table (has_coord = action has a meaningful coordinate):
    has_coord = True (click / long_press / swipe):
        coord_correct  action_correct  | r_g    r_a
        T              T               | +1.0   +1.0
        T              F               | +1.0   -1.0
        F              T               | -1.0   +1.0
        F              F               | -1.0   -0.5

    has_coord = False (wait / system_button / type / open / answer / key / terminate):
        action_correct                | r_g    r_a
        T                             |  0.0   +1.0
        F                             |  0.0   -1.0

Rationale:
- Coord/action correctness are measured independently so the matrix is observable.
- Non-coord actions: grounder can't be wrong about a coord that doesn't exist,
  so we don't update grounder (r_g=0). Cleaner than letting it absorb noise.
- (F, F) row: action gets -0.5 because grounder fed it a wrong coord, action
  being wrong is partly inherited.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np


# Match the AC matcher constants.
POINT_DISTANCE_THRESHOLD = 0.04   # normalized [0,1]
BBOX_ENLARGE_FACTOR = 1.2

# Action types where coordinate is meaningful for grounder credit.
HAS_COORD_ACTIONS = frozenset({"click", "long_press", "swipe"})

# Action types where there is no meaningful coordinate for grounder.
NO_COORD_ACTIONS = frozenset({
    "wait", "system_button", "type", "open", "answer", "key", "terminate",
})


def _enlarge_bbox(bbox_list, scale_factor=BBOX_ENLARGE_FACTOR):
    """Enlarge bboxes around their center by scale_factor."""
    bbox_array = np.asarray(bbox_list, dtype=np.float32)
    if bbox_array.ndim == 1:
        bbox_array = bbox_array[None, :]
    x_min, y_min, x_max, y_max = bbox_array.T
    x_c = (x_min + x_max) / 2
    y_c = (y_min + y_max) / 2
    w = (x_max - x_min) * scale_factor
    h = (y_max - y_min) * scale_factor
    out = np.vstack([x_c - w/2, y_c - h/2, x_c + w/2, y_c + h/2]).T
    return out.tolist()


def _check_click_normalized(pred_xy, candidate_bbox, gt_point):
    """Bbox-hit OR pixel-distance under normalized [0,1] coordinates."""
    if candidate_bbox:
        for bbox in _enlarge_bbox(candidate_bbox):
            if bbox[0] <= pred_xy[0] <= bbox[2] and bbox[1] <= pred_xy[1] <= bbox[3]:
                return True
    if gt_point is not None:
        d = ((pred_xy[0] - gt_point[0])**2 + (pred_xy[1] - gt_point[1])**2) ** 0.5
        return d <= POINT_DISTANCE_THRESHOLD
    return False


def _text_match(pred: str, gt: str) -> bool:
    """Lenient text match (substring either direction), matching AC matcher."""
    if pred is None or gt is None:
        return False
    p = str(pred).lower().strip()
    g = str(gt).lower().strip()
    return (p in g) or (g in p)


def parse_action_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract action JSON from model output.

    Looks for `<action>{...}</action>` first, then any JSON-looking blob.
    Returns None if parsing fails.
    """
    if not isinstance(text, str):
        return None
    m = re.search(r"<action>\s*(\{.*?\})\s*</action>", text, flags=re.DOTALL)
    if m:
        blob = m.group(1)
    else:
        m = re.search(r"\{[^{}]*\"action\"\s*:[^{}]*\}", text, flags=re.DOTALL)
        if not m:
            return None
        blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        return None


def analyze_correctness(
    pred_action: Optional[Dict[str, Any]],
    gt_action: Dict[str, Any],
    image_w: int,
    image_h: int,
    pred_image_w: Optional[int] = None,
    pred_image_h: Optional[int] = None,
) -> Tuple[Optional[bool], bool, bool]:
    """Independent measurement of (coord_correct, action_correct, has_coord).

    Args:
        pred_action: parsed action dict, or None on parse failure.
        gt_action: GT action dict from the dataset.
        image_w, image_h: pixel size used for GT coordinates.
        pred_image_w, pred_image_h: pixel size used for predicted coords (if
            different from GT, e.g. resized image). Defaults to (image_w, image_h).

    Returns:
        coord_correct: True / False / None (None when has_coord=False).
        action_correct: True / False.
        has_coord: True if GT action has a meaningful coordinate.
    """
    if pred_image_w is None:
        pred_image_w = image_w
    if pred_image_h is None:
        pred_image_h = image_h

    gt_type = gt_action.get("action")
    has_coord = gt_type in HAS_COORD_ACTIONS

    # ── action_correct: type matches AND non-coord params match ──
    action_correct = False
    if pred_action is not None and pred_action.get("action") == gt_type:
        if gt_type in ("click", "long_press", "swipe", "wait", "terminate"):
            # No non-coord params to check (or status for terminate is loose).
            action_correct = True
        elif gt_type == "system_button":
            action_correct = _text_match(
                pred_action.get("button"), gt_action.get("button"))
        elif gt_type in ("type", "open", "answer", "key"):
            action_correct = _text_match(
                pred_action.get("text"), gt_action.get("text"))
        else:
            action_correct = True  # unknown type but matches: be lenient

    # ── coord_correct: measured INDEPENDENTLY of action type ──
    coord_correct: Optional[bool] = None
    if has_coord:
        coord_correct = False
        if pred_action is not None:
            pred_coord = pred_action.get("coordinate")
            if pred_coord and len(pred_coord) >= 2:
                # Normalize predicted coord to [0,1] using the image size the
                # model saw.
                pcx = float(pred_coord[0]) / max(pred_image_w, 1)
                pcy = float(pred_coord[1]) / max(pred_image_h, 1)
                # Normalize GT coord using GT image size.
                gt_xy = gt_action.get("coordinate")
                gt_norm = None
                if gt_xy and len(gt_xy) >= 2:
                    gt_norm = [
                        float(gt_xy[0]) / max(image_w, 1),
                        float(gt_xy[1]) / max(image_h, 1),
                    ]
                bbox = gt_action.get("candidate_bbox") or []
                bbox_norm = [
                    [b[0]/image_w, b[1]/image_h, b[2]/image_w, b[3]/image_h]
                    for b in bbox
                ] if bbox else []
                coord_correct = _check_click_normalized(
                    [pcx, pcy], bbox_norm, gt_norm)

    return coord_correct, action_correct, has_coord


def compute_credit(
    coord_correct: Optional[bool],
    action_correct: bool,
    has_coord: bool,
) -> Tuple[float, float]:
    """Map correctness flags to (r_g, r_a) — the GRPO-style signed credit.

    See module docstring for the table. Used in step-2 (real GRPO).
    """
    if not has_coord:
        # Non-coord action: grounder neutral, action gets ±1.
        return 0.0, (1.0 if action_correct else -1.0)

    # Coord action: full 4-row table.
    if coord_correct and action_correct:
        return 1.0, 1.0
    if coord_correct and not action_correct:
        return 1.0, -1.0
    if not coord_correct and action_correct:
        return -1.0, 1.0
    return -1.0, -0.5  # both wrong: action partially excused


# ── Step-1 mistake-driven SFT weights ──────────────────────────────────
# Used during Reweighted-SFT (no rollout): we teacher-force on GT and
# UP-weight the loss on the component that was wrong, DOWN-weight the
# component that was already right.
#
#   has_coord = True (click / long_press / swipe):
#     coord_correct  action_correct | w_v   w_a
#     T              T              | 0.3   0.3
#     T              F              | 0.3   1.0
#     F              T              | 1.0   0.3
#     F              F              | 1.0   0.75   (action partially excused)
#
#   has_coord = False (wait / system_button / type / open / answer / key / terminate):
#     action_correct | w_v   w_a
#     T              | 0.0   0.3
#     F              | 0.0   1.0

_W_HIGH = 1.0   # weight when component is wrong → "fix it"
_W_LOW = 0.3    # weight when component is right → "light touch"
_W_FF_A = 0.75  # action excused when both wrong (grounder fed wrong coord)


def compute_sft_weights(
    coord_correct: Optional[bool],
    action_correct: bool,
    has_coord: bool,
) -> Tuple[float, float]:
    """Mistake-driven SFT loss weights (w_v, w_a).

    See module docstring for the table.
    """
    if not has_coord:
        # Non-coord action: grounder doesn't apply, action either fix or light.
        return 0.0, (_W_LOW if action_correct else _W_HIGH)

    if coord_correct and action_correct:
        return _W_LOW, _W_LOW
    if coord_correct and not action_correct:
        return _W_LOW, _W_HIGH
    if not coord_correct and action_correct:
        return _W_HIGH, _W_LOW
    # both wrong
    return _W_HIGH, _W_FF_A


def credit_for_sample(
    pred_action: Optional[Dict[str, Any]],
    gt_action: Dict[str, Any],
    image_w: int,
    image_h: int,
    pred_image_w: Optional[int] = None,
    pred_image_h: Optional[int] = None,
) -> Dict[str, Any]:
    """One-stop helper: returns the full credit record for a sample.

    Includes both the GRPO-style signed (r_g, r_a) (for step-2) and the
    mistake-driven SFT weights (w_v, w_a) (for step-1 reweighted SFT).
    """
    coord_correct, action_correct, has_coord = analyze_correctness(
        pred_action, gt_action, image_w, image_h, pred_image_w, pred_image_h,
    )
    r_g, r_a = compute_credit(coord_correct, action_correct, has_coord)
    w_v, w_a = compute_sft_weights(coord_correct, action_correct, has_coord)
    return {
        "has_coord": has_coord,
        "coord_correct": coord_correct,
        "action_correct": action_correct,
        "r_g": r_g,
        "r_a": r_a,
        "w_v": w_v,
        "w_a": w_a,
    }
