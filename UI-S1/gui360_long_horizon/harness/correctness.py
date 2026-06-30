"""Frozen GUI-360 step correctness helpers."""

from __future__ import annotations

import json
import re
from typing import Any, Optional, Tuple

ACTION_ALIASES = {
    "double_click": "click",
    "left_click": "click",
    "tap": "click",
    "input": "type",
    "drag": "swipe",
    "scroll": "swipe",
    "wheel_mouse_input": "swipe",
}


def _norm_function(value: Any) -> str:
    text = str(value or "").strip().lower()
    return ACTION_ALIASES.get(text, text)


def _pred_function(pred: Any) -> str:
    if isinstance(pred, dict):
        return _norm_function(pred.get("function") or pred.get("action"))
    return _norm_function(getattr(pred, "function", None) or getattr(pred, "action", None))


def _pred_xy(pred: Any) -> Optional[Tuple[float, float]]:
    if isinstance(pred, dict):
        xy = pred.get("xy") or pred.get("coordinate")
        args = pred.get("args") if isinstance(pred.get("args"), dict) else {}
        if xy is None and "coordinate" in args:
            xy = args.get("coordinate")
        if xy is None and "x" in args and "y" in args:
            xy = (args.get("x"), args.get("y"))
    else:
        xy = getattr(pred, "xy", None) or getattr(pred, "coordinate", None)
    if xy is None:
        return None
    try:
        return float(xy[0]), float(xy[1])
    except (TypeError, ValueError, IndexError):
        return None


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _text_from_action(action: Any) -> str:
    if not isinstance(action, dict):
        return ""
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    raw = action.get("raw_json") if isinstance(action.get("raw_json"), dict) else {}
    raw_args = raw.get("args") if isinstance(raw.get("args"), dict) else {}
    for source in (args, raw_args, action, raw):
        for key in ("text", "keys", "value", "query", "control_text"):
            if source.get(key) is not None:
                return str(source.get(key))
    return ""


def _json_action_from_text(text: str) -> dict:
    value = str(text or "")
    for pattern in (r"<tool_call>\s*(\{.*?\})\s*</tool_call>", r"```(?:json)?\s*(\{.*?\})\s*```"):
        match = re.search(pattern, value, flags=re.DOTALL)
        if not match:
            continue
        try:
            parsed = json.loads(match.group(1))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", value):
        try:
            parsed, _ = decoder.raw_decode(value[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _pred_text(pred: Any) -> str:
    if isinstance(pred, dict):
        return _text_from_action(pred)
    parsed = _json_action_from_text(str(getattr(pred, "text", "") or ""))
    return _text_from_action(parsed)


def _gt_text(step: Any) -> str:
    return _text_from_action(getattr(step, "gt_action", None) or {})


def function_match(pred: Any, gt_function: str) -> bool:
    """Return whether predicted and GT functions match under GUI aliases."""

    return bool(_pred_function(pred) and _pred_function(pred) == _norm_function(gt_function))


def coordinate_hit(pred_xy: Optional[Tuple[float, float]], gt_rect: Tuple[float, float, float, float], tol: float = 0.0) -> bool:
    """Return whether predicted xy lies inside the GT rectangle plus tolerance."""

    if pred_xy is None:
        return False
    x_val, y_val = pred_xy
    left, top, right, bottom = gt_rect
    return left - tol <= x_val <= right + tol and top - tol <= y_val <= bottom + tol


def step_correct(pred: Any, step: Any) -> bool:
    """Frozen step correctness: normalized function match and coordinate hit.

    Raises when called on off-trajectory/fail steps with no GT action rectangle.
    """

    if getattr(step, "gt_rect", None) is None or getattr(step, "gt_function", None) is None:
        raise ValueError("step_correct is undefined for GT-absent/off-trajectory steps")
    if not function_match(pred, step.gt_function):
        return False
    gt_function = _norm_function(step.gt_function)
    gt_text = _gt_text(step)
    if gt_function in {"type", "paste"} and gt_text and _norm_text(_pred_text(pred)) != _norm_text(gt_text):
        return False
    return coordinate_hit(_pred_xy(pred), step.gt_rect)
