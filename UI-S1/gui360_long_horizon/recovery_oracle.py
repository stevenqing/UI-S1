"""High-precision generic recovery oracle for off-trajectory GUI-360 screens."""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple


DISMISS_TEXT = {"ok", "cancel", "no", "close", "don't save", "dont save", "dismiss"}
ERROR_RE = re.compile(r"\b(error|cannot|can't|invalid|failed|failure|warning|unable)\b", re.I)
MODAL_TYPES = {"dialog", "messagebox", "alert"}
BUTTON_TYPES = {"button", "splitbutton"}


@dataclass(frozen=True)
class RecoveryAction:
    function: str
    key: Optional[str] = None
    xy: Optional[Tuple[float, float]] = None
    label: Optional[str] = None


@dataclass(frozen=True)
class RecoveryTarget:
    kind: str
    correct_action: RecoveryAction
    reason: str


def _controls(step: Any) -> List[dict]:
    control_infos = getattr(step, "control_infos", None) or {}
    controls = control_infos.get("uia_controls_info") or []
    return [ctrl for ctrl in controls if isinstance(ctrl, dict)] if isinstance(controls, list) else []


def _ctrl_text(ctrl: dict) -> str:
    values = []
    for key in ("control_text", "label", "name", "title", "automation_id"):
        value = ctrl.get(key)
        if value not in (None, ""):
            values.append(str(value))
    return " ".join(values).strip()


def _ctrl_type(ctrl: dict) -> str:
    return str(ctrl.get("control_type") or ctrl.get("type") or "").strip().lower()


def _rect_center(ctrl: dict) -> Optional[Tuple[float, float]]:
    rect = ctrl.get("control_rect") or ctrl.get("rectangle") or ctrl.get("bbox")
    if isinstance(rect, dict):
        try:
            left, top, right, bottom = float(rect["left"]), float(rect["top"]), float(rect["right"]), float(rect["bottom"])
            return (left + right) / 2.0, (top + bottom) / 2.0
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(rect, (list, tuple)) and len(rect) >= 4:
        try:
            left, top, right, bottom = map(float, rect[:4])
            return (left + right) / 2.0, (top + bottom) / 2.0
        except (TypeError, ValueError):
            return None
    return None


def _dismiss_buttons(controls: Sequence[dict]) -> List[dict]:
    out = []
    for ctrl in controls:
        text = _ctrl_text(ctrl).lower().strip()
        if _ctrl_type(ctrl) in BUTTON_TYPES and text in DISMISS_TEXT:
            out.append(ctrl)
    priority = {"don't save": 0, "dont save": 0, "cancel": 1, "no": 2, "close": 3, "dismiss": 4, "ok": 5}
    return sorted(out, key=lambda ctrl: priority.get(_ctrl_text(ctrl).lower().strip(), 99))


def _has_modal_signal(controls: Sequence[dict]) -> bool:
    for ctrl in controls:
        ctrl_type = _ctrl_type(ctrl)
        text = _ctrl_text(ctrl)
        if ctrl_type in MODAL_TYPES:
            return True
        if ctrl_type == "window" and ERROR_RE.search(text):
            return True
    return False


def _has_error_signal(controls: Sequence[dict]) -> bool:
    return any(ERROR_RE.search(_ctrl_text(ctrl)) for ctrl in controls)


def _content_independent_action(button: Optional[dict], fallback_key: str = "esc") -> RecoveryAction:
    if button is None:
        return RecoveryAction(function="press_key", key=fallback_key)
    xy = _rect_center(button)
    label = _ctrl_text(button) or None
    if xy is None:
        return RecoveryAction(function="press_key", key=fallback_key, label=label)
    return RecoveryAction(function="click", xy=xy, label=label)


def recovery_oracle(step: Any) -> Optional[RecoveryTarget]:
    """Return a content-independent generic recovery target, or None.

    The oracle deliberately favors precision over recall. It avoids normal top
    level windows with common buttons unless there is a modal/error signal.
    """

    controls = _controls(step)
    if not controls:
        return None
    buttons = _dismiss_buttons(controls)
    has_error = _has_error_signal(controls)
    has_modal = _has_modal_signal(controls)
    if has_error and buttons:
        return RecoveryTarget(kind="error_popup", correct_action=_content_independent_action(buttons[0]), reason="error dialog with dismiss button")
    if has_modal and buttons:
        return RecoveryTarget(kind="spurious_modal", correct_action=_content_independent_action(buttons[0]), reason="modal dialog with dismiss button")
    raw = getattr(step, "raw", {}) or {}
    if raw.get("_recovery_wrong_menu") is True:
        return RecoveryTarget(kind="wrong_menu", correct_action=RecoveryAction(function="press_key", key="esc"), reason="explicit wrong-menu audit flag")
    return None


def audit_precision(labeled_steps: Optional[Iterable[Tuple[Any, bool]]] = None, *, sample_n: int = 100, precision_min: float = 0.90, seed: int = 41) -> float:
    """Compute precision on hand-labeled `(step, should_recover)` samples.

    If no labeled samples are supplied, returns NaN; production callers should
    pass the audit set and enforce the pre-registered threshold.
    """

    if labeled_steps is None:
        return math.nan
    samples = list(labeled_steps)
    rng = random.Random(seed)
    rng.shuffle(samples)
    true_positive = 0
    predicted_positive = 0
    for step, should_recover in samples[:sample_n]:
        pred = recovery_oracle(step) is not None
        if pred:
            predicted_positive += 1
            true_positive += int(bool(should_recover))
    precision = true_positive / predicted_positive if predicted_positive else 1.0
    if precision < precision_min:
        raise ValueError(f"recovery oracle precision below threshold: {precision:.3f} < {precision_min:.3f}")
    return float(precision)
