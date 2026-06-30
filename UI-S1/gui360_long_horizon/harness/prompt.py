"""GUI-360 prompt builders with experiment knobs."""

from __future__ import annotations

import base64
import io
import os
from typing import Any, Dict, List, Optional

try:
        from v13_gui_360.eval_gui360_template import SUPPORTED_ACTIONS
except ImportError:  # pragma: no cover - only used in lightweight/offline envs
        SUPPORTED_ACTIONS = """<action>
- click
    - Args:
        - coordinate: [x, y], the absolute position on the screen you want to click at.
        - button: str, One of 'left', 'right', 'middle' or 'x' (Default: 'left')
        - double: bool, Whether to perform a double click (Default: False)
        - pressed: str|None, Keyboard key to press while clicking (Default: None)
- type
    - Args:
        - coordinate: [x, y], the absolute position on the screen you want to type at.
        - keys: str, The key to input.
        - clear_current_text: bool, Whether to clear the current text (Default: False)
        - control_focus: bool, Whether to focus on selected control before typing (Default: True)
- drag
    - Args:
        - start_coordinate: [x, y], where the drag starts.
        - end_coordinate: [x, y], where the drag ends.
        - button: str, 'left' or 'right' (Default: 'left')
        - duration: float, Duration in seconds (Default: 1.0)
- wheel_mouse_input
    - Args:
        - coordinate: [x, y], position on the screen to scroll.
        - wheel_dist: int, Wheel notches. Positive=up, negative=down.
</action>"""

DEFAULT_REPO = os.environ.get("GUI360_REPO", "vyokky/GUI-360")


class VisualOnlyPromptError(RuntimeError):
    pass

BASE_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, decide the next action to take.

The instruction is:
{instruction}

The current subtask is:
{subtask}

The history of actions and observations is:
{history}
{plan_text}{a11y_text}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen.

Output exactly one action within <tool_call></tool_call>.
"""

OFFTRACK_PROMPT = """Given the current GUI screenshot and task, answer whether the task is on track or whether something went wrong that should be undone first.

Instruction: {instruction}
Subtask: {subtask}

Reply with JSON: {"offtrack": true|false, "confidence": 0.0-1.0}.
"""


def _image_data_url(step: Any) -> str:
    if isinstance(getattr(step, "raw", None), dict) and step.raw.get("_image_data_url"):
        return step.raw["_image_data_url"]
    from gui360_long_horizon.data.loader import load_image

    image = load_image(DEFAULT_REPO, step.image_rel_path)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _history(step: Any, history_mode: str) -> str:
    if history_mode == "none":
        return "None"
    if history_mode not in {"full", "summary", "corrupt", "native", "native_last3", "action_only", "action_last3"}:
        raise ValueError(f"unknown history_mode: {history_mode}")
    raw = getattr(step, "raw", {}) or {}
    if history_mode == "summary":
        return str(raw.get("history_summary") or "Summary unavailable")
    if history_mode == "corrupt":
        return str(raw.get("corrupt_history") or "Step 1: wrong prior action injected")
    if history_mode == "native":
        return str(raw.get("history_native") or "None")
    if history_mode == "native_last3":
        return str(raw.get("history_native_last3") or "None")
    if history_mode == "action_only":
        return str(raw.get("history_action_only") or "None")
    if history_mode == "action_last3":
        return str(raw.get("history_action_last3") or "None")
    return str(raw.get("history_text") or "None")


def _a11y(step: Any, input_mode: str) -> str:
    if input_mode == "visual":
        return ""
    if input_mode == "visual_a11y":
        raise VisualOnlyPromptError("a11y text is referee-only and must not be placed in model prompts")
    raise ValueError(f"unknown input_mode: {input_mode}")


def build_messages(step: Any, history_mode: str, input_mode: str, plan: Optional[str] = None) -> List[Dict[str, Any]]:
    plan_text = f"\nOracle/global plan:\n{plan}\n" if plan else ""
    text = BASE_PROMPT.format(
        instruction=step.request,
        subtask=step.subtask or "(none)",
        history=_history(step, history_mode),
        plan_text=plan_text,
        a11y_text=_a11y(step, input_mode),
        actions=SUPPORTED_ACTIONS,
    )
    return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": _image_data_url(step)}}, {"type": "text", "text": text}]}]


def offtrack_probe(step: Any, input_mode: str) -> List[Dict[str, Any]]:
    text = OFFTRACK_PROMPT.format(instruction=step.request, subtask=step.subtask or "(none)") + _a11y(step, input_mode)
    return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": _image_data_url(step)}}, {"type": "text", "text": text}]}]
