"""Prediction parsing and cached step querying."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .prompt import build_messages, offtrack_probe


@dataclass(frozen=True)
class Pred:
    text: str
    function: str
    xy: Optional[Tuple[float, float]]
    control_label: Optional[str]
    top1_conf: float
    action_entropy: float
    recovery_class: str
    self_offtrack: float
    samples: Tuple[str, ...]


_CACHE: Dict[Tuple[Any, ...], Pred] = {}


def _parse_json_from_text(text: str) -> Dict[str, Any]:
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def _parse_action(text: str) -> Tuple[str, Optional[Tuple[float, float]], Optional[str]]:
    data = _parse_json_from_text(text)
    function = str(data.get("function") or data.get("action") or "").strip().lower()
    args = data.get("args") if isinstance(data.get("args"), dict) else data
    coord = args.get("coordinate") or args.get("xy")
    if coord is None and "x" in args and "y" in args:
        coord = [args.get("x"), args.get("y")]
    xy = None
    if coord is not None:
        try:
            xy = (float(coord[0]), float(coord[1]))
        except (TypeError, ValueError, IndexError):
            xy = None
    label = args.get("element_id") or args.get("control_label") or args.get("label")
    return function, xy, None if label is None else str(label)


def _signature(text: str) -> str:
    function, xy, label = _parse_action(text)
    if xy is not None:
        return f"{function}:{round(xy[0] / 20)}:{round(xy[1] / 20)}"
    return f"{function}:label:{label}"


def _entropy(signatures: List[str]) -> float:
    total = len(signatures)
    if total == 0:
        return 0.0
    counts = Counter(signatures)
    return float(-sum((count / total) * math.log((count / total) + 1e-12) for count in counts.values()))


def _recovery_class(function: str, text: str) -> str:
    normalized = (function + " " + text).lower()
    if any(token in normalized for token in ("undo", "ctrl+z", "esc", "cancel", "close", "back", "don't save", "dont save", "no")):
        return "recover"
    if function:
        return "progress"
    return "other"


def _offtrack_probability(model: Any, step: Any, input_mode: str) -> float:
    try:
        decodes = model.generate(offtrack_probe(step, input_mode), n=1, logprobs=False)
        text = decodes[0].text
        data = _parse_json_from_text(text)
        if "confidence" in data:
            conf = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
            return conf if data.get("offtrack") else 1.0 - conf
        return 1.0 if "offtrack" in text.lower() else 0.0
    except Exception:
        return 0.0


def _cache_key(model: Any, step: Any, history_mode: str, input_mode: str, plan: Optional[str]) -> Tuple[Any, ...]:
    plan_hash = hashlib.sha1((plan or "").encode("utf-8")).hexdigest()[:12]
    model_id = getattr(model, "cache_id", getattr(model, "model_name", model.__class__.__name__))
    return (model_id, step.exec_id, step.step_id, history_mode, input_mode, plan_hash)


def query_step(model: Any, step: Any, history_mode: str, input_mode: str, plan: Optional[str] = None, n: int = 8) -> Pred:
    key = _cache_key(model, step, history_mode, input_mode, plan)
    if key in _CACHE:
        return _CACHE[key]
    messages = build_messages(step, history_mode, input_mode, plan)
    decodes = model.generate(messages, n=n, logprobs=False)
    texts = [decode.text for decode in decodes]
    signatures = [_signature(text) for text in texts]
    top_sig, top_count = Counter(signatures).most_common(1)[0]
    top_idx = signatures.index(top_sig)
    top_text = texts[top_idx]
    function, xy, label = _parse_action(top_text)
    pred = Pred(
        text=top_text,
        function=function,
        xy=xy,
        control_label=label,
        top1_conf=float(top_count / max(len(signatures), 1)),
        action_entropy=_entropy(signatures),
        recovery_class=_recovery_class(function, top_text),
        self_offtrack=_offtrack_probability(model, step, input_mode),
        samples=tuple(texts),
    )
    _CACHE[key] = pred
    return pred
