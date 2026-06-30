"""Model harness for GUI-360 long-horizon experiments."""

from __future__ import annotations

from importlib import import_module


_LAZY_EXPORTS = {
    "Decode": ".model",
    "Pred": ".predict",
    "VLLMClient": ".model",
    "build_messages": ".prompt",
    "coordinate_hit": ".correctness",
    "function_match": ".correctness",
    "offtrack_probe": ".prompt",
    "query_step": ".predict",
    "step_correct": ".correctness",
}


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value

__all__ = [
    "Decode",
    "Pred",
    "VLLMClient",
    "build_messages",
    "coordinate_hit",
    "function_match",
    "offtrack_probe",
    "query_step",
    "step_correct",
]
