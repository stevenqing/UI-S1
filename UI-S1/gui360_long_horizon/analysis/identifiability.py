"""Runtime labels and guards for identified vs observational components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


COMPONENTS: Dict[str, Dict[str, Any]] = {
    "m_textmem": {"identified": True, "method": "paired success-step history ablation gate"},
    "m_core": {"identified": True, "method": "nested invariant feature decomposition; position block is descriptive"},
    "m_plan": {"identified": True, "method": "within-step oracle-plan intervention"},
    "m_textdrift": {"identified": True, "method": "randomized text-only injected error"},
    "m_recover": {"identified": True, "method": "content-independent recovery oracle slice"},
    "m_diag": {"identified": False, "method": "observational difficulty-matched upper bound only"},
}


@dataclass(frozen=True)
class LabeledOutput:
    module_name: str
    identified: bool
    method: str
    result: Any


def assert_no_causal_claim_from(module_name: str) -> None:
    component = COMPONENTS.get(module_name)
    if component is None:
        raise KeyError(f"unknown module: {module_name}")
    if not bool(component["identified"]):
        raise RuntimeError(f"{module_name} is non-identifying; causal claims are forbidden")


def label_output(module_name: str, result: Any) -> LabeledOutput:
    component = COMPONENTS.get(module_name)
    if component is None:
        raise KeyError(f"unknown module: {module_name}")
    return LabeledOutput(module_name=module_name, identified=bool(component["identified"]), method=str(component["method"]), result=result)
