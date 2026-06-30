"""Short-circuiting orchestrator for GUI-360 long-horizon stages."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


STAGE_ORDER = [
    "loader_smoke",
    "difficulty_gate",
    "divergence_audit",
    "recovery_audit",
    "textmem_gate",
    "textdrift_plan_core_recover",
    "diag_bound",
    "analysis_decision",
]

GATED_STAGES = {"loader_smoke", "difficulty_gate", "divergence_audit", "recovery_audit", "textmem_gate"}


@dataclass(frozen=True)
class StageResult:
    name: str
    passed: bool
    skipped: bool = False
    details: Dict[str, Any] | None = None


class GateFailed(RuntimeError):
    def __init__(self, result: StageResult):
        super().__init__(f"stage failed: {result.name}")
        self.result = result


def load_config(path: str | Path) -> Dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("config must be a YAML mapping")
    return data


def _passed_from(value: Any) -> bool:
    if isinstance(value, StageResult):
        return bool(value.passed)
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        if "passed" in value:
            return bool(value["passed"])
        if "gate_passed" in value:
            return bool(value["gate_passed"])
    if hasattr(value, "gate_passed"):
        return bool(value.gate_passed)
    if hasattr(value, "passed"):
        return bool(value.passed)
    return True


def _details_from(value: Any) -> Dict[str, Any]:
    if isinstance(value, StageResult):
        return value.details or {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return {"result": repr(value)}


def run_stages(config: Dict[str, Any], runners: Optional[Dict[str, Callable[[Dict[str, Any]], Any]]] = None, *, dry_run: bool = False, stages: Optional[Iterable[str]] = None) -> List[StageResult]:
    """Run stages in build order and short-circuit on pre-registered gates."""

    selected = list(stages) if stages is not None else list(STAGE_ORDER)
    unknown = [stage for stage in selected if stage not in STAGE_ORDER]
    if unknown:
        raise ValueError(f"unknown stages: {unknown}")
    if runners is None:
        from .stages import default_runners

        runners = default_runners()
    results: List[StageResult] = []
    for name in STAGE_ORDER:
        if name not in selected:
            continue
        if dry_run:
            result = StageResult(name=name, passed=True, skipped=True, details={"dry_run": True, "implemented_runner": name in runners})
        elif name not in runners:
            result = StageResult(name=name, passed=False, skipped=True, details={"missing_runner": True})
        else:
            raw = runners[name](config)
            result = raw if isinstance(raw, StageResult) else StageResult(name=name, passed=_passed_from(raw), details=_details_from(raw))
        results.append(result)
        if name in GATED_STAGES and not result.passed:
            raise GateFailed(result)
    return results


def _default_config_path() -> Path:
    return Path(__file__).with_name("configs") / "default.yaml"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run GUI-360 long-horizon stages with gate short-circuiting")
    parser.add_argument("--config", default=str(_default_config_path()))
    parser.add_argument("--dry-run", action="store_true", help="print the stage plan without running network/model work")
    parser.add_argument("--stage", action="append", choices=STAGE_ORDER, help="run only selected stage(s), preserving build-order sorting")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    try:
        results = run_stages(config, dry_run=args.dry_run, stages=args.stage)
    except GateFailed as exc:
        print(json.dumps({"ok": False, "failed": asdict(exc.result)}, indent=2, sort_keys=True))
        return 2
    print(json.dumps({"ok": True, "stages": [asdict(result) for result in results]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
