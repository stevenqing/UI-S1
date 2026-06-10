from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.config import REPO_ROOT


def write_gate_a_report(preflight: Dict[str, Any], output_path: str | Path = "chorus-n0n1/REPORT_GATE_A.md") -> Path:
    out = Path(output_path)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    entries = preflight.get("benchmarks") or [preflight]

    lines.append("# GATE A Report — Baseline Reproduction")
    lines.append("")
    status = "BLOCKED" if _all_blocking_issues(entries) else "READY_FOR_BASELINE_RUN"
    lines.append(f"**Status:** {status}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    for entry in entries:
        lines.append(
            f"- `{entry.get('benchmark', {}).get('name')}` / `{entry.get('benchmark', {}).get('split')}` "
            f"with `{entry.get('model', {}).get('id')}` from `{entry.get('config_path')}`"
        )
    lines.append("")

    lines.append("## Reproduction Metrics")
    lines.append("")
    lines.append("Not run yet. Phase A is blocked until all prerequisites below are satisfied.")
    lines.append("")

    lines.append("## Prerequisite Check")
    lines.append("")
    for entry in entries:
        lines.append(f"### {entry.get('benchmark', {}).get('name')} / {entry.get('benchmark', {}).get('split')}")
        lines.append("")
        for check in entry.get("checks", []):
            mark = "OK" if check.get("ok") else "MISSING"
            detail = check.get("detail", "")
            lines.append(f"- **{mark}** `{check.get('name')}`: {detail}")
        lines.append("")

    issues = _all_blocking_issues(entries)
    if issues:
        lines.append("## Blocking Issues")
        lines.append("")
        for issue in issues:
            lines.append(f"- {issue}")
        lines.append("")

    lines.append("## Truncation Summary")
    lines.append("")
    total_generations = sum(e.get("truncation_summary", {}).get("total_generations", 0) for e in entries)
    total_truncated = sum(e.get("truncation_summary", {}).get("truncated_generations", 0) for e in entries)
    trunc_rate = total_truncated / total_generations if total_generations else 0.0
    lines.append(f"- Total generations: {total_generations}")
    lines.append(f"- Truncated generations: {total_truncated}")
    lines.append(f"- Truncation rate: {trunc_rate:.2%}")
    lines.append("- Note: no model generations have been run in preflight mode.")
    lines.append("")

    lines.append("## Qualitative Step Records")
    lines.append("")
    examples = []
    for entry in entries:
        examples.extend(entry.get("qualitative_examples", []))
    if not examples:
        lines.append("No qualitative examples sampled because baseline inference has not run.")
    else:
        for example in examples:
            lines.append("```json")
            lines.append(json.dumps(example, ensure_ascii=False, indent=2))
            lines.append("```")
    lines.append("")

    lines.append("## Gate Decision")
    lines.append("")
    if issues:
        lines.append("STOP. Do not proceed to N0. Resolve the blocking issues, run Phase A baseline reproduction, then regenerate this report.")
    else:
        lines.append("STOP for human review after running baseline metrics and filling paper comparison fields.")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _all_blocking_issues(entries: List[Dict[str, Any]]) -> List[str]:
    issues: List[str] = []
    for entry in entries:
        bench = entry.get("benchmark", {}).get("name", "unknown")
        for issue in entry.get("blocking_issues", []):
            issues.append(f"[{bench}] {issue}")
    return issues
