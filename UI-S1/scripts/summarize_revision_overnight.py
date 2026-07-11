#!/usr/bin/env python3
"""Create a fail-soft summary for the overnight revision research pipeline."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def pp(value: float) -> str:
    return f"{100.0 * value:+.2f}pp"


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def optional_json(path: Path) -> Any | None:
    return read_json(path) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/multiagent_trajectory_revision/full_v1")
    parser.add_argument("--output-dir", default="outputs/multiagent_trajectory_revision/full_v1/overnight")
    parser.add_argument("--git-commit", default="")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.output_dir)
    a4 = optional_json(root / "causal_eval/a4_starting_student/analysis/student_relative_revision_summary.json")
    history = optional_json(root / "causal_analysis/history_intervention/student_relative_revision_summary.json")
    screen = optional_json(root / "lora_screen/report/summary.json")
    full_reports = []
    for path_name in sorted(glob.glob(str(root / "lora_screen/full_eval/*/report/summary.json"))):
        payload = read_json(Path(path_name))
        payload["path"] = path_name
        full_reports.append(payload)
    fullparam_reports = []
    for path_name in sorted(glob.glob(str(root / "fullparam_candidates/*/training_eval/report/summary.json"))):
        payload = read_json(Path(path_name))
        payload["path"] = path_name
        fullparam_reports.append(payload)

    stages = {
        "a4_starting_student": a4 is not None,
        "a5_history_intervention": history is not None,
        "lora_screen": screen is not None,
        "full_lora_confirmation": bool(full_reports),
        "fullparam_confirmation": bool(fullparam_reports),
    }
    summary: dict[str, Any] = {
        "git_commit": args.git_commit,
        "stages": stages,
        "a4_starting_student": a4,
        "a5_history_intervention": history,
        "lora_screen": screen,
        "full_lora_reports": full_reports,
        "fullparam_reports": fullparam_reports,
    }
    write_json(out_dir / "summary.json", summary)

    lines = [
        "# Overnight Revision Research Report",
        "",
        f"Git commit: `{args.git_commit or 'not recorded'}`",
        "",
        "## Stage Status",
        "",
        table(["stage", "complete"], [[name, complete] for name, complete in stages.items()]),
        "",
    ]
    if a4:
        utility = a4["student_relative_revision_utility"]
        boot = utility["cluster_bootstrap"]
        lines.extend([
            "## A4 Starting-Student Relative Utility",
            "",
            f"Student accuracy under revised history: **{pct(utility['student_accuracy'])}**. Revision accuracy: **{pct(utility['revision_accuracy'])}**.",
            f"Net student-relative revision utility: **{pp(utility['net_student_relative_revision_utility'])}**, trajectory-cluster interval **[{pp(boot['lo'])}, {pp(boot['hi'])}]**.",
            "",
        ])
    if history:
        intervention = history.get("history_intervention") or {}
        lines.extend([
            "## A5 Frozen-Student History Intervention",
            "",
            f"Balanced-grid GT-minus-revision-history delta: **{pp(intervention.get('gt_minus_revision_history_delta', 0.0))}**.",
        ])
        standardized = intervention.get("population_standardization")
        if standardized:
            lines.append(f"Population-standardized delta: **{pp(standardized['gt_minus_revision_history_delta'])}**.")
        lines.append("")
    if screen:
        lines.extend([
            "## Equal-Budget LoRA Screen",
            "",
            table(
                ["arm", "role", "ΔTSR", "Δstep", "gate"],
                [
                    [row["arm"], row.get("research_role"), pp(row["tsr_delta"]), pp(row["step_accuracy_delta"]), row["gate"]]
                    for row in screen["arms"]
                ],
            ),
            "",
            "Full-grid candidates: " + (", ".join(screen.get("full_eval_candidates", [])) or "none"),
            "",
        ])
    if full_reports:
        rows = []
        for report in full_reports:
            for arm in report.get("arms", []):
                rows.append([arm["arm"], pp(arm["tsr_delta"]), pp(arm["step_accuracy_delta"]), arm["gate"]])
        lines.extend([
            "## Full 1,000-Episode LoRA Confirmation",
            "",
            table(["arm", "ΔTSR", "Δstep", "gate"], rows),
            "",
        ])
    if fullparam_reports:
        rows = []
        for report in fullparam_reports:
            for arm in report.get("arms", []):
                rows.append([arm["arm"], pp(arm["tsr_delta"]), pp(arm["step_accuracy_delta"]), arm["gate"]])
        lines.extend([
            "## Full-Parameter Confirmation",
            "",
            table(["arm", "ΔTSR", "Δstep", "gate"], rows),
            "",
        ])
    if screen and not screen.get("full_eval_candidates"):
        lines.extend([
            "## Stop Decision",
            "",
            "No deployable LoRA screening arm crossed the predeclared HELPS gate. The pipeline stopped before additional full-parameter training.",
            "",
        ])
    (out_dir / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"stages": stages, "report": str(out_dir / "FINAL_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
