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
    factorial = optional_json(root / "lora_screen/factorial_analysis/summary.json")
    metadata_gate = optional_json(root / "utility_gate/metadata_v1/summary.json")
    a13 = optional_json(root / "utility_gate/a13_lora/report/summary.json")
    verifier = optional_json(root / "revision_verifier/eval/merged.summary.json")
    verifier_natural = optional_json(root / "revision_verifier_natural/eval/merged.summary.json")
    verifier_calibration = optional_json(root / "revision_verifier/calibration/summary.json")
    rescue_rankers = []
    for name, relative in (
        ("v1", "revision_rescue_ranker/scores"),
        ("v2_regress", "revision_rescue_ranker_v2_regress/scores"),
        ("v3_unique", "revision_rescue_ranker_v3_unique/scores"),
    ):
        calibration = optional_json(root / relative / "calibration/summary.json")
        dev_metrics = optional_json(root / relative / "dev.summary.json")
        test_metrics = optional_json(root / relative / "test.summary.json")
        if calibration and dev_metrics and test_metrics:
            rescue_rankers.append({"name": name, "calibration": calibration, "dev": dev_metrics, "test": test_metrics})
    transition_gate = optional_json(root / "transition_revision_gate/summary.json")
    replay_doses = []
    for path_name in sorted(glob.glob(str(root / "utility_gate/a1*_lora/report/summary.json"))):
        payload = read_json(Path(path_name)); payload["path"] = path_name; replay_doses.append(payload)
    a15_full = optional_json(root / "utility_gate/a15_student_rescue25_replay75_lora/full_eval/report/summary.json")
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
        "factorial_analysis": factorial is not None,
        "metadata_utility_gate": metadata_gate is not None,
        "oracle_student_rescue_ceiling": a13 is not None,
        "multimodal_verifier": verifier is not None,
        "natural_prior_verifier": verifier_natural is not None,
        "verifier_calibration": verifier_calibration is not None,
        "rescue_ranker_ablation": bool(rescue_rankers),
        "transition_revision_gate": transition_gate is not None,
        "replay_dose_ablation": bool(replay_doses),
        "a15_full_confirmation": a15_full is not None,
        "full_lora_confirmation": bool(full_reports),
        "fullparam_confirmation": bool(fullparam_reports),
    }
    summary: dict[str, Any] = {
        "git_commit": args.git_commit,
        "stages": stages,
        "a4_starting_student": a4,
        "a5_history_intervention": history,
        "lora_screen": screen,
        "factorial_analysis": factorial,
        "metadata_utility_gate": metadata_gate,
        "oracle_student_rescue_ceiling": a13,
        "multimodal_verifier": verifier,
        "natural_prior_verifier": verifier_natural,
        "verifier_calibration": verifier_calibration,
        "rescue_rankers": rescue_rankers,
        "transition_revision_gate": transition_gate,
        "replay_dose_reports": replay_doses,
        "a15_full_confirmation": a15_full,
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
    if factorial:
        step_effects = factorial["effects"]["step_accuracy"]
        lines.extend([
            "## Target × History Factorial",
            "",
            f"GT-history effect with GT targets: **{pp(step_effects['gt_history_effect_given_gt_target'])}**.",
            f"GT-history effect with revision targets: **{pp(step_effects['gt_history_effect_given_revision_target'])}**.",
            f"Revision-label effect under GT history: **{pp(step_effects['revision_label_effect_given_gt_history'])}**.",
            f"Label × history interaction: **{pp(step_effects['label_history_interaction'])}**.",
            "",
        ])
    if metadata_gate:
        test = metadata_gate["evaluations"]["test"]
        lines.extend([
            "## Metadata Utility Gate",
            "",
            f"Episode-disjoint test ROC-AUC **{test['roc_auc']:.4f}**, AP **{test['average_precision']:.4f}** at rescue base rate {pct(test['rescue_base_rate'])}.",
            "No nonzero-coverage operating point achieved positive net accepted utility.",
            "",
        ])
    if a13:
        arm = a13["arms"][0]
        lines.extend([
            "## Oracle Student-Rescue Ceiling",
            "",
            f"A13 ΔTSR **{pp(arm['tsr_delta'])}**, Δstep **{pp(arm['step_accuracy_delta'])}**, gate **{arm['gate']}**.",
            "",
        ])
    if verifier:
        lines.extend([
            "## Multimodal Verifier Agent",
            "",
            f"Decision accuracy **{pct(verifier['accuracy'])}**, macro-F1 **{verifier['macro_f1']:.4f}**, use-revision precision **{pct(verifier['per_class']['use_revision']['precision'])}**, recall **{pct(verifier['per_class']['use_revision']['recall'])}**.",
            f"Fallback-student routed accuracy **{pct(verifier['fallback_student_accuracy'])}** versus student baseline **{pct(verifier['student_baseline_accuracy'])}**.",
            "",
        ])
    if verifier_natural:
        lines.extend([
            "## Natural-Prior Verifier",
            "",
            f"Decision accuracy **{pct(verifier_natural['accuracy'])}**, macro-F1 **{verifier_natural['macro_f1']:.4f}**, use-revision recall **{pct(verifier_natural['per_class']['use_revision']['recall'])}**.",
            f"Fallback-student routed accuracy **{pct(verifier_natural['fallback_student_accuracy'])}** versus baseline **{pct(verifier_natural['student_baseline_accuracy'])}**.",
            "",
        ])
    if verifier_calibration:
        locked = verifier_calibration["locked_test_result"]
        lines.extend([
            "## Conservative Verifier Calibration",
            "",
            f"Locked rule: **{locked['rule']}**; coverage **{pct(locked['coverage'])}**, population utility **{pp(locked['population_net_utility'])}**.",
            "",
        ])
    if rescue_rankers:
        lines.extend([
            "## Calibrated Binary Rescue Rankers",
            "",
            table(
                ["ranker", "dev AUC", "dev AP", "test AUC", "test AP", "gate", "test utility"],
                [
                    [
                        item["name"], f"{item['dev']['roc_auc']:.3f}", f"{item['dev']['average_precision']:.3f}",
                        f"{item['test']['roc_auc']:.3f}", f"{item['test']['average_precision']:.3f}",
                        item["calibration"]["gate"], pp(item["calibration"]["test"]["population_net_utility"]),
                    ]
                    for item in rescue_rankers
                ],
            ),
            "",
        ])
    if transition_gate:
        locked = transition_gate["locked_test"]
        lines.extend([
            "## Visual-Transition Revision Gate",
            "",
            f"Gate **{transition_gate['gate']}**; accepted {locked['accepted']} test rows with {locked['rescue']} rescues and {locked['regress']} regressions, population utility **{pp(locked['population_net_utility'])}**.",
            "",
        ])
    if replay_doses:
        dose_rows = []
        for report in replay_doses:
            for arm in report.get("arms", []):
                dose_rows.append([arm["arm"], pp(arm["tsr_delta"]), pp(arm["step_accuracy_delta"]), arm["gate"]])
        lines.extend([
            "## Oracle Student-Rescue Replay Dose",
            "",
            table(["arm", "ΔTSR", "Δstep", "gate"], dose_rows),
            "",
        ])
    if a15_full:
        arm = a15_full["arms"][0]
        lines.extend([
            "## A15 Full 1,000-Episode Confirmation",
            "",
            f"ΔTSR **{pp(arm['tsr_delta'])}**, Δstep **{pp(arm['step_accuracy_delta'])}**, gate **{arm['gate']}**.",
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
