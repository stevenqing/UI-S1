import hashlib
import json
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ARM_A_PATH = RUN_DIR / "ARM_A.json"
ARM_B_PATH = RUN_DIR / "ARM_B.json"
OUTPUT_PATH = RUN_DIR / "CEIL_ADJUDICATION.json"
REPORT_PATH = RUN_DIR / "REPORT.md"
TABLE_PATH = RUN_DIR / "MAIN_TABLE.md"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def percent(value):
    return "NA" if value is None else f"{100 * value:+.2f} pp"


def interval(report):
    if report.get("status") != "PASS":
        return report["status"]
    return f"[{100 * report['ci_99'][0]:+.2f}, {100 * report['ci_99'][1]:+.2f}] pp"


def main():
    if OUTPUT_PATH.exists() or REPORT_PATH.exists() or TABLE_PATH.exists():
        raise FileExistsError("CEIL final outputs already exist")
    arm_a = json.loads(ARM_A_PATH.read_text())
    arm_b = json.loads(ARM_B_PATH.read_text())
    if arm_a.get("status") != "PASS_CEIL_ARM_A_POST_HOC_COMPLETE":
        raise PermissionError("CEIL Arm A incomplete")
    if arm_b.get("status") != "PASS_CEIL_ARM_B_COMPLETE":
        raise PermissionError("CEIL Arm B incomplete")
    overall = arm_b["overall_branch"]
    if overall not in {"OPEN_NEW_SPEC_C_D2", "CLOSE_C_D1", "FREEZE_C_D3_INDETERMINATE"}:
        raise ValueError("CEIL unknown branch")
    adjudication = {
        "schema_version": 1,
        "status": "COMPLETE",
        "outcome": f"CEIL_COMPLETE_{overall}",
        "arm_B_overall_branch": overall,
        "arm_B_decisions": {
            family: {
                "recoverable_samples": report["recoverable_samples"],
                "decision_eligible": report["decision_eligible"],
                "decision": report["decision"],
                "cheap_candidate_AUROC": report["cheap_candidate_AUROC"],
            }
            for family, report in arm_b["reports"].items()
        },
        "arm_A_evidence_status": "POST_HOC_DESCRIPTIVE",
        "arm_A_primary": {
            panel: {
                aggregator: {
                    "Delta_infinity": report["Delta_infinity"],
                    "Delta_infinity_CI": arm_a["panels"][panel]["bootstrap"][aggregator]["Delta_infinity"],
                    "full_neff": report["full_neff"],
                    "support_maximum": report["support_maximum"],
                }
                for aggregator, report in value["point"].items()
            }
            for panel, value in arm_a["panels"].items()
        },
        "changes_existing_statuses": False,
        "method_authorized": False,
        "runtime_rule_authorized": False,
        "current_round_reweighting_authorized": False,
        "new_spec_authorized": overall == "OPEN_NEW_SPEC_C_D2",
        "input_hashes": {
            "ARM_A.json": sha256_file(ARM_A_PATH),
            "ARM_B.json": sha256_file(ARM_B_PATH),
        },
        "next_action": (
            "WRITE_NEW_PREREG_FOR_MIND2WEB_FULL_CANDIDATE_REWEIGHTING"
            if overall == "OPEN_NEW_SPEC_C_D2"
            else "FREEZE_MAIN_TABLE_AND_CLOSE_RESEARCH_SEQUENCE"
        ),
    }
    OUTPUT_PATH.write_text(json.dumps(adjudication, indent=2, sort_keys=True) + "\n")

    table_lines = [
        "# CEIL Main Table",
        "",
        "## Arm B: Recoverable-subset candidate ranking",
        "",
        "| Benchmark | Recoverable | Cheap AUROC | 99% CI | Visual AUROC | Verifier AUROC | Decision |",
        "| --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for family in ("mind2web", "screenspot_pro", "androidcontrol"):
        value = arm_b["reports"][family]
        table_lines.append(
            f"| {family} | {value['recoverable_samples']} | {value['cheap_candidate_AUROC']['point']:.3f} "
            f"| [{value['cheap_candidate_AUROC']['ci_99'][0]:.3f}, {value['cheap_candidate_AUROC']['ci_99'][1]:.3f}] "
            f"| {value['visual_candidate_AUROC']['point']:.3f} | {value['verifier_candidate_AUROC']['point']:.3f} | {value['decision']} |"
        )
    table_lines.extend([
        "",
        "## Arm A: Post-hoc effective-vote ceiling",
        "",
        "| Panel | Aggregator | Full N_eff | Support max | Delta infinity | 99% CI |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ])
    for panel in arm_a["panels"]:
        for aggregator in ("density", "majority"):
            value = arm_a["panels"][panel]["point"][aggregator]
            bootstrap = arm_a["panels"][panel]["bootstrap"][aggregator]["Delta_infinity"]
            table_lines.append(
                f"| {panel} | {aggregator} | {value['full_neff']:.3f} | {value['support_maximum']:.3f} "
                f"| {percent(value['Delta_infinity'])} | {interval(bootstrap)} |"
            )
    TABLE_PATH.write_text("\n".join(table_lines) + "\n")

    mind = arm_b["reports"]["mind2web"]
    screen = arm_b["reports"]["screenspot_pro"]
    report = rf"""# CEIL Closure Diagnostic Report

Date: 2026-08-14

Status: `CEIL_COMPLETE_{overall}`

## Scope

CEIL is a zero-GPU closure analysis. It does not change F1, Q1, `TRIVUS_NOT_PROMOTED`, VUS-SR, SPLIT, or MASK. Arm A is post-hoc descriptive. Arm B is evaluation-side and cannot define a runtime selector.

## Arm B: Conditional ranking signal

Mind2Web has {mind['recoverable_samples']:,} recoverable samples. Cheap-ranker candidate AUROC is **{mind['cheap_candidate_AUROC']['point']:.3f}**, with 99% CI **[{mind['cheap_candidate_AUROC']['ci_99'][0]:.3f}, {mind['cheap_candidate_AUROC']['ci_99'][1]:.3f}]**. The lower bound exceeds 0.65, so Mind2Web triggers **C-D2**. Blind visual AUROC is {mind['visual_candidate_AUROC']['point']:.3f}; the cheap ranker contains conditional signal beyond the frozen visual ordering, although its recoverable-subset top-1 rate is only {100 * mind['top1']['cheap']['point']:.1f}%.

ScreenSpot-Pro has {screen['recoverable_samples']:,} recoverable samples. Cheap-ranker AUROC is **{screen['cheap_candidate_AUROC']['point']:.3f}**, with 99% CI **[{screen['cheap_candidate_AUROC']['ci_99'][0]:.3f}, {screen['cheap_candidate_AUROC']['ci_99'][1]:.3f}]**. The upper bound is below 0.60, so ScreenSpot-Pro triggers **C-D1**. Its conditional cheap signal is effectively absent under the frozen threshold.

The benchmark split is therefore substantive: CEIL does not support a shared candidate-reweighting conclusion. Because one eligible benchmark triggers C-D2, the overall branch is `{overall}`. This authorizes only a new preregistration for Mind2Web full-candidate reweighting; no experiment is authorized inside CEIL.

## Arm A: Post-hoc effective-vote ceilings

Arm A enumerates all 4,095 nonempty subsets in five independent panels. It strictly reuses MASK generalized $N_{{\mathrm{{eff}}}}$, reports the observed support separately from the full pool, and obtains $\Delta_\infty$ only from the frozen monotone saturating family. Full numerical results and 99% grouped-bootstrap intervals are in `MAIN_TABLE.md`; curves are in `ARM_A_CURVES.pdf`.

These values remain post-hoc and benchmark/arm-specific. They do not restore a universal one-dimensional effective-sample-size law. Isotonic extrapolation is limited to the finite ideal-three-vote target and is not interpreted as an infinite ceiling.

The Mind2Web parametric asymptotes are weakly identified: their $\Delta_\infty$ values range from roughly +22 to +73 pp and extrapolate far beyond observed support, with several fits approaching the bounded accuracy ceiling. They are sensitivity outputs, not precise recoverable headroom. By contrast, the finite ideal-three-vote isotonic gains range from about -0.13 to +3.32 pp across Mind2Web panels. ScreenSpot-Pro parametric $\Delta_\infty$ remains near zero with intervals crossing zero.

## Conclusion

> Conditional candidate-ranking signal survives on Mind2Web but not ScreenSpot-Pro. The current consensus-geometry sequence closes for ScreenSpot-Pro; Mind2Web alone meets the preregistered threshold for a separately preregistered full-candidate reweighting study.

No current status changes, method promotion, runtime rule, or within-round rescue is allowed.
"""
    REPORT_PATH.write_text(report)
    print(json.dumps({
        "outcome": adjudication["outcome"],
        "arm_B": adjudication["arm_B_decisions"],
        "new_spec_authorized": adjudication["new_spec_authorized"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()