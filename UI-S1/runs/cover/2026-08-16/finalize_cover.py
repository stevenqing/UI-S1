import hashlib
import json
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
REPORT_PATH = RUN_DIR / "REPORT.md"
TABLE_PATH = RUN_DIR / "MAIN_TABLES.md"
ADJUDICATION_PATH = RUN_DIR / "COVER_ADJUDICATION.json"
PILOT_PATH = ROOT / "runs/complementary-window/2026-08-17/SPEC.md"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if any(path.exists() for path in (REPORT_PATH, TABLE_PATH, ADJUDICATION_PATH)):
        raise FileExistsError("COVER final output exists")
    arm_a = json.loads((RUN_DIR / "ARM_A.json").read_text())
    arm_b = json.loads((RUN_DIR / "ARM_B.json").read_text())
    decision = json.loads((RUN_DIR / "FINAL_DECISION.json").read_text())
    strata_rows = ["| Target-center crop coverage | Rows | Fraction | B3 accuracy |", "| --- | ---: | ---: | ---: |"]
    for name, value in arm_a["target_strata"].items():
        accuracy = "NA" if value["b3_accuracy"] is None else f"{100 * value['b3_accuracy']:.2f}%"
        strata_rows.append(f"| {name} | {value['rows']} | {100 * value['fraction']:.2f}% | {accuracy} |")
    cross_rows = ["| Spatial stratum | Selected correct | Recoverable | Zero candidate-success coverage |", "| --- | ---: | ---: | ---: |"]
    for name, value in arm_a["row_class_cross_table"].items():
        cross_rows.append(f"| {name} | {value['selected_correct']} | {value['recoverable']} | {value['zero_candidate_success_coverage']} |")
    dependence_rows = ["| Benchmark | Within-model phi | Cross-model phi | Phi N_eff |", "| --- | ---: | ---: | ---: |", f"| ScreenSpot-Pro | {arm_b['references']['screenspot_pro']['rho_within']:.3f} | {arm_b['references']['screenspot_pro']['rho_cross']:.3f} | {arm_b['references']['screenspot_pro']['neff']:.3f} |", f"| Mind2Web | {arm_b['summaries']['within_model']['fold_mean']:.3f} | {arm_b['summaries']['cross_model']['fold_mean']:.3f} | {arm_b['empirical_phi_neff']:.3f} |", f"| AndroidControl reference | {arm_b['references']['androidcontrol']['rho_within']:.3f} | {arm_b['references']['androidcontrol']['rho_cross']:.3f} | NA |"]
    trend_rows = ["| Benchmark | Within-model cross-slot | Cross-model matched-role | Cross-model unmatched-role | Ordering |", "| --- | ---: | ---: | ---: | --- |", f"| ScreenSpot-Pro | {arm_b['trend']['screenspot_pro']['means']['within_model_cross_slot']:.3f} | {arm_b['trend']['screenspot_pro']['means']['cross_model_matched_role']:.3f} | {arm_b['trend']['screenspot_pro']['means']['cross_model_unmatched_role']:.3f} | {' > '.join(arm_b['trend']['screenspot_pro']['ordering'])} |", f"| Mind2Web | {arm_b['trend']['mind2web']['means']['within_model_cross_slot']:.3f} | {arm_b['trend']['mind2web']['means']['cross_model_matched_role']:.3f} | {arm_b['trend']['mind2web']['means']['cross_model_unmatched_role']:.3f} | {' > '.join(arm_b['trend']['mind2web']['ordering'])} |"]
    TABLE_PATH.write_text("\n".join(["# COVER Main Tables", "", "## Arm A target coverage", "", *strata_rows, "", "## Arm A row-class cross-table", "", *cross_rows, "", "## Arm B direct dependence", "", *dependence_rows, "", "## Arm B source/stage trend", "", *trend_rows, ""]))
    intersection = arm_a["area"]["intersection_fraction"]
    union = arm_a["area"]["union_fraction"]
    uncovered = arm_a["area"]["uncovered_fraction"]
    conditional = arm_a["conditional_accuracy"]
    report = f"""# COVER Proposer Coverage and Cross-Benchmark Dependence Report

Date: 2026-08-17

Outcome: `COVER_COMPLEMENTARY_SPEC_AUTHORIZED_COMMON_ORDERING_STRENGTH_SPLIT`

COVER is a zero-GPU post-selection diagnostic. It changes no prior result and makes no method claim.

## Arm A: crop-only coverage headroom

Arm A analyzes the 11 GTA1 proposer crop ranks. View 0 is the full-image baseline and is excluded from crop-only intersection/union. All three model lineages share these regions, so this is proposer-rank geometry, not lineage spatial diversity.

Across rows, the common 11-crop intersection occupies median **{100 * intersection['median']:.2f}%** of image area and the crop union occupies median **{100 * union['median']:.2f}%**. Median uncovered area is **{100 * uncovered['median']:.2f}%**. All 1,581 exact uint8 coverage maps are retained.

{chr(10).join(strata_rows)}

Low coverage (`partial_1_10 + uncovered_0`) contains **{100 * arm_a['low_coverage_fraction']:.2f}%** of rows. Common-coverage B3 accuracy exceeds low-coverage accuracy by **{100 * conditional['point_delta']:+.2f} pp**, 99% CI **[{100 * conditional['ci_99'][0]:+.2f},{100 * conditional['ci_99'][1]:+.2f}]**. A-G1 and A-G2 fail; A-G3 passes.

{chr(10).join(cross_rows)}

The 225 completely uncovered target centers have zero B3 successes; 54 are recoverable by another existing C-uni candidate and 171 have zero candidate-success coverage. Spatial coverage and candidate-success coverage are distinct.

The recorded human decision authorizes writing a complementary-window pilot specification only. GPU remains unauthorized. The design-only protocol is `runs/complementary-window/2026-08-17/SPEC.md` and requires a public gate plus a result-free net-benefit ledger before any inference.

## Arm B: cross-benchmark dependence

{chr(10).join(dependence_rows)}

M2W within-model phi is {arm_b['summaries']['within_model']['fold_mean']:.3f}, fold range [{arm_b['summaries']['within_model']['fold_range'][0]:.3f},{arm_b['summaries']['within_model']['fold_range'][1]:.3f}]. Cross-model phi is {arm_b['summaries']['cross_model']['fold_mean']:.3f}, range [{arm_b['summaries']['cross_model']['fold_range'][0]:.3f},{arm_b['summaries']['cross_model']['fold_range'][1]:.3f}]. Its empirical phi $N_{{\\mathrm{{eff}}}}$ is {arm_b['empirical_phi_neff']:.3f}, above ScreenSpot-Pro's 1.573.

{chr(10).join(trend_rows)}

Both benchmarks share the descriptive ordering `within-model > cross-model matched-role > cross-model unmatched-role`. Dependence strength is benchmark-specific: M2W cross-model phi 0.360 is materially below ScreenSpot-Pro 0.577. COVER therefore supports a common source/stage distance ordering, not a universal high-correlation level.

The M2W trend is not a model-scale law. TongUI-7B, CogAgent-18B, and UI-TARS-7B differ in family, architecture, training, and size; slot roles also mix full, view1, and stage2 crops.

## Boundaries

The complementary-window direction is not X2 or SPLIT revival: it minimizes overlap, adds proposals directly, forbids flips/verifiers/two-mode restrictions, and has no GPU authorization. All label-dependent COVER quantities are evaluation-side. Existing statuses remain unchanged.
"""
    REPORT_PATH.write_text(report)
    adjudication = {"schema_version": 1, "status": "COMPLETE", "outcome": "COVER_COMPLEMENTARY_SPEC_AUTHORIZED_COMMON_ORDERING_STRENGTH_SPLIT", "evidence_status": "POST_SELECTION_DIAGNOSTIC", "gpu_used": False, "followup_gpu_authorized": False, "changes_existing_statuses": False, "method_claim_allowed": False, "arm_a_gates": arm_a["gates"], "arm_a_low_coverage_fraction": arm_a["low_coverage_fraction"], "arm_a_accuracy_gap": conditional, "arm_b_mind2web_within_phi": arm_b["summaries"]["within_model"]["fold_mean"], "arm_b_mind2web_cross_phi": arm_b["summaries"]["cross_model"]["fold_mean"], "arm_b_mind2web_neff": arm_b["empirical_phi_neff"], "arm_b_ordering_consistent": arm_b["trend"]["ordering_consistent"], "arm_b_interpretation": decision["arm_b"]["decision"], "complementary_spec": {"path": str(PILOT_PATH.relative_to(ROOT)), "sha256": sha256_file(PILOT_PATH), "gpu_authorized": False}, "report": "runs/cover/2026-08-16/REPORT.md", "report_sha256": sha256_file(REPORT_PATH), "main_tables_sha256": sha256_file(TABLE_PATH), "next_action": "PREPARE_PUBLIC_GATE_AND_NET_LEDGER_NO_GPU"}
    ADJUDICATION_PATH.write_text(json.dumps(adjudication, indent=2, sort_keys=True) + "\n")
    print(json.dumps(adjudication, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()