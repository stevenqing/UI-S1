import hashlib
import json
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
REPORT_PATH = RUN_DIR / "REPORT.md"
TABLE_PATH = RUN_DIR / "MAIN_TABLES.md"
ADJUDICATION_PATH = RUN_DIR / "ICC_ADJUDICATION.json"
DISCLOSURE_PATH = ROOT / "docs/research_disclosures.md"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if any(path.exists() for path in (REPORT_PATH, TABLE_PATH, ADJUDICATION_PATH)):
        raise FileExistsError("ICC final output exists")
    arm_a = json.loads((RUN_DIR / "ARM_A.json").read_text())
    arm_b = json.loads((RUN_DIR / "ARM_B.json").read_text())
    arm_c = json.loads((RUN_DIR / "ARM_C.json").read_text())
    budget = json.loads((RUN_DIR / "SAME_BUDGET.json").read_text())
    decision = json.loads((RUN_DIR / "FINAL_DECISION.json").read_text())
    if decision["decision"] != "CLOSE_CORRELATION_DISCOUNT_DIRECTION":
        raise PermissionError("ICC closure decision mismatch")

    a_rows = ["| Fold | Selected rho_v | Selected rho_l | Accuracy | Class |", "| ---: | ---: | ---: | ---: | --- |"]
    for row in arm_a["folds"]:
        a_rows.append(f"| {row['outer_fold']} | {row['selected_rho_v']:.1f} | {row['selected_rho_l']:.1f} | {100 * row['selected_accuracy']:.2f}% | {row['endpoint_class']} |")
    c_rows = ["| Stratum | Phi fold mean | Fold range | AndroidControl reference | Difference |", "| --- | ---: | ---: | ---: | ---: |"]
    for name in ("within_lineage", "cross_lineage"):
        value = arm_c["summaries"][name]
        c_rows.append(f"| {name} | {value['fold_mean']:.3f} | [{value['fold_range'][0]:.3f},{value['fold_range'][1]:.3f}] | {value['androidcontrol_reference']:.3f} | {value['signed_difference']:+.3f} |")
    b_rows = ["| Direction | Rows | Wrong-to-correct | Correct-to-wrong | Net | Correction/all |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, value in arm_b["direction_summary"].items():
        rate = "NA" if value["wrong_to_correct_rate_all"] is None else f"{100 * value['wrong_to_correct_rate_all']:.2f}%"
        b_rows.append(f"| {name} | {value['rows']} | {value['wrong_to_correct']} | {value['correct_to_wrong']} | {value['net_correct']:+d} | {rate} |")
    budget_rows = ["| Omitted lineage | Method | Full 3x4 minus 2x6 | 99% CI |", "| --- | --- | ---: | ---: |"]
    for lineage, methods in budget["comparisons_full_minus_omit"].items():
        for method, value in methods.items():
            budget_rows.append(f"| {lineage} | {method} | {100 * value['point_delta']:+.3f} pp | [{100 * value['ci_99'][0]:+.3f},{100 * value['ci_99'][1]:+.3f}] |")
    TABLE_PATH.write_text("\n".join(["# ICC Main Tables", "", "## Arm A", "", *a_rows, "", "## Arm C", "", *c_rows, "", "## Arm B", "", *b_rows, "", "## Same-budget audit", "", *budget_rows, ""]))

    within = arm_c["summaries"]["within_lineage"]
    cross = arm_c["summaries"]["cross_lineage"]
    report = f"""# ICC EVID Premise Audit Report

Date: 2026-08-16

Outcome: `ICC_CLOSE_CORRELATION_DISCOUNT_DIRECTION_A2_TOTAL_APPROX_SUPPORTED`

ICC is a zero-GPU post-selection diagnostic. It changes no prior result or historical status and makes no method claim.

## Disclosure

EVID's fixed constants were AndroidControl failure kappas, not ScreenSpot-Pro error correlations. EVID therefore rejected its frozen transferred-constant variant rather than every possible direct-parameter version of the score family. EVID remains failed and Stage 2 remains blocked. This process error is recorded in `docs/research_disclosures.md`.

## Arm A: fitted rho endpoints

{chr(10).join(a_rows)}

Three of five folds select a low endpoint; two select $(0,0)$ and one selects $\\rho_\\ell=0$. No fold selects a high endpoint. Neighbor deltas are nonpositive around every selected cell. The fitted data therefore prefer little or no discount rather than stronger discount.

## Arm C: direct ScreenSpot-Pro dependence

{chr(10).join(c_rows)}

Direct ScreenSpot-Pro within-lineage phi is **{within['fold_mean']:.3f}**, {within['signed_difference']:+.3f} from the transferred 0.895. Cross-lineage phi is **{cross['fold_mean']:.3f}**, {cross['signed_difference']:+.3f} from 0.398. EVID simultaneously over-discounted within-lineage repeats and under-discounted cross-lineage dependence; the premise error is not a one-direction rescaling.

The empirical phi-matrix $N_{{\\mathrm{{eff}}}}$ is **{arm_c['neff']['empirical_phi']:.4f}**. MASK's empirical kappa-matrix value is **{arm_c['neff']['empirical_kappa']:.4f}**, relative error {100 * arm_c['neff']['kappa_vs_phi_relative_error']:.2f}%. The exchangeable two-level phi formula is {arm_c['neff']['structured_phi']:.4f}.

The structured formula equals the empirical phi-matrix value here by pair-count algebra: 18 within-lineage and 48 cross-lineage pairs with equal weighting exactly reconstruct $\\mathbf1^TR\\mathbf1$. It is not independent corroboration. The nontrivial diagnostic is kappa versus phi, which passes the frozen 10% tolerance. ICC therefore records retrospective A2 total-count approximation support, while GRAN G-P8 remains historically `NOT_ADJUDICABLE_PREREG_UNDERDEFINED`.

## Arm B: destination of changed rows

The fixed scorer changes 111/1,581 rows. It corrects 14 and harms 24, for a net **-10 rows (-0.633 pp)**; 22 remain correct and 51 remain wrong.

{chr(10).join(b_rows)}

104/111 changes increase represented lineage diversity. Those changes correct 13 rows and harm 23, net -10. No `same_L_concentration_increase` row exists, so there is no observed concentration-increase correction rate to compare; it is reported as unavailable rather than zero. On ScreenSpot-Pro, choosing the more lineage-diverse block is not a better correctness indicator than canonical count selection.

## Same-budget lineage audit

{chr(10).join(budget_rows)}

The historical saturation statement is composition-specific. Omitting UI-TARS changes full-minus-omit by only -0.063 pp for both B3 and M1, with intervals crossing zero. Omitting Qwen3 costs +1.645 pp B3 and +1.771 pp M1; omitting GTA1 costs +4.428 pp B3 and +3.416 pp M1. Thus the averaged DECOMP fixed-budget $2\\to3$ cell contrast can be positive while one specific third lineage, UI-TARS, is saturated. The estimands are not contradictory.

The historical 63.88% endpoint is `M1_ccm`, not source-priority majority. The source-priority bridge is separately reported and must not be substituted.

## Final interpretation

Arm A favors low rho. Arm B shows diversity-increase switches have negative net value. These two independent diagnostics satisfy the preregistered human closure rule. Arm C supports kappa as a close total effective-count approximation but does not rescue the row-level selector, and the transferred constants point in opposite errors across dependence strata.

The recorded decision is `CLOSE_CORRELATION_DISCOUNT_DIRECTION`. No correctly-anchored rho rescue round is authorized. All findings remain evaluation-side and post-selection.
"""
    REPORT_PATH.write_text(report)
    adjudication = {"schema_version": 1, "status": "COMPLETE", "outcome": "ICC_CLOSE_CORRELATION_DISCOUNT_DIRECTION_A2_TOTAL_APPROX_SUPPORTED", "evidence_status": "POST_SELECTION_DIAGNOSTIC", "gpu_used": False, "changes_existing_statuses": False, "method_claim_allowed": False, "arm_a_low_endpoint_folds": arm_a["endpoint_class_counts"]["low_endpoint"], "arm_b_net_correct_changes": arm_b["net_correct_changes"], "arm_c_within_phi": within["fold_mean"], "arm_c_cross_phi": cross["fold_mean"], "retrospective_A2_supported": arm_c["retrospective_A2_supported"], "historical_GRAN_G_P8_status": arm_c["historical_GRAN_G_P8_status"], "historical_status_changed": False, "decision": decision["decision"], "new_rho_rescue_round_authorized": False, "disclosure": {"path": str(DISCLOSURE_PATH.relative_to(ROOT)), "sha256": sha256_file(DISCLOSURE_PATH)}, "report": "runs/icc/2026-08-15/REPORT.md", "report_sha256": sha256_file(REPORT_PATH), "main_tables_sha256": sha256_file(TABLE_PATH), "next_action": "CLOSE_ICC_NO_RHO_RESCUE"}
    ADJUDICATION_PATH.write_text(json.dumps(adjudication, indent=2, sort_keys=True) + "\n")
    print(json.dumps(adjudication, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()