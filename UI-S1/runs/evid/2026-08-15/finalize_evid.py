import hashlib
import json
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
REPORT_PATH = RUN_DIR / "REPORT.md"
TABLE_PATH = RUN_DIR / "MAIN_TABLE.md"
ADJUDICATION_PATH = RUN_DIR / "EVID_ADJUDICATION.json"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def comparison_row(stage1, variant, baseline):
    value = stage1["comparisons"][variant][baseline]
    return f"| {variant} | {baseline} | {100 * value['point_delta']:+.3f} pp | [{100 * value['ci_99'][0]:+.3f},{100 * value['ci_99'][1]:+.3f}] pp |"


def main():
    if any(path.exists() for path in (REPORT_PATH, TABLE_PATH, ADJUDICATION_PATH)):
        raise FileExistsError("EVID final output exists")
    stage0 = json.loads((RUN_DIR / "STAGE0.json").read_text())
    selected = json.loads((RUN_DIR / "SELECTED_PARAMETERS.json").read_text())
    stage1 = json.loads((RUN_DIR / "STAGE1.json").read_text())
    if (
        stage0["status"] != "PASS_EVID_STAGE0_COMPLETE"
        or selected["status"] != "PASS_EVID_STAGE1_NESTED_SELECTIONS_BEFORE_OUTER_EVALUATION"
        or stage1["status"] != "PASS_EVID_STAGE1_COMPLETE"
        or stage1["stage2_authorized"] is not False
    ):
        raise PermissionError("EVID finalization contract mismatch")
    rows = [
        "| Variant | Accuracy | vs dev-selection | 99% CI | vs A4 | 99% CI | vs B3 | 99% CI |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, accuracy in stage1["accuracy"]["variants"].items():
        dev = stage1["comparisons"][variant]["nested_dev_selection"]
        a4 = stage1["comparisons"][variant]["A4"]
        b3 = stage1["comparisons"][variant]["B3"]
        rows.append(
            f"| {variant} | {100 * accuracy:.2f}% | {100 * dev['point_delta']:+.3f} pp | "
            f"[{100 * dev['ci_99'][0]:+.3f},{100 * dev['ci_99'][1]:+.3f}] | "
            f"{100 * a4['point_delta']:+.3f} pp | [{100 * a4['ci_99'][0]:+.3f},{100 * a4['ci_99'][1]:+.3f}] | "
            f"{100 * b3['point_delta']:+.3f} pp | [{100 * b3['ci_99'][0]:+.3f},{100 * b3['ci_99'][1]:+.3f}] |"
        )
    baseline_rows = [
        "| Baseline | Accuracy |",
        "| --- | ---: |",
        *[f"| {name} | {100 * value:.2f}% |" for name, value in stage1["accuracy"]["baselines"].items()],
    ]
    table = "\n".join(["# EVID Main Table", "", *rows, "", "## Baselines", "", *baseline_rows, ""])
    TABLE_PATH.write_text(table)

    fixed = stage1["comparisons"]["fixed"]["nested_dev_selection"]
    density_23 = stage0["lineage_transitions"]["density_B3"]["2_to_3"]
    majority_23 = stage0["lineage_transitions"]["F1_majority"]["2_to_3"]
    boundary_folds = sum(record["rho_boundary_selected"] for record in selected["folds"])
    report = f"""# EVID Source-Aware Effective-Evidence Aggregation Report

Date: 2026-08-15

Outcome: `EVID_FIXED_AGGREGATOR_FAILED_STAGE2_BLOCKED`

EVID is a zero-GPU, single-benchmark post-selection validation. The fixed rho values are AndroidControl failure-kappa heuristics, not validated ScreenSpot-Pro intraclass correlations. No existing project status changes.

## Stage 0

The rho-zero control reproduced canonical B3 row by row with zero mismatches. The fixed scorer selected a different block on 111/1,581 rows (**7.02%**), so E-G2 passed.

The fixed-output block oracle reaches **{100 * stage0['oracle']['output_correct_accuracy']:.2f}%**, or **{100 * stage0['oracle']['output_oracle_gain']:+.2f} pp** over nested dev-selection. Contains-any-correct coverage is {100 * stage0['oracle']['contains_correct_accuracy']:.2f}%; the difference is block-output loss and is not counted as attainable oracle accuracy. E-G1 passed.

The separated $2\to3$ lineage marginal is **{100 * density_23['pooled_point_delta']:+.3f} pp** for density B3, 99% CI **[{100 * density_23['ci_99'][0]:+.3f},{100 * density_23['ci_99'][1]:+.3f}]**, but only **{100 * majority_23['pooled_point_delta']:+.3f} pp** for F1 majority, CI **[{100 * majority_23['ci_99'][0]:+.3f},{100 * majority_23['ci_99'][1]:+.3f}]**. The conservative minimum is below 0.70 pp, so E-G3 failed and Stage 2 was permanently blocked before any GPU request.

## Stage 1

{chr(10).join(rows)}

The fixed primary reaches **{100 * stage1['accuracy']['variants']['fixed']:.2f}%**. Relative to nested dev-selection it is **{100 * fixed['point_delta']:+.3f} pp**, 99% CI **[{100 * fixed['ci_99'][0]:+.3f},{100 * fixed['ci_99'][1]:+.3f}]**. E-K1 triggers and the parameter-fixed theoretical variant fails.

The lineage-weighted variant is identical in aggregate to fixed EVID. The fitted-rho variant reaches {100 * stage1['accuracy']['variants']['rho_fitted']:.2f}% but remains below dev-selection, and {boundary_folds}/5 folds select a rho-grid endpoint. E-K5 triggers; the grid is not expanded and the fitted result cannot replace the primary failure.

The exact-singleton control equals the 59.84% source-priority majority/best-single endpoint and is substantially worse than finite EVID. E-K3 does not trigger because finite EVID itself is not positively distinguishable from dev-selection.

The diagonal additive-to-average path has Spearman correlation {stage1['path']['spearman']:.3f}, below the frozen 0.8 criterion. E-K4 triggers and the path-unification narrative is deleted. The endpoint jump is retained as sensitivity behavior, not evidence of a smooth majority transition.

## Interpretation

Stage 0 shows genuine block-selection headroom, but the frozen source-aware heuristic does not identify it. Discounting repeated same-lineage votes with AndroidControl-derived kappa anchors lowers accuracy below B3, A4, and nested dev-selection. This closes the fixed EVID score family on the current bank; neither fitted weights nor fitted rho rescues it under the preregistered rules.

Stage 2's proposed six-lineage equal-budget reallocation is not authorized because E-G3 failed before Stage 1 and Stage 1 was negative. No GPU forward was run.

## Boundaries

The result is ScreenSpot-Pro-only and post-selection. Mind2Web remains `BLOCKED_ALIGNED_POOL_UNAVAILABLE`. EVID changes none of F1, Q1, `TRIVUS_NOT_PROMOTED`, VUS-SR, SPLIT, MASK, CEIL, OTEXT, XSCR, DECOMP, or XSOFT.
"""
    REPORT_PATH.write_text(report)
    adjudication = {
        "schema_version": 1,
        "status": "COMPLETE",
        "outcome": "EVID_FIXED_AGGREGATOR_FAILED_STAGE2_BLOCKED",
        "evidence_status": "POST_SELECTION_SINGLE_BENCHMARK_VALIDATION",
        "gpu_used": False,
        "stage2_authorized": False,
        "stage2_blocked_by_E_G3": True,
        "changes_existing_statuses": False,
        "fixed_accuracy": stage1["accuracy"]["variants"]["fixed"],
        "fixed_minus_dev_selection": fixed,
        "positive_primary": False,
        "kill_conditions": stage1["kill_conditions"],
        "stage0_gates": stage0["gates"],
        "mind2web_status": "BLOCKED_ALIGNED_POOL_UNAVAILABLE",
        "report": "runs/evid/2026-08-15/REPORT.md",
        "report_sha256": sha256_file(REPORT_PATH),
        "main_table_sha256": sha256_file(TABLE_PATH),
        "next_action": "CLOSE_EVID_SCORE_FAMILY_NO_STAGE2_GPU",
    }
    ADJUDICATION_PATH.write_text(json.dumps(adjudication, indent=2, sort_keys=True) + "\n")
    print(json.dumps(adjudication, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()