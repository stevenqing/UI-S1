import hashlib
import json
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
REPORT_PATH = RUN_DIR / "REPORT.md"
TABLE_PATH = RUN_DIR / "MAIN_TABLES.md"
ADJUDICATION_PATH = RUN_DIR / "DECOMP_ADJUDICATION.json"
POLICY_PATH = ROOT / "docs/generation_trace_retention_policy.md"
ISOLATED_SPEC_PATH = ROOT / "runs/xscr-label-isolated/2026-08-15/SPEC.md"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percent(value):
    return "NA" if value is None else f"{100 * value:.2f}%"


def pp(value):
    return "NA" if value is None else f"{100 * value:+.3f}"


def ci_pp(value):
    return "NA" if value is None else f"[{100 * value[0]:+.3f},{100 * value[1]:+.3f}]"


def budget_table(arm1, aggregator):
    lines = [
        f"### {aggregator}",
        "",
        "| B | Accuracy | 99% CI | Fold-selected $(n_L,n_V)$ | Boundary folds | V-only comparator |",
        "| ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    historical = arm1["baselines"]["historical_v_only"]
    for row in arm1["budget_tables"][aggregator]:
        cells = ", ".join(f"({value['selected_cell'][0]},{value['selected_cell'][1]})" for value in row["fold_selections"])
        boundary = sum(value["boundary_selected"] for value in row["fold_selections"])
        comparator = historical.get(str(row["budget"]), {}).get(aggregator)
        lines.append(
            f"| {row['budget']} | {100 * row['point_accuracy']:.2f}% | "
            f"[{100 * row['ci_99'][0]:.2f},{100 * row['ci_99'][1]:.2f}]% | {cells} | "
            f"{boundary}/5 | {'NA' if comparator is None else f'{100 * comparator:.2f}%'} |"
        )
    return lines


def mechanism_table(arm1, aggregator):
    lines = [
        f"### {aggregator}",
        "",
        "| B | Lineage variance share | View variance share | Interaction share | Lineage marginal pp | 99% CI | View marginal pp | 99% CI |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for budget in map(str, range(2, 13)):
        variance = arm1["variance_decomposition"][aggregator][budget]
        marginal = arm1["marginal_contrasts"][aggregator][budget]
        lines.append(
            f"| {budget} | {percent(variance['lineage'])} | {percent(variance['view'])} | {percent(variance['interaction'])} | "
            f"{pp(marginal['lineage'])} | {ci_pp(marginal['lineage_ci_99'])} | "
            f"{pp(marginal['view'])} | {ci_pp(marginal['view_ci_99'])} |"
        )
    return lines


def main():
    if any(path.exists() for path in (REPORT_PATH, TABLE_PATH, ADJUDICATION_PATH)):
        raise FileExistsError("DECOMP final output exists")
    arm1 = json.loads((RUN_DIR / "ARM1.json").read_text())
    arm2 = json.loads((RUN_DIR / "ARM2.json").read_text())
    arm3 = json.loads((RUN_DIR / "ARM3_INVENTORY.json").read_text())
    decision = json.loads((RUN_DIR / "DECISION_ARM2.json").read_text())
    if (
        arm1["status"] != "PASS_DECOMP_ARM1_COMPLETE"
        or arm2["status"] != "PASS_DECOMP_ARM2_LABEL_FREE_COMPLETE_AWAITING_HUMAN_DECISION"
        or arm3["status"] != "DECOMP_ARM3_STOP_LOGPROB_CHANNEL_NOT_RETAINED"
        or decision["decision"] != "WRITE_PHYSICALLY_ISOLATED_LABEL_SPEC"
    ):
        raise PermissionError("DECOMP finalization contract mismatch")

    arm2_lines = [
        "| Grouping | Rows | Screens | Q1 / median / Q3 | Singleton screens | Rows on singleton screens |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for grouping, value in arm2["q1"].items():
        if not isinstance(value, dict):
            continue
        arm2_lines.append(
            f"| {grouping} | {value['rows']} | {value['screens']} | "
            f"{value['rows_per_screen_q1']:.1f} / {value['rows_per_screen_median']:.1f} / {value['rows_per_screen_q3']:.1f} | "
            f"{100 * value['singleton_screen_fraction']:.2f}% | {100 * value['rows_on_singleton_screen_fraction']:.2f}% |"
        )
    collision_lines = [
        "| Tolerance | Collision rows | Collision screens |",
        "| ---: | ---: | ---: |",
    ]
    for value in arm2["q2"]:
        collision_lines.append(
            f"| {value['tolerance_pixels']:.0f} px | {value['collision_rows']}/{value['rows']} "
            f"({100 * value['collision_row_fraction']:.3f}%) | {value['collision_screens']}/{value['screens']} "
            f"({100 * value['collision_screen_fraction']:.3f}%) |"
        )

    tables = [
        "# DECOMP Main Tables", "", "## Arm 1 budget configurations", "",
        *budget_table(arm1, "density_B3"), "", *budget_table(arm1, "F1_majority"), "",
        "## Arm 1 variance decomposition and marginal contrasts", "",
        *mechanism_table(arm1, "density_B3"), "", *mechanism_table(arm1, "F1_majority"), "",
        "## Arm 2 structure", "", *arm2_lines, "", *collision_lines, "",
    ]
    TABLE_PATH.write_text("\n".join(tables))

    baseline = arm1["baselines"]
    density_lineage = [
        arm1["marginal_contrasts"]["density_B3"][str(budget)]["lineage"]
        for budget in range(2, 9)
    ]
    majority_lineage = [
        arm1["marginal_contrasts"]["F1_majority"][str(budget)]["lineage"]
        for budget in range(2, 9)
    ]
    report = f"""# DECOMP Pool Allocation, Same-Screen Structure, and Logprob Report

Date: 2026-08-15

Outcome: `DECOMP_COMPLETE_LINEAGE_AXIS_DOMINANT_LOW_SCREEN_STRUCTURE_NO_LOGPROB`

DECOMP is a zero-GPU recomputation over frozen artifacts. It changes no existing result status and introduces no method claim.

## P0 reconciliation

P0 passed after correcting three apparent source issues. XSCR's 1,460/1,400 values are post-seal exploratory row counts, not full-bank sizes. SPLIT's Qwen2.5 name referred to a deferred probe, not a C-uni lineage. The historical 2,094-action Mind2Web DOM lane remains locally unavailable. Arm 1 is therefore ScreenSpot-Pro-only. Full details and hashes are in `LANE_RECONCILIATION.md`.

## Arm 1: allocation decomposition

The full-pool density B3 anchor reproduces at **{100 * arm1['anchors']['full_pool_density_B3']:.2f}%** over 1,581 rows. Arm 1 evaluates 4,083 subsets with budgets 2-12; the 12 singleton subsets are outside the requested budget range. Subsets overlap, so uncertainty resamples application groups/rows and never subsets.

Baseline accuracies are: full-pool majority/best-single **{100 * baseline['accuracy']['majority']:.2f}%**, full-pool density B3 **{100 * baseline['accuracy']['ours']:.2f}%**, A2/A3 **{100 * baseline['accuracy']['A2']:.2f}%**, A4 **{100 * baseline['accuracy']['A4']:.2f}%**, and nested dev-selection **{100 * baseline['nested_dev_selection_accuracy']:.2f}%**.

The lineage marginal is positive at every identifiable budget through B=8: density B3 ranges from **{100 * min(density_lineage):+.2f}** to **{100 * max(density_lineage):+.2f} pp**, and F1 majority from **{100 * min(majority_lineage):+.2f}** to **{100 * max(majority_lineage):+.2f} pp**. View marginals are much smaller and are usually negative beyond the smallest budgets. This direction is consistent with the historical failure kappas, view $\\kappa=0.895$ versus cross-lineage $\\kappa=0.398$.

The largest lineage variance shares occur for density B3 around intermediate budgets, while view and interaction shares are generally smaller. B=12 has only one configuration, so its ANOVA and marginal contrasts are `NA`. Every selected budget cell touches at least one supported-axis boundary in every fold; the budget table is therefore a boundary-sensitive descriptive recommendation, not a stable optimizer or new method.

Mind2Web remains `BLOCKED_ALIGNED_POOL_UNAVAILABLE` and has no Arm 1 table.

The complete budget and mechanism tables are in `MAIN_TABLES.md`.

## Arm 2: label-free same-screen structure

{chr(10).join(arm2_lines)}

{chr(10).join(collision_lines)}

Byte hashing reveals repeated screenshots hidden behind distinct source filenames: source IDs are all singletons, while byte hashes yield 1,551 screens. Even under byte identity, 98.52% of screens are singletons and 96.65% of rows have no same-screen partner. Collision rates are 0-0.253% across `[7,14,28]` pixels. No label, target bbox, evaluator, or prohibited path was opened.

The evidence-based default is to close this lane. The recorded human decision instead authorizes writing a physically isolated label-process specification. That specification is not authorized for execution and cannot restore confirmation because ScreenSpot-Pro labels were already used. See `runs/xscr-label-isolated/2026-08-15/SPEC.md`.

## Arm 3: logprob inventory

Arm 3 inspected {arm3['benchmarks']['screenspot_pro']['files']} ScreenSpot-Pro generating-trace files ({arm3['benchmarks']['screenspot_pro']['rows_across_files']:,} rows across files) and {arm3['benchmarks']['mind2web']['files']} Mind2Web files ({arm3['benchmarks']['mind2web']['rows_across_files']:,} rows across files). It found zero files with generating-model logprobs, generated token IDs, or coordinate-token spans. No labels were opened and no AUROC was computed.

Downstream selector logits exist but are explicitly classified as `DOWNSTREAM_CANDIDATE_SELECTOR_NOT_GENERATING_MODEL_LOGPROB`; they were not substituted. The arm stops as `LOGPROB_CHANNEL_NOT_RETAINED`.

Future forward retention requirements are now repository policy in `docs/generation_trace_retention_policy.md`.

## Boundaries

Arm 1 is a post-hoc descriptive decomposition of the existing +3.605 pp ScreenSpot-Pro pool result. Arm 2 is label-free structure only. Arm 3 is mechanism inventory only. None changes F1, Q1, `TRIVUS_NOT_PROMOTED`, VUS-SR, SPLIT, MASK, CEIL, OTEXT, XSCR, or XSOFT.
"""
    REPORT_PATH.write_text(report)
    adjudication = {
        "schema_version": 1,
        "status": "COMPLETE",
        "outcome": "DECOMP_COMPLETE_LINEAGE_AXIS_DOMINANT_LOW_SCREEN_STRUCTURE_NO_LOGPROB",
        "gpu_used": False,
        "changes_existing_statuses": False,
        "method_claim_allowed": False,
        "arm1": {
            "status": arm1["status"],
            "evidence_status": arm1["evidence_status"],
            "benchmark": arm1["benchmark"],
            "subsets": arm1["subsets_budget_2_to_12"],
            "mind2web_status": arm1["mind2web_status"],
            "full_pool_density_B3": arm1["anchors"]["full_pool_density_B3"],
            "lineage_direction_positive_all_identifiable_B2_B8": all(value > 0 for value in density_lineage + majority_lineage),
        },
        "arm2": {
            "status": arm2["status"],
            "labels_opened": False,
            "singleton_screen_fraction": arm2["q1"]["image_sha256"]["singleton_screen_fraction"],
            "maximum_collision_row_fraction": max(value["collision_row_fraction"] for value in arm2["q2"]),
            "human_decision": decision["decision"],
            "followup_execution_authorized": False,
        },
        "arm3": {
            "status": arm3["status"],
            "generating_model_logprob_available": False,
            "labels_opened": False,
            "auroc_computed": False,
        },
        "forward_retention_policy": {"path": str(POLICY_PATH.relative_to(ROOT)), "sha256": sha256_file(POLICY_PATH)},
        "isolated_label_spec": {"path": str(ISOLATED_SPEC_PATH.relative_to(ROOT)), "sha256": sha256_file(ISOLATED_SPEC_PATH), "execution_authorized": False},
        "report": "runs/decomp/2026-08-14/REPORT.md",
        "report_sha256": sha256_file(REPORT_PATH),
        "main_tables_sha256": sha256_file(TABLE_PATH),
        "next_action": "CLOSE_DECOMP_RETAIN_RESULTS_NO_NEW_EXPERIMENT_AUTHORIZED",
    }
    ADJUDICATION_PATH.write_text(json.dumps(adjudication, indent=2, sort_keys=True) + "\n")
    print(json.dumps(adjudication, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()