import hashlib
import json
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
REPORT_PATH = RUN_DIR / "REPORT.md"
TABLE_PATH = RUN_DIR / "MAIN_TABLE.md"
ADJUDICATION_PATH = RUN_DIR / "XSCR_ADJUDICATION.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percent(value):
    return f"{100 * value:.2f}%"


def main():
    if any(path.exists() for path in (REPORT_PATH, TABLE_PATH, ADJUDICATION_PATH)):
        raise FileExistsError("XSCR final output exists")
    q1 = json.loads((RUN_DIR / "Q1.json").read_text())
    q2 = json.loads((RUN_DIR / "Q2.json").read_text())
    bounds = json.loads((RUN_DIR / "Q3_Q4.json").read_text())
    decision = json.loads((RUN_DIR / "FINAL_DECISION.json").read_text())
    if (
        q1["status"] != "PASS_XSCR_Q1_COMPLETE_AWAITING_HUMAN_DECISION"
        or q2["status"] != "PASS_XSCR_Q2_COMPLETE_AWAITING_HUMAN_DECISION"
        or bounds["status"] != "PASS_XSCR_Q3_Q4_COMPLETE"
        or decision["decision"] != "AUTHORIZE_EXPLORATORY_METHOD_SPEC"
    ):
        raise PermissionError("XSCR finalization contract mismatch")

    q1_lines = [
        "| Lane | Rows | Screens | Q1 / median / Q3 | Singleton screens | Rows on singleton screens |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lane, value in q1["lanes"].items():
        q1_lines.append(
            f"| {lane} | {value['rows']} | {value['screens']} | "
            f"{value['rows_per_screen_q1']:.1f} / {value['rows_per_screen_median']:.1f} / {value['rows_per_screen_q3']:.1f} | "
            f"{percent(value['singleton_screen_fraction'])} | {percent(value['rows_on_singleton_screen_fraction'])} |"
        )
    bound_lines = [
        "| Lane | Tolerance | Collision | Repairable | Damageable | Signed proxy | Shared target |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    q2_map = {
        (lane, value["tolerance"]): value
        for lane, values in q2["lanes"].items() for value in values
    }
    for lane, values in bounds["lanes"].items():
        for value in values:
            collision = q2_map[(lane, value["tolerance"])]["collision_row_fraction"]
            shared = value["shared_target_fraction"]
            shared_text = "NA" if shared is None else percent(shared)
            bound_lines.append(
                f"| {lane} | {value['tolerance']:.6g} | {percent(collision)} | "
                f"{value['repairable_rows']} ({percent(value['repairable_over_all'])}) | "
                f"{value['damageable_rows']} ({percent(value['damageable_over_all'])}) | "
                f"{value['signed_screening_proxy_pp']:+.3f} pp | {shared_text} |"
            )
    table = "\n".join(["# XSCR Main Tables", "", "## Q1", "", *q1_lines, "", "## Q2-Q4", "", *bound_lines, ""])
    TABLE_PATH.write_text(table)

    mind_best = max(bounds["lanes"]["mind2web"], key=lambda value: value["signed_screening_proxy_pp"])
    android_best = max(
        bounds["lanes"]["androidcontrol_low"] + bounds["lanes"]["androidcontrol_high"],
        key=lambda value: value["signed_screening_proxy_pp"],
    )
    report = f"""# XSCR Same-Screen Cross-Row Feasibility Report

Date: 2026-08-14

Outcome: `XSCR_COMPLETE_BELOW_MDE_EXPLORATORY_SPEC_AUTHORIZED`

Evidence status: `POST_SELECTION_FEASIBILITY`. This round is descriptive, is not confirmatory, evaluates no method, and changes no existing project status.

## Structure

{chr(10).join(q1_lines)}

Byte-identical screens are overwhelmingly singletons: {percent(q1['lanes']['mind2web']['singleton_screen_fraction'])} for Mind2Web and {percent(q1['lanes']['androidcontrol_low']['singleton_screen_fraction'])} for each AndroidControl setting. The public-only seal audit also falsified the assumption that byte-identical Mind2Web screens never cross existing folds; a future transductive evaluation must isolate by screen rather than rely on row folds alone.

## Collision and paired bounds

{chr(10).join(bound_lines)}

AndroidControl's collision surface is at most 0.43%, and its signed screening proxy is never positive. Mind2Web collision ranges from 3.77% to 6.37%. Its best paired structural proxy is **{mind_best['signed_screening_proxy_pp']:+.3f} pp** at tolerance {mind_best['tolerance']:.6g}, below the preregistered 0.70 pp MDE. The best AndroidControl proxy is {android_best['signed_screening_proxy_pp']:+.3f} pp.

The Mind2Web shared-target diagnostic rises to {percent(bounds['lanes']['mind2web'][-2]['shared_target_fraction'])} at tolerance {bounds['lanes']['mind2web'][-2]['tolerance']:.6g}. Large tolerances therefore merge genuinely shared targets as well as competing locations, supporting soft rather than hard exclusion.

## Decision

The default evidence-based decision would be to close the method direction because the optimistic signed proxy is below MDE and AndroidControl supplies no positive net surface. The recorded human decision instead authorizes writing an **exploratory** soft-assignment specification. That future round remains post-selection, must evaluate only after freezing the method against the prospective internal holdout, cannot claim confirmation, and cannot enter the existing main table as a same-protocol improvement.

Q3, Q4, and shared-target diagnostics are evaluation-side only. They do not define a runtime gate.
"""
    REPORT_PATH.write_text(report)
    adjudication = {
        "schema_version": 1,
        "status": "COMPLETE",
        "outcome": "XSCR_COMPLETE_BELOW_MDE_EXPLORATORY_SPEC_AUTHORIZED",
        "evidence_status": "POST_SELECTION_FEASIBILITY",
        "confirmatory_claim_allowed": False,
        "method_claim_allowed": False,
        "changes_existing_statuses": False,
        "best_mind2web_signed_proxy_pp": mind_best["signed_screening_proxy_pp"],
        "best_androidcontrol_signed_proxy_pp": android_best["signed_screening_proxy_pp"],
        "mde_pp": 0.70,
        "exploratory_method_spec_authorized": True,
        "report": "runs/xscr/2026-08-14/REPORT.md",
        "report_sha256": sha256_file(REPORT_PATH),
        "main_table_sha256": sha256_file(TABLE_PATH),
    }
    ADJUDICATION_PATH.write_text(json.dumps(adjudication, indent=2, sort_keys=True) + "\n")
    print(json.dumps(adjudication, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()