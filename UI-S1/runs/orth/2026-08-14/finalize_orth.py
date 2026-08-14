import hashlib
import json
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
DECISION_PATH = RUN_DIR / "SCOPING_DECISION.json"
OUTPUT_PATH = RUN_DIR / "ORTH_ADJUDICATION.json"
REPORT_PATH = RUN_DIR / "REPORT.md"
TABLE_PATH = RUN_DIR / "MAIN_TABLES.md"
ALLOWED_DIRECTIONS = {
    "PREREGISTER_OCR_CONFIRMATORY",
    "RESTORE_DOM_BEFORE_STRUCTURED_CHANNEL_STUDY",
    "STOP_ORTHOGONAL_CHANNEL_FOLLOWUP",
}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setting_family(setting):
    return setting.split("/", 1)[0]


def range_value(values):
    finite = [float(value) for value in values if isinstance(value, (int, float))]
    return None if not finite else [min(finite), max(finite)]


def family_ranges(arm1, engine, family):
    settings = {
        name: value for name, value in arm1["engines"][engine].items()
        if setting_family(name) == family
    }
    output = {
        "settings": len(settings),
        "match_rate": range_value(value["match_rate"] for value in settings.values()),
        "all_row_accuracy": range_value(value["all_row_accuracy"] for value in settings.values()),
        "matched_only_accuracy": range_value(value["matched_only_accuracy"] for value in settings.values()),
        "text_accuracy": range_value(value["ui_type"]["text"]["all_row_accuracy"] for value in settings.values()),
        "icon_accuracy": range_value(value["ui_type"]["icon"]["all_row_accuracy"] for value in settings.values()),
        "all_error_kappa": range_value(
            value["error_kappa"]["all"] for value in settings.values()
            if isinstance(value["error_kappa"]["all"], (int, float))
        ),
        "class_match_rate": {},
    }
    for class_name in ("selected_correct", "recoverable", "zero_coverage"):
        rates = []
        for value in settings.values():
            row = value["class_match_table"][class_name]
            rates.append(row["matched"] / (row["matched"] + row["unmatched"]))
        output["class_match_rate"][class_name] = range_value(rates)
    return output


def format_range(value, percentage=False):
    if value is None:
        return "NA"
    scale = 100 if percentage else 1
    suffix = "%" if percentage else ""
    return f"[{scale * value[0]:.2f}, {scale * value[1]:.2f}]{suffix}"


def main():
    if OUTPUT_PATH.exists() or REPORT_PATH.exists() or TABLE_PATH.exists():
        raise FileExistsError("ORTH final outputs exist")
    arm0 = json.loads((RUN_DIR / "ARM0.json").read_text())
    arm1 = json.loads((RUN_DIR / "ARM1.json").read_text())
    arm2 = json.loads((RUN_DIR / "ARM2.json").read_text())
    arm3 = json.loads((RUN_DIR / "ARM3.json").read_text())
    decision = json.loads(DECISION_PATH.read_text())
    if decision.get("direction") not in ALLOWED_DIRECTIONS or not decision.get("rationale"):
        raise ValueError("ORTH invalid scoping decision")
    ranges = {
        engine: {
            family: family_ranges(arm1, engine, family)
            for family in ("exact", "normalized", "edit")
        }
        for engine in ("easyocr", "rapidocr")
    }
    outcome = f"ORTH_COMPLETE_{decision['direction']}"
    adjudication = {
        "schema_version": 1,
        "status": "COMPLETE_SCOPING",
        "outcome": outcome,
        "direction": decision["direction"],
        "rationale": decision["rationale"],
        "arm0": {
            family: {
                "group_clustered": value["group_clustered"],
                "row_clustered": value["row_clustered"],
                "unclustered_context_candidate": value["unclustered_context_candidate"],
                "recoverable_sample_keys": value["arm_expanded_recoverable_sample_keys"],
                "recoverable_base_row_union": value["recoverable_unique_base_row_union"],
            }
            for family, value in arm0["reports"].items()
        },
        "arm1_ranges": ranges,
        "arm1_row_classes": arm1["row_class_counts"],
        "arm2_status": arm2["status"],
        "arm3_identifiability_boundary": arm3["identifiability_boundary"],
        "paper_result_allowed": False,
        "method_claim_allowed": False,
        "runtime_rule_allowed": False,
        "changes_existing_statuses": False,
        "next_action": decision["direction"],
        "input_hashes": {
            name: sha256_file(RUN_DIR / f"{name}.json")
            for name in ("ARM0", "ARM1", "ARM2", "ARM3")
        },
    }
    OUTPUT_PATH.write_text(json.dumps(adjudication, indent=2, sort_keys=True) + "\n")

    lines = [
        "# ORTH Scoping Tables", "", "## Arm 0: CEIL analysis units", "",
        "| Benchmark | Point | Group 99% CI | Base-row 99% CI | IID pair 99% CI | Sample keys | Base rows |",
        "| --- | ---: | --- | --- | --- | ---: | ---: |",
    ]
    for family in ("mind2web", "screenspot_pro"):
        value = arm0["reports"][family]
        lines.append(
            f"| {family} | {value['group_clustered']['point']:.3f} | "
            f"[{value['group_clustered']['ci_99'][0]:.3f}, {value['group_clustered']['ci_99'][1]:.3f}] | "
            f"[{value['row_clustered']['ci_99'][0]:.3f}, {value['row_clustered']['ci_99'][1]:.3f}] | "
            f"[{value['unclustered_context_candidate']['ci_99'][0]:.3f}, {value['unclustered_context_candidate']['ci_99'][1]:.3f}] | "
            f"{value['arm_expanded_recoverable_sample_keys']} | {value['recoverable_unique_base_row_union']} |"
        )
    lines.extend([
        "", "## Arm 1: OCR matcher-family ranges", "",
        "| Engine | Matcher | Match rate | Selected-correct match | Recoverable match | Zero-coverage match | Text accuracy | Icon accuracy | Error kappa |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for engine in ("easyocr", "rapidocr"):
        for family in ("exact", "normalized", "edit"):
            value = ranges[engine][family]
            lines.append(
                f"| {engine} | {family} | {format_range(value['match_rate'], True)} | "
                f"{format_range(value['class_match_rate']['selected_correct'], True)} | "
                f"{format_range(value['class_match_rate']['recoverable'], True)} | "
                f"{format_range(value['class_match_rate']['zero_coverage'], True)} | "
                f"{format_range(value['text_accuracy'], True)} | {format_range(value['icon_accuracy'], True)} | "
                f"{format_range(value['all_error_kappa'])} |"
            )
    lines.extend([
        "", "## Arm 2: DOM/AX availability", "",
        f"`{arm2['status']}`. Historical complete HTML data was audited, but the local official dataset and scores are missing. The current XFER lane retains only label-selected positive snippets, which are not valid predictor input.",
        "", "## Arm 3: Identifiable headroom", "",
        f"{len(arm3['table'])} accuracy/kappa cells were evaluated; {sum(row['projected_to_feasible'] for row in arm3['table'])} requested cells required projection to a feasible joint error table. Bayes-fused grounding accuracy is not identified from marginal accuracy and error kappa alone.",
    ])
    TABLE_PATH.write_text("\n".join(lines) + "\n")

    report = rf"""# ORTH Orthogonal-Evidence Scoping Report

Date: 2026-08-14

Status: `{outcome}`

## Scope

ORTH is exploratory scoping only. No result is eligible for a paper claim, method claim, runtime rule, or change to any existing project status. A later confirmatory round must regenerate any claim-eligible evidence.

## Arm 0

CEIL's recoverable counts are unique arm-expanded sample keys, not candidate counts. Mind2Web has 2,021 recoverable sample keys but 891 unique base rows; ScreenSpot-Pro has 968 sample keys and 430 base rows. CEIL's primary interval was already group-clustered. The base-row-clustered sensitivity leaves the Mind2Web lower bound above 0.65, while the IID candidate-pair interval is much narrower and anti-conservative.

## Arm 1

Two independently implemented CPU OCR engines were run over all 1,581 ScreenSpot-Pro screenshots. The report uses matcher-family ranges over the complete frozen grid rather than selecting a best engine or threshold. Full per-setting results are in `ARM1.json`; range tables are in `MAIN_TABLES.md`.

OCR is evaluated separately on 977 text targets and 604 icon targets and projected onto selected-correct, recoverable, and zero-coverage row classes. All accuracy, overlap, and kappa values are evaluation-side.

## Arm 2

The official 2,094-action Mind2Web HTML lane was historically downloaded and completely audited, but its local dataset and candidate-score files are now absent. The current 2,080-row XFER lane has no full DOM/AX tree and retains only GT-selected positive snippets for 1,975 rows. No DOM predictor metric is computed; restoring and hashing the official data is a prerequisite.

## Arm 3

Marginal channel accuracy and error kappa identify a joint 2-by-2 error table, disagreement mass, and oracle selector headroom, but not Bayes-fused grounding accuracy. Visual weights 12 and 1.5937 both retain the visual channel on every disagreement when no row-level confidence exists. A confirmatory fusion study must define a common candidate space and calibrated per-candidate likelihoods.

## Scoping decision

Direction: `{decision['direction']}`.

{decision['rationale']}

This direction is a design recommendation only. It does not authorize a paper result or modify CEIL/SPLIT/MASK/TRIVUS/VUS-SR.
"""
    REPORT_PATH.write_text(report)
    print(json.dumps({"outcome": outcome, "direction": decision["direction"]}, indent=2))


if __name__ == "__main__":
    main()