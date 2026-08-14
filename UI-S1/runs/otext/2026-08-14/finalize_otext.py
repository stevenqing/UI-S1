import hashlib
import json
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
STAGE0_PATH = RUN_DIR / "STAGE0.json"
STAGE1_PATH = RUN_DIR / "STAGE1.json"
OUTPUT_PATH = RUN_DIR / "OTEXT_ADJUDICATION.json"
REPORT_PATH = RUN_DIR / "REPORT.md"
TABLE_PATH = RUN_DIR / "MAIN_TABLE.md"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ci(report):
    return f"[{100 * report['ci_99'][0]:+.2f}, {100 * report['ci_99'][1]:+.2f}] pp"


def main():
    if OUTPUT_PATH.exists() or REPORT_PATH.exists() or TABLE_PATH.exists():
        raise FileExistsError("OTEXT final outputs exist")
    stage0 = json.loads(STAGE0_PATH.read_text())
    if stage0.get("status") != "PASS_OTEXT_STAGE0_COMPLETE":
        raise PermissionError("OTEXT Stage0 incomplete")
    stage1 = json.loads(STAGE1_PATH.read_text()) if STAGE1_PATH.exists() else None
    if stage0["proceed_stage1"] and (stage1 is None or stage1.get("status") != "PASS_OTEXT_STAGE1_COMPLETE"):
        raise PermissionError("OTEXT Stage1 required but incomplete")
    if not stage0["proceed_stage1"] and stage1 is not None:
        raise PermissionError("OTEXT Stage1 exists despite O-K1")
    if not stage0["proceed_stage1"]:
        outcome = "OTEXT_STOPPED_O_K1_STAGE0"
        endpoints = {name: "NOT_RUN_O_K1" for name in ("O_P1", "O_P2", "O_P3", "O_P4", "O_P5")}
        kill = {"O_K1": True, "O_K7": stage0["kill_conditions"]["O_K7"]}
    else:
        endpoints = stage1["endpoints"]
        kill = {"O_K1": False, "O_K7": stage0["kill_conditions"]["O_K7"], **stage1["kill_conditions"]}
        outcome = "OTEXT_VALIDATION_O_P1_PASS" if endpoints["O_P1"] else "OTEXT_VALIDATION_O_P1_FAIL"
    adjudication = {
        "schema_version": 1, "status": "COMPLETE",
        "outcome": outcome, "evidence_status": "POST_SELECTION_VALIDATION",
        "confirmatory_claim_allowed": False,
        "stage0_O_G1": stage0["O_G1"], "stage1_run": stage1 is not None,
        "endpoints": endpoints, "kill_conditions": kill,
        "changes_existing_statuses": False, "paper_method_claim_allowed": False,
        "input_hashes": {"STAGE0.json": sha256_file(STAGE0_PATH), **({"STAGE1.json": sha256_file(STAGE1_PATH)} if stage1 else {})},
        "next_action": "REQUIRE_NEW_UNTOUCHED_DATA_FOR_CONFIRMATORY_OCR" if stage1 and endpoints["O_P1"] else "CLOSE_OR_REDESIGN_WITH_NEW_SPEC_NO_RETRY",
    }
    OUTPUT_PATH.write_text(json.dumps(adjudication, indent=2, sort_keys=True) + "\n")
    lines = ["# OTEXT Main Table", "", "## Stage 0", "", "| Engine | Gain vs majority | Gain vs dev-selection | Minimum | O-G1 |", "| --- | ---: | ---: | ---: | --- |"]
    for engine in ("easyocr", "rapidocr"):
        value = stage0["O_G1"][engine]
        lines.append(f"| {engine} | {100 * value['gains']['majority']:+.2f} pp | {100 * value['gains']['dev_selection']:+.2f} pp | {100 * value['objective']:+.2f} pp | {value['pass_O_G1']} |")
    if stage1:
        lines.extend(["", "## Stage 1 EasyOCR", "", "| Comparison | Delta | 99% CI |", "| --- | ---: | --- |"])
        for name, value in stage1["reports"]["easyocr"]["comparisons"].items():
            lines.append(f"| {name} | {100 * value['point']:+.2f} pp | {ci(value)} |")
    TABLE_PATH.write_text("\n".join(lines) + "\n")
    stage1_text = "Stage 1 was not run because EasyOCR failed O-G1." if stage1 is None else f"Stage 1 O-P1 was {'passed' if endpoints['O_P1'] else 'not passed'} under the dual-baseline rule. Detailed comparisons are in `MAIN_TABLE.md`."
    report = rf"""# OTEXT OCR Validation Report

Date: 2026-08-14

Status: `{outcome}`

## Evidence status

OTEXT is **post-selection validation**, not confirmatory evidence. ORTH used all 1,581 ScreenSpot-Pro labels to select the OCR/text direction. OTEXT preregisters and nests tuning, regenerates both OCR engines, and uses held-out folds, but a paper method claim still requires new untouched data.

## Stage 0

EasyOCR is the sole primary engine. Its weighted nested validation minimum gain across majority and nested dev-selection is {100 * stage0['O_G1']['easyocr']['objective']:+.2f} pp against the 0.70 pp gate. RapidOCR is replication only. Selected parameters and full inner-validation curves are retained in `SELECTED_PARAMETERS.json` and `STAGE0.json`.

{stage1_text}

## Boundaries

No `ui_type`, row class, GT overlap, or label-dependent statistic enters the runtime gate. Text/icon, gate-conditional accuracy, and conditional correctness remain evaluation-side. No existing project status changes, and failed settings cannot be rescued by retuning inside this round.
"""
    REPORT_PATH.write_text(report)
    print(json.dumps({"outcome": outcome, "stage1_run": stage1 is not None, "endpoints": endpoints}, indent=2))


if __name__ == "__main__":
    main()