import hashlib
import json
import os
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
ARM_A_PATH = RUN_DIR / "ARM_A.json"
ARM_B_PATH = RUN_DIR / "ARM_B.json"
COMMON_PATH = RUN_DIR / "COMMON_CALIBRATION.json"
REPORT_PATH = RUN_DIR / "REPORT.md"
ADJUDICATION_PATH = RUN_DIR / "OWIN_ADJUDICATION.json"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pp(value):
    return f"{100 * value:+.3f} pp"


def pct(value):
    return f"{100 * value:.2f}%"


def interval(value):
    return "NA" if value is None else f"[{100 * value[0]:+.3f}, {100 * value[1]:+.3f}] pp"


def atomic_text(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    if REPORT_PATH.exists() or ADJUDICATION_PATH.exists():
        raise FileExistsError("OWIN final output exists")
    arm_a = json.loads(ARM_A_PATH.read_text())
    arm_b = json.loads(ARM_B_PATH.read_text())
    common = json.loads(COMMON_PATH.read_text())
    if arm_a["status"] != "PASS_OWIN_ARM_A_COMPLETE" or arm_a["gpu"]["formal_calls"] != 6000 or not arm_a["common_frozen_reproduction"]["pass"]:
        raise ValueError("OWIN Arm A finalization mismatch")
    if arm_a["O_I"]["classification"] not in {"O_I1", "O_I2", "O_I3"} or arm_b["status"] != "PASS_OWIN_ARM_B_COMPLETE_ZERO_GPU":
        raise ValueError("OWIN adjudication mismatch")
    dependence_text = "Residual pool-dependence comparability could not be quantified for the affected stratum(s)." if arm_a["dependence_limitation_required"] else "Matched dependence diagnostics were quantifiable in every stratum/fold unit."
    lines = [
        "# OWIN Oracle Coverage Measurement Report",
        "",
        "Date: 2026-08-17",
        "",
        f"Outcome: `OWIN_{arm_a['O_I']['classification']}_GT_ORACLE_NON_DEPLOYABLE`",
        "",
        "OWIN is a post-selection, single-benchmark measurement, not a method. Every Arm A value below is `GT_ORACLE_NON_DEPLOYABLE`: GT bbox geometry constructs the windows and no runtime placement rule is implied.",
        "",
        "## Arm A: GT_ORACLE_NON_DEPLOYABLE pool measurement",
        "",
        "| Stratum | Existing B3 | Raw oracle-pool B3 | Corrected oracle-pool B3 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for stratum in ("uncovered_0", "partial_1_10", "common_11"):
        lines.append(f"| {stratum} | {pct(arm_a['B3']['existing'][stratum])} | {pct(arm_a['B3']['raw'][stratum])} | {pct(arm_a['B3']['corrected'][stratum])} |")
    lines.extend([
        "",
        f"Corrected B3 perfect-coverage opportunity is **{pp(arm_a['B3']['perfect_gain'])}**, 99% CI {interval(arm_a['bootstrap'].get('B3_perfect_gain', {}).get('ci_99'))}. It maps to **{arm_a['O_I']['classification']}** under the frozen 5/10 pp thresholds.",
        "",
        f"Raw M1_ccm and corrected M1_ccm opportunity are co-reported; corrected gain is {pp(arm_a['M1_ccm']['perfect_gain'])}, 99% CI {interval(arm_a['bootstrap'].get('M1_perfect_gain', {}).get('ci_99'))}. The corrected single-forward opportunity is {pp(arm_a['single_forward']['perfect_gain'])}, CI {interval(arm_a['bootstrap'].get('single_perfect_gain', {}).get('ci_99'))}. Neither drives O-I.",
        "",
        "### Named limitations",
        "",
        f"Constant-shift is not validated. Common small-minus-large calibration heterogeneity is {pp(common['size_heterogeneity']['contrast_small_minus_large'])}, CI {interval(common['size_heterogeneity']['contrast_ci_99'])}, labeled `{common['size_heterogeneity']['label']}`. Raw and corrected values must remain adjacent; small-target sensitivity is reported beside the primary estimate.",
        "",
        dependence_text,
        "",
        "No historical N_eff value substitutes for unavailable matched diagnostics. These limitations apply beside every pool-level oracle opportunity.",
        "",
        "## Arm B: fixed equal-budget geometry",
        "",
        "| N | Median union | Center covered | Full-bbox covered | Factorized G_N | 99% CI |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for count in range(4, 12):
        tiling = arm_b["summaries"][str(count)]["tiling"]
        factorized = arm_a["factorized_G_N"][str(count)]
        lines.append(f"| {count} | {pct(tiling['union_fraction']['median'])} | {tiling['center_covered_rows']}/1581 | {tiling['full_bbox_covered_rows']}/1581 | {pp(factorized['G_N'])} | {interval(factorized['ci_99'])} |")
    lines.extend([
        "",
        f"Frozen saturation status is `N_star={arm_b['N_star']}`. Existing 11-crop median union is 32.19%; fixed tiling reaches the values above without model signals. Factorized G_N is descriptive, not observed deployable gain.",
        "",
        "## Execution and boundaries",
        "",
        "Formal execution retained exactly 6,000 traces with zero final-shard failures. Passing smoke retained 36 traces. Two earlier smoke failures and the first inference-input isolation failure remain retained. Token logprobs, entropy, margins, coordinate spans, decoded output, and hashes follow the extended trace policy.",
        "",
        "OWIN changes no prior result or status. X2 and SPLIT remain closed, M2W is excluded, and any follow-up requires a new GT-free specification plus a net-benefit ledger on original-correct and crop-covered rows.",
    ])
    atomic_text(REPORT_PATH, "\n".join(lines) + "\n")
    adjudication = {"schema_version": 1, "date": "2026-08-17", "round": "owin", "status": "COMPLETE", "outcome": f"OWIN_{arm_a['O_I']['classification']}_GT_ORACLE_NON_DEPLOYABLE", "evidence_status": arm_a["evidence_status"], "method_claim_allowed": False, "deployable_rule_produced": False, "changes_existing_statuses": False, "formal_gpu_calls": 6000, "O_I": arm_a["O_I"], "arm_b_N_star": arm_b["N_star"], "constant_shift_limitation": True, "dependence_limitation": arm_a["dependence_limitation_required"], "report": str(REPORT_PATH.relative_to(ROOT)), "report_sha256": sha256_file(REPORT_PATH), "arm_a_sha256": sha256_file(ARM_A_PATH), "arm_b_sha256": sha256_file(ARM_B_PATH), "common_calibration_sha256": sha256_file(COMMON_PATH), "next_action": "HUMAN_REVIEW_O_I_AND_NO_FURTHER_GPU_WITHOUT_NEW_SPEC"}
    atomic_json(ADJUDICATION_PATH, adjudication)
    print(json.dumps(adjudication, indent=2))


if __name__ == "__main__":
    main()