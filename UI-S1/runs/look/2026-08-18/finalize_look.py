import hashlib
import json
import os
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
RESULT_PATH = RUN_DIR / "LOOK_RESULTS.json"
REPORT_PATH = RUN_DIR / "REPORT.md"
ADJUDICATION_PATH = RUN_DIR / "LOOK_ADJUDICATION.json"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pct(value):
    return f"{100 * value:.2f}%"


def pp(value):
    return f"{100 * value:+.2f} pp"


def ci(values):
    return f"[{values[0]:.3f}, {values[1]:.3f}]"


def atomic_text(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    if REPORT_PATH.exists() or ADJUDICATION_PATH.exists():
        raise FileExistsError("LOOK final output exists")
    result = json.loads(RESULT_PATH.read_text())
    if result["status"] != "PASS_LOOK_DIAGNOSTIC_COMPLETE" or result["sample"] != {"formal_calls": 1290, "pool_correct": 250, "recoverable": 180}:
        raise ValueError("LOOK finalization input mismatch")
    if result["future_method_authorized"] is not False or result["kill_conditions"]["L_K1"] is not True:
        raise ValueError("LOOK adjudication mismatch")
    lines = [
        "# LOOK Candidate-Confrontation Diagnostic Report",
        "",
        "Date: 2026-08-18",
        "",
        "Outcome: `LOOK_L_D3_L_K1_NULL_DOMINATES_NO_METHOD_AUTHORIZED`",
        "",
        "LOOK is a post-selection, single-benchmark diagnostic, not a method. It changes no prior result and authorizes no method experiment.",
        "",
        "## Primary discrimination",
        "",
        "| Endpoint | Point | 99% CI | Interpretation |",
        "| --- | ---: | ---: | --- |",
        f"| L-P1 main candidate AUROC | {result['L_P1']['main_AUROC']:.3f} | {ci(result['L_P1']['ci_99'])} | {result['decision']} |",
        f"| L-P2 main minus M1 correctness | {pp(result['L_P2']['difference'])} | [{100*result['L_P2']['ci_99'][0]:+.2f}, {100*result['L_P2']['ci_99'][1]:+.2f}] pp | descriptive recoverable gain |",
        f"| L-P4 main minus null AUROC | {result['L_P4']['difference']:+.3f} | {ci(result['L_P4']['ci_99'])} | L-K1 |",
        "",
        f"Main AUROC is {result['L_P1']['main_AUROC']:.3f}, close to CEIL's contextual 0.540, and its interval crosses both frozen directional boundaries. LOOK is therefore `{result['decision']}`. Null AUROC is {result['L_P4']['null_AUROC']:.3f}, exceeding main by {-result['L_P4']['difference']:.3f}; the paired interval excludes zero against main. L-K1 cancels candidate-identity signal wording regardless of L-P2.",
        "",
        "## Damage and sensitivity",
        "",
        f"On pool-correct rows, confrontation overturns the B3 mode on {pct(result['L_P3']['overturn_rate'])}; harmful overturn is {pct(result['L_P3']['harmful_overturn_rate'])}. Unmappable rate is {pct(result['L_P3']['unmappable_rate'])}.",
        "",
        f"Three-mode sensitivity AUROC is {result['sensitivity']['recoverable']['AUROC']:.3f} on recoverable rows and {result['sensitivity']['pool_correct']['AUROC']:.3f} on pool-correct rows. It is descriptive and cannot replace the failed primary identity control.",
        "",
        "## Separation-stratified results",
        "",
        "| Separation quartile | Rows | AUROC | Main minus M1 | Positive / negative records |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, value in result["L_P5"]["bins"].items():
        lines.append(f"| {key} | {value['rows']} | {value['AUROC']:.3f} | {pp(value['main_minus_M1'])} | {value['positive_records']} / {value['negative_records']} |")
    lines.extend([
        "",
        f"Frozen separation boundaries are `{result['L_P5']['boundaries']}`. The first quartile is strongest (AUROC {result['L_P5']['bins']['0']['AUROC']:.3f}); later quartiles are near chance. This pattern is descriptive and does not rescue L-K1.",
        "",
        "## Geometry",
        "",
        f"Main-window median area fraction is {pct(result['L_P6']['main_area_fraction']['median'])}; mean is {pct(result['L_P6']['main_area_fraction']['mean'])}. Only {pct(result['L_P6']['main_area_gt_0_8_fraction'])} exceed 80%, so L-K2 is false. Sensitivity-window median is {pct(result['L_P6']['sensitivity_area_fraction']['median'])}. Null area-ratio median is {result['L_P6']['null_area_ratio']['median']:.3f}; median null search attempt is {result['L_P6']['null_attempt']['median']:.1f}.",
        "",
        "## Execution and decision",
        "",
        "Formal execution completed 1,290/1,290 calls with zero failures; all outputs parsed and all token logprobs were retained. Realized samples were 180 recoverable and 250 pool-correct rows, so L-K3 is false.",
        "",
        "Final decision: L-D3 and L-K1. Local confrontation shows a descriptive M1-to-main correctness increase, but the random noncandidate control discriminates substantially better and the pool-correct damage rate is high. No candidate-identity mechanism claim and no follow-up method are authorized.",
    ])
    atomic_text(REPORT_PATH, "\n".join(lines) + "\n")
    adjudication = {"schema_version": 1, "round": "look", "date": "2026-08-18", "status": "COMPLETE", "outcome": "LOOK_L_D3_L_K1_NULL_DOMINATES_NO_METHOD_AUTHORIZED", "evidence_status": result["evidence_status"], "method_claim_allowed": False, "future_method_authorized": False, "changes_existing_statuses": False, "formal_gpu_calls": 1290, "decision": result["decision"], "kill_conditions": result["kill_conditions"], "report": str(REPORT_PATH.relative_to(ROOT)), "report_sha256": sha256_file(REPORT_PATH), "results_sha256": sha256_file(RESULT_PATH), "trace_sha256": result["trace"]["sha256"], "next_action": "CLOSE_LOOK_DIRECTION_NO_METHOD_SPEC"}
    atomic_json(ADJUDICATION_PATH, adjudication)
    print(json.dumps(adjudication, indent=2))


if __name__ == "__main__":
    main()