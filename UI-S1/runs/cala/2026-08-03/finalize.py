import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    if not path.exists(): raise FileNotFoundError(path)
    return json.loads(path.read_text())


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(value):
    return f"{100 * value:.2f}%"


def pp(value):
    return f"{100 * value:+.2f} pp"


def ci(record):
    return f"[{100 * record['ci_99'][0]:+.2f}, {100 * record['ci_99'][1]:+.2f}]"


def comparison_row(label, record, status):
    return f"| {label} | {pct(record['left_accuracy'])} | {pct(record['right_accuracy'])} | {pp(record['point_delta'])} | {ci(record)} | {record['p_one_sided_delta_le_zero']:.4g} | {status} |"


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path, default=Path(__file__).resolve().parent); args = parser.parse_args()
    run = args.run_dir
    paths = {"static": run/"cala_static_results.json", "adaptive": run/"cala_adaptive_results.json", "transfer": run/"cala_transfer_72b_results.json"}
    static, adaptive, transfer = (load(paths[key]) for key in ("static", "adaptive", "transfer"))
    if any(value["status"] != "PASS" for value in (static, adaptive, transfer)):
        raise ValueError("CALA result package incomplete")
    s12 = static["comparisons"]["CALA_S_N12_B3_mvp_vs_Uniform"]
    a12s = adaptive["comparisons"]["CALA_A_N12_B3_mvp_vs_CALA_S"]
    a8 = adaptive["comparisons"]["CALA_A_N8_B3_mvp_vs_Uniform_Mixed"]
    s72 = transfer["comparisons"]["CALA_S_N8_B3_mvp_vs_Uniform_Mixed_N8"]
    a72 = transfer["comparisons"]["CALA_A_N8_B3_mvp_vs_Uniform_Mixed_N8"]
    table = [
        "# CALA Main Table", "",
        "All learned policies are cross-fitted by application group. Every row within a comparison uses the same number of scored model-view or model-region forwards. B3 is unchanged.", "",
        "## Preregistered adjudication", "",
        "| Comparison | CALA | Baseline | Delta | 99% CI (pp) | One-sided p | Adjudication |",
        "|---|---:|---:|---:|---:|---:|---|",
        comparison_row("7B CALA-S N12 vs Uniform N12, B3", s12, "Primary FAIL"),
        comparison_row("7B CALA-A N12 vs CALA-S N12, B3", a12s, "Adaptive primary FAIL"),
        comparison_row("7B CALA-A N8 vs Uniform N8, B3", a8, "Preregistered secondary PASS"),
        comparison_row("72B CALA-S N8 vs Uniform N8, B3", s72, "Equal-budget transfer PASS"),
        comparison_row("72B CALA-A N8 vs Uniform N8, B3", a72, "Equal-budget transfer PASS"),
        "", "## Accuracy by budget", "",
        "| Scale | Policy | Budget | B3 | M1 | pass@N |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for policy in ("V_only", "Uniform_Mixed", "Quality_Only", "CALA_S"):
        for budget in (4, 8, 12, 16):
            value = static["accuracy"][policy][str(budget)]
            table.append(f"| 7B | {policy} | {budget} | {pct(value['B3_mvp'])} | {pct(value['M1_ccm'])} | {pct(value['pass_at_n'])} |")
    for budget in (8, 12, 16):
        value = adaptive["accuracy"][str(budget)]
        table.append(f"| 7B | CALA_A | {budget} | {pct(value['B3_mvp'])} | {pct(value['M1_ccm'])} | {pct(value['pass_at_n'])} |")
    for policy in ("GTA1_N8", "Uniform_Mixed_N8", "CALA_S_N8", "CALA_A_N8"):
        value = transfer["accuracy"][policy]
        table.append(f"| 72B | {policy} | 8 | {pct(value['B3_mvp'])} | {pct(value['M1_ccm'])} | {pct(value['pass_at_n'])} |")
    table.extend(["", "The 72B values are local equal-budget transfer results, not absolute SOTA results. The completed Scale-Up experiment remains below the paper-only 70.4/73.1 references.", ""])
    (run/"MAIN_TABLE.md").write_text("\n".join(table))
    report = f"""# CALA Method Report

Date: 2026-08-03

## Method

CALA treats a model-lineage/view pair as a budgeted action. CALA-S greedily maximizes development-only marginal candidate coverage. CALA-A spends six fixed scout forwards, then routes additional actions using a cross-fitted logistic predictor of novel-correct probability. Held-out routing can see only proposal metadata and predictions from actions already executed. Unchanged B3 is primary.

This is more specific than model ensembling: CALA is a fixed-budget action scheduler over model lineage and shared proposal geometry.

## Primary result

The preregistered CALA-S N12 primary failed. B3 changes from {pct(s12['right_accuracy'])} to {pct(s12['left_accuracy'])}: {pp(s12['point_delta'])}, 99% CI {ci(s12)} pp. CALA-S raises pass@12 from {pct(static['accuracy']['Uniform_Mixed']['12']['pass_at_n'])} to {pct(static['accuracy']['CALA_S']['12']['pass_at_n'])}, but B3 cannot realize the extra oracle headroom.

The preregistered CALA-A N12-over-CALA-S adaptive success criterion also failed: {pp(a12s['point_delta'])}, 99% CI {ci(a12s)} pp. Against Uniform N12, CALA-A is {pp(adaptive['comparisons']['CALA_A_N12_B3_mvp_vs_Uniform_Mixed']['point_delta'])}.

## Budget-specific positive result

At 7B N8, CALA-A improves unchanged B3 from {pct(a8['right_accuracy'])} to {pct(a8['left_accuracy'])}: {pp(a8['point_delta'])}, 99% CI {ci(a8)} pp, p={a8['p_one_sided_delta_le_zero']:.4g}. pass@8 changes by {pp(adaptive['comparisons']['CALA_A_N8_pass_at_n_vs_Uniform_Mixed']['point_delta'])} with a CI crossing zero.

The gain is budget-specific. Continuing the same router to N12/N16 does not improve Uniform Mixed. The method therefore supports adaptive early allocation and stopping, not monotonic gains from adding routed actions.

## 72B equal-budget transfer

All 72B policies use exactly eight scored model-region forwards. CALA-S improves B3 from {pct(s72['right_accuracy'])} to {pct(s72['left_accuracy'])}: {pp(s72['point_delta'])}, 99% CI {ci(s72)} pp. Its M1 delta is {pp(transfer['comparisons']['CALA_S_N8_M1_ccm_vs_Uniform_Mixed_N8']['point_delta'])}, and pass@8 delta is {pp(transfer['comparisons']['CALA_S_N8_pass_at_n_vs_Uniform_Mixed_N8']['point_delta'])}; both 99% CI lower bounds are positive.

CALA-A also improves 72B B3 over Uniform N8 by {pp(a72['point_delta'])}, 99% CI {ci(a72)} pp, but is below CALA-S by {pp(transfer['comparisons']['CALA_A_N8_B3_mvp_vs_CALA_S_N8']['point_delta'])} with a CI crossing zero.

This transfer validates the allocation algorithm direction at equal budget, but absolute 72B accuracy remains low and does not rescue the failed Scale-Up SOTA target.

## Contribution

The defensible method contribution is:

> A cross-fitted, complementarity-aware scheduler over model-lineage and shared-view actions improves unchanged GUI grounding aggregation at a fixed low inference budget, with significant N8 gains at both 7B and 72B scales.

Supporting methodological findings:

- candidate coverage is submodular and easy to increase, but coverage-only allocation can hurt the downstream aggregator;
- instance-adaptive routing is beneficial at the first two top-up decisions on 7B;
- static development complementarity transfers strongly at 72B;
- neither policy is universally better across budgets, so the result is an allocation-and-stopping method rather than a generic ensemble law.

## Boundaries

- The N12 primary method claim failed and remains visible.
- No post-result hyperparameter search was performed.
- B3 was not retuned.
- 72B N8 is an equal-forward transfer, separate from the failed N12 absolute-SOTA experiment.
- CALA does not claim that multi-model ensembling itself is novel.
"""
    (run/"REPORT.md").write_text(report)
    status = {
        "schema_version": 1, "date": "2026-08-03", "status": "COMPLETE",
        "primary_CALA_S_N12": "FAIL", "adaptive_CALA_A_N12": "FAIL",
        "secondary_CALA_A_7B_N8": "PASS", "transfer_CALA_S_72B_N8": "PASS", "transfer_CALA_A_72B_N8": "PASS",
        "claim_scope": "fixed_budget_N8_complementarity_aware_lineage_view_allocation",
        "artifacts": {key: {"path": path.name, "sha256": sha256_file(path)} for key, path in paths.items()},
    }
    (run/"STATUS.json").write_text(json.dumps(status, indent=2, sort_keys=True)+"\n")
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__": main()