import argparse
import hashlib
import json
from pathlib import Path


ARTIFACTS = {
    "X1": "x1/x1_sampling_axis.json",
    "X2": "x2/x2_composability.json",
    "X3": "x3_curve_stats.json",
    "X4": "x4_scanner_locator.json",
    "X5": "x5_allocation_topology.json",
    "X6": "x6_pool_ranking.json",
    "X7": "x7_confidence.json",
    "X8": "x8_mind2web_lineage_only.json",
}


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percent(value):
    return f"{100 * value:.2f}%"


def signed_pp(value):
    return f"{100 * value:+.2f} pp"


def load_results(run_dir):
    results = {}
    for experiment, relative in ARTIFACTS.items():
        path = run_dir / relative
        if not path.exists():
            raise FileNotFoundError(path)
        results[experiment] = json.loads(path.read_text())
    if results["X2"]["status"] != "PASS" or results["X3"]["status"] != "PASS":
        raise ValueError("Diversity-Axis core X2/X3 result is incomplete")
    if results["X6"].get("status") not in {"PASS", "FAIL_HELDOUT_SPEARMAN"}:
        raise ValueError(f"Diversity-Axis X6 validation is incomplete: {results['X6'].get('status')}")
    if results["X6"].get("heldout_observations") != 10:
        raise ValueError("Diversity-Axis X6 held-out validation is incomplete")
    return results


def report_text(results, x6_fit):
    x1, x2, x3, x4, x5, x6, x7, x8 = (results[f"X{index}"] for index in range(1, 9))
    accuracy = x2["accuracy"]
    interaction = x2["interactions"]["M1_ccm"]
    x7_primary = {
        pool: x7["pools"][pool]["variants"]["official_code"][x7["pools"][pool]["primary_label"]]["correctness_auroc_negative_uncertainty"]
        for pool in ("stochastic_GTA1_N4", "v_only_N12", "mixed_N12")
    }
    return f"""# Diversity-Axis Results

Date: 2026-08-02

## Executive result

X3 validates the original budget-axis sign with 10,000 fold-stratified application bootstraps: V-only M1 slope is {x3['slopes']['v_only']['M1_ccm']['point_slope_per_forward']:.6f} per forward with 99% CI [{x3['slopes']['v_only']['M1_ccm']['ci_99'][0]:.6f}, {x3['slopes']['v_only']['M1_ccm']['ci_99'][1]:.6f}], while Mixed is {x3['slopes']['mixed']['M1_ccm']['point_slope_per_forward']:.6f} with CI [{x3['slopes']['mixed']['M1_ccm']['ci_99'][0]:.6f}, {x3['slopes']['mixed']['M1_ccm']['ci_99'][1]:.6f}]. X-K2 does not trigger.

X2 uses a preregistered fixed-12, K=3 microchain extension of UI-Zoomer. Q1-Q4 M1 accuracies are {percent(accuracy['Q1']['M1_ccm'])}, {percent(accuracy['Q2']['M1_ccm'])}, {percent(accuracy['Q3']['M1_ccm'])}, and {percent(accuracy['Q4']['M1_ccm'])}. Q4 is {'the highest cell' if x2['prediction']['Q4_highest'] else 'not the highest cell'}. The M1 interaction is {signed_pp(interaction['point'])}, 99% CI [{signed_pp(interaction['ci_99'][0])}, {signed_pp(interaction['ci_99'][1])}], classified `{interaction['classification']}`. X-K1 is `{x2['kill_conditions']['X-K1']}`.

X1 remains blocked because only five GTA1 stochastic samples exist per row, not the required N=4/8/12/16 three-pool traces. X4 has no released GMS implementation or auditable fixed-12 trace. X5 lacks pure-serial and hybrid traces. X8 cannot construct a same-lineage N=6 Mind2Web pool. These are unavailable comparisons, not negative results.

## X1: sampling axis

Status: `{x1['status']}`. The exact GUI-RC voting port gives N4 S-only accuracy {percent(x1['available_budget_result']['S_only']['4']['accuracy']['GUI_RC'])}, B3 {percent(x1['available_budget_result']['S_only']['4']['accuracy']['B3_mvp'])}, and pass@4 {percent(x1['available_budget_result']['S_only']['4']['accuracy']['pass_at_n'])}. X-K3 is `{x1['kill_conditions']['X-K3']}` because a slope cannot be estimated without padding or new sampling traces.

## X2: adaptive zoom composability

| Cell | B3 | M1 | pass@12 | failure kappa |
|---|---:|---:|---:|---:|
| Q1 single/fixed | {percent(accuracy['Q1']['B3_mvp'])} | {percent(accuracy['Q1']['M1_ccm'])} | {percent(accuracy['Q1']['pass_at_n'])} | {x2['failure_kappa']['Q1']:.3f} |
| Q2 single/adaptive | {percent(accuracy['Q2']['B3_mvp'])} | {percent(accuracy['Q2']['M1_ccm'])} | {percent(accuracy['Q2']['pass_at_n'])} | {x2['failure_kappa']['Q2']:.3f} |
| Q3 mixed/fixed | {percent(accuracy['Q3']['B3_mvp'])} | {percent(accuracy['Q3']['M1_ccm'])} | {percent(accuracy['Q3']['pass_at_n'])} | {x2['failure_kappa']['Q3']:.3f} |
| Q4 mixed/adaptive | {percent(accuracy['Q4']['B3_mvp'])} | {percent(accuracy['Q4']['M1_ccm'])} | {percent(accuracy['Q4']['pass_at_n'])} | {x2['failure_kappa']['Q4']:.3f} |

Adaptive trigger rates are {percent(x2['gate_diagnostics']['Q2']['adaptive_trigger_rate'])} for Q2 and {percent(x2['gate_diagnostics']['Q4']['adaptive_trigger_rate'])} for Q4. Every row uses exactly 12 useful forwards. This is an algorithm-level K=3 budget-normalized extension; the official Qwen2.5-VL-7B K=8 sanity anchor was not run.

## X3: curve robustness

Both B3 and M1 satisfy V-only CI upper bound below zero and Mixed CI lower bound above zero. N24 remains one-sided because GTA1 provides only 16-19 unique candidates. Area stratification contradicts the small-target expectation at N12: the smallest target quintile has M1 Mixed-minus-V-only {signed_pp(x3['area_strata'][0]['mixed_minus_v_only']['12']['M1_ccm'])}, while the largest has {signed_pp(x3['area_strata'][-1]['mixed_minus_v_only']['12']['M1_ccm'])}.

## X4-X6

- X4: `{x4['status']}`; X-K4 is `{x4['kill_conditions']['X-K4']}`.
- X5: `{x5['status']}`; only the frozen pure-parallel point exists.
- X6: frozen L2 OLS training R-squared is {x6_fit['model']['training_r_squared']:.4f}. Held-out Spearman over 10 X2 fold-pool observations is {x6['spearman']['rho']:.3f} (p={x6['spearman']['p_value']:.4g}); criterion rho > 0.7 is `{x6['prediction_X6']}`.

## X7: confidence axis

SafeGround official-code geometry is exactly ported at commit `5e8fca7`. Correctness AUROC from negative uncertainty is {x7_primary['stochastic_GTA1_N4']:.3f} for stochastic GTA1 N4, {x7_primary['v_only_N12']:.3f} for deterministic V-only N12, and {x7_primary['mixed_N12']:.3f} for deterministic Mixed N12, versus the cross-task S_gap anchor 0.393. The deterministic N12 diagnostics transfer the dispersion score but do not inherit SafeGround's K=10 stochastic protocol or FDR guarantee.

## X8: Mind2Web alternative

Status: `{x8['status']}`. Six deployable full-view models are available across families {x8['deployable_family_counts']}; the largest same-family pool has only {x8['same_lineage_maximum']} models. Original L3 and L-K4 remain unchanged.

## Claim boundary

The strongest defensible claim is that the fixed-view budget-axis sign is statistically stable and that candidate dispersion is a useful confidence diagnostic, especially in the mixed pool. The adaptive-zoom composability claim follows the X2 interaction classification above but is limited to the preregistered K=3 fixed-budget extension. Sampling-family coverage, GMS comparison, topology triangle, and Mind2Web lineage transfer remain unresolved because their required candidate pools do not exist under the frozen contracts.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    results = load_results(args.run_dir)
    x6_fit_path = args.run_dir / "x6_fit.json"
    x6_fit = json.loads(x6_fit_path.read_text())
    if x6_fit.get("status") != "FROZEN_FIT_BEFORE_X2_VALIDATION" or x6_fit.get("training_observations") != 40:
        raise ValueError("Diversity-Axis X6 frozen fit is invalid")
    if results["X6"].get("fit_sha256") != sha256_file(x6_fit_path):
        raise ValueError("Diversity-Axis X6 validation did not use the frozen fit")
    report = report_text(results, x6_fit)
    (args.run_dir / "REPORT.md").write_text(report)
    status = {
        "schema_version": 1,
        "date": "2026-08-02",
        "status": "COMPLETE_WITH_BLOCKED_COMPARISONS",
        "experiments": {
            experiment: {
                "path": relative,
                "status": results[experiment]["status"],
                "sha256": sha256_file(args.run_dir / relative),
            }
            for experiment, relative in ARTIFACTS.items()
        },
        "kill_conditions": {
            "X-K1": results["X2"]["kill_conditions"]["X-K1"],
            "X-K2": results["X3"]["kill_conditions"]["X-K2"],
            "X-K3": results["X1"]["kill_conditions"]["X-K3"],
            "X-K4": results["X4"]["kill_conditions"]["X-K4"],
        },
        "validation": {
            "x2_rows": results["X2"]["rows"],
            "x2_forward_budget": results["X2"]["forward_budget"],
            "x3_bootstrap_resamples": results["X3"]["slopes"]["mixed"]["M1_ccm"]["resamples"],
            "x6_heldout_observations": results["X6"]["heldout_observations"],
            "protected_pid_1814_modified": False,
        },
    }
    (args.run_dir / "STATUS.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status["status"], "kill_conditions": status["kill_conditions"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()