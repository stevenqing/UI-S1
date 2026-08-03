import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    if not path.exists(): raise FileNotFoundError(path)
    return json.loads(path.read_text())


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(value): return f"{100*value:.2f}%"
def pp(value): return f"{100*value:+.2f} pp"
def ci(record): return f"[{100*record['ci_99'][0]:+.2f}, {100*record['ci_99'][1]:+.2f}]"


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--run-dir",type=Path,default=Path(__file__).resolve().parent); args=parser.parse_args(); run=args.run_dir
    paths={name:run/f"{name}.json" for name in ("n1_collapse","n2_upper_bound","n3_72b_repair","n4_noa","n5_stopping_gate")}
    n1,n2,n3,n4,n5=(load(paths[name]) for name in paths)
    if n1["status"]!="PASS" or n2["status"]!="BLOCKED_N1_COLLAPSE" or n3["status"]!="PASS_NO_GLOBAL_COORDINATE_BUG" or n4["status"]!="PASS" or n5["status"]!="PASS": raise ValueError("Neff result package incomplete")
    fits=n1["primary_panel"]["fits"]; best_one_name,best_one=min(fits.items(),key=lambda item:item[1]["N_eff"]["residual_sd"]); best_two_name,best_two=min(fits.items(),key=lambda item:item[1]["two_factor"]["residual_sd"])
    noa=n4["comparisons"]["NOA_static_N12_B3_mvp_vs_Uniform"]; stop=n4["comparisons"]["NOA_stop_B3_mvp_vs_Uniform_N12"]
    table=[
        "# Effective-Sample-Size Evidence Table","",
        "## New adjudications","",
        "| Test | Result | Criterion | Status |","|---|---|---|---|",
        f"| N1 one-factor collapse | best {best_one_name} residual SD {100*best_one['N_eff']['residual_sd']:.2f} pp; R2 {best_one['N_eff']['r_squared']:.3f} | residual SD <= 1.40 pp and better than K | FAIL |",
        f"| N1 two-factor diagnostic | best {best_two_name} residual SD {100*best_two['two_factor']['residual_sd']:.2f} pp; adjusted R2 {best_two['two_factor']['adjusted_r_squared']:.3f} | explanatory only | INSUFFICIENT |",
        "| N2 single-model upper bound | N1 did not pass | requires N1 collapse | BLOCKED |",
        "| N3 72B coordinates | all three bare anchors within 2 pp; two models have zero out-of-image points | no repair if anchors pass | PASS_NO_BUG |",
        f"| N4 NOA-static N12 B3 | {pct(noa['left_accuracy'])} vs {pct(noa['right_accuracy'])}; {pp(noa['point_delta'])}; CI {ci(noa)} | point estimate not lower | FAIL |",
        f"| N4 NOA-stop | {pct(stop['left_accuracy'])} at mean {n4['NOA_stop_compute']['mean_forwards']:.2f} forwards vs {pct(stop['right_accuracy'])} N12 | <=8 forwards and within 0.70 pp | FAIL |",
        f"| N5 stopping gate | high-disagreement pass@4 {pct(n5['bins'][4]['pass_at_n']['4'])} to pass@12 {pct(n5['bins'][4]['pass_at_n']['12'])}; {pp(n5['highest_disagreement_increment']['point_delta'])}; CI {ci(n5['highest_disagreement_increment'])} | positive point increment | PASS |",
        "","## Preserved upstream evidence","",
        "| Evidence | Value | Role after N1 |","|---|---|---|",
        "| H3 equal-compute mixed pool | M1 60.40% to 63.82%; +3.42 pp, 99% CI [+1.41,+5.67] | primary empirical allocation result |",
        "| H2 collision floor | view 0.895; cross-family 0.398; same-family scale 0.618 | external correlation evidence, not universal ScreenSpot rho |",
        "| ScreenSpot pool rho | V-only N12 failure-kappa 0.689; Uniform N12 0.594 | pool-specific correlation measurement |",
        "| X3 budget slopes | V-only -0.002467; Mixed +0.003052 with separated 99% CIs | robust sign-flip result |",
        "| L4 proposal quality | full-bbox containment 99.94% at rank 0 to 61.04% at rank 11 | quality-decay factor |",
        "| CALA-S N12 | pass@12 80.01%, B3 62.18% vs Uniform 63.69% | coverage/final-accuracy separation |",
        "| CALA 72B N8 | CALA-S B3 45.41% vs Uniform 41.24% | equal-budget transfer, not absolute SOTA |",
        "| X7 SafeGround | correctness AUROC 0.628 / 0.744 / 0.830 | disagreement remains useful |",
        "| X6 ranking | held-out Spearman 0.903 | unlabeled pool ranking evidence |",""
    ]; (run/"MAIN_TABLE.md").write_text("\n".join(table))
    report=f"""# Effective-Sample-Size Law Report

Date: 2026-08-03

## Executive result

The strong Effective-Sample-Size Law is **not established**. None of the three preregistered rho estimators collapses B3 pool accuracy to the required 1.40 pp residual scale. The best one-factor result uses `{best_one_name}` with residual SD {100*best_one['N_eff']['residual_sd']:.2f} pp and R-squared {best_one['N_eff']['r_squared']:.3f}. Raw K is worse, but N_eff remains far from sufficient.

Adding proposal quality improves adjusted R-squared to {best_two['two_factor']['adjusted_r_squared']:.3f} for `{best_two_name}`, yet residual SD remains {100*best_two['two_factor']['residual_sd']:.2f} pp. The framework therefore survives only as a qualitative two-factor explanation: correlation and candidate quality both matter, but they do not define a universal accuracy law.

## Why the strong law fails

Pool-specific rho does move in the expected direction. At 7B N12, V-only failure-kappa is {n1['pools']['7B/V_only/N12']['rho']['failure_kappa']:.3f} and Uniform Mixed is {n1['pools']['7B/Uniform_Mixed/N12']['rho']['failure_kappa']:.3f}; corresponding equicorrelation N_eff values are {n1['pools']['7B/V_only/N12']['N_eff']['failure_kappa']:.2f} and {n1['pools']['7B/Uniform_Mixed/N12']['N_eff']['failure_kappa']:.2f}. But CALA-S N12 reduces rho further and raises N_eff to {n1['pools']['7B/CALA_S/N12']['N_eff']['failure_kappa']:.2f} while B3 falls. Correlation reduction can buy oracle coverage without making an unchanged mode-like aggregator choose the correct cluster.

The external H2 view-axis value 0.895 is not the ScreenSpot pool rho: ScreenSpot V-only N12 measures 0.689. It remains evidence that repeated views are highly correlated across prior tasks, not a constant to substitute into every pool.

## N2 upper bound

N2 is `{n2['status']}`. The failure-kappa fit would diagnostically extrapolate {pct(n2['diagnostic_extrapolations']['failure_kappa']['predicted_accuracy'])} at 1/0.895, but its residual SD is 7.30 pp and the fit slope is negative. No impossibility upper-bound claim is made.

## 72B audit

N3 rejects the proposed global coordinate-bug diagnosis. Local full-image scores are GTA1 {pct(n3['models']['GTA1-72B']['full_image_accuracy'])}, UI-Venus {pct(n3['models']['UI-Venus-Ground-72B']['full_image_accuracy'])}, and Qwen3.5 {pct(n3['models']['Qwen3.5-122B-A10B']['full_image_accuracy'])}; all pass their paper-anchor tolerance. Existing 72B traces are retained without parser changes. Their low B3 is an aggregation/candidate-pollution boundary.

## NOA

NOA-static does not repair CALA-S. At N12, B3 is {pct(noa['left_accuracy'])} versus Uniform Mixed {pct(noa['right_accuracy'])}: {pp(noa['point_delta'])}, 99% CI {ci(noa)} pp.

NOA-stop uses an average {n4['NOA_stop_compute']['mean_forwards']:.2f} forwards (median {n4['NOA_stop_compute']['median_forwards']:.0f}) but reaches {pct(stop['left_accuracy'])}, {pp(stop['point_delta'])} below Uniform N12 with 99% CI {ci(stop)} pp. It saves compute but fails the frozen equal-accuracy tolerance, so no efficiency success is claimed.

N5 confirms that stopping was not doomed by absent headroom: in the highest SafeGround-disagreement quintile, pass@N rises from {pct(n5['bins'][4]['pass_at_n']['4'])} at N4 to {pct(n5['bins'][4]['pass_at_n']['12'])} at N12, {pp(n5['highest_disagreement_increment']['point_delta'])}, 99% CI {ci(n5['highest_disagreement_increment'])} pp. The failure lies in allocation/stopping realization, not a flat rescue curve.

## Consolidated contribution

The defensible paper contribution remains empirical and diagnostic:

1. Under equal 12-forward compute, cross-lineage allocation improves 7B grounding by 3.42 pp with a positive 99% CI.
2. Pool error correlation and proposal quality explain why repeated-view scaling saturates or reverses direction, but not through a universal one-dimensional N_eff curve.
3. Candidate union coverage is not final accuracy: both CALA-S and NOA can improve headroom or N_eff while hurting B3.
4. Low-budget CALA N8 gains transfer across 7B and 72B, but neither the 72B absolute-SOTA lane nor the generalized NOA objective succeeds.

## Execution boundary

N6 is not run because NOA-static underperforms Uniform Mixed at N12. Existing collision-law, allocation-law, diversity-axis, CALA and Scale-Up artifacts remain unchanged. Paper-only 62.8, 70.4 and 73.1 are context only.
"""; (run/"REPORT.md").write_text(report)
    status={"schema_version":1,"date":"2026-08-03","status":"COMPLETE","strong_Neff_law":"FAIL","two_factor_explanation":"QUALITATIVE_ONLY","N2_upper_bound":"BLOCKED","N3_72B":"PASS_NO_GLOBAL_COORDINATE_BUG","N4_NOA_static":"FAIL","N4_NOA_stop":"FAIL","N5_stopping_gate":"PASS","N6_action_space_extension":"NOT_RUN_N4_GATE_FAILED","paper_scope":"equal_compute_allocation_result_plus_qualitative_correlation_quality_mechanism","artifacts":{name:{"path":path.name,"sha256":sha256_file(path)} for name,path in paths.items()},"figure":{"path":"fig2_neff_collapse.pdf","sha256":sha256_file(run/"fig2_neff_collapse.pdf")}}
    (run/"STATUS.json").write_text(json.dumps(status,indent=2,sort_keys=True)+"\n"); print(json.dumps(status,indent=2,sort_keys=True))


if __name__=="__main__": main()