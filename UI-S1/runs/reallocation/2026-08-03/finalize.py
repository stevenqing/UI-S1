import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    if not path.exists(): raise FileNotFoundError(path)
    return json.loads(path.read_text())


def sha256_file(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def pct(value): return f"{100*value:.2f}%"
def pp(value): return f"{100*value:+.2f} pp"
def ci(record): return f"[{100*record['ci_99'][0]:+.2f}, {100*record['ci_99'][1]:+.2f}]"


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--run-dir",type=Path,default=Path(__file__).resolve().parent); args=parser.parse_args(); run=args.run_dir
    paths={name:run/f"{name}.json" for name in ("r1_stratified_accuracy","r2_reallocation","r3_conditional_proposal","r4_risk_coverage","r5_72b_pollution")}
    r1,r2,r3,r4,r5=(load(paths[name]) for name in paths)
    if r1["status"]!="PASS" or r2["status"]!="CANCELLED_R_K1" or r3["status"]!="CANCELLED_R_K1" or r4["status"]!="PASS" or r5["status"]!="PASS": raise ValueError("reallocation result package incomplete")
    high=r1["bins"][4]; b3=r1["highest_disagreement_N24_minus_N4"]["B3_mvp"]; oracle=r1["highest_disagreement_N24_minus_N4"]["pass_at_n"]
    mixed={value["coverage"]:value for value in r4["curves"]["Uniform_Mixed_N12"]}; vonly={value["coverage"]:value for value in r4["curves"]["V_only_N12"]}; pollution=r5["summaries"]["72B_Uniform_Mixed_N8"]
    table=["# Difficulty-Conditioned Reallocation Main Table","","## R1 stratified realization gate","","| Highest-disagreement metric | N4 | N8 | N12 | N16 | N24 | N24-N4 / 99% CI |","|---|---:|---:|---:|---:|---:|---|",f"| B3 | {pct(high['accuracy']['4']['B3_mvp'])} | {pct(high['accuracy']['8']['B3_mvp'])} | {pct(high['accuracy']['12']['B3_mvp'])} | {pct(high['accuracy']['16']['B3_mvp'])} | {pct(high['accuracy']['24']['B3_mvp'])} | {pp(b3['point_delta'])}; {ci(b3)} |",f"| M1 | {pct(high['accuracy']['4']['M1_ccm'])} | {pct(high['accuracy']['8']['M1_ccm'])} | {pct(high['accuracy']['12']['M1_ccm'])} | {pct(high['accuracy']['16']['M1_ccm'])} | {pct(high['accuracy']['24']['M1_ccm'])} | {pp(r1['highest_disagreement_N24_minus_N4']['M1_ccm']['point_delta'])}; {ci(r1['highest_disagreement_N24_minus_N4']['M1_ccm'])} |",f"| pass@N | {pct(high['accuracy']['4']['pass_at_n'])} | {pct(high['accuracy']['8']['pass_at_n'])} | {pct(high['accuracy']['12']['pass_at_n'])} | {pct(high['accuracy']['16']['pass_at_n'])} | {pct(high['accuracy']['24']['pass_at_n'])} | {pp(oracle['point_delta'])}; {ci(oracle)} |","",f"R1 status: **FAIL / R-K1**. Candidate headroom rises, but B3 realization ratio is {r1['realization_ratio_B3_over_pass']:.3f}.","","## R4 selective accuracy","","| Pool | Retained coverage | Retained B3 | Gain vs full | Random mean | Random 99% CI |","|---|---:|---:|---:|---:|---:|"]
    for name,curve in (("Uniform Mixed N12",mixed),("V-only N12",vonly)):
        for coverage in (.9,.8,.7):
            value=curve[coverage]; table.append(f"| {name} | {100*coverage:.0f}% | {pct(value['retained_accuracy'])} | {pp(value['accuracy_gain_vs_full'])} | {pct(value['random_rejection']['mean_accuracy'])} | {ci(value['random_rejection'])} |")
    table.extend(["","## R5 72B diagnostic","","| Diagnostic | 7B Uniform N8 | 72B Uniform N8 |","|---|---:|---:|",f"| B3 | {pct(r5['summaries']['7B_Uniform_Mixed_N8']['B3_accuracy'])} | {pct(pollution['B3_accuracy'])} |",f"| Mean normalized failed-pair distance | {r5['summaries']['7B_Uniform_Mixed_N8']['mean_failed_pair_distance']:.4f} | {pollution['mean_failed_pair_distance']:.4f} |",f"| Wrong B3 selected model, dominant | GTA1: {r5['summaries']['7B_Uniform_Mixed_N8']['wrong_selected_model']['GTA1-7B']} | GTA1: {pollution['wrong_selected_model']['GTA1-72B']} |","",f"Tight-error pollution hypothesis: **FAIL**. The 72B wrong winner composition is highly nonuniform, but failed candidates are more dispersed, not tighter.",""])
    (run/"MAIN_TABLE.md").write_text("\n".join(table))
    report=f"""# Difficulty-Conditioned Reallocation Report

Date: 2026-08-03

## Executive result

Difficulty-conditioned upward reallocation is not supported by the existing candidate bank. On the highest SafeGround-disagreement quintile, pass@N rises from {pct(high['accuracy']['4']['pass_at_n'])} at N4 to {pct(high['accuracy']['24']['pass_at_n'])} at N24, {pp(oracle['point_delta'])}, 99% CI {ci(oracle)} pp. B3 instead changes from {pct(high['accuracy']['4']['B3_mvp'])} to {pct(high['accuracy']['24']['B3_mvp'])}, {pp(b3['point_delta'])}, 99% CI {ci(b3)} pp.

R1 therefore fails and triggers R-K1. R2 budget reallocation and R3 conditional-proposal inference are cancelled exactly as preregistered. No S1-S4 result, random-budget control or new crop inference is fabricated after the failed gate.

This is the fifth direct collision-wall confirmation: additional candidates substantially increase oracle availability on difficult rows while unchanged B3 cannot realize the gain.

## Positive result: selective accuracy

SafeGround disagreement is highly useful for abstention even though it is not useful for deciding where to spend more of the existing fixed-view budget.

For Uniform Mixed N12, retaining the least-uncertain 90%, 80% and 70% yields B3 accuracies {pct(mixed[.9]['retained_accuracy'])}, {pct(mixed[.8]['retained_accuracy'])} and {pct(mixed[.7]['retained_accuracy'])}, compared with {pct(mixed[1.0]['retained_accuracy'])} at full coverage. At 80% coverage, the gain is {pp(mixed[.8]['accuracy_gain_vs_full'])}; random rejection has mean {pct(mixed[.8]['random_rejection']['mean_accuracy'])} and 99% interval {ci(mixed[.8]['random_rejection'])}.

V-only N12 also benefits, but reaches only {pct(vonly[.8]['retained_accuracy'])} at 80% coverage. Cross-lineage allocation thus improves both full-coverage grounding and the ranking of cases that should be deferred. The result supports a deployment workflow in which uncertain cases fall back to a human or a more expensive system.

R4 is a selective-prediction result, not evidence that the current uncertainty score can allocate additional fixed-view forwards effectively.

## 72B diagnostic

N3 already ruled out a global coordinate bug. R5 rejects the proposed tight-error-cluster explanation: 72B failed-candidate normalized pair distance is {pollution['mean_failed_pair_distance']:.4f} versus {r5['summaries']['7B_Uniform_Mixed_N8']['mean_failed_pair_distance']:.4f} at 7B, with paired 72B-minus-7B delta {r5['failed_distance_bootstrap']['point_delta_72B_minus_7B']:+.4f} and positive 99% CI.

However, B3 shows severe source bias. Among {pollution['B3_wrong_rows']} wrong 72B B3 rows, the selected candidate comes from GTA1 on {pollution['wrong_selected_model']['GTA1-72B']} rows, UI-Venus on {pollution['wrong_selected_model']['UI-Venus-Ground-72B']} and Qwen3.5 on only {pollution['wrong_selected_model']['Qwen3.5-122B-A10B']}. Winner-cluster model composition is highly nonuniform (`p={pollution['winner_group_model_uniformity']['p_value']:.2e}`).

The supported diagnosis is therefore model-source/coverage bias in B3 selection, not unusually tight strong-model errors.

## Preserved contribution

The paper retains:

1. equal-compute cross-lineage gains of +3.42 pp M1 and +3.54 pp unchanged B3 at 7B N12;
2. fixed-view budget-slope sign reversal;
3. weaker-model complementarity;
4. significant N8 CALA gains at 7B and 72B;
5. selective accuracy as a new deployment-facing positive result;
6. repeated evidence that oracle candidate headroom and final aggregation accuracy are distinct bottlenecks.

No R3 inference is launched, so this run consumes zero new model forwards.
"""; (run/"REPORT.md").write_text(report)
    status={"schema_version":1,"date":"2026-08-03","status":"COMPLETE","R1":"FAIL_R_K1","R2":"CANCELLED_R_K1","R3":"CANCELLED_R_K1_NO_NEW_INFERENCE","R4":"PASS_SELECTIVE_ACCURACY","R5":"FAIL_TIGHTNESS_HYPOTHESIS_WITH_B3_SOURCE_BIAS","new_model_forwards":0,"artifacts":{name:{"path":path.name,"sha256":sha256_file(path)} for name,path in paths.items()},"figures":{"stratified":{"path":"fig_stratified_curves.pdf","sha256":sha256_file(run/"fig_stratified_curves.pdf")},"risk_coverage":{"path":"fig_risk_coverage.pdf","sha256":sha256_file(run/"fig_risk_coverage.pdf")}}}
    (run/"STATUS.json").write_text(json.dumps(status,indent=2,sort_keys=True)+"\n"); print(json.dumps(status,indent=2,sort_keys=True))


if __name__=="__main__": main()