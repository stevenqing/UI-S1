import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    if not path.exists(): raise FileNotFoundError(path)
    return json.loads(path.read_text())


def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def pct(value): return f"{100*value:.2f}%"
def pp(value): return f"{100*value:+.2f} pp"
def ci(record): return f"[{100*record['ci_99'][0]:+.2f}, {100*record['ci_99'][1]:+.2f}]"


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--run-dir",type=Path,default=Path(__file__).resolve().parent); args=parser.parse_args(); run=args.run_dir
    paths={"B1":run/"results/b1_source_bias.json","B2":run/"results/b2_lineage_normalized.json","B4":run/"results/b4_attribution.json"}
    b1,b2,b4=(load(paths[key]) for key in ("B1","B2","B4"))
    if not b1["gate"]["B1_pass"] or b2["B2_primary_success"] or b2["B3x_action"]!="CANCEL": raise ValueError("source-bias gate state mismatch")
    primary=[]
    for scale in ("7B","72B"):
        report=b2["reports"][scale]; accuracy=report["accuracy"]
        for label,key in (("B3","vs_B3"),("M1","vs_M1")):
            comparison=report["comparisons"][key]
            primary.append(f"| {scale} nested LN vs {label} | {pct(accuracy['nested_LN'])} | {pct(comparison['right_accuracy'])} | {pp(comparison['point_delta'])} | {ci(comparison)} | {comparison['p_one_sided_delta_le_zero']:.4g} |")
        reference=report["comparisons"]["vs_best_single_reported"]
        primary.append(f"| {scale} nested LN vs reported best-single | {pct(reference['nested_accuracy'])} | {pct(reference['reported_best_single_accuracy'])} | {pp(reference['point_delta'])} | {'paired CI' if reference['paired_inference_available'] else 'independent trace; no paired CI'} | n/a |")
    bias=[]
    for scale,pool,gta in (("7B","7B_Uniform_Mixed_N12","GTA1-7B"),("72B","72B_Uniform_Mixed_N8","GTA1-72B")):
        report=b1["reports"][pool]["B3_mvp"]["incorrect"]
        bias.append(f"| {scale} B3 incorrect | {report['rows']} | {report['observed_winners'][gta]:d} | {report['expected_winners'][gta]:.2f} | {report['standardized_residuals'][gta]:+.2f} | {report['p_value']:.4g} | {report['cramers_V']:.3f} |")
    table="""# Source-Bias Main Table

## B1 source bias

| Pool/stratum | Rows | GTA observed | GTA expected | GTA residual | Chi-square p | Cramer's V |
|---|---:|---:|---:|---:|---:|---:|
"""+"\n".join(bias)+"""

## B2 nested lineage normalization

| Comparison | LN | Reference | Delta | 99% CI / availability | One-sided p |
|---|---:|---:|---:|---|---:|
"""+"\n".join(primary)+"\n"
    (run/"MAIN_TABLE.md").write_text(table)
    variants=list(b2["reports"]["7B"]["descriptive_crossfit_grid"])
    grid=["# B2 Descriptive 21-Variant Grid","","This cross-fitted grid is sensitivity analysis only. Its maximum is not the headline result.","","| Variant | 7B | 72B |","|---|---:|---:|"]
    for variant in variants: grid.append(f"| {variant} | {pct(b2['reports']['7B']['descriptive_crossfit_grid'][variant])} | {pct(b2['reports']['72B']['descriptive_crossfit_grid'][variant])} |")
    (run/"B2_VARIANT_GRID.md").write_text("\n".join(grid)+"\n")
    seven=b2["reports"]["7B"]; seventy=b2["reports"]["72B"]; balance=b4["count_balancing"]["72B_Uniform_Mixed_N8"]
    report=f"""# Source-Bias and Lineage-Normalized Aggregation Report

## Outcome

B1 passes at both scales. On incorrect B3 rows, GTA wins {b1['reports']['7B_Uniform_Mixed_N12']['B3_mvp']['incorrect']['observed_winners']['GTA1-7B']}/574 times at 7B and {b1['reports']['72B_Uniform_Mixed_N8']['B3_mvp']['incorrect']['observed_winners']['GTA1-72B']}/929 times at 72B, despite candidate-proportion expectations of 191.33 and 348.38. The standardized residuals are +26.36 and +35.49.

B4 does not support the stronger shared-proposer attribution at both scales. The view-0 GTA residual is not weaker than crop views at 7B, and 72B GTA within-lineage geometry is not significantly tighter than both alternatives. The defensible mechanism is a heterogeneous-pool aggregation effect. Candidate-count balancing nevertheless moves 72B B3 from {pct(balance['original_accuracy'])} to {pct(balance['balanced_accuracy'])}, a descriptive {pp(balance['delta'])} gain.

## Nested B2

At 72B, nested lineage normalization reaches {pct(seventy['accuracy']['nested_LN'])}, improving on B3 by {pp(seventy['comparisons']['vs_B3']['point_delta'])} with 99% CI {ci(seventy['comparisons']['vs_B3'])}, and on M1 by {pp(seventy['comparisons']['vs_M1']['point_delta'])} with CI {ci(seventy['comparisons']['vs_M1'])}. It nearly realizes the Qwen3.5 best-single candidate headroom but remains {pp(seventy['comparisons']['vs_best_single_reported']['point_delta'])} below it.

At 7B, nested lineage normalization reaches {pct(seven['accuracy']['nested_LN'])} and is worse than B3 by {pp(seven['comparisons']['vs_B3']['point_delta'])}, 99% CI {ci(seven['comparisons']['vs_B3'])}. The method therefore does not generalize across both frozen scales.

The preregistered B2 primary criterion fails because both scales were required. B-K4 triggers because the 72B nested result remains below best-single. B3x is not run by protocol.

## Claim boundary

The study establishes strong model-source voting bias in B3 and a large 72B correction from lineage normalization. It does not establish proposer-caused bias, a scale-general lineage-normalized method, or a result above best-single. Full-grid maxima remain descriptive.
"""
    (run/"REPORT.md").write_text(report)
    artifacts={key:{"path":str(path.relative_to(run)),"sha256":sha256(path)} for key,path in paths.items()}
    artifacts.update({name:{"path":name,"sha256":sha256(run/name)} for name in ("MAIN_TABLE.md","B2_VARIANT_GRID.md","REPORT.md")})
    status={"schema_version":1,"status":"COMPLETE","B1_source_bias":"PASS_BOTH_SCALES","B2_nested_primary":"FAIL_7B","B2_72B_bias_correction":"PASS","B_K4":True,"B3x":"NOT_RUN_B2_GATE_FAILED","proposal_source_attribution":"NOT_SUPPORTED_BOTH_SCALES","mechanism_scope":"heterogeneous_pool_aggregation_effect","artifacts":artifacts}
    (run/"STATUS.json").write_text(json.dumps(status,indent=2,sort_keys=True)+"\n")
    print(json.dumps(status,indent=2,sort_keys=True))


if __name__=="__main__": main()