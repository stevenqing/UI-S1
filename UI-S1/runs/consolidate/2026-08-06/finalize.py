import hashlib
import json
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
RESULTS = (
    "s1_pool_distribution",
    "s2_pool_selector",
    "s3_leave_one_lineage",
    "s4_slope_hardening",
    "s5_decline_attribution",
    "s6_anchors",
    "q1_sequential",
    "q2a_element_space",
    "q2b_verification",
)


def load(name):
    path = RUN_DIR / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"mandatory consolidation result missing: {path}")
    value = json.loads(path.read_text())
    if value.get("status") != "PASS":
        raise ValueError(f"mandatory consolidation result not PASS: {name}")
    return value


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(value):
    return f"{100 * value:.2f}%"


def pp(value):
    return f"{100 * value:+.2f} pp"


def ci(values):
    return f"[{100 * values[0]:+.2f}, {100 * values[1]:+.2f}] pp"


def main():
    data = {name: load(name) for name in RESULTS}
    s1 = data["s1_pool_distribution"]
    s2 = data["s2_pool_selector"]
    s3 = data["s3_leave_one_lineage"]
    s4 = data["s4_slope_hardening"]
    s5 = data["s5_decline_attribution"]
    s6 = data["s6_anchors"]
    q1 = data["q1_sequential"]
    q2a = data["q2a_element_space"]
    q2b = data["q2b_verification"]

    if not (s1["S_K1"] and s3["S_K2"] and s4["S_K3"]):
        raise ValueError("finalizer expects the adjudicated S-K1/S-K2/S-K3 path")
    if s2["success"] or q2a["claim"] != "ELEMENT_SPACE_NOT_SUPPORTED":
        raise ValueError("finalizer selector/Q2a gate mismatch")
    if s5["attribution"] != "RANK_DECAY_DOMINANT":
        raise ValueError("finalizer S5 attribution mismatch")

    s1_two = s1["reports"]["2"]["B3_mvp"]["distribution"]
    s1_three = s1["reports"]["3"]["B3_mvp"]["distribution"]
    s2_primary = s2["summary"]["primary"]
    s3_full = s3["pools"]["full_three_lineages_4x3"]["accuracy"]
    s3_without_ui = s3["pools"]["leave_out_UI_TARS"]["accuracy"]
    s4_v = s4["paired_N16_minus_N4"]["v_only"]
    s4_m = s4["paired_N16_minus_N4"]["mixed"]
    s6_b3 = s6["action_cluster_bootstrap"]["B3_minus_best"]
    s6_m1 = s6["action_cluster_bootstrap"]["M1_minus_best"]
    q1_primary = q1["comparisons"]["C_cond_minus_C_uni"]["B3_mvp"]
    q1_rand = q1["comparisons"]["C_cond_minus_C_rand"]["B3_mvp"]
    q1_self = q1["comparisons"]["C_cond_minus_C_self"]["B3_mvp"]
    q2b_primary = q2b["verified_B3_vs_Uniform_N12"]

    table = f"""# ScreenSpot-Pro Consolidation Main Table

| Experiment | Result | Decision |
|---|---|---|
| S1 two-lineage pools vs same-budget best single-lineage | positive {pct(s1_two['positive_share'])}; median {pp(s1_two['median'])}; IQR {ci([s1_two['q1'], s1_two['q3']])} | S-K1 triggered |
| S1 three-lineage pools vs same-budget best single-lineage | positive {pct(s1_three['positive_share'])}; median {pp(s1_three['median'])}; IQR {ci([s1_three['q1'], s1_three['q3']])} | only some configurations win |
| S2 held-out pool selector | mean Spearman {s2_primary['heldout_spearman_mean']:.3f}; top-decile delta {pp(s2_primary['top10_delta_bootstrap']['point'])}, 99% CI {ci(s2_primary['top10_delta_bootstrap']['ci_99'])} | enrichment, not reliable ranking |
| S3 full three-lineage N12 | B3/M1 {pct(s3_full['B3_mvp'])}/{pct(s3_full['M1_ccm'])} | reference |
| S3 leave UI-TARS N12 | B3/M1 {pct(s3_without_ui['B3_mvp'])}/{pct(s3_without_ui['M1_ccm'])} | S-K2 triggered; third lineage saturated |
| S4 V-only N16 minus N4 | B3 {pp(s4_v['B3_mvp']['point_delta'])}, CI {ci(s4_v['B3_mvp']['ci_99'])}; M1 {pp(s4_v['M1_ccm']['point_delta'])}, CI {ci(s4_v['M1_ccm']['ci_99'])} | paired decline |
| S4 Mixed N16 minus N4 | B3 {pp(s4_m['B3_mvp']['point_delta'])}, CI {ci(s4_m['B3_mvp']['ci_99'])}; M1 {pp(s4_m['M1_ccm']['point_delta'])}, CI {ci(s4_m['M1_ccm']['ci_99'])} | paired increase |
| S5 randomized view order | mean slope {s5['random_order']['slope_mean']:+.6f}; negative share {pct(s5['random_order']['negative_slope_share'])} | rank decay dominant |
| S6 B3 action-cluster bootstrap | raw CI {s6_b3['raw_rho_distribution']['ci_99']}; partial CI {s6_b3['partial_rho_distribution']['ci_99']} | negative direction robust |
| S6 M1 action-cluster bootstrap | raw CI {s6_m1['raw_rho_distribution']['ci_99']}; partial CI {s6_m1['partial_rho_distribution']['ci_99']} | mechanism evidence, not law |
| Q1 C-cond vs C-uni B3 | {pp(q1_primary['point_delta'])}, 99% CI {ci(q1_primary['ci_99'])} | {'PASS' if q1['primary_success'] else 'FAIL'} |
| Q1 C-cond vs C-rand B3 | {pp(q1_rand['point_delta'])}, 99% CI {ci(q1_rand['ci_99'])} | Q-K1={q1['Q_K1']} |
| Q1 C-cond vs C-self B3 | {pp(q1_self['point_delta'])}, 99% CI {ci(q1_self['ci_99'])} | Q-K2={q1['Q_K2']} |
| Q2a element-space combined25 vs combined24 | {pp(q2a['combined25_vs_combined24']['point_delta'])}, 99% CI {ci(q2a['combined25_vs_combined24']['ci_99'])} | not supported |
| Q2b verified B3 vs Uniform N12 | {pp(q2b_primary['point_delta'])}, 99% CI {ci(q2b_primary['ci_99'])} | {'PASS' if q2b['primary_success'] else 'FAIL'}; Q-K3={q2b['Q_K3']} |
| Q2b binary verifier | accuracy {pct(q2b['verification']['accuracy'])}; yes precision {pct(q2b['verification']['yes_precision']) if q2b['verification']['yes_precision'] is not None else 'n/a'}; yes recall {pct(q2b['verification']['yes_recall']) if q2b['verification']['yes_recall'] is not None else 'n/a'} | mandatory channel diagnostic |
"""
    (RUN_DIR / "MAIN_TABLE.md").write_text(table)

    q1_sentence = (
        "Sequential cross-lineage consensus RoIs pass the frozen B3 gate."
        if q1["primary_success"] else
        "Sequential cross-lineage consensus RoIs do not pass the frozen B3 gate."
    )
    q2b_sentence = (
        "Cross-lineage binary verification passes the equal-budget gate."
        if q2b["primary_success"] else
        "Cross-lineage binary verification does not pass the equal-budget gate."
    )
    report = f"""# ScreenSpot-Pro Consolidation Report

## Outcome

The zero-GPU consolidation weakens two prior claims and strengthens the budget-curve mechanism. S-K1, S-K2, and S-K3 all trigger: cross-lineage action-pool superiority is configuration-specific, UI-TARS has no positive marginal contribution at N12, and dense linear slopes cross zero. The primary budget statement therefore moves to paired N4-to-N16 differences.

## Distribution and selection

Only {pct(s1_three['positive_share'])} of three-lineage 3-forward pools beat the strongest same-budget single-lineage pool under B3; no two-lineage pool does. The reported N12 configuration is explicitly nonexchangeable with this 2/3-forward distribution. The unlabeled selector reaches mean held-out Spearman {s2_primary['heldout_spearman_mean']:.3f}, below the 0.7 method gate, but enriches the top decile by {pp(s2_primary['top10_delta_bootstrap']['point'])} with a positive 99% interval. It is an enrichment diagnostic, not an independent reliable ranker.

## Budget mechanism

V-only falls from N4 to N16 by {pp(s4_v['B3_mvp']['point_delta'])} B3 and {pp(s4_v['M1_ccm']['point_delta'])} M1, while Mixed changes by {pp(s4_m['B3_mvp']['point_delta'])} and {pp(s4_m['M1_ccm']['point_delta'])}. Randomizing GTA1 view order changes the mean slope from {s5['original']['slope']:+.6f} to {s5['random_order']['slope_mean']:+.6f}; only {pct(s5['random_order']['negative_slope_share'])} of random orders remain negative. Rank decay is the primary observed driver.

## Dependence-aware dominance evidence

Action-cluster bootstrap preserves a negative direction for both raw and controlled B3/M1 correlations. The M1 partial-rho-squared proxy is {s6['D1_positioning']['variance_explained_proxy_M1_partial_rho_squared']:.3f}, but this is rank-association evidence, not causal variance decomposition or a universal law. SafeGround remains an algorithm-level port because the numerical anchor does not match protocol.

## Q-series

{q1_sentence} C-cond minus C-uni is {pp(q1_primary['point_delta'])}, 99% CI {ci(q1_primary['ci_99'])}; mandatory random and self-consensus controls are retained. Q2a fails: adding patch-28 element mode changes nested accuracy by {pp(q2a['combined25_vs_combined24']['point_delta'])}. {q2b_sentence} Its binary channel accuracy is {pct(q2b['verification']['accuracy'])}.

## Claim boundary

The defensible positive result is the paired budget-curve divergence and its rank-decay diagnosis, plus the already established two-scale source bias and narrowed selective-prediction result. Pool superiority is not general, the weakest lineage is saturated at N12, the unlabeled selector is not a high-reliability method, and Q-series claims follow their frozen paired controls exactly.
"""
    (RUN_DIR / "REPORT.md").write_text(report)

    artifacts = {}
    for name in RESULTS:
        path = RUN_DIR / f"{name}.json"
        artifacts[name] = {"path": path.name, "sha256": sha256_file(path)}
    for name in (
        "SPEC.md",
        "CONSOLIDATED_SUMMARY_ZH.md",
        "MAIN_TABLE.md",
        "REPORT.md",
        "fig_pool_distribution.pdf",
        "configs/q1_arms.yaml",
        "configs/q2a_variant.yaml",
        "configs/q2b_verification.yaml",
    ):
        path = RUN_DIR / name
        artifacts[name] = {"path": name, "sha256": sha256_file(path)}
    status = {
        "schema_version": 1,
        "status": "COMPLETE",
        "S_K1": s1["S_K1"],
        "S_K2": s3["S_K2"],
        "S_K3": s4["S_K3"],
        "S2_success": s2["success"],
        "Q1_primary_success": q1["primary_success"],
        "Q_K1": q1["Q_K1"],
        "Q_K2": q1["Q_K2"],
        "Q2a_claim": q2a["claim"],
        "Q2b_primary_success": q2b["primary_success"],
        "Q_K3": q2b["Q_K3"],
        "artifacts": artifacts,
    }
    (RUN_DIR / "STATUS.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()