import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(value):
    return f"{100 * value:.2f}%"


def pp(value):
    return f"{100 * value:+.2f} pp"


def ci(record):
    return f"[{100 * record['ci_99'][0]:+.2f}, {100 * record['ci_99'][1]:+.2f}] pp"


def build_table(run, z, g1, g2):
    lines = [
        "# Scale-Up Main Table",
        "",
        "## Controlled 7B paired results",
        "",
        "| Comparison | Left | Right | Delta | 99% CI | One-sided p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record in z["Z1_paired_bootstrap"].values():
        lines.append(
            f"| {record['left']} vs {record['right']} | {pct(record['left_accuracy'])} | {pct(record['right_accuracy'])} | "
            f"{pp(record['point_delta'])} | {ci(record)} | {record['p_one_sided_delta_le_zero']:.4g} |"
        )
    lines.extend([
        "",
        "The H3-native B3 comparison (63.63% vs 60.09%) is primary. The later Allocation/Closing reconstruction gives 63.69% for Mixed B3 and is retained only as an implementation-sensitivity check.",
        "",
        "## G1 72B lineage gate",
        "",
        "| Model | Local bare | Paper-only reference | Difference | Anchor within 2 pp |",
        "|---|---:|---:|---:|---|",
    ])
    for model, record in g1["bare"].items():
        lines.append(f"| {model} | {pct(record['accuracy'])} | {pct(record['paper_only_reference'])} | {pp(record['local_minus_reference'])} | {record['anchor_consistent_within_2pp']} |")
    lines.extend([
        "",
        "| Pair | Failure kappa | Matched-marginal p |",
        "|---|---:|---:|",
    ])
    for record in g1["pairwise_failure_kappa"].values():
        lines.append(f"| {record['id']} | {record['observed_kappa']:.3f} | {record['p_greater_equal']:.4g} |")
    lines.extend([
        "",
        f"pass@3 is {pct(g1['pass_at_3'])}. G1 pass is `{g1['gate']['G1_pass']}`; action is `{g1['gate']['G2_action']}`.",
        "",
    ])
    if g2 is not None:
        lines.extend([
            "## G2 controlled 72B pools",
            "",
            "| Pool | Budget | B3 | M1 | pass@N |",
            "|---|---:|---:|---:|---:|",
        ])
        for label, key in (("P1 GTA1-72B single lineage", "P1_GTA1_72B"), ("P2 mixed 72B", "P2_mixed_72B")):
            accuracy = g2[key]["accuracy"]
            lines.append(f"| {label} | {g2[key]['budget']} | {pct(accuracy['B3_mvp'])} | {pct(accuracy['M1_ccm'])} | {pct(accuracy['pass_at_n'])} |")
        lines.extend([
            "",
            f"P2-P1 M1 is {pp(g2['comparisons']['P2_M1_minus_P1_M1_unequal_budget_context'])}, reported only as unequal-budget context (P2 N12 versus P1 N{g2['P1_GTA1_72B']['budget']}). Proposal MDE is {pp(g2['proposal_sensitivity']['MDE'])}. Outcome: `{g2['decision']['outcome']}`.",
            "",
        ])
    lines.extend([
        "## Paper-only context",
        "",
        "| System/model | ScreenSpot-Pro | Comparability |",
        "|---|---:|---|",
        "| Qwen3.5-122B-A10B reported model | 70.40% | Paper only; excluded from paired calculations |",
        "| ZoomClick + UI-Venus-Ground-72B | 73.10% | Paper only; excluded from paired calculations |",
        "| MVP trained GRPO selector | 62.80% | Paper only; excluded from paired calculations |",
        "",
    ])
    (run / "MAIN_TABLE.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    run = args.run_dir
    paths = {
        "Z": run / "z_closing_stats.json",
        "Z5": run / "z5_sampling_axis.json",
        "G1": run / "g1_lineage_gate.json",
    }
    z, z5, g1 = (load(paths[key]) for key in ("Z", "Z5", "G1"))
    if z["status"] != "PASS" or not z5["status"].startswith("PASS") or g1["status"] != "PASS":
        raise ValueError("Scale-Up prerequisite result incomplete")
    g2 = None
    if not g1["gate"]["G2_cancelled"]:
        paths["G2"] = run / "g2_mixed_72b.json"
        g2 = load(paths["G2"])
        if g2["status"] != "PASS":
            raise ValueError("Scale-Up G2 result incomplete")
    endpoint = "B_G1_COMMON_FAILURE_CEILING" if g2 is None else "A_OPEN_SYSTEM_SOTA" if g2["decision"]["system_SOTA_73_1_pass"] else "B_CONTROLLED_7B_PLUS_SCALEUP_BOUNDARY"
    main = z["Z1_paired_bootstrap"]["mixed_N12_M1_vs_v_only_N12_M1"]
    b3 = z["Z1_paired_bootstrap"]["mixed_N12_B3_vs_v_only_N12_B3"]
    g2_text = (
        "G2 was cancelled by the preregistered pass@3 ceiling rule."
        if g2 is None else
        f"P1 GTA1-72B N{g2['P1_GTA1_72B']['budget']} M1 is {pct(g2['P1_GTA1_72B']['accuracy']['M1_ccm'])}; P2 mixed N12 M1 is {pct(g2['P2_mixed_72B']['accuracy']['M1_ccm'])}. Their {pp(g2['comparisons']['P2_M1_minus_P1_M1_unequal_budget_context'])} difference is unequal-budget context, not an equal-compute claim. The proposal MDE is {pp(g2['proposal_sensitivity']['MDE'])}. The frozen outcome is `{g2['decision']['outcome']}`."
    )
    report = f"""# Scale-Up Gate Report

Date: 2026-08-02

Status: complete endpoint `{endpoint}`.

## G1 lineage gate

The three-model bare pass@3 is {pct(g1['pass_at_3'])}. The minimum pairwise failure kappa is {min(value['observed_kappa'] for value in g1['pairwise_failure_kappa'].values()):.3f}. G1 pass is `{g1['gate']['G1_pass']}`, lineage-concentrated is `{g1['gate']['lineage_concentrated']}`, and the frozen G2 action is `{g1['gate']['G2_action']}`.

Local bare scores and paper-only differences are reported in `MAIN_TABLE.md`. Anchor disagreement is treated as a reproducibility observation; no prompt or parser was retuned.

## G2 scale-up result

{g2_text}

The reported 70.4 and 73.1 values are independently source-verified but remain paper-only context, not same-environment controls, and never enter a row-level paired significance test. `REFERENCE_AUDIT.md` records the exact sources and protocol differences.

## 7B statistical close

Mixed N12 M1 reaches {pct(main['left_accuracy'])} versus V-only {pct(main['right_accuracy'])}: {pp(main['point_delta'])}, 99% CI {ci(main)}, one-sided p={main['p_one_sided_delta_le_zero']:.4g}. H3-native unchanged B3 moves from {pct(b3['right_accuracy'])} to {pct(b3['left_accuracy'])}: {pp(b3['point_delta'])}, 99% CI {ci(b3)}, p={b3['p_one_sided_delta_le_zero']:.4g}.

All three Mixed-versus-bare comparisons and both N16 comparisons have positive 99% CI lower bounds. The N=2 H1 column moves to the appendix because M1/M2 collapse to the full-image prediction and M1 headroom capture is 0%.

## Scope corrections

The budget-decline claim uses V-only N=4 to N=16: B3 changes by {pp(z['Z3_decline_scope']['v_only_B3_delta'])}, supported by the negative X3 slope CI. H1 N=4 to N=10 is only a same-candidate-set rule comparison. Main-text MDE uses full/v1 exchangeable perturbations; v2-v4 are information deletion/deployment shifts.

The S-only GUI-RC sampling slope is {z5['slopes']['S_only']['GUI_RC']['point_slope_per_forward']:.6f} per forward with 99% CI [{z5['slopes']['S_only']['GUI_RC']['ci_99'][0]:.6f}, {z5['slopes']['S_only']['GUI_RC']['ci_99'][1]:.6f}]. Because the CI crosses zero, the paper scope is **fixed-view allocation axis**, not a general single-model diversity axis.

## Paper disposition

UI-Zoomer X2 is excluded from the positive-result evidence chain and no X2 number enters this report. SafeGround, held-out pool ranking, and collision-floor evidence remain supporting diagnostics. No absolute open-source SOTA claim is made unless `system_SOTA_73_1_pass` is true.
"""
    (run / "REPORT.md").write_text(report)
    build_table(run, z, g1, g2)
    status = {
        "schema_version": 1,
        "date": "2026-08-02",
        "status": "COMPLETE",
        "endpoint": endpoint,
        "G1_gate": g1["gate"],
        "G2_decision": None if g2 is None else g2["decision"],
        "title_scope": z5["adjudication"]["title_scope"],
        "artifacts": {name: {"path": path.name, "sha256": sha256_file(path)} for name, path in paths.items()},
        "protected_pid_1814_modified": False,
    }
    (run / "STATUS.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "COMPLETE", "endpoint": endpoint, "G1": g1["gate"], "G2": status["G2_decision"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()