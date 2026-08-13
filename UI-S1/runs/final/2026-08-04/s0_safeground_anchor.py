import argparse
import hashlib
import json
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
X7_PATH = ROOT / "runs/diversity-axis/2026-08-02/x7_confidence.json"
R4_PATH = ROOT / "runs/reallocation/2026-08-03/r4_risk_coverage.json"

OFFICIAL_AUROC = 0.6344
OFFICIAL_TABLE_DECIMALS = 4
OFFICIAL_HALF_LAST_DIGIT = 0.5 * 10 ** (-OFFICIAL_TABLE_DECIMALS)


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output_path):
    x7 = json.loads(X7_PATH.read_text())
    r4 = json.loads(R4_PATH.read_text())
    local = x7["pools"]["stochastic_GTA1_N4"]["variants"]["official_code"]["GUI_RC"]
    local_auroc = float(local["correctness_auroc_negative_uncertainty"])
    rounded_local = round(local_auroc, OFFICIAL_TABLE_DECIMALS)
    anchor_delta = local_auroc - OFFICIAL_AUROC
    numerical_pass = abs(anchor_delta) <= OFFICIAL_HALF_LAST_DIGIT

    mixed_auroc = float(x7["pools"]["mixed_N12"]["variants"]["official_code"]["M1_ccm"]["correctness_auroc_negative_uncertainty"])
    v_only_auroc = float(x7["pools"]["v_only_N12"]["variants"]["official_code"]["M1_ccm"]["correctness_auroc_negative_uncertainty"])
    mixed_curve = {float(row["coverage"]): row for row in r4["curves"]["Uniform_Mixed_N12"]}
    v_only_curve = {float(row["coverage"]): row for row in r4["curves"]["V_only_N12"]}
    matched_coverage = 0.8
    b3_advantage = (
        float(mixed_curve[matched_coverage]["retained_accuracy"])
        - float(v_only_curve[matched_coverage]["retained_accuracy"])
    )
    if round(b3_advantage * 100, 2) != 7.12:
        raise ValueError(f"R4 matched-coverage anchor mismatch: {b3_advantage}")

    result = {
        "schema_version": 1,
        "status": (
            "PASS_NUMERICAL_ANCHOR"
            if numerical_pass
            else "NUMERICAL_ANCHOR_NOT_PASSED_ALGORITHM_LEVEL_PORT"
        ),
        "official_anchor": {
            "paper": "arXiv:2602.02419v2",
            "model": "GTA1-7B",
            "metric": "U_COM_AUROC",
            "value": OFFICIAL_AUROC,
            "table_decimals": OFFICIAL_TABLE_DECIMALS,
            "acceptance_half_band": OFFICIAL_HALF_LAST_DIGIT,
            "protocol": {
                "samples": 10,
                "temperature": 1.0,
                "patch_size": 14,
                "activation_ratio_beta": 0.3,
                "weights": {"concentration": 0.6, "entropy": 0.2, "margin": 0.2},
            },
        },
        "local_anchor_attempt": {
            "value": local_auroc,
            "rounded_four_decimals": rounded_local,
            "delta_from_official": anchor_delta,
            "within_table_precision_band": numerical_pass,
            "protocol": {
                "samples": 4,
                "temperature": 0.7,
                "patch_size": 28,
                "activation_threshold": 0.0,
                "primary_prediction": "GUI_RC",
            },
            "protocol_match": False,
            "interpretation": (
                "The existing local trace cannot validate the official numerical row: "
                "it differs in K, temperature, patch size, and thresholding. The local "
                "implementation remains an algorithm-level geometry transfer."
            ),
        },
        "r4_repositioning": {
            "claim": "Cross-lineage candidate pools strengthen a transferred dispersion signal.",
            "v_only_correctness_auroc": v_only_auroc,
            "mixed_correctness_auroc": mixed_auroc,
            "auroc_delta": mixed_auroc - v_only_auroc,
            "matched_coverage": matched_coverage,
            "v_only_retained_B3": float(v_only_curve[matched_coverage]["retained_accuracy"]),
            "mixed_retained_B3": float(mixed_curve[matched_coverage]["retained_accuracy"]),
            "mixed_minus_v_only_B3": b3_advantage,
            "prohibited_claim": "Disagreement itself is a novel selective-prediction method.",
        },
        "claim_boundaries": [
            "Deterministic N12 candidates are not K=10 stochastic samples.",
            "No local Clopper-Pearson calibration or FDR guarantee is inherited.",
            "The 5.38 percentage-point cascade gain is paper-only and non-comparable.",
        ],
        "official_asset_audit": {
            "commit_tree_contains_precomputed_GTA1_K10_outputs": False,
            "available_assets": "source_code_and_figures_only",
            "zero_GPU_exact_anchor_reproduction_from_official_repo": False,
        },
        "related_work_coordinates": {
            "SafeGround": "black-box spatial dispersion plus LTT/Clopper-Pearson FDR control",
            "HyperClick": "truncated-Gaussian spatial confidence head with calibration",
            "Argus": "post-hoc GUI grounding UQ benchmark spanning internal and frontier interfaces",
        },
        "sources": {
            "SafeGround_commit": x7["official_source"]["commit"],
            "X7_sha256": sha256_file(X7_PATH),
            "R4_sha256": sha256_file(R4_PATH),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RUN_DIR / "s0_safeground_anchor.json")
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()