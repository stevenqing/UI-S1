import hashlib
import json
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
STAGE1_PATH = RUN_DIR / "STAGE1.json"
ADJUDICATION_PATH = RUN_DIR / "MASK_ADJUDICATION.json"
REPORT_PATH = RUN_DIR / "REPORT.md"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if ADJUDICATION_PATH.exists() or REPORT_PATH.exists():
        raise FileExistsError("MASK final outputs already exist")
    stage1 = json.loads(STAGE1_PATH.read_text())
    if (
        stage1.get("status") != "STOP_M_K1_BEFORE_GPU"
        or stage1.get("M_G1", {}).get("pass") is not False
        or stage1.get("model_forward_started") is not False
        or stage1.get("subset_manifest_created") is not False
        or stage1.get("gpu_authorization_created") is not False
    ):
        raise PermissionError("MASK stage1 is not in frozen pre-GPU stop state")
    base = stage1["base_rates_and_masks"]
    calibration = stage1["neff_calibration"]
    verifier = stage1["verifier_curve"]
    later = {
        endpoint: {"status": "NOT_RUN_M_K1_PRE_GPU_STOP"}
        for endpoint in ("M_P2", "M_P1", "M_P3", "M_P4", "M_P5")
    }
    adjudication = {
        "schema_version": 1,
        "status": "COMPLETE_PRE_GPU_STOP",
        "outcome": "MASK_STOPPED_M_K1_IDEAL_NEFF_GAIN_BELOW_MDE",
        "exploratory_only": True,
        "M_G1": stage1["M_G1"],
        "verifier_curve": {
            "g_grid": [row["g"] for row in verifier["curves"]],
            "gate_rows": [row["gate_rows"] for row in verifier["curves"]],
            "positives": [row["positives"] for row in verifier["curves"]],
            "positive_rates": [row["positive_rate"] for row in verifier["curves"]],
            "g_0_25_net_gain_by_AUROC": {
                str(row["AUROC"]): row["net_gain"]
                for row in verifier["curves"][0]["hypothetical_discrimination"]
            },
        },
        "neff_calibration": {
            "subsets": calibration["subsets"],
            "valid_subsets": calibration["valid_subsets"],
            "full_pool_cross_fitted_neff": calibration["full_pool_cross_fitted_neff"],
            "ideal_neff_increment": calibration["ideal_neff_increment"],
            "density_B3_predicted_gain": calibration["aggregators"]["density_B3"]["predicted_gain"],
            "F1_majority_predicted_gain": calibration["aggregators"]["F1_majority"]["predicted_gain"],
        },
        "base_rates": {
            key: base[key] for key in (
                "rows", "density_B3_accuracy", "density_B3_error_rate",
                "pool_wrong_rows", "pool_correct_rows",
                "M2_correct_within_pool_wrong_rows",
                "M2_correct_rate_within_pool_wrong", "single_mode_rows",
                "empty_mask_infeasible_rows", "empty_mask_infeasible_rate",
            )
        },
        "endpoints": later,
        "kill_conditions": {
            "M_K1": True,
            "M_K2": None,
            "M_K3": None,
            "M_K4": None,
            "M_K5": None,
            "M_K6": None,
            "M_K7": None,
            "M_K8": False,
        },
        "model_forward_count": 0,
        "subset_manifest_created": False,
        "gpu_authorization_created": False,
        "input_hashes": {
            "STAGE1.json": sha256_file(STAGE1_PATH),
            "VERIFIER_CONTOURS.pdf": sha256_file(RUN_DIR / "VERIFIER_CONTOURS.pdf"),
        },
        "claim_boundary": {
            "occlusion_proposer_supported": False,
            "universal_neff_law_supported": False,
            "deployable_method_claim_allowed": False,
            "changes_F1_Q1_TRIVUS_or_VUS_SR": False,
        },
        "next_action": "CLOSE_MASK_NO_GPU_DO_NOT_RESCUE_WITH_ALTERNATE_MASK_OR_GATE",
    }
    ADJUDICATION_PATH.write_text(json.dumps(adjudication, indent=2, sort_keys=True) + "\n")
    g025 = verifier["curves"][0]
    selected_auc = {
        row["AUROC"]: row["net_gain"] for row in g025["hypothetical_discrimination"]
    }
    report = f"""# MASK Consensus-Occlusion Proposer Report

Date: 2026-08-14

Status: `MASK_STOPPED_M_K1_IDEAL_NEFF_GAIN_BELOW_MDE`

## Scope

MASK is an exploratory proposer study, independent from SPLIT. It does not change F1, Q1, `TRIVUS_NOT_PROMOTED`, or VUS-SR. The run stopped at the zero-GPU gate: no subset, GPU authorization, masked-model forward, or endpoint evaluation was created.

## Verifier closure

Sweeping the frozen SPLIT gate confirms the base-rate problem. At $g=0.25$, 1,187 rows enter the gate, but only 102 are M2-only positives: $\pi_0=8.59\%$ and $N/P=10.64$. Under the preregistered equal-variance Gaussian model, the best full-set net gain is:

| Hypothetical AUROC | Net gain |
| ---: | ---: |
| 0.75 | {100 * selected_auc[0.75]:.04f} pp |
| 0.80 | {100 * selected_auc[0.8]:.04f} pp |
| 0.85 | {100 * selected_auc[0.85]:.04f} pp |
| 0.88 | {100 * selected_auc[0.88]:.04f} pp |
| 0.90 | {100 * selected_auc[0.9]:.04f} pp |

Thus the verifier route does not exceed the 0.70 pp MDE until roughly AUROC 0.88. The same qualitative constraint remains across the full frozen $g$ grid. See `VERIFIER_CONTOURS.pdf`.

## Effective-vote calibration and M-G1

All 4,095 nonempty subsets of the 12 C-uni source slots were evaluated with five-fold cross-fitting. The full-pool generalized $N_{{\mathrm{{eff}}}}$ is {calibration['full_pool_cross_fitted_neff']:.4f}. Under the deliberately favorable ideal $\kappa_{{\mathrm{{new}}}}=0$, three new votes add 1.3636 effective votes.

The within-benchmark monotone calibration predicts:

- density B3: **+{100 * calibration['aggregators']['density_B3']['predicted_gain']:.3f} pp**;
- F1 majority: **+{100 * calibration['aggregators']['F1_majority']['predicted_gain']:.3f} pp**.

The maximum ideal prediction is +{100 * stage1['M_G1']['maximum_ideal_predicted_gain']:.3f} pp, below the preregistered 0.70 pp MDE. M-G1 fails and M-K1 stops the round before GPU. This calibration is benchmark-local and does not restore the previously rejected universal one-dimensional $N_{{\mathrm{{eff}}}}$ law.

## Base rates and mask control

Original C-uni density B3 accuracy is {100 * base['density_B3_accuracy']:.2f}%; 574/1,581 rows are pool-wrong. M2 is correct on only 78/574 pool-wrong rows ({100 * base['M2_correct_rate_within_pool_wrong']:.2f}%). Fifty-five rows have only one C-uni mode under inherited $\tau^*$.

The deterministic equal-area empty mask is infeasible on 5/1,581 rows ({100 * base['empty_mask_infeasible_rate']:.2f}%), below the 15% control limit. M-K8 is not triggered; geometry/control feasibility is not the reason for stopping.

## Conclusion

> Even under the ideal assumption that three masked proposals are uncorrelated with the original pool, the frozen within-benchmark calibration predicts at most +0.538 pp, below the 0.70 pp MDE. MASK therefore stops before GPU and provides no evidence that consensus occlusion is a useful orthogonal proposer channel.

M-P2, M-P1, M-P3, M-P4, and M-P5 are all `NOT_RUN_M_K1_PRE_GPU_STOP`. Replacing the mask, radius, fill, proposer, aggregation, or adding a runtime gate would be a new study rather than a MASK rescue.
"""
    REPORT_PATH.write_text(report)
    print(json.dumps({
        "outcome": adjudication["outcome"],
        "maximum_ideal_gain": stage1["M_G1"]["maximum_ideal_predicted_gain"],
        "MDE": stage1["M_G1"]["mde"],
        "model_forward_count": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()