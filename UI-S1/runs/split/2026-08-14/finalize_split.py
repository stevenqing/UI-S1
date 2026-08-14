import hashlib
import json
from collections import Counter
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
GATE_PATH = RUN_DIR / "ZERO_GPU_GATE.json"
GEOMETRY_PATH = RUN_DIR / "GEOMETRY_AUDIT.json"
ADJUDICATION_PATH = RUN_DIR / "SPLIT_ADJUDICATION.json"
REPORT_PATH = RUN_DIR / "REPORT.md"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if ADJUDICATION_PATH.exists() or REPORT_PATH.exists():
        raise FileExistsError("SPLIT final outputs already exist")
    gate = json.loads(GATE_PATH.read_text())
    geometry = json.loads(GEOMETRY_PATH.read_text())
    if gate.get("status") != "PASS_Z_G1_PROCEED_TO_GEOMETRY":
        raise PermissionError("SPLIT Z-G1 did not pass")
    if geometry.get("status") != "STOP_Z_K6_GEOMETRY":
        raise PermissionError("SPLIT geometry is not in frozen stop state")
    if gate.get("probe_forward_started") is not False or geometry.get("model_forward_started") is not False:
        raise PermissionError("SPLIT unexpected model forward")
    failures = Counter()
    combinations = Counter()
    for row in geometry["rows"].values():
        failed = tuple(
            name for name, passed in row["include_exclude"].items() if not passed
        )
        if failed:
            combinations["+".join(failed)] += 1
            failures.update(failed)
    endpoints = {
        endpoint: {
            "status": "NOT_RUN_PRE_GPU_STOP",
            "reason": "Z_K6_geometry_failure_rate_exceeded_15_percent",
        }
        for endpoint in ("Z_P3", "Z_P1", "Z_P2", "Z_P4", "Z_P5")
    }
    adjudication = {
        "schema_version": 1,
        "status": "COMPLETE_PRE_GPU_STOP",
        "outcome": "SPLIT_STOPPED_PRE_GPU_Z_K6_GEOMETRY_AND_Z_K7_LOW_N",
        "exploratory_only": True,
        "zero_gpu_gate": {
            "Z_G1_pass": True,
            "selected_g_by_fold": [fold["selected_g"] for fold in gate["folds"]],
            **gate["pooled"],
        },
        "geometry": {
            key: geometry[key]
            for key in (
                "gate_rows", "valid_rows", "geometry_failed_rows",
                "geometry_failure_rate", "maximum_failure_rate",
                "positive_rows_before_geometry", "positive_rows_after_geometry",
                "negative_rows_before_geometry", "negative_rows_after_geometry",
            )
        },
        "geometry_failure_predicates": dict(sorted(failures.items())),
        "geometry_failure_combinations": dict(sorted(combinations.items())),
        "kill_conditions": {
            "Z_K1": False,
            "Z_K2": None,
            "Z_K3": None,
            "Z_K4": None,
            "Z_K5": None,
            "Z_K6": True,
            "Z_K7": True,
        },
        "endpoints": endpoints,
        "models": {
            "Qwen3-VL-8B-Instruct": "NOT_RUN",
            "GTA1-7B": "NOT_RUN",
            "Qwen2.5-VL-7B-Instruct": "DEFERRED_CHECKPOINT_MISSING",
        },
        "gpu_authorization_created": False,
        "model_forward_count": 0,
        "balanced_subset_created": False,
        "input_hashes": {
            "ZERO_GPU_GATE.json": sha256_file(GATE_PATH),
            "GEOMETRY_AUDIT.json": sha256_file(GEOMETRY_PATH),
        },
        "claim_boundary": {
            "falsification_crop_channel_supported": False,
            "deployable_method_claim_allowed": False,
            "confirmation_claim_allowed": False,
            "changes_F1": False,
            "changes_Q1": False,
            "changes_TRIVUS_NOT_PROMOTED": False,
            "changes_VUS_SR": False,
        },
        "next_action": "CLOSE_SPLIT_NO_GPU_DO_NOT_RESCUE_GEOMETRY_POST_HOC",
    }
    ADJUDICATION_PATH.write_text(json.dumps(adjudication, indent=2, sort_keys=True) + "\n")
    report = f"""# SPLIT Falsification-Crop Probe Report

Date: 2026-08-14

Status: `SPLIT_STOPPED_PRE_GPU_Z_K6_GEOMETRY_AND_Z_K7_LOW_N`

## Scope

SPLIT was preregistered as an exploratory pilot. It does not change F1, Q1, `TRIVUS_NOT_PROMOTED`, or VUS-SR, and it does not authorize a deployable method. No model forward or GPU authorization occurred.

## Zero-GPU gate

Z-G1 passed. Across 1,581 held-out ScreenSpot-Pro rows, nested selection chose $g=0.25$ in all five folds. The gate triggered on 1,187 rows (75.08%). There were 102 M2-only positive rows, giving pooled $\\Delta_2=6.45$ pp and an 8.59% conditional positive rate inside the gate.

This is candidate-level headroom only. It does not show that falsification-crop confidence can identify the M2-only rows.

## Geometry stop

The corrected matched-window audit retained 869/1,187 gate rows and failed 318/1,187, a 26.79% failure rate above the preregistered 15% Z-K6 limit. Exactly 163 rows failed only `W1_excludes_M2`; 155 failed only `W2_excludes_M1`. All $W_0$ exclusion checks, area/aspect matching, and Qwen3/GTA1 resize equality checks passed.

The failures occur when image boundaries prevent the fixed minimum 512-pixel window from extending away from the neighboring mode. The frozen protocol prohibits shrinking the window, changing the separation axis, or rescuing failed rows. Z-K6 therefore stops the round before GPU.

After geometry, only 76 positive rows remain, below the preregistered minimum 120. Z-K7 independently limits any continuation to an observational report.

## Endpoints and conclusion

Z-P3, Z-P1, Z-P2, Z-P4, and Z-P5 are all `NOT_RUN_PRE_GPU_STOP`. Qwen3 and GTA1 forward counts are zero; Qwen2.5 remains deferred because its checkpoint is absent. No balanced subset or GPU authorization was created.

The strongest defensible conclusion is:

> A two-mode candidate headroom of 6.45 pp exists under the frozen gate, but the preregistered falsification-crop geometry is infeasible often enough to trigger Z-K6, and the surviving positive set is too small for endpoint decisions. SPLIT provides no evidence that crop confidence is an orthogonal channel.

Any alternate crop geometry is a new study and cannot rescue SPLIT post hoc.
"""
    REPORT_PATH.write_text(report)
    print(json.dumps({
        "outcome": adjudication["outcome"],
        "Delta2": gate["pooled"]["Delta2"],
        "geometry_failure_rate": geometry["geometry_failure_rate"],
        "positive_rows_after_geometry": geometry["positive_rows_after_geometry"],
        "model_forward_count": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()