import hashlib
import json
import os
from pathlib import Path

import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CONFIG_PATH = RUN_DIR / "configs/evid_prereg.yaml"
SPEC_PATH = RUN_DIR / "SPEC.md"
OUTPUT_PATH = RUN_DIR / "PREFLIGHT.json"

DEPENDENCIES = {
    "decomp_arm1": ROOT / "runs/decomp/2026-08-14/ARM1.json",
    "decomp_preflight": ROOT / "runs/decomp/2026-08-14/PREFLIGHT.json",
    "decomp_status": ROOT / "runs/decomp/2026-08-14/STATUS.json",
    "mask_common": ROOT / "runs/mask/2026-08-14/mask_common.py",
    "canonical_aggregator": ROOT / "runs/ccm-h2h/2026-07-31/h1/aggregators_coord.py",
    "sourcebias_common": ROOT / "runs/sourcebias/2026-08-03/sourcebias_common.py",
    "kappa_anchor": ROOT / "runs/ccm-h2h/2026-07-31/h2_collision_floor.json",
    "close_e1": ROOT / "runs/close/2026-08-08/e1_arm_aggregator_matrix.py",
    "generation_trace_policy": ROOT / "docs/generation_trace_retention_policy.md",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if config["status"] != "PREREGISTERED_BEFORE_ANY_EVID_RESULT":
        raise PermissionError("EVID preregistration mismatch")
    decomp = json.loads(DEPENDENCIES["decomp_arm1"].read_text())
    decomp_preflight = json.loads(DEPENDENCIES["decomp_preflight"].read_text())
    kappa = json.loads(DEPENDENCIES["kappa_anchor"].read_text())["summary"]["primary"]
    if (
        decomp["status"] != "PASS_DECOMP_ARM1_COMPLETE"
        or decomp["rows"] != 1581
        or decomp["anchors"]["full_pool_density_B3"] != config["method"]["rho_zero_anchor"]["expected_accuracy"]
        or decomp_preflight["status"] != "PASS_DECOMP_PREFLIGHT_NO_ARM_STARTED"
        or abs(kappa["view_axis_mean_kappa"] - 0.8946701841469047) > 1e-15
        or abs(kappa["cross_family_mean_kappa"] - 0.3981704074778036) > 1e-15
    ):
        raise ValueError("EVID input anchor mismatch")
    dependencies = {
        name: {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in DEPENDENCIES.items()
    }
    output = {
        "schema_version": 1,
        "status": "PASS_EVID_PREFLIGHT_NO_STAGE_RESULT",
        "gpu_used": False,
        "stage0_computed": False,
        "stage1_computed": False,
        "stage2_authorized": False,
        "spec": {"path": str(SPEC_PATH.relative_to(ROOT)), "sha256": sha256_file(SPEC_PATH)},
        "config": {"path": str(CONFIG_PATH.relative_to(ROOT)), "sha256": sha256_file(CONFIG_PATH)},
        "dependencies": dependencies,
        "bank": {"benchmark": "screenspot_pro", "rows": 1581, "candidates_per_row": 12, "canonical_order": "view_major_then_lineage"},
        "rho_anchors": {
            "configured": {"rho_v": config["method"]["primary"]["rho_v"], "rho_l": config["method"]["primary"]["rho_l"]},
            "exact_provenance": {"view_failure_kappa": kappa["view_axis_mean_kappa"], "cross_family_failure_kappa": kappa["cross_family_mean_kappa"]},
            "mapping_status": "FROZEN_HEURISTIC_NOT_VALIDATED_ICC",
        },
        "mandatory_baselines": config["stage1"]["mandatory_baselines"],
        "mind2web_status": "BLOCKED_ALIGNED_POOL_UNAVAILABLE",
    }
    atomic_json(OUTPUT_PATH, output)
    print(json.dumps({"status": output["status"], "rows": 1581, "stage0_computed": False, "stage2_authorized": False}, indent=2))


if __name__ == "__main__":
    main()