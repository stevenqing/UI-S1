import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CONFIG_PATH = RUN_DIR / "configs/ceil_prereg.yaml"
OUTPUT_PATH = RUN_DIR / "PREFLIGHT.json"
SEQUENTIAL_ROOT = ROOT / "runs/trivus/2026-08-13/sequential_exploratory"
XFER_ROOT = ROOT / "runs/xfer/2026-08-07"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path, sha256, size=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if size is not None and path.stat().st_size != int(size):
        raise ValueError(f"CEIL byte mismatch: {path}")
    if sha256_file(path) != sha256:
        raise ValueError(f"CEIL hash mismatch: {path}")


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if config.get("status") != "FROZEN_BEFORE_ANY_CEIL_RESULT":
        raise PermissionError("CEIL prereg status mismatch")
    spec = ROOT / config["canonical_spec"]["path"]
    verify(spec, config["canonical_spec"]["sha256"], config["canonical_spec"]["bytes"])
    for record in config["upstream"].values():
        verify(ROOT / record["path"], record["sha256"], record["bytes"])

    sequential_path = ROOT / config["upstream"]["sequential_manifest"]["path"]
    sequential = json.loads(sequential_path.read_text())
    if (
        sequential.get("status") != "PASS_EXPLORATORY_SEQUENTIAL_OOF_COMPLETE"
        or sequential.get("artifact_count") != 240
        or sequential.get("confirmatory") is not False
        or sequential.get("promotion_allowed") is not False
    ):
        raise PermissionError("CEIL sequential manifest boundary mismatch")
    for relative_path, record in sequential["artifacts"].items():
        verify(SEQUENTIAL_ROOT / relative_path, record["sha256"], record["bytes"])

    assembly_path = ROOT / config["upstream"]["assembly_prereg"]["path"]
    assembly = yaml.safe_load(assembly_path.read_text())
    for record in assembly["dependencies"].values():
        verify(ROOT / record["path"], record["sha256"])

    vus_private_path = ROOT / assembly["dependencies"]["vus_private_manifest"]["path"]
    vus_private = json.loads(vus_private_path.read_text())
    for record in vus_private["folds"].values():
        verify(vus_private_path.parent.parent / record["path"], record["sha256"])
    android_private_path = ROOT / assembly["dependencies"]["android_private_manifest"]["path"]
    android_private = json.loads(android_private_path.read_text())
    for record in android_private["folds"].values():
        verify(ROOT / record["path"], record["sha256"], record["bytes"])

    xfer_manifest_path = XFER_ROOT / "PUBLICATION_MANIFEST.json"
    xfer_manifest = json.loads(xfer_manifest_path.read_text())
    if xfer_manifest.get("artifact_count") != 47:
        raise PermissionError("CEIL XFER publication manifest mismatch")
    for relative_path, record in xfer_manifest["artifacts"].items():
        verify(XFER_ROOT / relative_path, record["sha256"], record["bytes"])

    result = {
        "schema_version": 1,
        "status": "PASS_CEIL_INPUT_PREFLIGHT",
        "gpu_used": False,
        "arm_A_statistics_computed": False,
        "arm_B_recoverable_subset_computed": False,
        "arm_B_AUROC_computed": False,
        "sequential_publication": {
            "manifest_path": sequential_path.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256_file(sequential_path),
            "artifact_count": 240,
            "all_artifacts_reverified": True,
            "total_bytes": sum(record["bytes"] for record in sequential["artifacts"].values()),
        },
        "trivus_dependencies": {
            "assembly_path": assembly_path.relative_to(ROOT).as_posix(),
            "dependency_count": len(assembly["dependencies"]),
            "all_dependencies_reverified": True,
            "vus_private_rows": int(vus_private["records"]),
            "android_private_rows": int(android_private["records"]),
            "private_shards": len(vus_private["folds"]) + len(android_private["folds"]),
        },
        "xfer_publication": {
            "manifest_path": xfer_manifest_path.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256_file(xfer_manifest_path),
            "artifact_count": 47,
            "all_artifacts_reverified": True,
            "total_bytes": sum(record["bytes"] for record in xfer_manifest["artifacts"].values()),
        },
        "blind_visual_source": {
            "field": "label_probabilities",
            "mapping": "display_to_candidate_via_restore_visual_values",
            "feature_column_inference": False,
        },
        "model_roles": {
            "Qwen2.5-VL-7B-Instruct": "SPLIT_DEFERRED_MISSING_CHECKPOINT_NOT_USED",
            "UI-TARS-7B-SFT": "SCREENSPOT_BANK_LINEAGE_NO_FORWARD",
            "conflict": False,
        },
        "environment": {
            "python_executable": sys.executable,
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "scipy": importlib.metadata.version("scipy"),
            "matplotlib": importlib.metadata.version("matplotlib"),
            "pyyaml": importlib.metadata.version("PyYAML"),
        },
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if json.loads(OUTPUT_PATH.read_text()) != result:
        raise ValueError("CEIL preflight readback mismatch")
    print(json.dumps({
        "status": result["status"],
        "sequential_artifacts": 240,
        "sequential_bytes": result["sequential_publication"]["total_bytes"],
        "private_shards": result["trivus_dependencies"]["private_shards"],
        "xfer_artifacts": 47,
        "model_roles": result["model_roles"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()