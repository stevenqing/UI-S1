import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CONFIG_PATH = RUN_DIR / "configs/mask_prereg.yaml"
OUTPUT_PATH = RUN_DIR / "PREFLIGHT.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_record(path, record):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"MASK byte mismatch: {path}")
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"MASK hash mismatch: {path}")


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if config.get("status") != "FROZEN_BEFORE_ANY_MASK_RESULT_OR_FORWARD":
        raise PermissionError("MASK prereg status mismatch")
    spec_path = ROOT / config["canonical_spec"]["path"]
    if sha256_file(spec_path) != config["canonical_spec"]["sha256"]:
        raise PermissionError("MASK spec hash mismatch")
    for record in config["upstream"].values():
        verify_record(ROOT / record["path"], record)

    gran_path = ROOT / config["upstream"]["gran_input_manifest"]["path"]
    gran = json.loads(gran_path.read_text())
    if (
        gran.get("status") != "LOCKED_BEFORE_GRAN_LABEL_STATISTICS_AND_TAU_SWEEP"
        or gran.get("gpu_used") is not False
        or gran.get("label_statistics_computed") is not False
    ):
        raise PermissionError("MASK GRAN manifest boundary mismatch")
    for relative_path, record in gran["files"].items():
        verify_record(ROOT / relative_path, record)

    split_path = ROOT / config["upstream"]["split_preflight"]["path"]
    split = json.loads(split_path.read_text())
    if (
        split.get("status")
        != "PASS_SPLIT_PREFLIGHT_QWEN3_GTA1_READY_QWEN25_DEFERRED"
        or split.get("screenspot_rows") != 1581
    ):
        raise PermissionError("MASK SPLIT preflight boundary mismatch")
    for record in split["images"].values():
        verify_record(ROOT / record["path"], record)

    gta1 = split["models"]["GTA1-7B"]
    if gta1.get("status") != "READY":
        raise PermissionError("MASK GTA1 is not ready")
    verify_record(ROOT / gta1["index"]["path"], gta1["index"])
    for record in gta1["shards"]:
        verify_record(ROOT / record["path"], record)
    if gta1["revision"] != config["model"]["revision"]:
        raise PermissionError("MASK GTA1 revision mismatch")

    role_audit = {
        "GTA1-7B": {
            "role": "ONLY_MASK_FORWARD_MODEL",
            "status": "READY",
            "path": gta1["path"],
            "revision": gta1["revision"],
            "index_sha256": gta1["index"]["sha256"],
            "weight_shards": len(gta1["shards"]),
            "weight_bytes": gta1["total_weight_bytes"],
        },
        "UI-TARS-7B-SFT": {
            "role": "CANDIDATE_BANK_LINEAGE_ONLY",
            "status": "LOCKED_BY_GRAN_INPUT_MANIFEST",
            "MASK_forward_allowed": False,
        },
        "Qwen2.5-VL-7B-Instruct": {
            "role": "SPLIT_DEFERRED_CHECKPOINT_NOT_IN_MASK",
            "status": split["models"]["Qwen2.5-VL-7B-Instruct"]["status"],
            "MASK_forward_allowed": False,
        },
    }
    result = {
        "schema_version": 1,
        "status": "PASS_MASK_PREFLIGHT_GTA1_READY_MODEL_ROLES_RESOLVED",
        "gpu_used": False,
        "mask_statistics_computed": False,
        "mask_constructed": False,
        "model_forward_started": False,
        "subset_manifest_created": False,
        "gpu_authorization_created": False,
        "gran_manifest": {
            "path": config["upstream"]["gran_input_manifest"]["path"],
            "sha256": sha256_file(gran_path),
            "file_count": gran["file_count"],
            "total_bytes": gran["total_bytes"],
            "role_counts": gran["role_counts"],
            "all_files_reverified": True,
        },
        "images": {
            "source_manifest": config["upstream"]["split_preflight"]["path"],
            "rows": len(split["images"]),
            "total_bytes": sum(record["bytes"] for record in split["images"].values()),
            "all_files_reverified": True,
        },
        "model_roles": role_audit,
        "environment": {
            "python_executable": sys.executable,
            "python": platform.python_version(),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "numpy": importlib.metadata.version("numpy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "pillow": importlib.metadata.version("Pillow"),
            "pyyaml": importlib.metadata.version("PyYAML"),
        },
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if json.loads(OUTPUT_PATH.read_text()) != result:
        raise ValueError("MASK preflight readback mismatch")
    print(json.dumps({
        "status": result["status"],
        "gran_files": result["gran_manifest"]["file_count"],
        "image_rows": result["images"]["rows"],
        "model_roles": {
            name: record["role"] for name, record in role_audit.items()
        },
        "gta1_weight_bytes": gta1["total_weight_bytes"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()