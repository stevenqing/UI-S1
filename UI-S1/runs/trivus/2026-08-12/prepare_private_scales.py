import json
import math
import subprocess
import sys
from pathlib import Path

from PIL import Image


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(RUN_DIR))

from context_common import (
    CONFIG_PATH, atomic_json_file, committed_file, git_blob_sha256, load_jsonl,
    load_prereg, publish_directory, repository_path, require_commit_order,
    sha256_rows, staging_directory, write_exclusive_json, write_jsonl_atomic,
)
from recovery_common import assert_protected_process, load_config as load_recovery_config, sha256_file


def validate_authorization(config):
    path = ROOT / config["output"]["private_scale_authorization"]
    if not path.is_file():
        raise PermissionError("TriVUS private-scale sealing is not implementation-authorized")
    value = json.loads(path.read_text())
    if (
        value.get("status") != "AUTHORIZED_TRIVUS_PRIVATE_SCALE_SEAL_ONCE"
        or value.get("result_must_not_exist") is not True
    ):
        raise PermissionError("TriVUS invalid private-scale authorization")
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", value["implementation_commit"], "HEAD"],
        cwd=ROOT, check=False,
    )
    if completed.returncode:
        raise PermissionError("TriVUS private-scale implementation commit is not an ancestor")
    authorization_commit = committed_file(path)
    require_commit_order(
        value["implementation_commit"], authorization_commit,
        "private-scale implementation must precede authorization",
    )
    expected = {
        "configs/fallback_contexts_prereg.yaml", "context_common.py",
        "prepare_private_scales.py", "prepare_fallback_contexts.py",
        "test_fallback_contexts.py",
    }
    if set(value.get("files", {})) != expected:
        raise PermissionError("TriVUS private-scale authorization file set mismatch")
    for name, expected_hash in value["files"].items():
        if sha256_file(RUN_DIR / name) != expected_hash:
            raise PermissionError(f"TriVUS private-scale implementation hash mismatch: {name}")
        if git_blob_sha256(value["implementation_commit"], RUN_DIR / name) != expected_hash:
            raise PermissionError(f"TriVUS private-scale implementation blob mismatch: {name}")
    output_directory = (ROOT / config["output"]["private_scale_manifest"]).parent
    receipt = authorization_receipt_path(config, value)
    if output_directory.exists() or receipt.exists():
        raise FileExistsError("TriVUS private-scale seal already exists")
    return value, authorization_commit


def authorization_receipt_path(config, authorization):
    nonce = authorization.get("authorization_nonce")
    if not isinstance(nonce, str) or len(nonce) != 64 or any(
        character not in "0123456789abcdef" for character in nonce
    ):
        raise PermissionError("TriVUS private-scale authorization nonce mismatch")
    return ROOT / config["output"]["private_scale_authorization_receipts"] / f"{nonce}.json"


def consume_authorization(config, authorization):
    path = authorization_receipt_path(config, authorization)
    value = {
        "schema_version": 1,
        "status": "CONSUMED_TRIVUS_PRIVATE_SCALE_AUTHORIZATION",
        "authorization_sha256": sha256_file(
            ROOT / config["output"]["private_scale_authorization"]
        ),
        "authorization_nonce": authorization["authorization_nonce"],
        "implementation_commit": authorization["implementation_commit"],
        "prereg_sha256": sha256_file(CONFIG_PATH),
    }
    write_exclusive_json(path, value, config["authorization_receipt_schema"])
    return path, value


def build_private_scales(config):
    public_rows = load_jsonl(ROOT / config["dependencies"]["vus_public"]["path"])
    mind_public = {
        row["row_id"]: row for row in public_rows
        if row["benchmark"] == "mind2web" and row["arm"] == "C_uni"
    }
    source_rows = load_jsonl(ROOT / config["dependencies"]["mind_targets"]["path"])
    source = {row["id"]: row for row in source_rows}
    if len(mind_public) != 2080 or set(source) != set(mind_public):
        raise ValueError("TriVUS private-scale source identity mismatch")
    output = {fold: [] for fold in range(5)}
    for row_id in sorted(mind_public):
        public = mind_public[row_id]
        target = source[row_id]
        image_path = repository_path(public["image_path"])
        if sha256_file(image_path) != public["image_sha256"]:
            raise ValueError(f"TriVUS private-scale image mismatch: {row_id}")
        with Image.open(image_path) as image:
            width, height = image.size
        bbox = target["step"]["bbox"]
        values = (float(bbox["width"]) / width, float(bbox["height"]) / height)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError(f"TriVUS invalid private scale: {row_id}")
        output[int(public["fold"])].append({
            "schema_version": 1,
            "row_id": row_id,
            "normalized_width": values[0],
            "normalized_height": values[1],
        })
    return output


def main():
    config = load_prereg()
    authorization, authorization_commit = validate_authorization(config)
    recovery = load_recovery_config()
    assert_protected_process(recovery)
    receipt_path, receipt = consume_authorization(config, authorization)
    manifest_path = ROOT / config["output"]["private_scale_manifest"]
    output_directory = manifest_path.parent
    if output_directory.exists():
        raise FileExistsError("TriVUS private-scale seal already exists")
    values = build_private_scales(config)
    with staging_directory(output_directory) as staging:
        folds = {}
        for fold in range(5):
            path = staging / f"private_scales_fold-{fold}.jsonl"
            rows = values[fold]
            expected = config["expected"]["mind_scale_rows_by_fold"][fold]
            if len(rows) != expected:
                raise ValueError(f"TriVUS private-scale fold coverage mismatch: {fold}/{len(rows)}")
            write_jsonl_atomic(path, rows)
            folds[str(fold)] = {
                "path": str((output_directory / path.name).relative_to(ROOT)),
                "rows": len(rows),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "row_ids_sha256": sha256_rows(row["row_id"] for row in rows),
            }
        manifest = {
            "schema_version": 1,
            "status": "PASS_TRIVUS_PHYSICALLY_SEALED_PRIVATE_SCALES",
            "schema": config["private_scale_schema"],
            "records": sum(item["rows"] for item in folds.values()),
            "folds": folds,
            "prereg_sha256": sha256_file(CONFIG_PATH),
            "source_public_sha256": config["dependencies"]["vus_public"]["sha256"],
            "source_targets_sha256": config["dependencies"]["mind_targets"]["sha256"],
            "authorization_sha256": sha256_file(
                ROOT / config["output"]["private_scale_authorization"]
            ),
            "authorization_receipt": str(receipt_path.relative_to(ROOT)),
            "authorization_receipt_sha256": sha256_file(receipt_path),
            "authorization_nonce": receipt["authorization_nonce"],
            "implementation_commit": authorization["implementation_commit"],
            "authorization_commit": authorization_commit,
            "environment": config["environment"],
            "aggregate_target_statistics_computed": False,
            "candidate_success_opened": False,
            "training_started": False,
        }
        atomic_json_file(staging / manifest_path.name, manifest)
        assert_protected_process(recovery)
        publish_directory(staging, output_directory)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()