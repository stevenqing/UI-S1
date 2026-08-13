import json
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
AUTHORIZATION_PATH = RUN_DIR / "FORMAL_AUTHORIZATION.json"
RECEIPT_ROOT = RUN_DIR / "formal_authorization_receipts"
OUTPUT_ROOT = RUN_DIR / "formal"

EXPECTED_IMPLEMENTATION_FILES = {
    "AMENDMENT_014_PHYSICAL_OUTER_RUNNER.md",
    "configs/formal_runner_prereg.yaml",
    "trivus_fit.py",
    "trivus_thresholds.py",
    "trivus_outer.py",
    "formal_authorization.py",
    "launch_trivus.py",
    "finalize_trivus.py",
    "test_formal_primitives.py",
    "test_trivus_outer.py",
    "test_formal_authorization.py",
    "test_launch_trivus.py",
    "test_finalize_trivus.py",
    "context_common.py",
    "recovery_common.py",
    "configs/recovery.yaml",
}


def _helpers():
    from context_common import (
        committed_file, git_blob_sha256, require_commit_order, sha256_file,
    )

    return committed_file, git_blob_sha256, require_commit_order, sha256_file


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def attempt_path(nonce):
    if not is_sha256(nonce):
        raise PermissionError("TriVUS formal authorization nonce mismatch")
    return RUN_DIR / f".formal-attempt-{nonce}"


def receipt_path(nonce):
    if not is_sha256(nonce):
        raise PermissionError("TriVUS formal authorization nonce mismatch")
    return RECEIPT_ROOT / f"{nonce}.json"


def load_bound_authorization():
    if not AUTHORIZATION_PATH.is_file():
        raise PermissionError("TriVUS formal execution is not authorized")
    value = json.loads(AUTHORIZATION_PATH.read_text())
    expected_fields = {
        "schema_version", "status", "implementation_commit",
        "authorization_nonce", "result_must_not_exist", "outer_folds",
        "training_authorized", "files",
    }
    if (
        set(value) != expected_fields
        or value["schema_version"] != 1
        or value["status"] != "AUTHORIZED_TRIVUS_FORMAL_EXECUTION_ONCE"
        or value["result_must_not_exist"] is not True
        or value["training_authorized"] is not True
        or value["outer_folds"] != list(range(5))
        or not is_sha256(value["authorization_nonce"])
        or set(value["files"]) != EXPECTED_IMPLEMENTATION_FILES
    ):
        raise PermissionError("TriVUS invalid formal authorization")
    committed_file, git_blob_sha256, require_commit_order, sha256_file = _helpers()
    authorization_commit = committed_file(AUTHORIZATION_PATH)
    require_commit_order(
        value["implementation_commit"], authorization_commit,
        "formal implementation must strictly precede authorization",
    )
    for name, expected_hash in value["files"].items():
        path = RUN_DIR / name
        if not is_sha256(expected_hash) or sha256_file(path) != expected_hash:
            raise PermissionError(f"TriVUS formal implementation hash mismatch: {name}")
        if git_blob_sha256(value["implementation_commit"], path) != expected_hash:
            raise PermissionError(f"TriVUS formal implementation blob mismatch: {name}")
    return value, authorization_commit


def validate_new_authorization():
    value, authorization_commit = load_bound_authorization()
    receipt = receipt_path(value["authorization_nonce"])
    attempt = attempt_path(value["authorization_nonce"])
    if receipt.exists() or attempt.exists() or OUTPUT_ROOT.exists():
        raise FileExistsError("TriVUS formal execution already attempted or published")
    return value, authorization_commit, receipt, attempt


def consume_authorization(value, authorization_commit, receipt):
    from context_common import sha256_file, write_exclusive_json

    record = {
        "schema_version": 1,
        "status": "CONSUMED_TRIVUS_FORMAL_AUTHORIZATION",
        "authorization_sha256": sha256_file(AUTHORIZATION_PATH),
        "authorization_nonce": value["authorization_nonce"],
        "implementation_commit": value["implementation_commit"],
        "authorization_commit": authorization_commit,
        "outer_folds": list(range(5)),
        "training_authorized": True,
    }
    write_exclusive_json(receipt, record, list(record))
    return record


def validate_worker_receipt(path, outer_fold):
    from context_common import sha256_file

    authorization, authorization_commit = load_bound_authorization()
    expected_path = receipt_path(authorization["authorization_nonce"]).resolve()
    path = Path(path).resolve()
    if path != expected_path or not path.is_file():
        raise PermissionError("TriVUS formal worker receipt is not canonical")
    value = json.loads(path.read_text())
    expected = {
        "schema_version": 1,
        "status": "CONSUMED_TRIVUS_FORMAL_AUTHORIZATION",
        "authorization_sha256": sha256_file(AUTHORIZATION_PATH),
        "authorization_nonce": authorization["authorization_nonce"],
        "implementation_commit": authorization["implementation_commit"],
        "authorization_commit": authorization_commit,
        "outer_folds": list(range(5)),
        "training_authorized": True,
    }
    if value != expected or outer_fold not in value["outer_folds"]:
        raise PermissionError("TriVUS invalid formal worker authorization receipt")
    return value
