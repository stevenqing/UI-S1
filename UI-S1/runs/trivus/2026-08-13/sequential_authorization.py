import json
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
PRIOR_DIR = RUN_DIR.parent / "2026-08-12"
sys.path.insert(0, str(PRIOR_DIR))
AUTHORIZATION_PATH = RUN_DIR / "SEQUENTIAL_OPTIMIZER_AUTHORIZATION.json"
RECEIPT_ROOT = RUN_DIR / "sequential_authorization_receipts"
OUTPUT_ROOT = RUN_DIR / "sequential_exploratory"

EXPECTED_IMPLEMENTATION_FILES = {
    "CORRECTION_001_VERIFIER_CHECKPOINT_SPLIT.md",
    "CORRECTION_002_POSITIVE_WEIGHT_BATCHING.md",
    "CORRECTION_003_GLOBAL_FIT_SCOPE_WEIGHTING.md",
    "configs/sequential_training_prereg.yaml",
    "sequential_model.py",
    "sequential_fit.py",
    "sequential_oof_runner.py",
    "sequential_verifier_oof_runner.py",
    "sequential_authorization.py",
    "launch_sequential.py",
    "test_sequential_authorization.py",
    "test_launch_sequential.py",
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
        raise PermissionError("Sequential authorization nonce mismatch")
    return RUN_DIR / f".sequential-attempt-{nonce}"


def receipt_path(nonce):
    if not is_sha256(nonce):
        raise PermissionError("Sequential authorization nonce mismatch")
    return RECEIPT_ROOT / f"{nonce}.json"


def load_bound_authorization():
    if not AUTHORIZATION_PATH.is_file():
        raise PermissionError("Sequential exploratory optimizer is not authorized")
    value = json.loads(AUTHORIZATION_PATH.read_text())
    if (
        set(value) != {
            "schema_version", "status", "implementation_commit",
            "authorization_nonce", "result_must_not_exist", "outer_folds",
            "families", "phases", "training_authorized", "confirmatory",
            "promotion_allowed", "files",
        }
        or value["schema_version"] != 1
        or value["status"] != "AUTHORIZED_EXPLORATORY_SEQUENTIAL_OPTIMIZER_ONCE"
        or value["result_must_not_exist"] is not True
        or value["outer_folds"] != list(range(5))
        or value["families"] != ["mind2web", "screenspot_pro", "androidcontrol"]
        or value["phases"] != ["cheap", "verifier"]
        or value["training_authorized"] is not True
        or value["confirmatory"] is not False
        or value["promotion_allowed"] is not False
        or not is_sha256(value["authorization_nonce"])
        or set(value["files"]) != EXPECTED_IMPLEMENTATION_FILES
    ):
        raise PermissionError("Invalid sequential optimizer authorization")
    committed_file, git_blob_sha256, require_commit_order, sha256_file = _helpers()
    authorization_commit = committed_file(AUTHORIZATION_PATH)
    require_commit_order(
        value["implementation_commit"], authorization_commit,
        "sequential implementation must strictly precede authorization",
    )
    for name, expected_hash in value["files"].items():
        path = RUN_DIR / name
        if not is_sha256(expected_hash) or sha256_file(path) != expected_hash:
            raise PermissionError(f"Sequential implementation hash mismatch: {name}")
        if git_blob_sha256(value["implementation_commit"], path) != expected_hash:
            raise PermissionError(f"Sequential implementation blob mismatch: {name}")
    return value, authorization_commit


def validate_new_authorization():
    value, commit = load_bound_authorization()
    receipt = receipt_path(value["authorization_nonce"])
    attempt = attempt_path(value["authorization_nonce"])
    if receipt.exists() or attempt.exists() or OUTPUT_ROOT.exists():
        raise FileExistsError("Sequential exploratory execution already attempted or published")
    return value, commit, receipt, attempt


def consume_authorization(value, authorization_commit, receipt):
    from context_common import sha256_file, write_exclusive_json
    record = {
        "schema_version": 1,
        "status": "CONSUMED_EXPLORATORY_SEQUENTIAL_OPTIMIZER",
        "authorization_sha256": sha256_file(AUTHORIZATION_PATH),
        "authorization_nonce": value["authorization_nonce"],
        "implementation_commit": value["implementation_commit"],
        "authorization_commit": authorization_commit,
        "outer_folds": list(range(5)),
        "families": value["families"],
        "phases": value["phases"],
        "training_authorized": True,
        "confirmatory": False,
        "promotion_allowed": False,
    }
    write_exclusive_json(receipt, record, list(record))
    return record


def validate_worker_receipt(path, outer_fold, holdout_fold, family, phase):
    from context_common import sha256_file
    authorization, authorization_commit = load_bound_authorization()
    expected_path = receipt_path(authorization["authorization_nonce"]).resolve()
    path = Path(path).resolve()
    if path != expected_path or not path.is_file():
        raise PermissionError("Sequential worker receipt is not canonical")
    value = json.loads(path.read_text())
    if (
        value.get("status") != "CONSUMED_EXPLORATORY_SEQUENTIAL_OPTIMIZER"
        or value.get("authorization_sha256") != sha256_file(AUTHORIZATION_PATH)
        or value.get("authorization_commit") != authorization_commit
        or value.get("implementation_commit") != authorization["implementation_commit"]
        or outer_fold not in value.get("outer_folds", ())
        or holdout_fold not in range(5)
        or holdout_fold == outer_fold
        or family not in value.get("families", ())
        or phase not in value.get("phases", ())
        or value.get("training_authorized") is not True
        or value.get("confirmatory") is not False
        or value.get("promotion_allowed") is not False
    ):
        raise PermissionError("Invalid sequential worker receipt")
    return value