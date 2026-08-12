import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CONFIG_PATH = RUN_DIR / "configs/real_data_smoke_prereg.yaml"
sys.path.insert(0, str(RUN_DIR))

from context_common import (
    atomic_json_file, committed_file, git_blob_sha256, require_commit_order,
    write_exclusive_json,
)
from recovery_common import assert_protected_process, load_config as load_recovery_config, sha256_file
from trivus_assembly import (
    assemble_phase_data, load_config as load_assembly_config,
    load_context_manifest, load_context_phase, load_locked_public_inputs,
    with_model_weights,
)
from trivus_data import FAMILIES, INPUT_DIMENSION, fit_standardizer, torch_batch
from trivus_model import TriVUSSetRanker


def load_config():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    phase = config.get("phase", {})
    if (
        config.get("status") != "FROZEN_AFTER_ASSEMBLY_IMPLEMENTATION_BEFORE_REAL_PRIVATE_SMOKE"
        or config.get("python") != ".venv-scaleup/bin/python"
        or config.get("seed") != 20260822
        or phase != {
            "outer_fold": 0,
            "role": "inner",
            "holdout_fold": 1,
            "checkpoint_fold": 2,
            "model_training_folds": [3, 4],
            "expected_rows": {
                "model_training": 7428,
                "checkpoint": 3838,
                "oof": 3666,
            },
        }
        or config.get("variant") != "JOINT3"
        or config.get("maximum_forward_rows") != 64
    ):
        raise ValueError("TriVUS real-data smoke contract mismatch")
    if Path(sys.executable).absolute() != (ROOT / config["python"]).absolute():
        raise RuntimeError("TriVUS real-data smoke interpreter mismatch")
    for item in config["dependencies"].values():
        if item["sha256"] == "TO_BE_FROZEN":
            raise ValueError("TriVUS smoke dependency contains an unresolved placeholder")
        if sha256_file(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"TriVUS smoke dependency mismatch: {item['path']}")
    return config


def authorization_receipt_path(config, authorization):
    nonce = authorization.get("authorization_nonce")
    if not isinstance(nonce, str) or len(nonce) != 64 or any(
        character not in "0123456789abcdef" for character in nonce
    ):
        raise PermissionError("TriVUS smoke authorization nonce mismatch")
    return ROOT / config["authorization_receipts"] / f"{nonce}.json"


def validate_authorization(config):
    path = ROOT / config["authorization"]
    if not path.is_file():
        raise PermissionError("TriVUS real-data smoke is not authorized")
    value = json.loads(path.read_text())
    if (
        value.get("status") != "AUTHORIZED_TRIVUS_REAL_DATA_SMOKE_ONCE"
        or value.get("result_must_not_exist") is not True
        or value.get("optimizer_authorized") is not False
        or value.get("training_authorized") is not False
    ):
        raise PermissionError("TriVUS invalid smoke authorization")
    authorization_commit = committed_file(path)
    require_commit_order(
        value["implementation_commit"], authorization_commit,
        "smoke implementation must strictly precede authorization",
    )
    expected = {
        "AMENDMENT_012_NO_OPTIMIZER_REAL_DATA_SMOKE.md",
        "configs/real_data_smoke_prereg.yaml",
        "run_real_data_smoke.py",
        "test_real_data_smoke.py",
    }
    if set(value.get("files", {})) != expected:
        raise PermissionError("TriVUS smoke authorization file set mismatch")
    for name, expected_hash in value["files"].items():
        path = RUN_DIR / name
        if sha256_file(path) != expected_hash:
            raise PermissionError(f"TriVUS smoke implementation hash mismatch: {name}")
        if git_blob_sha256(value["implementation_commit"], path) != expected_hash:
            raise PermissionError(f"TriVUS smoke implementation blob mismatch: {name}")
    output = ROOT / config["output"]
    receipt = authorization_receipt_path(config, value)
    if output.exists() or receipt.exists():
        raise FileExistsError("TriVUS real-data smoke already attempted")
    return value, authorization_commit


def consume_authorization(config, authorization):
    path = authorization_receipt_path(config, authorization)
    value = {
        "schema_version": 1,
        "status": "CONSUMED_TRIVUS_REAL_DATA_SMOKE_AUTHORIZATION",
        "authorization_sha256": sha256_file(ROOT / config["authorization"]),
        "authorization_nonce": authorization["authorization_nonce"],
        "implementation_commit": authorization["implementation_commit"],
        "prereg_sha256": sha256_file(CONFIG_PATH),
    }
    schema = [
        "schema_version", "status", "authorization_sha256",
        "authorization_nonce", "implementation_commit", "prereg_sha256",
    ]
    write_exclusive_json(path, value, schema)
    return path, value


def opened_provenance(paths):
    return [
        {
            "path": str(Path(path).relative_to(ROOT)),
            "sha256": sha256_file(path),
        }
        for path in sorted(paths)
    ]


def metric_free_smoke_loss(model, batch):
    utility, fallback_logit = model(
        batch.features, batch.candidate_mask, batch.fallback_indices
    )
    log_probabilities = torch.log_softmax(utility, dim=-1)
    token_mask = torch.cat((
        batch.candidate_mask,
        torch.ones(
            (len(batch.features), 1),
            dtype=torch.bool,
            device=batch.features.device,
        ),
    ), dim=1)
    safe_log_probabilities = torch.where(
        token_mask, log_probabilities, torch.zeros_like(log_probabilities)
    )
    listwise = -(batch.target_distribution * safe_log_probabilities).sum(dim=-1)
    auxiliary = torch.nn.functional.binary_cross_entropy_with_logits(
        fallback_logit, batch.fallback_correct, reduction="none"
    )
    denominator = batch.weights.sum().clamp_min(torch.finfo(listwise.dtype).eps)
    return ((listwise + 0.5 * auxiliary) * batch.weights).sum() / denominator


def build_result(config, authorization, authorization_commit, receipt_path, receipt, opened):
    result = {
        "schema_version": 1,
        "status": "PASS_TRIVUS_NO_OPTIMIZER_REAL_DATA_SMOKE",
        "phase": config["phase"],
        "public_row_counts": config["phase"]["expected_rows"],
        "opened_private_files": opened,
        "checks": {
            "public_and_blind_inputs_valid": True,
            "full_context_bank_and_phase_valid": True,
            "physical_fold_scoping_valid": True,
            "targets_valid": True,
            "weights_valid": True,
            "standardizer_finite": True,
            "checkpoint_transform_finite": True,
            "oof_transform_finite": True,
            "forward_finite": True,
            "loss_finite": True,
        },
        "implementation_commit": authorization["implementation_commit"],
        "authorization_commit": authorization_commit,
        "authorization_nonce": authorization["authorization_nonce"],
        "authorization_sha256": receipt["authorization_sha256"],
        "authorization_receipt": str(receipt_path.relative_to(ROOT)),
        "authorization_receipt_sha256": sha256_file(receipt_path),
        "optimizer_constructed": False,
        "backward_called": False,
        "performance_metric_computed": False,
        "training_started": False,
    }
    if set(result) != set(config["result_allowed_fields"]):
        raise ValueError("TriVUS smoke result schema mismatch")
    return result


def main():
    config = load_config()
    authorization, authorization_commit = validate_authorization(config)
    recovery = load_recovery_config()
    assert_protected_process(recovery)
    receipt_path, receipt = consume_authorization(config, authorization)

    assembly_config = load_assembly_config()
    public, predictions = load_locked_public_inputs(assembly_config)
    context_manifest = load_context_manifest(assembly_config)
    phase = load_context_phase(
        ROOT / assembly_config["dependencies"]["contexts"]["path"],
        context_manifest,
        public,
        outer_fold=0,
        role="inner",
        expected_fold_counts=assembly_config["expected"]["context_records_by_public_fold"],
        holdout_fold=1,
    )
    training, training_opened = assemble_phase_data(
        assembly_config, public, predictions, phase, (3, 4)
    )
    checkpoint, checkpoint_opened = assemble_phase_data(
        assembly_config, public, predictions, phase, (2,)
    )
    oof, oof_opened = assemble_phase_data(
        assembly_config, public, predictions, phase, (1,)
    )
    expected = config["phase"]["expected_rows"]
    if (
        len(training) != expected["model_training"]
        or len(checkpoint) != expected["checkpoint"]
        or len(oof) != expected["oof"]
    ):
        raise ValueError("TriVUS smoke row-count mismatch")

    weighted_training = with_model_weights(training, "JOINT3")
    standardizer = fit_standardizer(
        weighted_training, "JOINT3", included_families=FAMILIES
    )
    standardized_training = standardizer.transform(weighted_training)
    standardized_checkpoint = standardizer.transform(checkpoint)
    standardized_oof = standardizer.transform(oof)
    if (
        not np.isfinite(standardizer.mean).all()
        or not np.isfinite(standardizer.scale).all()
        or not np.isfinite(standardized_checkpoint.features).all()
        or not np.isfinite(standardized_oof.features).all()
    ):
        raise ValueError("TriVUS smoke standardization mismatch")

    indices = np.flatnonzero(standardized_training.weights > 0)[:config["maximum_forward_rows"]]
    if not len(indices):
        raise ValueError("TriVUS smoke has no positive-weight rows")
    batch = torch_batch(standardized_training, indices, torch.device("cpu"))
    torch.manual_seed(config["seed"])
    model = TriVUSSetRanker(
        INPUT_DIMENSION,
        width=64,
        heads=4,
        layers=2,
        dropout=0.1,
    ).eval()
    with torch.no_grad():
        loss = metric_free_smoke_loss(model, batch)
    if not bool(torch.isfinite(loss).item()):
        raise ValueError("TriVUS smoke loss is non-finite")

    opened = opened_provenance(
        set(training_opened) | set(checkpoint_opened) | set(oof_opened)
    )
    if len(opened) != 8:
        raise ValueError("TriVUS smoke opened-file provenance mismatch")
    assert_protected_process(recovery)
    result = build_result(
        config, authorization, authorization_commit,
        receipt_path, receipt, opened,
    )
    atomic_json_file(ROOT / config["output"], result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()