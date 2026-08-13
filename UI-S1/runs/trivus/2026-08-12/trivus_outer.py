import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CONFIG_PATH = RUN_DIR / "configs/formal_runner_prereg.yaml"
OUTPUT_ROOT = RUN_DIR / "formal"
sys.path.insert(0, str(RUN_DIR))

from context_common import (
    atomic_json_file, fsync_directory, safe_child_path, sha256_file,
    write_exclusive_json,
)
from recovery_common import assert_protected_process, load_config as load_recovery_config
from formal_authorization import attempt_path, validate_worker_receipt
from trivus_assembly import (
    assemble_phase_data, load_config as load_assembly_config, load_context_manifest,
    load_context_phase, load_locked_public_inputs, phase_contract,
)
from trivus_data import TriVUSStandardizer
from trivus_fit import (
    MODEL_SPECS, half_up_median, make_model, model_spec, predict_data,
    train_fixed_epochs, train_with_checkpoint,
)
from trivus_thresholds import (
    apply_selected_thresholds, compose_target_only, select_thresholds,
)


POLICY_SPECS = {
    "JOINT3": ("JOINT3", ("mind2web", "screenspot_pro", "androidcontrol")),
    "TARGET_ONLY": ("TARGET_ONLY", ("mind2web", "screenspot_pro", "androidcontrol")),
    "JOINT2_NO_ANDROID": ("JOINT2_NO_ANDROID", ("mind2web", "screenspot_pro")),
    "NO_VISUAL": ("NO_VISUAL", ("mind2web", "screenspot_pro", "androidcontrol")),
    "RANDOM_ID_PLACEBO": ("RANDOM_ID_PLACEBO", ("mind2web", "screenspot_pro", "androidcontrol")),
}


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def load_config():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if config.get("status") != "FROZEN_AFTER_REAL_DATA_SMOKE_BEFORE_ANY_TRIVUS_OPTIMIZER_STEP":
        raise ValueError("TriVUS formal runner protocol is not frozen")
    if set(config.get("model_specs", {})) != set(MODEL_SPECS):
        raise ValueError("TriVUS formal model-spec roster mismatch")
    for spec_id, expected in MODEL_SPECS.items():
        item = config["model_specs"][spec_id]
        observed = (
            item["variant"], item.get("target_family"), tuple(item["families"])
        )
        if observed != expected:
            raise ValueError(f"TriVUS formal model-spec mismatch: {spec_id}")
    if config["optimizer"]["optimizer_steps_per_epoch"] != 1:
        raise ValueError("TriVUS formal optimizer-step contract mismatch")
    for item in config["dependencies"].values():
        if sha256_file(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"TriVUS formal dependency mismatch: {item['path']}")
    return config


def standardizer_payload(standardizer):
    return {
        "variant": standardizer.variant,
        "mean": standardizer.mean.tolist(),
        "scale": standardizer.scale.tolist(),
    }


def write_final_artifact(path, artifact_root, model, standardizer, spec_id, epochs, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "schema_version": 1,
        "spec_id": spec_id,
        "epochs": int(epochs),
        "seed": int(seed),
        "state_dict": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
        "standardizer": standardizer_payload(standardizer),
    }, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)
    fsync_directory(path.parent)
    return {
        "relative_path": str(path.relative_to(artifact_root)),
        "sha256": sha256_file(path),
    }


def load_final_artifact(
    item, artifact_root, spec_id, expected_epochs, expected_seed, device,
):
    if set(item) != {"relative_path", "sha256"}:
        raise PermissionError(f"TriVUS final artifact reference mismatch: {spec_id}")
    path = safe_child_path(artifact_root, item["relative_path"])
    if sha256_file(path) != item["sha256"]:
        raise PermissionError(f"TriVUS final artifact hash mismatch: {spec_id}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (
        set(value) != {
            "schema_version", "spec_id", "epochs", "seed", "state_dict",
            "standardizer",
        }
        or value.get("schema_version") != 1
        or value.get("spec_id") != spec_id
        or value.get("epochs") != expected_epochs
        or value.get("seed") != expected_seed
    ):
        raise PermissionError(f"TriVUS final artifact metadata mismatch: {spec_id}")
    model = make_model().to(device)
    if set(value["state_dict"]) != set(model.state_dict()) or any(
        not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all()
        for tensor in value["state_dict"].values()
    ):
        raise PermissionError(f"TriVUS final artifact state mismatch: {spec_id}")
    model.load_state_dict(value["state_dict"])
    payload = value["standardizer"]
    if set(payload) != {"variant", "mean", "scale"}:
        raise PermissionError(f"TriVUS final standardizer schema mismatch: {spec_id}")
    mean = np.asarray(payload["mean"], dtype=np.float32)
    scale = np.asarray(payload["scale"], dtype=np.float32)
    if (
        payload["variant"] != model_spec(spec_id)["variant"]
        or mean.shape != (115,)
        or scale.shape != (115,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise PermissionError(f"TriVUS final standardizer mismatch: {spec_id}")
    standardizer = TriVUSStandardizer(
        mean=mean,
        scale=scale,
        variant=payload["variant"],
    )
    return model, standardizer


def target_only_expected(data):
    return {
        family: {
            context_key for context_key, value in zip(data.context_keys, data.families)
            if value == family
        }
        for family in ("mind2web", "screenspot_pro", "androidcontrol")
    }


def data_sha256(data):
    digest = hashlib.sha256()
    arrays = (
        data.features, data.candidate_mask, data.fallback_indices,
        data.target_distribution, data.fallback_correct, data.weights,
        data.active, data.labels, data.folds,
    )
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(str(values.dtype).encode())
        digest.update(json.dumps(values.shape).encode())
        digest.update(values.tobytes())
    metadata = (
        data.context_keys, data.sample_keys, data.families, data.cells,
        data.row_ids, data.groups,
    )
    digest.update(json.dumps(metadata, separators=(",", ":")).encode())
    return digest.hexdigest()


def rows_sha256(rows):
    payload = json.dumps(
        sorted(rows, key=lambda row: row["context_key"]),
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def write_jsonl_artifact(path, artifact_root, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    ordered = sorted(rows, key=lambda row: row["context_key"])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", buffering=1) as handle:
        for row in ordered:
            handle.write(json.dumps(
                row, ensure_ascii=True, sort_keys=True, allow_nan=False
            ) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    fsync_directory(path.parent)
    return {
        "relative_path": str(path.relative_to(artifact_root)),
        "sha256": sha256_file(path),
    }


def load_jsonl_artifact(item, artifact_root, expected_contexts=None):
    if set(item) != {"relative_path", "sha256"}:
        raise PermissionError("TriVUS OOF artifact reference mismatch")
    path = safe_child_path(artifact_root, item["relative_path"])
    if sha256_file(path) != item["sha256"]:
        raise PermissionError(f"TriVUS OOF artifact hash mismatch: {path}")
    rows = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    contexts = [row.get("context_key") for row in rows]
    if len(set(contexts)) != len(rows) or contexts != sorted(contexts):
        raise PermissionError("TriVUS OOF artifact context identity mismatch")
    if expected_contexts is not None and set(contexts) != set(expected_contexts):
        raise PermissionError("TriVUS OOF artifact coverage mismatch")
    return rows


def policy_predictions(predictions_by_spec, expected_by_family):
    target_only_specs = {
        spec_id: predictions_by_spec[spec_id]
        for spec_id in (
            "TARGET_ONLY_MIND2WEB", "TARGET_ONLY_SCREENSPOT_PRO",
            "TARGET_ONLY_ANDROIDCONTROL",
        )
    }
    return {
        "JOINT3": predictions_by_spec["JOINT3"],
        "TARGET_ONLY": compose_target_only(target_only_specs, expected_by_family),
        "JOINT2_NO_ANDROID": predictions_by_spec["JOINT2_NO_ANDROID"],
        "NO_VISUAL": predictions_by_spec["NO_VISUAL"],
        "RANDOM_ID_PLACEBO": predictions_by_spec["RANDOM_ID_PLACEBO"],
    }


def pretest_allowed_fields():
    return {
        "schema_version", "status", "outer_fold", "development_folds",
        "sealed_outer_fold", "opened_development_label_sha256",
        "sealed_outer_label_sha256", "code_and_data_sha256", "thresholds",
        "inner_epochs", "final_epochs", "final_seed", "final_artifacts",
        "inner_checkpoints", "data_sha256", "oof_prediction_sha256",
        "oof_artifacts",
        "optimizer_steps_per_epoch", "outer_labels_opened", "training_complete",
    }


def valid_threshold_value(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (math.isfinite(float(value)) or float(value) == float("inf"))
        and float(value) >= 0
    )


def validate_threshold_report(policy, report):
    families = POLICY_SPECS[policy][1]
    expected_cells = {
        "mind2web": ("C_uni", "C_cond", "C_rand", "C_self"),
        "screenspot_pro": ("C_uni", "C_cond", "C_rand", "C_self"),
        "androidcontrol": ("low", "high"),
    }
    if set(report) != {"families"} or set(report["families"]) != set(families):
        raise PermissionError(f"TriVUS pretest threshold family mismatch: {policy}")
    for family in families:
        family_report = report["families"][family]
        if set(family_report) != {"family_threshold", "family_selection", "cells"}:
            raise PermissionError(f"TriVUS pretest family threshold schema: {policy}/{family}")
        if set(family_report["cells"]) != set(expected_cells[family]):
            raise PermissionError(f"TriVUS pretest threshold cells: {policy}/{family}")
        if (
            len(family_report["family_threshold"]) != 2
            or not all(valid_threshold_value(value) for value in family_report["family_threshold"])
        ):
            raise PermissionError(f"TriVUS pretest family threshold values: {policy}/{family}")
        for cell, cell_report in family_report["cells"].items():
            if set(cell_report) != {
                "threshold", "threshold_source", "changed_opportunities", "selection",
            }:
                raise PermissionError(f"TriVUS pretest cell threshold schema: {policy}/{family}/{cell}")
            if (
                len(cell_report["threshold"]) != 2
                or not all(valid_threshold_value(value) for value in cell_report["threshold"])
                or cell_report["threshold_source"] not in {"cell", "family_backoff"}
                or type(cell_report["changed_opportunities"]) is not int
                or cell_report["changed_opportunities"] < 0
            ):
                raise PermissionError(f"TriVUS pretest cell threshold values: {policy}/{family}/{cell}")
    return True


def expected_oof_contexts(public, outer_fold, spec_id):
    families = set(model_spec(spec_id)["families"])
    return {
        f"outer-{outer_fold}/inner-{row['fold']}/{sample_key}"
        for sample_key, row in public.items()
        if int(row["fold"]) != outer_fold and row["benchmark"] in families
    }


def recompute_thresholds_from_pretest(
    record, config, artifact_root, outer_fold, public,
):
    oof_rows = {
        spec_id: load_jsonl_artifact(
            record["oof_artifacts"][spec_id], artifact_root,
            expected_contexts=expected_oof_contexts(public, outer_fold, spec_id),
        )
        for spec_id in MODEL_SPECS
    }
    expected_by_family = {
        family: expected_oof_contexts(public, outer_fold, spec_id)
        for family, spec_id in {
            "mind2web": "TARGET_ONLY_MIND2WEB",
            "screenspot_pro": "TARGET_ONLY_SCREENSPOT_PRO",
            "androidcontrol": "TARGET_ONLY_ANDROIDCONTROL",
        }.items()
    }
    policies = policy_predictions(oof_rows, expected_by_family)
    return {
        policy: select_thresholds(
            policies[policy], config["thresholds"]["mde"],
            config["thresholds"]["minimum_cell_opportunities"],
            included_families=families,
        )
        for policy, (_, families) in POLICY_SPECS.items()
    }


def recompute_data_hashes(outer_fold, assembly_config):
    public, blind = load_locked_public_inputs(assembly_config)
    context_manifest = load_context_manifest(assembly_config)
    context_path = ROOT / assembly_config["dependencies"]["contexts"]["path"]
    expected_counts = assembly_config["expected"]["context_records_by_public_fold"]
    development = tuple(fold for fold in range(5) if fold != outer_fold)
    output = {}
    for holdout in development:
        phase = load_context_phase(
            context_path, context_manifest, public, outer_fold, "inner",
            expected_counts, holdout_fold=holdout,
        )
        train, _ = assemble_phase_data(
            assembly_config, public, blind, phase, phase.fit_folds
        )
        checkpoint, _ = assemble_phase_data(
            assembly_config, public, blind, phase, (phase.checkpoint_fold,)
        )
        oof, _ = assemble_phase_data(
            assembly_config, public, blind, phase, (holdout,)
        )
        output[str(holdout)] = {
            "model_training": data_sha256(train),
            "checkpoint": data_sha256(checkpoint),
            "oof": data_sha256(oof),
        }
    final_phase = load_context_phase(
        context_path, context_manifest, public, outer_fold, "final",
        expected_counts,
    )
    final_train, _ = assemble_phase_data(
        assembly_config, public, blind, final_phase, development
    )
    output["final_training"] = data_sha256(final_train)
    return output


def validate_pretest(
    record, outer_fold, config, assembly_config, recovery,
    output_root=OUTPUT_ROOT,
):
    development = tuple(fold for fold in range(5) if fold != outer_fold)
    expected_seed = config["seed"] + outer_fold * 1000 + 999
    if (
        set(record) != pretest_allowed_fields()
        or record.get("schema_version") != 1
        or record.get("status") != "PASS_TRIVUS_PRETEST_FROZEN_BEFORE_OUTER_LABEL_ACCESS"
        or record.get("outer_fold") != outer_fold
        or record.get("sealed_outer_fold") != outer_fold
        or tuple(record.get("development_folds", ())) != development
        or record.get("outer_labels_opened") is not False
        or record.get("training_complete") is not True
        or record.get("optimizer_steps_per_epoch") != 1
        or set(record.get("final_artifacts", {})) != set(MODEL_SPECS)
        or set(record.get("inner_epochs", {})) != set(MODEL_SPECS)
        or set(record.get("final_epochs", {})) != set(MODEL_SPECS)
        or set(record.get("inner_checkpoints", {})) != set(MODEL_SPECS)
        or set(record.get("oof_prediction_sha256", {})) != set(MODEL_SPECS)
        or set(record.get("oof_artifacts", {})) != set(MODEL_SPECS)
        or set(record.get("thresholds", {})) != set(POLICY_SPECS)
        or record.get("final_seed") != expected_seed
        or record.get("opened_development_label_sha256") != private_fold_hashes(assembly_config, development)
        or record.get("sealed_outer_label_sha256") != private_fold_hashes(assembly_config, (outer_fold,))
        or record.get("code_and_data_sha256") != code_and_data_hashes(config, assembly_config)
    ):
        raise PermissionError("TriVUS invalid pretest record")
    for spec_id in MODEL_SPECS:
        epochs = record["inner_epochs"][spec_id]
        checkpoints = record["inner_checkpoints"][spec_id]
        if (
            not isinstance(epochs, list)
            or len(epochs) != 4
            or any(type(value) is not int or value < 1 for value in epochs)
            or not isinstance(checkpoints, list)
            or len(checkpoints) != 4
            or [item.get("holdout_fold") for item in checkpoints] != list(development)
            or [item.get("selected_epoch") for item in checkpoints] != epochs
            or any(
                set(item) != {
                    "holdout_fold", "selected_epoch", "selected_checkpoint_loss",
                    "epochs_run", "history",
                }
                for item in checkpoints
            )
            or not is_sha256(record["oof_prediction_sha256"][spec_id])
            or type(record["final_epochs"][spec_id]) is not int
            or record["final_epochs"][spec_id] != half_up_median(epochs)
        ):
            raise PermissionError(f"TriVUS pretest epoch mismatch: {spec_id}")
    data_hashes = record.get("data_sha256")
    if not isinstance(data_hashes, dict) or set(data_hashes) != {
        *(str(fold) for fold in development), "final_training",
    }:
        raise PermissionError("TriVUS pretest data fingerprint roster mismatch")
    if not is_sha256(data_hashes["final_training"]):
        raise PermissionError("TriVUS pretest final data fingerprint mismatch")
    for fold in development:
        values = data_hashes[str(fold)]
        if (
            not isinstance(values, dict)
            or set(values) != {"model_training", "checkpoint", "oof"}
            or not all(is_sha256(value) for value in values.values())
        ):
            raise PermissionError(f"TriVUS pretest inner data fingerprint mismatch: {fold}")
    if recompute_data_hashes(outer_fold, assembly_config) != data_hashes:
        raise PermissionError("TriVUS pretest data fingerprint recomputation mismatch")
    public, _ = load_locked_public_inputs(assembly_config)
    for spec_id in MODEL_SPECS:
        rows = load_jsonl_artifact(
            record["oof_artifacts"][spec_id],
            Path(output_root) / f"outer-{outer_fold}",
            expected_contexts=expected_oof_contexts(public, outer_fold, spec_id),
        )
        if rows_sha256(rows) != record["oof_prediction_sha256"][spec_id]:
            raise PermissionError(f"TriVUS pretest OOF content hash mismatch: {spec_id}")
    for policy, report in record["thresholds"].items():
        validate_threshold_report(policy, report)
    recomputed = recompute_thresholds_from_pretest(
        record, config, Path(output_root) / f"outer-{outer_fold}",
        outer_fold, public,
    )
    if recomputed != record["thresholds"]:
        raise PermissionError("TriVUS pretest threshold recomputation mismatch")
    for spec_id, item in record["final_artifacts"].items():
        artifact_root = Path(output_root) / f"outer-{outer_fold}"
        if item.get("relative_path") != f"{spec_id}.pt":
            raise PermissionError(f"TriVUS pretest artifact path mismatch: {spec_id}")
        if sha256_file(artifact_root / item["relative_path"]) != item["sha256"]:
            raise PermissionError(f"TriVUS pretest artifact drift: {spec_id}")
        loaded_model, _ = load_final_artifact(
            item, artifact_root, spec_id,
            record["final_epochs"][spec_id], expected_seed,
            torch.device("cpu"),
        )
        del loaded_model
    assert_protected_process(recovery)
    return True


def load_outer_after_pretest(
    path, outer_fold, config, assembly_config, public, predictions, final_phase,
    output_root=OUTPUT_ROOT,
):
    if not Path(path).is_file():
        raise PermissionError("TriVUS outer labels sealed until pretest exists")
    record = json.loads(Path(path).read_text())
    validate_pretest(
        record, outer_fold, config, assembly_config, load_recovery_config(),
        output_root,
    )
    return assemble_phase_data(
        assembly_config, public, predictions, final_phase, (outer_fold,)
    )


def opened_hashes(paths):
    return {
        str(Path(path).relative_to(ROOT)): sha256_file(path)
        for path in sorted(set(paths))
    }


def observed_fold_hashes(paths):
    output = {"vus": {}, "android": {}}
    for value in sorted(set(paths)):
        path = Path(value)
        name = path.name
        if not name.startswith("private_labels_fold-") or not name.endswith(".jsonl"):
            raise ValueError(f"TriVUS unexpected opened private file: {path}")
        fold = name.removeprefix("private_labels_fold-").removesuffix(".jsonl")
        family = "android" if RUN_DIR in path.parents else "vus"
        if fold in output[family]:
            raise ValueError(f"TriVUS duplicate observed private fold: {family}/{fold}")
        output[family][fold] = sha256_file(path)
    return output


def private_fold_hashes(assembly_config, folds):
    vus = json.loads((ROOT / assembly_config["dependencies"]["vus_private_manifest"]["path"]).read_text())
    android = json.loads((ROOT / assembly_config["dependencies"]["android_private_manifest"]["path"]).read_text())
    output = {"vus": {}, "android": {}}
    for family, manifest, base in (
        ("vus", vus, ROOT / "runs/visual-utility-selector/2026-08-11"),
        ("android", android, ROOT),
    ):
        for fold in folds:
            item = manifest["folds"][str(fold)]
            path = safe_child_path(base, item["path"])
            observed = sha256_file(path)
            if observed != item["sha256"]:
                raise PermissionError(f"TriVUS private fold drift: {family}/{fold}")
            output[family][str(fold)] = observed
    return output


def code_and_data_hashes(config, assembly_config):
    values = {
        f"formal/{name}": item["sha256"]
        for name, item in config["dependencies"].items()
    }
    values.update({
        f"assembly/{name}": item["sha256"]
        for name, item in assembly_config["dependencies"].items()
    })
    values["formal_config"] = sha256_file(CONFIG_PATH)
    return values


def outer_lock_path(outer_fold, output_root=OUTPUT_ROOT):
    return Path(output_root) / f"outer-{outer_fold}" / "STARTED.json"


def acquire_outer_lock(outer_fold, output_root=OUTPUT_ROOT):
    path = outer_lock_path(outer_fold, output_root)
    value = {
        "schema_version": 1,
        "status": "TRIVUS_OUTER_WORKER_STARTED",
        "outer_fold": outer_fold,
        "pid": os.getpid(),
    }
    write_exclusive_json(
        path, value, ["schema_version", "status", "outer_fold", "pid"]
    )
    return path


def require_outer_lock(outer_fold, output_root=OUTPUT_ROOT):
    path = outer_lock_path(outer_fold, output_root)
    if not path.is_file():
        raise PermissionError("TriVUS outer worker lock is missing")
    value = json.loads(path.read_text())
    if (
        set(value) != {"schema_version", "status", "outer_fold", "pid"}
        or value["schema_version"] != 1
        or value["status"] != "TRIVUS_OUTER_WORKER_STARTED"
        or value["outer_fold"] != outer_fold
        or value["pid"] != os.getpid()
    ):
        raise PermissionError("TriVUS outer worker lock is invalid")
    return path


def reload_before_outer_labels(reload_callbacks, label_callback):
    loaded = [callback() for callback in reload_callbacks]
    labels = label_callback()
    return loaded, labels


def run_outer(outer_fold, device, output_root=OUTPUT_ROOT):
    require_outer_lock(outer_fold, output_root)
    config = load_config()
    assembly_config = load_assembly_config()
    recovery = load_recovery_config()
    assert_protected_process(recovery)
    public, blind = load_locked_public_inputs(assembly_config)
    context_manifest = load_context_manifest(assembly_config)
    context_path = ROOT / assembly_config["dependencies"]["contexts"]["path"]
    expected_counts = assembly_config["expected"]["context_records_by_public_fold"]
    development = tuple(fold for fold in range(5) if fold != outer_fold)
    oof = {spec_id: [] for spec_id in MODEL_SPECS}
    inner_epochs = {spec_id: [] for spec_id in MODEL_SPECS}
    opened = set()
    phase_data_hashes = {}
    inner_checkpoints = {spec_id: [] for spec_id in MODEL_SPECS}

    for holdout in development:
        phase = load_context_phase(
            context_path, context_manifest, public, outer_fold, "inner",
            expected_counts, holdout_fold=holdout,
        )
        train, paths = assemble_phase_data(
            assembly_config, public, blind, phase, phase.fit_folds
        )
        opened.update(paths)
        checkpoint, paths = assemble_phase_data(
            assembly_config, public, blind, phase, (phase.checkpoint_fold,)
        )
        opened.update(paths)
        holdout_data, paths = assemble_phase_data(
            assembly_config, public, blind, phase, (holdout,)
        )
        opened.update(paths)
        phase_data_hashes[str(holdout)] = {
            "model_training": data_sha256(train),
            "checkpoint": data_sha256(checkpoint),
            "oof": data_sha256(holdout_data),
        }
        seed = config["seed"] + outer_fold * 1000 + holdout * 10
        for spec_id in MODEL_SPECS:
            model, standardizer, report = train_with_checkpoint(
                train, checkpoint, spec_id, config, seed, device
            )
            inner_epochs[spec_id].append(report["selected_epoch"])
            inner_checkpoints[spec_id].append({
                "holdout_fold": holdout,
                **report,
            })
            oof[spec_id].extend(predict_data(
                model, holdout_data, standardizer, spec_id,
                config["optimizer"]["evaluation_batch_size"], device,
            ))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    expected_by_family = {
        family: {
            f"outer-{outer_fold}/inner-{row['fold']}/{sample_key}"
            for sample_key, row in public.items()
            if row["fold"] != outer_fold and row["benchmark"] == family
        }
        for family in ("mind2web", "screenspot_pro", "androidcontrol")
    }
    outer_dir = Path(output_root) / f"outer-{outer_fold}"
    oof_artifacts = {
        spec_id: write_jsonl_artifact(
            outer_dir / "oof" / f"{spec_id}.jsonl", outer_dir, rows
        )
        for spec_id, rows in oof.items()
    }
    sealed_oof = {
        spec_id: load_jsonl_artifact(
            oof_artifacts[spec_id], outer_dir,
            expected_contexts=expected_oof_contexts(
                public, outer_fold, spec_id
            ),
        )
        for spec_id in oof
    }
    policies = policy_predictions(sealed_oof, expected_by_family)
    thresholds = {}
    for policy, (_, families) in POLICY_SPECS.items():
        thresholds[policy] = select_thresholds(
            policies[policy], config["thresholds"]["mde"],
            config["thresholds"]["minimum_cell_opportunities"],
            included_families=families,
        )

    final_phase = load_context_phase(
        context_path, context_manifest, public, outer_fold, "final",
        expected_counts,
    )
    final_train, paths = assemble_phase_data(
        assembly_config, public, blind, final_phase, development
    )
    opened.update(paths)
    phase_data_hashes["final_training"] = data_sha256(final_train)
    final_seed = config["seed"] + outer_fold * 1000 + 999
    final_epochs = {
        spec_id: half_up_median(values) for spec_id, values in inner_epochs.items()
    }
    final_artifacts = {}
    for spec_id in MODEL_SPECS:
        model, standardizer = train_fixed_epochs(
            final_train, spec_id, final_epochs[spec_id], config, final_seed, device
        )
        final_artifacts[spec_id] = write_final_artifact(
            outer_dir / f"{spec_id}.pt", outer_dir, model, standardizer,
            spec_id, final_epochs[spec_id], final_seed,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pretest = {
        "schema_version": 1,
        "status": "PASS_TRIVUS_PRETEST_FROZEN_BEFORE_OUTER_LABEL_ACCESS",
        "outer_fold": outer_fold,
        "development_folds": list(development),
        "sealed_outer_fold": outer_fold,
        "opened_development_label_sha256": observed_fold_hashes(opened),
        "sealed_outer_label_sha256": private_fold_hashes(assembly_config, (outer_fold,)),
        "code_and_data_sha256": code_and_data_hashes(config, assembly_config),
        "thresholds": thresholds,
        "inner_epochs": inner_epochs,
        "inner_checkpoints": inner_checkpoints,
        "final_epochs": final_epochs,
        "data_sha256": phase_data_hashes,
        "oof_prediction_sha256": {
            spec_id: rows_sha256(rows) for spec_id, rows in sealed_oof.items()
        },
        "oof_artifacts": oof_artifacts,
        "final_seed": final_seed,
        "final_artifacts": final_artifacts,
        "optimizer_steps_per_epoch": 1,
        "outer_labels_opened": False,
        "training_complete": True,
    }
    assert_protected_process(recovery)
    pretest_path = outer_dir / f"outer-{outer_fold}.pretest.json"
    atomic_json_file(pretest_path, pretest)
    atomic_json_file(outer_dir / "PRETEST_WRITTEN.json", {
        "schema_version": 1,
        "status": "TRIVUS_PRETEST_WRITTEN",
        "outer_fold": outer_fold,
        "pretest_sha256": sha256_file(pretest_path),
    })
    spec_order = tuple(MODEL_SPECS)
    loaded_values, outer_value = reload_before_outer_labels(
        [
            lambda spec_id=spec_id: load_final_artifact(
                final_artifacts[spec_id], outer_dir, spec_id,
                final_epochs[spec_id],
                final_seed, device,
            )
            for spec_id in spec_order
        ],
        lambda: load_outer_after_pretest(
            pretest_path, outer_fold, config, assembly_config,
            public, blind, final_phase, output_root,
        ),
    )
    loaded_final = dict(zip(spec_order, loaded_values))
    test_data, outer_paths = outer_value
    if set(outer_paths) & opened:
        raise PermissionError("TriVUS outer label was opened during development")
    observed_outer = observed_fold_hashes(outer_paths)
    if observed_outer != pretest["sealed_outer_label_sha256"]:
        raise PermissionError("TriVUS observed outer labels differ from pretest seal")
    predictions_by_spec = {}
    for spec_id in MODEL_SPECS:
        model, standardizer = loaded_final.pop(spec_id)
        predictions_by_spec[spec_id] = predict_data(
            model, test_data, standardizer, spec_id,
            config["optimizer"]["evaluation_batch_size"], device,
        )
        del model
    expected_test = target_only_expected(test_data)
    test_policies = policy_predictions(predictions_by_spec, expected_test)
    outputs = {}
    reports = {}
    for policy, (_, families) in POLICY_SPECS.items():
        values, report = apply_selected_thresholds(
            test_policies[policy], thresholds[policy], included_families=families
        )
        by_context = {row["context_key"]: row for row in test_policies[policy]}
        safe_by_sample = {
            by_context[key]["sample_key"]: bool(value) for key, value in values.items()
        }
        if len(safe_by_sample) != len(values):
            raise ValueError(f"TriVUS duplicate held-out sample output: {policy}")
        outputs[policy] = {
            "safe": safe_by_sample,
            "direct": {
                by_context[key]["sample_key"]: bool(by_context[key]["direct_success"])
                for key in values
            },
            "fallback": {
                by_context[key]["sample_key"]: bool(by_context[key]["fallback_success"])
                for key in values
            },
        }
        reports[policy] = report
    result = {
        "schema_version": 1,
        "status": "PASS_TRIVUS_OUTER_COMPLETE",
        "outer_fold": outer_fold,
        "pretest_sha256": sha256_file(pretest_path),
        "inner_epochs": inner_epochs,
        "final_epochs": final_epochs,
        "thresholds": thresholds,
        "opened_outer_label_sha256": observed_outer,
        "reports": reports,
        "outputs": outputs,
    }
    assert_protected_process(recovery)
    atomic_json_file(outer_dir / f"outer-{outer_fold}.json", result)
    atomic_json_file(outer_dir / "OUTER_COMPLETE.json", {
        "schema_version": 1,
        "status": "TRIVUS_OUTER_COMPLETE",
        "outer_fold": outer_fold,
        "result_sha256": sha256_file(outer_dir / f"outer-{outer_fold}.json"),
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-fold", type=int, required=True, choices=range(5))
    parser.add_argument("--device", required=True)
    parser.add_argument("--authorization-receipt", type=Path, required=True)
    args = parser.parse_args()
    authorization = validate_worker_receipt(
        args.authorization_receipt, args.outer_fold
    )
    attempt_root = attempt_path(authorization["authorization_nonce"])
    outer_dir = attempt_root / f"outer-{args.outer_fold}"
    if outer_dir.exists():
        raise FileExistsError(outer_dir)
    assert_protected_process(load_recovery_config())
    acquire_outer_lock(args.outer_fold, attempt_root)
    result = run_outer(args.outer_fold, torch.device(args.device), attempt_root)
    print(json.dumps({
        "status": result["status"],
        "outer_fold": result["outer_fold"],
        "pretest_sha256": result["pretest_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()