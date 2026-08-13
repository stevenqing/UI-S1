import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
PRIOR_DIR = RUN_DIR.parent / "2026-08-12"
CHEAP_ROOT = RUN_DIR / "sequential_oof"
OUTPUT_ROOT = RUN_DIR / "sequential_verifier_oof"
sys.path.insert(0, str(RUN_DIR))
sys.path.insert(0, str(PRIOR_DIR))

from context_common import checkpoint_and_fit_folds, sha256_file, write_jsonl_atomic
from sequential_fit import fit_with_checkpoint, require_real_data_optimizer_authorization
from sequential_model import augment_verifier_features, cheap_oof_features
from sequential_oof_runner import (
    FAMILIES, OOF_FIELDS, family_data, predict_rows, save_model,
)
from sequential_real_data_smoke import load_config
from trivus_assembly import (
    assemble_phase_data, load_config as load_assembly_config,
    load_context_manifest, load_context_phase, load_locked_public_inputs,
    with_model_weights,
)
from trivus_data import TriVUSData, fit_standardizer


def verifier_split(outer_fold, holdout_fold):
    checkpoint, fit_folds = checkpoint_and_fit_folds(outer_fold, holdout_fold)
    return {
        "fit_folds": fit_folds,
        "checkpoint_fold": checkpoint,
        "holdout_fold": holdout_fold,
    }


def load_cheap_rows(path, expected_fold, family):
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    contexts = set()
    for row in rows:
        count = len(row.get("candidate_logits", ()))
        if (
            set(row) != OOF_FIELDS
            or row["schema_version"] != 1
            or row["family"] != family
            or row["fold"] != expected_fold
            or count not in (3, 12)
            or len(row["candidate_probabilities"]) != count
            or sorted(row["candidate_order"]) != list(range(count))
            or row["context_key"] in contexts
        ):
            raise ValueError(f"Sequential cheap OOF artifact mismatch: {path}")
        contexts.add(row["context_key"])
    if not rows:
        raise ValueError(f"Sequential cheap OOF artifact is empty: {path}")
    return rows


def select_family_raw(data, family):
    selected = data.subset(np.asarray([
        value == family for value in data.families
    ], dtype=np.bool_))
    if not len(selected) or set(selected.families) != {family}:
        raise ValueError(f"Sequential verifier family missing: {family}")
    return selected


def concatenate_data(values):
    if not values:
        raise ValueError("Sequential verifier has no data to concatenate")
    array_fields = {
        "features", "candidate_mask", "fallback_indices", "target_distribution",
        "fallback_correct", "weights", "active", "labels", "folds",
    }
    tuple_fields = {
        "context_keys", "sample_keys", "families", "cells", "row_ids", "groups",
    }
    payload = {}
    for field in fields(TriVUSData):
        if field.name in array_fields:
            payload[field.name] = np.concatenate([
                getattr(value, field.name) for value in values
            ], axis=0)
        elif field.name in tuple_fields:
            payload[field.name] = tuple(
                item
                for value in values
                for item in getattr(value, field.name)
            )
        else:
            raise AssertionError(f"Unknown TriVUSData field: {field.name}")
    output = TriVUSData(**payload)
    if len(set(output.context_keys)) != len(output):
        raise ValueError("Sequential verifier duplicate concatenated context")
    return output


def augment_data(data, cheap_rows, device):
    by_context = {row["context_key"]: row for row in cheap_rows}
    if set(by_context) != set(data.context_keys):
        raise ValueError("Sequential verifier cheap/data context mismatch")
    logits = np.zeros(data.candidate_mask.shape, dtype=np.float32)
    for index, context_key in enumerate(data.context_keys):
        count = int(data.candidate_mask[index].sum())
        values = by_context[context_key]["candidate_logits"]
        if len(values) != count:
            raise ValueError("Sequential verifier cheap candidate count mismatch")
        logits[index, :count] = values
    feature_tensor = torch.as_tensor(data.features, dtype=torch.float32, device=device)
    mask = torch.as_tensor(data.candidate_mask, dtype=torch.bool, device=device)
    fallback = torch.as_tensor(data.fallback_indices, dtype=torch.long, device=device)
    cheap, _ = cheap_oof_features(
        torch.as_tensor(logits, dtype=torch.float32, device=device), mask, fallback
    )
    augmented = augment_verifier_features(feature_tensor, cheap, mask)
    return (
        augmented,
        mask,
        torch.as_tensor(data.labels, dtype=torch.bool, device=device),
        torch.as_tensor(data.weights, dtype=torch.float32, device=device),
    )


def load_fold_data(
    outer_fold, fold, family, assembly, public, predictions, manifest, cheap_root,
):
    context_path = ROOT / assembly["dependencies"]["contexts"]["path"]
    counts = assembly["expected"]["context_records_by_public_fold"]
    phase = load_context_phase(
        context_path, manifest, public, outer_fold, "inner", counts,
        holdout_fold=fold,
    )
    raw, _ = assemble_phase_data(
        assembly, public, predictions, phase, (fold,)
    )
    selected = select_family_raw(raw, family)
    cheap_path = Path(cheap_root) / f"outer-{outer_fold}" / f"holdout-{fold}" / f"{family}.jsonl"
    return selected, load_cheap_rows(cheap_path, fold, family)


def predict_augmented(model, data, augmented, batch_size):
    model.eval()
    output = []
    mask = torch.as_tensor(data.candidate_mask, dtype=torch.bool, device=augmented.device)
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            stop = min(start + batch_size, len(data))
            logits, _ = model(augmented[start:stop], mask[start:stop])
            probabilities = torch.sigmoid(logits)
            order = torch.argsort(
                logits.masked_fill(~mask[start:stop], torch.finfo(logits.dtype).min),
                dim=1, descending=True, stable=True,
            )
            for offset, index in enumerate(range(start, stop)):
                count = int(data.candidate_mask[index].sum())
                output.append({
                    "schema_version": 1,
                    "context_key": data.context_keys[index],
                    "sample_key": data.sample_keys[index],
                    "family": data.families[index],
                    "cell": data.cells[index],
                    "fold": int(data.folds[index]),
                    "candidate_logits": [float(value) for value in logits[offset, :count].cpu()],
                    "candidate_probabilities": [
                        float(value) for value in probabilities[offset, :count].cpu()
                    ],
                    "candidate_order": [int(value) for value in order[offset, :count].cpu()],
                })
    if any(set(row) != OOF_FIELDS for row in output):
        raise AssertionError("Sequential verifier OOF schema mismatch")
    return sorted(output, key=lambda row: row["context_key"])


def run_one(outer_fold, holdout_fold, family, device, receipt=None, output_root=None):
    config = load_config()
    require_real_data_optimizer_authorization(
        config, receipt, outer_fold, holdout_fold, family, "verifier"
    )
    split = verifier_split(outer_fold, holdout_fold)
    assembly = load_assembly_config()
    public, predictions = load_locked_public_inputs(assembly)
    manifest = load_context_manifest(assembly)
    root = OUTPUT_ROOT if output_root is None else Path(output_root)
    cheap_root = CHEAP_ROOT if output_root is None else root.parent / "cheap"
    loaded = {
        fold: load_fold_data(
            outer_fold, fold, family, assembly, public, predictions, manifest,
            cheap_root,
        )
        for fold in (*split["fit_folds"], split["checkpoint_fold"], holdout_fold)
    }
    train_raw = concatenate_data([loaded[fold][0] for fold in split["fit_folds"]])
    train, standardizer = family_data(train_raw, family)
    train_cheap = [row for fold in split["fit_folds"] for row in loaded[fold][1]]
    checkpoint, _ = family_data(
        loaded[split["checkpoint_fold"]][0], family, standardizer
    )
    checkpoint_cheap = loaded[split["checkpoint_fold"]][1]
    holdout, _ = family_data(loaded[holdout_fold][0], family, standardizer)
    holdout_cheap = loaded[holdout_fold][1]
    seed = int(config["seed"] + 2000 * outer_fold + 20 * holdout_fold + FAMILIES.index(family))
    model, report = fit_with_checkpoint(
        augment_data(train, train_cheap, device),
        augment_data(checkpoint, checkpoint_cheap, device),
        120, config, seed, device,
    )
    holdout_augmented = augment_data(holdout, holdout_cheap, device)[0]
    rows = predict_augmented(
        model, holdout, holdout_augmented,
        config["optimizer"]["evaluation_batch_size"],
    )
    directory = root / f"outer-{outer_fold}" / f"holdout-{holdout_fold}"
    model_path = directory / f"{family}.pt"
    prediction_path = directory / f"{family}.jsonl"
    model_sha256 = save_model(model_path, model, standardizer, {
        "outer_fold": outer_fold,
        "holdout_fold": holdout_fold,
        "family": family,
        "seed": seed,
        "selected_epoch": report["selected_epoch"],
        "input_dimension": 120,
    })
    write_jsonl_atomic(prediction_path, rows)
    return {
        "schema_version": 1,
        "status": "PASS_SEQUENTIAL_VERIFIER_OOF",
        "outer_fold": outer_fold,
        "holdout_fold": holdout_fold,
        "family": family,
        "fit_folds": list(split["fit_folds"]),
        "checkpoint_fold": split["checkpoint_fold"],
        "rows": len(rows),
        "model_sha256": model_sha256,
        "predictions_sha256": sha256_file(prediction_path),
        "selected_epoch": report["selected_epoch"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--holdout-fold", type=int, required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--authorization-receipt", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    print(json.dumps(run_one(
        args.outer_fold, args.holdout_fold, args.family,
        torch.device(args.device), args.authorization_receipt, args.output_root,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()