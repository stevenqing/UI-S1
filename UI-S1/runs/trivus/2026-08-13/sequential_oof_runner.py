import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
PRIOR_DIR = RUN_DIR.parent / "2026-08-12"
OUTPUT_ROOT = RUN_DIR / "sequential_oof"
sys.path.insert(0, str(RUN_DIR))
sys.path.insert(0, str(PRIOR_DIR))

from context_common import sha256_file, write_jsonl_atomic
from sequential_fit import fit_with_checkpoint, require_real_data_optimizer_authorization
from sequential_real_data_smoke import load_config
from trivus_assembly import (
    assemble_phase_data, load_config as load_assembly_config,
    load_context_manifest, load_context_phase, load_locked_public_inputs,
    with_model_weights,
)
from trivus_data import fit_standardizer


FAMILIES = ("mind2web", "screenspot_pro", "androidcontrol")
OOF_FIELDS = {
    "schema_version", "context_key", "sample_key", "family", "cell", "fold",
    "candidate_logits", "candidate_probabilities", "candidate_order",
}


def family_data(data, family, standardizer=None):
    if family not in FAMILIES:
        raise ValueError(f"Unknown sequential family: {family}")
    weighted = with_model_weights(data, "TARGET_ONLY", family)
    selected = weighted.subset(np.asarray([
        value == family for value in weighted.families
    ], dtype=np.bool_))
    if not len(selected) or set(selected.families) != {family}:
        raise ValueError(f"Sequential family data missing: {family}")
    if standardizer is None:
        standardizer = fit_standardizer(
            selected, "TARGET_ONLY", included_families=(family,)
        )
    transformed = standardizer.transform(selected)
    return transformed, standardizer


def tensors(data, device):
    return (
        torch.as_tensor(data.features, dtype=torch.float32, device=device),
        torch.as_tensor(data.candidate_mask, dtype=torch.bool, device=device),
        torch.as_tensor(data.labels, dtype=torch.bool, device=device),
        torch.as_tensor(data.weights, dtype=torch.float32, device=device),
    )


def predict_rows(model, data, batch_size, device):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            stop = min(start + batch_size, len(data))
            features = torch.as_tensor(
                data.features[start:stop], dtype=torch.float32, device=device
            )
            mask = torch.as_tensor(
                data.candidate_mask[start:stop], dtype=torch.bool, device=device
            )
            logits, _ = model(features, mask)
            probabilities = torch.sigmoid(logits)
            masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
            order = torch.argsort(masked, dim=1, descending=True, stable=True)
            logits = logits.cpu().numpy()
            probabilities = probabilities.cpu().numpy()
            order = order.cpu().numpy()
            for offset, index in enumerate(range(start, stop)):
                count = int(data.candidate_mask[index].sum())
                row = {
                    "schema_version": 1,
                    "context_key": data.context_keys[index],
                    "sample_key": data.sample_keys[index],
                    "family": data.families[index],
                    "cell": data.cells[index],
                    "fold": int(data.folds[index]),
                    "candidate_logits": [float(value) for value in logits[offset, :count]],
                    "candidate_probabilities": [
                        float(value) for value in probabilities[offset, :count]
                    ],
                    "candidate_order": [int(value) for value in order[offset, :count]],
                }
                if set(row) != OOF_FIELDS:
                    raise AssertionError("Sequential OOF schema mismatch")
                output.append(row)
    if len(output) != len(data) or len({row["context_key"] for row in output}) != len(output):
        raise ValueError("Sequential OOF output coverage mismatch")
    return sorted(output, key=lambda row: row["context_key"])


def save_model(path, model, standardizer, metadata):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "schema_version": 1,
        "metadata": metadata,
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "standardizer": {
            "variant": standardizer.variant,
            "mean": standardizer.mean.tolist(),
            "scale": standardizer.scale.tolist(),
        },
    }, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)
    return sha256_file(path)


def run_one(outer_fold, holdout_fold, family, device):
    config = load_config()
    require_real_data_optimizer_authorization(config)
    assembly = load_assembly_config()
    public, predictions = load_locked_public_inputs(assembly)
    manifest = load_context_manifest(assembly)
    context_path = ROOT / assembly["dependencies"]["contexts"]["path"]
    counts = assembly["expected"]["context_records_by_public_fold"]
    phase = load_context_phase(
        context_path, manifest, public, outer_fold, "inner", counts,
        holdout_fold=holdout_fold,
    )
    train_raw, _ = assemble_phase_data(
        assembly, public, predictions, phase, phase.fit_folds
    )
    checkpoint_raw, _ = assemble_phase_data(
        assembly, public, predictions, phase, (phase.checkpoint_fold,)
    )
    holdout_raw, _ = assemble_phase_data(
        assembly, public, predictions, phase, (phase.holdout_fold,)
    )
    train, standardizer = family_data(train_raw, family)
    checkpoint, _ = family_data(checkpoint_raw, family, standardizer)
    holdout, _ = family_data(holdout_raw, family, standardizer)
    seed = int(config["seed"] + 1000 * outer_fold + 10 * holdout_fold + FAMILIES.index(family))
    model, report = fit_with_checkpoint(
        tensors(train, device), tensors(checkpoint, device), 115,
        config, seed, device,
    )
    rows = predict_rows(
        model, holdout, config["optimizer"]["evaluation_batch_size"], device
    )
    directory = OUTPUT_ROOT / f"outer-{outer_fold}" / f"holdout-{holdout_fold}"
    model_path = directory / f"{family}.pt"
    prediction_path = directory / f"{family}.jsonl"
    model_sha256 = save_model(model_path, model, standardizer, {
        "outer_fold": outer_fold,
        "holdout_fold": holdout_fold,
        "family": family,
        "seed": seed,
        "selected_epoch": report["selected_epoch"],
    })
    write_jsonl_atomic(prediction_path, rows)
    return {
        "schema_version": 1,
        "status": "PASS_SEQUENTIAL_CHEAP_OOF",
        "outer_fold": outer_fold,
        "holdout_fold": holdout_fold,
        "family": family,
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
    args = parser.parse_args()
    if args.outer_fold not in range(5) or args.holdout_fold not in range(5):
        raise ValueError("Sequential OOF fold out of range")
    print(json.dumps(run_one(
        args.outer_fold, args.holdout_fold, args.family,
        torch.device(args.device),
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()