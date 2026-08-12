import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
VUS = ROOT / "runs/visual-utility-selector/2026-08-11"
UTILITY = ROOT / "runs/lsa-utility/2026-08-11"
sys.path.insert(0, str(VUS))
sys.path.insert(0, str(UTILITY))

from behavior_policy import cyclic_validation_fold, fit_final_policies, fit_inner_policies, load_cev_config
from utility_common import ARMS, BENCHMARKS, ids_for_folds, load_banks, load_cev, reliability_by_arm
from adjudicate_anchor import apply_threshold
from set_ranker_data import load_label_folds, load_public_predictions
from set_ranker_train import select_thresholds
from delta_data import build_delta_data, deterministic_permutations, fit_standardizer, load_channels, torch_batch
from delta_model import CHANNELS, VARIANT_CHANNELS, DeltaLateFusion, channel_mask, delta_loss, permute_batch, restore_candidate_order


CONFIG_PATH = RUN_DIR / "configs/delta_prereg.yaml"
TRAINED_VARIANTS = tuple(VARIANT_CHANNELS)
ALL_VARIANTS = (*TRAINED_VARIANTS, "FIXED_AVERAGE")


def validate_config(config):
    if config.get("status") != "FROZEN_AFTER_RAVEL_K4_BEFORE_DELTA_RESULTS":
        raise ValueError("DELTA protocol is not frozen")
    if tuple(config.get("channels", {})) != CHANNELS:
        raise ValueError("DELTA channel order mismatch")
    observed_variants = {
        name: tuple(values) for name, values in config.get("variants", {}).items()
        if name != "FIXED_AVERAGE"
    }
    if observed_variants != VARIANT_CHANNELS:
        raise ValueError("DELTA variant masks mismatch")
    if tuple(config["variants"].get("FIXED_AVERAGE", ())) != VARIANT_CHANNELS["FULL"]:
        raise ValueError("DELTA fixed-average channels mismatch")
    expected_losses = {
        "listwise_repair_or_keep": 1.0,
        "fallback_correct_bce": 0.5,
        "permutation_channel_consistency": 0.1,
        "expected_u_grpo": 0.1,
    }
    if config.get("losses") != expected_losses:
        raise ValueError("DELTA loss weights mismatch")
    optimizer = config["optimizer"]
    if optimizer.get("name") != "AdamW" or optimizer.get("optimizer_steps_per_epoch") != 1:
        raise ValueError("DELTA optimizer contract mismatch")
    safe_policy = config["safe_policy"]
    if (
        safe_policy.get("threshold_axes") != ["candidate_minus_KEEP", "fallback_wrong"]
        or safe_policy.get("threshold_candidates") != ["infinity", "zero", "positive_deciles"]
        or safe_policy.get("minimum_cell_opportunities") != 200
        or safe_policy.get("cell_loss_max_mde_fraction") != 0.5
        or safe_policy.get("benchmark_loss_max_mde_fraction") != 0.25
    ):
        raise ValueError("DELTA safe-policy contract mismatch")
    nested = config["nested_protocol"]
    if (
        nested.get("outer_folds") != 5
        or nested.get("model_train_folds") != 2
        or nested.get("checkpoint_folds") != 1
        or nested.get("oof_selection_folds") != 1
        or not nested.get("fold_sealed_labels")
        or not nested.get("atomic_pretest")
    ):
        raise ValueError("DELTA nested protocol mismatch")
    if config["statistics"].get("resamples") != 10000 or config["statistics"].get("confidence") != 0.99:
        raise ValueError("DELTA statistical protocol mismatch")


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(base_dim, config):
    values = config["architecture"]
    return DeltaLateFusion(
        base_dim=base_dim,
        channel_dim=7,
        channel_width=values["channel_encoder_width"],
        gate_width=values["channel_gate_width"],
        candidate_width=values["candidate_width"],
        layers=values["set_layers"],
        heads=values["attention_heads"],
        dropout=values["dropout"],
    )


def active_indices(data):
    indices = np.flatnonzero(data.active & (data.weights > 0))
    if not len(indices):
        raise ValueError("DELTA has no active weighted rows")
    return indices


def evaluate_loss(model, data, variant, config, seed, device):
    model.eval()
    indices = active_indices(data)
    permutations = deterministic_permutations(data.sample_keys, 0, seed, "validation_consistency")
    normalization = torch.as_tensor(float(data.weights[indices].sum()), device=device)
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(indices), 1024):
            selected = indices[start:start + 1024]
            batch = torch_batch(data, selected, device)
            permutation = torch.as_tensor(permutations[selected], device=device)
            loss, _ = delta_loss(model, batch, variant, permutation, normalization)
            total += float(loss)
    return total


def train_epoch(model, data, variant, optimizer, epoch, config, seed, device):
    model.train()
    values = config["optimizer"]
    indices = np.random.default_rng(seed + epoch).permutation(active_indices(data))
    permutations = deterministic_permutations(data.sample_keys, epoch, seed, "training_consistency")
    normalization = torch.as_tensor(float(data.weights[indices].sum()), device=device)
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for start in range(0, len(indices), values["batch_size"]):
        selected = indices[start:start + values["batch_size"]]
        batch = torch_batch(data, selected, device)
        permutation = torch.as_tensor(permutations[selected], device=device)
        loss, _ = delta_loss(model, batch, variant, permutation, normalization)
        loss.backward()
        total += float(loss.detach())
    torch.nn.utils.clip_grad_norm_(model.parameters(), values["gradient_clip_norm"])
    optimizer.step()
    return total


def train_checkpoint(train, validation, variant, config, seed, device):
    set_seed(seed)
    model = make_model(train.base_features.shape[-1], config).to(device)
    values = config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=values["learning_rate"], weight_decay=values["weight_decay"]
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, values["maximum_epochs"] + 1):
        training_loss = train_epoch(model, train, variant, optimizer, epoch, config, seed, device)
        validation_loss = evaluate_loss(model, validation, variant, config, seed, device)
        history.append({"epoch": epoch, "training_loss": training_loss, "validation_loss": validation_loss})
        if validation_loss < best_loss - values["minimum_improvement"]:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= values["patience"]:
            break
    if best_state is None:
        raise ValueError("DELTA checkpoint selection failed")
    model.load_state_dict(best_state)
    return model, {
        "selected_epoch": best_epoch,
        "selected_validation_loss": best_loss,
        "epochs_run": len(history),
        "history": history,
    }


def train_fixed(train, variant, epochs, config, seed, device):
    set_seed(seed)
    model = make_model(train.base_features.shape[-1], config).to(device)
    values = config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=values["learning_rate"], weight_decay=values["weight_decay"]
    )
    history = []
    for epoch in range(1, epochs + 1):
        history.append({
            "epoch": epoch,
            "training_loss": train_epoch(model, train, variant, optimizer, epoch, config, seed, device),
        })
    return model, history


def _prediction_rows(data, utility, wrong_scores, gate):
    output = []
    for index in range(len(data)):
        candidate_logits = utility[index, :12]
        direct = int(np.argmax(candidate_logits))
        fallback = int(data.fallback_indices[index])
        labels = data.labels[index]
        output.append({
            "sample_key": data.sample_keys[index],
            "benchmark": data.benchmarks[index],
            "arm": data.arms[index],
            "row_id": data.row_ids[index],
            "fold": int(data.folds[index]),
            "group": data.groups[index],
            "direct_index": direct,
            "fallback_index": fallback,
            "changed": direct != fallback,
            "margin": float(candidate_logits[direct] - utility[index, 12]),
            "wrong_score": float(wrong_scores[index]),
            "direct_success": bool(labels[direct]),
            "fallback_success": bool(labels[fallback]),
            "gate_mass": gate[index].mean(axis=0).tolist() if gate is not None else None,
        })
    return output


def predict_model(model, data, variant, device, active_override=None):
    model.eval()
    utilities = []
    wrong = []
    gates = []
    active = channel_mask(variant, device) if active_override is None else active_override.to(device)
    with torch.no_grad():
        for start in range(0, len(data), 1024):
            indices = np.arange(start, min(len(data), start + 1024))
            batch = torch_batch(data, indices, device)
            utility, fallback_logit, gate = model(
                batch.base_features, batch.channel_features, batch.fallback_indices, active
            )
            utilities.append(utility.float().cpu().numpy())
            wrong.append(torch.sigmoid(-fallback_logit).float().cpu().numpy())
            gates.append(gate.float().cpu().numpy())
    return _prediction_rows(
        data, np.concatenate(utilities), np.concatenate(wrong), np.concatenate(gates)
    )


def predict_fixed_average(data):
    centered = data.channel_features[:, :, :4, 0].mean(axis=2)
    utility = np.concatenate((centered, np.take_along_axis(
        centered, data.fallback_indices[:, None], axis=1
    )), axis=1)
    return _prediction_rows(data, utility, np.ones(len(data), dtype=np.float32), None)


def equivariance_error(model, data, config, seed, device):
    model.eval()
    count = min(64, len(data))
    indices = np.arange(count)
    batch = torch_batch(data, indices, device)
    permutations = torch.as_tensor(
        deterministic_permutations(data.sample_keys[:count], 0, seed, "formal_equivariance"),
        device=device,
    )
    changed = permute_batch(batch, permutations)
    active = channel_mask("FULL", device)
    with torch.no_grad():
        original, original_aux, original_gate = model(
            batch.base_features, batch.channel_features, batch.fallback_indices, active
        )
        moved, moved_aux, moved_gate = model(
            changed.base_features, changed.channel_features, changed.fallback_indices, active
        )
    errors = [
        (original[:, :12] - restore_candidate_order(moved[:, :12], permutations)).abs().max(),
        (original[:, 12] - moved[:, 12]).abs().max(),
        (original_aux - moved_aux).abs().max(),
        (original_gate - restore_candidate_order(moved_gate, permutations)).abs().max(),
    ]
    return float(torch.stack(errors).max().cpu())


def final_epoch(reports, variant):
    epochs = [report["variants"][variant]["checkpoint"]["selected_epoch"] for report in reports]
    return max(1, int(math.floor(float(np.median(epochs)) + 0.5))), epochs


def load_test_after_pretest(outer_fold, pretest_path):
    if not pretest_path.is_file():
        raise PermissionError("DELTA-K6 test labels sealed before pretest")
    record = json.loads(pretest_path.read_text())
    if (
        record.get("status") != "PASS_DELTA_SELECTION_FROZEN"
        or record.get("outer_fold") != outer_fold
        or outer_fold in record.get("opened_development_folds", [])
    ):
        raise PermissionError("DELTA-K6 invalid pretest record")
    return load_label_folds([outer_fold])


def run_outer(outer_fold, output, device):
    config = yaml.safe_load(CONFIG_PATH.read_text())
    validate_config(config)
    channels = load_channels(config)
    public, _ = load_public_predictions()
    banks = load_banks()
    cev = load_cev()
    cev_config = load_cev_config()
    dev_folds = [fold for fold in range(5) if fold != outer_fold]
    dev_labels = load_label_folds(dev_folds)
    oof = {variant: [] for variant in ALL_VARIANTS}
    inner_reports = []
    for holdout_fold in dev_folds:
        candidate_train = [fold for fold in dev_folds if fold != holdout_fold]
        validation_fold = cyclic_validation_fold(holdout_fold, candidate_train)
        model_train = [fold for fold in candidate_train if fold != validation_fold]
        train_ids = ids_for_folds(banks, model_train)
        validation_ids = ids_for_folds(banks, [validation_fold])
        holdout_ids = ids_for_folds(banks, [holdout_fold])
        reliability = reliability_by_arm(banks, train_ids)
        policies, behavior = fit_inner_policies(banks, model_train, validation_fold, cev_config)
        train_raw = build_delta_data(
            banks, train_ids, reliability, policies, public, channels, dev_labels,
            leave_one_ids={benchmark: set(values) for benchmark, values in train_ids.items()},
        )
        validation_raw = build_delta_data(
            banks, validation_ids, reliability, policies, public, channels, dev_labels
        )
        holdout_raw = build_delta_data(
            banks, holdout_ids, reliability, policies, public, channels, dev_labels
        )
        standardizer = fit_standardizer(train_raw)
        train = standardizer.transform(train_raw)
        validation = standardizer.transform(validation_raw)
        holdout = standardizer.transform(holdout_raw)
        seed = config["seed"] + outer_fold * 1000 + holdout_fold * 10
        report = {
            "holdout_fold": holdout_fold,
            "checkpoint_validation_fold": validation_fold,
            "model_train_folds": model_train,
            "behavior_policy": behavior,
            "rows": {"train": len(train), "validation": len(validation), "holdout": len(holdout)},
            "base_dimension": int(train.base_features.shape[-1]),
            "channel_dimension": list(train.channel_features.shape[2:]),
            "variants": {},
        }
        for variant in TRAINED_VARIANTS:
            model, checkpoint = train_checkpoint(train, validation, variant, config, seed, device)
            predictions = predict_model(model, holdout, variant, device)
            oof[variant].extend(predictions)
            report["variants"][variant] = {
                "checkpoint": checkpoint,
                "holdout_direct_accuracy": float(np.mean([row["direct_success"] for row in predictions])),
                "mean_gate_mass": np.mean([row["gate_mass"] for row in predictions], axis=0).tolist(),
            }
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        fixed_predictions = predict_fixed_average(holdout_raw)
        oof["FIXED_AVERAGE"].extend(fixed_predictions)
        report["variants"]["FIXED_AVERAGE"] = {
            "holdout_direct_accuracy": float(np.mean([row["direct_success"] for row in fixed_predictions]))
        }
        inner_reports.append(report)
        print(f"DELTA outer={outer_fold} holdout={holdout_fold} complete", flush=True)

    expected = sum(
        len(ids_for_folds(banks, [fold])["mind2web"])
        + len(ids_for_folds(banks, [fold])["screenspot_pro"])
        for fold in dev_folds
    ) * len(ARMS)
    if any(len(oof[variant]) != expected for variant in ALL_VARIANTS):
        raise ValueError("DELTA OOF coverage mismatch")
    thresholds = {variant: select_thresholds(oof[variant], config) for variant in ALL_VARIANTS}
    if any(not report["eligible"] for report in thresholds.values()):
        raise ValueError("DELTA infinite threshold eligibility failure")
    epochs = {}
    inner_epochs = {}
    for variant in TRAINED_VARIANTS:
        epochs[variant], inner_epochs[variant] = final_epoch(inner_reports, variant)

    dev_ids = ids_for_folds(banks, dev_folds)
    reliability = reliability_by_arm(banks, dev_ids)
    final_policies = fit_final_policies(banks, outer_fold, cev)
    final_raw = build_delta_data(
        banks, dev_ids, reliability, final_policies, public, channels, dev_labels,
        leave_one_ids={benchmark: set(values) for benchmark, values in dev_ids.items()},
    )
    standardizer = fit_standardizer(final_raw)
    final_train = standardizer.transform(final_raw)
    states = {}
    histories = {}
    equivariance = {}
    final_seed = config["seed"] + outer_fold * 1000 + 999
    for variant in TRAINED_VARIANTS:
        model, history = train_fixed(final_train, variant, epochs[variant], config, final_seed, device)
        states[variant] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        histories[variant] = history
        if variant == "FULL":
            equivariance[variant] = equivariance_error(model, final_train, config, final_seed, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    label_manifest = json.loads((VUS / "data/private_label_folds.manifest.json").read_text())
    channel_hashes = {name: config["channels"][name]["sha256"] for name in CHANNELS}
    pretest = output.with_name(f"outer-{outer_fold}.pretest.json")
    atomic_json(pretest, {
        "schema_version": 1,
        "status": "PASS_DELTA_SELECTION_FROZEN",
        "outer_fold": outer_fold,
        "opened_development_folds": dev_folds,
        "opened_development_label_sha256": {
            str(fold): label_manifest["folds"][str(fold)]["sha256"] for fold in dev_folds
        },
        "sealed_outer_label_sha256": label_manifest["folds"][str(outer_fold)]["sha256"],
        "channel_hashes": channel_hashes,
        "thresholds": thresholds,
        "final_epochs": epochs,
        "inner_epochs": inner_epochs,
    })
    test_labels = load_test_after_pretest(outer_fold, pretest)
    test_ids = ids_for_folds(banks, [outer_fold])
    test_raw = build_delta_data(
        banks, test_ids, reliability, final_policies, public, channels, test_labels
    )
    test = standardizer.transform(test_raw)
    outputs = {
        variant: {
            benchmark: {arm: {method: {} for method in ("safe", "direct", "fallback")} for arm in ARMS}
            for benchmark in BENCHMARKS
        }
        for variant in ALL_VARIANTS
    }
    test_reports = {variant: {benchmark: {} for benchmark in BENCHMARKS} for variant in ALL_VARIANTS}
    gate_mass = {}
    full_model = None
    for variant in TRAINED_VARIANTS:
        model = make_model(test.base_features.shape[-1], config).to(device)
        model.load_state_dict(states[variant])
        predictions = predict_model(model, test, variant, device)
        gate_mass[variant] = np.mean([row["gate_mass"] for row in predictions], axis=0).tolist()
        if variant == "FULL":
            full_model = model
        else:
            del model
        for benchmark in BENCHMARKS:
            for arm in ARMS:
                rows = [row for row in predictions if row["benchmark"] == benchmark and row["arm"] == arm]
                threshold = thresholds[variant]["benchmarks"][benchmark]["arms"][arm]["threshold"]
                safe, override = apply_threshold(rows, tuple(threshold))
                expected_fallback = cev["outputs"][benchmark][arm]["CEV_A"]
                mismatch = sum(row["fallback_success"] != bool(expected_fallback[row["row_id"]]) for row in rows)
                if mismatch:
                    raise ValueError(f"DELTA fallback mismatch: {variant}/{benchmark}/{arm}/{mismatch}")
                outputs[variant][benchmark][arm]["safe"] = {row["row_id"]: safe[row["sample_key"]] for row in rows}
                outputs[variant][benchmark][arm]["direct"] = {row["row_id"]: row["direct_success"] for row in rows}
                outputs[variant][benchmark][arm]["fallback"] = {row["row_id"]: row["fallback_success"] for row in rows}
                test_reports[variant][benchmark][arm] = {
                    "rows": len(rows), "fallback_mismatches": 0, **override,
                    "safe_accuracy": float(np.mean(list(outputs[variant][benchmark][arm]["safe"].values()))),
                    "direct_accuracy": float(np.mean(list(outputs[variant][benchmark][arm]["direct"].values()))),
                    "fallback_accuracy": float(np.mean(list(outputs[variant][benchmark][arm]["fallback"].values()))),
                }
    fixed_predictions = predict_fixed_average(test_raw)
    for benchmark in BENCHMARKS:
        for arm in ARMS:
            rows = [row for row in fixed_predictions if row["benchmark"] == benchmark and row["arm"] == arm]
            threshold = thresholds["FIXED_AVERAGE"]["benchmarks"][benchmark]["arms"][arm]["threshold"]
            safe, override = apply_threshold(rows, tuple(threshold))
            outputs["FIXED_AVERAGE"][benchmark][arm]["safe"] = {row["row_id"]: safe[row["sample_key"]] for row in rows}
            outputs["FIXED_AVERAGE"][benchmark][arm]["direct"] = {row["row_id"]: row["direct_success"] for row in rows}
            outputs["FIXED_AVERAGE"][benchmark][arm]["fallback"] = {row["row_id"]: row["fallback_success"] for row in rows}
            test_reports["FIXED_AVERAGE"][benchmark][arm] = {
                "rows": len(rows), **override,
                "safe_accuracy": float(np.mean(list(outputs["FIXED_AVERAGE"][benchmark][arm]["safe"].values()))),
                "direct_accuracy": float(np.mean(list(outputs["FIXED_AVERAGE"][benchmark][arm]["direct"].values()))),
                "fallback_accuracy": float(np.mean(list(outputs["FIXED_AVERAGE"][benchmark][arm]["fallback"].values()))),
            }

    dropout_reports = {}
    full_thresholds = thresholds["FULL"]
    for channel_index, channel_name in enumerate(CHANNELS[:4]):
        mask = channel_mask("FULL")
        mask[channel_index] = False
        predictions = predict_model(full_model, test, "FULL", device, active_override=mask)
        dropout_reports[channel_name] = {
            "mean_gate_mass": np.mean([row["gate_mass"] for row in predictions], axis=0).tolist(),
            "cells": {},
        }
        for benchmark in BENCHMARKS:
            for arm in ARMS:
                rows = [row for row in predictions if row["benchmark"] == benchmark and row["arm"] == arm]
                threshold = full_thresholds["benchmarks"][benchmark]["arms"][arm]["threshold"]
                safe, _ = apply_threshold(rows, tuple(threshold))
                dropout_reports[channel_name]["cells"][f"{benchmark}/{arm}"] = float(np.mean(list(safe.values())))
    del full_model

    result = {
        "schema_version": 1,
        "status": "PASS_DELTA_OUTER_COMPLETE",
        "outer_fold": outer_fold,
        "channel_hashes": channel_hashes,
        "thresholds": thresholds,
        "inner": inner_reports,
        "inner_epochs": inner_epochs,
        "final_epochs": epochs,
        "final_histories": histories,
        "equivariance_max_error": equivariance,
        "mean_gate_mass": gate_mass,
        "channel_dropout": dropout_reports,
        "test": test_reports,
        "outputs": outputs,
    }
    atomic_json(output, result)
    print(json.dumps({
        "status": result["status"], "outer_fold": outer_fold,
        "final_epochs": epochs, "FULL_gate_mass": gate_mass["FULL"],
        "equivariance_max_error": equivariance["FULL"],
    }, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not Path("/proc/2274").exists():
        raise RuntimeError("protected PID 2274 absent")
    pretest = args.output.with_name(f"outer-{args.outer_fold}.pretest.json")
    if args.output.exists() or pretest.exists():
        raise FileExistsError(args.output)
    run_outer(args.outer_fold, args.output, torch.device(args.device))


if __name__ == "__main__":
    main()
