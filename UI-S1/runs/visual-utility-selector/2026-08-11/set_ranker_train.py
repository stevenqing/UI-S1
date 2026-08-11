import argparse
import copy
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
UTILITY_DIR = ROOT / "runs/lsa-utility/2026-08-11"
sys.path.insert(0, str(UTILITY_DIR))

from behavior_policy import (
    cyclic_validation_fold,
    fit_final_policies,
    fit_inner_policies,
    load_cev_config,
)
from utility_common import ARMS, BENCHMARKS, ids_for_folds, load_banks, load_cev, reliability_by_arm
from adjudicate_anchor import apply_threshold, select_benchmark_threshold, select_cell_threshold
from set_ranker_data import (
    build_set_data,
    deterministic_epoch_permutations,
    fit_standardizer,
    load_label_folds,
    load_public_predictions,
    torch_batch,
)
from set_ranker_model import CONFIGS, VisualLogitSetRanker, permute_batch, ranker_loss


CONFIG_PATH = RUN_DIR / "configs/set_ranker_prereg.yaml"
CONFIG_IDS = ("S1", "S2", "S3")


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_test_labels_after_pretest(outer_fold, pretest_path, label_dir=RUN_DIR / "data"):
    if not pretest_path.is_file():
        raise PermissionError(f"V-K5 outer labels sealed until pretest selection exists: {pretest_path}")
    record = json.loads(pretest_path.read_text())
    if (
        record.get("status") != "PASS_SELECTION_FROZEN_BEFORE_OUTER_LABEL_ACCESS"
        or record.get("outer_fold") != outer_fold
        or outer_fold in record.get("opened_development_label_folds", [])
    ):
        raise PermissionError(f"V-K5 invalid pretest selection record: {pretest_path}")
    return load_label_folds([outer_fold], label_dir=label_dir)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_config(config):
    if config["status"] != "FROZEN_BEFORE_SET_RANKER_RESULTS":
        raise ValueError("VUS-SR protocol is not frozen")
    for config_id in CONFIG_IDS:
        observed = config["configurations"][config_id]
        expected = CONFIGS[config_id]
        pairs = {
            "learning_rate": "learning_rate",
            "weight_decay": "weight_decay",
            "maximum_epochs": "epochs",
            "auxiliary_weight": "aux_weight",
            "grpo_weight": "grpo_weight",
        }
        for observed_name, expected_name in pairs.items():
            if not math.isclose(float(observed[observed_name]), float(expected[expected_name])):
                raise ValueError(f"VUS-SR config mismatch: {config_id}/{observed_name}")


def make_model(input_dim, config):
    architecture = config["architecture"]
    return VisualLogitSetRanker(
        input_dim=input_dim,
        width=architecture["width"],
        heads=architecture["attention_heads"],
        layers=architecture["transformer_layers"],
        dropout=architecture["dropout"],
    )


def active_indices(data):
    indices = np.flatnonzero(data.active & (data.weights > 0))
    if not len(indices):
        raise ValueError("no active weighted VUS-SR rows")
    return indices


def evaluate_loss(model, data, config_id, batch_size, device):
    model.eval()
    indices = active_indices(data)
    normalization = torch.as_tensor(float(data.weights[indices].sum()), device=device)
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            batch = torch_batch(data, selected, device)
            loss, _ = ranker_loss(model, batch, config_id, normalization=normalization)
            total += float(loss)
    return total


def train_epoch(model, data, config_id, epoch, optimizer, config, seed, device):
    model.train()
    indices = active_indices(data)
    generator = np.random.default_rng(seed + epoch)
    indices = generator.permutation(indices)
    batch_size = config["optimizer"]["train_batch_size"]
    normalization = torch.as_tensor(float(data.weights[indices].sum()), device=device)
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    permutations = deterministic_epoch_permutations(data.sample_keys, epoch, seed)
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        batch = torch_batch(data, selected, device)
        permutation = torch.as_tensor(permutations[selected], device=device)
        batch = permute_batch(batch, permutation)
        loss, _ = ranker_loss(model, batch, config_id, normalization=normalization)
        loss.backward()
        total += float(loss.detach())
    torch.nn.utils.clip_grad_norm_(model.parameters(), config["optimizer"]["gradient_clip_norm"])
    optimizer.step()
    return total


def train_with_checkpoint(train_data, validation_data, config_id, config, seed, device):
    standardizer = fit_standardizer(train_data)
    train_data = standardizer.transform(train_data)
    validation_data = standardizer.transform(validation_data)
    set_seed(seed)
    model = make_model(train_data.features.shape[-1], config).to(device)
    values = config["configurations"][config_id]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=values["learning_rate"],
        weight_decay=values["weight_decay"],
    )
    patience = config["optimizer"]["patience_epochs"]
    minimum = config["optimizer"]["minimum_validation_improvement"]
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, values["maximum_epochs"] + 1):
        training_loss = train_epoch(model, train_data, config_id, epoch, optimizer, config, seed, device)
        validation_loss = evaluate_loss(
            model, validation_data, config_id,
            config["optimizer"]["evaluation_batch_size"], device,
        )
        history.append({"epoch": epoch, "training_loss": training_loss, "validation_loss": validation_loss})
        if validation_loss < best_loss - minimum:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None or best_epoch < 1:
        raise ValueError("VUS-SR checkpoint selection failed")
    model.load_state_dict(best_state)
    return model, standardizer, {
        "selected_epoch": best_epoch,
        "selected_validation_loss": best_loss,
        "epochs_run": len(history),
        "history": history,
    }


def train_fixed_epochs(train_data, config_id, epochs, config, seed, device):
    standardizer = fit_standardizer(train_data)
    train_data = standardizer.transform(train_data)
    set_seed(seed)
    model = make_model(train_data.features.shape[-1], config).to(device)
    values = config["configurations"][config_id]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=values["learning_rate"],
        weight_decay=values["weight_decay"],
    )
    history = []
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_data, config_id, epoch, optimizer, config, seed, device)
        history.append({"epoch": epoch, "training_loss": loss})
    return model, standardizer, history


def fallback_wrong_scores(auxiliary_logits, config_id):
    if config_id == "S1":
        return torch.ones_like(auxiliary_logits)
    if config_id not in ("S2", "S3"):
        raise ValueError(f"unknown VUS-SR configuration: {config_id}")
    return torch.sigmoid(-auxiliary_logits)


def predict_data(model, data, standardizer, config_id, batch_size, device):
    data = standardizer.transform(data)
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            stop = min(len(data), start + batch_size)
            indices = np.arange(start, stop)
            batch = torch_batch(data, indices, device)
            utility_logits, auxiliary_logits = model(batch.features, batch.fallback_indices)
            utility_logits = utility_logits.float().cpu().numpy()
            wrong_scores = fallback_wrong_scores(auxiliary_logits, config_id).float().cpu().numpy()
            for offset, index in enumerate(indices):
                candidate_logits = utility_logits[offset, :12]
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
                    "margin": float(candidate_logits[direct] - utility_logits[offset, 12]),
                    "wrong_score": float(wrong_scores[offset]),
                    "direct_success": bool(labels[direct]),
                    "fallback_success": bool(labels[fallback]),
                })
    return output


def select_thresholds(rows, config):
    report = {"benchmarks": {}}
    all_eligible = True
    objective = []
    for benchmark in BENCHMARKS:
        rows_by_arm = {
            arm: [row for row in rows if row["benchmark"] == benchmark and row["arm"] == arm]
            for arm in ARMS
        }
        benchmark_threshold, benchmark_selection = select_benchmark_threshold(
            rows_by_arm, config["mde"][benchmark]
        )
        arm_report = {}
        deltas = []
        for arm in ARMS:
            opportunities = sum(row["changed"] for row in rows_by_arm[arm])
            if opportunities >= 200:
                threshold, selection = select_cell_threshold(rows_by_arm[arm], config["mde"][benchmark])
                source = "cell"
            else:
                threshold = benchmark_threshold
                selection = apply_threshold(rows_by_arm[arm], threshold)[1]
                source = "benchmark_backoff"
            deltas.append(selection["point_delta"])
            all_eligible &= selection["point_delta"] >= -0.5 * config["mde"][benchmark] - 1e-15
            arm_report[arm] = {
                "threshold": list(threshold),
                "threshold_source": source,
                "changed_opportunities": opportunities,
                "selection": selection,
            }
        mean = float(np.mean(deltas))
        all_eligible &= mean >= -0.25 * config["mde"][benchmark] - 1e-15
        objective.append(mean / config["mde"][benchmark])
        report["benchmarks"][benchmark] = {
            "benchmark_threshold": list(benchmark_threshold),
            "benchmark_selection": benchmark_selection,
            "arms": arm_report,
            "equal_arm_delta": mean,
        }
    report["eligible"] = bool(all_eligible)
    report["selection_objective"] = float(np.mean(objective))
    return report


def select_configuration(oof_by_config, config):
    candidates = []
    order = config["safe_policy"]["configuration_tie_order"]
    for config_id in CONFIG_IDS:
        threshold_report = select_thresholds(oof_by_config[config_id], config)
        if threshold_report["eligible"]:
            candidates.append((
                threshold_report["selection_objective"],
                -order.index(config_id),
                config_id,
                threshold_report,
            ))
    if not candidates:
        raise AssertionError("VUS-SR infinite thresholds must make every configuration eligible")
    selected = max(candidates)
    return {
        "config_id": selected[2],
        "selection_objective": selected[0],
        "thresholds": selected[3],
        "all_configurations": {
            config_id: select_thresholds(oof_by_config[config_id], config)
            for config_id in CONFIG_IDS
        },
    }


def apply_selected_threshold(rows, benchmark, arm, selected):
    threshold = selected["thresholds"]["benchmarks"][benchmark]["arms"][arm]["threshold"]
    return apply_threshold(rows, tuple(threshold))


def final_epoch(inner_reports, config_id):
    epochs = [report["models"][config_id]["checkpoint"]["selected_epoch"] for report in inner_reports]
    return max(1, int(math.floor(float(np.median(epochs)) + 0.5))), epochs


def run_outer(outer_fold, config, device, pretest_output):
    banks = load_banks()
    cev = load_cev()
    cev_config = load_cev_config()
    public, visual_predictions = load_public_predictions()
    dev_folds = [fold for fold in range(5) if fold != outer_fold]
    development_labels = load_label_folds(dev_folds)
    oof_by_config = {config_id: [] for config_id in CONFIG_IDS}
    inner_reports = []
    for holdout_fold in dev_folds:
        candidate_train_folds = [fold for fold in dev_folds if fold != holdout_fold]
        validation_fold = cyclic_validation_fold(holdout_fold, candidate_train_folds)
        model_train_folds = [fold for fold in candidate_train_folds if fold != validation_fold]
        train_ids = ids_for_folds(banks, model_train_folds)
        validation_ids = ids_for_folds(banks, [validation_fold])
        holdout_ids = ids_for_folds(banks, [holdout_fold])
        reliability = reliability_by_arm(banks, train_ids)
        policies, behavior_report = fit_inner_policies(
            banks, model_train_folds, validation_fold, cev_config
        )
        train_data = build_set_data(
            banks, train_ids, reliability, policies, public, visual_predictions, development_labels,
            leave_one_ids={benchmark: set(values) for benchmark, values in train_ids.items()},
        )
        validation_data = build_set_data(
            banks, validation_ids, reliability, policies, public, visual_predictions, development_labels
        )
        holdout_data = build_set_data(
            banks, holdout_ids, reliability, policies, public, visual_predictions, development_labels
        )
        report = {
            "holdout_fold": holdout_fold,
            "checkpoint_validation_fold": validation_fold,
            "model_train_folds": model_train_folds,
            "behavior_policy": behavior_report,
            "rows": {
                "train": len(train_data),
                "validation": len(validation_data),
                "holdout": len(holdout_data),
            },
            "input_dimension": int(train_data.features.shape[-1]),
            "models": {},
        }
        for config_index, config_id in enumerate(CONFIG_IDS):
            seed = config["seed"] + outer_fold * 1000 + holdout_fold * 10
            model, standardizer, checkpoint = train_with_checkpoint(
                train_data, validation_data, config_id, config, seed, device
            )
            predictions = predict_data(
                model, holdout_data, standardizer, config_id,
                config["optimizer"]["evaluation_batch_size"], device,
            )
            oof_by_config[config_id].extend(predictions)
            report["models"][config_id] = {
                "checkpoint": checkpoint,
                "holdout_direct_accuracy": float(np.mean([row["direct_success"] for row in predictions])),
                "holdout_fallback_accuracy": float(np.mean([row["fallback_success"] for row in predictions])),
            }
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        inner_reports.append(report)
        print(
            f"VUS-SR outer={outer_fold} holdout={holdout_fold} train={model_train_folds} validation={validation_fold} complete",
            flush=True,
        )
    for config_id in CONFIG_IDS:
        expected = sum(len(ids_for_folds(banks, [fold])["mind2web"]) + len(ids_for_folds(banks, [fold])["screenspot_pro"]) for fold in dev_folds) * len(ARMS)
        if len(oof_by_config[config_id]) != expected:
            raise ValueError(f"VUS-SR OOF coverage mismatch: {config_id}/{len(oof_by_config[config_id])}/{expected}")
    selected = select_configuration(oof_by_config, config)
    epochs, inner_epochs = final_epoch(inner_reports, selected["config_id"])

    dev_ids = ids_for_folds(banks, dev_folds)
    reliability = reliability_by_arm(banks, dev_ids)
    final_policies = fit_final_policies(banks, outer_fold, cev)
    final_train = build_set_data(
        banks, dev_ids, reliability, final_policies, public, visual_predictions, development_labels,
        leave_one_ids={benchmark: set(values) for benchmark, values in dev_ids.items()},
    )
    final_seed = config["seed"] + outer_fold * 1000 + 999
    model, standardizer, final_history = train_fixed_epochs(
        final_train, selected["config_id"], epochs, config, final_seed, device
    )

    label_manifest = json.loads((RUN_DIR / "data/private_label_folds.manifest.json").read_text())
    pretest_record = {
        "schema_version": 1,
        "status": "PASS_SELECTION_FROZEN_BEFORE_OUTER_LABEL_ACCESS",
        "outer_fold": outer_fold,
        "opened_development_label_folds": dev_folds,
        "opened_development_label_sha256": {
            str(fold): label_manifest["folds"][str(fold)]["sha256"] for fold in dev_folds
        },
        "sealed_outer_label_sha256": label_manifest["folds"][str(outer_fold)]["sha256"],
        "selected": selected,
        "selected_inner_epochs": inner_epochs,
        "final_epochs": epochs,
        "blind_predictions_sha256": json.loads(
            (RUN_DIR / "zero_shot/predictions.manifest.json").read_text()
        )["predictions_sha256"],
    }
    atomic_json(pretest_output, pretest_record)
    test_labels = load_test_labels_after_pretest(outer_fold, pretest_output)

    test_ids = ids_for_folds(banks, [outer_fold])
    test_data = build_set_data(
        banks, test_ids, reliability, final_policies, public, visual_predictions, test_labels
    )
    test_predictions = predict_data(
        model, test_data, standardizer, selected["config_id"],
        config["optimizer"]["evaluation_batch_size"], device,
    )
    test_report = {benchmark: {} for benchmark in BENCHMARKS}
    outputs = {benchmark: {arm: {"safe": {}, "direct": {}, "fallback": {}} for arm in ARMS} for benchmark in BENCHMARKS}
    for benchmark in BENCHMARKS:
        for arm in ARMS:
            rows = [row for row in test_predictions if row["benchmark"] == benchmark and row["arm"] == arm]
            safe, override = apply_selected_threshold(rows, benchmark, arm, selected)
            expected_fallback = cev["outputs"][benchmark][arm]["CEV_A"]
            mismatch = sum(row["fallback_success"] != bool(expected_fallback[row["row_id"]]) for row in rows)
            if mismatch:
                raise ValueError(f"V-K2 final fallback mismatch: outer{outer_fold}/{benchmark}/{arm}/{mismatch}")
            outputs[benchmark][arm]["safe"] = {row["row_id"]: safe[row["sample_key"]] for row in rows}
            outputs[benchmark][arm]["direct"] = {row["row_id"]: row["direct_success"] for row in rows}
            outputs[benchmark][arm]["fallback"] = {row["row_id"]: row["fallback_success"] for row in rows}
            test_report[benchmark][arm] = {
                "rows": len(rows),
                "safe_accuracy": float(np.mean(list(outputs[benchmark][arm]["safe"].values()))),
                "direct_accuracy": float(np.mean(list(outputs[benchmark][arm]["direct"].values()))),
                "fallback_accuracy": float(np.mean(list(outputs[benchmark][arm]["fallback"].values()))),
                "fallback_mismatches": 0,
                **override,
            }
    return {
        "schema_version": 1,
        "status": "PASS_OUTER_COMPLETE",
        "outer_fold": outer_fold,
        "selected": selected,
        "selected_inner_epochs": inner_epochs,
        "final_epochs": epochs,
        "inner": inner_reports,
        "final_training_history": final_history,
        "test": test_report,
        "outputs": outputs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-fold", type=int, required=True, choices=range(5))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not Path("/proc/2274").exists():
        raise RuntimeError("protected PID 2274 is unexpectedly absent")
    config = yaml.safe_load(CONFIG_PATH.read_text())
    validate_config(config)
    device = torch.device(args.device)
    output = args.output or RUN_DIR / f"set_ranker/outer-{args.outer_fold}.json"
    pretest_output = output.with_name(f"outer-{args.outer_fold}.pretest.json")
    if pretest_output.exists():
        raise FileExistsError(pretest_output)
    result = run_outer(args.outer_fold, config, device, pretest_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "outer_fold": result["outer_fold"],
        "selected_config": result["selected"]["config_id"],
        "selection_objective": result["selected"]["selection_objective"],
        "final_epochs": result["final_epochs"],
        "test": result["test"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
