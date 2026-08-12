import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
VUS = ROOT / "runs/visual-utility-selector/2026-08-11"
sys.path.insert(0, str(VUS))

from set_ranker_data import load_label_folds
from civa_data import (
    ARMS, BENCHMARKS, EXPERTS, REAL_EXPERTS, attach_labels, build_base_data,
    load_inputs, sha256_array, sha256_file, validate_config,
)
from civa_model import fit_uplift_model


CONFIG_PATH = RUN_DIR / "configs/civa_prereg.yaml"
LEARNED_VARIANTS = ("REAL_FULL", "REAL_NO_TEXT", "REAL_TEXT_ONLY", "PLACEBO_FULL")
ALL_VARIANTS = (*LEARNED_VARIANTS, "MATCHED_RANDOM")


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def labels_for(base, labels):
    return {key: labels[key] for key in base.sample_keys}


def variant_features(base, variant):
    if variant in ("REAL_FULL", "PLACEBO_FULL"):
        return base.full_features
    if variant == "REAL_NO_TEXT":
        return base.no_text_features
    if variant == "REAL_TEXT_ONLY":
        return base.text_only_features
    raise ValueError(f"unknown CIVA learned variant: {variant}")


def expert_columns(variant):
    return (0, 1, 2) if variant != "PLACEBO_FULL" else (3,)


def fit_variant(data, variant, config, seed):
    columns = expert_columns(variant)
    model = fit_uplift_model(
        variant_features(data.base, variant),
        data.delta[:, columns],
        data.base.weights,
        config["learner"],
        seed,
    )
    return model, columns


def predict_rows(model, columns, data, variant):
    score, rescue, harm = model.predict(variant_features(data.base, variant))
    selected = np.argmax(score, axis=1)
    output = []
    for index, selected_column in enumerate(selected):
        expert_column = columns[int(selected_column)]
        baseline_index = int(data.base.baseline_indices[index])
        expert_index = int(data.base.expert_indices[index, expert_column])
        output.append({
            "sample_key": data.base.sample_keys[index],
            "benchmark": data.base.benchmarks[index],
            "arm": data.base.arms[index],
            "row_id": data.base.row_ids[index],
            "fold": int(data.base.folds[index]),
            "group": data.base.groups[index],
            "baseline_index": baseline_index,
            "expert_index": expert_index,
            "expert": EXPERTS[expert_column],
            "changed": baseline_index != expert_index,
            "score": float(score[index, selected_column]),
            "rescue_probability": float(rescue[index, selected_column]),
            "harm_probability": float(harm[index, selected_column]),
            "baseline_success": bool(data.baseline_success[index]),
            "expert_success": bool(data.expert_success[index, expert_column]),
        })
    return output


def threshold_candidates(rows):
    positive = [row["score"] for row in rows if row["changed"] and row["score"] > 0]
    values = {0.0, float("inf")}
    if positive:
        values.update(float(np.quantile(positive, quantile)) for quantile in np.linspace(0, 1, 11))
    return sorted(values)


def apply_threshold(rows, threshold):
    output = {}
    switched = {}
    wins = losses = 0
    expert_counts = {name: 0 for name in EXPERTS}
    for row in rows:
        switch = row["changed"] and row["score"] >= threshold
        success = row["expert_success"] if switch else row["baseline_success"]
        output[row["row_id"]] = bool(success)
        switched[row["row_id"]] = bool(switch)
        wins += int(switch and row["expert_success"] and not row["baseline_success"])
        losses += int(switch and row["baseline_success"] and not row["expert_success"])
        if switch:
            expert_counts[row["expert"]] += 1
    return output, switched, {
        "point_delta": (wins - losses) / len(rows),
        "wins": wins,
        "losses": losses,
        "switches": sum(switched.values()),
        "switch_rate": sum(switched.values()) / len(rows),
        "expert_counts": expert_counts,
    }


def select_cell_threshold(rows):
    candidates = []
    for threshold in threshold_candidates(rows):
        _, _, report = apply_threshold(rows, threshold)
        candidates.append((report["point_delta"], threshold, report))
    selected = max(candidates)
    return selected[1], selected[2]


def select_thresholds(rows, config):
    report = {"benchmarks": {}}
    objective = []
    eligible = True
    for benchmark in BENCHMARKS:
        arm_reports = {}
        points = []
        for arm in ARMS:
            cell = [row for row in rows if row["benchmark"] == benchmark and row["arm"] == arm]
            threshold, selection = select_cell_threshold(cell)
            if len(cell) < config["thresholds"]["minimum_cell_rows"]:
                raise ValueError(f"CIVA threshold cell too small: {benchmark}/{arm}/{len(cell)}")
            points.append(selection["point_delta"])
            eligible &= selection["point_delta"] >= -config["thresholds"]["cell_loss_max_mde_fraction"] * config["mde"][benchmark]
            arm_reports[arm] = {
                "threshold": threshold,
                "rows": len(cell),
                "selection": selection,
            }
        equal_arm = float(np.mean(points))
        eligible &= equal_arm >= -config["thresholds"]["benchmark_loss_max_mde_fraction"] * config["mde"][benchmark]
        objective.append(equal_arm / config["mde"][benchmark])
        report["benchmarks"][benchmark] = {"arms": arm_reports, "equal_arm_delta": equal_arm}
    report["eligible"] = bool(eligible)
    report["selection_objective"] = float(np.mean(objective))
    return report


def load_test_after_pretest(outer_fold, pretest_path):
    if not pretest_path.is_file():
        raise PermissionError("CIVA-K6 outer labels sealed before pretest")
    record = json.loads(pretest_path.read_text())
    if (
        record.get("status") != "PASS_CIVA_SELECTION_FROZEN"
        or record.get("outer_fold") != outer_fold
        or outer_fold in record.get("opened_development_folds", [])
    ):
        raise PermissionError("CIVA-K6 invalid pretest record")
    return load_label_folds([outer_fold])


def _hash_value(*values):
    return hashlib.sha256("/".join(str(value) for value in values).encode()).digest()


def matched_random(data, indices, desired_switches, seed):
    candidates = []
    for index in indices:
        changed = [
            expert for expert in range(len(REAL_EXPERTS))
            if data.base.expert_indices[index, expert] != data.base.baseline_indices[index]
        ]
        if changed:
            candidates.append((_hash_value(seed, data.base.sample_keys[index], "row"), index, changed))
    candidates.sort(key=lambda value: value[0])
    if desired_switches > len(candidates):
        raise ValueError("CIVA matched-random coverage infeasible")
    selected = {value[1]: value[2] for value in candidates[:desired_switches]}
    output = {}
    expert_counts = {name: 0 for name in REAL_EXPERTS}
    wins = losses = 0
    for index in indices:
        switch = index in selected
        if switch:
            choices = selected[index]
            digest = _hash_value(seed, data.base.sample_keys[index], "expert")
            expert = choices[int.from_bytes(digest[:8], "big") % len(choices)]
            success = bool(data.expert_success[index, expert])
            expert_counts[EXPERTS[expert]] += 1
            wins += int(success and not data.baseline_success[index])
            losses += int(data.baseline_success[index] and not success)
        else:
            success = bool(data.baseline_success[index])
        output[data.base.row_ids[index]] = success
    return output, {
        "point_delta": (wins - losses) / len(indices),
        "wins": wins,
        "losses": losses,
        "switches": desired_switches,
        "switch_rate": desired_switches / len(indices),
        "expert_counts": expert_counts,
    }


def run_outer(outer_fold, output):
    config = yaml.safe_load(CONFIG_PATH.read_text())
    validate_config(config)
    public, channels = load_inputs(config)
    base = build_base_data(public, channels, config["features"]["instruction_hash_dimensions"])
    dev_folds = [fold for fold in range(5) if fold != outer_fold]
    dev_labels = load_label_folds(dev_folds)
    oof = {variant: [] for variant in LEARNED_VARIANTS}
    inner = []
    for holdout_fold in dev_folds:
        train_folds = [fold for fold in dev_folds if fold != holdout_fold]
        train_base = base.subset(train_folds)
        holdout_base = base.subset([holdout_fold])
        train = attach_labels(train_base, labels_for(train_base, dev_labels))
        holdout = attach_labels(holdout_base, labels_for(holdout_base, dev_labels))
        seed = config["seed"] + outer_fold * 1000 + holdout_fold * 10
        report = {
            "holdout_fold": holdout_fold,
            "train_folds": train_folds,
            "rows": {"train": len(train), "holdout": len(holdout)},
            "variants": {},
        }
        for variant in LEARNED_VARIANTS:
            model, columns = fit_variant(train, variant, config, seed)
            predictions = predict_rows(model, columns, holdout, variant)
            oof[variant].extend(predictions)
            report["variants"][variant] = {
                "feature_dimension": int(variant_features(train.base, variant).shape[1]),
                "mean_score": float(np.mean([row["score"] for row in predictions])),
            }
        inner.append(report)
        print(f"CIVA outer={outer_fold} holdout={holdout_fold} complete", flush=True)

    expected = len(base.subset(dev_folds))
    if any(len(oof[variant]) != expected for variant in LEARNED_VARIANTS):
        raise ValueError("CIVA development OOF coverage mismatch")
    thresholds = {variant: select_thresholds(oof[variant], config) for variant in LEARNED_VARIANTS}
    if any(not report["eligible"] for report in thresholds.values()):
        raise ValueError("CIVA infinite-threshold eligibility failure")

    dev_base = base.subset(dev_folds)
    dev = attach_labels(dev_base, labels_for(dev_base, dev_labels))
    final_models = {}
    final_seed = config["seed"] + outer_fold * 1000 + 999
    for variant in LEARNED_VARIANTS:
        final_models[variant] = fit_variant(dev, variant, config, final_seed)

    label_manifest = json.loads((VUS / "data/private_label_folds.manifest.json").read_text())
    pretest = output.with_name(f"outer-{outer_fold}.pretest.json")
    atomic_json(pretest, {
        "schema_version": 1,
        "status": "PASS_CIVA_SELECTION_FROZEN",
        "outer_fold": outer_fold,
        "opened_development_folds": dev_folds,
        "opened_development_label_sha256": {
            str(fold): label_manifest["folds"][str(fold)]["sha256"] for fold in dev_folds
        },
        "sealed_outer_label_sha256": label_manifest["folds"][str(outer_fold)]["sha256"],
        "channel_hashes": {name: config["channels"][name]["sha256"] for name in config["channels"]},
        "public_sha256": sha256_file(VUS / "data/public_records.jsonl"),
        "feature_hashes": {
            "full": sha256_array(base.full_features),
            "no_text": sha256_array(base.no_text_features),
            "text_only": sha256_array(base.text_only_features),
            "baseline_indices": sha256_array(base.baseline_indices),
            "expert_indices": sha256_array(base.expert_indices),
        },
        "implementation_hashes": {
            name: sha256_file(RUN_DIR / name)
            for name in ("civa_data.py", "civa_model.py", "civa_train.py", "configs/civa_prereg.yaml")
        },
        "thresholds": thresholds,
        "learner": config["learner"],
    })

    test_labels = load_test_after_pretest(outer_fold, pretest)
    test_base = base.subset([outer_fold])
    test = attach_labels(test_base, labels_for(test_base, test_labels))
    outputs = {
        variant: {
            benchmark: {arm: {"policy": {}, "baseline": {}} for arm in ARMS}
            for benchmark in BENCHMARKS
        }
        for variant in ALL_VARIANTS
    }
    test_reports = {variant: {benchmark: {} for benchmark in BENCHMARKS} for variant in ALL_VARIANTS}
    predictions = {}
    for variant in LEARNED_VARIANTS:
        model, columns = final_models[variant]
        predictions[variant] = predict_rows(model, columns, test, variant)
        for benchmark in BENCHMARKS:
            for arm in ARMS:
                rows = [row for row in predictions[variant] if row["benchmark"] == benchmark and row["arm"] == arm]
                threshold = thresholds[variant]["benchmarks"][benchmark]["arms"][arm]["threshold"]
                policy, _, report = apply_threshold(rows, threshold)
                baseline = {row["row_id"]: row["baseline_success"] for row in rows}
                outputs[variant][benchmark][arm] = {"policy": policy, "baseline": baseline}
                test_reports[variant][benchmark][arm] = {
                    "rows": len(rows), "threshold": threshold,
                    "policy_accuracy": float(np.mean(list(policy.values()))),
                    "baseline_accuracy": float(np.mean(list(baseline.values()))),
                    **report,
                }

    for benchmark in BENCHMARKS:
        for arm in ARMS:
            full_rows = [row for row in predictions["REAL_FULL"] if row["benchmark"] == benchmark and row["arm"] == arm]
            desired = test_reports["REAL_FULL"][benchmark][arm]["switches"]
            indices = [
                index for index, (name, value) in enumerate(zip(test.base.benchmarks, test.base.arms))
                if name == benchmark and value == arm
            ]
            policy, report = matched_random(test, indices, desired, final_seed)
            baseline = {row["row_id"]: row["baseline_success"] for row in full_rows}
            outputs["MATCHED_RANDOM"][benchmark][arm] = {"policy": policy, "baseline": baseline}
            test_reports["MATCHED_RANDOM"][benchmark][arm] = {
                "rows": len(indices),
                "policy_accuracy": float(np.mean(list(policy.values()))),
                "baseline_accuracy": float(np.mean(list(baseline.values()))),
                **report,
            }

    result = {
        "schema_version": 1,
        "status": "PASS_CIVA_OUTER_COMPLETE",
        "outer_fold": outer_fold,
        "channel_hashes": {name: config["channels"][name]["sha256"] for name in config["channels"]},
        "inner": inner,
        "thresholds": thresholds,
        "test": test_reports,
        "outputs": outputs,
    }
    atomic_json(output, result)
    print(json.dumps({
        "status": result["status"], "outer_fold": outer_fold,
        "REAL_FULL": test_reports["REAL_FULL"],
    }, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not Path("/proc/2274").exists():
        raise RuntimeError("protected PID 2274 absent")
    pretest = args.output.with_name(f"outer-{args.outer_fold}.pretest.json")
    if args.output.exists() or pretest.exists():
        raise FileExistsError(args.output)
    run_outer(args.outer_fold, args.output)


if __name__ == "__main__":
    main()