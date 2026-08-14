import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
TRIVUS_DIR = ROOT / "runs/trivus/2026-08-13"
PRIOR_DIR = ROOT / "runs/trivus/2026-08-12"
OUTPUT_ROOT = TRIVUS_DIR / "sequential_exploratory"
CONFIG_PATH = RUN_DIR / "configs/ceil_prereg.yaml"
PREFLIGHT_PATH = RUN_DIR / "PREFLIGHT.json"
ROWS_PATH = RUN_DIR / "ARM_B_ROWS.jsonl"
OUTPUT_PATH = RUN_DIR / "ARM_B.json"
sys.path.insert(0, str(TRIVUS_DIR))
sys.path.insert(0, str(PRIOR_DIR))

from finalize_trivus import frozen_baselines, load_configs, load_public
from headroom_atlas import load_candidate_labels
from sequential_oof_diagnostic import FAMILIES, load_phase
from trivus_assembly import load_config as load_assembly_config, load_locked_public_inputs
from trivus_data import restore_visual_values


def write_jsonl_fsync(path, rows):
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", buffering=1) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    temporary.replace(path)


def group_pair_matrix(groups, scores_by_group, labels_by_group):
    positive_counts = np.zeros(len(groups), dtype=np.float64)
    negative_counts = np.zeros(len(groups), dtype=np.float64)
    matrix = np.zeros((len(groups), len(groups)), dtype=np.float64)
    positives = []
    negatives = []
    for index, group in enumerate(groups):
        scores = np.asarray(scores_by_group[group], dtype=np.float64)
        labels = np.asarray(labels_by_group[group], dtype=np.bool_)
        positive = scores[labels]
        negative = np.sort(scores[~labels])
        positives.append(positive)
        negatives.append(negative)
        positive_counts[index] = len(positive)
        negative_counts[index] = len(negative)
    for left, positive in enumerate(positives):
        for right, negative in enumerate(negatives):
            if len(positive) and len(negative):
                below = np.searchsorted(negative, positive, side="left")
                at_most = np.searchsorted(negative, positive, side="right")
                matrix[left, right] = float(np.sum(below + 0.5 * (at_most - below)))
    return matrix, positive_counts, negative_counts


def bootstrap_group_counts(groups, folds, resamples, seed):
    by_fold = defaultdict(list)
    for index, group in enumerate(groups):
        by_fold[folds[group]].append(index)
    generator = np.random.default_rng(seed)
    counts = np.zeros((resamples, len(groups)), dtype=np.float64)
    for replicate in range(resamples):
        for fold in sorted(by_fold):
            indices = np.asarray(by_fold[fold], dtype=np.int64)
            selected = generator.choice(indices, size=len(indices), replace=True)
            counts[replicate] += np.bincount(selected, minlength=len(groups))
    return counts


def grouped_auc(scores, labels, group_keys, folds, resamples, seed):
    scores_by_group = defaultdict(list)
    labels_by_group = defaultdict(list)
    for score, label, group in zip(scores, labels, group_keys):
        scores_by_group[group].append(float(score))
        labels_by_group[group].append(bool(label))
    groups = sorted(scores_by_group)
    matrix, positive, negative = group_pair_matrix(groups, scores_by_group, labels_by_group)
    counts = bootstrap_group_counts(groups, folds, resamples, seed)
    numerator = np.einsum("bi,ij,bj->b", counts, matrix, counts, optimize=True)
    denominator = (counts @ positive) * (counts @ negative)
    if np.any(denominator <= 0):
        raise ValueError("CEIL Arm B bootstrap has single-class replicate")
    samples = numerator / denominator
    point = float(roc_auc_score(np.asarray(labels, dtype=np.bool_), np.asarray(scores, dtype=np.float64)))
    return {
        "point": point,
        "ci_99": [float(np.quantile(samples, 0.005)), float(np.quantile(samples, 0.995))],
        "resamples": resamples,
        "seed": seed,
    }, counts, groups


def grouped_mean(values_by_sample, public, sample_keys, resamples, seed):
    by_group = defaultdict(list)
    folds = {}
    for key in sample_keys:
        row = public[key]
        group = (int(row["fold"]), str(row["group"]))
        by_group[group].append(float(values_by_sample[key]))
        folds[group] = int(row["fold"])
    groups = sorted(by_group)
    sums = np.asarray([sum(by_group[group]) for group in groups], dtype=np.float64)
    sizes = np.asarray([len(by_group[group]) for group in groups], dtype=np.float64)
    counts = bootstrap_group_counts(groups, folds, resamples, seed)
    samples = (counts @ sums) / (counts @ sizes)
    return {
        "point": float(np.mean([values_by_sample[key] for key in sample_keys])),
        "ci_99": [float(np.quantile(samples, 0.005)), float(np.quantile(samples, 0.995))],
        "resamples": resamples,
        "seed": seed,
    }


def decision(report):
    if not report["decision_eligible"]:
        return "DESCRIPTIVE_LOW_N"
    lower, upper = report["cheap_candidate_AUROC"]["ci_99"]
    if upper < 0.60:
        return "C_D1"
    if lower > 0.65:
        return "C_D2"
    return "C_D3"


def main():
    if OUTPUT_PATH.exists() or ROWS_PATH.exists():
        raise FileExistsError("CEIL Arm B outputs already exist")
    config = yaml.safe_load(CONFIG_PATH.read_text())
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    if (
        config.get("status") != "FROZEN_BEFORE_ANY_CEIL_RESULT"
        or preflight.get("status") != "PASS_CEIL_INPUT_PREFLIGHT"
        or preflight.get("arm_B_AUROC_computed") is not False
    ):
        raise PermissionError("CEIL Arm B preflight boundary mismatch")
    manifest = json.loads((OUTPUT_ROOT / "MANIFEST.json").read_text())
    if manifest.get("artifact_count") != 240:
        raise PermissionError("CEIL Arm B publication mismatch")
    public = load_public()
    labels, label_manifests = load_candidate_labels(public)
    _, training_config = load_configs()
    _, strongest = frozen_baselines(public, training_config)
    assembly = load_assembly_config()
    locked_public, blind_predictions = load_locked_public_inputs(assembly)
    if set(locked_public) != set(public):
        raise ValueError("CEIL Arm B public coverage mismatch")
    phases = {phase: load_phase(phase, public) for phase in ("cheap", "verifier")}
    reports = {}
    serialized = []
    for family_index, family in enumerate(FAMILIES):
        cheap_rows = phases["cheap"][family]
        verifier_by_context = {row["context_key"]: row for row in phases["verifier"][family]}
        cheap_by_sample = defaultdict(list)
        for row in cheap_rows:
            cheap_by_sample[row["sample_key"]].append(row)
        family_keys = sorted(key for key, row in public.items() if row["benchmark"] == family)
        if set(cheap_by_sample) != set(family_keys) or any(len(rows) != 4 for rows in cheap_by_sample.values()):
            raise ValueError(f"CEIL Arm B four-context mismatch: {family}")
        recoverable = [key for key in family_keys if not strongest[key] and any(labels[key])]
        candidate_values = {source: {"scores": [], "labels": [], "groups": []} for source in ("cheap", "visual", "verifier")}
        top1 = {source: {} for source in candidate_values}
        for key in recoverable:
            rows = sorted(cheap_by_sample[key], key=lambda row: row["context_key"])
            count = len(labels[key])
            _, visual_probability = restore_visual_values(blind_predictions[key], count)
            cheap_matrix = np.asarray([row["candidate_probabilities"] for row in rows], dtype=np.float64)
            verifier_matrix = np.asarray([
                verifier_by_context[row["context_key"]]["candidate_probabilities"] for row in rows
            ], dtype=np.float64)
            label_array = np.asarray(labels[key], dtype=np.bool_)
            group = (int(public[key]["fold"]), str(public[key]["group"]))
            for source, matrix in (
                ("cheap", cheap_matrix),
                ("visual", np.repeat(visual_probability[None, :], 4, axis=0)),
                ("verifier", verifier_matrix),
            ):
                candidate_values[source]["scores"].extend(matrix.ravel().tolist())
                candidate_values[source]["labels"].extend(np.tile(label_array, 4).tolist())
                candidate_values[source]["groups"].extend([group] * (4 * count))
                mean_probability = matrix.mean(axis=0)
                selected = int(np.argmax(mean_probability))
                top1[source][key] = bool(label_array[selected])
            serialized.append({
                "schema_version": 1,
                "family": family,
                "sample_key": key,
                "fold": group[0],
                "group": group[1],
                "candidate_labels": label_array.tolist(),
                "strongest_correct": bool(strongest[key]),
                "cheap_probabilities_by_context": cheap_matrix.tolist(),
                "visual_probabilities": visual_probability.tolist(),
                "verifier_probabilities_by_context": verifier_matrix.tolist(),
                "cheap_top1_correct": top1["cheap"][key],
                "visual_top1_correct": top1["visual"][key],
                "verifier_top1_correct": top1["verifier"][key],
            })
        seed = int(config["arm_B"]["bootstrap"]["seeds"][family])
        folds = {group: group[0] for group in set(candidate_values["cheap"]["groups"])}
        auc_reports = {}
        for offset, source in enumerate(("cheap", "visual", "verifier")):
            values = candidate_values[source]
            auc_reports[source], _, _ = grouped_auc(
                values["scores"], values["labels"], values["groups"], folds,
                int(config["arm_B"]["bootstrap"]["resamples"]), seed + offset,
            )
        top1_reports = {
            source: grouped_mean(values, public, recoverable, 10000, seed + 10 + offset)
            for offset, (source, values) in enumerate(top1.items())
        }
        forced_difference = {
            key: float(top1["cheap"].get(key, strongest[key])) - float(strongest[key])
            for key in family_keys
        }
        forced = grouped_mean(forced_difference, public, family_keys, 10000, seed + 20)
        report = {
            "family": family,
            "family_samples": len(family_keys),
            "recoverable_samples": len(recoverable),
            "positive_candidate_contexts": sum(candidate_values["cheap"]["labels"]),
            "negative_candidate_contexts": len(candidate_values["cheap"]["labels"]) - sum(candidate_values["cheap"]["labels"]),
            "decision_eligible": family in {"mind2web", "screenspot_pro"} and len(recoverable) >= 100,
            "cheap_candidate_AUROC": auc_reports["cheap"],
            "visual_candidate_AUROC": auc_reports["visual"],
            "verifier_candidate_AUROC": auc_reports["verifier"],
            "random_AUROC": 0.5,
            "top1": top1_reports,
            "evaluation_oracle_forced_override_minus_strongest": forced,
            "repeated_contexts_per_sample": 4,
        }
        report["decision"] = decision(report)
        reports[family] = report
    eligible = [report for report in reports.values() if report["decision_eligible"]]
    if any(report["decision"] == "C_D2" for report in eligible):
        overall = "OPEN_NEW_SPEC_C_D2"
    elif eligible and all(report["decision"] == "C_D1" for report in eligible):
        overall = "CLOSE_C_D1"
    else:
        overall = "FREEZE_C_D3_INDETERMINATE"
    write_jsonl_fsync(ROWS_PATH, sorted(serialized, key=lambda row: (row["family"], row["sample_key"])))
    result = {
        "schema_version": 1,
        "status": "PASS_CEIL_ARM_B_COMPLETE",
        "overall_branch": overall,
        "reports": reports,
        "label_manifests": label_manifests,
        "rows_jsonl": ROWS_PATH.relative_to(ROOT).as_posix(),
        "claim_boundary": {
            "evaluation_side_only": True,
            "runtime_rule_allowed": False,
            "current_round_reweighting_allowed": False,
        },
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "overall_branch": overall,
        "families": {
            family: {
                "recoverable": report["recoverable_samples"],
                "cheap_AUROC": report["cheap_candidate_AUROC"],
                "decision": report["decision"],
            }
            for family, report in reports.items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()