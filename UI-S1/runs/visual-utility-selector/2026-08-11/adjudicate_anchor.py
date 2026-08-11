import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from set_ranker_data import load_label_folds


RUN_DIR = Path(__file__).resolve().parent
ARMS = ("C_uni", "C_cond", "C_rand", "C_self")
BENCHMARKS = ("mind2web", "screenspot_pro")


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def join_context_rows(predictions, labels, contexts):
    output = []
    for context in contexts:
        key = context["sample_key"]
        if key not in predictions or key not in labels:
            raise ValueError(f"anchor context missing prediction or fold label: {key}")
        prediction = predictions[key]
        success = labels[key]["candidate_success"]
        permutation = prediction["display_to_candidate"]
        probabilities = prediction["label_probabilities"]
        direct_display = int(np.argmax(probabilities[:12]))
        direct_index = int(permutation[direct_display])
        fallback_index = int(context["fallback_index"])
        fallback_display = permutation.index(fallback_index)
        margin = float(probabilities[direct_display] - probabilities[fallback_display])
        wrong_score = float(max(0.0, 1.0 - probabilities[fallback_display]))
        output.append({
            **{field: prediction[field] for field in ("sample_key", "benchmark", "arm", "row_id", "fold", "group")},
            "context_key": context["context_key"],
            "outer_fold": context["outer_fold"],
            "role": context["role"],
            "probabilities": probabilities,
            "display_to_candidate": permutation,
            "direct_index": direct_index,
            "fallback_index": fallback_index,
            "changed": direct_index != fallback_index,
            "margin": margin,
            "wrong_score": wrong_score,
            "direct_success": bool(success[direct_index]),
            "fallback_success": bool(success[fallback_index]),
            "candidate_success": success,
        })
    return output


def load_anchor_test_labels_after_pretest(outer_fold, pretest_path, label_dir=RUN_DIR / "data"):
    if not pretest_path.is_file():
        raise PermissionError(f"V-K5 anchor outer labels sealed until pretest selection exists: {pretest_path}")
    record = json.loads(pretest_path.read_text())
    matches = [fold for fold in record.get("folds", []) if fold.get("outer_fold") == outer_fold]
    if (
        record.get("status") != "PASS_ALL_ANCHOR_SELECTIONS_FROZEN_BEFORE_OUTER_LABEL_ACCESS"
        or len(matches) != 1
        or outer_fold in matches[0].get("opened_development_label_folds", [])
    ):
        raise PermissionError(f"V-K5 invalid anchor pretest selection record: {pretest_path}")
    return load_label_folds([outer_fold], label_dir=label_dir)


def axis_candidates(values):
    positive = [value for value in values if value > 0]
    output = {0.0, float("inf")}
    if positive:
        output.update(float(np.quantile(positive, quantile)) for quantile in np.linspace(0, 1, 11))
    return sorted(output)


def threshold_grid(rows):
    changed = [row for row in rows if row["changed"]]
    margins = axis_candidates([row["margin"] for row in changed])
    wrong = axis_candidates([row["wrong_score"] for row in changed])
    return [(margin, wrong_score) for margin in margins for wrong_score in wrong]


def apply_threshold(rows, threshold):
    margin_threshold, wrong_threshold = threshold
    values = {}
    wins = losses = overrides = 0
    for row in rows:
        override = (
            row["changed"]
            and row["margin"] >= margin_threshold
            and row["wrong_score"] >= wrong_threshold
        )
        success = row["direct_success"] if override else row["fallback_success"]
        values[row["sample_key"]] = bool(success)
        wins += int(override and row["direct_success"] and not row["fallback_success"])
        losses += int(override and row["fallback_success"] and not row["direct_success"])
        overrides += int(override)
    delta = (wins - losses) / len(rows)
    return values, {
        "point_delta": delta,
        "wins": wins,
        "losses": losses,
        "overrides": overrides,
        "override_rate": overrides / len(rows),
    }


def select_cell_threshold(rows, mde):
    candidates = []
    for threshold in threshold_grid(rows):
        _, report = apply_threshold(rows, threshold)
        if report["point_delta"] >= -0.5 * mde - 1e-15:
            candidates.append((report["point_delta"], threshold[1], threshold[0], threshold, report))
    if not candidates:
        raise AssertionError("infinite cell threshold must be eligible")
    selected = max(candidates)
    return selected[3], selected[4]


def select_benchmark_threshold(rows_by_arm, mde):
    pooled = [row for arm_rows in rows_by_arm.values() for row in arm_rows]
    candidates = []
    for threshold in threshold_grid(pooled):
        reports = {arm: apply_threshold(rows, threshold)[1] for arm, rows in rows_by_arm.items()}
        mean = float(np.mean([reports[arm]["point_delta"] for arm in ARMS]))
        eligible = all(reports[arm]["point_delta"] >= -0.5 * mde - 1e-15 for arm in ARMS)
        eligible &= mean >= -0.25 * mde - 1e-15
        if eligible:
            candidates.append((mean, threshold[1], threshold[0], threshold, reports))
    if not candidates:
        raise AssertionError("infinite benchmark threshold must be eligible")
    selected = max(candidates)
    return selected[3], {"equal_arm_delta": selected[0], "arms": selected[4]}


def select_outer_thresholds(dev, outer_fold, config):
    minimum = config["eligibility_anchor"]["minimum_cell_opportunities"]
    fold_report = {"outer_fold": outer_fold, "benchmarks": {}}
    for benchmark in BENCHMARKS:
        dev_by_arm = {
            arm: [row for row in dev if row["benchmark"] == benchmark and row["arm"] == arm]
            for arm in ARMS
        }
        benchmark_threshold, benchmark_selection = select_benchmark_threshold(
            dev_by_arm, config["mde"][benchmark]
        )
        benchmark_report = {
            "benchmark_threshold": list(benchmark_threshold),
            "benchmark_selection": benchmark_selection,
            "arms": {},
        }
        for arm in ARMS:
            opportunities = sum(row["changed"] for row in dev_by_arm[arm])
            if opportunities >= minimum:
                threshold, selection = select_cell_threshold(dev_by_arm[arm], config["mde"][benchmark])
                source = "cell"
            else:
                threshold = benchmark_threshold
                selection = apply_threshold(dev_by_arm[arm], threshold)[1]
                source = "benchmark_backoff"
            benchmark_report["arms"][arm] = {
                "threshold": list(threshold),
                "threshold_source": source,
                "dev_changed_opportunities": opportunities,
                "dev_selection": selection,
            }
        fold_report["benchmarks"][benchmark] = benchmark_report
    return fold_report


def apply_outer_thresholds(test, fold_report):
    outputs = {}
    for benchmark in BENCHMARKS:
        for arm in ARMS:
            arm_report = fold_report["benchmarks"][benchmark]["arms"][arm]
            test_rows = [row for row in test if row["benchmark"] == benchmark and row["arm"] == arm]
            values, test_report = apply_threshold(test_rows, tuple(arm_report["threshold"]))
            outputs.update(values)
            arm_report["test"] = test_report
    return outputs


def nested_safe(rows, config):
    outputs = {}
    fold_reports = []
    for outer_fold in range(5):
        contexts = [row for row in rows if row["outer_fold"] == outer_fold]
        dev = [row for row in contexts if row["role"] == "dev"]
        test = [row for row in contexts if row["role"] == "test"]
        fold_report = select_outer_thresholds(dev, outer_fold, config)
        outputs.update(apply_outer_thresholds(test, fold_report))
        fold_reports.append(fold_report)
    expected = sum(row["role"] == "test" for row in rows)
    if len(outputs) != expected:
        raise ValueError(f"nested safe output coverage mismatch: {len(outputs)} != {expected}")
    return outputs, fold_reports


def candidate_auroc(rows, benchmark):
    labels = []
    scores = []
    for row in rows:
        if row["benchmark"] != benchmark or row["role"] != "test":
            continue
        fallback_success = row["fallback_success"]
        inverse = {candidate: display for display, candidate in enumerate(row["display_to_candidate"])}
        for candidate_index, success in enumerate(row["candidate_success"]):
            labels.append(int(bool(success) and not fallback_success))
            scores.append(float(row["probabilities"][inverse[candidate_index]]))
    if len(set(labels)) != 2:
        raise ValueError(f"anchor AUROC lacks two classes: {benchmark}")
    return float(roc_auc_score(labels, scores)), {"candidates": len(labels), "positives": sum(labels)}


def summarize(rows, safe):
    output = {benchmark: {} for benchmark in BENCHMARKS}
    for benchmark in BENCHMARKS:
        for arm in ARMS:
            selected = [row for row in rows if row["role"] == "test" and row["benchmark"] == benchmark and row["arm"] == arm]
            safe_accuracy = float(np.mean([safe[row["sample_key"]] for row in selected]))
            fallback_accuracy = float(np.mean([row["fallback_success"] for row in selected]))
            output[benchmark][arm] = {
                "rows": len(selected),
                "safe_accuracy": safe_accuracy,
                "fallback_accuracy": fallback_accuracy,
                "point_delta": safe_accuracy - fallback_accuracy,
            }
        output[benchmark]["equal_arm_delta"] = float(np.mean([
            output[benchmark][arm]["point_delta"] for arm in ARMS
        ]))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=RUN_DIR / "zero_shot/predictions.jsonl")
    parser.add_argument("--label-dir", type=Path, default=RUN_DIR / "data")
    parser.add_argument("--contexts", type=Path, default=RUN_DIR / "data/nested_fallbacks.jsonl")
    parser.add_argument("--pretest", type=Path, default=RUN_DIR / "zero_shot/anchor_pretest_selection.json")
    parser.add_argument("--output", type=Path, default=RUN_DIR / "zero_shot/anchor_adjudication.json")
    args = parser.parse_args()
    config = yaml.safe_load((RUN_DIR / "configs/vus_prereg.yaml").read_text())
    prediction_rows = load_jsonl(args.predictions)
    predictions = {row["sample_key"]: row for row in prediction_rows}
    if len(predictions) != len(prediction_rows):
        raise ValueError("duplicate anchor prediction keys")
    contexts = load_jsonl(args.contexts)
    if len(contexts) != 5 * len(predictions):
        raise ValueError("anchor nested-context coverage mismatch")
    label_manifest = json.loads((args.label_dir / "private_label_folds.manifest.json").read_text())

    folds = []
    pretest_folds = []
    for outer_fold in range(5):
        development_folds = [fold for fold in range(5) if fold != outer_fold]
        development_labels = load_label_folds(development_folds, label_dir=args.label_dir)
        development_contexts = [
            context for context in contexts
            if context["outer_fold"] == outer_fold and context["role"] == "dev"
        ]
        development_rows = join_context_rows(predictions, development_labels, development_contexts)
        fold_report = select_outer_thresholds(development_rows, outer_fold, config)
        folds.append(fold_report)
        pretest_folds.append({
            "outer_fold": outer_fold,
            "opened_development_label_folds": development_folds,
            "opened_development_label_sha256": {
                str(fold): label_manifest["folds"][str(fold)]["sha256"]
                for fold in development_folds
            },
            "sealed_outer_label_sha256": label_manifest["folds"][str(outer_fold)]["sha256"],
            "selection": fold_report,
        })
    atomic_json(args.pretest, {
        "schema_version": 1,
        "status": "PASS_ALL_ANCHOR_SELECTIONS_FROZEN_BEFORE_OUTER_LABEL_ACCESS",
        "folds": pretest_folds,
    })

    safe = {}
    test_rows = []
    for outer_fold, fold_report in enumerate(folds):
        test_labels = load_anchor_test_labels_after_pretest(
            outer_fold, args.pretest, label_dir=args.label_dir
        )
        test_contexts = [
            context for context in contexts
            if context["outer_fold"] == outer_fold and context["role"] == "test"
        ]
        rows = join_context_rows(predictions, test_labels, test_contexts)
        safe.update(apply_outer_thresholds(rows, fold_report))
        test_rows.extend(rows)
    if len(safe) != len(predictions) or len(test_rows) != len(predictions):
        raise ValueError("anchor held-out output coverage mismatch")
    summary = summarize(test_rows, safe)
    auroc = {}
    for benchmark in BENCHMARKS:
        value, counts = candidate_auroc(test_rows, benchmark)
        auroc[benchmark] = {"utility_positive_auroc": value, **counts}
    standardized = float(np.mean([
        summary[benchmark]["equal_arm_delta"] / config["mde"][benchmark]
        for benchmark in BENCHMARKS
    ]))
    no_cell_one_mde_loss = all(
        summary[benchmark][arm]["point_delta"] >= -config["mde"][benchmark] - 1e-15
        for benchmark in BENCHMARKS for arm in ARMS
    )
    a1 = (
        any(summary[benchmark]["equal_arm_delta"] > 0 for benchmark in BENCHMARKS)
        and no_cell_one_mde_loss
        and standardized > 0
    )
    threshold = config["eligibility_anchor"]["proceed_candidate_utility_auroc_each_benchmark"]
    a2 = all(auroc[benchmark]["utility_positive_auroc"] >= threshold for benchmark in BENCHMARKS)
    result = {
        "schema_version": 1,
        "status": "PASS_ADJUDICATED",
        "outcome": "PROCEED_TO_LORA" if a1 or a2 else "CLOSE_LORA_BRANCH",
        "gates": {
            "A1_safe_heldout_effect": a1,
            "A2_candidate_utility_auroc": a2,
            "no_cell_one_mde_loss": no_cell_one_mde_loss,
            "equal_benchmark_standardized_point": standardized,
        },
        "accuracy": summary,
        "candidate_ranking": auroc,
        "folds": folds,
        "outputs": {"safe": safe},
    }
    atomic_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("outcome", "gates", "accuracy", "candidate_ranking")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
