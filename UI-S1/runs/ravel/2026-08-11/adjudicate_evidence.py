import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
VUS = ROOT / "runs/visual-utility-selector/2026-08-11"
MODES = ("local", "random", "global_only", "fine_only", "context_only")
BENCHMARKS = ("mind2web", "screenspot_pro")
ARMS = ("C_uni", "C_cond", "C_rand", "C_self")


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def keyed(path):
    rows = load_jsonl(path)
    output = {row["sample_key"]: row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate keys: {path}")
    return output


def original_scores(row):
    output = np.empty(12, dtype=np.float64)
    for display_index, candidate_index in enumerate(row["display_to_candidate"]):
        output[candidate_index] = row["label_probabilities"][display_index]
    return output


def target_areas():
    mind = {
        row["id"]: row
        for row in load_jsonl(ROOT / "runs/xfer/2026-08-07/data/mind2web/mind2web_test_task.jsonl")
    }
    screen = {
        row["id"]: row
        for row in load_jsonl(ROOT / "runs/allocation-law/2026-08-01/raw/shared_regions_n12.jsonl")
    }
    output = {benchmark: {} for benchmark in BENCHMARKS}
    for row_id, row in mind.items():
        width, height = Image.open(ROOT / row["image"]).size
        bbox = row["step"]["bbox"]
        output["mind2web"][row_id] = bbox["width"] * bbox["height"] / (width * height)
    for row_id, row in screen.items():
        width, height = row["img_size"]
        left, top, right, bottom = row["target_bbox"]
        output["screenspot_pro"][row_id] = (right - left) * (bottom - top) / (width * height)
    return output


def arm_metrics(predictions, labels, fallback_outputs, areas, benchmark, arm):
    candidate_labels = []
    candidate_scores = []
    direct = []
    unique = []
    small = []
    cuts = np.quantile(list(areas[benchmark].values()), [0.25, 0.5, 0.75])
    for key, row in predictions.items():
        if row["benchmark"] != benchmark or row["arm"] != arm:
            continue
        success = np.asarray(labels[key]["candidate_success"], dtype=np.bool_)
        scores = original_scores(row)
        fallback_success = bool(fallback_outputs[benchmark][row["arm"]]["fallback"][row["row_id"]])
        utility_positive = success & (not fallback_success)
        candidate_labels.extend(utility_positive.astype(np.int8).tolist())
        candidate_scores.extend(scores.tolist())
        selected = int(np.argmax(scores))
        is_correct = bool(success[selected])
        direct.append(is_correct)
        if int(success.sum()) == 1:
            unique.append(is_correct)
        if areas[benchmark][row["row_id"]] <= cuts[0]:
            small.append((bool(success.any()), is_correct))
    covered_small = [correct for covered, correct in small if covered]
    return {
        "utility_positive_auroc": float(roc_auc_score(candidate_labels, candidate_scores)),
        "candidate_rows": len(candidate_labels),
        "utility_positive_candidates": int(sum(candidate_labels)),
        "direct_accuracy": float(np.mean(direct)),
        "unique_correct_rows": len(unique),
        "unique_correct_recall": float(np.mean(unique)),
        "small_quartile_covered_rows": len(covered_small),
        "small_quartile_recall_given_coverage": float(np.mean(covered_small)),
        "target_area_q25": float(cuts[0]),
    }


def metrics(predictions, labels, fallback_outputs, areas, benchmark):
    by_arm = {
        arm: arm_metrics(predictions, labels, fallback_outputs, areas, benchmark, arm)
        for arm in ARMS
    }
    primary_keys = (
        "utility_positive_auroc", "direct_accuracy",
        "unique_correct_recall", "small_quartile_recall_given_coverage",
    )
    equal_arm = {
        key: float(np.mean([by_arm[arm][key] for arm in ARMS]))
        for key in primary_keys
    }
    pooled_labels = []
    pooled_scores = []
    pooled_direct = []
    for key, row in predictions.items():
        if row["benchmark"] != benchmark:
            continue
        success = np.asarray(labels[key]["candidate_success"], dtype=np.bool_)
        scores = original_scores(row)
        fallback_success = bool(fallback_outputs[benchmark][row["arm"]]["fallback"][row["row_id"]])
        pooled_labels.extend((success & (not fallback_success)).astype(np.int8).tolist())
        pooled_scores.extend(scores.tolist())
        pooled_direct.append(bool(success[int(np.argmax(scores))]))
    return {
        "equal_arm": equal_arm,
        "by_arm": by_arm,
        "pooled_descriptive": {
            "utility_positive_auroc": float(roc_auc_score(pooled_labels, pooled_scores)),
            "candidate_rows": len(pooled_labels),
            "utility_positive_candidates": int(sum(pooled_labels)),
            "direct_accuracy": float(np.mean(pooled_direct)),
        },
    }


def main():
    manifests = {}
    predictions = {}
    for mode in MODES:
        manifest_path = RUN_DIR / f"evidence/{mode}/predictions.manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if manifest["status"] != "PASS_BLIND_EVIDENCE_LOCKED" or manifest["private_labels_opened"]:
            raise ValueError(f"RAVEL evidence not blind-locked: {mode}")
        manifests[mode] = manifest
        predictions[mode] = keyed(RUN_DIR / f"evidence/{mode}/predictions.jsonl")
    baseline_manifest = json.loads((VUS / "zero_shot/predictions.manifest.json").read_text())
    if baseline_manifest["private_labels_opened"]:
        raise ValueError("VUS baseline was not blind-locked")
    predictions["vus_full_screen"] = keyed(VUS / "zero_shot/predictions.jsonl")
    labels = {}
    for fold in range(5):
        labels.update(keyed(VUS / f"data/private_labels_fold-{fold}.jsonl"))
    fallback_outputs = json.loads((VUS / "set_ranker_adjudication.json").read_text())["outputs"]
    areas = target_areas()
    result = {
        "schema_version": 1,
        "status": "PASS_E0_DIAGNOSTIC_ADJUDICATED",
        "blind_manifests": manifests,
        "baseline_predictions_sha256": baseline_manifest["predictions_sha256"],
        "metrics": {},
    }
    for mode, values in predictions.items():
        if set(values) != set(labels):
            raise ValueError(f"RAVEL identity mismatch: {mode}")
        result["metrics"][mode] = {
            benchmark: metrics(values, labels, fallback_outputs, areas, benchmark)
            for benchmark in BENCHMARKS
        }
    local = result["metrics"]["local"]
    baseline = result["metrics"]["vus_full_screen"]
    random_control = result["metrics"]["random"]
    deltas = {
        benchmark: {
            "auroc_local_minus_vus": local[benchmark]["equal_arm"]["utility_positive_auroc"] - baseline[benchmark]["equal_arm"]["utility_positive_auroc"],
            "auroc_local_minus_random": local[benchmark]["equal_arm"]["utility_positive_auroc"] - random_control[benchmark]["equal_arm"]["utility_positive_auroc"],
            "unique_recall_local_minus_vus": local[benchmark]["equal_arm"]["unique_correct_recall"] - baseline[benchmark]["equal_arm"]["unique_correct_recall"],
            "small_recall_local_minus_vus": local[benchmark]["equal_arm"]["small_quartile_recall_given_coverage"] - baseline[benchmark]["equal_arm"]["small_quartile_recall_given_coverage"],
        }
        for benchmark in BENCHMARKS
    }
    result["deltas"] = deltas
    one_gain = any(deltas[benchmark]["auroc_local_minus_vus"] >= 0.03 for benchmark in BENCHMARKS)
    other_preserved = all(deltas[benchmark]["auroc_local_minus_vus"] >= -0.01 for benchmark in BENCHMARKS)
    random_not_explain = all(deltas[benchmark]["auroc_local_minus_random"] > 0 for benchmark in BENCHMARKS)
    result["E0_AUROC_gate"] = {
        "one_benchmark_gain_at_least_0_03": one_gain,
        "other_benchmark_loss_at_most_0_01": other_preserved,
        "random_center_control_not_explanatory": random_not_explain,
        "pass": one_gain and other_preserved and random_not_explain,
    }
    result["outcome"] = "PROCEED_TO_SAFE_STEP_EVALUATION" if result["E0_AUROC_gate"]["pass"] else "REQUIRE_SAFE_STEP_OR_STOP"
    path = RUN_DIR / "evidence_adjudication.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"outcome": result["outcome"], "gate": result["E0_AUROC_gate"], "deltas": deltas, "metrics": result["metrics"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
