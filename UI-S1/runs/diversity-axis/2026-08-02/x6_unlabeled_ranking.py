import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
ALLOCATION_DIR = ROOT / "runs/allocation-law/2026-08-01"
sys.path.insert(0, str(RUN_DIR / "x2"))
sys.path.insert(0, str(RUN_DIR))
sys.path.insert(0, str(ALLOCATION_DIR))
from x2_composability import build_q2, build_q4, load_trace, validate_source_identity
from x3_curve_stats import load_sources
from allocation_eval import build_pool, compact_evaluation, group_folds, l2_units


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_pairwise_normalized_distance(rows, image_sizes, row_filter=None):
    selected = rows if row_filter is None else [row for row in rows if row_filter(row)]
    if not selected:
        raise ValueError("X6 feature split is empty")
    values = []
    for row in selected:
        points = np.asarray([candidate["point"] for candidate in row["candidates"]], dtype=np.float64)
        if points.shape != (12, 2) or not np.all(np.isfinite(points)):
            raise ValueError(f"X6 requires 12 finite points: {row['id']}")
        width, height = image_sizes[row["id"]]
        diagonal = math.hypot(width, height)
        distances = [
            math.dist(points[left], points[right]) / diagonal
            for left in range(12)
            for right in range(left + 1, 12)
        ]
        values.append(float(np.mean(distances)))
    return float(np.mean(values))


def fit_ols(records):
    feature = np.asarray([record["feature"] for record in records], dtype=np.float64)
    target = np.asarray([record["target"] for record in records], dtype=np.float64)
    design = np.column_stack((np.ones(len(feature)), feature))
    intercept, coefficient = np.linalg.lstsq(design, target, rcond=None)[0]
    prediction = intercept + coefficient * feature
    residual = np.sum((target - prediction) ** 2)
    total = np.sum((target - target.mean()) ** 2)
    correlation = spearmanr(prediction, target)
    return {
        "intercept": float(intercept),
        "coefficient": float(coefficient),
        "training_r_squared": float(1 - residual / total),
        "training_spearman": {
            "rho": float(correlation.statistic),
            "p_value": float(correlation.pvalue),
        },
    }


def fit(output):
    gta1, generated, _ = load_sources()
    image_sizes = {row_id: row["img_size"] for row_id, row in gta1.items()}
    pools = l2_units(ALLOCATION_DIR / "configs/l2_pools.yaml")
    l2_path = ALLOCATION_DIR / "L2_RESULTS.json"
    l2 = json.loads(l2_path.read_text())
    records = []
    reference_mapping = None
    for pool_name, units in pools.items():
        rows = build_pool(gta1, generated, units)
        mapping, _ = group_folds(rows)
        if reference_mapping is None:
            reference_mapping = mapping
        elif mapping != reference_mapping:
            raise ValueError("X6 L2 fold mapping mismatch")
        source_folds = {record["fold"]: record for record in l2["pools"][pool_name]["folds"]}
        for fold in range(5):
            records.append({
                "pool": pool_name,
                "fold": fold,
                "feature": mean_pairwise_normalized_distance(
                    rows, image_sizes,
                    lambda row, fold=fold: mapping[row["application"]] != fold,
                ),
                "target": source_folds[fold]["heldout_accuracy"]["pass_at_n"],
            })
    if len(records) != 40:
        raise ValueError("X6 fit requires 40 L2 observations")
    model = fit_ols(records)
    result = {
        "schema_version": 1,
        "status": "FROZEN_FIT_BEFORE_X2_VALIDATION",
        "feature": "dev_mean_pairwise_normalized_distance",
        "target": "heldout_pass_at_12",
        "training_observations": len(records),
        "model": model,
        "records": records,
        "source": {"L2_RESULTS_sha256": sha256_file(l2_path)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "model": model}, indent=2, sort_keys=True))


def validate(args):
    fit_result = json.loads(args.fit.read_text())
    if fit_result["status"] != "FROZEN_FIT_BEFORE_X2_VALIDATION" or fit_result["training_observations"] != 40:
        raise ValueError("X6 frozen fit contract mismatch")
    gta1, _, _ = load_sources()
    image_sizes = {row_id: row["img_size"] for row_id, row in gta1.items()}
    q2_trace = load_trace(args.q2, "Q2", "GTA1-7B", 12)
    q4_traces = {
        "GTA1-7B": load_trace(args.q4_gta1, "Q4", "GTA1-7B", 4),
        "Qwen3-VL-8B-Instruct": load_trace(args.q4_qwen3, "Q4", "Qwen3-VL-8B-Instruct", 4),
        "UI-TARS-7B-SFT": load_trace(args.q4_uitars, "Q4", "UI-TARS-7B-SFT", 4),
    }
    validate_source_identity([q2_trace, *q4_traces.values()], gta1)
    pools = {"Q2": build_q2(gta1, q2_trace), "Q4": build_q4(gta1, q4_traces)}
    intercept = fit_result["model"]["intercept"]
    coefficient = fit_result["model"]["coefficient"]
    records = []
    reference_mapping = None
    for pool_name, rows in pools.items():
        mapping, _ = group_folds(rows)
        if reference_mapping is None:
            reference_mapping = mapping
        elif mapping != reference_mapping:
            raise ValueError("X6 X2 fold mapping mismatch")
        evaluation = compact_evaluation(rows)
        for fold in range(5):
            feature = mean_pairwise_normalized_distance(
                rows, image_sizes,
                lambda row, fold=fold: mapping[row["application"]] != fold,
            )
            records.append({
                "pool": pool_name,
                "fold": fold,
                "feature": feature,
                "predicted_pass_at_12": intercept + coefficient * feature,
                "actual_heldout_pass_at_12": evaluation["folds"][fold]["accuracy"]["pass_at_n"],
            })
    if len(records) != 10:
        raise ValueError("X6 validation requires 10 held-out observations")
    correlation = spearmanr(
        [record["predicted_pass_at_12"] for record in records],
        [record["actual_heldout_pass_at_12"] for record in records],
    )
    rho = float(correlation.statistic)
    result = {
        "schema_version": 1,
        "status": "PASS" if rho > 0.7 else "FAIL_HELDOUT_SPEARMAN",
        "fit_sha256": sha256_file(args.fit),
        "heldout_observations": len(records),
        "heldout_pools": ["Q2", "Q4"],
        "spearman": {"rho": rho, "p_value": float(correlation.pvalue)},
        "minimum_rho": 0.7,
        "prediction_X6": rho > 0.7,
        "records": records,
        "claim_boundary": "Ten observations from two new pools are a minimal low-power held-out validation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "spearman": result["spearman"],
        "prediction_X6": result["prediction_X6"],
    }, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--output", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--fit", type=Path, required=True)
    validate_parser.add_argument("--q2", type=Path, nargs="+", required=True)
    validate_parser.add_argument("--q4-gta1", type=Path, nargs="+", required=True)
    validate_parser.add_argument("--q4-qwen3", type=Path, nargs="+", required=True)
    validate_parser.add_argument("--q4-uitars", type=Path, nargs="+", required=True)
    validate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fit":
        fit(args.output)
    else:
        validate(args)


if __name__ == "__main__":
    main()