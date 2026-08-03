import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
ALLOCATION_DIR = ROOT / "runs/allocation-law/2026-08-01"
DIVERSITY_DIR = ROOT / "runs/diversity-axis/2026-08-02"
H1_DIR = ROOT / "runs/ccm-h2h/2026-07-31/h1"
sys.path.insert(0, str(DIVERSITY_DIR / "x1"))
sys.path.insert(0, str(DIVERSITY_DIR))
sys.path.insert(0, str(ALLOCATION_DIR))
sys.path.insert(0, str(H1_DIR))
from guirc_port import region_consistency_vote
from x3_curve_stats import slope
from allocation_eval import group_folds, load_gta1, load_manifest
from aggregators_coord import mvp_official
from run_l2 import stratified_group_sample_counts


BUDGETS = (4, 8, 12, 16)
METHODS = ("GUI_RC", "B3_mvp")
SEED = 20260802
RESAMPLES = 10000


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sample_seed(stable_index, sample_index):
    return SEED + stable_index * 16 + sample_index


def load_samples(paths):
    rows = {}
    for path in sorted(paths):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in rows:
                raise ValueError(f"F2 duplicate identity: {row['id']}")
            stable_index = row["stable_index"]
            if row["samples"] != 16 or len(row["predictions"]) != 16:
                raise ValueError(f"F2 sample budget mismatch: {row['id']}")
            if row["temperature"] != 0.5 or row["top_p"] != 0.95:
                raise ValueError(f"F2 sampling protocol mismatch: {row['id']}")
            if row["shard_index"] != stable_index % 8 or row["num_shards"] != 8:
                raise ValueError(f"F2 shard mismatch: {row['id']}")
            if canonical_hash(row["predictions"]) != row["prediction_sha256"]:
                raise ValueError(f"F2 prediction hash mismatch: {row['id']}")
            for index, prediction in enumerate(row["predictions"]):
                if prediction["sample_index"] != index or prediction["seed"] != sample_seed(stable_index, index):
                    raise ValueError(f"F2 seed/order mismatch: {row['id']}/{index}")
                point = prediction["point"]
                if point is not None and (len(point) != 2 or not all(math.isfinite(float(value)) for value in point)):
                    raise ValueError(f"F2 invalid point: {row['id']}/{index}")
            if row["valid_predictions"] != sum(value["point"] is not None for value in row["predictions"]):
                raise ValueError(f"F2 valid count mismatch: {row['id']}")
            if "bbox" in row or "target_bbox" in row:
                raise ValueError(f"F2 target leaked into trace: {row['id']}")
            rows[row["id"]] = row
    if len(rows) != 1581 or sorted(row["stable_index"] for row in rows.values()) != list(range(1581)):
        raise ValueError(f"F2 requires 1,581 identities, found {len(rows)}")
    return rows


def point_in_bbox(point, bbox):
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def aggregate(points, candidates, width, height):
    valid = [(point, candidate) for point, candidate in zip(points, candidates) if point is not None]
    valid_points = [value[0] for value in valid]
    valid_candidates = [value[1] for value in valid]
    guirc = region_consistency_vote(points, width, height)["point"]
    b3 = mvp_official(valid_points, valid_candidates) if valid_points else [0.0, 0.0]
    return {"GUI_RC": guirc, "B3_mvp": b3}, valid_points


def evaluate(gta1, samples, pool, budget):
    outputs = {method: {} for method in (*METHODS, "pass_at_n")}
    valid_counts = []
    for row_id in sorted(gta1):
        source = gta1[row_id]
        width, height = source["img_size"]
        sample_points = [prediction["point"] for prediction in samples[row_id]["predictions"]]
        if pool == "S_only":
            points = sample_points[:budget]
            candidates = [{"coverage": 0.0, "region": [0, 0, width, height]} for _ in points]
        else:
            points, candidates = [], []
            for index in range(budget // 2):
                points.append(sample_points[index])
                candidates.append({"coverage": 0.0, "region": [0, 0, width, height]})
                view = source["candidates"][index]
                points.append(view["point"])
                candidates.append({"coverage": view["coverage"], "region": view["region"]})
        predictions, valid_points = aggregate(points, candidates, width, height)
        valid_counts.append(len(valid_points))
        for method, point in predictions.items():
            outputs[method][row_id] = point_in_bbox(point, source["target_bbox"])
        outputs["pass_at_n"][row_id] = any(point_in_bbox(point, source["target_bbox"]) for point in valid_points)
    return {
        "accuracy": {method: sum(values.values()) / len(values) for method, values in outputs.items()},
        "outputs": outputs,
        "valid_candidates": {
            "mean": float(np.mean(valid_counts)),
            "minimum": int(min(valid_counts)),
            "complete_rows": sum(value == budget for value in valid_counts),
        },
    }


def bootstrap_slopes(rows, evaluations):
    mapping, fold_rows = group_folds(rows)
    groups = sorted(mapping)
    group_index = {group: index for index, group in enumerate(groups)}
    row_counts = np.zeros(len(groups), dtype=np.int64)
    for row in rows:
        row_counts[group_index[row["application"]]] += 1
    sample_counts = stratified_group_sample_counts(groups, mapping, RESAMPLES, np.random.default_rng(SEED))
    denominators = sample_counts @ row_counts
    x_values = np.asarray(BUDGETS, dtype=np.float64)
    centered = x_values - x_values.mean()
    denominator = np.dot(centered, centered)
    reports = {}
    for pool in evaluations:
        reports[pool] = {}
        for method in METHODS:
            successes = np.zeros((len(groups), len(BUDGETS)), dtype=np.int64)
            for budget_index, budget in enumerate(BUDGETS):
                output = evaluations[pool][budget]["outputs"][method]
                for row in rows:
                    successes[group_index[row["application"]], budget_index] += int(output[row["id"]])
            bootstrap_accuracy = (sample_counts @ successes) / denominators[:, None]
            values = (bootstrap_accuracy @ centered) / denominator
            point_values = [evaluations[pool][budget]["accuracy"][method] for budget in BUDGETS]
            reports[pool][method] = {
                "point_slope_per_forward": slope(point_values),
                "bootstrap_mean": float(np.mean(values)),
                "ci_99": [float(np.quantile(values, 0.005)), float(np.quantile(values, 0.995))],
                "p_slope_nonnegative": float((1 + np.sum(values >= 0)) / (RESAMPLES + 1)),
                "resamples": RESAMPLES,
                "seed": SEED,
            }
    return reports, {"groups": len(groups), "fold_rows": fold_rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples = load_samples(args.shards)
    manifest = load_manifest(ALLOCATION_DIR / "raw/shared_regions_n12.jsonl")
    gta1 = load_gta1(ROOT / "runs/ccm-h2h/2026-07-31/h1/shards/top18", manifest)
    if set(samples) != set(gta1):
        raise ValueError("F2 sample/GTA1 identity mismatch")
    evaluations = {
        pool: {budget: evaluate(gta1, samples, pool, budget) for budget in BUDGETS}
        for pool in ("S_only", "sampling_plus_views")
    }
    rows = [{"id": row_id, "application": gta1[row_id]["application"]} for row_id in sorted(gta1)]
    slopes, bootstrap = bootstrap_slopes(rows, evaluations)
    primary = slopes["S_only"]["GUI_RC"]
    covered = primary["ci_99"][1] < 0
    result = {
        "schema_version": 1,
        "status": "PASS",
        "budgets": list(BUDGETS),
        "pools": {
            pool: {
                str(budget): {
                    key: value for key, value in evaluation.items() if key != "outputs"
                }
                for budget, evaluation in values.items()
            }
            for pool, values in evaluations.items()
        },
        "slopes": slopes,
        "bootstrap_design": bootstrap,
        "mixed_sampling": "UNAVAILABLE_NO_PREEXISTING_CROSS_LINEAGE_RANDOM_TRACES",
        "prediction": {
            "primary_pool": "S_only",
            "primary_rule": "GUI_RC",
            "sampling_axis_covered": covered,
            "title_scope": "single_model_diversity_axis" if covered else "fixed_view_allocation_axis",
        },
        "source": {
            "model": "GTA1-7B",
            "samples_per_row": 16,
            "temperature": 0.5,
            "top_p": 0.95,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "accuracy": {pool: {str(n): value["accuracy"] for n, value in values.items()} for pool, values in evaluations.items()},
        "slopes": slopes,
        "prediction": result["prediction"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()