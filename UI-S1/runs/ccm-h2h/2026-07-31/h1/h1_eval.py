import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from aggregators_coord import (
    coordinate_mean,
    mvp_graph_centroid,
    mvp_official,
    mvp_paper_centroid,
    reguide_algorithm_level,
)
from ccm_coord import fit as fit_ccm, select as select_ccm


SEED = 20260731
N_VALUES = (2, 4, 10)


def point_in_bbox(point, bbox):
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def load_rows(path):
    rows = pq.read_table(path).to_pylist()
    rows.sort(key=lambda row: row["id"])
    if len(rows) != 1581 or len({row["id"] for row in rows}) != 1581:
        raise ValueError(f"H1 candidate coverage mismatch: {path}")
    if any(len(row["candidates"]) != row["candidate_count"] for row in rows):
        raise ValueError(f"H1 candidate count mismatch: {path}")
    return rows


def group_folds(rows, folds=5):
    counts = Counter(row["application"] for row in rows)
    loads = [0] * folds
    mapping = {}
    for group, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        fold = min(range(folds), key=lambda index: (loads[index], index))
        mapping[group] = fold
        loads[fold] += count
    if len(mapping) < folds:
        raise ValueError("H1 requires at least five application groups")
    return mapping, loads


def row_points(row):
    return [tuple(map(float, candidate["point"])) for candidate in row["candidates"]]


def calibration_rows(rows):
    output = []
    for row in rows:
        points = row_points(row)
        output.append((points, [point_in_bbox(point, row["target_bbox"]) for point in points]))
    return output


def choose_threshold(calibration, rows):
    records = []
    thresholds = set()
    for row in rows:
        points = row_points(row)
        winner, scores = select_ccm(calibration, points)
        gap = scores[winner] - scores[0]
        thresholds.add(gap)
        records.append((row, winner, gap))
    baseline = sum(point_in_bbox(row_points(row)[0], row["target_bbox"]) for row, _, _ in records)
    for threshold in sorted(value for value in thresholds if math.isfinite(value) and value >= 0) + [float("inf")]:
        success = 0
        for row, winner, gap in records:
            index = winner if gap >= threshold else 0
            success += point_in_bbox(row_points(row)[index], row["target_bbox"])
        if success >= baseline:
            return threshold, {"rows": len(rows), "baseline_successes": baseline, "selected_successes": success}
    raise AssertionError("infinite threshold must reproduce full image")


def aggregate_non_ccm(row, random_index):
    points = row_points(row)
    candidates = row["candidates"]
    return {
        "B0_full": points[0],
        "B1_random": points[random_index],
        "B2_mean": coordinate_mean(points, candidates),
        "B3_mvp_official": mvp_official(points, candidates),
        "B3_paper_centroid": mvp_paper_centroid(points, candidates),
        "B3_graph_centroid": mvp_graph_centroid(points, candidates),
        "B4_reguide_algorithm": reguide_algorithm_level(points, candidates),
    }


def evaluate_n(rows, count):
    mapping, fold_loads = group_folds(rows)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, count]))
    random_indices = {row["id"]: int(rng.integers(0, count)) for row in rows}
    methods = (
        "B0_full", "B1_random", "B2_mean", "B3_mvp_official",
        "B3_paper_centroid", "B3_graph_centroid", "B4_reguide_algorithm",
        "M1_ccm", "M2_ccm_risk", "pass_at_n",
    )
    outputs = {method: {} for method in methods}
    folds = []
    for fold in range(5):
        dev = [row for row in rows if mapping[row["application"]] != fold]
        test = [row for row in rows if mapping[row["application"]] == fold]
        calibration = fit_ccm(calibration_rows(dev))
        threshold_fold = (fold + 1) % 5
        threshold_dev = [row for row in dev if mapping[row["application"]] == threshold_fold]
        train = [row for row in dev if mapping[row["application"]] != threshold_fold]
        risk_calibration = fit_ccm(calibration_rows(train))
        threshold, threshold_report = choose_threshold(risk_calibration, threshold_dev)
        fold_counts = Counter()
        for row in test:
            points = row_points(row)
            predictions = aggregate_non_ccm(row, random_indices[row["id"]])
            winner, scores = select_ccm(calibration, points)
            predictions["M1_ccm"] = points[winner]
            risk_winner, risk_scores = select_ccm(risk_calibration, points)
            risk_gap = risk_scores[risk_winner] - risk_scores[0]
            predictions["M2_ccm_risk"] = points[risk_winner] if risk_gap >= threshold else points[0]
            labels = {
                method: point_in_bbox(point, row["target_bbox"])
                for method, point in predictions.items()
            }
            labels["pass_at_n"] = any(point_in_bbox(point, row["target_bbox"]) for point in points)
            for method, success in labels.items():
                outputs[method][row["id"]] = bool(success)
                fold_counts[method] += int(success)
        folds.append({
            "fold": fold,
            "dev_rows": len(dev),
            "test_rows": len(test),
            "threshold_fold": threshold_fold,
            "risk_train_rows": len(train),
            "risk_threshold_rows": len(threshold_dev),
            "risk_threshold": threshold,
            "risk_threshold_report": threshold_report,
            "ccm_calibration": {
                "successes": calibration.successes,
                "failures": calibration.failures,
                "boundaries": list(calibration.boundaries),
                "log_ratios": list(calibration.log_ratios),
            },
            "accuracy": {method: fold_counts[method] / len(test) for method in methods},
        })
    aggregate = {method: sum(values.values()) / len(rows) for method, values in outputs.items()}
    return {
        "candidate_count": count,
        "rows": len(rows),
        "groups": len(mapping),
        "fold_rows": fold_loads,
        "folds": folds,
        "accuracy": aggregate,
        "outputs": outputs,
    }


def grouped_bootstrap(rows, left, right, resamples=10000):
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["application"]].append(row["id"])
    groups = sorted(by_group)
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(resamples):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        ids = [row_id for group in sampled for row_id in by_group[group]]
        values.append(float(np.mean([left[row_id] - right[row_id] for row_id in ids])))
    return {
        "resamples": resamples,
        "seed": SEED,
        "mean": float(np.mean(values)),
        "ci_99": [float(np.quantile(values, 0.005)), float(np.quantile(values, 0.995))],
        "p_delta_le_zero": float(np.mean(np.asarray(values) <= 0)),
    }


def evaluate_seed_mde(paths):
    accuracies = []
    details = []
    for path in paths:
        rows = load_rows(path)
        result = evaluate_n(rows, 10)
        accuracy = result["accuracy"]["M1_ccm"]
        accuracies.append(accuracy)
        details.append({"path": str(path), "accuracy": accuracy})
    sample_sd = float(np.std(accuracies, ddof=1))
    return {"seeds": details, "sample_sd": sample_sd, "mde": 2 * sample_sd}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows_by_n = {count: load_rows(args.candidate_dir / f"candidates_N{count}.parquet") for count in N_VALUES}
    ids = {count: [row["id"] for row in rows] for count, rows in rows_by_n.items()}
    if len({tuple(value) for value in ids.values()}) != 1:
        raise ValueError("H1 N identity mismatch")
    results = {count: evaluate_n(rows_by_n[count], count) for count in N_VALUES}
    mde_paths = [args.candidate_dir / f"candidates_N10_seed{seed}.parquet" for seed in (20260731, 20260732, 20260733)]
    mde = evaluate_seed_mde(mde_paths)
    n10 = results[10]
    m1_vs_b3 = grouped_bootstrap(
        rows_by_n[10], n10["outputs"]["M1_ccm"], n10["outputs"]["B3_mvp_official"]
    )
    n10_n4 = grouped_bootstrap(
        rows_by_n[10], results[10]["outputs"]["M1_ccm"], results[4]["outputs"]["M1_ccm"]
    )
    accuracy = {str(count): results[count]["accuracy"] for count in N_VALUES}
    h1a_delta = accuracy["10"]["M1_ccm"] - accuracy["10"]["B3_mvp_official"]
    h1b_m1_delta = accuracy["10"]["M1_ccm"] - accuracy["4"]["M1_ccm"]
    h1b_b3_delta = accuracy["10"]["B3_mvp_official"] - accuracy["4"]["B3_mvp_official"]
    h1c_delta = accuracy["10"]["M1_ccm"] - 0.628
    predictions = {
        "P-H1a": {
            "delta_m1_minus_b3": h1a_delta,
            "mde": mde["mde"],
            "bootstrap": m1_vs_b3,
            "satisfied": h1a_delta > mde["mde"] and m1_vs_b3["p_delta_le_zero"] < 0.01,
        },
        "P-H1b": {
            "m1_n10_minus_n4": h1b_m1_delta,
            "m1_bootstrap": n10_n4,
            "b3_n10_minus_n4": h1b_b3_delta,
            "satisfied": h1b_m1_delta >= 0 and n10_n4["ci_99"][0] >= 0 and h1b_b3_delta < 0,
        },
        "P-H1c": {
            "paper_reference": 0.628,
            "delta": h1c_delta,
            "mde": mde["mde"],
            "satisfied": h1c_delta > mde["mde"],
            "paper_only": True,
        },
    }
    compact = {
        "status": "PASS",
        "accuracy": accuracy,
        "folds": {str(count): results[count]["folds"] for count in N_VALUES},
        "mde": mde,
        "predictions": predictions,
        "headroom_capture": {
            str(count): (
                (accuracy[str(count)]["M1_ccm"] - accuracy[str(count)]["B0_full"])
                / (accuracy[str(count)]["pass_at_n"] - accuracy[str(count)]["B0_full"])
                if accuracy[str(count)]["pass_at_n"] > accuracy[str(count)]["B0_full"] else None
            )
            for count in N_VALUES
        },
        "kill_conditions": {
            "H-K1": not predictions["P-H1a"]["satisfied"] and not predictions["P-H1c"]["satisfied"],
            "H-K2": not predictions["P-H1b"]["satisfied"] and h1b_b3_delta >= 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accuracy": accuracy, "mde": mde, "predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
