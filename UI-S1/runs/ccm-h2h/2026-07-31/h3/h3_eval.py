import argparse
import bisect
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from aggregators_coord import mvp_official


BINS = 8
MIN_CLASS = 32
H1_MDE = 0.007043345177520599


def point_in_bbox(point, bbox):
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def similarity(left, right):
    return math.exp(-(math.dist(left, right) ** 2) / (2 * 14.0**2))


def pair_type(left, right):
    if left["model"] == right["model"]:
        return "same-model-diff-view"
    if left["view_index"] == right["view_index"]:
        return "cross-lineage-same-view"
    return "cross-lineage-diff-view"


def fit_table(observations):
    labels = [label for _, label in observations]
    positive = sum(labels)
    negative = len(labels) - positive
    if positive < MIN_CLASS or negative < MIN_CLASS:
        raise ValueError("H3 LR cell lacks both classes")
    ordered = sorted(value for value, _ in observations)
    boundaries = tuple(ordered[math.ceil(len(ordered) * index / BINS) - 1] for index in range(1, BINS))
    pos_counts = [0] * BINS
    neg_counts = [0] * BINS
    for value, label in observations:
        index = bisect.bisect_right(boundaries, value)
        (pos_counts if label else neg_counts)[index] += 1
    ratios = tuple(
        math.log((pos_counts[index] + 1) / (positive + BINS))
        - math.log((neg_counts[index] + 1) / (negative + BINS))
        for index in range(BINS)
    )
    return {"boundaries": boundaries, "log_ratios": ratios, "successes": positive, "failures": negative}


def table_score(table, value):
    return table["log_ratios"][bisect.bisect_right(table["boundaries"], value)]


def fit_ccm(rows):
    source_counts = defaultdict(Counter)
    observations = defaultdict(list)
    for row in rows:
        candidates = row["candidates"]
        labels = [point_in_bbox(candidate["point"], row["target_bbox"]) for candidate in candidates]
        for candidate, label in zip(candidates, labels):
            source_counts[candidate["model"]]["rows"] += 1
            source_counts[candidate["model"]]["successes"] += int(label)
        for i, candidate in enumerate(candidates):
            for j, voter in enumerate(candidates):
                if i != j:
                    observations[pair_type(candidate, voter)].append((similarity(candidate["point"], voter["point"]), labels[i]))
    tables = {key: fit_table(values) for key, values in observations.items()}
    priors = {source: (counts["successes"] + 1) / (counts["rows"] + 2) for source, counts in source_counts.items()}
    return tables, priors


def ccm_select(row, tables, priors):
    candidates = row["candidates"]
    scores = []
    for i, candidate in enumerate(candidates):
        prior = priors[candidate["model"]]
        score = math.log(prior / (1 - prior))
        by_type = defaultdict(list)
        for j, voter in enumerate(candidates):
            if i == j:
                continue
            kind = pair_type(candidate, voter)
            by_type[kind].append(table_score(tables[kind], similarity(candidate["point"], voter["point"])))
        score += sum(sum(values) / len(values) for values in by_type.values())
        scores.append(score)
    return max(range(len(candidates)), key=lambda index: (scores[index], -index))


def group_folds(rows):
    counts = Counter(row["application"] for row in rows)
    loads = [0] * 5
    mapping = {}
    for group, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        fold = min(range(5), key=lambda index: (loads[index], index))
        mapping[group] = fold
        loads[fold] += count
    return mapping, loads


def evaluate_pool(rows):
    mapping, loads = group_folds(rows)
    outputs = {"B3_mvp": {}, "M1_ccm": {}, "pass_at_12": {}}
    folds = []
    for fold in range(5):
        dev = [row for row in rows if mapping[row["application"]] != fold]
        test = [row for row in rows if mapping[row["application"]] == fold]
        tables, priors = fit_ccm(dev)
        counts = Counter()
        for row in test:
            candidates = row["candidates"]
            points = [candidate["point"] for candidate in candidates]
            pseudo = [{"coverage": candidate.get("coverage", 0), "region": candidate["region"]} for candidate in candidates]
            b3 = mvp_official(points, pseudo)
            m1 = candidates[ccm_select(row, tables, priors)]["point"]
            labels = {
                "B3_mvp": point_in_bbox(b3, row["target_bbox"]),
                "M1_ccm": point_in_bbox(m1, row["target_bbox"]),
                "pass_at_12": any(point_in_bbox(point, row["target_bbox"]) for point in points),
            }
            for method, label in labels.items():
                outputs[method][row["id"]] = bool(label)
                counts[method] += int(label)
        folds.append({
            "fold": fold, "dev_rows": len(dev), "test_rows": len(test),
            "accuracy": {method: counts[method] / len(test) for method in outputs},
            "source_priors": priors,
            "tables": {
                key: {"boundaries": list(value["boundaries"]), "log_ratios": list(value["log_ratios"]), "successes": value["successes"], "failures": value["failures"]}
                for key, value in tables.items()
            },
        })
    return {"rows": len(rows), "fold_rows": loads, "folds": folds, "accuracy": {method: sum(values.values()) / len(rows) for method, values in outputs.items()}, "outputs": outputs}


def load_generated(shard_root, model):
    rows = {}
    for path in sorted(shard_root.glob("shard-*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip(): continue
            row = json.loads(line)
            if row["id"] in rows or row["views"] != 4:
                raise ValueError(f"H3 {model} identity/views mismatch")
            rows[row["id"]] = row
    if len(rows) != 1581:
        raise ValueError(f"H3 {model} requires 1,581 rows, found {len(rows)}")
    return rows


def build_pools(gta1_path, qwen3_root, uitars_root):
    gta_rows = pq.read_table(gta1_path).to_pylist()
    gta_rows.sort(key=lambda row: row["id"])
    qwen3 = load_generated(qwen3_root, "qwen3")
    uitars = load_generated(uitars_root, "uitars")
    d1, d2 = [], []
    for row in gta_rows:
        if len(row["candidates"]) < 12:
            raise ValueError(f"D1 requires 12 GTA1 candidates: {row['id']}")
        base = {key: row[key] for key in ("id", "application", "target_bbox")}
        d1_candidates = [
            {"model": "GTA1-7B", "view_index": index, "point": candidate["point"], "region": candidate["region"], "coverage": candidate["coverage"]}
            for index, candidate in enumerate(row["candidates"][:12])
        ]
        mixed = []
        for model, source in (("GTA1-7B", row), ("Qwen3-VL-8B-Instruct", qwen3[row["id"]]), ("UI-TARS-7B-SFT", uitars[row["id"]])):
            predictions = source["candidates"][:4] if model == "GTA1-7B" else source["predictions"]
            for index, prediction in enumerate(predictions):
                mixed.append({
                    "model": model, "view_index": index,
                    "point": prediction["point"], "region": prediction["region"],
                    "coverage": prediction.get("coverage", 0),
                })
        if len(mixed) != 12:
            raise ValueError("D2 requires exactly 12 candidates")
        d1.append({**base, "candidates": d1_candidates})
        d2.append({**base, "candidates": mixed})
    return d1, d2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gta1-superset", type=Path, required=True)
    parser.add_argument("--qwen3-shards", type=Path, required=True)
    parser.add_argument("--uitars-shards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    d1, d2 = build_pools(args.gta1_superset, args.qwen3_shards, args.uitars_shards)
    results = {"D1_pure_views": evaluate_pool(d1), "D2_mixed": evaluate_pool(d2)}
    delta = results["D2_mixed"]["accuracy"]["M1_ccm"] - results["D1_pure_views"]["accuracy"]["M1_ccm"]
    compact = {
        "status": "PASS", "forward_budget": 12,
        "models": ["GTA1-7B", "Qwen3-VL-8B-Instruct", "UI-TARS-7B-SFT"],
        "pools": {key: {k: v for k, v in value.items() if k != "outputs"} for key, value in results.items()},
        "comparison": {
            "d2_minus_d1_m1": delta,
            "mde": H1_MDE,
            "prediction_satisfied": delta > H1_MDE,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
