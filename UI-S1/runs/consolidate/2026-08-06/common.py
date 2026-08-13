import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
SOURCEBIAS_DIR = ROOT / "runs/sourcebias/2026-08-03"
sys.path.insert(0, str(SOURCEBIAS_DIR))

from sourcebias_common import fixed_rows, load_pools, point_in_bbox, rule_outputs, split_ids


MODELS = ("GTA1-7B", "Qwen3-VL-8B-Instruct", "UI-TARS-7B-SFT")
VIEWS = tuple(range(12))
ROWS = 1581
SEED = 20260806
RESAMPLES = 10000


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_context():
    contexts, _ = load_pools()
    context = contexts["7B"]
    required = {(model, view) for model in MODELS for view in VIEWS}
    if not required.issubset(context["bank"]):
        raise ValueError("consolidation requires complete 3x12 action bank")
    if len(context["row_ids"]) != ROWS:
        raise ValueError("consolidation row count mismatch")
    return context


def evaluate_actions(context, actions):
    actions = tuple(actions)
    if len(actions) != len(set(actions)):
        raise ValueError("duplicate actions")
    rows = fixed_rows(context, actions, split_ids)
    outputs, _ = rule_outputs(context, rows, split_ids)
    pass_outputs = {
        row["id"]: any(point_in_bbox(candidate["point"], row["target_bbox"]) for candidate in row["candidates"])
        for row in rows
    }
    return {
        "rows": len(rows),
        "actions": [list(action) for action in actions],
        "accuracy": {
            "B3_mvp": sum(value["correct"] for value in outputs["B3_mvp"].values()) / len(rows),
            "M1_ccm": sum(value["correct"] for value in outputs["M1_ccm"].values()) / len(rows),
            "pass_at_n": sum(pass_outputs.values()) / len(rows),
        },
        "outputs": {
            "B3_mvp": {row_id: value["correct"] for row_id, value in outputs["B3_mvp"].items()},
            "M1_ccm": {row_id: value["correct"] for row_id, value in outputs["M1_ccm"].items()},
            "pass_at_n": pass_outputs,
        },
        "row_metadata": {
            row["id"]: {"application": row["application"], "outer_fold": row["outer_fold"]}
            for row in rows
        },
    }


def action_correctness(context):
    return {
        (model, view): {
            row_id: bool(point_in_bbox(context["bank"][(model, view)][row_id]["point"], context["metadata"][row_id]["target_bbox"]))
            for row_id in context["row_ids"]
        }
        for model in MODELS for view in VIEWS
    }


def cohen_kappa(left, right):
    left = np.asarray(left, dtype=np.bool_)
    right = np.asarray(right, dtype=np.bool_)
    observed = float(np.mean(left == right))
    left_rate = float(np.mean(left))
    right_rate = float(np.mean(right))
    expected = left_rate * right_rate + (1 - left_rate) * (1 - right_rate)
    return 1.0 if math.isclose(expected, 1.0) else (observed - expected) / (1 - expected)


def paired_group_bootstrap(metadata, left, right, resamples=RESAMPLES, seed=SEED):
    by_fold_group = defaultdict(lambda: defaultdict(list))
    for row_id, row in metadata.items():
        by_fold_group[row["outer_fold"]][row["application"]].append(row_id)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(resamples):
        selected = []
        for fold in sorted(by_fold_group):
            groups = sorted(by_fold_group[fold])
            for group in rng.choice(groups, size=len(groups), replace=True):
                selected.extend(by_fold_group[fold][group])
        samples.append(float(np.mean([int(left[row_id]) - int(right[row_id]) for row_id in selected])))
    point = float(np.mean([int(left[row_id]) - int(right[row_id]) for row_id in metadata]))
    return {
        "point_delta": point,
        "ci_99": [float(np.quantile(samples, 0.005)), float(np.quantile(samples, 0.995))],
        "p_delta_le_zero_plus_one": float((1 + sum(value <= 0 for value in samples)) / (resamples + 1)),
        "resamples": resamples,
        "seed": seed,
    }


def mixed_sequence():
    return tuple((model, view) for view in VIEWS for model in MODELS)


def geometry_features(context, actions, row_ids):
    pair_distances = []
    largest_cluster_shares = []
    cross_distances = []
    within_distances = []
    for row_id in row_ids:
        points = [context["bank"][action][row_id]["point"] for action in actions]
        distances = [math.dist(points[left], points[right]) for left in range(len(points)) for right in range(left + 1, len(points))]
        pair_distances.extend(distances)
        if points:
            cluster_sizes = [sum(math.dist(point, other) <= 14 for other in points) for point in points]
            largest_cluster_shares.append(max(cluster_sizes) / len(points))
        for left in range(len(actions)):
            for right in range(left + 1, len(actions)):
                target = within_distances if actions[left][0] == actions[right][0] else cross_distances
                target.append(math.dist(points[left], points[right]))
    pair = np.asarray(pair_distances, dtype=np.float64)
    cross_mean = float(np.mean(cross_distances)) if cross_distances else 0.0
    within_mean = float(np.mean(within_distances)) if within_distances else 0.0
    return {
        "pair_mean": float(np.mean(pair)),
        "pair_std": float(np.std(pair)),
        "pair_median": float(np.median(pair)),
        "pair_q90": float(np.quantile(pair, 0.9)),
        "largest_cluster_share": float(np.mean(largest_cluster_shares)),
        "cross_lineage_mean": cross_mean,
        "within_lineage_mean": within_mean,
        "cross_within_ratio": cross_mean / within_mean if within_mean > 0 else 0.0,
        "lineage_count": len({action[0] for action in actions}),
        "pool_size": len(actions),
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
