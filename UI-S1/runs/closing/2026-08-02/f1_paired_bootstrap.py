import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
ALLOCATION_DIR = ROOT / "runs/allocation-law/2026-08-01"
H1_DIR = ROOT / "runs/ccm-h2h/2026-07-31/h1"
sys.path.insert(0, str(RUN_DIR))
sys.path.insert(0, str(ALLOCATION_DIR))
sys.path.insert(0, str(H1_DIR))
from closing_common import load_closing_pools
from allocation_eval import group_folds, point_in_bbox
from aggregators_coord import mvp_graph_centroid
from run_l2 import stratified_group_sample_counts


SEED = 20260802
RESAMPLES = 10000
MDE = 0.007043345177520599


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_bootstrap(rows, left, right, resamples=RESAMPLES, seed=SEED):
    mapping, fold_rows = group_folds(rows)
    groups = sorted(mapping)
    group_index = {group: index for index, group in enumerate(groups)}
    row_counts = np.zeros(len(groups), dtype=np.int64)
    group_deltas = np.zeros(len(groups), dtype=np.int64)
    for row in rows:
        index = group_index[row["application"]]
        row_counts[index] += 1
        group_deltas[index] += int(left[row["id"]]) - int(right[row["id"]])
    sample_counts = stratified_group_sample_counts(
        groups, mapping, resamples, np.random.default_rng(seed)
    )
    denominators = sample_counts @ row_counts
    values = (sample_counts @ group_deltas) / denominators
    point = float(sum(left.values()) / len(rows) - sum(right.values()) / len(rows))
    return {
        "rows": len(rows),
        "point_delta": point,
        "ci_99": [
            float(np.quantile(values, 0.005)),
            float(np.quantile(values, 0.995)),
        ],
        "p_one_sided_delta_le_zero": float((1 + np.sum(values <= 0)) / (resamples + 1)),
        "bootstrap_mean": float(np.mean(values)),
        "resamples": resamples,
        "seed": seed,
        "groups": len(groups),
        "fold_rows": fold_rows,
        "mde": MDE,
        "delta_over_mde": point / MDE,
    }


def comparison(pools, left_pool, left_method, right_pool, right_method):
    left = pools[left_pool]["evaluation"]
    right = pools[right_pool]["evaluation"]
    rows = pools[left_pool]["rows"]
    if [row["id"] for row in rows] != [row["id"] for row in pools[right_pool]["rows"]]:
        raise ValueError("F1 comparison identity mismatch")
    result = paired_bootstrap(
        rows, left["outputs"][left_method], right["outputs"][right_method]
    )
    result.update({
        "left": f"{left_pool}/{left_method}",
        "right": f"{right_pool}/{right_method}",
        "left_accuracy": left["accuracy"][left_method],
        "right_accuracy": right["accuracy"][right_method],
    })
    return result


def graph_centroid_outputs(pool):
    outputs = {}
    for row in pool["rows"]:
        points = [candidate["point"] for candidate in row["candidates"]]
        selected = mvp_graph_centroid(points, row["candidates"])
        outputs[row["id"]] = point_in_bbox(selected, row["target_bbox"])
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, pools = load_closing_pools()
    specifications = {
        "mixed_N12_M1_vs_v_only_GTA1_N12_M1": ("mixed_N12", "M1_ccm", "v_only_N12", "M1_ccm"),
        "mixed_N12_B3_vs_v_only_GTA1_N12_B3": ("mixed_N12", "B3_mvp", "v_only_N12", "B3_mvp"),
        "mixed_N12_M1_vs_Qwen3_N12_M1": ("mixed_N12", "M1_ccm", "qwen3_N12", "M1_ccm"),
        "mixed_N12_M1_vs_UI_TARS_N12_M1": ("mixed_N12", "M1_ccm", "uitars_N12", "M1_ccm"),
        "mixed_N16_M1_vs_v_only_N16_M1": ("mixed_N16", "M1_ccm", "v_only_N16", "M1_ccm"),
        "mixed_N16_B3_vs_v_only_N16_B3": ("mixed_N16", "B3_mvp", "v_only_N16", "B3_mvp"),
    }
    comparisons = {
        name: comparison(pools, *specification)
        for name, specification in specifications.items()
    }
    graph_v_only = graph_centroid_outputs(pools["v_only_N12"])
    graph_mixed = graph_centroid_outputs(pools["mixed_N12"])
    graph = paired_bootstrap(pools["mixed_N12"]["rows"], graph_mixed, graph_v_only)
    graph.update({
        "rule": "unchanged_H1_mvp_graph_centroid",
        "left": "mixed_N12/B3_graph_centroid",
        "right": "v_only_N12/B3_graph_centroid",
        "left_accuracy": sum(graph_mixed.values()) / len(graph_mixed),
        "right_accuracy": sum(graph_v_only.values()) / len(graph_v_only),
    })
    accuracies = {
        pool: values["evaluation"]["accuracy"]
        for pool, values in pools.items()
    }
    result = {
        "schema_version": 1,
        "status": "PASS",
        "comparisons": comparisons,
        "third_drop_in_graph_centroid": graph,
        "accuracies": accuracies,
        "lineage_quality": {
            "Qwen3_below_GTA1_M1": accuracies["v_only_N12"]["M1_ccm"] - accuracies["qwen3_N12"]["M1_ccm"],
            "UI_TARS_below_GTA1_M1": accuracies["v_only_N12"]["M1_ccm"] - accuracies["uitars_N12"]["M1_ccm"],
        },
        "sources": {
            "L1_RESULTS_sha256": sha256_file(ALLOCATION_DIR / "L1_RESULTS.json"),
            "L2_RESULTS_sha256": sha256_file(ALLOCATION_DIR / "L2_RESULTS.json"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        name: {
            "delta": value["point_delta"],
            "ci_99": value["ci_99"],
            "p": value["p_one_sided_delta_le_zero"],
        }
        for name, value in comparisons.items()
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()