import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from common import ROOT, UPSTREAM, load_module, write_json


SEED = 20260731
PERMUTATIONS = 1000


w1 = load_module(UPSTREAM / "w1_run.py", "h2_w1")
w2 = load_module(UPSTREAM / "w2_analyze.py", "h2_w2")


def pair_result(left, right, pair_id, axis):
    observed = w1.cohen_kappa(left, right)
    rng = np.random.default_rng(SEED + sum(ord(char) for char in pair_id))
    null = np.asarray([
        w1.cohen_kappa(left, rng.permutation(right)) for _ in range(PERMUTATIONS)
    ])
    return {
        "id": pair_id,
        "axis": axis,
        "rows": len(left),
        "observed_kappa": observed,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std()),
        "p_greater_equal": float((1 + np.count_nonzero(null >= observed)) / (PERMUTATIONS + 1)),
        "permutations": PERMUTATIONS,
    }


def android_view_pairs():
    output = []
    for model in w2.AC_MODELS:
        for setting in w2.AC_SETTINGS:
            clean = w2.clean_ac_indices(setting)
            full = w2.read_jsonl(w2.ac_prediction_path(model, setting, "full"))
            view = w2.read_jsonl(w2.ac_prediction_path(model, setting, "v1"))
            left = np.asarray([
                not w2.label_android_row(full[index])["step_success"] for index in clean
            ], dtype=np.int8)
            right = np.asarray([
                not w2.label_android_row(view[index])["step_success"] for index in clean
            ], dtype=np.int8)
            output.append(pair_result(
                left, right, f"androidcontrol/{setting}/{model}/full-v1", "same-model-view"
            ))
    return output


def mind2web_view_pair():
    full = w2.load_mind2web_full()
    view = w2.read_jsonl(w2.m2w_cell_path("v1"))
    left = np.asarray([not row["success"] for row in full], dtype=np.int8)
    right = np.asarray([not row["success"] for row in view], dtype=np.int8)
    return pair_result(left, right, "mind2web/visual/tongui-7b/full-v1", "same-model-view-descriptive")


def model_axis_pairs():
    output = []
    family_map = {
        "ui-agile-3b": "ui-agile", "ui-agile-7b": "ui-agile",
        "gui-r1-3b": "gui-r1", "gui-r1-7b": "gui-r1", "ui-r1-e-3b": "ui-r1-e",
        "tongui-3b": "tongui", "tongui-7b": "tongui", "tongui-32b": "tongui",
        "ui-tars-7b": "ui-tars", "ui-tars-72b": "ui-tars", "cogagent-18b": "cogagent",
    }
    for bench, setting in w1.POOLS:
        identities, available, pivot = w1.load_pool(bench, setting)
        models = w1.deployable_models(identities, available, pivot)
        for left_model, right_model in itertools.combinations(models, 2):
            left = np.asarray([not pivot[row_id][left_model]["success"] for row_id in identities], dtype=np.int8)
            right = np.asarray([not pivot[row_id][right_model]["success"] for row_id in identities], dtype=np.int8)
            same = family_map.get(left_model, left_model) == family_map.get(right_model, right_model)
            axis = "same-family-scale" if same else "cross-family"
            output.append(pair_result(
                left, right, f"{bench}/{setting}/{left_model}-{right_model}", axis
            ))
    return output


def primary_axis_test(view_pairs, model_pairs):
    view_values = [item["observed_kappa"] for item in view_pairs if item["id"].startswith("androidcontrol/")]
    cross_values = [
        item["observed_kappa"] for item in model_pairs
        if item["axis"] == "cross-family" and item["id"].startswith("androidcontrol/")
    ]
    observed = float(np.mean(view_values) - np.mean(cross_values))
    combined = np.asarray(view_values + cross_values)
    size = len(view_values)
    combinations = list(itertools.combinations(range(len(combined)), size))
    if len(combinations) <= 100000:
        null = []
        all_indices = set(range(len(combined)))
        for selected in combinations:
            selected_set = set(selected)
            other = sorted(all_indices - selected_set)
            null.append(float(np.mean(combined[list(selected)]) - np.mean(combined[other])))
        mode = "exact_label_enumeration"
    else:
        rng = np.random.default_rng(SEED)
        null = []
        for _ in range(100000):
            order = rng.permutation(len(combined))
            null.append(float(np.mean(combined[order[:size]]) - np.mean(combined[order[size:]])))
        mode = "100000_label_permutations"
    null = np.asarray(null)
    p = float(np.count_nonzero(null >= observed) / len(null))
    return {
        "view_axis_pairs": len(view_values),
        "cross_family_pairs": len(cross_values),
        "view_axis_mean_kappa": float(np.mean(view_values)),
        "cross_family_mean_kappa": float(np.mean(cross_values)),
        "difference": observed,
        "test": mode,
        "null_draws": len(null),
        "p_greater_equal": p,
        "prediction_satisfied": observed > 0 and p < 0.01,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    view_pairs = android_view_pairs()
    view_pairs.append(mind2web_view_pair())
    model_pairs = model_axis_pairs()
    primary = primary_axis_test(view_pairs, model_pairs)
    same_family = [item["observed_kappa"] for item in model_pairs if item["axis"] == "same-family-scale"]
    result = {
        "status": "PASS",
        "seed": SEED,
        "pairwise_permutations": PERMUTATIONS,
        "view_pairs": view_pairs,
        "model_pairs": model_pairs,
        "summary": {
            "primary": primary,
            "same_family_scale_mean_kappa": float(np.mean(same_family)) if same_family else None,
            "same_family_scale_pairs": len(same_family),
            "h2_positive": primary["prediction_satisfied"],
            "h3_gate": "OPEN" if primary["prediction_satisfied"] else "BLOCKED_HK3",
        },
    }
    write_json(args.output, result)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
