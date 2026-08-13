import json
import math
import sys
from collections import Counter
from pathlib import Path

import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
SOURCEBIAS_DIR = ROOT / "runs/sourcebias/2026-08-03"
sys.path.insert(0, str(SOURCEBIAS_DIR))

from b2_lineage_normalized import bootstrap, evaluate, fit_stats
from sourcebias_common import load_pools, point_in_bbox, split_ids


VARIANT = "E1_patch28_element_mode"


def element_predict(row, stats, patch_size):
    cells = {}
    for index, candidate in enumerate(row["candidates"]):
        x, y = candidate["point"]
        cell = (math.floor(x / patch_size), math.floor(y / patch_size))
        cells.setdefault(cell, []).append(index)
    reliability = stats["model_reliability"]
    winner = max(
        cells.values(),
        key=lambda indices: (
            len(indices),
            len({row["candidates"][index]["model"] for index in indices}),
            sum(reliability[row["candidates"][index]["model"]] for index in indices),
            -min(indices),
        ),
    )
    selected = max(
        winner,
        key=lambda index: (reliability[row["candidates"][index]["model"]], -index),
    )
    return list(row["candidates"][selected]["point"])


def evaluate_element(rows, stats, patch_size):
    return {
        row["id"]: bool(point_in_bbox(element_predict(row, stats, patch_size), row["target_bbox"]))
        for row in rows
    }


def main():
    config_path = RUN_DIR / "configs/q2a_variant.yaml"
    config = yaml.safe_load(config_path.read_text())
    if config["name"] != VARIANT or config["status"] != "result_blind_variant_freeze":
        raise ValueError("Q2a config mismatch")
    patch_size = config["proxy"]["patch_size"]
    b2_config = yaml.safe_load((SOURCEBIAS_DIR / "configs/b2_variants.yaml").read_text())
    original_variants = list(b2_config["combined_method_order"])
    variants = [*original_variants, VARIANT]
    contexts, pools = load_pools()
    context = contexts["7B"]
    rows = pools["7B_Uniform_Mixed_N12"]
    nested = {}
    selections = {}
    descriptive_element = {}
    for outer in range(5):
        test = [row for row in rows if row["outer_fold"] == outer]
        outer_dev = [row for row in rows if row["outer_fold"] != outer]
        inner_val_fold = (outer + 1) % 5
        inner_train = [row for row in rows if row["outer_fold"] not in (outer, inner_val_fold)]
        inner_val = [row for row in rows if row["outer_fold"] == inner_val_fold]
        inner_stats = fit_stats(inner_train)
        scores = []
        for order, variant in enumerate(variants):
            output = evaluate_element(inner_val, inner_stats, patch_size) if variant == VARIANT else evaluate(inner_val, variant, inner_stats)
            scores.append((sum(output.values()) / len(output), -order, variant))
        selected = max(scores)[2]
        refit = fit_stats(outer_dev)
        fold_output = evaluate_element(test, refit, patch_size) if selected == VARIANT else evaluate(test, selected, refit)
        nested.update(fold_output)
        descriptive_element.update(evaluate_element(test, refit, patch_size))
        selections[str(outer)] = {
            "inner_validation_fold": inner_val_fold,
            "selected_variant": selected,
            "inner_validation_accuracy": max(scores)[0],
            "element_inner_validation_accuracy": next(score[0] for score in scores if score[2] == VARIANT),
            "outer_test_rows": len(test),
        }

    original_path = SOURCEBIAS_DIR / "results/recovery_b2_lineage_normalized.json"
    original = json.loads(original_path.read_text())
    original_outputs = original["reports"]["7B"]["outputs"]["nested_LN"]
    original_outputs = {row_id: bool(value) for row_id, value in original_outputs.items()}
    if set(nested) != set(original_outputs) or len(nested) != 1581:
        raise ValueError("Q2a output coverage mismatch")
    comparison = bootstrap(rows, nested, original_outputs)
    result = {
        "schema_version": 1,
        "status": "PASS",
        "config": str(config_path.relative_to(ROOT)),
        "original_method_count": len(original_variants),
        "new_method_count": len(variants),
        "original_combined24": {
            "nested_accuracy": original["reports"]["7B"]["accuracy"]["nested_LN"],
            "selection_frequency": original["reports"]["7B"]["selection_frequency"],
        },
        "combined25": {
            "nested_accuracy": sum(nested.values()) / len(nested),
            "selection_frequency": dict(Counter(value["selected_variant"] for value in selections.values())),
            "outer_selections": selections,
        },
        "element_variant_descriptive_crossfit_accuracy": sum(descriptive_element.values()) / len(descriptive_element),
        "combined25_vs_combined24": comparison,
        "element_selected": any(value["selected_variant"] == VARIANT for value in selections.values()),
        "claim": (
            "ELEMENT_SPACE_IMPROVES_NESTED_SELECTION"
            if comparison["point_delta"] > 0 and comparison["ci_99"][0] > 0
            else "ELEMENT_SPACE_NOT_SUPPORTED"
        ),
        "outputs": {"combined25_nested": nested, "element_descriptive": descriptive_element},
    }
    (RUN_DIR / "q2a_element_space.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("original_combined24", "combined25", "element_variant_descriptive_crossfit_accuracy", "combined25_vs_combined24", "element_selected", "claim")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()