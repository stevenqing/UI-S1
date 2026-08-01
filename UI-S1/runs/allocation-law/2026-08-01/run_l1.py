import argparse
import json
from pathlib import Path

from allocation_eval import (
    BUDGETS,
    EXPECTED_ROWS,
    JOINT_BUDGETS,
    build_pool,
    compact_evaluation,
    l1_predictions,
    load_gta1,
    load_l1_units,
    load_manifest,
    load_model_views,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gta1-shards", type=Path, required=True)
    parser.add_argument("--qwen3-old", type=Path, required=True)
    parser.add_argument("--qwen3-extended", type=Path, nargs="+", required=True)
    parser.add_argument("--uitars-old", type=Path, required=True)
    parser.add_argument("--uitars-extended", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    gta1 = load_gta1(args.gta1_shards, manifest)
    gta1_counts = [len(row["candidates"]) for row in gta1.values()]
    if min(gta1_counts) >= 24:
        raise ValueError("V-only N24 was frozen unavailable but all rows now provide 24 candidates")
    generated = {
        "Qwen3-VL-8B-Instruct": load_model_views(
            args.qwen3_old, args.qwen3_extended, manifest, "Qwen3-VL-8B-Instruct"
        ),
        "UI-TARS-7B-SFT": load_model_views(
            args.uitars_old, args.uitars_extended, manifest, "UI-TARS-7B-SFT"
        ),
    }
    units = load_l1_units(args.config)
    evaluations = {"v_only": {}, "mixed": {}}
    curves = {"v_only": {}, "mixed": {}}
    for budget in BUDGETS:
        mixed = compact_evaluation(build_pool(gta1, generated, units[budget]))
        evaluations["mixed"][str(budget)] = {key: value for key, value in mixed.items() if key != "outputs"}
        curves["mixed"][budget] = mixed["accuracy"]
        if budget in JOINT_BUDGETS:
            v_units = [("GTA1-7B", view) for view in range(budget)]
            v_only = compact_evaluation(build_pool(gta1, generated, v_units))
            evaluations["v_only"][str(budget)] = {key: value for key, value in v_only.items() if key != "outputs"}
            curves["v_only"][budget] = v_only["accuracy"]
        else:
            evaluations["v_only"][str(budget)] = {
                "status": "TRUNCATED_UNAVAILABLE",
                "reason": "at least one row has fewer than 24 unique GTA1 candidates",
            }

    predictions = l1_predictions(curves)
    oracle_gap = {
        pool: {
            str(budget): {
                rule: accuracy["pass_at_n"] - accuracy[rule]
                for rule in ("B3_mvp", "M1_ccm")
            }
            for budget, accuracy in values.items()
        }
        for pool, values in curves.items()
    }
    result = {
        "schema_version": 1,
        "status": "PASS",
        "rows": EXPECTED_ROWS,
        "budgets": list(BUDGETS),
        "v_only_n24": "TRUNCATED_UNAVAILABLE",
        "gta1_candidate_count_range": [min(gta1_counts), max(gta1_counts)],
        "evaluations": evaluations,
        "oracle_gap": oracle_gap,
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "accuracy": {pool: {str(budget): accuracy for budget, accuracy in values.items()} for pool, values in curves.items()},
        "predictions": predictions,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
