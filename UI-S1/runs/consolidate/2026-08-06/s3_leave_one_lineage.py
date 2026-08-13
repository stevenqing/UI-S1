import json
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(RUN_DIR))

from common import MODELS, evaluate_actions, load_context, paired_group_bootstrap, write_json


def main():
    context = load_context()
    full_actions = tuple((model, view) for view in range(4) for model in MODELS)
    pools = {
        "full_three_lineages_4x3": full_actions,
        "leave_out_UI_TARS": tuple((model, view) for view in range(6) for model in MODELS if model != "UI-TARS-7B-SFT"),
        "leave_out_Qwen3": tuple((model, view) for view in range(6) for model in MODELS if model != "Qwen3-VL-8B-Instruct"),
        "leave_out_GTA1": tuple((model, view) for view in range(6) for model in MODELS if model != "GTA1-7B"),
    }
    evaluations = {name: evaluate_actions(context, actions) for name, actions in pools.items()}
    full = evaluations["full_three_lineages_4x3"]
    expected = {"B3_mvp": 0.6369386464263125, "M1_ccm": 0.6382036685641999}
    if any(abs(full["accuracy"][metric] - value) > 1e-15 for metric, value in expected.items()):
        raise ValueError("S3 full-pool anchor mismatch")

    comparisons = {}
    for name, evaluation in evaluations.items():
        if name == "full_three_lineages_4x3":
            continue
        comparisons[f"full_minus_{name}"] = {
            metric: {
                **paired_group_bootstrap(full["row_metadata"], full["outputs"][metric], evaluation["outputs"][metric]),
                "full_accuracy": full["accuracy"][metric],
                "leave_one_accuracy": evaluation["accuracy"][metric],
            }
            for metric in ("B3_mvp", "M1_ccm")
        }

    best_two_b3_name = max(
        (name for name in evaluations if name != "full_three_lineages_4x3"),
        key=lambda name: evaluations[name]["accuracy"]["B3_mvp"],
    )
    best_comparison = comparisons[f"full_minus_{best_two_b3_name}"]["B3_mvp"]
    s_k2 = best_comparison["point_delta"] <= 0 or best_comparison["ci_99"][0] <= 0
    positive_lineages = []
    for omitted in ("UI_TARS", "Qwen3", "GTA1"):
        comparison = comparisons[f"full_minus_leave_out_{omitted}"]["B3_mvp"]
        if comparison["point_delta"] > 0 and comparison["ci_99"][0] > 0:
            positive_lineages.append(omitted)
    result = {
        "schema_version": 1,
        "status": "PASS",
        "budget": 12,
        "pools": {
            name: {"actions": evaluation["actions"], "accuracy": evaluation["accuracy"]}
            for name, evaluation in evaluations.items()
        },
        "comparisons": comparisons,
        "best_two_lineage_pool_by_B3": best_two_b3_name,
        "S_K2": s_k2,
        "lineages_with_significantly_positive_marginal_B3": positive_lineages,
        "claim": (
            "LINEAGE_DIVERSITY_WITH_THIRD_LINEAGE_SATURATION"
            if s_k2 else "THREE_LINEAGES_SIGNIFICANTLY_OUTPERFORM_ALL_TWO_LINEAGE_ABLATIONS"
        ),
    }
    write_json(RUN_DIR / "s3_leave_one_lineage.json", result)
    print(json.dumps({
        "S_K2": s_k2,
        "best_two": best_two_b3_name,
        "accuracies": {name: value["accuracy"] for name, value in evaluations.items()},
        "positive_lineages": positive_lineages,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
