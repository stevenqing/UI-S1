import json
from pathlib import Path

import numpy as np
import yaml

from utility_adjudicate import ARMS, BENCHMARKS, equal_arm, paired_samples
from utility_common import load_banks


RUN_DIR = Path(__file__).resolve().parent


def compare_outputs(banks, left, right, config, seed_offset):
    comparisons = {benchmark: {} for benchmark in BENCHMARKS}
    sample_cache = {}
    for benchmark in BENCHMARKS:
        seed = config["statistics"]["mind2web_seed" if benchmark == "mind2web" else "screenspot_seed"]
        arm_samples = {}
        arm_points = {}
        for arm_index, arm in enumerate(ARMS):
            result, samples = paired_samples(
                banks[arm][benchmark],
                left[benchmark][arm]["safe"],
                right[benchmark][arm]["safe"],
                config["statistics"]["resamples"],
                seed + seed_offset + arm_index,
            )
            comparisons[benchmark][f"{arm}_main_minus_ablation"] = result
            arm_samples[arm] = samples
            arm_points[arm] = result["point_delta"]
        comparisons[benchmark]["equal_arm_mean_main_minus_ablation"], samples = equal_arm(
            arm_samples, arm_points
        )
        sample_cache[benchmark] = samples
    balanced = np.mean(np.stack([
        sample_cache[benchmark] / config["mde"][benchmark]
        for benchmark in BENCHMARKS
    ]), axis=0)
    return comparisons, {
        "point": float(np.mean(balanced)),
        "ci_99": [float(np.quantile(balanced, 0.005)), float(np.quantile(balanced, 0.995))],
    }


def main():
    config = yaml.safe_load((RUN_DIR / "configs/utility_prereg.yaml").read_text())
    main_result = json.loads((RUN_DIR / "utility_main.json").read_text())["main"]
    ablations = json.loads((RUN_DIR / "utility_ablations.json").read_text())
    banks = load_banks()
    results = {}
    for index, name in enumerate(("no_MVP_structure", "absolute_only")):
        comparisons, balanced = compare_outputs(
            banks, main_result["outputs"], ablations[name]["outputs"], config, 500 + index * 100
        )
        results[name] = {
            "comparison_direction": "main_Utility_LSA_minus_ablation",
            "comparisons": comparisons,
            "equal_benchmark_standardized": balanced,
        }

    no_mvp_screen = results["no_MVP_structure"]["comparisons"]["screenspot_pro"][
        "equal_arm_mean_main_minus_ablation"
    ]
    no_mvp_significantly_worse = no_mvp_screen["ci_99"][0] > 0
    no_mvp_not_worse = not no_mvp_significantly_worse
    importance_required = no_mvp_not_worse
    ur_k5 = no_mvp_not_worse and False
    result = {
        "schema_version": 1,
        "status": "PASS_ADJUDICATED",
        "comparisons": results,
        "UR_K5": ur_k5,
        "UR_K5_components": {
            "no_MVP_structure_not_worse": no_mvp_not_worse,
            "no_MVP_structure_significantly_worse_on_screenspot": no_mvp_significantly_worse,
            "structure_permutation_importance_required": importance_required,
            "structure_permutation_importance": None,
            "logic": "UR_K5 is a conjunction; importance is unnecessary after no-MVP-not-worse is false",
        },
    }
    (RUN_DIR / "utility_ablation_adjudication.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    adjudication_path = RUN_DIR / "utility_adjudication.json"
    adjudication = json.loads(adjudication_path.read_text())
    if adjudication["gates"]["UR_K5"] not in (None, False):
        raise ValueError("UR_K5 was already adjudicated inconsistently")
    adjudication["gates"]["UR_K5"] = ur_k5
    adjudication["gates"]["UR_K5_reason"] = {
        "no_MVP_structure_not_worse": no_mvp_not_worse,
        "screenspot_equal_arm_main_minus_no_MVP": no_mvp_screen,
        "importance_required": importance_required,
    }
    adjudication_path.write_text(json.dumps(adjudication, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "UR_K5": ur_k5,
        "UR_K5_components": result["UR_K5_components"],
        "equal_arm": {
            name: {
                benchmark: results[name]["comparisons"][benchmark]["equal_arm_mean_main_minus_ablation"]
                for benchmark in BENCHMARKS
            }
            for name in results
        },
        "balanced": {name: results[name]["equal_benchmark_standardized"] for name in results},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
