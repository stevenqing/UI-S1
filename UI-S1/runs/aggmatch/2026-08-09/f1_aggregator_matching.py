import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from aggmatch_common import RUN_DIR, atomic_json, load_cache, method_difference, paired_bootstrap, sha256_file


CONFIG_PATH = RUN_DIR / "configs/f1_families.yaml"


def benchmark_comparisons(benchmark, config):
    cache = load_cache()[benchmark]
    metadata = cache["metadata"]
    outputs = cache["outputs"]
    mapping = config["aggregators"][benchmark]["density_family"]
    result = {}
    for arm in ("C_uni", "C_cond", "C_rand", "C_self"):
        result[arm] = {}
        for label, method in mapping.items():
            differences = method_difference(outputs, arm, "majority", arm, method)
            result[arm][label] = paired_bootstrap(
                metadata,
                differences,
                config["bootstrap"]["resamples"],
                config["bootstrap"][benchmark]["seed"],
            )
    return result


def action_strata(config):
    cache = load_cache()["mind2web"]
    outputs = cache["outputs"]
    metadata = cache["metadata"]
    result = {}
    for action in config["action_spaces"]["mind2web"]["action_strata"]:
        row_ids = [row_id for row_id, row in metadata.items() if row["action"] == action]
        differences = method_difference(outputs, "C_uni", "majority", "C_uni", "ours", row_ids)
        result[action] = paired_bootstrap(
            metadata,
            differences,
            config["bootstrap"]["resamples"],
            config["bootstrap"]["mind2web"]["seed"] + 100 + len(result),
        )
    return result


def draw_figure(result, path):
    labels = ["Sequential/B3", "A1", "A2", "A3", "A4"]
    mind_keys = ["sequential", "A1_geometric_median", "A2_density_medoid", "A3_joint_PKA_medoid", "A4_continuous_PKA"]
    screen_keys = ["B3_official", "A1_geometric_median", "A2_density_medoid", "A3_joint_PKA_medoid", "A4_continuous_PKA"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    ax = axes[0]
    positions = np.arange(len(labels))
    for offset, benchmark, keys, color, marker in (
        (-0.10, "Mind2Web", mind_keys, "#19647e", "o"),
        (0.10, "ScreenSpot-Pro", screen_keys, "#c14924", "s"),
    ):
        values = result["benchmarks"][benchmark.lower().replace("-", "_")]["comparisons"]["C_uni"]
        points = np.asarray([values[key]["point_delta"] * 100 for key in keys])
        lower = np.asarray([values[key]["ci_99"][0] * 100 for key in keys])
        upper = np.asarray([values[key]["ci_99"][1] * 100 for key in keys])
        ax.errorbar(positions + offset, points, yerr=[points - lower, upper - points], fmt=marker, color=color, capsize=3, label=benchmark)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(positions, labels, rotation=24, ha="right")
    ax.set_ylabel("Majority minus density (pp)")
    ax.set_title("C-uni: action-space reversal")
    ax.legend(frameon=False)

    ax = axes[1]
    strata = result["mind2web_action_strata"]
    action_labels = list(strata)
    points = np.asarray([strata[action]["point_delta"] * 100 for action in action_labels])
    lower = np.asarray([strata[action]["ci_99"][0] * 100 for action in action_labels])
    upper = np.asarray([strata[action]["ci_99"][1] * 100 for action in action_labels])
    ax.errorbar(np.arange(len(action_labels)), points, yerr=[points - lower, upper - points], fmt="o", color="#19647e", capsize=4)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(np.arange(len(action_labels)), action_labels)
    ax.set_ylabel("Majority minus sequential (pp)")
    ax.set_title("Mind2Web by action type")
    fig.suptitle("Aggregator-action-space matching (99% paired bootstrap CI)", fontfamily="DejaVu Serif", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if config["status"] != "FROZEN_BEFORE_RESULTS":
        raise ValueError("F1 config is not frozen")
    mind = benchmark_comparisons("mind2web", config)
    screen = benchmark_comparisons("screenspot_pro", config)
    strata = action_strata(config)
    primary_mind = mind["C_uni"]["sequential"]
    primary_screen = screen["C_uni"][config["aggregators"]["screenspot_pro"]["primary_opposite_direction_member"]]
    mind_a1_a4_positive = all(mind["C_uni"][key]["point_delta"] > 0 for key in (
        "A1_geometric_median", "A2_density_medoid", "A3_joint_PKA_medoid", "A4_continuous_PKA"
    ))
    f_k1 = primary_mind["ci_99"][0] <= 0
    f_k2 = primary_screen["point_delta"] >= 0 or primary_screen["ci_99"][1] >= 0
    result = {
        "schema_version": 1,
        "status": "PASS",
        "analysis_role": "NEW_PRIMARY_RESULT",
        "config": "configs/f1_families.yaml",
        "config_sha256": sha256_file(CONFIG_PATH),
        "action_spaces": config["action_spaces"],
        "effect_convention": "majority_minus_density",
        "benchmarks": {
            "mind2web": {"comparisons": mind},
            "screenspot_pro": {"comparisons": screen},
        },
        "mind2web_action_strata": strata,
        "gates": {
            "mind2web_sequential_ci_lower_gt_zero": primary_mind["ci_99"][0] > 0,
            "mind2web_sequential_delta_gt_mde": primary_mind["point_delta"] > config["gates"]["mind2web_mde"],
            "mind2web_A1_A4_all_point_positive": mind_a1_a4_positive,
            "screenspot_A2_opposite_ci_upper_lt_zero": primary_screen["ci_99"][1] < 0,
            "F_K1": f_k1,
            "F_K2": f_k2,
            "new_primary_line_pass": not f_k1 and not f_k2 and mind_a1_a4_positive,
        },
    }
    atomic_json(RUN_DIR / "f1_aggregator_matching.json", result)
    draw_figure(result, RUN_DIR / "fig_aggregator_matching.pdf")
    print(json.dumps({"gates": result["gates"], "mind2web_primary": primary_mind, "screenspot_primary": primary_screen, "action_strata": strata}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
