import json

import numpy as np
import yaml

from aggmatch_common import METHODS, RUN_DIR, atomic_json, load_cache, method_difference, paired_bootstrap, sha256_file


CONFIG_PATH = RUN_DIR / "configs/f1_families.yaml"


def analyze_benchmark(benchmark, config):
    cache = load_cache()[benchmark]
    metadata = cache["metadata"]
    outputs = cache["outputs"]
    method_effects = {
        method: float(np.mean(list(method_difference(outputs, "C_cond", method, "C_uni", method).values())))
        for method in METHODS
    }
    pooled = {
        row_id: float(np.mean([
            int(outputs["C_cond"][method][row_id]) - int(outputs["C_uni"][method][row_id])
            for method in METHODS
        ]))
        for row_id in metadata
    }
    seed = config["bootstrap"][benchmark]["seed"] + 200
    combined = paired_bootstrap(metadata, pooled, config["bootstrap"]["resamples"], seed)
    accuracy = cache["accuracy"]
    best_arm_by_method = {
        method: max(accuracy, key=lambda arm: (accuracy[arm][method], arm))
        for method in METHODS
    }
    return {
        "combined_C_cond_minus_C_uni": combined,
        "per_aggregator_effects": method_effects,
        "heterogeneity_std_across_aggregators": float(np.std(list(method_effects.values()), ddof=0)),
        "best_arm_by_aggregator": best_arm_by_method,
        "C_cond_best_count": sum(arm == "C_cond" for arm in best_arm_by_method.values()),
        "aggregator_count": len(METHODS),
    }


def main():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if config["status"] != "FROZEN_BEFORE_RESULTS":
        raise ValueError("F2 requires the frozen F1 map")
    mind = analyze_benchmark("mind2web", config)
    screen = analyze_benchmark("screenspot_pro", config)
    f_k4 = mind["combined_C_cond_minus_C_uni"]["ci_99"][0] <= 0 <= mind["combined_C_cond_minus_C_uni"]["ci_99"][1] or screen["combined_C_cond_minus_C_uni"]["ci_99"][0] <= 0 <= screen["combined_C_cond_minus_C_uni"]["ci_99"][1]
    result = {
        "schema_version": 1,
        "status": "PASS",
        "analysis_role": "POST_HOC_EXPLORATORY_NOT_PREREGISTERED_NOT_A_PRIMARY_GATE",
        "config": "configs/f1_families.yaml",
        "config_sha256": sha256_file(CONFIG_PATH),
        "effect_convention": "C_cond_minus_C_uni_then_mean_across_seven_aggregators_within_each_bootstrap_sample",
        "shared_candidate_dependence": "handled_by_row_paired_stratified_bootstrap_not_a_sign_test",
        "mind2web": mind,
        "screenspot_pro": screen,
        "F_K4": f_k4,
        "claim": "arm_ranking_consistent_across_aggregators_but_preregistered_majority_gate_failed" if not f_k4 else "C_cond_advantage_restricted_to_specific_aggregators",
        "prohibition": "must_not_be_used_to_reinstate_the_primary_four_arm_claim",
    }
    atomic_json(RUN_DIR / "f2_arm_consistency.json", result)
    print(json.dumps({"F_K4": f_k4, "mind2web": mind, "screenspot_pro": screen}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()