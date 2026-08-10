import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from lsa_common import load_rows


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CEV_PATH = ROOT / "runs/cev/2026-08-09/cev_main.json"
BENCHMARKS = ("mind2web", "screenspot_pro")


def paired_bootstrap(rows, left, right, resamples, seed):
    by_fold_group = {}
    for row_id in left:
        row = rows[row_id]
        by_fold_group.setdefault(row.fold, {}).setdefault(row.group, []).append(row_id)
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for sample_index in range(resamples):
        selected = []
        for fold in sorted(by_fold_group):
            groups = sorted(by_fold_group[fold])
            for group in rng.choice(groups, size=len(groups), replace=True):
                selected.extend(by_fold_group[fold][group])
        samples[sample_index] = np.mean([int(left[row_id]) - int(right[row_id]) for row_id in selected])
    point = float(np.mean([int(left[row_id]) - int(right[row_id]) for row_id in left]))
    return {
        "point_delta": point,
        "ci_99": [float(np.quantile(samples, 0.005)), float(np.quantile(samples, 0.995))],
        "resamples": resamples,
        "seed": seed,
        "rows": len(left),
        "wins": sum(left[row_id] and not right[row_id] for row_id in left),
        "losses": sum(right[row_id] and not left[row_id] for row_id in left),
    }, samples


def balanced_bootstrap(samples_by_benchmark, mde):
    standardized = [samples_by_benchmark[benchmark] / mde[benchmark] for benchmark in BENCHMARKS]
    values = np.mean(np.stack(standardized), axis=0)
    return {
        "point_standardized_mean": float(np.mean(values)),
        "ci_99": [float(np.quantile(values, 0.005)), float(np.quantile(values, 0.995))],
    }


def main():
    config = yaml.safe_load((RUN_DIR / "configs/lsa_prereg.yaml").read_text())
    banks = load_rows()
    main = json.loads((RUN_DIR / "lsa_main.json").read_text())["pooled"]
    variants = json.loads((RUN_DIR / "lsa_variants.json").read_text())["variants"]
    cev = json.loads(CEV_PATH.read_text())
    comparisons = {benchmark: {} for benchmark in BENCHMARKS}
    sample_cache = {}
    variant_sample_cache = defaultdict(dict)
    for index, benchmark in enumerate(BENCHMARKS):
        seed = config["benchmarks"][benchmark]["bootstrap_seed"]
        outputs = main["outputs"][benchmark]
        controls = {
            "CEV_A": cev["outputs"][benchmark]["C_uni"]["CEV_A"],
            "dev_selection": cev["outputs"][benchmark]["C_uni"]["dev_selection"],
        }
        for offset, (name, left, right) in enumerate((
            ("pooled_safe_minus_CEV_A", outputs["safe"], controls["CEV_A"]),
            ("pooled_safe_minus_dev_selection", outputs["safe"], controls["dev_selection"]),
            ("pooled_safe_minus_direct", outputs["safe"], outputs["direct"]),
            ("pooled_direct_minus_CEV_A", outputs["direct"], controls["CEV_A"]),
        )):
            result, samples = paired_bootstrap(banks[benchmark], left, right, config["statistics"]["resamples"], seed + offset)
            comparisons[benchmark][name] = result
            sample_cache[(benchmark, name)] = samples
        within_key = f"within_safe_{benchmark}"
        within_outputs = variants[within_key]["outputs"][benchmark]["safe"]
        result, _ = paired_bootstrap(banks[benchmark], within_outputs, controls["CEV_A"], config["statistics"]["resamples"], seed + 10)
        comparisons[benchmark]["within_safe_minus_CEV_A"] = result
        for variant_offset, variant in enumerate(("reliability_only", "no_geometry", "no_action", "no_parameter")):
            left = variants[variant]["outputs"][benchmark]["safe"]
            result, samples = paired_bootstrap(banks[benchmark], left, controls["CEV_A"], config["statistics"]["resamples"], seed + 20 + variant_offset)
            comparisons[benchmark][f"{variant}_minus_CEV_A"] = result
            variant_sample_cache[variant][benchmark] = samples
            result, samples = paired_bootstrap(banks[benchmark], left, controls["dev_selection"], config["statistics"]["resamples"], seed + 30 + variant_offset)
            comparisons[benchmark][f"{variant}_minus_dev_selection"] = result
            variant_sample_cache[f"{variant}_devsel"][benchmark] = samples

    mde = {benchmark: config["benchmarks"][benchmark]["mde"] for benchmark in BENCHMARKS}
    l1 = {
        benchmark: comparisons[benchmark]["pooled_safe_minus_CEV_A"]["ci_99"][1] >= 0
        or abs(comparisons[benchmark]["pooled_safe_minus_CEV_A"]["point_delta"]) < mde[benchmark]
        for benchmark in BENCHMARKS
    }
    l2_significant = {
        benchmark: comparisons[benchmark]["pooled_safe_minus_CEV_A"]["ci_99"][0] > 0
        for benchmark in BENCHMARKS
    }
    l2 = all(l1.values()) and any(l2_significant.values())
    l3_strong = all(comparisons[benchmark]["pooled_safe_minus_dev_selection"]["ci_99"][0] > 0 for benchmark in BENCHMARKS)
    balanced = balanced_bootstrap(
        {benchmark: sample_cache[(benchmark, "pooled_safe_minus_dev_selection")] for benchmark in BENCHMARKS},
        mde,
    )
    no_mde_loss = all(comparisons[benchmark]["pooled_safe_minus_dev_selection"]["point_delta"] >= -mde[benchmark] for benchmark in BENCHMARKS)
    l3_safe = balanced["ci_99"][0] > 0 and no_mde_loss
    l4 = all(main["accuracy"][benchmark]["safe"] >= main["accuracy"][benchmark]["direct"] for benchmark in BENCHMARKS)
    infinity_folds = sum(not np.isfinite(fold["selected"]["threshold"]) for fold in main["folds"])
    within_pass = {
        benchmark: comparisons[benchmark]["within_safe_minus_CEV_A"]["ci_99"][1] >= 0
        or abs(comparisons[benchmark]["within_safe_minus_CEV_A"]["point_delta"]) < mde[benchmark]
        for benchmark in BENCHMARKS
    }
    gates = {
        "L1": {"pass": all(l1.values()), **l1},
        "L2": {"pass": l2, "significant_by_benchmark": l2_significant},
        "L3_strong": l3_strong,
        "L3_safe": {"pass": l3_safe, "balanced_standardized_delta": balanced, "no_MDE_loss": no_mde_loss},
        "L4": l4,
        "LSA_K1": False,
        "LSA_K2": not all(l1.values()),
        "LSA_K3": infinity_folds >= 3,
        "LSA_K4": all(within_pass.values()) and not all(l1.values()),
        "LSA_K5": None,
        "infinity_threshold_folds": infinity_folds,
    }
    variant_balanced = {
        variant: {
            "versus_CEV_A": balanced_bootstrap(
                {benchmark: variant_sample_cache[variant][benchmark] for benchmark in BENCHMARKS}, mde
            ),
            "versus_dev_selection": balanced_bootstrap(
                {benchmark: variant_sample_cache[f"{variant}_devsel"][benchmark] for benchmark in BENCHMARKS}, mde
            ),
        }
        for variant in ("reliability_only", "no_geometry", "no_action", "no_parameter")
    }
    if l3_strong:
        position = "STRONG_METHOD_CONTRIBUTION"
    elif l3_safe:
        position = "SAFE_LEARNED_CONTRIBUTION"
    elif l2:
        position = "CEV_IMPROVEMENT_NOT_DEVSEL_SUPERIOR"
    elif all(l1.values()):
        position = "SAFE_BUT_NO_SIGNIFICANT_GAIN"
    else:
        position = "FAILED_UNSAFE_LEARNED_AGGREGATOR"
    result = {
        "schema_version": 1,
        "status": "PASS_ADJUDICATED",
        "accuracy": main["accuracy"],
        "comparisons": comparisons,
        "gates": gates,
        "variant_descriptive_balanced": variant_balanced,
        "paper_position": position,
        "fold_selections": [
            {
                "outer_fold": fold["outer_fold"],
                "model_id": fold["selected"]["model_id"],
                "threshold": fold["selected"]["threshold"],
                "test": fold["test"],
            }
            for fold in main["folds"]
        ],
    }
    (RUN_DIR / "lsa_adjudication.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"paper_position": position, "gates": gates, "accuracy": main["accuracy"], "comparisons": comparisons}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()