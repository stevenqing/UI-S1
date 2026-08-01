import argparse
import json
from pathlib import Path

import numpy as np

from allocation_eval import (
    EXPECTED_ROWS,
    build_pool,
    compact_evaluation,
    failure_statistics,
    group_folds,
    l2_units,
    load_gta1,
    load_manifest,
    load_model_views,
    matched_marginal_permutation,
    point_in_bbox,
    rowwise_spearman,
    spearman,
)


SEED = 20260801
METHODS = ("pass_at_n", "B3_mvp", "M1_ccm")


def group_sufficient_statistics(rows, outputs, groups):
    group_index = {group: index for index, group in enumerate(groups)}
    row_counts = np.zeros(len(groups), dtype=np.int64)
    failures = np.zeros((len(groups), 12), dtype=np.int64)
    overlap = np.zeros((len(groups), 12, 12), dtype=np.int64)
    successes = {method: np.zeros(len(groups), dtype=np.int64) for method in METHODS}
    for row in rows:
        index = group_index[row["application"]]
        vector = np.asarray([
            not point_in_bbox(candidate["point"], row["target_bbox"])
            for candidate in row["candidates"]
        ], dtype=np.int64)
        row_counts[index] += 1
        failures[index] += vector
        overlap[index] += np.outer(vector, vector)
        for method in METHODS:
            successes[method][index] += int(outputs[method][row["id"]])
    return {"rows": row_counts, "failures": failures, "overlap": overlap, "successes": successes}


def weighted_kappa(sample_counts, statistics, group_mask):
    weights = sample_counts * group_mask
    total = weights @ statistics["rows"]
    failure = weights @ statistics["failures"]
    overlap = np.einsum("rg,gij->rij", weights, statistics["overlap"])
    values = []
    for left in range(12):
        for right in range(left + 1, 12):
            left_rate = np.divide(failure[:, left], total, out=np.full(total.shape, np.nan), where=total > 0)
            right_rate = np.divide(failure[:, right], total, out=np.full(total.shape, np.nan), where=total > 0)
            observed = np.divide(
                overlap[:, left, right] + total - failure[:, left] - failure[:, right] + overlap[:, left, right],
                total, out=np.full(total.shape, np.nan), where=total > 0,
            )
            expected = left_rate * right_rate + (1 - left_rate) * (1 - right_rate)
            values.append(np.divide(
                observed - expected, 1 - expected,
                out=np.full(total.shape, np.nan), where=~np.isclose(expected, 1.0),
            ))
    return np.nanmean(np.asarray(values), axis=0), total


def bootstrap_correlations(pool_statistics, groups, fold_for_group, resamples):
    rng = np.random.default_rng(SEED)
    sample_counts = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=resamples)
    kappa_columns = []
    outcome_columns = {method: [] for method in METHODS}
    for statistics in pool_statistics.values():
        for fold in range(5):
            dev_mask = np.asarray([fold_for_group[group] != fold for group in groups], dtype=np.int64)
            test_mask = 1 - dev_mask
            kappa, _ = weighted_kappa(sample_counts, statistics, dev_mask)
            kappa_columns.append(kappa)
            test_weights = sample_counts * test_mask
            test_rows = test_weights @ statistics["rows"]
            for method in METHODS:
                success = test_weights @ statistics["successes"][method]
                accuracy = np.divide(success, test_rows, out=np.full(test_rows.shape, np.nan), where=test_rows > 0)
                outcome_columns[method].append(accuracy)
    kappas = np.column_stack(kappa_columns)
    output = {}
    for method, columns in outcome_columns.items():
        outcomes = np.column_stack(columns)
        correlations = rowwise_spearman(kappas, outcomes)
        finite = correlations[np.isfinite(correlations)]
        if len(finite) < 0.99 * resamples:
            raise ValueError(f"L2 bootstrap has fewer than 99% finite {method} replicates")
        output[method] = {
            "resamples": resamples,
            "seed": SEED,
            "finite_replicates": len(finite),
            "rho_mean": float(np.mean(finite)),
            "rho_ci_99": [float(np.quantile(finite, 0.005)), float(np.quantile(finite, 0.995))],
            "p_rho_nonnegative": float(np.mean(finite >= 0)),
        }
    return output


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
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=1000)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    gta1 = load_gta1(args.gta1_shards, manifest)
    generated = {
        "Qwen3-VL-8B-Instruct": load_model_views(
            args.qwen3_old, args.qwen3_extended, manifest, "Qwen3-VL-8B-Instruct"
        ),
        "UI-TARS-7B-SFT": load_model_views(
            args.uitars_old, args.uitars_extended, manifest, "UI-TARS-7B-SFT"
        ),
    }
    pools = l2_units(args.config)
    base_rows = build_pool(gta1, generated, next(iter(pools.values())))
    fold_for_group, fold_rows = group_folds(base_rows)
    groups = sorted(fold_for_group)
    observations = []
    pool_reports = {}
    sufficient = {}
    rng = np.random.default_rng(SEED)
    for pool_name, units in pools.items():
        rows = build_pool(gta1, generated, units)
        evaluation = compact_evaluation(rows)
        folds = []
        for fold in range(5):
            statistics = failure_statistics(rows, lambda row, fold=fold: fold_for_group[row["application"]] != fold)
            permutation = matched_marginal_permutation(statistics, rng, args.permutations)
            accuracy = evaluation["folds"][fold]["accuracy"]
            record = {
                "pool": pool_name,
                "fold": fold,
                "dev_rows": statistics["rows"],
                "dev_mean_pairwise_failure_kappa": statistics["mean_pairwise_kappa"],
                "finite_pairs": statistics["finite_pairs"],
                "null_pairs": statistics["null_pairs"],
                "matched_marginal": permutation,
                "heldout_accuracy": {method: accuracy[method] for method in METHODS},
            }
            folds.append(record)
            observations.append(record)
        pool_reports[pool_name] = {
            "units": [f"{model}/view{view}" for model, view in units],
            "accuracy": evaluation["accuracy"],
            "folds": folds,
            "mean_dev_kappa": float(np.mean([record["dev_mean_pairwise_failure_kappa"] for record in folds])),
        }
        sufficient[pool_name] = group_sufficient_statistics(rows, evaluation["outputs"], groups)

    correlations = {}
    x_values = [record["dev_mean_pairwise_failure_kappa"] for record in observations]
    for method in METHODS:
        correlations[method] = spearman(x_values, [record["heldout_accuracy"][method] for record in observations])
    bootstrap = bootstrap_correlations(sufficient, groups, fold_for_group, args.bootstrap_resamples)
    primary = correlations["pass_at_n"]["rho"]
    result = {
        "schema_version": 1,
        "status": "PASS",
        "rows": EXPECTED_ROWS,
        "budget": 12,
        "pool_count": len(pools),
        "fold_rows": fold_rows,
        "observation_count": len(observations),
        "pools": pool_reports,
        "correlations": correlations,
        "bootstrap": bootstrap,
        "prediction": {
            "primary": "pass_at_n",
            "rho": primary,
            "minimum_absolute_rho": 0.7,
            "satisfied": primary < 0 and abs(primary) > 0.7,
        },
        "kill_conditions": {"L-K3": primary >= 0 or abs(primary) < 0.7},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"], "correlations": correlations,
        "bootstrap": bootstrap, "prediction": result["prediction"],
        "kill_conditions": result["kill_conditions"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
