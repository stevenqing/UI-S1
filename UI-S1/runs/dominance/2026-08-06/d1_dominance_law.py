import hashlib
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
SOURCEBIAS_DIR = ROOT / "runs/sourcebias/2026-08-03"
FINAL_DIR = ROOT / "runs/final/2026-08-04"
COMPLEMENTARITY_DIR = ROOT / "runs/complementarity/2026-07-30"
sys.path.insert(0, str(SOURCEBIAS_DIR))

from sourcebias_common import b3_select_index, load_pools, point_in_bbox, split_ids

H3_DIR = ROOT / "runs/ccm-h2h/2026-07-31/h3"
sys.path.insert(0, str(H3_DIR))
from h3_eval import ccm_select, fit_ccm


MODELS = ("GTA1-7B", "Qwen3-VL-8B-Instruct", "UI-TARS-7B-SFT")
VIEWS = tuple(range(12))
SEED = 20260806
RESAMPLES = 10000


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cohen_kappa(left, right):
    left = np.asarray(left, dtype=np.bool_)
    right = np.asarray(right, dtype=np.bool_)
    observed = float(np.mean(left == right))
    left_rate = float(np.mean(left))
    right_rate = float(np.mean(right))
    expected = left_rate * right_rate + (1 - left_rate) * (1 - right_rate)
    return 1.0 if math.isclose(expected, 1.0) else (observed - expected) / (1 - expected)


def pool_specs():
    specs = []
    for size in (2, 3):
        for models in itertools.combinations(MODELS, size):
            for views in itertools.product(VIEWS, repeat=size):
                actions = tuple(zip(models, views))
                specs.append({
                    "pool_id": "+".join(f"{model}/view{view}" for model, view in actions),
                    "pool_size": size,
                    "actions": actions,
                })
    if len(specs) != 2160:
        raise ValueError(f"D1 expected 2,160 ScreenSpot pools, found {len(specs)}")
    return specs


def row_for(context, row_id, actions, fold):
    metadata = context["metadata"][row_id]
    return {
        "id": row_id,
        "application": metadata["application"],
        "target_bbox": metadata["target_bbox"],
        "outer_fold": fold,
        "candidates": [context["bank"][action][row_id] for action in actions],
    }


def evaluate_pool(context, actions, folds):
    b3_correct = 0
    m1_correct = 0
    for fold in range(5):
        dev_ids, test_ids = folds[fold]
        dev_rows = [row_for(context, row_id, actions, fold) for row_id in dev_ids]
        tables, priors = fit_ccm(dev_rows)
        for row_id in test_ids:
            row = row_for(context, row_id, actions, fold)
            b3_index, _ = b3_select_index(row["candidates"])
            m1_index = ccm_select(row, tables, priors)
            b3_correct += int(point_in_bbox(row["candidates"][b3_index]["point"], row["target_bbox"]))
            m1_correct += int(point_in_bbox(row["candidates"][m1_index]["point"], row["target_bbox"]))
    rows = len(context["row_ids"])
    return {"B3_mvp": b3_correct / rows, "M1_ccm": m1_correct / rows}


def residualize(values, controls):
    values = rankdata(np.asarray(values, dtype=np.float64))
    controls = np.asarray([rankdata(np.asarray(control, dtype=np.float64)) for control in controls]).T
    design = np.column_stack([np.ones(len(values)), controls])
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    return values - design @ coefficients


def correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def bootstrap_correlations(records, delta_key):
    gap = np.asarray([row["dominance_gap"] for row in records], dtype=np.float64)
    delta = np.asarray([row[delta_key] for row in records], dtype=np.float64)
    mean_quality = np.asarray([row["mean_member_accuracy"] for row in records], dtype=np.float64)
    failure_kappa = np.asarray([row["mean_pairwise_failure_kappa"] for row in records], dtype=np.float64)
    gap_rank = rankdata(gap)
    delta_rank = rankdata(delta)
    gap_residual = residualize(gap, (mean_quality, failure_kappa))
    delta_residual = residualize(delta, (mean_quality, failure_kappa))
    raw_point = float(spearmanr(gap, delta).statistic)
    partial_point = correlation(gap_residual, delta_residual)

    strata = {
        size: np.asarray([index for index, row in enumerate(records) if row["pool_size"] == size], dtype=np.int64)
        for size in (2, 3)
    }
    rng = np.random.default_rng(SEED)
    raw_samples = []
    partial_samples = []
    for _ in range(RESAMPLES):
        selected = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in strata.values()
        ])
        raw_samples.append(correlation(gap_rank[selected], delta_rank[selected]))
        partial_samples.append(correlation(gap_residual[selected], delta_residual[selected]))
    raw_samples = np.asarray(raw_samples, dtype=np.float64)
    partial_samples = np.asarray(partial_samples, dtype=np.float64)
    return {
        "raw_spearman": {
            "rho": raw_point,
            "ci_99": [float(np.quantile(raw_samples, 0.005)), float(np.quantile(raw_samples, 0.995))],
        },
        "partial_spearman_controlling_mean_quality_and_failure_kappa": {
            "rho": partial_point,
            "ci_99": [float(np.quantile(partial_samples, 0.005)), float(np.quantile(partial_samples, 0.995))],
            "method": "Pearson correlation of full-sample rank residuals; pool bootstrap stratified by pool size",
        },
        "resamples": RESAMPLES,
        "seed": SEED,
    }


def manifest_pool_anchors(manifest, config):
    lane_sections = {
        "mind2web": manifest["mind2web"]["lanes"],
        "androidcontrol": manifest["androidcontrol"]["lanes"],
    }
    output = {}
    for section, bench, setting in (
        ("T1_mind2web", "mind2web", "visual"),
        ("T2_androidcontrol", "androidcontrol", "low"),
    ):
        output[section] = {}
        for pool_name, pool in config[section]["pools"].items():
            accuracies = {
                model: lane_sections[bench][f"{setting}/{model}"]["successes"]
                / lane_sections[bench][f"{setting}/{model}"]["rows"]
                for model in pool["models"]
            }
            ordered = sorted(accuracies.values(), reverse=True)
            output[section][pool_name] = {
                "models": pool["models"],
                "member_accuracy_unfiltered_manifest": accuracies,
                "mean_member_accuracy": float(np.mean(ordered)),
                "dominance_gap": ordered[0] - ordered[1],
                "mixed_metrics": None,
                "status": "BLOCKED_MISSING_ROW_LEVEL_TRACES",
            }
    return output


def figure(records, statistics, output):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    colors = {2: "#287271", 3: "#D97706"}
    for axis, metric, title in (
        (axes[0], "B3_minus_best", "B3 minus best action"),
        (axes[1], "M1_minus_best", "M1 minus best action"),
    ):
        for size in (2, 3):
            rows = [row for row in records if row["pool_size"] == size]
            axis.scatter(
                [100 * row["dominance_gap"] for row in rows],
                [100 * row[metric] for row in rows],
                s=8, alpha=0.28, color=colors[size], label=f"{size} lineages",
            )
        rho = statistics[metric]["raw_spearman"]["rho"]
        partial = statistics[metric]["partial_spearman_controlling_mean_quality_and_failure_kappa"]["rho"]
        axis.axhline(0, color="#444444", linewidth=0.8)
        axis.set_title(f"{title}\nSpearman={rho:+.3f}, partial={partial:+.3f}")
        axis.set_xlabel("Dominance gap (pp)")
        axis.set_ylabel("Aggregation delta (pp)")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    figure.suptitle("ScreenSpot-Pro action pools; cross-benchmark row traces unavailable")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def main():
    d0_path = RUN_DIR / "d0_r7_audit.json"
    d0 = json.loads(d0_path.read_text())
    if d0["status"] != "PASS" or d0["D_K1_triggered"]:
        raise ValueError("D1 blocked by D0")

    contexts, _ = load_pools()
    context = contexts["7B"]
    expected_actions = {(model, view) for model in MODELS for view in VIEWS}
    if not expected_actions.issubset(context["bank"]):
        raise ValueError("D1 missing frozen ScreenSpot 3x12 action bank")
    folds = {fold: split_ids(context, fold) for fold in range(5)}
    row_ids = list(context["row_ids"])
    correctness = {
        action: np.asarray([
            point_in_bbox(context["bank"][action][row_id]["point"], context["metadata"][row_id]["target_bbox"])
            for row_id in row_ids
        ], dtype=np.bool_)
        for action in sorted(expected_actions)
    }
    action_accuracy = {action: float(values.mean()) for action, values in correctness.items()}

    records = []
    for index, spec in enumerate(pool_specs(), start=1):
        actions = spec["actions"]
        member_accuracy = {f"{model}/view{view}": action_accuracy[(model, view)] for model, view in actions}
        ordered = sorted(member_accuracy.values(), reverse=True)
        kappas = [
            cohen_kappa(~correctness[left], ~correctness[right])
            for action_index, left in enumerate(actions)
            for right in actions[action_index + 1:]
        ]
        aggregate = evaluate_pool(context, actions, folds)
        record = {
            **spec,
            "actions": [list(action) for action in actions],
            "member_accuracy": member_accuracy,
            "best_single": ordered[0],
            "second_best": ordered[1],
            "dominance_gap": ordered[0] - ordered[1],
            "mean_member_accuracy": float(np.mean(ordered)),
            "mean_pairwise_failure_kappa": float(np.mean(kappas)),
            "B3_mvp": aggregate["B3_mvp"],
            "M1_ccm": aggregate["M1_ccm"],
            "B3_minus_best": aggregate["B3_mvp"] - ordered[0],
            "M1_minus_best": aggregate["M1_ccm"] - ordered[0],
        }
        records.append(record)
        if index % 100 == 0:
            print(json.dumps({"screen_pools": index, "total": 2160}), flush=True)

    statistics = {
        metric: bootstrap_correlations(records, metric)
        for metric in ("B3_minus_best", "M1_minus_best")
    }
    screen_gate = {
        metric: (
            values["raw_spearman"]["rho"] < -0.6
            and values["raw_spearman"]["ci_99"][1] < 0
            and values["partial_spearman_controlling_mean_quality_and_failure_kappa"]["ci_99"][1] < 0
        )
        for metric, values in statistics.items()
    }

    manifest_path = COMPLEMENTARITY_DIR / "rows_manifest.json"
    config_path = FINAL_DIR / "configs/t1_t2_pools.yaml"
    rows_path = COMPLEMENTARITY_DIR / "rows.parquet"
    import yaml
    manifest = json.loads(manifest_path.read_text())
    config = yaml.safe_load(config_path.read_text())
    cross_benchmark_status = "READY" if rows_path.is_file() else "BLOCKED_MISSING_ROWS_PARQUET_AND_LANE_TRACES"
    anchors = manifest_pool_anchors(manifest, config)
    required_benchmarks = ("ScreenSpot-Pro", "Mind2Web", "AndroidControl")
    available_benchmarks = ["ScreenSpot-Pro"]
    law_pass = False
    result = {
        "schema_version": 1,
        "status": "INCONCLUSIVE_BLOCKED_CROSS_BENCHMARK_ROWS",
        "hypothesis": "aggregation benefit decreases as within-pool dominance gap increases",
        "primary_metric": "B3_minus_best",
        "pool_definition": "one frozen action per retained lineage; all 3x12^2 pairs and 12^3 triples",
        "screen_spot": {
            "rows": len(row_ids),
            "pool_count": len(records),
            "pool_counts_by_size": {
                str(size): sum(row["pool_size"] == size for row in records) for size in (2, 3)
            },
            "statistics": statistics,
            "internal_gate": screen_gate,
            "pools": records,
        },
        "cross_benchmark": {
            "status": cross_benchmark_status,
            "required_benchmarks": list(required_benchmarks),
            "available_benchmarks": available_benchmarks,
            "missing_artifacts": [
                "runs/complementarity/2026-07-30/rows.parquet",
                "runs/androidcontrol-rft/2026-07-29/artifacts/*/*/predictions.jsonl",
                "runs/mind2web-*/2026-07-28/artifacts/*/predictions.jsonl",
            ],
            "manifest_member_quality_anchors": anchors,
            "note": "Member dominance gaps are recoverable from the manifest, but mixed-pool accuracy and failure kappa require row-level traces.",
        },
        "combined_gate": {
            "criterion": "combined rho < -0.6, 99% CI upper < 0, partial remains negative, and each benchmark direction agrees",
            "pass": law_pass,
            "adjudication": "NOT_ADJUDICATED_MISSING_TWO_BENCHMARKS",
            "D_K2": "BLOCKED_NOT_ADJUDICATED",
            "paper_action": "DO_NOT_CLAIM_DOMINANCE_LAW; retain scale split as an unexplained limitation",
        },
        "nonexchangeable_motivation_warning": {
            "7B_value": "63.69% B3",
            "72B_value": "70.52% nested LN",
            "warning": "These use different aggregators and are not inputs to the correlation analysis.",
        },
        "sources": {
            "D0": {"path": str(d0_path.relative_to(ROOT)), "sha256": sha256_file(d0_path)},
            "rows_manifest": {"path": str(manifest_path.relative_to(ROOT)), "sha256": sha256_file(manifest_path)},
            "pool_config": {"path": str(config_path.relative_to(ROOT)), "sha256": sha256_file(config_path)},
        },
    }
    output = RUN_DIR / "d1_dominance_law.json"
    figure_path = RUN_DIR / "fig_dominance.pdf"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    figure(records, statistics, figure_path)
    print(json.dumps({
        "status": result["status"],
        "screen_gate": screen_gate,
        "combined_pass": law_pass,
        "output": str(output.relative_to(ROOT)),
        "figure": str(figure_path.relative_to(ROOT)),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()