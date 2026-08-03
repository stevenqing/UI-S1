import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CALA_DIR = ROOT / "runs/cala/2026-08-03"
H1_RAW = ROOT / "runs/ccm-h2h/2026-07-31/h1/raw"
sys.path.insert(0, str(CALA_DIR))
sys.path.insert(0, str(ROOT / "runs/ccm-h2h/2026-07-31/h3"))
sys.path.insert(0, str(ROOT / "runs/allocation-law/2026-08-01"))
sys.path.insert(0, str(ROOT / "runs/diversity-axis/2026-08-02"))
from cala_common import BUDGETS, UNIFORM_SEQUENCE, V_ONLY_SEQUENCE, load_bank, split_ids
from cala_adaptive import SEED, development_statistics, route_row, training_matrix
from cala_transfer_72b import (
    GTA_N8, UNIFORM_N8, adaptive_route as route_72, development_stats as stats_72,
    load_context as load_72, split_ids as split_72, training_matrix as matrix_72,
)
from neff import estimate_pool, linear_fit, two_factor_fit


RHO_NAMES = ("failure_kappa", "rho_geom", "rho_cond")


def parse_action(name):
    model, view = name.rsplit("/view", 1)
    return model, int(view)


def rows_for_actions(context, actions_by_row):
    return [{
        "id": row_id,
        "application": context["metadata"][row_id]["application"],
        "target_bbox": context["metadata"][row_id]["target_bbox"],
        "candidates": [context["bank"][action][row_id] for action in actions_by_row[row_id]],
    } for row_id in context["row_ids"]]


def fixed_actions(context, sequence, budget):
    return {row_id: tuple(sequence[:budget]) for row_id in context["row_ids"]}


def fold_actions(context, sequences, method, budget):
    output = {}
    for fold in range(5):
        _, test_ids = split_ids(context, fold)
        actions = tuple(parse_action(name) for name in sequences[str(fold)]["sequences"][method][:budget])
        for row_id in test_ids:
            output[row_id] = actions
    if set(output) != set(context["row_ids"]):
        raise ValueError(f"N1 fold action coverage mismatch: {method}/N{budget}")
    return output


def adaptive_7b_actions(context):
    routes = {}
    for fold in range(5):
        dev_ids, test_ids = split_ids(context, fold)
        correct, accuracy, kappa = development_statistics(context, dev_ids)
        train_x, train_y = training_matrix(context, dev_ids, fold, correct, accuracy, kappa)
        classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, penalty="l2", class_weight="balanced", random_state=SEED, max_iter=500))
        classifier.fit(train_x, train_y)
        for row_id in test_ids:
            routes[row_id] = route_row(context, row_id, classifier, 16, accuracy, kappa)
    if set(routes) != set(context["row_ids"]):
        raise ValueError("N1 CALA-A route coverage mismatch")
    return routes


def h1_rows(count):
    rows = pq.read_table(H1_RAW / f"candidates_N{count}.parquet").to_pylist()
    rows.sort(key=lambda row: row["id"])
    if len(rows) != 1581 or any(len(row["candidates"]) != count for row in rows):
        raise ValueError(f"N1 H1 N{count} mismatch")
    return rows


def rows_72(context, actions_by_row):
    return [{
        "id": row_id, "application": context["metadata"][row_id]["application"],
        "target_bbox": context["metadata"][row_id]["target_bbox"],
        "candidates": [context["bank"][action][row_id] for action in actions_by_row[row_id]],
    } for row_id in context["row_ids"]]


def reconstruct_72_actions(context, transfer):
    static = {}
    adaptive = {}
    for fold in range(5):
        dev_ids, test_ids = split_72(context, fold)
        names = transfer["folds"][str(fold)]["CALA_S"]
        static_actions = tuple((name.rsplit("/region", 1)[0], int(name.rsplit("/region", 1)[1])) for name in names)
        correct, accuracy, kappa = stats_72(context, dev_ids)
        train_x, train_y = matrix_72(context, dev_ids, fold, correct, accuracy, kappa)
        classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, penalty="l2", class_weight="balanced", random_state=SEED, max_iter=500))
        classifier.fit(train_x, train_y)
        for row_id in test_ids:
            static[row_id] = static_actions
            adaptive[row_id] = route_72(context, row_id, classifier, accuracy, kappa)
    return static, adaptive


def add_pool(records, pool_id, scale, method, budget, rows, accuracies, rules):
    diagnostics = estimate_pool(rows)
    record = {"pool_id": pool_id, "scale": scale, "method": method, **diagnostics}
    records[pool_id] = record
    return [{
        "point_id": f"{pool_id}/{rule}", "pool_id": pool_id, "scale": scale,
        "method": method, "rule": rule, "K": budget, "accuracy": accuracies[rule],
        "quality": diagnostics["mean_proposal_full_bbox_containment"],
        **{f"N_eff_{name}": diagnostics["N_eff"][name] for name in RHO_NAMES},
    } for rule in rules]


def build_points():
    static = json.loads((CALA_DIR / "cala_static_results.json").read_text())
    adaptive = json.loads((CALA_DIR / "cala_adaptive_results.json").read_text())
    transfer = json.loads((CALA_DIR / "cala_transfer_72b_results.json").read_text())
    h1 = json.loads((ROOT / "runs/ccm-h2h/2026-07-31/h1_headtohead.json").read_text())
    n3 = json.loads((RUN_DIR / "n3_72b_repair.json").read_text())
    if n3["status"] != "PASS_NO_GLOBAL_COORDINATE_BUG":
        raise ValueError("N1 72B lane remains quarantined")
    context = load_bank()
    adaptive_routes = adaptive_7b_actions(context)
    pools = {}
    primary = []
    stress = []
    for method in ("V_only", "Uniform_Mixed", "Quality_Only", "CALA_S"):
        for budget in BUDGETS:
            if method == "V_only":
                actions = fixed_actions(context, V_ONLY_SEQUENCE, budget)
            elif method == "Uniform_Mixed":
                actions = fixed_actions(context, UNIFORM_SEQUENCE, budget)
            else:
                actions = fold_actions(context, static["fold_sequences"], method, budget)
            points = add_pool(pools, f"7B/{method}/N{budget}", "7B", method, budget, rows_for_actions(context, actions), static["accuracy"][method][str(budget)], ("B3_mvp", "M1_ccm", "pass_at_n"))
            primary.extend(point for point in points if point["rule"] == "B3_mvp")
            stress.extend(points)
    for budget in (8, 12, 16):
        actions = {row_id: route[:budget] for row_id, route in adaptive_routes.items()}
        points = add_pool(pools, f"7B/CALA_A/N{budget}", "7B", "CALA_A", budget, rows_for_actions(context, actions), adaptive["accuracy"][str(budget)], ("B3_mvp", "M1_ccm", "pass_at_n"))
        primary.extend(point for point in points if point["rule"] == "B3_mvp")
        stress.extend(points)
    for count in (2, 4, 10):
        diagnostics_rows = h1_rows(count)
        accuracies = h1["accuracy"][str(count)]
        selected = {"B3_mvp_official": accuracies["B3_mvp_official"], "B3_paper_centroid": accuracies["B3_paper_centroid"], "B3_graph_centroid": accuracies["B3_graph_centroid"]}
        points = add_pool(pools, f"7B/H1/N{count}", "7B", "H1", count, diagnostics_rows, selected, tuple(selected))
        primary.extend(points); stress.extend(points)
    context72 = load_72()
    static72, adaptive72 = reconstruct_72_actions(context72, transfer)
    actions72 = {
        "GTA1_N8": {row_id: GTA_N8 for row_id in context72["row_ids"]},
        "Uniform_Mixed_N8": {row_id: UNIFORM_N8 for row_id in context72["row_ids"]},
        "CALA_S_N8": static72,
        "CALA_A_N8": adaptive72,
    }
    for method, actions in actions72.items():
        points = add_pool(pools, f"72B/{method}", "72B", method, 8, rows_72(context72, actions), transfer["accuracy"][method], ("B3_mvp", "M1_ccm", "pass_at_n"))
        primary.extend(point for point in points if point["rule"] == "B3_mvp")
        stress.extend(points)
    return pools, primary, stress


def fit_panel(points):
    fits = {}
    budget_fit = linear_fit(points, "K")
    for rho_name in RHO_NAMES:
        key = f"N_eff_{rho_name}"
        neff_fit = linear_fit(points, key)
        fits[rho_name] = {
            "N_eff": neff_fit,
            "K": budget_fit,
            "collapse_success": neff_fit["residual_sd"] <= 0.014 and neff_fit["residual_sd"] < budget_fit["residual_sd"],
            "two_factor": two_factor_fit(points, key),
        }
    return fits


def make_figure(points, fits, output):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"7B": "#176B87", "72B": "#C84B31"}
    for scale in ("7B", "72B"):
        selected = [point for point in points if point["scale"] == scale]
        axes[0].scatter([point["K"] for point in selected], [100 * point["accuracy"] for point in selected], label=scale, color=colors[scale], alpha=.8)
        axes[1].scatter([point["N_eff_failure_kappa"] for point in selected], [100 * point["accuracy"] for point in selected], label=scale, color=colors[scale], alpha=.8)
    axes[0].set_xlabel("Raw forwards K"); axes[0].set_ylabel("B3 accuracy (%)"); axes[0].set_title("Raw budget")
    axes[1].set_xlabel("N_eff from failure kappa"); axes[1].set_ylabel("B3 accuracy (%)"); axes[1].set_title("Effective sample size")
    for axis in axes: axis.grid(alpha=.2); axis.legend()
    figure.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True); figure.savefig(output); plt.close(figure)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--figure", type=Path, required=True); args = parser.parse_args()
    pools, primary, stress = build_points()
    primary_fits = fit_panel(primary); stress_fits = fit_panel(stress)
    make_figure(primary, primary_fits, args.figure)
    result = {
        "schema_version": 1, "status": "PASS", "pools": pools,
        "primary_panel": {"points": primary, "fits": primary_fits, "collapse_any_estimator": any(value["collapse_success"] for value in primary_fits.values())},
        "stress_panel": {"points": stress, "fits": stress_fits, "collapse_any_estimator": any(value["collapse_success"] for value in stress_fits.values())},
        "criteria": {"maximum_residual_sd": 0.014, "require_better_than_K": True},
        "figure": str(args.figure.resolve().relative_to(ROOT)),
        "sources": {"rho_config_sha256": hashlib.sha256((RUN_DIR/"configs/rho_estimators.yaml").read_bytes()).hexdigest(), "criteria_sha256": hashlib.sha256((RUN_DIR/"configs/n1_criteria.yaml").read_bytes()).hexdigest()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"primary_observations": len(primary), "stress_observations": len(stress), "primary_fits": primary_fits, "collapse": result["primary_panel"]["collapse_any_estimator"]}, indent=2, sort_keys=True))


if __name__ == "__main__": main()