import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
H1_DIR = ROOT / "runs/ccm-h2h/2026-07-31/h1"
H3_DIR = ROOT / "runs/ccm-h2h/2026-07-31/h3"
CLOSING_DIR = ROOT / "runs/closing/2026-08-02"
sys.path.insert(0, str(H1_DIR))
sys.path.insert(0, str(H3_DIR))
sys.path.insert(0, str(CLOSING_DIR))
from aggregators_coord import mvp_official
from h3_eval import ccm_select, fit_ccm, point_in_bbox
from f1_paired_bootstrap import paired_bootstrap
from cala_common import (
    MODEL_ORDER,
    SHARED_ACTIONS,
    UNIFORM_SEQUENCE,
    action_name,
    build_rows,
    cohen_kappa,
    correctness,
    load_bank,
    sha256_file,
    split_ids,
)


SEED = 20260803
BUDGETS = (8, 12, 16)
SCOUT = (
    ("GTA1-7B", 0), ("Qwen3-VL-8B-Instruct", 0), ("UI-TARS-7B-SFT", 0),
    ("GTA1-7B", 1), ("Qwen3-VL-8B-Instruct", 1), ("UI-TARS-7B-SFT", 1),
)
MODEL_INDEX = {model: index for index, model in enumerate(MODEL_ORDER)}
ACTION_INDEX = {action: index for index, action in enumerate(SHARED_ACTIONS)}


def deterministic_order(row_id, fold, trajectory):
    digest = hashlib.sha256(f"{SEED}|{row_id}|{fold}|{trajectory}".encode()).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = np.random.default_rng(seed)
    remaining = [action for action in SHARED_ACTIONS if action not in SCOUT]
    return tuple(remaining[index] for index in rng.permutation(len(remaining)))


def selected_geometry(context, row_id, selected):
    width, height = context["metadata"][row_id]["img_size"]
    points = np.asarray([
        [context["bank"][action][row_id]["point"][0] / width, context["bank"][action][row_id]["point"][1] / height]
        for action in selected
    ], dtype=np.float64)
    centroid = points.mean(axis=0)
    spread = points.std(axis=0)
    distances = np.asarray([
        math.dist(points[left], points[right])
        for left in range(len(points))
        for right in range(left + 1, len(points))
    ], dtype=np.float64)
    if len(distances):
        mean_distance = float(distances.mean())
        max_distance = float(distances.max())
    else:
        mean_distance = max_distance = 0.0
    diagonal = math.hypot(width, height)
    threshold = 14.0 / diagonal
    largest_cluster_fraction = max(
        sum(math.dist(point, other) <= threshold for other in points) / len(points)
        for point in points
    )
    counts = [sum(action[0] == model for action in selected) / 16 for model in MODEL_ORDER]
    return [
        float(centroid[0]), float(centroid[1]), float(spread[0]), float(spread[1]),
        mean_distance, max_distance, largest_cluster_fraction, *counts,
    ]


def action_metadata(context, row_id, action):
    model, view = action
    width, height = context["metadata"][row_id]["img_size"]
    shared = context["bank"][("GTA1-7B", view)][row_id]
    left, top, right, bottom = shared["region"]
    return [
        *[float(model == value) for value in MODEL_ORDER],
        view / 11,
        view / 11,
        ((right - left) * (bottom - top)) / (width * height),
        (left + right) / (2 * width),
        (top + bottom) / (2 * height),
        float(shared.get("coverage", 0)) / 100,
    ]


def row_metadata(context, row_id):
    width, height = context["metadata"][row_id]["img_size"]
    instruction = context["metadata"][row_id]["instruction"]
    return [width / height, math.log(width * height), len(instruction) / 200]


def development_statistics(context, dev_ids):
    correct = {action: correctness(context, action, dev_ids) for action in SHARED_ACTIONS}
    accuracy = {action: float(np.mean(values)) for action, values in correct.items()}
    kappa = {}
    for left in SHARED_ACTIONS:
        for right in SHARED_ACTIONS:
            if left == right:
                kappa[left, right] = 1.0
            elif (right, left) in kappa:
                kappa[left, right] = kappa[right, left]
            else:
                kappa[left, right] = cohen_kappa(~correct[left], ~correct[right])
    return correct, accuracy, kappa


def feature(context, row_id, action, selected, dev_accuracy, dev_kappa, state_values=None, row_values=None):
    mean_kappa = float(np.mean([dev_kappa[action, value] for value in selected])) if selected else 0.0
    if state_values is None:
        state_values = selected_geometry(context, row_id, selected)
    if row_values is None:
        row_values = row_metadata(context, row_id)
    return [
        *action_metadata(context, row_id, action),
        dev_accuracy[action], mean_kappa,
        *state_values,
        *row_values,
    ]


def training_matrix(context, dev_ids, fold, correct, dev_accuracy, dev_kappa):
    features = []
    labels = []
    row_index = {row_id: index for index, row_id in enumerate(dev_ids)}
    for row_id in dev_ids:
        index = row_index[row_id]
        row_values = row_metadata(context, row_id)
        for trajectory in range(4):
            order = deterministic_order(row_id, fold, trajectory)
            selected = list(SCOUT)
            for history_length in range(10):
                selected_correct = any(correct[action][index] for action in selected)
                state_values = selected_geometry(context, row_id, selected)
                for action in SHARED_ACTIONS:
                    if action in selected:
                        continue
                    features.append(feature(
                        context, row_id, action, selected, dev_accuracy, dev_kappa,
                        state_values=state_values, row_values=row_values,
                    ))
                    labels.append(bool(correct[action][index] and not selected_correct))
                if history_length < 9:
                    selected.append(order[history_length])
    values = np.asarray(features, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.int8)
    if values.ndim != 2 or values.shape[1] != 24 or targets.sum() == 0 or targets.sum() == len(targets):
        raise ValueError(f"CALA-A training matrix invalid: {values.shape}, positives={targets.sum()}")
    return values, targets


def route_row(context, row_id, model, budget, dev_accuracy, dev_kappa):
    selected = list(SCOUT)
    row_values = row_metadata(context, row_id)
    while len(selected) < budget:
        actions = [action for action in SHARED_ACTIONS if action not in selected]
        state_values = selected_geometry(context, row_id, selected)
        features = np.asarray([
            feature(
                context, row_id, action, selected, dev_accuracy, dev_kappa,
                state_values=state_values, row_values=row_values,
            )
            for action in actions
        ], dtype=np.float32)
        probabilities = model.predict_proba(features)[:, 1]
        winner = max(
            range(len(actions)),
            key=lambda index: (probabilities[index], dev_accuracy[actions[index]], -ACTION_INDEX[actions[index]]),
        )
        selected.append(actions[winner])
    return tuple(selected)


def evaluate_variable_fold(context, dev_ids, test_ids, actions_by_row):
    dev_actions = tuple(UNIFORM_SEQUENCE[:max(len(value) for value in actions_by_row.values())])
    dev_rows = build_rows(context, dev_ids, dev_actions)
    test_rows = [
        {
            "id": row_id,
            "application": context["metadata"][row_id]["application"],
            "target_bbox": context["metadata"][row_id]["target_bbox"],
            "candidates": [context["bank"][action][row_id] for action in actions_by_row[row_id]],
        }
        for row_id in test_ids
    ]
    tables, priors = fit_ccm(dev_rows)
    outputs = {"B3_mvp": {}, "M1_ccm": {}, "pass_at_n": {}}
    for row in test_rows:
        candidates = row["candidates"]
        points = [candidate["point"] for candidate in candidates]
        pseudo = [{"coverage": candidate.get("coverage", 0), "region": candidate["region"]} for candidate in candidates]
        outputs["B3_mvp"][row["id"]] = point_in_bbox(mvp_official(points, pseudo), row["target_bbox"])
        outputs["M1_ccm"][row["id"]] = point_in_bbox(candidates[ccm_select(row, tables, priors)]["point"], row["target_bbox"])
        outputs["pass_at_n"][row["id"]] = any(point_in_bbox(point, row["target_bbox"]) for point in points)
    return outputs


def merge(target, source):
    for rule, values in source.items():
        if set(target[rule]) & set(values):
            raise ValueError("CALA-A duplicate held-out output")
        target[rule].update(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    static = json.loads(args.static.read_text())
    if static["status"] != "PASS":
        raise ValueError("CALA-A requires completed static result")
    context = load_bank()
    outputs = {budget: {rule: {} for rule in ("B3_mvp", "M1_ccm", "pass_at_n")} for budget in BUDGETS}
    route_counts = {str(budget): {action_name(action): 0 for action in SHARED_ACTIONS} for budget in BUDGETS}
    fold_reports = {}
    for fold in range(5):
        dev_ids, test_ids = split_ids(context, fold)
        correct, dev_accuracy, dev_kappa = development_statistics(context, dev_ids)
        train_x, train_y = training_matrix(context, dev_ids, fold, correct, dev_accuracy, dev_kappa)
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, penalty="l2", class_weight="balanced", random_state=SEED, max_iter=500),
        )
        classifier.fit(train_x, train_y)
        fold_reports[str(fold)] = {
            "development_rows": len(dev_ids), "test_rows": len(test_ids),
            "training_examples": len(train_y), "training_positives": int(train_y.sum()),
        }
        routed_to_16 = {
            row_id: route_row(context, row_id, classifier, 16, dev_accuracy, dev_kappa)
            for row_id in test_ids
        }
        for budget in BUDGETS:
            actions_by_row = {row_id: actions[:budget] for row_id, actions in routed_to_16.items()}
            for actions in actions_by_row.values():
                if len(actions) != budget or len(set(actions)) != budget or tuple(actions[:6]) != SCOUT:
                    raise ValueError(f"CALA-A route contract mismatch: fold{fold}/N{budget}")
                for action in actions:
                    route_counts[str(budget)][action_name(action)] += 1
            merge(outputs[budget], evaluate_variable_fold(context, dev_ids, test_ids, actions_by_row))
    rows = [context["metadata"][row_id] for row_id in context["row_ids"]]
    accuracy = {
        str(budget): {rule: sum(values.values()) / len(values) for rule, values in outputs[budget].items()}
        for budget in BUDGETS
    }
    comparisons = {}
    for budget in BUDGETS:
        for rule in ("B3_mvp", "M1_ccm", "pass_at_n"):
            for baseline in ("Uniform_Mixed", "CALA_S"):
                baseline_accuracy = static["accuracy"][baseline][str(budget)][rule]
                baseline_outputs = None
                # Reconstruct baseline outputs under the same frozen fold protocol.
                from cala_static import evaluate_fold
                reconstructed = {}
                for fold in range(5):
                    dev_ids, test_ids = split_ids(context, fold)
                    if baseline == "Uniform_Mixed":
                        selected = UNIFORM_SEQUENCE[:budget]
                    else:
                        names = static["fold_sequences"][str(fold)]["sequences"]["CALA_S"][:budget]
                        selected = tuple((name.rsplit("/view", 1)[0], int(name.rsplit("/view", 1)[1])) for name in names)
                    fold_output = evaluate_fold(context, dev_ids, test_ids, selected)[rule]
                    if set(reconstructed) & set(fold_output):
                        raise ValueError("CALA-A baseline reconstruction overlap")
                    reconstructed.update(fold_output)
                if abs(sum(reconstructed.values()) / len(reconstructed) - baseline_accuracy) > 1e-12:
                    raise ValueError(f"CALA-A baseline reconstruction mismatch: {baseline}/N{budget}/{rule}")
                record = paired_bootstrap(rows, outputs[budget][rule], reconstructed, resamples=10000, seed=SEED)
                record.update({
                    "left": f"CALA_A/N{budget}/{rule}", "right": f"{baseline}/N{budget}/{rule}",
                    "left_accuracy": accuracy[str(budget)][rule], "right_accuracy": baseline_accuracy,
                })
                comparisons[f"CALA_A_N{budget}_{rule}_vs_{baseline}"] = record
    primary = comparisons["CALA_A_N12_B3_mvp_vs_CALA_S"]
    result = {
        "schema_version": 1, "status": "PASS", "rows": 1581,
        "budgets": list(BUDGETS), "scout": [action_name(action) for action in SCOUT],
        "accuracy": accuracy, "comparisons": comparisons,
        "fold_reports": fold_reports, "route_counts": route_counts,
        "output_sha256": {
            str(budget): {
                rule: hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                for rule, values in outputs[budget].items()
            }
            for budget in BUDGETS
        },
        "sources": {
            "protocol_sha256": sha256_file(RUN_DIR / "configs/protocol.yaml"),
            "static_result_sha256": sha256_file(args.static),
        },
        "adaptive_success": {"comparison": "CALA_A_N12_B3_mvp_vs_CALA_S", "ci_99_lower_positive": primary["ci_99"][0] > 0, "success": primary["ci_99"][0] > 0},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accuracy": accuracy, "adaptive_success": result["adaptive_success"], "primary": primary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()