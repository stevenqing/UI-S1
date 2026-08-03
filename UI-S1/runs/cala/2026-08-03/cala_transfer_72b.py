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
SCALEUP_DIR = ROOT / "runs/scaleup/2026-08-02"
H1_DIR = ROOT / "runs/ccm-h2h/2026-07-31/h1"
H3_DIR = ROOT / "runs/ccm-h2h/2026-07-31/h3"
CLOSING_DIR = ROOT / "runs/closing/2026-08-02"
sys.path.insert(0, str(H1_DIR))
sys.path.insert(0, str(H3_DIR))
sys.path.insert(0, str(CLOSING_DIR))
from aggregators_coord import mvp_official
from h3_eval import ccm_select, fit_ccm, group_folds, point_in_bbox
from f1_paired_bootstrap import paired_bootstrap


SEED = 20260803
MODELS = ("GTA1-72B", "UI-Venus-Ground-72B", "Qwen3.5-122B-A10B")
MIXED_ACTIONS = tuple((model, region) for region in range(4) for model in MODELS)
UNIFORM_N8 = MIXED_ACTIONS[:8]
GTA_N8 = tuple(("GTA1-72B", region) for region in range(8))
SCOUT = tuple((model, region) for region in range(2) for model in MODELS)
ACTION_INDEX = {action: index for index, action in enumerate(MIXED_ACTIONS)}


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def action_name(action):
    return f"{action[0]}/region{action[1]}"


def cohen_kappa(left, right):
    left = np.asarray(left, dtype=np.bool_)
    right = np.asarray(right, dtype=np.bool_)
    observed = float(np.mean(left == right))
    left_rate, right_rate = float(np.mean(left)), float(np.mean(right))
    expected = left_rate * right_rate + (1 - left_rate) * (1 - right_rate)
    return 1.0 if math.isclose(expected, 1.0) else (observed - expected) / (1 - expected)


def load_unique(path):
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        if row["id"] in rows: raise ValueError(f"CALA-72 duplicate identity: {row['id']}")
        rows[row["id"]] = row
    if len(rows) != 1581: raise ValueError(f"CALA-72 requires 1,581 rows: {path}")
    return rows


def load_context():
    g2 = json.loads((SCALEUP_DIR / "g2_mixed_72b.json").read_text())
    paths = {
        "regions": SCALEUP_DIR / "raw/g2-regions.jsonl",
        "GTA1-72B": SCALEUP_DIR / "raw/g2-score-gta1.jsonl",
        "UI-Venus-Ground-72B": SCALEUP_DIR / "raw/g2-score-venus.jsonl",
        "Qwen3.5-122B-A10B": SCALEUP_DIR / "raw/g2-score-qwen35.jsonl",
    }
    for key, path in paths.items():
        source_key = "regions" if key == "regions" else key
        if sha256_file(path) != g2["sources"][source_key]["sha256"]:
            raise ValueError(f"CALA-72 source hash mismatch: {key}")
    regions = load_unique(paths["regions"])
    scores = {model: load_unique(paths[model]) for model in MODELS}
    label_rows = [json.loads(line) for line in (ROOT / "runs/ccm-h2h/2026-07-31/h3/raw/shared_regions_n4.jsonl").read_text().splitlines() if line.strip()]
    labels = {row["id"]: row for row in label_rows}
    if set(regions) != set(labels) or any(set(values) != set(labels) for values in scores.values()):
        raise ValueError("CALA-72 source identity mismatch")
    bank = {}
    for model in MODELS:
        for row_id, row in scores[model].items():
            predictions = {value["region_index"]: value for value in row["predictions"]}
            required = range(8) if model == "GTA1-72B" else range(4)
            if not set(required).issubset(predictions):
                raise ValueError(f"CALA-72 missing action: {model}/{row_id}")
            region_map = {value["region_index"]: value for value in regions[row_id]["regions"]}
            for region_index in required:
                prediction = predictions[region_index]
                point = prediction["point"] if prediction["parse_ok"] else [-1.0, -1.0]
                bank.setdefault((model, region_index), {})[row_id] = {
                    "model": model, "view_index": region_index,
                    "point": list(map(float, point)), "region": region_map[region_index]["region"],
                    "coverage": float(region_map[region_index]["coverage"]) if model == "GTA1-72B" else 0.0,
                }
    metadata = {
        row_id: {"id": row_id, "application": row["application"], "target_bbox": row["target_bbox"], "img_size": row["img_size"], "instruction": row["instruction"]}
        for row_id, row in labels.items()
    }
    fold_for_group, fold_rows = group_folds(list(metadata.values()))
    return {"row_ids": tuple(sorted(labels)), "metadata": metadata, "fold_for_group": fold_for_group, "fold_rows": fold_rows, "bank": bank, "regions": regions, "sources": {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in paths.items()}}


def split_ids(context, fold):
    dev = tuple(row_id for row_id in context["row_ids"] if context["fold_for_group"][context["metadata"][row_id]["application"]] != fold)
    test = tuple(row_id for row_id in context["row_ids"] if context["fold_for_group"][context["metadata"][row_id]["application"]] == fold)
    return dev, test


def correct(context, action, row_ids):
    return np.asarray([point_in_bbox(context["bank"][action][row_id]["point"], context["metadata"][row_id]["target_bbox"]) for row_id in row_ids], dtype=np.bool_)


def rows_for(context, row_ids, actions_by_row):
    return [{"id": row_id, "application": context["metadata"][row_id]["application"], "target_bbox": context["metadata"][row_id]["target_bbox"], "candidates": [context["bank"][action][row_id] for action in actions_by_row[row_id]]} for row_id in row_ids]


def constant_rows(context, row_ids, actions):
    return rows_for(context, row_ids, {row_id: actions for row_id in row_ids})


def b3_accuracy(context, row_ids, actions):
    count = 0
    for row_id in row_ids:
        candidates = [context["bank"][action][row_id] for action in actions]
        points = [value["point"] for value in candidates]
        pseudo = [{"coverage": value["coverage"], "region": value["region"]} for value in candidates]
        count += int(point_in_bbox(mvp_official(points, pseudo), context["metadata"][row_id]["target_bbox"]))
    return count / len(row_ids)


def development_stats(context, dev_ids):
    correctness = {action: correct(context, action, dev_ids) for action in MIXED_ACTIONS}
    accuracy = {action: float(values.mean()) for action, values in correctness.items()}
    kappa = {(left, right): (1.0 if left == right else cohen_kappa(~correctness[left], ~correctness[right])) for left in MIXED_ACTIONS for right in MIXED_ACTIONS}
    return correctness, accuracy, kappa


def static_sequence(context, dev_ids, correctness, accuracy, kappa):
    selected = []
    covered = np.zeros(len(dev_ids), dtype=np.bool_)
    while len(selected) < 8:
        choices = []
        for action in MIXED_ACTIONS:
            if action in selected: continue
            resulting = [*selected, action]
            coverage = float(np.mean(covered | correctness[action]))
            b3 = b3_accuracy(context, dev_ids, resulting)
            kappas = [kappa[left, right] for index, left in enumerate(resulting) for right in resulting[index + 1:]]
            mean_kappa = float(np.mean(kappas)) if kappas else 0.0
            choices.append(((coverage, b3, -mean_kappa, accuracy[action], -ACTION_INDEX[action]), action))
        _, winner = max(choices, key=lambda value: value[0])
        selected.append(winner); covered |= correctness[winner]
    return tuple(selected)


def geometry(context, row_id, selected):
    width, height = context["metadata"][row_id]["img_size"]
    points = np.asarray([[context["bank"][a][row_id]["point"][0] / width, context["bank"][a][row_id]["point"][1] / height] for a in selected])
    distances = [math.dist(points[i], points[j]) for i in range(len(points)) for j in range(i + 1, len(points))]
    counts = [sum(a[0] == model for a in selected) / 8 for model in MODELS]
    threshold = 14 / math.hypot(width, height)
    cluster = max(sum(math.dist(point, other) <= threshold for other in points) / len(points) for point in points)
    return [*points.mean(axis=0), *points.std(axis=0), float(np.mean(distances)), float(np.max(distances)), cluster, *counts]


def action_features(context, row_id, action):
    model, region_index = action
    width, height = context["metadata"][row_id]["img_size"]
    region = {value["region_index"]: value for value in context["regions"][row_id]["regions"]}[region_index]
    left, top, right, bottom = region["region"]
    return [*[float(model == value) for value in MODELS], region_index / 3, region_index / 3, (right-left)*(bottom-top)/(width*height), (left+right)/(2*width), (top+bottom)/(2*height), region["coverage"]/100]


def feature(context, row_id, action, selected, accuracy, kappa):
    width, height = context["metadata"][row_id]["img_size"]
    return [*action_features(context, row_id, action), accuracy[action], float(np.mean([kappa[action, value] for value in selected])), *geometry(context, row_id, selected), width/height, math.log(width*height), len(context["metadata"][row_id]["instruction"])/200]


def trajectory(row_id, fold, index):
    seed = int.from_bytes(hashlib.sha256(f"{SEED}|72B|{row_id}|{fold}|{index}".encode()).digest()[:8], "big")
    remaining = [action for action in MIXED_ACTIONS if action not in SCOUT]
    order = np.random.default_rng(seed).permutation(len(remaining))
    return tuple(remaining[value] for value in order)


def training_matrix(context, dev_ids, fold, correctness, accuracy, kappa):
    row_index = {row_id: index for index, row_id in enumerate(dev_ids)}
    x, y = [], []
    for row_id in dev_ids:
        for trajectory_index in range(4):
            order = trajectory(row_id, fold, trajectory_index)
            selected = list(SCOUT)
            for state in range(2):
                selected_correct = any(correctness[action][row_index[row_id]] for action in selected)
                for action in MIXED_ACTIONS:
                    if action in selected: continue
                    x.append(feature(context, row_id, action, selected, accuracy, kappa))
                    y.append(bool(correctness[action][row_index[row_id]] and not selected_correct))
                if state == 0: selected.append(order[0])
    values, labels = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int8)
    if values.shape[1] != 24 or not 0 < labels.sum() < len(labels): raise ValueError("CALA-72 training matrix invalid")
    return values, labels


def adaptive_route(context, row_id, classifier, accuracy, kappa):
    selected = list(SCOUT)
    while len(selected) < 8:
        actions = [action for action in MIXED_ACTIONS if action not in selected]
        probabilities = classifier.predict_proba(np.asarray([feature(context, row_id, action, selected, accuracy, kappa) for action in actions], dtype=np.float32))[:, 1]
        winner = max(range(len(actions)), key=lambda index: (probabilities[index], accuracy[actions[index]], -ACTION_INDEX[actions[index]]))
        selected.append(actions[winner])
    return tuple(selected)


def evaluate_fold(context, dev_ids, test_ids, actions_by_row, dev_actions):
    tables, priors = fit_ccm(constant_rows(context, dev_ids, dev_actions))
    outputs = {rule: {} for rule in ("B3_mvp", "M1_ccm", "pass_at_n")}
    for row in rows_for(context, test_ids, actions_by_row):
        candidates = row["candidates"]; points = [value["point"] for value in candidates]
        pseudo = [{"coverage": value["coverage"], "region": value["region"]} for value in candidates]
        outputs["B3_mvp"][row["id"]] = point_in_bbox(mvp_official(points, pseudo), row["target_bbox"])
        outputs["M1_ccm"][row["id"]] = point_in_bbox(candidates[ccm_select(row, tables, priors)]["point"], row["target_bbox"])
        outputs["pass_at_n"][row["id"]] = any(point_in_bbox(point, row["target_bbox"]) for point in points)
    return outputs


def merge(target, source):
    for rule, values in source.items():
        if set(target[rule]) & set(values): raise ValueError("CALA-72 duplicate output")
        target[rule].update(values)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    context = load_context()
    methods = ("GTA1_N8", "Uniform_Mixed_N8", "CALA_S_N8", "CALA_A_N8")
    outputs = {method: {rule: {} for rule in ("B3_mvp", "M1_ccm", "pass_at_n")} for method in methods}
    folds = {}
    for fold in range(5):
        dev_ids, test_ids = split_ids(context, fold)
        correctness, accuracy, kappa = development_stats(context, dev_ids)
        static = static_sequence(context, dev_ids, correctness, accuracy, kappa)
        train_x, train_y = training_matrix(context, dev_ids, fold, correctness, accuracy, kappa)
        classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, penalty="l2", class_weight="balanced", random_state=SEED, max_iter=500))
        classifier.fit(train_x, train_y)
        adaptive = {row_id: adaptive_route(context, row_id, classifier, accuracy, kappa) for row_id in test_ids}
        action_sets = {
            "GTA1_N8": {row_id: GTA_N8 for row_id in test_ids},
            "Uniform_Mixed_N8": {row_id: UNIFORM_N8 for row_id in test_ids},
            "CALA_S_N8": {row_id: static for row_id in test_ids},
            "CALA_A_N8": adaptive,
        }
        folds[str(fold)] = {"dev_rows": len(dev_ids), "test_rows": len(test_ids), "CALA_S": [action_name(value) for value in static], "training_examples": len(train_y), "training_positives": int(train_y.sum())}
        for method, values in action_sets.items():
            if any(len(actions) != 8 or len(set(actions)) != 8 for actions in values.values()): raise ValueError(f"CALA-72 N8 budget mismatch: {method}")
            dev_actions = GTA_N8 if method == "GTA1_N8" else UNIFORM_N8 if method in ("Uniform_Mixed_N8", "CALA_A_N8") else static
            merge(outputs[method], evaluate_fold(context, dev_ids, test_ids, values, dev_actions))
    accuracy = {method: {rule: sum(values.values())/1581 for rule, values in rules.items()} for method, rules in outputs.items()}
    rows = [context["metadata"][row_id] for row_id in context["row_ids"]]
    comparisons = {}
    for left in ("CALA_S_N8", "CALA_A_N8", "Uniform_Mixed_N8"):
        for right in ("Uniform_Mixed_N8", "CALA_S_N8", "GTA1_N8"):
            if left == right: continue
            for rule in ("B3_mvp", "M1_ccm", "pass_at_n"):
                record = paired_bootstrap(rows, outputs[left][rule], outputs[right][rule], resamples=10000, seed=SEED)
                record.update({"left": f"{left}/{rule}", "right": f"{right}/{rule}", "left_accuracy": accuracy[left][rule], "right_accuracy": accuracy[right][rule]})
                comparisons[f"{left}_{rule}_vs_{right}"] = record
    result = {"schema_version": 1, "status": "PASS", "rows": 1581, "budget": 8, "accuracy": accuracy, "comparisons": comparisons, "folds": folds, "sources": {**context["sources"], "operations_sha256": sha256_file(RUN_DIR / "AMENDMENT_001_72B_TRANSFER_OPERATIONS.md")}}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"accuracy": accuracy, "key_comparisons": {key: value for key, value in comparisons.items() if key in ("CALA_S_N8_B3_mvp_vs_Uniform_Mixed_N8", "CALA_A_N8_B3_mvp_vs_Uniform_Mixed_N8", "CALA_A_N8_B3_mvp_vs_CALA_S_N8")}}, indent=2, sort_keys=True))


if __name__ == "__main__": main()