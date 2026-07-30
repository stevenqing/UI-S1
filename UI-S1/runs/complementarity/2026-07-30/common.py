import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from scoring import ACTION_TO_ID, GROUNDING_ACTIONS, SIMPLE_ACTIONS, TEXT_ACTIONS, text_f1, token_f1


RUN_DIR = Path(__file__).resolve().parent
ROWS_PATH = RUN_DIR / "rows.parquet"
FOLDS_PATH = RUN_DIR / "folds.json"


def load_rows(bench: str, setting: str, include_quarantine: bool = False) -> list[dict]:
    table = pq.read_table(
        ROWS_PATH,
        filters=[("bench", "=", bench), ("setting", "=", setting)],
    )
    if not include_quarantine:
        table = table.filter(pc.invert(table["quarantine"]))
    return table.to_pylist()


def pivot_rows(rows: list[dict]) -> tuple[list[str], list[str], dict[str, dict[str, dict]]]:
    models = sorted({row["model"] for row in rows})
    by_identity = defaultdict(dict)
    for row in rows:
        if row["model"] in by_identity[row["row_id"]]:
            raise ValueError(f"duplicate tidy row: {row['row_id']}/{row['model']}")
        by_identity[row["row_id"]][row["model"]] = row
    if any(set(model_rows) != set(models) for model_rows in by_identity.values()):
        raise ValueError("model coverage differs across row identities")
    identities = sorted(by_identity)
    return identities, models, dict(by_identity)


def fold_for(pool: str, group_key: str) -> int:
    folds = json.loads(FOLDS_PATH.read_text())
    return folds["pools"][pool]["group_to_fold"][group_key]


def split_identities(pool: str, identities: list[str], pivot: dict, test_fold: int) -> tuple[list[str], list[str]]:
    train, test = [], []
    for row_id in identities:
        row = next(iter(pivot[row_id].values()))
        target = test if fold_for(pool, row["group_key"]) == test_fold else train
        target.append(row_id)
    return train, test


def score_prediction(reference: dict, action: str, x, y, parameter: str) -> bool:
    if reference["bench"] == "androidcontrol":
        if action != reference["gt_action"]:
            return False
        if reference["gt_action"] in GROUNDING_ACTIONS:
            if x is None or y is None or math.isnan(x) or math.isnan(y):
                return False
            return math.dist((x, y), (reference["gt_x"], reference["gt_y"])) < 0.14
        if reference["gt_action"] in TEXT_ACTIONS:
            return text_f1(parameter, reference["gt_param"]) >= 0.5
        if reference["gt_action"] in SIMPLE_ACTIONS:
            return True
        raise ValueError(f"unknown AndroidControl action: {reference['gt_action']}")

    if reference["bench"] != "mind2web":
        raise ValueError(reference["bench"])
    bbox = reference["gt_bbox"]
    element = bool(
        bbox is not None and x is not None and y is not None
        and not math.isnan(x) and not math.isnan(y)
        and bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]
    )
    if action not in ACTION_TO_ID:
        return False
    predicted = str(ACTION_TO_ID[action])
    if action in {"TYPE", "SELECT"}:
        predicted += " " + parameter.lower()
    expected = str(ACTION_TO_ID[reference["gt_action"]])
    if reference["gt_action"] in {"TYPE", "SELECT"}:
        expected += " " + reference["gt_param"].lower()
    return element and token_f1(predicted, expected) == 1.0


def micro(successes) -> float:
    values = list(successes)
    return sum(values) / len(values) if values else float("nan")


def episode_macro(success_by_row: dict[str, bool], pivot: dict) -> float:
    episode_values = defaultdict(list)
    for row_id, success in success_by_row.items():
        row = next(iter(pivot[row_id].values()))
        episode_values[row["episode_id"]].append(float(success))
    return float(np.mean([np.mean(values) for values in episode_values.values()]))


def model_success_sets(identities: list[str], models: list[str], pivot: dict) -> dict[str, set[str]]:
    return {
        model: {row_id for row_id in identities if pivot[row_id][model]["success"]}
        for model in models
    }


def parse_failure_rate(identities: list[str], model: str, pivot: dict) -> float:
    return 1 - micro(pivot[row_id][model]["parse_ok"] for row_id in identities)


def geometric_median(points: list[tuple[float, float]], weights=None, tolerance=1e-7, max_iter=50):
    if not points:
        return None
    values = np.asarray(points, dtype=np.float64)
    weight_values = np.ones(len(values), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if len(values) == 1:
        return float(values[0, 0]), float(values[0, 1])
    if len(values) == 2:
        if weight_values[0] > weight_values[1]:
            estimate = values[0]
        elif weight_values[1] > weight_values[0]:
            estimate = values[1]
        else:
            estimate = values.mean(axis=0)
        return float(estimate[0]), float(estimate[1])
    estimate = np.average(values, axis=0, weights=weight_values)
    for _ in range(max_iter):
        distances = np.linalg.norm(values - estimate, axis=1)
        if np.any(distances < tolerance):
            estimate = values[np.argmin(distances)]
            break
        adjusted = weight_values / distances
        updated = np.sum(values * adjusted[:, None], axis=0) / adjusted.sum()
        if np.linalg.norm(updated - estimate) < tolerance:
            estimate = updated
            break
        estimate = updated
    return float(estimate[0]), float(estimate[1])


def auc_roc(labels, scores) -> float | None:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    positive_rank_sum = ranks[labels == 1].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))