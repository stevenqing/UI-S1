import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
ALLOCATION_DIR = ROOT / "runs/allocation-law/2026-08-01"
DIVERSITY_DIR = ROOT / "runs/diversity-axis/2026-08-02"
H3_DIR = ROOT / "runs/ccm-h2h/2026-07-31/h3"
sys.path.insert(0, str(DIVERSITY_DIR))
sys.path.insert(0, str(ALLOCATION_DIR))
sys.path.insert(0, str(H3_DIR))
from allocation_eval import candidate_for_unit, group_folds, point_in_bbox
from x3_curve_stats import load_sources


MODEL_ORDER = ("GTA1-7B", "Qwen3-VL-8B-Instruct", "UI-TARS-7B-SFT")
SHARED_ACTIONS = tuple((model, view) for model in MODEL_ORDER for view in range(12))
UNIFORM_SEQUENCE = tuple((model, view) for view in range(12) for model in MODEL_ORDER)
V_ONLY_SEQUENCE = tuple(("GTA1-7B", view) for view in range(16))
BUDGETS = (4, 8, 12, 16)


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def action_name(action):
    return f"{action[0]}/view{action[1]}"


def load_bank():
    gta1, generated, _ = load_sources()
    row_ids = tuple(sorted(gta1))
    if len(row_ids) != 1581:
        raise ValueError("CALA requires 1,581 identities")
    metadata = {
        row_id: {
            "id": row_id,
            "application": gta1[row_id]["application"],
            "target_bbox": gta1[row_id]["target_bbox"],
            "img_size": gta1[row_id]["img_size"],
            "instruction": gta1[row_id]["instruction"],
        }
        for row_id in row_ids
    }
    fold_for_group, fold_rows = group_folds(list(metadata.values()))
    bank = {}
    for action in set(SHARED_ACTIONS) | set(V_ONLY_SEQUENCE):
        model, view = action
        bank[action] = {
            row_id: candidate_for_unit(row_id, model, view, gta1, generated)
            for row_id in row_ids
        }
    for action, values in bank.items():
        if len(values) != 1581:
            raise ValueError(f"CALA action identity mismatch: {action_name(action)}")
        for row_id, candidate in values.items():
            point = candidate["point"]
            if len(point) != 2 or not all(math.isfinite(float(value)) for value in point):
                raise ValueError(f"CALA invalid point: {action_name(action)}/{row_id}")
    return {
        "row_ids": row_ids,
        "metadata": metadata,
        "fold_for_group": fold_for_group,
        "fold_rows": fold_rows,
        "bank": bank,
    }


def split_ids(context, fold):
    dev = tuple(row_id for row_id in context["row_ids"] if context["fold_for_group"][context["metadata"][row_id]["application"]] != fold)
    test = tuple(row_id for row_id in context["row_ids"] if context["fold_for_group"][context["metadata"][row_id]["application"]] == fold)
    if len(dev) + len(test) != 1581 or not dev or not test:
        raise ValueError(f"CALA fold split mismatch: {fold}")
    return dev, test


def correctness(context, action, row_ids):
    return np.asarray([
        point_in_bbox(context["bank"][action][row_id]["point"], context["metadata"][row_id]["target_bbox"])
        for row_id in row_ids
    ], dtype=np.bool_)


def build_rows(context, row_ids, actions):
    if len(actions) != len(set(actions)):
        raise ValueError("CALA action duplication")
    return [
        {
            "id": row_id,
            "application": context["metadata"][row_id]["application"],
            "target_bbox": context["metadata"][row_id]["target_bbox"],
            "candidates": [context["bank"][action][row_id] for action in actions],
        }
        for row_id in row_ids
    ]


def cohen_kappa(left, right):
    left = np.asarray(left, dtype=np.bool_)
    right = np.asarray(right, dtype=np.bool_)
    observed = float(np.mean(left == right))
    left_rate = float(np.mean(left))
    right_rate = float(np.mean(right))
    expected = left_rate * right_rate + (1 - left_rate) * (1 - right_rate)
    if math.isclose(expected, 1.0):
        return 1.0
    return (observed - expected) / (1 - expected)


def mean_failure_kappa(correct_by_action, actions):
    if len(actions) < 2:
        return 0.0
    values = []
    for left_index, left in enumerate(actions):
        for right in actions[left_index + 1:]:
            values.append(cohen_kappa(~correct_by_action[left], ~correct_by_action[right]))
    return float(np.mean(values))
