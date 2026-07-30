import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from scoring import GROUNDING_ACTIONS, label_android_row


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
W2_ROOT = RUN_DIR / "w2_artifacts"
UPSTREAM_AC = ROOT / "runs/androidcontrol-rft/2026-07-29/artifacts"
COLLISION_ROWS = RUN_DIR / "rows.parquet"
VIEWS = ("full", "v1", "v2", "v3", "v4")
AC_MODELS = ("gui-r1-7b", "ui-agile-7b")
AC_SETTINGS = ("low", "high")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson(successes: int, total: int, z: float = 1.959963984540054):
    if total == 0:
        return None
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(probability * (1 - probability) / total + z * z / (4 * total * total)) / denominator
    return [center - spread, center + spread]


def clean_ac_indices(setting: str) -> list[int]:
    table = pq.read_table(
        COLLISION_ROWS,
        filters=[("bench", "=", "androidcontrol"), ("setting", "=", setting), ("model", "=", "gui-r1-7b")],
        columns=["row_id", "quarantine"],
    )
    return [int(row_id) for row_id, quarantine in zip(table["row_id"].to_pylist(), table["quarantine"].to_pylist()) if not quarantine]


def ac_prediction_path(model: str, setting: str, view: str) -> Path:
    if view == "full":
        return UPSTREAM_AC / model / setting / "predictions.jsonl"
    return W2_ROOT / "androidcontrol" / model / setting / view / "predictions.jsonl"


def coordinate(row: dict):
    values = row.get("pred_coord")
    if not values or len(values) < 2 or not all(math.isfinite(float(value)) for value in values[:2]):
        return None
    return float(values[0]), float(values[1])


def grounding_success(row: dict) -> bool | None:
    if row["gt_action"] not in GROUNDING_ACTIONS:
        return None
    point = coordinate(row)
    if point is None:
        return False
    width, height = row["image_size"]
    expected_x, expected_y = row["gt_bbox"][:2]
    distance = math.sqrt(((expected_x - point[0]) / width) ** 2 + ((expected_y - point[1]) / height) ** 2)
    return distance < 0.14


def summarize_cell(rows: list[dict], clean_indices: list[int]) -> dict:
    if len(rows) != 7708 or [row["index"] for row in rows] != list(range(7708)):
        raise ValueError("W2 completed cell must contain ordered indices 0..7707")
    selected = [rows[index] for index in clean_indices]
    labels = [label_android_row(row) for row in selected]
    return {
        "rows": len(selected),
        "step_successes": sum(label["step_success"] for label in labels),
        "step_sr": sum(label["step_success"] for label in labels) / len(labels),
        "action_accuracy": sum(label["action_correct"] for label in labels) / len(labels),
    }


def paired_view_flips(full_rows, view_rows, clean_indices):
    action_flips = 0
    action_valid = 0
    grounding_flips = 0
    grounding_stable_type = 0
    grounding_rows = 0
    for index in clean_indices:
        full, view = full_rows[index], view_rows[index]
        action_valid += 1
        action_flips += int(full.get("pred_action") != view.get("pred_action"))
        if full["gt_action"] not in GROUNDING_ACTIONS:
            continue
        grounding_rows += 1
        if full.get("pred_action") != view.get("pred_action"):
            continue
        grounding_stable_type += 1
        grounding_flips += int(grounding_success(full) != grounding_success(view))
    return {
        "action_type": {
            "flips": action_flips, "denominator": action_valid,
            "rate": action_flips / action_valid, "wilson_95": wilson(action_flips, action_valid),
        },
        "grounding_given_stable_type": {
            "gt_grounding_rows": grounding_rows,
            "stable_type_rows": grounding_stable_type,
            "flips": grounding_flips, "denominator": grounding_stable_type,
            "rate": grounding_flips / grounding_stable_type if grounding_stable_type else None,
            "wilson_95": wilson(grounding_flips, grounding_stable_type),
        },
    }


def analyze_androidcontrol():
    result = {"cells": {}, "full_to_view_flips": {}, "k1": {}, "noise": {}}
    for model in AC_MODELS:
        for setting in AC_SETTINGS:
            clean = clean_ac_indices(setting)
            full_path = ac_prediction_path(model, setting, "full")
            full_rows = read_jsonl(full_path)
            if len(full_rows) != 7708:
                raise ValueError(f"missing inherited full cell: {model}/{setting}")
            cell_key = f"{model}/{setting}/full"
            result["cells"][cell_key] = {
                **summarize_cell(full_rows, clean), "source": str(full_path.relative_to(ROOT)),
                "sha256": sha256_file(full_path),
            }
            available_scores = [result["cells"][cell_key]["step_sr"]]
            available_views = ["full"]
            for view in VIEWS[1:]:
                path = ac_prediction_path(model, setting, view)
                if not path.exists():
                    continue
                rows = read_jsonl(path)
                if len(rows) != 7708:
                    continue
                cell_key = f"{model}/{setting}/{view}"
                result["cells"][cell_key] = {
                    **summarize_cell(rows, clean), "source": str(path.relative_to(ROOT)),
                    "sha256": sha256_file(path),
                }
                flip_key = f"{model}/{setting}/full_to_{view}"
                result["full_to_view_flips"][flip_key] = paired_view_flips(full_rows, rows, clean)
                available_scores.append(result["cells"][cell_key]["step_sr"])
                available_views.append(view)
            noise_key = f"{model}/{setting}"
            noise = {"available_views": available_views, "complete": set(available_views) == set(VIEWS)}
            if noise["complete"]:
                noise.update({
                    "mean_step_sr": float(np.mean(available_scores)),
                    "sample_sd": float(np.std(available_scores, ddof=1)),
                    "mde": float(2 * np.std(available_scores, ddof=1)),
                })
            result["noise"][noise_key] = noise
            k1_key = f"{model}/{setting}"
            flip = result["full_to_view_flips"].get(f"{model}/{setting}/full_to_v1")
            if flip:
                action_rate = flip["action_type"]["rate"]
                grounding_rate = flip["grounding_given_stable_type"]["rate"]
                result["k1"][k1_key] = {
                    "action_flip_rate": action_rate,
                    "grounding_flip_rate": grounding_rate,
                    "difference": grounding_rate - action_rate,
                    "prediction_satisfied": grounding_rate > action_rate,
                }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flips", type=Path, required=True)
    parser.add_argument("--noise", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    args = parser.parse_args()
    androidcontrol = analyze_androidcontrol()
    complete_v1 = len(androidcontrol["k1"]) == len(AC_MODELS) * len(AC_SETTINGS)
    complete_noise = all(item["complete"] for item in androidcontrol["noise"].values())
    flips = {
        "status": "PASS" if complete_v1 else "PARTIAL",
        "contract": {
            "action_flip": "pred_action(view) != pred_action(full)",
            "grounding_flip": "GT grounding row; predicted type stable; grounding correctness changes",
            "quarantine_excluded": True,
        },
        "androidcontrol": androidcontrol,
        "mind2web": {"status": "PENDING_INFERENCE"},
    }
    noise = {
        "status": "PASS" if complete_noise else "PARTIAL",
        "definition": "MDE = 2 * sample SD over full,v1,v2,v3,v4",
        "androidcontrol": androidcontrol["noise"],
        "mind2web": {"status": "PENDING_INFERENCE"},
    }
    allocation = {
        "status": "PENDING_INFERENCE",
        "reason": "requires complete five-view prediction pools",
    }
    args.flips.write_text(json.dumps(flips, indent=2, sort_keys=True) + "\n")
    args.noise.write_text(json.dumps(noise, indent=2, sort_keys=True) + "\n")
    args.allocation.write_text(json.dumps(allocation, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"flips": flips["status"], "noise": noise["status"], "k1": androidcontrol["k1"]}, indent=2))


if __name__ == "__main__":
    main()