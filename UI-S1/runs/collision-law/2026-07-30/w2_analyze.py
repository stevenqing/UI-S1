import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from aggregators import pka_medoid, plurality_then_density
from pka import Prediction
from scoring import GROUNDING_ACTIONS, label_android_row
from w1_run import cohen_kappa, fold_map, load_pool, prediction_from_row, score_prediction, split_rows


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
W2_ROOT = RUN_DIR / "w2_artifacts"
UPSTREAM_AC = ROOT / "runs/androidcontrol-rft/2026-07-29/artifacts"
UPSTREAM_M2W_FULL = ROOT / "runs/mind2web-tongui/2026-07-28/artifacts/tongui-7b/merged/predictions.jsonl"
INHERITED_GUI_R1_HIGH_V4 = ROOT / "runs/complementarity/2026-07-30/e5_artifacts/androidcontrol/original_768/predictions.jsonl"
COLLISION_ROWS = RUN_DIR / "rows.parquet"
VIEWS = ("full", "v1", "v2", "v3", "v4")
AC_MODELS = ("gui-r1-7b", "ui-agile-7b")
AC_SETTINGS = ("low", "high")
P3_MODELS = {
    "androidcontrol": ("ui-agile-3b", "ui-agile-7b", "gui-r1-3b", "gui-r1-7b", "ui-r1-e-3b"),
    "mind2web": ("tongui-7b", "tongui-32b", "cogagent-18b", "tongui-3b", "ui-tars-72b"),
}
P3_REPRESENTATIVE = {"androidcontrol": "gui-r1-7b", "mind2web": "tongui-7b"}
P3_SEED = 20260730


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
    if (model, setting, view) == ("gui-r1-7b", "high", "v4"):
        return INHERITED_GUI_R1_HIGH_V4
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


def area_bin(area: float) -> str:
    if area <= 0.001:
        return "tiny"
    if area <= 0.005:
        return "small"
    return "regular"


def load_mind2web_full():
    table = pq.read_table(
        COLLISION_ROWS,
        filters=[("bench", "=", "mind2web"), ("setting", "=", "visual"), ("model", "=", "tongui-7b")],
    )
    by_identity = {row["row_id"]: row for row in table.to_pylist()}
    trace = read_jsonl(UPSTREAM_M2W_FULL)
    identities = [f"{row['annot_id']}__{row['action_uid']}" for row in trace]
    if len(by_identity) != 2080 or len(trace) != 2080 or set(by_identity) != set(identities):
        raise ValueError("expected 2,080 TongUI full rows")
    output = []
    for index, identity in enumerate(identities):
        row = by_identity[identity]
        output.append({
            "index": index, "row_id": row["row_id"], "pred_action": row["pred_action"],
            "pred_x": row["pred_x"], "pred_y": row["pred_y"], "parse_ok": row["parse_ok"],
            "element": float(row["bbox_dist"] == 0.0) if not math.isnan(row["bbox_dist"]) else 0.0,
            "success": row["success"], "gt_element_area": row["gt_element_area"],
        })
    return output


def m2w_cell_path(view: str):
    return W2_ROOT / "mind2web" / "tongui-7b" / view / "scored_rows.jsonl"


def summarize_m2w_cell(rows):
    if len(rows) != 2080 or [row["index"] for row in rows] != list(range(2080)):
        raise ValueError("W2 Mind2Web cell must contain ordered indices 0..2079")
    return {
        "rows": len(rows), "step_successes": sum(row["success"] for row in rows),
        "step_sr": sum(row["success"] for row in rows) / len(rows),
        "element_accuracy": sum(row["element"] for row in rows) / len(rows),
        "parse_rate": sum(row["parse_ok"] for row in rows) / len(rows),
    }


def paired_m2w_flips(full_rows, view_rows):
    action_flips = sum(full["pred_action"] != view["pred_action"] for full, view in zip(full_rows, view_rows))
    stable = [(full, view) for full, view in zip(full_rows, view_rows) if full["pred_action"] == view["pred_action"]]
    element_flips = sum(bool(full["element"]) != bool(view["element"]) for full, view in stable)
    bins = {}
    for name in ("tiny", "small", "regular"):
        members = [
            (full, view) for full, view in stable
            if area_bin(full["gt_element_area"]) == name
        ]
        flips = sum(bool(full["element"]) != bool(view["element"]) for full, view in members)
        bins[name] = {
            "rows": len(members), "flips": flips,
            "rate": flips / len(members) if members else None,
            "wilson_95": wilson(flips, len(members)),
        }
    return {
        "action_type": {
            "flips": action_flips, "denominator": len(full_rows),
            "rate": action_flips / len(full_rows), "wilson_95": wilson(action_flips, len(full_rows)),
        },
        "grounding_given_stable_type": {
            "stable_type_rows": len(stable), "flips": element_flips, "denominator": len(stable),
            "rate": element_flips / len(stable) if stable else None,
            "wilson_95": wilson(element_flips, len(stable)), "by_gt_element_area": bins,
        },
    }


def analyze_mind2web():
    full_rows = load_mind2web_full()
    full_summary = summarize_m2w_cell(full_rows)
    result = {
        "cells": {"tongui-7b/visual/full": {**full_summary, "source": "collision rows.parquet"}},
        "full_to_view_flips": {}, "k1": {}, "noise": {},
    }
    scores = [full_summary["step_sr"]]
    available = ["full"]
    for view in VIEWS[1:]:
        path = m2w_cell_path(view)
        if not path.exists():
            continue
        rows = read_jsonl(path)
        if len(rows) != 2080:
            continue
        result["cells"][f"tongui-7b/visual/{view}"] = {
            **summarize_m2w_cell(rows), "source": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
        }
        result["full_to_view_flips"][f"tongui-7b/visual/full_to_{view}"] = paired_m2w_flips(full_rows, rows)
        scores.append(result["cells"][f"tongui-7b/visual/{view}"]["step_sr"])
        available.append(view)
    noise = {"available_views": available, "complete": set(available) == set(VIEWS)}
    if noise["complete"]:
        noise.update({
            "mean_step_sr": float(np.mean(scores)),
            "sample_sd": float(np.std(scores, ddof=1)),
            "mde": float(2 * np.std(scores, ddof=1)),
        })
    result["noise"]["tongui-7b/visual"] = noise
    flip = result["full_to_view_flips"].get("tongui-7b/visual/full_to_v1")
    if flip:
        action_rate = flip["action_type"]["rate"]
        grounding_rate = flip["grounding_given_stable_type"]["rate"]
        result["k1"]["tongui-7b/visual"] = {
            "action_flip_rate": action_rate, "grounding_flip_rate": grounding_rate,
            "difference": grounding_rate - action_rate,
            "prediction_satisfied": grounding_rate > action_rate,
        }
    return result


def prediction_from_view_row(row: dict, source: str, bench: str) -> Prediction:
    if bench == "androidcontrol":
        width, height = row["image_size"]
        point = coordinate(row)
        x = row.get("pred_x")
        y = row.get("pred_y")
        if (x is None or y is None) and point is not None:
            x, y = point[0] / width, point[1] / height
        return Prediction(
            action=row.get("pred_action"), x=x, y=y,
            parameter=row.get("pred_input_text", ""), source=source,
            parse_ok=row.get("pred_action") is not None,
        )
    return Prediction(
        action=row.get("pred_action"), x=row.get("pred_x"), y=row.get("pred_y"),
        parameter=row.get("pred_param", ""), source=source,
        parse_ok=bool(row.get("parse_ok")),
    )


def p3_view_rows(bench: str, setting: str, view: str):
    if bench == "androidcontrol":
        path = ac_prediction_path(P3_REPRESENTATIVE[bench], setting, view)
        rows = read_jsonl(path)
        if len(rows) != 7708:
            return None
        return {row["index"]: row for row in rows}
    path = m2w_cell_path(view)
    rows = read_jsonl(path)
    if len(rows) != 2080:
        return None
    return {f"{row['annot_id']}__{row['action_uid']}": row for row in rows}


def build_p3_pool(bench: str, setting: str):
    identities, available_models, pivot = load_pool(bench, setting)
    models = P3_MODELS[bench]
    if not set(models).issubset(available_models):
        raise ValueError(f"P3 fixed model coverage mismatch: {bench}/{setting}")
    units = {}
    for model in models:
        key = f"{model}/full"
        units[key] = {}
        for row_id in identities:
            prediction = prediction_from_row(pivot[row_id][model])
            units[key][row_id] = Prediction(
                action=prediction.action, x=prediction.x, y=prediction.y,
                parameter=prediction.parameter, source=key, parse_ok=prediction.parse_ok,
            )
    representative = P3_REPRESENTATIVE[bench]
    for view in VIEWS[1:]:
        rows = p3_view_rows(bench, setting, view)
        if rows is None or not set(identities).issubset(rows):
            return None
        key = f"{representative}/{view}"
        units[key] = {
            row_id: prediction_from_view_row(rows[row_id], key, bench)
            for row_id in identities
        }
    return identities, pivot, units


def greedy_kappa_allocation(unit_failures, unit_step_sr, budget=5):
    keys = sorted(unit_failures)
    if budget > len(keys) or any(len(unit_failures[key]) == 0 for key in keys):
        raise ValueError("invalid P3 allocation pool")
    selected = [min(keys, key=lambda key: (-unit_step_sr[key], key))]
    while len(selected) < budget:
        candidates = [key for key in keys if key not in selected]
        selected.append(min(
            candidates,
            key=lambda key: (
                float(np.mean([
                    cohen_kappa(unit_failures[key], unit_failures[chosen])
                    for chosen in selected
                ])),
                -unit_step_sr[key],
                key,
            ),
        ))
    return selected


def evaluate_p3_pool(bench: str, setting: str, identities, pivot, units):
    pool = f"{bench}/{setting}"
    representative = P3_REPRESENTATIVE[bench]
    c1 = [f"{representative}/{view}" for view in VIEWS]
    c2 = [f"{model}/full" for model in P3_MODELS[bench]]
    candidate_keys = sorted(units)
    folds = []
    for test_fold in range(5):
        dev_ids, test_ids = split_rows(identities, pivot, fold_map(pool), test_fold)
        dev_success = {
            key: [
                score_prediction(next(iter(pivot[row_id].values())), units[key][row_id])
                for row_id in dev_ids
            ]
            for key in candidate_keys
        }
        dev_step_sr = {key: sum(values) / len(values) for key, values in dev_success.items()}
        failures = {key: [not value for value in values] for key, values in dev_success.items()}
        c3 = greedy_kappa_allocation(failures, dev_step_sr)
        rng = np.random.default_rng(np.random.SeedSequence([P3_SEED, test_fold]))
        c4 = sorted(rng.choice(candidate_keys, size=5, replace=False).tolist())
        selections = {"C1_single_model_five_views": c1, "C2_five_models_full": c2,
                      "C3_kappa_mixed": c3, "C4_random_mixed": c4}
        successes = {method: 0 for method in selections}
        for row_id in test_ids:
            reference = next(iter(pivot[row_id].values()))
            for method, selected in selections.items():
                predictions = [units[key][row_id] for key in selected]
                aggregate = (
                    plurality_then_density(bench, predictions, c1).prediction
                    if method == "C1_single_model_five_views"
                    else pka_medoid(bench, predictions).prediction
                )
                successes[method] += int(score_prediction(reference, aggregate))
        folds.append({
            "fold": test_fold, "dev_rows": len(dev_ids), "test_rows": len(test_ids),
            "selections": selections,
            "step_sr": {method: value / len(test_ids) for method, value in successes.items()},
        })
    total = sum(fold["test_rows"] for fold in folds)
    aggregate = {
        method: sum(fold["step_sr"][method] * fold["test_rows"] for fold in folds) / total
        for method in folds[0]["step_sr"]
    }
    return {"models": list(P3_MODELS[bench]), "representative": representative,
            "candidate_units": candidate_keys, "folds": folds, "aggregate_step_sr": aggregate}


def analyze_allocation():
    result = {
        "status": "PASS",
        "contract": {
            "budget": 5, "folds": "runs/complementarity/2026-07-30/folds.json",
            "c3_selection": "highest dev Step SR first; then minimum mean dev failure kappa; ties by dev Step SR then unit key",
            "c4_seed": P3_SEED, "test_label_tuning": False,
        },
        "pools": {},
    }
    for bench, setting in (("androidcontrol", "low"), ("androidcontrol", "high"), ("mind2web", "visual")):
        pool = build_p3_pool(bench, setting)
        if pool is None:
            return {"status": "PENDING_INFERENCE", "reason": "requires complete representative five-view prediction pools"}
        result["pools"][f"{bench}/{setting}"] = evaluate_p3_pool(bench, setting, *pool)
    for values in result["pools"].values():
        metrics = values["aggregate_step_sr"]
        values["p3_prediction_satisfied"] = (
            metrics["C3_kappa_mixed"] > max(
                metrics["C1_single_model_five_views"], metrics["C2_five_models_full"],
            )
        )
        values["c3_exceeds_random"] = metrics["C3_kappa_mixed"] > metrics["C4_random_mixed"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flips", type=Path, required=True)
    parser.add_argument("--noise", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    args = parser.parse_args()
    androidcontrol = analyze_androidcontrol()
    mind2web = analyze_mind2web()
    complete_v1 = (
        len(androidcontrol["k1"]) == len(AC_MODELS) * len(AC_SETTINGS)
        and len(mind2web["k1"]) == 1
    )
    complete_noise = (
        all(item["complete"] for item in androidcontrol["noise"].values())
        and all(item["complete"] for item in mind2web["noise"].values())
    )
    flips = {
        "status": "PASS" if complete_v1 else "PARTIAL",
        "contract": {
            "action_flip": "pred_action(view) != pred_action(full)",
            "grounding_flip": "GT grounding row; predicted type stable; grounding correctness changes",
            "quarantine_excluded": True,
        },
        "androidcontrol": androidcontrol,
        "mind2web": mind2web,
    }
    noise = {
        "status": "PASS" if complete_noise else "PARTIAL",
        "definition": "MDE = 2 * sample SD over full,v1,v2,v3,v4",
        "androidcontrol": androidcontrol["noise"],
        "mind2web": mind2web["noise"],
    }
    allocation = analyze_allocation()
    args.flips.write_text(json.dumps(flips, indent=2, sort_keys=True) + "\n")
    args.noise.write_text(json.dumps(noise, indent=2, sort_keys=True) + "\n")
    args.allocation.write_text(json.dumps(allocation, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"flips": flips["status"], "noise": noise["status"], "k1": androidcontrol["k1"]}, indent=2))


if __name__ == "__main__":
    main()