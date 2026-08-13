import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CLOSE_DIR = ROOT / "runs/close/2026-08-08"
CONSOLIDATE_DIR = ROOT / "runs/consolidate/2026-08-06"
CONFIG_PATH = RUN_DIR / "configs/gran_prereg.yaml"
OUTPUT_PATH = RUN_DIR / "TAU_SWEEP.json"
sys.path.insert(0, str(RUN_DIR))

from gran_common import (
    GranCandidate, attach_reliability, density_select, mechanism_values,
    prior_select, source_reliability, tau_label, tau_options,
)


ARMS = ("C_uni", "C_cond", "C_rand", "C_self")
SCREEN_MODELS = ("GTA1-7B", "Qwen3-VL-8B-Instruct", "UI-TARS-7B-SFT")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_contract():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if config.get("status") != "FROZEN_BEFORE_CLICK_COUNT_ANCHOR_AND_ANY_TAU_SWEEP":
        raise PermissionError("GRAN prereg status mismatch")
    anchors = json.loads((RUN_DIR / "ANCHORS.json").read_text())
    click = json.loads((RUN_DIR / "CLICK_SCOPE.json").read_text())
    manifest = json.loads((RUN_DIR / "INPUT_MANIFEST.json").read_text())
    if (
        anchors.get("status") != "PASS_GRAN_IMPLEMENTATION_ANCHORS"
        or anchors.get("tau_sweep_started") is not False
        or click.get("selected_strata") != 4
        or click.get("tau_sweep_started") is not False
        or manifest.get("status") != "LOCKED_BEFORE_GRAN_LABEL_STATISTICS_AND_TAU_SWEEP"
    ):
        raise PermissionError("GRAN pre-sweep gates are incomplete")
    return config


def load_screen_rows(e1):
    common = load_module(CONSOLIDATE_DIR / "common.py", "gran_screen_common")
    context = common.load_context()
    rows = {}
    for row_id in context["row_ids"]:
        width, height = context["metadata"][row_id]["img_size"]
        diagonal = math.hypot(width, height)
        target = context["metadata"][row_id]["target_bbox"]
        candidates = []
        for order, (model, view) in enumerate(
            (model, view) for view in range(12) for model in SCREEN_MODELS
        ):
            value = context["bank"][(model, view)][row_id]
            point = tuple(float(coordinate) for coordinate in value["point"])
            candidates.append(GranCandidate(
                source=f"{model}/view{view}",
                lineage=model,
                action="POINT",
                coordinate=(point[0] / diagonal, point[1] / diagonal),
                parameter="",
                parse_ok=True,
                order=order,
                correct=bool(e1.point_in_bbox(point, target)),
            ))
        rows[row_id] = {
            "fold": int(context["fold_for_group"][context["metadata"][row_id]["application"]]),
            "group": str(context["metadata"][row_id]["application"]),
            "candidates": tuple(candidates),
        }
    if len(rows) != 1581 or any(len(row["candidates"]) != 36 for row in rows.values()):
        raise ValueError("GRAN ScreenSpot full-bank mismatch")
    return rows


def load_mind_rows(e1):
    task_path = e1.XFER / "data/mind2web/mind2web_test_task.jsonl"
    task_rows = [json.loads(line) for line in task_path.read_text().splitlines() if line.strip()]
    rows_by_id = {str(row["id"]): row for row in task_rows}
    image_sizes = {str(row["id"]): Image.open(ROOT / row["image"]).size for row in task_rows}
    full = {model: e1.load_unique(e1.XFER / "raw/stage1" / directory) for model, directory in e1.MODEL_DIRS.items()}
    view1 = {model: e1.load_unique(e1.XFER / "raw/stage1/view1" / directory) for model, directory in e1.MODEL_DIRS.items()}
    stage2 = {model: e1.load_unique(e1.XFER / "raw/stage2" / directory) for model, directory in e1.MODEL_DIRS.items()}
    fold_map = json.loads((ROOT / "runs/complementarity/2026-07-30/folds.json").read_text())["pools"]["mind2web/visual"]["group_to_fold"]
    output = {arm: {} for arm in ARMS}
    for arm in ARMS:
        slots = e1.mind_slots(rows_by_id, full, view1, stage2, arm)
        for row_id, row in rows_by_id.items():
            if row["step"]["operation"]["op"] != "CLICK":
                continue
            width, height = image_sizes[row_id]
            diagonal = math.hypot(width, height)
            candidates = []
            for order, (source, lineage, prediction) in enumerate(slots[row_id]):
                position = prediction.get("position")
                coordinate = (
                    (float(position[0]) * width / diagonal, float(position[1]) * height / diagonal)
                    if position is not None else None
                )
                candidates.append(GranCandidate(
                    source=str(source),
                    lineage=str(lineage),
                    action=str(prediction.get("action") or ""),
                    coordinate=coordinate,
                    parameter=str(prediction.get("value") or ""),
                    parse_ok=bool(prediction.get("parse_ok")),
                    order=order,
                    correct=bool(e1.score_prediction(row, prediction, (width, height))),
                ))
            output[arm][row_id] = {
                "fold": int(fold_map[row["website"]]),
                "group": str(row["episode_id"]),
                "candidates": tuple(candidates),
            }
    counts = {arm: len(rows) for arm, rows in output.items()}
    if set(counts.values()) != {1774} or any(
        len(row["candidates"]) != 12 for rows in output.values() for row in rows.values()
    ):
        raise ValueError(f"GRAN Mind2Web CLICK bank mismatch: {counts}")
    return output


def evaluate(rows, row_ids, reliability, benchmark, option):
    outputs = {}
    priors = {}
    details = {}
    for row_id in row_ids:
        candidates = attach_reliability(rows[row_id]["candidates"], reliability)
        prior = prior_select(candidates)
        selected, selected_details = density_select(
            candidates, benchmark, option[0], option[1]
        )
        outputs[row_id] = bool(selected is not None and selected.correct)
        priors[row_id] = bool(prior is not None and prior.correct)
        details[row_id] = selected_details
    return outputs, priors, details


def choose_tau(rows, train_ids, validation_ids, benchmark, options):
    reliability = source_reliability(rows, train_ids)
    scores = []
    for index, option in enumerate(options):
        outputs, _, _ = evaluate(rows, validation_ids, reliability, benchmark, option)
        scores.append({
            "index": index,
            "tau": tau_label(option),
            "accuracy": float(np.mean(list(outputs.values()))),
        })
    selected = max(scores, key=lambda row: (row["accuracy"], -row["index"]))
    return options[selected["index"]], scores


def nested_primary(rows, benchmark, options):
    row_ids = tuple(sorted(rows))
    folds = []
    outputs = {}
    prior_outputs = {}
    mechanisms = {}
    for outer_fold in range(5):
        inner_validation_fold = (outer_fold + 1) % 5
        inner_train = [row_id for row_id in row_ids if rows[row_id]["fold"] not in {outer_fold, inner_validation_fold}]
        inner_validation = [row_id for row_id in row_ids if rows[row_id]["fold"] == inner_validation_fold]
        outer_development = [row_id for row_id in row_ids if rows[row_id]["fold"] != outer_fold]
        outer_test = [row_id for row_id in row_ids if rows[row_id]["fold"] == outer_fold]
        selected, scores = choose_tau(
            rows, inner_train, inner_validation, benchmark, options
        )
        reliability = source_reliability(rows, outer_development)
        fold_outputs, fold_prior, _ = evaluate(
            rows, outer_test, reliability, benchmark, selected
        )
        outputs.update(fold_outputs)
        prior_outputs.update(fold_prior)
        for row_id in outer_test:
            candidates = attach_reliability(rows[row_id]["candidates"], reliability)
            mechanisms[row_id] = {
                **mechanism_values(candidates, benchmark, selected[0], selected[1]),
                "density_correct": fold_outputs[row_id],
                "prior_correct": fold_prior[row_id],
                "margin": int(fold_outputs[row_id]) - int(fold_prior[row_id]),
                "fold": outer_fold,
                "group": rows[row_id]["group"],
                "selected_tau": tau_label(selected),
            }
        finite_grid = len(options) - 2
        selected_index = options.index(selected)
        folds.append({
            "outer_fold": outer_fold,
            "inner_validation_fold": inner_validation_fold,
            "inner_train_rows": len(inner_train),
            "inner_validation_rows": len(inner_validation),
            "outer_test_rows": len(outer_test),
            "selected_tau": tau_label(selected),
            "selected_index": selected_index,
            "finite_boundary_selected": selected_index in {1, finite_grid},
            "validation_scores": scores,
        })
    if set(outputs) != set(row_ids) or set(prior_outputs) != set(row_ids):
        raise ValueError("GRAN nested output coverage mismatch")
    return {
        "rows": len(row_ids),
        "accuracy": float(np.mean(list(outputs.values()))),
        "prior_accuracy": float(np.mean(list(prior_outputs.values()))),
        "point_margin": float(np.mean([
            int(outputs[row_id]) - int(prior_outputs[row_id]) for row_id in row_ids
        ])),
        "outputs": outputs,
        "prior_outputs": prior_outputs,
        "mechanisms": mechanisms,
        "folds": folds,
    }


def apply_primary_tau_to_arm(primary, rows, benchmark, options):
    row_ids = tuple(sorted(rows))
    outputs = {}
    prior_outputs = {}
    folds = []
    for fold_record in primary["folds"]:
        outer_fold = fold_record["outer_fold"]
        selected = options[fold_record["selected_index"]]
        development = [row_id for row_id in row_ids if rows[row_id]["fold"] != outer_fold]
        test = [row_id for row_id in row_ids if rows[row_id]["fold"] == outer_fold]
        reliability = source_reliability(rows, development)
        current, prior, _ = evaluate(rows, test, reliability, benchmark, selected)
        outputs.update(current)
        prior_outputs.update(prior)
        folds.append({
            "outer_fold": outer_fold,
            "selected_tau_from_primary": tau_label(selected),
            "test_rows": len(test),
        })
    return {
        "rows": len(row_ids),
        "accuracy": float(np.mean(list(outputs.values()))),
        "prior_accuracy": float(np.mean(list(prior_outputs.values()))),
        "point_margin": float(np.mean([
            int(outputs[row_id]) - int(prior_outputs[row_id]) for row_id in row_ids
        ])),
        "folds": folds,
    }


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    config = load_contract()
    e1 = load_module(CLOSE_DIR / "e1_arm_aggregator_matrix.py", "gran_tau_e1")
    options = tau_options(config["tau"]["finite_grid"]["values"])
    screen_rows = load_screen_rows(e1)
    mind_rows = load_mind_rows(e1)
    screen = nested_primary(screen_rows, "screenspot_pro", options)
    mind_primary = nested_primary(mind_rows["C_uni"], "mind2web", options)
    mind = {
        "C_uni": {key: value for key, value in mind_primary.items() if key != "outputs" and key != "prior_outputs"},
    }
    for arm in ARMS[1:]:
        mind[arm] = apply_primary_tau_to_arm(
            mind_primary, mind_rows[arm], "mind2web", options
        )
    result = {
        "schema_version": 1,
        "status": "PASS_GRAN_NESTED_TAU_SWEEP",
        "zero_gpu": True,
        "tau_options": [tau_label(option) for option in options],
        "screenspot_pro": {
            key: value for key, value in screen.items()
            if key not in {"outputs", "prior_outputs"}
        },
        "mind2web": mind,
        "kill_conditions": {
            "G_K6_screenspot_finite_boundary": any(
                fold["finite_boundary_selected"] for fold in screen["folds"]
            ),
            "G_K6_mind2web_finite_boundary": any(
                fold["finite_boundary_selected"] for fold in mind_primary["folds"]
            ),
        },
        "claim_boundary": {
            "explanatory_only": True,
            "runtime_selector_allowed": False,
            "p_hat_is_label_dependent": True,
        },
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "screenspot": {
            "accuracy": screen["accuracy"],
            "prior_accuracy": screen["prior_accuracy"],
            "margin": screen["point_margin"],
            "selected_tau": [fold["selected_tau"] for fold in screen["folds"]],
        },
        "mind2web": {
            arm: {
                "accuracy": value["accuracy"],
                "prior_accuracy": value["prior_accuracy"],
                "margin": value["point_margin"],
            }
            for arm, value in mind.items()
        },
        "kill_conditions": result["kill_conditions"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()