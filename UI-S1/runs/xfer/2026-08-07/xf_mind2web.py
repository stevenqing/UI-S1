import json
import math
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from xfer_common import aggregate


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
SEED = 20260807
RESAMPLES = 10000
ACTION_TO_ID = {"CLICK": 4, "SELECT": 2, "TYPE": 3}
MODEL_DIRS = {
    "TongUI-7B": "tongui",
    "CogAgent-18B": "cogagent",
    "UI-TARS-7B": "uitars",
}
ARMS = ("C_uni", "C_cond", "C_rand", "C_self")


def token_f1(prediction, reference):
    predicted = set(str(prediction).strip().split())
    expected = set(str(reference).strip().split())
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = len(predicted & expected)
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 0.0 if overlap == 0 else 2 * precision * recall / (precision + recall)


def score_prediction(row, prediction, image_size):
    if not prediction.get("parse_ok") or prediction.get("position") is None:
        return False
    action = prediction.get("action")
    if action not in ACTION_TO_ID:
        return False
    width, height = image_size
    bbox = row["step"]["bbox"]
    x, y = prediction["position"]
    element = (
        bbox["x"] / width <= x <= (bbox["x"] + bbox["width"]) / width
        and bbox["y"] / height <= y <= (bbox["y"] + bbox["height"]) / height
    )
    predicted_operation = str(ACTION_TO_ID[action])
    if action in {"TYPE", "SELECT"}:
        if not prediction.get("value"):
            return False
        predicted_operation += " " + prediction["value"].lower()
    reference_action = row["step"]["operation"]["op"]
    reference_operation = str(ACTION_TO_ID[reference_action])
    if reference_action in {"TYPE", "SELECT"}:
        reference_operation += " " + row["step"]["operation"]["value"].lower()
    return bool(element and token_f1(predicted_operation, reference_operation) == 1.0)


def micro(success_by_id):
    return float(np.mean(list(success_by_id.values())))


def episode_macro(rows_by_id, success_by_id):
    episodes = defaultdict(list)
    for row_id, success in success_by_id.items():
        episodes[rows_by_id[row_id]["episode_id"]].append(float(success))
    return float(np.mean([np.mean(values) for values in episodes.values()]))


def paired_episode_bootstrap(rows_by_id, fold_for_website, left, right, resamples=RESAMPLES, seed=SEED):
    by_fold_episode = defaultdict(lambda: defaultdict(list))
    for row_id, row in rows_by_id.items():
        fold = fold_for_website[row["website"]]
        by_fold_episode[fold][row["episode_id"]].append(row_id)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(resamples):
        selected = []
        for fold in sorted(by_fold_episode):
            episodes = sorted(by_fold_episode[fold])
            for episode in rng.choice(episodes, size=len(episodes), replace=True):
                selected.extend(by_fold_episode[fold][episode])
        samples.append(float(np.mean([int(left[row_id]) - int(right[row_id]) for row_id in selected])))
    point = float(np.mean([int(left[row_id]) - int(right[row_id]) for row_id in rows_by_id]))
    return {
        "point_delta": point,
        "ci_99": [float(np.quantile(samples, 0.005)), float(np.quantile(samples, 0.995))],
        "p_delta_le_zero_plus_one": float((1 + sum(value <= 0 for value in samples)) / (resamples + 1)),
        "resamples": resamples,
        "seed": seed,
        "unit": "episode_stratified_by_website_fold",
    }


def summarize_candidate_slots(rows_by_id, slot_success):
    rates = {
        slot: float(np.mean([int(value) for value in values.values()]))
        for slot, values in slot_success.items()
    }
    best_slot = max(rates, key=lambda slot: (rates[slot], slot))
    pool_values = [
        int(value)
        for values in slot_success.values()
        for value in values.values()
    ]
    return {
        "best_single": {"slot": best_slot, "step_sr": rates[best_slot]},
        "pool_mean_step_sr": float(np.mean(pool_values)),
        "slot_step_sr": dict(sorted(rates.items())),
        "slot_coverage": {slot: len(values) / len(rows_by_id) for slot, values in sorted(slot_success.items())},
    }


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_unique(directory, expected_rows=2080):
    rows = {}
    for path in sorted(directory.glob("shard-*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in rows:
                raise ValueError(f"duplicate row: {directory}/{row['id']}")
            rows[row["id"]] = row
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows in {directory}, found {len(rows)}")
    return rows


def crop_prediction(row, set_name, crop_index=0):
    return row["predictions"][set_name][crop_index]["prediction"]


def model_candidate(prediction, model, view_index, stage):
    return {**prediction, "model": model, "view_index": view_index, "stage": stage}


def evaluate_predictions(rows_by_id, predictions, image_sizes):
    successes = {
        row_id: score_prediction(rows_by_id[row_id], prediction, image_sizes[row_id])
        for row_id, prediction in predictions.items()
    }
    return {
        "micro_step_sr": micro(successes),
        "episode_macro_step_sr": episode_macro(rows_by_id, successes),
        "successes": successes,
    }


def aggregate_candidates(rows_by_id, candidates_by_id, model_order, image_sizes):
    return {
        row_id: aggregate(candidates_by_id[row_id], model_order, image_sizes[row_id])
        for row_id in rows_by_id
    }


def build_stage1(rows_by_id, full_lanes, view1_lanes):
    output = {}
    for row_id in rows_by_id:
        output[row_id] = []
        for view_index in (0, 1):
            for model in full_lanes:
                prediction = (
                    full_lanes[model][row_id]["prediction"]
                    if view_index == 0
                    else crop_prediction(view1_lanes[model][row_id], "view1")
                )
                output[row_id].append(model_candidate(prediction, model, view_index, "stage1"))
    return output


def evaluate_arms(rows_by_id, consensus, stage1, stage2, model_order, image_sizes):
    evaluations = {}
    candidate_summaries = {}
    for arm in ARMS:
        candidates_by_id = {}
        slot_success = defaultdict(dict)
        executed = 0
        for row_id, row in rows_by_id.items():
            source = consensus[row_id]
            candidates = list(stage1[row_id])
            for index, candidate in enumerate(candidates):
                slot = f"stage1_{candidate['model']}_view{candidate['view_index']}"
                slot_success[slot][row_id] = score_prediction(row, candidate, image_sizes[row_id])
                executed += 1
            if source["stage2_trigger"]:
                for crop_index in range(2):
                    for model in model_order:
                        prediction = crop_prediction(stage2[model][row_id], arm, crop_index)
                        candidate = model_candidate(prediction, model, crop_index, "stage2")
                        candidates.append(candidate)
                        slot = f"stage2_{model}_crop{crop_index}"
                        slot_success[slot][row_id] = score_prediction(row, candidate, image_sizes[row_id])
                        executed += 1
            candidates_by_id[row_id] = candidates
        predictions = aggregate_candidates(rows_by_id, candidates_by_id, model_order, image_sizes)
        evaluation = evaluate_predictions(rows_by_id, predictions, image_sizes)
        triggered_ids = [row_id for row_id in rows_by_id if consensus[row_id]["stage2_trigger"]]
        nontriggered_ids = [row_id for row_id in rows_by_id if not consensus[row_id]["stage2_trigger"]]
        evaluation.update({
            "triggered_rows": len(triggered_ids),
            "nontriggered_rows": len(nontriggered_ids),
            "triggered_micro_step_sr": float(np.mean([evaluation["successes"][row_id] for row_id in triggered_ids])) if triggered_ids else None,
            "nontriggered_micro_step_sr": float(np.mean([evaluation["successes"][row_id] for row_id in nontriggered_ids])) if nontriggered_ids else None,
            "mean_forwards": executed / len(rows_by_id),
        })
        evaluations[arm] = evaluation
        candidate_summaries[arm] = summarize_candidate_slots(rows_by_id, slot_success)
    return evaluations, candidate_summaries


def view_candidate(full_lanes, view1_lanes, tongui_views, shared_views, model, view_index, row_id):
    if view_index == 0:
        return model_candidate(full_lanes[model][row_id]["prediction"], model, view_index, "curve")
    if view_index == 1:
        return model_candidate(crop_prediction(view1_lanes[model][row_id], "view1"), model, view_index, "curve")
    lane = tongui_views if model == "TongUI-7B" else shared_views[model]
    return model_candidate(crop_prediction(lane[row_id], f"view{view_index}"), model, view_index, "curve")


def evaluate_curves(rows_by_id, full_lanes, view1_lanes, tongui_views, shared_views, model_order, image_sizes):
    curve_config = yaml.safe_load((RUN_DIR / "configs/curves.yaml").read_text())
    if curve_config["status"] != "RESULT_BLIND_BEFORE_CURVE_SCORING":
        raise ValueError("curve protocol is not frozen")
    v_sequence = [("TongUI-7B", view) for view in range(16)]
    mixed_sequence = [(model, view) for view in range(4) for model in model_order] + [("TongUI-7B", view) for view in range(4, 8)]
    reports = {"v_only": {}, "mixed": {}}
    for name, sequence in (("v_only", v_sequence), ("mixed", mixed_sequence)):
        for budget in curve_config["budgets"]:
            candidates_by_id = {
                row_id: [
                    view_candidate(full_lanes, view1_lanes, tongui_views, shared_views, model, view, row_id)
                    for model, view in sequence[:budget]
                ]
                for row_id in rows_by_id
            }
            predictions = aggregate_candidates(rows_by_id, candidates_by_id, model_order, image_sizes)
            reports[name][str(budget)] = evaluate_predictions(rows_by_id, predictions, image_sizes)
    return reports


def strip_successes(value):
    if isinstance(value, dict):
        return {key: strip_successes(item) for key, item in value.items() if key != "successes"}
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--tongui-views", type=Path, required=True)
    parser.add_argument("--shared-views-root", type=Path, required=True)
    parser.add_argument("--mde", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roster = yaml.safe_load((RUN_DIR / "configs/xfer_roster.yaml").read_text())
    model_order = [model["id"] for model in roster["mind2web"]["models"]]
    rows = [json.loads(line) for line in (RUN_DIR / "data/mind2web/mind2web_test_task.jsonl").read_text().splitlines() if line.strip()]
    rows_by_id = {row["id"]: row for row in rows}
    image_sizes = {row["id"]: Image.open(ROOT / row["image"]).size for row in rows}
    full_lanes = {model: load_unique(RUN_DIR / "raw/stage1" / directory) for model, directory in MODEL_DIRS.items()}
    view1_lanes = {model: load_unique(RUN_DIR / "raw/stage1/view1" / directory) for model, directory in MODEL_DIRS.items()}
    consensus_rows = load_unique(RUN_DIR / "raw", expected_rows=0) if False else {
        row["id"]: row for row in map(json.loads, (RUN_DIR / "raw/mind2web-consensus-roi.jsonl").read_text().splitlines())
    }
    if len(consensus_rows) != 2080:
        raise ValueError("consensus row coverage mismatch")
    stage2 = {model: load_unique(args.stage2_root / directory) for model, directory in MODEL_DIRS.items()}
    for row_id, source in consensus_rows.items():
        for model in model_order:
            row = stage2[model][row_id]
            if row["source_hashes"] != {arm: source["arms_sha256"] for arm in ARMS}:
                raise ValueError(f"stage2 arm provenance mismatch: {model}/{row_id}")
            if canonical_hash(row["predictions"]) != row["predictions_sha256"]:
                raise ValueError(f"stage2 prediction hash mismatch: {model}/{row_id}")
    stage1 = build_stage1(rows_by_id, full_lanes, view1_lanes)
    evaluations, candidate_summaries = evaluate_arms(rows_by_id, consensus_rows, stage1, stage2, model_order, image_sizes)
    fold_for_website = json.loads((ROOT / "runs/complementarity/2026-07-30/folds.json").read_text())["pools"]["mind2web/visual"]["group_to_fold"]
    comparisons = {
        reference: paired_episode_bootstrap(
            rows_by_id, fold_for_website,
            evaluations["C_cond"]["successes"], evaluations[reference]["successes"],
        )
        for reference in ("C_uni", "C_rand", "C_self")
    }
    tongui_views = load_unique(args.tongui_views)
    shared_views = {
        model: load_unique(args.shared_views_root / MODEL_DIRS[model])
        for model in model_order if model != "TongUI-7B"
    }
    curves = evaluate_curves(rows_by_id, full_lanes, view1_lanes, tongui_views, shared_views, model_order, image_sizes)
    curve_deltas = {
        pool: paired_episode_bootstrap(
            rows_by_id, fold_for_website,
            values["16"]["successes"], values["4"]["successes"],
        )
        for pool, values in curves.items()
    }
    full_single = {
        model: evaluate_predictions(
            rows_by_id,
            {row_id: full_lanes[model][row_id]["prediction"] for row_id in rows_by_id},
            image_sizes,
        )
        for model in model_order
    }
    mde = json.loads(args.mde.read_text())
    xf1 = comparisons["C_uni"]["point_delta"] > mde["micro_mde"] and comparisons["C_uni"]["ci_99"][0] > 0
    xf2 = comparisons["C_rand"]["ci_99"][0] > 0 and comparisons["C_self"]["ci_99"][0] > 0
    xf4 = curve_deltas["v_only"]["ci_99"][1] < 0 and curve_deltas["mixed"]["ci_99"][0] > 0
    result = {
        "schema_version": 1,
        "status": "PASS",
        "rows": len(rows),
        "episodes": len({row["episode_id"] for row in rows}),
        "evaluations": strip_successes(evaluations),
        "comparisons": comparisons,
        "candidate_summaries": candidate_summaries,
        "full_image_single_models": strip_successes(full_single),
        "curves": strip_successes(curves),
        "paired_N16_minus_N4": curve_deltas,
        "mde": mde,
        "XF1": xf1,
        "XF2": xf2,
        "XF4": xf4,
        "XF_K1": evaluations["C_cond"]["micro_step_sr"] <= evaluations["C_uni"]["micro_step_sr"],
        "XF_K2": evaluations["C_cond"]["micro_step_sr"] > evaluations["C_uni"]["micro_step_sr"] and evaluations["C_cond"]["micro_step_sr"] <= evaluations["C_self"]["micro_step_sr"],
        "XF_K3": evaluations["C_cond"]["triggered_rows"] / len(rows) < 0.60,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("evaluations", "comparisons", "XF1", "XF2", "XF4", "XF_K1", "XF_K2", "XF_K3")}, indent=2, sort_keys=True))


def test_contracts():
    base = {
        "episode_id": "episode-a",
        "website": "site-a",
        "step": {
            "bbox": {"x": 100.0, "y": 50.0, "width": 100.0, "height": 100.0},
            "operation": {"op": "TYPE", "value": "new york"},
        },
    }
    exact = {"action": "TYPE", "value": "york new", "position": [0.15, 0.125], "parse_ok": True}
    partial = {"action": "TYPE", "value": "new", "position": [0.15, 0.125], "parse_ok": True}
    wrong_point = {"action": "TYPE", "value": "new york", "position": [0.9, 0.9], "parse_ok": True}
    assert score_prediction(base, exact, (1000, 800))
    assert not score_prediction(base, partial, (1000, 800))
    assert not score_prediction(base, wrong_point, (1000, 800))
    rows = {
        "a": base,
        "b": {**base, "episode_id": "episode-b"},
    }
    left, right = {"a": True, "b": False}, {"a": False, "b": False}
    bootstrap = paired_episode_bootstrap(rows, {"site-a": 0}, left, right, resamples=100, seed=1)
    assert math.isclose(bootstrap["point_delta"], 0.5)
    summary = summarize_candidate_slots(rows, {"m0": left, "m1": right})
    assert summary["best_single"]["slot"] == "m0"
    assert math.isclose(summary["pool_mean_step_sr"], 0.25)


if __name__ == "__main__":
    test_contracts()
    if len(__import__("sys").argv) > 1:
        main()
    else:
        print(json.dumps({"status": "PASS"}))