import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CLOSE = ROOT / "runs/close/2026-08-08"
AGGMATCH = ROOT / "runs/aggmatch/2026-08-09"
CONFIG_PATH = RUN_DIR / "configs/cev_prereg.yaml"
sys.path.insert(0, str(RUN_DIR))
sys.path.insert(0, str(AGGMATCH))

from aggmatch_common import atomic_json, paired_bootstrap, sha256_file
from cev import Candidate, select


ARMS = ("C_uni", "C_cond", "C_rand", "C_self")
E1_METHODS = ("majority", "A0", "ours", "A1", "A2", "A3", "A4")
LINKAGE = "complete"
LINEAGE_VOTE_CAP = None


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configuration_key(configuration):
    return (
        configuration["granularity"],
        configuration.get("coordinate_multiplier"),
        configuration.get("parameter_threshold"),
    )


def configuration_label(configuration):
    values = [configuration["granularity"]]
    if "coordinate_multiplier" in configuration:
        values.append(f"c{configuration['coordinate_multiplier']}")
    if "parameter_threshold" in configuration:
        values.append(f"p{configuration['parameter_threshold']}")
    return "_".join(values)


def configurations(config):
    output = [{"granularity": "G0"}]
    multipliers = config["benchmarks"]["mind2web"]["coordinate_multipliers"]
    thresholds = config["benchmarks"]["mind2web"]["parameter_thresholds"]
    output.extend({"granularity": "G1", "coordinate_multiplier": value} for value in multipliers)
    output.extend({"granularity": "G2", "parameter_threshold": value} for value in thresholds)
    output.extend(
        {"granularity": "G3", "coordinate_multiplier": multiplier, "parameter_threshold": threshold}
        for multiplier in multipliers for threshold in thresholds
    )
    return output


def config_rank(configuration, config):
    granularity = config["cev_a"]["granularity_tie_order"].index(configuration["granularity"])
    threshold = configuration.get("parameter_threshold")
    threshold_rank = config["cev_a"]["parameter_threshold_tie_order"].index(threshold) if threshold is not None else -1
    multiplier = configuration.get("coordinate_multiplier")
    multiplier_rank = config["cev_a"]["coordinate_multiplier_tie_order"].index(multiplier) if multiplier is not None else -1
    return granularity, threshold_rank, multiplier_rank


def choose_configuration(scores, by_key, config):
    return min(scores, key=lambda key: (-scores[key], config_rank(by_key[key], config)))


def mind_scale(row_ids, rows_by_id, image_sizes):
    widths = []
    heights = []
    for row_id in row_ids:
        bbox = rows_by_id[row_id]["step"]["bbox"]
        width, height = image_sizes[row_id]
        widths.append(bbox["width"] / width)
        heights.append(bbox["height"] / height)
    return float(np.median(widths)), float(np.median(heights))


def mind_candidates(e1, slots, reliability, row_id):
    output = []
    for order, (slot, model, prediction) in enumerate(slots[row_id]):
        position = prediction.get("position")
        output.append(Candidate(
            action=str(prediction.get("action") or ""),
            coordinate=tuple(position) if position is not None else None,
            parameter=str(prediction.get("value") or ""),
            source=slot,
            reliability=reliability.get(slot, 0.0),
            order=order,
            payload=prediction,
            parse_ok=bool(prediction.get("parse_ok")),
            lineage=model,
        ))
    return output


def mind_prediction(candidate):
    if candidate is None:
        return {"action": None, "value": None, "position": None, "parse_ok": False}
    return {
        "action": candidate.action,
        "value": candidate.parameter or None,
        "position": list(candidate.coordinate) if candidate.coordinate is not None else None,
        "parse_ok": candidate.parse_ok,
    }


def threshold_for(configuration, scale):
    multiplier = configuration.get("coordinate_multiplier", 1.0)
    return scale[0] * multiplier, scale[1] * multiplier


def apply_mind_configuration(candidates, configuration, scale, restrict_action=None):
    selected_candidates = candidates if restrict_action is None else [candidate for candidate in candidates if candidate.action == restrict_action]
    prediction, details = select(
        selected_candidates,
        configuration["granularity"],
        threshold_for(configuration, scale),
        configuration.get("parameter_threshold", 1.0),
        linkage=LINKAGE,
        lineage_vote_cap=LINEAGE_VOTE_CAP,
    )
    return prediction, details


def predicted_plurality_action(candidates, scale):
    prediction, _ = apply_mind_configuration(candidates, {"granularity": "G0"}, scale)
    return prediction.action if prediction is not None else ""


def score_mind(e1, row, image_size, prediction):
    return bool(e1.score_prediction(row, mind_prediction(prediction), image_size))


def evaluate_mind_config(e1, row_ids, slots, reliability, rows_by_id, image_sizes, configuration, scale, restrict_by_plurality=False):
    outputs = {}
    actions = {}
    for row_id in row_ids:
        candidates = mind_candidates(e1, slots, reliability, row_id)
        action = predicted_plurality_action(candidates, scale)
        prediction, _ = apply_mind_configuration(candidates, configuration, scale, action if restrict_by_plurality else None)
        outputs[row_id] = score_mind(e1, rows_by_id[row_id], image_sizes[row_id], prediction)
        actions[row_id] = action
    return outputs, actions


def fit_mind_action_configs(e1, val_ids, slots, reliability, rows_by_id, image_sizes, options, scale, global_configuration, config):
    action_rows = {}
    row_candidates = {}
    for row_id in val_ids:
        candidates = mind_candidates(e1, slots, reliability, row_id)
        row_candidates[row_id] = candidates
        action = predicted_plurality_action(candidates, scale)
        action_rows.setdefault(action, []).append(row_id)
    selected = {}
    validation_outputs = {}
    minimum = config["cev_a"]["action_min_inner_validation_rows"]
    for action, ids in action_rows.items():
        if len(ids) < minimum:
            selected[action] = {**global_configuration, "backoff": True, "validation_rows": len(ids)}
            choice = global_configuration
        else:
            scores = {}
            for option in options:
                successes = []
                for row_id in ids:
                    prediction, _ = apply_mind_configuration(row_candidates[row_id], option, scale, action)
                    successes.append(score_mind(e1, rows_by_id[row_id], image_sizes[row_id], prediction))
                scores[configuration_key(option)] = float(np.mean(successes))
            by_key = {configuration_key(option): option for option in options}
            chosen_key = choose_configuration(scores, by_key, config)
            choice = by_key[chosen_key]
            selected[action] = {**choice, "backoff": False, "validation_rows": len(ids), "validation_accuracy": scores[chosen_key]}
        for row_id in ids:
            prediction, _ = apply_mind_configuration(row_candidates[row_id], choice, scale, action)
            validation_outputs[row_id] = score_mind(e1, rows_by_id[row_id], image_sizes[row_id], prediction)
    return selected, validation_outputs


def evaluate_mind_action_strategy(e1, row_ids, slots, reliability, rows_by_id, image_sizes, action_configs, global_configuration, scale):
    outputs = {}
    selected_actions = {}
    for row_id in row_ids:
        candidates = mind_candidates(e1, slots, reliability, row_id)
        action = predicted_plurality_action(candidates, scale)
        configuration = action_configs.get(action, global_configuration)
        prediction, _ = apply_mind_configuration(candidates, configuration, scale, action)
        outputs[row_id] = score_mind(e1, rows_by_id[row_id], image_sizes[row_id], prediction)
        selected_actions[row_id] = action
    return outputs, selected_actions


def nested_mind(e1, config, options_override=None, arms=ARMS):
    rows = [json.loads(line) for line in (e1.XFER / "data/mind2web/mind2web_test_task.jsonl").read_text().splitlines() if line.strip()]
    rows_by_id = {row["id"]: row for row in rows}
    image_sizes = {row["id"]: Image.open(ROOT / row["image"]).size for row in rows}
    full = {model: e1.load_unique(e1.XFER / "raw/stage1" / directory) for model, directory in e1.MODEL_DIRS.items()}
    view1 = {model: e1.load_unique(e1.XFER / "raw/stage1/view1" / directory) for model, directory in e1.MODEL_DIRS.items()}
    stage2 = {model: e1.load_unique(e1.XFER / "raw/stage2" / directory) for model, directory in e1.MODEL_DIRS.items()}
    fold_map = json.loads((ROOT / "runs/complementarity/2026-07-30/folds.json").read_text())["pools"]["mind2web/visual"]["group_to_fold"]
    fold_for_id = {row_id: fold_map[row["website"]] for row_id, row in rows_by_id.items()}
    options = configurations(config) if options_override is None else options_override
    outputs = {arm: {"global": {}, "action": {}, "CEV_A": {}, "dev_selection": {}} for arm in arms}
    records = []
    for outer_fold in range(5):
        inner_val_fold = (outer_fold + 1) % 5
        inner_train_ids = [row_id for row_id in rows_by_id if fold_for_id[row_id] not in {outer_fold, inner_val_fold}]
        inner_val_ids = [row_id for row_id in rows_by_id if fold_for_id[row_id] == inner_val_fold]
        outer_dev_ids = [row_id for row_id in rows_by_id if fold_for_id[row_id] != outer_fold]
        test_ids = [row_id for row_id in rows_by_id if fold_for_id[row_id] == outer_fold]
        fold_record = {"outer_fold": outer_fold, "inner_validation_fold": inner_val_fold, "arms": {}}
        for arm in arms:
            slots = e1.mind_slots(rows_by_id, full, view1, stage2, arm)
            inner_priority, inner_weights = e1.dev_mind_statistics(inner_train_ids, slots, rows_by_id, image_sizes)
            inner_reliability = {slot: float(np.mean([e1.score_prediction(rows_by_id[row_id], prediction, image_sizes[row_id]) for row_id in inner_train_ids for candidate_slot, _, prediction in slots[row_id] if candidate_slot == slot])) for slot in inner_priority}
            inner_scale = mind_scale(inner_train_ids, rows_by_id, image_sizes)
            global_scores = {}
            global_val_outputs = {}
            for option in options:
                successes, _ = evaluate_mind_config(e1, inner_val_ids, slots, inner_reliability, rows_by_id, image_sizes, option, inner_scale)
                key = configuration_key(option)
                global_scores[key] = float(np.mean(list(successes.values())))
                global_val_outputs[key] = successes
            by_key = {configuration_key(option): option for option in options}
            global_key = choose_configuration(global_scores, by_key, config)
            global_choice = by_key[global_key]
            action_choices, action_val_outputs = fit_mind_action_configs(
                e1, inner_val_ids, slots, inner_reliability, rows_by_id, image_sizes,
                options, inner_scale, global_choice, config,
            )
            global_val_accuracy = global_scores[global_key]
            action_val_accuracy = float(np.mean(list(action_val_outputs.values())))
            selected_variant = "action" if action_val_accuracy > global_val_accuracy else "global"

            devsel_scores = {}
            for method in E1_METHODS:
                successes = [e1.evaluate_mind_method(method, row_id, slots, inner_priority, inner_weights, rows_by_id, image_sizes, list(e1.MODEL_DIRS)) for row_id in inner_val_ids]
                devsel_scores[method] = float(np.mean(successes))
            selected_method = max(E1_METHODS, key=lambda method: (devsel_scores[method], -E1_METHODS.index(method)))

            outer_priority, outer_weights = e1.dev_mind_statistics(outer_dev_ids, slots, rows_by_id, image_sizes)
            outer_reliability = {slot: float(np.mean([e1.score_prediction(rows_by_id[row_id], prediction, image_sizes[row_id]) for row_id in outer_dev_ids for candidate_slot, _, prediction in slots[row_id] if candidate_slot == slot])) for slot in outer_priority}
            outer_scale = mind_scale(outer_dev_ids, rows_by_id, image_sizes)
            global_test, _ = evaluate_mind_config(e1, test_ids, slots, outer_reliability, rows_by_id, image_sizes, global_choice, outer_scale)
            action_test, selected_actions = evaluate_mind_action_strategy(e1, test_ids, slots, outer_reliability, rows_by_id, image_sizes, action_choices, global_choice, outer_scale)
            for row_id in test_ids:
                outputs[arm]["global"][row_id] = global_test[row_id]
                outputs[arm]["action"][row_id] = action_test[row_id]
                outputs[arm]["CEV_A"][row_id] = action_test[row_id] if selected_variant == "action" else global_test[row_id]
                outputs[arm]["dev_selection"][row_id] = e1.evaluate_mind_method(selected_method, row_id, slots, outer_priority, outer_weights, rows_by_id, image_sizes, list(e1.MODEL_DIRS))
            fold_record["arms"][arm] = {
                "global_configuration": global_choice,
                "global_validation_accuracy": global_val_accuracy,
                "global_validation_scores": {
                    configuration_label(by_key[key]): value
                    for key, value in global_scores.items()
                },
                "action_configurations": action_choices,
                "action_validation_accuracy": action_val_accuracy,
                "selected_variant": selected_variant,
                "selected_actions_on_test": dict(Counter(selected_actions.values())),
                "dev_selection_method": selected_method,
                "dev_selection_validation_accuracy": devsel_scores[selected_method],
                "dev_selection_scores": devsel_scores,
                "inner_scale": list(inner_scale),
                "outer_refit_scale": list(outer_scale),
                "inner_train_rows": len(inner_train_ids),
                "inner_validation_rows": len(inner_val_ids),
                "outer_test_rows": len(test_ids),
            }
        records.append(fold_record)
    accuracy = {arm: {method: float(np.mean(list(values.values()))) for method, values in methods.items()} for arm, methods in outputs.items()}
    metadata = {row_id: {"fold": fold_for_id[row_id], "group": rows_by_id[row_id]["episode_id"], "action": rows_by_id[row_id]["step"]["operation"]["op"]} for row_id in rows_by_id}
    return {"rows": len(rows_by_id), "outputs": outputs, "accuracy": accuracy, "metadata": metadata, "folds": records}


def nested_screen(e1, config, arms=ARMS):
    context_module = e1.load_module(e1.CONSOLIDATE / "common.py", "cev_main_screen_common")
    context = context_module.load_context()
    regions = {row["id"]: row for row in map(json.loads, (e1.CONSOLIDATE / "raw/q1_regions.jsonl").read_text().splitlines())}
    q1 = {model: e1.load_screen_q1(model) for model in e1.SCREEN_MODELS}
    row_ids = context["row_ids"]
    targets = {row_id: context["metadata"][row_id]["target_bbox"] for row_id in row_ids}
    fold_for_group = context["fold_for_group"]
    fold_for_id = {row_id: fold_for_group[context["metadata"][row_id]["application"]] for row_id in row_ids}
    threshold = config["benchmarks"]["screenspot_pro"]["coordinate_base_tolerance"]
    outputs = {arm: {"CEV_A": {}, "dev_selection": {}} for arm in arms}
    records = []
    for outer_fold in range(5):
        inner_val_fold = (outer_fold + 1) % 5
        inner_train_ids = [row_id for row_id in row_ids if fold_for_id[row_id] not in {outer_fold, inner_val_fold}]
        inner_val_ids = [row_id for row_id in row_ids if fold_for_id[row_id] == inner_val_fold]
        outer_dev_ids = [row_id for row_id in row_ids if fold_for_id[row_id] != outer_fold]
        test_ids = [row_id for row_id in row_ids if fold_for_id[row_id] == outer_fold]
        fold_record = {"outer_fold": outer_fold, "inner_validation_fold": inner_val_fold, "arms": {}}
        for arm in arms:
            slots = {row_id: e1.screen_slots(context, regions, q1, arm, row_id) for row_id in row_ids}
            inner_priority, inner_reliability = e1.screen_dev_priority(inner_train_ids, slots, targets)
            devsel_scores = {}
            for method in E1_METHODS:
                devsel_scores[method] = float(np.mean([e1.evaluate_screen_method(method, row_id, slots, inner_priority, targets[row_id]) for row_id in inner_val_ids]))
            selected_method = max(E1_METHODS, key=lambda method: (devsel_scores[method], -E1_METHODS.index(method)))
            outer_priority, outer_reliability = e1.screen_dev_priority(outer_dev_ids, slots, targets)
            for row_id in test_ids:
                candidates = [Candidate(action="POINT", coordinate=tuple(candidate["point"]), parameter="", source=slot, reliability=outer_reliability[slot], order=order, payload=candidate, parse_ok=True, lineage=candidate["model"]) for order, (slot, candidate) in enumerate(slots[row_id])]
                prediction, _ = select(candidates, "G4", threshold, linkage=LINKAGE, lineage_vote_cap=LINEAGE_VOTE_CAP)
                outputs[arm]["CEV_A"][row_id] = bool(e1.point_in_bbox(prediction.coordinate, targets[row_id]))
                outputs[arm]["dev_selection"][row_id] = bool(e1.evaluate_screen_method(selected_method, row_id, slots, outer_priority, targets[row_id]))
            fold_record["arms"][arm] = {
                "global_configuration": {"granularity": "G4", "coordinate_tolerance": threshold},
                "selected_variant": "global",
                "dev_selection_method": selected_method,
                "dev_selection_validation_accuracy": devsel_scores[selected_method],
                "dev_selection_scores": devsel_scores,
                "inner_train_rows": len(inner_train_ids),
                "inner_validation_rows": len(inner_val_ids),
                "outer_test_rows": len(test_ids),
            }
        records.append(fold_record)
    accuracy = {arm: {method: float(np.mean(list(values.values()))) for method, values in methods.items()} for arm, methods in outputs.items()}
    metadata = {row_id: {"fold": fold_for_id[row_id], "group": context["metadata"][row_id]["application"]} for row_id in row_ids}
    return {"rows": len(row_ids), "outputs": outputs, "accuracy": accuracy, "metadata": metadata, "folds": records}


def comparison(metadata, left, right, config, benchmark, seed_offset=0):
    differences = {row_id: int(left[row_id]) - int(right[row_id]) for row_id in metadata}
    return paired_bootstrap(metadata, differences, config["statistics"]["paired_bootstrap_resamples"], config["benchmarks"][benchmark]["bootstrap_seed"] + seed_offset)


def gate_noninferior(value, mde):
    return value["ci_99"][1] >= 0 or abs(value["point_delta"]) < mde


def strip_outputs(result):
    return {key: value for key, value in result.items() if key != "outputs"}


def main():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if config["status"] != "POST_LEAKAGE_RECONSTRUCTED_FROZEN_BEFORE_CEV_RESULTS":
        raise ValueError("CEV preregistration is not frozen")
    anchor = json.loads((RUN_DIR / "v1_anchor.json").read_text())
    if not anchor["exact_aggregate_match"] or anchor["C_K1"]:
        raise ValueError("C-K1 blocks nested calibration")
    e1 = load_module(CLOSE / "e1_arm_aggregator_matrix.py", "cev_main_e1")
    baseline_config = yaml.safe_load((CLOSE / "configs/aggregator_map.yaml").read_text())
    baseline_mind = e1.mind2web_matrix(baseline_config)
    baseline_screen = e1.screenspot_matrix(baseline_config)
    mind = nested_mind(e1, config)
    screen = nested_screen(e1, config)

    comparisons = {
        "mind2web": {
            "CEV_A_minus_majority": comparison(mind["metadata"], mind["outputs"]["C_uni"]["CEV_A"], baseline_mind["outputs"]["C_uni"]["majority"], config, "mind2web"),
            "CEV_A_minus_sequential": comparison(mind["metadata"], mind["outputs"]["C_uni"]["CEV_A"], baseline_mind["outputs"]["C_uni"]["ours"], config, "mind2web", 1),
            "CEV_A_minus_dev_selection": comparison(mind["metadata"], mind["outputs"]["C_uni"]["CEV_A"], mind["outputs"]["C_uni"]["dev_selection"], config, "mind2web", 2),
        },
        "screenspot_pro": {
            "CEV_A_minus_A2": comparison(screen["metadata"], screen["outputs"]["C_uni"]["CEV_A"], baseline_screen["outputs"]["C_uni"]["A2"], config, "screenspot_pro"),
            "CEV_A_minus_majority": comparison(screen["metadata"], screen["outputs"]["C_uni"]["CEV_A"], baseline_screen["outputs"]["C_uni"]["majority"], config, "screenspot_pro", 1),
            "CEV_A_minus_dev_selection": comparison(screen["metadata"], screen["outputs"]["C_uni"]["CEV_A"], screen["outputs"]["C_uni"]["dev_selection"], config, "screenspot_pro", 2),
        },
    }
    comparisons["mind2web"]["arm_robustness_CEV_A"] = {
        arm: comparison(
            mind["metadata"],
            mind["outputs"][arm]["CEV_A"],
            mind["outputs"]["C_uni"]["CEV_A"],
            config,
            "mind2web",
            10 + index,
        )
        for index, arm in enumerate(("C_cond", "C_rand", "C_self"))
    }
    comparisons["screenspot_pro"]["arm_robustness_CEV_A"] = {
        arm: comparison(
            screen["metadata"],
            screen["outputs"][arm]["CEV_A"],
            screen["outputs"]["C_uni"]["CEV_A"],
            config,
            "screenspot_pro",
            10 + index,
        )
        for index, arm in enumerate(("C_cond", "C_rand", "C_self"))
    }
    v2_screen = gate_noninferior(comparisons["screenspot_pro"]["CEV_A_minus_A2"], config["benchmarks"]["screenspot_pro"]["mde"])
    v2_mind = gate_noninferior(comparisons["mind2web"]["CEV_A_minus_majority"], config["benchmarks"]["mind2web"]["mde"])
    v3_screen = comparisons["screenspot_pro"]["CEV_A_minus_majority"]["ci_99"][0] > 0
    v3_mind = comparisons["mind2web"]["CEV_A_minus_sequential"]["ci_99"][0] > 0
    dev_values = {
        "mind2web": comparisons["mind2web"]["CEV_A_minus_dev_selection"],
        "screenspot_pro": comparisons["screenspot_pro"]["CEV_A_minus_dev_selection"],
    }
    method_contribution = all(value["ci_99"][0] > 0 for value in dev_values.values())
    failed_unification = any(value["ci_99"][1] < 0 and abs(value["point_delta"]) >= config["benchmarks"][benchmark]["mde"] for benchmark, value in dev_values.items())
    v4 = "METHOD_CONTRIBUTION" if method_contribution else "FAILED_UNIFICATION" if failed_unification else "EXPLANATORY_CONTRIBUTION"
    mind_primary_configs = [fold["arms"]["C_uni"]["global_configuration"]["granularity"] for fold in mind["folds"]]
    modal_count = max(Counter(mind_primary_configs).values())
    central_best = {"G1": [], "G3": []}
    for fold in mind["folds"]:
        scores = fold["arms"]["C_uni"]["global_validation_scores"]
        for granularity in central_best:
            central = {
                label: value
                for label, value in scores.items()
                if label.startswith(granularity) and any(f"c{multiplier}" in label for multiplier in (0.75, 1.0, 1.25))
            }
            if central:
                central_best[granularity].append(max(central, key=lambda label: (central[label], label)))
    tolerance_rank_flip = any(len(set(labels)) > 1 for labels in central_best.values())
    c_k5 = modal_count < 3 or tolerance_rank_flip
    gates = {
        "V1": True,
        "V2": {"pass": v2_screen and v2_mind, "screenspot_pro": v2_screen, "mind2web": v2_mind},
        "V3": {"pass": v3_screen and v3_mind, "screenspot_pro": v3_screen, "mind2web": v3_mind},
        "V4": v4,
        "C_K1": False,
        "C_K2": not v2_screen,
        "C_K3": not v2_mind,
        "C_K4": v4 == "FAILED_UNIFICATION",
        "C_K5": c_k5,
        "C_K5_diagnostics": {
            "selected_granularity_modal_count": modal_count,
            "selected_granularities": mind_primary_configs,
            "central_tolerance_best_by_fold": central_best,
            "central_tolerance_rank_flip": tolerance_rank_flip,
        },
    }
    result = {
        "schema_version": 1,
        "status": "PASS_ANALYSIS_COMPLETE",
        "config": "configs/cev_prereg.yaml",
        "config_sha256": sha256_file(CONFIG_PATH),
        "preregistration_commits": ["d873c41", "de5b125"],
        "leakage_boundary": config["leakage"],
        "mind2web": strip_outputs(mind),
        "screenspot_pro": strip_outputs(screen),
        "baseline_accuracy": {"mind2web": baseline_mind["accuracy"], "screenspot_pro": baseline_screen["accuracy"]},
        "comparisons": comparisons,
        "gates": gates,
        "paper_position": "CEV_METHOD_CONTRIBUTION" if v4 == "METHOD_CONTRIBUTION" else "CEV_EXPLANATORY_CONTRIBUTION" if v4 == "EXPLANATORY_CONTRIBUTION" else "KEEP_F1_PRIMARY_CEV_FAILED_ATTEMPT",
        "outputs": {"mind2web": mind["outputs"], "screenspot_pro": screen["outputs"]},
    }
    atomic_json(RUN_DIR / "cev_main.json", result)
    print(json.dumps({"gates": gates, "paper_position": result["paper_position"], "accuracy": {"mind2web": mind["accuracy"], "screenspot_pro": screen["accuracy"]}, "comparisons": comparisons}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()