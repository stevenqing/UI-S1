import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CLOSE_DIR = ROOT / "runs/close/2026-08-08"
AGGMATCH_DIR = ROOT / "runs/aggmatch/2026-08-09"
CONFIG_PATH = RUN_DIR / "configs/eqv_equivalence.yaml"
sys.path.insert(0, str(RUN_DIR))
sys.path.insert(0, str(AGGMATCH_DIR))

from aggmatch_common import atomic_json, paired_bootstrap, sha256_file
from eqv import Candidate, aggregate, equivalent


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_abl4_self_check(config):
    e1 = load_module(CLOSE_DIR / "e1_arm_aggregator_matrix.py", "eqv_close_e1")
    context_module = e1.load_module(e1.CONSOLIDATE / "common.py", "eqv_consolidate_common")
    context = context_module.load_context()
    row_ids = context["row_ids"]
    targets = {row_id: context["metadata"][row_id]["target_bbox"] for row_id in row_ids}
    fold_for_group = context["fold_for_group"]
    actions = [(model, view) for view in range(4) for model in e1.SCREEN_MODELS]
    slots = {
        row_id: [(f"{model}_view{view}", dict(context["bank"][(model, view)][row_id])) for model, view in actions]
        for row_id in row_ids
    }
    variants = {
        "complete_lineage_dedup": {"linkage": "complete", "lineage_dedup": True, "outputs": {}},
        "complete_candidate_votes": {"linkage": "complete", "lineage_dedup": False, "outputs": {}},
        "single_lineage_dedup": {"linkage": "single", "lineage_dedup": True, "outputs": {}},
        "single_candidate_votes": {"linkage": "single", "lineage_dedup": False, "outputs": {}},
    }
    a2_outputs = {}
    fold_records = []
    threshold = config["equivalence"]["coordinate"]["screenspot_pro"]["threshold"]
    lineage_order = tuple(config["lineage"]["frozen_order"]["screenspot_pro"])
    for test_fold in range(5):
        dev_ids = [row_id for row_id in row_ids if fold_for_group[context["metadata"][row_id]["application"]] != test_fold]
        test_ids = [row_id for row_id in row_ids if fold_for_group[context["metadata"][row_id]["application"]] == test_fold]
        priority, reliability = e1.screen_dev_priority(dev_ids, slots, targets)
        for row_id in test_ids:
            candidates = [
                Candidate(
                    action="POINT",
                    coordinate=tuple(candidate["point"]),
                    parameter="",
                    lineage=candidate["model"],
                    source=slot,
                    reliability=reliability[slot],
                    order=index,
                    payload=candidate,
                )
                for index, (slot, candidate) in enumerate(slots[row_id])
            ]
            relation = lambda left, right: equivalent(
                left,
                right,
                use_action=False,
                use_coordinate=True,
                use_parameter=False,
                coordinate_threshold=threshold,
                coordinate_metric="euclidean",
                parameter_threshold=1.0,
            )
            for variant in variants.values():
                selected = aggregate(
                    candidates,
                    equivalence=relation,
                    linkage=variant["linkage"],
                    lineage_dedup=variant["lineage_dedup"],
                    lineage_order=lineage_order,
                )["prediction"]
                variant["outputs"][row_id] = e1.point_in_bbox(selected.coordinate, targets[row_id])
            a2_point = e1.density_medoid([candidate["point"] for _, candidate in slots[row_id]])
            a2_outputs[row_id] = e1.point_in_bbox(a2_point, targets[row_id])
        fold_records.append({"fold": test_fold, "dev_rows": len(dev_ids), "test_rows": len(test_ids), "priority": priority, "reliability": reliability})
    metadata = {
        row_id: {
            "fold": fold_for_group[context["metadata"][row_id]["application"]],
            "group": context["metadata"][row_id]["application"],
        }
        for row_id in row_ids
    }
    a2_accuracy = float(np.mean(list(a2_outputs.values())))
    expected_a2 = json.loads((CLOSE_DIR / "e1_arm_aggregator_matrix.json").read_text())["screenspot_pro"]["accuracy"]["C_uni"]["A2"]
    if abs(a2_accuracy - expected_a2) > 1e-15:
        raise ValueError(f"ABL-4 A2 anchor mismatch: {a2_accuracy} != {expected_a2}")
    diagnostics = {}
    for name, variant in variants.items():
        differences = {row_id: int(variant["outputs"][row_id]) - int(a2_outputs[row_id]) for row_id in row_ids}
        diagnostics[name] = {
            "accuracy": float(np.mean(list(variant["outputs"].values()))),
            "minus_A2": paired_bootstrap(
                metadata,
                differences,
                config["bootstrap"]["resamples"],
                config["bootstrap"]["screenspot_pro"]["seed"],
            ),
        }
    main = diagnostics["complete_lineage_dedup"]
    triggered = abs(main["minus_A2"]["point_delta"]) > config["mde"]["screenspot_pro"]
    return {
        "status": "FAIL_U_K4" if triggered else "PASS",
        "benchmark": "screenspot_pro",
        "arm": "C_uni",
        "definition": "coordinate_only_complete_link_EQV_with_lineage_dedup",
        "EQV_ABL4_accuracy": main["accuracy"],
        "A2_accuracy": a2_accuracy,
        "EQV_ABL4_minus_A2": main["minus_A2"],
        "mde": config["mde"]["screenspot_pro"],
        "U_K4": triggered,
        "debug_diagnostics": {
            name: {"accuracy": value["accuracy"], "minus_A2": value["minus_A2"]}
            for name, value in diagnostics.items()
        },
        "debug_scope": "ABL1_and_ABL2_factorial_only_no_threshold_change_no_primary_adjudication",
        "folds": fold_records,
    }


def main():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if config["status"] != "FROZEN_BEFORE_RESULTS":
        raise ValueError("EQV config is not frozen")
    self_check = run_abl4_self_check(config)
    result = {
        "schema_version": 1,
        "status": "PAUSED_U_K4_IMPLEMENTATION_SELF_CHECK" if self_check["U_K4"] else "SELF_CHECK_PASS_READY_FOR_PRIMARY",
        "config": "configs/eqv_equivalence.yaml",
        "config_sha256": sha256_file(CONFIG_PATH),
        "definition_boundary": config["definition_boundary"],
        "ABL4_self_check": self_check,
        "primary_adjudication_started": False,
        "dev_selection_started": False,
    }
    atomic_json(RUN_DIR / "eqv_main.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()