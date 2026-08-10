import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CLOSE = ROOT / "runs/close/2026-08-08"
CONFIG_PATH = RUN_DIR / "configs/cev_prereg.yaml"
sys.path.insert(0, str(RUN_DIR))

from cev import Candidate, select


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if config["status"] != "POST_LEAKAGE_RECONSTRUCTED_FROZEN_BEFORE_CEV_RESULTS":
        raise ValueError("CEV preregistration is not frozen")
    e1 = load_module(CLOSE / "e1_arm_aggregator_matrix.py", "cev_v1_e1")
    context_module = e1.load_module(e1.CONSOLIDATE / "common.py", "cev_v1_common")
    context = context_module.load_context()
    row_ids = context["row_ids"]
    targets = {row_id: context["metadata"][row_id]["target_bbox"] for row_id in row_ids}
    fold_for_group = context["fold_for_group"]
    actions = [(model, view) for view in range(4) for model in e1.SCREEN_MODELS]
    slots = {
        row_id: [(f"{model}_view{view}", dict(context["bank"][(model, view)][row_id])) for model, view in actions]
        for row_id in row_ids
    }
    cev_outputs = {}
    a2_outputs = {}
    selected_sources = {}
    folds = []
    threshold = config["benchmarks"]["screenspot_pro"]["coordinate_base_tolerance"]
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
                    source=slot,
                    reliability=reliability[slot],
                    order=index,
                    payload=candidate,
                )
                for index, (slot, candidate) in enumerate(slots[row_id])
            ]
            prediction, _ = select(candidates, "G4", threshold)
            cev_outputs[row_id] = bool(e1.point_in_bbox(prediction.coordinate, targets[row_id]))
            selected_sources[row_id] = prediction.source
            a2_point = e1.density_medoid([candidate["point"] for _, candidate in slots[row_id]])
            a2_outputs[row_id] = bool(e1.point_in_bbox(a2_point, targets[row_id]))
        folds.append({"fold": test_fold, "dev_rows": len(dev_ids), "test_rows": len(test_ids), "priority": priority})
    cev_accuracy = float(np.mean(list(cev_outputs.values())))
    a2_accuracy = float(np.mean(list(a2_outputs.values())))
    anchor = config["leakage"]["disclosed_screenspot_c_uni_accuracies"]["A2"]
    differing_ids = [row_id for row_id in row_ids if cev_outputs[row_id] != a2_outputs[row_id]]
    passed = cev_accuracy == anchor and a2_accuracy == anchor
    result = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL_C_K1",
        "config": "configs/cev_prereg.yaml",
        "benchmark": "screenspot_pro",
        "arm": "C_uni",
        "granularity": "G4",
        "CEV_accuracy": cev_accuracy,
        "A2_accuracy": a2_accuracy,
        "frozen_anchor": anchor,
        "exact_aggregate_match": passed,
        "rowwise_agreement": 1 - len(differing_ids) / len(row_ids),
        "rowwise_disagreements": len(differing_ids),
        "rowwise_disagreement_ids": differing_ids,
        "selected_sources": selected_sources,
        "folds": folds,
        "C_K1": not passed,
        "downstream": "STOP_AND_DEBUG" if not passed else "CONTINUE_NESTED_CALIBRATION",
    }
    (RUN_DIR / "v1_anchor.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "CEV_accuracy", "A2_accuracy", "frozen_anchor", "rowwise_agreement", "rowwise_disagreements", "C_K1", "downstream")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()