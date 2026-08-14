import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
GRAN_DIR = ROOT / "runs/gran/2026-08-14"
CONFIG_PATH = RUN_DIR / "configs/split_prereg.yaml"
PREFLIGHT_PATH = RUN_DIR / "PREFLIGHT.json"
OUTPUT_PATH = RUN_DIR / "ZERO_GPU_GATE.json"
sys.path.insert(0, str(GRAN_DIR))

from gran_common import attach_reliability, partition, source_reliability


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_contract():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    upstream = config["upstream"]
    expected = {
        "gran_tau_source": "runs/gran/2026-08-14/TAU_SWEEP.json",
        "gran_adjudication": "runs/gran/2026-08-14/GRAN_ADJUDICATION.json",
        "gran_input_manifest": "runs/gran/2026-08-14/INPUT_MANIFEST.json",
        "gran_config": "runs/gran/2026-08-14/configs/gran_prereg.yaml",
    }
    if config.get("status") != "FROZEN_BEFORE_ZERO_GPU_GATE_AND_ANY_PROBE_FORWARD":
        raise PermissionError("SPLIT prereg status mismatch")
    for key, relative_path in expected.items():
        if upstream[key]["path"] != relative_path:
            raise PermissionError(f"SPLIT upstream path mismatch: {key}")
        if sha256_file(ROOT / relative_path) != upstream[key]["sha256"]:
            raise PermissionError(f"SPLIT upstream hash mismatch: {key}")
    if (
        preflight.get("status")
        != "PASS_SPLIT_PREFLIGHT_QWEN3_GTA1_READY_QWEN25_DEFERRED"
        or preflight.get("gpu_used") is not False
        or preflight.get("Delta2_computed") is not False
        or preflight.get("probe_forward_started") is not False
    ):
        raise PermissionError("SPLIT preflight boundary mismatch")
    return config


def class_score(parsed, members):
    return (
        len(members),
        sum(parsed[index].reliability for index in members),
        max(parsed[index].reliability for index in members),
        -min(parsed[index].order for index in members),
    )


def top_two_modes(candidates, tau):
    parsed, classes = partition(
        candidates, "screenspot_pro", "finite", float(tau)
    )
    ranked = sorted(
        classes, key=lambda members: class_score(parsed, members), reverse=True
    )
    if len(ranked) < 2:
        return None
    output = []
    for members in ranked[:2]:
        output.append({
            "votes": len(members),
            "correct": any(parsed[index].correct for index in members),
            "members": [parsed[index].order for index in members],
        })
    return output


def evaluate(rows, row_ids, reliability, tau, gate):
    output = {}
    for row_id in row_ids:
        candidates = attach_reliability(rows[row_id]["candidates"], reliability)
        modes = top_two_modes(candidates, tau)
        if modes is None:
            parsed, classes = partition(
                candidates, "screenspot_pro", "finite", float(tau)
            )
            output[row_id] = {
                "fold": int(rows[row_id]["fold"]),
                "application": str(rows[row_id]["group"]),
                "mode_count": len(classes),
                "w1": len(classes[0]) if classes else 0,
                "w2": None,
                "w2_over_w1": None,
                "M1_correct": (
                    any(parsed[index].correct for index in classes[0])
                    if classes else False
                ),
                "M2_correct": False,
                "gate": False,
                "positive": False,
                "negative": False,
                "M1_members": (
                    [parsed[index].order for index in classes[0]]
                    if classes else []
                ),
                "M2_members": [],
            }
            continue
        ratio = float(modes[1]["votes"] / modes[0]["votes"])
        triggered = ratio >= float(gate)
        positive = bool(triggered and modes[1]["correct"] and not modes[0]["correct"])
        negative = bool(triggered and modes[0]["correct"] and not modes[1]["correct"])
        output[row_id] = {
            "fold": int(rows[row_id]["fold"]),
            "application": str(rows[row_id]["group"]),
            "mode_count": None,
            "w1": int(modes[0]["votes"]),
            "w2": int(modes[1]["votes"]),
            "w2_over_w1": ratio,
            "M1_correct": bool(modes[0]["correct"]),
            "M2_correct": bool(modes[1]["correct"]),
            "gate": triggered,
            "positive": positive,
            "negative": negative,
            "M1_members": modes[0]["members"],
            "M2_members": modes[1]["members"],
        }
    return output


def metrics(outputs):
    values = list(outputs.values())
    gate_count = sum(row["gate"] for row in values)
    positive_count = sum(row["positive"] for row in values)
    negative_count = sum(row["negative"] for row in values)
    insufficient_mode_count = sum(row["w2"] is None for row in values)
    return {
        "rows": len(values),
        "insufficient_mode_rows": insufficient_mode_count,
        "gate_rows": gate_count,
        "gate_prevalence": float(gate_count / len(values)),
        "positive_rows": positive_count,
        "negative_rows": negative_count,
        "conditional_positive_rate": (
            float(positive_count / gate_count) if gate_count else 0.0
        ),
        "Delta2": float(positive_count / len(values)),
    }


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    config = load_contract()
    gran_runner = load_module(GRAN_DIR / "run_tau_sweep.py", "split_gran_runner")
    e1 = gran_runner.load_module(
        gran_runner.CLOSE_DIR / "e1_arm_aggregator_matrix.py", "split_gate_e1"
    )
    rows = gran_runner.load_screen_rows(e1)
    row_ids = tuple(sorted(rows))
    gate_grid = tuple(float(value) for value in config["trigger_gate"]["grid"])
    tau_by_fold = {
        int(fold): float(value)
        for fold, value in config["upstream"]["gran_tau_source"][
            "screenspot_selected_tau_by_outer_fold"
        ].items()
    }
    all_outputs = {}
    fold_records = []
    for outer_fold in range(5):
        inner_validation_fold = (outer_fold + 1) % 5
        inner_train = [
            row_id for row_id in row_ids
            if rows[row_id]["fold"] not in {outer_fold, inner_validation_fold}
        ]
        inner_validation = [
            row_id for row_id in row_ids
            if rows[row_id]["fold"] == inner_validation_fold
        ]
        outer_development = [
            row_id for row_id in row_ids if rows[row_id]["fold"] != outer_fold
        ]
        outer_test = [
            row_id for row_id in row_ids if rows[row_id]["fold"] == outer_fold
        ]
        tau = tau_by_fold[outer_fold]
        inner_reliability = source_reliability(rows, inner_train)
        gate_scores = []
        for gate in gate_grid:
            validation_outputs = evaluate(
                rows, inner_validation, inner_reliability, tau, gate
            )
            gate_scores.append({"g": gate, **metrics(validation_outputs)})
        selected = max(gate_scores, key=lambda score: (score["Delta2"], score["g"]))
        outer_reliability = source_reliability(rows, outer_development)
        heldout = evaluate(rows, outer_test, outer_reliability, tau, selected["g"])
        if set(all_outputs) & set(heldout):
            raise ValueError("SPLIT duplicate held-out output")
        all_outputs.update(heldout)
        fold_records.append({
            "outer_fold": outer_fold,
            "inner_validation_fold": inner_validation_fold,
            "inner_train_rows": len(inner_train),
            "inner_validation_rows": len(inner_validation),
            "outer_development_rows": len(outer_development),
            "outer_test_rows": len(outer_test),
            "selected_tau": tau,
            "selected_g": selected["g"],
            "inner_validation_scores": gate_scores,
            "heldout": metrics(heldout),
        })
    if set(all_outputs) != set(row_ids):
        raise ValueError("SPLIT held-out coverage mismatch")
    pooled = metrics(all_outputs)
    threshold = float(config["zero_gpu_gate"]["threshold"])
    passed = pooled["Delta2"] >= threshold
    result = {
        "schema_version": 1,
        "status": "PASS_Z_G1_PROCEED_TO_GEOMETRY" if passed else "STOP_Z_K1_BEFORE_GPU",
        "zero_gpu": True,
        "probe_forward_started": False,
        "selection_rule": "maximize_inner_validation_Delta2_tie_larger_g",
        "gate_grid": list(gate_grid),
        "threshold": threshold,
        "folds": fold_records,
        "pooled": pooled,
        "Z_G1_pass": passed,
        "Z_K1_triggered": not passed,
        "heldout_rows": all_outputs,
        "claim_boundary": {
            "exploratory_only": True,
            "runtime_gate_uses_labels": False,
            "Delta2_uses_labels_for_offline_gate_adjudication": True,
            "gpu_authorized": False,
        },
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "selected_g": [fold["selected_g"] for fold in fold_records],
        "pooled": pooled,
        "Z_G1_pass": passed,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()