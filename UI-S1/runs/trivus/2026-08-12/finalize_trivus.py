import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from context_common import atomic_json_file, sha256_file


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CONFIG_PATH = RUN_DIR / "configs/formal_runner_prereg.yaml"
OUTPUT_ROOT = RUN_DIR / "formal"
POLICIES = (
    "JOINT3", "TARGET_ONLY", "JOINT2_NO_ANDROID", "NO_VISUAL",
    "RANDOM_ID_PLACEBO",
)
CELL_ORDER = (
    ("mind2web", "C_uni"), ("mind2web", "C_cond"),
    ("mind2web", "C_rand"), ("mind2web", "C_self"),
    ("screenspot_pro", "C_uni"), ("screenspot_pro", "C_cond"),
    ("screenspot_pro", "C_rand"), ("screenspot_pro", "C_self"),
    ("androidcontrol", "low"), ("androidcontrol", "high"),
)


def load_public():
    paths = (
        ROOT / "runs/visual-utility-selector/2026-08-11/data/public_records.jsonl",
        RUN_DIR / "data/public_records.jsonl",
    )
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]
    output = {row["sample_key"]: row for row in rows}
    if len(output) != 18644:
        raise ValueError("TriVUS final public coverage mismatch")
    return output


def load_configs():
    formal = yaml.safe_load(CONFIG_PATH.read_text())
    training_path = RUN_DIR / "configs/training_prereg.yaml"
    training = yaml.safe_load(training_path.read_text())
    for config in (formal, training):
        for item in config["dependencies"].values():
            if sha256_file(ROOT / item["path"]) != item["sha256"]:
                raise PermissionError(f"TriVUS final dependency drift: {item['path']}")
    if formal["statistics"] != {
        "resamples": 10000,
        "confidence": 0.99,
        "bootstrap_seed_base": 20260900,
        "control_offsets": {
            "primary": 0, "target_only": 100,
            "no_visual": 200, "strongest": 300,
        },
        "same_index_cell_composition_within_comparison": True,
    }:
        raise ValueError("TriVUS final statistics contract mismatch")
    return formal, training


def validate_outer_result(row, marker_value, public, fold, result_sha256, pretest_sha256):
    if marker_value != {
        "schema_version": 1,
        "status": "TRIVUS_OUTER_COMPLETE",
        "outer_fold": fold,
        "result_sha256": result_sha256,
    }:
        raise ValueError(f"TriVUS outer completion marker mismatch: {fold}")
    if (
        set(row) != {
            "schema_version", "status", "outer_fold", "pretest_sha256",
            "inner_epochs", "final_epochs", "thresholds",
            "opened_outer_label_sha256", "reports", "outputs",
        }
        or row.get("schema_version") != 1
        or row.get("status") != "PASS_TRIVUS_OUTER_COMPLETE"
        or row.get("outer_fold") != fold
        or row.get("pretest_sha256") != pretest_sha256
        or set(row.get("outputs", {})) != set(POLICIES)
    ):
        raise ValueError(f"TriVUS incomplete outer result: {fold}")
    for policy in POLICIES:
        if set(row["outputs"][policy]) != {"safe", "direct", "fallback"}:
            raise ValueError(f"TriVUS outer output schema mismatch: {fold}/{policy}")
        for method, incoming in row["outputs"][policy].items():
            if (
                not isinstance(incoming, dict)
                or any(
                    key not in public
                    or int(public[key]["fold"]) != fold
                    or type(value) is not bool
                    for key, value in incoming.items()
                )
            ):
                raise ValueError(f"TriVUS held-out output mismatch: {fold}/{policy}/{method}")
    return True


def merge_outers(public):
    values = {policy: {method: {} for method in ("safe", "direct", "fallback")} for policy in POLICIES}
    reports = []
    for fold in range(5):
        path = OUTPUT_ROOT / f"outer-{fold}" / f"outer-{fold}.json"
        pretest = OUTPUT_ROOT / f"outer-{fold}" / f"outer-{fold}.pretest.json"
        marker = OUTPUT_ROOT / f"outer-{fold}" / "OUTER_COMPLETE.json"
        if not path.is_file() or not pretest.is_file() or not marker.is_file():
            raise FileNotFoundError(path if not path.is_file() else pretest if not pretest.is_file() else marker)
        row = json.loads(path.read_text())
        marker_value = json.loads(marker.read_text())
        validate_outer_result(
            row, marker_value, public, fold,
            sha256_file(path), sha256_file(pretest),
        )
        for policy in POLICIES:
            for method in values[policy]:
                incoming = row["outputs"][policy][method]
                overlap = set(values[policy][method]) & set(incoming)
                if overlap:
                    raise ValueError(f"TriVUS duplicate outer outputs: {policy}/{method}")
                values[policy][method].update(incoming)
        reports.append(row)
    return values, reports


def frozen_baselines(public, training_config):
    vus_path = ROOT / training_config["dependencies"]["vus_baseline"]["path"]
    r1_path = ROOT / training_config["dependencies"]["android_r1"]["path"]
    if (
        sha256_file(vus_path) != training_config["dependencies"]["vus_baseline"]["sha256"]
        or sha256_file(r1_path) != training_config["dependencies"]["android_r1"]["sha256"]
    ):
        raise PermissionError("TriVUS frozen baseline hash mismatch")
    vus = json.loads(vus_path.read_text())
    r1 = json.loads(r1_path.read_text())
    primary = {}
    strongest = {}
    for sample_key, row in public.items():
        family = row["benchmark"]
        if family == "androidcontrol":
            setting = row["setting"]
            row_id = row["row_id"]
            primary[sample_key] = bool(r1["settings"][setting]["outputs"]["majority"][row_id])
            strongest[sample_key] = bool(r1["settings"][setting]["outputs"]["UI-AGILE-7B"][row_id])
        else:
            value = bool(vus["outputs"][family][row["arm"]]["safe"][row["row_id"]])
            primary[sample_key] = value
            strongest[sample_key] = value
    return primary, strongest


def cell_keys(public, family, cell):
    return sorted(
        key for key, row in public.items()
        if row["benchmark"] == family
        and (row["setting"] if family == "androidcontrol" else row["arm"]) == cell
    )


def paired_samples(public, keys, left, right, resamples, seed):
    if set(keys) != set(left) or set(keys) != set(right):
        raise ValueError("TriVUS paired sample coverage mismatch")
    by_fold_group = defaultdict(lambda: defaultdict(list))
    for key in keys:
        row = public[key]
        by_fold_group[int(row["fold"])][str(row["group"])].append(key)
    differences = {key: int(left[key]) - int(right[key]) for key in keys}
    generator = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        selected = []
        for fold in sorted(by_fold_group):
            groups = sorted(by_fold_group[fold])
            for group in generator.choice(groups, size=len(groups), replace=True):
                selected.extend(by_fold_group[fold][group])
        samples[index] = np.mean([differences[key] for key in selected])
    point = float(np.mean(list(differences.values())))
    return {
        "point_delta": point,
        "ci_99": [float(np.quantile(samples, 0.005)), float(np.quantile(samples, 0.995))],
        "rows": len(keys),
        "groups": len({public[key]["group"] for key in keys}),
        "resamples": resamples,
        "seed": seed,
    }, samples


def compare_policy(public, left, right, comparison_name, config):
    cells = {}
    samples = {}
    for index, (family, cell) in enumerate(CELL_ORDER):
        keys = cell_keys(public, family, cell)
        left_cell = {key: left[key] for key in keys}
        right_cell = {key: right[key] for key in keys}
        report, replicate = paired_samples(
            public, keys, left_cell, right_cell,
            config["statistics"]["resamples"],
            config["statistics"]["bootstrap_seed_base"]
            + config["statistics"]["control_offsets"][comparison_name]
            + index,
        )
        cells[f"{family}/{cell}"] = report
        samples[(family, cell)] = replicate
    family_samples = {
        family: np.mean(np.stack([
            samples[(name, cell)] for name, cell in CELL_ORDER if name == family
        ]), axis=0)
        for family in ("mind2web", "screenspot_pro", "androidcontrol")
    }
    standardized = np.mean(np.stack([
        family_samples[family] / config["thresholds"]["mde"][family]
        for family in family_samples
    ]), axis=0)
    return {
        "comparison": comparison_name,
        "cells": cells,
        "families": {
            family: {
                "point_delta": float(np.mean(values)),
                "ci_99": [float(np.quantile(values, 0.005)), float(np.quantile(values, 0.995))],
            }
            for family, values in family_samples.items()
        },
        "standardized_three_family": {
            "point_delta": float(np.mean(standardized)),
            "ci_99": [float(np.quantile(standardized, 0.005)), float(np.quantile(standardized, 0.995))],
        },
    }


def adjudicate(outputs, public, primary, strongest, config):
    joint = outputs["JOINT3"]["safe"]
    comparisons = {
        "primary": compare_policy(public, joint, primary, "primary", config),
        "target_only": compare_policy(public, joint, outputs["TARGET_ONLY"]["safe"], "target_only", config),
        "no_visual": compare_policy(public, joint, outputs["NO_VISUAL"]["safe"], "no_visual", config),
        "strongest": compare_policy(public, joint, strongest, "strongest", config),
    }
    mde = config["thresholds"]["mde"]
    m2w_screen_cells = [f"{family}/{cell}" for family, cell in CELL_ORDER if family != "androidcontrol"]
    android_cells = ["androidcontrol/low", "androidcontrol/high"]
    gate1 = all(
        comparisons["primary"]["cells"][cell]["ci_99"][0] > -mde[cell.split("/")[0]]
        for cell in m2w_screen_cells
    )
    gate2 = all(
        comparisons["primary"]["cells"][cell]["ci_99"][0] > -0.01
        for cell in android_cells
    )
    gate3 = any(
        value["ci_99"][0] > 0 for value in comparisons["primary"]["families"].values()
    )
    gate4 = comparisons["primary"]["standardized_three_family"]["ci_99"][0] > 0
    gate5 = comparisons["target_only"]["standardized_three_family"]["ci_99"][0] > 0
    gate6 = comparisons["no_visual"]["standardized_three_family"]["ci_99"][0] > 0
    gate7 = gate1 and all(
        comparisons["strongest"]["cells"][cell]["ci_99"][0] > -0.01
        for cell in android_cells
    )
    gates = {
        "G1_vus_sr_cell_noninferiority": gate1,
        "G2_android_majority_noninferiority": gate2,
        "G3_one_family_positive": gate3,
        "G4_three_family_primary_positive": gate4,
        "G5_joint_minus_target_only_positive": gate5,
        "G6_joint_minus_no_visual_positive": gate6,
        "G7_strongest_baseline_safety": gate7,
    }
    return comparisons, gates


def main():
    config, training_config = load_configs()
    public = load_public()
    outputs, outers = merge_outers(public)
    expected_all = set(public)
    expected_joint2 = {
        key for key, row in public.items() if row["benchmark"] != "androidcontrol"
    }
    for policy in POLICIES:
        expected = expected_joint2 if policy == "JOINT2_NO_ANDROID" else expected_all
        for method in outputs[policy]:
            if set(outputs[policy][method]) != expected:
                raise ValueError(f"TriVUS final policy coverage mismatch: {policy}/{method}")
    primary, strongest = frozen_baselines(public, training_config)
    comparisons, gates = adjudicate(outputs, public, primary, strongest, config)
    result = {
        "schema_version": 1,
        "status": "PASS_TRIVUS_ADJUDICATED",
        "outcome": "TRIVUS_PROMOTED" if all(gates.values()) else "TRIVUS_NOT_PROMOTED",
        "gates": gates,
        "comparisons": comparisons,
        "outputs": outputs,
        "outer_epochs": [
            {
                "outer_fold": row["outer_fold"],
                "inner_epochs": row["inner_epochs"],
                "final_epochs": row["final_epochs"],
            }
            for row in outers
        ],
        "claim_boundary": {
            "vus_sr": "paired_success_bits_only_not_action_semantic_compatibility",
            "androidcontrol": "paired_2000_rows_per_setting_not_full_7650",
            "external_confirmation": False,
        },
    }
    path = RUN_DIR / "TRIVUS_ADJUDICATION.json"
    if path.exists():
        raise FileExistsError(path)
    atomic_json_file(path, result)
    print(json.dumps({"outcome": result["outcome"], "gates": gates}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()