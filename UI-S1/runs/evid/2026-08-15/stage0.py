import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
MASK_DIR = ROOT / "runs/mask/2026-08-14"
H1_DIR = ROOT / "runs/ccm-h2h/2026-07-31/h1"
DECOMP_ARM1_PATH = ROOT / "runs/decomp/2026-08-14/arm1.py"
CONFIG_PATH = RUN_DIR / "configs/evid_prereg.yaml"
PREFLIGHT_PATH = RUN_DIR / "PREFLIGHT.json"
OUTPUT_PATH = RUN_DIR / "STAGE0.json"
RAW_PATH = RUN_DIR / "raw/stage0_rows.jsonl"

sys.path.insert(0, str(MASK_DIR))
sys.path.insert(0, str(H1_DIR))
from aggregators_coord import official_groups
from mask_common import load_rows, source_reliability


MODELS = ("GTA1-7B", "Qwen3-VL-8B-Instruct", "UI-TARS-7B-SFT")
SLOTS = tuple((model, view) for view in range(4) for model in MODELS)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def representative(candidates, group):
    return max(group, key=lambda index: (float(candidates[index].get("coverage", 0)), -index))


def block_mean_coverage(candidates, group):
    return float(np.mean([float(candidates[index].get("coverage", 0)) for index in group]))


def effective_score(group, rho_v, rho_l, weights):
    counts = {model: sum(SLOTS[index][0] == model for index in group) for model in MODELS}
    represented = [model for model, count in counts.items() if count]
    numerator = sum(
        weights[model] * counts[model] / (1 + (counts[model] - 1) * rho_v)
        for model in represented
    )
    return float(numerator / (1 + (len(represented) - 1) * rho_l))


def select_group(candidates, rho_v, rho_l, weights, singleton=False):
    groups = [(index,) for index in range(len(candidates))] if singleton else official_groups([candidate["point"] for candidate in candidates])
    selected_order, selected = max(enumerate(groups), key=lambda item: (
        effective_score(item[1], rho_v, rho_l, weights),
        block_mean_coverage(candidates, item[1]),
        -item[0],
    ))
    return tuple(selected), representative(candidates, selected), selected_order


def b3_group(candidates):
    groups = official_groups([candidate["point"] for candidate in candidates])
    order, group = max(enumerate(groups), key=lambda item: (len(item[1]), block_mean_coverage(candidates, item[1]), -item[0]))
    return tuple(group), representative(candidates, group), order


def transition_pairs(features, transition):
    by_cell = defaultdict(list)
    for index, feature in enumerate(features):
        by_cell[(feature["budget"], feature["lineage_count"], feature["view_count"])].append(index)
    pairs = []
    left_count, right_count = transition
    for budget in range(2, 13):
        for view_count in range(1, 5):
            left = (budget, left_count, view_count)
            right = (budget, right_count, view_count)
            if left in by_cell and right in by_cell:
                pairs.append({"budget": budget, "view_count": view_count, "left": by_cell[left], "right": by_cell[right]})
    return pairs


def transition_report(matrix, features, rows, row_ids, transition, multiplicities, applications, app_indices):
    pairs = transition_pairs(features, transition)
    point_values = []
    by_budget = defaultdict(list)
    for pair in pairs:
        value = float(np.mean(matrix[:, pair["right"]]) - np.mean(matrix[:, pair["left"]]))
        point_values.append(value)
        by_budget[pair["budget"]].append(value)
    point = float(np.mean(point_values))
    app_pair_sums = np.asarray([
        [
            float(np.mean(matrix[np.ix_(app_indices[app], pair["right"])], axis=1).sum() - np.mean(matrix[np.ix_(app_indices[app], pair["left"])], axis=1).sum())
            for pair in pairs
        ]
        for app in applications
    ])
    app_rows = np.asarray([len(app_indices[app]) for app in applications])
    denominators = multiplicities @ app_rows
    replicate_pairs = (multiplicities @ app_pair_sums) / denominators[:, None]
    replicates = np.mean(replicate_pairs, axis=1)
    return {
        "transition": list(transition),
        "supported_pairs": len(pairs),
        "pairs": [{"budget": pair["budget"], "view_count": pair["view_count"], "point_delta": value} for pair, value in zip(pairs, point_values)],
        "by_budget": {str(budget): float(np.mean(values)) for budget, values in sorted(by_budget.items())},
        "pooled_point_delta": point,
        "ci_99": [float(np.quantile(replicates, 0.005)), float(np.quantile(replicates, 0.995))],
    }


def write_jsonl_fsynced(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def sha256_file(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUTPUT_PATH.exists() or RAW_PATH.exists():
        raise FileExistsError("EVID Stage 0 output exists")
    config = yaml.safe_load(CONFIG_PATH.read_text())
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    if preflight["status"] != "PASS_EVID_PREFLIGHT_NO_STAGE_RESULT" or preflight["stage0_computed"] is not False:
        raise PermissionError("EVID Stage 0 preflight mismatch")
    rows = load_rows()
    row_ids = tuple(sorted(rows))
    weights = {model: 1.0 for model in MODELS}
    raw = []
    rho_zero_mismatch = 0
    disagreement = 0
    output_oracle = 0
    contains_oracle = 0
    path_changes = {f"{value:.1f}": 0 for value in config["method"]["diagonal_path"]["values"]}
    previous_path = {}
    for row_id in row_ids:
        candidates = rows[row_id]["candidates"]
        b3_block, b3_rep, _ = b3_group(candidates)
        zero_block, zero_rep, _ = select_group(candidates, 0.0, 0.0, weights)
        if zero_block != b3_block or zero_rep != b3_rep:
            rho_zero_mismatch += 1
        fixed_block, fixed_rep, _ = select_group(candidates, config["method"]["primary"]["rho_v"], config["method"]["primary"]["rho_l"], weights)
        disagreement += int(fixed_block != b3_block)
        groups = official_groups([candidate["point"] for candidate in candidates])
        output_correct = any(candidates[representative(candidates, group)]["correct"] for group in groups)
        contains_correct = any(any(candidates[index]["correct"] for index in group) for group in groups)
        output_oracle += int(output_correct)
        contains_oracle += int(contains_correct)
        path_blocks = {}
        for value in config["method"]["diagonal_path"]["values"]:
            block, _, _ = select_group(candidates, value, value, weights)
            label = f"{value:.1f}"
            path_blocks[label] = list(block)
            if value > 0 and block != previous_path.get(row_id, b3_block):
                path_changes[label] += 1
            previous_path[row_id] = block
        raw.append({
            "row_id": row_id,
            "application": rows[row_id]["application"],
            "fold": rows[row_id]["fold"],
            "b3_block": list(b3_block),
            "b3_representative": b3_rep,
            "fixed_block": list(fixed_block),
            "fixed_representative": fixed_rep,
            "fixed_disagrees": fixed_block != b3_block,
            "output_correct_oracle": bool(output_correct),
            "contains_correct_oracle": bool(contains_correct),
            "path_blocks": path_blocks,
        })
    if rho_zero_mismatch:
        raise ValueError(f"EVID E-K2 rho-zero mismatch rows: {rho_zero_mismatch}")

    decomp = load_module(DECOMP_ARM1_PATH, "evid_decomp_arm1")
    features = [decomp.subset_features(mask) for mask in range(1, 1 << 12) if mask.bit_count() >= 2]
    b3_matrix = decomp.full_predictions(rows, row_ids, features)
    majority_matrix = np.zeros_like(b3_matrix)
    for fold in range(5):
        development = [row_id for row_id in row_ids if rows[row_id]["fold"] != fold]
        current, _ = decomp.majority_predictions(rows, row_ids, features, development)
        test_indices = [index for index, row_id in enumerate(row_ids) if rows[row_id]["fold"] == fold]
        majority_matrix[test_indices] = current[test_indices]
    applications = sorted({rows[row_id]["application"] for row_id in row_ids})
    app_indices = {app: np.asarray([index for index, row_id in enumerate(row_ids) if rows[row_id]["application"] == app], dtype=np.int64) for app in applications}
    app_fold = {app: rows[next(row_id for row_id in row_ids if rows[row_id]["application"] == app)]["fold"] for app in applications}
    multiplicities = decomp.application_multiplicities(applications, app_fold, config["stage0"]["transition_bootstrap"]["resamples"], 20260816)
    transitions = {
        "density_B3": {
            "1_to_2": transition_report(b3_matrix, features, rows, row_ids, (1, 2), multiplicities, applications, app_indices),
            "2_to_3": transition_report(b3_matrix, features, rows, row_ids, (2, 3), multiplicities, applications, app_indices),
        },
        "F1_majority": {
            "1_to_2": transition_report(majority_matrix, features, rows, row_ids, (1, 2), multiplicities, applications, app_indices),
            "2_to_3": transition_report(majority_matrix, features, rows, row_ids, (2, 3), multiplicities, applications, app_indices),
        },
    }
    oracle_accuracy = output_oracle / len(row_ids)
    contains_accuracy = contains_oracle / len(row_ids)
    dev_selection = config["stage1"]["mandatory_baselines"]["nested_dev_selection"]
    disagreement_rate = disagreement / len(row_ids)
    g1 = oracle_accuracy - dev_selection >= config["stage0"]["gates"]["E_G1_min_oracle_gain"]
    g2 = disagreement_rate >= config["stage0"]["gates"]["E_G2_min_disagreement"]
    g3_points = [transitions[name]["2_to_3"]["pooled_point_delta"] for name in ("density_B3", "F1_majority")]
    g3 = min(g3_points) > config["stage0"]["gates"]["E_G3_min_pooled_2_to_3_point_each_aggregator"]
    write_jsonl_fsynced(RAW_PATH, raw)
    output = {
        "schema_version": 1,
        "status": "PASS_EVID_STAGE0_COMPLETE",
        "rho_zero_rowwise_mismatches": 0,
        "E_K2": False,
        "lineage_transitions": transitions,
        "oracle": {
            "output_correct_accuracy": oracle_accuracy,
            "contains_correct_accuracy": contains_accuracy,
            "nested_dev_selection_accuracy": dev_selection,
            "output_oracle_gain": oracle_accuracy - dev_selection,
        },
        "disagreement": {"rows": disagreement, "total_rows": len(row_ids), "fraction": disagreement_rate, "label_free": True},
        "diagonal_path": {"step_change_rows": path_changes},
        "gates": {"E_G1": g1, "E_G2": g2, "E_G3": g3, "E_G3_min_point": min(g3_points)},
        "proceed_stage1": bool(g1 and g2),
        "stage2_permanently_blocked": not g3,
        "raw": {"path": str(RAW_PATH.relative_to(ROOT)), "rows": len(raw), "bytes": RAW_PATH.stat().st_size, "sha256": sha256_file(RAW_PATH), "write_flush_fsync_per_row": True},
    }
    temporary = OUTPUT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUTPUT_PATH)
    print(json.dumps({"status": output["status"], "gates": output["gates"], "proceed_stage1": output["proceed_stage1"], "stage2_blocked": output["stage2_permanently_blocked"]}, indent=2))


if __name__ == "__main__":
    main()