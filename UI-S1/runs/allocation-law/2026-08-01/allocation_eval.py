import hashlib
import json
import math
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml
import numpy as np
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[3]
H3_DIR = ROOT / "runs/ccm-h2h/2026-07-31/h3"
H1_DIR = ROOT / "runs/ccm-h2h/2026-07-31/h1"
sys.path.insert(0, str(H3_DIR))
sys.path.insert(0, str(H1_DIR))
from h3_eval import evaluate_pool, group_folds, point_in_bbox


MODEL_NAMES = {
    "GTA1-7B": "GTA1-7B",
    "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B-Instruct",
    "UI-TARS-7B-SFT": "UI-TARS-7B-SFT",
}
EXPECTED_REVISIONS = {
    "Qwen3-VL-8B-Instruct": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    "UI-TARS-7B-SFT": "3434901a9dd04dd3625617d839a5724fe5e2db20",
}
EXPECTED_ROWS = 1581
BUDGETS = (4, 8, 12, 16, 24)
JOINT_BUDGETS = (4, 8, 12, 16)
MDE = 0.007043345177520599


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_jsonl(paths):
    rows = []
    for path in sorted(paths):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def index_unique(rows, source, expected_rows=EXPECTED_ROWS):
    indexed = {}
    for row in rows:
        row_id = row["id"]
        if row_id in indexed:
            raise ValueError(f"{source} duplicate identity: {row_id}")
        indexed[row_id] = row
    if len(indexed) != expected_rows:
        raise ValueError(f"{source} requires {expected_rows} identities, found {len(indexed)}")
    return indexed


def load_manifest(path, expected_rows=EXPECTED_ROWS):
    rows = load_jsonl([path])
    indexed = index_unique(rows, "N12 manifest", expected_rows)
    ordered = sorted(rows, key=lambda row: row["id"])
    for stable_index, row in enumerate(ordered):
        if row["stable_index"] != stable_index or len(row["regions"]) != 12:
            raise ValueError(f"N12 manifest index/region mismatch: {row['id']}")
    return indexed


def load_gta1(shard_root, manifest, expected_rows=EXPECTED_ROWS):
    rows = load_jsonl(shard_root.glob("shard-*.jsonl"))
    indexed = index_unique(rows, "GTA1 superset", expected_rows)
    if set(indexed) != set(manifest):
        raise ValueError("GTA1/manifest identity mismatch")
    output = {}
    for row_id, row in indexed.items():
        candidates = row["candidates"]
        if canonical_hash(candidates) != row["candidate_sha256"] or len(candidates) < 16:
            raise ValueError(f"GTA1 candidate integrity mismatch: {row_id}")
        if [candidate["region"] for candidate in candidates[:12]] != manifest[row_id]["regions"]:
            raise ValueError(f"GTA1/N12 region mismatch: {row_id}")
        output[row_id] = row
    return output


def load_model_views(old_root, extended_paths, manifest, model, expected_rows=EXPECTED_ROWS):
    old = index_unique(load_jsonl(old_root.glob("shard-*.jsonl")), f"{model} views0-3", expected_rows)
    extended = index_unique(load_jsonl(extended_paths), f"{model} views4-11", expected_rows)
    if set(old) != set(manifest) or set(extended) != set(manifest):
        raise ValueError(f"{model} identity mismatch")
    revision = EXPECTED_REVISIONS[model]
    output = {}
    for row_id in sorted(manifest):
        old_row, new_row = old[row_id], extended[row_id]
        stable_index = manifest[row_id]["stable_index"]
        if old_row["model_id"] != model or new_row["model_id"] != model:
            raise ValueError(f"{model} model identity mismatch: {row_id}")
        if old_row["model_revision"] != revision or new_row["model_revision"] != revision:
            raise ValueError(f"{model} revision mismatch: {row_id}")
        for source_name, source in (("old", old_row), ("extended", new_row)):
            if source["stable_index"] != stable_index:
                raise ValueError(f"{model} {source_name} stable-index mismatch: {row_id}")
            if source["num_shards"] != 4 or source["shard_index"] != stable_index % 4:
                raise ValueError(f"{model} {source_name} shard mismatch: {row_id}")
            if canonical_hash(source["predictions"]) != source["prediction_sha256"]:
                raise ValueError(f"{model} {source_name} prediction hash mismatch: {row_id}")
        if "target_bbox" in new_row:
            raise ValueError(f"{model} extended trace contains target field: {row_id}")
        old_predictions = old_row["predictions"]
        new_predictions = new_row["predictions"]
        predictions = old_predictions + new_predictions
        if [item["view_index"] for item in old_predictions] != list(range(4)):
            raise ValueError(f"{model} old view mismatch: {row_id}")
        if [item["view_index"] for item in new_predictions] != list(range(4, 12)):
            raise ValueError(f"{model} extended view mismatch: {row_id}")
        if new_row["shared_region_candidate_sha256"] != manifest[row_id]["shared_region_candidate_sha256"]:
            raise ValueError(f"{model} N12 hash mismatch: {row_id}")
        for view_index, prediction in enumerate(predictions):
            if prediction["region"] != manifest[row_id]["regions"][view_index]:
                raise ValueError(f"{model} region mismatch: {row_id}/view{view_index}")
            point = prediction["point"]
            if len(point) != 2 or not all(math.isfinite(float(value)) for value in point):
                raise ValueError(f"{model} invalid point: {row_id}/view{view_index}")
        output[row_id] = predictions
    return output


def candidate(model, view_index, source, coverage):
    return {
        "model": model,
        "view_index": view_index,
        "point": [float(value) for value in source["point"]],
        "region": list(source["region"]),
        "coverage": float(coverage),
    }


def candidate_for_unit(row_id, model, view_index, gta1, generated):
    if model == "GTA1-7B":
        source = gta1[row_id]["candidates"][view_index]
        return candidate(model, view_index, source, source["coverage"])
    source = generated[model][row_id][view_index]
    if "coverage" in source:
        raise ValueError(f"unexpected generated-model coverage: {model}/view{view_index}/{row_id}")
    return candidate(model, view_index, source, 0)


def parse_unit(unit):
    model, view = unit.rsplit("/view", 1)
    if model not in MODEL_NAMES:
        raise ValueError(f"unknown allocation model: {model}")
    return model, int(view)


def build_pool(gta1, generated, units):
    parsed = [parse_unit(unit) if isinstance(unit, str) else unit for unit in units]
    if len(parsed) != len(set(parsed)):
        raise ValueError("candidate unit duplication")
    rows = []
    for row_id in sorted(gta1):
        source = gta1[row_id]
        candidates = [
            candidate_for_unit(row_id, model, view_index, gta1, generated)
            for model, view_index in parsed
        ]
        rows.append({
            "id": row_id,
            "application": source["application"],
            "target_bbox": source["target_bbox"],
            "candidates": candidates,
        })
    return rows


def load_l1_units(path):
    config = yaml.safe_load(path.read_text())
    sequence = config["allocation_sequence"]
    units = {}
    for budget in BUDGETS:
        frozen = config["budget_prefixes"][budget]
        expected = sequence[:budget]
        if isinstance(frozen, list) and frozen != expected:
            raise ValueError(f"L1 frozen prefix mismatch at N={budget}")
        if isinstance(frozen, str):
            expected_token = {
                12: "first_12_allocation_sequence",
                16: "first_16_allocation_sequence",
                24: "full_allocation_sequence",
            }.get(budget)
            if frozen != expected_token:
                raise ValueError(f"L1 prefix token mismatch at N={budget}")
        units[budget] = expected
    return units


def l2_units(path):
    config = yaml.safe_load(path.read_text())
    if config["budget"] != 12 or len(config["pools"]) != 8:
        raise ValueError("L2 frozen pool count/budget mismatch")
    units = {
        "gta1_12views": [("GTA1-7B", view) for view in range(12)],
        "qwen3_12views": [("Qwen3-VL-8B-Instruct", view) for view in range(12)],
        "uitars_12views": [("UI-TARS-7B-SFT", view) for view in range(12)],
        "gta1_qwen3_6x2": [(model, view) for view in range(6) for model in ("GTA1-7B", "Qwen3-VL-8B-Instruct")],
        "gta1_uitars_6x2": [(model, view) for view in range(6) for model in ("GTA1-7B", "UI-TARS-7B-SFT")],
        "qwen3_uitars_6x2": [(model, view) for view in range(6) for model in ("Qwen3-VL-8B-Instruct", "UI-TARS-7B-SFT")],
        "three_lineages_4x3": [(model, view) for view in range(4) for model in MODEL_NAMES],
        "three_lineages_4x3_shuffled_control": [(model, view) for view in range(4, 8) for model in MODEL_NAMES],
    }
    expected_descriptors = {
        "gta1_12views": "GTA1_views_0_to_11",
        "qwen3_12views": "Qwen3_views_0_to_11",
        "uitars_12views": "UI-TARS_views_0_to_11",
        "gta1_qwen3_6x2": "round_robin_GTA1_Qwen3_views_0_to_5",
        "gta1_uitars_6x2": "round_robin_GTA1_UITARS_views_0_to_5",
        "qwen3_uitars_6x2": "round_robin_Qwen3_UITARS_views_0_to_5",
        "three_lineages_4x3": "round_robin_all_models_views_0_to_3",
        "three_lineages_4x3_shuffled_control": "each_model_views_4_to_7",
    }
    if list(units) != list(config["pools"]):
        raise ValueError("L2 implementation/config pool mismatch")
    for pool_name, descriptor in expected_descriptors.items():
        if config["pools"][pool_name]["units"] != descriptor:
            raise ValueError(f"L2 descriptor mismatch: {pool_name}")
    if any(len(pool) != 12 or len(set(pool)) != 12 for pool in units.values()):
        raise ValueError("L2 pool budget/duplication mismatch")
    return units


def compact_evaluation(rows):
    result = evaluate_pool(rows)
    accuracy = dict(result["accuracy"])
    outputs = dict(result["outputs"])
    accuracy["pass_at_n"] = accuracy.pop("pass_at_12")
    outputs["pass_at_n"] = outputs.pop("pass_at_12")
    folds = []
    for fold in result["folds"]:
        value = dict(fold)
        value["accuracy"] = dict(value["accuracy"])
        value["accuracy"]["pass_at_n"] = value["accuracy"].pop("pass_at_12")
        folds.append(value)
    return {
        "rows": result["rows"],
        "fold_rows": result["fold_rows"],
        "folds": folds,
        "accuracy": accuracy,
        "outputs": outputs,
    }


def l1_predictions(curves):
    intervals = tuple(zip(JOINT_BUDGETS, JOINT_BUDGETS[1:]))
    increments = {}
    rule_passes = {}
    for rule in ("B3_mvp", "M1_ccm"):
        records = []
        for left, right in intervals:
            v_increment = curves["v_only"][right][rule] - curves["v_only"][left][rule]
            mixed_increment = curves["mixed"][right][rule] - curves["mixed"][left][rule]
            records.append({
                "interval": f"{left}->{right}",
                "v_only_increment": v_increment,
                "mixed_increment": mixed_increment,
                "satisfied": v_increment < MDE and mixed_increment > MDE,
            })
        increments[rule] = records
        rule_passes[rule] = any(record["satisfied"] for record in records)

    allocation_gaps = []
    for budget in JOINT_BUDGETS:
        b3_gap = curves["mixed"][budget]["B3_mvp"] - curves["v_only"][budget]["B3_mvp"]
        m1_gap = curves["mixed"][budget]["M1_ccm"] - curves["v_only"][budget]["M1_ccm"]
        same_sign = (b3_gap > 0 and m1_gap > 0) or (b3_gap < 0 and m1_gap < 0)
        magnitude = abs(b3_gap - m1_gap) < min(abs(b3_gap), abs(m1_gap))
        allocation_gaps.append({
            "budget": budget,
            "B3_mixed_minus_v_only": b3_gap,
            "M1_mixed_minus_v_only": m1_gap,
            "same_sign": same_sign,
            "magnitude_condition": magnitude,
            "satisfied": same_sign and magnitude,
        })
    p_l1a_passes = sum(rule_passes.values())
    p_l1b = all(record["satisfied"] for record in allocation_gaps)
    return {
        "P-L1a": {
            "mde": MDE,
            "increments": increments,
            "rule_passes": rule_passes,
            "status": "PASS" if p_l1a_passes == 2 else "PARTIAL_PASS" if p_l1a_passes == 1 else "FAIL",
        },
        "P-L1b": {
            "allocation_gaps": allocation_gaps,
            "status": "PASS" if p_l1b else "FAIL",
        },
        "kill_conditions": {
            "L-K1": p_l1a_passes == 0,
            "L-K2": not p_l1b,
        },
    }


def cohen_kappa_from_counts(total, left_failures, right_failures, overlap):
    if total <= 0:
        return None
    observed = (overlap + total - left_failures - right_failures + overlap) / total
    left_rate = left_failures / total
    right_rate = right_failures / total
    expected = left_rate * right_rate + (1 - left_rate) * (1 - right_rate)
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1 - expected)


def failure_statistics(rows, row_filter=None):
    selected = rows if row_filter is None else [row for row in rows if row_filter(row)]
    if not selected:
        raise ValueError("empty failure-statistics split")
    failures = np.asarray([
        [not point_in_bbox(candidate["point"], row["target_bbox"]) for candidate in row["candidates"]]
        for row in selected
    ], dtype=np.int64)
    pair_values = []
    pair_counts = []
    for left in range(failures.shape[1]):
        for right in range(left + 1, failures.shape[1]):
            left_count = int(failures[:, left].sum())
            right_count = int(failures[:, right].sum())
            overlap = int((failures[:, left] * failures[:, right]).sum())
            value = cohen_kappa_from_counts(len(selected), left_count, right_count, overlap)
            pair_counts.append((left_count, right_count, overlap))
            if value is not None:
                pair_values.append(value)
    if not pair_values:
        raise ValueError("all candidate failure pairs are constant")
    return {
        "rows": len(selected),
        "mean_pairwise_kappa": float(np.mean(pair_values)),
        "finite_pairs": len(pair_values),
        "null_pairs": len(pair_counts) - len(pair_values),
        "pair_counts": pair_counts,
    }


def matched_marginal_permutation(statistics, rng, permutations=1000):
    observed = statistics["mean_pairwise_kappa"]
    null_means = []
    total = statistics["rows"]
    for _ in range(permutations):
        values = []
        for left, right, _ in statistics["pair_counts"]:
            overlap = int(rng.hypergeometric(left, total - left, right))
            value = cohen_kappa_from_counts(total, left, right, overlap)
            if value is not None:
                values.append(value)
        null_means.append(float(np.mean(values)))
    null = np.asarray(null_means)
    return {
        "permutations": permutations,
        "null_mean": float(null.mean()),
        "null_ci_99": [float(np.quantile(null, 0.005)), float(np.quantile(null, 0.995))],
        "p_greater_equal_observed": float((1 + np.sum(null >= observed)) / (permutations + 1)),
    }


def spearman(x_values, y_values):
    result = spearmanr(x_values, y_values)
    return {"rho": float(result.statistic), "p_value": float(result.pvalue), "observations": len(x_values)}


def rowwise_spearman(left, right):
    left_rank = rankdata(left, axis=1)
    right_rank = rankdata(right, axis=1)
    left_centered = left_rank - left_rank.mean(axis=1, keepdims=True)
    right_centered = right_rank - right_rank.mean(axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(np.sum(left_centered**2, axis=1) * np.sum(right_centered**2, axis=1))
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
