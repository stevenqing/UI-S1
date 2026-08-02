import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parents[1]
ROOT = RUN_DIR.parents[2]
ALLOCATION_DIR = ROOT / "runs/allocation-law/2026-08-01"
sys.path.insert(0, str(RUN_DIR))
sys.path.insert(0, str(ROOT / "runs/diversity-axis/2026-08-02"))
sys.path.insert(0, str(ALLOCATION_DIR))
from zoom_port import adaptive_crop, deterministic_seed, gate, point_to_box
from x3_curve_stats import load_sources, reconstruct
from allocation_eval import compact_evaluation, failure_statistics, group_folds
from run_l2 import stratified_group_sample_counts


MODEL_SPECS = {
    "GTA1-7B": "701bedc80b447863bd60e3318ae44f6cbbfafd78",
    "Qwen3-VL-8B-Instruct": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    "UI-TARS-7B-SFT": "3434901a9dd04dd3625617d839a5724fe5e2db20",
}
MODEL_ORDER = tuple(MODEL_SPECS)
MODEL_PATHS = {
    "GTA1-7B": ROOT / "runs/collision-law/2026-07-30/w3_assets/GTA1-7B",
    "Qwen3-VL-8B-Instruct": ROOT / "runs/ccm-h2h/2026-07-31/h3/models/Qwen3-VL-8B-Instruct",
    "UI-TARS-7B-SFT": ROOT / "runs/mind2web-tongui/2026-07-28/models/UI-TARS-7B-SFT",
}
METHODS = ("B3_mvp", "M1_ccm", "pass_at_n")
SEED = 20260802
RESAMPLES = 10000


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(paths):
    rows = []
    for path in sorted(paths):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def load_trace(paths, cell, model, expected_forwards):
    rows = read_jsonl(paths)
    indexed = {}
    expected_model_hash = sha256_file(MODEL_PATHS[model] / "model.safetensors.index.json")
    for row in rows:
        row_id = row["id"]
        if row_id in indexed:
            raise ValueError(f"X2 duplicate identity: {cell}/{model}/{row_id}")
        if "target_bbox" in row or "bbox" in row:
            raise ValueError(f"X2 trace contains target: {cell}/{model}/{row_id}")
        if row["cell"] != cell or row["model_id"] != model or row["model_revision"] != MODEL_SPECS[model]:
            raise ValueError(f"X2 model/cell identity mismatch: {row_id}")
        if row["model_index_sha256"] != expected_model_hash:
            raise ValueError(f"X2 model index hash mismatch: {cell}/{model}/{row_id}")
        stable_index = row["stable_index"]
        if row["num_shards"] != 2 or row["shard_index"] != stable_index % 2:
            raise ValueError(f"X2 shard mismatch: {cell}/{model}/{row_id}")
        predictions = row["predictions"]
        if row["forward_count"] != expected_forwards or len(predictions) != expected_forwards:
            raise ValueError(f"X2 forward budget mismatch: {cell}/{model}/{row_id}")
        actual_valid = sum(prediction["point"] is not None for prediction in predictions)
        if row["valid_candidate_count"] != actual_valid:
            raise ValueError(f"X2 valid-candidate count mismatch: {cell}/{model}/{row_id}")
        if canonical_hash(predictions) != row["prediction_sha256"]:
            raise ValueError(f"X2 prediction hash mismatch: {cell}/{model}/{row_id}")
        expected_pairs = [(chain, slot) for chain in range(3 if cell == "Q2" else 1) for slot in range(4)]
        if [(prediction["chain_index"], prediction["slot"]) for prediction in predictions] != expected_pairs:
            raise ValueError(f"X2 chain/slot order mismatch: {cell}/{model}/{row_id}")
        for prediction in predictions:
            chain, slot = prediction["chain_index"], prediction["slot"]
            expected_seed = deterministic_seed(row_id, cell, model, chain, slot)
            if prediction["seed"] != expected_seed:
                raise ValueError(f"X2 seed mismatch: {cell}/{model}/{row_id}/{chain}/{slot}")
            point = prediction["point"]
            if point is None:
                if prediction["box"] is not None:
                    raise ValueError(f"X2 invalid point has non-null box: {cell}/{model}/{row_id}/{chain}/{slot}")
            elif len(point) != 2 or not all(math.isfinite(float(value)) for value in point):
                raise ValueError(f"X2 invalid point: {cell}/{model}/{row_id}/{chain}/{slot}")
            elif prediction["box"] != point_to_box(point, *row["img_size"]):
                raise ValueError(f"X2 point-box mismatch: {cell}/{model}/{row_id}/{chain}/{slot}")
            if not math.isfinite(prediction["confidence"]) or not 0 <= prediction["confidence"] <= 1:
                raise ValueError(f"X2 invalid confidence: {cell}/{model}/{row_id}/{chain}/{slot}")
            if slot < 3 and (prediction["branch"] != "global_sample" or prediction["temperature"] != 0.9):
                raise ValueError(f"X2 global slot mismatch: {cell}/{model}/{row_id}/{chain}/{slot}")
            if slot < 3 and prediction["region"] != [0, 0, *row["img_size"]]:
                raise ValueError(f"X2 global region mismatch: {cell}/{model}/{row_id}/{chain}/{slot}")
            if slot == 3 and prediction["branch"] == "adaptive_crop_refine" and prediction["temperature"] != 0.0:
                raise ValueError(f"X2 crop temperature mismatch: {cell}/{model}/{row_id}/{chain}")
            if slot == 3 and prediction["branch"] == "global_confirmation" and prediction["temperature"] != 0.9:
                raise ValueError(f"X2 confirmation temperature mismatch: {cell}/{model}/{row_id}/{chain}")
        if len(row["chain_reports"]) != (3 if cell == "Q2" else 1):
            raise ValueError(f"X2 chain report count mismatch: {cell}/{model}/{row_id}")
        for report in row["chain_reports"]:
            chain = report["chain_index"]
            chain_predictions = predictions[chain * 4:(chain + 1) * 4]
            expected_gate = gate([{"box": value["box"], "confidence": value["confidence"]} for value in chain_predictions[:3]])
            expected_crop = adaptive_crop(
                [{"box": value["box"], "confidence": value["confidence"]} for value in chain_predictions[:3]],
                *row["img_size"],
            ) if not expected_gate["reliable"] else None
            expected_branch = "adaptive_crop_refine" if expected_crop is not None else "global_confirmation"
            for key in ("reliable", "spatial_consistency", "mean_confidence", "score", "valid_candidates"):
                if report[key] != expected_gate[key]:
                    raise ValueError(f"X2 gate recomputation mismatch: {cell}/{model}/{row_id}/{chain}/{key}")
            if report["crop"] != expected_crop or report["branch"] != expected_branch or chain_predictions[3]["branch"] != expected_branch:
                raise ValueError(f"X2 crop/branch recomputation mismatch: {cell}/{model}/{row_id}/{chain}")
            expected_region = expected_crop if expected_crop is not None else [0, 0, *row["img_size"]]
            if chain_predictions[3]["region"] != expected_region:
                raise ValueError(f"X2 branch region mismatch: {cell}/{model}/{row_id}/{chain}")
        indexed[row_id] = row
    if len(indexed) != 1581:
        raise ValueError(f"X2 requires 1,581 identities: {cell}/{model}, found {len(indexed)}")
    if sorted(row["stable_index"] for row in indexed.values()) != list(range(1581)):
        raise ValueError(f"X2 stable index mismatch: {cell}/{model}")
    return indexed


def validate_source_identity(traces, gta1):
    stable_indices = {row_id: index for index, row_id in enumerate(sorted(gta1))}
    for trace in traces:
        if set(trace) != set(gta1):
            raise ValueError("X2 trace/GTA1 identity mismatch")
        for row_id, row in trace.items():
            source = gta1[row_id]
            expected = {
                "stable_index": stable_indices[row_id],
                "application": source["application"],
                "img_filename": source["img_filename"],
                "img_size": source["img_size"],
                "instruction": source["instruction"],
            }
            if any(row[key] != value for key, value in expected.items()):
                raise ValueError(f"X2 trace source metadata mismatch: {row_id}")


def candidate(prediction, model, view_index):
    point = prediction["point"] if prediction["point"] is not None else [0.0, 0.0]
    return {
        "model": model,
        "view_index": view_index,
        "point": [float(value) for value in point],
        "region": list(prediction["region"]),
        "coverage": 0.0,
    }


def build_q2(gta1, trace):
    rows = []
    for row_id in sorted(gta1):
        source = gta1[row_id]
        predictions = trace[row_id]["predictions"]
        rows.append({
            "id": row_id,
            "application": source["application"],
            "target_bbox": source["target_bbox"],
            "candidates": [candidate(prediction, "GTA1-7B", index) for index, prediction in enumerate(predictions)],
        })
    return rows


def build_q4(gta1, traces):
    rows = []
    for row_id in sorted(gta1):
        source = gta1[row_id]
        candidates = []
        for slot in range(4):
            for model in MODEL_ORDER:
                candidates.append(candidate(traces[model][row_id]["predictions"][slot], model, slot))
        rows.append({
            "id": row_id,
            "application": source["application"],
            "target_bbox": source["target_bbox"],
            "candidates": candidates,
        })
    return rows


def interaction_classification(interval):
    lower, upper = interval
    if lower > 0:
        return "SUPER_ADDITIVE"
    if upper < 0:
        return "SUB_ADDITIVE"
    return "NEAR_ADDITIVE"


def bootstrap_interactions(rows, evaluations):
    mapping, _ = group_folds(rows)
    groups = sorted(mapping)
    group_index = {group: index for index, group in enumerate(groups)}
    row_counts = np.zeros(len(groups), dtype=np.int64)
    for row in rows:
        row_counts[group_index[row["application"]]] += 1
    sample_counts = stratified_group_sample_counts(groups, mapping, RESAMPLES, np.random.default_rng(SEED))
    denominators = sample_counts @ row_counts
    reports = {}
    for method in METHODS:
        cell_success = {}
        for cell in ("Q1", "Q2", "Q3", "Q4"):
            values = np.zeros(len(groups), dtype=np.int64)
            outputs = evaluations[cell]["outputs"][method]
            for row in rows:
                values[group_index[row["application"]]] += int(outputs[row["id"]])
            cell_success[cell] = values
        contrast = cell_success["Q4"] - cell_success["Q3"] - cell_success["Q2"] + cell_success["Q1"]
        bootstrap = (sample_counts @ contrast) / denominators
        point = (
            evaluations["Q4"]["accuracy"][method]
            - evaluations["Q3"]["accuracy"][method]
            - evaluations["Q2"]["accuracy"][method]
            + evaluations["Q1"]["accuracy"][method]
        )
        interval = [float(np.quantile(bootstrap, 0.005)), float(np.quantile(bootstrap, 0.995))]
        reports[method] = {
            "point": point,
            "bootstrap_mean": float(np.mean(bootstrap)),
            "ci_99": interval,
            "classification": interaction_classification(interval),
            "p_interaction_negative": float(np.mean(bootstrap < 0)),
            "resamples": RESAMPLES,
            "seed": SEED,
        }
    return reports


def gate_diagnostics(q2_trace, q4_traces):
    reports = {}
    for cell, traces in (("Q2", {"GTA1-7B": q2_trace}), ("Q4", q4_traces)):
        chain_reports = [report for trace in traces.values() for row in trace.values() for report in row["chain_reports"]]
        branches = Counter(report["branch"] for report in chain_reports)
        reports[cell] = {
            "chains": len(chain_reports),
            "adaptive_trigger_rate": branches["adaptive_crop_refine"] / len(chain_reports),
            "branch_counts": dict(sorted(branches.items())),
            "mean_gate_score": float(np.mean([report["score"] for report in chain_reports])),
            "realized_forwards_per_row": 12,
            "invalid_forwards": sum(
                prediction["point"] is None
                for trace in traces.values()
                for row in trace.values()
                for prediction in row["predictions"]
            ),
            "rows_with_invalid_forwards": len({
                row_id
                for trace in traces.values()
                for row_id, row in trace.items()
                if any(prediction["point"] is None for prediction in row["predictions"])
            }),
        }
    return reports


def area_strata(gta1, evaluations):
    area = {}
    for row_id, row in gta1.items():
        width, height = row["img_size"]
        left, top, right, bottom = row["target_bbox"]
        area[row_id] = (right - left) * (bottom - top) / (width * height)
    ordered = sorted(area, key=lambda row_id: (area[row_id], row_id))
    reports = []
    for index, values in enumerate(np.array_split(np.asarray(ordered, dtype=object), 5)):
        ids = values.tolist()
        reports.append({
            "bin": index,
            "rows": len(ids),
            "area_ratio_mean": float(np.mean([area[row_id] for row_id in ids])),
            "Q2_minus_Q1_M1": float(np.mean([evaluations["Q2"]["outputs"]["M1_ccm"][row_id] - evaluations["Q1"]["outputs"]["M1_ccm"][row_id] for row_id in ids])),
            "Q4_minus_Q3_M1": float(np.mean([evaluations["Q4"]["outputs"]["M1_ccm"][row_id] - evaluations["Q3"]["outputs"]["M1_ccm"][row_id] for row_id in ids])),
        })
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q2", type=Path, nargs="+", required=True)
    parser.add_argument("--q4-gta1", type=Path, nargs="+", required=True)
    parser.add_argument("--q4-qwen3", type=Path, nargs="+", required=True)
    parser.add_argument("--q4-uitars", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gta1, generated, units = load_sources()
    fixed_rows, fixed_evaluations = reconstruct(gta1, generated, units)
    q2_trace = load_trace(args.q2, "Q2", "GTA1-7B", 12)
    q4_traces = {
        "GTA1-7B": load_trace(args.q4_gta1, "Q4", "GTA1-7B", 4),
        "Qwen3-VL-8B-Instruct": load_trace(args.q4_qwen3, "Q4", "Qwen3-VL-8B-Instruct", 4),
        "UI-TARS-7B-SFT": load_trace(args.q4_uitars, "Q4", "UI-TARS-7B-SFT", 4),
    }
    validate_source_identity([q2_trace, *q4_traces.values()], gta1)
    cell_rows = {
        "Q1": fixed_rows["v_only"][12],
        "Q2": build_q2(gta1, q2_trace),
        "Q3": fixed_rows["mixed"][12],
        "Q4": build_q4(gta1, q4_traces),
    }
    evaluations = {
        "Q1": fixed_evaluations["v_only"][12],
        "Q2": compact_evaluation(cell_rows["Q2"]),
        "Q3": fixed_evaluations["mixed"][12],
        "Q4": compact_evaluation(cell_rows["Q4"]),
    }
    interactions = bootstrap_interactions(cell_rows["Q1"], evaluations)
    primary = interactions["M1_ccm"]
    accuracy = {cell: evaluation["accuracy"] for cell, evaluation in evaluations.items()}
    highest = max(accuracy, key=lambda cell: (accuracy[cell]["M1_ccm"], cell))
    result = {
        "schema_version": 1,
        "status": "PASS",
        "method": "UI_Zoomer_fixed12_microchains_algorithm_level_extension",
        "sanity_anchor": {
            "benchmark": "ScreenSpot-Pro",
            "model": "Qwen2.5-VL-7B-Instruct",
            "baseline_accuracy": 0.276,
            "UI_Zoomer_accuracy": 0.410,
            "tolerance_absolute": 0.01,
            "status": "NOT_RUN_MISSING_OFFICIAL_BACKBONE_CHECKPOINT_AND_GENERATIONS",
        },
        "rows": 1581,
        "forward_budget": 12,
        "accuracy": accuracy,
        "interactions": interactions,
        "failure_kappa": {
            cell: failure_statistics(rows)["mean_pairwise_kappa"]
            for cell, rows in cell_rows.items()
        },
        "gate_diagnostics": gate_diagnostics(q2_trace, q4_traces),
        "area_strata": area_strata(gta1, evaluations),
        "prediction": {
            "highest_cell": highest,
            "Q4_highest": highest == "Q4",
            "primary_interaction_classification": primary["classification"],
            "composability_success": highest == "Q4" and primary["classification"] != "SUB_ADDITIVE",
        },
        "kill_conditions": {"X-K1": primary["classification"] == "SUB_ADDITIVE"},
        "claim_boundary": (
            "Budget-normalized K=3 microchains are an algorithm-level UI-Zoomer extension, not the official K=8 protocol. "
            "Q2/Q4 use UI-Zoomer resize and standard checkpoint classes; Q1/Q3 are frozen MVP/fixed-region upstream pools."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "accuracy": accuracy,
        "interactions": interactions,
        "failure_kappa": result["failure_kappa"],
        "prediction": result["prediction"],
        "kill_conditions": result["kill_conditions"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()