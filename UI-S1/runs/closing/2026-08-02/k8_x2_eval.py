import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
DIVERSITY_DIR = ROOT / "runs/diversity-axis/2026-08-02"
ALLOCATION_DIR = ROOT / "runs/allocation-law/2026-08-01"
sys.path.insert(0, str(DIVERSITY_DIR / "x2"))
sys.path.insert(0, str(DIVERSITY_DIR))
sys.path.insert(0, str(ALLOCATION_DIR))
from generate_microchains import MODEL_SPECS
from zoom_port import adaptive_crop, deterministic_seed, gate, point_to_box
from x2_composability import bootstrap_interactions, interaction_classification
from x3_curve_stats import load_sources
from allocation_eval import compact_evaluation, failure_statistics


MODEL_ORDER = ("GTA1-7B", "Qwen3-VL-8B-Instruct", "UI-TARS-7B-SFT")
MODEL_PATHS = {
    "GTA1-7B": ROOT / "runs/collision-law/2026-07-30/w3_assets/GTA1-7B",
    "Qwen3-VL-8B-Instruct": ROOT / "runs/ccm-h2h/2026-07-31/h3/models/Qwen3-VL-8B-Instruct",
    "UI-TARS-7B-SFT": ROOT / "runs/mind2web-tongui/2026-07-28/models/UI-TARS-7B-SFT",
}
REVISION = {value["id"]: value["revision"] for value in MODEL_SPECS.values()}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_trace(paths, family, model, expected_chains, num_shards):
    rows = {}
    expected_model_hash = sha256_file(MODEL_PATHS[model] / "model.safetensors.index.json")
    for path in sorted(paths):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = row["id"]
            if row_id in rows:
                raise ValueError(f"K8 duplicate identity: {family}/{model}/{row_id}")
            if "bbox" in row or "target_bbox" in row:
                raise ValueError(f"K8 target leak: {family}/{model}/{row_id}")
            if row["family"] != family or row["model_id"] != model or row["model_revision"] != REVISION[model]:
                raise ValueError(f"K8 model/family mismatch: {row_id}")
            if row["model_index_sha256"] != expected_model_hash:
                raise ValueError(f"K8 model hash mismatch: {row_id}")
            if row["num_shards"] != num_shards or row["shard_index"] != row["stable_index"] % num_shards:
                raise ValueError(f"K8 shard mismatch: {row_id}")
            chains = row["chains"]
            if len(chains) != expected_chains or row["fixed_cell_forwards"] != 9 * expected_chains or row["adaptive_cell_forwards"] != 9 * expected_chains:
                raise ValueError(f"K8 chain/budget mismatch: {row_id}")
            if canonical_hash(chains) != row["chains_sha256"]:
                raise ValueError(f"K8 chain hash mismatch: {row_id}")
            width, height = row["img_size"]
            for chain_index, chain in enumerate(chains):
                if chain["chain_index"] != chain_index or len(chain["global_K8"]) != 8:
                    raise ValueError(f"K8 chain order mismatch: {row_id}/{chain_index}")
                expected_seed = deterministic_seed(row_id, family, model, chain_index, "global_K8")
                if chain["global_seed"] != expected_seed:
                    raise ValueError(f"K8 global seed mismatch: {row_id}/{chain_index}")
                candidates = []
                for sample_index, value in enumerate(chain["global_K8"]):
                    if value["sample_index"] != sample_index or value["seed"] != expected_seed:
                        raise ValueError(f"K8 sample order mismatch: {row_id}/{chain_index}/{sample_index}")
                    point = value["point"]
                    if point is not None and (len(point) != 2 or not all(math.isfinite(float(x)) for x in point)):
                        raise ValueError(f"K8 invalid global point: {row_id}/{chain_index}/{sample_index}")
                    if value["box"] != point_to_box(point, width, height):
                        raise ValueError(f"K8 global box mismatch: {row_id}/{chain_index}/{sample_index}")
                    candidates.append({"box": value["box"], "confidence": value["confidence"]})
                expected_gate = gate(candidates)
                expected_crop = adaptive_crop(candidates, width, height) if not expected_gate["reliable"] else None
                report = chain["report"]
                for key in ("reliable", "spatial_consistency", "mean_confidence", "score", "valid_candidates"):
                    if report[key] != expected_gate[key]:
                        raise ValueError(f"K8 gate mismatch: {row_id}/{chain_index}/{key}")
                expected_use = "refinement" if expected_crop is not None else "confirmation"
                if report["crop"] != expected_crop or report["adaptive_uses"] != expected_use:
                    raise ValueError(f"K8 branch mismatch: {row_id}/{chain_index}")
                confirmation = chain["confirmation"]
                if confirmation["seed"] != deterministic_seed(row_id, family, model, chain_index, "confirmation"):
                    raise ValueError(f"K8 confirmation seed mismatch: {row_id}/{chain_index}")
                if confirmation["box"] != point_to_box(confirmation["point"], width, height):
                    raise ValueError(f"K8 confirmation box mismatch: {row_id}/{chain_index}")
                refinement = chain["refinement"]
                if expected_crop is None:
                    if refinement is not None:
                        raise ValueError(f"K8 unexpected refinement: {row_id}/{chain_index}")
                else:
                    if refinement is None or refinement["region"] != expected_crop:
                        raise ValueError(f"K8 refinement mismatch: {row_id}/{chain_index}")
            rows[row_id] = row
    if len(rows) != 1581:
        raise ValueError(f"K8 requires 1,581 identities: {family}/{model}, found {len(rows)}")
    return rows


def candidate(value, model, view_index):
    point = value["point"] if value["point"] is not None else [0.0, 0.0]
    return {"model": model, "view_index": view_index, "point": list(map(float, point)), "region": list(value["region"]), "coverage": 0.0}


def chain_candidates(chain, model, chain_index, adaptive):
    values = [candidate(value, model, chain_index * 9 + index) for index, value in enumerate(chain["global_K8"])]
    branch = chain["refinement"] if adaptive and chain["refinement"] is not None else chain["confirmation"]
    values.append(candidate(branch, model, chain_index * 9 + 8))
    return values


def build_single(gta1, trace, adaptive):
    return [{
        "id": row_id,
        "application": gta1[row_id]["application"],
        "target_bbox": gta1[row_id]["target_bbox"],
        "candidates": [value for chain_index, chain in enumerate(trace[row_id]["chains"]) for value in chain_candidates(chain, "GTA1-7B", chain_index, adaptive)],
    } for row_id in sorted(gta1)]


def build_mixed(gta1, traces, adaptive):
    rows = []
    for row_id in sorted(gta1):
        by_model = {}
        for model in MODEL_ORDER:
            chain = traces[model][row_id]["chains"][0]
            values = [candidate(value, model, index) for index, value in enumerate(chain["global_K8"])]
            branch = chain["refinement"] if adaptive and chain["refinement"] is not None else chain["confirmation"]
            values.append(candidate(branch, model, 8))
            by_model[model] = values
        candidates = [by_model[model][slot] for slot in range(9) for model in MODEL_ORDER]
        rows.append({"id": row_id, "application": gta1[row_id]["application"], "target_bbox": gta1[row_id]["target_bbox"], "candidates": candidates})
    return rows


def diagnostics(single, mixed):
    result = {}
    for family, traces in (("single", {"GTA1-7B": single}), ("mixed", mixed)):
        reports = [chain["report"] for trace in traces.values() for row in trace.values() for chain in row["chains"]]
        invalid = sum(
            value["point"] is None
            for trace in traces.values()
            for row in trace.values()
            for chain in row["chains"]
            for value in [*chain["global_K8"], chain["confirmation"], *([chain["refinement"]] if chain["refinement"] else [])]
        )
        result[family] = {
            "chains": len(reports),
            "adaptive_trigger_rate": sum(report["adaptive_uses"] == "refinement" for report in reports) / len(reports),
            "mean_gate_score": float(np.mean([report["score"] for report in reports])),
            "invalid_union_outputs": invalid,
            "fixed_cell_forwards_per_row": 27,
            "adaptive_cell_forwards_per_row": 27,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-gta1", type=Path, nargs="+", required=True)
    parser.add_argument("--mixed-gta1", type=Path, nargs="+", required=True)
    parser.add_argument("--mixed-qwen3", type=Path, nargs="+", required=True)
    parser.add_argument("--mixed-uitars", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gta1, _, _ = load_sources()
    single = load_trace(args.single_gta1, "single", "GTA1-7B", 3, 4)
    mixed = {
        "GTA1-7B": load_trace(args.mixed_gta1, "mixed", "GTA1-7B", 1, 1),
        "Qwen3-VL-8B-Instruct": load_trace(args.mixed_qwen3, "mixed", "Qwen3-VL-8B-Instruct", 1, 1),
        "UI-TARS-7B-SFT": load_trace(args.mixed_uitars, "mixed", "UI-TARS-7B-SFT", 1, 2),
    }
    cells = {
        "Q1_K8_single_fixed": build_single(gta1, single, False),
        "Q2_K8_single_adaptive": build_single(gta1, single, True),
        "Q3_K8_mixed_fixed": build_mixed(gta1, mixed, False),
        "Q4_K8_mixed_adaptive": build_mixed(gta1, mixed, True),
    }
    short = {"Q1": cells["Q1_K8_single_fixed"], "Q2": cells["Q2_K8_single_adaptive"], "Q3": cells["Q3_K8_mixed_fixed"], "Q4": cells["Q4_K8_mixed_adaptive"]}
    evaluations = {name: compact_evaluation(rows) for name, rows in short.items()}
    interactions = bootstrap_interactions(short["Q1"], evaluations)
    accuracy = {name: value["accuracy"] for name, value in evaluations.items()}
    highest = max(accuracy, key=lambda name: (accuracy[name]["M1_ccm"], name))
    primary = interactions["M1_ccm"]
    effects = {
        "adaptive_single_Q2_minus_Q1": accuracy["Q2"]["M1_ccm"] - accuracy["Q1"]["M1_ccm"],
        "adaptive_mixed_Q4_minus_Q3": accuracy["Q4"]["M1_ccm"] - accuracy["Q3"]["M1_ccm"],
        "allocation_fixed_Q3_minus_Q1": accuracy["Q3"]["M1_ccm"] - accuracy["Q1"]["M1_ccm"],
        "allocation_adaptive_Q4_minus_Q2": accuracy["Q4"]["M1_ccm"] - accuracy["Q2"]["M1_ccm"],
    }
    result = {
        "schema_version": 1,
        "status": "PASS",
        "rows": 1581,
        "budget_per_cell": 27,
        "accuracy": accuracy,
        "M1_effects": effects,
        "interactions": interactions,
        "failure_kappa": {name: failure_statistics(rows)["mean_pairwise_kappa"] for name, rows in short.items()},
        "diagnostics": diagnostics(single, mixed),
        "prediction": {
            "highest_cell": highest,
            "Q4_highest": highest == "Q4",
            "interaction_classification": primary["classification"],
            "composability_success": highest == "Q4" and primary["classification"] != "SUB_ADDITIVE",
        },
        "kill_conditions": {"X-K1": primary["classification"] == "SUB_ADDITIVE"},
        "replaces_K3_X2_conclusion": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accuracy": accuracy, "effects": effects, "interactions": interactions, "prediction": result["prediction"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()