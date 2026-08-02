import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
ALLOCATION_DIR = ROOT / "runs/allocation-law/2026-08-01"
sys.path.insert(0, str(ALLOCATION_DIR))
from allocation_eval import compact_evaluation


MODEL_ORDER = ("GTA1-72B", "UI-Venus-Ground-72B", "Qwen3.5-122B-A10B")


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_unique(path, source):
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        if row["id"] in rows: raise ValueError(f"G2 duplicate {source}: {row['id']}")
        rows[row["id"]] = row
    if len(rows) != 1581: raise ValueError(f"G2 {source} requires 1,581 rows, found {len(rows)}")
    return rows


def validate_scores(regions, scores, model):
    if set(regions) != set(scores): raise ValueError(f"G2 {model} identity mismatch")
    for row_id, row in scores.items():
        source = regions[row_id]
        if "bbox" in row or "target_bbox" in row or row["model_id"] != model:
            raise ValueError(f"G2 score target/model mismatch: {model}/{row_id}")
        if row["region_manifest_sha256"] != source["regions_sha256"] or canonical_hash(row["predictions"]) != row["predictions_sha256"]:
            raise ValueError(f"G2 score hash mismatch: {model}/{row_id}")
        expected = source["required_region_indices_by_model"][model]
        if [value["region_index"] for value in row["predictions"]] != expected:
            raise ValueError(f"G2 score coverage mismatch: {model}/{row_id}")


def candidate(score, region, model, view_index):
    point = score["point"] if score["parse_ok"] else [-1.0, -1.0]
    return {
        "model": model, "view_index": view_index,
        "point": list(map(float, point)), "region": list(region["region"]),
        "coverage": float(region["coverage"]) if model == "GTA1-72B" else 0.0,
    }


def build_pool(labels, regions, scores, kind, seed=None, p1_budget=8):
    rows = []
    for row_id in sorted(labels):
        source = regions[row_id]
        by_region = {value["region_index"]: value for value in source["regions"]}
        by_score = {model: {value["region_index"]: value for value in scores[model][row_id]["predictions"]} for model in MODEL_ORDER}
        candidates = []
        if kind == "P1":
            for slot, region_index in enumerate(range(p1_budget)):
                candidates.append(candidate(by_score["GTA1-72B"][region_index], by_region[region_index], "GTA1-72B", slot))
        else:
            indices = list(range(4)) if kind == "P2" else [0, *source["perturbed_region_indices"][str(seed)]]
            for slot, region_index in enumerate(indices):
                for model in MODEL_ORDER:
                    candidates.append(candidate(by_score[model][region_index], by_region[region_index], model, slot))
        rows.append({"id": row_id, "application": labels[row_id]["application"], "target_bbox": labels[row_id]["target_bbox"], "candidates": candidates})
    return rows


def compact(value):
    return {"rows": value["rows"], "fold_rows": value["fold_rows"], "accuracy": value["accuracy"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1", type=Path, required=True)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--gta1", type=Path, required=True)
    parser.add_argument("--venus", type=Path, required=True)
    parser.add_argument("--qwen35", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = yaml.safe_load((RUN_DIR / "configs/g2_protocol.yaml").read_text())
    g1 = json.loads(args.g1.read_text())
    if g1["status"] != "PASS" or g1["gate"]["G2_cancelled"]:
        raise ValueError("G2 is blocked by G1")
    label_rows = [json.loads(line) for line in args.labels.read_text().splitlines() if line.strip()]
    labels = {row["id"]: {"application": row["application"], "target_bbox": row["target_bbox"]} for row in label_rows}
    if len(labels) != 1581: raise ValueError("G2 label identity mismatch")
    regions = load_unique(args.regions, "regions")
    if set(regions) != set(labels) or any("bbox" in row or "target_bbox" in row for row in regions.values()):
        raise ValueError("G2 region identity/target leak")
    paths = dict(zip(MODEL_ORDER, (args.gta1, args.venus, args.qwen35)))
    scores = {model: load_unique(paths[model], model) for model in MODEL_ORDER}
    for model in MODEL_ORDER: validate_scores(regions, scores[model], model)
    p1_budget = protocol["cells"]["P1"]["selected_budget"]
    p1 = compact_evaluation(build_pool(labels, regions, scores, "P1", p1_budget=p1_budget))
    p2 = compact_evaluation(build_pool(labels, regions, scores, "P2"))
    seeds = protocol["proposal_sensitivity"]["seeds"]
    perturbed = {str(seed): compact_evaluation(build_pool(labels, regions, scores, "MDE", seed)) for seed in seeds}
    seed_accuracies = [perturbed[str(seed)]["accuracy"]["M1_ccm"] for seed in seeds]
    mde = 2 * statistics.stdev(seed_accuracies)
    p2_m1 = p2["accuracy"]["M1_ccm"]
    threshold = g1["gate"]["G2_effective_threshold"]
    system_sota = p2_m1 > protocol["decisions"]["standard_system_sota"]["threshold"] and p2_m1 - protocol["decisions"]["standard_system_sota"]["threshold"] > mde
    effective = p2_m1 > threshold and p2_m1 - threshold > mde
    if system_sota:
        outcome = "SYSTEM_SOTA"
    elif effective:
        outcome = "EFFECTIVE_THRESHOLD_RESULT"
    elif p2_m1 > 0.704:
        outcome = "ABOVE_PAPER_MODEL_REFERENCE_ONLY"
    else:
        outcome = "BELOW_PAPER_MODEL_REFERENCE"
    unique_counts = {model: sum(len(row["required_region_indices_by_model"][model]) for row in regions.values()) for model in MODEL_ORDER}
    result = {
        "schema_version": 1, "status": "PASS", "rows": 1581,
        "P0_bare": g1["bare"], "P1_GTA1_72B": {"budget": p1_budget, **compact(p1)}, "P2_mixed_72B": {"budget": 12, **compact(p2)},
        "proposal_sensitivity": {"seeds": seeds, "M1_accuracies": seed_accuracies, "sample_sd": mde / 2, "MDE": mde, "evaluations": {seed: compact(value) for seed, value in perturbed.items()}},
        "compute": {"primary_P2_scoring_forwards": 1581 * 12, "unique_scored_regions_by_model": unique_counts, "total_unique_scoring_forwards": sum(unique_counts.values())},
        "comparisons": {"P2_M1_minus_P1_M1_unequal_budget_context": p2_m1 - p1["accuracy"]["M1_ccm"], "P2_M1_minus_73_1": p2_m1 - 0.731, "P2_M1_minus_70_4": p2_m1 - 0.704},
        "decision": {"effective_threshold": threshold, "effective_threshold_pass": effective, "system_SOTA_73_1_pass": system_sota, "equal_budget_allocation_adjudication": "UNAVAILABLE_P1_N8_FALLBACK", "P2_above_P1_unequal_budget_context": p2_m1 > p1["accuracy"]["M1_ccm"], "outcome": outcome},
        "paper_only_references": protocol["paper_only_references"],
        "sources": {"G1": {"path": str(args.g1), "sha256": sha256_file(args.g1)}, "regions": {"path": str(args.regions), "sha256": sha256_file(args.regions)}, **{model: {"path": str(paths[model]), "sha256": sha256_file(paths[model])} for model in MODEL_ORDER}},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"P1": result["P1_GTA1_72B"]["accuracy"], "P2": result["P2_mixed_72B"]["accuracy"], "MDE": mde, "decision": result["decision"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()