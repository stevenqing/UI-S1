import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import yaml


RUN_DIR = Path(__file__).resolve().parent
CONFIG_PATH = RUN_DIR / "configs/g1_roster.yaml"
MODEL_ORDER = ("UI-Venus-Ground-72B", "GTA1-72B", "Qwen3.5-122B-A10B")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def point_in_bbox(point, bbox):
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def cohen_kappa(left, right):
    left = np.asarray(left, dtype=np.int8)
    right = np.asarray(right, dtype=np.int8)
    observed = float(np.mean(left == right))
    left_rate = float(np.mean(left))
    right_rate = float(np.mean(right))
    expected = left_rate * right_rate + (1 - left_rate) * (1 - right_rate)
    if math.isclose(expected, 1.0):
        raise ValueError("G1 constant failure pair")
    return (observed - expected) / (1 - expected)


def pair_result(left, right, pair_id, permutations, seed):
    observed = cohen_kappa(left, right)
    rng = np.random.default_rng(seed + sum(map(ord, pair_id)))
    null = np.asarray([cohen_kappa(left, rng.permutation(right)) for _ in range(permutations)])
    return {
        "id": pair_id,
        "rows": len(left),
        "observed_kappa": observed,
        "null_mean": float(np.mean(null)),
        "null_sd": float(np.std(null)),
        "p_greater_equal": float((1 + np.count_nonzero(null >= observed)) / (permutations + 1)),
        "permutations": permutations,
    }


def adjudicate_gate(pass_at_3, kappas, gate):
    gate_pass = pass_at_3 >= gate["pass_requires"]["minimum_pass_at_3"] and min(kappas) < gate["pass_requires"]["at_least_one_pairwise_kappa_below"]
    concentrated = all(value >= gate["lineage_concentrated_if_all_pairwise_kappa_at_least"] for value in kappas)
    cancelled = pass_at_3 < gate["cancel_g2_if_pass_at_3_below"]
    if cancelled:
        action = "CANCEL_G2_COMMON_FAILURE_CEILING"
        threshold = None
    elif concentrated:
        action = "RUN_G2_LINEAGE_CONCENTRATED_RELAXED_THRESHOLD"
        threshold = gate["concentrated_g2_effective_threshold"]
    elif gate_pass:
        action = "RUN_G2_STANDARD_SYSTEM_SOTA_THRESHOLD"
        threshold = gate["default_g2_threshold"]
    else:
        action = "RUN_G2_MARGINAL_GATE_STANDARD_THRESHOLD"
        threshold = gate["default_g2_threshold"]
    return {
        "G1_pass": gate_pass,
        "lineage_concentrated": concentrated,
        "G2_cancelled": cancelled,
        "G2_action": action,
        "G2_effective_threshold": threshold,
        "G2_stretch_threshold": gate["stretch_threshold"],
    }


def load_trace(path, model, spec, labels):
    rows = {}
    expected_protocol = canonical_hash(spec)
    expected_index_hash = sha256_file(RUN_DIR / "models" / model / "model.safetensors.index.json")
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row_id = row["id"]
        if row_id in rows or "bbox" in row or "target_bbox" in row:
            raise ValueError(f"G1 trace identity/target leak: {model}/{row_id}")
        if row["model_id"] != model or row["model_revision"] != spec["revision"]:
            raise ValueError(f"G1 model identity mismatch: {model}/{row_id}")
        if row["model_index_sha256"] != expected_index_hash or row["protocol_sha256"] != expected_protocol:
            raise ValueError(f"G1 model/protocol hash mismatch: {model}/{row_id}")
        if canonical_hash(row["prediction"]) != row["prediction_sha256"]:
            raise ValueError(f"G1 prediction hash mismatch: {model}/{row_id}")
        point = row["prediction"]["point"]
        if len(point) != 2 or not all(math.isfinite(float(value)) for value in point):
            raise ValueError(f"G1 invalid point: {model}/{row_id}")
        if row_id not in labels or row["application"] != labels[row_id]["application"]:
            raise ValueError(f"G1 label identity mismatch: {model}/{row_id}")
        rows[row_id] = row
    if len(rows) != 1581 or set(rows) != set(labels):
        raise ValueError(f"G1 requires 1,581 identities: {model}, found {len(rows)}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--venus", type=Path, required=True)
    parser.add_argument("--gta1", type=Path, required=True)
    parser.add_argument("--qwen35", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text())
    label_rows = [json.loads(line) for line in args.labels.read_text().splitlines() if line.strip()]
    labels = {row["id"]: {"application": row["application"], "target_bbox": row["target_bbox"]} for row in label_rows}
    if len(labels) != 1581:
        raise ValueError("G1 label source identity mismatch")
    paths = dict(zip(MODEL_ORDER, (args.venus, args.gta1, args.qwen35)))
    traces = {model: load_trace(paths[model], model, config["models"][model], labels) for model in MODEL_ORDER}
    ordered_ids = sorted(labels)
    successes = {}
    failures = {}
    bare = {}
    for model in MODEL_ORDER:
        success = np.asarray([
            traces[model][row_id]["prediction"]["parse_ok"]
            and point_in_bbox(traces[model][row_id]["prediction"]["point"], labels[row_id]["target_bbox"])
            for row_id in ordered_ids
        ], dtype=np.int8)
        successes[model] = success
        failures[model] = 1 - success
        parse_successes = sum(traces[model][row_id]["prediction"]["parse_ok"] for row_id in ordered_ids)
        local_accuracy = float(np.mean(success))
        reference = config["models"][model]["paper_only_screenspot_pro"]
        bare[model] = {
            "accuracy": local_accuracy,
            "correct": int(np.sum(success)),
            "parse_successes": parse_successes,
            "parse_rate": parse_successes / len(ordered_ids),
            "paper_only_reference": reference,
            "local_minus_reference": local_accuracy - reference,
            "anchor_consistent_within_2pp": abs(local_accuracy - reference) <= config["statistics"]["anchor_tolerance_absolute"],
        }
    permutations = config["statistics"]["matched_marginal_permutations"]
    seed = config["statistics"]["permutation_seed"]
    pairs = {}
    for left, right in itertools.combinations(MODEL_ORDER, 2):
        pair_id = f"{left}__{right}"
        pairs[pair_id] = pair_result(failures[left], failures[right], pair_id, permutations, seed)
    pass_at_3 = float(np.mean(np.logical_or.reduce([successes[model].astype(bool) for model in MODEL_ORDER])))
    kappas = [record["observed_kappa"] for record in pairs.values()]
    gate = config["gate"]
    gate_result = adjudicate_gate(pass_at_3, kappas, gate)
    result = {
        "schema_version": 1,
        "status": "PASS",
        "rows": 1581,
        "models": list(MODEL_ORDER),
        "bare": bare,
        "pairwise_failure_kappa": pairs,
        "pass_at_3": pass_at_3,
        "gate": gate_result,
        "paper_only_anchors": config["paper_only_references"],
        "sources": {model: {"path": str(paths[model]), "sha256": sha256_file(paths[model])} for model in MODEL_ORDER},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"bare": bare, "pairwise_failure_kappa": pairs, "pass_at_3": pass_at_3, "gate": result["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()