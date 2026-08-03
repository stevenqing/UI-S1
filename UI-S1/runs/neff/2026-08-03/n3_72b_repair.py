import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
SCALEUP = ROOT / "runs/scaleup/2026-08-02"
LABELS = ROOT / "runs/ccm-h2h/2026-07-31/h3/raw/shared_regions_n4.jsonl"
MODEL_SOURCES = {
    "GTA1-72B": ("gta1", 0.584),
    "UI-Venus-Ground-72B": ("venus", 0.619),
    "Qwen3.5-122B-A10B": ("qwen35", 0.704),
}
TOLERANCE = 0.02


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_unique(path):
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] in rows:
            raise ValueError(f"N3 duplicate identity: {row['id']}")
        rows[row["id"]] = row
    if len(rows) != 1581:
        raise ValueError(f"N3 requires 1,581 identities: {path}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = load_unique(LABELS)
    scaleup = json.loads((SCALEUP / "g2_mixed_72b.json").read_text())
    models = {}
    all_anchor_pass = True
    zero_out_of_image_models = 0
    trace_paths = {}
    for model, (slug, anchor) in MODEL_SOURCES.items():
        path = SCALEUP / f"raw/g2-score-{slug}.jsonl"
        expected_hash = scaleup["sources"][model]["sha256"]
        if sha256_file(path) != expected_hash:
            raise ValueError(f"N3 source hash mismatch: {model}")
        trace_paths[model] = path
        rows = load_unique(path)
        if set(rows) != set(labels):
            raise ValueError(f"N3 identity mismatch: {model}")
        correct = 0
        parse_success = 0
        out_of_image = 0
        normalized_x = []
        normalized_y = []
        per_region = {}
        for row_id, row in rows.items():
            width, height = row["img_size"]
            bbox = labels[row_id]["target_bbox"]
            for prediction in row["predictions"]:
                region_index = prediction["region_index"]
                value = per_region.setdefault(region_index, {"rows": 0, "parse_successes": 0, "out_of_image": 0, "correct": 0})
                x, y = prediction["point"]
                parse_ok = bool(prediction["parse_ok"])
                inside_image = 0 <= x <= width and 0 <= y <= height
                is_correct = parse_ok and bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]
                value["rows"] += 1
                value["parse_successes"] += int(parse_ok)
                value["out_of_image"] += int(not inside_image)
                value["correct"] += int(is_correct)
                if region_index == 0:
                    parse_success += int(parse_ok)
                    out_of_image += int(not inside_image)
                    correct += int(is_correct)
                    normalized_x.append(x / width)
                    normalized_y.append(y / height)
        accuracy = correct / 1581
        anchor_pass = abs(accuracy - anchor) <= TOLERANCE
        all_anchor_pass &= anchor_pass
        zero_out_of_image_models += int(out_of_image == 0)
        models[model] = {
            "rows": 1581,
            "full_image_accuracy": accuracy,
            "paper_only_anchor": anchor,
            "local_minus_anchor": accuracy - anchor,
            "anchor_tolerance": TOLERANCE,
            "anchor_pass": anchor_pass,
            "parse_rate": parse_success / 1581,
            "out_of_image_points": out_of_image,
            "normalized_coordinate_range": {
                "x": [min(normalized_x), max(normalized_x)],
                "y": [min(normalized_y), max(normalized_y)],
            },
            "per_region": {
                str(index): {
                    **value,
                    "parse_rate": value["parse_successes"] / value["rows"],
                    "accuracy": value["correct"] / value["rows"],
                }
                for index, value in sorted(per_region.items())
            },
            "trace_sha256": expected_hash,
        }
    no_global_bug = all_anchor_pass and zero_out_of_image_models >= 2
    if no_global_bug:
        status = "PASS_NO_GLOBAL_COORDINATE_BUG"
        action = "NO_REPAIR_PERMITTED_RETAIN_EXISTING_72B_RESULTS"
    else:
        status = "FAIL_COORDINATE_SOURCE_REQUIRES_AMENDMENT"
        action = "QUARANTINE_FAILING_MODEL_AND_FREEZE_MODEL_CARD_CORRECTION"
    result = {
        "schema_version": 1,
        "status": status,
        "rows": 1581,
        "models": models,
        "all_anchors_pass": all_anchor_pass,
        "models_with_zero_out_of_image_points": zero_out_of_image_models,
        "adjudication": {
            "global_coordinate_bug_supported": not no_global_bug,
            "action": action,
            "72B_points_eligible_for_N1": no_global_bug,
            "low_B3_interpretation": "observed_aggregation_and_candidate_pollution_boundary" if no_global_bug else "quarantined",
        },
        "frozen_evaluator_parity": {
            "P1_GTA1_N8": scaleup["P1_GTA1_72B"],
            "P2_Mixed_N12": scaleup["P2_mixed_72B"],
            "CALA_72B_source_sha256": sha256_file(ROOT / "runs/cala/2026-08-03/cala_transfer_72b_results.json"),
        },
        "sources": {
            "labels": {"path": str(LABELS.relative_to(ROOT)), "sha256": sha256_file(LABELS)},
            "ScaleUp_G2": {"path": "runs/scaleup/2026-08-02/g2_mixed_72b.json", "sha256": sha256_file(SCALEUP / "g2_mixed_72b.json")},
            **{model: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)} for model, path in trace_paths.items()},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "models": {model: {key: value[key] for key in ("full_image_accuracy", "anchor_pass", "parse_rate", "out_of_image_points")} for model, value in models.items()}, "adjudication": result["adjudication"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()