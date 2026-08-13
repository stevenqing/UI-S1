import hashlib
import json
import sys
from pathlib import Path

import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(RUN_DIR))

from common import load_context


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def parse_action(value):
    model, view = value.rsplit("/view", 1)
    return model, int(view)


def crop_around(point, image_size, crop_size):
    width, height = image_size
    crop_width = min(crop_size[0], width)
    crop_height = min(crop_size[1], height)
    left = min(max(0, int(round(point[0] - crop_width / 2))), width - crop_width)
    top = min(max(0, int(round(point[1] - crop_height / 2))), height - crop_height)
    return [left, top, left + crop_width, top + crop_height]


def main():
    config_path = RUN_DIR / "configs/q2b_verification.yaml"
    config = yaml.safe_load(config_path.read_text())
    if config["status"] != "result_blind_design_freeze":
        raise ValueError("Q2b config is not frozen")
    context = load_context()
    manifest_path = ROOT / "runs/allocation-law/2026-08-01/raw/shared_regions_n12.jsonl"
    manifest_rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    manifest = {row["id"]: row for row in manifest_rows}
    if len(manifest) != 1581 or set(manifest) != set(context["row_ids"]):
        raise ValueError("Q2b source manifest identity mismatch")
    crop_config = config["stage_2"]["crop_geometry"]
    crop_size = (crop_config["width"], crop_config["height"])
    mapping = {parse_action(action): verifier for action, verifier in config["stage_2"]["mapping"].items()}
    expected_actions = tuple(parse_action(value) for value in config["stage_1"]["generation_actions"])
    if set(mapping) != set(expected_actions) or any(action[0] == verifier for action, verifier in mapping.items()):
        raise ValueError("Q2b verifier mapping mismatch or self-check")

    output_rows = []
    for stable_index, row_id in enumerate(context["row_ids"]):
        metadata = context["metadata"][row_id]
        source = manifest[row_id]
        if source["img_size"] != metadata["img_size"] or source["instruction"] != metadata["instruction"]:
            raise ValueError(f"Q2b source metadata mismatch: {row_id}")
        checks = []
        for check_index, action in enumerate(expected_actions):
            candidate = context["bank"][action][row_id]
            point = list(map(float, candidate["point"]))
            checks.append({
                "check_index": check_index,
                "candidate_model": action[0],
                "candidate_view": action[1],
                "verifier_model": mapping[action],
                "candidate_point": point,
                "candidate_region": list(candidate["region"]),
                "candidate_coverage": float(candidate.get("coverage", 0.0)),
                "verification_crop": crop_around(point, metadata["img_size"], crop_size),
            })
        artifact = {
            "stable_index": stable_index,
            "id": row_id,
            "application": metadata["application"],
            "img_filename": source["img_filename"],
            "img_size": metadata["img_size"],
            "instruction": metadata["instruction"],
            "checks": checks,
        }
        if any("target" in key or "bbox" in key for key in artifact):
            raise ValueError("Q2b crop artifact target leak")
        artifact["checks_sha256"] = canonical_hash(checks)
        output_rows.append(artifact)
    output = RUN_DIR / "raw/q2b_crops.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in output_rows))
    print(json.dumps({
        "status": "PASS",
        "rows": len(output_rows),
        "checks": sum(len(row["checks"]) for row in output_rows),
        "output": str(output.relative_to(ROOT)),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()