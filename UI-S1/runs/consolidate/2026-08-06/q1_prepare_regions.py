import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(RUN_DIR))

from common import MODELS, load_context


STAGE1 = tuple((model, view) for view in range(2) for model in MODELS)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def official_groups(points):
    groups = []
    assigned = set()
    for index in range(len(points)):
        if index in assigned:
            continue
        members = [index]
        assigned.add(index)
        for candidate in range(len(points)):
            if candidate in assigned:
                continue
            if all(abs(points[member][0] - points[candidate][0]) <= 14 and abs(points[member][1] - points[candidate][1]) <= 14 for member in members):
                members.append(candidate)
                assigned.add(candidate)
        groups.append(tuple(members))
    groups.sort(key=lambda group: (-len(group), min(group)))
    return groups


def crop_around(center, image_size, crop_size):
    width, height = image_size
    crop_width = min(crop_size[0], width)
    crop_height = min(crop_size[1], height)
    left = min(max(0, int(round(center[0] - crop_width / 2))), width - crop_width)
    top = min(max(0, int(round(center[1] - crop_height / 2))), height - crop_height)
    return [left, top, left + crop_width, top + crop_height]


def main():
    config = yaml.safe_load((RUN_DIR / "configs/q1_arms.yaml").read_text())
    if config["status"] != "result_blind_design_freeze":
        raise ValueError("Q1 config is not frozen")
    crop_size = (config["stage_2"]["crop_geometry"]["width"], config["stage_2"]["crop_geometry"]["height"])
    seed = config["arms"]["C_rand"]["seed"]
    context = load_context()
    manifest_path = ROOT / "runs/allocation-law/2026-08-01/raw/shared_regions_n12.jsonl"
    manifest_rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    manifest = {row["id"]: row for row in manifest_rows}
    if len(manifest) != 1581 or set(manifest) != set(context["row_ids"]):
        raise ValueError("Q1 shared-region manifest identity mismatch")
    rows = []
    for stable_index, row_id in enumerate(context["row_ids"]):
        metadata = context["metadata"][row_id]
        source = manifest[row_id]
        if source["img_size"] != metadata["img_size"] or source["instruction"] != metadata["instruction"]:
            raise ValueError(f"Q1 manifest metadata mismatch: {row_id}")
        points = [list(map(float, context["bank"][action][row_id]["point"])) for action in STAGE1]
        groups = official_groups(points)
        centers = [[float(np.mean([points[index][axis] for index in group])) for axis in (0, 1)] for group in groups[:2]]
        if len(centers) == 1:
            farthest = max(points, key=lambda point: (math.dist(point, centers[0]), point[0], point[1]))
            centers.append(list(farthest))
        self_centers = [points[0], points[3]]
        width, height = metadata["img_size"]
        rng = np.random.default_rng(np.random.SeedSequence([seed, stable_index]))
        crop_width, crop_height = min(crop_size[0], width), min(crop_size[1], height)
        random_regions = []
        for _ in range(2):
            left = int(rng.integers(0, width - crop_width + 1))
            top = int(rng.integers(0, height - crop_height + 1))
            random_regions.append([left, top, left + crop_width, top + crop_height])
        regions = {
            "C_cond": [crop_around(center, metadata["img_size"], crop_size) for center in centers],
            "C_rand": random_regions,
            "C_self": [crop_around(center, metadata["img_size"], crop_size) for center in self_centers],
        }
        artifact = {
            "stable_index": stable_index,
            "id": row_id,
            "application": metadata["application"],
            "img_filename": source["img_filename"],
            "img_size": metadata["img_size"],
            "instruction": metadata["instruction"],
            "stage1_actions": [list(action) for action in STAGE1],
            "stage1_points": points,
            "stage1_groups": [list(group) for group in groups],
            "arms": regions,
        }
        if any("target" in key or "bbox" in key for key in artifact):
            raise ValueError("Q1 region artifact target leak")
        artifact["arms_sha256"] = canonical_hash(regions)
        rows.append(artifact)
    output = RUN_DIR / "raw/q1_regions.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows))
    print(json.dumps({"status": "PASS", "rows": len(rows), "output": str(output.relative_to(ROOT)), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
