import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


RUN_DIR = Path(__file__).resolve().parents[1]
ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(RUN_DIR))

from xfer_common import COORDINATE_ACTIONS, complete_link_groups, plurality_action


SEED = 20260807
MODEL_DIRS = {
    "TongUI-7B": "tongui",
    "CogAgent-18B": "cogagent",
    "UI-TARS-7B": "uitars",
}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_lane(directory):
    rows = {}
    for path in sorted(directory.glob("shard-*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip(): continue
            row = json.loads(line)
            if row["id"] in rows: raise ValueError(f"duplicate stage1 id: {row['id']}")
            rows[row["id"]] = row
    if len(rows) != 2080:
        raise ValueError(f"stage1 lane incomplete: {directory}, rows={len(rows)}")
    return rows


def crop_around(point, image_size, crop_size=(448, 448)):
    width, height = image_size
    crop_width, crop_height = min(crop_size[0], width), min(crop_size[1], height)
    left = min(max(0, int(round(point[0] - crop_width / 2))), width - crop_width)
    top = min(max(0, int(round(point[1] - crop_height / 2))), height - crop_height)
    return [left, top, left + crop_width, top + crop_height]


def farthest(points, center):
    return max(points, key=lambda point: (math.dist(point, center), point[0], point[1]))


def main():
    roster = yaml.safe_load((RUN_DIR / "configs/xfer_roster.yaml").read_text())
    model_order = [model["id"] for model in roster["mind2web"]["models"]]
    lanes = {model: load_lane(RUN_DIR / "raw/stage1" / directory) for model, directory in MODEL_DIRS.items()}
    view1_lanes = {model: load_lane(RUN_DIR / "raw/stage1/view1" / directory) for model, directory in MODEL_DIRS.items()}
    proposer_regions = load_lane(RUN_DIR / "raw/proposer-regions")
    rows = [json.loads(line) for line in (RUN_DIR / "data/mind2web/mind2web_test_task.jsonl").read_text().splitlines() if line.strip()]
    output_rows = []
    triggers = 0
    for row in rows:
        image = Image.open(ROOT / row["image"])
        image_size = image.size
        proposer_source = proposer_regions[row["id"]]
        if len(proposer_source.get("regions", [])) != 16:
            raise ValueError(f"proposer region count mismatch: {row['id']}")
        candidates = []
        for view_index in (0, 1):
            for model in model_order:
                source = lanes[model][row["id"]] if view_index == 0 else view1_lanes[model][row["id"]]
                if source["model_id"] != model or source["stable_index"] != row["stable_index"] or source["image_sha256"] != row["image_sha256"]:
                    raise ValueError(f"stage1 provenance mismatch: {model}/{row['id']}")
                if view_index == 0:
                    prediction = source["prediction"]
                else:
                    if source["source_hashes"]["view1"] != proposer_source["regions_sha256"]:
                        raise ValueError(f"view1 proposer provenance mismatch: {model}/{row['id']}")
                    prediction = source["predictions"]["view1"][0]["prediction"]
                candidates.append({**prediction, "model": model, "view_index": view_index})
        winning_type = plurality_action(candidates, model_order)
        retained = [candidate for candidate in candidates if candidate.get("parse_ok") and candidate["action"] == winning_type and candidate.get("position") is not None]
        trigger = winning_type in COORDINATE_ACTIONS and len(retained) >= 1
        arms = {"C_uni": [], "C_cond": [], "C_rand": [], "C_self": []}
        cluster_fallback = None
        if trigger:
            triggers += 1
            points = [[candidate["position"][0] * image_size[0], candidate["position"][1] * image_size[1]] for candidate in retained]
            groups = complete_link_groups(points)
            centers = [[float(np.mean([points[index][axis] for index in group])) for axis in (0, 1)] for group in groups[:2]]
            if len(centers) == 1:
                centers.append(list(farthest(points, centers[0])))
                cluster_fallback = "single_cluster_farthest_winning_type_point"
            proposer_predictions = [
                candidate for candidate in candidates
                if candidate["model"] == "TongUI-7B" and candidate.get("parse_ok") and candidate.get("position") is not None
            ]
            proposer_by_view = {candidate["view_index"]: candidate for candidate in proposer_predictions}
            view1_region = proposer_source["regions"][0]["region"]
            self_fallbacks = {
                0: [image_size[0] / 2, image_size[1] / 2],
                1: [(view1_region[0] + view1_region[2]) / 2, (view1_region[1] + view1_region[3]) / 2],
            }
            self_centers = []
            for view_index in (0, 1):
                candidate = proposer_by_view.get(view_index)
                self_centers.append(
                    [candidate["position"][0] * image_size[0], candidate["position"][1] * image_size[1]]
                    if candidate is not None else self_fallbacks[view_index]
                )
            rng = np.random.default_rng(np.random.SeedSequence([SEED, row["stable_index"]]))
            crop_width, crop_height = min(448, image_size[0]), min(448, image_size[1])
            random_regions = []
            for _ in range(2):
                left = int(rng.integers(0, image_size[0] - crop_width + 1))
                top = int(rng.integers(0, image_size[1] - crop_height + 1))
                random_regions.append([left, top, left + crop_width, top + crop_height])
            arms = {
                "C_uni": [proposer_source["regions"][index]["region"] for index in (1, 2)],
                "C_cond": [crop_around(center, image_size) for center in centers],
                "C_rand": random_regions,
                "C_self": [crop_around(center, image_size) for center in self_centers],
            }
        artifact = {
            "stable_index": row["stable_index"],
            "id": row["id"],
            "image": row["image"],
            "image_sha256": row["image_sha256"],
            "task": row["task"],
            "step_history": row["step_history"],
            "image_size": list(image_size),
            "stage1_predictions": candidates,
            "winning_type": winning_type,
            "stage2_trigger": trigger,
            "trigger_candidate_count": len(retained),
            "stage1_group_count": len(groups) if trigger else 0,
            "cluster_fallback": cluster_fallback,
            "proposer_regions_sha256": proposer_source["regions_sha256"],
            "arms": arms,
        }
        if any("target" in key or "bbox" in key or key == "step" for key in artifact):
            raise ValueError("consensus ROI target leak")
        artifact["arms_sha256"] = canonical_hash(arms)
        output_rows.append(artifact)
    output = RUN_DIR / "raw/mind2web-consensus-roi.jsonl"
    with output.open("w", buffering=1) as handle:
        for artifact in output_rows:
            handle.write(json.dumps(artifact, ensure_ascii=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    print(json.dumps({
        "status": "PASS",
        "rows": len(output_rows),
        "triggered": triggers,
        "trigger_rate": triggers / len(output_rows),
        "mean_forwards_C_cond": 6 + 6 * triggers / len(output_rows),
        "output": str(output.relative_to(ROOT)),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }, indent=2, sort_keys=True))


if __name__ == "__main__": main()