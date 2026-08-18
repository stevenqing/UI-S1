import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from owin_common import (
    ROOT,
    R_GRID,
    RUN_DIR,
    atomic_json,
    jitter_offsets,
    median_pairwise_iou,
    oracle_window,
    read_jsonl,
    sha256_file,
    write_jsonl_fsynced,
)


ALLOCATION_DIR = ROOT / "runs/allocation-law/2026-08-01"
REGION_PATH = ALLOCATION_DIR / "raw/shared_regions_n12.jsonl"
GTA1_ROOT = ROOT / "runs/ccm-h2h/2026-07-31/h1/shards/top18"
COVER_ROWS_PATH = ROOT / "runs/cover/2026-08-16/raw/arm_a_rows.jsonl"
COVER_RESULT_PATH = ROOT / "runs/cover/2026-08-16/ARM_A.json"
CWIN_ROWS_PATH = ROOT / "runs/cwin/2026-08-17/raw/stage0_rows.jsonl"
DATA_ROOT = ROOT / "runs/collision-law/2026-07-30/w3_assets/ScreenSpot-Pro"
MODEL_ROOT = ROOT / "runs/collision-law/2026-07-30/w3_assets/GTA1-7B"
MODEL_INDEX_PATH = MODEL_ROOT / "model.safetensors.index.json"
H1_GENERATOR_PATH = ROOT / "runs/ccm-h2h/2026-07-31/h1/generate_candidates.py"
GENERATION_CONTRACT_PATH = ROOT / "runs/ccm-h2h/2026-07-31/h1/generation_contract.py"
MVP_SOURCE_PATH = ROOT / "runs/collision-law/2026-07-30/w3_assets/MVP/mvp_sspro.py"
CONFIG_PATHS = [RUN_DIR / "configs/owin_prereg.yaml", *[RUN_DIR / f"configs/amendment_{index:03d}.yaml" for index in range(1, 4)], RUN_DIR / "configs/amendment_004_part_a.yaml"]
SPEC_PATHS = [RUN_DIR / "SPEC.md", *[RUN_DIR / f"AMENDMENT_{index:03d}_{name}.md" for index, name in ((1, "CALIBRATION_AND_BUDGET"), (2, "RADIUS_SAMPLE_SHIFT"), (3, "DEPENDENCE_DEGENERACY"))], RUN_DIR / "AMENDMENT_004_PART_A_CALIBRATION_IDENTITY.md"]
OUTPUT_PATH = RUN_DIR / "PREFLIGHT.json"
IMAGE_MANIFEST_PATH = RUN_DIR / "INPUT_IMAGE_MANIFEST.jsonl"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


allocation = load_module(ALLOCATION_DIR / "allocation_eval.py", "owin_preflight_allocation")


def main():
    if OUTPUT_PATH.exists() or IMAGE_MANIFEST_PATH.exists():
        raise FileExistsError("OWIN preflight output exists")
    configs = [yaml.safe_load(path.read_text()) for path in CONFIG_PATHS]
    if configs[-1]["part_b"]["gpu_authorized"] is not False or configs[-1]["unchanged"]["exact_formal_calls"] != 6000:
        raise PermissionError("OWIN authorization boundary mismatch")
    manifest = allocation.load_manifest(REGION_PATH)
    gta1 = allocation.load_gta1(GTA1_ROOT, manifest)
    pool = allocation.build_pool(gta1, {}, [("GTA1-7B", view) for view in range(12)])
    pool_by_id = {row["id"]: row for row in pool}
    regions = {row["id"]: row for row in read_jsonl(REGION_PATH)}
    cover_rows = {row["row_id"]: row for row in read_jsonl(COVER_ROWS_PATH)}
    cwin_rows = {row["row_id"]: row for row in read_jsonl(CWIN_ROWS_PATH)}
    identities = set(pool_by_id)
    if len(identities) != 1581 or identities != set(regions) or identities != set(cover_rows) or identities != set(cwin_rows):
        raise ValueError("OWIN identity mismatch")
    if Counter(row["target_stratum"] for row in cover_rows.values()) != Counter({"common_11": 931, "partial_1_10": 425, "uncovered_0": 225}):
        raise ValueError("OWIN COVER stratum mismatch")

    existing_row_medians = []
    candidate_row_medians = {radius: [] for radius in R_GRID}
    offsets = {radius: jitter_offsets(radius) for radius in R_GRID}
    for row_id in sorted(identities):
        row_regions = regions[row_id]
        width, height = row_regions["img_size"]
        crops = row_regions["regions"][1:]
        existing_row_medians.append(median_pairwise_iou(crops))
        target_bbox = pool_by_id[row_id]["target_bbox"]
        for radius in R_GRID:
            windows = [oracle_window(width, height, target_bbox, offset)["final_window"] for offset in offsets[radius]]
            candidate_row_medians[radius].append(median_pairwise_iou(windows))
    iou_star = float(np.median(existing_row_medians))
    iou_by_radius = {radius: float(np.median(candidate_row_medians[radius])) for radius in R_GRID}
    selected_radius = min(R_GRID, key=lambda radius: (abs(iou_by_radius[radius] - iou_star), radius))

    common_ids = [row_id for row_id in identities if cover_rows[row_id]["target_stratum"] == "common_11"]
    ordered_common = sorted(
        common_ids,
        key=lambda row_id: (
            max(0.0, pool_by_id[row_id]["target_bbox"][2] - pool_by_id[row_id]["target_bbox"][0])
            * max(0.0, pool_by_id[row_id]["target_bbox"][3] - pool_by_id[row_id]["target_bbox"][1]),
            row_id,
        ),
    )
    small_ids, large_ids = set(ordered_common[:465]), set(ordered_common[465:])
    median_bbox_area = max(0.0, pool_by_id[ordered_common[465]]["target_bbox"][2] - pool_by_id[ordered_common[465]]["target_bbox"][0]) * max(0.0, pool_by_id[ordered_common[465]]["target_bbox"][3] - pool_by_id[ordered_common[465]]["target_bbox"][1])
    b3_half = {
        "common_small": float(np.mean([cover_rows[row_id]["b3_correct"] for row_id in small_ids])),
        "common_large": float(np.mean([cover_rows[row_id]["b3_correct"] for row_id in large_ids])),
    }
    m1_common = float(np.mean([cwin_rows[row_id]["original_m1_correct"] for row_id in common_ids]))
    single = {}
    for stratum in ("uncovered_0", "partial_1_10", "common_11"):
        row_ids = [row_id for row_id in identities if cover_rows[row_id]["target_stratum"] == stratum]
        values = [
            int(allocation.point_in_bbox(candidate["point"], pool_by_id[row_id]["target_bbox"]))
            for row_id in row_ids
            for candidate in pool_by_id[row_id]["candidates"][1:]
        ]
        single[stratum] = {"rows": len(row_ids), "slot_observations": len(values), "accuracy": float(np.mean(values))}

    image_records = []
    for row_id in sorted(identities):
        source = gta1[row_id]
        image_path = DATA_ROOT / "images" / source["img_filename"]
        image_records.append({"row_id": row_id, "path": str(image_path.relative_to(ROOT)), "bytes": image_path.stat().st_size, "sha256": sha256_file(image_path)})
    write_jsonl_fsynced(IMAGE_MANIFEST_PATH, image_records)
    output = {
        "schema_version": 1,
        "status": "PASS_OWIN_GEOMETRY_CALIBRATION_PREFLIGHT_NO_MODEL_OUTPUT",
        "gpu_used": False,
        "gpu_authorized": False,
        "owin_model_outputs_computed": False,
        "rows": 1581,
        "strata": dict(Counter(row["target_stratum"] for row in cover_rows.values())),
        "radius_calibration": {
            "status": "EVALUATION_SIDE_GT_GEOMETRY_ZERO_GPU_NOT_LABEL_FREE",
            "IoU_star": iou_star,
            "candidates": {str(radius): {"IoU": iou_by_radius[radius], "signed_difference": iou_by_radius[radius] - iou_star, "absolute_difference": abs(iou_by_radius[radius] - iou_star), "normative_integer_offsets": offsets[radius]} for radius in R_GRID},
            "selected_R": selected_radius,
            "selected_offsets": offsets[selected_radius],
            "endpoint": selected_radius in (R_GRID[0], R_GRID[-1]),
        },
        "common_area_split": {"population_rows": 931, "small_rows": 465, "large_rows": 466, "numeric_median_area": median_bbox_area, "small_ids": sorted(small_ids), "large_ids": sorted(large_ids), "w": 465 / 931},
        "anchors": {"B_common_M1_v_only_N12": m1_common, "B3_common_halves_C_uni": b3_half, "B_single_GTA1_views1_11": single, "Delta_ident_tolerance": 1e-9, "Delta_decomp_reference_scale": 0.005},
        "dependencies": {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in [*CONFIG_PATHS, *SPEC_PATHS, REGION_PATH, COVER_ROWS_PATH, COVER_RESULT_PATH, CWIN_ROWS_PATH, H1_GENERATOR_PATH, GENERATION_CONTRACT_PATH, MVP_SOURCE_PATH]},
        "gta1_shards": {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(GTA1_ROOT.glob("shard-*.jsonl"))},
        "model_root": {"path": str(MODEL_ROOT.relative_to(ROOT)), "model_revision": "701bedc80b447863bd60e3318ae44f6cbbfafd78", "index_path": str(MODEL_INDEX_PATH.relative_to(ROOT)), "index_bytes": MODEL_INDEX_PATH.stat().st_size, "index_sha256": sha256_file(MODEL_INDEX_PATH)},
        "image_manifest": {"path": str(IMAGE_MANIFEST_PATH.relative_to(ROOT)), "rows": len(image_records), "bytes": IMAGE_MANIFEST_PATH.stat().st_size, "sha256": sha256_file(IMAGE_MANIFEST_PATH)},
    }
    atomic_json(OUTPUT_PATH, output)
    print(json.dumps({"status": output["status"], "radius_calibration": output["radius_calibration"], "anchors": output["anchors"], "gpu_authorized": False}, indent=2))


if __name__ == "__main__":
    main()