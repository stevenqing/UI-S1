import hashlib
import json
import os
from pathlib import Path

import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CONFIG_PATH = RUN_DIR / "configs/cwin_prereg.yaml"
SPEC_PATH = RUN_DIR / "SPEC.md"
P0_PATH = RUN_DIR / "CONTAINMENT_RECONCILIATION.md"
OUTPUT_PATH = RUN_DIR / "PREFLIGHT.json"
REGION_PATH = ROOT / "runs/allocation-law/2026-08-01/raw/shared_regions_n12.jsonl"
COVER_ARM_A_PATH = ROOT / "runs/cover/2026-08-16/ARM_A.json"
ALLOCATION_L4_PATH = ROOT / "runs/allocation-law/2026-08-01/L4_RESULTS.json"
ALLOCATION_MODULE_PATH = ROOT / "runs/allocation-law/2026-08-01/allocation_eval.py"
H3_MODULE_PATH = ROOT / "runs/ccm-h2h/2026-07-31/h3/h3_eval.py"
GTA1_ROOT = ROOT / "runs/ccm-h2h/2026-07-31/h1/shards/top18"
GRAN_MANIFEST_PATH = ROOT / "runs/gran/2026-08-14/INPUT_MANIFEST.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if config["status"] != "PREREGISTERED_AFTER_P0_BEFORE_ANY_CWIN_RESULT" or config["stage1"]["authorized"] is not False:
        raise PermissionError("CWIN preregistration mismatch")
    regions = read_jsonl(REGION_PATH)
    if len(regions) != 1581:
        raise ValueError("CWIN region row mismatch")
    crop_sizes = []
    invalid = []
    for row in regions:
        width, height = row["img_size"]
        if row["regions"][0] != [0, 0, width, height] or len(row["regions"]) != 12:
            invalid.append(row["id"])
            continue
        sizes = {(right - left, bottom - top) for left, top, right, bottom in row["regions"][1:]}
        if len(sizes) != 1:
            invalid.append(row["id"])
        else:
            crop_sizes.append(next(iter(sizes)))
    if invalid:
        raise ValueError(f"CWIN inconsistent crop geometry rows: {len(invalid)}")
    cover = json.loads(COVER_ARM_A_PATH.read_text())
    allocation = json.loads(ALLOCATION_L4_PATH.read_text())
    if (
        cover["geometry"]["crop_views"] != list(range(1, 12))
        or allocation["E1_shared_gta1"]["diagnostic"]["per_rank_full_bbox_containment"][0] != 0.9993674889310563
        or allocation["E1_shared_gta1"]["diagnostic"]["per_rank_full_bbox_containment"][11] != 0.6103731815306768
    ):
        raise ValueError("CWIN containment anchor mismatch")
    gran = json.loads(GRAN_MANIFEST_PATH.read_text())
    gta_files = {path: info for path, info in gran["files"].items() if "screenspot_gta1_views_0_15" in info["roles"]}
    if not gta_files:
        raise ValueError("CWIN GTA1 input manifest missing")
    output = {
        "schema_version": 1,
        "status": "PASS_CWIN_PREFLIGHT_NO_GEOMETRY_OR_RESULT",
        "gpu_used": False,
        "stage0_computed": False,
        "stage1_authorized": False,
        "dependencies": {
            "spec": {"path": str(SPEC_PATH.relative_to(ROOT)), "sha256": sha256_file(SPEC_PATH)},
            "config": {"path": str(CONFIG_PATH.relative_to(ROOT)), "sha256": sha256_file(CONFIG_PATH)},
            "p0": {"path": str(P0_PATH.relative_to(ROOT)), "sha256": sha256_file(P0_PATH)},
            "regions": {"path": str(REGION_PATH.relative_to(ROOT)), "bytes": REGION_PATH.stat().st_size, "sha256": sha256_file(REGION_PATH)},
            "cover_arm_a": {"path": str(COVER_ARM_A_PATH.relative_to(ROOT)), "sha256": sha256_file(COVER_ARM_A_PATH)},
            "allocation_l4": {"path": str(ALLOCATION_L4_PATH.relative_to(ROOT)), "sha256": sha256_file(ALLOCATION_L4_PATH)},
            "allocation_module": {"path": str(ALLOCATION_MODULE_PATH.relative_to(ROOT)), "sha256": sha256_file(ALLOCATION_MODULE_PATH)},
            "h3_module": {"path": str(H3_MODULE_PATH.relative_to(ROOT)), "sha256": sha256_file(H3_MODULE_PATH)},
            "gran_manifest": {"path": str(GRAN_MANIFEST_PATH.relative_to(ROOT)), "sha256": sha256_file(GRAN_MANIFEST_PATH)},
        },
        "gta1_trace_files": gta_files,
        "geometry": {"rows": 1581, "full_image_view": 0, "crop_views": list(range(1, 12)), "all_rows_equal_crop_size": True, "crop_size_counts": {f"{width}x{height}": crop_sizes.count((width, height)) for width, height in sorted(set(crop_sizes))}, "invalid_rows": 0},
        "containment_reconciliation": {"rank0_full_image": True, "rank0_full_bbox": 0.9993674889310563, "rank11_full_bbox": 0.6103731815306768, "cover_crop_only_uncovered_center": cover["target_strata"]["uncovered_0"]["fraction"], "contradiction": False},
        "pool_scope": {"stage1a": "GTA1_v_only_N12", "stage1b": "full_36_requires_new_amendment", "mixed_C_uni_is_not_replacement_pool": True},
    }
    atomic_json(OUTPUT_PATH, output)
    print(json.dumps({"status": output["status"], "geometry": output["geometry"], "stage1_authorized": False}, indent=2))


if __name__ == "__main__":
    main()