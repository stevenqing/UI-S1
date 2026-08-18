import hashlib
import json
from collections import defaultdict
from pathlib import Path

from owin_common import ROOT, RUN_DIR, atomic_json, oracle_window, read_jsonl, sha256_file, write_jsonl_fsynced


PREFLIGHT_PATH = RUN_DIR / "PREFLIGHT.json"
ARM_B_PATH = RUN_DIR / "ARM_B.json"
COVER_ROWS_PATH = ROOT / "runs/cover/2026-08-16/raw/arm_a_rows.jsonl"
REGION_PATH = ROOT / "runs/allocation-law/2026-08-01/raw/shared_regions_n12.jsonl"
GTA1_ROOT = ROOT / "runs/ccm-h2h/2026-07-31/h1/shards/top18"
IMAGE_MANIFEST_PATH = RUN_DIR / "INPUT_IMAGE_MANIFEST.jsonl"
OUTPUT_PATH = RUN_DIR / "SAMPLE_SUMMARY.json"
MANIFEST_PATH = RUN_DIR / "SAMPLE_WINDOW_MANIFEST.jsonl"
TARGETS = {"uncovered_0": 150, "partial_1_10": 150, "common_11": 200}


def allocate_counts(populations, target):
    applications = sorted(populations)
    if target < len(applications):
        raise ValueError("OWIN target smaller than nonempty application count")
    allocations = {application: 1 for application in applications}
    capacities = {application: populations[application] - 1 for application in applications}
    remaining = target - len(applications)
    while remaining:
        capacity_total = sum(capacities.values())
        if capacity_total < remaining or capacity_total <= 0:
            raise ValueError("OWIN insufficient sample capacity")
        quotas = {application: remaining * capacities[application] / capacity_total for application in applications}
        floors = {application: min(capacities[application], int(quotas[application])) for application in applications}
        assigned = sum(floors.values())
        for application in applications:
            allocations[application] += floors[application]
            capacities[application] -= floors[application]
        remaining -= assigned
        if not remaining:
            break
        order = sorted(applications, key=lambda application: (-(quotas[application] - floors[application]), application))
        progressed = False
        for application in order:
            if remaining == 0:
                break
            if capacities[application] > 0:
                allocations[application] += 1
                capacities[application] -= 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise ValueError("OWIN allocation made no progress")
    if sum(allocations.values()) != target or any(allocations[application] > populations[application] for application in applications):
        raise ValueError("OWIN invalid allocation")
    return allocations


def row_hash(stratum, application, row_id):
    value = f"OWIN|20260817|{stratum}|{application}|{row_id}"
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    if OUTPUT_PATH.exists() or MANIFEST_PATH.exists():
        raise FileExistsError("OWIN sample output exists")
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    arm_b = json.loads(ARM_B_PATH.read_text())
    if preflight["gpu_authorized"] is not False or arm_b["gpu_authorized"] is not False or arm_b["status"] != "PASS_OWIN_ARM_B_COMPLETE_ZERO_GPU":
        raise PermissionError("OWIN sampling authorization boundary mismatch")
    selected_radius = preflight["radius_calibration"]["selected_R"]
    offsets = preflight["radius_calibration"]["selected_offsets"]
    cover = {row["row_id"]: row for row in read_jsonl(COVER_ROWS_PATH)}
    regions = {row["id"]: row for row in read_jsonl(REGION_PATH)}
    images = {row["row_id"]: row for row in read_jsonl(IMAGE_MANIFEST_PATH)}
    gta1 = {}
    for path in sorted(GTA1_ROOT.glob("shard-*.jsonl")):
        for row in read_jsonl(path):
            gta1[row["id"]] = row
    if len(gta1) != 1581 or set(gta1) != set(cover) or set(gta1) != set(regions) or set(gta1) != set(images):
        raise ValueError("OWIN sample identity mismatch")

    records = []
    allocation_records = {}
    for stratum, target in TARGETS.items():
        by_application = defaultdict(list)
        for row_id, row in cover.items():
            if row["target_stratum"] == stratum:
                by_application[row["application"]].append(row_id)
        populations = {application: len(row_ids) for application, row_ids in by_application.items()}
        allocations = allocate_counts(populations, target)
        allocation_records[stratum] = {}
        for application in sorted(by_application):
            ordered = sorted(by_application[application], key=lambda row_id: (row_hash(stratum, application, row_id), row_id))
            selected = ordered[: allocations[application]]
            inclusion_probability = allocations[application] / populations[application]
            allocation_records[stratum][application] = {"population": populations[application], "sample": allocations[application], "inclusion_probability": inclusion_probability, "inverse_probability_weight": 1 / inclusion_probability}
            for row_id in selected:
                width, height = regions[row_id]["img_size"]
                windows = [{"slot": 0, "kind": "full_image", "crop_jitter_index": None, "requested_offset": None, "initial_window": [0, 0, width, height], "final_window": [0, 0, width, height], "translation": [0, 0], "target_center_contained": True, "target_bbox_contained": True}]
                windows.extend({"slot": index + 1, "kind": "oracle_crop", "crop_jitter_index": index, **oracle_window(width, height, gta1[row_id]["target_bbox"], offset)} for index, offset in enumerate(offsets))
                if len(windows) != 12 or any(not window["target_center_contained"] for window in windows):
                    raise ValueError(f"OWIN invalid oracle windows: {row_id}")
                records.append({"sample_id": f"owin-{row_id}", "row_id": row_id, "stratum": stratum, "application": application, "row_hash": row_hash(stratum, application, row_id), "cell_population": populations[application], "cell_sample": allocations[application], "inclusion_probability": inclusion_probability, "inverse_probability_weight": 1 / inclusion_probability, "selected_R": selected_radius, "image": images[row_id], "width": width, "height": height, "windows": windows})
    records.sort(key=lambda row: ({"common_11": 0, "partial_1_10": 1, "uncovered_0": 2}[row["stratum"]], row["row_id"]))
    if len(records) != 500 or len({row["row_id"] for row in records}) != 500:
        raise ValueError("OWIN sample row mismatch")
    write_jsonl_fsynced(MANIFEST_PATH, records)
    output = {"schema_version": 1, "status": "PASS_OWIN_SAMPLE_WINDOW_MANIFEST_FROZEN", "gpu_used": False, "gpu_authorized": False, "selected_R": selected_radius, "rows": len(records), "calls": len(records) * 12, "counts": {stratum: sum(row["stratum"] == stratum for row in records) for stratum in TARGETS}, "allocation_method": "minimum_one_then_residual_capacity_proportional_largest_remainder", "allocations": allocation_records, "manifest": {"path": str(MANIFEST_PATH.relative_to(ROOT)), "rows": len(records), "bytes": MANIFEST_PATH.stat().st_size, "sha256": sha256_file(MANIFEST_PATH), "write_flush_fsync_per_row": True}, "dependencies": {"preflight_sha256": sha256_file(PREFLIGHT_PATH), "arm_b_sha256": sha256_file(ARM_B_PATH), "image_manifest_sha256": sha256_file(IMAGE_MANIFEST_PATH)}, "next_action": "COMMIT_RUNNER_AND_TESTS_NO_GPU_BEFORE_PART_B"}
    atomic_json(OUTPUT_PATH, output)
    print(json.dumps({"status": output["status"], "counts": output["counts"], "calls": output["calls"], "selected_R": selected_radius, "gpu_authorized": False}, indent=2))


if __name__ == "__main__":
    main()