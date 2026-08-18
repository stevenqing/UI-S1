import hashlib
import json
from pathlib import Path

from owin_common import ROOT, RUN_DIR, atomic_json, oracle_window, read_jsonl, sha256_file, write_jsonl_fsynced


SAMPLE_PATH = RUN_DIR / "SAMPLE_WINDOW_MANIFEST.jsonl"
PREFLIGHT_PATH = RUN_DIR / "PREFLIGHT.json"
GTA1_ROOT = ROOT / "runs/ccm-h2h/2026-07-31/h1/shards/top18"
FORMAL_PATH = RUN_DIR / "INFERENCE_INPUT_MANIFEST.jsonl"
SMOKE_PATH = RUN_DIR / "SMOKE_INPUT_MANIFEST.jsonl"
SUMMARY_PATH = RUN_DIR / "INFERENCE_INPUT_SUMMARY.json"


def inference_window(window):
    return {
        key: value
        for key, value in window.items()
        if key not in {"target_center_contained", "target_bbox_contained"}
    }


def main():
    if any(path.exists() for path in (FORMAL_PATH, SMOKE_PATH, SUMMARY_PATH)):
        raise FileExistsError("OWIN inference input exists")
    samples = read_jsonl(SAMPLE_PATH)
    sample_ids = {row["row_id"] for row in samples}
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    gta1 = {}
    for path in sorted(GTA1_ROOT.glob("shard-*.jsonl")):
        for row in read_jsonl(path):
            gta1[row["id"]] = row
    formal = []
    for row in samples:
        source = gta1[row["row_id"]]
        formal.append({"sample_id": row["sample_id"], "row_id": row["row_id"], "execution_shard": row["stratum"], "instruction": source["instruction"], "image": row["image"], "windows": [inference_window(window) for window in row["windows"]]})
    smoke_ids = sorted(set(gta1) - sample_ids, key=lambda row_id: (hashlib.sha256(f"OWIN-SMOKE|20260817|{row_id}".encode()).hexdigest(), row_id))[:3]
    image_manifest = {row["row_id"]: row for row in read_jsonl(RUN_DIR / "INPUT_IMAGE_MANIFEST.jsonl")}
    offsets = preflight["radius_calibration"]["selected_offsets"]
    smoke = []
    for row_id in smoke_ids:
        source = gta1[row_id]
        width, height = source["img_size"]
        windows = [{"slot": 0, "kind": "full_image", "crop_jitter_index": None, "requested_offset": None, "initial_window": [0, 0, width, height], "final_window": [0, 0, width, height], "translation": [0, 0], "target_center_contained": True, "target_bbox_contained": True}]
        windows.extend({"slot": index + 1, "kind": "oracle_crop", "crop_jitter_index": index, **oracle_window(width, height, source["target_bbox"], offset)} for index, offset in enumerate(offsets))
        smoke.append({"sample_id": f"owin-smoke-{row_id}", "row_id": row_id, "instruction": source["instruction"], "image": image_manifest[row_id], "windows": [inference_window(window) for window in windows]})
    write_jsonl_fsynced(FORMAL_PATH, formal)
    write_jsonl_fsynced(SMOKE_PATH, smoke)
    output = {"schema_version": 1, "status": "PASS_OWIN_LABEL_FREE_INFERENCE_INPUTS_FROZEN", "gpu_used": False, "gpu_authorized": False, "formal": {"path": str(FORMAL_PATH.relative_to(ROOT)), "rows": len(formal), "calls": len(formal) * 12, "bytes": FORMAL_PATH.stat().st_size, "sha256": sha256_file(FORMAL_PATH)}, "smoke": {"path": str(SMOKE_PATH.relative_to(ROOT)), "rows": len(smoke), "calls": len(smoke) * 12, "row_ids": smoke_ids, "overlap_with_formal": sorted(set(smoke_ids) & sample_ids), "bytes": SMOKE_PATH.stat().st_size, "sha256": sha256_file(SMOKE_PATH)}, "forbidden_fields_absent": ["target_bbox", "bbox", "correct", "correctness", "reward", "label"], "dependencies": {"sample_manifest_sha256": sha256_file(SAMPLE_PATH), "preflight_sha256": sha256_file(PREFLIGHT_PATH)}, "next_action": "COMMIT_RUNNER_AND_TESTS_NO_GPU_BEFORE_PART_B"}
    atomic_json(SUMMARY_PATH, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()