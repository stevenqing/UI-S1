import hashlib
import json
import os
from pathlib import Path

import yaml
from PIL import Image


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CONFIG_PATH = RUN_DIR / "configs/decomp_prereg.yaml"
SPEC_PATH = RUN_DIR / "SPEC.md"
RECONCILIATION_PATH = RUN_DIR / "LANE_RECONCILIATION.md"
AMENDMENT_PATH = RUN_DIR / "AMENDMENT_001_PUBLIC_TIE_BREAK.md"
OUTPUT_PATH = RUN_DIR / "PREFLIGHT.json"
PUBLIC_PATH = ROOT / "runs/visual-utility-selector/2026-08-11/data/public_records.jsonl"

DEPENDENCIES = {
    "gran_input_manifest": ROOT / "runs/gran/2026-08-14/INPUT_MANIFEST.json",
    "vus_data_manifest": ROOT / "runs/visual-utility-selector/2026-08-11/data/MANIFEST.json",
    "vus_private_fold_manifest": ROOT / "runs/visual-utility-selector/2026-08-11/data/private_label_folds.manifest.json",
    "m0_result": ROOT / "runs/final/2026-08-04/m0_manifest_diff.json",
    "allocation_l1": ROOT / "runs/allocation-law/2026-08-01/L1_RESULTS.json",
    "mask_adjudication": ROOT / "runs/mask/2026-08-14/MASK_ADJUDICATION.json",
    "split_preflight": ROOT / "runs/split/2026-08-14/PREFLIGHT.json",
    "orth_arm2": ROOT / "runs/orth/2026-08-14/ARM2.json",
    "xfer_publication_manifest": ROOT / "runs/xfer/2026-08-07/PUBLICATION_MANIFEST.json",
    "canonical_aggregator": ROOT / "runs/ccm-h2h/2026-07-31/h1/aggregators_coord.py",
}

MODELS = ("GTA1-7B", "Qwen3-VL-8B-Instruct", "UI-TARS-7B-SFT")
CANONICAL_SLOTS = tuple((model, view) for view in range(4) for model in MODELS)
FORBIDDEN_PUBLIC_KEYS = {
    "bbox", "candidate_success", "correct", "ground_truth", "gt", "label",
    "reward", "success", "target", "target_bbox", "ui_type",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def contains_forbidden_key(value):
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = key.lower()
            if normalized in FORBIDDEN_PUBLIC_KEYS or normalized.startswith("gt_"):
                return True
            if contains_forbidden_key(child):
                return True
    elif isinstance(value, list):
        return any(contains_forbidden_key(child) for child in value)
    return False


def resolve_image(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def load_public_rows():
    rows = []
    with PUBLIC_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("benchmark") == "screenspot_pro" and row.get("arm") == "C_uni":
                rows.append(row)
    if len(rows) != 1581:
        raise ValueError(f"DECOMP SSPro public row mismatch: {len(rows)}")
    if len({row["sample_key"] for row in rows}) != 1581:
        raise ValueError("DECOMP duplicate SSPro public sample key")
    if any(len(row["candidates"]) != 12 for row in rows):
        raise ValueError("DECOMP SSPro public candidate width mismatch")
    if any(contains_forbidden_key(row) for row in rows):
        raise ValueError("DECOMP SSPro public bank contains prohibited evaluation field")
    return rows


def image_snapshot(rows):
    images = {}
    for row in rows:
        digest = row["image_sha256"]
        path = resolve_image(row["image_path"])
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"DECOMP image mismatch: {row['sample_key']}")
        with Image.open(path) as image:
            width, height = image.size
        source_id = row["image_path"].split("/images/", 1)[-1]
        record = images.setdefault(digest, {
            "bytes": path.stat().st_size,
            "height": height,
            "source_ids": set(),
            "width": width,
        })
        if (record["width"], record["height"]) != (width, height):
            raise ValueError(f"DECOMP image alias dimensions differ: {digest}")
        record["source_ids"].add(source_id)
    return {
        digest: {
            "bytes": record["bytes"],
            "height": record["height"],
            "source_ids": sorted(record["source_ids"]),
            "width": record["width"],
        }
        for digest, record in images.items()
    }


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if config["status"] != "PREREGISTERED_AFTER_P0_BEFORE_ANY_DECOMP_ARM":
        raise PermissionError("DECOMP preregistration status mismatch")
    if config["arm2"]["mode_tie_break"] != "earliest_canonical_index":
        raise PermissionError("DECOMP Arm 2 public tie-break mismatch")
    rows = load_public_rows()
    images = image_snapshot(rows)
    dependencies = {
        name: {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in DEPENDENCIES.items()
    }
    output = {
        "schema_version": 1,
        "status": "PASS_DECOMP_PREFLIGHT_NO_ARM_STARTED",
        "gpu_used": False,
        "arm_statistics_computed": False,
        "labels_opened": False,
        "target_bbox_opened": False,
        "logprob_inventory_started": False,
        "dependencies": {
            "reconciliation": {"path": str(RECONCILIATION_PATH.relative_to(ROOT)), "sha256": sha256_file(RECONCILIATION_PATH)},
            "spec": {"path": str(SPEC_PATH.relative_to(ROOT)), "sha256": sha256_file(SPEC_PATH)},
            "config": {"path": str(CONFIG_PATH.relative_to(ROOT)), "sha256": sha256_file(CONFIG_PATH)},
            "amendment_001": {"path": str(AMENDMENT_PATH.relative_to(ROOT)), "sha256": sha256_file(AMENDMENT_PATH)},
            "public_records": {"path": str(PUBLIC_PATH.relative_to(ROOT)), "bytes": PUBLIC_PATH.stat().st_size, "sha256": sha256_file(PUBLIC_PATH)},
            **dependencies,
        },
        "authorized_lanes": {
            "arm1": "screenspot_pro_C_uni_1581x12",
            "arm2": "screenspot_pro_C_uni_public_1581x12",
            "arm3": "screenspot_pro_and_mind2web_inventory_only",
        },
        "canonical_slots": [list(slot) for slot in CANONICAL_SLOTS],
        "public_bank": {
            "rows": len(rows),
            "candidates_per_row": 12,
            "candidate_schema": sorted(rows[0]["candidates"][0]),
            "public_coverage_available": False,
            "forbidden_evaluation_fields": False,
        },
        "dataset_snapshot": {
            "image_count": len(images),
            "image_bytes": sum(record["bytes"] for record in images.values()),
            "images": images,
        },
        "mind2web_arm1_status": "BLOCKED_ALIGNED_POOL_UNAVAILABLE",
        "mind2web_dom_status": "FULL_DOM_AX_UNAVAILABLE_HISTORICAL_DATA_CURRENTLY_MISSING",
        "split_name_reconciliation": "BANK_LINEAGES_DISTINCT_FROM_PROBE_MODELS",
    }
    atomic_json(OUTPUT_PATH, output)
    print(json.dumps({
        "status": output["status"],
        "rows": output["public_bank"]["rows"],
        "images": output["dataset_snapshot"]["image_count"],
        "labels_opened": False,
        "arm_statistics_computed": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()