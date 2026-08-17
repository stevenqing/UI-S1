import glob
import hashlib
import json
import os
from pathlib import Path

import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CONFIG_PATH = RUN_DIR / "configs/cover_prereg.yaml"
SPEC_PATH = RUN_DIR / "SPEC.md"
OUTPUT_PATH = RUN_DIR / "ARM_B_FEASIBILITY.json"
PUBLIC_PATH = ROOT / "runs/visual-utility-selector/2026-08-11/data/public_records.jsonl"
PRIVATE_GLOB = ROOT / "runs/visual-utility-selector/2026-08-11/data/private_labels_fold-*.jsonl"
ROLE_CODE_PATH = ROOT / "runs/close/2026-08-08/e1_arm_aggregator_matrix.py"
LSA_CODE_PATH = ROOT / "runs/lsa/2026-08-10/lsa_common.py"
FOLDS_PATH = ROOT / "runs/complementarity/2026-07-30/folds.json"
XFER_MANIFEST_PATH = ROOT / "runs/xfer/2026-08-07/PUBLICATION_MANIFEST.json"
REGION_MANIFEST_PATH = ROOT / "runs/allocation-law/2026-08-01/raw/shared_regions_n12.jsonl"


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
    if config["status"] != "PREREGISTERED_BEFORE_ANY_COVER_RESULT":
        raise PermissionError("COVER preregistration mismatch")
    public = [row for row in read_jsonl(PUBLIC_PATH) if row.get("sample_key", "").startswith("mind2web/C_uni/")]
    if len(public) != config["arm_b"]["rows"] or any(len(row["candidates"]) != 12 for row in public):
        raise ValueError("COVER M2W public bank mismatch")
    public_keys = {row["sample_key"] for row in public}
    if len(public_keys) != len(public) or set(row["fold"] for row in public) != set(range(5)):
        raise ValueError("COVER M2W public identity/fold mismatch")
    private_paths = [Path(path) for path in sorted(glob.glob(str(PRIVATE_GLOB)))]
    if len(private_paths) != 5:
        raise ValueError("COVER private fold count mismatch")
    private_keys = set()
    private_records = 0
    private_files = []
    for path in private_paths:
        rows = read_jsonl(path)
        selected = [row for row in rows if row.get("sample_key", "").startswith("mind2web/C_uni/")]
        for row in selected:
            if set(row) != {"schema_version", "sample_key", "candidate_success"} or len(row["candidate_success"]) != 12:
                raise ValueError(f"COVER private schema mismatch: {path}")
            private_keys.add(row["sample_key"])
        private_records += len(selected)
        private_files.append({"path": str(path.relative_to(ROOT)), "rows_total": len(rows), "rows_selected": len(selected), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    if private_records != 2080 or private_keys != public_keys:
        raise ValueError("COVER public/private identity mismatch")
    role_code = ROLE_CODE_PATH.read_text()
    required_tokens = ("for view_index in (0, 1)", "for crop_index in range(2)", "stage1_", "stage2_")
    if any(token not in role_code for token in required_tokens):
        raise ValueError("COVER M2W slot-role code mismatch")
    region_rows = read_jsonl(REGION_MANIFEST_PATH)
    if len(region_rows) != 1581 or any(len(row["regions"]) != 12 or row["regions"][0] != [0, 0, *row["img_size"]] for row in region_rows):
        raise ValueError("COVER SSPro region manifest mismatch")
    output = {
        "schema_version": 1,
        "status": "PASS_COVER_ARM_B_FEASIBILITY_AND_INPUT_LOCK",
        "arm_statistics_computed": False,
        "gpu_used": False,
        "spec": {"path": str(SPEC_PATH.relative_to(ROOT)), "sha256": sha256_file(SPEC_PATH)},
        "config": {"path": str(CONFIG_PATH.relative_to(ROOT)), "sha256": sha256_file(CONFIG_PATH)},
        "mind2web": {
            "status": "READY_2080x12",
            "public_rows": len(public),
            "private_rows": private_records,
            "folds": 5,
            "models": config["arm_b"]["models"],
            "slot_roles": config["arm_b"]["slot_roles"],
            "public": {"path": str(PUBLIC_PATH.relative_to(ROOT)), "bytes": PUBLIC_PATH.stat().st_size, "sha256": sha256_file(PUBLIC_PATH)},
            "private_files": private_files,
            "role_code": {"path": str(ROLE_CODE_PATH.relative_to(ROOT)), "sha256": sha256_file(ROLE_CODE_PATH)},
            "lsa_loader": {"path": str(LSA_CODE_PATH.relative_to(ROOT)), "sha256": sha256_file(LSA_CODE_PATH)},
            "folds_file": {"path": str(FOLDS_PATH.relative_to(ROOT)), "sha256": sha256_file(FOLDS_PATH)},
            "xfer_manifest": {"path": str(XFER_MANIFEST_PATH.relative_to(ROOT)), "sha256": sha256_file(XFER_MANIFEST_PATH)},
        },
        "screenspot_regions": {
            "status": "READY_1581x12_VIEW0_FULL_VIEW1_11_CROPS",
            "rows": len(region_rows),
            "path": str(REGION_MANIFEST_PATH.relative_to(ROOT)),
            "bytes": REGION_MANIFEST_PATH.stat().st_size,
            "sha256": sha256_file(REGION_MANIFEST_PATH),
        },
    }
    atomic_json(OUTPUT_PATH, output)
    print(json.dumps({"status": output["status"], "mind2web": output["mind2web"]["status"], "screenspot": output["screenspot_regions"]["status"], "arm_statistics_computed": False}, indent=2))


if __name__ == "__main__":
    main()