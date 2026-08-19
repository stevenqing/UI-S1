import importlib.util
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

from tile_common import N_GRID, ROOT, RUN_DIR, atomic_json, read_jsonl, sha256_file


ALLOCATION_DIR = ROOT / "runs/allocation-law/2026-08-01"
REGION_PATH = ALLOCATION_DIR / "raw/shared_regions_n12.jsonl"
GTA1_ROOT = ROOT / "runs/ccm-h2h/2026-07-31/h1/shards/top18"
COVER_PATH = ROOT / "runs/cover/2026-08-16/raw/arm_a_rows.jsonl"
CWIN_PATH = ROOT / "runs/cwin/2026-08-17/raw/stage0_rows.jsonl"
OWIN_RESULT_PATH = ROOT / "runs/owin/2026-08-17/ARM_B.json"
OWIN_RAW_PATH = ROOT / "runs/owin/2026-08-17/raw/arm_b_rows.jsonl"
SPEC_PATH = RUN_DIR / "SPEC.md"
CONFIG_PATH = RUN_DIR / "configs/tile_prereg.yaml"
AMENDMENT_PATH = RUN_DIR / "AMENDMENT_001_STAGE0_OPERATIONS.md"
AMENDMENT_CONFIG_PATH = RUN_DIR / "configs/amendment_001.yaml"
BASELINE_AMENDMENT_PATH = RUN_DIR / "AMENDMENT_002_BASELINE_RECONCILIATION.md"
BASELINE_AMENDMENT_CONFIG_PATH = RUN_DIR / "configs/amendment_002.yaml"
OUTPUT_PATH = RUN_DIR / "PREFLIGHT.json"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


allocation = load_module(ALLOCATION_DIR / "allocation_eval.py", "tile_preflight_allocation")


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    config = yaml.safe_load(CONFIG_PATH.read_text())
    amendment = yaml.safe_load(AMENDMENT_CONFIG_PATH.read_text())
    if config["gpu"]["stage1_authorized"] is not False or amendment["status"] != "FROZEN_BEFORE_ANY_TILE_RESULT":
        raise PermissionError("TILE preflight config mismatch")
    manifest = allocation.load_manifest(REGION_PATH)
    gta1 = allocation.load_gta1(GTA1_ROOT, manifest)
    pool = allocation.build_pool(gta1, {}, [("GTA1-7B", view) for view in range(12)])
    fold_for_group, fold_loads = allocation.group_folds(pool)
    cover = {row["row_id"]: row for row in read_jsonl(COVER_PATH)}
    cwin = {row["row_id"]: row for row in read_jsonl(CWIN_PATH)}
    if len(pool) != 1581 or set(gta1) != set(cover) or set(gta1) != set(cwin):
        raise ValueError("TILE identity mismatch")
    for row in pool:
        row_id = row["id"]
        if cover[row_id]["fold"] != cwin[row_id]["outer_fold"] or cover[row_id]["fold"] != fold_for_group[row["application"]]:
            raise ValueError(f"TILE fold mismatch: {row_id}")
    disagreement_ids = sorted(row_id for row_id in cover if bool(cover[row_id]["b3_correct"]) != bool(cwin[row_id]["original_b3_correct"]))
    if sum(row["b3_correct"] for row in cover.values()) != 1007 or sum(row["original_b3_correct"] for row in cwin.values()) != 950 or len(disagreement_ids) != 143 or sum(row["target_coverage_count"] > 0 for row in cover.values()) != 1356:
        raise ValueError("TILE anchor mismatch")
    owin = json.loads(OWIN_RESULT_PATH.read_text())
    raw = read_jsonl(OWIN_RAW_PATH)
    counts = Counter(row["N"] for row in raw)
    if any(counts[value] != 1581 for value in range(4, 12)) or len(raw) != 12648 or sha256_file(OWIN_RAW_PATH) != owin["raw"]["sha256"]:
        raise ValueError("TILE OWIN raw mismatch")
    selected = [row for row in raw if row["N"] in N_GRID]
    if len({(row["row_id"], row["N"]) for row in selected}) != 1581 * len(N_GRID):
        raise ValueError("TILE OWIN layout identity mismatch")
    dependencies = [SPEC_PATH, CONFIG_PATH, AMENDMENT_PATH, AMENDMENT_CONFIG_PATH, BASELINE_AMENDMENT_PATH, BASELINE_AMENDMENT_CONFIG_PATH, REGION_PATH, COVER_PATH, CWIN_PATH, OWIN_RESULT_PATH, OWIN_RAW_PATH, ALLOCATION_DIR / "allocation_eval.py"]
    output = {"schema_version": 1, "status": "PASS_TILE_PREFLIGHT_NO_TILE_STATISTIC", "gpu_used": False, "stage0_computed": False, "stage1_authorized": False, "rows": 1581, "fold_loads": fold_loads, "anchors": {"V_only_B3_correct_rows": 950, "C_uni_B3_correct_rows": 1007, "B3_disagreement_rows": 143, "B3_disagreement_ids_sha256": hashlib.sha256(json.dumps(disagreement_ids, separators=(",", ":")).encode()).hexdigest(), "crop_center_covered_rows": 1356}, "OWIN_layouts": {"N_grid": list(N_GRID), "rows_per_N": 1581, "raw_sha256": sha256_file(OWIN_RAW_PATH)}, "GTA1_shards": {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(GTA1_ROOT.glob("shard-*.jsonl"))}, "dependencies": {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in dependencies}}
    atomic_json(OUTPUT_PATH, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()