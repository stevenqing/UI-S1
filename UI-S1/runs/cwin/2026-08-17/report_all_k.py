import importlib.util
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
STAGE0_PATH = RUN_DIR / "STAGE0.json"
GEOMETRY_PATH = RUN_DIR / "raw/geometry_all_k.jsonl"
REGION_PATH = ROOT / "runs/allocation-law/2026-08-01/raw/shared_regions_n12.jsonl"
ALLOCATION_DIR = ROOT / "runs/allocation-law/2026-08-01"
H3_PATH = ROOT / "runs/ccm-h2h/2026-07-31/h3/h3_eval.py"
SOURCEBIAS_PATH = ROOT / "runs/sourcebias/2026-08-03/sourcebias_common.py"
MANIFEST_PATH = ALLOCATION_DIR / "raw/shared_regions_n12.jsonl"
GTA1_ROOT = ROOT / "runs/ccm-h2h/2026-07-31/h1/shards/top18"
OUTPUT_PATH = RUN_DIR / "STAGE0_ALL_K.json"
K_GRID = (2, 3, 4)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


allocation = load_module(ALLOCATION_DIR / "allocation_eval.py", "cwin_report_allocation")
h3 = load_module(H3_PATH, "cwin_report_h3")
sourcebias = load_module(SOURCEBIAS_PATH, "cwin_report_sourcebias")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def contains_center(region, target_bbox):
    x = (target_bbox[0] + target_bbox[2]) / 2
    y = (target_bbox[1] + target_bbox[3]) / 2
    return region[0] <= x < region[2] and region[1] <= y < region[3]


def filtered_row(row, dropped_crop_indices):
    dropped_candidate_indices = {index + 1 for index in dropped_crop_indices}
    return {
        **row,
        "candidates": [
            candidate
            for index, candidate in enumerate(row["candidates"])
            if index not in dropped_candidate_indices
        ],
    }


def b3_correct(row):
    selected, _ = sourcebias.b3_select_index(row["candidates"])
    return bool(h3.point_in_bbox(row["candidates"][selected]["point"], row["target_bbox"]))


def m1_outputs(rows, fold_for_group):
    outputs = {}
    for fold in range(5):
        development = [row for row in rows if fold_for_group[row["application"]] != fold]
        test = [row for row in rows if fold_for_group[row["application"]] == fold]
        tables, priors = h3.fit_ccm(development)
        for row in test:
            selected = h3.ccm_select(row, tables, priors)
            outputs[row["id"]] = bool(
                h3.point_in_bbox(row["candidates"][selected]["point"], row["target_bbox"])
            )
    return outputs


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    stage0 = json.loads(STAGE0_PATH.read_text())
    if stage0["status"] != "PASS_CWIN_STAGE0_COMPLETE" or stage0["gpu_authorized"] is not False:
        raise PermissionError("CWIN Stage-0 result mismatch")
    manifest = allocation.load_manifest(MANIFEST_PATH)
    gta1 = allocation.load_gta1(GTA1_ROOT, manifest)
    original_rows = allocation.build_pool(gta1, {}, [("GTA1-7B", view) for view in range(12)])
    original = allocation.compact_evaluation(original_rows)
    fold_for_group, _ = allocation.group_folds(original_rows)
    geometry = {row["row_id"]: row for row in read_jsonl(GEOMETRY_PATH)}
    regions = {row["id"]: row for row in read_jsonl(REGION_PATH)}
    if len(geometry) != 1581 or set(geometry) != set(regions):
        raise ValueError("CWIN all-K identity mismatch")

    per_k = {}
    for K in K_GRID:
        dropped_rows = [
            filtered_row(row, geometry[row["id"]]["dropped_crop_indices"][:K])
            for row in original_rows
        ]
        b3 = {row["id"]: b3_correct(row) for row in dropped_rows}
        m1 = m1_outputs(dropped_rows, fold_for_group)
        transition = Counter()
        newly_covered = 0
        partial_to_higher = 0
        lost_all = 0
        for row in original_rows:
            row_id = row["id"]
            target = row["target_bbox"]
            crops = regions[row_id]["regions"][1:]
            dropped = set(geometry[row_id]["dropped_crop_indices"][:K])
            retained = [region for index, region in enumerate(crops) if index not in dropped]
            new = geometry[row_id]["new_windows"][:K]
            before = sum(contains_center(region, target) for region in crops)
            after = sum(contains_center(region, target) for region in retained + new)
            transition[(before, after)] += 1
            newly_covered += before == 0 and after > 0
            partial_to_higher += before > 0 and after > before
            lost_all += before > 0 and after == 0
        b3_accuracy = float(np.mean(list(b3.values())))
        m1_accuracy = float(np.mean(list(m1.values())))
        per_k[str(K)] = {
            "L1": {
                "newly_covered_rows": newly_covered,
                "partial_to_higher_rows": partial_to_higher,
                "lost_all_coverage_rows": lost_all,
                "coverage_transition": {
                    f"{before}->{after}": count
                    for (before, after), count in sorted(transition.items())
                },
            },
            "L3": {
                "B3_drop_accuracy": b3_accuracy,
                "B3_drop_delta": b3_accuracy - original["accuracy"]["B3_mvp"],
                "M1_drop_accuracy": m1_accuracy,
                "M1_drop_delta": m1_accuracy - original["accuracy"]["M1_ccm"],
            },
        }
    l1_keys = ("newly_covered_rows", "partial_to_higher_rows", "lost_all_coverage_rows")
    if any(per_k["4"]["L1"][key] != stage0["L1"][key] for key in l1_keys) or any(
        not math.isclose(
            per_k["4"]["L3"][key], stage0["L3"][key], rel_tol=0.0, abs_tol=1e-15
        )
        for key in ("B3_drop_delta", "M1_drop_delta")
    ):
        raise ValueError("CWIN selected-K reconstruction mismatch")
    output = {
        "schema_version": 1,
        "status": "CWIN_STAGE0_ALL_K_REPORTING_RECOVERY_COMPLETE",
        "created_after_stage0": True,
        "changes_nested_gate": False,
        "gpu_used": False,
        "gpu_authorized": False,
        "rows": 1581,
        "original_accuracy": original["accuracy"],
        "per_K": per_k,
        "selected_K_reproduces_stage0": True,
    }
    atomic_json(OUTPUT_PATH, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()