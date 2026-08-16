import hashlib
import json
import os
from pathlib import Path

import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CONFIG_PATH = RUN_DIR / "configs/icc_prereg.yaml"
SPEC_PATH = RUN_DIR / "SPEC.md"
SOURCE_PATH = ROOT / "runs/evid/2026-08-15/SELECTED_PARAMETERS.json"
OUTPUT_PATH = RUN_DIR / "ARM_A.json"
RAW_PATH = RUN_DIR / "raw/arm_a_rho_curves.jsonl"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def endpoint_class(rho_v, rho_l):
    low_v, low_l = rho_v == 0.0, rho_l == 0.0
    high_v, high_l = rho_v == 1.0, rho_l == 1.0
    if low_v or low_l:
        return "mixed_endpoint" if high_v or high_l else "low_endpoint"
    if high_v or high_l:
        return "high_endpoint"
    return "interior"


def surface_record(fold, scores, selected, grid):
    lookup = {(row["rho_v"], row["rho_l"]): row["accuracy"] for row in scores}
    rho_v, rho_l = selected["rho_v"], selected["rho_l"]
    v_index, l_index = grid.index(rho_v), grid.index(rho_l)
    neighbors = {}
    for name, v_offset, l_offset in (
        ("rho_v_lower", -1, 0), ("rho_v_higher", 1, 0),
        ("rho_l_lower", 0, -1), ("rho_l_higher", 0, 1),
    ):
        new_v, new_l = v_index + v_offset, l_index + l_offset
        neighbors[name] = (
            None if not (0 <= new_v < len(grid) and 0 <= new_l < len(grid))
            else float(lookup[(grid[new_v], grid[new_l])] - selected["accuracy"])
        )
    row_max = {str(value): max(lookup[(value, other)] for other in grid) for value in grid}
    column_max = {str(value): max(lookup[(other, value)] for other in grid) for value in grid}
    return {
        "outer_fold": fold,
        "selected_rho_v": rho_v,
        "selected_rho_l": rho_l,
        "selected_accuracy": selected["accuracy"],
        "endpoint_class": endpoint_class(rho_v, rho_l),
        "neighbor_accuracy_deltas": neighbors,
        "rho_v_row_maxima": row_max,
        "rho_l_column_maxima": column_max,
        "surface": scores,
    }


def write_jsonl_fsynced(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def main():
    if OUTPUT_PATH.exists() or RAW_PATH.exists():
        raise FileExistsError("ICC Arm A output exists")
    config = yaml.safe_load(CONFIG_PATH.read_text())
    source = json.loads(SOURCE_PATH.read_text())
    if (
        config["status"] != "PREREGISTERED_BEFORE_ANY_ICC_RESULT"
        or source["status"] != "PASS_EVID_STAGE1_NESTED_SELECTIONS_BEFORE_OUTER_EVALUATION"
        or source["outer_test_outputs_computed"] is not False
    ):
        raise PermissionError("ICC Arm A source mismatch")
    grid = config["arm_a"]["grid"]
    records = []
    for fold in source["folds"]:
        scores = fold["rho_scores"]
        pairs = [(row["rho_v"], row["rho_l"]) for row in scores]
        expected = [(rho_v, rho_l) for rho_v in grid for rho_l in grid]
        if pairs != expected:
            raise ValueError(f"ICC Arm A grid order mismatch: {fold['outer_fold']}")
        records.append(surface_record(fold["outer_fold"], scores, fold["selected_rho"], grid))
    write_jsonl_fsynced(RAW_PATH, records)
    counts = {name: sum(row["endpoint_class"] == name for row in records) for name in config["arm_a"]["endpoint_classes"]}
    output = {
        "schema_version": 1,
        "status": "PASS_ICC_ARM_A_TABLE_AUDIT",
        "source": {"path": str(SOURCE_PATH.relative_to(ROOT)), "sha256": sha256_file(SOURCE_PATH)},
        "spec_sha256": sha256_file(SPEC_PATH),
        "grid": grid,
        "folds": [{key: value for key, value in row.items() if key != "surface"} for row in records],
        "endpoint_class_counts": counts,
        "new_accuracy_computed": False,
        "raw": {"path": str(RAW_PATH.relative_to(ROOT)), "rows": len(records), "bytes": RAW_PATH.stat().st_size, "sha256": sha256_file(RAW_PATH), "write_flush_fsync_per_row": True},
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": output["status"], "endpoint_class_counts": counts, "selected": [[row["selected_rho_v"], row["selected_rho_l"]] for row in records]}, indent=2))


if __name__ == "__main__":
    main()