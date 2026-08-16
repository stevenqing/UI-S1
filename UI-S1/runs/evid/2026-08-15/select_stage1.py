import json
from pathlib import Path

import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CONFIG_PATH = RUN_DIR / "configs/evid_prereg.yaml"
PREFLIGHT_PATH = RUN_DIR / "PREFLIGHT.json"
STAGE0_PATH = RUN_DIR / "STAGE0.json"
AMENDMENT_PATH = RUN_DIR / "AMENDMENT_001_STAGE1_TIES_PATH.md"
OUTPUT_PATH = RUN_DIR / "SELECTED_PARAMETERS.json"


def load_module(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    config = yaml.safe_load(CONFIG_PATH.read_text())
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    stage0 = json.loads(STAGE0_PATH.read_text())
    if (
        preflight["status"] != "PASS_EVID_PREFLIGHT_NO_STAGE_RESULT"
        or stage0["status"] != "PASS_EVID_STAGE0_COMPLETE"
        or stage0["proceed_stage1"] is not True
    ):
        raise PermissionError("EVID Stage 1 selection authorization mismatch")
    stage1 = load_module(RUN_DIR / "stage1.py", "evid_stage1_selection_helpers")
    stage0_impl = load_module(RUN_DIR / "stage0.py", "evid_stage0_selection_helpers")
    rows = stage1.load_rows()
    row_ids = tuple(sorted(rows))
    fold_for_row = {row_id: int(rows[row_id]["fold"]) for row_id in row_ids}
    grid = config["method"]["secondary_rho_grid"]["values"]
    selections = []
    for outer_fold in range(5):
        inner_validation_fold = (outer_fold + 1) % 5
        inner_train = [row_id for row_id in row_ids if fold_for_row[row_id] not in {outer_fold, inner_validation_fold}]
        inner_validation = [row_id for row_id in row_ids if fold_for_row[row_id] == inner_validation_fold]
        outer_development = [row_id for row_id in row_ids if fold_for_row[row_id] != outer_fold]
        selected_rho, rho_scores = stage1.select_rho(stage0_impl, rows, inner_validation, grid)
        selections.append({
            "outer_fold": outer_fold,
            "inner_validation_fold": inner_validation_fold,
            "selected_rho": selected_rho,
            "rho_scores": rho_scores,
            "inner_lineage_weights": stage1.lineage_weights(rows, inner_train),
            "outer_lineage_weights": stage1.lineage_weights(rows, outer_development),
            "rho_boundary_selected": selected_rho["rho_v"] in {0.0, 1.0} or selected_rho["rho_l"] in {0.0, 1.0},
            "outer_test_labels_opened": False,
            "outer_test_outputs_computed": False,
        })
    output = {
        "schema_version": 1,
        "status": "PASS_EVID_STAGE1_NESTED_SELECTIONS_BEFORE_OUTER_EVALUATION",
        "stage0_sha256": sha256_file(STAGE0_PATH),
        "amendment_sha256": sha256_file(AMENDMENT_PATH),
        "outer_test_outputs_computed": False,
        "folds": selections,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": output["status"], "selections": [{"fold": row["outer_fold"], "rho": row["selected_rho"], "boundary": row["rho_boundary_selected"]} for row in selections]}, indent=2))


if __name__ == "__main__":
    main()