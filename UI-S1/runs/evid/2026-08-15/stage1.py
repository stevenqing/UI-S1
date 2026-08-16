import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
MASK_DIR = ROOT / "runs/mask/2026-08-14"
DECOMP_ARM1_PATH = ROOT / "runs/decomp/2026-08-14/arm1.py"
STAGE0_IMPL_PATH = RUN_DIR / "stage0.py"
CONFIG_PATH = RUN_DIR / "configs/evid_prereg.yaml"
PREFLIGHT_PATH = RUN_DIR / "PREFLIGHT.json"
STAGE0_PATH = RUN_DIR / "STAGE0.json"
AMENDMENT_PATH = RUN_DIR / "AMENDMENT_001_STAGE1_TIES_PATH.md"
OUTPUT_PATH = RUN_DIR / "STAGE1.json"
SELECTION_PATH = RUN_DIR / "SELECTED_PARAMETERS.json"
RAW_PATH = RUN_DIR / "raw/stage1_rows.jsonl"

sys.path.insert(0, str(MASK_DIR))
from mask_common import load_rows, source_reliability


MODELS = ("GTA1-7B", "Qwen3-VL-8B-Instruct", "UI-TARS-7B-SFT")
METHODS = ("majority", "A0", "ours", "A1", "A2", "A3", "A4")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lineage_weights(rows, fit_ids):
    values = {model: [] for model in MODELS}
    for row_id in fit_ids:
        for index, candidate in enumerate(rows[row_id]["candidates"]):
            values[MODELS[index % 3]].append(float(candidate["correct"]))
    raw = {model: float(np.mean(current)) for model, current in values.items()}
    mean = float(np.mean(list(raw.values())))
    if mean <= 0:
        raise ValueError("EVID zero lineage reliability")
    return {model: value / mean for model, value in raw.items()}


def output_for_row(stage0_impl, row, rho_v, rho_l, weights, singleton=False):
    block, representative, _ = stage0_impl.select_group(row["candidates"], rho_v, rho_l, weights, singleton=singleton)
    return bool(row["candidates"][representative]["correct"]), block, representative


def select_rho(stage0_impl, rows, validation_ids, grid):
    weights = {model: 1.0 for model in MODELS}
    scores = []
    for rho_v in grid:
        for rho_l in grid:
            accuracy = float(np.mean([output_for_row(stage0_impl, rows[row_id], rho_v, rho_l, weights)[0] for row_id in validation_ids]))
            scores.append({"rho_v": float(rho_v), "rho_l": float(rho_l), "accuracy": accuracy})
    selected = max(range(len(scores)), key=lambda index: (scores[index]["accuracy"], -index))
    return scores[selected], scores


def grouped_bootstrap(rows, differences, resamples, seed):
    applications = sorted({row["application"] for row in rows.values()})
    app_values = {
        app: np.asarray([differences[row_id] for row_id, row in rows.items() if row["application"] == app], dtype=np.float64)
        for app in applications
    }
    app_fold = {app: next(row["fold"] for row in rows.values() if row["application"] == app) for app in applications}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        total = 0.0
        count = 0
        for fold in range(5):
            fold_apps = [app for app in applications if app_fold[app] == fold]
            for app in rng.choice(fold_apps, size=len(fold_apps), replace=True):
                total += float(app_values[app].sum())
                count += len(app_values[app])
        values.append(total / count)
    point = float(np.mean(list(differences.values())))
    return {"point_delta": point, "ci_99": [float(np.quantile(values, 0.005)), float(np.quantile(values, 0.995))], "resamples": resamples, "unit": "application_group"}


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_jsonl_fsynced(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def sha256_file(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUTPUT_PATH.exists() or RAW_PATH.exists():
        raise FileExistsError("EVID Stage 1 output exists")
    config = yaml.safe_load(CONFIG_PATH.read_text())
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    stage0 = json.loads(STAGE0_PATH.read_text())
    selection_manifest = json.loads(SELECTION_PATH.read_text())
    if (
        preflight["status"] != "PASS_EVID_PREFLIGHT_NO_STAGE_RESULT"
        or stage0["status"] != "PASS_EVID_STAGE0_COMPLETE"
        or stage0["proceed_stage1"] is not True
        or config["stage2"]["authorized"] is not False
        or selection_manifest["status"] != "PASS_EVID_STAGE1_NESTED_SELECTIONS_BEFORE_OUTER_EVALUATION"
        or selection_manifest["stage0_sha256"] != sha256_file(STAGE0_PATH)
        or selection_manifest["amendment_sha256"] != sha256_file(AMENDMENT_PATH)
        or selection_manifest["outer_test_outputs_computed"] is not False
    ):
        raise PermissionError("EVID Stage 1 authorization mismatch")
    rows = load_rows()
    row_ids = tuple(sorted(rows))
    stage0_impl = load_module(STAGE0_IMPL_PATH, "evid_stage0_impl")
    decomp = load_module(DECOMP_ARM1_PATH, "evid_decomp_impl")
    fold_for_row = {row_id: int(rows[row_id]["fold"]) for row_id in row_ids}
    baseline_outputs, dev_selection_records = decomp.baseline_outputs(rows, list(row_ids), fold_for_row)
    dev_selection = {}
    for record in dev_selection_records:
        for row_id in row_ids:
            if fold_for_row[row_id] == record["outer_fold"]:
                dev_selection[row_id] = bool(baseline_outputs[record["selected_method"]][row_ids.index(row_id)])

    variants = {name: {} for name in ("fixed", "weighted", "rho_fitted", "exact_singleton")}
    selections = []
    uniform = {model: 1.0 for model in MODELS}
    selections = selection_manifest["folds"]
    if [record["outer_fold"] for record in selections] != list(range(5)):
        raise ValueError("EVID Stage 1 selection fold mismatch")
    for selection in selections:
        outer_fold = selection["outer_fold"]
        outer_test = [row_id for row_id in row_ids if fold_for_row[row_id] == outer_fold]
        selected_rho = selection["selected_rho"]
        outer_weights = selection["outer_lineage_weights"]
        for row_id in outer_test:
            variants["fixed"][row_id] = output_for_row(stage0_impl, rows[row_id], config["method"]["primary"]["rho_v"], config["method"]["primary"]["rho_l"], uniform)[0]
            variants["weighted"][row_id] = output_for_row(stage0_impl, rows[row_id], config["method"]["secondary_weighted"]["rho_v"], config["method"]["secondary_weighted"]["rho_l"], outer_weights)[0]
            variants["rho_fitted"][row_id] = output_for_row(stage0_impl, rows[row_id], selected_rho["rho_v"], selected_rho["rho_l"], uniform)[0]
            variants["exact_singleton"][row_id] = output_for_row(stage0_impl, rows[row_id], config["method"]["primary"]["rho_v"], config["method"]["primary"]["rho_l"], uniform, singleton=True)[0]

    baselines = {
        "B3": {row_id: bool(baseline_outputs["ours"][index]) for index, row_id in enumerate(row_ids)},
        "A2_A3": {row_id: bool(baseline_outputs["A2"][index]) for index, row_id in enumerate(row_ids)},
        "A4": {row_id: bool(baseline_outputs["A4"][index]) for index, row_id in enumerate(row_ids)},
        "majority_best_single": {row_id: bool(baseline_outputs["majority"][index]) for index, row_id in enumerate(row_ids)},
        "nested_dev_selection": dev_selection,
    }
    comparisons = {}
    for variant, outputs in variants.items():
        comparisons[variant] = {}
        for baseline, baseline_values in baselines.items():
            differences = {row_id: int(outputs[row_id]) - int(baseline_values[row_id]) for row_id in row_ids}
            comparisons[variant][baseline] = grouped_bootstrap(rows, differences, config["stage1"]["bootstrap"]["resamples"], 20260820 + 10 * list(variants).index(variant) + list(baselines).index(baseline))

    stage0_raw = [json.loads(line) for line in (ROOT / stage0["raw"]["path"]).read_text().splitlines() if line.strip()]
    path_disagreement = {}
    b3_by_row = {row["row_id"]: tuple(row["b3_block"]) for row in stage0_raw}
    for value in config["method"]["diagonal_path"]["values"]:
        label = f"{value:.1f}"
        path_disagreement[label] = float(np.mean([tuple(row["path_blocks"][label]) != b3_by_row[row["row_id"]] for row in stage0_raw]))
    path_t = np.asarray(config["method"]["diagonal_path"]["values"], dtype=np.float64)
    path_d = np.asarray([path_disagreement[f"{value:.1f}"] for value in path_t], dtype=np.float64)
    path_spearman = float(spearmanr(path_t, path_d).statistic)
    path_systematic = bool(path_spearman > 0.8 and path_disagreement["1.0"] > path_disagreement["0.1"])

    primary = comparisons["fixed"]["nested_dev_selection"]
    exact = comparisons["exact_singleton"]["nested_dev_selection"]
    fixed_distinguishable = primary["ci_99"][0] > 0
    exact_distinguishable = exact["ci_99"][0] > 0
    kill = {
        "E_K1": primary["ci_99"][0] <= 0 <= primary["ci_99"][1],
        "E_K2": False,
        "E_K3": bool(fixed_distinguishable and not exact_distinguishable),
        "E_K4": not path_systematic,
        "E_K5": any(record["rho_boundary_selected"] for record in selections),
        "E_K6": False,
    }
    positive_primary = bool(primary["ci_99"][0] > 0 and primary["point_delta"] >= config["stage1"]["practical_threshold"])
    raw_rows = []
    for row_id in row_ids:
        raw_rows.append({
            "row_id": row_id,
            "application": rows[row_id]["application"],
            "fold": fold_for_row[row_id],
            "variants": {name: bool(values[row_id]) for name, values in variants.items()},
            "baselines": {name: bool(values[row_id]) for name, values in baselines.items()},
        })
    write_jsonl_fsynced(RAW_PATH, raw_rows)
    output = {
        "schema_version": 1,
        "status": "PASS_EVID_STAGE1_COMPLETE",
        "evidence_status": "POST_SELECTION_SINGLE_BENCHMARK_VALIDATION",
        "stage2_authorized": False,
        "stage2_blocked_by_E_G3": stage0["stage2_permanently_blocked"],
        "accuracy": {"variants": {name: float(np.mean(list(values.values()))) for name, values in variants.items()}, "baselines": {name: float(np.mean(list(values.values()))) for name, values in baselines.items()}},
        "comparisons": comparisons,
        "path": {"disagreement_from_B3": path_disagreement, "spearman": path_spearman, "systematic": path_systematic},
        "kill_conditions": kill,
        "positive_primary": positive_primary,
        "single_benchmark_claim_only": True,
        "raw": {"path": str(RAW_PATH.relative_to(ROOT)), "rows": len(raw_rows), "bytes": RAW_PATH.stat().st_size, "sha256": sha256_file(RAW_PATH), "write_flush_fsync_per_row": True},
    }
    atomic_json(OUTPUT_PATH, output)
    print(json.dumps({"status": output["status"], "primary": primary, "positive_primary": positive_primary, "kill_conditions": kill, "stage2_authorized": False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()