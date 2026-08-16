import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
ALLOCATION_DIR = ROOT / "runs/allocation-law/2026-08-01"
CONFIG_PATH = RUN_DIR / "configs/icc_prereg.yaml"
SPEC_PATH = RUN_DIR / "SPEC.md"
AMENDMENT_PATH = RUN_DIR / "AMENDMENT_001_SAME_BUDGET_METHOD_NAMES.md"
OUTPUT_PATH = RUN_DIR / "SAME_BUDGET.json"
RAW_PATH = RUN_DIR / "raw/same_budget_rows.jsonl"

MANIFEST_PATH = ALLOCATION_DIR / "raw/shared_regions_n12.jsonl"
GTA1_ROOT = ROOT / "runs/ccm-h2h/2026-07-31/h1/shards/top18"
QWEN_OLD = ROOT / "runs/ccm-h2h/2026-07-31/h3/shards/qwen3_views"
UITARS_OLD = ROOT / "runs/ccm-h2h/2026-07-31/h3/shards/uitars_views"
QWEN_EXTENDED = tuple(sorted((ALLOCATION_DIR / "shards").glob("qwen3-views-4-11-*.jsonl")))
UITARS_EXTENDED = tuple(sorted((ALLOCATION_DIR / "shards").glob("uitars-views-4-11-*.jsonl")))
L2_CONFIG = ALLOCATION_DIR / "configs/l2_pools.yaml"

POOL_NAMES = ("three_lineages_4x3", "qwen3_uitars_6x2", "gta1_uitars_6x2", "gta1_qwen3_6x2")
OMITTED = {"qwen3_uitars_6x2": "GTA1-7B", "gta1_uitars_6x2": "Qwen3-VL-8B-Instruct", "gta1_qwen3_6x2": "UI-TARS-7B-SFT"}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_priority_outputs(module, rows, fold_for_group):
    outputs = {}
    fold_priorities = []
    for fold in range(5):
        development = [row for row in rows if fold_for_group[row["application"]] != fold]
        test = [row for row in rows if fold_for_group[row["application"]] == fold]
        reliability = []
        for index in range(12):
            reliability.append(float(np.mean([module.point_in_bbox(row["candidates"][index]["point"], row["target_bbox"]) for row in development])))
        selected = max(range(12), key=lambda index: (reliability[index], -index))
        fold_priorities.append({"outer_fold": fold, "selected_index": selected, "selected_source": [rows[0]["candidates"][selected]["model"], rows[0]["candidates"][selected]["view_index"]], "reliability": reliability})
        for row in test:
            outputs[row["id"]] = bool(module.point_in_bbox(row["candidates"][selected]["point"], row["target_bbox"]))
    return outputs, fold_priorities


def paired_bootstrap(rows, left, right, fold_for_group, resamples, seed):
    groups = sorted(fold_for_group)
    group_values = {group: np.asarray([int(left[row["id"]]) - int(right[row["id"]]) for row in rows if row["application"] == group], dtype=np.float64) for group in groups}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        total = 0.0
        count = 0
        for fold in range(5):
            current = [group for group in groups if fold_for_group[group] == fold]
            for group in rng.choice(current, size=len(current), replace=True):
                total += float(group_values[group].sum())
                count += len(group_values[group])
        values.append(total / count)
    return {"point_delta": float(np.mean([int(left[row["id"]]) - int(right[row["id"]]) for row in rows])), "ci_99": [float(np.quantile(values, 0.005)), float(np.quantile(values, 0.995))], "resamples": resamples, "unit": "application_group"}


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
        raise FileExistsError("ICC same-budget output exists")
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if config["same_budget"]["aggregators"] != ["B3_mvp", "M1_ccm", "source_priority"]:
        raise PermissionError("ICC same-budget method-name mismatch")
    allocation = load_module(ALLOCATION_DIR / "allocation_eval.py", "icc_allocation_eval")
    manifest = allocation.load_manifest(MANIFEST_PATH)
    gta1 = allocation.load_gta1(GTA1_ROOT, manifest)
    generated = {
        "Qwen3-VL-8B-Instruct": allocation.load_model_views(QWEN_OLD, QWEN_EXTENDED, manifest, "Qwen3-VL-8B-Instruct"),
        "UI-TARS-7B-SFT": allocation.load_model_views(UITARS_OLD, UITARS_EXTENDED, manifest, "UI-TARS-7B-SFT"),
    }
    units = allocation.l2_units(L2_CONFIG)
    pools = {name: allocation.build_pool(gta1, generated, units[name]) for name in POOL_NAMES}
    fold_for_group, _ = allocation.group_folds(pools["three_lineages_4x3"])
    reports = {}
    outputs = {}
    priorities = {}
    for name, rows in pools.items():
        evaluation = allocation.compact_evaluation(rows)
        source_outputs, fold_priorities = source_priority_outputs(allocation, rows, fold_for_group)
        reports[name] = {"units": [f"{model}/view{view}" for model, view in units[name]], "accuracy": {"B3_mvp": evaluation["accuracy"]["B3_mvp"], "M1_ccm": evaluation["accuracy"]["M1_ccm"], "source_priority": float(np.mean(list(source_outputs.values())))} }
        outputs[name] = {"B3_mvp": evaluation["outputs"]["B3_mvp"], "M1_ccm": evaluation["outputs"]["M1_ccm"], "source_priority": source_outputs}
        priorities[name] = fold_priorities
    if abs(reports["three_lineages_4x3"]["accuracy"]["B3_mvp"] - 0.6369386464263125) > 1e-15 or abs(reports["gta1_qwen3_6x2"]["accuracy"]["B3_mvp"] - 0.6375711574952562) > 1e-15 or abs(reports["gta1_qwen3_6x2"]["accuracy"]["M1_ccm"] - 0.6388361796331435) > 1e-15:
        raise ValueError("ICC same-budget historical anchor mismatch")
    comparisons = {}
    full_rows = pools["three_lineages_4x3"]
    for pool, omitted in OMITTED.items():
        comparisons[omitted] = {}
        for method in config["same_budget"]["aggregators"]:
            comparisons[omitted][method] = paired_bootstrap(full_rows, outputs["three_lineages_4x3"][method], outputs[pool][method], fold_for_group, config["same_budget"]["bootstrap"]["resamples"], 20260900 + 10 * list(OMITTED).index(pool) + config["same_budget"]["aggregators"].index(method))
    raw = []
    rows_by_pool = {name: {row["id"]: row for row in rows} for name, rows in pools.items()}
    for row in full_rows:
        row_id = row["id"]
        raw.append({"row_id": row_id, "application": row["application"], "fold": fold_for_group[row["application"]], "outputs": {pool: {method: bool(outputs[pool][method][row_id]) for method in config["same_budget"]["aggregators"]} for pool in POOL_NAMES}})
    write_jsonl_fsynced(RAW_PATH, raw)
    output = {"schema_version": 1, "status": "PASS_ICC_SAME_BUDGET_AUDIT", "budget": 12, "pools": reports, "source_priority_folds": priorities, "comparisons_full_minus_omit": comparisons, "historical_method_name": "M1_ccm", "decomp_bridge_method_name": "source_priority", "estimand_distinction": "DECOMP_equal_cell_2_to_3_is_not_named_lineage_3x4_minus_2x6", "source_hashes": {"spec": sha256_file(SPEC_PATH), "amendment": sha256_file(AMENDMENT_PATH), "l2_config": sha256_file(L2_CONFIG), "manifest": sha256_file(MANIFEST_PATH)}, "raw": {"path": str(RAW_PATH.relative_to(ROOT)), "rows": len(raw), "bytes": RAW_PATH.stat().st_size, "sha256": sha256_file(RAW_PATH), "write_flush_fsync_per_row": True}}
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": output["status"], "accuracy": {name: value["accuracy"] for name, value in reports.items()}, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()