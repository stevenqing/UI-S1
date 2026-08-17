import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
LSA_DIR = ROOT / "runs/lsa/2026-08-10"
ICC_DIR = ROOT / "runs/icc/2026-08-15"
CONFIG_PATH = RUN_DIR / "configs/cover_prereg.yaml"
FEASIBILITY_PATH = RUN_DIR / "ARM_B_FEASIBILITY.json"
AMENDMENT_PATH = RUN_DIR / "AMENDMENT_001_TREND_ORDER.md"
ICC_ARM_C_PATH = ICC_DIR / "ARM_C.json"
ICC_RAW_PATH = ICC_DIR / "raw/arm_c_pairwise.jsonl"
OUTPUT_PATH = RUN_DIR / "ARM_B.json"
RAW_PATH = RUN_DIR / "raw/arm_b_pairwise.jsonl"

sys.path.insert(0, str(LSA_DIR))
from lsa_common import load_rows


TREND_ORDER = ("within_model_cross_slot", "cross_model_matched_role", "cross_model_unmatched_role")
PAIRS = tuple((left, right) for left in range(12) for right in range(left + 1, 12))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


icc = load_module(ICC_DIR / "arm_c.py", "cover_icc_formulas")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair_primary_stratum(candidates, left, right):
    return "within_model" if candidates[left].lineage == candidates[right].lineage else "cross_model"


def pair_trend_stratum(candidates, left, right):
    same_model = candidates[left].lineage == candidates[right].lineage
    left_role, right_role = left // 3, right // 3
    if same_model:
        return "within_model_cross_slot"
    return "cross_model_matched_role" if left_role == right_role else "cross_model_unmatched_role"


def sufficient(rows, row_ids, groups):
    failures = np.asarray([[not candidate.success for candidate in rows[row_id].candidates] for row_id in row_ids], dtype=np.float64)
    group_for_row = [rows[row_id].group for row_id in row_ids]
    n = np.zeros(len(groups), dtype=np.int64)
    sums = np.zeros((len(groups), 12), dtype=np.float64)
    cross = np.zeros((len(groups), 12, 12), dtype=np.float64)
    for index, group in enumerate(groups):
        selected = failures[np.asarray([value == group for value in group_for_row])]
        n[index] = len(selected)
        sums[index] = selected.sum(axis=0)
        cross[index] = selected.T @ selected
    return n, sums, cross


def fold_stats(n, sums, cross, group_mask, candidates):
    total_n = int(n[group_mask].sum())
    total_sums = sums[group_mask].sum(axis=0)
    total_cross = cross[group_mask].sum(axis=0)
    phi_matrix = np.eye(12, dtype=np.float64)
    records = []
    for left, right in PAIRS:
        phi = icc.phi_from_counts(total_n, total_sums[left], total_sums[right], total_cross[left, right])
        kappa = icc.kappa_from_counts(total_n, total_sums[left], total_sums[right], total_cross[left, right])
        phi_matrix[left, right] = phi_matrix[right, left] = 0.0 if phi is None else phi
        records.append({"left": left, "right": right, "left_source": candidates[left].source, "right_source": candidates[right].source, "left_model": candidates[left].lineage, "right_model": candidates[right].lineage, "left_role": left // 3, "right_role": right // 3, "primary_stratum": pair_primary_stratum(candidates, left, right), "trend_stratum": pair_trend_stratum(candidates, left, right), "phi": phi, "kappa": kappa, "phi_zero_fill": 0.0 if phi is None else phi, "kappa_zero_fill": 0.0 if kappa is None else kappa})
    summaries = {}
    for stratum in ("within_model", "cross_model", *TREND_ORDER):
        key = "primary_stratum" if stratum in {"within_model", "cross_model"} else "trend_stratum"
        current = [record for record in records if record[key] == stratum]
        valid = [record["phi"] for record in current if record["phi"] is not None]
        summaries[stratum] = {"pairs": len(current), "valid_phi_pairs": len(valid), "undefined_phi_pairs": len(current) - len(valid), "phi_mean_valid": float(np.mean(valid)) if valid else None, "phi_mean_zero_fill": float(np.mean([record["phi_zero_fill"] for record in current])), "kappa_mean_zero_fill": float(np.mean([record["kappa_zero_fill"] for record in current]))}
    return records, summaries, icc.neff(phi_matrix)


def bootstrap(n, sums, cross, groups, group_fold, outer_fold, candidates, resamples, seed):
    rng = np.random.default_rng(seed)
    strata = ("within_model", "cross_model", *TREND_ORDER)
    values = {name: [] for name in strata}
    for _ in range(resamples):
        multiplicity = np.zeros(len(groups), dtype=np.int64)
        for fold in range(5):
            if fold == outer_fold:
                continue
            indices = np.asarray([index for index, group in enumerate(groups) if group_fold[group] == fold], dtype=np.int64)
            sampled = rng.choice(indices, size=len(indices), replace=True)
            multiplicity += np.bincount(sampled, minlength=len(groups))
        total_n = int(multiplicity @ n)
        total_sums = multiplicity @ sums
        total_cross = np.tensordot(multiplicity, cross, axes=(0, 0))
        pair_values = {name: [] for name in strata}
        for left, right in PAIRS:
            phi = icc.phi_from_counts(total_n, total_sums[left], total_sums[right], total_cross[left, right])
            if phi is not None:
                pair_values[pair_primary_stratum(candidates, left, right)].append(phi)
                pair_values[pair_trend_stratum(candidates, left, right)].append(phi)
        for name in strata:
            if not pair_values[name]:
                raise ValueError(f"COVER Arm B bootstrap no valid {name} pairs")
            values[name].append(float(np.mean(pair_values[name])))
    return {name: [float(np.quantile(current, 0.005)), float(np.quantile(current, 0.995))] for name, current in values.items()}


def ordering(values):
    return sorted(TREND_ORDER, key=lambda name: (-values[name], TREND_ORDER.index(name)))


def sspro_trend():
    records = [json.loads(line) for line in ICC_RAW_PATH.read_text().splitlines() if line.strip()]
    fold_values = {name: [] for name in TREND_ORDER}
    for fold in range(5):
        current = [record for record in records if record["outer_fold"] == fold]
        mapping = {tuple(record["left_source"]): record["left"] for record in current}
        by_name = {name: [] for name in TREND_ORDER}
        for record in current:
            left_model, left_view = record["left_source"]
            right_model, right_view = record["right_source"]
            if left_model == right_model:
                name = "within_model_cross_slot"
            else:
                name = "cross_model_matched_role" if left_view == right_view else "cross_model_unmatched_role"
            if record["phi"] is not None:
                by_name[name].append(record["phi"])
        for name in TREND_ORDER:
            fold_values[name].append(float(np.mean(by_name[name])))
    means = {name: float(np.mean(values)) for name, values in fold_values.items()}
    return {"fold_values": fold_values, "means": means, "ordering": ordering(means)}


def write_jsonl_fsynced(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def main():
    if OUTPUT_PATH.exists() or RAW_PATH.exists():
        raise FileExistsError("COVER Arm B output exists")
    config = yaml.safe_load(CONFIG_PATH.read_text())
    feasibility = json.loads(FEASIBILITY_PATH.read_text())
    if feasibility["mind2web"]["status"] != "READY_2080x12":
        raise PermissionError("COVER Arm B feasibility mismatch")
    rows = load_rows("C_uni")["mind2web"]
    row_ids = tuple(sorted(rows))
    if len(row_ids) != 2080 or any(len(rows[row_id].candidates) != 12 for row_id in row_ids):
        raise ValueError("COVER Arm B row/candidate mismatch")
    template = rows[row_ids[0]].candidates
    expected_roles = ["stage1_TongUI-7B_view0", "stage1_CogAgent-18B_view0", "stage1_UI-TARS-7B_view0", "stage1_TongUI-7B_view1", "stage1_CogAgent-18B_view1", "stage1_UI-TARS-7B_view1", "stage2_TongUI-7B_crop0", "stage2_CogAgent-18B_crop0", "stage2_UI-TARS-7B_crop0", "stage2_TongUI-7B_crop1", "stage2_CogAgent-18B_crop1", "stage2_UI-TARS-7B_crop1"]
    if [candidate.source for candidate in template] != expected_roles:
        raise ValueError("COVER Arm B slot-role order mismatch")
    groups = sorted({rows[row_id].group for row_id in row_ids})
    group_fold = {}
    for group in groups:
        folds = {rows[row_id].fold for row_id in row_ids if rows[row_id].group == group}
        if len(folds) != 1:
            raise ValueError(f"COVER episode crosses folds: {group}")
        group_fold[group] = next(iter(folds))
    n, sums, cross = sufficient(rows, row_ids, groups)
    folds = []
    raw = []
    heldout_sizes = []
    for outer_fold in range(5):
        mask = np.asarray([group_fold[group] != outer_fold for group in groups])
        pair_records, summaries, phi_neff = fold_stats(n, sums, cross, mask, template)
        cis = bootstrap(n, sums, cross, groups, group_fold, outer_fold, template, config["arm_b"]["bootstrap"]["resamples"], 20261100 + outer_fold)
        for record in pair_records:
            raw.append({"outer_fold": outer_fold, **record})
        folds.append({"outer_fold": outer_fold, "development_rows": int(n[mask].sum()), "summaries": summaries, "phi_ci_99": cis, "empirical_phi_neff": phi_neff})
        heldout_sizes.append(sum(rows[row_id].fold == outer_fold for row_id in row_ids))
    weights = np.asarray(heldout_sizes) / sum(heldout_sizes)
    summary = {}
    for name in ("within_model", "cross_model", *TREND_ORDER):
        values = [fold["summaries"][name]["phi_mean_valid"] for fold in folds]
        summary[name] = {"fold_values": values, "fold_mean": float(np.mean(values)), "fold_range": [float(min(values)), float(max(values))], "fold_ci_99": [fold["phi_ci_99"][name] for fold in folds], "undefined_phi_pairs_by_fold": [fold["summaries"][name]["undefined_phi_pairs"] for fold in folds], "kappa_zero_fill_fold_values": [fold["summaries"][name]["kappa_mean_zero_fill"] for fold in folds]}
    neff = float(np.average([fold["empirical_phi_neff"] for fold in folds], weights=weights))
    mind_means = {name: summary[name]["fold_mean"] for name in TREND_ORDER}
    screen = sspro_trend()
    mind_order = ordering(mind_means)
    write_jsonl_fsynced(RAW_PATH, raw)
    output = {"schema_version": 1, "status": "PASS_COVER_ARM_B_DEPENDENCE_COMPLETE", "evidence_status": "POST_SELECTION_DIAGNOSTIC", "rows": len(row_ids), "candidates_per_row": 12, "models": config["arm_b"]["models"], "slot_roles": config["arm_b"]["slot_roles"], "folds": folds, "summaries": summary, "empirical_phi_neff": neff, "references": config["references"], "trend": {"mind2web": {"means": mind_means, "ordering": mind_order}, "screenspot_pro": screen, "ordering_consistent": mind_order == screen["ordering"], "interpretation": "source_stage_not_model_scale"}, "source_hashes": {"feasibility": sha256_file(FEASIBILITY_PATH), "amendment": sha256_file(AMENDMENT_PATH), "icc_arm_c": sha256_file(ICC_ARM_C_PATH), "icc_raw": sha256_file(ICC_RAW_PATH)}, "raw": {"path": str(RAW_PATH.relative_to(ROOT)), "rows": len(raw), "bytes": RAW_PATH.stat().st_size, "sha256": sha256_file(RAW_PATH), "write_flush_fsync_per_row": True}}
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": output["status"], "summary": {name: value["fold_mean"] for name, value in summary.items()}, "neff": neff, "trend": output["trend"]}, indent=2))


if __name__ == "__main__":
    main()