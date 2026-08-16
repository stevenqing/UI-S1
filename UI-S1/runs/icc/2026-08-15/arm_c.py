import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
MASK_DIR = ROOT / "runs/mask/2026-08-14"
CONFIG_PATH = RUN_DIR / "configs/icc_prereg.yaml"
SPEC_PATH = RUN_DIR / "SPEC.md"
MASK_STAGE1_PATH = ROOT / "runs/mask/2026-08-14/STAGE1.json"
OUTPUT_PATH = RUN_DIR / "ARM_C.json"
RAW_PATH = RUN_DIR / "raw/arm_c_pairwise.jsonl"

sys.path.insert(0, str(MASK_DIR))
from mask_common import load_rows


MODELS = ("GTA1-7B", "Qwen3-VL-8B-Instruct", "UI-TARS-7B-SFT")
SLOTS = tuple((model, view) for view in range(4) for model in MODELS)
PAIRS = tuple((left, right) for left in range(12) for right in range(left + 1, 12))


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair_stratum(left, right):
    return "within_lineage" if SLOTS[left][0] == SLOTS[right][0] else "cross_lineage"


def phi_from_counts(n, sx, sy, sxy):
    denominator = (n * sx - sx * sx) * (n * sy - sy * sy)
    if denominator <= 0:
        return None
    return float((n * sxy - sx * sy) / np.sqrt(denominator))


def kappa_from_counts(n, sx, sy, sxy):
    if n <= 0:
        return None
    both_one = sxy
    both_zero = n - sx - sy + sxy
    observed = (both_one + both_zero) / n
    px, py = sx / n, sy / n
    expected = px * py + (1 - px) * (1 - py)
    if expected >= 1:
        return None
    return float((observed - expected) / (1 - expected))


def neff(matrix):
    denominator = float(np.ones(len(matrix)) @ matrix @ np.ones(len(matrix)))
    return float(len(matrix) ** 2 / denominator)


def structured_neff(rho_v, rho_l):
    return float(144 / (12 + 36 * rho_v + 96 * rho_l))


def app_sufficient(rows, row_ids, applications):
    failures = np.asarray([[not candidate["correct"] for candidate in rows[row_id]["candidates"]] for row_id in row_ids], dtype=np.float64)
    app_for_row = [rows[row_id]["application"] for row_id in row_ids]
    n = np.zeros(len(applications), dtype=np.int64)
    sums = np.zeros((len(applications), 12), dtype=np.float64)
    cross = np.zeros((len(applications), 12, 12), dtype=np.float64)
    for index, app in enumerate(applications):
        selected = failures[np.asarray([value == app for value in app_for_row])]
        n[index] = len(selected)
        sums[index] = selected.sum(axis=0)
        cross[index] = selected.T @ selected
    return n, sums, cross


def fold_statistics(n, sums, cross, app_mask):
    total_n = int(n[app_mask].sum())
    total_sums = sums[app_mask].sum(axis=0)
    total_cross = cross[app_mask].sum(axis=0)
    phi_matrix = np.eye(12, dtype=np.float64)
    kappa_matrix = np.eye(12, dtype=np.float64)
    records = []
    for left, right in PAIRS:
        phi = phi_from_counts(total_n, total_sums[left], total_sums[right], total_cross[left, right])
        kappa = kappa_from_counts(total_n, total_sums[left], total_sums[right], total_cross[left, right])
        phi_matrix[left, right] = phi_matrix[right, left] = 0.0 if phi is None else phi
        kappa_matrix[left, right] = kappa_matrix[right, left] = 0.0 if kappa is None else kappa
        records.append({"left": left, "right": right, "left_source": list(SLOTS[left]), "right_source": list(SLOTS[right]), "stratum": pair_stratum(left, right), "phi": phi, "kappa": kappa, "phi_zero_fill": 0.0 if phi is None else phi, "kappa_zero_fill": 0.0 if kappa is None else kappa})
    summary = {}
    for stratum in ("within_lineage", "cross_lineage"):
        current = [record for record in records if record["stratum"] == stratum]
        valid_phi = [record["phi"] for record in current if record["phi"] is not None]
        summary[stratum] = {
            "pairs": len(current),
            "valid_phi_pairs": len(valid_phi),
            "undefined_phi_pairs": len(current) - len(valid_phi),
            "phi_mean_valid": float(np.mean(valid_phi)) if valid_phi else None,
            "phi_mean_zero_fill": float(np.mean([record["phi_zero_fill"] for record in current])),
            "kappa_mean_zero_fill": float(np.mean([record["kappa_zero_fill"] for record in current])),
        }
    return records, summary, neff(phi_matrix), neff(kappa_matrix)


def bootstrap_phi(n, sums, cross, applications, app_fold, outer_fold, resamples, seed):
    rng = np.random.default_rng(seed)
    included_folds = [fold for fold in range(5) if fold != outer_fold]
    values = {"within_lineage": [], "cross_lineage": []}
    for _ in range(resamples):
        multiplicity = np.zeros(len(applications), dtype=np.int64)
        for fold in included_folds:
            indices = np.asarray([index for index, app in enumerate(applications) if app_fold[app] == fold], dtype=np.int64)
            selected = rng.choice(indices, size=len(indices), replace=True)
            multiplicity += np.bincount(selected, minlength=len(applications))
        total_n = int(multiplicity @ n)
        total_sums = multiplicity @ sums
        total_cross = np.tensordot(multiplicity, cross, axes=(0, 0))
        pair_values = {"within_lineage": [], "cross_lineage": []}
        for left, right in PAIRS:
            value = phi_from_counts(total_n, total_sums[left], total_sums[right], total_cross[left, right])
            if value is not None:
                pair_values[pair_stratum(left, right)].append(value)
        for stratum in values:
            if not pair_values[stratum]:
                raise ValueError(f"ICC Arm C bootstrap has no valid {stratum} phi pairs")
            values[stratum].append(float(np.mean(pair_values[stratum])))
    return {stratum: [float(np.quantile(current, 0.005)), float(np.quantile(current, 0.995))] for stratum, current in values.items()}


def write_jsonl_fsynced(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def main():
    if OUTPUT_PATH.exists() or RAW_PATH.exists():
        raise FileExistsError("ICC Arm C output exists")
    config = yaml.safe_load(CONFIG_PATH.read_text())
    mask_stage1 = json.loads(MASK_STAGE1_PATH.read_text())
    if config["status"] != "PREREGISTERED_BEFORE_ANY_ICC_RESULT" or mask_stage1["neff_calibration"]["full_pool_cross_fitted_neff"] != config["arm_c"]["neff"]["mask_anchor"]:
        raise PermissionError("ICC Arm C input mismatch")
    rows = load_rows()
    row_ids = tuple(sorted(rows))
    applications = sorted({rows[row_id]["application"] for row_id in row_ids})
    app_fold = {app: next(rows[row_id]["fold"] for row_id in row_ids if rows[row_id]["application"] == app) for app in applications}
    n, sums, cross = app_sufficient(rows, row_ids, applications)
    raw = []
    folds = []
    heldout_sizes = []
    for outer_fold in range(5):
        development_mask = np.asarray([app_fold[app] != outer_fold for app in applications])
        pair_records, summary, phi_neff, kappa_neff = fold_statistics(n, sums, cross, development_mask)
        bootstrap = bootstrap_phi(n, sums, cross, applications, app_fold, outer_fold, config["arm_c"]["bootstrap"]["resamples"], 20260830 + outer_fold)
        for record in pair_records:
            raw.append({"outer_fold": outer_fold, **record})
        rho_v = summary["within_lineage"]["phi_mean_valid"]
        rho_l = summary["cross_lineage"]["phi_mean_valid"]
        folds.append({"outer_fold": outer_fold, "development_rows": int(n[development_mask].sum()), "strata": summary, "phi_ci_99": bootstrap, "empirical_phi_neff": phi_neff, "empirical_kappa_neff": kappa_neff, "structured_phi_neff": structured_neff(rho_v, rho_l)})
        heldout_sizes.append(sum(rows[row_id]["fold"] == outer_fold for row_id in row_ids))
    weights = np.asarray(heldout_sizes, dtype=np.float64) / sum(heldout_sizes)
    phi_neff = float(np.average([fold["empirical_phi_neff"] for fold in folds], weights=weights))
    kappa_neff = float(np.average([fold["empirical_kappa_neff"] for fold in folds], weights=weights))
    structured = float(np.average([fold["structured_phi_neff"] for fold in folds], weights=weights))
    mask_anchor = config["arm_c"]["neff"]["mask_anchor"]
    if abs(kappa_neff - mask_anchor) > 1e-12:
        raise ValueError(f"ICC MASK N_eff anchor mismatch: {kappa_neff} != {mask_anchor}")
    summaries = {}
    for stratum, reference in (("within_lineage", 0.895), ("cross_lineage", 0.398)):
        values = [fold["strata"][stratum]["phi_mean_valid"] for fold in folds]
        summaries[stratum] = {"fold_values": values, "fold_mean": float(np.mean(values)), "fold_range": [float(min(values)), float(max(values))], "androidcontrol_reference": reference, "signed_difference": float(np.mean(values) - reference), "fold_ci_99": [fold["phi_ci_99"][stratum] for fold in folds], "undefined_phi_pairs_by_fold": [fold["strata"][stratum]["undefined_phi_pairs"] for fold in folds], "kappa_zero_fill_fold_values": [fold["strata"][stratum]["kappa_mean_zero_fill"] for fold in folds]}
    kappa_error = abs(kappa_neff - phi_neff) / phi_neff
    structured_error = abs(structured - phi_neff) / phi_neff
    threshold = config["arm_c"]["neff"]["relative_error_threshold"]
    support = bool(kappa_error <= threshold and structured_error <= threshold)
    write_jsonl_fsynced(RAW_PATH, raw)
    output = {"schema_version": 1, "status": "PASS_ICC_ARM_C_DIRECT_ERROR_DEPENDENCE", "evidence_status": "POST_SELECTION_DIAGNOSTIC", "folds": folds, "summaries": summaries, "neff": {"empirical_phi": phi_neff, "empirical_kappa": kappa_neff, "structured_phi": structured, "mask_anchor": mask_anchor, "kappa_vs_phi_relative_error": kappa_error, "structured_vs_phi_relative_error": structured_error, "threshold": threshold}, "retrospective_A2_supported": support, "historical_GRAN_G_P8_status": "NOT_ADJUDICABLE_PREREG_UNDERDEFINED", "historical_status_changed": False, "raw": {"path": str(RAW_PATH.relative_to(ROOT)), "rows": len(raw), "bytes": RAW_PATH.stat().st_size, "sha256": sha256_file(RAW_PATH), "write_flush_fsync_per_row": True}, "spec_sha256": sha256_file(SPEC_PATH)}
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": output["status"], "rho": summaries, "neff": output["neff"], "A2_supported": support}, indent=2))


if __name__ == "__main__":
    main()