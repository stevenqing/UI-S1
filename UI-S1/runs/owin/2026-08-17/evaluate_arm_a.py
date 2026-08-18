import importlib.util
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
SAMPLE_PATH = RUN_DIR / "SAMPLE_WINDOW_MANIFEST.jsonl"
PREFLIGHT_PATH = RUN_DIR / "PREFLIGHT.json"
ARM_B_PATH = RUN_DIR / "ARM_B.json"
COMMON_PATH = RUN_DIR / "COMMON_CALIBRATION.json"
COVER_ROWS_PATH = ROOT / "runs/cover/2026-08-16/raw/arm_a_rows.jsonl"
CWIN_ROWS_PATH = ROOT / "runs/cwin/2026-08-17/raw/stage0_rows.jsonl"
GTA1_ROOT = ROOT / "runs/ccm-h2h/2026-07-31/h1/shards/top18"
SOURCEBIAS_PATH = ROOT / "runs/sourcebias/2026-08-03/sourcebias_common.py"
H3_PATH = ROOT / "runs/ccm-h2h/2026-07-31/h3/h3_eval.py"
OUTPUT_PATH = RUN_DIR / "ARM_A.json"
PRIVATE_ROWS_PATH = RUN_DIR / "raw/private_arm_a_evaluation.jsonl"
STRATA = ("uncovered_0", "partial_1_10", "common_11")
SHARD_PATHS = {stratum: RUN_DIR / f"raw/arm_a_{stratum}.jsonl" for stratum in STRATA}
SHARD_STATUS_PATHS = {stratum: RUN_DIR / f"raw/arm_a_{stratum}_status.json" for stratum in STRATA}
RESAMPLES = 10000


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sourcebias = load_module(SOURCEBIAS_PATH, "owin_final_sourcebias")
h3 = load_module(H3_PATH, "owin_final_h3")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_jsonl_fsynced(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=json_scalar)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def json_scalar(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def weighted_ratio(rows, field, multiplicity=None):
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        multiple = 1 if multiplicity is None else multiplicity.get(row["application"], 0)
        weight = row["inverse_probability_weight"] * multiple
        numerator += weight * float(row[field])
        denominator += weight
    return numerator / denominator if denominator > 0 else None


def population_ratio(rows, field, multiplicity=None):
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        multiple = 1 if multiplicity is None else multiplicity.get(row["application"], 0)
        numerator += multiple * float(row[field])
        denominator += multiple
    return numerator / denominator if denominator > 0 else None


def phi_from_weighted(values_left, values_right, weights):
    total = float(np.sum(weights))
    sx = float(weights @ values_left)
    sy = float(weights @ values_right)
    sxy = float(weights @ (values_left * values_right))
    denominator = (total * sx - sx * sx) * (total * sy - sy * sy)
    if denominator <= 0:
        return None
    return float((total * sxy - sx * sy) / math.sqrt(denominator))


def dependence_endpoint(error_matrix, weights):
    slot_count = error_matrix.shape[1]
    pairs = []
    constant_slots = int(sum(bool(np.all(error_matrix[:, slot] == error_matrix[0, slot])) for slot in range(slot_count)))
    for left in range(slot_count):
        for right in range(left + 1, slot_count):
            pairs.append(phi_from_weighted(error_matrix[:, left], error_matrix[:, right], weights))
    valid = [value for value in pairs if value is not None]
    undefined = len(pairs) - len(valid)
    mean_valid = float(np.mean(valid)) if valid else None
    output = {"pair_count": len(pairs), "valid_pair_count": len(valid), "undefined_pair_count": undefined, "constant_slot_count": constant_slots, "nonconstant_slot_count": slot_count - constant_slots, "mean_valid_signed_phi": mean_valid, "mean_zero_filled_off_diagonal_phi": float(np.mean([0.0 if value is None else value for value in pairs]))}
    matrix_zero = np.eye(slot_count)
    matrix_mean = np.eye(slot_count) if valid else None
    index = 0
    for left in range(slot_count):
        for right in range(left + 1, slot_count):
            value = pairs[index]
            matrix_zero[left, right] = matrix_zero[right, left] = 0.0 if value is None else value
            if matrix_mean is not None:
                matrix_mean[left, right] = matrix_mean[right, left] = mean_valid if value is None else value
            index += 1
    for name, matrix in (("zero", matrix_zero), ("mean", matrix_mean)):
        if matrix is None:
            output[f"{name}_denominator"] = None
            output[f"neff_{name}"] = None
        else:
            denominator = float(np.ones(slot_count) @ matrix @ np.ones(slot_count))
            output[f"{name}_denominator"] = denominator
            output[f"neff_{name}"] = slot_count**2 / denominator if math.isfinite(denominator) and denominator > 0 else None
    output["reliability"] = "DEPENDENCE_DIAGNOSTIC_UNRELIABLE" if undefined > 11 else "RELIABLE"
    return output


def dependence_label(interval):
    if interval[0] > 0.10:
        return "MATERIAL_DEPENDENCE_MISMATCH"
    if interval[1] < 0.10:
        return "APPROXIMATELY_MATCHED"
    return "DEPENDENCE_MATCH_INDETERMINATE"


def coverage_stratum(count):
    if count == 0:
        return "uncovered_0"
    if count == 11:
        return "common_11"
    return "partial_1_10"


def dependence_bootstrap(rows, resamples, seed):
    applications = sorted({row["application"] for row in rows})
    rng = np.random.default_rng(seed)
    values = {"zero": [], "mean": []}
    for _ in range(resamples):
        multiplicity = Counter(rng.choice(applications, size=len(applications), replace=True))
        active = [row for row in rows if multiplicity[row["application"]] > 0]
        if not active:
            continue
        weights = np.asarray([row["inverse_probability_weight"] * multiplicity[row["application"]] for row in active], dtype=float)
        oracle = dependence_endpoint(np.asarray([row["oracle_crop_errors"] for row in active], dtype=float), weights)
        existing = dependence_endpoint(np.asarray([row["existing_crop_errors"] for row in active], dtype=float), weights)
        if oracle["reliability"] != "RELIABLE" or existing["reliability"] != "RELIABLE" or existing["constant_slot_count"] == 11:
            continue
        for fill in ("zero", "mean"):
            oracle_value = oracle[f"neff_{fill}"]
            existing_value = existing[f"neff_{fill}"]
            if oracle_value is not None and existing_value is not None and math.isfinite(existing_value) and existing_value > 0:
                values[fill].append(abs(oracle_value - existing_value) / existing_value)
    output = {}
    for fill, current in values.items():
        if len(current) < 0.99 * resamples:
            output[fill] = {"finite_reliable_replicates": len(current), "ci_99": None, "label": "DEPENDENCE_BOOTSTRAP_UNRELIABLE"}
        else:
            interval = [float(np.quantile(current, 0.005)), float(np.quantile(current, 0.995))]
            output[fill] = {"finite_reliable_replicates": len(current), "ci_99": interval, "label": dependence_label(interval)}
    labels = [value["label"] for value in output.values()]
    if "MATERIAL_DEPENDENCE_MISMATCH" in labels:
        combined = "MATERIAL_DEPENDENCE_MISMATCH"
    elif any(label in {"DEPENDENCE_MATCH_INDETERMINATE", "DEPENDENCE_BOOTSTRAP_UNRELIABLE"} for label in labels):
        combined = "DEPENDENCE_MATCH_INDETERMINATE"
    else:
        combined = "APPROXIMATELY_MATCHED"
    output["combined_label"] = combined
    return output


def fit_m1(rows):
    outputs = {}
    for fold in range(5):
        development = [row for row in rows if row["fold"] != fold]
        test = [row for row in rows if row["fold"] == fold]
        tables, priors = h3.fit_ccm(development)
        for row in test:
            selected = h3.ccm_select(row, tables, priors)
            outputs[row["id"]] = bool(h3.point_in_bbox(row["candidates"][selected]["point"], row["target_bbox"]))
    return outputs


def corrected_values(sample_rows, population_rows, method, multiplicity=None, delta_override=None):
    raw = {stratum: weighted_ratio([row for row in sample_rows if row["stratum"] == stratum], f"oracle_{method}_correct", multiplicity) for stratum in STRATA}
    existing = {stratum: population_ratio([row for row in population_rows if row["stratum"] == stratum], f"existing_{method}_correct", multiplicity) for stratum in STRATA}
    delta = raw["common_11"] - existing["common_11"] if delta_override is None else delta_override
    corrected = {stratum: min(1.0, max(0.0, raw[stratum] - delta)) for stratum in STRATA}
    fractions = {stratum: sum(row["stratum"] == stratum for row in population_rows) / len(population_rows) for stratum in STRATA}
    perfect = sum(fractions[stratum] * corrected[stratum] for stratum in STRATA)
    baseline = sum(fractions[stratum] * existing[stratum] for stratum in STRATA)
    return {"raw": raw, "existing": existing, "delta_common": delta, "corrected": corrected, "perfect_accuracy": perfect, "perfect_gain": perfect - baseline}


def bootstrap_endpoints(sample_rows, population_rows, single_population_rows, arm_b, resamples=RESAMPLES, seed=20260817):
    applications = sorted({row["application"] for row in population_rows})
    rng = np.random.default_rng(seed)
    values = defaultdict(list)
    for _ in range(resamples):
        multiplicity = Counter(rng.choice(applications, size=len(applications), replace=True))
        b3 = corrected_values(sample_rows, population_rows, "b3", multiplicity)
        m1 = corrected_values(sample_rows, population_rows, "m1", multiplicity)
        if None in [*b3["raw"].values(), *b3["existing"].values(), *m1["raw"].values(), *m1["existing"].values()]:
            continue
        values["B3_perfect_gain"].append(b3["perfect_gain"])
        values["M1_perfect_gain"].append(m1["perfect_gain"])
        single_raw = {stratum: weighted_ratio([row for row in sample_rows if row["stratum"] == stratum], "zero_jitter_correct", multiplicity) for stratum in STRATA}
        single_existing = {stratum: population_ratio([row for row in single_population_rows if row["stratum"] == stratum], "single_slot_correct", multiplicity) for stratum in STRATA}
        if None not in [*single_raw.values(), *single_existing.values()]:
            delta_single = single_raw["common_11"] - single_existing["common_11"]
            single_corrected = {stratum: min(1.0, max(0.0, single_raw[stratum] - delta_single)) for stratum in STRATA}
            baseline_single = sum(sum(multiplicity.get(row["application"], 0) for row in population_rows if row["stratum"] == stratum) / sum(multiplicity.get(row["application"], 0) for row in population_rows) * single_existing[stratum] for stratum in STRATA)
            perfect_single = sum(sum(multiplicity.get(row["application"], 0) for row in population_rows if row["stratum"] == stratum) / sum(multiplicity.get(row["application"], 0) for row in population_rows) * single_corrected[stratum] for stratum in STRATA)
            values["single_perfect_gain"].append(perfect_single - baseline_single)
        for stratum in STRATA:
            values[f"B3_raw_{stratum}"].append(b3["raw"][stratum])
            values[f"B3_corrected_{stratum}"].append(b3["corrected"][stratum])
        common_small = [row for row in sample_rows if row["size_half"] == "common_small"]
        delta_small = weighted_ratio(common_small, "oracle_b3_correct", multiplicity) - population_ratio([row for row in population_rows if row["size_half"] == "common_small"], "existing_b3_correct", multiplicity)
        sensitivity = corrected_values(sample_rows, population_rows, "b3", multiplicity, delta_small)
        values["B3_small_sensitivity_gain"].append(sensitivity["perfect_gain"])
        fractions = {stratum: sum(multiplicity.get(row["application"], 0) for row in population_rows if row["stratum"] == stratum) / sum(multiplicity.get(row["application"], 0) for row in population_rows) for stratum in STRATA}
        for count in range(4, 12):
            q = {stratum: population_ratio([row for row in population_rows if row["stratum"] == stratum], f"tiling_center_{count}", multiplicity) for stratum in STRATA}
            if None not in q.values():
                values[f"G_N_{count}"].append(sum(fractions[stratum] * q[stratum] * (b3["corrected"][stratum] - b3["existing"][stratum]) for stratum in STRATA))
    return {key: {"finite_replicates": len(current), "ci_99": [float(np.quantile(current, 0.005)), float(np.quantile(current, 0.995))]} for key, current in values.items() if len(current) >= 0.99 * resamples}


def main():
    if OUTPUT_PATH.exists() or PRIVATE_ROWS_PATH.exists():
        raise FileExistsError("OWIN Arm A output exists")
    for stratum in STRATA:
        status = json.loads(SHARD_STATUS_PATHS[stratum].read_text())
        expected = {"common_11": 2400, "partial_1_10": 1800, "uncovered_0": 1800}[stratum]
        if status["failures"] != 0 or status["calls"] != expected or sha256_file(SHARD_PATHS[stratum]) != status["trace_sha256"]:
            raise PermissionError(f"OWIN shard integrity mismatch: {stratum}")
    samples = {row["row_id"]: row for row in read_jsonl(SAMPLE_PATH)}
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    small_ids = set(preflight["common_area_split"]["small_ids"])
    traces = defaultdict(dict)
    for stratum in STRATA:
        for trace in read_jsonl(SHARD_PATHS[stratum]):
            traces[trace["row_id"]][trace["slot"]] = trace
    if set(traces) != set(samples) or any(set(value) != set(range(12)) for value in traces.values()):
        raise ValueError("OWIN complete trace/sample mismatch")
    gta1 = {}
    for path in sorted(GTA1_ROOT.glob("shard-*.jsonl")):
        for row in read_jsonl(path):
            gta1[row["id"]] = row
    cover = {row["row_id"]: row for row in read_jsonl(COVER_ROWS_PATH)}
    cwin = {row["row_id"]: row for row in read_jsonl(CWIN_ROWS_PATH)}
    oracle_pool = []
    private_rows = []
    for row_id in sorted(samples):
        sample = samples[row_id]
        target_bbox = gta1[row_id]["target_bbox"]
        candidates = []
        slot_correct = []
        for slot in range(12):
            trace = traces[row_id][slot]
            point = trace["parsed"]["full_image_point"] if trace["parsed"]["parse_status"] == "parsed" else [0, 0]
            correct = bool(h3.point_in_bbox(point, target_bbox)) if trace["parsed"]["parse_status"] == "parsed" else False
            candidates.append({"model": "GTA1-7B", "view_index": slot, "point": point, "coverage": 0.0})
            slot_correct.append(correct)
        selected, group = sourcebias.b3_select_index(candidates)
        oracle_pool.append({"id": row_id, "application": sample["application"], "fold": cover[row_id]["fold"], "target_bbox": target_bbox, "candidates": candidates})
        existing_full_bbox_count = sum(candidate["region"][0] <= target_bbox[0] and candidate["region"][1] <= target_bbox[1] and target_bbox[2] <= candidate["region"][2] and target_bbox[3] <= candidate["region"][3] for candidate in gta1[row_id]["candidates"][1:12])
        private_rows.append({"row_id": row_id, "application": sample["application"], "fold": cover[row_id]["fold"], "stratum": sample["stratum"], "existing_full_bbox_stratum": coverage_stratum(existing_full_bbox_count), "size_half": "common_small" if row_id in small_ids else ("common_large" if sample["stratum"] == "common_11" else "NA"), "inverse_probability_weight": sample["inverse_probability_weight"], "oracle_b3_correct": slot_correct[selected], "oracle_b3_selected_slot": selected, "oracle_b3_group": list(group), "zero_jitter_correct": slot_correct[1], "oracle_crop_errors": [not value for value in slot_correct[1:]], "existing_crop_errors": [not bool(h3.point_in_bbox(candidate["point"], target_bbox)) for candidate in gta1[row_id]["candidates"][1:12]], "existing_b3_correct": cover[row_id]["b3_correct"], "existing_m1_correct": cwin[row_id]["original_m1_correct"]})
    m1 = fit_m1(oracle_pool)
    for row in private_rows:
        row["oracle_m1_correct"] = m1[row["row_id"]]
    write_jsonl_fsynced(PRIVATE_ROWS_PATH, private_rows)
    arm_b = json.loads(ARM_B_PATH.read_text())
    arm_b_raw = read_jsonl(Path(arm_b["raw"]["path"]))
    tiling = {(row["row_id"], row["N"]): int(row["tiling"]["center_count"] > 0) for row in arm_b_raw}
    population_rows = []
    for row_id in sorted(cover):
        row = {"row_id": row_id, "application": cover[row_id]["application"], "stratum": cover[row_id]["target_stratum"], "size_half": "common_small" if row_id in small_ids else ("common_large" if cover[row_id]["target_stratum"] == "common_11" else "NA"), "existing_b3_correct": cover[row_id]["b3_correct"], "existing_m1_correct": cwin[row_id]["original_m1_correct"]}
        row.update({f"tiling_center_{count}": tiling[(row_id, count)] for count in range(4, 12)})
        population_rows.append(row)
    single_population_rows = []
    for row_id in sorted(cover):
        for candidate in gta1[row_id]["candidates"][1:12]:
            single_population_rows.append({"row_id": row_id, "application": cover[row_id]["application"], "stratum": cover[row_id]["target_stratum"], "single_slot_correct": bool(h3.point_in_bbox(candidate["point"], gta1[row_id]["target_bbox"]))})
    b3 = corrected_values(private_rows, population_rows, "b3")
    m1_values = corrected_values(private_rows, population_rows, "m1")
    common_small = [row for row in private_rows if row["size_half"] == "common_small"]
    delta_small = weighted_ratio(common_small, "oracle_b3_correct") - population_ratio([row for row in population_rows if row["size_half"] == "common_small"], "existing_b3_correct")
    sensitivity = corrected_values(private_rows, population_rows, "b3", delta_override=delta_small)
    single_raw = {stratum: weighted_ratio([row for row in private_rows if row["stratum"] == stratum], "zero_jitter_correct") for stratum in STRATA}
    single_existing = {stratum: population_ratio([row for row in single_population_rows if row["stratum"] == stratum], "single_slot_correct") for stratum in STRATA}
    delta_single = single_raw["common_11"] - single_existing["common_11"]
    single_corrected = {stratum: min(1.0, max(0.0, single_raw[stratum] - delta_single)) for stratum in STRATA}
    fractions = {stratum: sum(row["stratum"] == stratum for row in population_rows) / len(population_rows) for stratum in STRATA}
    single_values = {"raw": single_raw, "existing": single_existing, "delta_common": delta_single, "corrected": single_corrected, "perfect_accuracy": sum(fractions[stratum] * single_corrected[stratum] for stratum in STRATA), "perfect_gain": sum(fractions[stratum] * (single_corrected[stratum] - single_existing[stratum]) for stratum in STRATA)}
    factorized = {}
    for count in range(4, 12):
        q = {stratum: arm_b["summaries"][str(count)]["tiling"]["center_coverage_by_existing_stratum"][stratum]["coverage_fraction"] for stratum in STRATA}
        factorized[str(count)] = {"q": q, "G_N": sum(fractions[stratum] * q[stratum] * (b3["corrected"][stratum] - b3["existing"][stratum]) for stratum in STRATA)}
    dependence = {}
    for stratum in STRATA:
        rows = [row for row in private_rows if row["stratum"] == stratum]
        dependence[stratum] = {"folds": []}
        for outer_fold in range(5):
            current = [row for row in rows if row["fold"] != outer_fold]
            weights = np.asarray([row["inverse_probability_weight"] for row in current], dtype=float)
            oracle_errors = np.asarray([row["oracle_crop_errors"] for row in current], dtype=float)
            existing_errors = np.asarray([row["existing_crop_errors"] for row in current], dtype=float)
            oracle_endpoint = dependence_endpoint(oracle_errors, weights)
            existing_endpoint = dependence_endpoint(existing_errors, weights)
            if existing_endpoint["constant_slot_count"] == 11:
                comparator_status = "UNDEFINED_DEGENERATE_COMPARATOR"
            elif existing_endpoint["valid_pair_count"] == 0:
                comparator_status = "NO_VALID_PAIR_COMPARATOR"
            else:
                comparator_status = "AVAILABLE"
            interval_labels = dependence_bootstrap(current, RESAMPLES, 20261000 + STRATA.index(stratum) * 5 + outer_fold) if oracle_endpoint["reliability"] == "RELIABLE" and existing_endpoint["reliability"] == "RELIABLE" and comparator_status == "AVAILABLE" else {"zero": {"ci_99": None, "label": comparator_status if comparator_status != "AVAILABLE" else "DEPENDENCE_DIAGNOSTIC_UNRELIABLE"}, "mean": {"ci_99": None, "label": comparator_status if comparator_status != "AVAILABLE" else "DEPENDENCE_DIAGNOSTIC_UNRELIABLE"}, "combined_label": comparator_status if comparator_status != "AVAILABLE" else "DEPENDENCE_DIAGNOSTIC_UNRELIABLE"}
            dependence[stratum]["folds"].append({"outer_fold": outer_fold, "rows": len(current), "oracle": oracle_endpoint, "existing": existing_endpoint, "comparator_status": comparator_status, "match_intervals": interval_labels})
            reliable_folds = [fold for fold in dependence[stratum]["folds"] if fold["oracle"]["reliability"] == "RELIABLE"]
            dependence[stratum]["oracle_fold_summary"] = {fill: {"mean": float(np.mean([fold["oracle"][f"neff_{fill}"] for fold in reliable_folds if fold["oracle"][f"neff_{fill}"] is not None])) if any(fold["oracle"][f"neff_{fill}"] is not None for fold in reliable_folds) else None, "range": [float(min(fold["oracle"][f"neff_{fill}"] for fold in reliable_folds if fold["oracle"][f"neff_{fill}"] is not None)), float(max(fold["oracle"][f"neff_{fill}"] for fold in reliable_folds if fold["oracle"][f"neff_{fill}"] is not None))] if any(fold["oracle"][f"neff_{fill}"] is not None for fold in reliable_folds) else None} for fill in ("zero", "mean")}
            intervals = bootstrap_endpoints(private_rows, population_rows, single_population_rows, arm_b)
    gain = b3["perfect_gain"]
    interpretation = "O_I1" if gain < 0.05 else ("O_I2" if gain <= 0.10 else "O_I3")
    for count in range(4, 12):
        factorized[str(count)]["ci_99"] = intervals.get(f"G_N_{count}", {}).get("ci_99")
    full_bbox_domains = {}
    for stratum in STRATA:
        selected = [row for row in private_rows if row["existing_full_bbox_stratum"] == stratum]
        full_bbox_domains[stratum] = {"rows": len(selected), "oracle_B3": weighted_ratio(selected, "oracle_b3_correct")}
    unreliable = [f"{stratum}/fold{fold['outer_fold']}:{fold['match_intervals']['combined_label']}" for stratum, value in dependence.items() for fold in value["folds"] if fold["match_intervals"]["combined_label"] in {"DEPENDENCE_DIAGNOSTIC_UNRELIABLE", "UNDEFINED_DEGENERATE_COMPARATOR", "NO_VALID_PAIR_COMPARATOR", "DEPENDENCE_MATCH_INDETERMINATE"}]
    common_frozen = json.loads(COMMON_PATH.read_text())
    common_reproduction = {"B3_raw_absolute_difference": abs(b3["raw"]["common_11"] - common_frozen["raw"]["oracle_B3"]), "B3_delta_absolute_difference": abs(b3["delta_common"] - common_frozen["calibration"]["delta_pool_B3"]), "single_raw_absolute_difference": abs(single_values["raw"]["common_11"] - common_frozen["raw"]["zero_jitter_single"]), "single_delta_absolute_difference": abs(single_values["delta_common"] - common_frozen["calibration"]["delta_single"]), "tolerance": 1e-12}
    common_reproduction["pass"] = all(value <= common_reproduction["tolerance"] for key, value in common_reproduction.items() if key.endswith("difference"))
    if not common_reproduction["pass"]:
        raise ValueError("OWIN final evaluator does not reproduce frozen common calibration")
    output = {"schema_version": 1, "status": "PASS_OWIN_ARM_A_COMPLETE", "evidence_status": "GT_ORACLE_NON_DEPLOYABLE_POST_SELECTION_SINGLE_BENCHMARK", "gpu": {"formal_calls": 6000, "failures": 0, "smoke_success_calls": 36, "failed_smoke_attempts_retained": 36}, "B3": b3, "M1_ccm": m1_values, "single_forward": single_values, "pool_minus_single": {"B3_raw_by_stratum": {stratum: b3["raw"][stratum] - single_values["raw"][stratum] for stratum in STRATA}, "M1_minus_single_raw_by_stratum": {stratum: m1_values["raw"][stratum] - single_values["raw"][stratum] for stratum in STRATA}}, "common_frozen_reproduction": common_reproduction, "small_target_sensitivity_B3": sensitivity, "bootstrap": intervals, "O_I": {"point_gain": gain, "classification": interpretation, "thresholds": {"O_I1": "<0.05", "O_I2": "[0.05,0.10]", "O_I3": ">0.10"}, "uses": "corrected_B3_pool_level_point_using_full_common_delta"}, "factorized_G_N": factorized, "secondary_full_bbox_domains": full_bbox_domains, "dependence": dependence, "dependence_unavailable_units": unreliable, "dependence_limitation_required": bool(unreliable), "constant_shift_limitation_required": True, "arm_b_N_star": arm_b["N_star"], "private_rows": {"path": str(PRIVATE_ROWS_PATH.relative_to(ROOT)), "rows": len(private_rows), "bytes": PRIVATE_ROWS_PATH.stat().st_size, "sha256": sha256_file(PRIVATE_ROWS_PATH), "write_flush_fsync_per_row": True}, "trace_shards": {stratum: {"path": str(SHARD_PATHS[stratum].relative_to(ROOT)), "bytes": SHARD_PATHS[stratum].stat().st_size, "sha256": sha256_file(SHARD_PATHS[stratum])} for stratum in STRATA}}
    atomic_json(OUTPUT_PATH, output)
    print(json.dumps({"status": output["status"], "B3": b3, "M1_ccm": m1_values, "sensitivity": sensitivity, "O_I": output["O_I"], "G_N": {key: value["G_N"] for key, value in factorized.items()}, "dependence_unavailable_units": unreliable}, indent=2))


if __name__ == "__main__":
    main()