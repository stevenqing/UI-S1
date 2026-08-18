import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
TRACE_PATH = RUN_DIR / "raw/arm_a_common_11.jsonl"
TRACE_STATUS_PATH = RUN_DIR / "raw/arm_a_common_11_status.json"
SAMPLE_PATH = RUN_DIR / "SAMPLE_WINDOW_MANIFEST.jsonl"
PREFLIGHT_PATH = RUN_DIR / "PREFLIGHT.json"
COVER_ROWS_PATH = ROOT / "runs/cover/2026-08-16/raw/arm_a_rows.jsonl"
GTA1_ROOT = ROOT / "runs/ccm-h2h/2026-07-31/h1/shards/top18"
SOURCEBIAS_PATH = ROOT / "runs/sourcebias/2026-08-03/sourcebias_common.py"
H3_PATH = ROOT / "runs/ccm-h2h/2026-07-31/h3/h3_eval.py"
OUTPUT_PATH = RUN_DIR / "COMMON_CALIBRATION.json"
PRIVATE_ROWS_PATH = RUN_DIR / "raw/private_common_evaluation.jsonl"
RESAMPLES = 10000


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sourcebias = load_module(SOURCEBIAS_PATH, "owin_common_sourcebias")
h3 = load_module(H3_PATH, "owin_common_h3")


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
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def weighted_ratio(rows, field, multiplicity=None):
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        multiple = 1 if multiplicity is None else multiplicity.get(row["application"], 0)
        weight = row["inverse_probability_weight"] * multiple
        numerator += weight * float(row[field])
        denominator += weight
    if denominator <= 0:
        return None
    return numerator / denominator


def unweighted_population_ratio(rows, field, multiplicity=None):
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        multiple = 1 if multiplicity is None else multiplicity.get(row["application"], 0)
        numerator += multiple * float(row[field])
        denominator += multiple
    if denominator <= 0:
        return None
    return numerator / denominator


def identity_delta(rows, multiplicity=None):
    small = [row for row in rows if row["size_half"] == "common_small"]
    large = [row for row in rows if row["size_half"] == "common_large"]
    all_weight = sum(row["inverse_probability_weight"] * (1 if multiplicity is None else multiplicity.get(row["application"], 0)) for row in rows)
    small_weight = sum(row["inverse_probability_weight"] * (1 if multiplicity is None else multiplicity.get(row["application"], 0)) for row in small)
    if all_weight <= 0 or small_weight <= 0 or small_weight >= all_weight:
        return None
    achieved = small_weight / all_weight
    full_delta = weighted_ratio(rows, "oracle_b3_correct", multiplicity) - weighted_ratio(rows, "existing_b3_correct", multiplicity)
    small_delta = weighted_ratio(small, "oracle_b3_correct", multiplicity) - weighted_ratio(small, "existing_b3_correct", multiplicity)
    large_delta = weighted_ratio(large, "oracle_b3_correct", multiplicity) - weighted_ratio(large, "existing_b3_correct", multiplicity)
    return full_delta - (achieved * small_delta + (1 - achieved) * large_delta)


def bootstrap(sample_rows, population_rows, single_population_rows, population_anchors, resamples=RESAMPLES, seed=20260817):
    applications = sorted({row["application"] for row in population_rows})
    rng = np.random.default_rng(seed)
    values = defaultdict(list)
    nonfinite_identity = 0
    for _ in range(resamples):
        selected = rng.choice(applications, size=len(applications), replace=True)
        multiplicity = Counter(selected)
        sample_small = [row for row in sample_rows if row["size_half"] == "common_small"]
        sample_large = [row for row in sample_rows if row["size_half"] == "common_large"]
        population_small = [row for row in population_rows if row["size_half"] == "common_small"]
        population_large = [row for row in population_rows if row["size_half"] == "common_large"]
        oracle = weighted_ratio(sample_rows, "oracle_b3_correct", multiplicity)
        oracle_small = weighted_ratio(sample_small, "oracle_b3_correct", multiplicity)
        oracle_large = weighted_ratio(sample_large, "oracle_b3_correct", multiplicity)
        existing = unweighted_population_ratio(population_rows, "existing_b3_correct", multiplicity)
        existing_small = unweighted_population_ratio(population_small, "existing_b3_correct", multiplicity)
        existing_large = unweighted_population_ratio(population_large, "existing_b3_correct", multiplicity)
        zero = weighted_ratio(sample_rows, "zero_jitter_correct", multiplicity)
        single = unweighted_population_ratio(single_population_rows, "single_slot_correct", multiplicity)
        if None not in (oracle, oracle_small, oracle_large, existing, existing_small, existing_large, zero, single):
            delta = oracle - existing
            delta_small = oracle_small - existing_small
            delta_large = oracle_large - existing_large
            values["oracle_b3"].append(oracle)
            values["delta_pool_b3"].append(delta)
            values["delta_small_b3"].append(delta_small)
            values["delta_large_b3"].append(delta_large)
            values["heterogeneity"].append(delta_small - delta_large)
            values["zero_jitter"].append(zero)
            values["delta_single"].append(zero - single)
        identity = identity_delta(sample_rows, multiplicity)
        if identity is None:
            nonfinite_identity += 1
        else:
            values["identity"].append(identity)
    intervals = {key: [float(np.quantile(current, 0.005)), float(np.quantile(current, 0.995))] for key, current in values.items() if key != "identity" and len(current) >= 0.99 * resamples}
    return intervals, {"finite_replicates": len(values["identity"]), "nonfinite_replicates": nonfinite_identity, "maximum_absolute_finite_residual": max(map(abs, values["identity"]), default=None)}


def main():
    if OUTPUT_PATH.exists() or PRIVATE_ROWS_PATH.exists():
        raise FileExistsError("OWIN common evaluation output exists")
    trace_status = json.loads(TRACE_STATUS_PATH.read_text())
    if trace_status["failures"] != 0 or trace_status["calls"] != 2400 or sha256_file(TRACE_PATH) != trace_status["trace_sha256"]:
        raise PermissionError("OWIN common trace integrity mismatch")
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    small_ids = set(preflight["common_area_split"]["small_ids"])
    large_ids = set(preflight["common_area_split"]["large_ids"])
    samples = {row["row_id"]: row for row in read_jsonl(SAMPLE_PATH) if row["stratum"] == "common_11"}
    traces = read_jsonl(TRACE_PATH)
    by_row = defaultdict(dict)
    for trace in traces:
        by_row[trace["row_id"]][trace["slot"]] = trace
    if set(by_row) != set(samples) or any(set(slots) != set(range(12)) for slots in by_row.values()):
        raise ValueError("OWIN common trace/sample mismatch")
    gta1 = {}
    for path in sorted(GTA1_ROOT.glob("shard-*.jsonl")):
        for row in read_jsonl(path):
            gta1[row["id"]] = row
    cover = {row["row_id"]: row for row in read_jsonl(COVER_ROWS_PATH)}

    private_rows = []
    for row_id in sorted(samples):
        target_bbox = gta1[row_id]["target_bbox"]
        candidates = []
        slot_correct = []
        for slot in range(12):
            trace = by_row[row_id][slot]
            point = trace["parsed"]["full_image_point"] if trace["parsed"]["parse_status"] == "parsed" else [0, 0]
            correct = bool(h3.point_in_bbox(point, target_bbox)) if trace["parsed"]["parse_status"] == "parsed" else False
            candidates.append({"model": "GTA1-7B", "view_index": slot, "point": point, "coverage": 0.0})
            slot_correct.append(correct)
        selected, group = sourcebias.b3_select_index(candidates)
        sample = samples[row_id]
        private_rows.append({"row_id": row_id, "application": sample["application"], "fold": cover[row_id]["fold"], "size_half": "common_small" if row_id in small_ids else "common_large", "inverse_probability_weight": sample["inverse_probability_weight"], "oracle_b3_correct": slot_correct[selected], "oracle_b3_selected_slot": selected, "oracle_b3_group": list(group), "zero_jitter_correct": slot_correct[1], "existing_b3_correct": cover[row_id]["b3_correct"], "slot_correct": slot_correct})
    if sum(row["size_half"] == "common_small" for row in private_rows) == 0 or sum(row["size_half"] == "common_large" for row in private_rows) == 0:
        raise ValueError("OWIN common size half empty")

    population_rows = []
    single_population_rows = []
    for row_id in sorted(small_ids | large_ids):
        size_half = "common_small" if row_id in small_ids else "common_large"
        population_rows.append({"row_id": row_id, "application": cover[row_id]["application"], "size_half": size_half, "existing_b3_correct": cover[row_id]["b3_correct"]})
        for candidate in gta1[row_id]["candidates"][1:12]:
            single_population_rows.append({"row_id": row_id, "application": cover[row_id]["application"], "single_slot_correct": bool(h3.point_in_bbox(candidate["point"], gta1[row_id]["target_bbox"]))})
    write_jsonl_fsynced(PRIVATE_ROWS_PATH, private_rows)

    sample_small = [row for row in private_rows if row["size_half"] == "common_small"]
    sample_large = [row for row in private_rows if row["size_half"] == "common_large"]
    population_small = [row for row in population_rows if row["size_half"] == "common_small"]
    population_large = [row for row in population_rows if row["size_half"] == "common_large"]
    oracle = weighted_ratio(private_rows, "oracle_b3_correct")
    oracle_small = weighted_ratio(sample_small, "oracle_b3_correct")
    oracle_large = weighted_ratio(sample_large, "oracle_b3_correct")
    existing = unweighted_population_ratio(population_rows, "existing_b3_correct")
    existing_small = unweighted_population_ratio(population_small, "existing_b3_correct")
    existing_large = unweighted_population_ratio(population_large, "existing_b3_correct")
    zero = weighted_ratio(private_rows, "zero_jitter_correct")
    single = unweighted_population_ratio(single_population_rows, "single_slot_correct")
    delta = oracle - existing
    delta_small = oracle_small - existing_small
    delta_large = oracle_large - existing_large
    identity = identity_delta(private_rows)
    all_weight = sum(row["inverse_probability_weight"] for row in private_rows)
    small_weight = sum(row["inverse_probability_weight"] for row in sample_small)
    achieved = small_weight / all_weight
    population_w = 465 / 931
    delta_decomp = delta - (population_w * delta_small + (1 - population_w) * delta_large)
    intervals, identity_bootstrap = bootstrap(private_rows, population_rows, single_population_rows, preflight["anchors"])
    identity_pass = identity is not None and abs(identity) <= 1e-9
    output = {"schema_version": 1, "status": "PASS_OWIN_COMMON_CALIBRATION_FROZEN" if identity_pass else "CALIBRATION_IDENTITY_CHECK_FAILED", "evidence_status": "GT_ORACLE_NON_DEPLOYABLE_POST_SELECTION", "M1_status": "DEFERRED_BY_AMENDMENT_007_UNTIL_ALL_500_ROWS", "sample_rows": len(private_rows), "raw": {"oracle_B3": oracle, "oracle_B3_ci_99": intervals.get("oracle_b3"), "zero_jitter_single": zero, "zero_jitter_single_ci_99": intervals.get("zero_jitter")}, "anchors": {"existing_common_B3": existing, "existing_single_GTA1_views1_11": single}, "calibration": {"delta_pool_B3": delta, "delta_pool_B3_ci_99": intervals.get("delta_pool_b3"), "delta_single": zero - single, "delta_single_ci_99": intervals.get("delta_single")}, "size_heterogeneity": {"oracle_small": oracle_small, "oracle_large": oracle_large, "existing_small": existing_small, "existing_large": existing_large, "delta_small_B3": delta_small, "delta_large_B3": delta_large, "contrast_small_minus_large": delta_small - delta_large, "contrast_ci_99": intervals.get("heterogeneity"), "label": "CONSTANT_SHIFT_SIZE_HETEROGENEITY_DETECTED" if intervals.get("heterogeneity") and (intervals["heterogeneity"][0] > 0 or intervals["heterogeneity"][1] < 0) else "NO_DETECTED_SIZE_HETEROGENEITY_AT_99_PERCENT"}, "identity": {"Delta_ident": identity, "tolerance": 1e-9, "pass": identity_pass, "bootstrap": identity_bootstrap}, "representativeness": {"Delta_decomp": delta_decomp, "population_w": population_w, "achieved_IPW_w": achieved, "raw_sample_small_share": len(sample_small) / len(private_rows), "reference_scale": 0.005, "exceeds_reference_scale": abs(delta_decomp) > 0.005}, "gpu_generation_complete": {"common": True, "partial": False, "uncovered": False}, "O_I_status": "NOT_COMPUTABLE_BEFORE_LOW_STRATA", "private_rows": {"path": str(PRIVATE_ROWS_PATH.relative_to(ROOT)), "rows": len(private_rows), "bytes": PRIVATE_ROWS_PATH.stat().st_size, "sha256": sha256_file(PRIVATE_ROWS_PATH), "write_flush_fsync_per_row": True}, "trace_sha256": trace_status["trace_sha256"]}
    atomic_json(OUTPUT_PATH, output)
    print(json.dumps({"status": output["status"], "raw": output["raw"], "calibration": output["calibration"], "heterogeneity": output["size_heterogeneity"], "identity": output["identity"], "representativeness": output["representativeness"], "M1": output["M1_status"], "O_I": output["O_I_status"]}, indent=2))
    if not identity_pass:
        raise RuntimeError("OWIN calibration identity failed")


if __name__ == "__main__":
    main()