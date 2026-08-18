import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from look_common import ROOT, RUN_DIR, atomic_json, nearest_choice, read_jsonl, sha256_file, write_jsonl_fsynced


TRACE_PATH = RUN_DIR / "raw/formal_traces.jsonl"
TRACE_STATUS_PATH = RUN_DIR / "raw/formal_status.json"
PRIVATE_PATH = RUN_DIR / "raw/private_preflight_rows.jsonl"
SAMPLE_PATH = RUN_DIR / "SAMPLE_WINDOW_MANIFEST.jsonl"
SUMMARY_PATH = RUN_DIR / "SAMPLE_SUMMARY.json"
OUTPUT_PATH = RUN_DIR / "LOOK_RESULTS.json"
PRIVATE_EVAL_PATH = RUN_DIR / "raw/private_evaluation_rows.jsonl"
RESAMPLES = 10000


def weighted_mean(values, weights):
    denominator = sum(weights)
    return sum(value * weight for value, weight in zip(values, weights)) / denominator if denominator > 0 else None


def weighted_auc(records, multiplicity=None):
    labels = []
    scores = []
    weights = []
    for record in records:
        multiple = 1 if multiplicity is None else multiplicity.get(record["application"], 0)
        weight = record["inverse_probability_weight"] * multiple
        if weight <= 0:
            continue
        labels.append(int(record["label"]))
        scores.append(float(record["score"]))
        weights.append(weight)
    if not labels or len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores, sample_weight=weights))


def row_weighted(rows, field, multiplicity=None):
    values = []
    weights = []
    for row in rows:
        multiple = 1 if multiplicity is None else multiplicity.get(row["application"], 0)
        if multiple:
            values.append(float(row[field]))
            weights.append(row["inverse_probability_weight"] * multiple)
    return weighted_mean(values, weights)


def bootstrap(rows, main_records, null_records, resamples=RESAMPLES, seed=20260818):
    applications = sorted({row["application"] for row in rows})
    recoverable = [row for row in rows if row["stratum"] == "recoverable"]
    main_recoverable = [record for record in main_records if record["stratum"] == "recoverable"]
    null_recoverable = [record for record in null_records if record["stratum"] == "recoverable"]
    rng = np.random.default_rng(seed)
    values = defaultdict(list)
    for _ in range(resamples):
        multiplicity = Counter(rng.choice(applications, size=len(applications), replace=True))
        main_auc = weighted_auc(main_recoverable, multiplicity)
        null_auc = weighted_auc(null_recoverable, multiplicity)
        if main_auc is not None:
            values["L_P1_AUROC"].append(main_auc)
        if main_auc is not None and null_auc is not None:
            values["L_P4_main_minus_null"].append(main_auc - null_auc)
        main_correct = row_weighted(recoverable, "main_selected_correct", multiplicity)
        m1_correct = row_weighted(recoverable, "M1_correct", multiplicity)
        if main_correct is not None and m1_correct is not None:
            values["L_P2_main_minus_M1"].append(main_correct - m1_correct)
    output = {}
    for key, current in values.items():
        if len(current) >= 0.99 * resamples:
            output[key] = {"finite_replicates": len(current), "ci_99": [float(np.quantile(current, 0.005)), float(np.quantile(current, 0.995))]}
        else:
            output[key] = {"finite_replicates": len(current), "ci_99": None}
    return output


def main():
    if OUTPUT_PATH.exists() or PRIVATE_EVAL_PATH.exists():
        raise FileExistsError("LOOK evaluation output exists")
    status = json.loads(TRACE_STATUS_PATH.read_text())
    if status["calls"] != 1290 or status["failures"] != 0 or sha256_file(TRACE_PATH) != status["trace_sha256"]:
        raise PermissionError("LOOK trace integrity mismatch")
    sample = {row["row_id"]: row for row in read_jsonl(SAMPLE_PATH)}
    private = {row["row_id"]: row for row in read_jsonl(PRIVATE_PATH)}
    summary = json.loads(SUMMARY_PATH.read_text())
    traces = defaultdict(dict)
    for trace in read_jsonl(TRACE_PATH):
        traces[trace["row_id"]][trace["window"]["kind"]] = trace
    if set(traces) != set(sample) or any(set(value) != {"main", "sensitivity", "null"} for value in traces.values()):
        raise ValueError("LOOK trace/sample mismatch")
    rows = []
    main_records = []
    null_records = []
    sensitivity_records = []
    boundaries = summary["separation_quartile_boundaries"]
    for row_id in sorted(sample):
        sampled = sample[row_id]
        source = private[row_id]
        modes = source["modes"]
        with Image.open(ROOT / sampled["image"]["path"]) as image:
            image_width, image_height = image.size
        diagonal = math.hypot(image_width, image_height)
        mapped = {}
        for kind in ("main", "sensitivity", "null"):
            trace = traces[row_id][kind]
            mapped[kind] = trace["parsed"]["full_image_point"] if trace["parsed"]["parse_status"] == "parsed" else None
        mode_centroids = [mode["centroid"] for mode in modes]
        main_choice = nearest_choice(mapped["main"], mode_centroids[:2])
        sensitivity_choice = nearest_choice(mapped["sensitivity"], mode_centroids[:3])
        null_centroids = [mode_centroids[0], source["null_window"]["random_point"]]
        null_choice = nearest_choice(mapped["null"], null_centroids)
        b3_mode = next((index for index, mode in enumerate(modes) if source["b3_selected_index"] in mode["members"]), None)
        separation_bin = int(np.searchsorted(boundaries, sampled["normalized_M1_M2_separation"], side="right"))
        record = {"row_id": row_id, "application": sampled["application"], "fold": sampled["fold"], "stratum": sampled["stratum"], "inverse_probability_weight": sampled["inverse_probability_weight"], "separation": sampled["normalized_M1_M2_separation"], "separation_bin": separation_bin, "main_area_fraction": sampled["windows"][0]["area_fraction"], "sensitivity_area_fraction": sampled["windows"][1]["area_fraction"], "null_area_ratio": source["null_window"]["area_ratio"], "null_attempt": source["null_window"]["attempt"], "M1_correct": bool(modes[0]["correct"]), "main_choice": main_choice, "main_unmappable": main_choice is None, "main_selected_correct": bool(modes[main_choice]["correct"]) if main_choice is not None else False, "sensitivity_choice": sensitivity_choice, "sensitivity_unmappable": sensitivity_choice is None, "sensitivity_selected_correct": bool(modes[sensitivity_choice]["correct"]) if sensitivity_choice is not None else False, "null_choice": null_choice, "null_unmappable": null_choice is None, "b3_mode": b3_mode, "overturn": main_choice is not None and b3_mode is not None and main_choice != b3_mode, "harmful_overturn": main_choice is not None and b3_mode is not None and main_choice != b3_mode and not bool(modes[main_choice]["correct"])}
        rows.append(record)
        if mapped["main"] is not None:
            for index in range(2):
                score = -math.dist(mapped["main"], mode_centroids[index]) / diagonal
                main_records.append({"row_id": row_id, "application": sampled["application"], "stratum": sampled["stratum"], "inverse_probability_weight": sampled["inverse_probability_weight"], "candidate": f"M{index+1}", "score": score, "label": bool(modes[index]["correct"])})
        if mapped["null"] is not None:
            for index in range(2):
                score = -math.dist(mapped["null"], null_centroids[index]) / diagonal
                label = bool(modes[0]["correct"]) if index == 0 else bool(point_inside(source["null_window"]["random_point"], source["target_bbox"]))
                null_records.append({"row_id": row_id, "application": sampled["application"], "stratum": sampled["stratum"], "inverse_probability_weight": sampled["inverse_probability_weight"], "candidate": "M1" if index == 0 else "random", "score": score, "label": label})
        if mapped["sensitivity"] is not None:
            for index in range(3):
                score = -math.dist(mapped["sensitivity"], mode_centroids[index]) / diagonal
                sensitivity_records.append({"row_id": row_id, "application": sampled["application"], "stratum": sampled["stratum"], "inverse_probability_weight": sampled["inverse_probability_weight"], "candidate": f"M{index+1}", "score": score, "label": bool(modes[index]["correct"])})
    write_jsonl_fsynced(PRIVATE_EVAL_PATH, rows)
    recoverable = [row for row in rows if row["stratum"] == "recoverable"]
    pool_correct = [row for row in rows if row["stratum"] == "pool_correct"]
    main_recoverable = [record for record in main_records if record["stratum"] == "recoverable"]
    null_recoverable = [record for record in null_records if record["stratum"] == "recoverable"]
    L_P1 = weighted_auc(main_recoverable)
    null_auc = weighted_auc(null_recoverable)
    L_P2 = row_weighted(recoverable, "main_selected_correct") - row_weighted(recoverable, "M1_correct")
    L_P4 = L_P1 - null_auc if L_P1 is not None and null_auc is not None else None
    intervals = bootstrap(rows, main_records, null_records)
    L_P3 = {"overturn_rate": row_weighted(pool_correct, "overturn"), "harmful_overturn_rate": row_weighted(pool_correct, "harmful_overturn"), "unmappable_rate": row_weighted(pool_correct, "main_unmappable")}
    separation = {}
    for bin_index in range(4):
        selected_rows = [row for row in recoverable if row["separation_bin"] == bin_index]
        selected_ids = {row["row_id"] for row in selected_rows}
        selected_records = [record for record in main_recoverable if record["row_id"] in selected_ids]
        separation[str(bin_index)] = {"rows": len(selected_rows), "AUROC": weighted_auc(selected_records), "main_minus_M1": row_weighted(selected_rows, "main_selected_correct") - row_weighted(selected_rows, "M1_correct") if selected_rows else None, "positive_records": sum(record["label"] for record in selected_records), "negative_records": sum(not record["label"] for record in selected_records)}
    geometry = {"main_area_gt_0_8_fraction": row_weighted(rows, "main_area_fraction_gt_0_8") if False else weighted_mean([row["main_area_fraction"] > 0.8 for row in rows], [row["inverse_probability_weight"] for row in rows]), "main_area_fraction": summary_values([row["main_area_fraction"] for row in rows]), "sensitivity_area_fraction": summary_values([row["sensitivity_area_fraction"] for row in rows]), "null_area_ratio": summary_values([row["null_area_ratio"] for row in rows]), "null_attempt": summary_values([row["null_attempt"] for row in rows])}
    unmappable = {stratum: row_weighted([row for row in rows if row["stratum"] == stratum], "main_unmappable") for stratum in ("recoverable", "pool_correct")}
    L_K1 = intervals.get("L_P4_main_minus_null", {}).get("ci_99") is None or intervals["L_P4_main_minus_null"]["ci_99"][0] <= 0
    L_K2 = geometry["main_area_gt_0_8_fraction"] > 0.5
    L_K3 = len(recoverable) < 150 or len(pool_correct) < 150
    L_K4 = any(value > 0.10 for value in unmappable.values())
    L_K5 = status["failure_rate"] > 0.01
    if L_K3 or L_P1 is None or intervals.get("L_P1_AUROC", {}).get("ci_99") is None:
        decision = "L_K3_OBSERVATIONAL_NO_DIRECTIONAL_ADJUDICATION" if L_K3 else "L_P1_UNDEFINED_NO_DIRECTIONAL_ADJUDICATION"
    else:
        bounds = intervals["L_P1_AUROC"]["ci_99"]
        decision = "L_D1" if bounds[1] < 0.60 else ("L_D2" if bounds[0] > 0.65 else "L_D3")
    output = {"schema_version": 1, "status": "PASS_LOOK_DIAGNOSTIC_COMPLETE", "evidence_status": "POST_SELECTION_SINGLE_BENCHMARK_DIAGNOSTIC_NOT_METHOD", "sample": {"recoverable": len(recoverable), "pool_correct": len(pool_correct), "formal_calls": 1290}, "L_P1": {"main_AUROC": L_P1, "ci_99": intervals.get("L_P1_AUROC", {}).get("ci_99"), "finite_replicates": intervals.get("L_P1_AUROC", {}).get("finite_replicates"), "positive_records": sum(record["label"] for record in main_recoverable), "negative_records": sum(not record["label"] for record in main_recoverable)}, "L_P2": {"M1_correct": row_weighted(recoverable, "M1_correct"), "main_correct": row_weighted(recoverable, "main_selected_correct"), "difference": L_P2, "ci_99": intervals.get("L_P2_main_minus_M1", {}).get("ci_99")}, "L_P3": L_P3, "L_P4": {"main_AUROC": L_P1, "null_AUROC": null_auc, "difference": L_P4, "ci_99": intervals.get("L_P4_main_minus_null", {}).get("ci_99")}, "L_P5": {"boundaries": summary["separation_quartile_boundaries"], "bins": separation}, "L_P6": geometry, "sensitivity": {stratum: {"AUROC": weighted_auc([record for record in sensitivity_records if record["stratum"] == stratum]), "selected_correct": row_weighted([row for row in rows if row["stratum"] == stratum], "sensitivity_selected_correct")} for stratum in ("recoverable", "pool_correct")}, "unmappable": unmappable, "decision": decision, "kill_conditions": {"L_K1": L_K1, "L_K2": L_K2, "L_K3": L_K3, "L_K4": L_K4, "L_K5": L_K5}, "method_claim_allowed": False, "future_method_authorized": decision == "L_D2" and not L_K1 and not L_K2 and not L_K3 and not L_K4 and not L_K5, "private_rows": {"path": str(PRIVATE_EVAL_PATH.relative_to(ROOT)), "rows": len(rows), "bytes": PRIVATE_EVAL_PATH.stat().st_size, "sha256": sha256_file(PRIVATE_EVAL_PATH)}, "trace": {"path": str(TRACE_PATH.relative_to(ROOT)), "bytes": TRACE_PATH.stat().st_size, "sha256": sha256_file(TRACE_PATH)}}
    atomic_json(OUTPUT_PATH, output)
    print(json.dumps({"status": output["status"], "L_P1": output["L_P1"], "L_P2": output["L_P2"], "L_P3": output["L_P3"], "L_P4": output["L_P4"], "decision": decision, "kills": output["kill_conditions"], "future_method_authorized": output["future_method_authorized"]}, indent=2))


def point_inside(point, bbox):
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def summary_values(values):
    array = np.asarray(values, dtype=float)
    return {"minimum": float(array.min()), "q1": float(np.quantile(array, 0.25)), "median": float(np.quantile(array, 0.5)), "mean": float(array.mean()), "q3": float(np.quantile(array, 0.75)), "maximum": float(array.max())}


if __name__ == "__main__":
    main()