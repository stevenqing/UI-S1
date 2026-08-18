import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from qwen_vl_utils import smart_resize

from look_common import ROOT, RUN_DIR, TAU_BY_FOLD, allocate_counts, atomic_json, confrontation_window, find_null_window, read_jsonl, sample_hash, sha256_file, write_jsonl_fsynced


MASK_DIR = ROOT / "runs/mask/2026-08-14"
GRAN_DIR = ROOT / "runs/gran/2026-08-14"
SOURCEBIAS_DIR = ROOT / "runs/sourcebias/2026-08-03"
CONFIG_PATH = RUN_DIR / "configs/look_prereg.yaml"
SPEC_PATH = RUN_DIR / "SPEC.md"
TAU_PATH = GRAN_DIR / "TAU_SWEEP.json"
GRAN_COMMON_PATH = GRAN_DIR / "gran_common.py"
MASK_COMMON_PATH = MASK_DIR / "mask_common.py"
IMAGE_MANIFEST_PATH = ROOT / "runs/owin/2026-08-17/INPUT_IMAGE_MANIFEST.jsonl"
OUTPUT_PATH = RUN_DIR / "PREFLIGHT.json"
PRIVATE_PATH = RUN_DIR / "raw/private_preflight_rows.jsonl"
SAMPLE_PATH = RUN_DIR / "SAMPLE_WINDOW_MANIFEST.jsonl"
INFERENCE_PATH = RUN_DIR / "INFERENCE_INPUT_MANIFEST.jsonl"
SUMMARY_PATH = RUN_DIR / "SAMPLE_SUMMARY.json"
TARGETS = {"recoverable": 250, "pool_correct": 250}

sys.path.insert(0, str(MASK_DIR))
sys.path.insert(0, str(SOURCEBIAS_DIR))
from mask_common import load_rows, mode_center, ranked_modes, source_reliability
from sourcebias_common import b3_select_index, point_in_bbox


def main():
    if any(path.exists() for path in (OUTPUT_PATH, PRIVATE_PATH, SAMPLE_PATH, INFERENCE_PATH, SUMMARY_PATH)):
        raise FileExistsError("LOOK preparation output exists")
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if config["gpu"]["authorized"] is not False or config["status"] != "PREREGISTERED_BEFORE_ANY_LOOK_RESULT":
        raise PermissionError("LOOK preregistration mismatch")
    tau = json.loads(TAU_PATH.read_text())["screenspot_pro"]["folds"]
    observed_tau = tuple(float(row["selected_tau"].split(":", 1)[1]) for row in sorted(tau, key=lambda row: row["outer_fold"]))
    if observed_tau != TAU_BY_FOLD:
        raise ValueError("LOOK tau anchor mismatch")
    rows = load_rows()
    images = {row["row_id"]: row for row in read_jsonl(IMAGE_MANIFEST_PATH)}
    if len(rows) != 1581 or set(rows) != set(images):
        raise ValueError("LOOK identity mismatch")
    reliability = {}
    for fold in range(5):
        development = [row_id for row_id, row in rows.items() if row["fold"] != fold]
        reliability[fold] = source_reliability(rows, development)

    private_records = []
    eligible_by_stratum = defaultdict(list)
    b3_correct_count = 0
    for row_id in sorted(rows):
        row = rows[row_id]
        fold = row["fold"]
        modes = ranked_modes(row["gran_candidates"], reliability[fold], TAU_BY_FOLD[fold])
        candidates = row["candidates"]
        selected_index, selected_group = b3_select_index(candidates)
        b3_correct = bool(candidates[selected_index]["correct"])
        b3_correct_count += b3_correct
        recoverable = not b3_correct and any(candidate["correct"] for candidate in candidates)
        stratum = "pool_correct" if b3_correct else ("recoverable" if recoverable else "other")
        record = {"row_id": row_id, "application": row["application"], "fold": fold, "stratum": stratum, "b3_selected_index": selected_index, "b3_selected_group": list(selected_group), "b3_correct": b3_correct, "candidate_correct": [bool(candidate["correct"]) for candidate in candidates], "target_bbox": row["target_bbox"], "mode_count": len(modes), "eligible": False, "ineligibility": []}
        if len(modes) < 3:
            record["ineligibility"].append("FEWER_THAN_THREE_MODES")
            private_records.append(record)
            continue
        mode_records = []
        for rank, mode in enumerate(modes):
            centroid = mode_center(candidates, mode["members"])
            mode_records.append({"rank": rank + 1, "members": mode["members"], "centroid": list(centroid), "correct": bool(mode["correct"]), "representative_order": mode["representative_order"], "representative_correct": bool(mode["representative_correct"])})
        width, height = row["image_size"]
        main_window = confrontation_window(width, height, [mode_records[0]["centroid"], mode_records[1]["centroid"]])
        sensitivity_window = confrontation_window(width, height, [mode_records[0]["centroid"], mode_records[1]["centroid"], mode_records[2]["centroid"]])
        if main_window["status"] != "FEASIBLE":
            record["ineligibility"].append("MAIN_WINDOW_INFEASIBLE")
        if sensitivity_window["status"] != "FEASIBLE":
            record["ineligibility"].append("SENSITIVITY_WINDOW_INFEASIBLE")
        null_window = None
        if main_window["status"] == "FEASIBLE":
            null_window = find_null_window(row_id, width, height, mode_records[0]["centroid"], [candidate["point"] for candidate in candidates], main_window["area"])
            if null_window["status"] != "FEASIBLE":
                record["ineligibility"].append("NULL_WINDOW_INFEASIBLE")
        if stratum not in TARGETS:
            record["ineligibility"].append("NOT_SAMPLING_STRATUM")
        record.update({"modes": mode_records, "main_window": main_window, "sensitivity_window": sensitivity_window, "null_window": null_window, "normalized_M1_M2_separation": math.dist(mode_records[0]["centroid"], mode_records[1]["centroid"]) / math.hypot(width, height)})
        record["eligible"] = not record["ineligibility"]
        if record["eligible"]:
            eligible_by_stratum[stratum].append(row_id)
        private_records.append(record)
    if b3_correct_count != 1007:
        raise ValueError(f"LOOK B3 anchor mismatch: {b3_correct_count}")
    write_jsonl_fsynced(PRIVATE_PATH, private_records)
    private_by_id = {row["row_id"]: row for row in private_records}

    recoverable_distances = [private_by_id[row_id]["normalized_M1_M2_separation"] for row_id in eligible_by_stratum["recoverable"]]
    separation_boundaries = [float(value) for value in np.quantile(recoverable_distances, [0.25, 0.5, 0.75], method="linear")] if recoverable_distances else []
    sample_records = []
    allocations = {}
    for stratum, target in TARGETS.items():
        by_application = defaultdict(list)
        for row_id in eligible_by_stratum[stratum]:
            by_application[private_by_id[row_id]["application"]].append(row_id)
        populations = {application: len(row_ids) for application, row_ids in by_application.items()}
        allocation = allocate_counts(populations, min(target, sum(populations.values())))
        allocations[stratum] = {}
        for application in sorted(by_application):
            ordered = sorted(by_application[application], key=lambda row_id: (sample_hash(stratum, application, row_id), row_id))
            selected = ordered[: allocation[application]]
            probability = allocation[application] / populations[application]
            allocations[stratum][application] = {"population": populations[application], "sample": allocation[application], "inclusion_probability": probability, "inverse_probability_weight": 1 / probability}
            for row_id in selected:
                private = private_by_id[row_id]
                row = rows[row_id]
                windows = []
                for kind, window in (("main", private["main_window"]), ("sensitivity", private["sensitivity_window"]), ("null", private["null_window"]["window"])):
                    crop_width, crop_height = window["dimensions"]
                    resized_height, resized_width = smart_resize(crop_height * 2, crop_width * 2, factor=28, min_pixels=3136, max_pixels=4096 * 2160)
                    windows.append({"kind": kind, "final_window": window["final_window"], "dimensions": window["dimensions"], "area_fraction": window["area_fraction"], "processor_resized_size": [resized_width, resized_height]})
                sample_records.append({"sample_id": f"look-{row_id}", "row_id": row_id, "stratum": stratum, "application": application, "fold": private["fold"], "sample_hash": sample_hash(stratum, application, row_id), "cell_population": populations[application], "cell_sample": allocation[application], "inclusion_probability": probability, "inverse_probability_weight": 1 / probability, "normalized_M1_M2_separation": private["normalized_M1_M2_separation"], "image": images[row_id], "windows": windows})
    sample_records.sort(key=lambda row: ({"recoverable": 0, "pool_correct": 1}[row["stratum"]], row["row_id"]))
    write_jsonl_fsynced(SAMPLE_PATH, sample_records)
    inference_records = []
    for sample in sample_records:
        row = rows[sample["row_id"]]
        inference_records.append({"sample_id": sample["sample_id"], "row_id": sample["row_id"], "instruction": next(candidate for candidate in row["candidates"])["instruction"] if "instruction" in row["candidates"][0] else None, "image": sample["image"], "windows": sample["windows"]})
    # Instruction is sourced separately below because MASK candidates do not carry it.
    gta1 = {}
    for path in sorted((ROOT / "runs/ccm-h2h/2026-07-31/h1/shards/top18").glob("shard-*.jsonl")):
        for value in read_jsonl(path):
            gta1[value["id"]] = value
    for record in inference_records:
        record["instruction"] = gta1[record["row_id"]]["instruction"]
    write_jsonl_fsynced(INFERENCE_PATH, inference_records)
    summary = {"schema_version": 1, "status": "PASS_LOOK_SAMPLE_AND_WINDOWS_FROZEN_NO_GPU", "gpu_used": False, "gpu_authorized": False, "eligible_counts": {stratum: len(values) for stratum, values in eligible_by_stratum.items()}, "sample_counts": {stratum: sum(row["stratum"] == stratum for row in sample_records) for stratum in TARGETS}, "formal_calls": len(sample_records) * 3, "separation_quartile_boundaries": separation_boundaries, "allocations": allocations, "observational_L_K3": any(sum(row["stratum"] == stratum for row in sample_records) < 150 for stratum in TARGETS), "private_rows": {"path": str(PRIVATE_PATH.relative_to(ROOT)), "rows": len(private_records), "bytes": PRIVATE_PATH.stat().st_size, "sha256": sha256_file(PRIVATE_PATH)}, "sample_manifest": {"path": str(SAMPLE_PATH.relative_to(ROOT)), "rows": len(sample_records), "bytes": SAMPLE_PATH.stat().st_size, "sha256": sha256_file(SAMPLE_PATH)}, "inference_manifest": {"path": str(INFERENCE_PATH.relative_to(ROOT)), "rows": len(inference_records), "bytes": INFERENCE_PATH.stat().st_size, "sha256": sha256_file(INFERENCE_PATH)}, "next_action": "COMMIT_RUNNER_AND_TESTS_THEN_SEPARATE_GPU_AUTHORIZATION"}
    atomic_json(SUMMARY_PATH, summary)
    preflight = {"schema_version": 1, "status": "PASS_LOOK_PREFLIGHT_AND_EVALUATION_SIDE_PREPARATION", "gpu_used": False, "gpu_authorized": False, "rows": len(rows), "B3_correct_rows": b3_correct_count, "tau_by_fold": list(TAU_BY_FOLD), "dependencies": {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in (CONFIG_PATH, SPEC_PATH, TAU_PATH, GRAN_COMMON_PATH, MASK_COMMON_PATH, IMAGE_MANIFEST_PATH)}, "sample_summary_sha256": sha256_file(SUMMARY_PATH)}
    atomic_json(OUTPUT_PATH, preflight)
    print(json.dumps({"status": preflight["status"], "eligible": summary["eligible_counts"], "sample": summary["sample_counts"], "calls": summary["formal_calls"], "L_K3": summary["observational_L_K3"], "gpu_authorized": False}, indent=2))


if __name__ == "__main__":
    main()