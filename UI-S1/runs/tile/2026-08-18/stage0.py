import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from tile_common import N_GRID, ROOT, RUN_DIR, atomic_json, atomic_text, eccentricity, fit_curve, ledger_record, read_jsonl, score_layout, select_n, sha256_file, write_jsonl_fsynced


ALLOCATION_DIR = ROOT / "runs/allocation-law/2026-08-01"
GTA1_ROOT = ROOT / "runs/ccm-h2h/2026-07-31/h1/shards/top18"
REGION_PATH = ALLOCATION_DIR / "raw/shared_regions_n12.jsonl"
COVER_PATH = ROOT / "runs/cover/2026-08-16/raw/arm_a_rows.jsonl"
CWIN_PATH = ROOT / "runs/cwin/2026-08-17/raw/stage0_rows.jsonl"
OWIN_RAW_PATH = ROOT / "runs/owin/2026-08-17/raw/arm_b_rows.jsonl"
PREFLIGHT_PATH = RUN_DIR / "PREFLIGHT.json"
CONFIG_PATH = RUN_DIR / "configs/tile_prereg.yaml"
PAIRS_PATH = RUN_DIR / "raw/eccentricity_pairs.jsonl"
CURVES_PATH = RUN_DIR / "raw/fold_curves.jsonl"
SCORES_PATH = RUN_DIR / "raw/row_scores.jsonl"
OUTPUT_PATH = RUN_DIR / "STAGE0.json"
REPORT_PATH = RUN_DIR / "REPORT.md"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


allocation = load_module(ALLOCATION_DIR / "allocation_eval.py", "tile_stage0_allocation")


def curve_records(curve, outer_fold, phase, fit_folds):
    records = []
    for scale, value in curve["scales"].items():
        for row in value["bins"]:
            records.append({"outer_fold": outer_fold, "phase": phase, "fit_folds": fit_folds, "scale": scale, "area_median": curve["area_median"], "boundaries": curve["boundaries"], "scale_pairs": value["pairs"], "scale_pooled_correctness": value["pooled_correctness"], **row})
    return records


def summarize_rows(rows):
    repair = sum(row["expected_repair"] for row in rows)
    damage = sum(row["expected_damage"] for row in rows)
    return {"rows": len(rows), "expected_repair": repair, "expected_damage": damage, "expected_net": repair - damage, "expected_net_pp": 100 * (repair - damage) / len(rows), "hard_below_0_5_rows": sum(row["hard_below_0_5"] for row in rows)}


def grouped_bootstrap(rows, keys, resamples, seed):
    applications = sorted({row["application"] for row in rows})
    by_app = {application: [row for row in rows if row["application"] == application] for application in applications}
    rng = np.random.default_rng(seed)
    values = {key: [] for key in keys}
    for _ in range(resamples):
        selected = rng.choice(applications, size=len(applications), replace=True)
        current = [row for application in selected for row in by_app[application]]
        for key in keys:
            values[key].append(float(np.mean([row[key] for row in current])))
    return {key: {"point": float(np.mean([row[key] for row in rows])), "ci_99": [float(np.quantile(value, 0.005)), float(np.quantile(value, 0.995))], "resamples": resamples} for key, value in values.items()}


def main():
    if any(path.exists() for path in (PAIRS_PATH, CURVES_PATH, SCORES_PATH, OUTPUT_PATH, REPORT_PATH)):
        raise FileExistsError("TILE Stage 0 output exists")
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if preflight["status"] != "PASS_TILE_PREFLIGHT_NO_TILE_STATISTIC" or preflight["stage0_computed"] is not False or config["gpu"]["stage1_authorized"] is not False:
        raise PermissionError("TILE Stage 0 preflight mismatch")
    manifest = allocation.load_manifest(REGION_PATH)
    gta1 = allocation.load_gta1(GTA1_ROOT, manifest)
    cover = {row["row_id"]: row for row in read_jsonl(COVER_PATH)}
    cwin = {row["row_id"]: row for row in read_jsonl(CWIN_PATH)}
    layouts = {(row["row_id"], row["N"]): row["tiling"]["rectangles"] for row in read_jsonl(OWIN_RAW_PATH) if row["N"] in N_GRID}
    pairs = []
    areas = {}
    for row_id in sorted(gta1):
        bbox = gta1[row_id]["target_bbox"]
        area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
        areas[row_id] = area
        for slot, candidate in enumerate(gta1[row_id]["candidates"][1:12], start=1):
            rectangle = candidate["region"]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            if rectangle[0] <= center_x < rectangle[2] and rectangle[1] <= center_y < rectangle[3]:
                pairs.append({"row_id": row_id, "application": cover[row_id]["application"], "fold": cover[row_id]["fold"], "slot": slot, "eccentricity": eccentricity(rectangle, bbox), "target_area": area, "correct": bool(allocation.point_in_bbox(candidate["point"], bbox))})
    write_jsonl_fsynced(PAIRS_PATH, pairs)
    curve_rows = []
    score_rows = []
    fold_selections = []
    for outer_fold in range(5):
        inner_fold = (outer_fold + 1) % 5
        inner_train_folds = [fold for fold in range(5) if fold not in {outer_fold, inner_fold}]
        inner_pairs = [row for row in pairs if row["fold"] in inner_train_folds]
        inner_ids = {row["row_id"] for row in inner_pairs}
        inner_curve = fit_curve(inner_pairs, {row_id: areas[row_id] for row_id in inner_ids})
        curve_rows.extend(curve_records(inner_curve, outer_fold, "inner_train", inner_train_folds))
        inner_scores = {}
        for N in N_GRID:
            values = []
            for row_id in sorted(row_id for row_id in gta1 if cover[row_id]["fold"] == inner_fold):
                score = score_layout(layouts[(row_id, N)], gta1[row_id]["target_bbox"], areas[row_id], inner_curve)
                values.append(score["p_hat"] - int(cwin[row_id]["original_b3_correct"]))
                ledger = ledger_record(score["p_hat"], cwin[row_id]["original_b3_correct"])
                contextual = ledger_record(score["p_hat"], cover[row_id]["b3_correct"])
                score_rows.append({"row_id": row_id, "application": cover[row_id]["application"], "outer_fold": outer_fold, "phase": "inner_validation", "N": N, "V_only_B3_correct": bool(cwin[row_id]["original_b3_correct"]), "C_uni_B3_correct": bool(cover[row_id]["b3_correct"]), "target_stratum": cover[row_id]["target_stratum"], "crop_covered": cover[row_id]["target_coverage_count"] > 0, "tile_rectangles_sha256": __import__("hashlib").sha256(json.dumps(layouts[(row_id, N)], separators=(",", ":")).encode()).hexdigest(), **score, **ledger, "C_uni_expected_repair": contextual["expected_repair"], "C_uni_expected_damage": contextual["expected_damage"], "C_uni_expected_net": contextual["expected_net"]})
            inner_scores[N] = float(np.mean(values))
        selected_N = select_n(inner_scores)
        development_folds = [fold for fold in range(5) if fold != outer_fold]
        development_pairs = [row for row in pairs if row["fold"] in development_folds]
        development_ids = {row["row_id"] for row in development_pairs}
        outer_curve = fit_curve(development_pairs, {row_id: areas[row_id] for row_id in development_ids})
        curve_rows.extend(curve_records(outer_curve, outer_fold, "outer_development", development_folds))
        fold_selections.append({"outer_fold": outer_fold, "inner_validation_fold": inner_fold, "inner_train_folds": inner_train_folds, "inner_expected_net": {str(key): value for key, value in inner_scores.items()}, "selected_N": selected_N, "endpoint": selected_N in {4, 11}})
        for N in N_GRID:
            for row_id in sorted(row_id for row_id in gta1 if cover[row_id]["fold"] == outer_fold):
                score = score_layout(layouts[(row_id, N)], gta1[row_id]["target_bbox"], areas[row_id], outer_curve)
                ledger = ledger_record(score["p_hat"], cwin[row_id]["original_b3_correct"])
                contextual = ledger_record(score["p_hat"], cover[row_id]["b3_correct"])
                score_rows.append({"row_id": row_id, "application": cover[row_id]["application"], "outer_fold": outer_fold, "phase": "outer_test", "N": N, "selected_policy": N == selected_N, "V_only_B3_correct": bool(cwin[row_id]["original_b3_correct"]), "C_uni_B3_correct": bool(cover[row_id]["b3_correct"]), "target_stratum": cover[row_id]["target_stratum"], "crop_covered": cover[row_id]["target_coverage_count"] > 0, "tile_rectangles_sha256": __import__("hashlib").sha256(json.dumps(layouts[(row_id, N)], separators=(",", ":")).encode()).hexdigest(), **score, **ledger, "C_uni_expected_repair": contextual["expected_repair"], "C_uni_expected_damage": contextual["expected_damage"], "C_uni_expected_net": contextual["expected_net"]})
    write_jsonl_fsynced(CURVES_PATH, curve_rows)
    write_jsonl_fsynced(SCORES_PATH, score_rows)
    outer_rows = [row for row in score_rows if row["phase"] == "outer_test"]
    fixed = {}
    bootstrap = {}
    for N in N_GRID:
        current = [row for row in outer_rows if row["N"] == N]
        fixed[str(N)] = {"full": summarize_rows(current), "V_only_original_correct": summarize_rows([row for row in current if row["V_only_B3_correct"]]), "C_uni_original_correct_contextual": summarize_rows([row for row in current if row["C_uni_B3_correct"]]), "C_uni_contextual_full": {"rows": len(current), "expected_repair": sum(row["C_uni_expected_repair"] for row in current), "expected_damage": sum(row["C_uni_expected_damage"] for row in current), "expected_net": sum(row["C_uni_expected_net"] for row in current), "expected_net_pp": 100 * float(np.mean([row["C_uni_expected_net"] for row in current]))}, "crop_covered": summarize_rows([row for row in current if row["crop_covered"]]), "strata": {stratum: summarize_rows([row for row in current if row["target_stratum"] == stratum]) for stratum in ("uncovered_0", "partial_1_10", "common_11")}, "fold_expected_net_pp": {str(fold): 100 * float(np.mean([row["expected_net"] for row in current if row["outer_fold"] == fold])) for fold in range(5)}}
        bootstrap[str(N)] = grouped_bootstrap(current, ("expected_net", "expected_repair", "expected_damage"), 10000, 20261800 + N)
    selected_rows = [row for row in outer_rows if row["selected_policy"]]
    selected_summary = {"full": summarize_rows(selected_rows), "V_only_original_correct": summarize_rows([row for row in selected_rows if row["V_only_B3_correct"]]), "C_uni_original_correct_contextual": summarize_rows([row for row in selected_rows if row["C_uni_B3_correct"]]), "C_uni_contextual_full": {"rows": len(selected_rows), "expected_repair": sum(row["C_uni_expected_repair"] for row in selected_rows), "expected_damage": sum(row["C_uni_expected_damage"] for row in selected_rows), "expected_net": sum(row["C_uni_expected_net"] for row in selected_rows), "expected_net_pp": 100 * float(np.mean([row["C_uni_expected_net"] for row in selected_rows]))}, "crop_covered": summarize_rows([row for row in selected_rows if row["crop_covered"]]), "strata": {stratum: summarize_rows([row for row in selected_rows if row["target_stratum"] == stratum]) for stratum in ("uncovered_0", "partial_1_10", "common_11")}}
    repair = selected_summary["full"]["expected_repair"]
    ratio = selected_summary["full"]["expected_damage"] / repair if repair > 0 else None
    t_g1 = all(fixed[str(N)]["full"]["expected_net"] / 1581 < 0.007 for N in N_GRID)
    t_g2_review = ratio is None or ratio > 0.5
    output = {"schema_version": 1, "status": "PASS_TILE_STAGE0_COMPLETE_AWAITING_HUMAN_REVIEW" if t_g2_review else "PASS_TILE_STAGE0_COMPLETE", "evidence_status": "POST_SELECTION_EVALUATION_SIDE_OPTIMISTIC_PROXY", "gpu_used": False, "stage1_authorized": False, "preflight_sha256": sha256_file(PREFLIGHT_PATH), "pairs": {"rows": len(pairs), "path": str(PAIRS_PATH.relative_to(ROOT)), "bytes": PAIRS_PATH.stat().st_size, "sha256": sha256_file(PAIRS_PATH)}, "curves": {"rows": len(curve_rows), "path": str(CURVES_PATH.relative_to(ROOT)), "bytes": CURVES_PATH.stat().st_size, "sha256": sha256_file(CURVES_PATH)}, "row_scores": {"rows": len(score_rows), "path": str(SCORES_PATH.relative_to(ROOT)), "bytes": SCORES_PATH.stat().st_size, "sha256": sha256_file(SCORES_PATH)}, "fold_selections": fold_selections, "fixed_N": fixed, "selected_policy": selected_summary, "bootstrap": bootstrap, "T_G1": t_g1, "T_G2": {"damage_to_repair_ratio": ratio, "review_required": t_g2_review, "decision": "PENDING_HUMAN_REVIEW" if t_g2_review else "NOT_REQUIRED"}, "T_K5": any(row["endpoint"] for row in fold_selections), "next_action": "STOP_BEFORE_GPU" if t_g1 else ("HUMAN_T_G2_REVIEW_BEFORE_STAGE1" if t_g2_review else "MAY_WRITE_STAGE1_AMENDMENT")}
    atomic_json(OUTPUT_PATH, output)
    lines = ["# TILE Stage-0 Eccentricity Proxy Report", "", "Date: 2026-08-18", "", f"Status: `{output['status']}`", "", "Stage 0 is a zero-GPU, post-selection optimistic proxy. Max curve probability is not B3/M1 accuracy.", "", "| N | Expected net | Repair | Damage |", "| ---: | ---: | ---: | ---: |"]
    for N in N_GRID:
        value = fixed[str(N)]["full"]
        lines.append(f"| {N} | {value['expected_net_pp']:+.3f} pp | {value['expected_repair']:.2f} | {value['expected_damage']:.2f} |")
    lines.extend(["", f"Fold selections: `{[row['selected_N'] for row in fold_selections]}`. T-G1={t_g1}; T-G2 review={t_g2_review}; ratio={ratio}; T-K5={output['T_K5']}.", "", "All repair/damage values are fractional expectations, not observed flips. Stage 1 remains unauthorized."])
    atomic_text(REPORT_PATH, "\n".join(lines) + "\n")
    print(json.dumps({"status": output["status"], "pairs": len(pairs), "fixed_net_pp": {str(N): fixed[str(N)]["full"]["expected_net_pp"] for N in N_GRID}, "selected_N": [row["selected_N"] for row in fold_selections], "selected": selected_summary["full"], "T_G1": t_g1, "T_G2": output["T_G2"], "T_K5": output["T_K5"], "stage1_authorized": False}, indent=2))


if __name__ == "__main__":
    main()