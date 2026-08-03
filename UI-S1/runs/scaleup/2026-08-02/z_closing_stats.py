import argparse
import hashlib
import json
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
H3_DIR = ROOT / "runs/ccm-h2h/2026-07-31/h3"
CLOSING_DIR = ROOT / "runs/closing/2026-08-02"
sys.path.insert(0, str(H3_DIR))
sys.path.insert(0, str(CLOSING_DIR))
from h3_eval import build_pools, evaluate_pool, point_in_bbox
from score_eligibility import annotations, load_generated, load_qwen3
from closing_common import load_closing_pools
from f1_paired_bootstrap import paired_bootstrap


EXPECTED_ROWS = 1581
QWEN_SOURCE_SHA256 = "bde27a80602066974d411399266b7723175b18f9b5755dfe3526b4f251680776"


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    return json.loads(path.read_text())


def labels_from_points(rows, points):
    if set(points) != {row["id"] for row in rows}:
        raise ValueError("Z1 bare/mixed identity mismatch")
    return {
        row["id"]: bool(points[row["id"]] is not None and point_in_bbox(points[row["id"]], row["target_bbox"]))
        for row in rows
    }


def comparison(rows, left, right, left_name, right_name):
    result = paired_bootstrap(rows, left, right)
    result.update({
        "left": left_name,
        "right": right_name,
        "left_accuracy": sum(left.values()) / len(rows),
        "right_accuracy": sum(right.values()) / len(rows),
    })
    return result


def load_bare_outputs(reference_rows):
    collision = ROOT / "runs/collision-law/2026-07-30"
    gta_path = collision / "w3_artifacts/gta1_screenspot_pro/predictions.jsonl"
    gta_score = load_json(collision / "w3_artifacts/gta1_screenspot_pro/score.json")
    if sha256_file(gta_path) != gta_score["predictions_sha256"]:
        raise ValueError("Z1 GTA1 bare source hash mismatch")
    gta_rows = [json.loads(line) for line in gta_path.read_text().splitlines() if line.strip()]
    gta_points = {row["id"]: row["pred_point_original"] if row["parse_ok"] else None for row in gta_rows}

    annotation_root = collision / "w3_assets/ScreenSpot-Pro/annotations"
    annotation_rows = annotations(annotation_root)
    qwen_path = collision / "w3_assets/MVP/mvp_sspro_qwen3vl_8b/mvp_qwen3vl8b.json"
    if sha256_file(qwen_path) != QWEN_SOURCE_SHA256:
        raise ValueError("Z1 Qwen3 bare source hash mismatch")
    qwen_rows = load_qwen3(qwen_path, annotation_rows)
    qwen_points = {row["id"]: row["point"] if row["parse_ok"] else None for row in qwen_rows}

    uitars_rows = load_generated(H3_DIR / "shards/uitars_bare", 8)
    uitars_points = {row["id"]: row["predictions"][0]["point"] for row in uitars_rows}
    outputs = {
        "GTA1-7B": labels_from_points(reference_rows, gta_points),
        "Qwen3-VL-8B-Instruct": labels_from_points(reference_rows, qwen_points),
        "UI-TARS-7B-SFT": labels_from_points(reference_rows, uitars_points),
    }
    expected = {
        "GTA1-7B": 0.4939911448450348,
        "Qwen3-VL-8B-Instruct": 0.5464895635673624,
        "UI-TARS-7B-SFT": 0.33459835547122074,
    }
    for model, labels in outputs.items():
        accuracy = sum(labels.values()) / EXPECTED_ROWS
        if accuracy != expected[model]:
            raise ValueError(f"Z1 {model} bare accuracy mismatch: {accuracy}")
    return outputs, {
        "GTA1_predictions_sha256": sha256_file(gta_path),
        "Qwen3_trace_sha256": sha256_file(qwen_path),
        "UI_TARS_prediction_hashes_sha256": hashlib.sha256(
            "".join(sorted(row["prediction_sha256"] for row in uitars_rows)).encode()
        ).hexdigest(),
    }


def pct(value):
    return f"{100 * value:.2f}"


def write_main_table(path, result):
    comparisons = result["Z1_paired_bootstrap"]
    lines = [
        "# Scale-Up Main Table",
        "",
        "## Controlled 7B results",
        "",
        "| Comparison | Left | Right | Delta | 99% CI | One-sided p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record in comparisons.values():
        lines.append(
            f"| {record['left']} vs {record['right']} | {pct(record['left_accuracy'])} | {pct(record['right_accuracy'])} | "
            f"{100 * record['point_delta']:+.2f} pp | [{100 * record['ci_99'][0]:+.2f}, {100 * record['ci_99'][1]:+.2f}] | "
            f"{record['p_one_sided_delta_le_zero']:.4g} |"
        )
    lines.extend([
        "",
        "The H3-native B3 row is the primary drop-in comparison (63.63 vs 60.09). A later Allocation/Closing reconstruction gives 63.69 for the mixed side; that one-row implementation sensitivity is reported but is not substituted into the primary H3 statistic.",
        "",
        "## Reporting dispositions",
        "",
        "| Item | Main-text disposition |",
        "|---|---|",
        "| H1 N=2 | Appendix only: M1/M2 collapse to B0 and M1 headroom capture is 0% |",
        "| Budget decline | Use L1 N=4 to N=16 and X3 slope CI; H1 N=4 to N=10 is rule comparison only |",
        "| MDE | Use v1-only 0.09-1.16 pp; v2-v4 are deployment/information shifts |",
        "| Sampling | GUI-RC point slope is negative but 99% CI crosses zero; title scope is fixed-view allocation axis |",
        "",
        "## 72B gate",
        "",
        "G1 and G2 are pending local checkpoint acquisition and inference. Paper-only 70.4 and 73.1 remain context rows and are excluded from paired calculations.",
        "",
    ])
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--main-table", type=Path, required=True)
    args = parser.parse_args()

    h3_root = ROOT / "runs/ccm-h2h/2026-07-31"
    d1, d2 = build_pools(
        H3_DIR / "raw/gta1_N12.parquet",
        H3_DIR / "shards/qwen3_views",
        H3_DIR / "shards/uitars_views",
    )
    d1_eval = evaluate_pool(d1)
    d2_eval = evaluate_pool(d2)
    published_h3 = load_json(h3_root / "h3_mixed_pool.json")
    for pool_name, evaluation in (("D1_pure_views", d1_eval), ("D2_mixed", d2_eval)):
        if evaluation["accuracy"] != published_h3["pools"][pool_name]["accuracy"]:
            raise ValueError(f"Z1 H3 reconstruction mismatch: {pool_name}")

    _, closing_pools = load_closing_pools()
    n16_mixed = closing_pools["mixed_N16"]
    n16_v_only = closing_pools["v_only_N16"]
    bare, bare_sources = load_bare_outputs(d2)
    comparisons = {
        "mixed_N12_M1_vs_v_only_N12_M1": comparison(
            d2, d2_eval["outputs"]["M1_ccm"], d1_eval["outputs"]["M1_ccm"], "Mixed N12 M1", "V-only N12 M1"
        ),
        "mixed_N12_B3_vs_v_only_N12_B3": comparison(
            d2, d2_eval["outputs"]["B3_mvp"], d1_eval["outputs"]["B3_mvp"], "Mixed N12 B3 H3-native", "V-only N12 B3"
        ),
    }
    for model, labels in bare.items():
        comparisons[f"mixed_N12_M1_vs_{model}_bare"] = comparison(
            d2, d2_eval["outputs"]["M1_ccm"], labels, "Mixed N12 M1", f"{model} bare"
        )
    for method in ("M1_ccm", "B3_mvp"):
        comparisons[f"mixed_N16_{method}_vs_v_only_N16_{method}"] = comparison(
            n16_mixed["rows"], n16_mixed["evaluation"]["outputs"][method],
            n16_v_only["evaluation"]["outputs"][method], f"Mixed N16 {method}", f"V-only N16 {method}"
        )

    h1 = load_json(h3_root / "h1_headtohead.json")
    x3_path = ROOT / "runs/diversity-axis/2026-08-02/x3_curve_stats.json"
    x3 = load_json(x3_path)
    mde_path = h3_root / "fixes/c1_mde.json"
    mde = load_json(mde_path)
    z5_path = CLOSING_DIR / "f2_sampling_axis.json"
    z5 = load_json(z5_path)
    result = {
        "schema_version": 1,
        "status": "PASS",
        "rows": EXPECTED_ROWS,
        "Z1_paired_bootstrap": comparisons,
        "Z1_B3_reconstruction_sensitivity": {
            "primary_H3_native": d2_eval["accuracy"]["B3_mvp"],
            "later_allocation_reconstruction": closing_pools["mixed_N12"]["evaluation"]["accuracy"]["B3_mvp"],
            "disposition": "retain_H3_native_for_requested_drop_in_comparison",
        },
        "Z2_N2_degenerate": {
            "accuracy": h1["accuracy"]["2"],
            "M1_headroom_capture": 0.0,
            "disposition": "remove_from_main_table_and_report_in_appendix",
        },
        "Z3_decline_scope": {
            "primary_interval": [4, 16],
            "v_only_B3_start": x3["curve_accuracy"]["v_only"]["4"]["B3_mvp"],
            "v_only_B3_end": x3["curve_accuracy"]["v_only"]["16"]["B3_mvp"],
            "v_only_B3_delta": x3["curve_accuracy"]["v_only"]["16"]["B3_mvp"] - x3["curve_accuracy"]["v_only"]["4"]["B3_mvp"],
            "v_only_M1_slope": x3["slopes"]["v_only"]["M1_ccm"],
            "H1_N4_to_N10_B3_delta": h1["accuracy"]["10"]["B3_mvp_official"] - h1["accuracy"]["4"]["B3_mvp_official"],
            "H1_MDE": published_h3["comparison"]["mde"],
            "H1_disposition": "same_candidate_set_rule_comparison_not_trend_evidence",
        },
        "Z4_MDE_scope": {
            "definition": mde["definition"],
            "cells": mde["cells"],
            "main_text_range": [min(value["absolute_delta"] for value in mde["cells"].values()), max(value["mde_v1_only"] for value in mde["cells"].values())],
            "excluded_views": mde["excluded_distribution_shift_views"],
            "disposition": "v1_only_main_text_v2_to_v4_appendix_distribution_shift",
        },
        "Z5_sampling_disposition": {
            "primary_slope": z5["slopes"]["S_only"]["GUI_RC"],
            "sampling_axis_covered": z5["prediction"]["sampling_axis_covered"],
            "title_scope": z5["prediction"]["title_scope"],
            "new_inference_required": False,
        },
        "sources": {
            **bare_sources,
            "h3_mixed_pool_sha256": sha256_file(h3_root / "h3_mixed_pool.json"),
            "x3_curve_stats_sha256": sha256_file(x3_path),
            "c1_mde_sha256": sha256_file(mde_path),
            "closing_f2_sampling_axis_sha256": sha256_file(z5_path),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_main_table(args.main_table, result)
    print(json.dumps({name: {"delta": value["point_delta"], "ci_99": value["ci_99"], "p": value["p_one_sided_delta_le_zero"]} for name, value in comparisons.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
