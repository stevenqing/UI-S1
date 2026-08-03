import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ALLOCATION_DIR = ROOT / "runs/allocation-law/2026-08-01"
sys.path.insert(0, str(ALLOCATION_DIR))
from allocation_eval import (
    build_pool,
    compact_evaluation,
    group_folds,
    load_gta1,
    load_l1_units,
    load_manifest,
    load_model_views,
)
from run_l2 import stratified_group_sample_counts


BUDGETS = (4, 8, 12, 16)
METHODS = ("B3_mvp", "M1_ccm")
SEED = 20260802
RESAMPLES = 10000


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slope(values):
    x_values = np.asarray(BUDGETS, dtype=np.float64)
    centered = x_values - x_values.mean()
    return float(np.dot(centered, np.asarray(values)) / np.dot(centered, centered))


def load_sources():
    manifest = load_manifest(ALLOCATION_DIR / "raw/shared_regions_n12.jsonl")
    gta1 = load_gta1(ROOT / "runs/ccm-h2h/2026-07-31/h1/shards/top18", manifest)
    shard_root = ALLOCATION_DIR / "shards"
    generated = {
        "Qwen3-VL-8B-Instruct": load_model_views(
            ROOT / "runs/ccm-h2h/2026-07-31/h3/shards/qwen3_views",
            sorted(shard_root.glob("qwen3-views-4-11-*.jsonl")),
            manifest,
            "Qwen3-VL-8B-Instruct",
        ),
        "UI-TARS-7B-SFT": load_model_views(
            ROOT / "runs/ccm-h2h/2026-07-31/h3/shards/uitars_views",
            sorted(shard_root.glob("uitars-views-4-11-*.jsonl")),
            manifest,
            "UI-TARS-7B-SFT",
        ),
    }
    units = load_l1_units(ALLOCATION_DIR / "configs/l1_pools.yaml")
    return gta1, generated, units


def reconstruct(gta1, generated, units):
    evaluations = {"v_only": {}, "mixed": {}}
    rows_by_pool = {"v_only": {}, "mixed": {}}
    for budget in BUDGETS:
        pool_units = {
            "v_only": [("GTA1-7B", view) for view in range(budget)],
            "mixed": units[budget],
        }
        for pool, selected_units in pool_units.items():
            rows = build_pool(gta1, generated, selected_units)
            rows_by_pool[pool][budget] = rows
            evaluations[pool][budget] = compact_evaluation(rows)
    return rows_by_pool, evaluations


def validate_parity(evaluations):
    l1_path = ALLOCATION_DIR / "L1_RESULTS.json"
    l1 = json.loads(l1_path.read_text())
    differences = {}
    for pool in evaluations:
        for budget, evaluation in evaluations[pool].items():
            for method in (*METHODS, "pass_at_n"):
                expected = l1["evaluations"][pool][str(budget)]["accuracy"][method]
                actual = evaluation["accuracy"][method]
                difference = actual - expected
                differences[f"{pool}/N{budget}/{method}"] = difference
                if abs(difference) > 1e-15:
                    raise ValueError(f"X3 L1 parity mismatch: {pool}/N{budget}/{method}: {difference}")
    return {"source_sha256": sha256_file(l1_path), "max_absolute_difference": max(map(abs, differences.values()))}


def bootstrap_slopes(rows, evaluations):
    base_rows = rows["v_only"][4]
    fold_for_group, fold_rows = group_folds(base_rows)
    groups = sorted(fold_for_group)
    group_index = {group: index for index, group in enumerate(groups)}
    row_counts = np.zeros(len(groups), dtype=np.int64)
    for row in base_rows:
        row_counts[group_index[row["application"]]] += 1
    rng = np.random.default_rng(SEED)
    sample_counts = stratified_group_sample_counts(groups, fold_for_group, RESAMPLES, rng)
    denominators = sample_counts @ row_counts
    if np.any(denominators <= 0):
        raise ValueError("X3 stratified bootstrap produced an empty replicate")

    x_values = np.asarray(BUDGETS, dtype=np.float64)
    centered = x_values - x_values.mean()
    denominator = np.dot(centered, centered)
    reports = {}
    for pool in evaluations:
        reports[pool] = {}
        for method in METHODS:
            group_successes = np.zeros((len(groups), len(BUDGETS)), dtype=np.int64)
            for budget_index, budget in enumerate(BUDGETS):
                outputs = evaluations[pool][budget]["outputs"][method]
                for row in rows[pool][budget]:
                    group_successes[group_index[row["application"]], budget_index] += int(outputs[row["id"]])
            bootstrap_accuracy = (sample_counts @ group_successes) / denominators[:, None]
            bootstrap_values = (bootstrap_accuracy @ centered) / denominator
            point_values = [evaluations[pool][budget]["accuracy"][method] for budget in BUDGETS]
            reports[pool][method] = {
                "point_slope_per_forward": slope(point_values),
                "bootstrap_mean": float(np.mean(bootstrap_values)),
                "ci_99": [
                    float(np.quantile(bootstrap_values, 0.005)),
                    float(np.quantile(bootstrap_values, 0.995)),
                ],
                "p_slope_nonnegative": float(np.mean(bootstrap_values >= 0)),
                "resamples": RESAMPLES,
                "seed": SEED,
            }
    return reports, {
        "groups": len(groups),
        "fold_rows": fold_rows,
        "replicate_row_count_range": [int(denominators.min()), int(denominators.max())],
    }


def area_strata(gta1, evaluations):
    area = {}
    for row_id, row in gta1.items():
        width, height = row["img_size"]
        left, top, right, bottom = row["target_bbox"]
        value = max(0, right - left) * max(0, bottom - top) / (width * height)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"X3 invalid target area: {row_id}")
        area[row_id] = value
    ordered = sorted(area, key=lambda row_id: (area[row_id], row_id))
    bins = np.array_split(np.asarray(ordered, dtype=object), 5)
    reports = []
    for index, values in enumerate(bins):
        ids = values.tolist()
        record = {
            "bin": index,
            "rows": len(ids),
            "area_ratio_min": min(area[row_id] for row_id in ids),
            "area_ratio_mean": float(np.mean([area[row_id] for row_id in ids])),
            "area_ratio_max": max(area[row_id] for row_id in ids),
            "mixed_minus_v_only": {},
        }
        for budget in BUDGETS:
            record["mixed_minus_v_only"][str(budget)] = {}
            for method in METHODS:
                mixed = evaluations["mixed"][budget]["outputs"][method]
                v_only = evaluations["v_only"][budget]["outputs"][method]
                record["mixed_minus_v_only"][str(budget)][method] = float(
                    np.mean([mixed[row_id] for row_id in ids]) - np.mean([v_only[row_id] for row_id in ids])
                )
        reports.append(record)
    return reports


def pdf_escape(value):
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_figure(curves, output):
    width, height = 760, 480
    left, right, bottom, top = 75, 720, 70, 430
    series = []
    for pool in ("v_only", "mixed"):
        for method in METHODS:
            label = f"{pool} {method}"
            values = [(budget, 100 * curves[pool][str(budget)][method]) for budget in BUDGETS]
            series.append((label, values))
    all_y = [value for _, values in series for _, value in values]
    y_min = min(all_y) - 2
    y_max = max(all_y) + 2

    def px(value):
        return left + (value - BUDGETS[0]) / (BUDGETS[-1] - BUDGETS[0]) * (right - left)

    def py(value):
        return bottom + (value - y_min) / (y_max - y_min) * (top - bottom)

    colors = ((0.7, 0.15, 0.15), (0.9, 0.45, 0.1), (0.1, 0.35, 0.7), (0.1, 0.6, 0.4))
    content = ["0.8 w", f"{left} {bottom} m {left} {top} l {right} {top} l {right} {bottom} l h S"]
    for series_index, (name, values) in enumerate(series):
        red, green, blue = colors[series_index]
        content.append(f"{red} {green} {blue} RG 1.8 w")
        for index, (x_value, y_value) in enumerate(values):
            content.append(f"{px(x_value):.2f} {py(y_value):.2f} {'m' if index == 0 else 'l'}")
        content.append("S")
        for x_value, y_value in values:
            content.append(f"{px(x_value)-2:.2f} {py(y_value)-2:.2f} 4 4 re f")
        legend_y = top - 17 * series_index
        content.extend([
            f"{red} {green} {blue} RG {right-190} {legend_y} m {right-170} {legend_y} l S",
            "0 0 0 RG", "BT /F1 9 Tf", f"{right-165} {legend_y-3} Td ({pdf_escape(name)}) Tj ET",
        ])
    for budget in BUDGETS:
        content.extend(["0 0 0 RG", "BT /F1 9 Tf", f"{px(budget)-5:.2f} {bottom-16} Td ({budget}) Tj ET"])
    content.extend([
        "BT /F1 11 Tf", f"{left+245} 28 Td (Forward budget N) Tj ET",
        "BT /F1 11 Tf 0 1 -1 0 18 155 Tm (Accuracy percent) Tj ET",
        "BT /F1 9 Tf", f"{left-35} {bottom} Td ({y_min:.1f}) Tj ET",
        "BT /F1 9 Tf", f"{left-35} {top} Td ({y_max:.1f}) Tj ET",
    ])
    stream = "\n".join(content).encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>".encode(),
        f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    document = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, value in enumerate(objects, start=1):
        offsets.append(len(document))
        document.extend(f"{index} 0 obj\n".encode() + value + b"\nendobj\n")
    xref = len(document)
    document.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f\n".encode())
    for offset in offsets:
        document.extend(f"{offset:010d} 00000 n\n".encode())
    document.extend(f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(document)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    gta1, generated, units = load_sources()
    rows, evaluations = reconstruct(gta1, generated, units)
    parity = validate_parity(evaluations)
    slopes, bootstrap_design = bootstrap_slopes(rows, evaluations)
    strata = area_strata(gta1, evaluations)
    curves = {
        pool: {
            str(budget): {method: evaluations[pool][budget]["accuracy"][method] for method in METHODS}
            for budget in BUDGETS
        }
        for pool in evaluations
    }
    primary_v = slopes["v_only"]["M1_ccm"]["ci_99"]
    primary_mixed = slopes["mixed"]["M1_ccm"]["ci_99"]
    satisfied = primary_v[1] < 0 < primary_mixed[0]
    result = {
        "schema_version": 1,
        "status": "PASS",
        "budgets": list(BUDGETS),
        "curve_accuracy": curves,
        "l1_parity": parity,
        "bootstrap_design": bootstrap_design,
        "slopes": slopes,
        "prediction": {
            "primary_rule": "M1_ccm",
            "requires": "v_only_CI_upper_lt_0_lt_mixed_CI_lower",
            "satisfied": satisfied,
        },
        "kill_conditions": {"X-K2": not satisfied},
        "area_strata": strata,
        "N24": {
            "v_only": "STRUCTURALLY_UNAVAILABLE_16_TO_19_UNIQUE_CANDIDATES",
            "mixed_accuracy": json.loads((ALLOCATION_DIR / "L1_RESULTS.json").read_text())["evaluations"]["mixed"]["24"]["accuracy"],
            "inference": "ONE_SIDED_DISPLAY_ONLY_EXCLUDED_FROM_SLOPE_AND_BILATERAL_TESTS",
        },
        "figure": str(args.figure),
    }
    write_figure(curves, args.figure)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "slopes": slopes,
        "prediction": result["prediction"],
        "kill_conditions": result["kill_conditions"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
