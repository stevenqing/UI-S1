import hashlib
import json
import math
import sys
from pathlib import Path

import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
SOURCEBIAS_DIR = ROOT / "runs/sourcebias/2026-08-03"
sys.path.insert(0, str(SOURCEBIAS_DIR))

from b2_lineage_normalized import centroid, run_scale
from sourcebias_common import load_pools, split_ids, split_72


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def legacy_weighted_centroid(points, indices, weights):
    values = [[float(points[index][axis]) for axis in (0, 1)] for index in indices]
    return [sum(value[axis] * weight for value, weight in zip(values, weights)) for axis in (0, 1)]


def selected_variants(report):
    return [report["outer_selections"][str(fold)]["selected_variant"] for fold in range(5)]


def compact_report(report):
    return {
        "accuracy": report["accuracy"],
        "selected_variants": selected_variants(report),
        "selection_frequency": report["selection_frequency"],
        "r7_grid": {
            key: value
            for key, value in report["descriptive_crossfit_grid"].items()
            if key.startswith("R7_")
        },
    }


def main():
    config_path = SOURCEBIAS_DIR / "configs/b2_variants.yaml"
    frozen_path = SOURCEBIAS_DIR / "results/b2_lineage_normalized.json"
    recovery_path = SOURCEBIAS_DIR / "results/recovery_b2_lineage_normalized.json"
    source_path = SOURCEBIAS_DIR / "b2_lineage_normalized.py"
    config = yaml.safe_load(config_path.read_text())
    frozen = json.loads(frozen_path.read_text())
    recovery = json.loads(recovery_path.read_text())

    points = [[0.0, 0.0], [10.0, 0.0]]
    indices = [0, 1]
    weights = [2.0, 1.0]
    legacy_probe = legacy_weighted_centroid(points, indices, weights)
    fixed_probe = centroid(points, indices, weights)
    expected_probe = [10.0 / 3.0, 0.0]
    if legacy_probe != [10.0, 0.0]:
        raise ValueError("D0 legacy weighted-centroid probe mismatch")
    if not all(math.isclose(value, expected, abs_tol=1e-15) for value, expected in zip(fixed_probe, expected_probe)):
        raise ValueError("D0 fixed weighted-centroid normalization mismatch")

    variants = list(config["variant_order"])
    if len(variants) != 21 or variants[-3:] != ["R7_D1", "R7_D2", "R7_D3"]:
        raise ValueError("D0 frozen 21-method order mismatch")
    variants_without_r7 = [variant for variant in variants if not variant.startswith("R7_")]
    contexts, pools = load_pools()
    scale_specs = {
        "7B": (pools["7B_Uniform_Mixed_N12"], split_ids, "Qwen3-VL-8B-Instruct"),
        "72B": (pools["72B_Uniform_Mixed_N8"], split_72, "Qwen3.5-122B-A10B"),
    }
    fixed_21 = {}
    fixed_without_r7 = {}
    for scale, (rows, splitter, best_model) in scale_specs.items():
        reported_best = config["baselines"][scale]["best_single"]["accuracy"]
        fixed_21[scale] = run_scale(scale, contexts[scale], rows, splitter, variants, best_model, reported_best)
        fixed_without_r7[scale] = run_scale(
            scale, contexts[scale], rows, splitter, variants_without_r7, best_model, reported_best
        )

    headline_changed_by_r7_fix = {}
    for scale in scale_specs:
        full = fixed_21[scale]
        without = fixed_without_r7[scale]
        same_predictions = full["outputs"]["nested_LN"] == without["outputs"]["nested_LN"]
        same_selections = selected_variants(full) == selected_variants(without)
        headline_changed_by_r7_fix[scale] = not (same_predictions and same_selections)
        if any(variant.startswith("R7_") for variant in selected_variants(full)):
            raise ValueError(f"D0 repaired R7 unexpectedly selected on {scale}")

    frozen_r7 = {
        scale: {
            key: value
            for key, value in frozen["reports"][scale]["descriptive_crossfit_grid"].items()
            if key.startswith("R7_")
        }
        for scale in ("7B", "72B")
    }
    recovery_r7 = {
        scale: {
            key: value
            for key, value in recovery["reports"][scale]["descriptive_crossfit_grid"].items()
            if key.startswith("R7_")
        }
        for scale in ("7B", "72B")
    }
    if not all(value < 0.03 for values in frozen_r7.values() for value in values.values()):
        raise ValueError("D0 frozen R7 degeneracy signature missing")
    if not all(value > 0.50 for values in recovery_r7.values() for value in values.values()):
        raise ValueError("D0 repaired R7 remains degenerate")

    result = {
        "schema_version": 1,
        "status": "PASS",
        "audit": "D0_R7_IMPLEMENTATION",
        "finding": "IMPLEMENTATION_FAULT_CONFIRMED",
        "root_cause": "weighted centroid returned sum(w_i*x_i) without division by sum(w_i)",
        "analytic_probe": {
            "points": points,
            "weights": weights,
            "legacy_output": legacy_probe,
            "fixed_output": fixed_probe,
            "expected_fixed_output": expected_probe,
        },
        "historical_broken_r7_grid": frozen_r7,
        "recovery_fixed_r7_grid": recovery_r7,
        "fixed_21_method_rerun": {scale: compact_report(report) for scale, report in fixed_21.items()},
        "fixed_18_method_without_r7": {
            scale: compact_report(report) for scale, report in fixed_without_r7.items()
        },
        "headline_changed_by_r7_fix": headline_changed_by_r7_fix,
        "D_K1_triggered": any(headline_changed_by_r7_fix.values()),
        "decision": (
            "REPORT_PRE_AND_POST_FIX_SIDE_BY_SIDE"
            if any(headline_changed_by_r7_fix.values())
            else "REPORT_FAULT_AND_BOTH_RESULTS;_B2_GATE_UNCHANGED"
        ),
        "sources": {
            "implementation": {"path": str(source_path.relative_to(ROOT)), "sha256": sha256_file(source_path)},
            "config": {"path": str(config_path.relative_to(ROOT)), "sha256": sha256_file(config_path)},
            "frozen_b2": {"path": str(frozen_path.relative_to(ROOT)), "sha256": sha256_file(frozen_path)},
            "recovery_b2": {"path": str(recovery_path.relative_to(ROOT)), "sha256": sha256_file(recovery_path)},
        },
        "notes": [
            "The historical broken and recovery combined-24 headline results are not treated as a controlled pair because the method set and recovered bank differ.",
            "The controlled fixed-21 versus fixed-18 comparison isolates whether repaired R7 changes nested selection on the recovered bank.",
        ],
    }
    output = RUN_DIR / "d0_r7_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "finding": result["finding"],
        "headline_changed_by_r7_fix": headline_changed_by_r7_fix,
        "D_K1_triggered": result["D_K1_triggered"],
        "output": str(output.relative_to(ROOT)),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()