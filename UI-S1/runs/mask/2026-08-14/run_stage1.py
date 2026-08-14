import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

from mask_common import (
    b3_correct, empty_mask, failure_matrix, generalized_neff,
    informative_mask_pixels, load_rows, mode_center, ranked_modes,
    source_reliability,
)


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CONFIG_PATH = RUN_DIR / "configs/mask_prereg.yaml"
PREFLIGHT_PATH = RUN_DIR / "PREFLIGHT.json"
SPLIT_GATE_PATH = ROOT / "runs/split/2026-08-14/ZERO_GPU_GATE.json"
OUTPUT_PATH = RUN_DIR / "STAGE1.json"
FIGURE_PATH = RUN_DIR / "VERIFIER_CONTOURS.pdf"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_contract():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    if config.get("status") != "FROZEN_BEFORE_ANY_MASK_RESULT_OR_FORWARD":
        raise PermissionError("MASK prereg status mismatch")
    if sha256_file(ROOT / config["canonical_spec"]["path"]) != config["canonical_spec"]["sha256"]:
        raise PermissionError("MASK spec hash mismatch")
    if (
        preflight.get("status") != "PASS_MASK_PREFLIGHT_GTA1_READY_MODEL_ROLES_RESOLVED"
        or preflight.get("gpu_used") is not False
        or preflight.get("mask_statistics_computed") is not False
        or preflight.get("model_forward_started") is not False
    ):
        raise PermissionError("MASK preflight boundary mismatch")
    return config


def verifier_curve(config):
    split = json.loads(SPLIT_GATE_PATH.read_text())
    if split.get("status") != "PASS_Z_G1_PROCEED_TO_GEOMETRY":
        raise PermissionError("MASK requires locked SPLIT gate rows")
    rows = list(split["heldout_rows"].values())
    auc_grid = config["stage1"]["verifier_curve"]["auc_grid"]
    threshold_spec = config["stage1"]["verifier_curve"]["threshold_grid"]
    thresholds = np.arange(
        threshold_spec["start"],
        threshold_spec["stop"] + threshold_spec["step"] / 2,
        threshold_spec["step"],
    )
    curves = []
    for gate_value in config["stage1"]["verifier_curve"]["g_grid"]:
        gated = [
            row for row in rows
            if row["w2_over_w1"] is not None and row["w2_over_w1"] >= gate_value
        ]
        positives = sum(row["M2_correct"] and not row["M1_correct"] for row in gated)
        negatives = len(gated) - positives
        auc_results = []
        for auc in auc_grid:
            d_prime = math.sqrt(2) * float(norm.ppf(auc))
            tpr = norm.sf(thresholds - d_prime)
            fpr = norm.sf(thresholds)
            gains = (positives * tpr - negatives * fpr) / len(rows)
            best_gain = float(np.max(gains))
            best_indices = np.flatnonzero(np.isclose(gains, best_gain, rtol=0, atol=1e-15))
            best = int(best_indices[-1])
            auc_results.append({
                "AUROC": float(auc),
                "d_prime": d_prime,
                "best_threshold": float(thresholds[best]),
                "TPR": float(tpr[best]),
                "FPR": float(fpr[best]),
                "net_gain": best_gain,
            })
        curves.append({
            "g": float(gate_value),
            "gate_rows": len(gated),
            "positives": positives,
            "negatives": negatives,
            "positive_rate": float(positives / len(gated)) if gated else 0.0,
            "negative_positive_ratio": float(negatives / positives) if positives else None,
            "hypothetical_discrimination": auc_results,
        })
    return {"rows": len(rows), "curves": curves}


def plot_verifier(values):
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for curve in values["curves"]:
        axis.plot(
            [row["AUROC"] for row in curve["hypothetical_discrimination"]],
            [100 * row["net_gain"] for row in curve["hypothetical_discrimination"]],
            marker="o",
            linewidth=1.4,
            markersize=3,
            label=f"g={curve['g']:.2f}, n={curve['gate_rows']}",
        )
    axis.axhline(0.7, color="black", linestyle="--", linewidth=1, label="MDE 0.70 pp")
    axis.axhline(0, color="gray", linewidth=0.8)
    axis.set_xlabel("Hypothetical AUROC")
    axis.set_ylabel("Best full-set net gain (pp)")
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(FIGURE_PATH)
    plt.close(figure)


def subset_indices(mask):
    return tuple(index for index in range(12) if mask & (1 << index))


def isotonic_report(x_values, y_values, baseline_x, target_x):
    grouped = defaultdict(list)
    for x_value, y_value in zip(x_values, y_values):
        grouped[float(x_value)].append(float(y_value))
    unique_x = np.asarray(sorted(grouped), dtype=np.float64)
    unique_y = np.asarray([np.mean(grouped[x_value]) for x_value in unique_x], dtype=np.float64)
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    model.fit(unique_x, unique_y)
    baseline_prediction, target_prediction = model.predict([baseline_x, target_x])
    return {
        "unique_x": len(unique_x),
        "x_min": float(unique_x.min()),
        "x_max": float(unique_x.max()),
        "baseline_x": float(baseline_x),
        "ideal_target_x": float(target_x),
        "baseline_prediction": float(baseline_prediction),
        "ideal_target_prediction": float(target_prediction),
        "predicted_gain": float(target_prediction - baseline_prediction),
        "thresholds_x": model.X_thresholds_.tolist(),
        "thresholds_y": model.y_thresholds_.tolist(),
    }


def neff_calibration(rows, config):
    row_ids = tuple(sorted(rows))
    masks = tuple(range(1, 1 << 12))
    fold_sizes = np.zeros(5, dtype=np.int64)
    neff_by_fold = np.full((5, len(masks)), np.nan, dtype=np.float64)
    b3_correct_by_fold = np.zeros((5, len(masks)), dtype=np.int64)
    majority_correct_by_fold = np.zeros((5, len(masks)), dtype=np.int64)
    undefined_by_fold = np.zeros((5, len(masks)), dtype=np.int64)
    for fold in range(5):
        development = [row_id for row_id in row_ids if rows[row_id]["fold"] != fold]
        test = [row_id for row_id in row_ids if rows[row_id]["fold"] == fold]
        fold_sizes[fold] = len(test)
        reliability = source_reliability(rows, development)
        source_order = [candidate.source for candidate in next(iter(rows.values()))["gran_candidates"]]
        for offset, mask in enumerate(masks):
            indices = subset_indices(mask)
            matrix, undefined = failure_matrix(rows, development, indices)
            neff_by_fold[fold, offset] = generalized_neff(matrix)
            undefined_by_fold[fold, offset] = undefined
            selected_source = min(indices, key=lambda index: (-reliability[source_order[index]], index))
            majority_correct_by_fold[fold, offset] = sum(
                rows[row_id]["candidates"][selected_source]["correct"] for row_id in test
            )
            b3_correct_by_fold[fold, offset] = sum(
                b3_correct(
                    [rows[row_id]["candidates"][index] for index in indices],
                    rows[row_id]["target_bbox"],
                )
                for row_id in test
            )
    valid = np.all(np.isfinite(neff_by_fold), axis=0)
    weights = fold_sizes / fold_sizes.sum()
    aggregate_neff = np.sum(neff_by_fold * weights[:, None], axis=0)
    aggregate_b3 = np.sum(b3_correct_by_fold, axis=0) / fold_sizes.sum()
    aggregate_majority = np.sum(majority_correct_by_fold, axis=0) / fold_sizes.sum()
    full_offset = masks.index((1 << 12) - 1)
    baseline_x = float(aggregate_neff[full_offset])
    ideal_increment = float(config["stage1"]["neff_calibration"]["ideal_x_increment"])
    target_x = baseline_x + ideal_increment
    reports = {
        "density_B3": isotonic_report(
            aggregate_neff[valid], aggregate_b3[valid], baseline_x, target_x
        ),
        "F1_majority": isotonic_report(
            aggregate_neff[valid], aggregate_majority[valid], baseline_x, target_x
        ),
    }
    sensitivity = {name: [] for name in reports}
    for omitted_fold in range(5):
        keep = np.asarray([fold != omitted_fold for fold in range(5)])
        kept_sizes = fold_sizes[keep]
        kept_weights = kept_sizes / kept_sizes.sum()
        current_neff = np.sum(neff_by_fold[keep] * kept_weights[:, None], axis=0)
        current_valid = np.all(np.isfinite(neff_by_fold[keep]), axis=0)
        current_baseline = float(current_neff[full_offset])
        current_target = current_baseline + ideal_increment
        for name, correct in (
            ("density_B3", b3_correct_by_fold),
            ("F1_majority", majority_correct_by_fold),
        ):
            current_accuracy = np.sum(correct[keep], axis=0) / kept_sizes.sum()
            report = isotonic_report(
                current_neff[current_valid], current_accuracy[current_valid],
                current_baseline, current_target,
            )
            sensitivity[name].append({
                "omitted_fold": omitted_fold,
                "predicted_gain": report["predicted_gain"],
                "baseline_x": current_baseline,
                "ideal_target_x": current_target,
            })
    records = []
    for offset, mask in enumerate(masks):
        records.append({
            "mask": mask,
            "indices": list(subset_indices(mask)),
            "K": int(mask.bit_count()),
            "valid": bool(valid[offset]),
            "cross_fitted_neff": float(aggregate_neff[offset]) if valid[offset] else None,
            "density_B3_accuracy": float(aggregate_b3[offset]),
            "F1_majority_accuracy": float(aggregate_majority[offset]),
            "undefined_pair_kappa_by_fold": undefined_by_fold[:, offset].tolist(),
        })
    return {
        "subsets": len(masks),
        "valid_subsets": int(valid.sum()),
        "fold_sizes": fold_sizes.tolist(),
        "full_pool_mask": (1 << 12) - 1,
        "full_pool_cross_fitted_neff": baseline_x,
        "ideal_neff_increment": ideal_increment,
        "aggregators": reports,
        "leave_one_fold_out_sensitivity": sensitivity,
        "records": records,
    }


def base_rates_and_masks(rows, config):
    tau_by_fold = {
        int(fold): float(value) for fold, value in config["tau"]["selected_by_outer_fold"].items()
    }
    row_ids = tuple(sorted(rows))
    outputs = {}
    for fold in range(5):
        development = [row_id for row_id in row_ids if rows[row_id]["fold"] != fold]
        test = [row_id for row_id in row_ids if rows[row_id]["fold"] == fold]
        reliability = source_reliability(rows, development)
        tau = tau_by_fold[fold]
        for row_id in test:
            row = rows[row_id]
            modes = ranked_modes(row["gran_candidates"], reliability, tau)
            if not modes:
                raise ValueError(f"MASK no C-uni mode: {row_id}")
            centers = [mode_center(row["candidates"], mode["members"]) for mode in modes]
            width, height = row["image_size"]
            radius = 2 * tau * math.hypot(width, height)
            information_pixels = informative_mask_pixels(
                width, height, centers[0], radius
            )
            control = empty_mask(
                width, height, len(information_pixels), centers[0], centers
            )
            pool_correct = b3_correct(row["candidates"], row["target_bbox"])
            outputs[row_id] = {
                "fold": fold,
                "application": row["application"],
                "pool_correct": pool_correct,
                "M1_correct": bool(modes[0]["correct"]),
                "M1_representative_correct": bool(modes[0]["representative_correct"]),
                "M2_exists": len(modes) >= 2,
                "M2_correct": bool(modes[1]["correct"]) if len(modes) >= 2 else False,
                "M1_members": modes[0]["members"],
                "M1_center": list(centers[0]),
                "mode_centers": [list(center) for center in centers],
                "tau_normalized": tau,
                "radius_pixels": radius,
                "information_mask_pixels": len(information_pixels),
                "empty_mask_feasible": control is not None,
                "empty_mask_angle_degrees": control["angle_degrees"] if control else None,
                "empty_mask_center": list(control["center"]) if control else None,
            }
    pool_wrong = [row for row in outputs.values() if not row["pool_correct"]]
    pool_correct = [row for row in outputs.values() if row["pool_correct"]]
    infeasible = sum(not row["empty_mask_feasible"] for row in outputs.values())
    return {
        "rows": len(outputs),
        "density_B3_accuracy": float(len(pool_correct) / len(outputs)),
        "density_B3_error_rate": float(len(pool_wrong) / len(outputs)),
        "pool_wrong_rows": len(pool_wrong),
        "pool_correct_rows": len(pool_correct),
        "M2_correct_within_pool_wrong_rows": sum(row["M2_correct"] for row in pool_wrong),
        "M2_correct_rate_within_pool_wrong": float(np.mean([row["M2_correct"] for row in pool_wrong])),
        "single_mode_rows": sum(not row["M2_exists"] for row in outputs.values()),
        "empty_mask_infeasible_rows": infeasible,
        "empty_mask_infeasible_rate": float(infeasible / len(outputs)),
        "rows_detail": outputs,
    }


def main():
    if OUTPUT_PATH.exists() or FIGURE_PATH.exists():
        raise FileExistsError("MASK stage1 outputs already exist")
    config = load_contract()
    verifier = verifier_curve(config)
    plot_verifier(verifier)
    rows = load_rows()
    calibration = neff_calibration(rows, config)
    base_rates = base_rates_and_masks(rows, config)
    gains = {
        name: report["predicted_gain"]
        for name, report in calibration["aggregators"].items()
    }
    maximum_gain = max(gains.values())
    mde = float(config["stage1"]["M_G1"]["mde"])
    mask_control_pass = (
        base_rates["empty_mask_infeasible_rate"]
        <= float(config["empty_mask"]["maximum_infeasible_rate"])
    )
    result = {
        "schema_version": 1,
        "status": (
            "PASS_M_G1_PROCEED_TO_SUBSET"
            if maximum_gain >= mde and mask_control_pass
            else "STOP_M_K8_EMPTY_MASK_CONTROL"
            if not mask_control_pass
            else "STOP_M_K1_BEFORE_GPU"
        ),
        "zero_gpu": True,
        "model_forward_started": False,
        "subset_manifest_created": False,
        "gpu_authorization_created": False,
        "verifier_curve": verifier,
        "neff_calibration": calibration,
        "base_rates_and_masks": base_rates,
        "M_G1": {
            "predicted_gain_by_aggregator": gains,
            "maximum_ideal_predicted_gain": maximum_gain,
            "mde": mde,
            "pass": maximum_gain >= mde,
        },
        "mask_control_gate": {
            "maximum_infeasible_rate": float(config["empty_mask"]["maximum_infeasible_rate"]),
            "pass": mask_control_pass,
        },
        "kill_conditions": {
            "M_K1": maximum_gain < mde,
            "M_K8_empty_mask": not mask_control_pass,
        },
        "claim_boundary": {
            "exploratory_only": True,
            "universal_neff_law_claim": False,
            "gpu_authorized": False,
        },
        "input_hashes": {
            "PREFLIGHT.json": sha256_file(PREFLIGHT_PATH),
            "ZERO_GPU_GATE.json": sha256_file(SPLIT_GATE_PATH),
        },
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "M_G1": result["M_G1"],
        "mask_control_gate": result["mask_control_gate"],
        "base_rates": {
            key: value for key, value in base_rates.items() if key != "rows_detail"
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()