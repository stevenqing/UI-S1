import json
import sys
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(RUN_DIR))

from common import SEED, load_context, write_json

SOURCEBIAS_DIR = ROOT / "runs/sourcebias/2026-08-03"
sys.path.insert(0, str(SOURCEBIAS_DIR))
from sourcebias_common import b3_select_index, point_in_bbox


BUDGETS = tuple(range(2, 17))
PERMUTATIONS = 1000


def slope(values):
    x = np.asarray(BUDGETS, dtype=np.float64)
    centered = x - x.mean()
    return float(centered @ np.asarray(values, dtype=np.float64) / (centered @ centered))


def curve(context, order):
    correct = np.zeros(len(BUDGETS), dtype=np.int64)
    for row_id in context["row_ids"]:
        target = context["metadata"][row_id]["target_bbox"]
        candidates = []
        for view in order:
            candidates.append(context["bank"][("GTA1-7B", int(view))][row_id])
            if len(candidates) >= 2:
                selected, _ = b3_select_index(candidates)
                correct[len(candidates) - 2] += int(point_in_bbox(candidates[selected]["point"], target))
    return correct / len(context["row_ids"])


def main():
    context = load_context()
    original_order = np.arange(16, dtype=np.int64)
    original_curve = curve(context, original_order)
    rng = np.random.default_rng(SEED)
    slopes = np.empty(PERMUTATIONS, dtype=np.float64)
    endpoint_deltas = np.empty(PERMUTATIONS, dtype=np.float64)
    mean_curve = np.zeros(len(BUDGETS), dtype=np.float64)
    example_extremes = []
    for index in range(PERMUTATIONS):
        order = rng.permutation(original_order)
        values = curve(context, order)
        slopes[index] = slope(values)
        endpoint_deltas[index] = values[-1] - values[2]
        mean_curve += values
        if index < 5:
            example_extremes.append({"order": order.tolist(), "curve": values.tolist(), "slope": slopes[index]})
        if (index + 1) % 100 == 0:
            print(json.dumps({"permutations": index + 1, "total": PERMUTATIONS}), flush=True)
    mean_curve /= PERMUTATIONS
    original_slope = slope(original_curve)
    original_endpoint = original_curve[-1] - original_curve[2]
    random_still_declines = float(np.mean(slopes < 0)) >= 0.95 and float(np.quantile(slopes, 0.995)) < 0
    attenuation = 1 - abs(float(np.mean(slopes))) / abs(original_slope) if original_slope else None
    if random_still_declines:
        attribution = "FAILURE_CORRELATION_DOMINANT"
    elif float(np.mean(slopes)) >= 0 or (attenuation is not None and attenuation >= 0.5):
        attribution = "RANK_DECAY_DOMINANT"
    else:
        attribution = "BOTH_RANK_DECAY_AND_CORRELATION"
    result = {
        "schema_version": 1,
        "status": "PASS",
        "method": "GTA1 B3 under 1,000 random permutations of views 0-15",
        "budgets": list(BUDGETS),
        "original": {
            "order": original_order.tolist(),
            "curve": original_curve.tolist(),
            "slope": original_slope,
            "N16_minus_N4": original_endpoint,
        },
        "random_order": {
            "permutations": PERMUTATIONS,
            "seed": SEED,
            "mean_curve": mean_curve.tolist(),
            "slope_mean": float(np.mean(slopes)),
            "slope_median": float(np.median(slopes)),
            "slope_ci_99": [float(np.quantile(slopes, 0.005)), float(np.quantile(slopes, 0.995))],
            "negative_slope_share": float(np.mean(slopes < 0)),
            "N16_minus_N4_mean": float(np.mean(endpoint_deltas)),
            "N16_minus_N4_ci_99": [float(np.quantile(endpoint_deltas, 0.005)), float(np.quantile(endpoint_deltas, 0.995))],
        },
        "absolute_slope_attenuation_fraction": attenuation,
        "attribution": attribution,
        "interpretation": (
            "Randomizing rank order removes or strongly attenuates the decline, so proposer rank decay is the primary observed driver."
            if attribution == "RANK_DECAY_DOMINANT"
            else "The randomized curve retains a material decline; failure correlation contributes beyond rank decay."
        ),
        "examples": example_extremes,
    }
    write_json(RUN_DIR / "s5_decline_attribution.json", result)
    print(json.dumps({"attribution": attribution, "original_slope": original_slope, "random": result["random_order"], "attenuation": attenuation}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
