import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(RUN_DIR))

from common import SEED, write_json


RESAMPLES = 10000


def weighted_corr(left, right, weights):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.sum() <= 0:
        return float("nan")
    weights /= weights.sum()
    left_mean = float(weights @ left)
    right_mean = float(weights @ right)
    left_centered = left - left_mean
    right_centered = right - right_mean
    denominator = np.sqrt(float(weights @ (left_centered**2)) * float(weights @ (right_centered**2)))
    return float(weights @ (left_centered * right_centered) / denominator) if denominator > 0 else float("nan")


def weighted_residual(values, controls, weights):
    design = np.column_stack([np.ones(len(values)), *controls])
    sqrt_weights = np.sqrt(weights)[:, None]
    coefficients = np.linalg.lstsq(design * sqrt_weights, values * sqrt_weights[:, 0], rcond=None)[0]
    return values - design @ coefficients


def action_cluster_bootstrap(pools, metric):
    actions = sorted({tuple(action) for pool in pools for action in pool["actions"]})
    action_index = {action: index for index, action in enumerate(actions)}
    pool_actions = [[action_index[tuple(action)] for action in pool["actions"]] for pool in pools]
    gap = rankdata(np.asarray([pool["dominance_gap"] for pool in pools], dtype=np.float64))
    delta = rankdata(np.asarray([pool[metric] for pool in pools], dtype=np.float64))
    quality = rankdata(np.asarray([pool["mean_member_accuracy"] for pool in pools], dtype=np.float64))
    kappa = rankdata(np.asarray([pool["mean_pairwise_failure_kappa"] for pool in pools], dtype=np.float64))
    rng = np.random.default_rng(SEED)
    raw = []
    partial = []
    for _ in range(RESAMPLES):
        counts = np.bincount(rng.integers(0, len(actions), size=len(actions)), minlength=len(actions))
        weights = np.asarray([np.prod([counts[index] for index in indices]) for indices in pool_actions], dtype=np.float64)
        active = weights > 0
        active_weights = weights[active]
        raw.append(weighted_corr(gap[active], delta[active], active_weights))
        gap_residual = weighted_residual(gap[active], (quality[active], kappa[active]), active_weights)
        delta_residual = weighted_residual(delta[active], (quality[active], kappa[active]), active_weights)
        partial.append(weighted_corr(gap_residual, delta_residual, active_weights))
    raw = np.asarray([value for value in raw if np.isfinite(value)])
    partial = np.asarray([value for value in partial if np.isfinite(value)])
    return {
        "raw_rho_distribution": {
            "mean": float(np.mean(raw)),
            "median": float(np.median(raw)),
            "ci_99": [float(np.quantile(raw, 0.005)), float(np.quantile(raw, 0.995))],
            "negative_share": float(np.mean(raw < 0)),
        },
        "partial_rho_distribution": {
            "mean": float(np.mean(partial)),
            "median": float(np.median(partial)),
            "ci_99": [float(np.quantile(partial, 0.005)), float(np.quantile(partial, 0.995))],
            "negative_share": float(np.mean(partial < 0)),
        },
        "resamples": RESAMPLES,
        "seed": SEED,
        "cluster_unit": "action",
        "actions": len(actions),
        "weighting": "product of sampled action multiplicities for each pool",
    }


def main():
    d1_path = ROOT / "runs/dominance/2026-08-06/d1_dominance_law.json"
    s0_path = ROOT / "runs/final/2026-08-04/s0_safeground_anchor.json"
    d1 = json.loads(d1_path.read_text())
    s0 = json.loads(s0_path.read_text())
    pools = d1["screen_spot"]["pools"]
    cluster = {
        metric: action_cluster_bootstrap(pools, metric)
        for metric in ("B3_minus_best", "M1_minus_best")
    }
    result = {
        "schema_version": 1,
        "status": "PASS",
        "safeground_anchor": {
            "status": s0["status"],
            "official_value": s0["official_anchor"]["value"],
            "local_value": s0["local_anchor_attempt"]["value"],
            "delta": s0["local_anchor_attempt"]["delta_from_official"],
            "protocol_match": s0["local_anchor_attempt"]["protocol_match"],
            "interpretation": s0["local_anchor_attempt"]["interpretation"],
            "writing": "ALGORITHM_LEVEL_PORT_ONLY",
        },
        "action_cluster_bootstrap": cluster,
        "D1_positioning": {
            "direction_robust_B3": cluster["B3_minus_best"]["partial_rho_distribution"]["ci_99"][1] < 0,
            "direction_robust_M1": cluster["M1_minus_best"]["partial_rho_distribution"]["ci_99"][1] < 0,
            "main_text": "MECHANISM_EVIDENCE_NOT_LAW",
            "variance_explained_proxy_M1_partial_rho_squared": d1["screen_spot"]["statistics"]["M1_minus_best"]["partial_spearman_controlling_mean_quality_and_failure_kappa"]["rho"] ** 2,
            "boundary": "Correlation explains rank association, not causal variance decomposition.",
        },
        "sources": {
            "D1": str(d1_path.relative_to(ROOT)),
            "SafeGround": str(s0_path.relative_to(ROOT)),
        },
    }
    write_json(RUN_DIR / "s6_anchors.json", result)
    print(json.dumps({"SafeGround": result["safeground_anchor"], "cluster": cluster, "positioning": result["D1_positioning"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()