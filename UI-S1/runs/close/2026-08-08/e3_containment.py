import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
SCREEN_REGIONS = ROOT / "runs/allocation-law/2026-08-01/raw/shared_regions_n12.jsonl"
SCREEN_CURVES = ROOT / "runs/consolidate/2026-08-06/s4_slope_hardening.json"
MIND_RESULT = ROOT / "runs/xfer/2026-08-07/xf_mind2web.json"


def full_bbox_hit(region, bbox):
    return region[0] <= bbox[0] and region[1] <= bbox[1] and region[2] >= bbox[2] and region[3] >= bbox[3]


def screen_containment():
    rows = [json.loads(line) for line in SCREEN_REGIONS.read_text().splitlines() if line.strip()]
    if len(rows) != 1581 or any(len(row["regions"]) < 12 for row in rows):
        raise ValueError("ScreenSpot E3 region coverage mismatch")
    return [float(np.mean([full_bbox_hit(row["regions"][rank], row["target_bbox"]) for row in rows])) for rank in range(12)]


def main():
    config = yaml.safe_load((RUN_DIR / "configs/e3_mechanism.yaml").read_text())
    if config["status"] != "RESULT_BLIND_BEFORE_E3_COMPUTATION":
        raise ValueError("E3 config is not frozen")
    screen_curve = screen_containment()
    screen = json.loads(SCREEN_CURVES.read_text())
    mind = json.loads(MIND_RESULT.read_text())
    mind_curve = mind["proposer_rank_containment"]["full_bbox_containment_by_rank"][:12]
    screen_budget = {str(budget): screen["curves"]["v_only"][str(budget)]["B3_mvp"] for budget in (4, 8, 12, 16)}
    mind_budget = {str(budget): mind["curves"]["v_only"][str(budget)]["micro_step_sr"] for budget in (4, 8, 12, 16)}
    screen_delta = screen["paired_N16_minus_N4"]["v_only"]["B3_mvp"]
    mind_delta = mind["paired_N16_minus_N4"]["v_only"]
    screen_drop = screen_curve[0] - screen_curve[11]
    mind_drop = mind_curve[0] - mind_curve[11]
    conditions = {
        "screenspot_rank0_containment_gt_mind2web_rank0_containment": screen_curve[0] > mind_curve[0],
        "screenspot_rank0_to_rank11_drop_gt_mind2web_rank0_to_rank11_drop": screen_drop > mind_drop,
        "screenspot_V_only_N16_minus_N4_ci_upper_lt_zero": screen_delta["ci_99"][1] < 0,
        "mind2web_V_only_N16_minus_N4_ci_includes_zero": mind_delta["ci_99"][0] <= 0 <= mind_delta["ci_99"][1],
    }
    supported = all(conditions.values())
    result = {
        "schema_version": 1,
        "status": "PASS",
        "config": "configs/e3_mechanism.yaml",
        "screenspot_pro": {
            "rows": 1581,
            "full_bbox_containment_by_rank": screen_curve,
            "rank0": screen_curve[0],
            "rank11": screen_curve[11],
            "rank0_minus_rank11": screen_drop,
            "v_only_curve": screen_budget,
            "N16_minus_N4": screen_delta,
        },
        "mind2web": {
            "rows": 2080,
            "full_bbox_containment_by_rank": mind_curve,
            "rank0": mind_curve[0],
            "rank11": mind_curve[11],
            "rank0_minus_rank11": mind_drop,
            "v_only_curve": mind_budget,
            "N16_minus_N4": mind_delta,
        },
        "conditions": conditions,
        "mechanism_supported_with_high_start_condition": supported,
        "E_K4": not supported,
        "interpretation": (
            "MECHANISM_SUPPORTED_WITH_HIGH_START_CONDITION"
            if supported else "XF4_UNSUPPORTED_RANK_DECAY_LIMITED_TO_SCREENSPOT"
        ),
        "claim_boundary": "two_benchmark_qualitative_support_not_a_law",
    }
    output = RUN_DIR / "e3_containment_mechanism.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    ranks = np.arange(12)
    budgets = [4, 8, 12, 16]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ranks, np.asarray(screen_curve) * 100, marker="o", label="ScreenSpot-Pro")
    axes[0].plot(ranks, np.asarray(mind_curve) * 100, marker="s", label="Mind2Web")
    axes[0].set_xlabel("Proposal rank")
    axes[0].set_ylabel("Full-bbox containment (%)")
    axes[0].set_xticks(ranks)
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(budgets, [screen_budget[str(value)] * 100 for value in budgets], marker="o", label="ScreenSpot-Pro B3")
    axes[1].plot(budgets, [mind_budget[str(value)] * 100 for value in budgets], marker="s", label="Mind2Web Step SR")
    axes[1].set_xlabel("V-only forward budget")
    axes[1].set_ylabel("Performance (%)")
    axes[1].set_xticks(budgets)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(RUN_DIR / config["plot"]["output"], bbox_inches="tight")
    plt.close(figure)
    print(json.dumps({
        "conditions": conditions,
        "supported": supported,
        "E_K4": not supported,
        "screen_rank0_rank11": [screen_curve[0], screen_curve[11]],
        "mind_rank0_rank11": [mind_curve[0], mind_curve[11]],
        "screen_delta": screen_delta,
        "mind_delta": mind_delta,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
