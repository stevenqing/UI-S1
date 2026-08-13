import itertools
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(RUN_DIR))

from common import MODELS, VIEWS, evaluate_actions, load_context, write_json


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "q1": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q3": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
        "positive_share": float(np.mean(values > 0)),
        "zero_share": float(np.mean(values == 0)),
    }


def main():
    d1_path = ROOT / "runs/dominance/2026-08-06/d1_dominance_law.json"
    d1 = json.loads(d1_path.read_text())
    pools = d1["screen_spot"]["pools"]
    if len(pools) != 2160:
        raise ValueError("S1 requires all 2,160 D1 pools")
    context = load_context()

    references = {}
    for size in (2, 3):
        candidates = []
        for model in MODELS:
            for views in itertools.combinations(VIEWS, size):
                evaluation = evaluate_actions(context, [(model, view) for view in views])
                candidates.append({
                    "model": model,
                    "views": list(views),
                    "accuracy": evaluation["accuracy"],
                })
        references[str(size)] = {
            metric: max(candidates, key=lambda row: (row["accuracy"][metric], row["model"], row["views"]))
            for metric in ("B3_mvp", "M1_ccm")
        }

    records = []
    for pool in pools:
        size = str(pool["pool_size"])
        record = {
            "pool_id": pool["pool_id"],
            "pool_size": pool["pool_size"],
            "actions": pool["actions"],
            "B3_mvp": pool["B3_mvp"],
            "M1_ccm": pool["M1_ccm"],
        }
        for metric in ("B3_mvp", "M1_ccm"):
            baseline = references[size][metric]["accuracy"][metric]
            record[f"{metric}_same_budget_best_single_lineage"] = baseline
            record[f"{metric}_delta"] = pool[metric] - baseline
        records.append(record)

    reports = {}
    for size in (2, 3):
        rows = [row for row in records if row["pool_size"] == size]
        reports[str(size)] = {"pools": len(rows)}
        for metric in ("B3_mvp", "M1_ccm"):
            delta_key = f"{metric}_delta"
            ordered = sorted(rows, key=lambda row: (row[delta_key], row["pool_id"]))
            reports[str(size)][metric] = {
                "reference": references[str(size)][metric],
                "distribution": summarize([row[delta_key] for row in rows]),
                "minimum_pool": ordered[0],
                "maximum_pool": ordered[-1],
            }

    positive_share_primary = {
        size: reports[str(size)]["B3_mvp"]["distribution"]["positive_share"]
        for size in (2, 3)
    }
    s_k1 = any(value < 0.60 for value in positive_share_primary.values())
    result = {
        "schema_version": 1,
        "status": "PASS",
        "analysis_unit": "2_or_3_forward_action_pool",
        "pool_count": len(records),
        "same_budget_single_lineage_references": references,
        "reports": reports,
        "S_K1": s_k1,
        "claim": (
            "SOME_CROSS_LINEAGE_CONFIGURATIONS_OUTPERFORM_SAME_BUDGET_SINGLE_LINEAGE"
            if s_k1 else "CROSS_LINEAGE_POOLS_USUALLY_OUTPERFORM_SAME_BUDGET_SINGLE_LINEAGE"
        ),
        "reported_N12_configuration_position": {
            "status": "NONEXCHANGEABLE_BUDGET",
            "B3_mvp": 0.6369386464263125,
            "reason": "D1 enumerates 2/3-forward action pools; the reported configuration uses 12 forwards.",
        },
        "pools": records,
        "source": str(d1_path.relative_to(ROOT)),
    }
    output = RUN_DIR / "s1_pool_distribution.json"
    write_json(output, result)

    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    colors = {2: "#287271", 3: "#D97706"}
    for column, metric in enumerate(("B3_mvp", "M1_ccm")):
        for size in (2, 3):
            values = np.asarray([
                row[f"{metric}_delta"] for row in records if row["pool_size"] == size
            ]) * 100
            axes[0, column].hist(values, bins=40, alpha=0.55, density=True, color=colors[size], label=f"{size} lineages")
            ordered = np.sort(values)
            axes[1, column].plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered), color=colors[size], label=f"{size} lineages")
        axes[0, column].axvline(0, color="#333333", linewidth=0.8)
        axes[1, column].axvline(0, color="#333333", linewidth=0.8)
        axes[0, column].set_title(f"{metric} delta histogram")
        axes[1, column].set_title(f"{metric} delta empirical CDF")
        axes[1, column].set_xlabel("Mixed minus best same-budget single-lineage pool (pp)")
        axes[0, column].grid(alpha=0.2)
        axes[1, column].grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    axes[1, 0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(RUN_DIR / "fig_pool_distribution.pdf")
    plt.close(figure)
    print(json.dumps({"S_K1": s_k1, "positive_share": positive_share_primary, "output": str(output.relative_to(ROOT))}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()