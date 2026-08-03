import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from reallocation_common import MIXED_BUDGETS, load_pools, ordered_bins, sha256_file, subset_bootstrap, uncertainty_scores


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
MDE = 0.007043345177520599


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--figure", type=Path, required=True); args = parser.parse_args()
    context = load_pools(); rows12 = context["mixed"][12]; image_sizes = {row_id: row["img_size"] for row_id, row in context["gta1"].items()}
    scores = uncertainty_scores(rows12, image_sizes); bins = ordered_bins(rows12, scores, 5)
    records = []
    for index, ids in enumerate(bins):
        record = {"bin": index, "label": "highest_disagreement" if index == 4 else "ordered_uncertainty_quintile", "rows": len(ids), "uncertainty_min": min(scores[row_id] for row_id in ids), "uncertainty_max": max(scores[row_id] for row_id in ids), "accuracy": {}}
        for budget in MIXED_BUDGETS:
            evaluation = context["evaluations"]["mixed"][budget]
            record["accuracy"][str(budget)] = {rule: sum(evaluation["outputs"][rule][row_id] for row_id in ids) / len(ids) for rule in ("B3_mvp", "M1_ccm", "pass_at_n")}
        records.append(record)
    high_ids = bins[-1]
    bootstraps = {}
    for rule in ("B3_mvp", "M1_ccm", "pass_at_n"):
        bootstraps[rule] = subset_bootstrap(rows12, high_ids, context["evaluations"]["mixed"][24]["outputs"][rule], context["evaluations"]["mixed"][4]["outputs"][rule], context["fold_for_group"])
    pass_delta = bootstraps["pass_at_n"]["point_delta"]
    b3_delta = bootstraps["B3_mvp"]["point_delta"]
    gate_pass = b3_delta > MDE and bootstraps["B3_mvp"]["ci_99"][0] > 0
    result = {
        "schema_version": 1, "status": "PASS", "rows": 1581, "budgets": list(MIXED_BUDGETS),
        "score": {"method": "SafeGround_official_code_transfer", "patch_size": 28, "activation_threshold": 0.0},
        "bins": records, "highest_disagreement_N24_minus_N4": bootstraps,
        "realization_ratio_B3_over_pass": b3_delta / pass_delta if pass_delta > 0 else None,
        "gate": {"MDE": MDE, "delta_above_MDE": b3_delta > MDE, "ci_99_lower_positive": bootstraps["B3_mvp"]["ci_99"][0] > 0, "R1_pass": gate_pass, "R_K1": not gate_pass, "R2_R3_action": "RUN" if gate_pass else "CANCEL"},
        "sources": {"strata_config_sha256": sha256_file(RUN_DIR / "configs/strata.yaml"), "L1_sha256": sha256_file(ROOT / "runs/allocation-law/2026-08-01/L1_RESULTS.json")},
    }
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for axis, rule, title in zip(axes, ("B3_mvp", "M1_ccm", "pass_at_n"), ("B3 accuracy", "M1 accuracy", "Candidate pass@N")):
        for record in records:
            axis.plot(MIXED_BUDGETS, [100 * record["accuracy"][str(budget)][rule] for budget in MIXED_BUDGETS], marker="o", label=f"Q{record['bin']+1}")
        axis.set_xlabel("Forwards N"); axis.set_ylabel("Percent"); axis.set_title(title); axis.grid(alpha=.2)
    axes[0].legend(ncol=2); figure.tight_layout(); args.figure.parent.mkdir(parents=True, exist_ok=True); figure.savefig(args.figure); plt.close(figure)
    result["figure"] = str(args.figure.resolve().relative_to(ROOT)); args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"highest_bin": records[-1], "bootstrap": bootstraps, "realization_ratio": result["realization_ratio_B3_over_pass"], "gate": result["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
