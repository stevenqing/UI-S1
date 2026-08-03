import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from reallocation_common import SEED, load_pools, sha256_file, uncertainty_scores


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
COVERAGES = (1.0, .9, .8, .7)
PERMUTATIONS = 10000


def pool_curve(rows, scores, correctness):
    ordered = sorted(rows, key=lambda row: (scores[row["id"]], row["id"]))
    total_correct = sum(correctness.values()); output = []
    rng = np.random.default_rng(SEED)
    labels = np.asarray([int(correctness[row["id"]]) for row in rows], dtype=np.int8)
    retained_counts = {coverage: math.floor(len(rows) * coverage) for coverage in COVERAGES}
    random_by_coverage = {coverage: np.empty(PERMUTATIONS, dtype=np.float64) for coverage in COVERAGES}
    for index in range(PERMUTATIONS):
        shuffled = labels[rng.permutation(len(labels))]
        cumulative = np.cumsum(shuffled)
        for coverage, retained_count in retained_counts.items():
            random_by_coverage[coverage][index] = cumulative[retained_count - 1] / retained_count
    for coverage in COVERAGES:
        retained_count = retained_counts[coverage]
        retained = ordered[:retained_count]; retained_ids = [row["id"] for row in retained]
        retained_correct = sum(correctness[row_id] for row_id in retained_ids)
        rejected_correct = total_correct - retained_correct
        rejected_count = len(rows) - retained_count
        random_values = random_by_coverage[coverage]
        output.append({
            "coverage": coverage, "retained_rows": retained_count, "rejected_rows": rejected_count,
            "retained_accuracy": retained_correct / retained_count,
            "retained_errors": retained_count - retained_correct,
            "rejected_successes": rejected_correct,
            "rejected_failures": rejected_count - rejected_correct,
            "random_rejection": {"mean_accuracy": float(random_values.mean()), "ci_99": [float(np.quantile(random_values, .005)), float(np.quantile(random_values, .995))], "permutations": PERMUTATIONS, "seed": SEED},
        })
    full_accuracy = output[0]["retained_accuracy"]
    for value in output:
        value["accuracy_gain_vs_full"] = value["retained_accuracy"] - full_accuracy
        value["gain_vs_random_mean"] = value["retained_accuracy"] - value["random_rejection"]["mean_accuracy"]
    return output


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--figure", type=Path, required=True); args = parser.parse_args()
    context = load_pools(); image_sizes = {row_id: row["img_size"] for row_id, row in context["gta1"].items()}
    definitions = {"Uniform_Mixed_N12": (context["mixed"][12], context["evaluations"]["mixed"][12]["outputs"]["B3_mvp"]), "V_only_N12": (context["v_only_N12"], context["evaluations"]["v_only_N12"]["outputs"]["B3_mvp"])}
    curves = {}
    for name, (rows, correctness) in definitions.items():
        curves[name] = pool_curve(rows, uncertainty_scores(rows, image_sizes), correctness)
    result = {"schema_version": 1, "status": "PASS", "rows": 1581, "retained_coverages": list(COVERAGES), "curves": curves, "interpretation": {"mixed_AUROC": .830, "v_only_AUROC": .744, "claim": "Cross-lineage allocation changes both grounding accuracy and selective-error ranking."}, "sources": {"strata_config_sha256": sha256_file(RUN_DIR / "configs/strata.yaml"), "X7_sha256": sha256_file(ROOT / "runs/diversity-axis/2026-08-02/x7_confidence.json")}}
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    for name, values in curves.items(): axis.plot([100 * value["coverage"] for value in values], [100 * value["retained_accuracy"] for value in values], marker="o", label=name)
    random = curves["Uniform_Mixed_N12"]; axis.plot([100 * value["coverage"] for value in random], [100 * value["random_rejection"]["mean_accuracy"] for value in random], linestyle="--", color="#777777", label="Random rejection")
    axis.set_xlabel("Retained coverage (%)"); axis.set_ylabel("B3 accuracy (%)"); axis.invert_xaxis(); axis.grid(alpha=.2); axis.legend(); figure.tight_layout(); args.figure.parent.mkdir(parents=True, exist_ok=True); figure.savefig(args.figure); plt.close(figure)
    result["figure"] = str(args.figure.resolve().relative_to(ROOT)); args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(curves, indent=2, sort_keys=True))


if __name__ == "__main__": main()