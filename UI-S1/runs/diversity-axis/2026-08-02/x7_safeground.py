import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(RUN_DIR))
sys.path.insert(0, str(RUN_DIR / "x1"))
from x1_sampling_axis import evaluate as evaluate_sampling, original_points
from x3_curve_stats import load_sources, reconstruct
from x7_safeground_port import OFFICIAL_COMMIT, compute_uncertainty


VARIANTS = {
    "official_code": {"patch_size": 28, "activation_threshold": 0.0},
    "paper_v1": {"patch_size": 14, "activation_threshold": 0.3},
}
S_GAP_CORRECTNESS_AUROC = 0.39310293492742826


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(scores, successes):
    failures = [not value for value in successes]
    if len(set(failures)) != 2:
        raise ValueError("X7 AUROC requires both outcome classes")
    failure_auc = float(roc_auc_score(failures, scores))
    correctness_auc = float(roc_auc_score(successes, [-value for value in scores]))
    return {
        "rows": len(scores),
        "failure_auroc_uncertainty": failure_auc,
        "correctness_auroc_negative_uncertainty": correctness_auc,
        "mean_uncertainty": float(np.mean(scores)),
        "std_uncertainty": float(np.std(scores)),
        "delta_vs_cross_task_s_gap_correctness_auroc": correctness_auc - S_GAP_CORRECTNESS_AUROC,
    }


def score_rows(rows, outputs):
    reports = {variant: {method: None for method in outputs} for variant in VARIANTS}
    variant_scores = {variant: [] for variant in VARIANTS}
    for row in rows:
        width, height = row["image_size"]
        points = [candidate["point"] for candidate in row["candidates"]]
        for variant, parameters in VARIANTS.items():
            variant_scores[variant].append(compute_uncertainty(points, width, height, **parameters)["combined"])
    for variant in VARIANTS:
        for method, labels in outputs.items():
            reports[variant][method] = summarize(variant_scores[variant], [bool(labels[row["id"]]) for row in rows])
    return reports


def deterministic_pools():
    gta1, generated, units = load_sources()
    rows_by_pool, evaluations = reconstruct(gta1, generated, units)
    output = {}
    for pool in ("v_only", "mixed"):
        rows = rows_by_pool[pool][12]
        for row in rows:
            row["image_size"] = gta1[row["id"]]["img_size"]
        output[f"{pool}_N12"] = {
            "source_kind": "DETERMINISTIC_CANDIDATE_SET_NOT_STOCHASTIC_SAMPLES",
            "candidates": 12,
            "primary_label": "M1_ccm",
            "variants": score_rows(rows, {
                "M1_ccm": evaluations[pool][12]["outputs"]["M1_ccm"],
                "B3_mvp": evaluations[pool][12]["outputs"]["B3_mvp"],
            }),
        }
    return output


def stochastic_pool(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 1581 or any(row["samples"] != 5 for row in rows):
        raise ValueError("X7 stochastic source coverage mismatch")
    evaluation = evaluate_sampling(rows, 4)
    normalized = []
    for row in rows:
        normalized.append({
            "id": row["id"],
            "image_size": row["img_size"],
            "candidates": [{"point": point} for point in original_points(row, 4) if point is not None],
        })
    return {
        "source_kind": "STOCHASTIC_SAMPLES_PROTOCOL_MISMATCH_K4_T0.7",
        "source_sha256": sha256_file(path),
        "candidates": 4,
        "primary_label": "GUI_RC",
        "variants": score_rows(normalized, evaluation["outputs"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pools = {"stochastic_GTA1_N4": stochastic_pool(args.sampling), **deterministic_pools()}
    result = {
        "schema_version": 1,
        "status": "PASS_MIGRATION_DIAGNOSTIC_NO_FDR_GUARANTEE",
        "official_source": {
            "repository": "UCSB-AI/SAFEGROUND",
            "commit": OFFICIAL_COMMIT,
            "official_code": VARIANTS["official_code"],
            "paper_v1": VARIANTS["paper_v1"],
            "weights": {"margin": 0.2, "entropy": 0.2, "concentration": 0.6},
        },
        "official_protocol": {"samples": 10, "temperature": 1.0},
        "cross_task_s_gap_correctness_auroc": S_GAP_CORRECTNESS_AUROC,
        "pools": pools,
        "claim_boundary": "N12 pools are deterministic views/lineages, so scores are candidate-dispersion transfers rather than SafeGround stochastic uncertainty with FDR control.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: value["variants"] for name, value in pools.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()