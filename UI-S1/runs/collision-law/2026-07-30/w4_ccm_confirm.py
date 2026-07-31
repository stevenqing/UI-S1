import argparse
import json
import math
from collections import Counter
from pathlib import Path

from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from ccm import calibration_from_dict, candidate_class, score_candidates
from w4_analyze import load_official_utils, load_setting, score_prediction


SETTINGS = ("low", "high")


def safe_auc(labels, scores):
    finite = [(label, score) for label, score in zip(labels, scores) if math.isfinite(score)]
    if not finite or len({label for label, _ in finite}) < 2:
        return None
    return float(roc_auc_score([label for label, _ in finite], [score for _, score in finite]))


def paired_test(wins, losses):
    total = wins + losses
    return {
        "wins": wins,
        "losses": losses,
        "superiority_p_one_sided": float(binomtest(wins, total, 0.5, alternative="greater").pvalue) if total else 1.0,
        "inferiority_p_one_sided": float(binomtest(losses, total, 0.5, alternative="greater").pvalue) if total else 1.0,
    }


def confirm_setting(setting, rows, frozen, discovery, utils):
    models = frozen["models"]
    best_source = frozen["fixed_best_source"]
    threshold = frozen["oof_threshold"]
    calibration = calibration_from_dict(frozen["final_calibration"])
    if set(calibration.source_priors) != set(models):
        raise ValueError(f"frozen source mismatch: {setting}")
    successes = 0
    baseline_successes = 0
    overrides = 0
    override_successes = 0
    wins = 0
    losses = 0
    gaps = []
    labels = []
    backoffs = Counter()
    selected_classes = Counter()
    selected_sources = Counter()
    for row in rows:
        predictions = [row["predictions"][model] for model in models]
        baseline = row["predictions"][best_source]
        scores, row_backoffs = score_candidates(calibration, predictions, family_dedup=True)
        backoffs.update(row_backoffs)
        if scores:
            winner_position = max(range(len(scores)), key=lambda index: (scores[index][0], -index))
            winner_score, _, winner = scores[winner_position]
            baseline_scores = [
                score for score, _, prediction in scores if prediction.source == best_source
            ]
            baseline_score = baseline_scores[0] if baseline_scores else float("-inf")
            gap = winner_score - baseline_score
        else:
            winner = baseline
            gap = float("-inf")
        use_winner = threshold is not None and gap >= threshold
        selected = winner if use_winner else baseline
        success = bool(score_prediction(row, selected, utils)["step"])
        baseline_success = bool(score_prediction(row, baseline, utils)["step"])
        override = selected != baseline
        successes += int(success)
        baseline_successes += int(baseline_success)
        overrides += int(override)
        override_successes += int(override and success)
        wins += int(success and not baseline_success)
        losses += int(baseline_success and not success)
        gaps.append(gap)
        labels.append(success)
        selected_classes[candidate_class("androidcontrol", selected)] += 1
        selected_sources[selected.source] += 1
    ccm_sr = successes / len(rows)
    baseline_sr = baseline_successes / len(rows)
    discovery_delta = (
        discovery["aggregate_step_sr"]["A5d_risk"]
        - discovery["aggregate_step_sr"]["A0_heldout_best"]
    )
    confirmation_delta = ccm_sr - baseline_sr
    return {
        "rows": len(rows),
        "models": models,
        "excluded_models": sorted(set(rows[0]["predictions"]) - set(models)),
        "fixed_best_source": best_source,
        "oof_threshold": threshold,
        "w4_labels_used_for_calibration": False,
        "best_source_step_sr": baseline_sr,
        "ccm_step_sr": ccm_sr,
        "delta_ccm_minus_best": confirmation_delta,
        "discovery_delta": discovery_delta,
        "direction_preserved": (
            (discovery_delta > 0 and confirmation_delta > 0)
            or (discovery_delta < 0 and confirmation_delta < 0)
            or (discovery_delta == 0 and confirmation_delta == 0)
        ),
        "paired": paired_test(wins, losses),
        "override_rows": overrides,
        "override_rate": overrides / len(rows),
        "override_conditional_step_sr": override_successes / overrides if overrides else None,
        "s_gap_correctness_auroc": safe_auc(labels, gaps),
        "backoff_counts": dict(backoffs),
        "selected_candidate_classes": dict(selected_classes),
        "selected_sources": dict(selected_sources),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads(args.frozen.read_text())
    discovery = json.loads(args.discovery.read_text())
    if frozen.get("w4_labels_read") is not False:
        raise ValueError("confirmation artifact must attest zero W4 label access")
    utils = load_official_utils()
    loaded = {setting: load_setting(setting, utils) for setting in SETTINGS}
    if any(rows is None for rows in loaded.values()):
        result = {"status": "PENDING_INFERENCE", "reason": "requires all ten W4 cells"}
    else:
        result = {
            "status": "PASS",
            "protocol": "AMENDMENT_008_CCM_CONFIRMATION_DEPLOYMENT.md",
            "frozen_calibration": str(args.frozen),
            "settings": {
                setting: confirm_setting(
                    setting,
                    loaded[setting],
                    frozen["settings"][setting],
                    discovery["pools"][f"androidcontrol/{setting}"],
                    utils,
                )
                for setting in SETTINGS
            },
        }
        result["direction_preserved_settings"] = sum(
            value["direction_preserved"] for value in result["settings"].values()
        )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()