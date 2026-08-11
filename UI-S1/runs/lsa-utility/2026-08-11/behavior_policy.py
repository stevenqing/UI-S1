import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CEV_ROOT = ROOT / "runs/cev/2026-08-09"
XFER = ROOT / "runs/xfer/2026-08-07"
sys.path.insert(0, str(CEV_ROOT))

from cev import Candidate as CEVCandidate
from cev import select as cev_select
from cev_main import config_rank, configuration_key, configurations


@dataclass(frozen=True)
class Policy:
    benchmark: str
    arm: str
    configuration: dict
    reliability: dict[str, float]
    scale: tuple[float, float] | None
    fit_folds: tuple[int, ...]
    config_validation_fold: int | None


def source_reliability(rows, row_ids):
    sums = {}
    counts = {}
    for row_id in row_ids:
        for candidate in rows[row_id].candidates:
            sums[candidate.source] = sums.get(candidate.source, 0.0) + float(candidate.success)
            counts[candidate.source] = counts.get(candidate.source, 0) + 1
    return {source: sums[source] / counts[source] for source in sums}


def mind_scales():
    rows = [json.loads(line) for line in (XFER / "data/mind2web/mind2web_test_task.jsonl").read_text().splitlines() if line.strip()]
    output = {}
    for row in rows:
        width, height = Image.open(ROOT / row["image"]).size
        bbox = row["step"]["bbox"]
        output[row["id"]] = (bbox["width"] / width, bbox["height"] / height)
    return output


MIND_SCALES = None


def fit_scale(row_ids):
    global MIND_SCALES
    if MIND_SCALES is None:
        MIND_SCALES = mind_scales()
    return (
        float(np.median([MIND_SCALES[row_id][0] for row_id in row_ids])),
        float(np.median([MIND_SCALES[row_id][1] for row_id in row_ids])),
    )


def policy_candidate(candidate, reliability):
    return CEVCandidate(
        action=candidate.action,
        coordinate=candidate.baseline_coordinate,
        parameter=candidate.parameter,
        source=candidate.source,
        reliability=reliability[candidate.source],
        order=candidate.order,
        payload=candidate.order,
        parse_ok=candidate.parse_ok,
        lineage=candidate.lineage,
    )


def apply_policy(row, policy):
    candidates = [policy_candidate(candidate, policy.reliability) for candidate in row.candidates]
    configuration = policy.configuration
    if policy.benchmark == "screenspot_pro":
        threshold = configuration.get("coordinate_tolerance", 14.0)
    else:
        multiplier = configuration.get("coordinate_multiplier", 1.0)
        threshold = (policy.scale[0] * multiplier, policy.scale[1] * multiplier)
    prediction, _ = cev_select(
        candidates,
        configuration["granularity"],
        threshold,
        configuration.get("parameter_threshold", 1.0),
    )
    return int(prediction.payload)


def choose_config(rows, train_ids, validation_ids, cev_config):
    reliability = source_reliability(rows, train_ids)
    scale = fit_scale(train_ids)
    options = configurations(cev_config)
    scores = {}
    by_key = {}
    for option in options:
        key = configuration_key(option)
        by_key[key] = option
        policy = Policy("mind2web", "", option, reliability, scale, (), None)
        scores[key] = float(np.mean([
            rows[row_id].candidates[apply_policy(rows[row_id], policy)].success
            for row_id in validation_ids
        ]))
    selected_key = min(scores, key=lambda key: (-scores[key], config_rank(by_key[key], cev_config)))
    return by_key[selected_key], scores[selected_key]


def cyclic_validation_fold(holdout_fold, fit_folds):
    for offset in range(1, 6):
        candidate = (holdout_fold + offset) % 5
        if candidate in fit_folds:
            return candidate
    raise ValueError("no CEV configuration validation fold")


def fit_inner_policies(banks, fit_folds, holdout_fold, cev_config):
    config_validation_fold = cyclic_validation_fold(holdout_fold, fit_folds)
    config_train_folds = [fold for fold in fit_folds if fold != config_validation_fold]
    policies = {benchmark: {} for benchmark in ("mind2web", "screenspot_pro")}
    report = {"fit_folds": fit_folds, "config_validation_fold": config_validation_fold, "arms": {}}
    for benchmark in policies:
        for arm, rows_by_benchmark in banks.items():
            rows = rows_by_benchmark[benchmark]
            fit_ids = [row_id for row_id, row in rows.items() if row.fold in fit_folds]
            reliability = source_reliability(rows, fit_ids)
            if benchmark == "screenspot_pro":
                configuration = {"granularity": "G4", "coordinate_tolerance": 14.0}
                scale = None
                validation_accuracy = None
            else:
                config_train_ids = [row_id for row_id, row in rows.items() if row.fold in config_train_folds]
                config_validation_ids = [row_id for row_id, row in rows.items() if row.fold == config_validation_fold]
                configuration, validation_accuracy = choose_config(rows, config_train_ids, config_validation_ids, cev_config)
                scale = fit_scale(fit_ids)
            policies[benchmark][arm] = Policy(
                benchmark, arm, configuration, reliability, scale,
                tuple(sorted(fit_folds)), config_validation_fold,
            )
            report["arms"][f"{benchmark}/{arm}"] = {
                "configuration": configuration,
                "validation_accuracy": validation_accuracy,
                "fit_rows": len(fit_ids),
            }
    return policies, report


def fit_final_policies(banks, outer_fold, cev_result):
    policies = {benchmark: {} for benchmark in ("mind2web", "screenspot_pro")}
    for benchmark in policies:
        for arm, rows_by_benchmark in banks.items():
            rows = rows_by_benchmark[benchmark]
            fit_ids = [row_id for row_id, row in rows.items() if row.fold != outer_fold]
            reliability = source_reliability(rows, fit_ids)
            fold_record = cev_result[benchmark]["folds"][outer_fold]["arms"][arm]
            configuration = fold_record["global_configuration"]
            scale = tuple(fold_record["outer_refit_scale"]) if benchmark == "mind2web" else None
            policies[benchmark][arm] = Policy(
                benchmark, arm, configuration, reliability, scale,
                tuple(fold for fold in range(5) if fold != outer_fold), None,
            )
    return policies


def load_cev_config():
    return yaml.safe_load((CEV_ROOT / "configs/cev_prereg.yaml").read_text())
