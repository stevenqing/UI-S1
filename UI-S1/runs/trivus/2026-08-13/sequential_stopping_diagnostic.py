import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml


RUN_DIR = Path(__file__).resolve().parent
PRIOR_DIR = RUN_DIR.parent / "2026-08-12"
OUTPUT_ROOT = RUN_DIR / "sequential_exploratory"
CONFIG_PATH = RUN_DIR / "configs/sequential_training_prereg.yaml"
RESULT_PATH = RUN_DIR / "SEQUENTIAL_STOPPING_DIAGNOSTIC.json"
sys.path.insert(0, str(RUN_DIR))
sys.path.insert(0, str(PRIOR_DIR))

from context_common import atomic_json_file, sha256_file
from finalize_trivus import frozen_baselines, load_configs, load_public
from headroom_atlas import load_candidate_labels
from sequential_oof_runner import FAMILIES


def cell_name(public_row):
    return (
        public_row["setting"]
        if public_row["benchmark"] == "androidcontrol"
        else public_row["arm"]
    )


def load_scope(outer, holdout, family, public):
    phases = {}
    for phase in ("cheap", "verifier"):
        path = (
            OUTPUT_ROOT / phase / f"outer-{outer}" / f"holdout-{holdout}"
            / f"{family}.jsonl"
        )
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        phases[phase] = {row["context_key"]: row for row in rows}
        if len(phases[phase]) != len(rows):
            raise ValueError("Sequential stopping duplicate context")
    if set(phases["cheap"]) != set(phases["verifier"]):
        raise ValueError("Sequential stopping phase context mismatch")
    output = []
    for context in sorted(phases["cheap"]):
        cheap = phases["cheap"][context]
        verifier = phases["verifier"][context]
        sample_key = cheap["sample_key"]
        if (
            verifier["sample_key"] != sample_key
            or cheap["fold"] != holdout
            or verifier["fold"] != holdout
            or public[sample_key]["benchmark"] != family
        ):
            raise ValueError("Sequential stopping scope mismatch")
        output.append({
            "context_key": context,
            "sample_key": sample_key,
            "cell": cell_name(public[sample_key]),
            "order": cheap["candidate_order"],
            "cheap_probability": cheap["candidate_probabilities"],
            "verifier_probability": verifier["candidate_probabilities"],
        })
    return output


def fallback_reliability(rows, strongest):
    values = defaultdict(list)
    for row in rows:
        values[row["cell"]].append(bool(strongest[row["sample_key"]]))
    if not values:
        raise ValueError("Sequential stopping has no fallback reliability rows")
    return {cell: float(np.mean(items)) for cell, items in values.items()}


def apply_policy(rows, labels, strongest, reliability, source, parameters):
    outputs = []
    for row in rows:
        fallback = bool(strongest[row["sample_key"]])
        selected = None
        if parameters is not None:
            budget, minimum_delta, maximum_loss_risk = parameters
            baseline_probability = reliability[row["cell"]]
            probabilities = row[f"{source}_probability"]
            for candidate in row["order"][:budget]:
                probability = float(probabilities[candidate])
                if (
                    probability - baseline_probability >= minimum_delta
                    and baseline_probability * (1.0 - probability) <= maximum_loss_risk
                ):
                    selected = int(candidate)
                    break
        direct = (
            bool(labels[row["sample_key"]][selected])
            if selected is not None else fallback
        )
        outputs.append({
            "cell": row["cell"],
            "success": direct,
            "fallback": fallback,
            "override": selected is not None,
            "win": selected is not None and direct and not fallback,
            "loss": selected is not None and fallback and not direct,
        })
    return outputs


def summarize(outputs):
    by_cell = defaultdict(list)
    for row in outputs:
        by_cell[row["cell"]].append(row)
    cells = {}
    for cell, rows in sorted(by_cell.items()):
        accuracy = float(np.mean([row["success"] for row in rows]))
        baseline = float(np.mean([row["fallback"] for row in rows]))
        cells[cell] = {
            "rows": len(rows),
            "accuracy": accuracy,
            "strongest": baseline,
            "delta": accuracy - baseline,
            "override_rate": float(np.mean([row["override"] for row in rows])),
            "wins": sum(row["win"] for row in rows),
            "losses": sum(row["loss"] for row in rows),
        }
    return {
        "cells": cells,
        "equal_cell_delta": float(np.mean([row["delta"] for row in cells.values()])),
    }


def select_parameters(rows, labels, strongest, source, config, family):
    reliability = fallback_reliability(rows, strongest)
    candidates = [None]
    candidates.extend(
        (budget, minimum_delta, maximum_loss_risk)
        for budget in config["sequential_policy"]["budget_grid"][family]
        for minimum_delta in config["sequential_policy"]["minimum_delta_grid"]
        for maximum_loss_risk in config["sequential_policy"]["maximum_loss_risk_grid"]
    )
    mde = float(config["safety"]["mde"][family])
    best = None
    for parameters in candidates:
        report = summarize(apply_policy(
            rows, labels, strongest, reliability, source, parameters
        ))
        if any(value["delta"] < -mde for value in report["cells"].values()):
            continue
        losses = sum(value["losses"] for value in report["cells"].values())
        if parameters is None:
            tie = (report["equal_cell_delta"], -losses, 0, 0.0, 0.0)
        else:
            budget, minimum_delta, maximum_loss_risk = parameters
            tie = (
                report["equal_cell_delta"], -losses, -budget,
                minimum_delta, -maximum_loss_risk,
            )
        if best is None or tie > best[0]:
            best = (tie, parameters, report, reliability)
    if best is None:
        raise ValueError("Sequential stopping calibration found no safe policy")
    return {
        "parameters": list(best[1]) if best[1] is not None else None,
        "calibration": best[2],
        "fallback_reliability": best[3],
    }


def cross_fitted_family(family, source, public, labels, strongest, config):
    evaluations = []
    selections = []
    for outer in range(5):
        development = [fold for fold in range(5) if fold != outer]
        scopes = {
            holdout: load_scope(outer, holdout, family, public)
            for holdout in development
        }
        for holdout in development:
            calibration = [
                row for fold in development if fold != holdout
                for row in scopes[fold]
            ]
            selected = select_parameters(
                calibration, labels, strongest, source, config, family
            )
            outputs = apply_policy(
                scopes[holdout], labels, strongest,
                selected["fallback_reliability"], source,
                tuple(selected["parameters"]) if selected["parameters"] is not None else None,
            )
            evaluations.extend(outputs)
            selections.append({
                "outer_fold": outer,
                "holdout_fold": holdout,
                "parameters": selected["parameters"],
                "calibration_equal_cell_delta": selected["calibration"]["equal_cell_delta"],
            })
    report = summarize(evaluations)
    report["parameter_counts"] = {
        json.dumps(parameters): count
        for parameters, count in Counter(
            tuple(row["parameters"]) if row["parameters"] is not None else None
            for row in selections
        ).items()
    }
    report["selections"] = selections
    report["repeated_sample_contexts"] = 4
    return report


def main():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    public = load_public()
    labels, manifests = load_candidate_labels(public)
    _, training = load_configs()
    _, strongest = frozen_baselines(public, training)
    results = {
        source: {
            family: cross_fitted_family(
                family, source, public, labels, strongest, config
            )
            for family in FAMILIES
        }
        for source in ("cheap", "verifier")
    }
    result = {
        "schema_version": 1,
        "status": "PASS_EXPLORATORY_CROSS_FITTED_STOPPING_DIAGNOSTIC",
        "outcome": "EXPLORATORY_ONLY_NO_PROMOTION",
        "config_sha256": sha256_file(CONFIG_PATH),
        "label_manifests": manifests,
        "results": results,
        "claim_boundary": {
            "confirmatory": False,
            "promotion_allowed": False,
            "repeated_sample_contexts": 4,
        },
    }
    if RESULT_PATH.exists():
        raise FileExistsError(RESULT_PATH)
    atomic_json_file(RESULT_PATH, result)
    print(json.dumps({
        source: {
            family: {
                "equal_cell_delta": value["equal_cell_delta"],
                "cells": {
                    cell: {
                        "delta": row["delta"],
                        "override_rate": row["override_rate"],
                    }
                    for cell, row in value["cells"].items()
                },
                "parameter_counts": value["parameter_counts"],
            }
            for family, value in families.items()
        }
        for source, families in results.items()
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()