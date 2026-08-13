import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
PRIOR_DIR = RUN_DIR.parent / "2026-08-12"
CONFIG_PATH = RUN_DIR / "configs/sequential_training_prereg.yaml"
OUTPUT_PATH = RUN_DIR / "SEQUENTIAL_REAL_DATA_SMOKE.json"
sys.path.insert(0, str(PRIOR_DIR))

from context_common import atomic_json_file, sha256_file
from trivus_assembly import (
    assemble_phase_data, load_config as load_assembly_config,
    load_context_manifest, load_context_phase, load_locked_public_inputs,
)
from trivus_data import validate_trivus_data


def load_config():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if (
        config.get("status") != "FROZEN_BEFORE_SEQUENTIAL_REAL_DATA_OPTIMIZER"
        or config["execution"] != {
            "real_data_optimizer_authorized": False,
            "confirmation_authorized": False,
            "promotion_allowed": False,
        }
        or config["features"]["public_candidate_dimensions"] != 115
        or config["features"]["verifier_dimensions"] != 120
        or config["cross_fitting"]["verifier_inputs_must_be_cheap_oof"] is not True
        or config["cross_fitting"]["calibration_inputs_must_be_verifier_oof"] is not True
    ):
        raise PermissionError("Sequential training prereg mismatch")
    if Path(sys.executable).absolute() != (ROOT / config["python"]).absolute():
        raise RuntimeError("Sequential smoke interpreter mismatch")
    for item in config["dependencies"].values():
        if sha256_file(ROOT / item["path"]) != item["sha256"]:
            raise PermissionError(f"Sequential dependency mismatch: {item['path']}")
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", config["implementation_commit_floor"], "HEAD"],
        cwd=ROOT, check=False,
    )
    if completed.returncode:
        raise PermissionError("Sequential implementation commit is not an ancestor")
    return config


def summarize(data, expected_folds):
    validate_trivus_data(data)
    folds = sorted(set(int(value) for value in data.folds))
    if folds != sorted(expected_folds):
        raise ValueError("Sequential smoke fold mismatch")
    family_rows = {
        family: int(sum(value == family for value in data.families))
        for family in ("mind2web", "screenspot_pro", "androidcontrol")
    }
    valid_candidates = data.candidate_mask.sum(axis=1)
    return {
        "rows": len(data),
        "folds": folds,
        "family_rows": family_rows,
        "feature_shape": list(data.features.shape),
        "candidate_counts": {
            str(count): int(np.sum(valid_candidates == count))
            for count in (3, 12)
        },
        "candidate_labels": int(valid_candidates.sum()),
        "positive_candidate_labels": int(data.labels[data.candidate_mask].sum()),
    }


def main():
    config = load_config()
    assembly = load_assembly_config()
    public, predictions = load_locked_public_inputs(assembly)
    manifest = load_context_manifest(assembly)
    context_path = ROOT / assembly["dependencies"]["contexts"]["path"]
    counts = assembly["expected"]["context_records_by_public_fold"]
    outer_fold = 0
    holdout_fold = 1
    phase = load_context_phase(
        context_path, manifest, public, outer_fold, "inner", counts,
        holdout_fold=holdout_fold,
    )
    train, train_opened = assemble_phase_data(
        assembly, public, predictions, phase, phase.fit_folds
    )
    checkpoint, checkpoint_opened = assemble_phase_data(
        assembly, public, predictions, phase, (phase.checkpoint_fold,)
    )
    holdout, holdout_opened = assemble_phase_data(
        assembly, public, predictions, phase, (phase.holdout_fold,)
    )
    opened = {
        "train": sorted(str(Path(path).relative_to(ROOT)) for path in train_opened),
        "checkpoint": sorted(str(Path(path).relative_to(ROOT)) for path in checkpoint_opened),
        "holdout": sorted(str(Path(path).relative_to(ROOT)) for path in holdout_opened),
    }
    result = {
        "schema_version": 1,
        "status": "PASS_SEQUENTIAL_NO_OPTIMIZER_REAL_DATA_SMOKE",
        "config_sha256": sha256_file(CONFIG_PATH),
        "outer_fold": outer_fold,
        "holdout_fold": holdout_fold,
        "fit_folds": list(phase.fit_folds),
        "checkpoint_fold": phase.checkpoint_fold,
        "train": summarize(train, phase.fit_folds),
        "checkpoint": summarize(checkpoint, (phase.checkpoint_fold,)),
        "holdout": summarize(holdout, (phase.holdout_fold,)),
        "opened_private_files": opened,
        "optimizer_constructed": False,
        "backward_called": False,
        "model_parameters_created": False,
        "real_data_optimizer_authorized": config["execution"]["real_data_optimizer_authorized"],
        "confirmation_authorized": config["execution"]["confirmation_authorized"],
    }
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    atomic_json_file(OUTPUT_PATH, result)
    print(json.dumps({
        "status": result["status"],
        "folds": {
            "fit": result["fit_folds"],
            "checkpoint": result["checkpoint_fold"],
            "holdout": result["holdout_fold"],
        },
        "rows": {
            name: result[name]["rows"] for name in ("train", "checkpoint", "holdout")
        },
        "candidate_labels": {
            name: result[name]["candidate_labels"]
            for name in ("train", "checkpoint", "holdout")
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()