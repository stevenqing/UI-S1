import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CONFIG_PATH = RUN_DIR / "configs/private_labels_prereg.yaml"
sys.path.insert(0, str(RUN_DIR))

from recovery_common import atomic_json, load_config as load_recovery_config, load_jsonl, sha256_file
from selector_data import audit_public_record


def load_config():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if config.get("status") != "FROZEN_AFTER_COMMITTED_BLIND_LOCK_BEFORE_PRIVATE_LABELS":
        raise ValueError("TriVUS private-label protocol is not frozen")
    if config.get("expected_records") != 4000 or config.get("candidate_count") != 3:
        raise ValueError("TriVUS private-label coverage contract mismatch")
    if config.get("expected_fold_rows") != {0: 792, 1: 754, 2: 826, 3: 870, 4: 758}:
        raise ValueError("TriVUS private-label fold contract mismatch")
    expected_prohibitions = {
        "no_success_rate", "no_model_score", "no_selector_accuracy", "no_oracle",
        "no_majority_comparison", "no_auroc", "no_training",
    }
    if set(config.get("prohibitions", ())) != expected_prohibitions:
        raise ValueError("TriVUS private-label prohibition contract mismatch")
    for item in config["dependencies"].values():
        if sha256_file(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"TriVUS private-label dependency hash mismatch: {item['path']}")
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", config["blind_lock_commit"], "HEAD"],
        cwd=ROOT, check=False,
    )
    if completed.returncode:
        raise PermissionError("TriVUS blind-lock commit is not an ancestor of HEAD")
    blind = json.loads((ROOT / config["dependencies"]["blind_manifest"]["path"]).read_text())
    if (
        blind.get("status") != "PASS_TRIVUS_SELECTOR_BLIND_LOCK"
        or blind.get("private_labels_created") is not False
        or blind.get("ground_truth_fields_used") is not False
        or blind.get("label_metrics_computed") is not False
        or blind.get("predictions_sha256") != config["dependencies"]["selector_predictions"]["sha256"]
    ):
        raise PermissionError("TriVUS invalid committed blind lock")
    return config


def load_scoring(config):
    path = ROOT / config["dependencies"]["scoring"]["path"]
    spec = importlib.util.spec_from_file_location("trivus_private_scoring", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_candidate(reference, candidate, scoring, config):
    action = candidate["action"]
    if not candidate["parse_ok"] or action != reference["gt_action"]:
        return False
    if action in scoring.GROUNDING_ACTIONS:
        coordinate = candidate["coordinate"]
        if coordinate is None:
            return False
        width, height = reference["image_size"]
        target = (reference["gt_bbox"][0] / width, reference["gt_bbox"][1] / height)
        return math.dist(tuple(coordinate), target) < config["coordinate_radius"]
    if action in scoring.TEXT_ACTIONS:
        expected = str(reference["gt_input_text"])
        if action == "scroll":
            expected = expected.lower()
        return scoring.text_f1(candidate["parameter"], expected) >= config["text_f1_threshold"]
    if action in scoring.SIMPLE_ACTIONS:
        return True
    return False


def build_labels(config, scoring):
    public_rows = load_jsonl(ROOT / config["dependencies"]["public_records"]["path"])
    public = {row["sample_key"]: row for row in public_rows}
    if len(public) != len(public_rows) or len(public) != config["expected_records"]:
        raise ValueError("TriVUS private-label public identity mismatch")
    recovery = load_recovery_config()
    references = {}
    for setting in ("low", "high"):
        rows = load_jsonl(ROOT / recovery["references"][setting]["path"])
        references.update({f"androidcontrol/{setting}/{row['id']}": row for row in rows})
    if set(references) != set(public):
        raise ValueError("TriVUS private-label reference identity mismatch")
    output = {fold: [] for fold in range(config["folds"])}
    for sample_key in sorted(public):
        row = public[sample_key]
        audit_public_record(row)
        reference = references[sample_key]
        values = [score_candidate(reference, candidate, scoring, config) for candidate in row["candidates"]]
        if len(values) != config["candidate_count"]:
            raise ValueError(f"TriVUS private-label width mismatch: {sample_key}")
        output[row["fold"]].append({
            "schema_version": 1,
            "sample_key": sample_key,
            "candidate_success": [bool(value) for value in values],
        })
    for fold, rows in output.items():
        if len(rows) != config["expected_fold_rows"][fold]:
            raise ValueError(f"TriVUS private-label fold coverage mismatch: {fold}/{len(rows)}")
    return output


def write_fold(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", buffering=1) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main():
    config = load_config()
    outputs = [RUN_DIR / f"data/private_labels_fold-{fold}.jsonl" for fold in range(config["folds"])]
    manifest_path = RUN_DIR / "data/PRIVATE_LABEL_MANIFEST.json"
    if any(path.exists() for path in (*outputs, manifest_path)):
        raise FileExistsError("TriVUS private-label output already exists")
    scoring = load_scoring(config)
    labels = build_labels(config, scoring)
    folds = {}
    for fold, path in enumerate(outputs):
        write_fold(path, labels[fold])
        folds[str(fold)] = {
            "path": str(path.relative_to(ROOT)),
            "rows": len(labels[fold]),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema_version": 1,
        "status": "PASS_TRIVUS_FOLD_SEALED_PRIVATE_LABELS",
        "records": sum(item["rows"] for item in folds.values()),
        "candidate_count": config["candidate_count"],
        "schema": config["private_schema"],
        "folds": folds,
        "blind_lock_commit": config["blind_lock_commit"],
        "dependencies": {name: item["sha256"] for name, item in config["dependencies"].items()},
        "aggregate_success_statistics_computed": False,
        "selector_metric_computed": False,
        "oracle_metric_computed": False,
        "training_started": False,
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()