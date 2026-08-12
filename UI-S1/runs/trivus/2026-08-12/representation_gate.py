import collections
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CONFIG_PATH = RUN_DIR / "configs/representation_gate.yaml"
sys.path.insert(0, str(RUN_DIR))

from recovery_common import atomic_json, load_jsonl, sha256_file
from selector_data import audit_public_record, load_config as load_selector_config, public_candidate_permutation
from finalize_selector import validate_prediction


AUTHORIZATION_PATH = RUN_DIR / "REPRESENTATION_GATE_AUTHORIZATION.json"


def validate_authorization(config, path=AUTHORIZATION_PATH):
    path = Path(path)
    if not path.is_file():
        raise PermissionError("TriVUS representation gate is not execution-authorized")
    authorization = json.loads(path.read_text())
    if (
        authorization.get("status") != "AUTHORIZED_TRIVUS_REPRESENTATION_GATE_ONCE"
        or authorization.get("result_must_not_exist") is not True
    ):
        raise PermissionError("TriVUS invalid representation authorization")
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", authorization["implementation_commit"], "HEAD"],
        cwd=ROOT, check=False,
    )
    if completed.returncode:
        raise PermissionError("TriVUS representation implementation commit is not an ancestor of HEAD")
    expected_files = {
        "AMENDMENT_007_SELECTOR_REPRESENTATION_GATE.md",
        "configs/representation_gate.yaml",
        "representation_gate.py",
        "test_representation_gate.py",
    }
    if set(authorization.get("files", {})) != expected_files:
        raise PermissionError("TriVUS representation authorization file set mismatch")
    for name, expected_hash in authorization["files"].items():
        if sha256_file(RUN_DIR / name) != expected_hash:
            raise PermissionError(f"TriVUS representation implementation hash mismatch: {name}")
    if (RUN_DIR / "REPRESENTATION_GATE.json").exists():
        raise FileExistsError(RUN_DIR / "REPRESENTATION_GATE.json")
    return authorization


def load_config(require_authorization=True):
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if config.get("status") != "FROZEN_AFTER_PRIVATE_LABEL_SEAL_BEFORE_SELECTOR_METRICS":
        raise ValueError("TriVUS representation gate is not frozen")
    if config.get("settings") != ["low", "high"] or config.get("models") != [
        "UI-AGILE-7B", "GUI-R1-7B", "UI-R1-E-3B"
    ]:
        raise ValueError("TriVUS representation gate roster mismatch")
    if config.get("noninferiority_margin") != 0.01 or config.get("repair_auroc_threshold") != 0.55:
        raise ValueError("TriVUS representation gate threshold mismatch")
    if config["bootstrap"].get("resamples") != 10000 or config["bootstrap"].get("confidence") != 0.99:
        raise ValueError("TriVUS representation bootstrap mismatch")
    for item in config["dependencies"].values():
        if sha256_file(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"TriVUS representation dependency mismatch: {item['path']}")
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", config["private_label_seal_commit"], "HEAD"],
        cwd=ROOT, check=False,
    )
    if completed.returncode:
        raise PermissionError("TriVUS private-label seal commit is not an ancestor of HEAD")
    public_manifest = json.loads((ROOT / config["dependencies"]["public_manifest"]["path"]).read_text())
    blind_manifest = json.loads((ROOT / config["dependencies"]["blind_manifest"]["path"]).read_text())
    if (
        public_manifest.get("status") != "PASS_TRIVUS_PUBLIC_BANK_LOCKED"
        or public_manifest.get("public_sha256") != config["dependencies"]["public_records"]["sha256"]
        or blind_manifest.get("status") != "PASS_TRIVUS_SELECTOR_BLIND_LOCK"
        or blind_manifest.get("public_sha256") != config["dependencies"]["public_records"]["sha256"]
        or blind_manifest.get("predictions_sha256") != config["dependencies"]["selector_predictions"]["sha256"]
        or blind_manifest.get("private_labels_created") is not False
        or blind_manifest.get("label_metrics_computed") is not False
    ):
        raise PermissionError("TriVUS invalid public/blind manifest linkage")
    private_manifest = json.loads((ROOT / config["dependencies"]["private_manifest"]["path"]).read_text())
    if (
        private_manifest.get("status") != "PASS_TRIVUS_FOLD_SEALED_PRIVATE_LABELS"
        or private_manifest.get("records") != 4000
        or private_manifest.get("selector_metric_computed") is not False
        or private_manifest.get("training_started") is not False
    ):
        raise PermissionError("TriVUS invalid private-label seal")
    if require_authorization:
        validate_authorization(config)
    return config, private_manifest


def load_locked_public_predictions(config):
    public_path = ROOT / config["dependencies"]["public_records"]["path"]
    prediction_path = ROOT / config["dependencies"]["selector_predictions"]["path"]
    if sha256_file(public_path) != config["dependencies"]["public_records"]["sha256"]:
        raise PermissionError("TriVUS representation public JSONL hash mismatch")
    if sha256_file(prediction_path) != config["dependencies"]["selector_predictions"]["sha256"]:
        raise PermissionError("TriVUS representation prediction JSONL hash mismatch")
    public_rows = load_jsonl(public_path)
    prediction_rows = load_jsonl(prediction_path)
    public = {row["sample_key"]: row for row in public_rows}
    predictions = {row["sample_key"]: row for row in prediction_rows}
    if (
        len(public_rows) != 4000
        or len(public) != 4000
        or len(prediction_rows) != 4000
        or len(predictions) != 4000
        or set(public) != set(predictions)
    ):
        raise ValueError("TriVUS representation locked identity mismatch")
    selector_config = load_selector_config()
    for sample_key, row in public.items():
        audit_public_record(row)
        validate_prediction(predictions[sample_key], row, selector_config)
    return public_rows, predictions


def load_bootstrap(config):
    path = ROOT / config["dependencies"]["bootstrap"]["path"]
    spec = importlib.util.spec_from_file_location("trivus_representation_bootstrap", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.paired_bootstrap


def load_labels(private_manifest):
    labels = {}
    for fold in range(5):
        item = private_manifest["folds"][str(fold)]
        path = ROOT / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"TriVUS representation label hash mismatch: {fold}")
        rows = load_jsonl(path)
        if len(rows) != item["rows"]:
            raise ValueError(f"TriVUS representation label count mismatch: {fold}")
        for row in rows:
            if row["sample_key"] in labels or len(row["candidate_success"]) != 3:
                raise ValueError(f"TriVUS representation label identity mismatch: {row['sample_key']}")
            labels[row["sample_key"]] = row["candidate_success"]
    if len(labels) != 4000:
        raise ValueError("TriVUS representation label coverage mismatch")
    return labels


def hash_random_index(sample_key, seed):
    digest = hashlib.sha256(f"{sample_key}/{seed}/representation-random".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 3


def probability_by_public_candidate(prediction):
    output = [None] * 3
    for display_index, public_index in enumerate(prediction["display_to_candidate"]):
        output[public_index] = float(prediction["label_probabilities"][display_index])
    if any(value is None for value in output):
        raise ValueError("TriVUS representation probability permutation mismatch")
    return output


def majority_public_index(row, reliability, config):
    public_order = public_candidate_permutation(row["sample_key"], config["seed"])
    canonical_to_public = {canonical: public for public, canonical in enumerate(public_order)}
    parsed = [
        (canonical, canonical_to_public[canonical], row["candidates"][canonical_to_public[canonical]])
        for canonical in range(3)
        if row["candidates"][canonical_to_public[canonical]]["parse_ok"]
    ]
    if not parsed:
        return canonical_to_public[0]
    counts = collections.Counter(candidate["action"] for _, _, candidate in parsed)
    highest = max(counts.values())
    tied = {action for action, count in counts.items() if count == highest}
    priority = sorted(range(3), key=lambda index: (-reliability[index], index))
    return next(
        public_index for canonical in priority
        for source_index, public_index, candidate in parsed
        if source_index == canonical and candidate["action"] in tied
    )


def build_rows(config, labels, public_rows=None, predictions=None):
    if public_rows is None or predictions is None:
        public_rows, predictions = load_locked_public_predictions(config)
    if len(public_rows) != 4000 or set(predictions) != {row["sample_key"] for row in public_rows} or set(labels) != set(predictions):
        raise ValueError("TriVUS representation identity mismatch")
    output = []
    for setting in config["settings"]:
        setting_rows = [row for row in public_rows if row["setting"] == setting]
        for outer_fold in range(5):
            dev = [row for row in setting_rows if row["fold"] != outer_fold]
            test = [row for row in setting_rows if row["fold"] == outer_fold]
            reliability = []
            for canonical in range(3):
                values = []
                for row in dev:
                    public_order = public_candidate_permutation(row["sample_key"], config["seed"])
                    public_index = public_order.index(canonical)
                    values.append(labels[row["sample_key"]][public_index])
                reliability.append(float(np.mean(values)))
            for row in test:
                prediction = predictions[row["sample_key"]]
                probabilities = probability_by_public_candidate(prediction)
                direct = int(max(range(3), key=probabilities.__getitem__))
                fallback = majority_public_index(row, reliability, config)
                random_index = hash_random_index(row["sample_key"], config["seed"])
                success = labels[row["sample_key"]]
                output.append({
                    "sample_key": row["sample_key"],
                    "setting": setting,
                    "row_id": row["row_id"],
                    "fold": row["fold"],
                    "group": row["group"],
                    "probabilities": probabilities,
                    "direct_index": direct,
                    "fallback_index": fallback,
                    "random_index": random_index,
                    "direct_success": bool(success[direct]),
                    "fallback_success": bool(success[fallback]),
                    "random_success": bool(success[random_index]),
                    "candidate_success": success,
                })
    if len(output) != 4000 or len({row["sample_key"] for row in output}) != 4000:
        raise ValueError("TriVUS representation output coverage mismatch")
    return output


def compare(rows, setting, left, right, seed, paired_bootstrap):
    selected = [row for row in rows if row["setting"] == setting]
    metadata = {row["row_id"]: {"fold": row["fold"], "group": row["group"]} for row in selected}
    differences = {row["row_id"]: int(row[left]) - int(row[right]) for row in selected}
    return paired_bootstrap(metadata, differences, 10000, seed)


def repair_auroc(rows, setting):
    labels = []
    scores = []
    for row in rows:
        if row["setting"] != setting:
            continue
        for index, success in enumerate(row["candidate_success"]):
            labels.append(int(bool(success) and not row["fallback_success"]))
            scores.append(row["probabilities"][index])
    if len(set(labels)) != 2:
        raise ValueError(f"TriVUS representation AUROC lacks both classes: {setting}")
    return {
        "value": float(roc_auc_score(labels, scores)),
        "candidates": len(labels),
        "positives": int(sum(labels)),
    }


def adjudicate(rows, config, paired_bootstrap):
    comparisons = {}
    auroc = {}
    for setting in config["settings"]:
        comparisons[setting] = {
            "direct_minus_majority": compare(
                rows, setting, "direct_success", "fallback_success",
                config["bootstrap"]["seed"][f"{setting}_majority"], paired_bootstrap,
            ),
            "direct_minus_hash_random": compare(
                rows, setting, "direct_success", "random_success",
                config["bootstrap"]["seed"][f"{setting}_random"], paired_bootstrap,
            ),
        }
        auroc[setting] = repair_auroc(rows, setting)
    random_sanity = all(
        comparisons[setting]["direct_minus_hash_random"]["ci_99"][0] > 0
        for setting in config["settings"]
    )
    positive = [
        setting for setting in config["settings"]
        if comparisons[setting]["direct_minus_majority"]["ci_99"][0] > 0
    ]
    other_noninferior = all(
        comparisons[setting]["direct_minus_majority"]["ci_99"][0] > -config["noninferiority_margin"]
        for setting in config["settings"] if setting not in positive
    )
    point_mean = float(np.mean([
        comparisons[setting]["direct_minus_majority"]["point_delta"]
        for setting in config["settings"]
    ]))
    route_a1 = bool(positive) and other_noninferior and point_mean > 0
    route_a2 = all(auroc[setting]["value"] >= config["repair_auroc_threshold"] for setting in config["settings"])
    gates = {"RG_S_random_sanity": random_sanity, "RG_A1_direct_safe": route_a1, "RG_A2_repair_auroc": route_a2}
    return {
        "schema_version": 1,
        "status": "PASS_TRIVUS_REPRESENTATION_ADJUDICATED",
        "outcome": "PROCEED_TO_TRIVUS_TRAINING_IMPLEMENTATION" if random_sanity and (route_a1 or route_a2) else "STOP_TRIVUS_REPRESENTATION",
        "gates": gates,
        "comparisons": comparisons,
        "repair_auroc": auroc,
        "direct_minus_majority_equal_setting_point": point_mean,
        "config_sha256": sha256_file(CONFIG_PATH),
    }


def main():
    config, private_manifest = load_config()
    public_rows, predictions = load_locked_public_predictions(config)
    labels = load_labels(private_manifest)
    rows = build_rows(config, labels, public_rows, predictions)
    result = adjudicate(rows, config, load_bootstrap(config))
    output = RUN_DIR / "REPRESENTATION_GATE.json"
    if output.exists():
        raise FileExistsError(output)
    atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()