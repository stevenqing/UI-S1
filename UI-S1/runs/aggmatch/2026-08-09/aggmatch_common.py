import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CLOSE_DIR = ROOT / "runs/close/2026-08-08"
E1_SCRIPT = CLOSE_DIR / "e1_arm_aggregator_matrix.py"
E1_CONFIG = CLOSE_DIR / "configs/aggregator_map.yaml"
E1_RESULT = CLOSE_DIR / "e1_arm_aggregator_matrix.json"
CACHE_PATH = RUN_DIR / "derived/e1_row_outputs.json.gz"
ARMS = ("C_uni", "C_cond", "C_rand", "C_self")
METHODS = ("majority", "A0", "ours", "A1", "A2", "A3", "A4")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def provenance():
    return {
        "e1_script_sha256": sha256_file(E1_SCRIPT),
        "e1_config_sha256": sha256_file(E1_CONFIG),
        "e1_result_sha256": sha256_file(E1_RESULT),
    }


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def serialize_outputs(outputs):
    return {
        arm: {
            method: {row_id: bool(value) for row_id, value in rows.items()}
            for method, rows in methods.items()
        }
        for arm, methods in outputs.items()
    }


def build_cache():
    e1 = load_module(E1_SCRIPT, "aggmatch_close_e1")
    config = yaml.safe_load(E1_CONFIG.read_text())
    mind = e1.mind2web_matrix(config)
    screen = e1.screenspot_matrix(config)
    mind_rows = {
        row["id"]: row
        for row in map(json.loads, (e1.XFER / "data/mind2web/mind2web_test_task.jsonl").read_text().splitlines())
    }
    mind_folds = json.loads((ROOT / "runs/complementarity/2026-07-30/folds.json").read_text())["pools"]["mind2web/visual"]["group_to_fold"]
    screen_common = e1.load_module(e1.CONSOLIDATE / "common.py", "aggmatch_consolidate_common")
    screen_context = screen_common.load_context()
    cache = {
        "schema_version": 1,
        "status": "DERIVED_ZERO_GPU",
        "provenance": provenance(),
        "methods": list(METHODS),
        "arms": list(ARMS),
        "mind2web": {
            "metadata": {
                row_id: {
                    "fold": mind_folds[row["website"]],
                    "group": row["episode_id"],
                    "action": row["step"]["operation"]["op"],
                }
                for row_id, row in mind_rows.items()
            },
            "accuracy": mind["accuracy"],
            "outputs": serialize_outputs(mind["outputs"]),
        },
        "screenspot_pro": {
            "metadata": {
                row_id: {
                    "fold": screen_context["fold_for_group"][screen_context["metadata"][row_id]["application"]],
                    "group": screen_context["metadata"][row_id]["application"],
                }
                for row_id in screen_context["row_ids"]
            },
            "accuracy": screen["accuracy"],
            "outputs": serialize_outputs(screen["outputs"]),
        },
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CACHE_PATH.with_suffix(CACHE_PATH.suffix + ".tmp")
    with gzip.open(temporary, "wt") as handle:
        json.dump(cache, handle, sort_keys=True, separators=(",", ":"))
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(CACHE_PATH)
    return cache


def load_cache():
    if CACHE_PATH.exists():
        with gzip.open(CACHE_PATH, "rt") as handle:
            cache = json.load(handle)
        if cache.get("provenance") == provenance():
            return cache
    return build_cache()


def paired_bootstrap(metadata, differences, resamples, seed):
    by_fold_group = {}
    for row_id in differences:
        row = metadata[row_id]
        by_fold_group.setdefault(row["fold"], {}).setdefault(row["group"], []).append(row_id)
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for sample_index in range(resamples):
        selected = []
        for fold in sorted(by_fold_group):
            groups = sorted(by_fold_group[fold])
            for group in rng.choice(groups, size=len(groups), replace=True):
                selected.extend(by_fold_group[fold][group])
        samples[sample_index] = np.mean([differences[row_id] for row_id in selected])
    point = float(np.mean(list(differences.values())))
    return {
        "point_delta": point,
        "ci_99": [float(np.quantile(samples, 0.005)), float(np.quantile(samples, 0.995))],
        "p_delta_le_zero_plus_one": float((1 + np.count_nonzero(samples <= 0)) / (resamples + 1)),
        "resamples": resamples,
        "seed": seed,
        "rows": len(differences),
        "groups": len({row["group"] for row_id, row in metadata.items() if row_id in differences}),
    }


def method_difference(outputs, left_arm, left_method, right_arm, right_method, row_ids=None):
    left = outputs[left_arm][left_method]
    right = outputs[right_arm][right_method]
    identities = sorted(set(left) & set(right)) if row_ids is None else sorted(row_ids)
    return {row_id: int(left[row_id]) - int(right[row_id]) for row_id in identities}
