import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CONFIG_PATH = RUN_DIR / "configs/fallback_contexts_prereg.yaml"
UTILITY_DIR = ROOT / "runs/lsa-utility/2026-08-11"
sys.path.insert(0, str(UTILITY_DIR))
sys.path.insert(0, str(RUN_DIR))

import behavior_policy
from behavior_policy import Policy, apply_policy, load_cev_config
from recovery_common import sha256_file
from selector_data import public_candidate_permutation


ARMS = ("C_uni", "C_cond", "C_rand", "C_self")
BENCHMARKS = ("mind2web", "screenspot_pro")
MIND_MODELS = ("TongUI-7B", "CogAgent-18B", "UI-TARS-7B")
SCREEN_MODELS = ("GTA1-7B", "Qwen3-VL-8B-Instruct", "UI-TARS-7B-SFT")
CONTEXT_FIELDS = {
    "schema_version", "context_key", "sample_key", "outer_fold", "role",
    "holdout_fold", "fit_folds", "fallback_index",
}


@dataclass(frozen=True)
class ContextCandidate:
    source: str
    lineage: str
    action: str
    baseline_coordinate: tuple[float, float] | None
    parameter: str
    parse_ok: bool
    order: int
    success: bool | None = None


@dataclass(frozen=True)
class ContextRow:
    row_id: str
    benchmark: str
    fold: int
    group: str
    candidates: tuple[ContextCandidate, ...]


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_prereg():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if config.get("status") != "FROZEN_AFTER_AMENDMENT_009_BEFORE_PRIVATE_SCALE_SEAL_AND_CONTEXTS":
        raise ValueError("TriVUS fallback-context protocol is not frozen")
    expected = config["expected"]
    if (
        config.get("python") != ".venv-scaleup/bin/python"
        or expected.get("contexts") != expected.get("total_records") * expected.get("contexts_per_record")
        or expected.get("contexts") != 391524
        or config.get("context_schema") != [
            "schema_version", "context_key", "sample_key", "outer_fold", "role",
            "holdout_fold", "fit_folds", "fallback_index",
        ]
        or set(config.get("context_schema", ())) != CONTEXT_FIELDS
    ):
        raise ValueError("TriVUS fallback-context contract mismatch")
    assert_context_environment(config)
    for item in config["dependencies"].values():
        path = ROOT / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"TriVUS fallback-context dependency mismatch: {item['path']}")
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", config["protocol_commit"], "HEAD"],
        cwd=ROOT, check=False,
    )
    if completed.returncode:
        raise PermissionError("TriVUS fallback-context protocol commit is not an ancestor")
    return config


@lru_cache(maxsize=1)
def git_root():
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return Path(completed.stdout.strip()).resolve()


def git_relative_path(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(git_root()).as_posix()
    except ValueError as error:
        raise PermissionError(f"TriVUS file is outside Git repository: {path}") from error


def committed_file(path):
    path = Path(path).resolve()
    relative = git_relative_path(path)
    completed = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", relative],
        cwd=git_root(), check=True, capture_output=True, text=True,
    )
    commit = completed.stdout.strip()
    if not commit:
        raise PermissionError(f"TriVUS file is not committed: {relative}")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=git_root(), check=False,
    )
    if ancestor.returncode:
        raise PermissionError(f"TriVUS file commit is not an ancestor: {relative}")
    blob = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=git_root(), check=True, capture_output=True,
    ).stdout
    if blob != path.read_bytes():
        raise PermissionError(f"TriVUS committed file differs from working tree: {relative}")
    return commit


def git_blob_sha256(commit, path):
    relative = git_relative_path(path)
    blob = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=git_root(), check=True, capture_output=True,
    ).stdout
    return hashlib.sha256(blob).hexdigest()


def require_commit_order(ancestor, descendant, description):
    if ancestor == descendant:
        raise PermissionError(f"TriVUS commit order is not strict: {description}")
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT, check=False,
    )
    if completed.returncode:
        raise PermissionError(f"TriVUS invalid commit order: {description}")


def assert_context_environment(config):
    expected_executable = (ROOT / config["python"]).absolute()
    observed_executable = Path(sys.executable).absolute()
    if observed_executable != expected_executable:
        raise RuntimeError(
            f"TriVUS fallback-context interpreter mismatch: "
            f"{observed_executable}/{expected_executable}"
        )
    observed = {
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "torch": importlib.metadata.version("torch"),
        "pyyaml": importlib.metadata.version("PyYAML"),
    }
    if observed != config["environment"]:
        raise RuntimeError(f"TriVUS fallback-context environment mismatch: {observed}")
    return observed


def fsync_directory(path):
    descriptor = os.open(Path(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json_file(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("w") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_exclusive_json(path, value, schema):
    if list(value) != schema:
        raise ValueError("TriVUS exclusive JSON schema mismatch")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)


def publish_directory(staging, destination):
    staging = Path(staging)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if staging.parent.resolve() != destination.parent.resolve():
        raise ValueError("TriVUS publication must use a same-parent directory rename")
    fsync_directory(staging)
    staging.rename(destination)
    fsync_directory(destination.parent)


@contextmanager
def staging_directory(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.staging-",
        dir=destination.parent,
    ))
    try:
        yield staging
    finally:
        if staging.exists():
            shutil.rmtree(staging)
            fsync_directory(destination.parent)


def repository_path(value):
    raw = Path(value)
    path = raw if raw.is_absolute() else ROOT / raw
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"TriVUS path escapes repository: {value}") from error
    return resolved


def safe_child_path(base, value):
    raw = Path(value)
    if raw.is_absolute():
        raise ValueError(f"TriVUS sealed path must be relative: {value}")
    base = Path(base).resolve()
    resolved = (base / raw).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as error:
        raise ValueError(f"TriVUS sealed path escapes base: {value}") from error
    return resolved


def sha256_rows(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def write_jsonl_atomic(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("w", buffering=1) as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def checkpoint_and_fit_folds(outer_fold, holdout_fold):
    if outer_fold not in range(5) or holdout_fold not in range(5) or outer_fold == holdout_fold:
        raise ValueError(f"invalid nested folds: {outer_fold}/{holdout_fold}")
    development = tuple(fold for fold in range(5) if fold != outer_fold)
    candidates = set(development) - {holdout_fold}
    checkpoint = next(
        fold for offset in range(1, 6)
        if (fold := (holdout_fold + offset) % 5) in candidates
    )
    fit_folds = tuple(fold for fold in development if fold not in {holdout_fold, checkpoint})
    if len(fit_folds) != 2 or set(fit_folds) | {holdout_fold, checkpoint} != set(development):
        raise AssertionError("TriVUS nested split construction failed")
    return checkpoint, fit_folds


def mind_layout():
    values = []
    for view in (0, 1):
        values.extend((f"stage1_{model}_view{view}", model) for model in MIND_MODELS)
    for crop in range(2):
        values.extend((f"stage2_{model}_crop{crop}", model) for model in MIND_MODELS)
    return tuple(values)


def screen_layout(record, region):
    if record["arm"] == "C_uni":
        actions = [(model, view) for view in range(4) for model in SCREEN_MODELS]
        return tuple((f"{model}_view{view}", model) for model, view in actions)
    actions = [(str(model), int(view)) for model, view in region["stage1_actions"]]
    values = [(f"{model}_view{view}", model) for model, view in actions]
    for crop in range(2):
        values.extend((f"{model}_{record['arm']}_crop{crop}", model) for model in SCREEN_MODELS)
    if len(values) != 12:
        raise ValueError(f"TriVUS screen source layout mismatch: {record['sample_key']}")
    return tuple(values)


def restore_coordinate(coordinate, size=None):
    if coordinate is None:
        return None
    values = tuple(float(value) for value in coordinate)
    if size is None:
        return values
    return (
        float(round(values[0] * size[0])),
        float(round(values[1] * size[1])),
    )


def build_vus_banks(public_rows, screen_regions):
    regions = {row["id"]: row for row in screen_regions}
    public = {}
    banks = {arm: {benchmark: {} for benchmark in BENCHMARKS} for arm in ARMS}
    for record in public_rows:
        sample_key = record["sample_key"]
        if sample_key in public or record["arm"] not in ARMS or record["benchmark"] not in BENCHMARKS:
            raise ValueError(f"TriVUS VUS public identity mismatch: {sample_key}")
        if len(record["candidates"]) != 12:
            raise ValueError(f"TriVUS VUS candidate width mismatch: {sample_key}")
        if record["benchmark"] == "mind2web":
            layout = mind_layout()
            size = None
        else:
            region = regions.get(record["row_id"])
            if region is None:
                raise KeyError(record["row_id"])
            layout = screen_layout(record, region)
            size = tuple(region["img_size"])
        candidates = []
        for order, (candidate, (source, lineage)) in enumerate(zip(record["candidates"], layout)):
            coordinate = restore_coordinate(candidate["coordinate"], size)
            candidates.append(ContextCandidate(
                source=source,
                lineage=lineage,
                action=str(candidate["action"]),
                baseline_coordinate=coordinate,
                parameter=str(candidate["parameter"]),
                parse_ok=bool(candidate["parse_ok"]),
                order=order,
            ))
        row = ContextRow(
            row_id=record["row_id"],
            benchmark=record["benchmark"],
            fold=int(record["fold"]),
            group=str(record["group"]),
            candidates=tuple(candidates),
        )
        arm_rows = banks[record["arm"]][record["benchmark"]]
        if record["row_id"] in arm_rows:
            raise ValueError(f"TriVUS duplicate VUS row: {sample_key}")
        arm_rows[record["row_id"]] = row
        public[sample_key] = record
    if len(public) != 14644 or any(
        set(banks[arm][benchmark]) != set(banks["C_uni"][benchmark])
        for arm in ARMS for benchmark in BENCHMARKS
    ):
        raise ValueError("TriVUS VUS public coverage mismatch")
    return banks, public


def inject_vus_labels(banks, public, labels, fit_folds):
    fit_folds = set(fit_folds)
    expected = {key for key, row in public.items() if int(row["fold"]) in fit_folds}
    if set(labels) != expected:
        raise ValueError("TriVUS VUS fit-label identity mismatch")
    output = {
        arm: {benchmark: dict(rows) for benchmark, rows in by_benchmark.items()}
        for arm, by_benchmark in banks.items()
    }
    for sample_key, label in labels.items():
        record = public[sample_key]
        values = label.get("candidate_success")
        if len(values) != 12 or any(type(value) is not bool for value in values):
            raise ValueError(f"TriVUS VUS private-label width mismatch: {sample_key}")
        row = output[record["arm"]][record["benchmark"]][record["row_id"]]
        candidates = tuple(
            replace(candidate, success=value)
            for candidate, value in zip(row.candidates, values)
        )
        output[record["arm"]][record["benchmark"]][record["row_id"]] = replace(
            row, candidates=candidates,
        )
    return output


def set_private_scales(scales):
    if not scales or any(
        not all(math.isfinite(float(value)) and float(value) >= 0 for value in pair)
        for pair in scales.values()
    ):
        raise ValueError("TriVUS invalid private Mind2Web scales")
    behavior_policy.MIND_SCALES = {
        row_id: (float(values[0]), float(values[1])) for row_id, values in scales.items()
    }


def fit_inner_vus_policies(banks, fit_folds, checkpoint_fold, scales):
    set_private_scales(scales)
    try:
        policies, _ = behavior_policy.fit_inner_policies(
            banks, list(fit_folds), int(checkpoint_fold), load_cev_config(),
        )
        return policies
    finally:
        behavior_policy.MIND_SCALES = None


def fit_final_vus_policies(banks, outer_fold, scales):
    development = tuple(fold for fold in range(5) if fold != outer_fold)
    inner_validation = (outer_fold + 1) % 5
    inner_training = tuple(fold for fold in development if fold != inner_validation)
    set_private_scales(scales)
    try:
        config = load_cev_config()
        policies = {benchmark: {} for benchmark in BENCHMARKS}
        for benchmark in BENCHMARKS:
            for arm in ARMS:
                rows = banks[arm][benchmark]
                fit_ids = [row_id for row_id, row in rows.items() if row.fold in development]
                reliability = behavior_policy.source_reliability(rows, fit_ids)
                if benchmark == "screenspot_pro":
                    configuration = {"granularity": "G4", "coordinate_tolerance": 14.0}
                    scale = None
                    config_validation = None
                else:
                    training_ids = [row_id for row_id, row in rows.items() if row.fold in inner_training]
                    validation_ids = [row_id for row_id, row in rows.items() if row.fold == inner_validation]
                    configuration, _ = behavior_policy.choose_config(
                        rows, training_ids, validation_ids, config,
                    )
                    scale = behavior_policy.fit_scale(fit_ids)
                    config_validation = inner_validation
                policies[benchmark][arm] = Policy(
                    benchmark=benchmark,
                    arm=arm,
                    configuration=configuration,
                    reliability=reliability,
                    scale=scale,
                    fit_folds=tuple(development),
                    config_validation_fold=config_validation,
                )
        return policies
    finally:
        behavior_policy.MIND_SCALES = None


def apply_vus_policies(banks, public, policies, allowed_folds):
    allowed_folds = set(allowed_folds)
    output = {}
    for sample_key in sorted(public):
        record = public[sample_key]
        if int(record["fold"]) not in allowed_folds:
            continue
        row = banks[record["arm"]][record["benchmark"]][record["row_id"]]
        fallback = apply_policy(row, policies[record["benchmark"]][record["arm"]])
        if not 0 <= fallback < 12:
            raise ValueError(f"TriVUS invalid VUS fallback: {sample_key}/{fallback}")
        output[sample_key] = fallback
    return output


def android_reliability(public, labels, fit_folds, seed):
    fit_folds = set(fit_folds)
    expected = {key for key, row in public.items() if int(row["fold"]) in fit_folds}
    if set(labels) != expected:
        raise ValueError("TriVUS Android fit-label identity mismatch")
    sums = {setting: np.zeros(3, dtype=np.float64) for setting in ("low", "high")}
    counts = {setting: np.zeros(3, dtype=np.int64) for setting in ("low", "high")}
    for sample_key, label in labels.items():
        row = public[sample_key]
        values = label.get("candidate_success")
        if len(values) != 3 or any(type(value) is not bool for value in values):
            raise ValueError(f"TriVUS Android private-label width mismatch: {sample_key}")
        public_order = public_candidate_permutation(sample_key, seed)
        for public_index, canonical in enumerate(public_order):
            sums[row["setting"]][canonical] += float(values[public_index])
            counts[row["setting"]][canonical] += 1
    output = {}
    for setting in ("low", "high"):
        if np.any(counts[setting] == 0):
            raise ValueError(f"TriVUS Android empty reliability cell: {setting}")
        output[setting] = tuple((sums[setting] / counts[setting]).tolist())
    return output


def android_majority_index(row, reliability, seed):
    public_order = public_candidate_permutation(row["sample_key"], seed)
    canonical_to_public = {canonical: public for public, canonical in enumerate(public_order)}
    parsed = [
        (canonical, canonical_to_public[canonical], row["candidates"][canonical_to_public[canonical]])
        for canonical in range(3)
        if row["candidates"][canonical_to_public[canonical]]["parse_ok"]
    ]
    if not parsed:
        return canonical_to_public[0]
    action_counts = Counter(candidate["action"] for _, _, candidate in parsed)
    highest = max(action_counts.values())
    tied = {action for action, count in action_counts.items() if count == highest}
    priority = sorted(range(3), key=lambda index: (-reliability[index], index))
    return next(
        public_index for canonical in priority
        for source_index, public_index, candidate in parsed
        if source_index == canonical and candidate["action"] in tied
    )


def apply_android_policy(public, reliability, allowed_folds, seed):
    allowed_folds = set(allowed_folds)
    output = {}
    for sample_key in sorted(public):
        row = public[sample_key]
        if int(row["fold"]) not in allowed_folds:
            continue
        fallback = android_majority_index(row, reliability[row["setting"]], seed)
        if not 0 <= fallback < 3:
            raise ValueError(f"TriVUS invalid Android fallback: {sample_key}/{fallback}")
        output[sample_key] = fallback
    return output


def context_record(outer_fold, role, holdout_fold, fit_folds, sample_key, fallback_index):
    if role not in {"inner", "final"} or (role == "inner") != (holdout_fold is not None):
        raise ValueError("TriVUS invalid context role")
    phase = "final" if role == "final" else f"inner-{holdout_fold}"
    record = {
        "schema_version": 1,
        "context_key": f"outer-{outer_fold}/{phase}/{sample_key}",
        "sample_key": sample_key,
        "outer_fold": int(outer_fold),
        "role": role,
        "holdout_fold": None if holdout_fold is None else int(holdout_fold),
        "fit_folds": [int(fold) for fold in fit_folds],
        "fallback_index": int(fallback_index),
    }
    if set(record) != CONTEXT_FIELDS:
        raise AssertionError("TriVUS context schema drift")
    return record


def load_sealed_rows(manifest, folds, base, expected_rows, expected_keys=None):
    folds = tuple(sorted(set(int(fold) for fold in folds)))
    if not folds or any(fold not in range(5) for fold in folds):
        raise ValueError(f"TriVUS invalid sealed folds: {folds}")
    output = {}
    opened = []
    for fold in folds:
        item = manifest["folds"][str(fold)]
        path = safe_child_path(base, item["path"])
        rows = load_jsonl(path)
        count = item.get("rows", item.get("records"))
        if len(rows) != count or count != expected_rows[fold] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"TriVUS sealed fold mismatch: {path}")
        for row in rows:
            key = row["sample_key"] if "sample_key" in row else row["row_id"]
            if key in output:
                raise ValueError(f"TriVUS duplicate sealed key: {key}")
            output[key] = row
        opened.append(str(path))
    if expected_keys is not None and set(output) != set(expected_keys):
        raise ValueError("TriVUS sealed fold identity mismatch")
    return output, opened