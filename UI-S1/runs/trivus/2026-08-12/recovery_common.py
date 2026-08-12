import hashlib
import json
import os
from pathlib import Path

import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CONFIG_PATH = RUN_DIR / "configs/recovery.yaml"
REQUIRED_PREDICTION_FIELDS = {"action", "value", "position", "parse_ok", "pixel_position"}
FORBIDDEN_IMPORT_PREFIXES = (
    "evaluation", "qwenvl_utils", "scoring", "f3_androidcontrol_aggregator",
)


def assert_protected_process(config):
    expected = config["protected_process"]
    process = Path(f"/proc/{expected['pid']}")
    if not process.is_dir():
        raise RuntimeError("protected process absent")
    stat = (process / "stat").read_text()
    closing_parenthesis = stat.rfind(")")
    fields = stat[closing_parenthesis + 2:].split()
    observed = {
        "pid": expected["pid"],
        "start_ticks": int(fields[19]),
        "comm": (process / "comm").read_text().strip(),
        "cmdline_sha256": hashlib.sha256((process / "cmdline").read_bytes()).hexdigest(),
        "executable": str((process / "exe").resolve()),
    }
    if observed != expected:
        raise RuntimeError(f"protected process identity mismatch: {observed}")
    return observed


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def _verify_file(item):
    path = ROOT / item["path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != item["sha256"]:
        raise ValueError(f"TriVUS R0 hash mismatch: {path}/{observed}/{item['sha256']}")
    return path


def load_config():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    validate_config(config)
    return config


def references(config, setting):
    return load_jsonl(ROOT / config["references"][setting]["path"])


def validate_config(config):
    if config.get("status") != "FROZEN_BEFORE_RECOVERY":
        raise ValueError("TriVUS R0 protocol is not frozen")
    if config.get("expected_rows_per_lane") != 2000:
        raise ValueError("TriVUS R0 row contract mismatch")
    if set(config.get("protected_process", {})) != {
        "pid", "start_ticks", "comm", "cmdline_sha256", "executable"
    }:
        raise ValueError("TriVUS R0 protected-process contract mismatch")
    if config.get("execution") != {
        "num_shards": 1,
        "shard_index": 0,
        "batch_size": 8,
        "resume": True,
        "sampling_temperature": 0.0,
        "max_tokens": 256,
        "dtype": "bfloat16",
        "kv_cache_bytes": 2147483648,
    }:
        raise ValueError("TriVUS R0 execution contract mismatch")
    expected_prohibitions = {
        "no_scoring", "no_evaluator_import", "no_private_label_join",
        "no_accuracy_or_oracle_computation", "no_historical_file_mutation",
    }
    if set(config.get("prohibitions", ())) != expected_prohibitions:
        raise ValueError("TriVUS R0 prohibition contract mismatch")
    _verify_file(config["source_script"])
    roster_path = _verify_file(config["roster"])
    roster = yaml.safe_load(roster_path.read_text())
    model_specs = {model["id"]: model for model in roster["androidcontrol"]["models"]}
    for setting, item in config["references"].items():
        path = _verify_file(item)
        rows = load_jsonl(path)
        if len(rows) != 2000 or len({row["id"] for row in rows}) != 2000:
            raise ValueError(f"TriVUS R0 reference identity mismatch: {setting}")
        if [row["stable_index"] for row in rows] != list(range(2000)):
            raise ValueError(f"TriVUS R0 reference order mismatch: {setting}")
    for name, lane in config["lanes"].items():
        seed = ROOT / lane["seed_path"]
        if sha256_file(seed) != lane["seed_sha256"] or seed.stat().st_size != lane["seed_bytes"]:
            raise ValueError(f"TriVUS R0 seed byte mismatch: {name}")
        rows = load_jsonl(seed)
        if len(rows) != lane["seed_rows"] or lane["seed_rows"] + lane["missing_rows"] != 2000:
            raise ValueError(f"TriVUS R0 seed row mismatch: {name}")
        model = model_specs[lane["model_id"]]
        model_path = ROOT / model["local_path"] / "model.safetensors.index.json"
        if model["revision"] != lane["model_revision"] or sha256_file(model_path) != lane["model_index_sha256"]:
            raise ValueError(f"TriVUS R0 model mismatch: {name}")
        destination = ROOT / lane["destination"]
        if destination.resolve() == seed.resolve():
            raise ValueError(f"TriVUS R0 destination aliases history: {name}")
        validate_lane_rows(rows, references(config, lane["setting"]), lane, require_complete=False)
    for name, lane in config["complete_lanes"].items():
        model = model_specs[lane["model_id"]]
        model_path = ROOT / model["local_path"] / "model.safetensors.index.json"
        if model["revision"] != lane["model_revision"] or sha256_file(model_path) != lane["model_index_sha256"]:
            raise ValueError(f"TriVUS R0 complete model mismatch: {name}")
        rows = []
        for item in lane["shards"]:
            path = _verify_file(item)
            if path.stat().st_size != item["bytes"]:
                raise ValueError(f"TriVUS R0 complete byte mismatch: {path}")
            shard_rows = load_jsonl(path)
            if len(shard_rows) != item["rows"]:
                raise ValueError(f"TriVUS R0 complete row mismatch: {path}")
            rows.extend(shard_rows)
        validate_lane_rows(rows, references(config, lane["setting"]), lane, require_complete=True)


def validate_lane_rows(rows, reference_rows, lane, require_complete):
    expected_count = 2000 if require_complete else lane["seed_rows"]
    if len(rows) != expected_count:
        raise ValueError(f"TriVUS R0 lane count mismatch: {lane['model_id']}/{lane['setting']}/{len(rows)}")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"TriVUS R0 duplicate IDs: {lane['model_id']}/{lane['setting']}")
    stable_indices = [row["stable_index"] for row in rows]
    if len(set(stable_indices)) != len(rows) or sorted(stable_indices) != list(range(expected_count)):
        raise ValueError(f"TriVUS R0 stable-index coverage mismatch: {lane['model_id']}/{lane['setting']}")
    ordered = sorted(rows, key=lambda row: row["stable_index"])
    expected_references = reference_rows if require_complete else reference_rows[:expected_count]
    if [
        (row["stable_index"], row["id"]) for row in ordered
    ] != [
        (row["stable_index"], row["id"]) for row in expected_references
    ]:
        raise ValueError(f"TriVUS R0 row identity/order mismatch: {lane['model_id']}/{lane['setting']}")
    for row, reference in zip(ordered, expected_references):
        expected = {
            "stable_index": reference["stable_index"],
            "id": reference["id"],
            "episode_id": reference["episode_id"],
            "setting": lane["setting"],
            "source_index": reference["source_index"],
            "source_sha256": reference["source_sha256"],
            "image_sha256": reference["image_sha256"],
            "image_size": reference["image_size"],
            "model_id": lane["model_id"],
            "model_revision": lane["model_revision"],
            "model_index_sha256": lane["model_index_sha256"],
        }
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(f"TriVUS R0 provenance mismatch: {lane['model_id']}/{lane['setting']}/{reference['id']}/{key}")
        if not REQUIRED_PREDICTION_FIELDS.issubset(row.get("prediction", {})):
            raise ValueError(f"TriVUS R0 prediction schema mismatch: {reference['id']}")
        for key in ("text_prompt_sha256", "model_prompt_sha256"):
            value = row.get(key, "")
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"TriVUS R0 prompt hash mismatch: {reference['id']}/{key}")
        if require_complete and lane["model_id"] == "UI-AGILE-7B":
            if row.get("num_shards") != 2 or row.get("shard_index") != reference["stable_index"] % 2:
                raise ValueError(f"TriVUS R0 complete shard mismatch: {reference['id']}")
        elif row.get("num_shards") != 1 or row.get("shard_index") != 0:
            raise ValueError(f"TriVUS R0 single shard mismatch: {reference['id']}")
    return ordered