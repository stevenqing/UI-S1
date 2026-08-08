import fcntl
import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
SOURCE_ROOT = ROOT / "runs/xfer/2026-08-07/raw/ac-stage1"
BACKUP_ROOT = Path("/scratch/workspaceblobstore/aggmatch-traces/2026-08-09")
MANIFEST_PATH = BACKUP_ROOT / "BACKUP_MANIFEST.json"
UPSTREAM_STATUS = ROOT / "runs/close/2026-08-08/AC_PAUSE_STATUS.json"
SOURCE_WRITER = ROOT / "runs/xfer/2026-08-07/infer/ac_stage1_vllm.py"
MODELS = {
    "ui-agile": "UI-AGILE-7B",
    "gui-r1": "GUI-R1-7B",
    "ui-r1-e": "UI-R1-E-3B",
}
REQUIRED_FIELDS = {
    "id", "episode_id", "setting", "source_sha256", "image_sha256",
    "image_size", "model_id", "model_revision", "prediction",
}
REQUIRED_PREDICTION_FIELDS = {"action", "value", "position", "parse_ok"}


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
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


@contextmanager
def manifest_lock():
    lock_path = MANIFEST_PATH.with_suffix(".json.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def validate_shard(path, expected_model, expected_setting):
    identities = set()
    rows = 0
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = REQUIRED_FIELDS - set(row)
            if missing:
                raise ValueError(f"missing fields at {path}:{line_number}: {sorted(missing)}")
            prediction_missing = REQUIRED_PREDICTION_FIELDS - set(row["prediction"])
            if prediction_missing:
                raise ValueError(f"missing prediction fields at {path}:{line_number}: {sorted(prediction_missing)}")
            if row["id"] in identities:
                raise ValueError(f"duplicate row id within shard: {path}/{row['id']}")
            if row["model_id"] != expected_model or row["setting"] != expected_setting:
                raise ValueError(f"lane identity mismatch at {path}:{line_number}")
            identities.add(row["id"])
            rows += 1
    return rows, identities


def copy_fsynced(source, destination, expected_sha):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    if sha256_file(temporary) != expected_sha:
        raise ValueError(f"backup hash mismatch: {source}")
    temporary.replace(destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def lane_sha(paths):
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def verify_writer_contract():
    source = SOURCE_WRITER.read_text()
    required = ('output.write(', 'output.flush()', 'os.fsync(output.fileno())')
    if not all(fragment in source for fragment in required):
        raise ValueError("AC source writer no longer demonstrates row write/flush/fsync")
    return {"path": str(SOURCE_WRITER.relative_to(ROOT)), "sha256": sha256_file(SOURCE_WRITER)}


def main():
    upstream = json.loads(UPSTREAM_STATUS.read_text())
    if upstream["status"] != "CANCELLED_BY_E_K1":
        raise ValueError("AndroidControl must remain cancelled during F0")
    writer = verify_writer_contract()
    archived_at = datetime.now(timezone.utc).isoformat()
    lanes = {}
    artifacts = {}
    with manifest_lock():
        for directory, model_id in MODELS.items():
            for setting in ("low", "high"):
                lane_key = f"{directory}/{setting}"
                paths = sorted((SOURCE_ROOT / directory / setting).glob("*.jsonl"))
                if not paths:
                    raise FileNotFoundError(lane_key)
                lane_ids = set()
                shard_records = []
                for source in paths:
                    rows, identities = validate_shard(source, model_id, setting)
                    overlap = lane_ids & identities
                    if overlap:
                        raise ValueError(f"duplicate ids across shards in {lane_key}: {sorted(overlap)[:3]}")
                    lane_ids.update(identities)
                    digest = sha256_file(source)
                    relative = source.relative_to(ROOT)
                    destination = BACKUP_ROOT / relative
                    copy_fsynced(source, destination, digest)
                    record = {
                        "source_path": str(relative),
                        "backup_path": str(destination),
                        "rows": rows,
                        "bytes": source.stat().st_size,
                        "sha256": digest,
                    }
                    shard_records.append(record)
                    artifacts[str(relative)] = record
                expected = upstream["completed_rows"][lane_key]
                if len(lane_ids) != expected:
                    raise ValueError(f"lane row mismatch: {lane_key}/{len(lane_ids)} != {expected}")
                lanes[lane_key] = {
                    "model_id": model_id,
                    "setting": setting,
                    "rows": len(lane_ids),
                    "shards": shard_records,
                    "lane_sha256": lane_sha(paths),
                    "row_ids_sha256": hashlib.sha256("\n".join(sorted(lane_ids)).encode()).hexdigest(),
                }
        preserved_artifacts = {}
        if MANIFEST_PATH.exists():
            existing_manifest = json.loads(MANIFEST_PATH.read_text())
            preserved_artifacts = {
                key: value
                for key, value in existing_manifest.get("artifacts", {}).items()
                if not key.startswith("runs/xfer/2026-08-07/raw/ac-stage1/")
            }
        manifest = {
            "schema_version": 1,
            "status": "LOCKED",
            "archived_at_utc": archived_at,
            "backup_root": str(BACKUP_ROOT),
            "artifacts": {**preserved_artifacts, **artifacts},
            "lanes": lanes,
        }
        atomic_json(MANIFEST_PATH, manifest)

    result = {
        "schema_version": 1,
        "status": "PASS",
        "archived_at_utc": archived_at,
        "source_root": str(SOURCE_ROOT.relative_to(ROOT)),
        "backup_root": str(BACKUP_ROOT),
        "backup_manifest": str(MANIFEST_PATH),
        "source_writer_row_flush_fsync_verified": writer,
        "lanes": lanes,
        "total_rows_across_lanes": sum(lane["rows"] for lane in lanes.values()),
        "androidcontrol_four_arm_status": "CANCELLED_BY_E_K1",
        "cleanup_protection": {
            "recursive_raw_deletion": "PROHIBITED",
            "protected_patterns": ["raw/", "predictions*.jsonl", "*.jsonl"],
        },
    }
    atomic_json(RUN_DIR / "f0_ac_archive.json", result)
    status = {
        "schema_version": 1,
        "status": "F0_ARCHIVED",
        "workdir": str(RUN_DIR.relative_to(ROOT)),
        "ac_archive": {
            "result": "f0_ac_archive.json",
            "backup_root": str(BACKUP_ROOT),
            "backup_manifest": str(MANIFEST_PATH),
            "lanes": {key: {"rows": value["rows"], "lane_sha256": value["lane_sha256"]} for key, value in lanes.items()},
        },
        "androidcontrol_four_arm_status": "CANCELLED_NO_RESUME",
        "cleanup_protection": result["cleanup_protection"],
    }
    atomic_json(RUN_DIR / "STATUS.json", status)
    result_sha = sha256_file(RUN_DIR / "f0_ac_archive.json")
    copy_fsynced(RUN_DIR / "f0_ac_archive.json", BACKUP_ROOT / "results/f0_ac_archive.json", result_sha)
    print(json.dumps({"status": "PASS", "lanes": {key: value["rows"] for key, value in lanes.items()}, "backup_root": str(BACKUP_ROOT)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
