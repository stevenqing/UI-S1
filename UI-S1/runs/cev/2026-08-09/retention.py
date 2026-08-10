import argparse
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
BACKUP_ROOT = Path("/scratch/workspaceblobstore/cev-traces/2026-08-10")
MANIFEST_PATH = BACKUP_ROOT / "BACKUP_MANIFEST.json"


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


def backup(source, relative):
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = sha256_file(source)
    destination = BACKUP_ROOT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    if sha256_file(temporary) != digest:
        raise ValueError(f"backup mismatch: {source}")
    temporary.replace(destination)
    try:
        recorded_source = str(source.relative_to(ROOT))
    except ValueError:
        recorded_source = str(source)
    with manifest_lock():
        manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {"schema_version": 1, "status": "LOCKED", "artifacts": {}}
        record = {
            "source_path": recorded_source,
            "backup_path": str(destination),
            "bytes": source.stat().st_size,
            "sha256": digest,
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest["artifacts"][str(relative)] = record
        atomic_json(MANIFEST_PATH, manifest)
    return record


def verify():
    with manifest_lock():
        manifest = json.loads(MANIFEST_PATH.read_text())
        for relative, record in manifest["artifacts"].items():
            source = Path(record["source_path"])
            if not source.is_absolute():
                source = ROOT / source
            destination = Path(record["backup_path"])
            if not source.is_file() or not destination.is_file():
                raise FileNotFoundError(relative)
            if sha256_file(source) != record["sha256"] or sha256_file(destination) != record["sha256"]:
                raise ValueError(f"retention mismatch: {relative}")
    return {"status": "PASS", "artifacts": len(manifest["artifacts"]), "backup_root": str(BACKUP_ROOT)}


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    backup_parser = commands.add_parser("backup")
    backup_parser.add_argument("--source", type=Path, required=True)
    backup_parser.add_argument("--relative", type=Path, required=True)
    commands.add_parser("verify")
    args = parser.parse_args()
    result = backup(args.source, args.relative) if args.command == "backup" else verify()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()