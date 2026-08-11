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
BACKUP_ROOT = Path("/scratch/workspaceblobstore/visual-utility-selector/2026-08-11")
MANIFEST_PATH = BACKUP_ROOT / "BACKUP_MANIFEST.json"
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache"}


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


def source_files():
    return sorted(
        path for path in RUN_DIR.rglob("*")
        if path.is_file()
        and not any(part in EXCLUDED_PARTS for part in path.relative_to(RUN_DIR).parts)
        and path.suffix != ".pyc"
    )


def copy_verified(source, destination, digest):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256_file(destination) == digest:
        return
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    if sha256_file(temporary) != digest:
        raise ValueError(f"backup mismatch: {source}")
    temporary.replace(destination)


def snapshot():
    artifacts = {}
    total_bytes = 0
    for source in source_files():
        relative = source.relative_to(RUN_DIR)
        digest = sha256_file(source)
        destination = BACKUP_ROOT / "run" / relative
        copy_verified(source, destination, digest)
        size = source.stat().st_size
        total_bytes += size
        artifacts[str(relative)] = {
            "source_path": str(source.relative_to(ROOT)),
            "backup_path": str(destination),
            "bytes": size,
            "sha256": digest,
        }
    manifest = {
        "schema_version": 1,
        "status": "LOCKED",
        "source_root": str(RUN_DIR.relative_to(ROOT)),
        "backup_root": str(BACKUP_ROOT),
        "artifact_count": len(artifacts),
        "total_bytes": total_bytes,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
    }
    with manifest_lock():
        atomic_json(MANIFEST_PATH, manifest)
    return {
        "status": "PASS",
        "artifacts": len(artifacts),
        "bytes": total_bytes,
        "backup_root": str(BACKUP_ROOT),
    }


def verify():
    with manifest_lock():
        manifest = json.loads(MANIFEST_PATH.read_text())
        current = {str(path.relative_to(RUN_DIR)): path for path in source_files()}
        recorded = manifest["artifacts"]
        if set(current) != set(recorded):
            raise ValueError(f"retention file-set mismatch: current={len(current)} recorded={len(recorded)}")
        for relative, source in current.items():
            record = recorded[relative]
            destination = Path(record["backup_path"])
            if not destination.is_file():
                raise FileNotFoundError(destination)
            digest = sha256_file(source)
            if digest != record["sha256"] or sha256_file(destination) != digest:
                raise ValueError(f"retention digest mismatch: {relative}")
    return {
        "status": "PASS",
        "artifacts": len(recorded),
        "bytes": sum(record["bytes"] for record in recorded.values()),
        "backup_root": str(BACKUP_ROOT),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("snapshot", "verify"))
    args = parser.parse_args()
    result = snapshot() if args.command == "snapshot" else verify()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
