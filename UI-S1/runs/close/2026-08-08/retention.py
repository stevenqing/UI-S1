import argparse
import fcntl
import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import yaml


RUN_DIR = Path(__file__).resolve().parent
CONFIG = RUN_DIR / "configs/retention.yaml"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_rows(path):
    if path.suffix != ".jsonl":
        return None
    rows = 0
    identities = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        identity = value.get("id", value.get("row_id", value.get("index")))
        if identity is None or identity in identities:
            raise ValueError(f"invalid JSONL identity at {path}:{line_number}")
        identities.add(identity)
        rows += 1
    return rows


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


@contextmanager
def locked(path, exclusive=True):
    lock = path.with_suffix(path.suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_manifest(path):
    return json.loads(path.read_text()) if path.exists() else {"schema_version": 1, "status": "ACTIVE", "artifacts": {}}


def backup(source, relative):
    config = yaml.safe_load(CONFIG.read_text())
    root = Path(config["backup_root"])
    manifest_path = Path(config["manifest"])
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with locked(manifest_path):
        digest = sha256_file(source)
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        if sha256_file(temporary) != digest:
            raise ValueError(f"backup hash mismatch: {source}")
        temporary.replace(destination)
        manifest = load_manifest(manifest_path)
        manifest["artifacts"][str(relative)] = {
            "local_path": str(source),
            "backup_path": str(destination),
            "rows": jsonl_rows(source),
            "bytes": source.stat().st_size,
            "sha256": digest,
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(manifest_path, manifest)
        return manifest["artifacts"][str(relative)]


def verify():
    config = yaml.safe_load(CONFIG.read_text())
    manifest_path = Path(config["manifest"])
    with locked(manifest_path, exclusive=False):
        manifest = load_manifest(manifest_path)
        for relative, record in manifest["artifacts"].items():
            local = Path(record["local_path"])
            remote = Path(record["backup_path"])
            if not local.is_file() or not remote.is_file():
                raise FileNotFoundError(relative)
            if sha256_file(local) != record["sha256"] or sha256_file(remote) != record["sha256"]:
                raise ValueError(f"verification mismatch: {relative}")
        return {"status": "PASS", "artifacts": len(manifest["artifacts"])}


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