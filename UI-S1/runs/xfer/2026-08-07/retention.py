import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CONFIG_PATH = RUN_DIR / "configs/retention.yaml"


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
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            identity = row.get("id", row.get("row_id", row.get("index")))
            if identity is None:
                raise ValueError(f"missing identity at {path}:{line_number}")
            if identity in identities:
                raise ValueError(f"duplicate identity at {path}:{line_number}: {identity}")
            identities.add(identity)
            rows += 1
    return rows


def atomic_write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_manifest(path):
    if not path.exists():
        return {"schema_version": 1, "status": "ACTIVE", "artifacts": {}}
    return json.loads(path.read_text())


def backup(source, relative):
    config = yaml.safe_load(CONFIG_PATH.read_text())
    backup_root = Path(config["backup"]["root"])
    manifest_path = Path(config["backup"]["manifest"])
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = jsonl_rows(source)
    digest = sha256_file(source)
    destination = backup_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    if sha256_file(temporary) != digest:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"backup hash mismatch: {source}")
    temporary.replace(destination)
    manifest = load_manifest(manifest_path)
    manifest["artifacts"][str(relative)] = {
        "local_path": str(source),
        "backup_path": str(destination),
        "rows": rows,
        "bytes": source.stat().st_size,
        "sha256": digest,
        "backup_verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest["artifacts"][str(relative)]


def verify():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    manifest_path = Path(config["backup"]["manifest"])
    manifest = load_manifest(manifest_path)
    for relative, record in manifest["artifacts"].items():
        local = Path(record["local_path"])
        remote = Path(record["backup_path"])
        if not local.is_file() or not remote.is_file():
            raise FileNotFoundError(relative)
        local_hash = sha256_file(local)
        remote_hash = sha256_file(remote)
        if local_hash != record["sha256"] or remote_hash != record["sha256"]:
            raise ValueError(f"retention verification mismatch: {relative}")
    return {"status": "PASS", "artifacts": len(manifest["artifacts"])}


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    backup_parser = subparsers.add_parser("backup")
    backup_parser.add_argument("--source", type=Path, required=True)
    backup_parser.add_argument("--relative", type=Path, required=True)
    subparsers.add_parser("verify")
    args = parser.parse_args()
    result = backup(args.source, args.relative) if args.command == "backup" else verify()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
