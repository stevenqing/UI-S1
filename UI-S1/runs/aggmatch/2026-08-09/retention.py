import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from f0_ac_archive import BACKUP_ROOT, MANIFEST_PATH, ROOT, atomic_json, copy_fsynced, manifest_lock, sha256_file


def source_path(record):
    path = Path(record["source_path"])
    return path if path.is_absolute() else ROOT / path


def backup(source, relative):
    source = source.resolve()
    try:
        recorded_source = str(source.relative_to(ROOT))
    except ValueError:
        recorded_source = str(source)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = BACKUP_ROOT / relative
    digest = sha256_file(source)
    copy_fsynced(source, destination, digest)
    with manifest_lock():
        manifest = json.loads(MANIFEST_PATH.read_text())
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
            source = source_path(record)
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