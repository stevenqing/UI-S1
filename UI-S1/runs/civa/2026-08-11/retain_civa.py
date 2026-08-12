import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
BACKUP_ROOT = Path("/scratch/workspaceblobstore/civa/2026-08-11")
BACKUP_RUN = BACKUP_ROOT / "run"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retained_files():
    return sorted(
        path for path in RUN_DIR.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )


def main():
    if not Path("/proc/2274").exists():
        raise RuntimeError("protected PID 2274 absent")
    artifacts = {}
    for source in retained_files():
        relative = source.relative_to(RUN_DIR)
        destination = BACKUP_RUN / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hash = sha256_file(source)
        backup_hash = sha256_file(destination)
        if source_hash != backup_hash:
            raise ValueError(f"CIVA retention hash mismatch: {relative}")
        artifacts[relative.as_posix()] = {
            "source_path": source.relative_to(ROOT).as_posix(),
            "backup_path": str(destination),
            "bytes": source.stat().st_size,
            "sha256": source_hash,
        }
    manifest = {
        "schema_version": 1,
        "status": "LOCKED",
        "source_root": RUN_DIR.relative_to(ROOT).as_posix(),
        "backup_root": str(BACKUP_ROOT),
        "artifact_count": len(artifacts),
        "total_bytes": sum(value["bytes"] for value in artifacts.values()),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
    }
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = BACKUP_ROOT / "BACKUP_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("CIVA retention manifest readback mismatch")
    print(json.dumps({
        "status": "CIVA_RETENTION_VERIFIED",
        "artifact_count": manifest["artifact_count"],
        "total_bytes": manifest["total_bytes"],
        "manifest": str(manifest_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()