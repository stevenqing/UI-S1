import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
BACKUP_ROOT = Path("/scratch/workspaceblobstore/cwin/2026-08-17")
BACKUP_RUN = BACKUP_ROOT / "run"
STATUS_PATH = RUN_DIR / "STATUS.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main():
    if BACKUP_ROOT.exists() or STATUS_PATH.exists():
        raise FileExistsError("CWIN retention destination exists")
    adjudication = json.loads((RUN_DIR / "CWIN_ADJUDICATION.json").read_text())
    files = sorted(
        path
        for path in RUN_DIR.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path != STATUS_PATH
    )
    artifacts = {}
    for source in files:
        relative = source.relative_to(RUN_DIR)
        destination = BACKUP_RUN / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        digest = sha256_file(source)
        if digest != sha256_file(destination):
            raise ValueError(f"CWIN retention mismatch: {relative}")
        artifacts[relative.as_posix()] = {
            "source_path": source.relative_to(ROOT).as_posix(),
            "backup_path": str(destination),
            "bytes": source.stat().st_size,
            "sha256": digest,
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
    manifest_path = BACKUP_ROOT / "BACKUP_MANIFEST.json"
    atomic_json(manifest_path, manifest)
    status = {
        "schema_version": 1,
        "date": "2026-08-17",
        "round": "cwin",
        "status": "COMPLETE_STAGE0",
        "outcome": adjudication["outcome"],
        "evidence_status": adjudication["evidence_status"],
        "gpu_used": False,
        "gpu_authorized": False,
        "changes_existing_statuses": False,
        "method_claim_allowed": False,
        "report": adjudication["report"],
        "report_sha256": adjudication["report_sha256"],
        "adjudication_sha256": sha256_file(RUN_DIR / "CWIN_ADJUDICATION.json"),
        "retention": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "verified": True,
            "artifact_count": len(artifacts),
            "total_bytes": manifest["total_bytes"],
        },
        "W_G1": adjudication["W_G1"],
        "W_K5": adjudication["W_K5"],
        "next_action": adjudication["next_action"],
    }
    atomic_json(STATUS_PATH, status)
    print(
        json.dumps(
            {
                "status": "CWIN_RETENTION_VERIFIED",
                "artifact_count": len(artifacts),
                "total_bytes": manifest["total_bytes"],
                "gpu_authorized": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()