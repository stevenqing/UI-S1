import hashlib
import json
import os
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
BACKUP_ROOT = Path("/scratch/workspaceblobstore/cwin/2026-08-17")
PARTIAL_RUN = BACKUP_ROOT / "run"
ARCHIVE_PATH = BACKUP_ROOT / "cwin-run.tar"
MANIFEST_PATH = BACKUP_ROOT / "BACKUP_MANIFEST.json"
LOCAL_ARCHIVE_PATH = Path("/tmp/cwin-2026-08-17-run.tar")
STATUS_PATH = RUN_DIR / "STATUS.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_stream(handle):
    digest = hashlib.sha256()
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
    if not PARTIAL_RUN.exists():
        raise FileNotFoundError("CWIN partial retention attempt is missing")
    if any(path.exists() for path in (ARCHIVE_PATH, MANIFEST_PATH, STATUS_PATH, LOCAL_ARCHIVE_PATH)):
        raise FileExistsError("CWIN archive recovery destination exists")
    adjudication = json.loads((RUN_DIR / "CWIN_ADJUDICATION.json").read_text())
    files = sorted(
        path
        for path in RUN_DIR.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path != STATUS_PATH
    )
    artifacts = {}
    with tarfile.open(LOCAL_ARCHIVE_PATH, "w", format=tarfile.PAX_FORMAT) as archive:
        for source in files:
            relative = source.relative_to(RUN_DIR).as_posix()
            digest = sha256_file(source)
            artifacts[relative] = {
                "source_path": source.relative_to(ROOT).as_posix(),
                "archive_path": str(ARCHIVE_PATH),
                "archive_member": relative,
                "bytes": source.stat().st_size,
                "sha256": digest,
            }
            archive.add(source, arcname=relative, recursive=False)
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LOCAL_ARCHIVE_PATH, ARCHIVE_PATH)
    archive_sha256 = sha256_file(ARCHIVE_PATH)
    if archive_sha256 != sha256_file(LOCAL_ARCHIVE_PATH):
        raise ValueError("CWIN archive copy mismatch")
    seen = set()
    with tarfile.open(ARCHIVE_PATH, "r") as archive:
        for member in archive:
            if not member.isfile() or member.name not in artifacts or member.name in seen:
                raise ValueError(f"CWIN invalid archive member: {member.name}")
            expected = artifacts[member.name]
            extracted = archive.extractfile(member)
            if extracted is None or member.size != expected["bytes"] or sha256_stream(extracted) != expected["sha256"]:
                raise ValueError(f"CWIN archive member mismatch: {member.name}")
            seen.add(member.name)
    if seen != set(artifacts):
        raise ValueError("CWIN archive member set mismatch")
    partial_files = sorted(path for path in PARTIAL_RUN.rglob("*") if path.is_file())
    manifest = {
        "schema_version": 1,
        "status": "LOCKED_ARCHIVE_RECOVERY",
        "source_root": RUN_DIR.relative_to(ROOT).as_posix(),
        "backup_root": str(BACKUP_ROOT),
        "archive": {
            "path": str(ARCHIVE_PATH),
            "bytes": ARCHIVE_PATH.stat().st_size,
            "sha256": archive_sha256,
            "member_count": len(artifacts),
            "all_members_verified": True,
        },
        "failed_attempt": {
            "tool": "retain_cwin.py",
            "partial_root": str(PARTIAL_RUN),
            "partial_file_count_retained": len(partial_files),
            "partial_files_deleted": False,
            "reason": "high_latency_per_file_scratch_copy_interrupted",
        },
        "artifact_count": len(artifacts),
        "total_bytes": sum(value["bytes"] for value in artifacts.values()),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
    }
    atomic_json(MANIFEST_PATH, manifest)
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
            "manifest": str(MANIFEST_PATH),
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "archive": str(ARCHIVE_PATH),
            "archive_sha256": archive_sha256,
            "verified": True,
            "artifact_count": len(artifacts),
            "total_bytes": manifest["total_bytes"],
            "failed_attempt_retained": True,
        },
        "W_G1": adjudication["W_G1"],
        "W_K5": adjudication["W_K5"],
        "next_action": adjudication["next_action"],
    }
    atomic_json(STATUS_PATH, status)
    LOCAL_ARCHIVE_PATH.unlink()
    print(
        json.dumps(
            {
                "status": "CWIN_ARCHIVE_RETENTION_VERIFIED",
                "artifact_count": len(artifacts),
                "total_bytes": manifest["total_bytes"],
                "archive_bytes": ARCHIVE_PATH.stat().st_size,
                "partial_file_count_retained": len(partial_files),
                "gpu_authorized": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()