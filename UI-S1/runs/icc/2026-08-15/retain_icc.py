import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
BACKUP_ROOT = Path("/scratch/workspaceblobstore/icc/2026-08-15")
BACKUP_RUN = BACKUP_ROOT / "run"
STATUS_PATH = RUN_DIR / "STATUS.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if BACKUP_ROOT.exists() or STATUS_PATH.exists():
        raise FileExistsError("ICC retention destination exists")
    adjudication = json.loads((RUN_DIR / "ICC_ADJUDICATION.json").read_text())
    files = sorted(path for path in RUN_DIR.rglob("*") if path.is_file() and "__pycache__" not in path.parts and path != STATUS_PATH)
    artifacts = {}
    for source in files:
        relative = source.relative_to(RUN_DIR)
        destination = BACKUP_RUN / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        digest = sha256_file(source)
        if digest != sha256_file(destination):
            raise ValueError(f"ICC retention mismatch: {relative}")
        artifacts[relative.as_posix()] = {"source_path": source.relative_to(ROOT).as_posix(), "backup_path": str(destination), "bytes": source.stat().st_size, "sha256": digest}
    disclosure = ROOT / "docs/research_disclosures.md"
    disclosure_backup = BACKUP_ROOT / "external/docs/research_disclosures.md"
    disclosure_backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(disclosure, disclosure_backup)
    disclosure_digest = sha256_file(disclosure)
    if disclosure_digest != sha256_file(disclosure_backup):
        raise ValueError("ICC disclosure retention mismatch")
    manifest = {"schema_version": 1, "status": "LOCKED", "source_root": RUN_DIR.relative_to(ROOT).as_posix(), "backup_root": str(BACKUP_ROOT), "artifact_count": len(artifacts), "total_bytes": sum(value["bytes"] for value in artifacts.values()), "external_artifacts": {"docs/research_disclosures.md": {"backup_path": str(disclosure_backup), "bytes": disclosure.stat().st_size, "sha256": disclosure_digest}}, "verified_at_utc": datetime.now(timezone.utc).isoformat(), "artifacts": artifacts}
    manifest_path = BACKUP_ROOT / "BACKUP_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    status = {"schema_version": 1, "date": "2026-08-16", "round": "icc", "status": "COMPLETE", "outcome": adjudication["outcome"], "evidence_status": adjudication["evidence_status"], "gpu_used": False, "changes_existing_statuses": False, "method_claim_allowed": False, "historical_GRAN_G_P8_status": adjudication["historical_GRAN_G_P8_status"], "historical_status_changed": False, "report": adjudication["report"], "report_sha256": adjudication["report_sha256"], "adjudication_sha256": sha256_file(RUN_DIR / "ICC_ADJUDICATION.json"), "retention": {"manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path), "verified": True, "artifact_count": len(artifacts), "total_bytes": manifest["total_bytes"]}, "next_action": adjudication["next_action"]}
    STATUS_PATH.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "ICC_RETENTION_VERIFIED", "artifact_count": len(artifacts), "total_bytes": manifest["total_bytes"]}, indent=2))


if __name__ == "__main__":
    main()