import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
BACKUP_ROOT = Path("/scratch/workspaceblobstore/orth/2026-08-14")
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
        raise FileExistsError("ORTH retention destination exists")
    adjudication = json.loads((RUN_DIR / "ORTH_ADJUDICATION.json").read_text())
    if adjudication.get("status") != "COMPLETE_SCOPING":
        raise PermissionError("ORTH adjudication incomplete")
    files = sorted(
        path for path in RUN_DIR.rglob("*") if path.is_file()
        and "__pycache__" not in path.parts and path != STATUS_PATH
    )
    artifacts = {}
    for source in files:
        relative = source.relative_to(RUN_DIR)
        destination = BACKUP_RUN / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        digest = sha256_file(source)
        if digest != sha256_file(destination):
            raise ValueError(f"ORTH retention mismatch: {relative}")
        artifacts[relative.as_posix()] = {
            "source_path": source.relative_to(ROOT).as_posix(),
            "backup_path": str(destination), "bytes": source.stat().st_size,
            "sha256": digest,
        }
    manifest = {
        "schema_version": 1, "status": "LOCKED",
        "source_root": RUN_DIR.relative_to(ROOT).as_posix(),
        "backup_root": str(BACKUP_ROOT), "artifact_count": len(artifacts),
        "total_bytes": sum(value["bytes"] for value in artifacts.values()),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_images_copied": False, "ocr_models_copied": False,
        "artifacts": artifacts,
    }
    manifest_path = BACKUP_ROOT / "BACKUP_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    status = {
        "schema_version": 1, "date": "2026-08-14", "round": "orth",
        "status": "COMPLETE", "outcome": adjudication["outcome"],
        "exploratory_scoping_only": True, "paper_result_allowed": False,
        "runtime_rule_allowed": False, "changes_existing_statuses": False,
        "gpu_used": False,
        "adjudication": "runs/orth/2026-08-14/ORTH_ADJUDICATION.json",
        "adjudication_sha256": sha256_file(RUN_DIR / "ORTH_ADJUDICATION.json"),
        "report": "runs/orth/2026-08-14/REPORT.md",
        "report_sha256": sha256_file(RUN_DIR / "REPORT.md"),
        "retention": {
            "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
            "verified": True, "artifact_count": manifest["artifact_count"],
            "total_bytes": manifest["total_bytes"],
        },
        "next_action": adjudication["next_action"],
    }
    STATUS_PATH.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "ORTH_RETENTION_VERIFIED", "artifact_count": manifest["artifact_count"], "total_bytes": manifest["total_bytes"]}, indent=2))


if __name__ == "__main__":
    main()