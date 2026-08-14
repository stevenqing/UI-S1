import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
BACKUP_ROOT = Path("/scratch/workspaceblobstore/mask/2026-08-14")
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
        raise FileExistsError("MASK retention destination or status exists")
    adjudication_path = RUN_DIR / "MASK_ADJUDICATION.json"
    adjudication = json.loads(adjudication_path.read_text())
    if adjudication.get("outcome") != "MASK_STOPPED_M_K1_IDEAL_NEFF_GAIN_BELOW_MDE":
        raise PermissionError("MASK adjudication is not final")
    files = sorted(
        path for path in RUN_DIR.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts and path != STATUS_PATH
    )
    artifacts = {}
    for source in files:
        relative = source.relative_to(RUN_DIR)
        destination = BACKUP_RUN / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        digest = sha256_file(source)
        if digest != sha256_file(destination):
            raise ValueError(f"MASK retention hash mismatch: {relative}")
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
        "total_bytes": sum(row["bytes"] for row in artifacts.values()),
        "upstream_inputs_copied": False,
        "model_weights_copied": False,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
    }
    manifest_path = BACKUP_ROOT / "BACKUP_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("MASK retention readback mismatch")
    status = {
        "schema_version": 1,
        "date": "2026-08-14",
        "round": "mask",
        "status": "COMPLETE_PRE_GPU_STOP",
        "outcome": adjudication["outcome"],
        "M_G1": adjudication["M_G1"],
        "base_rates": adjudication["base_rates"],
        "kill_conditions": adjudication["kill_conditions"],
        "model_forward_count": 0,
        "subset_manifest_created": False,
        "gpu_authorization_created": False,
        "adjudication": "runs/mask/2026-08-14/MASK_ADJUDICATION.json",
        "adjudication_sha256": sha256_file(adjudication_path),
        "report": "runs/mask/2026-08-14/REPORT.md",
        "report_sha256": sha256_file(RUN_DIR / "REPORT.md"),
        "retention": {
            "backup_root": str(BACKUP_ROOT),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "verified": True,
            "artifact_count": manifest["artifact_count"],
            "total_bytes": manifest["total_bytes"],
            "upstream_inputs_copied": False,
            "model_weights_copied": False,
        },
        "next_action": "CLOSE_MASK_NO_GPU_DO_NOT_RESCUE_WITH_ALTERNATE_MASK_OR_GATE",
    }
    STATUS_PATH.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "MASK_RETENTION_VERIFIED",
        "artifact_count": manifest["artifact_count"],
        "total_bytes": manifest["total_bytes"],
        "manifest": str(manifest_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()