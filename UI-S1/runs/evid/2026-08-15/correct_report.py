import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
REPORT_PATH = RUN_DIR / "REPORT.md"
ADJUDICATION_PATH = RUN_DIR / "EVID_ADJUDICATION.json"
STATUS_PATH = RUN_DIR / "STATUS.json"
NOTE_PATH = RUN_DIR / "CORRECTION_001_REPORT_KATEX.md"
SUPPLEMENT_ROOT = Path("/scratch/workspaceblobstore/evid/2026-08-15/correction-001")

EXPECTED_REPORT_SHA = "293fb824417a0c501b0bf9dc2de2b06f038f98f0a650f880bdd4104e26593213"
EXPECTED_ADJUDICATION_SHA = "caf5c5573910f863c19491e4f2d3651a65f8af887450b588962dedf4c8d909bf"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if SUPPLEMENT_ROOT.exists():
        raise FileExistsError(SUPPLEMENT_ROOT)
    if sha256_file(REPORT_PATH) != EXPECTED_REPORT_SHA or sha256_file(ADJUDICATION_PATH) != EXPECTED_ADJUDICATION_SHA:
        raise ValueError("EVID pre-correction hashes mismatch")
    report = REPORT_PATH.read_text()
    old = "The separated $2\to3$ lineage marginal"
    new = "The separated $2\\to3$ lineage marginal"
    if report.count(old) != 1:
        raise ValueError("EVID report correction anchor mismatch")
    REPORT_PATH.write_text(report.replace(old, new))
    adjudication = json.loads(ADJUDICATION_PATH.read_text())
    adjudication["report_sha256"] = sha256_file(REPORT_PATH)
    adjudication["formatting_correction_001"] = True
    ADJUDICATION_PATH.write_text(json.dumps(adjudication, indent=2, sort_keys=True) + "\n")
    status = json.loads(STATUS_PATH.read_text())
    status["report_sha256"] = adjudication["report_sha256"]
    status["adjudication_sha256"] = sha256_file(ADJUDICATION_PATH)

    artifacts = {}
    for source in (REPORT_PATH, ADJUDICATION_PATH, NOTE_PATH):
        destination = SUPPLEMENT_ROOT / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        digest = sha256_file(source)
        if digest != sha256_file(destination):
            raise ValueError(f"EVID correction retention mismatch: {source.name}")
        artifacts[source.name] = {"backup_path": str(destination), "bytes": source.stat().st_size, "sha256": digest}
    manifest = {
        "schema_version": 1,
        "status": "LOCKED_EVID_FORMATTING_CORRECTION",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_count": len(artifacts),
        "total_bytes": sum(value["bytes"] for value in artifacts.values()),
        "artifacts": artifacts,
    }
    manifest_path = SUPPLEMENT_ROOT / "BACKUP_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    status["retention"]["original_manifest_preserved"] = True
    status["retention"]["correction_manifest"] = str(manifest_path)
    status["retention"]["correction_manifest_sha256"] = sha256_file(manifest_path)
    status["retention"]["correction_verified"] = True
    STATUS_PATH.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "EVID_REPORT_CORRECTION_COMPLETE", "report_sha256": status["report_sha256"], "correction_manifest_sha256": status["retention"]["correction_manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    main()