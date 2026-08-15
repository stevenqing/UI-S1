import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
REPORT_PATH = RUN_DIR / "REPORT.md"
ADJUDICATION_PATH = RUN_DIR / "XSCR_ADJUDICATION.json"
STATUS_PATH = RUN_DIR / "STATUS.json"
CORRECTION_PATH = RUN_DIR / "CORRECTION_004_HOLDOUT_LABEL_ACCESS.md"
SUPPLEMENT_ROOT = Path("/scratch/workspaceblobstore/xscr/2026-08-14/correction-001")

EXPECTED_REPORT_SHA = "0979270c8946575d2d21a9f2d4d176f3c3fd0a2251efa84061ec3cd91595d2ac"
EXPECTED_ADJUDICATION_SHA = "000ef78f9c57db729b48e57aed9f7d264c7b1cceff6de3bef0782524dbbf4c63"
EXPECTED_STATUS_OUTCOME = "XSCR_COMPLETE_BELOW_MDE_EXPLORATORY_SPEC_AUTHORIZED"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if SUPPLEMENT_ROOT.exists():
        raise FileExistsError(SUPPLEMENT_ROOT)
    if sha256_file(REPORT_PATH) != EXPECTED_REPORT_SHA or sha256_file(ADJUDICATION_PATH) != EXPECTED_ADJUDICATION_SHA:
        raise ValueError("XSCR pre-correction report/adjudication mismatch")
    status = json.loads(STATUS_PATH.read_text())
    if status["outcome"] != EXPECTED_STATUS_OUTCOME:
        raise ValueError("XSCR pre-correction status mismatch")

    report = REPORT_PATH.read_text()
    old = "That future round remains post-selection, must evaluate only after freezing the method against the prospective internal holdout, cannot claim confirmation, and cannot enter the existing main table as a same-protocol improvement."
    new = "That future round remains post-selection and cannot claim confirmation or enter the existing main table as a same-protocol improvement. Correction 004 determined that all private-label files were parsed during input locking, so the nominal 30% subset is not an unread prospective holdout. Any current-data follow-up must use explicitly post-selection nested evaluation; independent validation requires new untouched data."
    if report.count(old) != 1:
        raise ValueError("XSCR report correction anchor mismatch")
    report = report.replace(old, new)
    report += "\n## Protocol correction\n\nThe seal excluded holdout screens from every reported aggregate, but the private-input locker and Q3/Q4 loader read all private-label and reference rows into memory. The holdout is therefore contaminated for future evaluation. See `CORRECTION_004_HOLDOUT_LABEL_ACCESS.md`.\n"
    REPORT_PATH.write_text(report)

    adjudication = json.loads(ADJUDICATION_PATH.read_text())
    adjudication.update({
        "outcome": "XSCR_COMPLETE_BELOW_MDE_HOLDOUT_CONTAMINATED_EXPLORATORY_SPEC_AUTHORIZED",
        "holdout_excluded_from_reported_statistics": True,
        "holdout_labels_read": True,
        "prospective_internal_holdout_valid": False,
        "independent_validation_requires_new_data": True,
        "report_sha256": sha256_file(REPORT_PATH),
    })
    ADJUDICATION_PATH.write_text(json.dumps(adjudication, indent=2, sort_keys=True) + "\n")

    files = [REPORT_PATH, ADJUDICATION_PATH, CORRECTION_PATH]
    artifacts = {}
    for source in files:
        destination = SUPPLEMENT_ROOT / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        digest = sha256_file(source)
        if digest != sha256_file(destination):
            raise ValueError(f"XSCR correction retention mismatch: {source.name}")
        artifacts[source.name] = {"backup_path": str(destination), "bytes": source.stat().st_size, "sha256": digest}
    supplement = {
        "schema_version": 1,
        "status": "LOCKED_XSCR_HOLDOUT_CORRECTION",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_count": len(artifacts),
        "total_bytes": sum(value["bytes"] for value in artifacts.values()),
        "artifacts": artifacts,
    }
    supplement_path = SUPPLEMENT_ROOT / "BACKUP_MANIFEST.json"
    supplement_path.write_text(json.dumps(supplement, indent=2, sort_keys=True) + "\n")

    status.update({
        "outcome": adjudication["outcome"],
        "report_sha256": adjudication["report_sha256"],
        "adjudication_sha256": sha256_file(ADJUDICATION_PATH),
        "holdout_excluded_from_reported_statistics": True,
        "holdout_labels_read": True,
        "prospective_internal_holdout_valid": False,
        "independent_validation_requires_new_data": True,
        "next_action": "WRITE_NESTED_EXPLORATORY_SOFT_ASSIGNMENT_SPEC_NO_HOLDOUT_OR_METHOD_CLAIM",
    })
    status["retention"]["original_manifest_preserved"] = True
    status["retention"]["correction_manifest"] = str(supplement_path)
    status["retention"]["correction_manifest_sha256"] = sha256_file(supplement_path)
    status["retention"]["correction_verified"] = True
    STATUS_PATH.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "XSCR_HOLDOUT_CORRECTION_COMPLETE",
        "outcome": status["outcome"],
        "prospective_internal_holdout_valid": False,
        "correction_manifest": str(supplement_path),
        "correction_manifest_sha256": status["retention"]["correction_manifest_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()