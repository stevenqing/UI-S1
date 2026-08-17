import hashlib
import json
import os
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
OUTPUT_PATH = RUN_DIR / "CWIN_ADJUDICATION.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    stage0 = json.loads((RUN_DIR / "STAGE0.json").read_text())
    all_k = json.loads((RUN_DIR / "STAGE0_ALL_K.json").read_text())
    if (
        stage0["W_G1"] is not True
        or stage0["W_K5"] is not True
        or stage0["gpu_authorized"] is not False
        or [row["selected_K"] for row in stage0["selections"]] != [4] * 5
        or all_k["selected_K_reproduces_stage0"] is not True
    ):
        raise ValueError("CWIN adjudication input mismatch")
    report_path = RUN_DIR / "REPORT.md"
    output = {
        "schema_version": 1,
        "date": "2026-08-17",
        "round": "cwin",
        "status": "COMPLETE_STAGE0",
        "outcome": "CWIN_STAGE0_W_G1_PASS_W_K5_ENDPOINT_AMENDMENT_REQUIRED_GPU_UNAUTHORIZED",
        "evidence_status": "POST_SELECTION_EXPLORATORY_PILOT",
        "changes_existing_statuses": False,
        "gpu_used": False,
        "gpu_authorized": False,
        "method_claim_allowed": False,
        "selected_K": {str(row["outer_fold"]): row["selected_K"] for row in stage0["selections"]},
        "W_G1": True,
        "W_K5": True,
        "L4_upper": stage0["L4_upper"],
        "L4_conservative": stage0["L4_conservative"],
        "report": str(report_path.relative_to(ROOT)),
        "report_sha256": sha256_file(report_path),
        "stage0_sha256": sha256_file(RUN_DIR / "STAGE0.json"),
        "all_k_sha256": sha256_file(RUN_DIR / "STAGE0_ALL_K.json"),
        "window_manifest_sha256": sha256_file(RUN_DIR / "WINDOW_MANIFEST.json"),
        "reporting_recovery_declared": True,
        "next_action": "WRITE_SEPARATE_STAGE1_AMENDMENT_NO_GPU_BEFORE_EXPLICIT_AUTHORIZATION",
    }
    temporary = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(OUTPUT_PATH)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()