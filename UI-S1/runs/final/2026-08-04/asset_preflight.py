import hashlib
import json
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]

ASSETS = {
    "M0_B1_B2_7B": [
        "runs/allocation-law/2026-08-01/raw/shared_regions_n12.jsonl",
        "runs/ccm-h2h/2026-07-31/h1/shards/top18",
        "runs/ccm-h2h/2026-07-31/h3/shards/qwen3_views",
        "runs/ccm-h2h/2026-07-31/h3/shards/uitars_views",
        "runs/allocation-law/2026-08-01/shards",
    ],
    "B1_B2_72B": [
        "runs/scaleup/2026-08-02/raw/g2-regions.jsonl",
        "runs/scaleup/2026-08-02/raw/g2-score-gta1.jsonl",
        "runs/scaleup/2026-08-02/raw/g2-score-venus.jsonl",
        "runs/scaleup/2026-08-02/raw/g2-score-qwen35.jsonl",
        "runs/ccm-h2h/2026-07-31/h3/raw/shared_regions_n4.jsonl",
    ],
    "T1_T2": [
        "runs/complementarity/2026-07-30/rows.parquet",
    ],
    "S0": [
        "runs/diversity-axis/2026-08-02/x7_confidence.json",
        "runs/reallocation/2026-08-03/r4_risk_coverage.json",
    ],
    "X1": [
        "runs/scaleup/2026-08-02/z5_sampling_axis.json",
    ],
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(path):
    absolute = ROOT / path
    exists = absolute.exists()
    if not exists:
        return {"path": path, "exists": False}
    if absolute.is_file():
        return {
            "path": path,
            "exists": True,
            "kind": "file",
            "bytes": absolute.stat().st_size,
            "sha256": sha256_file(absolute),
        }
    files = [value for value in absolute.rglob("*") if value.is_file()]
    return {
        "path": path,
        "exists": True,
        "kind": "directory",
        "files": len(files),
        "bytes": sum(value.stat().st_size for value in files),
    }


def main():
    groups = {}
    for name, paths in ASSETS.items():
        records = [inspect(path) for path in paths]
        groups[name] = {
            "status": "READY" if all(record["exists"] for record in records) else "BLOCKED_MISSING_ASSETS",
            "assets": records,
        }
    result = {
        "schema_version": 1,
        "status": "READY" if all(group["status"] == "READY" for group in groups.values()) else "PARTIAL",
        "groups": groups,
        "restore_policy": "copy exact Git-ignored frozen artifacts from the source machine; do not regenerate from aggregate summaries",
    }
    output = RUN_DIR / "ASSET_PREFLIGHT.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
