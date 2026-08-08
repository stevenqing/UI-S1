import hashlib
import json
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
PATTERNS = (
    "raw/mind2web-proposer-ablation.jsonl",
    "raw/ac-proposer-ablation.jsonl",
    "raw/mind2web-consensus-roi.jsonl",
    "raw/stage1/tongui/shard-*.jsonl",
    "raw/stage1/uitars/shard-*.jsonl",
    "raw/stage1/cogagent/shard-*.jsonl",
    "raw/stage1/view1/tongui/shard-*.jsonl",
    "raw/stage1/view1/uitars/shard-*.jsonl",
    "raw/stage1/view1/cogagent/shard-*.jsonl",
    "raw/stage2/tongui/shard-*.jsonl",
    "raw/stage2/uitars/shard-*.jsonl",
    "raw/stage2/cogagent/shard-*.jsonl",
    "raw/views/tongui/shard-*.jsonl",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_rows(path):
    rows = 0
    identities = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        identity = value.get("id", value.get("row_id", value.get("index")))
        if identity is None or identity in identities:
            raise ValueError(f"invalid identity at {path}:{line_number}")
        identities.add(identity)
        rows += 1
    return rows


def main():
    paths = sorted({path for pattern in PATTERNS for path in RUN_DIR.glob(pattern)})
    if len(paths) != 47:
        raise ValueError(f"expected 47 publication traces, found {len(paths)}")
    artifacts = {}
    for path in paths:
        relative = str(path.relative_to(RUN_DIR))
        artifacts[relative] = {
            "bytes": path.stat().st_size,
            "rows": jsonl_rows(path),
            "sha256": sha256_file(path),
        }
    result = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "Mind2Web Q1 paired row-level reproducibility plus AC proposer ablation",
        "artifact_count": len(artifacts),
        "total_bytes": sum(value["bytes"] for value in artifacts.values()),
        "artifacts": artifacts,
        "excluded": [
            "model weights",
            "benchmark images and archives",
            "AndroidControl source parquet files",
            "incomplete AndroidControl formal inference traces",
        ],
        "durable_full_trace_backup": "/scratch/workspaceblobstore/xfer-traces/2026-08-07/BACKUP_MANIFEST.json",
    }
    output = RUN_DIR / "PUBLICATION_MANIFEST.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "artifacts": result["artifact_count"],
        "total_bytes": result["total_bytes"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()