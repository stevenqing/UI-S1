import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
STAGE_ROOT = Path("/scratch/workspaceblobstore/hf-trajectory-publication/2026-08-14")
XFER_MANIFEST = Path("/scratch/workspaceblobstore/xfer-traces/2026-08-07/BACKUP_MANIFEST.json")
AGGMATCH_MANIFEST = Path("/scratch/workspaceblobstore/aggmatch-traces/2026-08-09/BACKUP_MANIFEST.json")
TRIVUS_MANIFEST = Path("/scratch/workspaceblobstore/trivus/2026-08-12/BACKUP_MANIFEST.json")

RELEASES = {
    "mind2web": {
        "repo_id": "Stevenshuqing/UI-S1-Mind2Web-Trajectories",
        "expected_files": 44,
    },
    "androidcontrol": {
        "repo_id": "Stevenshuqing/UI-S1-AndroidControl-Trajectories",
        "expected_files": 12,
    },
}

FORBIDDEN_KEYS = {
    "api_key",
    "candidate_success",
    "correct",
    "correctness",
    "ground_truth",
    "gt_action",
    "gt_point",
    "label",
    "labels",
    "private_label",
    "reward",
    "secret",
    "target_bbox",
    "token",
}
LOCAL_PATH_PREFIXES = ("/home/", "/mnt/", "/scratch/", "/tmp/")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def relative_source(info):
    source = info.get("source_path") or info.get("local_path")
    if not source:
        raise ValueError("retention artifact lacks source provenance")
    marker = "runs/"
    position = source.find(marker)
    if position < 0:
        raise ValueError(f"source is not under runs/: {source}")
    return source[position:]


def selected_artifacts():
    selected = {name: [] for name in RELEASES}
    sources = [
        ("xfer-2026-08-07", XFER_MANIFEST, "mind2web"),
        ("aggmatch-2026-08-09", AGGMATCH_MANIFEST, "androidcontrol"),
        ("trivus-2026-08-12", TRIVUS_MANIFEST, "androidcontrol"),
    ]
    for source_run, manifest_path, benchmark in sources:
        manifest = load_manifest(manifest_path)
        manifest_sha = sha256_file(manifest_path)
        for artifact_key, info in manifest["artifacts"].items():
            source_path = relative_source(info)
            include = False
            destination = None
            if benchmark == "mind2web":
                prefixes = (
                    "runs/xfer/2026-08-07/raw/stage1/",
                    "runs/xfer/2026-08-07/raw/stage2/",
                    "runs/xfer/2026-08-07/raw/views/",
                )
                include = source_path.startswith(prefixes) and source_path.endswith(".jsonl")
                if include:
                    destination = "data/xfer/" + source_path.split("/raw/", 1)[1]
            elif source_run.startswith("aggmatch"):
                marker = "runs/xfer/2026-08-07/raw/ac-stage1/"
                include = source_path.startswith(marker) and source_path.endswith(".jsonl")
                if include:
                    destination = "data/aggmatch/" + source_path[len(marker):]
            else:
                marker = "runs/trivus/2026-08-12/recovery/ac-stage1/"
                include = source_path.startswith(marker) and source_path.endswith(".jsonl")
                if include:
                    destination = "data/trivus-recovery/" + source_path[len(marker):]
            if include:
                selected[benchmark].append(
                    {
                        "artifact_key": artifact_key,
                        "backup_path": info["backup_path"],
                        "bytes": info["bytes"],
                        "destination": destination,
                        "manifest_sha256": manifest_sha,
                        "rows": info.get("rows"),
                        "sha256": info["sha256"],
                        "source_path": source_path,
                        "source_run": source_run,
                    }
                )
    for benchmark, records in selected.items():
        hashes = [record["sha256"] for record in records]
        destinations = [record["destination"] for record in records]
        if len(hashes) != len(set(hashes)):
            raise ValueError(f"duplicate SHA-256 in {benchmark} release")
        if len(destinations) != len(set(destinations)):
            raise ValueError(f"duplicate destination in {benchmark} release")
        if len(records) != RELEASES[benchmark]["expected_files"]:
            raise ValueError(f"unexpected {benchmark} file count: {len(records)}")
    return selected


def validate_value(value, key_path=""):
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = key.lower()
            if normalized in FORBIDDEN_KEYS or normalized.startswith("gt_") or normalized.startswith("private_"):
                raise ValueError(f"forbidden key at {key_path}{key}")
            validate_value(child, f"{key_path}{key}.")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            validate_value(child, f"{key_path}{index}.")
    elif isinstance(value, str) and value.startswith(LOCAL_PATH_PREFIXES):
        raise ValueError(f"absolute local path at {key_path.rstrip('.')}")


def validate_jsonl(path):
    rows = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL {path}:{line_number}") from exc
            validate_value(value)
            rows += 1
    if rows == 0:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def dataset_card(benchmark, repo_id, file_count, total_rows, total_bytes):
    title = "Mind2Web" if benchmark == "mind2web" else "AndroidControl"
    return f"""---
pretty_name: UI-S1 {title} Model Trajectories
---

# UI-S1 {title} Model Trajectories

Private archival release of label-blind GUI model trajectories produced by UI-S1 experiments.

## Contents

- Repository: `{repo_id}`
- JSONL shards: {file_count}
- Rows across shards: {total_rows}
- Bytes across shards: {total_bytes}
- Integrity metadata: `RELEASE_MANIFEST.json`

Each row contains stable identifiers and image hashes, model provenance, the raw model response, and a parsed prediction. Benchmark images, source archives, model weights, ground truth, candidate-success labels, rewards, correctness fields, selector outputs, and derived benchmark statistics are excluded.

The files do not include benchmark images. Join them to a separately licensed benchmark copy using stable IDs and image SHA-256 values.

## Access and redistribution

This repository is private pending review of benchmark and model-output redistribution terms. Do not make it public without a separate license review.

## Provenance

The release was assembled only from SHA-256-verified retention manifests. Raw JSONL bytes were preserved without record rewriting, and every row was parsed and checked for prohibited evaluation-side or local-path fields before upload.
"""


def atomic_write(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main():
    if STAGE_ROOT.exists():
        raise FileExistsError(f"publication stage already exists: {STAGE_ROOT}")
    selected = selected_artifacts()
    summary = {}
    for benchmark, records in selected.items():
        package = STAGE_ROOT / benchmark
        package.mkdir(parents=True)
        artifacts = {}
        total_rows = 0
        for record in sorted(records, key=lambda item: item["destination"]):
            source = Path(record["backup_path"])
            if source.stat().st_size != record["bytes"] or sha256_file(source) != record["sha256"]:
                raise ValueError(f"retention mismatch: {record['source_path']}")
            rows = validate_jsonl(source)
            if record["rows"] is not None and rows != record["rows"]:
                raise ValueError(f"row mismatch: {record['source_path']}")
            destination = package / record["destination"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if sha256_file(destination) != record["sha256"]:
                raise ValueError(f"copy mismatch: {record['destination']}")
            total_rows += rows
            artifacts[record["destination"]] = {
                "bytes": record["bytes"],
                "rows": rows,
                "sha256": record["sha256"],
                "source_artifact_key": record["artifact_key"],
                "source_manifest_sha256": record["manifest_sha256"],
                "source_path": record["source_path"],
                "source_run": record["source_run"],
            }
        total_bytes = sum(value["bytes"] for value in artifacts.values())
        release_manifest = {
            "schema_version": 1,
            "status": "VALIDATED_LABEL_BLIND_TRAJECTORY_RELEASE",
            "benchmark": benchmark,
            "repo_id": RELEASES[benchmark]["repo_id"],
            "private": True,
            "artifact_count": len(artifacts),
            "total_rows": total_rows,
            "total_bytes": total_bytes,
            "raw_bytes_preserved": True,
            "images_included": False,
            "ground_truth_included": False,
            "private_labels_included": False,
            "artifacts": artifacts,
        }
        atomic_write(package / "RELEASE_MANIFEST.json", json.dumps(release_manifest, indent=2, sort_keys=True) + "\n")
        atomic_write(package / "README.md", dataset_card(benchmark, RELEASES[benchmark]["repo_id"], len(artifacts), total_rows, total_bytes))
        summary[benchmark] = {
            "artifact_count": len(artifacts),
            "total_rows": total_rows,
            "total_bytes": total_bytes,
            "release_manifest_sha256": sha256_file(package / "RELEASE_MANIFEST.json"),
            "readme_sha256": sha256_file(package / "README.md"),
        }
    package_manifest = {
        "schema_version": 1,
        "status": "PASS_HF_TRAJECTORY_PACKAGES_VALIDATED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage_root": str(STAGE_ROOT),
        "releases": summary,
    }
    atomic_write(RUN_DIR / "PACKAGE_MANIFEST.json", json.dumps(package_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(package_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()