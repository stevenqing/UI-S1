#!/usr/bin/env python3
"""Build a deterministic archive of Git-ignored Pass@8 bridge assets."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Iterable


UPSTREAM_FILES = (
    "outputs/rl_feasibility/per_step.jsonl",
    "outputs/multiagent_complementarity/target_ids.json",
    "outputs/multiagent_complementarity/qwen3_vl_candidates.jsonl",
    "outputs/multiagent_complementarity/qwen35_candidates.jsonl",
    "outputs/multiagent_complementarity/extra_tiers/llava15_candidates.jsonl",
)

BRIDGE_FILES = (
    "outputs/multiagent_trajectory_revision/full_v1/causal_arms/a1_gt_target_gt_history.jsonl",
    "outputs/multiagent_trajectory_revision/full_v1/causal_arms/a5_revision_target_gt_history.jsonl",
    "outputs/multiagent_trajectory_revision/full_v1/causal_eval/a5_gt_history_grid/merged.jsonl",
    "outputs/multiagent_trajectory_revision/full_v1/utility_gate/a13_oracle_student_rescue_gt_history.jsonl",
    "outputs/multiagent_trajectory_revision/full_v1/utility_gate/a15_student_rescue25_replay75.jsonl",
)

SAFETY_FILES = (
    "outputs/validation_2k/data/train_episodes.jsonl",
)

PASS8_DIRECTORY = "outputs/pass8_selector_study"
METADATA_PREFIX = "_migration_bundle"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalized_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.mode = 0o755 if info.isdir() else 0o644
    return info


def add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    normalized_info(info)
    archive.addfile(info, io.BytesIO(payload))


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except subprocess.CalledProcessError:
        return None


def collect_files(root: Path) -> tuple[list[str], dict[str, list[str]]]:
    pass8_root = root / PASS8_DIRECTORY
    if not pass8_root.is_dir():
        raise FileNotFoundError(pass8_root)
    pass8_files = sorted(str(path.relative_to(root)) for path in pass8_root.rglob("*") if path.is_file())
    groups = {
        "pass8_upstream_must_copy": list(UPSTREAM_FILES),
        "pass8_frozen_directory_replace": pass8_files,
        "purity_bridge_must_copy": list(BRIDGE_FILES),
        "train_manifest_safety_copy": list(SAFETY_FILES),
    }
    files = sorted({path for paths in groups.values() for path in paths})
    missing = [relative for relative in files if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError("missing bundle inputs: " + ", ".join(missing))
    return files, groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="outputs/migration_bundle")
    parser.add_argument("--name", default="ui_s1_bridge_missing_assets_v1")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    files, groups = collect_files(root)
    rows = []
    for relative in files:
        path = root / relative
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest: dict[str, Any] = {
        "version": "ui-s1-bridge-missing-assets-v1",
        "source_git_commit": git_commit(root),
        "restore_root": "UI-S1 project root",
        "groups": groups,
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "notes": [
            "The train manifest is included as a safety copy; a correct existing hash may be kept.",
            "GUI screenshots, parquet datasets, environments, and checkpoints are intentionally excluded.",
            "Pass@8 runtime logs are included because the source directory is only about 14 MB; they are not required for analysis.",
        ],
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    sums_text = "".join(f"{row['sha256']}  {row['path']}\n" for row in rows)
    sums_bytes = sums_text.encode("utf-8")
    archive_path = out_dir / f"{args.name}.tar.gz"
    with archive_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                add_bytes(archive, f"{METADATA_PREFIX}/asset_manifest.json", manifest_bytes)
                add_bytes(archive, f"{METADATA_PREFIX}/SHA256SUMS", sums_bytes)
                for row in rows:
                    path = root / row["path"]
                    info = normalized_info(archive.gettarinfo(str(path), arcname=row["path"]))
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)
    archive_hash = sha256(archive_path)
    external_manifest = {
        **manifest,
        "archive": archive_path.name,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_hash,
        "embedded_manifest_sha256": sha256_bytes(manifest_bytes),
        "embedded_sha256s_sha256": sha256_bytes(sums_bytes),
    }
    manifest_path = out_dir / f"{args.name}.manifest.json"
    sums_path = out_dir / f"{args.name}.SHA256SUMS"
    manifest_path.write_text(json.dumps(external_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sums_path.write_text(sums_text, encoding="utf-8")
    print(json.dumps({
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_hash,
        "manifest": str(manifest_path),
        "files": len(rows),
        "uncompressed_bytes": manifest["total_bytes"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()