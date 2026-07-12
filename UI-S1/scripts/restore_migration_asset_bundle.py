#!/usr/bin/env python3
"""Safely restore and verify a UI-S1 bridge migration asset bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


METADATA_PATH = "_migration_bundle/asset_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(name: str) -> Path:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe archive path: {name}")
    return Path(*pure.parts)


def read_manifest(archive: tarfile.TarFile) -> dict[str, Any]:
    member = archive.getmember(METADATA_PATH)
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError("asset manifest is unreadable")
    return json.loads(handle.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--report", default="outputs/migration_verification/restored_assets.json")
    args = parser.parse_args()

    bundle = Path(args.bundle).resolve()
    root = Path(args.root).resolve()
    with tarfile.open(bundle, mode="r:gz") as archive:
        manifest = read_manifest(archive)
        expected = {str(row["path"]): dict(row) for row in manifest.get("files") or []}
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        missing_members = sorted(set(expected) - set(members))
        if missing_members:
            raise ValueError(f"bundle is missing {len(missing_members)} payload members")
        actions = []
        with tempfile.TemporaryDirectory(prefix="ui_s1_restore_") as temporary:
            temp_root = Path(temporary)
            for relative, row in expected.items():
                safe = safe_relative(relative)
                destination = root / safe
                if destination.exists() and sha256(destination) == row["sha256"]:
                    actions.append({"path": relative, "action": "kept", "ok": True})
                    continue
                if args.verify_only:
                    actions.append({"path": relative, "action": "missing_or_mismatched", "ok": False})
                    continue
                if destination.exists() and not args.overwrite:
                    actions.append({"path": relative, "action": "conflict_requires_overwrite", "ok": False})
                    continue
                member = members[relative]
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot read payload: {relative}")
                temporary_path = temp_root / safe
                temporary_path.parent.mkdir(parents=True, exist_ok=True)
                with temporary_path.open("wb") as handle:
                    shutil.copyfileobj(source, handle)
                actual = sha256(temporary_path)
                if actual != row["sha256"] or temporary_path.stat().st_size != int(row["bytes"]):
                    raise ValueError(f"payload checksum mismatch: {relative}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary_path.replace(destination)
                actions.append({"path": relative, "action": "restored", "ok": True})
    failures = [row for row in actions if not row["ok"]]
    report = {
        "bundle": str(bundle),
        "bundle_sha256": sha256(bundle),
        "root": str(root),
        "verify_only": args.verify_only,
        "overwrite": args.overwrite,
        "files": len(actions),
        "kept": sum(row["action"] == "kept" for row in actions),
        "restored": sum(row["action"] == "restored" for row in actions),
        "failures": failures,
        "ok": not failures,
        "actions": actions,
    }
    report_path = root / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("bundle", "root", "files", "kept", "restored", "ok", "failures")}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()