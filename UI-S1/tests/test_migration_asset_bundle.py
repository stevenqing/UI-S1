from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
for path in (str(REPO_ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from build_migration_asset_bundle import BRIDGE_FILES, SAFETY_FILES, UPSTREAM_FILES, collect_files
from restore_migration_asset_bundle import safe_relative


def test_safe_relative_rejects_traversal() -> None:
    assert safe_relative("outputs/example.jsonl") == Path("outputs/example.jsonl")
    for unsafe in ("/absolute/path", "../escape", "outputs/../../escape"):
        try:
            safe_relative(unsafe)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe path was accepted: {unsafe}")


def test_bundle_collects_all_required_groups() -> None:
    files, groups = collect_files(REPO_ROOT)
    assert set(UPSTREAM_FILES) <= set(files)
    assert set(BRIDGE_FILES) <= set(files)
    assert set(SAFETY_FILES) <= set(files)
    assert groups["pass8_frozen_directory_replace"]
    assert len(files) == len(set(files)) == 62
