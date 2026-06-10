from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> Dict[str, Any]:
    cfg_path = resolve_path(path)
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    data["_config_path"] = str(cfg_path)
    return data


def resolve_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    out = resolve_path(path) if not Path(path).is_absolute() else Path(path)
    assert out is not None
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
