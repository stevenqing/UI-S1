#!/usr/bin/env python3
"""Attach raw GUI-360 UIA control metadata to the balanced test JSONL.

The balanced/reconstructed test JSONL used by the original-template evaluator
contains local screenshot copies and GT actions, but it does not carry the raw
`control_infos`. This script joins each balanced test step back to the raw
GUI-360 JSONL by using the original screenshot path from the balanced parquet
files, then writes an enriched JSONL with `control_infos` on each step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


DEFAULT_RECONSTRUCTED = "outputs/gui360_history_ab/original_eval/gui360_test_1000_balanced_reconstructed.jsonl"
DEFAULT_BALANCED_PARQUETS = "datasets/gui360-balanced/data/test-*.parquet"
DEFAULT_RAW_ROOT = "datasets/GUI-360-raw-jsonl/test/data"
DEFAULT_OUTPUT = "outputs/gui360_history_ab/original_eval/gui360_test_1000_balanced_uia.jsonl"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_steps(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, list):
        return value
    raise TypeError(f"unsupported steps value: {type(value)!r}")


def load_balanced_steps(pattern: str) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    paths = sorted(Path().glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no parquet files matched {pattern}")
    for path in paths:
        frame = pd.read_parquet(path)
        for _, row in frame.iterrows():
            out[str(row["episode_id"])] = parse_steps(row["steps"])
    return out


def parse_original_screenshot(path_value: str) -> Optional[Tuple[str, str, str, str, int]]:
    """Return (app, category, status, execution_id, step_id) from original image path."""

    parts = Path(path_value).parts
    try:
        idx = parts.index("image")
    except ValueError:
        return None
    if idx + 5 >= len(parts):
        return None
    app = parts[idx + 1]
    category = parts[idx + 2]
    status = parts[idx + 3]
    execution_id = parts[idx + 4]
    filename = parts[idx + 5]
    if not filename.startswith("action_step") or not filename.endswith(".png"):
        return None
    try:
        step_id = int(filename.removeprefix("action_step").removesuffix(".png"))
    except ValueError:
        return None
    return app, category, status, execution_id, step_id


def action_step_id(path_value: str) -> Optional[int]:
    filename = Path(path_value).name
    if not filename.startswith("action_step") or not filename.endswith(".png"):
        return None
    try:
        return int(filename.removeprefix("action_step").removesuffix(".png").removesuffix("_annotated"))
    except ValueError:
        return None


def load_raw_episode(raw_root: Path, app: str, category: str, status: str, execution_id: str) -> Dict[int, Dict[str, Any]]:
    raw_path = raw_root / app / category / status / f"{execution_id}.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"missing raw trajectory {raw_path}")
    out = {}
    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            raw_step = row.get("step") or {}
            key = action_step_id(str(raw_step.get("screenshot_clean") or ""))
            if key is None:
                key = int(row.get("step_id") or 0)
            out[key] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstructed", default=DEFAULT_RECONSTRUCTED)
    parser.add_argument("--balanced-parquets", default=DEFAULT_BALANCED_PARQUETS)
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    reconstructed = read_jsonl(Path(args.reconstructed))
    balanced_steps = load_balanced_steps(args.balanced_parquets)
    raw_root = Path(args.raw_root)
    raw_cache: Dict[Tuple[str, str, str, str], Dict[int, Dict[str, Any]]] = {}

    total_steps = 0
    joined_steps = 0
    missing_controls = 0
    enriched = []
    for episode in reconstructed:
        episode_id = str(episode["episode_id"])
        original_steps = balanced_steps.get(episode_id)
        if original_steps is None:
            raise ValueError(f"missing balanced parquet episode {episode_id}")
        if len(original_steps) != len(episode.get("steps") or []):
            raise ValueError(f"step count mismatch for episode {episode_id}: {len(original_steps)} vs {len(episode.get('steps') or [])}")
        new_episode = dict(episode)
        new_steps = []
        for step, original_step in zip(episode.get("steps") or [], original_steps):
            total_steps += 1
            parsed = parse_original_screenshot(str(original_step.get("screenshot") or ""))
            if parsed is None:
                raise ValueError(f"could not parse original screenshot for episode {episode_id} step {step.get('step_idx')}: {original_step.get('screenshot')}")
            app, category, status, execution_id, step_id = parsed
            cache_key = (app, category, status, execution_id)
            if cache_key not in raw_cache:
                raw_cache[cache_key] = load_raw_episode(raw_root, app, category, status, execution_id)
            raw_row = raw_cache[cache_key].get(step_id)
            if raw_row is None:
                raise ValueError(f"missing raw step {step_id} for {cache_key}")
            raw_step = raw_row.get("step") or {}
            control_infos = raw_step.get("control_infos") or {}
            controls = control_infos.get("uia_controls_info") or control_infos.get("merged_controls_info") or []
            if not controls:
                missing_controls += 1
            else:
                joined_steps += 1
            new_step = dict(step)
            new_step["control_infos"] = control_infos
            new_step["raw_execution_id"] = execution_id
            new_step["raw_step_id"] = step_id
            new_step["raw_app"] = app
            new_step["raw_category"] = category
            new_step["raw_status"] = status
            new_step["raw_screenshot_clean"] = raw_step.get("screenshot_clean")
            new_step["raw_screenshot_annotated"] = raw_step.get("screenshot_annotated")
            new_step["raw_action"] = raw_step.get("action")
            new_steps.append(new_step)
        new_episode["steps"] = new_steps
        enriched.append(new_episode)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in enriched:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "output": str(output),
        "episodes": len(enriched),
        "total_steps": total_steps,
        "joined_steps_with_controls": joined_steps,
        "missing_controls": missing_controls,
        "raw_trajectories_loaded": len(raw_cache),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()