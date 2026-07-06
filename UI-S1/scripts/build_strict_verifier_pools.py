#!/usr/bin/env python3
"""Build strict TRAIN/TEST verifier per-step pools from sampled candidate pools."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.critstep_verifier_full_action import (  # noqa: E402
    build_step_record,
    key_for,
    load_index,
    load_scope_index,
    read_episodes,
    read_jsonl,
    write_jsonl,
)


DEFAULT_TRAIN_SAMPLES = "outputs/critstep_elicit_train/per_step.jsonl"
DEFAULT_TEST_SAMPLES = "outputs/critstep_elicit_uia/per_step.jsonl"
DEFAULT_TRAIN_DATA = "outputs/gui360_history_ab/original_eval/gui360_train_balanced_uia.jsonl"
DEFAULT_TEST_DATA = "outputs/gui360_history_ab/original_eval/gui360_test_1000_balanced_uia.jsonl"
DEFAULT_TEST_UIA = "outputs/critstep_reward_structure_uia/per_step.jsonl"
DEFAULT_TEST_SCOPE = "outputs/critstep_scope/per_step.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/critstep_verifier_v2/strict/pools"


def filtered_rows(path: Path, *, temperature: float, population: str, recoverable_only: bool) -> list[Dict[str, Any]]:
    rows = []
    for row in read_jsonl(path):
        if str(row.get("population")) != population:
            continue
        try:
            row_temperature = float(row.get("temperature"))
        except (TypeError, ValueError):
            continue
        if row_temperature != temperature:
            continue
        if recoverable_only and not row.get("recoverable"):
            continue
        rows.append(row)
    return rows


def build_pool(
    *,
    samples_path: Path,
    data_path: Path,
    output_path: Path,
    temperature: float,
    population: str,
    uia_path: Optional[Path] = None,
    scope_path: Optional[Path] = None,
    split_name: str,
) -> Dict[str, Any]:
    episodes = read_episodes(data_path)
    uia_by_key = load_index(uia_path) if uia_path and uia_path.exists() else {}
    scope_by_key = load_scope_index(scope_path) if scope_path and scope_path.exists() else {}
    rows = filtered_rows(samples_path, temperature=temperature, population=population, recoverable_only=True)
    records = []
    missing_episodes = []
    missing_uia = []
    for sample_row in rows:
        episode = episodes.get(str(sample_row.get("episode_id")))
        if episode is None:
            missing_episodes.append(str(sample_row.get("target_id")))
            continue
        row_key = key_for(sample_row)
        uia_row = uia_by_key.get(row_key) if uia_by_key else None
        if uia_by_key and uia_row is None:
            missing_uia.append(str(sample_row.get("target_id")))
        record = build_step_record(sample_row, episode, uia_row, scope_by_key.get(row_key))
        record["split"] = split_name
        record["episode_key"] = f"{split_name}:{record.get('episode_id')}"
        records.append(record)
    records.sort(key=lambda row: (str(row.get("episode_id")), int(row.get("step_idx") or 0), str(row.get("target_id"))))
    write_jsonl(output_path, records)
    subsets = Counter(str(row.get("subset")) for row in records)
    depths = Counter(str(row.get("depth_bin")) for row in records)
    episodes_out = {str(row.get("episode_key")) for row in records}
    return {
        "samples_path": str(samples_path),
        "data_path": str(data_path),
        "output_path": str(output_path),
        "temperature": temperature,
        "population": population,
        "split": split_name,
        "recoverable_rows_in": len(rows),
        "records_out": len(records),
        "episodes": len(episodes_out),
        "subset_counts": dict(subsets),
        "depth_counts": dict(depths),
        "missing_episodes": missing_episodes[:20],
        "missing_episode_count": len(missing_episodes),
        "missing_uia_count": len(missing_uia),
        "missing_uia_examples": missing_uia[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-samples", default=DEFAULT_TRAIN_SAMPLES)
    parser.add_argument("--test-samples", default=DEFAULT_TEST_SAMPLES)
    parser.add_argument("--train-data", default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--test-data", default=DEFAULT_TEST_DATA)
    parser.add_argument("--test-uia", default=DEFAULT_TEST_UIA)
    parser.add_argument("--test-scope", default=DEFAULT_TEST_SCOPE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--population", default="critical")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = build_pool(
        samples_path=Path(args.train_samples),
        data_path=Path(args.train_data),
        output_path=output_dir / "train_per_step.jsonl",
        temperature=args.temperature,
        population=args.population,
        split_name="train",
    )
    test = build_pool(
        samples_path=Path(args.test_samples),
        data_path=Path(args.test_data),
        output_path=output_dir / "test_per_step.jsonl",
        temperature=args.temperature,
        population=args.population,
        uia_path=Path(args.test_uia),
        scope_path=Path(args.test_scope),
        split_name="test",
    )
    train_episodes = {str(row["episode_key"]) for row in read_jsonl(output_dir / "train_per_step.jsonl")}
    test_episodes = {str(row["episode_key"]) for row in read_jsonl(output_dir / "test_per_step.jsonl")}
    manifest = {
        "train": train,
        "test": test,
        "episode_split": {
            "train_episodes": len(train_episodes),
            "test_episodes": len(test_episodes),
            "intersection": sorted(train_episodes & test_episodes)[:20],
            "intersection_count": len(train_episodes & test_episodes),
            "split_source": "upstream GUI-360 train/test episode boundary",
        },
        "leakage_note": "Subset/depth labels are retained for evaluation only; scorer prompts use instruction, screenshot, history, candidate actions, and UIA metadata.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()