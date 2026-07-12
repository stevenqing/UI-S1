#!/usr/bin/env python3
"""Build a deployable ranker-selected revision plus clean-replay training arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["correction_id"]), int(row["step_idx"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-data", required=True, help="A5 revision target + GT history rows")
    parser.add_argument("--clean-replay-data", required=True)
    parser.add_argument("--train-scores", required=True)
    parser.add_argument("--calibration-summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", default="a17_ranker25_replay75")
    parser.add_argument("--revision-rows", type=int, default=200)
    parser.add_argument("--replay-rows", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    calibration = read_json(Path(args.calibration_summary))
    if calibration.get("gate") != "POSITIVE_TEST_UTILITY":
        raise ValueError(f"ranker gate does not authorize training: {calibration.get('gate')}")
    threshold = float(calibration["selected_threshold"])
    revisions = {key(row): row for row in read_jsonl(Path(args.revision_data))}
    scores = read_jsonl(Path(args.train_scores))
    eligible = [row for row in scores if float(row["score"]) >= threshold and key(row) in revisions]
    eligible.sort(key=lambda row: (-float(row["score"]), str(row["correction_id"]), int(row["step_idx"])))
    if len(eligible) < args.revision_rows:
        raise ValueError(f"ranker accepted only {len(eligible)} train rows; need {args.revision_rows}")
    selected_scores = eligible[: args.revision_rows]
    selected_keys = {key(row) for row in selected_scores}
    revision_rows = []
    score_by_key = {key(row): row for row in selected_scores}
    for paired in selected_keys:
        row = dict(revisions[paired])
        row.update({
            "sample_id": f"{args.arm}:ranker_revision:{row['correction_id']}:{row['step_idx']}",
            "treatment_arm": args.arm,
            "screen_arm": args.arm,
            "research_role": "deployable_ranker_selected_revision_with_clean_replay",
            "mixture_source": "ranker_revision",
            "ranker_score": float(score_by_key[paired]["score"]),
            "ranker_threshold": threshold,
            "selection_uses_matcher": False,
            "semantic_quality_filter_used": False,
            "oracle_target_used": False,
        })
        revision_rows.append(row)

    replay_source = [row for row in read_jsonl(Path(args.clean_replay_data)) if key(row) not in selected_keys]
    actors = sorted({str(row["actor"]) for row in replay_source})
    if args.replay_rows % (2 * len(actors)):
        raise ValueError("replay rows must divide actor x prefix strata")
    per_stratum = args.replay_rows // (2 * len(actors))
    replay_rows = []
    for actor in actors:
        for clean in (True, False):
            group = [row for row in replay_source if str(row["actor"]) == actor and bool(row["prefix_clean"]) is clean]
            random.Random(f"{args.seed}:{actor}:{int(clean)}").shuffle(group)
            if len(group) < per_stratum:
                raise ValueError(f"insufficient clean replay stratum {actor}/{clean}")
            for source in group[:per_stratum]:
                row = dict(source)
                row.update({
                    "sample_id": f"{args.arm}:clean_replay:{row['correction_id']}:{row['step_idx']}",
                    "treatment_arm": args.arm,
                    "screen_arm": args.arm,
                    "research_role": "deployable_ranker_selected_revision_with_clean_replay",
                    "mixture_source": "clean_replay",
                })
                replay_rows.append(row)

    rows = revision_rows + replay_rows
    random.Random(args.seed).shuffle(rows)
    if len(rows) != args.revision_rows + args.replay_rows or len({str(row["sample_id"]) for row in rows}) != len(rows):
        raise ValueError("ranker/replay arm size or uniqueness failure")
    output = Path(args.output); write_jsonl(output, rows)
    summary = {
        "arm": args.arm,
        "research_role": "deployable_ranker_selected_revision_with_clean_replay",
        "rows": len(rows),
        "optimizer_steps": len(rows) // 8,
        "gradient_accumulation_steps": 8,
        "mixture_counts": dict(Counter(str(row["mixture_source"]) for row in rows)),
        "ranker_threshold": threshold,
        "ranker_eligible_train_rows": len(eligible),
        "ranker_selected_rows": len(revision_rows),
        "diagnostic_revision_accuracy_not_used_for_selection": sum(bool(row["diagnostic_matcher_correct"]) for row in revision_rows) / len(revision_rows),
        "overall_diagnostic_accuracy_not_used_for_selection": sum(bool(row["diagnostic_matcher_correct"]) for row in rows) / len(rows),
        "selection_uses_matcher": False,
        "semantic_quality_filter_used": False,
        "oracle_target_used": False,
        "output": str(output),
        "output_sha256": sha256(output),
        "score_source": args.train_scores,
        "score_source_sha256": sha256(Path(args.train_scores)),
        "calibration_summary": args.calibration_summary,
        "calibration_summary_sha256": sha256(Path(args.calibration_summary)),
    }
    write_json(output.with_suffix(".summary.json"), summary)
    write_json(output.with_name(f"{args.arm}_screen_manifest.json"), {
        "version": "ranker-selected-replay-screen-v1",
        "training_policy": {"same_update_budget": True, "optimizer_steps": summary["optimizer_steps"]},
        "arms": {args.arm: summary},
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
