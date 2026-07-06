#!/usr/bin/env python3
"""Create prefix-N verifier pools from an all-step E2E candidate pool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_critstep_verifier_v2_cot_voting import candidate_distinct_key  # noqa: E402


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_value(candidate: Mapping[str, Any], vote_k: int) -> float:
    value = candidate.get(f"stage1_score_k{vote_k}")
    if value is None:
        value = candidate.get("stage1_score")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def recompute_stage1_fields(row: Dict[str, Any], vote_ks: Sequence[int]) -> None:
    candidates = row.get("candidates") if isinstance(row.get("candidates"), list) else []
    if not candidates:
        return
    for vote_k in vote_ks:
        best = max(candidates, key=lambda candidate: (score_value(candidate, vote_k), str(candidate.get("candidate_id"))))
        row[f"stage1_k{vote_k}_candidate_id"] = best.get("candidate_id")
        row[f"stage1_k{vote_k}_distinct_key"] = candidate_distinct_key(best)
        row[f"stage1_k{vote_k}_correct"] = bool(best.get("is_correct"))
    if vote_ks:
        best_k = max(vote_ks)
        row["stage1_best_k"] = best_k
        row["stage1_best_candidate_id"] = row.get(f"stage1_k{best_k}_candidate_id")
        row["stage1_best_distinct_key"] = row.get(f"stage1_k{best_k}_distinct_key")
        row["stage1_best_correct"] = row.get(f"stage1_k{best_k}_correct")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-candidates", type=int, required=True)
    parser.add_argument("--vote-ks", default="8", help="comma-separated Stage1 vote counts to recompute if score fields exist")
    args = parser.parse_args()

    vote_ks = [int(item.strip()) for item in args.vote_ks.split(",") if item.strip()]
    rows = []
    for row in read_jsonl(Path(args.input)):
        row = dict(row)
        candidates = row.get("candidates") if isinstance(row.get("candidates"), list) else []
        row["candidates"] = candidates[: args.n_candidates]
        row["n_candidates"] = len(row["candidates"])
        row["n_correct_candidates"] = sum(1 for candidate in row["candidates"] if candidate.get("is_correct"))
        row["correct_candidate_ids"] = [candidate.get("candidate_id") for candidate in row["candidates"] if candidate.get("is_correct")]
        recompute_stage1_fields(row, vote_ks)
        rows.append(row)
    write_jsonl(Path(args.output), rows)
    summary = {
        "input": args.input,
        "output": args.output,
        "rows": len(rows),
        "n_candidates": args.n_candidates,
        "vote_ks": vote_ks,
    }
    Path(args.output).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()