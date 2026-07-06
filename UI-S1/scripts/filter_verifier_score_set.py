#!/usr/bin/env python3
"""Filter an all-step candidate pool to steps whose verifier outcome can affect lift.

Under analysis-time bin gating, verifier compute is applied only to bin steps
where greedy is wrong. If the N-pool has no correct candidate, the verifier
cannot fix the step. Therefore only greedy-wrong + recoverable steps need
verifier scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


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


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--n-candidates", type=int, default=50)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    score_rows = []
    greedy_correct = 0
    missing = 0
    score_set_candidates = 0
    total_candidates = 0
    for row in rows:
        candidates = row.get("candidates") if isinstance(row.get("candidates"), list) else []
        candidates = candidates[: args.n_candidates]
        total_candidates += len(candidates)
        is_greedy_correct = bool(row.get("greedy_correct") if row.get("greedy_correct") is not None else (candidates and candidates[0].get("is_correct")))
        recoverable = any(bool(candidate.get("is_correct")) for candidate in candidates)
        if is_greedy_correct:
            greedy_correct += 1
            continue
        if not recoverable:
            missing += 1
            continue
        out = dict(row)
        out["candidates"] = candidates
        out["score_set_requires_verifier"] = True
        out["filter_reason"] = "greedy_wrong_recoverable"
        score_rows.append(out)
        score_set_candidates += len(candidates)
    summary = {
        "input": args.input,
        "output": args.output,
        "n_candidates": args.n_candidates,
        "total_steps": len(rows),
        "greedy_correct_skipped": greedy_correct,
        "greedy_wrong_missing_skipped": missing,
        "score_set_steps": len(score_rows),
        "total_candidate_jobs_unfiltered": total_candidates,
        "score_set_candidate_jobs": score_set_candidates,
        "candidate_job_reduction_fraction": 1.0 - (score_set_candidates / total_candidates if total_candidates else 0.0),
        "analysis_time_semantics": "verifier is applied only to greedy-wrong bin steps; greedy-correct stays correct and MISSING stays wrong deterministically",
        "deployment_gap": "at inference, greedy correctness is unknown; verifier damage on greedy-correct bin steps must be estimated separately",
    }
    write_jsonl(Path(args.output), score_rows)
    write_json(Path(args.summary) if args.summary else Path(args.output).with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()