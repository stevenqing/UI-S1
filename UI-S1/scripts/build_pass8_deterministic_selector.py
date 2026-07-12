#!/usr/bin/env python3
"""Build deterministic, label-blind support baselines for frozen Pass@8 packets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from run_pass8_selector import read_jsonl, verify_frozen_blind


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite deterministic selector output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def rank(candidate: Mapping[str, Any]) -> tuple[int, int, int, str]:
    return (
        int(candidate.get("source_count") or 0),
        int(candidate.get("neighborhood_support_count") or candidate.get("support_count") or 0),
        int(candidate.get("support_count") or 0),
        str(candidate.get("candidate_id") or ""),
    )


def choose(row: Mapping[str, Any], rule: str) -> tuple[Mapping[str, Any], str]:
    candidates = list(row.get("candidates") or [])
    baseline = next(candidate for candidate in candidates if candidate.get("candidate_id") == "BASELINE")
    alternatives = [candidate for candidate in candidates if candidate.get("candidate_id") != "BASELINE"]
    if rule == "exact_plurality":
        eligible = [candidate for candidate in alternatives if int(candidate.get("support_count") or 0) >= 2]
        reason = "highest anonymized exact-action support with at least two votes"
        rank_key = lambda candidate: (  # noqa: E731 - rule-local ordering is clearer inline
            int(candidate.get("support_count") or 0),
            int(candidate.get("source_count") or 0),
            int(candidate.get("neighborhood_support_count") or 0),
            str(candidate.get("candidate_id") or ""),
        )
    elif rule == "cross_source_consensus":
        eligible = [candidate for candidate in alternatives if int(candidate.get("source_count") or 0) >= 2]
        reason = "highest independent-generator agreement, then highest nearby-action support"
        rank_key = rank
    else:
        raise ValueError(f"unsupported deterministic rule: {rule}")
    if not eligible:
        return baseline, "no candidate met the conservative support requirement"
    # Candidate order was randomized before labels were sealed; the ID is only a deterministic final tie-break.
    return max(eligible, key=rank_key), reason


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--blind", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rule", choices=("exact_plurality", "cross_source_consensus"), required=True)
    args = parser.parse_args()

    blind_path = Path(args.blind)
    rows = read_jsonl(blind_path)
    verify_frozen_blind(Path(args.manifest), blind_path, rows)
    outputs = []
    for row in rows:
        selected, reason = choose(row, args.rule)
        support = int(selected.get("support_count") or 1)
        neighborhood = int(selected.get("neighborhood_support_count") or support)
        outputs.append({
            "protocol_version": row.get("protocol_version"),
            "prompt_version": row.get("prompt_version"),
            "selector_name": args.rule,
            "model": f"deterministic:{args.rule}",
            "target_id": row["target_id"],
            "episode_id": row["episode_id"],
            "step_idx": row["step_idx"],
            "packet_sha256": row["packet_sha256"],
            "attempted_candidate_id": selected["candidate_id"],
            "selected_candidate_id": selected["candidate_id"],
            "selected_action": selected.get("action"),
            "confidence": min(1.0, neighborhood / 8.0),
            "reason": reason,
            "parse_ok": True,
            "fallback_reason": None,
            "request_error": None,
            "raw_output": "__deterministic_label_blind_rule__",
        })
    write_jsonl(Path(args.output), outputs)
    print(json.dumps({"rule": args.rule, "rows": len(outputs), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
