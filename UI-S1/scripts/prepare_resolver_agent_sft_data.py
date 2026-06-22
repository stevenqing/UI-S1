#!/usr/bin/env python3
"""Prepare Resolver Agent SFT data from verifier replan states.

The resolver is the agent called after the verifier refuses to execute the
current candidate packet. Its target is a corrected GUI action.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

SYSTEM_PROMPT = """You are a Resolver Agent in a multi-agent GUI controller.
The verifier refused the current candidate actions or requested replanning.
Your job is to propose one corrected low-level GUI action for the current step.

Return exactly one JSON object:
{"action": action_object, "reason_codes": [strings], "rationale": short_string}

Allowed action objects include:
- {"action": "click", "coordinate": [x, y]}
- {"action": "long_press", "coordinate": [x, y]}
- {"action": "swipe", "coordinate": [x1, y1], "coordinate2": [x2, y2]}
- {"action": "type", "text": "..."}
- {"action": "system_button", "button": "Home|Back|Menu|..."}
- {"action": "terminate", "status": "success|failure"}
"""


def iter_jsonl(path: Path) -> list[JsonDict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def target_action(metadata: JsonDict) -> JsonDict:
    action = metadata.get("gt_action") or {}
    return action if isinstance(action, dict) else {"action": str(action)}


def resolver_response(action: JsonDict) -> JsonDict:
    return {
        "action": action,
        "reason_codes": ["ground_truth_resolver_target"],
        "rationale": "The corrected action is the supervised target for this verifier replan state.",
    }


def replan_request_from_packet(row: JsonDict) -> JsonDict:
    packet = row.get("packet", {}) or {}
    target = row.get("target", {}) or {}
    return {
        "reason": "oracle_replan_training_state",
        "verifier_decision": target.get("decision", "replan"),
        "verifier_output": json.dumps(target, ensure_ascii=False),
        "task": packet.get("task", {}),
        "memory": packet.get("memory", {}),
        "candidate_agents": packet.get("candidate_agents", {}),
        "computed_evidence": packet.get("computed_evidence", {}),
        "recommended_next_steps": [
            "generate_alternative_candidate",
            "recover_missing_carried_value",
            "rewrite_current_instruction",
            "rerun_verifier_on_new_packet",
        ],
    }


def prompt_for_request(request: JsonDict, metadata: JsonDict) -> str:
    payload = {
        "replan_request": request,
        "metadata": {
            "dominant_capability": metadata.get("dominant_capability"),
            "step_index": metadata.get("step_index"),
            "total_steps": metadata.get("total_steps"),
            "carried_values": metadata.get("carried_values", []),
            "screenshot": metadata.get("screenshot"),
        },
    }
    return SYSTEM_PROMPT + "\n\nResolve this verifier replan request:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def record_from_request(request: JsonDict, metadata: JsonDict, source: str) -> JsonDict:
    action = target_action(metadata)
    return {
        "messages": [
            {"role": "user", "content": prompt_for_request(request, metadata)},
            {"role": "assistant", "content": json.dumps(resolver_response(action), ensure_ascii=False)},
        ],
        "target_action": action,
        "metadata": {**metadata, "resolver_source": source},
    }


def records_from_hard_rows(rows: list[JsonDict], source: str, only_target_replan: bool) -> list[JsonDict]:
    output = []
    for row in rows:
        target = (row.get("target", {}) or {}).get("decision")
        if only_target_replan and target != "replan":
            continue
        metadata = row.get("metadata", {}) or {}
        output.append(record_from_request(replan_request_from_packet(row), metadata, source))
    return output


def records_from_hybrid_commands(rows: list[JsonDict], source: str, hard_only: bool) -> list[JsonDict]:
    output = []
    for row in rows:
        if row.get("status") != "replan":
            continue
        if hard_only and not row.get("hybrid_hard_state"):
            continue
        request = row.get("replan_request") or {}
        metadata = row.get("metadata", {}) or {}
        output.append(record_from_request(request, metadata, source))
    return output


def parquet_row(row: JsonDict) -> JsonDict:
    return {
        "messages": json.dumps(row.get("messages", []), ensure_ascii=False),
        "target_action": json.dumps(row.get("target_action", {}), ensure_ascii=False),
        "metadata": json.dumps(row.get("metadata", {}), ensure_ascii=False),
    }


def write_parquet(path: Path, rows: list[JsonDict]) -> bool:
    try:
        import pandas as pd

        frame = pd.DataFrame([parquet_row(row) for row in rows])
        frame.to_parquet(path, index=False, engine="pyarrow")
        return True
    except Exception as exc:  # pragma: no cover
        print(f"warning: failed to write parquet {path}: {exc}")
        return False


def summarize(rows: list[JsonDict]) -> JsonDict:
    actions = Counter(str((row.get("target_action", {}) or {}).get("action", "unknown")) for row in rows)
    caps = Counter(str((row.get("metadata", {}) or {}).get("dominant_capability", "unknown")) for row in rows)
    sources = Counter(str((row.get("metadata", {}) or {}).get("resolver_source", "unknown")) for row in rows)
    user_lens = [len(str((row.get("messages", [{}])[0] or {}).get("content", ""))) for row in rows]
    return {
        "rows": len(rows),
        "target_actions": dict(actions.most_common()),
        "capabilities": dict(caps.most_common(20)),
        "sources": dict(sources.most_common()),
        "avg_user_chars": sum(user_lens) / len(user_lens) if user_lens else 0.0,
        "max_user_chars": max(user_lens) if user_lens else 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Resolver Agent SFT data")
    parser.add_argument("--hard-dir", required=True, help="Hard-only verifier data directory")
    parser.add_argument("--hybrid-dir", required=True, help="Hybrid policy directory containing balanced/dev|test commands")
    parser.add_argument("--mode", default="balanced", help="Hybrid mode subdirectory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-all-hard", action="store_true", help="Use all hard train rows, not only target replan rows")
    parser.add_argument("--no-parquet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hard_dir = Path(args.hard_dir)
    hybrid_dir = Path(args.hybrid_dir) / args.mode
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": records_from_hard_rows(
            iter_jsonl(hard_dir / "train.jsonl"),
            source="hard_train_oracle_packet",
            only_target_replan=not args.train_all_hard,
        ),
        "dev": records_from_hybrid_commands(
            iter_jsonl(hybrid_dir / "dev" / "hybrid_commands.jsonl"),
            source=f"hybrid_{args.mode}_dev_replan_request",
            hard_only=True,
        ),
        "test": records_from_hybrid_commands(
            iter_jsonl(hybrid_dir / "test" / "hybrid_commands.jsonl"),
            source=f"hybrid_{args.mode}_test_replan_request",
            hard_only=True,
        ),
    }

    stats = {}
    parquet_written = {}
    for split, rows in splits.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)
        parquet_written[split] = False if args.no_parquet else write_parquet(output_dir / f"{split}.parquet", rows)
        stats[split] = summarize(rows)

    manifest = {
        "hard_dir": str(hard_dir),
        "hybrid_dir": str(hybrid_dir),
        "mode": args.mode,
        "train_all_hard": args.train_all_hard,
        "parquet_written": parquet_written,
        "splits": stats,
        "recommended_training": {
            "train_file": str(output_dir / "train.parquet"),
            "val_file": str(output_dir / "dev.parquet"),
            "messages_key": "messages",
            "task": "Resolver Agent corrected-action SFT",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "splits": stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()