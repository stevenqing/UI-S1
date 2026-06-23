#!/usr/bin/env python3
"""Audit candidate definitions for the GUI-Odyssey 3057 A0-wrong set.

This script does not run model inference. It reports which existing local
artifacts can or cannot reproduce the slide subset size, and writes candidate
row keys for the closest available source.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


JsonDict = dict[str, Any]


def iter_jsonl(path: Path) -> Iterable[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_split_counts(gui_odyssey_dir: Path, split_name: str) -> JsonDict:
    split_path = gui_odyssey_dir / "splits" / f"{split_name}.json"
    if not split_path.exists():
        return {"exists": False, "path": str(split_path)}
    split = json.loads(split_path.read_text(encoding="utf-8"))
    out: JsonDict = {"exists": True, "path": str(split_path)}
    for subset, ids in split.items():
        out[f"{subset}_episodes"] = len(ids)
        step_count = 0
        missing = 0
        for item in ids:
            episode_id = str(item).removesuffix(".json")
            ann_path = gui_odyssey_dir / "annotations" / f"{episode_id}.json"
            if not ann_path.exists():
                missing += 1
                continue
            ann = json.loads(ann_path.read_text(encoding="utf-8"))
            step_count += len(ann.get("steps", []))
        out[f"{subset}_steps"] = step_count
        out[f"{subset}_missing_annotations"] = missing
    return out


def behavior_stats(behavior_path: Path) -> tuple[JsonDict, list[JsonDict]]:
    if not behavior_path.exists():
        return {"exists": False, "path": str(behavior_path)}, []
    counts: Counter[tuple[str, str, str]] = Counter()
    wrong: Counter[tuple[str, str, str]] = Counter()
    wrong_episodes: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    candidate_rows: list[JsonDict] = []
    for row in iter_jsonl(behavior_path):
        if row.get("model_key") != "qwen3_vl_8b" or row.get("condition") != "no_history":
            continue
        key = (str(row.get("thinking_mode")), str(row.get("case_kind")), str(row.get("condition")))
        counts[key] += 1
        if not row.get("value_match"):
            wrong[key] += 1
            wrong_episodes[key].add(str(row.get("episode_id")))
            candidate_rows.append(
                {
                    "source": "model_behavior_no_history_wrong",
                    "model_key": row.get("model_key"),
                    "thinking_mode": row.get("thinking_mode"),
                    "case_kind": row.get("case_kind"),
                    "case_id": row.get("case_id"),
                    "episode_id": row.get("episode_id"),
                    "step_index": row.get("step_index"),
                    "screenshot": row.get("screenshot"),
                    "gt_action": row.get("gt_action"),
                    "pred_action": row.get("pred_action"),
                    "type_match": row.get("type_match"),
                    "value_match": row.get("value_match"),
                    "parse_ok": row.get("parse_ok"),
                }
            )
    by_bucket = []
    for key, total in sorted(counts.items()):
        by_bucket.append(
            {
                "thinking_mode": key[0],
                "case_kind": key[1],
                "condition": key[2],
                "rows": total,
                "wrong_rows": wrong[key],
                "wrong_episodes": len(wrong_episodes[key]),
            }
        )
    return (
        {
            "exists": True,
            "path": str(behavior_path),
            "qwen3_vl_no_history_rows": sum(counts.values()),
            "qwen3_vl_no_history_wrong_rows": sum(wrong.values()),
            "by_bucket": by_bucket,
        },
        candidate_rows,
    )


def verifier_stats(path: Path) -> JsonDict:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    rows = list(iter_jsonl(path))
    decisions = Counter(str((row.get("target") or {}).get("decision")) for row in rows)
    by_mode = Counter(str((row.get("metadata") or {}).get("thinking_mode")) for row in rows)
    no_history_wrong = 0
    episodes = set()
    for row in rows:
        metadata = row.get("metadata") or {}
        condition_value_match = metadata.get("condition_value_match") or {}
        if condition_value_match and not condition_value_match.get("no_history"):
            no_history_wrong += 1
        if metadata.get("episode_id") is not None:
            episodes.add(str(metadata.get("episode_id")))
    return {
        "exists": True,
        "path": str(path),
        "rows": len(rows),
        "episodes": len(episodes),
        "thinking_modes": dict(sorted(by_mode.items())),
        "target_decisions": dict(sorted(decisions.items())),
        "no_history_wrong_rows": no_history_wrong,
    }


def scan_near_3057(root: Path, max_size_bytes: int = 200_000_000) -> list[JsonDict]:
    excluded_prefixes = {
        ".venv",
        ".venv-qwen3-vllm",
        "datasets/GUI-Odyssey",
        "verl/third_party",
    }
    matches: list[JsonDict] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in {".jsonl", ".json"}:
            continue
        rel = path.relative_to(root).as_posix()
        if any(rel.startswith(prefix) for prefix in excluded_prefixes):
            continue
        try:
            if path.stat().st_size > max_size_bytes:
                continue
        except OSError:
            continue
        try:
            if path.suffix == ".jsonl":
                rows = 0
                wrongish: Counter[str] = Counter()
                with path.open("r", encoding="utf-8", errors="ignore") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        rows += 1
                        if rows > 10_000:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(row, dict):
                            continue
                        for key in [
                            "value_match",
                            "extract_match",
                            "is_correct",
                            "success",
                            "correct",
                            "coord_correct",
                            "action_correct",
                            "type_match",
                        ]:
                            if key in row and row.get(key) in {False, "no", 0, None}:
                                wrongish[key] += 1
                if abs(rows - 3057) <= 20 or any(abs(value - 3057) <= 20 for value in wrongish.values()):
                    matches.append({"path": rel, "rows": rows, "wrongish": dict(wrongish)})
            elif path.stat().st_size <= 50_000_000:
                data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
                lengths = []
                if isinstance(data, list):
                    lengths.append(["list", len(data)])
                elif isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, (list, dict)):
                            lengths.append([key, len(value)])
                if any(abs(length - 3057) <= 20 for _, length in lengths):
                    matches.append({"path": rel, "lengths": lengths})
        except Exception:
            continue
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit candidate sources for GUI-Odyssey 3057 A0-wrong rows")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--split", default="random_split")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gui_odyssey_3057_audit"))
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    gui_odyssey_dir = root / "datasets" / "GUI-Odyssey"
    behavior_path = root / "datasets" / "model_bottleneck_validation_qwen3vl_restore_20260620_sharded" / "merged" / "model_behavior_results.jsonl"
    verifier_all_test = root / "datasets" / "verifier_agent_gui_odyssey_all_restore_20260620" / "test.jsonl"
    verifier_hard_test = root / "datasets" / "verifier_agent_gui_odyssey_hard_restore_20260620" / "test.jsonl"

    behavior_summary, behavior_candidate_rows = behavior_stats(behavior_path)
    candidate_path = output_dir / "qwen3_vl_behavior_no_history_wrong_candidate_keys.jsonl"
    written = write_jsonl(candidate_path, behavior_candidate_rows)

    summary = {
        "target_slide_rows": 3057,
        "conclusion": {
            "exact_3057_artifact_found": False,
            "closest_local_candidate": "qwen3_vl behavior no_history wrong rows",
            "closest_local_candidate_rows": written,
            "row_delta_vs_3057": written - 3057,
            "important_caveat": "The closest candidate comes from the bottleneck no_history prompt, not the GUI-360/SFTv2 baseline prompt template.",
        },
        "gui_odyssey_split": load_split_counts(gui_odyssey_dir, args.split),
        "qwen3_vl_behavior": behavior_summary,
        "verifier_all_test": verifier_stats(verifier_all_test),
        "verifier_hard_test": verifier_stats(verifier_hard_test),
        "near_3057_local_json_artifacts": scan_near_3057(root),
        "outputs": {"candidate_keys": str(candidate_path)},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()