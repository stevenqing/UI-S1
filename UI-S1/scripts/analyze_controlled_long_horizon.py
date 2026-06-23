#!/usr/bin/env python3
"""Controlled long-horizon analysis for bottleneck validation results."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


JsonDict = dict[str, Any]
CONDITIONS = ("no_history", "segment_summary", "full_history", "wrong_summary")


def iter_jsonl(path: Path) -> Iterable[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_segments(path: Path) -> dict[str, JsonDict]:
    return {str(row.get("episode_id")): row for row in iter_jsonl(path)}


def segment_for_step(episode: JsonDict | None, step_index: int) -> JsonDict | None:
    if not episode:
        return None
    for segment in episode.get("segments", []) or []:
        if int(segment.get("start_step", 0)) <= step_index <= int(segment.get("end_step", -1)):
            return segment
    return None


def features(row: JsonDict, episodes: dict[str, JsonDict]) -> JsonDict:
    episode = episodes.get(str(row.get("episode_id")))
    step_index = int(row.get("step_index") or 0)
    total_steps = int((episode or {}).get("num_steps") or 0)
    segment = segment_for_step(episode, step_index)
    prev_segments = 0
    if episode:
        prev_segments = sum(1 for seg in episode.get("segments", []) or [] if int(seg.get("end_step", -1)) < step_index)
    memory = (segment.get("memory_need", {}) or {}) if segment else {}
    return {
        "episode_num_steps": total_steps,
        "step_index": step_index,
        "step_pos_ratio": step_index / max(total_steps - 1, 1) if total_steps else None,
        "prev_segments": prev_segments,
        "segment_id": segment.get("segment_id") if segment else None,
        "segment_start": segment.get("start_step") if segment else None,
        "segment_end": segment.get("end_step") if segment else None,
        "segment_len_so_far": step_index - int(segment.get("start_step", step_index)) + 1 if segment else None,
        "memory_strength": memory.get("strength", "unknown"),
        "carried_values": segment.get("carried_values", []) if segment else [],
        "dominant_capability": segment.get("dominant_capability", "unknown") if segment else "unknown",
    }


def group_rows(rows: Iterable[JsonDict]) -> dict[tuple[str, str, int], dict[str, JsonDict]]:
    grouped: dict[tuple[str, str, int], dict[str, JsonDict]] = defaultdict(dict)
    for row in rows:
        if row.get("condition") not in CONDITIONS:
            continue
        key = (str(row.get("thinking_mode", "unknown")), str(row.get("case_kind", "unknown")), int(row.get("case_id")))
        grouped[key][str(row.get("condition"))] = row
    return {key: by_cond for key, by_cond in grouped.items() if all(cond in by_cond for cond in CONDITIONS)}


def ok(by_cond: dict[str, JsonDict], condition: str) -> bool:
    return bool(by_cond[condition].get("value_match"))


def action(row: JsonDict) -> str:
    return str((row.get("gt_action") or {}).get("action", "unknown"))


def subset_name(feat: JsonDict) -> list[str]:
    names = ["all"]
    n = int(feat.get("episode_num_steps") or 0)
    step = int(feat.get("step_index") or 0)
    prev = int(feat.get("prev_segments") or 0)
    memory = feat.get("memory_strength")
    carried = bool(feat.get("carried_values"))
    for threshold in (6, 10, 15, 20, 25):
        if n > threshold:
            names.append(f"episode_gt_{threshold}")
    for threshold in (3, 6, 10, 15):
        if step >= threshold:
            names.append(f"step_ge_{threshold}")
    if prev >= 1:
        names.append("prev_segments_ge_1")
    if prev >= 2:
        names.append("prev_segments_ge_2")
    if carried:
        names.append("has_carried_values")
    if memory in {"medium", "high"}:
        names.append("memory_medium_high")
    if memory == "high":
        names.append("memory_high")
    return names


def ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def summarize(cases: list[tuple[tuple[str, str, int], dict[str, JsonDict], JsonDict]]) -> JsonDict:
    den = len(cases)
    no_wrong = [case for case in cases if not ok(case[1], "no_history")]
    segment_rescue = [case for case in cases if not ok(case[1], "no_history") and ok(case[1], "segment_summary")]
    full_rescue = [case for case in cases if not ok(case[1], "no_history") and ok(case[1], "full_history")]
    any_memory_rescue = [case for case in cases if not ok(case[1], "no_history") and (ok(case[1], "segment_summary") or ok(case[1], "full_history"))]
    specific_segment_rescue = [case for case in segment_rescue if not ok(case[1], "wrong_summary")]
    specific_any_rescue = [case for case in any_memory_rescue if not ok(case[1], "wrong_summary")]
    segment_regression = [case for case in cases if ok(case[1], "no_history") and not ok(case[1], "segment_summary")]
    wrong_beats_segment = [case for case in cases if not ok(case[1], "segment_summary") and ok(case[1], "wrong_summary")]
    all_wrong = [case for case in cases if not any(ok(case[1], cond) for cond in CONDITIONS)]
    return {
        "cases": den,
        "no_history_acc": ratio(sum(ok(case[1], "no_history") for case in cases), den),
        "segment_summary_acc": ratio(sum(ok(case[1], "segment_summary") for case in cases), den),
        "full_history_acc": ratio(sum(ok(case[1], "full_history") for case in cases), den),
        "wrong_summary_acc": ratio(sum(ok(case[1], "wrong_summary") for case in cases), den),
        "no_history_wrong": len(no_wrong),
        "segment_rescue": len(segment_rescue),
        "segment_rescue_rate_of_no_wrong": ratio(len(segment_rescue), len(no_wrong)),
        "full_rescue": len(full_rescue),
        "full_rescue_rate_of_no_wrong": ratio(len(full_rescue), len(no_wrong)),
        "any_memory_rescue": len(any_memory_rescue),
        "any_memory_rescue_rate_of_no_wrong": ratio(len(any_memory_rescue), len(no_wrong)),
        "specific_segment_rescue": len(specific_segment_rescue),
        "specific_segment_rescue_rate_of_no_wrong": ratio(len(specific_segment_rescue), len(no_wrong)),
        "specific_any_rescue": len(specific_any_rescue),
        "specific_any_rescue_rate_of_no_wrong": ratio(len(specific_any_rescue), len(no_wrong)),
        "segment_regression": len(segment_regression),
        "wrong_beats_segment": len(wrong_beats_segment),
        "all_conditions_wrong": len(all_wrong),
    }


def compact_case(key: tuple[str, str, int], by_cond: dict[str, JsonDict], feat: JsonDict, category: str) -> JsonDict:
    mode, case_kind, case_id = key
    return {
        "category": category,
        "thinking_mode": mode,
        "case_kind": case_kind,
        "case_id": case_id,
        "episode_id": by_cond["no_history"].get("episode_id"),
        "step_index": by_cond["no_history"].get("step_index"),
        "episode_num_steps": feat.get("episode_num_steps"),
        "prev_segments": feat.get("prev_segments"),
        "memory_strength": feat.get("memory_strength"),
        "carried_values": feat.get("carried_values"),
        "dominant_capability": feat.get("dominant_capability"),
        "gt_action": by_cond["no_history"].get("gt_action"),
        "condition_value_match": {cond: ok(by_cond, cond) for cond in CONDITIONS},
        "condition_type_match": {cond: bool(by_cond[cond].get("type_match")) for cond in CONDITIONS},
        "pred_actions": {cond: by_cond[cond].get("pred_action") for cond in CONDITIONS},
        "raw_outputs": {cond: by_cond[cond].get("raw_output") for cond in CONDITIONS},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-key", default="qwen3_vl_8b")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_segments(args.segments)
    rows = [row for row in iter_jsonl(args.results) if row.get("model_key") == args.model_key]
    grouped = group_rows(rows)
    enriched: list[tuple[tuple[str, str, int], dict[str, JsonDict], JsonDict]] = []
    for key, by_cond in grouped.items():
        enriched.append((key, by_cond, features(by_cond["no_history"], episodes)))

    by_subset: dict[tuple[str, str, str], list[tuple[tuple[str, str, int], dict[str, JsonDict], JsonDict]]] = defaultdict(list)
    for item in enriched:
        mode, case_kind, _ = item[0]
        for name in subset_name(item[2]):
            by_subset[(mode, case_kind, name)].append(item)

    summary_rows = []
    for (mode, case_kind, name), cases in sorted(by_subset.items()):
        row = {"thinking_mode": mode, "case_kind": case_kind, "subset": name}
        row.update(summarize(cases))
        summary_rows.append(row)

    by_action: dict[tuple[str, str, str, str], list[tuple[tuple[str, str, int], dict[str, JsonDict], JsonDict]]] = defaultdict(list)
    for item in enriched:
        mode, case_kind, _ = item[0]
        gt_action = action(item[1]["no_history"])
        for name in subset_name(item[2]):
            by_action[(mode, case_kind, name, gt_action)].append(item)
    action_rows = []
    for (mode, case_kind, name, gt_action), cases in sorted(by_action.items()):
        row = {"thinking_mode": mode, "case_kind": case_kind, "subset": name, "action": gt_action}
        row.update(summarize(cases))
        action_rows.append(row)

    hard_examples = []
    for key, by_cond, feat in enriched:
        categories = []
        if not ok(by_cond, "no_history") and ok(by_cond, "segment_summary"):
            categories.append("segment_rescue")
        if not ok(by_cond, "no_history") and ok(by_cond, "segment_summary") and not ok(by_cond, "wrong_summary"):
            categories.append("specific_segment_rescue")
        if not ok(by_cond, "no_history") and (ok(by_cond, "segment_summary") or ok(by_cond, "full_history")):
            categories.append("any_memory_rescue")
        if not ok(by_cond, "no_history") and (ok(by_cond, "segment_summary") or ok(by_cond, "full_history")) and not ok(by_cond, "wrong_summary"):
            categories.append("specific_any_memory_rescue")
        if ok(by_cond, "no_history") and not ok(by_cond, "segment_summary"):
            categories.append("segment_regression")
        if not ok(by_cond, "segment_summary") and ok(by_cond, "wrong_summary"):
            categories.append("wrong_beats_segment")
        if not any(ok(by_cond, cond) for cond in CONDITIONS):
            categories.append("all_wrong")
        for category in categories:
            hard_examples.append(compact_case(key, by_cond, feat, category))

    (args.output_dir / "controlled_long_horizon_summary.json").write_text(
        json.dumps({"summary": summary_rows, "action_breakdown": action_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(args.output_dir / "controlled_long_horizon_cases.jsonl", hard_examples)

    selected_subsets = ["all", "episode_gt_6", "episode_gt_10", "episode_gt_15", "step_ge_6", "step_ge_10", "prev_segments_ge_2", "has_carried_values", "memory_medium_high", "memory_high"]
    lines = ["# Controlled Long-Horizon Bottleneck Analysis", ""]
    lines.append(f"Source: `{args.results}`")
    lines.append(f"Segments: `{args.segments}`")
    lines.append("")
    lines.append("## Summary By Subset")
    lines.append("")
    lines.append("| thinking | case | subset | cases | no acc | seg acc | full acc | wrong acc | no wrong | seg rescue | seg rescue/no wrong | specific seg rescue | specific seg/no wrong | any memory rescue | specific any rescue | all wrong | seg regression | wrong>seg |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        if row["subset"] not in selected_subsets:
            continue
        lines.append(
            f"| {row['thinking_mode']} | {row['case_kind']} | {row['subset']} | {row['cases']} | "
            f"{row['no_history_acc']:.4f} | {row['segment_summary_acc']:.4f} | {row['full_history_acc']:.4f} | {row['wrong_summary_acc']:.4f} | "
            f"{row['no_history_wrong']} | {row['segment_rescue']} | {row['segment_rescue_rate_of_no_wrong']:.4f} | "
            f"{row['specific_segment_rescue']} | {row['specific_segment_rescue_rate_of_no_wrong']:.4f} | "
            f"{row['any_memory_rescue']} | {row['specific_any_rescue']} | {row['all_conditions_wrong']} | {row['segment_regression']} | {row['wrong_beats_segment']} |"
        )
    lines.append("")
    lines.append("## Interpretation Notes")
    lines.append("")
    lines.append("- `segment_rescue`: no_history failed, segment_summary succeeded.")
    lines.append("- `specific_segment_rescue`: segment_rescue and wrong_summary failed; this is stronger evidence that the right memory helped.")
    lines.append("- `any_memory_rescue`: no_history failed, segment_summary or full_history succeeded.")
    lines.append("- `specific_any_rescue`: any_memory_rescue and wrong_summary failed.")
    lines.append("- `segment_regression` and `wrong>seg` are counter-evidence/noise indicators.")
    lines.append("")
    lines.append("## Example counts")
    lines.append("")
    counts = Counter(row["category"] for row in hard_examples)
    for key, value in counts.most_common():
        lines.append(f"- {key}: {value}")
    lines.append("")
    args.output_dir.joinpath("controlled_long_horizon_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"rows={len(rows)} complete_cases={len(grouped)} examples={len(hard_examples)} output={args.output_dir}")


if __name__ == "__main__":
    main()