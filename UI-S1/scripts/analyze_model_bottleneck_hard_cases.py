#!/usr/bin/env python3
"""Analyze hard cases for model bottleneck behavior validation results."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


JsonDict = dict[str, Any]
CONDITIONS = ["no_history", "segment_summary", "full_history", "wrong_summary"]


def iter_jsonl(path: Path) -> Iterable[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def action_name(row: JsonDict) -> str:
    return str(row.get("gt_action", {}).get("action", "unknown"))


def prediction_text(row: JsonDict | None) -> str:
    if not row:
        return "missing"
    if row.get("error"):
        return "ERROR"
    prediction = row.get("pred_action")
    if prediction is None:
        return "UNPARSED"
    return json.dumps(prediction, ensure_ascii=False)


def group_rows(rows: list[JsonDict]) -> dict[tuple[str, str, int], dict[str, JsonDict]]:
    grouped: dict[tuple[str, str, int], dict[str, JsonDict]] = defaultdict(dict)
    for row in rows:
        key = (row.get("thinking_mode", "unknown"), row["case_kind"], int(row["case_id"]))
        grouped[key][row["condition"]] = row
    return {key: value for key, value in grouped.items() if all(condition in value for condition in CONDITIONS)}


def load_episode_index(paths: list[Path]) -> dict[str, JsonDict]:
    episodes = {}
    for path in paths:
        for episode in iter_jsonl(path):
            episodes[str(episode.get("episode_id"))] = episode
    return episodes


def segment_for_step(episode: JsonDict | None, step_index: int) -> JsonDict | None:
    if not episode:
        return None
    for segment in episode.get("segments", []):
        if segment.get("start_step", 0) <= step_index <= segment.get("end_step", -1):
            return segment
    return None


def long_horizon_features(row: JsonDict, episode_index: dict[str, JsonDict], step_threshold: int, prev_segments_threshold: int) -> JsonDict:
    episode = episode_index.get(str(row.get("episode_id")))
    step_index = int(row.get("step_index") or 0)
    prev_segments = 0
    total_steps = None
    segment_id = None
    segment_start = None
    segment_len_so_far = None
    carried_values = []
    memory_strength = "unknown"
    dominant_capability = "unknown"
    if episode:
        total_steps = len(episode.get("steps", []))
        prev_segments = sum(1 for segment in episode.get("segments", []) if segment.get("end_step", -1) < step_index)
        segment = segment_for_step(episode, step_index)
        if segment:
            segment_id = segment.get("segment_id")
            segment_start = segment.get("start_step")
            segment_len_so_far = step_index - int(segment.get("start_step", step_index)) + 1
            carried_values = segment.get("carried_values", []) or []
            memory_strength = (segment.get("memory_need", {}) or {}).get("strength", "unknown")
            dominant_capability = segment.get("dominant_capability", "unknown")
    is_long_horizon = step_index >= step_threshold or prev_segments >= prev_segments_threshold or bool(carried_values) or memory_strength in {"medium", "high"}
    return {
        "step_index": step_index,
        "total_steps": total_steps,
        "prev_segments": prev_segments,
        "segment_id": segment_id,
        "segment_start": segment_start,
        "segment_len_so_far": segment_len_so_far,
        "carried_values": carried_values,
        "memory_strength": memory_strength,
        "dominant_capability": dominant_capability,
        "is_long_horizon": is_long_horizon,
    }


def bool_metric(row: JsonDict, metric: str) -> bool:
    return bool(row.get(metric))


def case_record(
    key: tuple[str, str, int],
    rows_by_condition: dict[str, JsonDict],
    category: str,
    episode_index: dict[str, JsonDict],
    step_threshold: int,
    prev_segments_threshold: int,
) -> JsonDict:
    mode, case_kind, case_id = key
    base_row = rows_by_condition["no_history"]
    features = long_horizon_features(base_row, episode_index, step_threshold, prev_segments_threshold)
    return {
        "category": category,
        "thinking_mode": mode,
        "case_kind": case_kind,
        "case_id": case_id,
        "episode_id": base_row.get("episode_id"),
        "step_index": base_row.get("step_index"),
        "gt_action": base_row.get("gt_action"),
        "screenshot": base_row.get("screenshot", ""),
        "condition_value_match": {condition: bool_metric(rows_by_condition[condition], "value_match") for condition in CONDITIONS},
        "condition_type_match": {condition: bool_metric(rows_by_condition[condition], "type_match") for condition in CONDITIONS},
        "condition_parse_ok": {condition: bool_metric(rows_by_condition[condition], "parse_ok") for condition in CONDITIONS},
        "pred_actions": {condition: rows_by_condition[condition].get("pred_action") for condition in CONDITIONS},
        "errors": {condition: rows_by_condition[condition].get("error", "") for condition in CONDITIONS},
        "long_horizon": features,
    }


def rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def condition_rates(rows: list[JsonDict], metric: str) -> dict[tuple[str, str, str], float]:
    grouped: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("thinking_mode", "unknown"), row["case_kind"], row["condition"])].append(bool_metric(row, metric))
    return {key: rate(values) for key, values in grouped.items()}


def summarize_grouped_cases(
    grouped_cases: dict[tuple[str, str, int], dict[str, JsonDict]],
    episode_index: dict[str, JsonDict],
    step_threshold: int,
    prev_segments_threshold: int,
) -> tuple[list[JsonDict], list[JsonDict]]:
    summary_rows = []
    hard_cases = []
    grouped_by_mode_kind: dict[tuple[str, str], list[tuple[tuple[str, str, int], dict[str, JsonDict]]]] = defaultdict(list)
    for key, rows_by_condition in grouped_cases.items():
        mode, case_kind, _case_id = key
        grouped_by_mode_kind[(mode, case_kind)].append((key, rows_by_condition))

    for (mode, case_kind), cases in sorted(grouped_by_mode_kind.items()):
        no_wrong = [item for item in cases if not bool_metric(item[1]["no_history"], "value_match")]
        segment_rescue = [item for item in cases if not bool_metric(item[1]["no_history"], "value_match") and bool_metric(item[1]["segment_summary"], "value_match")]
        segment_regression = [item for item in cases if bool_metric(item[1]["no_history"], "value_match") and not bool_metric(item[1]["segment_summary"], "value_match")]
        memory_specific = [item for item in cases if bool_metric(item[1]["segment_summary"], "value_match") and not bool_metric(item[1]["wrong_summary"], "value_match")]
        wrong_beats_segment = [item for item in cases if not bool_metric(item[1]["segment_summary"], "value_match") and bool_metric(item[1]["wrong_summary"], "value_match")]
        full_rescue = [item for item in cases if not bool_metric(item[1]["no_history"], "value_match") and bool_metric(item[1]["full_history"], "value_match")]
        all_wrong = [item for item in cases if not any(bool_metric(item[1][condition], "value_match") for condition in CONDITIONS)]
        non_obvious_no_wrong = [
            item for item in no_wrong
            if action_name(item[1]["no_history"]) not in {"click", "system_button"}
        ]
        long_horizon = [item for item in cases if long_horizon_features(item[1]["no_history"], episode_index, step_threshold, prev_segments_threshold)["is_long_horizon"]]
        long_horizon_no_wrong = [item for item in no_wrong if long_horizon_features(item[1]["no_history"], episode_index, step_threshold, prev_segments_threshold)["is_long_horizon"]]
        long_horizon_segment_rescue = [item for item in segment_rescue if long_horizon_features(item[1]["no_history"], episode_index, step_threshold, prev_segments_threshold)["is_long_horizon"]]
        summary_rows.append(
            {
                "thinking_mode": mode,
                "case_kind": case_kind,
                "cases": len(cases),
                "no_history_wrong": len(no_wrong),
                "segment_rescue": len(segment_rescue),
                "segment_regression": len(segment_regression),
                "memory_specific_segment_over_wrong": len(memory_specific),
                "wrong_beats_segment": len(wrong_beats_segment),
                "full_history_rescue": len(full_rescue),
                "all_conditions_wrong": len(all_wrong),
                "non_obvious_no_history_wrong": len(non_obvious_no_wrong),
                "long_horizon_cases": len(long_horizon),
                "long_horizon_no_history_wrong": len(long_horizon_no_wrong),
                "long_horizon_segment_rescue": len(long_horizon_segment_rescue),
            }
        )
        for category, selected_cases in [
            ("segment_rescue", segment_rescue),
            ("segment_regression", segment_regression),
            ("memory_specific_segment_over_wrong", memory_specific),
            ("wrong_beats_segment", wrong_beats_segment),
            ("full_history_rescue", full_rescue),
            ("all_conditions_wrong", all_wrong),
            ("non_obvious_no_history_wrong", non_obvious_no_wrong),
            ("long_horizon_no_history_wrong", long_horizon_no_wrong),
            ("long_horizon_segment_rescue", long_horizon_segment_rescue),
        ]:
            for key, rows_by_condition in selected_cases:
                hard_cases.append(case_record(key, rows_by_condition, category, episode_index, step_threshold, prev_segments_threshold))
    return summary_rows, hard_cases


def action_breakdown(grouped_cases: dict[tuple[str, str, int], dict[str, JsonDict]]) -> list[JsonDict]:
    buckets: dict[tuple[str, str, str], list[dict[str, JsonDict]]] = defaultdict(list)
    for key, rows_by_condition in grouped_cases.items():
        mode, case_kind, _case_id = key
        buckets[(mode, case_kind, action_name(rows_by_condition["no_history"]))].append(rows_by_condition)
    rows = []
    for (mode, case_kind, action), cases in sorted(buckets.items()):
        row = {"thinking_mode": mode, "case_kind": case_kind, "action": action, "n": len(cases)}
        for condition in CONDITIONS:
            row[f"{condition}_value_acc"] = rate([bool_metric(item[condition], "value_match") for item in cases])
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_float(value: float) -> str:
    return f"{value:.3f}"


def write_report(
    path: Path,
    model_key: str,
    input_path: Path,
    filtered_rows: list[JsonDict],
    summary_rows: list[JsonDict],
    hard_cases: list[JsonDict],
    action_rows: list[JsonDict],
) -> None:
    metric_rates = condition_rates(filtered_rows, "value_match")
    lines = [f"# Hard-Case Bottleneck Analysis: {model_key}", ""]
    lines.append(f"Source: `{input_path}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Rows: {len(filtered_rows)}")
    lines.append(f"- Complete paired cases: {sum(row['cases'] for row in summary_rows)}")
    lines.append(f"- Errors: {sum(1 for row in filtered_rows if row.get('error'))}")
    lines.append("")
    lines.append("## Value Accuracy")
    lines.append("")
    lines.append("| thinking | case | no_history | segment_summary | full_history | wrong_summary |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for mode in sorted({row.get("thinking_mode", "unknown") for row in filtered_rows}):
        for case_kind in ["real_boundary", "random_control"]:
            values = [metric_rates.get((mode, case_kind, condition), 0.0) for condition in CONDITIONS]
            lines.append(f"| {mode} | {case_kind} | " + " | ".join(format_float(value) for value in values) + " |")
    lines.append("")
    lines.append("## Hard-Case Counts")
    lines.append("")
    lines.append("| thinking | case | cases | no wrong | segment rescue | segment regression | segment > wrong | wrong > segment | full rescue | all wrong | non-obvious no wrong | long-horizon | long no wrong | long rescue |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            f"| {row['thinking_mode']} | {row['case_kind']} | {row['cases']} | {row['no_history_wrong']} | "
            f"{row['segment_rescue']} | {row['segment_regression']} | {row['memory_specific_segment_over_wrong']} | "
            f"{row['wrong_beats_segment']} | {row['full_history_rescue']} | {row['all_conditions_wrong']} | {row['non_obvious_no_history_wrong']} | "
            f"{row.get('long_horizon_cases', 0)} | {row.get('long_horizon_no_history_wrong', 0)} | {row.get('long_horizon_segment_rescue', 0)} |"
        )
    lines.append("")
    lines.append("## Action Breakdown")
    lines.append("")
    lines.append("| thinking | case | action | n | no_history | segment_summary | full_history | wrong_summary |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for row in action_rows:
        if row["n"] < 5:
            continue
        lines.append(
            f"| {row['thinking_mode']} | {row['case_kind']} | {row['action']} | {row['n']} | "
            f"{format_float(row['no_history_value_acc'])} | {format_float(row['segment_summary_value_acc'])} | "
            f"{format_float(row['full_history_value_acc'])} | {format_float(row['wrong_summary_value_acc'])} |"
        )
    lines.append("")
    lines.append("## Representative Hard Cases")
    lines.append("")
    category_counts = Counter(row["category"] for row in hard_cases)
    for category, count in category_counts.most_common():
        lines.append(f"- {category}: {count}")
    lines.append("")
    for category in ["segment_rescue", "long_horizon_segment_rescue", "memory_specific_segment_over_wrong", "segment_regression", "all_conditions_wrong", "non_obvious_no_history_wrong", "long_horizon_no_history_wrong"]:
        selected = [row for row in hard_cases if row["category"] == category][:5]
        if not selected:
            continue
        lines.append(f"### {category}")
        lines.append("")
        for row in selected:
            preds = "; ".join(f"{condition}={prediction_text({'pred_action': row['pred_actions'].get(condition), 'error': row['errors'].get(condition, '')})}" for condition in CONDITIONS)
            values = ", ".join(f"{condition}:{row['condition_value_match'][condition]}" for condition in CONDITIONS)
            long = row.get("long_horizon", {})
            lines.append(
                f"- {row['thinking_mode']} {row['case_kind']} case={row['case_id']} episode={row['episode_id']} "
                f"step={row['step_index']} prev_segments={long.get('prev_segments')} memory={long.get('memory_strength')} "
                f"capability={long.get('dominant_capability')} carried={json.dumps(long.get('carried_values', []), ensure_ascii=False)} "
                f"gt={json.dumps(row['gt_action'], ensure_ascii=False)} values=[{values}] preds=[{preds}]"
            )
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("Use `segment_rescue` as the strongest positive evidence: the current state alone failed, while the segment summary succeeded. Use `segment_regression` and `wrong_beats_segment` as counter-evidence. A strong bottleneck result should show more segment rescues on real boundaries than random controls, fewer regressions, and a meaningful gap between segment_summary and wrong_summary.")
    lines.append("Long-horizon rows are marked by high step index, multiple previous segments, carried values, or medium/high memory annotations. These are the most relevant samples for testing whether subtask memory is needed beyond current-screen recognition.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze hard cases in model bottleneck behavior results")
    parser.add_argument("--results", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", nargs="*", default=[])
    parser.add_argument("--long-step-threshold", type=int, default=10)
    parser.add_argument("--long-prev-segments-threshold", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [row for row in iter_jsonl(input_path) if row.get("model_key") == args.model_key]
    episode_index = load_episode_index([Path(path) for path in args.episodes]) if args.episodes else {}
    grouped_cases = group_rows(rows)
    summary_rows, hard_cases = summarize_grouped_cases(grouped_cases, episode_index, args.long_step_threshold, args.long_prev_segments_threshold)
    action_rows = action_breakdown(grouped_cases)

    write_jsonl(output_dir / f"{args.model_key}_hard_cases.jsonl", hard_cases)
    (output_dir / f"{args.model_key}_hard_summary.json").write_text(
        json.dumps({"summary": summary_rows, "action_breakdown": action_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(output_dir / f"{args.model_key}_hard_case_report.md", args.model_key, input_path, rows, summary_rows, hard_cases, action_rows)
    print(f"rows={len(rows)} complete_cases={len(grouped_cases)} hard_cases={len(hard_cases)} output={output_dir}")


if __name__ == "__main__":
    main()