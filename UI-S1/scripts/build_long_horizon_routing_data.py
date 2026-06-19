#!/usr/bin/env python3
"""Build selective long-horizon memory routing data from bottleneck validation results."""

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


def load_episodes(paths: list[Path]) -> dict[str, JsonDict]:
    episodes = {}
    for path in paths:
        for episode in iter_jsonl(path):
            episodes[str(episode.get("episode_id"))] = episode
    return episodes


def find_segment(episode: JsonDict | None, step_index: int) -> JsonDict | None:
    if not episode:
        return None
    for segment in episode.get("segments", []):
        if int(segment.get("start_step", 0)) <= step_index <= int(segment.get("end_step", -1)):
            return segment
    return None


def action_name(row: JsonDict) -> str:
    return str(row.get("gt_action", {}).get("action", "unknown"))


def group_rows(rows: Iterable[JsonDict]) -> dict[tuple[str, str, str, int], dict[str, JsonDict]]:
    grouped: dict[tuple[str, str, str, int], dict[str, JsonDict]] = defaultdict(dict)
    for row in rows:
        key = (
            str(row.get("model_key", "unknown")),
            str(row.get("thinking_mode", "unknown")),
            str(row.get("case_kind", "unknown")),
            int(row.get("case_id", -1)),
        )
        grouped[key][str(row.get("condition"))] = row
    return {key: value for key, value in grouped.items() if all(condition in value for condition in CONDITIONS)}


def row_ok(row: JsonDict, metric: str = "value_match") -> bool:
    return bool(row.get(metric)) and not row.get("error")


def extract_features(rows_by_condition: dict[str, JsonDict], episodes: dict[str, JsonDict], long_step_threshold: int, prev_segments_threshold: int) -> JsonDict:
    base = rows_by_condition["no_history"]
    episode = episodes.get(str(base.get("episode_id")))
    step_index = int(base.get("step_index") or 0)
    segment = find_segment(episode, step_index)
    prev_segments = 0
    total_steps = None
    if episode:
        total_steps = len(episode.get("steps", []))
        prev_segments = sum(1 for item in episode.get("segments", []) if int(item.get("end_step", -1)) < step_index)
    carried_values = []
    memory_strength = "unknown"
    dominant_capability = "unknown"
    segment_start = None
    segment_len_so_far = None
    if segment:
        carried_values = segment.get("carried_values", []) or []
        memory_strength = (segment.get("memory_need", {}) or {}).get("strength", "unknown")
        dominant_capability = str(segment.get("dominant_capability", "unknown"))
        segment_start = int(segment.get("start_step", step_index))
        segment_len_so_far = step_index - segment_start + 1
    is_long_horizon = (
        step_index >= long_step_threshold
        or prev_segments >= prev_segments_threshold
        or bool(carried_values)
        or memory_strength in {"medium", "high"}
    )
    return {
        "episode_id": base.get("episode_id"),
        "step_index": step_index,
        "total_steps": total_steps,
        "case_kind": base.get("case_kind"),
        "gt_action": base.get("gt_action"),
        "gt_action_type": action_name(base),
        "screenshot": base.get("screenshot", ""),
        "prev_segments": prev_segments,
        "segment_start": segment_start,
        "segment_len_so_far": segment_len_so_far,
        "carried_values": carried_values,
        "carried_value_count": len(carried_values),
        "memory_strength": memory_strength,
        "dominant_capability": dominant_capability,
        "is_long_horizon": is_long_horizon,
    }


def choose_route(rows_by_condition: dict[str, JsonDict]) -> tuple[str, str]:
    no_ok = row_ok(rows_by_condition["no_history"])
    segment_ok = row_ok(rows_by_condition["segment_summary"])
    full_ok = row_ok(rows_by_condition["full_history"])
    wrong_ok = row_ok(rows_by_condition["wrong_summary"])

    if segment_ok and not no_ok:
        return "use_segment_summary", "segment_rescue"
    if full_ok and not no_ok and not segment_ok:
        return "use_full_history", "full_history_rescue_only"
    if no_ok and not segment_ok:
        return "use_no_history", "segment_regression"
    if segment_ok and not wrong_ok:
        return "use_segment_summary", "segment_beats_wrong"
    if no_ok:
        return "use_no_history", "current_screen_sufficient"
    if wrong_ok and not segment_ok:
        return "avoid_segment_summary", "wrong_beats_segment"
    return "escalate_or_replan", "all_conditions_wrong"


def build_examples(results_paths: list[Path], episodes: dict[str, JsonDict], long_step_threshold: int, prev_segments_threshold: int) -> list[JsonDict]:
    examples = []
    for results_path in results_paths:
        rows = list(iter_jsonl(results_path))
        grouped = group_rows(rows)
        for (model_key, thinking_mode, case_kind, case_id), rows_by_condition in grouped.items():
            features = extract_features(rows_by_condition, episodes, long_step_threshold, prev_segments_threshold)
            route, reason = choose_route(rows_by_condition)
            condition_value_match = {condition: row_ok(rows_by_condition[condition]) for condition in CONDITIONS}
            condition_type_match = {condition: bool(rows_by_condition[condition].get("type_match")) for condition in CONDITIONS}
            example = {
                "source_results": str(results_path),
                "model_key": model_key,
                "thinking_mode": thinking_mode,
                "case_id": case_id,
                "case_kind": case_kind,
                "route_label": route,
                "route_reason": reason,
                "use_memory": route in {"use_segment_summary", "use_full_history"},
                "condition_value_match": condition_value_match,
                "condition_type_match": condition_type_match,
                "features": features,
            }
            examples.append(example)
    return examples


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_report(path: Path, examples: list[JsonDict]) -> None:
    route_counts = Counter(item["route_label"] for item in examples)
    reason_counts = Counter(item["route_reason"] for item in examples)
    model_route_counts = Counter((item["model_key"], item["thinking_mode"], item["route_label"]) for item in examples)
    long_route_counts = Counter((item["features"]["is_long_horizon"], item["route_label"]) for item in examples)
    action_route_counts = Counter((item["features"]["gt_action_type"], item["route_label"]) for item in examples)
    lines = ["# Long-Horizon Routing Data Report", ""]
    lines.append(f"- examples: {len(examples)}")
    lines.append(f"- long_horizon: {sum(1 for item in examples if item['features']['is_long_horizon'])}")
    lines.append(f"- use_memory: {sum(1 for item in examples if item['use_memory'])}")
    lines.append("")
    lines.append("## Route Labels")
    lines.append("")
    lines.append("| route | n | rate |")
    lines.append("|---|---:|---:|")
    for route, count in route_counts.most_common():
        lines.append(f"| {route} | {count} | {count / len(examples):.3f} |")
    lines.append("")
    lines.append("## Route Reasons")
    lines.append("")
    lines.append("| reason | n |")
    lines.append("|---|---:|")
    for reason, count in reason_counts.most_common():
        lines.append(f"| {reason} | {count} |")
    lines.append("")
    lines.append("## Model And Thinking Mode")
    lines.append("")
    lines.append("| model | thinking | route | n |")
    lines.append("|---|---|---|---:|")
    for (model, mode, route), count in sorted(model_route_counts.items()):
        lines.append(f"| {model} | {mode} | {route} | {count} |")
    lines.append("")
    lines.append("## Long-Horizon Split")
    lines.append("")
    lines.append("| is_long_horizon | route | n |")
    lines.append("|---|---|---:|")
    for (is_long, route), count in sorted(long_route_counts.items()):
        lines.append(f"| {is_long} | {route} | {count} |")
    lines.append("")
    lines.append("## Action-Type Route Counts")
    lines.append("")
    lines.append("| action | route | n |")
    lines.append("|---|---|---:|")
    for (action, route), count in sorted(action_route_counts.items()):
        if count >= 10:
            lines.append(f"| {action} | {route} | {count} |")
    lines.append("")
    lines.append("## Suggested Use")
    lines.append("")
    lines.append("Train a selective memory/router head to predict `route_label` from current step, segment state, carried values, and local action/capability features. The highest-value positives are `use_segment_summary` with `segment_rescue` or `segment_beats_wrong`, especially when `is_long_horizon=true`.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build long-horizon selective-memory routing supervision")
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--episodes", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--long-step-threshold", type=int, default=10)
    parser.add_argument("--long-prev-segments-threshold", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = load_episodes([Path(path) for path in args.episodes])
    examples = build_examples([Path(path) for path in args.results], episodes, args.long_step_threshold, args.long_prev_segments_threshold)
    write_jsonl(output_dir / "routing_examples.jsonl", examples)
    write_report(output_dir / "routing_report.md", examples)
    print(f"examples={len(examples)} output={output_dir}")


if __name__ == "__main__":
    main()