#!/usr/bin/env python3
"""Validate whether discovered segment boundaries behave like bottlenecks.

This is an offline proxy validation. It compares real segment boundaries against
random non-boundary positions from the same episodes, then reports whether real
boundaries concentrate action/text shifts, system navigation, route changes,
memory need, and carried values.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_trajectory_segments import text_blob, tokenize  # noqa: E402

JsonDict = dict[str, Any]


def iter_jsonl(path: Path) -> Iterable[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def action_type(step: JsonDict) -> str:
    return str(step.get("action", {}).get("type", "unknown"))


def button_name(step: JsonDict) -> str:
    return str(step.get("action", {}).get("args", {}).get("button", "")).lower()


def action_distribution(steps: list[JsonDict]) -> Counter[str]:
    return Counter(action_type(step) for step in steps)


def js_divergence(left: Counter[str], right: Counter[str]) -> float:
    keys = set(left) | set(right)
    left_total = sum(left.values()) or 1
    right_total = sum(right.values()) or 1
    left_probs = {key: left[key] / left_total for key in keys}
    right_probs = {key: right[key] / right_total for key in keys}
    mid = {key: 0.5 * (left_probs[key] + right_probs[key]) for key in keys}

    def kl(probs: dict[str, float], base: dict[str, float]) -> float:
        total = 0.0
        for key, prob in probs.items():
            if prob > 0 and base[key] > 0:
                total += prob * math.log(prob / base[key], 2)
        return total

    return 0.5 * kl(left_probs, mid) + 0.5 * kl(right_probs, mid)


def token_set(steps: list[JsonDict]) -> set[str]:
    tokens: set[str] = set()
    for step in steps:
        tokens.update(tokenize(text_blob(step)))
        args = step.get("action", {}).get("args", {})
        for key in ("text", "button", "status"):
            value = args.get(key)
            if isinstance(value, str):
                tokens.update(tokenize(value))
    return tokens


def jaccard_distance(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    if not left or not right:
        return 1.0
    return 1.0 - len(left & right) / len(left | right)


def local_windows(steps: list[JsonDict], boundary_step: int, radius: int) -> tuple[list[JsonDict], list[JsonDict]]:
    before = steps[max(0, boundary_step - radius) : boundary_step]
    after = steps[boundary_step : min(len(steps), boundary_step + radius)]
    return before, after


def raw_transition_features(steps: list[JsonDict], boundary_step: int) -> dict[str, Any]:
    prev_step = steps[boundary_step - 1]
    curr_step = steps[boundary_step]
    prev_action = action_type(prev_step)
    curr_action = action_type(curr_step)
    row = {
        "action_bigram": f"{prev_action}->{curr_action}",
        "prev_system_nav": prev_action == "system_button" or button_name(prev_step) in {"home", "back", "menu", "appselect"},
        "curr_system_nav": curr_action == "system_button" or button_name(curr_step) in {"home", "back", "menu", "appselect"},
        "curr_terminal": curr_action == "terminate",
        "prev_type": prev_action == "type",
        "curr_type": curr_action == "type",
        "changed_action_type": prev_action != curr_action,
    }
    if row["prev_system_nav"] or row["curr_system_nav"]:
        category = "surface_navigation"
    elif row["curr_terminal"]:
        category = "terminal"
    elif prev_action == "type" and curr_action != "type":
        category = "after_value_entry"
    elif prev_action in {"swipe", "wait"} and curr_action in {"click", "type"}:
        category = "browse_to_interact"
    elif row["changed_action_type"]:
        category = "capability_shift"
    else:
        category = "same_action"
    row["boundary_category"] = category
    return row


def typed_values(steps: list[JsonDict]) -> set[str]:
    values = set()
    for step in steps:
        args = step.get("action", {}).get("args", {})
        value = args.get("text")
        if isinstance(value, str) and value.strip():
            values.add(value.strip().lower())
    return values


def segment_by_start(episode: JsonDict) -> dict[int, JsonDict]:
    return {segment["start_step"]: segment for segment in episode.get("segments", [])}


def route_distance(prev_segment: JsonDict | None, curr_segment: JsonDict | None) -> float:
    if not prev_segment or not curr_segment:
        return 0.0
    left = set(prev_segment.get("candidate_routes", []))
    right = set(curr_segment.get("candidate_routes", []))
    return jaccard_distance(left, right)


def memory_strength(curr_segment: JsonDict | None) -> int:
    if not curr_segment:
        return 0
    value = curr_segment.get("memory_need", {}).get("strength", "none")
    return {"none": 0, "medium": 1, "high": 2}.get(value, 0)


def real_boundary_metrics(episode: JsonDict, boundary_step: int, radius: int) -> JsonDict:
    steps = episode["steps"]
    before, after = local_windows(steps, boundary_step, radius)
    segments = episode.get("segments", [])
    by_start = segment_by_start(episode)
    curr_segment = by_start.get(boundary_step)
    prev_segment = None
    for segment in segments:
        if segment["end_step"] == boundary_step - 1:
            prev_segment = segment
            break
    raw = raw_transition_features(steps, boundary_step)
    prev_values = typed_values(prev_segment.get("steps", []) if prev_segment else before)
    after_text = " ".join(text_blob(step).lower() for step in after)
    value_consumed_after = any(value and value in after_text for value in prev_values)
    return {
        "kind": "real",
        "benchmark": episode.get("benchmark"),
        "episode_id": episode.get("episode_id"),
        "boundary_step": boundary_step,
        "action_js": js_divergence(action_distribution(before), action_distribution(after)),
        "text_shift": jaccard_distance(token_set(before), token_set(after)),
        "route_shift": route_distance(prev_segment, curr_segment),
        "memory_strength": memory_strength(curr_segment),
        "value_consumed_after": value_consumed_after,
        **raw,
    }


def random_boundary_metrics(episode: JsonDict, boundary_step: int, radius: int) -> JsonDict:
    steps = episode["steps"]
    before, after = local_windows(steps, boundary_step, radius)
    raw = raw_transition_features(steps, boundary_step)
    return {
        "kind": "random",
        "benchmark": episode.get("benchmark"),
        "episode_id": episode.get("episode_id"),
        "boundary_step": boundary_step,
        "action_js": js_divergence(action_distribution(before), action_distribution(after)),
        "text_shift": jaccard_distance(token_set(before), token_set(after)),
        "route_shift": 0.0,
        "memory_strength": 0,
        "value_consumed_after": False,
        **raw,
    }


def sample_random_positions(episode: JsonDict, count: int, rng: random.Random) -> list[int]:
    steps = episode.get("steps", [])
    if len(steps) < 3 or count <= 0:
        return []
    real_starts = {segment.get("start_step") for segment in episode.get("segments", [])}
    candidates = [idx for idx in range(1, len(steps)) if idx not in real_starts]
    if not candidates:
        return []
    if len(candidates) <= count:
        return candidates
    return rng.sample(candidates, count)


def collect_metrics(paths: list[Path], radius: int, seed: int) -> list[JsonDict]:
    rng = random.Random(seed)
    rows: list[JsonDict] = []
    for path in paths:
        for episode in iter_jsonl(path):
            steps = episode.get("steps", [])
            if len(steps) < 2:
                continue
            real_starts = [segment["start_step"] for segment in episode.get("segments", []) if segment.get("start_step", 0) > 0]
            for start in real_starts:
                if 0 < start < len(steps):
                    rows.append(real_boundary_metrics(episode, start, radius))
            for start in sample_random_positions(episode, len(real_starts), rng):
                rows.append(random_boundary_metrics(episode, start, radius))
    return rows


def summarize(rows: list[JsonDict]) -> JsonDict:
    by_kind: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        by_kind[row["kind"]].append(row)

    numeric_fields = ["action_js", "text_shift", "route_shift", "memory_strength"]
    bool_fields = ["prev_system_nav", "curr_system_nav", "curr_terminal", "prev_type", "curr_type", "changed_action_type", "value_consumed_after"]
    summary: JsonDict = {}
    for kind, subset in by_kind.items():
        item: JsonDict = {"count": len(subset)}
        for field in numeric_fields:
            values = [float(row[field]) for row in subset]
            item[field] = round(sum(values) / len(values), 4) if values else 0.0
        for field in bool_fields:
            item[field] = round(sum(1 for row in subset if row[field]) / len(subset), 4) if subset else 0.0
        item["top_action_bigrams"] = dict(Counter(row["action_bigram"] for row in subset).most_common(12))
        item["benchmark_counts"] = dict(Counter(row["benchmark"] for row in subset).most_common())
        summary[kind] = item

    real = summary.get("real", {})
    rand = summary.get("random", {})
    lifts = {}
    for field in numeric_fields + bool_fields:
        rand_value = rand.get(field, 0.0)
        real_value = real.get(field, 0.0)
        lifts[field] = round(real_value / rand_value, 3) if rand_value else None
    summary["real_vs_random_lift"] = lifts
    category_summary = {}
    for category in sorted(set(row["boundary_category"] for row in rows)):
        subset = [row for row in rows if row["boundary_category"] == category]
        by_kind_category: dict[str, list[JsonDict]] = defaultdict(list)
        for row in subset:
            by_kind_category[row["kind"]].append(row)
        category_summary[category] = {}
        for kind, kind_rows in by_kind_category.items():
            category_summary[category][kind] = {
                "count": len(kind_rows),
                "action_js": round(sum(float(row["action_js"]) for row in kind_rows) / len(kind_rows), 4),
                "text_shift": round(sum(float(row["text_shift"]) for row in kind_rows) / len(kind_rows), 4),
                "top_action_bigrams": dict(Counter(row["action_bigram"] for row in kind_rows).most_common(8)),
            }
    summary["by_boundary_category"] = category_summary
    return summary


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, summary: JsonDict, rows: list[JsonDict]) -> None:
    lines = ["# Segmentation Bottleneck Validation", ""]
    lines.append("This is an offline proxy validation. It compares real segment boundaries against random non-boundary positions from the same episodes.")
    lines.append("")
    lines.append("## Aggregate Metrics")
    lines.append("")
    fields = ["action_js", "text_shift", "route_shift", "memory_strength", "prev_system_nav", "curr_system_nav", "curr_terminal", "changed_action_type", "value_consumed_after"]
    lines.append("| metric | real | random | lift |")
    lines.append("|---|---:|---:|---:|")
    for field in fields:
        real_value = summary.get("real", {}).get(field, 0.0)
        random_value = summary.get("random", {}).get(field, 0.0)
        lift = summary.get("real_vs_random_lift", {}).get(field)
        lift_text = "n/a" if lift is None else f"{lift:.2f}"
        lines.append(f"| `{field}` | {real_value:.4f} | {random_value:.4f} | {lift_text} |")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append(f"- Real boundaries: {summary.get('real', {}).get('count', 0)}")
    lines.append(f"- Random controls: {summary.get('random', {}).get('count', 0)}")
    lines.append("")
    lines.append("## Top Real Boundary Action Bigrams")
    lines.append("")
    for bigram, count in summary.get("real", {}).get("top_action_bigrams", {}).items():
        lines.append(f"- `{bigram}`: {count}")
    lines.append("")
    lines.append("## Boundary Category Breakdown")
    lines.append("")
    for category, category_summary in summary.get("by_boundary_category", {}).items():
        real = category_summary.get("real", {})
        random = category_summary.get("random", {})
        lines.append(f"### `{category}`")
        lines.append(f"- real_count: {real.get('count', 0)}")
        lines.append(f"- random_count: {random.get('count', 0)}")
        lines.append(f"- real_action_js: {real.get('action_js', 0.0):.4f}")
        lines.append(f"- random_action_js: {random.get('action_js', 0.0):.4f}")
        lines.append(f"- real_text_shift: {real.get('text_shift', 0.0):.4f}")
        lines.append(f"- random_text_shift: {random.get('text_shift', 0.0):.4f}")
        lines.append(f"- real_top_bigrams: {real.get('top_action_bigrams', {})}")
        lines.append("")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("A boundary is bottleneck-like if it has larger action/text shifts than random controls and also concentrates route shift, memory need, or value carry. Strong system-navigation lifts support surface-transition bottlenecks; semantic bottlenecks need additional text/screen or model-error validation.")
    lines.append("")
    lines.append("## Example Real Boundaries")
    lines.append("")
    for row in [row for row in rows if row["kind"] == "real"][:20]:
        lines.append("- " + json.dumps({k: row[k] for k in ("benchmark", "episode_id", "boundary_step", "action_bigram", "action_js", "text_shift", "route_shift", "memory_strength", "prev_system_nav", "curr_system_nav")}, ensure_ascii=False))
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate whether segment boundaries behave like bottlenecks")
    parser.add_argument("--inputs", nargs="+", default=[
        str(PROJECT_ROOT / "datasets" / "segmentation_train" / "gui_odyssey_segments.jsonl"),
        str(PROJECT_ROOT / "datasets" / "segmentation_train" / "android_control_segments.jsonl"),
    ])
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "datasets" / "bottleneck_validation_train"))
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(path) for path in args.inputs]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_metrics(input_paths, args.radius, args.seed)
    summary = summarize(rows)
    write_jsonl(output_dir / "boundary_metrics.jsonl", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(output_dir / "bottleneck_validation_report.md", summary, rows)
    print(f"rows={len(rows)} output={output_dir}")


if __name__ == "__main__":
    main()
