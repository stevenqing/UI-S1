#!/usr/bin/env python3
"""Mine raw transition features that predict segment boundaries.

Input is the train segmentation JSONL. The labels are weak segment starts, but
features are computed only from raw/canonical step fields: action, text,
coordinates, bbox, and goal overlap.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from discover_segmentation_schema import (  # noqa: E402
    action_type,
    button_name,
    coord_region,
    jaccard,
    text_tokens,
)
from analyze_trajectory_segments import tokenize  # noqa: E402

JsonDict = dict[str, Any]


def bucket(value: float, bins: list[tuple[float, str]]) -> str:
    for threshold, name in bins:
        if value <= threshold:
            return name
    return bins[-1][1]


def transition_features(prev_step: JsonDict, curr_step: JsonDict, goal_tokens: set[str]) -> set[str]:
    prev_tokens = text_tokens(prev_step)
    curr_tokens = text_tokens(curr_step)
    prev_action = action_type(prev_step)
    curr_action = action_type(curr_step)
    text_shift = 1.0 - jaccard(prev_tokens, curr_tokens)
    prev_goal = jaccard(prev_tokens, goal_tokens)
    curr_goal = jaccard(curr_tokens, goal_tokens)
    goal_delta = curr_goal - prev_goal

    features = {
        f"prev_action:{prev_action}",
        f"curr_action:{curr_action}",
        f"action_bigram:{prev_action}->{curr_action}",
        "same_action_type" if prev_action == curr_action else "changed_action_type",
        f"text_shift:{bucket(text_shift, [(0.25, 'low'), (0.55, 'mid'), (0.80, 'high'), (1.01, 'very_high')])}",
        f"goal_delta:{bucket(goal_delta, [(-0.10, 'down'), (0.10, 'flat'), (1.01, 'up')])}",
    }
    if button_name(prev_step) in {"home", "back", "menu", "appselect"}:
        features.add("prev_system_nav")
    if button_name(curr_step) in {"home", "back", "menu", "appselect"}:
        features.add("curr_system_nav")
    if curr_action == "terminate":
        features.add("curr_terminal")
    if curr_action == "open":
        features.add("curr_open_surface")
    if prev_action == "type":
        features.add("prev_entered_value")
    if curr_action == "type":
        features.add("curr_enters_value")
    if prev_step.get("grounding", {}).get("bbox"):
        features.add("prev_has_bbox")
    if curr_step.get("grounding", {}).get("bbox"):
        features.add("curr_has_bbox")
    prev_region = coord_region(prev_step)
    curr_region = coord_region(curr_step)
    if prev_region:
        features.add(f"prev_{prev_region}")
    if curr_region:
        features.add(f"curr_{curr_region}")
    for token, _ in Counter((prev_tokens | curr_tokens) & goal_tokens).most_common(8):
        features.add(f"goal_tok:{token}")
    return features


def iter_jsonl(path: Path) -> Iterable[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def collect_counts(paths: list[Path]) -> tuple[Counter[str], Counter[str], Counter[str], dict[str, list[JsonDict]], Counter[str]]:
    feature_total: Counter[str] = Counter()
    feature_boundary: Counter[str] = Counter()
    feature_nonboundary: Counter[str] = Counter()
    examples: dict[str, list[JsonDict]] = defaultdict(list)
    benchmark_counts: Counter[str] = Counter()

    for path in paths:
        for episode in iter_jsonl(path):
            benchmark = episode.get("benchmark", path.stem)
            benchmark_counts[benchmark] += 1
            steps = episode.get("steps", [])
            segment_starts = {segment.get("start_step") for segment in episode.get("segments", []) if segment.get("start_step") not in (None, 0)}
            goal_tokens = set(tokenize(episode.get("task_goal", "")))
            for index in range(1, len(steps)):
                is_boundary = index in segment_starts
                features = transition_features(steps[index - 1], steps[index], goal_tokens)
                for feature in features:
                    feature_total[feature] += 1
                    if is_boundary:
                        feature_boundary[feature] += 1
                    else:
                        feature_nonboundary[feature] += 1
                    if is_boundary and len(examples[feature]) < 5:
                        examples[feature].append(
                            {
                                "benchmark": benchmark,
                                "episode_id": episode.get("episode_id"),
                                "step_index": index,
                                "prev_action": action_type(steps[index - 1]),
                                "curr_action": action_type(steps[index]),
                                "prev_instruction": steps[index - 1].get("instruction") or steps[index - 1].get("text_fields", {}).get("instruction", ""),
                                "curr_instruction": steps[index].get("instruction") or steps[index].get("text_fields", {}).get("instruction", ""),
                            }
                        )
    return feature_total, feature_boundary, feature_nonboundary, examples, benchmark_counts


def score_features(feature_total: Counter[str], feature_boundary: Counter[str], feature_nonboundary: Counter[str], min_count: int) -> list[JsonDict]:
    total_boundary = sum(feature_boundary.values())
    total_all = sum(feature_total.values())
    base_rate = total_boundary / total_all if total_all else 0.0
    rows = []
    for feature, total in feature_total.items():
        if total < min_count:
            continue
        boundary = feature_boundary[feature]
        nonboundary = feature_nonboundary[feature]
        precision = boundary / total if total else 0.0
        recall = boundary / total_boundary if total_boundary else 0.0
        lift = precision / base_rate if base_rate else 0.0
        # Log odds with add-one smoothing, useful for rare but decisive features.
        odds_feature = (boundary + 1) / (nonboundary + 1)
        odds_global = (total_boundary + 1) / (total_all - total_boundary + 1)
        log_odds = math.log(odds_feature / odds_global)
        rows.append(
            {
                "feature": feature,
                "count": total,
                "boundary": boundary,
                "nonboundary": nonboundary,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "lift": round(lift, 2),
                "log_odds": round(log_odds, 3),
            }
        )
    return sorted(rows, key=lambda row: (row["lift"], row["precision"], row["count"]), reverse=True)


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, rows: list[JsonDict], examples: dict[str, list[JsonDict]], benchmark_counts: Counter[str]) -> None:
    lines = [
        "# Boundary Signal Mining Report",
        "",
        "This report treats train segmentation starts as weak boundary labels and mines which raw transition features predict those boundaries.",
        "",
        "## Inputs",
        "",
    ]
    for benchmark, count in benchmark_counts.most_common():
        lines.append(f"- `{benchmark}` episodes: {count}")
    lines.extend(["", "## Strongest Boundary-Predictive Raw Features", ""])
    lines.append("| feature | count | boundary | precision | recall | lift |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in rows[:40]:
        lines.append(f"| `{row['feature']}` | {row['count']} | {row['boundary']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['lift']:.2f} |")
    lines.extend(["", "## Examples", ""])
    for row in rows[:12]:
        feature = row["feature"]
        lines.append(f"### `{feature}`")
        lines.append(f"precision={row['precision']:.3f}, recall={row['recall']:.3f}, lift={row['lift']:.2f}")
        for example in examples.get(feature, [])[:3]:
            lines.append("- " + json.dumps(example, ensure_ascii=False))
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine raw transition features that predict weak segment boundaries")
    parser.add_argument("--inputs", nargs="+", default=[
        str(PROJECT_ROOT / "datasets" / "segmentation_train" / "gui_odyssey_segments.jsonl"),
        str(PROJECT_ROOT / "datasets" / "segmentation_train" / "android_control_segments.jsonl"),
    ])
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "datasets" / "schema_discovery_train"))
    parser.add_argument("--min-count", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(path) for path in args.inputs]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_total, feature_boundary, feature_nonboundary, examples, benchmark_counts = collect_counts(input_paths)
    rows = score_features(feature_total, feature_boundary, feature_nonboundary, args.min_count)
    write_jsonl(output_dir / "boundary_signal_scores.jsonl", rows)
    write_report(output_dir / "boundary_signal_report.md", rows, examples, benchmark_counts)
    print(f"features={len(rows)} output={output_dir}")


if __name__ == "__main__":
    main()
