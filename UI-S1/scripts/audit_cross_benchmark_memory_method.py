#!/usr/bin/env python3
"""Audit whether benchmarks support the counterfactual memory-router method.

This script checks canonical segmented episodes, not model accuracy. It answers:
which benchmarks expose the trajectory, action, screenshot, instruction, segment,
and carried-value structure needed to run no-history / segment-memory /
wrong-memory context interventions and train a memory-specific candidate router.
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


def load_records(paths: list[Path], max_records: int) -> list[JsonDict]:
    records = []
    for path in paths:
        count = 0
        for row in iter_jsonl(path):
            records.append(row)
            count += 1
            if max_records and count >= max_records:
                break
    return records


def has_text(value: Any) -> bool:
    return bool(str(value or "").strip())


def step_fields(step: JsonDict) -> JsonDict:
    fields = step.get("text_fields", {}) or {}
    action = step.get("action", {}) or {}
    args = action.get("args", {}) or {}
    grounding = step.get("grounding", {}) or {}
    return {
        "has_screenshot": has_text(step.get("screenshot")),
        "has_action_type": has_text(action.get("type")),
        "has_action_value": any(has_text(args.get(key)) for key in ["text", "button", "status"]),
        "has_coordinate": bool(args.get("coordinate") or grounding.get("coordinate")),
        "has_bbox": bool(grounding.get("bbox")),
        "has_instruction": has_text(fields.get("instruction")),
        "has_observation": has_text(fields.get("observation")),
        "has_local_hint": has_text(fields.get("thought")),
        "action_type": str(action.get("type", "unknown")),
    }


def percent(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize_benchmark(records: list[JsonDict]) -> JsonDict:
    steps = [step for record in records for step in record.get("steps", [])]
    segments = [segment for record in records for segment in record.get("segments", [])]
    step_stats = [step_fields(step) for step in steps]
    action_counts = Counter(item["action_type"] for item in step_stats)
    memory_counts = Counter((segment.get("memory_need", {}) or {}).get("strength", "unknown") for segment in segments)
    capability_counts = Counter(segment.get("dominant_capability", "unknown") for segment in segments)
    carried_segments = [segment for segment in segments if segment.get("carried_values")]
    non_initial_segments = [segment for segment in segments if int(segment.get("segment_id", 0)) > 0]
    benchmark = records[0].get("benchmark", "unknown") if records else "unknown"
    totals = {
        "benchmark": benchmark,
        "episodes": len(records),
        "steps": len(steps),
        "segments": len(segments),
        "avg_steps_per_episode": len(steps) / len(records) if records else 0.0,
        "avg_segments_per_episode": len(segments) / len(records) if records else 0.0,
        "step_instruction_rate": percent(sum(item["has_instruction"] for item in step_stats), len(step_stats)),
        "step_observation_rate": percent(sum(item["has_observation"] for item in step_stats), len(step_stats)),
        "step_local_hint_rate": percent(sum(item["has_local_hint"] for item in step_stats), len(step_stats)),
        "screenshot_rate": percent(sum(item["has_screenshot"] for item in step_stats), len(step_stats)),
        "action_type_rate": percent(sum(item["has_action_type"] for item in step_stats), len(step_stats)),
        "action_value_rate": percent(sum(item["has_action_value"] for item in step_stats), len(step_stats)),
        "coordinate_rate": percent(sum(item["has_coordinate"] for item in step_stats), len(step_stats)),
        "bbox_rate": percent(sum(item["has_bbox"] for item in step_stats), len(step_stats)),
        "non_initial_segment_rate": percent(len(non_initial_segments), len(segments)),
        "carried_value_segment_rate": percent(len(carried_segments), len(segments)),
        "memory_medium_high_rate": percent(sum(memory_counts[key] for key in ["medium", "high"]), len(segments)),
        "action_types": dict(action_counts.most_common()),
        "memory_need": dict(memory_counts.most_common()),
        "dominant_capabilities": dict(capability_counts.most_common(12)),
    }
    totals["supports_segmentation"] = bool(records and steps and segments and totals["action_type_rate"] > 0.95)
    totals["supports_context_interventions"] = bool(totals["supports_segmentation"] and totals["screenshot_rate"] > 0.80)
    totals["supports_specificity_test"] = bool(totals["supports_context_interventions"] and len(records) > 1 and len(segments) > 1)
    totals["supports_progress_test"] = bool(totals["step_instruction_rate"] > 0.50)
    totals["portable_core_ready"] = bool(totals["supports_specificity_test"])
    totals["portable_full_ready"] = bool(totals["supports_specificity_test"] and totals["supports_progress_test"])
    if not totals["supports_progress_test"]:
        totals["progress_gap"] = "low current-step instruction coverage; use goal/action-derived intents or add benchmark adapter fields"
    else:
        totals["progress_gap"] = "none"
    return totals


def group_by_benchmark(records: list[JsonDict]) -> dict[str, list[JsonDict]]:
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("benchmark", "unknown"))].append(record)
    return grouped


def bool_mark(value: bool) -> str:
    return "yes" if value else "no"


def write_report(path: Path, summaries: list[JsonDict]) -> None:
    lines = [
        "# Cross-Benchmark Memory Method Audit",
        "",
        "This audit checks whether each canonical benchmark has the structural ingredients needed for the non-OCR counterfactual memory-router method.",
        "It does not claim final model transfer; it identifies which benchmarks are ready for behavior-intervention validation.",
        "",
        "## Portability Matrix",
        "",
        "| benchmark | episodes | steps | instr rate | screenshot rate | segmentation | context interventions | specificity test | progress test | core ready | full ready |",
        "|---|---:|---:|---:|---:|---|---|---|---|---|---|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['benchmark']} | {item['episodes']} | {item['steps']} | {item['step_instruction_rate']:.1%} | "
            f"{item['screenshot_rate']:.1%} | {bool_mark(item['supports_segmentation'])} | "
            f"{bool_mark(item['supports_context_interventions'])} | {bool_mark(item['supports_specificity_test'])} | "
            f"{bool_mark(item['supports_progress_test'])} | {bool_mark(item['portable_core_ready'])} | {bool_mark(item['portable_full_ready'])} |"
        )
    lines.extend(["", "## Details", ""])
    for item in summaries:
        lines.extend([
            f"### {item['benchmark']}",
            "",
            f"- avg steps / episode: {item['avg_steps_per_episode']:.2f}",
            f"- avg segments / episode: {item['avg_segments_per_episode']:.2f}",
            f"- action value rate: {item['action_value_rate']:.1%}",
            f"- coordinate rate: {item['coordinate_rate']:.1%}",
            f"- bbox rate: {item['bbox_rate']:.1%}",
            f"- non-initial segment rate: {item['non_initial_segment_rate']:.1%}",
            f"- carried-value segment rate: {item['carried_value_segment_rate']:.1%}",
            f"- medium/high memory segment rate: {item['memory_medium_high_rate']:.1%}",
            f"- progress gap: {item['progress_gap']}",
            "",
            "Top action types:",
        ])
        for name, count in list(item["action_types"].items())[:10]:
            lines.append(f"- `{name}`: {count}")
        lines.extend(["", "Top capabilities:"])
        for name, count in list(item["dominant_capabilities"].items())[:10]:
            lines.append(f"- `{name}`: {count}")
        lines.append("")
    lines.extend([
        "## Interpretation",
        "",
        "Core portability means the benchmark can support no-history / segment-summary / wrong-summary interventions and candidate specificity analysis.",
        "Full portability additionally means current-step instruction text is available for the instruction-progress compatibility test.",
        "A benchmark that lacks instruction text can still test the core research method, but needs an adapter or a goal/action-derived intent proxy before using progress features.",
    ])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit cross-benchmark readiness for counterfactual memory routing")
    parser.add_argument("--segments", nargs="+", required=True, help="Canonical segmented episode JSONL files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-records-per-file", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(path) for path in args.segments]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(paths, args.max_records_per_file)
    grouped = group_by_benchmark(records)
    summaries = [summarize_benchmark(grouped[name]) for name in sorted(grouped)]
    (output_dir / "cross_benchmark_memory_method_audit.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(output_dir / "cross_benchmark_memory_method_audit.md", summaries)
    print(json.dumps({"benchmarks": [item["benchmark"] for item in summaries], "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
