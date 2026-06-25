#!/usr/bin/env python3
"""Summarize unified text-space error-horizon lower-bound probe rows."""

from __future__ import annotations

import argparse
import json
import math
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


def mean_ci(values: list[float]) -> JsonDict:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    mean = sum(values) / n
    if n == 1:
        return {"n": n, "mean": mean, "ci95_low": mean, "ci95_high": mean}
    var = sum((value - mean) ** 2 for value in values) / (n - 1)
    se = math.sqrt(var / n)
    return {"n": n, "mean": mean, "ci95_low": mean - 1.96 * se, "ci95_high": mean + 1.96 * se}


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def prefix_bin(count: int) -> str:
    if count <= 5:
        return str(count)
    if count <= 10:
        return "6-10"
    if count <= 20:
        return "11-20"
    return "21+"


def distance_bin(distance: int | None) -> str:
    if distance is None:
        return "no_prior_error"
    if distance <= 3:
        return str(distance)
    if distance <= 5:
        return "4-5"
    if distance <= 10:
        return "6-10"
    return "11+"


def absolute_depth_bin(depth: int) -> str:
    if depth <= 2:
        return str(depth)
    if depth <= 4:
        return "3-4"
    if depth <= 7:
        return "5-7"
    if depth <= 10:
        return "8-10"
    if depth <= 15:
        return "11-15"
    if depth <= 20:
        return "16-20"
    return "21+"


def normalized_depth_bin(value: float) -> str:
    edges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    for low, high in edges:
        if low <= value < high:
            return f"[{low:.1f},{min(high, 1.0):.1f})"
    return "unknown"


def sort_bins(items: dict[str, JsonDict]) -> dict[str, JsonDict]:
    order = {
        "no_prior_error": -1,
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "3-4": 3,
        "4-5": 4,
        "5-7": 5,
        "6-10": 6,
        "8-10": 8,
        "11-15": 11,
        "11-20": 11,
        "16-20": 16,
        "21+": 21,
    }
    return dict(sorted(items.items(), key=lambda item: (order.get(item[0], 999), item[0])))


def table(title: str, grouped: dict[str, JsonDict]) -> list[str]:
    lines = [f"## {title}", "", "| bin | n steps | gap mean | 95% CI |", "|---|---:|---:|---:|"]
    for label, stats in grouped.items():
        lines.append(f"| {label} | {stats['n']} | {pct(stats['mean'])} | [{pct(stats['ci95_low'])}, {pct(stats['ci95_high'])}] |")
    lines.append("")
    return lines


def summarize_rows(rows: list[JsonDict], zero_gate_threshold: float) -> JsonDict:
    groups_value_prefix: dict[str, list[float]] = defaultdict(list)
    groups_type_prefix: dict[str, list[float]] = defaultdict(list)
    groups_distance: dict[str, list[float]] = defaultdict(list)
    groups_abs_depth: dict[str, list[float]] = defaultdict(list)
    groups_norm_depth: dict[str, list[float]] = defaultdict(list)
    family_value: dict[str, list[float]] = defaultdict(list)
    family_type: dict[str, list[float]] = defaultdict(list)
    zero_value = []
    zero_type = []
    for row in rows:
        value_gap = float(row.get("gap_value_match", 0))
        type_gap = float(row.get("gap_type_match", 0))
        prefix_count = int(row.get("prefix_error_count") or 0)
        groups_value_prefix[prefix_bin(prefix_count)].append(value_gap)
        groups_type_prefix[prefix_bin(prefix_count)].append(type_gap)
        groups_distance[distance_bin(row.get("nearest_error_distance"))].append(value_gap)
        groups_abs_depth[absolute_depth_bin(int(row.get("absolute_depth") or row.get("step_index") or 0))].append(value_gap)
        groups_norm_depth[normalized_depth_bin(float(row.get("normalized_depth") or 0.0))].append(value_gap)
        family = str(row.get("action_family", "unknown"))
        family_value[family].append(value_gap)
        family_type[family].append(type_gap)
        if prefix_count == 0:
            zero_value.append(value_gap)
            zero_type.append(type_gap)
    zero_summary = {"value_gap": mean_ci(zero_value), "type_gap": mean_ci(zero_type)}
    by_family = {}
    for family in sorted(family_value):
        by_family[family] = {
            "n": len(family_value[family]),
            "value_gap": mean_ci(family_value[family]),
            "type_gap": mean_ci(family_type[family]),
        }
    return {
        "episodes": len({row.get("episode_id") for row in rows}),
        "steps": len(rows),
        "zero_gate_threshold": zero_gate_threshold,
        "zero_prefix_error_gate": zero_summary,
        "zero_gate_status": "PASS" if abs(zero_summary["value_gap"]["mean"]) <= zero_gate_threshold else "FAIL",
        "value_gap_by_prefix_error_count": sort_bins({key: mean_ci(values) for key, values in groups_value_prefix.items()}),
        "type_gap_by_prefix_error_count": sort_bins({key: mean_ci(values) for key, values in groups_type_prefix.items()}),
        "value_gap_by_nearest_error_distance": sort_bins({key: mean_ci(values) for key, values in groups_distance.items()}),
        "value_gap_by_absolute_depth": sort_bins({key: mean_ci(values) for key, values in groups_abs_depth.items()}),
        "value_gap_by_normalized_depth": dict(sorted({key: mean_ci(values) for key, values in groups_norm_depth.items()}.items())),
        "by_family": by_family,
        "prefix_error_count_distribution": dict(Counter(int(row.get("prefix_error_count") or 0) for row in rows).most_common()),
        "history_policy": "action_only_oracle_corrected",
    }


def write_report(path: Path, summary: JsonDict, probe_rows_path: Path) -> None:
    lines = ["# Text-Space Error-Horizon Lower-Bound Probe", ""]
    lines.append("## Setup")
    lines.append("")
    lines.append("This is an offline lower-bound probe: every current screen is the GT GUI-Odyssey screen. Only textual action history differs between A and B.")
    lines.append("")
    lines.append("A and B are both re-run in the same probe with matched action-only history format. This avoids comparing verbose model reasoning in A against terse GT history in B.")
    lines.append("")
    lines.append("| condition | history policy |")
    lines.append("|---|---|")
    lines.append("| A self/predicted | previous A predictions serialized as `<action>{...}</action>` |")
    lines.append("| B oracle/corrected | same as A when A's previous step was matcher-correct; GT action replaces A only after A is wrong |")
    lines.append("")
    lines.append("Therefore, when `prefix_error_count == 0`, A and B histories are identical by construction. The zero-prefix-error gap is the format/length artifact gate.")
    lines.append("")
    lines.append(f"Raw rows: `{probe_rows_path}`")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    zero = summary["zero_prefix_error_gate"]
    lines.append(f"- episodes: `{summary['episodes']}`")
    lines.append(f"- steps compared: `{summary['steps']}`")
    lines.append(f"- zero-prefix-error value gap: `{pct(zero['value_gap']['mean'])}` over `{zero['value_gap']['n']}` steps")
    lines.append(f"- zero-prefix-error type gap: `{pct(zero['type_gap']['mean'])}` over `{zero['type_gap']['n']}` steps")
    lines.append(f"- zero-prefix-error gate threshold: `{pct(summary['zero_gate_threshold'])}`")
    lines.append(f"- zero-prefix-error gate status: `{summary['zero_gate_status']}`")
    lines.append("")
    lines.extend(table("Primary: Value Gap vs Prefix Error Count", summary["value_gap_by_prefix_error_count"]))
    lines.extend(table("Secondary: Type Gap vs Prefix Error Count", summary["type_gap_by_prefix_error_count"]))
    lines.extend(table("Value Gap vs Nearest Error Distance", summary["value_gap_by_nearest_error_distance"]))
    lines.extend(table("Value Gap vs Absolute Depth", summary["value_gap_by_absolute_depth"]))
    lines.extend(table("Value Gap vs Normalized Depth", summary["value_gap_by_normalized_depth"]))
    lines.append("## Family Split")
    lines.append("")
    lines.append("| family | n steps | value gap | type gap |")
    lines.append("|---|---:|---:|---:|")
    for family, stats in summary["by_family"].items():
        lines.append(f"| {family} | {stats['n']} | {pct(stats['value_gap']['mean'])} | {pct(stats['type_gap']['mean'])} |")
    lines.append("")
    lines.append("## Interpretation Guardrail")
    lines.append("")
    lines.append("This is a lower bound on error-horizon: the GT screen at every step pulls the model back toward the correct state. Any positive gap is therefore text-space compounding despite screen rescue; a flat gap means text history errors are shallow/offline-benign, not that online off-GT screen drift is absent.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize text-space error-horizon probe rows")
    parser.add_argument("--probe-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zero-gate-threshold", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(iter_jsonl(args.probe_rows))
    summary = summarize_rows(rows, args.zero_gate_threshold)
    (args.output_dir / "text_error_horizon_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args.output_dir / "text_error_horizon_report.md", summary, args.probe_rows)
    print(json.dumps({"output_dir": str(args.output_dir), "zero_gate_status": summary["zero_gate_status"], "zero_value_gap": summary["zero_prefix_error_gate"]["value_gap"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()