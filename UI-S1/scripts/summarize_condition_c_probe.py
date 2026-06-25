#!/usr/bin/env python3
"""Summarize Condition C Stage 0/1 results."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


JsonDict = dict[str, Any]

NATURAL_VALUE_GAP = {
    "1": 0.4220,
    "2": 0.2183,
    "3": 0.1398,
    "4-5": 0.0571,
    "6-10": 0.0556,
}


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


def ordered_bins(items: dict[str, Any]) -> list[str]:
    order = {"1": 1, "2": 2, "3": 3, "4-5": 4, "6-10": 6, "11-20": 11, "21+": 21}
    return sorted(items, key=lambda key: (order.get(key, 999), key))


def summarize(rows: list[JsonDict], stage0_rows: list[JsonDict] | None = None) -> JsonDict:
    by_bin_value: dict[str, list[float]] = defaultdict(list)
    by_bin_type: dict[str, list[float]] = defaultdict(list)
    by_family: dict[str, list[float]] = defaultdict(list)
    by_family_type: dict[str, list[float]] = defaultdict(list)
    by_family_bin: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    errors = []
    for row in rows:
        if row.get("error"):
            errors.append(row)
            continue
        bin_name = str(row["distance_bin"])
        family = str(row.get("target_family", "unknown"))
        by_bin_value[bin_name].append(float(row.get("gap_value", 0.0)))
        by_bin_type[bin_name].append(float(row.get("gap_type", 0.0)))
        by_family[family].append(float(row.get("gap_value", 0.0)))
        by_family_type[family].append(float(row.get("gap_type", 0.0)))
        by_family_bin[family][bin_name].append(float(row.get("gap_value", 0.0)))
    summary = {
        "pairs": len(rows),
        "errors": len(errors),
        "value_gap_by_distance_bin": {key: mean_ci(by_bin_value[key]) for key in ordered_bins(by_bin_value)},
        "type_gap_by_distance_bin": {key: mean_ci(by_bin_type[key]) for key in ordered_bins(by_bin_type)},
        "family_split": {
            family: {
                "value_gap": mean_ci(values),
                "type_gap": mean_ci(by_family_type[family]),
            }
            for family, values in sorted(by_family.items())
        },
        "family_by_distance_bin": {
            family: {key: mean_ci(values_by_bin[key]) for key in ordered_bins(values_by_bin)}
            for family, values_by_bin in sorted(by_family_bin.items())
        },
        "natural_value_gap_overlay": NATURAL_VALUE_GAP,
    }
    if stage0_rows is not None:
        stage0_by_d: dict[str, list[float]] = defaultdict(list)
        stage0_type_by_d: dict[str, list[float]] = defaultdict(list)
        for row in stage0_rows:
            if row.get("error"):
                continue
            key = str(row["distance"])
            stage0_by_d[key].append(float(row.get("gap_value", 0.0)))
            stage0_type_by_d[key].append(float(row.get("gap_type", 0.0)))
        stage0 = {
            "value_gap_by_d": {key: mean_ci(stage0_by_d[key]) for key in ordered_bins(stage0_by_d)},
            "type_gap_by_d": {key: mean_ci(stage0_type_by_d[key]) for key in ordered_bins(stage0_type_by_d)},
        }
        stage0["gate_pass"] = all(abs(item["mean"]) <= 0.01 for item in stage0["value_gap_by_d"].values())
        summary["stage0_recap"] = stage0
    return summary


def write_table(lines: list[str], title: str, values: dict[str, JsonDict], natural: dict[str, float] | None = None) -> None:
    lines.append(f"## {title}")
    lines.append("")
    if natural:
        lines.append("| bin | n pairs | C gap mean | 95% CI | natural gap |")
        lines.append("|---|---:|---:|---:|---:|")
    else:
        lines.append("| bin | n pairs | gap mean | 95% CI |")
        lines.append("|---|---:|---:|---:|")
    for key in ordered_bins(values):
        item = values[key]
        ci = f"[{pct(item['ci95_low'])}, {pct(item['ci95_high'])}]"
        if natural:
            natural_value = natural.get(key)
            natural_text = pct(natural_value) if natural_value is not None else "n/a"
            lines.append(f"| {key} | {item['n']} | {pct(item['mean'])} | {ci} | {natural_text} |")
        else:
            lines.append(f"| {key} | {item['n']} | {pct(item['mean'])} | {ci} |")
    lines.append("")


def write_markdown(path: Path, summary: JsonDict, args: argparse.Namespace) -> None:
    lines = [f"# Condition C {args.stage.upper()} Summary", ""]
    lines.append("## Setup")
    lines.append("")
    lines.append("Condition C uses GT action history except for one externally injected action. The injected wrong action is A's real model-produced wrong `pred_action` at source step `j=k-d`. The current screen remains the GT screenshot at target step `k`.")
    lines.append("")
    lines.append(f"- rows: `{args.results}`")
    lines.append(f"- pairs: `{summary['pairs']}`")
    lines.append(f"- errors: `{summary['errors']}`")
    lines.append("")
    if "stage0_recap" in summary:
        lines.append("## Stage 0 Recap")
        lines.append("")
        lines.append(f"- gate pass: `{summary['stage0_recap']['gate_pass']}`")
        write_table(lines, "Stage 0 Zero-Point Value Gap", summary["stage0_recap"]["value_gap_by_d"])
        write_table(lines, "Stage 0 Zero-Point Type Gap", summary["stage0_recap"]["type_gap_by_d"])
    write_table(lines, "Stage 1 Value Gap vs Distance Bin", summary["value_gap_by_distance_bin"], summary.get("natural_value_gap_overlay"))
    write_table(lines, "Stage 1 Type Gap vs Distance Bin", summary["type_gap_by_distance_bin"])
    lines.append("## Family Split")
    lines.append("")
    lines.append("| family | n pairs | value gap | type gap |")
    lines.append("|---|---:|---:|---:|")
    for family, item in summary["family_split"].items():
        lines.append(f"| {family} | {item['value_gap']['n']} | {pct(item['value_gap']['mean'])} | {pct(item['type_gap']['mean'])} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("Compare C to the natural nearest-error curve on the same distance bins. If C is similar, the natural decay is mostly causal textual damage from one error. If C is much flatter/lower, the natural curve is driven by selection and/or accumulation of multiple errors.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Condition C")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["stage0", "stage1"], required=True)
    parser.add_argument("--stage0-results", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(iter_jsonl(args.results))
    stage0_rows = list(iter_jsonl(args.stage0_results)) if args.stage0_results else None
    summary = summarize(rows, stage0_rows=stage0_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_dir / "summary.md", summary, args)
    print(json.dumps({"output_dir": str(args.output_dir), "pairs": summary["pairs"], "errors": summary["errors"], "stage0_gate_pass": summary.get("stage0_recap", {}).get("gate_pass")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()