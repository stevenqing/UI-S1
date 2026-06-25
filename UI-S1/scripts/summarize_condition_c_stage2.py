#!/usr/bin/env python3
"""Summarize Condition C Stage2 dose-response results."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
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


def dose_summary(rows: list[JsonDict], headline_only: bool) -> dict[str, JsonDict]:
    by_dose_value: dict[int, list[float]] = defaultdict(list)
    by_dose_type: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("error"):
            continue
        if headline_only and not row.get("headline_dose3_eligible"):
            continue
        dose = int(row["dose"])
        by_dose_value[dose].append(float(row.get("gap_value", 0.0)))
        by_dose_type[dose].append(float(row.get("gap_type", 0.0)))
    return {
        str(dose): {
            "value_gap": mean_ci(by_dose_value[dose]),
            "type_gap": mean_ci(by_dose_type[dose]),
        }
        for dose in sorted(by_dose_value)
    }


def family_summary(rows: list[JsonDict]) -> dict[str, dict[str, JsonDict]]:
    by_family_dose: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_family_dose_type: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("error") or not row.get("headline_dose3_eligible"):
            continue
        family = str(row.get("target_family", "unknown"))
        dose = int(row["dose"])
        by_family_dose[family][dose].append(float(row.get("gap_value", 0.0)))
        by_family_dose_type[family][dose].append(float(row.get("gap_type", 0.0)))
    return {
        family: {
            str(dose): {
                "value_gap": mean_ci(values),
                "type_gap": mean_ci(by_family_dose_type[family][dose]),
            }
            for dose, values in sorted(doses.items())
        }
        for family, doses in sorted(by_family_dose.items())
    }


def summarize(zero_rows: list[JsonDict], main_rows: list[JsonDict]) -> JsonDict:
    zero_value = [float(row.get("gap_value", 0.0)) for row in zero_rows if not row.get("error")]
    zero_type = [float(row.get("gap_type", 0.0)) for row in zero_rows if not row.get("error")]
    zero = {"value_gap": mean_ci(zero_value), "type_gap": mean_ci(zero_type)}
    zero["gate_pass"] = abs(zero["value_gap"]["mean"]) <= 0.01
    headline = dose_summary(main_rows, headline_only=True)
    per_dose = dose_summary(main_rows, headline_only=False)
    return {
        "zero_point_multi": zero,
        "main_rows": len(main_rows),
        "main_errors": sum(1 for row in main_rows if row.get("error")),
        "headline_dose3_eligible_targets": len({(row["episode_id"], row["target_step"]) for row in main_rows if row.get("headline_dose3_eligible")}),
        "headline_within_target_by_dose": headline,
        "per_dose_all_eligible": per_dose,
        "family_headline_by_dose": family_summary(main_rows),
    }


def write_dose_table(lines: list[str], title: str, summary: dict[str, JsonDict]) -> None:
    lines.append(f"## {title}")
    lines.append("")
    lines.append("| dose | n pairs | value gap | value 95% CI | type gap | type 95% CI |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for dose in sorted(summary, key=lambda item: int(item)):
        item = summary[dose]
        v = item["value_gap"]
        t = item["type_gap"]
        lines.append(f"| {dose} | {v['n']} | {pct(v['mean'])} | [{pct(v['ci95_low'])}, {pct(v['ci95_high'])}] | {pct(t['mean'])} | [{pct(t['ci95_low'])}, {pct(t['ci95_high'])}] |")
    lines.append("")


def write_report(path: Path, summary: JsonDict, args: argparse.Namespace) -> None:
    lines = ["# Condition C Stage2 Dose-Response Summary", ""]
    lines.append("## Zero-Point Multi-Injection Gate")
    lines.append("")
    z = summary["zero_point_multi"]
    lines.append(f"- gate pass: `{z['gate_pass']}`")
    lines.append(f"- value gap: `{pct(z['value_gap']['mean'])}` over `{z['value_gap']['n']}` targets, CI [{pct(z['value_gap']['ci95_low'])}, {pct(z['value_gap']['ci95_high'])}]")
    lines.append(f"- type gap: `{pct(z['type_gap']['mean'])}` over `{z['type_gap']['n']}` targets, CI [{pct(z['type_gap']['ci95_low'])}, {pct(z['type_gap']['ci95_high'])}]")
    lines.append("")
    lines.append("## Support")
    lines.append("")
    lines.append(f"- headline dose3-eligible targets: `{summary['headline_dose3_eligible_targets']}`")
    lines.append(f"- main rows: `{summary['main_rows']}`")
    lines.append(f"- main errors: `{summary['main_errors']}`")
    lines.append("")
    write_dose_table(lines, "Headline Within-Target Dose Curve", summary["headline_within_target_by_dose"])
    write_dose_table(lines, "Secondary Per-Dose All-Eligible Sets", summary["per_dose_all_eligible"])
    lines.append("## Family Split (Headline Set)")
    lines.append("")
    for family, by_dose in summary["family_headline_by_dose"].items():
        lines.append(f"### {family}")
        lines.append("")
        lines.append("| dose | n pairs | value gap | type gap |")
        lines.append("|---|---:|---:|---:|")
        for dose in sorted(by_dose, key=lambda item: int(item)):
            item = by_dose[dose]
            lines.append(f"| {dose} | {item['value_gap']['n']} | {pct(item['value_gap']['mean'])} | {pct(item['type_gap']['mean'])} |")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("If the headline dose curve is flat near zero, multi-error textual damage is absent under GT-screen rescue. If it rises with dose, recent multi-error accumulation has a causal text-history effect.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Condition C Stage2 dose response")
    parser.add_argument("--zero-results", type=Path, required=True)
    parser.add_argument("--main-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    zero_rows = list(iter_jsonl(args.zero_results))
    main_rows = list(iter_jsonl(args.main_results))
    summary = summarize(zero_rows, main_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args.output_dir / "summary.md", summary, args)
    print(json.dumps({"output_dir": str(args.output_dir), "zero_gate_pass": summary["zero_point_multi"]["gate_pass"], "headline_targets": summary["headline_dose3_eligible_targets"], "main_rows": summary["main_rows"], "main_errors": summary["main_errors"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()