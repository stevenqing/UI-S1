#!/usr/bin/env python3
"""Summarize full-population on-policy state-repair probe results."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from run_on_policy_state_repair_probe import CONDITIONS, summarize


JsonDict = dict[str, Any]


def iter_jsonl(path: Path) -> Iterable[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_json(path: Path | None) -> JsonDict:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def pct(num: float, den: float) -> float:
    return num / den if den else 0.0


def fmt_rate(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def by_probe(rows: list[JsonDict]) -> dict[str, dict[str, JsonDict]]:
    grouped: dict[str, dict[str, JsonDict]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["probe_id"])][str(row["condition"])] = row
    return grouped


def family_probe_counts(grouped: dict[str, dict[str, JsonDict]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for conds in grouped.values():
        row = conds.get("screen_only") or next(iter(conds.values()))
        counts[str(row.get("rollout_failure_family", "unknown"))] += 1
    return counts


def family_condition_value_rates(grouped: dict[str, dict[str, JsonDict]]) -> dict[str, dict[str, float]]:
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    matches: dict[str, Counter[str]] = defaultdict(Counter)
    for conds in grouped.values():
        base = conds.get("screen_only") or next(iter(conds.values()))
        family = str(base.get("rollout_failure_family", "unknown"))
        for condition in CONDITIONS:
            if condition not in conds:
                continue
            totals[family][condition] += 1
            if conds[condition].get("value_match"):
                matches[family][condition] += 1
    return {
        family: {condition: pct(matches[family][condition], totals[family][condition]) for condition in CONDITIONS}
        for family in totals
    }


def write_markdown(path: Path, report: JsonDict) -> None:
    summary = report["summary"]
    manifest = report["manifest"]
    denominators = report["denominators"]
    rescue = summary.get("rescue", {})
    lines = ["# Full Test On-Policy State Repair Population", ""]
    lines.append("## Population")
    lines.append("")
    lines.append("| quantity | count |")
    lines.append("|---|---:|")
    lines.append(f"| GUI-Odyssey test episodes | {denominators['total_test_episodes']} |")
    lines.append(f"| rollout success episodes | {denominators['rollout_success_episodes']} |")
    lines.append(f"| rollout failed episodes with first-error probe | {denominators['failed_episodes_with_first_error']} |")
    lines.append(f"| probes evaluated | {summary.get('probes', 0)} |")
    lines.append(f"| screen-only wrong probes | {rescue.get('screen_only_wrong', 0)} |")
    lines.append("")
    lines.append("## Condition Accuracy")
    lines.append("")
    lines.append("| condition | rows | value match | type match | parse OK |")
    lines.append("|---|---:|---:|---:|---:|")
    for condition, stats in summary.get("condition_stats", {}).items():
        rows = stats.get("rows", 0)
        lines.append(
            f"| {condition} | {rows} | {stats.get('value_match', 0)} / {rows} ({fmt_rate(stats.get('value_match_rate', 0.0))}) | "
            f"{stats.get('type_match', 0)} / {rows} ({fmt_rate(stats.get('type_match_rate', 0.0))}) | "
            f"{stats.get('parse_ok', 0)} / {rows} ({fmt_rate(stats.get('parse_rate', 0.0))}) |"
        )
    lines.append("")
    lines.append("## Rescue Denominators")
    lines.append("")
    lines.append("| metric | count | over screen-only wrong | over failed episodes | over all test episodes |")
    lines.append("|---|---:|---:|---:|---:|")
    screen_wrong = denominators["screen_only_wrong"]
    failed = denominators["failed_episodes_with_first_error"]
    total = denominators["total_test_episodes"]
    for key in ["state_rescue", "clean_state_rescue", "wrong_state_rescue", "full_history_rescue", "local_unsolved", "rollout_drift_or_recoverable_screen_only"]:
        count = rescue.get(key, 0)
        lines.append(f"| {key} | {count} | {fmt_rate(pct(count, screen_wrong))} | {fmt_rate(pct(count, failed))} | {fmt_rate(pct(count, total))} |")
    lines.append("")
    lines.append("## Failure Family Distribution")
    lines.append("")
    lines.append("| family | probes | share of failed probes | screen-only wrong | state rescue | clean rescue | clean / screen-only wrong |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    family_counts = report["family_probe_counts"]
    rescue_by_family = summary.get("rescue_by_family", {})
    for family, probes in sorted(family_counts.items(), key=lambda item: (-item[1], item[0])):
        counts = rescue_by_family.get(family, {})
        family_screen_wrong = counts.get("screen_only_wrong", 0)
        lines.append(
            f"| {family} | {probes} | {fmt_rate(pct(probes, summary.get('probes', 0)))} | "
            f"{family_screen_wrong} | {counts.get('state_rescue', 0)} | {counts.get('clean_state_rescue', 0)} | "
            f"{fmt_rate(pct(counts.get('clean_state_rescue', 0), family_screen_wrong))} |"
        )
    lines.append("")
    lines.append("## Run Metadata")
    lines.append("")
    for key in ["selection_mode", "state_mode", "min_num_steps", "max_probes", "candidate_steps_before_cap", "probes_written"]:
        if key in manifest:
            lines.append(f"- {key}: `{manifest[key]}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize full-population state-repair probe results")
    parser.add_argument("--probe-results", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--total-test-episodes", type=int, default=1666)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(iter_jsonl(args.probe_results))
    summary = summarize(rows)
    manifest = read_json(args.probe_manifest)
    grouped = by_probe(rows)
    families = family_probe_counts(grouped)
    rescue = summary.get("rescue", {})
    denominators = {
        "total_test_episodes": int(manifest.get("dataset_episodes") or args.total_test_episodes),
        "rollout_success_episodes": int(manifest.get("rollout_success_episodes") or 0),
        "failed_episodes_with_first_error": int(manifest.get("failed_episodes_with_first_error") or summary.get("probes", 0)),
        "screen_only_wrong": int(rescue.get("screen_only_wrong", 0)),
    }
    report = {
        "manifest": manifest,
        "denominators": denominators,
        "summary": summary,
        "family_probe_counts": dict(families),
        "family_condition_value_rates": family_condition_value_rates(grouped),
    }
    output_json = args.output_json or args.probe_results.with_name("population_summary.json")
    output_md = args.output_md or args.probe_results.with_name("population_report.md")
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(output_md, report)
    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md), "denominators": denominators}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()