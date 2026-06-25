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
    lines.append("## How We Did It")
    lines.append("")
    lines.append("This is a full-test first-error probe over the Qwen3.5 GUI-Odyssey rollout, not the earlier 400-sample enriched probe.")
    lines.append("")
    lines.append("Source artifacts:")
    lines.append("")
    lines.append("| artifact | path |")
    lines.append("|---|---|")
    lines.append("| corrected Qwen3.5 rollout | `outputs/qwen35_9b_baseline_template_gui_odyssey_full_random_split_test_8gpu_512threads/merged_corrected_direct1k/trajectory_results.jsonl` |")
    lines.append("| GUI-Odyssey test set | `datasets/GUI-Odyssey/gui_odyssey_random_split_test.jsonl` |")
    lines.append("| test segmentation/state metadata | `datasets/segmentation_test/gui_odyssey_segments.jsonl` |")
    lines.append("| generated probe file | `outputs/gui_odyssey_on_policy_state_repair_probe/qwen35_prefix_only_no_future_full1666/probes_first_error_all_prefix_only_no_future.jsonl` |")
    lines.append("| raw probe results | `outputs/gui_odyssey_on_policy_state_repair_probe/qwen35_prefix_only_no_future_full1666/probe_results.jsonl` |")
    lines.append("")
    lines.append("Procedure:")
    lines.append("")
    lines.append("1. Start from all 1666 GUI-Odyssey random-split test episodes and the corrected direct-1k Qwen3.5 rollout results.")
    lines.append("2. Keep successful rollout episodes only in the population denominator. Do not create probes for them.")
    lines.append("3. For every failed rollout episode, select exactly the first step whose rollout action failed GUI-Odyssey matching. This produces one first-error probe per failed episode.")
    lines.append("4. Do not length-filter or family-enrich this run. `selection_mode=first_error_all`, `min_num_steps=0`, and `max_probes=0` mean the probe set is the full failed-episode population.")
    lines.append("5. For each probe, keep the global task, current screenshot, ground-truth action/check options, and rollout failure family fixed. Only the supplied task-state text changes across conditions.")
    lines.append("6. Use the strict `prefix_only_no_future` state construction: completed segments, current step index, prior steps inside the current segment, and carried values already available from the goal or prior steps. Upcoming segments and the future part of the current segment are withheld.")
    lines.append("7. Evaluate four conditions with the same Qwen3.5 model endpoint: `screen_only`, `correct_task_state`, `wrong_task_state`, and `full_history`.")
    lines.append("8. Parse each model response into one GUI action and evaluate it with the corrected GUI-Odyssey direct `[0,1000]` action matcher.")
    lines.append("")
    lines.append("Condition definitions:")
    lines.append("")
    lines.append("| condition | prompt state |")
    lines.append("|---|---|")
    lines.append("| `screen_only` | global task + current screenshot, no extra task state |")
    lines.append("| `correct_task_state` | global task + current screenshot + prefix-only no-future state from the same episode |")
    lines.append("| `wrong_task_state` | global task + current screenshot + prefix-only no-future state sampled from another episode |")
    lines.append("| `full_history` | global task + current screenshot + ground-truth prior progress + correct prefix-only state |")
    lines.append("")
    lines.append("Rescue definitions:")
    lines.append("")
    lines.append("| metric | definition |")
    lines.append("|---|---|")
    lines.append("| `state_rescue` | `screen_only` is wrong and `correct_task_state` is correct |")
    lines.append("| `clean_state_rescue` | `screen_only` is wrong, `correct_task_state` is correct, and `wrong_task_state` is wrong |")
    lines.append("| `wrong_state_rescue` | `screen_only` is wrong and `wrong_task_state` is correct |")
    lines.append("| `full_history_rescue` | `screen_only` is wrong and `full_history` is correct |")
    lines.append("| `local_unsolved` | `screen_only`, `correct_task_state`, and `full_history` are all wrong |")
    lines.append("| `rollout_drift_or_recoverable_screen_only` | the original rollout failed, but the fresh `screen_only` probe succeeds |")
    lines.append("")
    lines.append("Reproduction commands:")
    lines.append("")
    lines.append("```bash")
    lines.append(".venv/bin/python scripts/build_on_policy_state_repair_probes.py \\")
    lines.append("  --trajectory-results outputs/qwen35_9b_baseline_template_gui_odyssey_full_random_split_test_8gpu_512threads/merged_corrected_direct1k/trajectory_results.jsonl \\")
    lines.append("  --jsonl-file datasets/GUI-Odyssey/gui_odyssey_random_split_test.jsonl \\")
    lines.append("  --segments datasets/segmentation_test/gui_odyssey_segments.jsonl \\")
    lines.append("  --output outputs/gui_odyssey_on_policy_state_repair_probe/qwen35_prefix_only_no_future_full1666/probes_first_error_all_prefix_only_no_future.jsonl \\")
    lines.append("  --selection-mode first_error_all \\")
    lines.append("  --state-mode prefix_only_no_future \\")
    lines.append("  --min-num-steps 0 \\")
    lines.append("  --max-probes 0")
    lines.append("")
    lines.append("PORT_BASE=8090 FORCE_BUILD=0 FORCE_RUN=1 FORCE_SUMMARY=1 \\")
    lines.append("  bash scripts/run_qwen35_full1666_prefix_only_state_repair_probe.sh")
    lines.append("```")
    lines.append("")
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