#!/usr/bin/env python3
"""Characterize non-analyzable recoverable critical steps.

This diagnostic scopes the UIA element-selection verifier method by separating
true click element-selection failures from non-click action/content failures and
from click rows that remained non-analyzable in the spatial control-identity
analysis. It uses only existing sampled outputs and frozen matcher labels.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_SAMPLES = "outputs/critstep_elicit_uia/per_step.jsonl"
DEFAULT_UIA_PER_STEP = "outputs/critstep_reward_structure_uia/per_step.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/critstep_scope"

ACTION_ALIASES = {
    "tap": "click",
    "left_click": "click",
    "double_click": "click",
    "double_click_input": "click",
    "double_click_on_coordinates": "click",
    "input": "type",
    "drag": "swipe",
    "scroll": "swipe",
    "wheel_mouse_input": "swipe",
    "press": "key",
    "hotkey": "key",
    "shortcut": "key",
    "back": "system_button",
    "home": "system_button",
}

PRIMARY_SCOPE_ORDER = [
    "click_element_selection",
    "click_coordinate",
    "non_click_distinct_class",
    "click_non_analyzable_gap",
    "click_analyzable_other",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_action_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    return ACTION_ALIASES.get(text, text)


def action_category(value: Any) -> str:
    atype = normalize_action_type(value)
    if atype in {"click", "long_press"}:
        return "click"
    if atype == "type":
        return "type"
    if atype == "swipe":
        return "swipe"
    if atype in {"key", "system_button"}:
        return "special-key"
    return "other"


def pct(numer: float, denom: float) -> str:
    if not denom:
        return "0.00%"
    return f"{100.0 * numer / denom:.2f}%"


def bool_mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get(field)) / len(rows)


def number_summary(values: Sequence[float]) -> Dict[str, Optional[float]]:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return {"n": 0.0, "mean": None, "median": None}
    mid = len(vals) // 2
    median = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0
    return {"n": float(len(vals)), "mean": sum(vals) / len(vals), "median": median}


def key_for(row: Mapping[str, Any]) -> Tuple[str, int, str]:
    return (str(row.get("episode_id")), int(row.get("step_idx")), str(row.get("target_id")))


def uia_cause(uia_row: Mapping[str, Any]) -> Optional[str]:
    if uia_row.get("analyzable"):
        return None
    if int(uia_row.get("n_controls") or 0) == 0:
        return "no_controls"
    if str(uia_row.get("greedy_assignment") or "") == "no_point":
        return "greedy_no_point"
    if int(uia_row.get("n_correct_with_control") or 0) == 0:
        return "no_correct_control"
    return "other_non_analyzable"


def gt_type(sample_row: Mapping[str, Any]) -> str:
    greedy = sample_row.get("greedy") if isinstance(sample_row.get("greedy"), dict) else {}
    return normalize_action_type(greedy.get("gt_type") or sample_row.get("action_type"))


def greedy_type(sample_row: Mapping[str, Any]) -> str:
    greedy = sample_row.get("greedy") if isinstance(sample_row.get("greedy"), dict) else {}
    return normalize_action_type(greedy.get("pred_type") or "")


def sample_pred_type(sample: Mapping[str, Any]) -> str:
    pred_action = sample.get("pred_action") if isinstance(sample.get("pred_action"), dict) else {}
    return normalize_action_type(sample.get("pred_type") or pred_action.get("action") or "")


def correct_samples(row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    samples = row.get("samples") if isinstance(row.get("samples"), list) else []
    return [sample for sample in samples if isinstance(sample, dict) and sample.get("success")]


def first_correct_rank(row: Mapping[str, Any]) -> Optional[int]:
    value = row.get("first_correct_rank")
    if value is None:
        return None
    try:
        rank = int(value)
    except (TypeError, ValueError):
        return None
    return rank if rank > 0 else None


def rank_bin(rank: Optional[int]) -> str:
    if rank is None:
        return "missing@50"
    if rank == 1:
        return "1"
    if rank <= 5:
        return "2-5"
    if rank <= 10:
        return "6-10"
    if rank <= 20:
        return "11-20"
    return "21-50"


def failure_kind_for_non_click(gt_cat: str, greedy_cat: str, greedy_bucket: str) -> str:
    if greedy_bucket == "format_error" or greedy_cat == "other":
        return "FORMAT/PARSE error"
    if gt_cat != greedy_cat:
        return "ACTION-TYPE mismatch"
    if gt_cat == "type":
        return "TYPE-CONTENT error"
    if gt_cat == "swipe":
        return "SWIPE error"
    return "SPECIAL/other"


def failure_kind(sample_row: Mapping[str, Any], cause: str) -> str:
    gt_cat = action_category(gt_type(sample_row))
    greedy_cat = action_category(greedy_type(sample_row))
    greedy_bucket = str(sample_row.get("greedy_bucket") or (sample_row.get("greedy") or {}).get("bucket") or "")
    if gt_cat != "click":
        return failure_kind_for_non_click(gt_cat, greedy_cat, greedy_bucket)
    if cause == "greedy_no_point":
        return "CLICK non-analyzable: greedy no point"
    if cause == "no_correct_control":
        return "CLICK non-analyzable: no correct control"
    return "CLICK non-analyzable: other"


def scope_flag(gt_cat: str, cause: Optional[str]) -> str:
    if gt_cat != "click":
        return "true_non_click_out_of_scope"
    if cause == "greedy_no_point":
        return "click_parse_or_action_format_gap"
    if cause == "no_correct_control":
        return "click_correct_control_coverage_gap"
    if cause:
        return "click_other_coverage_gap"
    return "analyzable_or_not_applicable"


def scope_bucket(sample_row: Mapping[str, Any], uia_row: Mapping[str, Any]) -> str:
    gt_cat = action_category(gt_type(sample_row))
    if gt_cat != "click":
        return "non_click_distinct_class"
    if uia_row.get("analyzable") and uia_row.get("different_control_majority"):
        return "click_element_selection"
    if uia_row.get("analyzable") and uia_row.get("same_control_majority"):
        return "click_coordinate"
    if uia_row.get("analyzable"):
        return "click_analyzable_other"
    return "click_non_analyzable_gap"


def counter_table(counter: Counter, order: Optional[Sequence[str]] = None) -> List[Tuple[str, int]]:
    keys = list(order or []) + sorted(key for key in counter if not order or key not in order)
    return [(key, counter[key]) for key in keys if counter[key]]


def markdown_count_table(title: str, rows: Sequence[Tuple[str, int]], denom: int, key_label: str = "bucket") -> List[str]:
    lines = [f"## {title}", "", f"| {key_label} | count | share |", "|---|---:|---:|"]
    for key, count in rows:
        lines.append(f"| {key} | {count} | {pct(count, denom)} |")
    lines.append("")
    return lines


def markdown_crosstab(title: str, matrix: Mapping[str, Counter], row_order: Sequence[str], col_order: Sequence[str]) -> List[str]:
    lines = [f"## {title}", ""]
    header = "| GT \\ greedy | " + " | ".join(col_order) + " | total |"
    sep = "|---" + "|---:" * (len(col_order) + 1) + "|"
    lines.extend([header, sep])
    for row_key in row_order:
        row = matrix.get(row_key, Counter())
        total = sum(row.values())
        if not total:
            continue
        values = [str(row.get(col, 0)) for col in col_order]
        lines.append(f"| {row_key} | " + " | ".join(values) + f" | {total} |")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default=DEFAULT_SAMPLES)
    parser.add_argument("--uia-per-step", default=DEFAULT_UIA_PER_STEP)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--population", default="critical")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rows_all = [row for row in read_jsonl(Path(args.samples)) if str(row.get("population")) == args.population]
    sample_rows = [row for row in sample_rows_all if float(row.get("temperature")) == args.temperature]
    sample_by_key = {key_for(row): row for row in sample_rows}
    uia_rows = [row for row in read_jsonl(Path(args.uia_per_step)) if float(row.get("temperature")) == args.temperature]
    uia_by_key = {key_for(row): row for row in uia_rows}

    recoverable_rows = [row for row in sample_rows if row.get("recoverable")]
    recoverable_keys = {key_for(row) for row in recoverable_rows}
    missing_uia = sorted(recoverable_keys - set(uia_by_key))
    if missing_uia:
        raise SystemExit(f"UIA per-step is missing {len(missing_uia)} recoverable rows")

    non_analyzable_records: List[Dict[str, Any]] = []
    cause_counter = Counter()
    cause_gt = defaultdict(Counter)
    cause_greedy = defaultdict(Counter)
    gt_greedy_matrix = defaultdict(Counter)
    non_click_failure_kind = Counter()
    non_click_failure_by_gt = defaultdict(Counter)
    scope_counter = Counter()
    non_analyzable_scope_flags = Counter()

    for row in recoverable_rows:
        key = key_for(row)
        uia = uia_by_key[key]
        scope_counter[scope_bucket(row, uia)] += 1
        cause = uia_cause(uia)
        if cause is None:
            continue
        gt_raw = gt_type(row)
        greedy_raw = greedy_type(row)
        gt_cat = action_category(gt_raw)
        greedy_cat = action_category(greedy_raw)
        kind = failure_kind(row, cause)
        flag = scope_flag(gt_cat, cause)
        correct = correct_samples(row)
        correct_type_counter = Counter(action_category(sample_pred_type(sample)) for sample in correct)
        record = {
            "target_id": row.get("target_id"),
            "episode_id": row.get("episode_id"),
            "step_idx": row.get("step_idx"),
            "temperature": float(row.get("temperature")),
            "cause": cause,
            "gt_action_type": gt_raw,
            "gt_action_category": gt_cat,
            "greedy_action_type": greedy_raw or "parse_or_missing",
            "greedy_action_category": greedy_cat,
            "greedy_bucket": row.get("greedy_bucket"),
            "failure_kind": kind,
            "scope_flag": flag,
            "first_correct_rank": first_correct_rank(row),
            "success_count": int(row.get("success_count") or 0),
            "pass_at_1": bool(row.get("pass_at_1")),
            "pass_at_5": bool(row.get("pass_at_5")),
            "pass_at_10": bool(row.get("pass_at_10")),
            "pass_at_20": bool(row.get("pass_at_20")),
            "pass_at_50": bool(row.get("pass_at_50")),
            "n_correct_with_control": int(uia.get("n_correct_with_control") or 0),
            "greedy_assignment": uia.get("greedy_assignment"),
            "correct_action_category_counts": dict(correct_type_counter),
        }
        non_analyzable_records.append(record)
        cause_counter[cause] += 1
        cause_gt[cause][gt_cat] += 1
        cause_greedy[cause][greedy_cat] += 1
        gt_greedy_matrix[gt_cat][greedy_cat] += 1
        non_analyzable_scope_flags[flag] += 1
        if gt_cat != "click":
            non_click_failure_kind[kind] += 1
            non_click_failure_by_gt[gt_cat][kind] += 1

    non_click_primary = [row for row in sample_rows if action_category(gt_type(row)) != "click"]
    non_click_recoverable = [row for row in non_click_primary if row.get("recoverable")]
    non_click_non_analyzable = [record for record in non_analyzable_records if record["gt_action_category"] != "click"]
    click_non_analyzable = [record for record in non_analyzable_records if record["gt_action_category"] == "click"]
    rank_bins_non_click = Counter(rank_bin(first_correct_rank(row)) for row in non_click_primary)
    rank_bins_non_click_recoverable = Counter(rank_bin(first_correct_rank(row)) for row in non_click_recoverable)
    success_counts_non_click_recoverable = [float(row.get("success_count") or 0) for row in non_click_recoverable]

    total_recoverable = len(recoverable_rows)
    total_non_analyzable = len(non_analyzable_records)
    non_click_share = len(non_click_non_analyzable) / total_non_analyzable if total_non_analyzable else 0.0
    click_gap_share = len(click_non_analyzable) / total_non_analyzable if total_non_analyzable else 0.0
    if non_click_share >= 0.70 and click_gap_share < 0.25:
        gate = "SCOPE CLARIFIED"
        gate_reason = "Non-analyzable rows are dominantly true non-click actions; the element-selection verifier scope is cleanly bounded."
    elif click_gap_share >= 0.25:
        gate = "SCOPE NARROWER THAN EXPECTED"
        gate_reason = "A large share of non-analyzable rows are GT-click rows, so part of the gap is click parse/control coverage rather than a pure non-click scope boundary."
    else:
        gate = "MIXED"
        gate_reason = "The boundary between click element-selection and other failure classes is graded; report the split explicitly."

    summary = {
        "samples": args.samples,
        "uia_per_step": args.uia_per_step,
        "temperature": args.temperature,
        "population": args.population,
        "total_primary_failures": len(sample_rows),
        "total_recoverable_primary": total_recoverable,
        "total_non_analyzable": total_non_analyzable,
        "cause_counts": dict(cause_counter),
        "scope_counts": dict(scope_counter),
        "scope_fractions_of_recoverable": {key: scope_counter[key] / total_recoverable for key in scope_counter},
        "non_analyzable_scope_flags": dict(non_analyzable_scope_flags),
        "non_click_primary": {
            "rows": len(non_click_primary),
            "recoverable": len(non_click_recoverable),
            "recoverable_fraction": len(non_click_recoverable) / len(non_click_primary) if non_click_primary else 0.0,
            "pass_at_1": bool_mean(non_click_primary, "pass_at_1"),
            "pass_at_5": bool_mean(non_click_primary, "pass_at_5"),
            "pass_at_10": bool_mean(non_click_primary, "pass_at_10"),
            "pass_at_20": bool_mean(non_click_primary, "pass_at_20"),
            "pass_at_50": bool_mean(non_click_primary, "pass_at_50"),
            "rank_bins_all": dict(rank_bins_non_click),
            "rank_bins_recoverable": dict(rank_bins_non_click_recoverable),
            "success_count_recoverable": number_summary(success_counts_non_click_recoverable),
        },
        "gate": gate,
        "gate_reason": gate_reason,
    }

    write_jsonl(output_dir / "per_step.jsonl", non_analyzable_records)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    cat_order = ["click", "type", "swipe", "special-key", "other"]
    cause_order = ["greedy_no_point", "no_correct_control", "no_controls", "other_non_analyzable"]
    rank_order = ["1", "2-5", "6-10", "11-20", "21-50", "missing@50"]

    lines: List[str] = []
    lines.append("# Critical-Step Scope Diagnostic")
    lines.append("")
    lines.append("Diagnostic only: existing UIA sampled pool + frozen matcher + existing UIA true-control identity output. No training was performed.")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- samples: `{args.samples}`")
    lines.append(f"- UIA per-step identity: `{args.uia_per_step}`")
    lines.append(f"- population: `{args.population}`")
    lines.append(f"- primary temperature: `{args.temperature:.1f}`")
    lines.append(f"- recoverable critical steps analyzed: `{total_recoverable}`")
    lines.append(f"- non-analyzable recoverable steps: `{total_non_analyzable}`")
    lines.append("")
    lines.append("## Metric 1: Non-analyzable Action-Type Breakdown")
    lines.append("")
    lines.append("### By Cause and GT Action Category")
    lines.append("")
    lines.append("| cause | total | click | type | swipe | special-key | other | non-click share |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for cause in cause_order:
        total = cause_counter[cause]
        if not total:
            continue
        row = cause_gt[cause]
        non_click = total - row.get("click", 0)
        values = [row.get(cat, 0) for cat in cat_order]
        lines.append(f"| {cause} | {total} | " + " | ".join(str(v) for v in values) + f" | {pct(non_click, total)} |")
    lines.append("")
    lines.append("### By Cause and Greedy Action Category")
    lines.append("")
    lines.append("| cause | total | click | type | swipe | special-key | other | no-point-producing share |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for cause in cause_order:
        total = cause_counter[cause]
        if not total:
            continue
        row = cause_greedy[cause]
        no_point_like = row.get("type", 0) + row.get("swipe", 0) + row.get("special-key", 0) + row.get("other", 0)
        values = [row.get(cat, 0) for cat in cat_order]
        lines.append(f"| {cause} | {total} | " + " | ".join(str(v) for v in values) + f" | {pct(no_point_like, total)} |")
    lines.append("")
    lines.extend(markdown_crosstab("Metric 1: GT Action Category x Greedy Action Category", gt_greedy_matrix, cat_order, cat_order))

    lines.append("## Metric 2: Failure Kind of Non-click Non-analyzable Steps")
    lines.append("")
    lines.append(f"Non-click non-analyzable rows: `{len(non_click_non_analyzable)} / {total_non_analyzable}` ({pct(len(non_click_non_analyzable), total_non_analyzable)} of non-analyzable; {pct(len(non_click_non_analyzable), total_recoverable)} of all recoverable critical steps).")
    lines.append("")
    lines.append("| failure kind | count | share of non-click non-analyzable |")
    lines.append("|---|---:|---:|")
    for kind, count in non_click_failure_kind.most_common():
        lines.append(f"| {kind} | {count} | {pct(count, len(non_click_non_analyzable))} |")
    lines.append("")
    lines.append("### Failure Kind by GT Category")
    lines.append("")
    failure_kinds = sorted(non_click_failure_kind)
    header = "| GT category | total | " + " | ".join(failure_kinds) + " |"
    sep = "|---|---:" + "|---:" * len(failure_kinds) + "|"
    lines.extend([header, sep])
    for gt_cat in ["type", "swipe", "special-key", "other"]:
        row = non_click_failure_by_gt[gt_cat]
        total = sum(row.values())
        if not total:
            continue
        lines.append(f"| {gt_cat} | {total} | " + " | ".join(str(row.get(kind, 0)) for kind in failure_kinds) + " |")
    lines.append("")

    lines.append("## Metric 3: True Scope Split over All Recoverable Critical Steps")
    lines.append("")
    lines.append("| scope bucket | count | fraction of 488 | read |")
    lines.append("|---|---:|---:|---|")
    scope_reads = {
        "click_element_selection": "verifier-addressable click element-selection",
        "click_coordinate": "same-control coordinate component",
        "non_click_distinct_class": "type/swipe/content class, outside click-control verifier scope",
        "click_non_analyzable_gap": "GT-click but not spatially analyzable; parse/control coverage gap to inspect",
        "click_analyzable_other": "click analyzable but neither majority bucket",
    }
    for bucket in PRIMARY_SCOPE_ORDER:
        count = scope_counter[bucket]
        if count:
            lines.append(f"| {bucket} | {count} | {pct(count, total_recoverable)} | {scope_reads[bucket]} |")
    lines.append("")
    verifier_count = scope_counter["click_element_selection"]
    non_click_count = scope_counter["non_click_distinct_class"]
    coordinate_count = scope_counter["click_coordinate"]
    click_gap_count = scope_counter["click_non_analyzable_gap"]
    lines.append(f"Precise scope sentence: the current UIA generative-verifier method directly addresses `{verifier_count} / {total_recoverable}` recoverable critical-step failures ({pct(verifier_count, total_recoverable)}) as click element-selection; `{non_click_count} / {total_recoverable}` ({pct(non_click_count, total_recoverable)}) are non-click type/swipe/content failures requiring a different method; `{coordinate_count} / {total_recoverable}` ({pct(coordinate_count, total_recoverable)}) are click coordinate/same-control cases; `{click_gap_count} / {total_recoverable}` ({pct(click_gap_count, total_recoverable)}) are GT-click non-analyzable coverage gaps that should not be counted as non-click scope.")
    lines.append("")
    lines.extend(markdown_count_table("Metric 3: Non-analyzable Scope Flags", counter_table(non_analyzable_scope_flags), total_non_analyzable, "flag"))

    lines.append("## Metric 4: Non-click Recoverability and Depth")
    lines.append("")
    lines.append("### All Critical Non-click Failures")
    lines.append("")
    lines.append("| rows | recoverable@50 | pass@1 | pass@5 | pass@10 | pass@20 | pass@50 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(f"| {len(non_click_primary)} | {len(non_click_recoverable)} ({pct(len(non_click_recoverable), len(non_click_primary))}) | {pct(sum(1 for row in non_click_primary if row.get('pass_at_1')), len(non_click_primary))} | {pct(sum(1 for row in non_click_primary if row.get('pass_at_5')), len(non_click_primary))} | {pct(sum(1 for row in non_click_primary if row.get('pass_at_10')), len(non_click_primary))} | {pct(sum(1 for row in non_click_primary if row.get('pass_at_20')), len(non_click_primary))} | {pct(sum(1 for row in non_click_primary if row.get('pass_at_50')), len(non_click_primary))} |")
    lines.append("")
    lines.append("### First-correct Rank Bins for Non-click Critical Failures")
    lines.append("")
    lines.append("| first-correct rank bin | all non-click failures | recoverable non-click failures |")
    lines.append("|---|---:|---:|")
    for item in rank_order:
        lines.append(f"| {item} | {rank_bins_non_click.get(item, 0)} | {rank_bins_non_click_recoverable.get(item, 0)} |")
    lines.append("")
    success_summary = summary["non_click_primary"]["success_count_recoverable"]
    mean_success = success_summary["mean"]
    median_success = success_summary["median"]
    lines.append(f"Among recoverable non-click rows, success-count mean/median over 50 samples is `{mean_success:.2f}` / `{median_success:.2f}`.")
    lines.append("")

    lines.append("## Gate")
    lines.append("")
    lines.append(f"**{gate}**")
    lines.append("")
    lines.append(gate_reason)
    lines.append("")
    lines.append("Read: classify, do not force spatial control identity onto point-less actions. Non-click rows are a real distinct class, but GT-click non-analyzable rows are a separate fixable coverage/parse/control-assignment issue and should be reported separately from the method boundary.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{output_dir / 'scope.md'}`")
    lines.append(f"- `{output_dir / 'summary.json'}`")
    lines.append(f"- `{output_dir / 'per_step.jsonl'}`")
    lines.append("")
    lines.append("STOP for review.")
    (output_dir / "scope.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "report": str(output_dir / "scope.md"),
        "summary": str(output_dir / "summary.json"),
        "per_step": str(output_dir / "per_step.jsonl"),
        "gate": gate,
        "scope_counts": dict(scope_counter),
        "non_analyzable": total_non_analyzable,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()