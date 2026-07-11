#!/usr/bin/env python3
"""Analyze revision utility relative to the starting student on matched states."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def paired_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["correction_id"]), int(row["step_idx"])


def classify(reference_correct: bool, revision_correct: bool) -> str:
    if reference_correct and revision_correct:
        return "preserve_correct"
    if reference_correct and not revision_correct:
        return "regress"
    if not reference_correct and revision_correct:
        return "rescue"
    return "unresolved"


def percentile(values: Sequence[float], q: float) -> float:
    idx = min(len(values) - 1, max(0, int(q * len(values))))
    return float(values[idx])


def bootstrap_trajectory_utility(rows: Sequence[Mapping[str, Any]], draws: int, seed: int) -> dict[str, Any]:
    grouped: dict[str, Counter[str]] = {}
    for row in rows:
        counts = grouped.setdefault(str(row["correction_id"]), Counter())
        counts[classify(bool(row["student_correct"]), bool(row["revision_correct"]))] += 1
        counts["steps"] += 1
    units = list(grouped.values())
    rng = random.Random(seed)
    values = []
    for _ in range(draws):
        rescue = regress = steps = 0
        for _ in range(len(units)):
            unit = units[rng.randrange(len(units))]
            rescue += unit["rescue"]
            regress += unit["regress"]
            steps += unit["steps"]
        values.append((rescue - regress) / steps)
    values.sort()
    return {
        "mean": sum(values) / len(values),
        "lo": percentile(values, 0.025),
        "hi": percentile(values, 0.975),
        "draws": draws,
        "sampling_unit": "correction_trajectory",
    }


def utility_summary(rows: Sequence[Mapping[str, Any]], draws: int, seed: int) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    trajectories: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        category = classify(bool(row["student_correct"]), bool(row["revision_correct"]))
        counts[category] += 1
        counts["steps"] += 1
        counts["student_correct"] += int(bool(row["student_correct"]))
        counts["revision_correct"] += int(bool(row["revision_correct"]))
        trajectories.setdefault(str(row["correction_id"]), []).append(row)
    steps = counts["steps"]
    trajectory_student_success = sum(all(bool(row["student_correct"]) for row in group) for group in trajectories.values())
    trajectory_revision_success = sum(all(bool(row["revision_correct"]) for row in group) for group in trajectories.values())
    return {
        "steps": steps,
        "trajectories": len(trajectories),
        "outcomes": {
            name: {"count": counts[name], "fraction": counts[name] / steps}
            for name in ("rescue", "regress", "preserve_correct", "unresolved")
        },
        "student_accuracy": counts["student_correct"] / steps,
        "revision_accuracy": counts["revision_correct"] / steps,
        "net_student_relative_revision_utility": (counts["rescue"] - counts["regress"]) / steps,
        "revision_rescue_rate_given_student_wrong": counts["rescue"] / max(1, steps - counts["student_correct"]),
        "revision_regression_rate_given_student_correct": counts["regress"] / max(1, counts["student_correct"]),
        "student_trajectory_success_rate": trajectory_student_success / max(1, len(trajectories)),
        "revision_trajectory_success_rate": trajectory_revision_success / max(1, len(trajectories)),
        "cluster_bootstrap": bootstrap_trajectory_utility(rows, draws, seed),
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def pp(value: float) -> str:
    return f"{100.0 * value:+.2f}pp"


def render_report(summary: Mapping[str, Any]) -> str:
    overall = summary["student_relative_revision_utility"]
    lines = [
        "# Student-Relative Revision Utility",
        "",
        "For the same state/history grid, classify the global revision relative to the frozen starting student:",
        "",
        "$$u_t^{stu}=M(a_t^{revision})-M(a_t^{student}).$$",
        "",
        "## Overall",
        "",
        "| outcome | count | fraction |",
        "|---|---:|---:|",
    ]
    for name in ("rescue", "regress", "preserve_correct", "unresolved"):
        row = overall["outcomes"][name]
        lines.append(f"| {name} | {row['count']} | {pct(row['fraction'])} |")
    bootstrap = overall["cluster_bootstrap"]
    lines.extend(
        [
            "",
            f"- Starting student accuracy: **{pct(overall['student_accuracy'])}**.",
            f"- Revision accuracy: **{pct(overall['revision_accuracy'])}**.",
            f"- Net student-relative revision utility: **{pp(overall['net_student_relative_revision_utility'])}**.",
            f"- Trajectory-cluster bootstrap: **[{pp(bootstrap['lo'])}, {pp(bootstrap['hi'])}]**.",
            f"- Rescue rate given student-wrong: **{pct(overall['revision_rescue_rate_given_student_wrong'])}**.",
            f"- Regression rate given student-correct: **{pct(overall['revision_regression_rate_given_student_correct'])}**.",
            "",
            "## By Source Actor",
            "",
            "| source | steps | student acc | revision acc | net utility | rescue / regress |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for source, row in summary["by_actor"].items():
        lines.append(
            f"| {source} | {row['steps']} | {pct(row['student_accuracy'])} | "
            f"{pct(row['revision_accuracy'])} | {pp(row['net_student_relative_revision_utility'])} | "
            f"{row['outcomes']['rescue']['count']} / {row['outcomes']['regress']['count']} |"
        )
    lines.extend(
        [
            "",
            "## By Prefix",
            "",
            "| prefix | steps | student acc | revision acc | net utility |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for prefix, row in summary["by_prefix"].items():
        lines.append(
            f"| {prefix} | {row['steps']} | {pct(row['student_accuracy'])} | "
            f"{pct(row['revision_accuracy'])} | {pp(row['net_student_relative_revision_utility'])} |"
        )
    if summary.get("history_intervention"):
        history = summary["history_intervention"]
        lines.extend(
            [
                "",
                "## Frozen-Student History Intervention",
                "",
                f"- Revision-history accuracy: **{pct(history['revision_history_accuracy'])}**.",
                f"- GT-history accuracy: **{pct(history['gt_history_accuracy'])}**.",
                f"- GT-history delta: **{pp(history['gt_minus_revision_history_delta'])}**.",
                f"- Wrong→right flips: **{history['wrong_to_right']}**.",
                f"- Right→wrong flips: **{history['right_to_wrong']}**.",
                f"- Student action disagreement: **{pct(history['action_disagreement'])}**.",
            ]
        )
        if history.get("population_standardization"):
            standardized = history["population_standardization"]
            lines.append(
                f"- Actor×prefix population-standardized GT-history delta: **{pp(standardized['gt_minus_revision_history_delta'])}** over a {standardized['population_rows']}-row target population."
            )
    lines.extend(
        [
            "",
            "## Three-Way Correctness Patterns",
            "",
            "| actor / revision / student | count | fraction |",
            "|---|---:|---:|",
        ]
    )
    for pattern, item in sorted(summary["three_way_patterns"].items()):
        lines.append(f"| {pattern} | {item['count']} | {pct(item['fraction'])} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-history-input", required=True)
    parser.add_argument("--revision-history-eval", required=True)
    parser.add_argument("--gt-history-eval")
    parser.add_argument("--population-input")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    source_rows = read_jsonl(Path(args.revision_history_input))
    if args.max_rows > 0:
        source_rows = source_rows[: args.max_rows]
    source = {paired_key(row): row for row in source_rows}
    evaluated_rows = read_jsonl(Path(args.revision_history_eval))
    evaluated_all = {paired_key(row): row for row in evaluated_rows}
    if len(source) != len(source_rows) or len(evaluated_all) != len(evaluated_rows):
        raise ValueError("duplicate paired key")
    missing_evaluated = set(source) - set(evaluated_all)
    if missing_evaluated:
        raise ValueError(f"source/evaluation grid missing {list(missing_evaluated)[:10]}")
    evaluated = {key: evaluated_all[key] for key in source}

    joined: list[dict[str, Any]] = []
    pattern_counts: Counter[str] = Counter()
    for key in source:
        src = source[key]
        row = dict(evaluated[key])
        row["revision_correct"] = bool(src["revision_correct"])
        row["actor_correct"] = bool(src["actor_correct"])
        row["prefix_clean"] = bool(src["prefix_clean"])
        joined.append(row)
        pattern = "/".join("C" if bool(value) else "W" for value in (src["actor_correct"], src["revision_correct"], row["student_correct"]))
        pattern_counts[pattern] += 1

    summary: dict[str, Any] = {
        "definition": "matcher(revision) - matcher(starting_student)",
        "revision_history_input": args.revision_history_input,
        "revision_history_eval": args.revision_history_eval,
        "student_relative_revision_utility": utility_summary(joined, args.bootstrap_draws, args.seed),
        "by_actor": {
            actor: utility_summary([row for row in joined if str(row["actor"]) == actor], args.bootstrap_draws, args.seed)
            for actor in sorted({str(row["actor"]) for row in joined})
        },
        "by_prefix": {
            name: utility_summary([row for row in joined if bool(row["prefix_clean"]) is clean], args.bootstrap_draws, args.seed)
            for name, clean in (("clean", True), ("dirty", False))
        },
        "three_way_patterns": {
            pattern: {"count": count, "fraction": count / len(joined)}
            for pattern, count in sorted(pattern_counts.items())
        },
    }

    if args.gt_history_eval:
        gt_rows = read_jsonl(Path(args.gt_history_eval))
        gt_all = {paired_key(row): row for row in gt_rows}
        if len(gt_all) != len(gt_rows):
            raise ValueError("duplicate GT-history paired key")
        missing_gt = set(source) - set(gt_all)
        if missing_gt:
            raise ValueError(f"GT-history grid missing {list(missing_gt)[:10]}")
        gt = {key: gt_all[key] for key in source}
        revision_correct = [bool(evaluated[key]["student_correct"]) for key in source]
        gt_correct = [bool(gt[key]["student_correct"]) for key in source]
        history_intervention: dict[str, Any] = {
            "rows": len(source),
            "revision_history_accuracy": sum(revision_correct) / len(source),
            "gt_history_accuracy": sum(gt_correct) / len(source),
            "gt_minus_revision_history_delta": (sum(gt_correct) - sum(revision_correct)) / len(source),
            "wrong_to_right": sum(not before and after for before, after in zip(revision_correct, gt_correct)),
            "right_to_wrong": sum(before and not after for before, after in zip(revision_correct, gt_correct)),
            "action_disagreement": sum(
                str(evaluated[key].get("student_action_key")) != str(gt[key].get("student_action_key"))
                for key in source
            ) / len(source),
        }
        stratum_effects: dict[str, Any] = {}
        for actor in sorted({str(row["actor"]) for row in source.values()}):
            for clean in (True, False):
                keys = [
                    key for key, row in source.items()
                    if str(row["actor"]) == actor and bool(row["prefix_clean"]) is clean
                ]
                if not keys:
                    continue
                before = [int(bool(evaluated[key]["student_correct"])) for key in keys]
                after = [int(bool(gt[key]["student_correct"])) for key in keys]
                name = f"{actor}:{'clean' if clean else 'dirty'}"
                stratum_effects[name] = {
                    "rows": len(keys),
                    "revision_history_accuracy": sum(before) / len(keys),
                    "gt_history_accuracy": sum(after) / len(keys),
                    "delta": (sum(after) - sum(before)) / len(keys),
                    "wrong_to_right": sum(not b and a for b, a in zip(before, after)),
                    "right_to_wrong": sum(b and not a for b, a in zip(before, after)),
                }
        history_intervention["by_actor_prefix"] = stratum_effects
        if args.population_input:
            population_rows = read_jsonl(Path(args.population_input))
            population_counts = Counter(
                f"{row['actor']}:{'clean' if bool(row['prefix_clean']) else 'dirty'}"
                for row in population_rows
            )
            if set(population_counts) - set(stratum_effects):
                raise ValueError("sample lacks a population actor/prefix stratum")
            population_total = sum(population_counts.values())
            history_intervention["population_standardization"] = {
                "population_rows": population_total,
                "stratum_weights": {
                    name: count / population_total for name, count in sorted(population_counts.items())
                },
                "gt_minus_revision_history_delta": sum(
                    population_counts[name] / population_total * stratum_effects[name]["delta"]
                    for name in population_counts
                ),
                "note": "standardized over the full actor x diagnostic-prefix distribution",
            }
        summary["history_intervention"] = history_intervention

    output_dir = Path(args.output_dir)
    write_json(output_dir / "student_relative_revision_summary.json", summary)
    (output_dir / "student_relative_revision_report.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps({
        "steps": len(joined),
        "student_accuracy": summary["student_relative_revision_utility"]["student_accuracy"],
        "revision_accuracy": summary["student_relative_revision_utility"]["revision_accuracy"],
        "net_student_relative_utility": summary["student_relative_revision_utility"]["net_student_relative_revision_utility"],
        "history_intervention": summary.get("history_intervention"),
        "output_dir": str(output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
