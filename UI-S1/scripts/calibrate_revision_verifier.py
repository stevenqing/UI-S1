#!/usr/bin/env python3
"""Calibrate a conservative use-revision rule on dev and lock it for test."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rl_feasibility_sampling import action_key  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["correction_id"]), int(row["step_idx"])


def action_type(action: Any) -> str:
    return str((action or {}).get("action") or "unparsed").lower()


def enrich(
    eval_rows: Sequence[Mapping[str, Any]],
    source: Mapping[tuple[str, int], Mapping[str, Any]],
    student_candidates: Mapping[tuple[str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for evaluated in eval_rows:
        paired = key(evaluated)
        if paired not in source:
            raise ValueError(f"missing source row: {paired}")
        if paired not in student_candidates:
            raise ValueError(f"missing student candidate: {paired}")
        src = source[paired]
        student = student_candidates[paired]
        student_key = str(student.get("student_action_key") or "__unparsed__")
        revision_key = str(src["chosen_action_key"])
        actor_key = action_key(src.get("actor_action"), 25)
        student_correct = bool(evaluated["student_correct"])
        revision_correct = bool(evaluated["revision_correct"])
        rows.append({
            **dict(evaluated),
            "actor": str(src["actor"]),
            "revision_type": action_type(src.get("revision_action")),
            "actor_type": action_type(src.get("actor_action")),
            "student_type": action_type(student.get("student_action")),
            "revision_changed": bool(src["revision_changed_from_actor"]),
            "student_parse_ok": bool(student.get("parse_ok")),
            "revision_student_same": revision_key == student_key,
            "actor_student_same": actor_key == student_key,
            "actor_revision_same": actor_key == revision_key,
            "confidence": float(src.get("correction_confidence") or 0.0),
            "relative_step": int(src["step_idx"]) / max(1, int(src["num_steps"]) - 1),
            "rescue": (not student_correct) and revision_correct,
            "regress": student_correct and (not revision_correct),
        })
    return rows


def stats(rows: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rescue = sum(bool(row["rescue"]) for row in selected)
    regress = sum(bool(row["regress"]) for row in selected)
    accepted = len(selected)
    baseline = sum(bool(row["student_correct"]) for row in rows) / len(rows)
    return {
        "rows": len(rows),
        "accepted": accepted,
        "coverage": accepted / len(rows),
        "rescue": rescue,
        "regress": regress,
        "neutral": accepted - rescue - regress,
        "rescue_precision": rescue / max(1, accepted),
        "regress_rate": regress / max(1, accepted),
        "accepted_net_utility": (rescue - regress) / max(1, accepted),
        "population_net_utility": (rescue - regress) / len(rows),
        "student_baseline_accuracy": baseline,
        "fallback_student_accuracy": baseline + (rescue - regress) / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-input", required=True)
    parser.add_argument("--dev-eval", required=True)
    parser.add_argument("--test-eval", required=True)
    parser.add_argument("--student-candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-dev-accepted", type=int, default=10)
    args = parser.parse_args()

    source_rows = read_jsonl(Path(args.causal_input))
    source = {key(row): row for row in source_rows}
    student_rows = read_jsonl(Path(args.student_candidates))
    student_candidates = {key(row): row for row in student_rows}
    dev = enrich(read_jsonl(Path(args.dev_eval)), source, student_candidates)
    test = enrich(read_jsonl(Path(args.test_eval)), source, student_candidates)

    predicates: dict[str, Callable[[Mapping[str, Any]], bool]] = {
        "student_parsed": lambda row: bool(row["student_parse_ok"]),
        "revision_changed": lambda row: bool(row["revision_changed"]),
        "revision_differs_student": lambda row: not bool(row["revision_student_same"]),
        "actor_differs_student": lambda row: not bool(row["actor_student_same"]),
        "actor_revision_same": lambda row: bool(row["actor_revision_same"]),
        "actor=internvl3": lambda row: row["actor"] == "internvl3",
        "actor=qwen3_vl": lambda row: row["actor"] == "qwen3_vl",
        "confidence>=0.98": lambda row: float(row["confidence"]) >= 0.98,
        "confidence>=1.00": lambda row: float(row["confidence"]) >= 1.0,
        "relative_step<=0.25": lambda row: float(row["relative_step"]) <= 0.25,
        "relative_step<=0.50": lambda row: float(row["relative_step"]) <= 0.50,
    }
    for field in ("revision_type", "student_type", "actor_type"):
        for value in ("click", "type", "swipe", "press", "key"):
            predicates[f"{field}={value}"] = lambda row, field=field, value=value: row[field] == value

    base = lambda row: row.get("predicted_decision") == "use_revision"
    candidates = [("semantic_use_revision", tuple())]
    names = list(predicates)
    candidates.extend((f"semantic_use_revision & {name}", (name,)) for name in names)
    candidates.extend(
        (f"semantic_use_revision & {left} & {right}", (left, right))
        for left, right in itertools.combinations(names, 2)
    )
    results = []
    for label, conditions in candidates:
        selected = [row for row in dev if base(row) and all(predicates[name](row) for name in conditions)]
        item = {"rule": label, "conditions": list(conditions), **stats(dev, selected)}
        results.append(item)
    viable = [
        row for row in results
        if row["accepted"] >= args.min_dev_accepted and row["rescue"] > row["regress"]
    ]
    viable.sort(key=lambda row: (row["population_net_utility"], row["accepted_net_utility"], row["accepted"]), reverse=True)
    selected_rule = viable[0] if viable else None

    if selected_rule:
        conditions = selected_rule["conditions"]
        selected_test = [row for row in test if base(row) and all(predicates[name](row) for name in conditions)]
        test_result = {"rule": selected_rule["rule"], "conditions": conditions, **stats(test, selected_test)}
    else:
        test_result = {"rule": "no_positive_dev_rule", "conditions": [], **stats(test, [])}
    baseline_dev = stats(dev, [row for row in dev if base(row)])
    baseline_test = stats(test, [row for row in test if base(row)])
    summary = {
        "selection_uses_test_labels": False,
        "features_exclude_gt_and_matcher": True,
        "dev_rows": len(dev),
        "test_rows": len(test),
        "min_dev_accepted": args.min_dev_accepted,
        "semantic_use_revision_dev": baseline_dev,
        "semantic_use_revision_test": baseline_test,
        "selected_dev_rule": selected_rule,
        "locked_test_result": test_result,
        "top_dev_rules": viable[:20],
    }
    out_dir = Path(args.output_dir)
    write_json(out_dir / "summary.json", summary)
    lines = [
        "# Conservative Revision-Verifier Calibration",
        "",
        "Rules are selected on episode-disjoint dev labels and evaluated once on test. Inference features exclude GT and matcher outcomes.",
        "",
        f"Selected rule: **{test_result['rule']}**",
        "",
        "| split/rule | accepted | coverage | rescue | regress | population utility | fallback accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in (("dev semantic", baseline_dev), ("test semantic", baseline_test), ("test locked", test_result)):
        lines.append(
            f"| {name} | {row['accepted']} | {100*row['coverage']:.2f}% | {row['rescue']} | {row['regress']} | "
            f"{100*row['population_net_utility']:+.2f}pp | {100*row['fallback_student_accuracy']:.2f}% |"
        )
    lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"selected_rule": test_result["rule"], "test": test_result, "report": str(out_dir / "report.md")}, indent=2))


if __name__ == "__main__":
    main()
