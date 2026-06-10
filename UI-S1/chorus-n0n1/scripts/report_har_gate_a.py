#!/usr/bin/env python3
"""Build a CHORUS Gate A report from HAR GUI-Odyssey episode JSONL rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_har_gui_odyssey import PAPER_GUI_ODYSSEY_OVERALL, summarize  # noqa: E402


def main() -> int:
    args = parse_args()
    results_path = resolve_workspace_path(args.results_jsonl)
    rows = load_jsonl(results_path)
    summary = summarize(rows)
    output_path = resolve_workspace_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_report(args, results_path, rows, summary), encoding="utf-8")

    json_path = output_path.with_suffix(".json")
    payload = {
        "status": gate_status(summary, args.expected_episodes),
        "results_jsonl": str(results_path),
        "expected_episodes": args.expected_episodes,
        "summary": summary,
        "sota_proxy": True,
        "hiconagent_status": args.hiconagent_status,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote report: {output_path}")
    print(f"Wrote machine summary: {json_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Gate A HAR proxy report from resumable JSONL results")
    parser.add_argument("--results_jsonl", required=True, help="Episode-level HAR GUI-Odyssey JSONL results")
    parser.add_argument("--output", default="chorus-n0n1/REPORT_GATE_A_HAR_PROXY.md")
    parser.add_argument("--expected_episodes", type=int, default=1666)
    parser.add_argument("--model_name", default="HAR-GUI-3B-GUI-Odyssey")
    parser.add_argument("--test_data", default="datasets/GUI-Odyssey/gui_odyssey_random_split_test.jsonl")
    parser.add_argument("--hiconagent_status", default="checkpoint_unavailable")
    parser.add_argument("--qualitative_examples", type=int, default=10)
    return parser.parse_args()


def resolve_workspace_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return WORKSPACE_ROOT / path


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def gate_status(summary: Dict[str, Any], expected_episodes: int) -> str:
    if summary.get("episodes", 0) < expected_episodes:
        return "IN_PROGRESS"
    truncation = summary.get("truncation", {})
    if float(truncation.get("truncated_generation_percent", 0.0)) > 1.0:
        return "INVALID_TRUNCATION"
    return "READY_FOR_GATE_REVIEW"


def build_report(
    args: argparse.Namespace,
    results_path: Path,
    rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> str:
    status = gate_status(summary, args.expected_episodes)
    truncation = summary.get("truncation", {})
    lines: List[str] = []
    lines.append("# GATE A Report - HAR GUI-Odyssey Proxy")
    lines.append("")
    lines.append(f"**Status:** {status}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Benchmark: GUI-Odyssey random split test (`{args.test_data}`)")
    lines.append(f"- Model: `{args.model_name}`")
    lines.append("- Evaluation setting: HAR native prompts, Act2Sum history, all steps, no first-error stop")
    lines.append("- SOTA status: `sota_proxy=true` because HiconAgent checkpoint is unavailable locally")
    lines.append(f"- HiconAgent status: `{args.hiconagent_status}`")
    lines.append(f"- Resumable results: `{workspace_relative(results_path)}`")
    lines.append("")
    lines.append("## Progress")
    lines.append("")
    lines.append(f"- Episodes completed: {summary.get('episodes', 0)} / {args.expected_episodes}")
    lines.append(f"- Steps evaluated: {summary.get('steps_evaluated', 0)} / {summary.get('total_steps', 0)}")
    lines.append(f"- Correct steps: {summary.get('correct_steps', 0)}")
    lines.append(f"- Step SR: {summary.get('step_sr_percent', 0.0):.2f}%")
    lines.append(f"- Task SR: {summary.get('tsr_percent', 0.0):.2f}%")
    lines.append("")
    lines.append("## Paper Comparison")
    lines.append("")
    lines.append(f"- Paper GUI-Odyssey Overall SSR: {PAPER_GUI_ODYSSEY_OVERALL:.2f}%")
    lines.append(f"- Current Overall SSR: {summary.get('step_sr_percent', 0.0):.2f}%")
    lines.append(f"- Delta: {summary.get('delta_vs_paper_overall_points', 0.0):+.2f} points")
    lines.append("")
    lines.append("## Category Breakdown")
    lines.append("")
    lines.append("| Category | Paper column | Episodes | Steps | Step SR | Task SR |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for category, values in sorted(summary.get("by_category", {}).items()):
        lines.append(
            f"| {category} | {values.get('paper_column', category)} | {values.get('episodes', 0)} | "
            f"{values.get('steps', 0)} | {values.get('step_sr_percent', 0.0):.2f}% | "
            f"{values.get('tsr_percent', 0.0):.2f}% |"
        )
    if not summary.get("by_category"):
        lines.append("| n/a | n/a | 0 | 0 | 0.00% | 0.00% |")
    lines.append("")
    lines.append("## Truncation Summary")
    lines.append("")
    lines.append(f"- Generations: {truncation.get('generations', 0)}")
    lines.append(f"- Truncated generations: {truncation.get('truncated_generations', 0)}")
    lines.append(f"- Truncated generation rate: {truncation.get('truncated_generation_percent', 0.0):.2f}%")
    lines.append(f"- Action generations: {truncation.get('action_generations', 0)}")
    lines.append(f"- Truncated action generations: {truncation.get('truncated_action_generations', 0)}")
    lines.append(f"- Truncated action rate: {truncation.get('truncated_action_percent', 0.0):.2f}%")
    lines.append("- Gate rule: >1% truncated generations invalidates the run")
    lines.append("")
    lines.append("## Qualitative Step Records")
    lines.append("")
    examples = qualitative_examples(rows, args.qualitative_examples)
    if examples:
        for example in examples:
            lines.append("```json")
            lines.append(json.dumps(example, ensure_ascii=False, indent=2))
            lines.append("```")
    else:
        lines.append("No episode rows have been written yet.")
    lines.append("")
    lines.append("## Gate Decision")
    lines.append("")
    if status == "IN_PROGRESS":
        lines.append("STOP. Full Phase A reproduction is still running; do not proceed to N0/N1 from this partial proxy result.")
    elif status == "INVALID_TRUNCATION":
        lines.append("STOP. The run violates the truncation gate and must be rerun or diagnosed before N0/N1.")
    else:
        lines.append("STOP for human review. Proceed to N0 only after accepting this Phase A proxy report and the HiconAgent limitation.")
    lines.append("")
    return "\n".join(lines)


def qualitative_examples(rows: Iterable[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    sorted_rows = sorted(rows, key=lambda row: (row.get("task_success", False), row.get("episode_id", "")))
    for row in sorted_rows:
        if len(selected) >= limit:
            break
        selected.append(compact_episode(row))
    return selected


def compact_episode(row: Dict[str, Any]) -> Dict[str, Any]:
    compact_steps = []
    for step in row.get("steps", [])[:3]:
        compact_steps.append(
            {
                "step_idx": step.get("step_idx"),
                "extract_match": step.get("extract_match"),
                "type_match": step.get("type_match"),
                "gt_action": step.get("gt_action"),
                "pred_action": step.get("pred_action"),
                "answer": step.get("answer"),
                "finish_reason": step.get("finish_reason"),
                "truncated": step.get("truncated"),
                "error": step.get("error"),
            }
        )
    return {
        "episode_id": row.get("episode_id"),
        "category": row.get("category"),
        "goal": row.get("goal"),
        "num_steps": row.get("num_steps"),
        "steps_evaluated": row.get("steps_evaluated"),
        "correct_steps": row.get("correct_steps"),
        "task_success": row.get("task_success"),
        "first_error_step": row.get("first_error_step"),
        "steps": compact_steps,
    }


def workspace_relative(path: Path) -> str:
    try:
        return str(path.relative_to(WORKSPACE_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())