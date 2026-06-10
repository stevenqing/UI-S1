#!/usr/bin/env python3
"""Prepare offline N0/N1 inputs from HAR GUI-Odyssey episode JSONL rows."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bench.har_odyssey_results import flatten_episode_rows, load_episode_rows, summarize_step_rows, write_jsonl  # noqa: E402
from src.metrics.prevalence_check import build_prevalence_manifest  # noqa: E402
from src.probes.headroom import build_headroom_manifest, build_headroom_probe_queue  # noqa: E402
from src.readers.disagreement import build_reader_queue  # noqa: E402


def main() -> int:
    args = parse_args()
    results_path = resolve_workspace_path(args.results_jsonl)
    output_dir = resolve_workspace_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_rows = load_episode_rows(results_path)
    step_rows = flatten_episode_rows(episode_rows)

    step_rows_path = output_dir / "har_gui_odyssey_steps.jsonl"
    n0_queue_path = output_dir / "n0_headroom_queue.jsonl"
    n1_queue_path = output_dir / "n1_reader_inputs_queue.jsonl"
    manifest_path = output_dir / "manifest.json"
    report_path = output_dir / "README.md"

    write_jsonl(step_rows_path, step_rows)
    n0_queue = build_headroom_probe_queue(
        step_rows,
        include_correct=args.include_correct_in_n0,
        limit=args.queue_limit,
    )
    n1_queue = build_reader_queue(step_rows, limit=args.queue_limit)
    write_jsonl(n0_queue_path, n0_queue)
    write_jsonl(n1_queue_path, n1_queue)

    summary = summarize_step_rows(step_rows, episodes_completed=len(episode_rows))
    status = prep_status(summary, len(episode_rows), args.half_episodes, args.expected_episodes)
    n0_manifest = build_headroom_manifest(
        step_rows,
        queue_path=workspace_relative(n0_queue_path),
        include_correct=args.include_correct_in_n0,
        limit=args.queue_limit,
    )
    n1_manifest = build_prevalence_manifest(
        step_rows,
        queue_path=workspace_relative(n1_queue_path),
        queue_items=len(n1_queue),
        limit=args.queue_limit,
    )
    manifest: Dict[str, Any] = {
        "status": status,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results_jsonl": workspace_relative(results_path),
        "output_dir": workspace_relative(output_dir),
        "expected_episodes": args.expected_episodes,
        "half_episodes": args.half_episodes,
        "episodes_completed": len(episode_rows),
        "episodes_remaining_to_half": max(0, args.half_episodes - len(episode_rows)),
        "episodes_remaining_to_full": max(0, args.expected_episodes - len(episode_rows)),
        "sota_proxy": True,
        "hiconagent_status": args.hiconagent_status,
        "phase_gate": {
            "gate_a_required": True,
            "current_phase": "Phase A HAR proxy running" if len(episode_rows) < args.expected_episodes else "Phase A HAR proxy complete",
            "may_start_offline_n0n1_at_half": status in {"READY_FOR_N0N1_OFFLINE_PREP_HALF", "READY_FOR_N0N1_OFFLINE_PREP_FULL"},
            "may_start_model_probe_calls": False,
            "model_probe_note": "Keep model-based N0/N1 probes stopped until Gate A half/full review and GPU resource decision.",
        },
        "files": {
            "step_rows": workspace_relative(step_rows_path),
            "n0_headroom_queue": workspace_relative(n0_queue_path),
            "n1_reader_inputs_queue": workspace_relative(n1_queue_path),
            "manifest": workspace_relative(manifest_path),
            "report": workspace_relative(report_path),
        },
        "summary": summary,
        "n0": n0_manifest,
        "n1_prevalence": n1_manifest,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(build_report(manifest), encoding="utf-8")
    print(json.dumps({"status": status, "episodes_completed": len(episode_rows), "output_dir": workspace_relative(output_dir)}, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare offline N0/N1 inputs from HAR GUI-Odyssey JSONL results")
    parser.add_argument(
        "--results_jsonl",
        default="related_work/har/outputs/gui_odyssey_paper/full_har_gui_odyssey_20260610.jsonl",
        help="Episode-level HAR GUI-Odyssey JSONL results",
    )
    parser.add_argument("--output_dir", default="chorus-n0n1/runs/n0n1_inputs/har_gui_odyssey_latest")
    parser.add_argument("--expected_episodes", type=int, default=1666)
    parser.add_argument("--half_episodes", type=int, default=833)
    parser.add_argument("--queue_limit", type=int, default=None)
    parser.add_argument("--include_correct_in_n0", action="store_true")
    parser.add_argument("--hiconagent_status", default="checkpoint_unavailable")
    return parser.parse_args()


def resolve_workspace_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return WORKSPACE_ROOT / path


def workspace_relative(path: Path) -> str:
    try:
        return str(path.relative_to(WORKSPACE_ROOT))
    except ValueError:
        return str(path)


def prep_status(summary: Dict[str, Any], episodes: int, half_episodes: int, expected_episodes: int) -> str:
    if episodes == 0:
        return "NO_RESULTS"
    truncation = summary.get("truncation", {})
    if float(truncation.get("truncated_generation_percent", 0.0)) > 1.0:
        return "INVALID_TRUNCATION"
    if episodes < half_episodes:
        return "WAIT_FOR_HALF"
    if episodes < expected_episodes:
        return "READY_FOR_N0N1_OFFLINE_PREP_HALF"
    return "READY_FOR_N0N1_OFFLINE_PREP_FULL"


def build_report(manifest: Dict[str, Any]) -> str:
    summary = manifest.get("summary", {})
    truncation = summary.get("truncation", {})
    n1 = manifest.get("n1_prevalence", {})
    lines = [
        "# N0/N1 Offline Inputs - HAR GUI-Odyssey",
        "",
        f"**Status:** {manifest.get('status')}",
        "",
        "## Progress",
        "",
        f"- Episodes completed: {manifest.get('episodes_completed')} / {manifest.get('expected_episodes')}",
        f"- Episodes remaining to half: {manifest.get('episodes_remaining_to_half')}",
        f"- Steps materialized: {summary.get('steps', 0)}",
        f"- Step SR: {summary.get('step_sr_percent', 0.0):.2f}%",
        f"- Baseline error steps: {summary.get('baseline_error_steps', 0)}",
        f"- Truncated generation rate: {truncation.get('truncated_generation_percent', 0.0):.2f}%",
        "",
        "## Files",
        "",
    ]
    for name, path in manifest.get("files", {}).items():
        lines.append(f"- {name}: `{path}`")
    lines.extend(
        [
            "",
            "## N0 Headroom",
            "",
            f"- Queue items: {manifest.get('n0', {}).get('queue_items', 0)}",
            "- Status: model teacher probes not started",
            "- Rule: all teacher calls must use `src/infer/wrapper.py`",
            "",
            "## N1 Prevalence Check",
            "",
            f"- Queue items: {n1.get('queue_items', 0)}",
            f"- Baseline error steps: {n1.get('baseline_error_steps', 0)}",
            f"- Baseline error rate: {n1.get('baseline_error_percent', 0.0):.2f}%",
            "- AUROC: not computed in prevalence checks",
            "- Status: independent reader calls not started",
            "- Rule: all reader calls must use `src/infer/wrapper.py`",
            "",
            "## Gate Note",
            "",
            "These files are offline preparation artifacts. Model-based N0/N1 probes remain stopped until the half/full Gate A review and a resource decision, because the full HAR baseline is still using GPUs 4-7.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())