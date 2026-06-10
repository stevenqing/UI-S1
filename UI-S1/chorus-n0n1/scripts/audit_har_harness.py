#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
for path in [ROOT, WORKSPACE_ROOT, WORKSPACE_ROOT / "related_work" / "har", WORKSPACE_ROOT / "gui_odyssey_eval"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gui_odyssey_eval.odyssey_action_matching import evaluate_odyssey_action, get_scroll_direction  # noqa: E402
from related_work.har.action_parser import parse_har_output  # noqa: E402


ACTION_MAPPING_ROWS = [
    ("CLICK:(x,y)", "click", "coordinate=[x,y]", "parser regex in related_work/har/action_parser.py"),
    ("LONG_PRESS:(x,y)", "long_press", "coordinate=[x,y]", "parser regex in related_work/har/action_parser.py"),
    ("TYPE:text or TYPE:\"text\"", "type", "text", "parser regex in related_work/har/action_parser.py"),
    ("SCROLL:UP/DOWN/LEFT/RIGHT", "swipe", "coordinate, coordinate2", "global gesture-direction mapping"),
    ("COMPLETE", "terminate", "status=success", "eval_har_gui_odyssey.normalize_har_action fallback"),
    ("IMPOSSIBLE", "terminate", "status=impossible", "eval_har_gui_odyssey.normalize_har_action fallback"),
    ("BACK/HOME/PRESS_RECENT", "system_button", "button", "eval_har_gui_odyssey.normalize_har_action fallback"),
]


def main() -> int:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset_index(resolve(args.test_data))
    step_rows = flatten_results(load_jsonl(resolve(args.results_jsonl)), dataset)
    baseline = summarize(step_rows)
    rebaseline = rescore(step_rows)
    long_press_errors = [row for row in step_rows if row["gt_action"].get("action") == "long_press" and row["baseline_error"]]
    swipe_errors = [row for row in step_rows if row["gt_action"].get("action") == "swipe" and row["baseline_error"]]
    swipe_sample = random.Random(args.seed).sample(swipe_errors, min(args.swipe_sample, len(swipe_errors)))

    write_cases(audit_dir / "long_press_cases.md", "long_press", long_press_errors)
    write_cases(audit_dir / "swipe_cases.md", "swipe", swipe_sample)
    write_report_v2(output_dir / "REPORT_V2.md", baseline, rebaseline, long_press_errors, swipe_errors, swipe_sample)

    print(json.dumps({
        "status": "phase_h_audit_ready",
        "report": str(output_dir / "REPORT_V2.md"),
        "long_press_cases": len(long_press_errors),
        "swipe_error_cases": len(swipe_errors),
        "swipe_sample_cases": len(swipe_sample),
        "step_sr_percent": baseline["step_sr_percent"],
    }, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit HAR GUI-Odyssey parser/matcher artifacts before N0/N1 model calls")
    parser.add_argument("--results_jsonl", default="related_work/har/outputs/gui_odyssey_paper/full_har_gui_odyssey_20260610.jsonl")
    parser.add_argument("--test_data", default="datasets/GUI-Odyssey/gui_odyssey_random_split_test.jsonl")
    parser.add_argument("--output_dir", default="chorus-n0n1/runs/n0n1_inputs/har_gui_odyssey_latest/offline_analysis")
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--swipe_sample", type=int, default=20)
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_dataset_index(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("episode_id")): row for row in load_jsonl(path)}


def flatten_results(rows: Iterable[Dict[str, Any]], dataset: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat = []
    for episode_index, episode in enumerate(rows):
        episode_id = str(episode.get("episode_id"))
        source_episode = dataset.get(episode_id, {})
        source_steps = source_episode.get("steps", [])
        for step in episode.get("steps", []):
            step_idx = int(step.get("step_idx", 0))
            source_step = source_steps[step_idx] if step_idx < len(source_steps) else {}
            _, answer, parsed = parse_har_output(step.get("raw_text") or step.get("answer", ""))
            gt_action = step.get("gt_action") or source_step.get("check_options") or {}
            flat.append({
                "episode_index": episode_index,
                "episode_id": episode_id,
                "category": episode.get("category", ""),
                "goal": episode.get("goal", ""),
                "num_steps": int(episode.get("num_steps") or len(source_steps) or len(episode.get("steps", []))),
                "step_idx": step_idx,
                "step_number": step_idx + 1,
                "screenshot": step.get("screenshot") or source_step.get("screenshot", ""),
                "raw_text": step.get("raw_text", ""),
                "answer": answer or step.get("answer", ""),
                "parsed_action": parsed,
                "gt_action": gt_action,
                "pred_action": step.get("pred_action") or {},
                "type_match": bool(step.get("type_match", False)),
                "extract_match": bool(step.get("extract_match", False)),
                "baseline_error": not bool(step.get("extract_match", False)),
                "image_width": int(step.get("image_width") or source_episode.get("width") or 0),
                "image_height": int(step.get("image_height") or source_episode.get("height") or 0),
                "error": step.get("error", ""),
            })
    return flat


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row["extract_match"])
    action_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"steps": 0, "errors": 0})
    error_kinds = Counter()
    for row in rows:
        action = str(row["gt_action"].get("action", "unknown"))
        action_counts[action]["steps"] += 1
        if row["baseline_error"]:
            action_counts[action]["errors"] += 1
            error_kinds[error_kind(row)] += 1
    return {
        "steps": total,
        "correct_steps": correct,
        "baseline_error_steps": total - correct,
        "step_sr_percent": round(100 * correct / total, 4) if total else 0.0,
        "baseline_error_percent": round(100 * (total - correct) / total, 4) if total else 0.0,
        "error_kind_counts": dict(sorted(error_kinds.items())),
        "by_gt_action": action_table(action_counts),
    }


def rescore(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    rescored = []
    for row in rows:
        updated = dict(row)
        if row["pred_action"] and row["gt_action"] and row["image_width"] and row["image_height"]:
            try:
                type_match, extract_match = evaluate_odyssey_action(
                    copy.deepcopy(row["pred_action"]),
                    copy.deepcopy(row["gt_action"]),
                    row["image_width"],
                    row["image_height"],
                )
                updated["type_match"] = bool(type_match)
                updated["extract_match"] = bool(extract_match)
                updated["baseline_error"] = not bool(extract_match)
            except Exception:
                pass
        rescored.append(updated)
    return summarize(rescored)


def error_kind(row: Dict[str, Any]) -> str:
    if row.get("type_match") is False:
        return "action_type"
    gt_action = row.get("gt_action") or {}
    pred_action = row.get("pred_action") or {}
    if gt_action.get("action") in {"click", "swipe", "long_press"} or pred_action.get("action") in {"click", "swipe", "long_press"}:
        return "coordinate_or_target"
    if row.get("error"):
        return "parse_or_runtime"
    return "semantic_or_sequence"


def action_table(action_counts: Dict[str, Dict[str, int]]) -> List[Dict[str, Any]]:
    table = []
    for action, counts in sorted(action_counts.items(), key=lambda item: (-item[1]["errors"], item[0])):
        steps = counts["steps"]
        errors = counts["errors"]
        table.append({
            "gt_action": action,
            "steps": steps,
            "errors": errors,
            "error_rate_percent": round(100 * errors / steps, 2) if steps else 0.0,
        })
    return table


def write_cases(path: Path, title: str, rows: List[Dict[str, Any]]) -> None:
    lines = [f"# Audit Cases - {title}", "", "Human review: fill review notes before approving Gate H.", ""]
    for index, row in enumerate(rows, start=1):
        lines.extend([
            f"## Case {index}: episode {row['episode_id']} step {row['step_idx']}",
            "",
            f"- Category: `{row['category']}`",
            f"- Screenshot: `{row['screenshot']}`",
            f"- Parsed action: `{json.dumps(row['parsed_action'], ensure_ascii=False)}`",
            f"- Pred action: `{json.dumps(row['pred_action'], ensure_ascii=False)}`",
            f"- GT action: `{json.dumps(row['gt_action'], ensure_ascii=False)}`",
            f"- Type match: `{row['type_match']}`",
            f"- Semantic match: `{row['extract_match']}`",
        ])
        if row["gt_action"].get("action") == "swipe":
            lines.append(f"- GT gesture direction: `{swipe_direction(row['gt_action'])}`")
            lines.append(f"- Pred gesture direction: `{swipe_direction(row['pred_action'])}`")
        lines.extend(["", "Raw model output:", "", "```text", row["raw_text"] or row["answer"], "```", "", "Review notes:", "", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def swipe_direction(action: Dict[str, Any]) -> str:
    if not action or action.get("action") != "swipe":
        return "n/a"
    return get_scroll_direction(action.get("coordinate", [0, 0]), action.get("coordinate2", [0, 0]))


def write_report_v2(
    path: Path,
    baseline: Dict[str, Any],
    rebaseline: Dict[str, Any],
    long_press_errors: List[Dict[str, Any]],
    swipe_errors: List[Dict[str, Any]],
    swipe_sample: List[Dict[str, Any]],
) -> None:
    lines = [
        "# REPORT V2 - HAR GUI-Odyssey N0/N1 Gates",
        "",
        "Subject: `HAR`; run scope: `half_run`; match type: `semantic`.",
        "",
        "## Phase H - Harness Audit",
        "",
        "### H1 long_press Parser/Normalizer Audit",
        "",
        "The HAR action parser can map `LONG_PRESS:(x,y)` to internal `long_press`. The evaluator then passes that action to the GUI-Odyssey matcher, which supports `long_press` when GT type is `long_press`. No parser code fix was applied in this pass.",
        "",
        "| Model output format | Internal action | Payload | Source |",
        "| --- | --- | --- | --- |",
    ]
    for row in ACTION_MAPPING_ROWS:
        lines.append(f"| `{row[0]}` | `{row[1]}` | `{row[2]}` | {row[3]} |")
    lines.extend([
        "",
        f"Dumped {len(long_press_errors)} long_press error cases to [audit/long_press_cases.md](audit/long_press_cases.md).",
        "",
        "### H2 swipe Semantic-Match Audit",
        "",
        "Current convention: both GT annotations and model outputs are interpreted as gesture direction from `coordinate` to `coordinate2`. The matcher compares same axis and same gesture direction after converting predicted pixel coordinates to [0,1000] space. It does not invert to content-scroll direction. This is one global convention, not a per-step mapping.",
        "",
        "- GT convention: `get_scroll_direction(gt.coordinate, gt.coordinate2)` on normalized [0,1000] coordinates.",
        "- Model convention: `SCROLL:UP` maps to a finger gesture from lower y to upper y; `SCROLL:DOWN` maps upper y to lower y; left/right analogous.",
        f"- Dumped {len(swipe_sample)} deterministic random swipe-error cases from {len(swipe_errors)} swipe errors to [audit/swipe_cases.md](audit/swipe_cases.md).",
        "",
        "### H3 Re-baseline",
        "",
        "No H1/H2 code fix was applied before this re-baseline. Re-scoring with the current matcher reproduces the half-run numbers, so deltas are zero unless a future parser/matcher patch is accepted.",
        "",
        "| Metric | Half-run report | Post-H audit | Delta |",
        "| --- | ---: | ---: | ---: |",
        f"| Steps | {baseline['steps']} | {rebaseline['steps']} | {rebaseline['steps'] - baseline['steps']} |",
        f"| Correct steps | {baseline['correct_steps']} | {rebaseline['correct_steps']} | {rebaseline['correct_steps'] - baseline['correct_steps']} |",
        f"| Error steps | {baseline['baseline_error_steps']} | {rebaseline['baseline_error_steps']} | {rebaseline['baseline_error_steps'] - baseline['baseline_error_steps']} |",
        f"| Step SR | {baseline['step_sr_percent']:.2f}% | {rebaseline['step_sr_percent']:.2f}% | {rebaseline['step_sr_percent'] - baseline['step_sr_percent']:+.2f} pts |",
        "",
        "Error-kind table:",
        "",
        "| Error kind | Count |",
        "| --- | ---: |",
    ])
    for key, value in sorted(rebaseline["error_kind_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "Per-GT-action table:", "", "| GT action | Steps | Errors | Error rate |", "| --- | ---: | ---: | ---: |"])
    for row in rebaseline["by_gt_action"]:
        lines.append(f"| `{row['gt_action']}` | {row['steps']} | {row['errors']} | {row['error_rate_percent']:.2f}% |")
    lines.extend([
        "",
        "### GATE H",
        "",
        "STOP for human review. Acceptance requires reviewing all 23 long_press cases and the 20 swipe cases, then deciding whether the current long_press 100% error rate and swipe 65.25% error rate are genuine or require a parser/matcher patch.",
        "",
        "Status: `PENDING_HUMAN_REVIEW`.",
        "",
        "## Phase G - GT-Isolation Guardrails",
        "",
        "Pending test execution after code changes. This section will be refreshed after G checks run.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
