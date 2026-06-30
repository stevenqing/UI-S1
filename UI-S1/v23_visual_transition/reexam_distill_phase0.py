#!/usr/bin/env python3
"""Phase 0 data viability for re-examination self-distillation.

Runs V, real full-a11y teacher, and placebo full-a11y teacher on GUI-360 train
states. Counts teacher-corrected positives: V wrong but teacher right, grouped by
V failure bucket. No training is performed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pyarrow.parquet as pq
from openai import OpenAI

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from v13_gui_360.eval_gui360_template import _format_action_for_history  # noqa: E402
from v23_visual_transition.modality_jaccard import (  # noqa: E402
    attach_controls,
    build_messages,
    classify_prediction,
)
from v23_visual_transition.placebo_a11y import build_placebo_full_messages  # noqa: E402

PRIMARY_BUCKETS = ("far_miss", "type_mismatch")
TEACHERS = ("real_a11y", "placebo_a11y")


def safe_classify(pred_text: str, state: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    try:
        return classify_prediction(
            pred_text,
            state["gt_action"],
            state["image_w"],
            state["image_h"],
            args.match_threshold,
            args.near_px,
            args.far_px,
        )
    except Exception as exc:
        return {
            "success": False,
            "bucket": "classify_error",
            "reward": 0.0,
            "pred_action": None,
            "pred_type": "",
            "gt_type": str((state.get("gt_action") or {}).get("action") or ""),
            "pred_text": pred_text[:800],
            "classify_error": str(exc)[:240],
        }


def raw_path_from_screenshot(screenshot: str) -> str:
    parts = Path(screenshot).parts
    idx = parts.index("image")
    split = parts[idx - 1]
    app, category, status, exec_id = parts[idx + 1], parts[idx + 2], parts[idx + 3], parts[idx + 4]
    return f"{split}/data/{app}/{category}/{status}/{exec_id}.jsonl"


def raw_screenshot_clean_from_screenshot(screenshot: str, step_idx: int) -> str:
    parts = Path(screenshot).parts
    exec_id = parts[-2]
    filename = parts[-1]
    if filename:
        return f"success/{exec_id}/{filename}"
    return f"success/{exec_id}/action_step{step_idx + 1}.png"


def read_split_states(data_dir: str, split: str, max_episodes: int = 0) -> List[Dict[str, Any]]:
    states: List[Dict[str, Any]] = []
    episode_count = 0
    for parquet_path in sorted(Path(data_dir).glob(f"{split}-*.parquet")):
        pf = pq.ParquetFile(parquet_path)
        for batch in pf.iter_batches(batch_size=16, columns=["episode_id", "goal", "steps", "screenshots"]):
            for row in batch.to_pylist():
                episode_count += 1
                if max_episodes and episode_count > max_episodes:
                    return states
                steps = json.loads(row["steps"])
                screenshots = row.get("screenshots") or []
                history: List[str] = []
                for step in steps:
                    step_idx = int(step.get("step_idx", 0))
                    if step_idx >= len(screenshots) or not screenshots[step_idx].get("bytes"):
                        continue
                    screenshot = step.get("screenshot") or ""
                    states.append({
                        "state_id": f"{split}:{row['episode_id']}:{step_idx}",
                        "episode_id": str(row["episode_id"]),
                        "step_idx": step_idx,
                        "goal": row.get("goal", ""),
                        "gt_action": step.get("action") or {},
                        "image_w": int(step.get("image_w") or 1040),
                        "image_h": int(step.get("image_h") or 736),
                        "image_bytes": screenshots[step_idx]["bytes"],
                        "screenshot": screenshot,
                        "raw_path": raw_path_from_screenshot(screenshot),
                        "raw_screenshot_clean": raw_screenshot_clean_from_screenshot(screenshot, step_idx),
                        "history": list(history),
                    })
                    history.append(_format_action_for_history(step.get("action"), step_idx + 1))
    return states


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists():
        return []
    with file_path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate_one(args: argparse.Namespace, state: Dict[str, Any]) -> Dict[str, Any]:
    client = OpenAI(base_url=args.api_url, api_key="dummy", timeout=args.request_timeout)
    row: Dict[str, Any] = {
        "state_id": state["state_id"],
        "episode_id": state["episode_id"],
        "step_idx": state["step_idx"],
        "goal": state["goal"],
        "gt_action": state["gt_action"],
        "a11y_present": bool(state.get("controls")),
        "num_controls": len(state.get("controls") or []),
    }
    request_specs = [
        ("V", lambda: build_messages(state, "V", args.max_controls, args.image_max_pixels, "directive")),
        ("real_a11y", lambda: build_messages(state, "VA", args.max_controls, args.image_max_pixels, "directive")),
        ("placebo_a11y", lambda: build_placebo_full_messages(state, args)[0]),
    ]
    for source, message_builder in request_specs:
        try:
            messages = message_builder()
            response = client.chat.completions.create(
                model=args.model_name,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            pred_text = response.choices[0].message.content or ""
        except Exception as exc:
            pred_text = ""
            row[f"{source}_api_error"] = str(exc)[:240]
        result = safe_classify(pred_text, state, args)
        row[source] = result
    row["V_bucket"] = row["V"].get("bucket")
    row["distill_positive_real_a11y"] = (not row["V"]["success"]) and row["real_a11y"]["success"]
    row["distill_positive_placebo_a11y"] = (not row["V"]["success"]) and row["placebo_a11y"]["success"]
    row["preserve_v_correct"] = bool(row["V"]["success"])
    return row


def evaluate_states(args: argparse.Namespace, states: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if args.threads <= 1:
        for idx, state in enumerate(states, 1):
            rows.append(evaluate_one(args, state))
            if args.log_every and idx % args.log_every == 0:
                print(f"evaluated {idx}/{len(states)} states", flush=True)
        return rows
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(evaluate_one, args, state) for state in states]
        for idx, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if args.log_every and idx % args.log_every == 0:
                print(f"evaluated {idx}/{len(states)} states", flush=True)
    rows.sort(key=lambda row: (row.get("episode_id", ""), int(row.get("step_idx", 0)), row["state_id"]))
    return rows


def summarize(rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    v_bucket_counts = Counter(row["V_bucket"] for row in rows)
    teacher_counts: Dict[str, Dict[str, Any]] = {}
    for teacher in TEACHERS:
        positives = [row for row in rows if row[f"distill_positive_{teacher}"]]
        by_bucket = Counter(row["V_bucket"] for row in positives)
        teacher_counts[teacher] = {
            "positives_total": len(positives),
            "positives_by_v_bucket": dict(by_bucket),
            "primary_positive_total": sum(by_bucket[bucket] for bucket in PRIMARY_BUCKETS),
            "teacher_correct_total": sum(1 for row in rows if row[teacher]["success"]),
            "teacher_breaks_v_correct": sum(1 for row in rows if row["V"]["success"] and not row[teacher]["success"]),
        }
    v_correct_count = sum(1 for row in rows if row["V"]["success"])
    api_errors = {
        source: sum(1 for row in rows if row.get(f"{source}_api_error"))
        for source in ("V", "real_a11y", "placebo_a11y")
    }
    viable = all(teacher_counts[teacher]["primary_positive_total"] >= args.min_primary_positives for teacher in TEACHERS)
    if viable:
        verdict = "DATA-VIABLE"
        consequent = "enough teacher-corrected train positives; proceed to BC/logit distillation after review"
    else:
        verdict = "DATA-STARVED"
        consequent = "too few teacher-corrected positives; expand train slice/pool before training"
    return {
        "n": len(rows),
        "v_correct_count": v_correct_count,
        "v_correct_rate": v_correct_count / max(len(rows), 1),
        "v_bucket_counts": dict(v_bucket_counts),
        "teacher_counts": teacher_counts,
        "api_errors": api_errors,
        "min_primary_positives": args.min_primary_positives,
        "verdict": verdict,
        "consequent": consequent,
    }


def render(summary: Dict[str, Any], args: argparse.Namespace) -> str:
    lines = [
        "# Re-examination Self-Distillation Phase 0 Data Viability",
        "",
        "## Gate Verdict",
        "",
        f"**{summary['verdict']}**",
        "",
        summary["consequent"],
        "",
        "## Inputs",
        "",
        f"- split: `{args.split}`",
        f"- evaluated train states: `{summary['n']}`",
        f"- max train states requested: `{args.limit}`",
        f"- real teacher: full real `uia_controls_info` prompt",
        f"- placebo teacher: scrambled full-a11y prompt with same format/framing",
        f"- teacher target: teacher-generated action only when frozen matcher says correct; no GT action target injected",
        f"- min primary positives per teacher: `{summary['min_primary_positives']}`",
        f"- API errors: `{summary['api_errors']}`",
        "",
        "## V Baseline Distribution",
        "",
        f"- V-correct preservation pool: `{summary['v_correct_count']}` ({summary['v_correct_rate']:.4f})",
        "",
        "| V bucket | count |",
        "|---|---:|",
    ]
    for bucket, count in sorted(summary["v_bucket_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {bucket} | {count} |")
    lines += ["", "## Teacher-Corrected Positives", "", "| teacher | positives total | primary far/type positives | far_miss | type_mismatch | teacher correct | breaks V-correct |", "|---|---:|---:|---:|---:|---:|---:|"]
    for teacher in TEACHERS:
        info = summary["teacher_counts"][teacher]
        by_bucket = info["positives_by_v_bucket"]
        lines.append(
            f"| {teacher} | {info['positives_total']} | {info['primary_positive_total']} | "
            f"{by_bucket.get('far_miss', 0)} | {by_bucket.get('type_mismatch', 0)} | "
            f"{info['teacher_correct_total']} | {info['teacher_breaks_v_correct']} |"
        )
    lines += ["", "## Decision", "", f"{summary['consequent']}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced_data_dir", default="datasets/gui360-balanced/data")
    parser.add_argument("--raw_repo", default="vyokky/GUI-360")
    parser.add_argument("--raw_local_dir", default="datasets/GUI-360-raw-jsonl")
    parser.add_argument("--output_dir", default="outputs/reexam_distill")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--model_name", default="checkpoints/gui360-fullparam-sft-step250")
    parser.add_argument("--max_controls", type=int, default=256)
    parser.add_argument("--max_full_controls", type=int, default=256)
    parser.add_argument("--image_max_pixels", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--request_timeout", type=float, default=600.0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--near_px", type=float, default=50.0)
    parser.add_argument("--far_px", type=float, default=150.0)
    parser.add_argument("--min_primary_positives", type=int, default=50)
    parser.add_argument("--resume_rows", default="")
    parser.add_argument("--log_every", type=int, default=25)
    args = parser.parse_args()

    states = read_split_states(args.balanced_data_dir, args.split, args.max_episodes)
    rng = random.Random(args.seed)
    rng.shuffle(states)
    if args.limit:
        states = states[: args.limit]
    coverage = attach_controls(states, args.raw_repo, args.raw_local_dir, args.log_every)
    states = [state for state in states if state.get("controls")]
    print(f"coverage: {coverage}", flush=True)
    print(f"states with controls: {len(states)}", flush=True)
    resumed_rows = read_jsonl(args.resume_rows)
    resumed_by_id = {row["state_id"]: row for row in resumed_rows}
    states_to_eval = [state for state in states if state["state_id"] not in resumed_by_id]
    if resumed_by_id:
        print(f"resumed rows: {len(resumed_by_id)}; evaluating new states: {len(states_to_eval)}", flush=True)
    new_rows = evaluate_states(args, states_to_eval)
    selected_ids = {state["state_id"] for state in states}
    rows = [row for row in resumed_by_id.values() if row["state_id"] in selected_ids] + new_rows
    rows.sort(key=lambda row: (row.get("episode_id", ""), int(row.get("step_idx", 0)), row["state_id"]))
    summary = summarize(rows, args)
    summary["coverage"] = coverage
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "phase0_per_state.jsonl", rows)
    (output_dir / "phase0_data.json").write_text(json.dumps({"summary": summary, "args": vars(args)}, ensure_ascii=False, indent=2) + "\n")
    (output_dir / "phase0_data.md").write_text(render(summary, args))
    print(f"Wrote {output_dir / 'phase0_data.md'}")
    print(f"Wrote {output_dir / 'phase0_per_state.jsonl'}")
    print(f"PHASE0: {summary['verdict']} - {summary['consequent']}")


if __name__ == "__main__":
    main()
