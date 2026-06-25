#!/usr/bin/env python3
"""Phase 1 far-miss source viability gate for GUI-360.

Tests only the hard half of candidate orthogonality: whether a cheap
far-miss-targeted prompt-bias source can reduce TEST-split far-miss errors
relative to frozen full-SFT base, without merely trading them for other errors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from v13_gui_360.eval_gui360_template import (  # noqa: E402
    _format_action_for_history,
    build_step_prompt,
    parse_tool_call,
)
from v13_gui_360.reward import compute_step_reward, parse_action_from_text  # noqa: E402

GROUNDING_FIRST_GUIDANCE = """Use an element-grounding-first procedure for this step.

Before choosing coordinates, explicitly identify the semantic UI element needed for the current instruction and history. Compare nearby or visually similar candidate elements, reject distractors, and choose the element whose label, role, or context best matches the next required step. Then emit exactly one <tool_call> using the center of that selected element. Do not change the action type unless the screenshot clearly requires it; this source is only meant to improve element disambiguation for click targets."""

ACTION_ALIASES = {
    "drag": "swipe",
    "scroll": "swipe",
    "wheel_mouse_input": "swipe",
    "input": "type",
    "left_click": "click",
    "tap": "click",
    "double_click": "click",
}


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_episodes(path: str) -> Dict[str, Dict[str, Any]]:
    return {str(row["episode_id"]): row for row in read_jsonl(path)}


def normalize_action_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    return ACTION_ALIASES.get(text, text)


def action_coord(action: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    if not action:
        return None
    coord = action.get("coordinate") or action.get("start_coordinate") or action.get("startCoordinate")
    if coord is None or len(coord) < 2:
        return None
    try:
        return float(coord[0]), float(coord[1])
    except (TypeError, ValueError):
        return None


def coord_distance_px(pred_action: Optional[Dict[str, Any]], gt_action: Dict[str, Any]) -> Optional[float]:
    pred = action_coord(pred_action)
    gt = action_coord(gt_action)
    if pred is None or gt is None:
        return None
    return math.sqrt((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2)


def first_bad_step(eval_episode: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for step in eval_episode.get("steps", []):
        if not step.get("success"):
            return step
    return None


def classify_action(
    pred_text: str,
    gt_action: Dict[str, Any],
    image_w: int,
    image_h: int,
    near_px: float,
    far_px: float,
    match_threshold: float,
) -> Dict[str, Any]:
    pred_action = parse_tool_call(pred_text)
    if pred_action is None:
        pred_action = parse_action_from_text(pred_text)
    fake_text = f"<action>{json.dumps(pred_action)}</action>" if pred_action else pred_text
    reward, info = compute_step_reward(fake_text, gt_action, image_w=image_w, image_h=image_h)
    success = reward >= match_threshold
    gt_type = normalize_action_type(info.get("gt_type") or gt_action.get("action"))
    pred_type = normalize_action_type(info.get("pred_type") or (pred_action or {}).get("action"))
    distance = None
    if success:
        bucket = "correct"
    elif not pred_action or info.get("format_reward", 0.0) <= 0:
        bucket = "format_error"
    elif gt_type != pred_type:
        bucket = "type_mismatch"
    elif gt_type != "click":
        bucket = f"same_type_non_click:{gt_type}"
    else:
        distance = coord_distance_px(pred_action, gt_action)
        if distance is None:
            bucket = "grounding_missing_coord"
        elif distance <= near_px:
            bucket = "near_miss"
        elif distance >= far_px:
            bucket = "far_miss"
        else:
            bucket = "mid_miss"
    return {
        "success": success,
        "bucket": bucket,
        "reward": reward,
        "pred_action": pred_action,
        "pred_type": pred_type,
        "gt_type": gt_type,
        "distance_px": distance,
        "format_reward": info.get("format_reward", 0.0),
        "type_reward": info.get("type_reward", 0.0),
        "content_reward": info.get("content_reward", 0.0),
        "pred_text": pred_text[:1000],
    }


def classify_eval_step(
    eval_step: Dict[str, Any],
    gt_action: Dict[str, Any],
    near_px: float,
    far_px: float,
    image_w: int,
    image_h: int,
) -> Dict[str, Any]:
    pred_action = eval_step.get("pred_action") or parse_action_from_text(eval_step.get("pred_text", "") or "")
    gt_type = normalize_action_type(eval_step.get("gt_type") or gt_action.get("action"))
    pred_type = normalize_action_type(eval_step.get("pred_type") or (pred_action or {}).get("action"))
    if eval_step.get("success"):
        bucket = "correct"
        distance = None
    elif not pred_action or eval_step.get("format_reward", 0.0) <= 0:
        bucket = "format_error"
        distance = None
    elif gt_type != pred_type:
        bucket = "type_mismatch"
        distance = None
    elif gt_type != "click":
        bucket = f"same_type_non_click:{gt_type}"
        distance = None
    else:
        distance = coord_distance_px(pred_action, gt_action)
        if distance is None:
            bucket = "grounding_missing_coord"
        elif distance <= near_px:
            bucket = "near_miss"
        elif distance >= far_px:
            bucket = "far_miss"
        else:
            bucket = "mid_miss"
    return {
        "success": bool(eval_step.get("success")),
        "bucket": bucket,
        "reward": eval_step.get("reward", 0.0),
        "pred_action": pred_action,
        "pred_type": pred_type,
        "gt_type": gt_type,
        "distance_px": distance,
        "format_reward": eval_step.get("format_reward", 0.0),
        "type_reward": eval_step.get("type_reward", 0.0),
        "content_reward": eval_step.get("content_reward", 0.0),
        "pred_text": (eval_step.get("pred_text") or "")[:1000],
    }


def build_base_farmiss_set(args: argparse.Namespace, episodes: Dict[str, Dict[str, Any]], base_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    states = []
    for episode_id, eval_episode in base_results.items():
        if eval_episode.get("task_success"):
            continue
        episode = episodes.get(str(episode_id))
        bad = first_bad_step(eval_episode)
        if episode is None or bad is None:
            continue
        step_idx = int(bad.get("step_idx", 0))
        if step_idx >= len(episode.get("steps", [])):
            continue
        step = episode["steps"][step_idx]
        base = classify_eval_step(
            bad,
            step.get("action") or {},
            args.near_px,
            args.far_px,
            int(step.get("image_w") or 1040),
            int(step.get("image_h") or 736),
        )
        if base["bucket"] != "far_miss":
            continue
        history = []
        for prior in eval_episode.get("steps", [])[:step_idx]:
            history.append(_format_action_for_history(prior.get("pred_action"), int(prior.get("step_idx", len(history))) + 1))
        states.append({
            "episode_id": str(episode_id),
            "goal": episode.get("goal", ""),
            "step_idx": step_idx,
            "screenshot": step.get("screenshot"),
            "image_w": int(step.get("image_w") or 1040),
            "image_h": int(step.get("image_h") or 736),
            "gt_action": step.get("action") or {},
            "history": history,
            "base": base,
        })
    states.sort(key=lambda row: int(row["episode_id"]) if str(row["episode_id"]).isdigit() else row["episode_id"])
    if args.shuffle_seed is not None:
        rng = random.Random(args.shuffle_seed)
        rng.shuffle(states)
    if args.limit:
        states = states[: args.limit]
    return states


def call_source(client: Any, args: argparse.Namespace, state: Dict[str, Any]) -> Dict[str, Any]:
    messages = build_step_prompt(
        state["goal"],
        state["screenshot"],
        state["step_idx"],
        state["history"],
        guidance=GROUNDING_FIRST_GUIDANCE,
        image_max_pixels=args.image_max_pixels,
    )
    response = client.chat.completions.create(
        model=args.model_name,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    pred_text = response.choices[0].message.content or ""
    return classify_action(
        pred_text,
        state["gt_action"],
        state["image_w"],
        state["image_h"],
        args.near_px,
        args.far_px,
        args.match_threshold,
    )


def bootstrap_ci(values: Sequence[float], seed: int, samples: int = 10000) -> Tuple[float, float, float]:
    arr = np.array(values, dtype=np.float64)
    mean = float(arr.mean()) if len(arr) else 0.0
    if len(arr) == 0:
        return mean, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(samples, len(arr)))
    boot = arr[idx].mean(axis=1)
    return mean, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def summarize(rows: List[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    n = len(rows)
    base_counts = Counter(row["base"]["bucket"] for row in rows)
    source_counts = Counter(row["source"]["bucket"] for row in rows)
    base_far = [1.0 if row["base"]["bucket"] == "far_miss" else 0.0 for row in rows]
    source_far = [1.0 if row["source"]["bucket"] == "far_miss" else 0.0 for row in rows]
    source_correct = [1.0 if row["source"]["bucket"] == "correct" else 0.0 for row in rows]
    reduction = [b - s for b, s in zip(base_far, source_far)]
    correct_gain = [c - (1.0 if row["base"]["bucket"] == "correct" else 0.0) for c, row in zip(source_correct, rows)]
    reduction_mean, reduction_lo, reduction_hi = bootstrap_ci(reduction, seed)
    correct_mean, correct_lo, correct_hi = bootstrap_ci(correct_gain, seed + 1)
    source_far_mean, source_far_lo, source_far_hi = bootstrap_ci(source_far, seed + 2)
    type_mismatch_rate = source_counts.get("type_mismatch", 0) / max(n, 1)
    format_error_rate = source_counts.get("format_error", 0) / max(n, 1)
    nonfar_incorrect = sum(count for bucket, count in source_counts.items() if bucket not in {"far_miss", "correct"}) / max(n, 1)
    viable = reduction_lo > 0 and correct_mean > 0 and nonfar_incorrect <= reduction_mean
    if viable:
        verdict = "VIABLE"
        consequent = "hard far-miss source bias exists; proceed to type-mismatch source only after review"
    elif reduction_lo <= 0 or correct_mean <= 0 or nonfar_incorrect > reduction_mean:
        verdict = "NOT VIABLE"
        consequent = "prompt-bias source does not establish an effective far-miss source; do not proceed to Jaccard"
    else:
        verdict = "INCONCLUSIVE"
        consequent = "effect is marginal or redistribution is ambiguous; try next ladder mechanism before declaring viability"
    return {
        "n": n,
        "base_counts": dict(base_counts),
        "source_counts": dict(source_counts),
        "base_far_miss_rate": sum(base_far) / max(n, 1),
        "source_far_miss_rate": source_far_mean,
        "source_far_miss_rate_ci95": [source_far_lo, source_far_hi],
        "far_miss_reduction": reduction_mean,
        "far_miss_reduction_ci95": [reduction_lo, reduction_hi],
        "source_correct_rate": sum(source_correct) / max(n, 1),
        "correct_gain": correct_mean,
        "correct_gain_ci95": [correct_lo, correct_hi],
        "source_type_mismatch_rate": type_mismatch_rate,
        "source_format_error_rate": format_error_rate,
        "source_nonfar_incorrect_rate": nonfar_incorrect,
        "verdict": verdict,
        "consequent": consequent,
    }


def write_per_state(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps({
                "episode_id": row["episode_id"],
                "step_idx": row["step_idx"],
                "goal": row["goal"],
                "screenshot": row["screenshot"],
                "gt_action": row["gt_action"],
                "base_action": row["base"].get("pred_action"),
                "source_action": row["source"].get("pred_action"),
                "base_bucket": row["base"].get("bucket"),
                "source_bucket": row["source"].get("bucket"),
                "base_far_miss": row["base"].get("bucket") == "far_miss",
                "source_far_miss": row["source"].get("bucket") == "far_miss",
                "source_success": row["source"].get("success"),
                "base_distance_px": row["base"].get("distance_px"),
                "source_distance_px": row["source"].get("distance_px"),
                "base_pred_type": row["base"].get("pred_type"),
                "source_pred_type": row["source"].get("pred_type"),
                "base_reward": row["base"].get("reward"),
                "source_reward": row["source"].get("reward"),
                "source_pred_text": row["source"].get("pred_text"),
            }, ensure_ascii=False) + "\n")


def render_summary(args: argparse.Namespace, summary: Dict[str, Any], total_far_states: int) -> str:
    lines = [
        "# Phase 1 Far-Miss Source Viability (GUI-360)",
        "",
        "## Gate Verdict",
        "",
        f"**{summary['verdict']}**",
        "",
        summary["consequent"],
        "",
        "## Construction Mechanism",
        "",
        "Section 1.1: decoding/prompt bias toward element grounding.",
        "",
        "The source uses the same frozen full-SFT model and adds a grounding-first guidance block that forces semantic element selection before coordinate emission. No training, no crop source, no type-mismatch source, and no Jaccard are built in this phase.",
        "",
        "## Inputs",
        "",
        f"- test data: `{args.test_data}`",
        f"- base eval results: `{args.base_eval_results}`",
        f"- model name: `{args.model_name}`",
        f"- api url: `{args.api_url}`",
        f"- output dir: `{args.output_dir}`",
        f"- total TEST far-miss states available: `{total_far_states}`",
        f"- evaluated states: `{summary['n']}`",
        f"- 200-slice mode: `{bool(args.limit and args.limit <= 200)}`",
        "",
        "## Primary Metric",
        "",
        f"- base far-miss rate: `{summary['base_far_miss_rate']:.4f}`",
        f"- source far-miss rate: `{summary['source_far_miss_rate']:.4f}` (95% CI `{summary['source_far_miss_rate_ci95'][0]:.4f}` / `{summary['source_far_miss_rate_ci95'][1]:.4f}`)",
        f"- far-miss reduction: `{summary['far_miss_reduction']:.4f}` (95% CI `{summary['far_miss_reduction_ci95'][0]:.4f}` / `{summary['far_miss_reduction_ci95'][1]:.4f}`)",
        f"- source correctness on set: `{summary['source_correct_rate']:.4f}`",
        f"- correct gain: `{summary['correct_gain']:.4f}` (95% CI `{summary['correct_gain_ci95'][0]:.4f}` / `{summary['correct_gain_ci95'][1]:.4f}`)",
        "",
        "## Anti-Degradation Redistribution",
        "",
        "| bucket | base count | source count | source rate |",
        "|---|---:|---:|---:|",
    ]
    buckets = sorted(set(summary["base_counts"]) | set(summary["source_counts"]))
    for bucket in buckets:
        base = summary["base_counts"].get(bucket, 0)
        source = summary["source_counts"].get(bucket, 0)
        lines.append(f"| {bucket} | {base} | {source} | {source / max(summary['n'], 1):.4f} |")
    lines += [
        "",
        "## Gate Checks",
        "",
        f"- source type-mismatch rate: `{summary['source_type_mismatch_rate']:.4f}`",
        f"- source format-error rate: `{summary['source_format_error_rate']:.4f}`",
        f"- source non-far incorrect rate: `{summary['source_nonfar_incorrect_rate']:.4f}`",
        "",
        "Pre-registered viable condition in this implementation: far-miss reduction CI lower bound > 0, correct gain > 0, and non-far incorrect redistribution does not exceed the far-miss reduction. This prevents counting a source that only converts far-miss into type-mismatch/format errors as viable.",
        "",
        "## Notes",
        "",
        "This phase tests only the far-miss source. It intentionally does not build the type-mismatch source and does not compute Jaccard/verifier selection.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", default="datasets/gui360-balanced/gui360_test_1000_balanced.jsonl")
    parser.add_argument("--base_eval_results", default="outputs/gui360_fullparam_sft_step250_balanced_8gpu64_stop_bounded/eval_results_20260624_153846.json")
    parser.add_argument("--output_dir", default="outputs/candidate_orthogonality/phase1_farmiss_source")
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--model_name", default="checkpoints/gui360-fullparam-sft-step250")
    parser.add_argument("--near_px", type=float, default=50.0)
    parser.add_argument("--far_px", type=float, default=150.0)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--request_timeout", type=float, default=600.0)
    parser.add_argument("--image_max_pixels", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shuffle_seed", type=int, default=None)
    parser.add_argument("--bootstrap_seed", type=int, default=17)
    parser.add_argument("--log_every", type=int, default=20)
    args = parser.parse_args()

    from openai import OpenAI

    episodes = load_episodes(args.test_data)
    with open(args.base_eval_results) as handle:
        base_results = json.load(handle)
    all_args = argparse.Namespace(**{**vars(args), "limit": 0})
    all_states = build_base_farmiss_set(all_args, episodes, base_results)
    states = build_base_farmiss_set(args, episodes, base_results)
    if len(states) < 30:
        raise SystemExit(f"insufficient far-miss states: {len(states)}")

    client = OpenAI(base_url=args.api_url, api_key="dummy", timeout=args.request_timeout)
    rows = []
    for idx, state in enumerate(states, 1):
        source = call_source(client, args, state)
        row = dict(state)
        row["source"] = source
        rows.append(row)
        if args.log_every and idx % args.log_every == 0:
            print(f"evaluated {idx}/{len(states)} far-miss states", flush=True)

    summary = summarize(rows, args.bootstrap_seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps({"summary": summary, "args": vars(args), "total_far_states": len(all_states)}, ensure_ascii=False, indent=2) + "\n")
    (output_dir / "summary.md").write_text(render_summary(args, summary, len(all_states)))
    write_per_state(output_dir / "per_state.jsonl", rows)
    print(f"Wrote {output_dir / 'summary.md'}")
    print(f"Wrote {output_dir / 'per_state.jsonl'}")
    print(f"GATE: {summary['verdict']} - {summary['consequent']}")


if __name__ == "__main__":
    main()
