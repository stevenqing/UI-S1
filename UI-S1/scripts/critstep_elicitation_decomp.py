#!/usr/bin/env python3
"""Sampling decomposition for critical-step failures.

This is a diagnostic only: it samples the frozen SFT baseline on already-defined
GUI-360 critical-step failures, scores each sample with the frozen matcher, and
summarizes whether failures are elicitation-recoverable or capability-missing.
No training is performed.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v13_gui_360.eval_gui360_template import (  # noqa: E402
    _format_action_for_history,
    build_step_prompt,
    parse_tool_call,
)
from v13_gui_360.reward import compute_step_reward  # noqa: E402
from v23_visual_transition.modality_jaccard import classify_prediction  # noqa: E402


DEFAULT_TEST_DATA = "outputs/gui360_history_ab/original_eval/gui360_test_1000_balanced_reconstructed.jsonl"
DEFAULT_CRIT_PER_TASK = "outputs/critstep_eval/per_task.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/critstep_elicit"
DEFAULT_MODEL = "checkpoints/gui360-fullparam-sft-step250"
DEFAULT_EVAL_GLOBS = [
    "outputs/gui360_history_ab/original_sft_template_gt_history_retry_part*_20260630/eval_results_*.json",
    "outputs/gui360_history_ab/original_sft_template_gt_history_part5_20260630/eval_results_*.json",
    "outputs/gui360_history_ab/original_sft_template_gt_history_part6_20260630/eval_results_*.json",
    "outputs/gui360_history_ab/original_sft_template_gt_history_part7_20260630/eval_results_*.json",
]
N_SWEEP = (1, 5, 10, 20, 50)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_test_episodes(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row["episode_id"]): row for row in read_jsonl(path)}


def read_crit_tasks(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row["episode_id"]): row for row in read_jsonl(path)}


def discover_eval_paths(patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(Path(path) for path in sorted(glob.glob(pattern)))
    unique: List[Path] = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def read_eval_results(paths: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, value in data.items():
            episode_id = str(value.get("episode_id", key))
            if episode_id in out:
                raise ValueError(f"duplicate eval result for episode {episode_id} in {path}")
            out[episode_id] = value
    return out


def safe_classify(pred_text: str, gt_action: Dict[str, Any], image_w: int, image_h: int, match_threshold: float, near_px: float, far_px: float) -> Dict[str, Any]:
    try:
        return classify_prediction(pred_text, gt_action, image_w, image_h, match_threshold, near_px, far_px)
    except Exception as exc:
        return {
            "success": False,
            "bucket": "classify_error",
            "reward": 0.0,
            "pred_action": None,
            "pred_type": "",
            "gt_type": str(gt_action.get("action") or ""),
            "pred_text": pred_text[:800],
            "classify_error": str(exc)[:240],
        }


def classify_eval_step(step_result: Dict[str, Any], gt_action: Dict[str, Any], image_w: int, image_h: int, match_threshold: float, near_px: float, far_px: float) -> Dict[str, Any]:
    pred_text = str(step_result.get("pred_text") or "")
    return safe_classify(pred_text, gt_action, image_w, image_h, match_threshold, near_px, far_px)


def build_history(steps: Sequence[Dict[str, Any]], target_idx: int) -> List[str]:
    return [_format_action_for_history(step.get("action"), idx + 1) for idx, step in enumerate(steps[:target_idx])]


def action_type(action: Dict[str, Any], fallback: str = "unknown") -> str:
    return str(action.get("action") or fallback or "unknown").strip().lower() or "unknown"


def target_id(population: str, episode_id: str, step_idx: int) -> str:
    return f"{population}:{episode_id}:{step_idx}"


def build_targets(
    episodes: Dict[str, Dict[str, Any]],
    crit_tasks: Dict[str, Dict[str, Any]],
    eval_results: Dict[str, Dict[str, Any]],
    *,
    noncritical_max: int,
    seed: int,
    match_threshold: float,
    near_px: float,
    far_px: float,
    max_critical: int,
) -> List[Dict[str, Any]]:
    critical: List[Dict[str, Any]] = []
    noncritical: List[Dict[str, Any]] = []
    for episode_id, task in sorted(crit_tasks.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]):
        episode = episodes.get(episode_id)
        result = eval_results.get(episode_id)
        if not episode or not result:
            raise ValueError(f"missing episode/eval result for {episode_id}")
        steps = episode.get("steps") or []
        eval_steps = result.get("steps") or []
        bottom1 = set(int(idx) for idx in task.get("bottom1_critical_indices", []))
        bottom2 = set(int(idx) for idx in task.get("bottom2_critical_indices", []))
        per_success = [bool(value) for value in task.get("per_step_success", [])]
        per_p = [float(value) for value in task.get("per_step_p_heldout_cv", [])]
        for step_idx, success in enumerate(per_success):
            if success:
                continue
            if step_idx >= len(steps) or step_idx >= len(eval_steps):
                continue
            step = steps[step_idx]
            gt_action = step.get("action") if isinstance(step.get("action"), dict) else {}
            greedy = classify_eval_step(
                eval_steps[step_idx],
                gt_action,
                int(step.get("image_w") or 1040),
                int(step.get("image_h") or 736),
                match_threshold,
                near_px,
                far_px,
            )
            is_critical = step_idx in bottom2
            population = "critical" if is_critical else "noncritical"
            payload = {
                "target_id": target_id(population, episode_id, step_idx),
                "population": population,
                "episode_id": episode_id,
                "step_idx": int(step_idx),
                "task_k": int(task.get("k") or len(steps)),
                "bottom1": step_idx in bottom1,
                "bottom2": is_critical,
                "p_hat_heldout": per_p[step_idx] if step_idx < len(per_p) else None,
                "goal": episode.get("goal", ""),
                "screenshot": step.get("screenshot"),
                "gt_action": gt_action,
                "image_w": int(step.get("image_w") or 1040),
                "image_h": int(step.get("image_h") or 736),
                "history": build_history(steps, step_idx),
                "greedy": greedy,
                "greedy_bucket": greedy.get("bucket"),
                "action_type": action_type(gt_action, greedy.get("gt_type", "unknown")),
                "click_far_miss": bool(greedy.get("bucket") == "far_miss" and action_type(gt_action, greedy.get("gt_type")) == "click"),
            }
            if is_critical:
                critical.append(payload)
            else:
                noncritical.append(payload)
    if max_critical > 0:
        critical = critical[:max_critical]
    if noncritical_max > 0 and len(noncritical) > noncritical_max:
        rng = random.Random(seed)
        noncritical = sorted(rng.sample(noncritical, noncritical_max), key=lambda row: (int(row["episode_id"]) if str(row["episode_id"]).isdigit() else row["episode_id"], row["step_idx"]))
    return critical + noncritical


def build_phase0_targets(
    episodes: Dict[str, Dict[str, Any]],
    phase0_rows: Sequence[Dict[str, Any]],
    *,
    wrong_source: str,
    max_critical: int,
) -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []
    for row in phase0_rows:
        if not row.get("bottom2"):
            continue
        greedy = row.get(wrong_source) if isinstance(row.get(wrong_source), dict) else {}
        if greedy.get("success"):
            continue
        episode_id = str(row.get("episode_id"))
        step_idx = int(row.get("step_idx"))
        episode = episodes.get(episode_id)
        if not episode:
            raise ValueError(f"missing episode {episode_id} in --test_data for Phase0 row")
        steps = episode.get("steps") or []
        if step_idx >= len(steps):
            raise ValueError(f"step index out of range for episode {episode_id}: {step_idx} >= {len(steps)}")
        step = steps[step_idx]
        gt_action = step.get("action") if isinstance(step.get("action"), dict) else row.get("gt_action") or {}
        action = action_type(gt_action, greedy.get("gt_type", "unknown"))
        targets.append({
            "target_id": target_id("critical", episode_id, step_idx),
            "population": "critical",
            "episode_id": episode_id,
            "step_idx": step_idx,
            "task_k": int(row.get("task_k") or len(steps)),
            "bottom1": bool(row.get("bottom1")),
            "bottom2": bool(row.get("bottom2")),
            "p_hat_heldout": row.get("p_hat_heldout_bucket"),
            "goal": episode.get("goal") or row.get("goal", ""),
            "screenshot": step.get("screenshot"),
            "gt_action": gt_action,
            "image_w": int(step.get("image_w") or 1040),
            "image_h": int(step.get("image_h") or 736),
            "history": build_history(steps, step_idx),
            "greedy": greedy,
            "greedy_bucket": greedy.get("bucket") or row.get(f"{wrong_source}_bucket") or row.get("V_bucket"),
            "action_type": action,
            "click_far_miss": bool((greedy.get("bucket") or row.get("V_bucket")) == "far_miss" and action == "click"),
        })
        if max_critical > 0 and len(targets) >= max_critical:
            break
    return targets


def sample_batch(
    client: OpenAI,
    args: argparse.Namespace,
    messages: List[Dict[str, Any]],
    temperature: float,
    n: int,
) -> List[str]:
    response = client.chat.completions.create(
        model=args.model_name,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=temperature,
        top_p=args.top_p,
        n=n,
    )
    return [(choice.message.content or "") for choice in response.choices]


def score_sample(pred_text: str, target: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    pred_action = parse_tool_call(pred_text)
    fake_text = f"<action>{json.dumps(pred_action)}</action>" if pred_action else pred_text
    reward, info = compute_step_reward(fake_text, target["gt_action"], target["image_w"], target["image_h"])
    classified = safe_classify(pred_text, target["gt_action"], target["image_w"], target["image_h"], args.match_threshold, args.near_px, args.far_px)
    return {
        "success": bool(reward >= args.match_threshold),
        "reward": float(reward),
        "bucket": classified.get("bucket"),
        "pred_action": info.get("pred_action") or classified.get("pred_action"),
        "pred_type": info.get("pred_type") or classified.get("pred_type"),
        "gt_type": info.get("gt_type") or classified.get("gt_type"),
        "pred_text": pred_text[: args.store_pred_text_chars],
    }


def pass_at(successes: Sequence[bool], k: int) -> bool:
    return any(successes[: min(k, len(successes))])


def first_correct_rank(successes: Sequence[bool]) -> Optional[int]:
    for idx, value in enumerate(successes, 1):
        if value:
            return idx
    return None


def sample_target_temperature(target: Dict[str, Any], temperature: float, args: argparse.Namespace) -> Dict[str, Any]:
    client = OpenAI(base_url=args.api_url, api_key="dummy", timeout=args.request_timeout)
    messages = build_step_prompt(
        target["goal"],
        target["screenshot"],
        int(target["step_idx"]),
        list(target.get("history") or []),
        image_max_pixels=args.image_max_pixels,
    )
    samples: List[Dict[str, Any]] = []
    remaining = args.n_samples
    errors: List[str] = []
    while remaining > 0:
        take = min(args.samples_per_request, remaining)
        pred_texts: List[str] = []
        for attempt in range(args.max_retries + 1):
            try:
                pred_texts = sample_batch(client, args, messages, temperature, take)
                break
            except Exception as exc:  # noqa: BLE001 - report API failure in artifact
                errors.append(str(exc)[:300])
                if attempt >= args.max_retries:
                    pred_texts = [""] * take
                    break
                time.sleep(min(10.0, 1.5 ** attempt))
        samples.extend(score_sample(text, target, args) for text in pred_texts)
        remaining -= take
    successes = [bool(sample["success"]) for sample in samples]
    first_rank = first_correct_rank(successes)
    pass_curve = {f"pass_at_{k}": pass_at(successes, k) for k in N_SWEEP if k <= args.n_samples}
    return {
        "target_id": target["target_id"],
        "population": target["population"],
        "episode_id": target["episode_id"],
        "step_idx": target["step_idx"],
        "task_k": target["task_k"],
        "bottom1": target["bottom1"],
        "bottom2": target["bottom2"],
        "p_hat_heldout": target.get("p_hat_heldout"),
        "action_type": target["action_type"],
        "click_far_miss": target["click_far_miss"],
        "greedy_bucket": target["greedy_bucket"],
        "greedy": target["greedy"],
        "temperature": temperature,
        "n_samples": args.n_samples,
        "samples_per_request": args.samples_per_request,
        "success_count": int(sum(successes)),
        "recoverable": bool(any(successes)),
        "first_correct_rank": first_rank,
        **pass_curve,
        "samples": samples,
        "api_errors": errors[:10],
        "api_error_count": len(errors),
    }


def done_keys(per_step_path: Path) -> set[Tuple[str, float]]:
    if not per_step_path.exists():
        return set()
    keys = set()
    with per_step_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add((str(row.get("target_id")), float(row.get("temperature"))))
    return keys


def done_keys_many(paths: Sequence[Path]) -> set[Tuple[str, float]]:
    keys: set[Tuple[str, float]] = set()
    for path in paths:
        keys.update(done_keys(path))
    return keys


def append_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()


def load_sample_rows(path: Path) -> List[Dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_group(rows: Sequence[Dict[str, Any]], n_samples: int) -> Dict[str, Any]:
    total = len(rows)
    recovered = sum(1 for row in rows if row.get("recoverable"))
    pass_curve = {}
    for k in N_SWEEP:
        if k <= n_samples:
            key = f"pass_at_{k}"
            pass_curve[key] = fraction(sum(1 for row in rows if row.get(key)), total)
    ranks = [int(row["first_correct_rank"]) for row in rows if row.get("first_correct_rank") is not None]
    rank_bins = {
        "rank_1": sum(1 for value in ranks if value == 1),
        "rank_2_5": sum(1 for value in ranks if 2 <= value <= 5),
        "rank_6_10": sum(1 for value in ranks if 6 <= value <= 10),
        "rank_11_20": sum(1 for value in ranks if 11 <= value <= 20),
        "rank_21_50": sum(1 for value in ranks if 21 <= value <= 50),
        "missing": total - recovered,
    }
    success_counts = [int(row.get("success_count") or 0) for row in rows]
    return {
        "total": total,
        "recoverable": recovered,
        "missing": total - recovered,
        "recoverable_fraction": fraction(recovered, total),
        "missing_fraction": fraction(total - recovered, total),
        "pass_curve": pass_curve,
        "saturation_delta_20_50": pass_curve.get("pass_at_50", 0.0) - pass_curve.get("pass_at_20", 0.0),
        "first_correct_rank_bins": rank_bins,
        "success_count_mean": sum(success_counts) / total if total else 0.0,
        "success_count_median": sorted(success_counts)[total // 2] if total else 0.0,
    }


def table_line(label: str, summary: Dict[str, Any]) -> str:
    curve = summary["pass_curve"]
    return (
        f"| {label} | {summary['total']} | {summary['recoverable']} ({summary['recoverable_fraction']*100:.2f}%) | "
        f"{summary['missing']} ({summary['missing_fraction']*100:.2f}%) | "
        f"{curve.get('pass_at_1', 0.0)*100:.2f}% | {curve.get('pass_at_5', 0.0)*100:.2f}% | "
        f"{curve.get('pass_at_10', 0.0)*100:.2f}% | {curve.get('pass_at_20', 0.0)*100:.2f}% | "
        f"{curve.get('pass_at_50', 0.0)*100:.2f}% | {summary['saturation_delta_20_50']*100:.2f}pp |"
    )


def decide_verdict(critical_summary: Dict[str, Any]) -> Tuple[str, str]:
    recovered = critical_summary["recoverable_fraction"]
    missing = critical_summary["missing_fraction"]
    delta = critical_summary["saturation_delta_20_50"]
    if delta > 0.05:
        return (
            "MIXED / UNSATURATED",
            f"pass@50 is still {delta*100:.2f}pp above pass@20, so MISSING@50 is an upper bound; larger N or a tail-temperature pass may still find more recoverable critical steps.",
        )
    if recovered >= 0.50:
        return (
            "ELICITATION-DOMINATED",
            f"{recovered*100:.2f}% of critical failures are recoverable@50 and the curve is near-saturated; the main problem is selection.",
        )
    if missing >= 0.50:
        return (
            "CAPABILITY-DOMINATED",
            f"{missing*100:.2f}% of critical failures are still missing@50 with near-saturation; verifier/BoN has limited ceiling unless more samples or higher temperature reveal hidden positives.",
        )
    return (
        "MIXED",
        f"recoverable@50 is {recovered*100:.2f}% and missing@50 is {missing*100:.2f}%; neither side dominates cleanly.",
    )


def render_rank_table(summary: Dict[str, Any]) -> List[str]:
    bins = summary["first_correct_rank_bins"]
    total = summary["total"]
    lines = ["| first-correct rank bin | count | share |", "|---|---:|---:|"]
    labels = [
        ("1", "rank_1"),
        ("2-5", "rank_2_5"),
        ("6-10", "rank_6_10"),
        ("11-20", "rank_11_20"),
        ("21-50", "rank_21_50"),
        ("missing@50", "missing"),
    ]
    for label, key in labels:
        count = int(bins.get(key, 0))
        lines.append(f"| {label} | {count} | {fraction(count, total)*100:.2f}% |")
    return lines


def render_report(args: argparse.Namespace, rows: Sequence[Dict[str, Any]], target_counts: Dict[str, int]) -> str:
    by_temp: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_temp[float(row.get("temperature"))].append(row)
    primary_temp = float(args.temperatures[0])
    primary_rows = by_temp.get(primary_temp, [])
    critical_rows = [row for row in primary_rows if row.get("population") == "critical"]
    noncritical_rows = [row for row in primary_rows if row.get("population") == "noncritical"]
    click_far_rows = [row for row in critical_rows if row.get("click_far_miss")]

    summaries = {
        "critical": summarize_group(critical_rows, args.n_samples),
        "critical_click_far_miss": summarize_group(click_far_rows, args.n_samples),
        "noncritical_reference": summarize_group(noncritical_rows, args.n_samples),
    }
    verdict, verdict_reason = decide_verdict(summaries["critical"])
    lines: List[str] = []
    lines.append("# Critical-Step Elicitation-vs-Capability Decomposition")
    lines.append("")
    lines.append("Diagnostic only: sampling + frozen matcher scoring. No training was performed.")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- model: `{args.model_name}`")
    lines.append(f"- test data: `{args.test_data}`")
    lines.append(f"- critical definition: bottom-2 by held-out p_i from `{args.crit_per_task}`")
    lines.append(f"- primary temperature: `{primary_temp}`")
    lines.append(f"- samples per target and temperature: `{args.n_samples}`")
    lines.append(f"- N sweep: `{', '.join(map(str, [k for k in N_SWEEP if k <= args.n_samples]))}`")
    lines.append(f"- target counts requested: `{target_counts}`")
    lines.append(f"- sample rows completed: `{len(rows)}`")
    lines.append("")
    if args.noncritical_max > 0:
        lines.append(f"Non-critical reference is a seeded sample capped at `{args.noncritical_max}` failures, not all non-critical failures.")
        lines.append("")
    lines.append("## Metric 1: pass@N Decomposition")
    lines.append("")
    lines.append("| population | failing steps | RECOVERABLE@50 | MISSING@50 | pass@1 | pass@5 | pass@10 | pass@20 | pass@50 | pass@50 - pass@20 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(table_line("critical bottom-2 failures", summaries["critical"]))
    lines.append(table_line("critical click-far_miss failures", summaries["critical_click_far_miss"]))
    lines.append(table_line("non-critical failure reference", summaries["noncritical_reference"]))
    lines.append("")
    lines.append("Saturation read: if pass@50 is still meaningfully above pass@20, MISSING@50 is an upper bound rather than a settled capability ceiling.")
    lines.append("")
    lines.append("## Metric 2: Elicitation Depth")
    lines.append("")
    lines.append("Critical bottom-2 failures:")
    lines.append("")
    lines.extend(render_rank_table(summaries["critical"]))
    lines.append("")
    lines.append("Critical click-far_miss failures:")
    lines.append("")
    lines.extend(render_rank_table(summaries["critical_click_far_miss"]))
    lines.append("")
    lines.append("## Metric 3: Verifier Recovery")
    lines.append("")
    rec_frac = summaries["critical"].get("recoverable_fraction", 0.0)
    lines.append(f"Oracle recoverable ceiling on critical failures at N={args.n_samples}: `{rec_frac*100:.2f}%`.")
    lines.append("")
    lines.append("No existing verifier artifact was run in this diagnostic. If the oracle recoverable fraction is non-trivial, the next measurement is verifier selection accuracy restricted to these RECOVERABLE critical-step pools.")
    lines.append("")
    lines.append("## Solution-Direction Verdict")
    lines.append("")
    lines.append(f"**{verdict}**")
    lines.append("")
    lines.append(verdict_reason)
    lines.append("")
    if verdict.startswith("ELICITATION"):
        lines.append("Consequence: focus on verifier / BoN / RL selection on critical steps; the model distribution often contains a correct action.")
    elif verdict.startswith("CAPABILITY"):
        lines.append("Consequence: critical-step grounding is mostly a base-capability ceiling under this sampling budget; verifier/RL has limited ceiling because there is often no correct sample to select.")
    else:
        lines.append("Consequence: report the split honestly; if unsaturated, larger-N or a tail-temperature probe is needed before declaring a capability ceiling.")
    lines.append("")
    lines.append("## Additional Temperatures")
    lines.append("")
    if len(by_temp) <= 1:
        lines.append("Only the primary temperature was completed in this run.")
    else:
        lines.append("| temperature | population | failing steps | RECOVERABLE@50 | MISSING@50 | pass@1 | pass@5 | pass@10 | pass@20 | pass@50 | pass@50 - pass@20 |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for temp, temp_rows in sorted(by_temp.items()):
            crit = [row for row in temp_rows if row.get("population") == "critical"]
            click_far = [row for row in crit if row.get("click_far_miss")]
            noncrit = [row for row in temp_rows if row.get("population") == "noncritical"]
            for label, group_rows in (
                ("critical bottom-2 failures", crit),
                ("critical click-far_miss failures", click_far),
                ("non-critical failure reference", noncrit),
            ):
                summary = summarize_group(group_rows, args.n_samples)
                curve = summary["pass_curve"]
                lines.append(
                    f"| {temp} | {label} | {summary['total']} | "
                    f"{summary['recoverable']} ({summary['recoverable_fraction']*100:.2f}%) | "
                    f"{summary['missing']} ({summary['missing_fraction']*100:.2f}%) | "
                    f"{curve.get('pass_at_1', 0.0)*100:.2f}% | {curve.get('pass_at_5', 0.0)*100:.2f}% | "
                    f"{curve.get('pass_at_10', 0.0)*100:.2f}% | {curve.get('pass_at_20', 0.0)*100:.2f}% | "
                    f"{curve.get('pass_at_50', 0.0)*100:.2f}% | {summary['saturation_delta_20_50']*100:.2f}pp |"
                )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{Path(args.output_dir) / 'decomposition.md'}`")
    lines.append(f"- `{Path(args.output_dir) / 'per_step.jsonl'}`")
    lines.append("")
    lines.append("STOP for review.")
    return "\n".join(lines) + "\n"


def summarize_only(args: argparse.Namespace, target_counts: Optional[Dict[str, int]] = None) -> None:
    output_dir = Path(args.output_dir)
    per_step = output_dir / "per_step.jsonl"
    rows = load_sample_rows(per_step)
    if not rows:
        raise SystemExit(f"no rows to summarize at {per_step}")
    if target_counts is None:
        targets_by_population: Dict[str, set[str]] = defaultdict(set)
        for row in rows:
            population = str(row.get("population") or "unknown")
            targets_by_population[population].add(str(row.get("target_id") or ""))
        target_counts = {population: len(ids) for population, ids in sorted(targets_by_population.items())}
    report = render_report(args, rows, target_counts)
    (output_dir / "decomposition.md").write_text(report, encoding="utf-8")
    row_counts = Counter((str(row.get("population") or "unknown"), str(row.get("temperature"))) for row in rows)
    recoverable_counts = Counter((str(row.get("population") or "unknown"), str(row.get("temperature"))) for row in rows if row.get("recoverable"))
    seen = set()
    duplicates = 0
    for row in rows:
        key = (str(row.get("target_id")), str(row.get("temperature")))
        duplicates += int(key in seen)
        seen.add(key)
    summary = {
        "rows": len(rows),
        "target_counts": target_counts,
        "row_counts_by_population_temperature": {f"{population}@{temperature}": count for (population, temperature), count in sorted(row_counts.items())},
        "recoverable_by_population_temperature": {f"{population}@{temperature}": count for (population, temperature), count in sorted(recoverable_counts.items())},
        "duplicates": duplicates,
        "sampling_errors": sum(1 for row in rows if "sampling_error" in row),
        "api_error_count": sum(int(row.get("api_error_count") or 0) for row in rows),
        "report": str(output_dir / "decomposition.md"),
        "per_step": str(per_step),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def parse_temperatures(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_data", default=DEFAULT_TEST_DATA)
    parser.add_argument("--crit_per_task", default=DEFAULT_CRIT_PER_TASK)
    parser.add_argument("--phase0_per_state", default="", help="optional Phase0 per-state JSONL; when set, build bottom-2 wrong TRAIN targets from this file")
    parser.add_argument("--phase0_wrong_source", default="V", help="Phase0 prediction field used as greedy hard negative")
    parser.add_argument("--eval_results", nargs="*", default=[])
    parser.add_argument("--api_url", default="http://127.0.0.1:8141/v1")
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--temperatures", type=parse_temperatures, default=[0.7])
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--samples_per_request", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--request_timeout", type=float, default=900.0)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--near_px", type=float, default=50.0)
    parser.add_argument("--far_px", type=float, default=150.0)
    parser.add_argument("--image_max_pixels", type=int, default=None)
    parser.add_argument("--noncritical_max", type=int, default=0, help="0 means all non-critical failures; positive value samples that many")
    parser.add_argument("--max_critical", type=int, default=0, help="debug cap; 0 means all critical failures")
    parser.add_argument("--critical_only", action="store_true", help="sample only critical failures after target construction")
    parser.add_argument("--num_shards", type=int, default=1, help="split targets across this many sampler shards")
    parser.add_argument("--shard_index", type=int, default=0, help="0-based sampler shard index")
    parser.add_argument("--resume_from", nargs="*", default=[], help="additional per_step JSONL files whose target/temp keys should be skipped")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--store_pred_text_chars", type=int, default=500)
    parser.add_argument("--summarize_only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_step = output_dir / "per_step.jsonl"
    if args.summarize_only:
        summarize_only(args)
        return

    episodes = read_test_episodes(Path(args.test_data))
    if args.phase0_per_state:
        eval_paths: List[Path] = []
        targets = build_phase0_targets(
            episodes,
            read_jsonl(Path(args.phase0_per_state)),
            wrong_source=args.phase0_wrong_source,
            max_critical=args.max_critical,
        )
    else:
        eval_paths = [Path(path) for path in args.eval_results] if args.eval_results else discover_eval_paths(DEFAULT_EVAL_GLOBS)
        crit_tasks = read_crit_tasks(Path(args.crit_per_task))
        eval_results = read_eval_results(eval_paths)
        targets = build_targets(
            episodes,
            crit_tasks,
            eval_results,
            noncritical_max=args.noncritical_max,
            seed=args.seed,
            match_threshold=args.match_threshold,
            near_px=args.near_px,
            far_px=args.far_px,
            max_critical=args.max_critical,
        )
    if args.critical_only:
        targets = [target for target in targets if target.get("population") == "critical"]
    if args.num_shards < 1:
        raise SystemExit("--num_shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit("--shard_index must satisfy 0 <= shard_index < num_shards")
    if args.num_shards > 1:
        targets = [target for idx, target in enumerate(targets) if idx % args.num_shards == args.shard_index]
    target_counts = dict(Counter(target["population"] for target in targets))
    manifest = {
        "test_data": args.test_data,
        "crit_per_task": args.crit_per_task,
        "phase0_per_state": args.phase0_per_state,
        "phase0_wrong_source": args.phase0_wrong_source,
        "eval_results": [str(path) for path in eval_paths],
        "target_counts": target_counts,
        "temperatures": args.temperatures,
        "n_samples": args.n_samples,
        "samples_per_request": args.samples_per_request,
        "noncritical_max": args.noncritical_max,
        "max_critical": args.max_critical,
        "critical_only": args.critical_only,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "resume_from": args.resume_from,
        "model_name": args.model_name,
        "api_url": args.api_url,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    existing = done_keys_many([per_step] + [Path(path) for path in args.resume_from])
    work: List[Tuple[Dict[str, Any], float]] = []
    for target in targets:
        for temp in args.temperatures:
            key = (target["target_id"], float(temp))
            if key not in existing:
                work.append((target, float(temp)))
    print(f"targets={target_counts} temperatures={args.temperatures} remaining_jobs={len(work)}", flush=True)
    if work:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = {executor.submit(sample_target_temperature, target, temp, args): (target["target_id"], temp) for target, temp in work}
            completed = 0
            for future in as_completed(futures):
                target_key, temp = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001 - preserve failed job in JSONL
                    row = {
                        "target_id": target_key,
                        "temperature": temp,
                        "recoverable": False,
                        "n_samples": args.n_samples,
                        "sampling_error": str(exc)[:1000],
                    }
                append_jsonl(per_step, [row])
                completed += 1
                if completed % 25 == 0 or completed == len(work):
                    print(f"completed {completed}/{len(work)} jobs", flush=True)
    summarize_only(args, target_counts)


if __name__ == "__main__":
    main()