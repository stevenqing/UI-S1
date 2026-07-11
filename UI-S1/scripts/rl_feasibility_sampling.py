#!/usr/bin/env python3
"""Sampling-only GRPO feasibility diagnostic for GUI-360 critical steps.

The script asks whether GRPO has a learning signal on critical steps: sampled
groups must contain positives and, more importantly, have non-zero binary reward
variance. It uses GUI-360's original prompt, the frozen matcher, and the same
sampling config as the reflex decisiveness run.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Optional, Sequence

import pyarrow.parquet as pq
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v13_gui_360.eval_gui360_template import (  # noqa: E402
    SUPPORTED_ACTIONS,
    USER_PROMPT_TEMPLATE,
    _format_action_for_history,
    parse_tool_call,
)
from v13_gui_360.reward import compute_step_reward, _normalize_action_type  # noqa: E402


@dataclass(frozen=True)
class StepJob:
    episode_id: str
    goal: str
    num_steps: int
    step_idx: int
    gt_action: dict[str, Any]
    screenshot_bytes: bytes
    history: tuple[str, ...]

    @property
    def target_id(self) -> str:
        return f"{self.episode_id}:{self.step_idx}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_jsonable(data), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sanitize_jsonable(value: Any) -> Any:
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(value, list):
        return [sanitize_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {sanitize_jsonable(key): sanitize_jsonable(item) for key, item in value.items()}
    return value


def append_jsonl(path: Path, row: Mapping[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sanitize_jsonable(row), ensure_ascii=False) + "\n")


def iter_parquet_files(data_dir: Path, split: str) -> list[Path]:
    files = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not files and split == "test":
        files = sorted(data_dir.glob("test-*.parquet"))
    return files


def load_jobs(data_dir: Path, split: str, max_episodes: Optional[int], start_episode: int = 0) -> list[StepJob]:
    jobs: list[StepJob] = []
    seen_episodes = 0
    for path in iter_parquet_files(data_dir, split):
        table = pq.read_table(path, columns=["episode_id", "goal", "num_steps", "steps", "screenshots"])
        for episode in table.to_pylist():
            if seen_episodes < start_episode:
                seen_episodes += 1
                continue
            if max_episodes is not None and (seen_episodes - start_episode) >= max_episodes:
                return jobs
            steps = json.loads(episode["steps"]) if isinstance(episode["steps"], str) else episode["steps"]
            screenshots = episode["screenshots"]
            history: list[str] = []
            for index, step in enumerate(steps):
                screenshot_item = screenshots[index]
                screenshot_bytes = screenshot_item.get("bytes") if isinstance(screenshot_item, Mapping) else screenshot_item
                gt_action = step.get("action") if isinstance(step, Mapping) else None
                if not isinstance(gt_action, dict) or not isinstance(screenshot_bytes, (bytes, bytearray)):
                    continue
                jobs.append(StepJob(
                    episode_id=str(episode["episode_id"]),
                    goal=str(episode["goal"]),
                    num_steps=int(episode.get("num_steps") or len(steps)),
                    step_idx=int(step.get("step_idx", index)),
                    gt_action=gt_action,
                    screenshot_bytes=bytes(screenshot_bytes),
                    history=tuple(history),
                ))
                history.append(_format_action_for_history(gt_action, index + 1))
            seen_episodes += 1
    return jobs


def screenshot_to_b64(screenshot_bytes: bytes, image_max_pixels: Optional[int]) -> tuple[str, int, int]:
    image = Image.open(BytesIO(screenshot_bytes)).convert("RGB")
    original_w, original_h = image.size
    if image_max_pixels:
        pixels = original_w * original_h
        if pixels > image_max_pixels:
            scale = math.sqrt(image_max_pixels / pixels)
            image = image.resize((max(1, int(original_w * scale)), max(1, int(original_h * scale))), Image.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8"), original_w, original_h


def build_messages(job: StepJob, image_max_pixels: Optional[int]) -> tuple[list[dict[str, Any]], int, int]:
    history_text = "\n".join(job.history) if job.history else "None"
    prompt_text = USER_PROMPT_TEMPLATE.format(
        instruction=job.goal,
        history=history_text,
        actions=SUPPORTED_ACTIONS,
    )
    b64, image_w, image_h = screenshot_to_b64(job.screenshot_bytes, image_max_pixels)
    return ([{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt_text},
        ],
    }], image_w, image_h)


def fake_action_text(action: Optional[Mapping[str, Any]], fallback: str) -> str:
    if action:
        return f"<action>{json.dumps(action, ensure_ascii=False)}</action>"
    return fallback or ""


def rounded_coord(coord: Any, bucket: int) -> Optional[list[int]]:
    if not isinstance(coord, (list, tuple)) or len(coord) < 2:
        return None
    try:
        return [int(round(float(coord[0]) / bucket) * bucket), int(round(float(coord[1]) / bucket) * bucket)]
    except (TypeError, ValueError):
        return None


def action_key(action: Optional[Mapping[str, Any]], coord_bucket: int) -> str:
    if not isinstance(action, Mapping):
        return "__unparsed__"
    atype = _normalize_action_type(str(action.get("action", "")))
    payload: dict[str, Any] = {"type": atype}
    if atype in {"click", "long_press"}:
        payload["coord"] = rounded_coord(action.get("coordinate"), coord_bucket)
    elif atype == "swipe":
        payload["start"] = rounded_coord(action.get("startCoordinate") or action.get("coordinate"), coord_bucket)
        payload["end"] = rounded_coord(action.get("endCoordinate"), coord_bucket)
    elif atype in {"type", "open", "answer", "key"}:
        payload["text"] = str(action.get("text", "")).strip().lower()[:160]
        coord = rounded_coord(action.get("coordinate"), coord_bucket)
        if coord:
            payload["coord"] = coord
    elif atype == "system_button":
        payload["button"] = str(action.get("button", "")).strip().lower()
    else:
        payload["raw"] = json.dumps(action, sort_keys=True, ensure_ascii=False)[:240]
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def parse_any_action(text: str) -> Optional[dict[str, Any]]:
    return parse_tool_call(text)


def score_text(text: str, gt_action: Mapping[str, Any], image_w: int, image_h: int, match_threshold: float) -> dict[str, Any]:
    parsed = parse_any_action(text)
    reward, info = compute_step_reward(fake_action_text(parsed, text), dict(gt_action), image_w=image_w, image_h=image_h)
    pred_action = info.get("pred_action")
    return {
        "raw_output": text[:1000],
        "pred_action": pred_action,
        "reward": reward,
        "correct": reward >= match_threshold,
        "pred_type": info.get("pred_type"),
        "gt_type": info.get("gt_type"),
    }


def request_with_retries(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    n: int,
    retries: int,
) -> list[str]:
    last_error: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
            )
            return [choice.message.content or "" for choice in response.choices]
        except BaseException as exc:  # noqa: BLE001 - keep long jobs resilient
            last_error = exc
            time.sleep(min(2.0 * (attempt + 1), 10.0))
    raise RuntimeError(f"OpenAI-compatible request failed after retries: {last_error}")


def eval_probe_step(job: StepJob, client: OpenAI, args: argparse.Namespace) -> dict[str, Any]:
    messages, image_w, image_h = build_messages(job, args.image_max_pixels)
    greedy_text = request_with_retries(
        client,
        model=args.model_name,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
        n=1,
        retries=args.retries,
    )[0]
    greedy = score_text(greedy_text, job.gt_action, image_w, image_h, args.match_threshold)
    sample_texts = request_with_retries(
        client,
        model=args.model_name,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.sample_temperature,
        top_p=args.top_p,
        n=args.initial_sample_n,
        retries=args.retries,
    )
    samples = [score_text(text, job.gt_action, image_w, image_h, args.match_threshold) for text in sample_texts]
    candidate_actions = [greedy.get("pred_action")] + [sample.get("pred_action") for sample in samples]
    candidate_correct = [bool(greedy.get("correct"))] + [bool(sample.get("correct")) for sample in samples]
    keys = [action_key(action, args.coord_bucket) for action in candidate_actions]
    key_counts = Counter(keys)
    greedy_key = keys[0]
    total_candidates = max(1, len(keys))
    return {
        "target_id": job.target_id,
        "episode_id": job.episode_id,
        "step_idx": job.step_idx,
        "num_steps": job.num_steps,
        "gt_action": job.gt_action,
        "gt_type": greedy.get("gt_type"),
        "image_w": image_w,
        "image_h": image_h,
        "greedy_correct": greedy.get("correct"),
        "greedy_reward": greedy.get("reward"),
        "greedy_pred_action": greedy.get("pred_action"),
        "greedy_key": greedy_key,
        "candidate_key_counts": dict(key_counts),
        "candidate_total": total_candidates,
        "p_i_initial": sum(candidate_correct) / total_candidates,
        "sample_correct_rate_initial": sum(bool(sample.get("correct")) for sample in samples) / max(1, len(samples)),
        "greedy_decode_share": key_counts[greedy_key] / total_candidates,
        "modal_decode_frac": max(key_counts.values()) / total_candidates,
        "initial_samples": compact_samples(samples, args.coord_bucket),
    }


def eval_extra_samples(job: StepJob, client: OpenAI, args: argparse.Namespace, n: int) -> tuple[list[dict[str, Any]], int, int]:
    if n <= 0:
        return [], 0, 0
    messages, image_w, image_h = build_messages(job, args.image_max_pixels)
    texts = request_with_retries(
        client,
        model=args.model_name,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.sample_temperature,
        top_p=args.top_p,
        n=n,
        retries=args.retries,
    )
    samples = [score_text(text, job.gt_action, image_w, image_h, args.match_threshold) for text in texts]
    return samples, image_w, image_h


def compact_samples(samples: Sequence[Mapping[str, Any]], coord_bucket: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        pred_action = sample.get("pred_action")
        rows.append({
            "reward": sample.get("reward"),
            "correct": bool(sample.get("correct")),
            "pred_type": sample.get("pred_type"),
            "pred_action": pred_action,
            "action_key": action_key(pred_action, coord_bucket),
            "raw_output": sample.get("raw_output", ""),
        })
    return rows


def by_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("target_id")): dict(row) for row in rows if row.get("target_id") is not None}


def resolve_api_urls(args: argparse.Namespace) -> list[str]:
    urls = list(getattr(args, "api_urls", None) or [])
    if not urls:
        urls = [args.api_url]
    return urls


def run_all_step_probe(jobs: list[StepJob], args: argparse.Namespace, output_path: Path) -> list[dict[str, Any]]:
    existing = by_id(read_jsonl(output_path))
    pending = [job for job in jobs if job.target_id not in existing]
    api_urls = resolve_api_urls(args)
    print(json.dumps({"stage": "all_step_probe", "jobs": len(jobs), "existing": len(existing), "pending": len(pending), "api_urls": api_urls}, indent=2), flush=True)
    if pending and not args.recompute_only:
        clients = [OpenAI(base_url=url, api_key="dummy", timeout=args.request_timeout) for url in api_urls]
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futures = [pool.submit(eval_probe_step, job, clients[index % len(clients)], args) for index, job in enumerate(pending)]
            for index, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="all-step"), start=1):
                row = future.result()
                append_jsonl(output_path, row, lock)
                if index % 100 == 0:
                    print(f"all-step completed new {index}/{len(pending)}", flush=True)
    rows = read_jsonl(output_path)
    expected_ids = {job.target_id for job in jobs}
    filtered = [row for row in rows if str(row.get("target_id")) in expected_ids]
    have = {str(row.get("target_id")) for row in filtered}
    if len(have) < len(expected_ids):
        raise RuntimeError(f"all-step probe incomplete: missing {len(expected_ids) - len(have)} rows")
    return filtered


def select_critical_steps(rows: Sequence[Mapping[str, Any]], max_k: int, seed: int) -> tuple[set[str], dict[str, list[str]], dict[str, list[dict[str, Any]]]]:
    del seed  # deterministic rankings; kept for CLI/report compatibility
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[str(row["episode_id"])].append(dict(row))
    for episode_rows in by_episode.values():
        episode_rows.sort(key=lambda row: int(row["step_idx"]))

    critical_ids: set[str] = set()
    sources: dict[str, list[str]] = defaultdict(list)
    selected_rows: dict[str, list[dict[str, Any]]] = {"S_REFLEX_WRONG": [], "S_DIFFICULTY": []}

    for episode_rows in by_episode.values():
        wrong_rows = [row for row in episode_rows if not bool(row.get("greedy_correct"))]
        budget = min(max_k, len(wrong_rows))
        if budget <= 0:
            continue
        reflex_ranked = sorted(
            wrong_rows,
            key=lambda row: (float(row.get("greedy_decode_share") or 0.0), float(row.get("modal_decode_frac") or 0.0), -int(row["step_idx"])),
            reverse=True,
        )[:budget]
        difficulty_ranked = sorted(
            episode_rows,
            key=lambda row: (float(row.get("p_i_initial") if row.get("p_i_initial") is not None else 1.0), -float(row.get("greedy_decode_share") or 0.0), int(row["step_idx"])),
        )[:budget]
        for source, picked in (("S_REFLEX_WRONG", reflex_ranked), ("S_DIFFICULTY", difficulty_ranked)):
            for row in picked:
                target_id = str(row["target_id"])
                critical_ids.add(target_id)
                if source not in sources[target_id]:
                    sources[target_id].append(source)
                selected_rows[source].append(row)
    return critical_ids, sources, selected_rows


def group_metrics(samples: Sequence[Mapping[str, Any]], group_sizes: Sequence[int]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for group_size in group_sizes:
        group = list(samples[:group_size])
        if len(group) < group_size:
            metrics[str(group_size)] = {"available": False, "sample_count": len(group)}
            continue
        correct_values = [1.0 if bool(sample.get("correct")) else 0.0 for sample in group]
        rewards = [float(sample.get("reward") or 0.0) for sample in group]
        correct_count = int(sum(correct_values))
        dense_mean = sum(rewards) / max(1, len(rewards))
        binary_mean = sum(correct_values) / max(1, len(correct_values))
        dense_var = sum((value - dense_mean) ** 2 for value in rewards) / max(1, len(rewards))
        binary_var = sum((value - binary_mean) ** 2 for value in correct_values) / max(1, len(correct_values))
        metrics[str(group_size)] = {
            "available": True,
            "sample_count": len(group),
            "correct_count": correct_count,
            "positive_present": correct_count >= 1,
            "all_wrong": correct_count == 0,
            "all_correct": correct_count == group_size,
            "binary_reward_variance": binary_var,
            "nonzero_binary_variance": binary_var > 1e-12,
            "dense_reward_variance": dense_var,
            "nonzero_dense_variance": dense_var > 1e-12,
            "mean_dense_reward": dense_mean,
        }
    return metrics


def split_reflex_strata(per_step_rows: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, Any]]:
    ranked = sorted(per_step_rows, key=lambda row: (float(row.get("greedy_decode_share") or 0.0), str(row["target_id"])))
    midpoint = len(ranked) // 2
    strata: dict[str, str] = {}
    for index, row in enumerate(ranked):
        strata[str(row["target_id"])] = "weak_reflex" if index < midpoint else "strong_reflex"
    strong_values = [float(row.get("greedy_decode_share") or 0.0) for row in ranked[midpoint:]]
    weak_values = [float(row.get("greedy_decode_share") or 0.0) for row in ranked[:midpoint]]
    info = {
        "method": "rank_median_by_greedy_decode_share",
        "weak_count": len(weak_values),
        "strong_count": len(strong_values),
        "weak_range": [min(weak_values), max(weak_values)] if weak_values else None,
        "strong_range": [min(strong_values), max(strong_values)] if strong_values else None,
    }
    return strata, info


def build_critical_row(job: StepJob, probe_row: Mapping[str, Any], source: list[str], samples: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    metrics = group_metrics(samples, args.group_sizes)
    return {
        "target_id": job.target_id,
        "episode_id": job.episode_id,
        "step_idx": job.step_idx,
        "num_steps": job.num_steps,
        "critical_source": source,
        "gt_action": job.gt_action,
        "gt_type": probe_row.get("gt_type"),
        "image_w": probe_row.get("image_w"),
        "image_h": probe_row.get("image_h"),
        "greedy_correct": probe_row.get("greedy_correct"),
        "greedy_reward": probe_row.get("greedy_reward"),
        "greedy_pred_action": probe_row.get("greedy_pred_action"),
        "greedy_decode_share": probe_row.get("greedy_decode_share"),
        "modal_decode_frac": probe_row.get("modal_decode_frac"),
        "p_i_initial": probe_row.get("p_i_initial"),
        "sample_count": len(samples),
        "sample_correct_count": sum(1 for sample in samples if bool(sample.get("correct"))),
        "sample_correct_rate": sum(1 for sample in samples if bool(sample.get("correct"))) / max(1, len(samples)),
        "samples": samples,
        "group_metrics": metrics,
    }


def run_critical_sampling(
    jobs: list[StepJob],
    probe_rows: list[dict[str, Any]],
    critical_ids: set[str],
    sources: Mapping[str, list[str]],
    args: argparse.Namespace,
    output_path: Path,
) -> list[dict[str, Any]]:
    jobs_by_id = {job.target_id: job for job in jobs}
    probe_by_id = by_id(probe_rows)
    existing = by_id(read_jsonl(output_path))
    max_group_size = max(args.group_sizes)
    pending_ids = [target_id for target_id in sorted(critical_ids) if len(existing.get(target_id, {}).get("samples", [])) < max_group_size]
    api_urls = resolve_api_urls(args)
    print(json.dumps({"stage": "critical_sampling", "critical": len(critical_ids), "existing": len(existing), "pending": len(pending_ids), "max_group_size": max_group_size, "api_urls": api_urls}, indent=2), flush=True)

    if pending_ids and not args.recompute_only:
        clients = [OpenAI(base_url=url, api_key="dummy", timeout=args.request_timeout) for url in api_urls]
        lock = threading.Lock()

        def worker(target_id: str, client: OpenAI) -> dict[str, Any]:
            job = jobs_by_id[target_id]
            probe_row = probe_by_id[target_id]
            initial_samples = list(probe_row.get("initial_samples") or [])
            if target_id in existing:
                initial_samples = list(existing[target_id].get("samples") or initial_samples)
            missing = max_group_size - len(initial_samples)
            extra_samples: list[dict[str, Any]] = []
            if missing > 0:
                scored, _, _ = eval_extra_samples(job, client, args, missing)
                extra_samples = compact_samples(scored, args.coord_bucket)
            samples = (initial_samples + extra_samples)[:max_group_size]
            return build_critical_row(job, probe_row, list(sources.get(target_id, [])), samples, args)

        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futures = [pool.submit(worker, target_id, clients[index % len(clients)]) for index, target_id in enumerate(pending_ids)]
            for index, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="critical"), start=1):
                row = future.result()
                append_jsonl(output_path, row, lock)
                if index % 50 == 0:
                    print(f"critical completed new {index}/{len(pending_ids)}", flush=True)
    rows = [row for row in by_id(read_jsonl(output_path)).values() if str(row.get("target_id")) in critical_ids]
    have = {str(row.get("target_id")) for row in rows}
    if len(have) < len(critical_ids):
        raise RuntimeError(f"critical sampling incomplete: missing {len(critical_ids) - len(have)} rows")
    return rows


def summarize_group(rows: Sequence[Mapping[str, Any]], group_sizes: Sequence[int]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for group_size in group_sizes:
        key = str(group_size)
        available = [row for row in rows if row.get("group_metrics", {}).get(key, {}).get("available")]
        if not available:
            summary.append({"G": group_size, "n_steps": 0})
            continue
        metrics = [row["group_metrics"][key] for row in available]
        summary.append({
            "G": group_size,
            "n_steps": len(available),
            "positive_coverage": mean([1.0 if metric.get("positive_present") else 0.0 for metric in metrics]),
            "advantage_signal_fraction": mean([1.0 if metric.get("nonzero_binary_variance") else 0.0 for metric in metrics]),
            "all_wrong_fraction": mean([1.0 if metric.get("all_wrong") else 0.0 for metric in metrics]),
            "all_correct_fraction": mean([1.0 if metric.get("all_correct") else 0.0 for metric in metrics]),
            "dense_variance_fraction": mean([1.0 if metric.get("nonzero_dense_variance") else 0.0 for metric in metrics]),
            "mean_correct_count": mean([float(metric.get("correct_count") or 0.0) for metric in metrics]),
            "mean_dense_reward": mean([float(metric.get("mean_dense_reward") or 0.0) for metric in metrics]),
        })
    return summary


def summarize(per_step_rows: list[dict[str, Any]], group_sizes: Sequence[int]) -> dict[str, Any]:
    strata, strata_info = split_reflex_strata(per_step_rows)
    for row in per_step_rows:
        row["reflex_stratum"] = strata[str(row["target_id"])]
    by_stratum = {
        "weak_reflex": [row for row in per_step_rows if row.get("reflex_stratum") == "weak_reflex"],
        "strong_reflex": [row for row in per_step_rows if row.get("reflex_stratum") == "strong_reflex"],
    }
    by_source = {
        source: [row for row in per_step_rows if source in row.get("critical_source", [])]
        for source in ["S_REFLEX_WRONG", "S_DIFFICULTY"]
    }
    return {
        "num_critical_steps": len(per_step_rows),
        "group_sizes": list(group_sizes),
        "strata_info": strata_info,
        "overall": summarize_group(per_step_rows, group_sizes),
        "by_reflex_stratum": {key: summarize_group(value, group_sizes) for key, value in by_stratum.items()},
        "by_source": {key: summarize_group(value, group_sizes) for key, value in by_source.items()},
        "source_counts": {key: len(value) for key, value in by_source.items()},
    }


def pct(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * value:.2f}%"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---" for _ in headers]) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def rows_for_table(summary_rows: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    return [[
        row.get("G"),
        row.get("n_steps"),
        pct(row.get("positive_coverage")),
        pct(row.get("advantage_signal_fraction")),
        pct(row.get("all_wrong_fraction")),
        pct(row.get("all_correct_fraction")),
        f"{row.get('mean_correct_count', 0.0):.2f}",
        pct(row.get("dense_variance_fraction")),
    ] for row in summary_rows]


def gate(summary: Mapping[str, Any], feasible_max_g: int, adequate_threshold: float, starved_threshold: float) -> tuple[str, str, Optional[int]]:
    strong_rows = summary["by_reflex_stratum"].get("strong_reflex", [])
    viable_g: Optional[int] = None
    for row in strong_rows:
        group_size = int(row.get("G") or 0)
        if group_size <= feasible_max_g:
            coverage = float(row.get("positive_coverage") or 0.0)
            signal = float(row.get("advantage_signal_fraction") or 0.0)
            if coverage >= adequate_threshold and signal >= adequate_threshold:
                viable_g = group_size
                break
    if viable_g is not None:
        return (
            "GRPO VIABLE (signal present where needed)",
            f"Strong-reflex critical steps reach >= {pct(adequate_threshold)} positive coverage and advantage-signal fraction by G={viable_g}, within the feasible group-size budget.",
            viable_g,
        )
    max_row = max(strong_rows, key=lambda row: int(row.get("G") or 0), default={})
    max_coverage = float(max_row.get("positive_coverage") or 0.0)
    max_signal = float(max_row.get("advantage_signal_fraction") or 0.0)
    if max_coverage < starved_threshold or max_signal < starved_threshold:
        return (
            "POSITIVE-STARVED ON STRONG-REFLEX (needs injection/warmup)",
            f"Even at max G={max_row.get('G')}, strong-reflex coverage={pct(max_coverage)} and signal={pct(max_signal)}, below the starvation threshold {pct(starved_threshold)}.",
            None,
        )
    return (
        "MIXED",
        f"Strong-reflex steps have some signal at large G but do not reach {pct(adequate_threshold)} within G<={feasible_max_g}; use larger groups selectively or add positive injection/warmup for the hardest tail.",
        None,
    )


def render_report(summary: Mapping[str, Any], verdict: str, reason: str, needed_g: Optional[int], args: argparse.Namespace) -> str:
    strong = summary["by_reflex_stratum"].get("strong_reflex", [])
    weak = summary["by_reflex_stratum"].get("weak_reflex", [])
    lines = [
        "# RL Feasibility On Critical Steps",
        "",
        "No training was run. This is sampling statistics only, using GUI-360's original template and the frozen matcher.",
        "Primary GRPO signal is measured as non-zero variance of binary matcher correctness within a sampled group. Dense reward variance is reported as a supplement.",
        "Sampling config matches the reflex decisiveness run: temperature/top-p/max tokens/image resize are kept aligned unless shown below.",
        "",
        "## Run Setup",
        "",
        markdown_table(
            ["field", "value"],
            [
                ["model", args.model_name],
                ["api", ", ".join(resolve_api_urls(args))],
                ["data", str(args.data_dir)],
                ["critical max k", args.critical_k],
                ["critical steps", summary["num_critical_steps"]],
                ["group sizes", ", ".join(map(str, args.group_sizes))],
                ["initial probe samples", args.initial_sample_n],
                ["temperature", args.sample_temperature],
                ["top_p", args.top_p],
                ["reflex split", summary["strata_info"]["method"]],
                ["weak reflex count/range", f"{summary['strata_info']['weak_count']} / {summary['strata_info']['weak_range']}"],
                ["strong reflex count/range", f"{summary['strata_info']['strong_count']} / {summary['strata_info']['strong_range']}"],
            ],
        ),
        "",
        "## Metric 1 And 2 - Positive Coverage And Advantage Signal",
        "",
        markdown_table(
            ["G", "steps", ">=1 correct", "nonzero binary variance", "all wrong", "all correct", "mean correct", "dense variance"],
            rows_for_table(summary["overall"]),
        ),
        "",
        "## Metric 3 - Reflex-Strength Strata",
        "",
        "### Weak Reflex Critical Steps",
        "",
        markdown_table(
            ["G", "steps", ">=1 correct", "nonzero binary variance", "all wrong", "all correct", "mean correct", "dense variance"],
            rows_for_table(weak),
        ),
        "",
        "### Strong Reflex Critical Steps",
        "",
        markdown_table(
            ["G", "steps", ">=1 correct", "nonzero binary variance", "all wrong", "all correct", "mean correct", "dense variance"],
            rows_for_table(strong),
        ),
        "",
        "## Critical-Set Source Breakdown",
        "",
    ]
    for source, source_rows in summary["by_source"].items():
        lines.extend([
            f"### {source}",
            "",
            markdown_table(
                ["G", "steps", ">=1 correct", "nonzero binary variance", "all wrong", "all correct", "mean correct", "dense variance"],
                rows_for_table(source_rows),
            ),
            "",
        ])

    implication = (
        f"Pure GRPO with critical-step weighting is viable at G={needed_g}; proceed to RL design with critical-step weighting."
        if needed_g is not None
        else "Pure GRPO should be helped on the hardest critical steps: use larger groups selectively, inject known long-tail positives, or run an SFT warmup to raise positive probability before GRPO."
    )
    lines.extend([
        "## Metric 4 - Implication For RL Design",
        "",
        implication,
        "",
        "## Gate",
        "",
        verdict,
        "",
        reason,
        "",
        "STOP for review.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/gui360-balanced/data"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rl_feasibility"))
    parser.add_argument("--api-url", default="http://127.0.0.1:8177/v1")
    parser.add_argument("--api-urls", nargs="+", default=None)
    parser.add_argument("--model-name", default="gui360-sft")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--initial-sample-n", type=int, default=4)
    parser.add_argument("--sample-temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--image-max-pixels", type=int, default=602112)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--match-threshold", type=float, default=0.5)
    parser.add_argument("--coord-bucket", type=int, default=25)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--critical-k", type=int, default=3)
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--feasible-max-g", type=int, default=16)
    parser.add_argument("--adequate-threshold", type=float, default=0.5)
    parser.add_argument("--starved-threshold", type=float, default=0.25)
    parser.add_argument("--recompute-only", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_step_path = args.output_dir / "all_steps_probe.jsonl"
    per_step_path = args.output_dir / "per_step.jsonl"
    summary_path = args.output_dir / "summary.json"
    report_path = args.output_dir / "feasibility.md"

    jobs = load_jobs(args.data_dir, args.split, args.max_episodes, args.start_episode)
    if not jobs:
        raise RuntimeError(f"No jobs loaded from {args.data_dir} split={args.split}")
    probe_rows = run_all_step_probe(jobs, args, all_step_path)
    critical_ids, sources, selected_rows = select_critical_steps(probe_rows, args.critical_k, args.seed)
    selection_summary = {source: len(rows) for source, rows in selected_rows.items()}
    print(json.dumps({"stage": "critical_selection", "critical_union": len(critical_ids), "source_counts_with_duplicates": selection_summary}, indent=2), flush=True)
    critical_rows = run_critical_sampling(jobs, probe_rows, critical_ids, sources, args, per_step_path)
    summary = summarize(critical_rows, args.group_sizes)
    summary["selection"] = {
        "critical_k": args.critical_k,
        "critical_union": len(critical_ids),
        "source_counts_with_duplicates": selection_summary,
    }
    verdict, reason, needed_g = gate(summary, args.feasible_max_g, args.adequate_threshold, args.starved_threshold)
    summary["gate"] = verdict
    summary["gate_reason"] = reason
    summary["needed_group_size"] = needed_g
    write_json(summary_path, summary)
    report_path.write_text(render_report(summary, verdict, reason, needed_g, args), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "report": str(report_path), "per_step": str(per_step_path), "gate": verdict, "needed_group_size": needed_g}, indent=2), flush=True)


if __name__ == "__main__":
    main()