#!/usr/bin/env python3
"""Sample all GUI-360 TEST steps for teacher-forced verifier E2E TSR.

The output is a verifier-ready per-step pool. Candidate 0 is the canonical
GT-history greedy prediction from the frozen 22.20% baseline; candidates 1..N-1
are sampled from the same base model/prompt under GT history.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.critstep_verifier_full_action import (  # noqa: E402
    action_category,
    action_signature,
    candidate_from_payload,
    history_for_step,
    normalize_action_type,
)
from scripts.critstep_reward_structure_uia import controls_for_step  # noqa: E402
from v13_gui_360.eval_gui360_template import build_step_prompt, parse_tool_call  # noqa: E402
from v13_gui_360.reward import compute_step_reward  # noqa: E402


DEFAULT_TEST_DATA = "outputs/gui360_history_ab/original_eval/gui360_test_1000_balanced_uia.jsonl"
DEFAULT_BASELINE_SUMMARY = "outputs/gui360_history_ab/original_sft_template_gt_history_merged_20260630/summary.json"
DEFAULT_MODEL = "checkpoints/gui360-fullparam-sft-step250"
DEFAULT_OUTPUT_DIR = "outputs/verifier_e2e/slice200/candidates"
N_SWEEP = (5, 10, 20, 50)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_csv_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_api_urls(value: str) -> List[str]:
    urls = [item.strip().rstrip("/") for item in value.split(",") if item.strip()]
    return urls or ["http://127.0.0.1:8141/v1"]


def load_baseline_results(summary_path: Path, result_paths: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    paths: List[Path]
    if result_paths:
        paths = [Path(path) for path in result_paths]
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        paths = [Path(item["results"]) for item in summary.get("shards", []) if item.get("results")]
    if not paths:
        raise ValueError("no baseline eval result paths were provided or found in --baseline-summary")
    out: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, value in data.items():
            episode_id = str(value.get("episode_id", key))
            if episode_id in out:
                raise ValueError(f"duplicate baseline episode {episode_id} in {path}")
            out[episode_id] = value
    return out


def parse_action(pred_text: str) -> Optional[Dict[str, Any]]:
    action = parse_tool_call(pred_text)
    if action is not None:
        return action
    match = re.search(r"<action>\s*(\{.*?\})\s*</action>", pred_text, flags=re.DOTALL)
    if match:
        try:
            value = json.loads(match.group(1))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def score_prediction(pred_text: str, gt_action: Mapping[str, Any], image_w: int, image_h: int, match_threshold: float, store_chars: int) -> Dict[str, Any]:
    pred_action = parse_action(pred_text)
    fake_text = f"<action>{json.dumps(pred_action, ensure_ascii=False)}</action>" if pred_action else pred_text
    reward, info = compute_step_reward(fake_text, dict(gt_action), image_w, image_h)
    pred_type = info.get("pred_type") or (pred_action.get("action") if isinstance(pred_action, dict) else None)
    return {
        "success": bool(float(reward) >= match_threshold),
        "reward": float(reward),
        "bucket": None,
        "pred_action": info.get("pred_action") or pred_action,
        "pred_type": normalize_action_type(pred_type),
        "gt_type": normalize_action_type(info.get("gt_type") or gt_action.get("action")),
        "format_reward": info.get("format_reward", 0.0),
        "type_reward": info.get("type_reward", 0.0),
        "content_reward": info.get("content_reward", 0.0),
        "pred_text": pred_text[:store_chars],
    }


def baseline_payload(baseline_step: Mapping[str, Any], gt_action: Mapping[str, Any], image_w: int, image_h: int, match_threshold: float, store_chars: int) -> Dict[str, Any]:
    pred_action = baseline_step.get("pred_action") if isinstance(baseline_step.get("pred_action"), dict) else None
    pred_text = str(baseline_step.get("pred_text") or "")
    if pred_action is not None:
        fake_text = f"<action>{json.dumps(pred_action, ensure_ascii=False)}</action>"
        reward, info = compute_step_reward(fake_text, dict(gt_action), image_w, image_h)
        return {
            "success": bool(float(reward) >= match_threshold),
            "reward": float(reward),
            "bucket": None,
            "pred_action": info.get("pred_action") or pred_action,
            "pred_type": normalize_action_type(info.get("pred_type") or baseline_step.get("pred_type") or pred_action.get("action")),
            "gt_type": normalize_action_type(info.get("gt_type") or baseline_step.get("gt_type") or gt_action.get("action")),
            "format_reward": info.get("format_reward", baseline_step.get("format_reward", 0.0)),
            "type_reward": info.get("type_reward", baseline_step.get("type_reward", 0.0)),
            "content_reward": info.get("content_reward", baseline_step.get("content_reward", 0.0)),
            "pred_text": pred_text[:store_chars],
        }
    return score_prediction(pred_text, gt_action, image_w, image_h, match_threshold, store_chars)


def choice_logprob_stats(choice: Any) -> Dict[str, Any]:
    logprobs = getattr(choice, "logprobs", None)
    content = getattr(logprobs, "content", None) if logprobs is not None else None
    if not content:
        return {"model_logprob_sum": None, "model_logprob_avg": None, "model_logprob_tokens": 0}
    values = []
    for item in content:
        value = getattr(item, "logprob", None)
        if value is not None:
            values.append(float(value))
    total = sum(values)
    return {
        "model_logprob_sum": total if values else None,
        "model_logprob_avg": (total / len(values)) if values else None,
        "model_logprob_tokens": len(values),
    }


def sample_batch(client: OpenAI, args: argparse.Namespace, messages: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    kwargs: Dict[str, Any] = {
        "model": args.model_name,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.sample_temperature,
        "top_p": args.top_p,
        "n": n,
    }
    if args.collect_logprobs:
        kwargs.update({"logprobs": True, "top_logprobs": 1})
    response = client.chat.completions.create(**kwargs)
    out = []
    for choice in response.choices:
        text = choice.message.content or ""
        item = {"text": text}
        item.update(choice_logprob_stats(choice))
        out.append(item)
    return out


def build_candidate(candidate_id: str, source: str, payload: Mapping[str, Any], step: Mapping[str, Any], controls: Sequence[Dict[str, Any]], sample_rank: int) -> Dict[str, Any]:
    candidate = candidate_from_payload(candidate_id=candidate_id, source=source, payload=payload, step=step, controls=controls)
    candidate["sample_rank"] = sample_rank
    candidate["model_logprob_sum"] = payload.get("model_logprob_sum")
    candidate["model_logprob_avg"] = payload.get("model_logprob_avg")
    candidate["model_logprob_tokens"] = payload.get("model_logprob_tokens")
    candidate["temperature"] = payload.get("temperature")
    return candidate


def build_record(
    *,
    episode: Mapping[str, Any],
    episode_order: int,
    step_idx: int,
    greedy_payload: Mapping[str, Any],
    sample_payloads: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    api_url: Optional[str],
    api_errors: Sequence[str],
) -> Dict[str, Any]:
    steps = episode.get("steps") if isinstance(episode.get("steps"), list) else []
    step = steps[step_idx]
    controls = controls_for_step(step)
    candidates = [build_candidate("greedy", "greedy", greedy_payload, step, controls, 0)]
    for sample_idx, payload in enumerate(sample_payloads, 1):
        candidates.append(build_candidate(f"sample_{sample_idx:03d}", "sample", payload, step, controls, sample_idx))
    target_id = f"test:{episode.get('episode_id')}:{step_idx}"
    correct_ids = [candidate["candidate_id"] for candidate in candidates if candidate.get("is_correct")]
    return {
        "target_id": target_id,
        "split": "test",
        "episode_id": str(episode.get("episode_id")),
        "episode_key": f"test:{episode.get('episode_id')}",
        "episode_order": int(episode_order),
        "step_idx": int(step_idx),
        "instruction": episode.get("goal", ""),
        "goal": episode.get("goal", ""),
        "screenshot": step.get("screenshot"),
        "history": history_for_step(steps, step_idx),
        "gt_action": step.get("action") if isinstance(step.get("action"), dict) else {},
        "gt_action_type": normalize_action_type((step.get("action") or {}).get("action") if isinstance(step.get("action"), dict) else None),
        "gt_action_category": action_category((step.get("action") or {}).get("action") if isinstance(step.get("action"), dict) else None),
        "image_w": int(step.get("image_w") or 1040),
        "image_h": int(step.get("image_h") or 736),
        "n_candidates": len(candidates),
        "n_requested_candidates": int(args.n_candidates),
        "sample_temperature": float(args.sample_temperature),
        "samples_per_request": int(args.samples_per_request),
        "greedy_correct": bool(candidates[0].get("is_correct")),
        "greedy_reward": float(candidates[0].get("reward") or 0.0),
        "n_correct_candidates": len(correct_ids),
        "correct_candidate_ids": correct_ids,
        "api_url": api_url,
        "api_error_count": len(api_errors),
        "api_errors": list(api_errors)[:10],
        "candidates": candidates,
    }


def done_target_ids(paths: Sequence[Path]) -> set[str]:
    done: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                target_id = row.get("target_id")
                if target_id:
                    done.add(str(target_id))
    return done


def build_work_items(episodes: Sequence[Mapping[str, Any]], baseline: Mapping[str, Any], args: argparse.Namespace) -> List[Tuple[int, Mapping[str, Any], int]]:
    selected = list(enumerate(episodes))
    if args.start > 0 or args.end is not None:
        selected = selected[args.start : args.end]
    if args.max_episodes > 0:
        selected = selected[: args.max_episodes]
    work: List[Tuple[int, Mapping[str, Any], int]] = []
    for episode_order, episode in selected:
        episode_id = str(episode.get("episode_id"))
        base_episode = baseline.get(episode_id)
        if base_episode is None:
            raise ValueError(f"missing baseline result for episode {episode_id}")
        steps = episode.get("steps") if isinstance(episode.get("steps"), list) else []
        base_steps = base_episode.get("steps") if isinstance(base_episode.get("steps"), list) else []
        if len(base_steps) < len(steps):
            raise ValueError(f"baseline result for episode {episode_id} has {len(base_steps)} steps, expected {len(steps)}")
        for step_idx in range(len(steps)):
            work.append((episode_order, episode, step_idx))
    if args.num_shards > 1:
        work = [item for idx, item in enumerate(work) if idx % args.num_shards == args.shard_index]
    return work


def sample_work_item(item: Tuple[int, Mapping[str, Any], int], job_index: int, baseline: Mapping[str, Any], api_urls: Sequence[str], args: argparse.Namespace) -> Dict[str, Any]:
    episode_order, episode, step_idx = item
    episode_id = str(episode.get("episode_id"))
    steps = episode.get("steps") if isinstance(episode.get("steps"), list) else []
    step = steps[step_idx]
    gt_action = step.get("action") if isinstance(step.get("action"), dict) else {}
    image_w = int(step.get("image_w") or 1040)
    image_h = int(step.get("image_h") or 736)
    baseline_step = baseline[episode_id]["steps"][step_idx]
    greedy = baseline_payload(baseline_step, gt_action, image_w, image_h, args.match_threshold, args.store_pred_text_chars)
    greedy["temperature"] = 0.0
    greedy["model_logprob_sum"] = None
    greedy["model_logprob_avg"] = None
    greedy["model_logprob_tokens"] = 0

    remaining = max(0, int(args.n_candidates) - 1)
    sample_payloads: List[Dict[str, Any]] = []
    api_errors: List[str] = []
    api_url = api_urls[job_index % len(api_urls)] if remaining > 0 else None
    if remaining > 0:
        client = OpenAI(base_url=api_url, api_key="dummy", timeout=args.request_timeout)
        messages = build_step_prompt(
            str(episode.get("goal") or ""),
            str(step.get("screenshot")),
            int(step_idx),
            history_for_step(steps, step_idx),
            image_max_pixels=args.image_max_pixels,
        )
        while remaining > 0:
            take = min(int(args.samples_per_request), remaining)
            batch: List[Dict[str, Any]] = []
            for attempt in range(args.max_retries + 1):
                try:
                    batch = sample_batch(client, args, messages, take)
                    break
                except Exception as exc:  # noqa: BLE001 - preserve API errors in artifact
                    api_errors.append(str(exc)[:400])
                    if attempt >= args.max_retries:
                        batch = [{"text": "", "model_logprob_sum": None, "model_logprob_avg": None, "model_logprob_tokens": 0} for _ in range(take)]
                        break
                    time.sleep(min(30.0, 1.5 ** attempt))
            for sampled in batch:
                scored = score_prediction(str(sampled.get("text") or ""), gt_action, image_w, image_h, args.match_threshold, args.store_pred_text_chars)
                scored["temperature"] = float(args.sample_temperature)
                scored["model_logprob_sum"] = sampled.get("model_logprob_sum")
                scored["model_logprob_avg"] = sampled.get("model_logprob_avg")
                scored["model_logprob_tokens"] = sampled.get("model_logprob_tokens")
                sample_payloads.append(scored)
            remaining -= take
    return build_record(
        episode=episode,
        episode_order=episode_order,
        step_idx=step_idx,
        greedy_payload=greedy,
        sample_payloads=sample_payloads,
        args=args,
        api_url=api_url,
        api_errors=api_errors,
    )


def summarize_rows(rows: Sequence[Mapping[str, Any]], output_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    episodes = defaultdict(list)
    total_steps = 0
    greedy_correct = 0
    oracle_by_n = {n: 0 for n in N_SWEEP if n <= args.n_candidates}
    for row in rows:
        episodes[str(row.get("episode_id"))].append(row)
        total_steps += 1
        greedy_correct += int(bool(row.get("greedy_correct")))
        candidates = row.get("candidates") if isinstance(row.get("candidates"), list) else []
        for n in oracle_by_n:
            oracle_by_n[n] += int(any(bool(candidate.get("is_correct")) for candidate in candidates[:n]))
    episode_success = 0
    for episode_rows in episodes.values():
        if all(bool(row.get("greedy_correct")) for row in episode_rows):
            episode_success += 1
    summary = {
        "output_dir": str(output_dir),
        "rows": len(rows),
        "episodes": len(episodes),
        "n_candidates": int(args.n_candidates),
        "sample_temperature": float(args.sample_temperature),
        "greedy_tsr_from_pool": episode_success / len(episodes) if episodes else 0.0,
        "greedy_stepsr_from_pool": greedy_correct / total_steps if total_steps else 0.0,
        "oracle_step_recoverable_by_n": {str(n): oracle_by_n[n] / total_steps if total_steps else 0.0 for n in oracle_by_n},
        "api_error_count": sum(int(row.get("api_error_count") or 0) for row in rows),
        "candidate_count_histogram": dict(Counter(int(row.get("n_candidates") or 0) for row in rows)),
        "logprob_candidate_count": sum(
            1
            for row in rows
            for candidate in (row.get("candidates") if isinstance(row.get("candidates"), list) else [])
            if candidate.get("model_logprob_avg") is not None
        ),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def summarize_only(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    per_step = output_dir / "per_step.jsonl"
    rows = read_jsonl(per_step) if per_step.exists() else []
    summary = summarize_rows(rows, output_dir, args)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-data", default=DEFAULT_TEST_DATA)
    parser.add_argument("--baseline-summary", default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--baseline-results", nargs="*", default=[])
    parser.add_argument("--api-url", default="http://127.0.0.1:8141/v1")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-candidates", type=int, default=50)
    parser.add_argument("--samples-per-request", type=int, default=5)
    parser.add_argument("--sample-temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--match-threshold", type=float, default=0.5)
    parser.add_argument("--image-max-pixels", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=200)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume-from", nargs="*", default=[])
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--store-pred-text-chars", type=int, default=700)
    parser.add_argument("--collect-logprobs", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    if args.n_candidates < 1:
        raise ValueError("--n-candidates must be >= 1")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_step = output_dir / "per_step.jsonl"
    if args.num_shards > 1:
        per_step = output_dir / f"per_step.shard_{args.shard_index:02d}_of_{args.num_shards:02d}.jsonl"
    if args.summarize_only:
        summarize_only(args)
        return

    episodes = read_jsonl(Path(args.test_data))
    baseline = load_baseline_results(Path(args.baseline_summary), args.baseline_results)
    work = build_work_items(episodes, baseline, args)
    existing = done_target_ids([per_step] + [Path(path) for path in args.resume_from])
    work = [item for item in work if f"test:{item[1].get('episode_id')}:{item[2]}" not in existing]
    api_urls = parse_api_urls(args.api_url)
    manifest = {
        "test_data": args.test_data,
        "baseline_summary": args.baseline_summary,
        "baseline_results": args.baseline_results,
        "model_name": args.model_name,
        "api_urls": api_urls,
        "n_candidates_total_including_greedy": args.n_candidates,
        "sample_temperature": args.sample_temperature,
        "collect_logprobs": args.collect_logprobs,
        "start": args.start,
        "end": args.end,
        "max_episodes": args.max_episodes,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "remaining_steps": len(work),
        "protocol": "candidate 0 is canonical GT-history greedy baseline; candidates 1..N-1 are stochastic samples under GT-history prompt",
    }
    write_json(output_dir / ("manifest.json" if args.num_shards == 1 else f"manifest.shard_{args.shard_index:02d}.json"), manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)

    lock = threading.Lock()
    completed = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.threads)) as executor:
        futures = {
            executor.submit(sample_work_item, item, job_index, baseline, api_urls, args): item
            for job_index, item in enumerate(work)
        }
        for future in as_completed(futures):
            row = future.result()
            append_jsonl(per_step, [row], lock)
            completed += 1
            if completed % 25 == 0 or completed == len(work):
                elapsed = max(1e-6, time.time() - started)
                rate = completed / elapsed
                print(f"completed {completed}/{len(work)} steps rate={rate:.3f}/s", flush=True)

    if args.num_shards == 1:
        rows = read_jsonl(per_step)
        summary = summarize_rows(rows, output_dir, args)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    else:
        print(json.dumps({"per_step": str(per_step), "completed": completed}, indent=2), flush=True)


if __name__ == "__main__":
    main()