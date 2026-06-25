#!/usr/bin/env python3
"""Sample K offline action candidates on GT screens and score them with matcher.

This script is the offline-only exploration/distillation step. It does not need
online environment access: candidates are sampled on GT screenshots and scored
against the GT action using the existing GUI-360 matcher reward.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI
from tqdm import tqdm

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from v13_gui_360.eval_gui360_template import build_step_prompt, parse_tool_call  # noqa: E402
from v13_gui_360.reward import compute_step_reward  # noqa: E402
from v23_visual_transition.prepare_offline_data import format_action_for_history  # noqa: E402


def load_episode_jsonl(path: str) -> Dict[str, Dict[str, Any]]:
    episodes: Dict[str, Dict[str, Any]] = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                episode = json.loads(line)
                episodes[str(episode.get("episode_id"))] = episode
    return episodes


def load_jsonl(path: str, max_rows: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def build_gt_history(steps: List[Dict[str, Any]], step_idx: int) -> List[str]:
    history: List[str] = []
    for idx in range(step_idx):
        history.append(format_action_for_history(steps[idx].get("action", {}) or {}, idx + 1))
    return history


def score_text(pred_text: str, gt_action: Dict[str, Any], image_w: int, image_h: int) -> Dict[str, Any]:
    pred_action = parse_tool_call(pred_text)
    if pred_action is not None:
        reward_text = f"<action>{json.dumps(pred_action, ensure_ascii=False)}</action>"
    else:
        reward_text = pred_text
    reward, info = compute_step_reward(reward_text, gt_action, image_w, image_h)
    return {
        "text": pred_text,
        "reward": reward,
        "success": reward >= 0.5,
        "pred_action": info.get("pred_action"),
        "pred_type": info.get("pred_type"),
        "gt_type": info.get("gt_type"),
        "format_reward": info.get("format_reward", 0.0),
        "type_reward": info.get("type_reward", 0.0),
        "content_reward": info.get("content_reward", 0.0),
    }


def sample_one(
    row: Dict[str, Any],
    episodes: Dict[str, Dict[str, Any]],
    client: OpenAI,
    model_name: str,
    num_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    request_sleep: float,
) -> Dict[str, Any]:
    episode_id = str(row.get("episode_id"))
    episode = episodes[episode_id]
    steps = episode["steps"]
    step_idx = int(row.get("step_idx", 0))
    step = steps[step_idx]
    history = build_gt_history(steps, step_idx)
    messages = build_step_prompt(
        episode["goal"],
        step["screenshot"],
        step_idx,
        history,
        image_max_pixels=None,
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=num_samples,
    )
    if request_sleep > 0:
        time.sleep(request_sleep)

    candidates = []
    for choice in response.choices:
        pred_text = choice.message.content or ""
        candidates.append(score_text(
            pred_text,
            step.get("action", {}) or {},
            int(step.get("image_w", 1040)),
            int(step.get("image_h", 736)),
        ))

    candidates.sort(key=lambda item: item["reward"], reverse=True)
    best = candidates[0] if candidates else None
    return {
        "episode_id": episode_id,
        "step_idx": step_idx,
        "num_steps": len(steps),
        "goal": episode.get("goal", ""),
        "screenshot": step.get("screenshot"),
        "gt_action": step.get("action"),
        "hard_state": row,
        "num_samples": len(candidates),
        "any_success": any(candidate["success"] for candidate in candidates),
        "best_reward": best["reward"] if best else 0.0,
        "best_candidate": best,
        "candidates": candidates,
    }


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def row_key(row: Dict[str, Any]) -> str:
    return f"{row.get('episode_id')}:{row.get('step_idx')}"


def load_existing_outputs(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    return load_jsonl(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample K matcher-scored candidates for hard states")
    parser.add_argument("--hard_states", required=True)
    parser.add_argument("--episode_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--api_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--request_timeout", type=float, default=300.0)
    parser.add_argument("--request_sleep", type=float, default=0.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "matcher_candidates.jsonl")
    err_path = os.path.join(args.output_dir, "errors.jsonl")
    hard_rows = load_jsonl(args.hard_states, args.max_rows)
    episodes = load_episode_jsonl(args.episode_data)
    client = OpenAI(base_url=args.api_url, api_key="dummy", timeout=args.request_timeout)

    existing_outputs = load_existing_outputs(out_path) if args.resume else []
    done_keys = {row_key(row) for row in existing_outputs}
    pending_rows = [
        row for row in hard_rows
        if str(row.get("episode_id")) in episodes and row_key(row) not in done_keys
    ]

    outputs: List[Dict[str, Any]] = list(existing_outputs)
    errors: List[Dict[str, Any]] = []
    write_mode = "a" if args.resume and os.path.exists(out_path) else "w"
    with open(out_path, write_mode) as out_handle, open(err_path, "a") as err_handle, ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(
                sample_one,
                row,
                episodes,
                client,
                args.model_name,
                args.num_samples,
                args.temperature,
                args.top_p,
                args.max_tokens,
                args.request_sleep,
            ): row
            for row in pending_rows
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Sampling"):
            row = futures[future]
            try:
                result = future.result()
                outputs.append(result)
                out_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_handle.flush()
            except Exception as exc:  # keep long jobs resumable by preserving failures
                error_row = {
                    "episode_id": row.get("episode_id"),
                    "step_idx": row.get("step_idx"),
                    "error": repr(exc),
                }
                errors.append(error_row)
                err_handle.write(json.dumps(error_row, ensure_ascii=False) + "\n")
                err_handle.flush()

    summary = {
        "num_requested": len(hard_rows),
        "num_outputs": len(outputs),
        "num_errors": len(errors),
        "num_existing_outputs": len(existing_outputs),
        "num_pending_rows": len(pending_rows),
        "num_samples": args.num_samples,
        "any_success_rate": (sum(1 for row in outputs if row["any_success"]) / len(outputs)) if outputs else 0.0,
        "mean_best_reward": (sum(float(row["best_reward"]) for row in outputs) / len(outputs)) if outputs else 0.0,
        "model_name": args.model_name,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()