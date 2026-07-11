#!/usr/bin/env python3
"""Greedy teacher-forced evaluation for the noisy global-revision pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.hetero_inject_phase1_eval import generate_batch, score_output
from scripts.minimal_validation import load_model_for_eval
from scripts.rl_feasibility_sampling import sanitize_jsonable
from v13_gui_360.eval_gui360_template import _format_action_for_history
from v23_visual_transition.train_offline_grpo import build_eval_style_messages


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(sanitize_jsonable(dict(row)), ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_jobs(episodes: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    jobs = []
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        history: list[str] = []
        for step_idx, step in enumerate(episode.get("steps", [])):
            jobs.append({
                "target_id": f"{episode_id}:{step_idx}",
                "episode_id": episode_id,
                "step_idx": step_idx,
                "messages": build_eval_style_messages(episode.get("goal", ""), history, step["screenshot"]),
                "image": step["screenshot"],
                "image_w": int(step.get("image_w") or 1040),
                "image_h": int(step.get("image_h") or 736),
                "gt_action": step.get("action") or {},
            })
            history.append(_format_action_for_history(step.get("action") or {}, step_idx + 1))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--episode-data", default="outputs/multiagent_trajectory_revision/pilot_v1/test_eval_episodes.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", default="checkpoints/gui360-fullparam-sft-step250")
    parser.add_argument("--adapter-dir", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--image-max-pixels", type=int, default=602112)
    parser.add_argument("--match-threshold", type=float, default=0.5)
    parser.add_argument("--coord-bucket", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--num-comm-rounds", type=int, default=2)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    args = parser.parse_args()

    all_episodes = read_jsonl(Path(args.episode_data))
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    episodes = [row for index, row in enumerate(all_episodes) if index % args.shard_count == args.shard_index]
    jobs = build_jobs(episodes)
    model, processor, device = load_model_for_eval(args)
    processor.tokenizer.padding_side = "left"
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for start in tqdm(range(0, len(jobs), args.batch_size), desc=args.arm):
        batch = jobs[start:start + args.batch_size]
        texts = generate_batch(
            model, processor, device, batch,
            do_sample=False, temperature=0.0, top_p=1.0,
            max_new_tokens=args.max_new_tokens, seed=args.seed + start,
        )
        for job, text in zip(batch, texts):
            scored = score_output(text, job, args.coord_bucket, args.match_threshold)
            by_episode[job["episode_id"]].append({
                "target_id": job["target_id"],
                "step_idx": job["step_idx"],
                "success": scored["success"],
                "reward": scored["reward"],
                "parse_ok": scored["parse_ok"],
                "parser_mode": scored["parser_mode"],
                "action_key": scored["action_key"],
                "pred_action": scored["pred_action"],
                "raw_output": scored["raw_output"],
            })
        torch.cuda.empty_cache()

    rows = []
    source_by_id = {str(row["episode_id"]): row for row in episodes}
    for episode_id in [str(row["episode_id"]) for row in episodes]:
        steps = sorted(by_episode[episode_id], key=lambda row: row["step_idx"])
        first_error = next((step["step_idx"] + 1 for step in steps if not step["success"]), None)
        rows.append({
            "arm": args.arm,
            "episode_id": episode_id,
            "goal": source_by_id[episode_id].get("goal", ""),
            "num_steps": len(steps),
            "correct_steps": sum(step["success"] for step in steps),
            "task_success": first_error is None,
            "first_error_step": first_error,
            "teacher_forced": True,
            "steps": steps,
        })
    output = Path(args.output)
    write_jsonl(output, rows)
    all_steps = [step for row in rows for step in row["steps"]]
    summary = {
        "arm": args.arm,
        "episodes": len(rows),
        "steps": len(all_steps),
        "task_successes": sum(row["task_success"] for row in rows),
        "tsr": sum(row["task_success"] for row in rows) / max(1, len(rows)),
        "step_accuracy": sum(step["success"] for step in all_steps) / max(1, len(all_steps)),
        "parse_rate": sum(step["parse_ok"] for step in all_steps) / max(1, len(all_steps)),
        "mean_reward": sum(float(step["reward"]) for step in all_steps) / max(1, len(all_steps)),
        "episode_data": args.episode_data,
        "episode_data_sha256": sha256(Path(args.episode_data)),
        "model_path": args.model_path,
        "adapter_dir": args.adapter_dir,
        "seed": args.seed,
        "teacher_forced": True,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "output_sha256": sha256(output),
    }
    write_json(output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
