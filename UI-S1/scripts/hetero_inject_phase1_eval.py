#!/usr/bin/env python3
"""Batched teacher-forced Phase-1 evaluation with arm-specific sampling temperature."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.hetero_inject_e02r import lenient_parse_action  # noqa: E402
from scripts.hetero_inject_eval import parse_prediction  # noqa: E402
from scripts.minimal_validation import load_model_for_eval  # noqa: E402
from scripts.rl_feasibility_sampling import action_key, sanitize_jsonable  # noqa: E402
from v13_gui_360.eval_gui360_template import _format_action_for_history  # noqa: E402
from v13_gui_360.reward import compute_step_reward  # noqa: E402
from v23_visual_transition.train_offline_grpo import build_eval_style_messages  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def adapter_file_hashes(adapter_dir: str) -> dict[str, str]:
    if not adapter_dir:
        return {}
    root = Path(adapter_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"adapter directory missing: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"adapter directory is empty: {root}")
    return {str(path.relative_to(root)): sha256(path) for path in files}


def load_chosen(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["target_id"]): row for row in read_jsonl(path)}


def stop_config(processor: Any) -> dict[str, Any]:
    tokenizer = processor.tokenizer
    stop_ids = [tokenizer.eos_token_id]
    tool_end = tokenizer.encode("</tool_call>", add_special_tokens=False)
    if len(tool_end) == 1:
        stop_ids.append(tool_end[0])
    return {"eos_token_id": stop_ids, "stop_strings": ["</tool_call>"], "tokenizer": tokenizer}


def generate_batch(
    model: Any,
    processor: Any,
    device: torch.device,
    jobs: Sequence[Mapping[str, Any]],
    *,
    do_sample: bool,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
) -> list[str]:
    prompts = [processor.apply_chat_template(job["messages"], tokenize=False, add_generation_prompt=True) for job in jobs]
    images = [Image.open(job["image"]).convert("RGB") for job in jobs]
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_width = int(inputs["input_ids"].shape[1])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        **stop_config(processor),
    }
    if do_sample:
        kwargs.update({"temperature": temperature, "top_p": top_p})
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        output_ids = model.generate(**kwargs)
    return [
        processor.tokenizer.decode(output_ids[index, prompt_width:], skip_special_tokens=True)
        for index in range(len(jobs))
    ]


def score_output(text: str, job: Mapping[str, Any], coord_bucket: int, match_threshold: float) -> dict[str, Any]:
    strict_action = parse_prediction(text)
    strict_key = action_key(strict_action, coord_bucket)
    lenient_action, parser_mode = lenient_parse_action(text)
    lenient_key = action_key(lenient_action, coord_bucket)
    fake_text = f"<action>{json.dumps(lenient_action, ensure_ascii=False)}</action>" if lenient_action else text
    reward, info = compute_step_reward(
        fake_text,
        dict(job["gt_action"]),
        image_w=int(job["image_w"]),
        image_h=int(job["image_h"]),
    )
    return {
        "raw_output": text,
        "strict_parse_ok": strict_action is not None,
        "strict_action_key": strict_key,
        "parse_ok": lenient_action is not None,
        "parser_mode": parser_mode,
        "action_key": lenient_key,
        "pred_action": info.get("pred_action"),
        "reward": float(reward),
        "success": bool(reward >= match_threshold),
    }


def flatten_jobs(episodes: Sequence[Mapping[str, Any]], chosen: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    jobs = []
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        history: list[str] = []
        for step_idx, step in enumerate(episode.get("steps", [])):
            target_id = f"{episode_id}:{step_idx}"
            jobs.append({
                "target_id": target_id,
                "episode_id": episode_id,
                "step_idx": step_idx,
                "num_steps": len(episode.get("steps", [])),
                "goal": episode.get("goal", ""),
                "messages": build_eval_style_messages(episode.get("goal", ""), history, step["screenshot"]),
                "image": step["screenshot"],
                "image_w": int(step.get("image_w") or 1040),
                "image_h": int(step.get("image_h") or 736),
                "gt_action": step.get("action") or {},
                "chosen_action_key": chosen.get(target_id, {}).get("chosen_action_key"),
                "is_blind_injected": target_id in chosen,
            })
            history.append(_format_action_for_history(step.get("action") or {}, step_idx + 1))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--episode-data", required=True)
    parser.add_argument("--chosen-pairs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", default="checkpoints/gui360-fullparam-sft-step250")
    parser.add_argument("--adapter-dir", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-k", type=int, default=4)
    parser.add_argument("--sample-temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--image-max-pixels", type=int, default=602112)
    parser.add_argument("--match-threshold", type=float, default=0.5)
    parser.add_argument("--coord-bucket", type=int, default=25)
    parser.add_argument("--conf-threshold", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--num-comm-rounds", type=int, default=2)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    args = parser.parse_args()

    episodes = read_jsonl(Path(args.episode_data))
    if args.max_episodes > 0:
        episodes = episodes[:args.max_episodes]
    chosen = load_chosen(Path(args.chosen_pairs))
    jobs = flatten_jobs(episodes, chosen)
    adapter_hashes = adapter_file_hashes(args.adapter_dir)
    model, processor, device = load_model_for_eval(args)
    processor.tokenizer.padding_side = "left"

    greedy_by_target: dict[str, dict[str, Any]] = {}
    for batch_index in tqdm(range(0, len(jobs), args.batch_size), desc=f"{args.arm}-greedy"):
        batch = jobs[batch_index:batch_index + args.batch_size]
        texts = generate_batch(
            model, processor, device, batch,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed + batch_index,
        )
        for job, text in zip(batch, texts):
            greedy_by_target[job["target_id"]] = {**job, **score_output(text, job, args.coord_bucket, args.match_threshold)}
        torch.cuda.empty_cache()

    blind_jobs = [job for job in jobs if job["is_blind_injected"]]
    samples_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    positions_per_batch = max(1, args.batch_size // args.sample_k)
    for start in tqdm(range(0, len(blind_jobs), positions_per_batch), desc=f"{args.arm}-sample"):
        position_batch = blind_jobs[start:start + positions_per_batch]
        expanded = [job for job in position_batch for _ in range(args.sample_k)]
        seed_text = f"{args.arm}|{args.sample_temperature}|{start}|{args.seed}"
        batch_seed = args.seed + int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16)
        texts = generate_batch(
            model, processor, device, expanded,
            do_sample=True,
            temperature=args.sample_temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            seed=batch_seed,
        )
        for job, text in zip(expanded, texts):
            samples_by_target[job["target_id"]].append(score_output(text, job, args.coord_bucket, args.match_threshold))
        torch.cuda.empty_cache()

    episode_steps: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        greedy = greedy_by_target[job["target_id"]]
        samples = samples_by_target.get(job["target_id"], [])
        counts = Counter(sample["action_key"] for sample in samples)
        chosen_key = job.get("chosen_action_key")
        chosen_count = counts.get(chosen_key, 0) if chosen_key else 0
        greedy_share = counts.get(greedy["action_key"], 0) / max(1, len(samples)) if samples else None
        episode_steps[job["episode_id"]].append({
            "target_id": job["target_id"],
            "step_idx": job["step_idx"],
            "success": greedy["success"],
            "reward": greedy["reward"],
            "greedy_raw_output": greedy["raw_output"],
            "greedy_pred_action": greedy["pred_action"],
            "parse_ok": greedy["parse_ok"],
            "strict_parse_ok": greedy["strict_parse_ok"],
            "parser_mode": greedy["parser_mode"],
            "action_key": greedy["action_key"],
            "strict_action_key": greedy["strict_action_key"],
            "strict_lenient_action_key_agree": greedy["strict_action_key"] == greedy["action_key"],
            "is_blind_injected": job["is_blind_injected"],
            "chosen_action_key": chosen_key,
            "strict_lenient_chosen_verdict_agree": (
                (greedy["strict_action_key"] == chosen_key) == (greedy["action_key"] == chosen_key)
                if chosen_key else None
            ),
            "greedy_is_chosen": bool(chosen_key and greedy["action_key"] == chosen_key),
            "sample_count": len(samples),
            "sample_parse_count": sum(1 for sample in samples if sample["parse_ok"]),
            "sample_chosen_count": chosen_count,
            "sample_chosen_frequency": chosen_count / max(1, len(samples)) if samples else None,
            "sample_chosen_any": chosen_count > 0 if samples else None,
            "sample_greedy_share": greedy_share,
            "sample_confident_wrong": bool(samples and not greedy["success"] and greedy_share is not None and greedy_share >= args.conf_threshold),
            "sample_ece_abs": abs(float(greedy_share) - (1.0 if greedy["success"] else 0.0)) if greedy_share is not None else None,
            "samples": samples,
        })

    episode_rows = []
    episode_lookup = {str(row["episode_id"]): row for row in episodes}
    for episode_id in [str(row["episode_id"]) for row in episodes]:
        steps = sorted(episode_steps[episode_id], key=lambda row: row["step_idx"])
        first_error = next((step["step_idx"] + 1 for step in steps if not step["success"]), None)
        episode_rows.append({
            "arm": args.arm,
            "episode_id": episode_id,
            "goal": episode_lookup[episode_id].get("goal", ""),
            "num_steps": len(steps),
            "correct_steps": sum(1 for step in steps if step["success"]),
            "task_success": first_error is None,
            "first_error_step": first_error,
            "teacher_forced": True,
            "sample_temperature": args.sample_temperature,
            "sample_k": args.sample_k,
            "parser": "lenient-tool-call-v1",
            "steps": steps,
        })
    output_path = Path(args.output)
    write_jsonl(output_path, episode_rows)

    all_steps = [step for row in episode_rows for step in row["steps"]]
    blind_steps = [step for step in all_steps if step["is_blind_injected"]]
    summary = {
        "arm": args.arm,
        "episodes": len(episode_rows),
        "steps": len(all_steps),
        "tsr": sum(1 for row in episode_rows if row["task_success"]) / max(1, len(episode_rows)),
        "step_accuracy": sum(1 for step in all_steps if step["success"]) / max(1, len(all_steps)),
        "blind_steps": len(blind_steps),
        "blind_accuracy": sum(1 for step in blind_steps if step["success"]) / max(1, len(blind_steps)),
        "chosen_frequency": sum(float(step["sample_chosen_frequency"] or 0.0) for step in blind_steps) / max(1, len(blind_steps)),
        "chosen_sampled_any": sum(1 for step in blind_steps if step["sample_chosen_any"]) / max(1, len(blind_steps)),
        "confident_wrong": sum(1 for step in blind_steps if step["sample_confident_wrong"]) / max(1, len(blind_steps)),
        "sample_ece_abs": sum(float(step["sample_ece_abs"] or 0.0) for step in blind_steps) / max(1, len(blind_steps)),
        "sample_parse_rate": sum(int(step["sample_parse_count"] or 0) for step in blind_steps) / max(1, sum(int(step["sample_count"] or 0) for step in blind_steps)),
        "greedy_parser_parity": sum(1 for step in all_steps if step["strict_lenient_action_key_agree"]) / max(1, len(all_steps)),
        "greedy_chosen_verdict_parity": sum(1 for step in blind_steps if step["strict_lenient_chosen_verdict_agree"]) / max(1, len(blind_steps)),
        "sample_temperature": args.sample_temperature,
        "sample_k": args.sample_k,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "coord_bucket": args.coord_bucket,
        "match_threshold": args.match_threshold,
        "confidence_threshold": args.conf_threshold,
        "teacher_forced": True,
        "parser": "lenient-tool-call-v1",
        "model_path": args.model_path,
        "adapter_dir": args.adapter_dir,
        "adapter_files_sha256": adapter_hashes,
        "episode_data_sha256": sha256(Path(args.episode_data)),
        "chosen_pairs_sha256": sha256(Path(args.chosen_pairs)),
        "eval_jsonl_sha256": sha256(output_path),
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
