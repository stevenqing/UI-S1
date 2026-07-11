#!/usr/bin/env python3
"""Minimal B0 vs M2 validation utilities for GUI-360 offline GRPO.

This script intentionally keeps the experiment small and explicit:

- export: parquet GUI-360 split -> episode JSONL with PNG screenshots
- sample: SFT/vLLM -> matcher-scored offline candidate groups
- critical: training-time critical-step selection from B0 candidate accuracy
- eval-local: teacher-forced local greedy evaluation with optional cooperative adapter
- report: B0/M2 task-level TSR comparison report
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pyarrow.parquet as pq
import torch
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rl_feasibility_sampling import sanitize_jsonable  # noqa: E402
from v13_gui_360.eval_gui360_template import (  # noqa: E402
    SUPPORTED_ACTIONS,
    USER_PROMPT_TEMPLATE,
    _format_action_for_history,
    parse_tool_call,
)
from v13_gui_360.reward import compute_step_reward  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitize_jsonable(dict(row)), ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_jsonable(data), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def parquet_files(data_dir: Path, split: str) -> list[Path]:
    return sorted(data_dir.glob(f"{split}-*.parquet"))


def export_episodes(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    image_dir = out_dir / "images" / args.split
    image_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}_episodes.jsonl"
    rng = random.Random(args.seed)

    raw_episodes: list[dict[str, Any]] = []
    for parquet_path in parquet_files(data_dir, args.split):
        table = pq.read_table(parquet_path, columns=["episode_id", "goal", "num_steps", "steps", "screenshots"])
        raw_episodes.extend(table.to_pylist())

    if args.shuffle:
        rng.shuffle(raw_episodes)
    if args.max_episodes > 0:
        raw_episodes = raw_episodes[: args.max_episodes]

    count = 0
    step_count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for episode in tqdm(raw_episodes, desc=f"export-{args.split}"):
            steps = json.loads(episode["steps"]) if isinstance(episode["steps"], str) else episode["steps"]
            screenshots = episode["screenshots"]
            episode_id = str(episode["episode_id"])
            episode_dir = image_dir / safe_name(episode_id)
            episode_dir.mkdir(parents=True, exist_ok=True)
            out_steps = []
            for step_idx, step in enumerate(steps):
                gt_action = step.get("action") if isinstance(step, Mapping) else None
                screenshot_item = screenshots[step_idx]
                screenshot_bytes = screenshot_item.get("bytes") if isinstance(screenshot_item, Mapping) else screenshot_item
                if not isinstance(gt_action, dict) or not isinstance(screenshot_bytes, (bytes, bytearray)):
                    continue
                image = Image.open(BytesIO(bytes(screenshot_bytes))).convert("RGB")
                image_path = episode_dir / f"step_{step_idx:03d}.png"
                if not image_path.exists() or args.overwrite_images:
                    image.save(image_path)
                image_w, image_h = image.size
                out_steps.append({
                    "step_idx": int(step.get("step_idx", step_idx)),
                    "screenshot": str(image_path),
                    "action": gt_action,
                    "image_w": image_w,
                    "image_h": image_h,
                })
            if out_steps:
                row = {
                    "episode_id": episode_id,
                    "goal": str(episode.get("goal", "")),
                    "num_steps": len(out_steps),
                    "steps": out_steps,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
                step_count += len(out_steps)

    write_json(out_dir / f"{args.split}_export_summary.json", {
        "split": args.split,
        "episodes": count,
        "steps": step_count,
        "output": str(out_path),
        "image_dir": str(image_dir),
    })
    print(json.dumps({"episodes": count, "steps": step_count, "output": str(out_path)}, indent=2))


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:160]


def encode_image(path: str, image_max_pixels: Optional[int]) -> tuple[str, int, int]:
    image = Image.open(path).convert("RGB")
    image_w, image_h = image.size
    if image_max_pixels:
        pixels = image_w * image_h
        if pixels > image_max_pixels:
            scale = (image_max_pixels / pixels) ** 0.5
            image = image.resize((max(1, int(image_w * scale)), max(1, int(image_h * scale))), Image.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8"), image_w, image_h


def build_messages(goal: str, history: Sequence[str], screenshot: str, image_max_pixels: Optional[int]) -> tuple[list[dict[str, Any]], int, int]:
    history_text = "\n".join(history) if history else "None"
    prompt_text = USER_PROMPT_TEMPLATE.format(
        instruction=goal,
        history=history_text,
        actions=SUPPORTED_ACTIONS,
    )
    b64, image_w, image_h = encode_image(screenshot, image_max_pixels)
    return ([{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt_text},
        ],
    }], image_w, image_h)


def load_target_ids(path: Optional[str]) -> Optional[set[str]]:
    if not path:
        return None
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        values = data.get("target_ids") or data.get("critical_ids") or []
    else:
        values = data
    return {str(item) for item in values}


def request_texts(client: OpenAI, model_name: str, messages: list[dict[str, Any]], *, n: int, temperature: float, top_p: float, max_tokens: int, retries: int) -> list[str]:
    last_error: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                n=n,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            return [choice.message.content or "" for choice in response.choices]
        except BaseException as exc:  # noqa: BLE001 - long candidate jobs should retry transient API errors
            last_error = exc
            time.sleep(min(2.0 * (attempt + 1), 10.0))
    return [f"ERROR: {last_error}"] * n


def parse_prediction(text: str) -> Optional[dict[str, Any]]:
    try:
        pred_action = parse_tool_call(text)
    except Exception:
        pred_action = None
    if pred_action is not None:
        return pred_action
    match = re.search(r"<action>\s*(\{.*?\})\s*</action>", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None


def score_candidate(text: str, gt_action: Mapping[str, Any], image_w: int, image_h: int, match_threshold: float, temperature: float) -> dict[str, Any]:
    pred_action = parse_prediction(text)
    fake_text = f"<action>{json.dumps(pred_action, ensure_ascii=False)}</action>" if pred_action else text
    reward, info = compute_step_reward(fake_text, dict(gt_action), image_w=image_w, image_h=image_h)
    return {
        "text": text,
        "response": text,
        "reward": float(reward),
        "is_correct": bool(reward >= match_threshold),
        "type_reward": float(info.get("type_reward", 0.0)),
        "content_reward": float(info.get("content_reward", 0.0)),
        "pred_action": info.get("pred_action"),
        "pred_type": info.get("pred_type"),
        "gt_type": info.get("gt_type"),
        "temperature": temperature,
    }


def sample_episode(episode: Mapping[str, Any], client: OpenAI, args: argparse.Namespace, target_ids: Optional[set[str]]) -> list[dict[str, Any]]:
    rows = []
    history: list[str] = []
    for step_idx, step in enumerate(episode.get("steps", [])):
        target_id = f"{episode['episode_id']}:{step_idx}"
        if target_ids is not None and target_id not in target_ids:
            history.append(_format_action_for_history(step.get("action", {}) or {}, step_idx + 1))
            continue
        messages, image_w, image_h = build_messages(episode["goal"], history, step["screenshot"], args.image_max_pixels)
        texts: list[tuple[str, float]] = []
        if args.include_greedy:
            greedy = request_texts(
                client, args.model_name, messages,
                n=1, temperature=0.0, top_p=1.0,
                max_tokens=args.max_tokens, retries=args.retries,
            )[0]
            texts.append((greedy, 0.0))
        n_sample = args.n_candidates - len(texts)
        if n_sample > 0:
            samples = request_texts(
                client, args.model_name, messages,
                n=n_sample, temperature=args.sample_temperature, top_p=args.top_p,
                max_tokens=args.max_tokens, retries=args.retries,
            )
            texts.extend((text, args.sample_temperature) for text in samples)
        candidates = [score_candidate(text, step["action"], image_w, image_h, args.match_threshold, temp) for text, temp in texts]
        n_correct = sum(1 for candidate in candidates if candidate["is_correct"])
        row = {
            "target_id": target_id,
            "episode_id": str(episode["episode_id"]),
            "step_idx": step_idx,
            "num_steps": len(episode.get("steps", [])),
            "goal": episode["goal"],
            "screenshot": step["screenshot"],
            "history": list(history),
            "gt_action": step["action"],
            "image_w": image_w,
            "image_h": image_h,
            "sample_temperature": args.sample_temperature,
            "n_candidates": len(candidates),
            "n_correct": n_correct,
            "has_positive": n_correct > 0,
            "has_variance": 0 < n_correct < len(candidates),
            "candidates": candidates,
        }
        rows.append(row)
        history.append(_format_action_for_history(step.get("action", {}) or {}, step_idx + 1))
    return rows


def sample_candidates(args: argparse.Namespace) -> None:
    episodes = read_jsonl(Path(args.episode_data))
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    target_ids = load_target_ids(args.target_ids)
    existing_ids = {row.get("target_id") for row in read_jsonl(Path(args.output)) if row.get("target_id")}
    clients = [OpenAI(base_url=url, api_key="dummy", timeout=args.request_timeout) for url in args.api_urls]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    filtered = []
    for episode in episodes:
        if target_ids is None:
            filtered.append(episode)
            continue
        step_ids = {f"{episode['episode_id']}:{idx}" for idx in range(len(episode.get("steps", [])))}
        if step_ids & target_ids:
            filtered.append(episode)
    print(json.dumps({
        "stage": "sample_candidates",
        "episodes": len(filtered),
        "target_filter": len(target_ids) if target_ids is not None else None,
        "existing_rows": len(existing_ids),
        "temperature": args.sample_temperature,
        "n_candidates": args.n_candidates,
        "output": str(out_path),
    }, indent=2), flush=True)

    def worker(index_episode: tuple[int, dict[str, Any]]) -> list[dict[str, Any]]:
        index, episode = index_episode
        return sample_episode(episode, clients[index % len(clients)], args, target_ids)

    total_rows = 0
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = [pool.submit(worker, item) for item in enumerate(filtered)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="sample-candidates"):
            for row in future.result():
                if row["target_id"] in existing_ids:
                    continue
                append_jsonl(out_path, row)
                existing_ids.add(row["target_id"])
                total_rows += 1
    print(json.dumps({"written_new_rows": total_rows, "total_rows": len(existing_ids)}, indent=2), flush=True)


def derive_critical(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.candidates))
    selected = []
    for row in rows:
        candidates = row.get("candidates", [])
        if not candidates:
            continue
        n_correct = sum(1 for item in candidates if item.get("is_correct") or float(item.get("reward") or 0.0) >= args.match_threshold)
        correct_rate = n_correct / max(1, len(candidates))
        greedy_correct = bool(candidates[0].get("is_correct")) if candidates else False
        keep = False
        if args.criterion == "all_wrong":
            keep = n_correct == 0
        elif args.criterion == "greedy_wrong":
            keep = not greedy_correct
        elif args.criterion == "correct_rate_le":
            keep = correct_rate <= args.correct_rate_threshold
        if keep:
            selected.append({
                "target_id": row.get("target_id") or f"{row.get('episode_id')}:{row.get('step_idx')}",
                "episode_id": row.get("episode_id"),
                "step_idx": row.get("step_idx"),
                "n_candidates": len(candidates),
                "n_correct": n_correct,
                "correct_rate": correct_rate,
                "greedy_correct": greedy_correct,
            })
    if args.max_critical > 0:
        selected = selected[: args.max_critical]
    payload = {
        "criterion": args.criterion,
        "correct_rate_threshold": args.correct_rate_threshold,
        "source": args.candidates,
        "total_rows": len(rows),
        "critical_rows": len(selected),
        "target_ids": [row["target_id"] for row in selected],
        "rows": selected,
    }
    write_json(Path(args.output), payload)
    print(json.dumps({"critical_rows": len(selected), "output": args.output}, indent=2), flush=True)


def load_model_for_eval(args: argparse.Namespace):
    from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from v13_gui_360.iterative_cooperative_wrapper import IterativeCooperativeVLMWrapper
    from v15_gui_360.train_trajectory_gspo import V15TrajectoryGSPOTrainer

    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    if args.image_max_pixels > 0:
        processor.image_processor.max_pixels = args.image_max_pixels
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    V15TrajectoryGSPOTrainer._patch_legacy_mrope_config(config)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if args.adapter_dir:
        cfg_path = Path(args.adapter_dir) / "cooperative_config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        model = IterativeCooperativeVLMWrapper(
            base_model=base,
            lora_r=int(cfg.get("lora_r", args.lora_r)),
            lora_alpha=int(cfg.get("lora_alpha", args.lora_alpha)),
            lora_dropout=0.0,
            target_modules=list(cfg.get("target_modules", args.target_modules)),
            balance_weight=float(cfg.get("balance_weight", 0.0)),
            num_comm_rounds=int(cfg.get("num_comm_rounds", args.num_comm_rounds)),
        )
        model.load_cooperative(args.adapter_dir, device=device)
    else:
        model = base
    model = model.to(device).eval()
    return model, processor, device


def local_generate(model: Any, processor: Any, device: torch.device, messages: list[dict[str, Any]], image_path: str, args: argparse.Namespace) -> str:
    image = Image.open(image_path).convert("RGB")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=False)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    tokenizer = processor.tokenizer
    stop_ids = [tokenizer.eos_token_id]
    tool_end = tokenizer.encode("</tool_call>", add_special_tokens=False)
    if len(tool_end) == 1:
        stop_ids.append(tool_end[0])
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=stop_ids,
            stop_strings=["</tool_call>"],
            tokenizer=tokenizer,
        )
    response_ids = output_ids[0, prompt_len:]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


def evaluate_local(args: argparse.Namespace) -> None:
    episodes = read_jsonl(Path(args.episode_data))
    episodes = episodes[args.start: args.end]
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    model, processor, device = load_model_for_eval(args)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        done = {row.get("episode_id") for row in read_jsonl(out_path)}
    else:
        done = set()
        out_path.write_text("", encoding="utf-8")
    for episode in tqdm(episodes, desc=f"eval-{args.model_key}"):
        if episode["episode_id"] in done:
            continue
        history: list[str] = []
        steps_out = []
        correct = 0
        first_error = None
        for idx, step in enumerate(episode.get("steps", [])):
            messages, image_w, image_h = build_messages(episode["goal"], history, step["screenshot"], args.image_max_pixels)
            pred_text = local_generate(model, processor, device, messages, step["screenshot"], args)
            pred_action = parse_prediction(pred_text)
            fake_text = f"<action>{json.dumps(pred_action, ensure_ascii=False)}</action>" if pred_action else pred_text
            reward, info = compute_step_reward(fake_text, step["action"], image_w=image_w, image_h=image_h)
            success = reward >= args.match_threshold
            if success:
                correct += 1
            elif first_error is None:
                first_error = idx + 1
            steps_out.append({
                "step_idx": idx,
                "success": bool(success),
                "reward": float(reward),
                "pred_text": pred_text[:500],
                "pred_action": info.get("pred_action"),
                "pred_type": info.get("pred_type"),
                "gt_type": info.get("gt_type"),
            })
            history.append(_format_action_for_history(step.get("action", {}) or {}, idx + 1))
        num_steps = len(episode.get("steps", []))
        row = {
            "model_key": args.model_key,
            "episode_id": episode["episode_id"],
            "goal": episode.get("goal"),
            "num_steps": num_steps,
            "steps_evaluated": num_steps,
            "correct_steps": correct,
            "step_sr": correct / max(1, num_steps),
            "task_success": first_error is None,
            "first_error_step": first_error,
            "teacher_forced": True,
            "steps": steps_out,
        }
        append_jsonl(out_path, row)
    summarize_eval_file(out_path)


def summarize_eval_file(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    n = len(rows)
    total_steps = sum(int(row.get("num_steps") or 0) for row in rows)
    correct_steps = sum(int(row.get("correct_steps") or 0) for row in rows)
    summary = {
        "path": str(path),
        "num_episodes": n,
        "tsr": sum(1 for row in rows if row.get("task_success")) / max(1, n),
        "step_sr": correct_steps / max(1, total_steps),
        "total_steps": total_steps,
        "correct_steps": correct_steps,
    }
    write_json(path.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def report(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    b0 = read_jsonl(Path(args.b0_eval))
    m2 = read_jsonl(Path(args.m2_eval))
    sft = read_jsonl(Path(args.sft_eval)) if args.sft_eval else []
    critical_ids = set(load_target_ids(args.critical_ids) or [])
    b0_by = {row["episode_id"]: row for row in b0}
    m2_by = {row["episode_id"]: row for row in m2}
    sft_by = {row["episode_id"]: row for row in sft}

    per_task = []
    for episode_id in sorted(set(b0_by) & set(m2_by)):
        b = b0_by[episode_id]
        m = m2_by[episode_id]
        critical_steps = [tid for tid in critical_ids if tid.startswith(f"{episode_id}:")]
        per_task.append({
            "episode_id": episode_id,
            "b0_success": bool(b.get("task_success")),
            "m2_success": bool(m.get("task_success")),
            "sft_success": bool(sft_by.get(episode_id, {}).get("task_success")) if sft_by else None,
            "b0_first_error_step": b.get("first_error_step"),
            "m2_first_error_step": m.get("first_error_step"),
            "b0_step_sr": b.get("step_sr"),
            "m2_step_sr": m.get("step_sr"),
            "critical_steps": critical_steps,
            "m2_fixed_vs_b0": bool((not b.get("task_success")) and m.get("task_success")),
        })
    per_task_path = out_dir / "per_task.jsonl"
    per_task_path.write_text("", encoding="utf-8")
    for row in per_task:
        append_jsonl(per_task_path, row)

    def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        total_steps = sum(int(row.get("num_steps") or row.get("steps_evaluated") or 0) for row in rows)
        correct_steps = sum(int(row.get("correct_steps") or 0) for row in rows)
        return {
            "episodes": n,
            "tsr": sum(1 for row in rows if row.get("task_success")) / max(1, n),
            "step_sr": correct_steps / max(1, total_steps),
            "correct_steps": correct_steps,
            "total_steps": total_steps,
        }

    b0_m = metrics(b0)
    m2_m = metrics(m2)
    sft_m = metrics(sft) if sft else None
    if b0_m["episodes"] == 0 or m2_m["episodes"] == 0:
        gate = "PIPELINE ISSUE"
        reason = "Missing B0 or M2 evaluation rows."
    elif m2_m["tsr"] > b0_m["tsr"]:
        gate = "DIAGNOSIS CONVERTS"
        reason = "M2 beats B0 on teacher-forced task-level TSR at matched reported budget."
    else:
        gate = "M2 DOES NOT BEAT B0"
        reason = "M2 does not exceed B0 on teacher-forced task-level TSR at matched reported budget."

    lines = [
        "# Minimal Validation - Diagnosis-Driven Offline RL vs Uniform RL",
        "",
        "Teacher-forced offline evaluation on GT screens; frozen GUI-360 matcher. Confident-wrong countering is deferred.",
        "",
        "## Budget And Seeds",
        "",
        f"- Seed: `{args.seed}`",
        f"- Budget: `{args.budget_note}`",
        f"- Critical IDs: `{args.critical_ids}`",
        f"- SFT greedy floor reference: `{args.sft_floor}`",
        "",
        "## Results",
        "",
        "| model | episodes | task TSR | step accuracy | correct/steps |",
        "|---|---:|---:|---:|---:|",
    ]
    if sft_m:
        lines.append(f"| SFT greedy | {sft_m['episodes']} | {pct(sft_m['tsr'])} | {pct(sft_m['step_sr'])} | {sft_m['correct_steps']}/{sft_m['total_steps']} |")
    lines.append(f"| B0 uniform GRPO | {b0_m['episodes']} | {pct(b0_m['tsr'])} | {pct(b0_m['step_sr'])} | {b0_m['correct_steps']}/{b0_m['total_steps']} |")
    lines.append(f"| M2 targeted+exploration GRPO | {m2_m['episodes']} | {pct(m2_m['tsr'])} | {pct(m2_m['step_sr'])} | {m2_m['correct_steps']}/{m2_m['total_steps']} |")
    lines.extend([
        "",
        "## Stage 1 - B0 Pipeline Sanity",
        "",
        f"B0 TSR: {pct(b0_m['tsr'])}; B0 step accuracy: {pct(b0_m['step_sr'])}. Compare against SFT greedy floor/reference before interpreting Stage 2.",
        "",
        "## Stage 2 - M2 vs B0",
        "",
        f"TSR delta M2-B0: {pct(m2_m['tsr'] - b0_m['tsr'])}; step-accuracy delta: {pct(m2_m['step_sr'] - b0_m['step_sr'])}.",
        "",
        "## Gate",
        "",
        gate,
        "",
        reason,
        "",
        "STOP for review.",
        "",
    ])
    validation_path = out_dir / "validation.md"
    validation_path.write_text("\n".join(lines), encoding="utf-8")
    write_json(out_dir / "validation_summary.json", {"gate": gate, "b0": b0_m, "m2": m2_m, "sft": sft_m, "per_task": str(per_task_path)})
    print(json.dumps({"validation": str(validation_path), "per_task": str(per_task_path), "gate": gate}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("export")
    p.add_argument("--data-dir", default="datasets/gui360-balanced/data")
    p.add_argument("--split", choices=["train", "test"], required=True)
    p.add_argument("--output-dir", default="outputs/minimal_validation/data")
    p.add_argument("--max-episodes", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--overwrite-images", action="store_true")
    p.set_defaults(func=export_episodes)

    p = sub.add_parser("sample")
    p.add_argument("--episode-data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--api-urls", nargs="+", default=["http://127.0.0.1:8177/v1"])
    p.add_argument("--model-name", required=True)
    p.add_argument("--target-ids", default=None)
    p.add_argument("--max-episodes", type=int, default=0)
    p.add_argument("--n-candidates", type=int, default=8)
    p.add_argument("--include-greedy", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sample-temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--match-threshold", type=float, default=0.5)
    p.add_argument("--image-max-pixels", type=int, default=602112)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--request-timeout", type=float, default=120.0)
    p.set_defaults(func=sample_candidates)

    p = sub.add_parser("critical")
    p.add_argument("--candidates", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--criterion", choices=["all_wrong", "greedy_wrong", "correct_rate_le"], default="correct_rate_le")
    p.add_argument("--correct-rate-threshold", type=float, default=0.125)
    p.add_argument("--match-threshold", type=float, default=0.5)
    p.add_argument("--max-critical", type=int, default=0)
    p.set_defaults(func=derive_critical)

    p = sub.add_parser("eval-local")
    p.add_argument("--episode-data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model-key", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--adapter-dir", default="")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--max-episodes", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--image-max-pixels", type=int, default=602112)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--match-threshold", type=float, default=0.5)
    p.add_argument("--lora-r", type=int, default=128)
    p.add_argument("--lora-alpha", type=int, default=256)
    p.add_argument("--num-comm-rounds", type=int, default=2)
    p.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    p.set_defaults(func=evaluate_local)

    p = sub.add_parser("report")
    p.add_argument("--output-dir", default="outputs/minimal_validation")
    p.add_argument("--b0-eval", required=True)
    p.add_argument("--m2-eval", required=True)
    p.add_argument("--sft-eval", default="")
    p.add_argument("--critical-ids", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--budget-note", default="matched max_steps and candidate-group budget; see training logs")
    p.add_argument("--sft-floor", default="22.2% full-test greedy floor from prior run")
    p.set_defaults(func=report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()