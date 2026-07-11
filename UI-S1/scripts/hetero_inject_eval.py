#!/usr/bin/env python3
"""Teacher-forced local evaluation and blind-step sampling for hetero injection."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rl_feasibility_sampling import action_key, sanitize_jsonable  # noqa: E402
from v13_gui_360.eval_gui360_template import _format_action_for_history, parse_tool_call  # noqa: E402
from v13_gui_360.reward import compute_step_reward  # noqa: E402
from scripts.minimal_validation import build_messages, load_model_for_eval  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitize_jsonable(dict(row)), ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_jsonable(data), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_chosen_map(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    return {str(row["target_id"]): row for row in rows}


def parse_prediction(text: str) -> Optional[dict[str, Any]]:
    try:
        action = parse_tool_call(text)
    except Exception:
        action = None
    if action is not None:
        return action
    match = re.search(r"<action>\s*(\{.*?\})\s*</action>", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None


def generate_text(
    model: Any,
    processor: Any,
    device: torch.device,
    messages: list[dict[str, Any]],
    image_path: str,
    args: argparse.Namespace,
    *,
    do_sample: bool,
) -> str:
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
            do_sample=do_sample,
            temperature=args.sample_temperature if do_sample else None,
            top_p=args.top_p if do_sample else None,
            eos_token_id=stop_ids,
            stop_strings=["</tool_call>"],
            tokenizer=tokenizer,
        )
    response_ids = output_ids[0, prompt_len:]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


def score_text(text: str, gt_action: Mapping[str, Any], image_w: int, image_h: int, match_threshold: float, coord_bucket: int) -> dict[str, Any]:
    pred_action = parse_prediction(text)
    fake_text = f"<action>{json.dumps(pred_action, ensure_ascii=False)}</action>" if pred_action else text
    reward, info = compute_step_reward(fake_text, dict(gt_action), image_w=image_w, image_h=image_h)
    parsed_action = info.get("pred_action")
    return {
        "raw_output": text[:1000],
        "reward": float(reward),
        "success": bool(reward >= match_threshold),
        "correct": bool(reward >= match_threshold),
        "pred_action": parsed_action,
        "pred_type": info.get("pred_type"),
        "gt_type": info.get("gt_type"),
        "action_key": action_key(parsed_action, coord_bucket),
        "parse_ok": parsed_action is not None,
    }


def summarize_samples(greedy_key: str, greedy_success: bool, chosen_key: Optional[str], samples: Sequence[Mapping[str, Any]], conf_threshold: float) -> dict[str, Any]:
    counts = Counter(str(sample.get("action_key")) for sample in samples)
    total = len(samples)
    modal_count = max(counts.values(), default=0)
    greedy_count = counts.get(greedy_key, 0)
    chosen_count = counts.get(chosen_key or "", 0)
    confidence = greedy_count / max(1, total)
    return {
        "sample_count": total,
        "sample_parse_count": sum(1 for sample in samples if sample.get("parse_ok")),
        "sample_correct_count": sum(1 for sample in samples if sample.get("success")),
        "sample_modal_share": modal_count / max(1, total),
        "sample_greedy_share": confidence,
        "sample_chosen_count": chosen_count,
        "sample_chosen_frequency": chosen_count / max(1, total),
        "sample_chosen_any": chosen_count > 0,
        "sample_confident_wrong": (not greedy_success) and confidence >= conf_threshold,
        "sample_ece_abs": abs(confidence - (1.0 if greedy_success else 0.0)),
    }


def evaluate(args: argparse.Namespace) -> None:
    episodes = read_jsonl(Path(args.episode_data))
    episodes = episodes[args.start: args.end]
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    chosen_map = load_chosen_map(Path(args.chosen_pairs)) if args.chosen_pairs else {}
    model, processor, device = load_model_for_eval(args)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        done = {row.get("episode_id") for row in read_jsonl(out_path)}
    else:
        done = set()
        out_path.write_text("", encoding="utf-8")

    for episode in tqdm(episodes, desc=f"eval-{args.model_key}"):
        episode_id = str(episode["episode_id"])
        if episode_id in done:
            continue
        history: list[str] = []
        steps_out: list[dict[str, Any]] = []
        correct = 0
        first_error = None
        for idx, step in enumerate(episode.get("steps", [])):
            tid = f"{episode_id}:{idx}"
            messages, image_w, image_h = build_messages(episode["goal"], history, step["screenshot"], args.image_max_pixels)
            greedy_text = generate_text(model, processor, device, messages, step["screenshot"], args, do_sample=False)
            greedy = score_text(greedy_text, step["action"], image_w, image_h, args.match_threshold, args.coord_bucket)
            success = bool(greedy["success"])
            if success:
                correct += 1
            elif first_error is None:
                first_error = idx + 1

            chosen_row = chosen_map.get(tid)
            samples: list[dict[str, Any]] = []
            sample_summary: dict[str, Any] = {}
            if chosen_row and args.sample_n > 0:
                for _ in range(args.sample_n):
                    sample_text = generate_text(model, processor, device, messages, step["screenshot"], args, do_sample=True)
                    samples.append(score_text(sample_text, step["action"], image_w, image_h, args.match_threshold, args.coord_bucket))
                sample_summary = summarize_samples(
                    str(greedy["action_key"]),
                    success,
                    str(chosen_row.get("chosen_action_key")),
                    samples,
                    args.conf_threshold,
                )

            steps_out.append({
                "target_id": tid,
                "step_idx": idx,
                "success": success,
                "reward": greedy["reward"],
                "pred_text": greedy_text[:500],
                "pred_action": greedy["pred_action"],
                "pred_type": greedy["pred_type"],
                "gt_type": greedy["gt_type"],
                "action_key": greedy["action_key"],
                "parse_ok": greedy["parse_ok"],
                "is_blind_injected": chosen_row is not None,
                "chosen_action_key": chosen_row.get("chosen_action_key") if chosen_row else None,
                "chosen_teacher": chosen_row.get("chosen_teacher") if chosen_row else None,
                "greedy_is_chosen": bool(chosen_row and greedy["action_key"] == chosen_row.get("chosen_action_key")),
                **sample_summary,
                "samples": samples[: args.store_samples],
            })
            history.append(_format_action_for_history(step.get("action", {}) or {}, idx + 1))

        num_steps = len(episode.get("steps", []))
        append_jsonl(out_path, {
            "model_key": args.model_key,
            "episode_id": episode_id,
            "goal": episode.get("goal"),
            "num_steps": num_steps,
            "correct_steps": correct,
            "step_sr": correct / max(1, num_steps),
            "task_success": first_error is None,
            "first_error_step": first_error,
            "teacher_forced": True,
            "sample_n_on_blind": args.sample_n,
            "steps": steps_out,
        })
        torch.cuda.empty_cache()

    summarize(out_path)


def summarize(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    total_steps = sum(int(row.get("num_steps") or 0) for row in rows)
    correct_steps = sum(int(row.get("correct_steps") or 0) for row in rows)
    blind_steps = [step for row in rows for step in row.get("steps", []) if step.get("is_blind_injected")]
    summary = {
        "path": str(path),
        "episodes": len(rows),
        "steps": total_steps,
        "tsr": sum(1 for row in rows if row.get("task_success")) / max(1, len(rows)),
        "step_accuracy": correct_steps / max(1, total_steps),
        "blind_steps": len(blind_steps),
        "blind_accuracy": sum(1 for step in blind_steps if step.get("success")) / max(1, len(blind_steps)),
        "blind_greedy_is_chosen": sum(1 for step in blind_steps if step.get("greedy_is_chosen")) / max(1, len(blind_steps)),
        "blind_sample_chosen_frequency": sum(float(step.get("sample_chosen_frequency") or 0.0) for step in blind_steps) / max(1, len(blind_steps)),
        "blind_sample_confident_wrong": sum(1 for step in blind_steps if step.get("sample_confident_wrong")) / max(1, len(blind_steps)),
        "blind_sample_ece_abs": sum(float(step.get("sample_ece_abs") or 0.0) for step in blind_steps) / max(1, len(blind_steps)),
    }
    write_json(path.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-data", required=True)
    parser.add_argument("--chosen-pairs", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-dir", default="")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-max-pixels", type=int, default=602112)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--match-threshold", type=float, default=0.5)
    parser.add_argument("--coord-bucket", type=int, default=25)
    parser.add_argument("--sample-n", type=int, default=0)
    parser.add_argument("--sample-temperature", type=float, default=1.5)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--conf-threshold", type=float, default=0.75)
    parser.add_argument("--store-samples", type=int, default=0)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--num-comm-rounds", type=int, default=2)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()