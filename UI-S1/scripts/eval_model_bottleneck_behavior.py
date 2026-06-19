#!/usr/bin/env python3
"""Model-behavior bottleneck validation for segmentation boundaries.

Runs intervention-style next-action prediction at real boundaries and random
non-boundaries under multiple context conditions:
  - no_history: goal + current step only
  - segment_summary: goal + previous segment summaries + current step
  - full_history: goal + full previous action history + current step
  - wrong_summary: goal + unrelated segment summary + current step

The script talks to OpenAI-compatible endpoints (vLLM servers), so Qwen3-VL and
Qwen3.5 multimodal models can be evaluated with the same protocol.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent

JsonDict = dict[str, Any]

ACTION_SPACE = """Return exactly one action in this format:
<action>
{"action": "click", "coordinate": [x, y]}
</action>

Available actions:
- click: tap a coordinate [x, y]
- long_press: long press a coordinate [x, y]
- swipe: swipe from coordinate to coordinate2
- type: type text
- open: open an app by name
- wait: wait for loading
- system_button: press Home, Back, or Menu
- terminate: finish when the task is complete
"""

NO_THINK_ACTION_OUTPUT_INSTRUCTION = """CRITICAL: Do not explain. Do not provide analysis. Return only this XML block:
<action>
{"action": "system_button", "button": "Home"}
</action>
Use the correct action and arguments for the current state."""

THINK_ACTION_OUTPUT_INSTRUCTION = """Use thinking mode if it helps. The final answer must still contain exactly one parseable action block:
<action>
{"action": "system_button", "button": "Home"}
</action>
Use the correct action and arguments for the current state."""


def iter_jsonl(path: Path) -> Iterable[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def action_type(step: JsonDict) -> str:
    return str(step.get("action", {}).get("type", "unknown"))


def action_to_short_text(action: JsonDict) -> str:
    action_type_ = action.get("type") or action.get("action") or "unknown"
    args = action.get("args", {})
    if action_type_ in {"click", "long_press"}:
        return f"{action_type_} at {args.get('coordinate')}"
    if action_type_ == "swipe":
        return f"swipe from {args.get('coordinate')} to {args.get('coordinate2')}"
    if action_type_ == "type":
        return f"type {args.get('text', '')!r}"
    if action_type_ == "open":
        return f"open {args.get('text', '')!r}"
    if action_type_ == "system_button":
        return f"system_button {args.get('button', '')!r}"
    if action_type_ == "terminate":
        return f"terminate {args.get('status', '')!r}"
    return json.dumps(action, ensure_ascii=False)


def find_segment_for_step(episode: JsonDict, step_index: int) -> JsonDict | None:
    for segment in episode.get("segments", []):
        if segment["start_step"] <= step_index <= segment["end_step"]:
            return segment
    return None


def previous_segments(episode: JsonDict, step_index: int) -> list[JsonDict]:
    return [segment for segment in episode.get("segments", []) if segment["end_step"] < step_index]


def full_history_text(episode: JsonDict, step_index: int, max_steps: int = 6) -> str:
    steps = episode.get("steps", [])
    start = max(0, step_index - max_steps)
    lines = []
    for step in steps[start:step_index]:
        instruction = step.get("instruction") or step.get("text_fields", {}).get("instruction", "")
        lines.append(f"Step {step['step_index']}: {action_to_short_text(step['action'])}. {instruction}".strip())
    return "\n".join(lines)


def segment_summary_text(segments: list[JsonDict], max_segments: int = 4) -> str:
    if not segments:
        return "None."
    lines = []
    for segment in segments[-max_segments:]:
        memory = segment.get("memory_need", {})
        values = segment.get("carried_values", [])
        lines.append(
            f"Segment {segment['segment_id']} steps {segment['start_step']}-{segment['end_step']}: "
            f"{segment.get('summary', '')}; capability={segment.get('dominant_capability')}; "
            f"memory={memory.get('strength', 'none')}; carried_values={values}"
        )
    return "\n".join(lines)


def current_step_text(step: JsonDict) -> str:
    instruction = step.get("instruction") or step.get("text_fields", {}).get("instruction", "")
    thought = step.get("thought") or step.get("text_fields", {}).get("thought", "")
    grounding = step.get("grounding", {})
    parts = [f"Current step index: {step['step_index']}"]
    if instruction:
        parts.append(f"Current step instruction: {instruction}")
    if thought:
        parts.append(f"Current local hint: {thought[:180]}")
    if grounding.get("bbox"):
        parts.append(f"Grounding bbox hint from data format: {grounding.get('bbox')}")
    return "\n".join(parts)


def build_prompt(episode: JsonDict, step_index: int, condition: str, wrong_summary: str = "", thinking_enabled: bool = False) -> str:
    step = episode["steps"][step_index]
    current_segment = find_segment_for_step(episode, step_index)
    prev_segments = previous_segments(episode, step_index)
    prompt = [
        "You are validating whether trajectory boundaries are useful bottlenecks for GUI action prediction.",
        "Predict the next GUI action for the current screenshot/state.",
        ACTION_SPACE,
        THINK_ACTION_OUTPUT_INSTRUCTION if thinking_enabled else NO_THINK_ACTION_OUTPUT_INSTRUCTION,
        f"Overall task: {episode.get('task_goal', '')}",
    ]
    if condition == "full_history":
        prompt.append("Previous action history:")
        prompt.append(full_history_text(episode, step_index))
    elif condition == "segment_summary":
        prompt.append("Relevant completed segment memory:")
        prompt.append(segment_summary_text(prev_segments))
    elif condition == "wrong_summary":
        prompt.append("Potentially relevant completed segment memory:")
        prompt.append(wrong_summary or "None.")
    elif condition == "no_history":
        prompt.append("No previous trajectory history is available.")
    else:
        raise ValueError(f"unknown condition: {condition}")
    if current_segment:
        prompt.append(f"Current segment hypothesis: {current_segment.get('summary', '')}")
    prompt.append(current_step_text(step))
    prompt.append("Return only the <action>...</action> block.")
    return "\n\n".join(item for item in prompt if item)


def image_to_data_url(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def resolve_screenshot(path_text: str) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
        if str(path).startswith("/datasets/"):
            candidates.append(PROJECT_ROOT / str(path).lstrip("/"))
    else:
        candidates.append(PROJECT_ROOT / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_messages(prompt: str, screenshot: Path | None, use_image: bool) -> list[JsonDict]:
    content: list[JsonDict] = [{"type": "text", "text": prompt}]
    if use_image and screenshot is not None:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(screenshot)}})
    return [{"role": "user", "content": content}]


def call_openai_compatible(
    api_url: str,
    model: str,
    messages: list[JsonDict],
    max_tokens: int,
    temperature: float,
    timeout: int,
    chat_template_kwargs: JsonDict | None = None,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    url = api_url.rstrip("/") + "/chat/completions"
    response = requests.post(url, headers={"Authorization": "Bearer EMPTY"}, json=payload, timeout=timeout)
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:2000]}")
    data = response.json()
    message = data["choices"][0]["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    if reasoning:
        return f"<think>\n{reasoning}\n</think>\n\n{content}"
    return content


def parse_action(text: str) -> JsonDict | None:
    match = re.search(r"<action>\s*(\{.*?\})\s*</action>", text, re.S)
    if not match:
        match = re.search(r"(\{[^{}]*\})", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except Exception:
        return None


def normalize_gt_action(step: JsonDict) -> JsonDict:
    action = step.get("action", {})
    row = {"action": action.get("type", "unknown")}
    row.update(action.get("args", {}))
    return row


def check_prediction(pred: JsonDict | None, gt: JsonDict) -> dict[str, bool]:
    if pred is None:
        return {"parse_ok": False, "type_match": False, "value_match": False}
    pred_type = str(pred.get("action", "")).lower()
    gt_type = str(gt.get("action", "")).lower()
    type_match = pred_type == gt_type
    value_match = type_match
    if type_match and gt_type in {"type", "open"}:
        pred_text = str(pred.get("text", "")).lower().strip()
        gt_text = str(gt.get("text", "")).lower().strip()
        value_match = bool(pred_text and gt_text and (pred_text in gt_text or gt_text in pred_text))
    elif type_match and gt_type == "system_button":
        value_match = str(pred.get("button", "")).lower().strip() == str(gt.get("button", "")).lower().strip()
    elif type_match and gt_type == "terminate":
        value_match = str(pred.get("status", "")).lower().strip() == str(gt.get("status", "")).lower().strip() or "status" not in gt
    return {"parse_ok": True, "type_match": type_match, "value_match": value_match}


def load_episodes(paths: list[Path]) -> list[JsonDict]:
    episodes = []
    for path in paths:
        episodes.extend(iter_jsonl(path))
    return episodes


def choose_cases(episodes: list[JsonDict], max_cases: int, seed: int, require_image: bool) -> list[JsonDict]:
    rng = random.Random(seed)
    real_cases = []
    random_cases = []
    for episode in episodes:
        steps = episode.get("steps", [])
        real_starts = [s["start_step"] for s in episode.get("segments", []) if s.get("start_step", 0) > 0]
        real_start_set = set(real_starts)
        for step_index in real_starts:
            screenshot = resolve_screenshot(steps[step_index].get("screenshot", "")) if step_index < len(steps) else None
            if require_image and screenshot is None:
                continue
            real_cases.append({"case_kind": "real_boundary", "episode": episode, "step_index": step_index})
        candidates = [idx for idx in range(1, len(steps)) if idx not in real_start_set]
        if candidates:
            step_index = rng.choice(candidates)
            screenshot = resolve_screenshot(steps[step_index].get("screenshot", ""))
            if not require_image or screenshot is not None:
                random_cases.append({"case_kind": "random_control", "episode": episode, "step_index": step_index})
    rng.shuffle(real_cases)
    rng.shuffle(random_cases)
    n_real = max_cases // 2
    n_random = max_cases - n_real
    return real_cases[:n_real] + random_cases[:n_random]


def wrong_summary_pool(episodes: list[JsonDict]) -> list[str]:
    pool = []
    for episode in episodes:
        for segment in episode.get("segments", []):
            text = segment_summary_text([segment], max_segments=1)
            if text:
                pool.append(text)
    return pool


def evaluate_one_request(
    args: argparse.Namespace,
    model_key: str,
    model_name: str,
    api_url: str,
    use_image: bool,
    job: JsonDict,
) -> JsonDict:
    case = job["case"]
    case_id = job["case_id"]
    condition = job["condition"]
    wrong_summary = job["wrong_summary"]
    thinking_mode = job["thinking_mode"]
    thinking_enabled = thinking_mode == "thinking"
    episode = case["episode"]
    step_index = case["step_index"]
    step = episode["steps"][step_index]
    screenshot = resolve_screenshot(step.get("screenshot", ""))
    gt = normalize_gt_action(step)
    prompt = build_prompt(episode, step_index, condition, wrong_summary, thinking_enabled=thinking_enabled)
    messages = build_messages(prompt, screenshot, use_image)
    output = ""
    error = ""
    parsed = None
    checks = {"parse_ok": False, "type_match": False, "value_match": False}
    try:
        chat_template_kwargs = {"enable_thinking": thinking_enabled}
        output = call_openai_compatible(
            api_url,
            model_name,
            messages,
            args.max_tokens,
            args.temperature,
            args.timeout,
            chat_template_kwargs=chat_template_kwargs,
        )
        parsed = parse_action(output)
        checks = check_prediction(parsed, gt)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    if args.sleep > 0:
        time.sleep(args.sleep)
    return {
        "model_key": model_key,
        "model_name": model_name,
        "thinking_mode": thinking_mode,
        "case_id": case_id,
        "case_kind": case["case_kind"],
        "benchmark": episode.get("benchmark"),
        "episode_id": episode.get("episode_id"),
        "step_index": step_index,
        "condition": condition,
        "use_image": use_image,
        "screenshot": str(screenshot) if screenshot else "",
        "gt_action": gt,
        "pred_action": parsed,
        "raw_output": output,
        "error": error,
        **checks,
    }


def evaluate_model(args: argparse.Namespace, model_key: str, model_name: str, api_url: str, use_image: bool, cases: list[JsonDict], summary_pool: list[str]) -> list[JsonDict]:
    rng = random.Random(args.seed + 17)
    conditions = ["no_history", "segment_summary", "full_history", "wrong_summary"]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / f"{model_key}_partial_results.jsonl"
    progress_path = output_dir / f"{model_key}_progress.json"
    if not args.resume_partial or not partial_path.exists():
        partial_path.write_text("", encoding="utf-8")
    jobs = []
    for local_case_id, case in enumerate(cases):
        case_id = int(case.get("case_id", local_case_id))
        for thinking_mode in args.thinking_modes:
            for condition in conditions:
                wrong = rng.choice(summary_pool) if summary_pool else ""
                jobs.append({
                    "job_index": len(jobs),
                    "case_id": case_id,
                    "case": case,
                    "condition": condition,
                    "wrong_summary": wrong,
                    "thinking_mode": thinking_mode,
                })
    rows_by_index: dict[int, JsonDict] = {}
    job_index_by_key = {
        (job["thinking_mode"], job["case"]["case_kind"], int(job["case_id"]), job["condition"]): int(job["job_index"])
        for job in jobs
    }
    if args.resume_partial and partial_path.exists():
        for row in iter_jsonl(partial_path):
            key = (row.get("thinking_mode", "unknown"), row.get("case_kind"), int(row.get("case_id", -1)), row.get("condition"))
            job_index = job_index_by_key.get(key)
            if job_index is not None:
                rows_by_index[job_index] = row
    request_workers = max(1, args.request_workers)

    def record_progress(row: JsonDict | None = None) -> None:
        if row is not None:
            with partial_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        errors = sum(1 for item in rows_by_index.values() if item.get("error"))
        progress = {
            "model_key": model_key,
            "model_name": model_name,
            "completed_requests": len(rows_by_index),
            "total_requests": len(jobs),
            "error_requests": errors,
            "request_workers": request_workers,
            "partial_results": str(partial_path),
        }
        progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    record_progress()
    pending_jobs = [job for job in jobs if int(job["job_index"]) not in rows_by_index]
    if rows_by_index:
        print(f"[{model_key}] resumed {len(rows_by_index)}/{len(jobs)} completed requests")
    if request_workers == 1:
        for job in pending_jobs:
            row = evaluate_one_request(args, model_key, model_name, api_url, use_image, job)
            rows_by_index[job["job_index"]] = row
            record_progress(row)
            if len(rows_by_index) % 20 == 0:
                print(f"[{model_key}] {len(rows_by_index)}/{len(jobs)} requests done")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=request_workers) as executor:
            future_to_index = {
                executor.submit(evaluate_one_request, args, model_key, model_name, api_url, use_image, job): job["job_index"]
                for job in pending_jobs
            }
            for future in concurrent.futures.as_completed(future_to_index):
                job_index = future_to_index[future]
                row = future.result()
                rows_by_index[job_index] = row
                record_progress(row)
                if len(rows_by_index) % 20 == 0:
                    print(f"[{model_key}] {len(rows_by_index)}/{len(jobs)} requests done")
    record_progress()
    return [rows_by_index[index] for index in sorted(rows_by_index)]


def summarize(rows: list[JsonDict]) -> JsonDict:
    groups: dict[tuple[str, str, str, str], list[JsonDict]] = defaultdict(list)
    for row in rows:
        groups[(row["model_key"], row.get("thinking_mode", "unknown"), row["case_kind"], row["condition"])].append(row)
    summary = {}
    for key, subset in groups.items():
        model_key, thinking_mode, case_kind, condition = key
        total = len(subset)
        parsed = sum(row["parse_ok"] for row in subset)
        type_match = sum(row["type_match"] for row in subset)
        value_match = sum(row["value_match"] for row in subset)
        summary["|".join(key)] = {
            "model_key": model_key,
            "thinking_mode": thinking_mode,
            "case_kind": case_kind,
            "condition": condition,
            "n": total,
            "parse_rate": parsed / total if total else 0.0,
            "type_acc": type_match / total if total else 0.0,
            "value_acc": value_match / total if total else 0.0,
        }
    return summary


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, summary: JsonDict, args: argparse.Namespace) -> None:
    lines = ["# Qwen Bottleneck Model-Behavior Validation", ""]
    lines.append("Compares no-history, segment-summary, full-history, and wrong-summary contexts at real boundaries versus random controls.")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- vl_model: `{args.vl_model}` @ `{args.vl_api_url}`")
    lines.append(f"- qwen35_model: `{args.text_model}` @ `{args.text_api_url}`")
    lines.append(f"- thinking_modes: `{args.thinking_modes}`")
    lines.append("- qwen35_use_image: `True`")
    lines.append(f"- max_cases: {args.max_cases}")
    lines.append(f"- request_workers: {args.request_workers}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| model | thinking | case | condition | n | parse | type_acc | value_acc |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|")
    for item in sorted(summary.values(), key=lambda x: (x["model_key"], x["thinking_mode"], x["case_kind"], x["condition"])):
        lines.append(
            f"| {item['model_key']} | {item['thinking_mode']} | {item['case_kind']} | {item['condition']} | {item['n']} | "
            f"{item['parse_rate']:.3f} | {item['type_acc']:.3f} | {item['value_acc']:.3f} |"
        )
    lines.append("")
    lines.append("## Bottleneck Evidence Criterion")
    lines.append("")
    lines.append("A boundary is behaviorally bottleneck-like if segment_summary improves over no_history at real boundaries, wrong_summary hurts, and the same gains are weaker on random controls. Full_history is the high-cost upper baseline.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model-behavior bottleneck validation with Qwen models")
    parser.add_argument("--inputs", nargs="+", default=[str(PROJECT_ROOT / "datasets" / "segmentation_train" / "gui_odyssey_segments.jsonl")])
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "datasets" / "model_bottleneck_validation"))
    parser.add_argument("--vl-api-url", default="http://localhost:8000/v1")
    parser.add_argument("--vl-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--text-api-url", default="http://localhost:8001/v1")
    parser.add_argument("--text-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--models", nargs="+", choices=["vl", "text", "qwen35"], default=["vl", "qwen35"])
    parser.add_argument("--thinking-modes", nargs="+", choices=["non_thinking", "thinking"], default=["non_thinking", "thinking"])
    parser.add_argument("--max-cases", type=int, default=40)
    parser.add_argument("--case-shard-index", type=int, default=0)
    parser.add_argument("--case-shard-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--request-workers", type=int, default=1)
    parser.add_argument("--resume-partial", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--sleep", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_episodes([Path(path) for path in args.inputs])
    summary_pool = wrong_summary_pool(episodes)
    all_rows = []
    if "vl" in args.models:
        vl_cases = choose_cases(episodes, args.max_cases, args.seed, require_image=True)
        for case_id, case in enumerate(vl_cases):
            case["case_id"] = case_id
        if args.case_shard_count > 1:
            vl_cases = [case for case in vl_cases if int(case["case_id"]) % args.case_shard_count == args.case_shard_index]
        all_rows.extend(evaluate_model(args, "qwen3_vl_8b", args.vl_model, args.vl_api_url, True, vl_cases, summary_pool))
    if "text" in args.models or "qwen35" in args.models:
        qwen35_cases = choose_cases(episodes, args.max_cases, args.seed, require_image=True)
        for case_id, case in enumerate(qwen35_cases):
            case["case_id"] = case_id
        if args.case_shard_count > 1:
            qwen35_cases = [case for case in qwen35_cases if int(case["case_id"]) % args.case_shard_count == args.case_shard_index]
        all_rows.extend(evaluate_model(args, "qwen3_5_9b", args.text_model, args.text_api_url, True, qwen35_cases, summary_pool))

    write_jsonl(output_dir / "model_behavior_results.jsonl", all_rows)
    summary = summarize(all_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(output_dir / "model_behavior_report.md", summary, args)
    print(f"rows={len(all_rows)} output={output_dir}")


if __name__ == "__main__":
    main()
