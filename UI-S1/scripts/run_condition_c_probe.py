#!/usr/bin/env python3
"""Run Condition C paired baseline/injection queries."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))
sys.path.insert(0, str(PROJECT_ROOT / "gui_odyssey_eval"))

import evaluation.qwenvl_utils as qwen_utils  # noqa: E402
from gui_odyssey_eval.eval_ar_trajectory import safe_parse_response  # noqa: E402
from gui_odyssey_eval.odyssey_action_matching import evaluate_odyssey_action  # noqa: E402
from x.data.agent.json import JsonFormat  # noqa: E402
from x.data.agent.space.std_space import RAW_SPACE  # noqa: E402
from x.qwen.data_format import slim_messages  # noqa: E402


JsonDict = dict[str, Any]
write_lock = Lock()


def iter_jsonl(path: Path) -> Iterable[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_dataset_by_id(path: Path) -> dict[str, JsonDict]:
    return {str(row["episode_id"]): row for row in iter_jsonl(path)}


def action_response(action: JsonDict) -> str:
    return "<action>\n" + json.dumps(action, ensure_ascii=False) + "\n</action>"


def build_state(
    fm: JsonFormat,
    episode: JsonDict,
    target_step: int,
    inject_actions_by_step: dict[int, JsonDict] | None,
) -> JsonDict:
    state = None
    previous_response = None
    for step_index in range(target_step + 1):
        state = fm.gen_next_round(episode, state, previous_model_response=previous_response)
        if state is None:
            raise RuntimeError(f"state became None before target_step={target_step}")
        if step_index == target_step:
            return state
        if inject_actions_by_step is not None and step_index in inject_actions_by_step:
            action = inject_actions_by_step[step_index]
        else:
            action = episode["steps"][step_index]["action_content"]
        if not isinstance(action, dict):
            raise TypeError(f"history action is not dict at step {step_index}: {action!r}")
        previous_response = action_response(action)
    raise AssertionError("unreachable")


def call_once(fm: JsonFormat, episode: JsonDict, state: JsonDict, target_step: int, args: argparse.Namespace) -> JsonDict:
    messages = slim_messages(messages=state["messages"], num_image_limit=args.n_history_image_limit)
    _, _width, _height, resized_width, resized_height = qwen_utils.find_last_image_ele(messages)
    raw_output = qwen_utils.call_mobile_agent_vllm(messages=messages, model_name=args.model_name)
    pred_action = None
    parse_ok = False
    parse_error = ""
    type_match = False
    value_match = False
    try:
        parsed = safe_parse_response(fm, raw_output)
        pred_action = parsed["action_content"]
        type_match, value_match = evaluate_odyssey_action(
            pred_action,
            episode["steps"][target_step]["check_options"],
            resized_width,
            resized_height,
        )
        parse_ok = True
    except Exception as exc:
        parse_error = repr(exc)
    return {
        "parse_ok": bool(parse_ok),
        "parse_error": parse_error,
        "pred_action": pred_action,
        "value_match": bool(value_match),
        "type_match": bool(type_match),
        "raw_response_prefix": str(raw_output or "")[:500],
    }


def sample_condition(
    fm: JsonFormat,
    episode: JsonDict,
    target_step: int,
    inject_actions_by_step: dict[int, JsonDict] | None,
    args: argparse.Namespace,
) -> list[JsonDict]:
    outputs = []
    state = build_state(fm, episode, target_step, inject_actions_by_step)
    for sample_index in range(args.n_samples):
        row = call_once(fm, episode, state, target_step, args)
        row["sample_index"] = sample_index
        outputs.append(row)
    return outputs


def mean_bool(samples: list[JsonDict], key: str) -> float:
    return sum(bool(sample.get(key)) for sample in samples) / len(samples) if samples else 0.0


def compact_samples(samples: list[JsonDict]) -> list[JsonDict]:
    return [
        {
            "sample_index": sample.get("sample_index"),
            "parse_ok": sample.get("parse_ok"),
            "value_match": sample.get("value_match"),
            "type_match": sample.get("type_match"),
            "pred_action": sample.get("pred_action"),
            "parse_error": sample.get("parse_error", ""),
        }
        for sample in samples
    ]


def normalized_injections(pair: JsonDict) -> dict[int, JsonDict] | None:
    if "inject_actions_by_step" in pair:
        return {int(step): action for step, action in (pair.get("inject_actions_by_step") or {}).items()}
    if pair.get("source_step") is not None and isinstance(pair.get("inject_action"), dict):
        return {int(pair["source_step"]): pair["inject_action"]}
    return None


def get_baseline_samples(
    fm: JsonFormat,
    episode: JsonDict,
    target_step: int,
    cache_key: tuple[str, int],
    baseline_cache: dict[tuple[str, int], list[JsonDict]],
    baseline_locks: dict[tuple[str, int], Lock],
    baseline_locks_lock: Lock,
    args: argparse.Namespace,
) -> list[JsonDict]:
    with baseline_locks_lock:
        key_lock = baseline_locks.setdefault(cache_key, Lock())
    with key_lock:
        if cache_key not in baseline_cache:
            baseline_cache[cache_key] = sample_condition(fm, episode, target_step, None, args)
        return baseline_cache[cache_key]


def process_pair(
    pair: JsonDict,
    episode: JsonDict,
    baseline_cache: dict[tuple[str, int], list[JsonDict]],
    baseline_locks: dict[tuple[str, int], Lock],
    baseline_locks_lock: Lock,
    args: argparse.Namespace,
) -> JsonDict:
    fm = JsonFormat(RAW_SPACE, add_thought=True, force_add_thought=True)
    episode_id = str(pair["episode_id"])
    target_step = int(pair["target_step"])
    cache_key = (episode_id, target_step)
    baseline_samples = get_baseline_samples(fm, episode, target_step, cache_key, baseline_cache, baseline_locks, baseline_locks_lock, args)
    injected_samples = sample_condition(fm, episode, target_step, normalized_injections(pair), args)
    baseline_value = mean_bool(baseline_samples, "value_match")
    baseline_type = mean_bool(baseline_samples, "type_match")
    injected_value = mean_bool(injected_samples, "value_match")
    injected_type = mean_bool(injected_samples, "type_match")
    return {
        **pair,
        "n_samples": args.n_samples,
        "baseline_value_mean": baseline_value,
        "injected_value_mean": injected_value,
        "gap_value": baseline_value - injected_value,
        "baseline_type_mean": baseline_type,
        "injected_type_mean": injected_type,
        "gap_type": baseline_type - injected_type,
        "baseline_samples": compact_samples(baseline_samples),
        "injected_samples": compact_samples(injected_samples),
    }


def write_jsonl_append(path: Path, row: JsonDict) -> None:
    with write_lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_completed_pair_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("pair_id")) for row in iter_jsonl(path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Condition C pairs")
    parser.add_argument("--jsonl-file", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="qwen3.5-9b")
    parser.add_argument("--endpoint", default=os.environ.get("QWENVL_ENDPOINT", "http://localhost:8000/v1"))
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--n-history-image-limit", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-pairs", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qwen_utils.END_POINT = args.endpoint
    episodes = load_dataset_by_id(args.jsonl_file)
    pairs = list(iter_jsonl(args.pairs))
    if args.limit_pairs > 0:
        pairs = pairs[: args.limit_pairs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        completed = load_completed_pair_ids(args.output)
        pairs = [pair for pair in pairs if str(pair["pair_id"]) not in completed]
        print(f"resume: skipped_pairs={len(completed)} remaining={len(pairs)}")
    elif args.output.exists():
        args.output.unlink()
    baseline_cache: dict[tuple[str, int], list[JsonDict]] = {}
    baseline_locks: dict[tuple[str, int], Lock] = {}
    baseline_locks_lock = Lock()
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_pair, pair, episodes[str(pair["episode_id"])], baseline_cache, baseline_locks, baseline_locks_lock, args): pair
            for pair in pairs
        }
        for completed_count, future in enumerate(as_completed(futures), start=1):
            pair = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {**pair, "error": repr(exc), "n_samples": args.n_samples}
            write_jsonl_append(args.output, result)
            if completed_count % 50 == 0:
                print(f"completed_pairs={completed_count}/{len(pairs)}")
    manifest = {
        "jsonl_file": str(args.jsonl_file),
        "pairs": str(args.pairs),
        "output": str(args.output),
        "n_samples": args.n_samples,
        "model_name": args.model_name,
        "endpoint": args.endpoint,
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()