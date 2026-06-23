#!/usr/bin/env python3
"""Run on-policy state repair probes against an OpenAI-compatible vLLM server."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))
sys.path.insert(0, str(PROJECT_ROOT / "gui_odyssey_eval"))

from evaluation.qwenvl_utils import image_to_data_url  # noqa: E402
from gui_odyssey_eval.odyssey_action_matching import evaluate_odyssey_action  # noqa: E402


JsonDict = dict[str, Any]
CONDITIONS = ("screen_only", "correct_task_state", "wrong_task_state", "full_history")


SYSTEM_PROMPT = """You are an Android GUI control agent.
Return exactly one action in this format and no extra text:
<action>{"action": ...}</action>

Coordinates must be normalized to the GUI-Odyssey [0,1000] coordinate system.
Allowed actions:
- {"action":"click", "coordinate":[x,y]}
- {"action":"long_press", "coordinate":[x,y]}
- {"action":"swipe", "coordinate":[x1,y1], "coordinate2":[x2,y2]}
- {"action":"type", "text":"..."}
- {"action":"system_button", "button":"Back"|"Home"|"Menu"}
- {"action":"wait", "time":2}
- {"action":"terminate", "status":"success"}
""".strip()


def iter_jsonl(path: Path) -> Iterable[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def safe_parse_action(text: str) -> tuple[bool, JsonDict | None, str]:
    match = re.search(r"<action>\s*(\{.*?\})\s*</action>", text, re.DOTALL)
    if not match:
        match = re.search(r"(\{[^{}]*\})", text, re.DOTALL)
    if not match:
        return False, None, "no_json_action"
    raw = match.group(1)
    while raw.endswith("}}"):
        raw = raw[:-1]
    try:
        action = json.loads(raw)
    except Exception as exc:
        return False, None, repr(exc)
    return True, action, ""


def condition_state_text(probe: JsonDict, condition: str) -> str:
    if condition == "screen_only":
        return probe["screen_only_state_text"]
    if condition == "correct_task_state":
        return probe["correct_task_state_text"]
    if condition == "wrong_task_state":
        return probe["wrong_task_state_text"]
    if condition == "full_history":
        return probe["full_history_text"] + "\n\n" + probe["correct_task_state_text"]
    raise KeyError(condition)


def build_messages(probe: JsonDict, condition: str) -> list[JsonDict]:
    state = condition_state_text(probe, condition)
    user_text = f"""Global task:
{probe['goal']}

Current step index: {probe['step_index']} of {probe['num_steps']}.

{state}

Look at the current screenshot and choose the next GUI action. Do not explain. Return exactly one <action> JSON block.
""".strip()
    with Image.open(probe["screenshot"]) as image:
        data_url = image_to_data_url(image)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]


def call_model(client: OpenAI, model_name: str, messages: list[JsonDict], max_tokens: int, temperature: float) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"top_k": 1},
    )
    return response.choices[0].message.content or ""


def eval_condition(probe: JsonDict, condition: str, client: OpenAI, model_name: str, max_tokens: int, temperature: float) -> JsonDict:
    error = ""
    raw_output = ""
    pred_action = None
    parse_ok = False
    type_match = False
    value_match = False
    try:
        raw_output = call_model(client, model_name, build_messages(probe, condition), max_tokens, temperature)
        parse_ok, pred_action, error = safe_parse_action(raw_output)
        if parse_ok and pred_action is not None:
            type_match, value_match = evaluate_odyssey_action(pred_action, probe["check_options"], 1000, 1000)
    except Exception as exc:
        error = repr(exc)
    return {
        "probe_id": probe["probe_id"],
        "condition": condition,
        "episode_id": probe["episode_id"],
        "step_index": probe["step_index"],
        "num_steps": probe["num_steps"],
        "category": probe.get("category"),
        "probe_kind": probe.get("probe_kind"),
        "rollout_failure_family": probe.get("rollout_failure_family"),
        "gt_action": probe.get("gt_action"),
        "rollout_pred_action": probe.get("rollout_pred_action"),
        "pred_action": pred_action,
        "raw_output": raw_output,
        "error": error,
        "parse_ok": parse_ok,
        "type_match": bool(type_match),
        "value_match": bool(value_match),
    }


def run_probe(probe: JsonDict, endpoint: str, model_name: str, max_tokens: int, temperature: float) -> list[JsonDict]:
    client = OpenAI(api_key="EMPTY", base_url=endpoint, timeout=600)
    return [eval_condition(probe, condition, client, model_name, max_tokens, temperature) for condition in CONDITIONS]


def summarize(rows: list[JsonDict]) -> JsonDict:
    by_probe: dict[str, dict[str, JsonDict]] = defaultdict(dict)
    by_condition: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        by_probe[row["probe_id"]][row["condition"]] = row
        by_condition[row["condition"]].append(row)

    condition_stats = {}
    for condition in CONDITIONS:
        selected = by_condition[condition]
        den = len(selected)
        condition_stats[condition] = {
            "rows": den,
            "parse_ok": sum(bool(row.get("parse_ok")) for row in selected),
            "type_match": sum(bool(row.get("type_match")) for row in selected),
            "value_match": sum(bool(row.get("value_match")) for row in selected),
            "parse_rate": sum(bool(row.get("parse_ok")) for row in selected) / den if den else 0,
            "type_match_rate": sum(bool(row.get("type_match")) for row in selected) / den if den else 0,
            "value_match_rate": sum(bool(row.get("value_match")) for row in selected) / den if den else 0,
        }

    rescue = Counter()
    rescue_by_family: dict[str, Counter[str]] = defaultdict(Counter)
    for probe_id, conds in by_probe.items():
        if not all(condition in conds for condition in CONDITIONS):
            continue
        screen = bool(conds["screen_only"].get("value_match"))
        correct = bool(conds["correct_task_state"].get("value_match"))
        wrong = bool(conds["wrong_task_state"].get("value_match"))
        full = bool(conds["full_history"].get("value_match"))
        family = str(conds["screen_only"].get("rollout_failure_family"))
        if not screen:
            rescue["screen_only_wrong"] += 1
            rescue_by_family[family]["screen_only_wrong"] += 1
        if not screen and correct:
            rescue["state_rescue"] += 1
            rescue_by_family[family]["state_rescue"] += 1
        if not screen and correct and not wrong:
            rescue["clean_state_rescue"] += 1
            rescue_by_family[family]["clean_state_rescue"] += 1
        if not screen and wrong:
            rescue["wrong_state_rescue"] += 1
            rescue_by_family[family]["wrong_state_rescue"] += 1
        if not screen and full:
            rescue["full_history_rescue"] += 1
            rescue_by_family[family]["full_history_rescue"] += 1
        if not screen and not correct and not full:
            rescue["local_unsolved"] += 1
            rescue_by_family[family]["local_unsolved"] += 1
        if screen:
            rescue["rollout_drift_or_recoverable_screen_only"] += 1
            rescue_by_family[family]["rollout_drift_or_recoverable_screen_only"] += 1

    den = rescue["screen_only_wrong"]
    return {
        "probes": len(by_probe),
        "condition_stats": condition_stats,
        "rescue": dict(rescue),
        "state_rescue_rate": rescue["state_rescue"] / den if den else 0,
        "clean_state_rescue_rate": rescue["clean_state_rescue"] / den if den else 0,
        "specificity_gap": condition_stats["correct_task_state"]["value_match_rate"] - condition_stats["wrong_task_state"]["value_match_rate"],
        "rescue_by_family": {family: dict(counts) for family, counts in rescue_by_family.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run on-policy state repair probes")
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="qwen3.5-9b")
    parser.add_argument("--endpoint", default=os.environ.get("QWENVL_ENDPOINT", "http://localhost:8000/v1"))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    probes = list(iter_jsonl(args.probes))
    if args.limit > 0:
        probes = probes[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "probe_results.jsonl"
    rows = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(run_probe, probe, args.endpoint, args.model_name, args.max_tokens, args.temperature) for probe in probes]
        for index, future in enumerate(as_completed(futures), start=1):
            batch = future.result()
            rows.extend(batch)
            with out_path.open("a", encoding="utf-8") as handle:
                for row in batch:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if index % 25 == 0:
                print(f"completed_probes={index}/{len(probes)}")

    summary = summarize(rows)
    (args.output_dir / "probe_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()