#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
for path in [ROOT, WORKSPACE_ROOT, WORKSPACE_ROOT / "gui_odyssey_eval"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gui_odyssey_eval.odyssey_action_matching import evaluate_odyssey_action  # noqa: E402
from related_work.har.action_parser import parse_har_output  # noqa: E402
from src.infer.wrapper import CachedOpenAIClient, InferenceRequest  # noqa: E402


PROBES = ("T_screen", "T_act", "T_full")
BUCKETS = ("B1", "B2", "B3", "B4a", "B4b")


def main() -> int:
    args = parse_args()
    inputs_path = resolve(args.inputs)
    error_set_path = resolve(args.error_set)
    output_path = resolve(args.output)
    cost_log = resolve(args.cost_log)
    cache_dir = resolve(args.cache_dir)

    inputs = [row for row in load_jsonl(inputs_path) if row.get("teacher_probe_status") == "queued"]
    if args.limit is not None:
        inputs = inputs[: args.limit]
    completed = load_completed(output_path)
    pending = [row for row in inputs if row_key(row) not in completed]
    scorer_rows = load_error_rows(error_set_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, args.threads)) as executor:
        futures = [
            executor.submit(run_one, row, scorer_rows, args, cache_dir, cost_log)
            for row in pending
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Phase P teacher probes"):
            result = future.result()
            append_jsonl(output_path, result)

    total_done = len(load_completed(output_path))
    print(
        json.dumps(
            {
                "status": "done",
                "output": workspace_relative(output_path),
                "new_rows": len(pending),
                "total_rows": total_done,
                "requested_rows": len(inputs),
                "model": args.model,
                "api_url": args.api_url,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase P N0 teacher probes through src/infer/wrapper.py")
    parser.add_argument("--inputs", default="chorus-n0n1/runs/n0n1_inputs/har_gui_odyssey_latest/offline_analysis/phase_p_teacher_probe_inputs.jsonl")
    parser.add_argument("--error_set", default="chorus-n0n1/runs/n0n1_inputs/har_gui_odyssey_latest/offline_analysis/error_sets/error_set_E_v1.jsonl")
    parser.add_argument("--output", default="chorus-n0n1/runs/n0n1_inputs/har_gui_odyssey_latest/offline_analysis/phase_p_teacher_results.jsonl")
    parser.add_argument("--cache_dir", default="chorus-n0n1/cache/phase_p_teacher")
    parser.add_argument("--cost_log", default="chorus-n0n1/runs/n0n1_inputs/har_gui_odyssey_latest/offline_analysis/phase_p_teacher_cost_log.jsonl")
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--model", default="har-gui-3b-gui-odyssey")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_history_frames", type=int, default=1)
    parser.add_argument("--image_max_pixels", type=int, default=602112)
    return parser.parse_args()


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def workspace_relative(path: Path) -> str:
    try:
        return str(path.relative_to(WORKSPACE_ROOT))
    except ValueError:
        return str(path)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL row in {path} at line {line_number}: {exc}") from exc
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def load_completed(path: Path) -> set[Tuple[str, int]]:
    completed: set[Tuple[str, int]] = set()
    for row in load_jsonl(path):
        completed.add((str(row.get("episode_id", "")), int(row.get("step_idx", 0))))
    return completed


def row_key(row: Dict[str, Any]) -> Tuple[str, int]:
    return (str(row.get("episode_id", "")), int(row.get("step_idx", 0)))


def load_error_rows(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    rows = {}
    for row in load_jsonl(path):
        rows[(str(row.get("episode_id", "")), int(row.get("step_idx", 0)))] = row
    return rows


def run_one(
    input_row: Dict[str, Any],
    scorer_rows: Dict[Tuple[str, int], Dict[str, Any]],
    args: argparse.Namespace,
    cache_dir: Path,
    cost_log: Path,
) -> Dict[str, Any]:
    key = row_key(input_row)
    scorer_row = scorer_rows[key]
    client = CachedOpenAIClient(args.api_url, cache_dir, cost_log, api_key=args.api_key, retries=1, retry_sleep=1.0)
    probe_results: Dict[str, Any] = {}
    for probe_name in PROBES:
        request = build_request(input_row, probe_name, args)
        response = client.chat(request)
        width, height = image_size(resolve(input_row["teacher_inputs"][probe_name]["current_screenshot"]))
        parsed = parse_teacher_output(response.get("text", ""), width, height)
        pred_action = parsed.get("action") if isinstance(parsed.get("action"), dict) else None
        match = score_action(pred_action, scorer_row, width, height)
        probe_results[probe_name] = {
            "text": response.get("text", ""),
            "finish_reason": response.get("finish_reason"),
            "truncated": bool(response.get("truncated", False)),
            "cache_hit": bool(response.get("cache_hit", False)),
            "error": response.get("error"),
            "parsed": parsed,
            "pred_action": pred_action,
            "semantic_match": match["semantic_match"],
            "type_match": match["type_match"],
            "match_error": match["match_error"],
        }
    bucket = assign_bucket(probe_results)
    return {
        "benchmark": input_row.get("benchmark"),
        "episode_id": input_row.get("episode_id"),
        "step_idx": input_row.get("step_idx"),
        "category": input_row.get("category"),
        "bucket": bucket,
        "sampling_weight": 1.0,
        "probes": probe_results,
        "finish_reason": aggregate_finish_reason(probe_results),
        "truncated": any(bool(probe_results[name].get("truncated")) for name in PROBES),
        "error_set_version": input_row.get("error_set_version"),
        "run_scope": input_row.get("run_scope"),
        "sample_name": input_row.get("sample_name"),
    }


def build_request(input_row: Dict[str, Any], probe_name: str, args: argparse.Namespace) -> InferenceRequest:
    probe_input = input_row["teacher_inputs"][probe_name]
    image_paths = [probe_input["current_screenshot"]]
    if probe_name == "T_full":
        history_frames = [item.get("screenshot", "") for item in probe_input.get("observation_history", []) if item.get("screenshot")]
        image_paths = history_frames[-args.max_history_frames :] + image_paths
    content: List[Dict[str, Any]] = []
    for image_path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": image_data_url(resolve(image_path), args.image_max_pixels)}})
    content.append({"type": "text", "text": build_prompt(input_row, probe_name, probe_input)})
    return InferenceRequest(
        model=args.model,
        messages=[
            {"role": "system", "content": "You are a careful GUI action teacher. Return exactly one compact JSON object."},
            {"role": "user", "content": content},
        ],
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
        extra={"stop": [";", "\u000e"]},
    )


def build_prompt(input_row: Dict[str, Any], probe_name: str, probe_input: Dict[str, Any]) -> str:
    history_clause = ""
    if probe_name in {"T_act", "T_full"}:
        history_clause = "\nAction history, generated by the baseline agent:\n" + json.dumps(
            probe_input.get("action_history", []), ensure_ascii=False
        )
    if probe_name == "T_full":
        history_clause += "\nThe image sequence contains the most recent prior observation frame when available, followed by the current screenshot."
    return "\n".join(
        [
            "Predict the next GUI action for this task step.",
            f"Probe: {probe_name}.",
            f"Goal: {probe_input.get('goal', input_row.get('goal', ''))}",
            history_clause,
            "Return a flat action JSON object using this schema:",
            '{"action":"click|long_press|swipe|type|system_button|terminate","coordinate":[x,y],"coordinate2":[x,y],"text":"...","button":"Back|Home|Recent","status":"success|impossible"}',
            "Use only the fields required for the chosen action. For click/long_press use coordinate. For swipe use coordinate and coordinate2. For type use text. For system_button use button. For terminate use status. Do not include text unless action is type. Return one JSON object only; do not repeat it.",
        ]
    )


def image_data_url(path: Path, max_pixels: int) -> str:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    if max_pixels and width * height > max_pixels:
        scale = (max_pixels / (width * height)) ** 0.5
        width = max(1, int(width * scale))
        height = max(1, int(height * scale))
        image = image.resize((width, height), Image.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image.close()
    blob = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{blob}"


def image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def parse_teacher_output(text: str, image_width: int, image_height: int) -> Dict[str, Any]:
    parsed_json = parse_teacher_json(text)
    if isinstance(parsed_json.get("action"), dict):
        return parsed_json
    flat_action = normalize_flat_action_json(parsed_json)
    if flat_action is not None:
        return {"action": flat_action, "alternative_valid_actions": [], "needs_history": "none", "rationale": "parsed from flat action JSON"}
    relaxed_action = parse_relaxed_flat_action(text)
    if relaxed_action is not None:
        return {"action": relaxed_action, "alternative_valid_actions": [], "needs_history": "none", "rationale": "parsed from relaxed flat action JSON"}
    _, answer, parsed_har = parse_har_output(text)
    action = normalize_har_action(answer, parsed_har, image_width, image_height)
    if action is not None:
        return {
            "action": action,
            "alternative_valid_actions": [],
            "needs_history": "none",
            "rationale": "parsed from HAR action output fallback",
            "fallback_parser": "har_action_parser",
        }
    return parsed_json


def parse_teacher_json(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    stripped = stripped.strip().rstrip(";").rstrip("'").strip()
    first_segment = stripped.split(";", 1)[0].strip()
    try:
        value = json.loads(first_segment)
        return value if isinstance(value, dict) else {"parse_error": "json_not_object", "raw": text}
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", first_segment or stripped, flags=re.DOTALL)
        if match:
            try:
                value = json.loads(match.group(0))
                return value if isinstance(value, dict) else {"parse_error": "json_not_object", "raw": text}
            except json.JSONDecodeError as exc:
                return {"parse_error": repr(exc), "raw": text}
        return {"parse_error": "no_json_object", "raw": text}


def normalize_flat_action_json(value: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    action_type = value.get("action")
    if not isinstance(action_type, str):
        return None
    action_type = action_type.split("|", 1)[0].strip()
    if action_type not in {"click", "long_press", "swipe", "type", "system_button", "terminate", "wait", "open"}:
        return None
    action: Dict[str, Any] = {"action": action_type}
    for key in ["coordinate", "coordinate2", "text", "button", "status"]:
        if key in value:
            action[key] = value[key]
    return action


def parse_relaxed_flat_action(text: str) -> Optional[Dict[str, Any]]:
    segment = text.strip().split(";", 1)[0].split("\u000e", 1)[0].strip()
    action_match = re.search(r'"action"\s*:\s*"([^"]+)"', segment)
    if not action_match:
        return None
    action_type = action_match.group(1).split("|", 1)[0].strip()
    if action_type not in {"click", "long_press", "swipe", "type", "system_button", "terminate", "wait", "open"}:
        return None
    action: Dict[str, Any] = {"action": action_type}
    coordinate = extract_relaxed_coordinate(segment, "coordinate")
    coordinate2 = extract_relaxed_coordinate(segment, "coordinate2")
    if coordinate is not None:
        action["coordinate"] = coordinate
    if coordinate2 is not None:
        action["coordinate2"] = coordinate2
    for key in ["text", "button", "status"]:
        value_match = re.search(rf'"{key}"\s*:\s*"([^"]*)"', segment)
        if value_match:
            action[key] = value_match.group(1)
    return action


def extract_relaxed_coordinate(text: str, key: str) -> Optional[List[int]]:
    number = r"-?\d+(?:\.\d+)?"
    list_match = re.search(rf'"{key}"\s*:\s*\[\s*({number})\s*,\s*({number})\s*\]', text)
    if list_match:
        return [int(float(list_match.group(1))), int(float(list_match.group(2)))]
    bare_match = re.search(rf'"{key}"\s*:\s*({number})\s*,\s*({number})', text)
    if bare_match:
        return [int(float(bare_match.group(1))), int(float(bare_match.group(2)))]
    return None


def normalize_har_action(answer: str, parsed: Optional[Dict[str, Any]], width: int, height: int) -> Optional[Dict[str, Any]]:
    text = answer.strip()
    upper = text.upper()
    if parsed:
        action = dict(parsed)
        if action.get("action") == "swipe" and "endCoordinate" in action:
            action["coordinate2"] = action.pop("endCoordinate")
        return action
    match = re.match(r"SCROLL:\s*(UP|DOWN|LEFT|RIGHT)", text, re.IGNORECASE)
    if match:
        return scroll_to_action(match.group(1), width, height)
    if upper == "COMPLETE":
        return {"action": "terminate", "status": "success"}
    if upper == "IMPOSSIBLE":
        return {"action": "terminate", "status": "impossible"}
    if upper in {"BACK", "HOME", "PRESS_RECENT"}:
        button = {"BACK": "Back", "HOME": "Home", "PRESS_RECENT": "Recent"}[upper]
        return {"action": "system_button", "button": button}
    return None


def scroll_to_action(direction: str, width: int, height: int) -> Dict[str, Any]:
    x_mid = width / 2
    y_mid = height / 2
    x_left = width * 0.3
    x_right = width * 0.7
    y_top = height * 0.3
    y_bottom = height * 0.7
    direction = direction.upper()
    if direction == "UP":
        start, end = [x_mid, y_bottom], [x_mid, y_top]
    elif direction == "DOWN":
        start, end = [x_mid, y_top], [x_mid, y_bottom]
    elif direction == "LEFT":
        start, end = [x_right, y_mid], [x_left, y_mid]
    else:
        start, end = [x_left, y_mid], [x_right, y_mid]
    return {"action": "swipe", "coordinate": [int(start[0]), int(start[1])], "coordinate2": [int(end[0]), int(end[1])]}


def score_action(pred_action: Optional[Dict[str, Any]], scorer_row: Dict[str, Any], image_width: int, image_height: int) -> Dict[str, Any]:
    if pred_action is None:
        return {"type_match": False, "semantic_match": False, "match_error": "missing_action"}
    try:
        type_match, semantic_match = evaluate_odyssey_action(
            dict(pred_action),
            dict(scorer_row.get("gt_action") or {}),
            image_width,
            image_height,
        )
        return {"type_match": bool(type_match), "semantic_match": bool(semantic_match), "match_error": None}
    except Exception as exc:
        return {"type_match": False, "semantic_match": False, "match_error": repr(exc)}


def assign_bucket(probe_results: Dict[str, Any]) -> str:
    screen = bool(probe_results["T_screen"].get("semantic_match"))
    act = bool(probe_results["T_act"].get("semantic_match"))
    full = bool(probe_results["T_full"].get("semantic_match"))
    alternatives = any(has_matching_alternative(probe_results[name]) for name in PROBES)
    if screen:
        return "B1"
    if act:
        return "B2"
    if full:
        return "B3"
    if alternatives:
        return "B4a"
    return "B4b"


def has_matching_alternative(probe_result: Dict[str, Any]) -> bool:
    parsed = probe_result.get("parsed") or {}
    alternatives = parsed.get("alternative_valid_actions") if isinstance(parsed, dict) else None
    return isinstance(alternatives, list) and len(alternatives) > 0


def aggregate_finish_reason(probe_results: Dict[str, Any]) -> str:
    reasons = [str(probe_results[name].get("finish_reason") or "") for name in PROBES]
    if any(reason == "error" for reason in reasons):
        return "error"
    if any(reason == "length" for reason in reasons):
        return "length"
    return "+".join(reasons)


if __name__ == "__main__":
    raise SystemExit(main())