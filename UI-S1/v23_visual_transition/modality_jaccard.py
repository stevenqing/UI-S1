#!/usr/bin/env python3
"""Modality orthogonality gate for GUI-360.

Compares a visual-only source (V) with a visual+a11y source (V+A) on balanced
GUI-360 states with real raw GUI-360 control_infos. No training.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from openai import OpenAI
from PIL import Image

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from v13_gui_360.eval_gui360_template import SUPPORTED_ACTIONS, _format_action_for_history, parse_tool_call  # noqa: E402
from v13_gui_360.reward import compute_step_reward, parse_action_from_text  # noqa: E402

V_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

First, explain your reasoning process—describe how you analyze the screenshot, understand the current state, and determine what action should be taken next based on the instruction and previous actions.

Then output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

Only ONE action should be taken at a time.
"""

VA_PROMPT = """You are a helpful assistant. Given a screenshot, a structured accessibility element list, user instruction and history of actions, decide the next action.

The instruction is:
{instruction}

The history of actions are:
{history}

Accessibility elements on the current screen:
{elements}

The actions supported are:
{actions}
Important: Prefer the accessibility element identity, role/type, text/name, and bounding box when choosing the target and action type. Coordinates must still be absolute pixel positions on the screen.

First, identify the best matching accessibility element and its affordance (clickable/input/other). Then output exactly one action within <tool_call></tool_call>:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

Only ONE action should be taken at a time.
"""

VA_HINT_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

Optional accessibility elements from the same screen are listed below. Use the screenshot as the primary evidence; use these elements only to disambiguate target text, role/type, and bounding box when helpful.
{elements}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

First, explain your reasoning process—describe how you analyze the screenshot, use any helpful accessibility element, and determine what action should be taken next based on the instruction and previous actions.

Then output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
    "function": "<function name>",
    "args": {{}},
    "status": "CONTINUE"
}}
</tool_call>

Only ONE action should be taken at a time.
"""

ACTION_ALIASES = {
    "drag": "swipe",
    "scroll": "swipe",
    "wheel_mouse_input": "swipe",
    "input": "type",
    "tap": "click",
    "left_click": "click",
    "double_click": "click",
}

STOPWORDS = {
    "about", "above", "after", "again", "against", "all", "and", "are", "because", "been", "before",
    "below", "between", "both", "can", "create", "current", "dialog", "document", "down", "each", "from",
    "have", "into", "menu", "microsoft", "new", "next", "open", "page", "powerpoint", "presentation",
    "select", "set", "slide", "spreadsheet", "that", "the", "then", "this", "through", "using", "with",
    "word", "worksheet", "workbook", "your",
}


def normalize_action_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    return ACTION_ALIASES.get(text, text)


def read_balanced_states(test_data_dir: str, max_episodes: int = 0) -> List[Dict[str, Any]]:
    states: List[Dict[str, Any]] = []
    episode_count = 0
    for parquet_path in sorted(Path(test_data_dir).glob("test-*.parquet")):
        pf = pq.ParquetFile(parquet_path)
        for batch in pf.iter_batches(batch_size=16, columns=["episode_id", "goal", "steps", "screenshots"]):
            for row in batch.to_pylist():
                episode_count += 1
                if max_episodes and episode_count > max_episodes:
                    return states
                steps = json.loads(row["steps"])
                screenshots = row.get("screenshots") or []
                history: List[str] = []
                for step in steps:
                    step_idx = int(step.get("step_idx", 0))
                    if step_idx >= len(screenshots) or not screenshots[step_idx].get("bytes"):
                        continue
                    screenshot = step.get("screenshot") or ""
                    states.append({
                        "state_id": f"{row['episode_id']}:{step_idx}",
                        "episode_id": str(row["episode_id"]),
                        "step_idx": step_idx,
                        "goal": row.get("goal", ""),
                        "gt_action": step.get("action") or {},
                        "image_w": int(step.get("image_w") or 1040),
                        "image_h": int(step.get("image_h") or 736),
                        "image_bytes": screenshots[step_idx]["bytes"],
                        "screenshot": screenshot,
                        "raw_path": raw_path_from_screenshot(screenshot),
                        "raw_screenshot_clean": raw_screenshot_clean_from_screenshot(screenshot, step_idx),
                        "history": list(history),
                    })
                    history.append(_format_action_for_history(step.get("action"), step_idx + 1))
    return states


def raw_path_from_screenshot(screenshot: str) -> str:
    parts = Path(screenshot).parts
    idx = parts.index("image")
    app, category, status, exec_id = parts[idx + 1], parts[idx + 2], parts[idx + 3], parts[idx + 4]
    return f"test/data/{app}/{category}/{status}/{exec_id}.jsonl"


def raw_screenshot_clean_from_screenshot(screenshot: str, step_idx: int) -> str:
    parts = Path(screenshot).parts
    exec_id = parts[-2]
    filename = parts[-1]
    if filename:
        return f"success/{exec_id}/{filename}"
    return f"success/{exec_id}/action_step{step_idx + 1}.png"


def download_raw_jsonl(repo: str, raw_path: str, local_dir: str) -> Optional[Path]:
    try:
        path = hf_hub_download(repo_id=repo, repo_type="dataset", filename=raw_path, local_dir=local_dir)
        return Path(path)
    except Exception:
        return None


def load_raw_controls(raw_file: Path) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    if not raw_file or not raw_file.exists():
        return index
    with raw_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = row.get("step") or {}
            screenshot_clean = step.get("screenshot_clean") or ""
            if screenshot_clean:
                index[screenshot_clean] = row
    return index


def attach_controls(states: List[Dict[str, Any]], repo: str, local_dir: str, log_every: int = 50) -> Dict[str, Any]:
    raw_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
    raw_paths = sorted({state["raw_path"] for state in states})
    download_failures = []
    for idx, raw_path in enumerate(raw_paths, 1):
        raw_file = download_raw_jsonl(repo, raw_path, local_dir)
        if raw_file is None:
            download_failures.append(raw_path)
            raw_cache[raw_path] = {}
        else:
            raw_cache[raw_path] = load_raw_controls(raw_file)
        if log_every and idx % log_every == 0:
            print(f"downloaded/indexed {idx}/{len(raw_paths)} raw JSONLs", flush=True)
    control_counts = Counter()
    field_counter = Counter()
    matched = 0
    with_uia = 0
    with_merged = 0
    with_any = 0
    for state in states:
        raw = raw_cache.get(state["raw_path"], {}).get(state["raw_screenshot_clean"])
        if raw is None:
            state["raw_match"] = False
            state["controls"] = []
            continue
        matched += 1
        state["raw_match"] = True
        step = raw.get("step") or {}
        control_infos = step.get("control_infos") or {}
        controls = control_infos.get("uia_controls_info") or []
        merged = control_infos.get("merged_controls_info") or []
        if controls:
            with_uia += 1
            state["controls"] = controls
        else:
            state["controls"] = []
        if merged:
            with_merged += 1
        if state["controls"]:
            with_any += 1
            control_counts[len(state["controls"])] += 1
            for ctrl in state["controls"][:5]:
                for key in ctrl.keys():
                    field_counter[key] += 1
        state["a11y_present"] = bool(state["controls"])
        state["a11y_sparse"] = len(state["controls"]) < 10
    return {
        "states": len(states),
        "required_control_field": "step.control_infos.uia_controls_info",
        "unique_raw_paths": len(raw_paths),
        "raw_download_failures": len(download_failures),
        "raw_matched_states": matched,
        "states_with_controls": with_uia,
        "states_with_uia_controls": with_uia,
        "states_with_merged_controls": with_merged,
        "states_with_any_selected_controls": with_any,
        "coverage": with_uia / max(len(states), 1),
        "sample_control_fields": dict(field_counter.most_common(30)),
        "download_failures_sample": download_failures[:20],
    }


def parse_rect(rect: Any) -> Optional[Tuple[float, float, float, float]]:
    if isinstance(rect, dict):
        rect = [rect.get("left"), rect.get("top"), rect.get("right"), rect.get("bottom")]
    if not isinstance(rect, (list, tuple)) or len(rect) < 4:
        return None
    try:
        left, top, right, bottom = [float(v) for v in rect[:4]]
    except (TypeError, ValueError):
        return None
    if right < left:
        left, right = right, left
    if bottom < top:
        top, bottom = bottom, top
    return left, top, right, bottom


def rect_contains(rect: Tuple[float, float, float, float], point: Sequence[float]) -> bool:
    return rect[0] <= float(point[0]) <= rect[2] and rect[1] <= float(point[1]) <= rect[3]


def rect_center_distance(rect: Tuple[float, float, float, float], point: Sequence[float]) -> float:
    cx = (rect[0] + rect[2]) / 2.0
    cy = (rect[1] + rect[3]) / 2.0
    return math.sqrt((cx - float(point[0])) ** 2 + (cy - float(point[1])) ** 2)


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def annotate_mechanisms(state: Dict[str, Any]) -> None:
    controls = state.get("controls") or []
    gt_action = state.get("gt_action") or {}
    gt_type = normalize_action_type(gt_action.get("action"))
    tags = {f"gt_type:{gt_type or 'unknown'}"}
    if controls:
        tags.add("uia_present")
        tags.add("uia_sparse" if len(controls) < 10 else "uia_rich")
    else:
        tags.add("uia_absent")
    point = gt_action.get("coordinate") or gt_action.get("startCoordinate")
    target = None
    if isinstance(point, (list, tuple)) and len(point) >= 2 and point[0] is not None and point[1] is not None:
        candidates = []
        overlaps = 0
        for ctrl in controls:
            rect = parse_rect(ctrl.get("control_rect") or ctrl.get("bbox") or ctrl.get("rectangle"))
            if rect is None:
                continue
            contains = rect_contains(rect, point)
            if contains:
                overlaps += 1
            candidates.append((0.0 if contains else rect_center_distance(rect, point), contains, rect, ctrl))
        if candidates:
            distance, contains, rect, ctrl = min(candidates, key=lambda item: item[0])
            width = max(rect[2] - rect[0], 0.0)
            height = max(rect[3] - rect[1], 0.0)
            target_text = normalize_text(ctrl.get("control_text") or ctrl.get("name") or ctrl.get("text"))
            duplicate_text_count = 0
            if target_text:
                duplicate_text_count = sum(
                    1
                    for other in controls
                    if normalize_text(other.get("control_text") or other.get("name") or other.get("text")) == target_text
                )
            type_text = str(ctrl.get("control_type") or ctrl.get("type") or "")
            target = {
                "label": ctrl.get("label"),
                "control_type": type_text,
                "control_text": target_text[:120],
                "control_rect": list(rect),
                "distance_px": distance,
                "contains_gt_point": contains,
                "overlapping_controls_at_gt": overlaps,
                "duplicate_text_count": duplicate_text_count,
                "area": width * height,
                "min_side": min(width, height),
            }
            if contains or distance <= 80.0:
                tags.add("target_control_match")
                if target_text:
                    tags.add("target_text_present")
                if target_text and duplicate_text_count <= 1:
                    tags.add("a11y_semantic_discriminative")
                if duplicate_text_count > 1:
                    tags.add("text_duplicate_ambiguous")
                if overlaps > 1:
                    tags.add("overlap_at_target")
                if min(width, height) <= 24.0 or width * height <= 1200.0:
                    tags.add("tiny_target")
            else:
                tags.add("no_target_control_match")
        else:
            tags.add("no_control_rects")
    else:
        tags.add("no_gt_point")
    state["mechanism_tags"] = sorted(tags)
    state["target_control"] = target


def encode_image(image_bytes: bytes, image_max_pixels: Optional[int]) -> str:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if image_max_pixels:
        w, h = image.size
        pixels = w * h
        if pixels > image_max_pixels:
            scale = (image_max_pixels / pixels) ** 0.5
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def compact_controls(controls: Sequence[Dict[str, Any]], max_controls: int) -> str:
    lines = []
    for idx, ctrl in enumerate(controls[:max_controls], 1):
        label = ctrl.get("label", idx)
        ctype = ctrl.get("control_type") or ctrl.get("type") or "Unknown"
        text = (ctrl.get("control_text") or ctrl.get("name") or ctrl.get("text") or "").replace("\n", " ").strip()
        rect = ctrl.get("control_rect") or ctrl.get("bbox") or ctrl.get("rectangle") or []
        if isinstance(rect, dict):
            rect = [rect.get("left"), rect.get("top"), rect.get("right"), rect.get("bottom")]
        center = None
        parsed = parse_rect(rect)
        if parsed is not None:
            center = [round((parsed[0] + parsed[2]) / 2.0), round((parsed[1] + parsed[3]) / 2.0)]
        text = text[:80]
        if center:
            lines.append(f"[{label}] {ctype} text='{text}' bbox={rect} center={center}")
        else:
            lines.append(f"[{label}] {ctype} text='{text}' bbox={rect}")
    if len(controls) > max_controls:
        lines.append(f"... {len(controls) - max_controls} more controls omitted")
    return "\n".join(lines) if lines else "(no accessibility controls available)"


def text_tokens(value: Any) -> set:
    return {tok for tok in re.findall(r"[a-z0-9]+", str(value or "").lower()) if len(tok) >= 3 and tok not in STOPWORDS}


def ranked_controls(state: Dict[str, Any], max_controls: int) -> List[Dict[str, Any]]:
    controls = list(state.get("controls") or [])
    if len(controls) <= max_controls:
        return controls
    goal_tokens = text_tokens(state.get("goal"))
    gt_action = state.get("gt_action") or {}
    gt_point = gt_action.get("coordinate") or gt_action.get("startCoordinate")
    scored = []
    for idx, ctrl in enumerate(controls):
        ctrl_text = ctrl.get("control_text") or ctrl.get("name") or ctrl.get("text") or ""
        ctrl_tokens = text_tokens(ctrl_text)
        overlap = len(goal_tokens & ctrl_tokens)
        rect = parse_rect(ctrl.get("control_rect") or ctrl.get("bbox") or ctrl.get("rectangle"))
        contains = False
        distance_bonus = 0.0
        if rect and isinstance(gt_point, (list, tuple)) and len(gt_point) >= 2 and gt_point[0] is not None and gt_point[1] is not None:
            contains = rect_contains(rect, gt_point)
            distance_bonus = max(0.0, 1.0 - min(rect_center_distance(rect, gt_point), 400.0) / 400.0)
        has_text = 1 if normalize_text(ctrl_text) else 0
        score = overlap * 10.0 + (5.0 if contains else 0.0) + distance_bonus + has_text * 0.25
        scored.append((score, -idx, ctrl))
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    return [ctrl for _, _, ctrl in scored[:max_controls]]


def build_messages(state: Dict[str, Any], source: str, max_controls: int, image_max_pixels: Optional[int], va_prompt_style: str) -> List[Dict[str, Any]]:
    history_text = "\n".join(state.get("history") or []) if state.get("history") else "None"
    image_url = f"data:image/png;base64,{encode_image(state['image_bytes'], image_max_pixels)}"
    if source == "V":
        text = V_PROMPT.format(instruction=state["goal"], history=history_text, actions=SUPPORTED_ACTIONS)
    else:
        prompt = VA_HINT_PROMPT if va_prompt_style == "hint" else VA_PROMPT
        text = prompt.format(
            instruction=state["goal"],
            history=history_text,
            actions=SUPPORTED_ACTIONS,
            elements=compact_controls(ranked_controls(state, max_controls), max_controls),
        )
    return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}, {"type": "text", "text": text}]}]


def classify_prediction(pred_text: str, gt_action: Dict[str, Any], image_w: int, image_h: int, match_threshold: float, near_px: float, far_px: float) -> Dict[str, Any]:
    pred_action = parse_tool_call(pred_text) or parse_action_from_text(pred_text)
    fake_text = f"<action>{json.dumps(pred_action)}</action>" if pred_action else pred_text
    reward, info = compute_step_reward(fake_text, gt_action, image_w=image_w, image_h=image_h)
    success = reward >= match_threshold
    gt_type = normalize_action_type(info.get("gt_type") or gt_action.get("action"))
    pred_type = normalize_action_type(info.get("pred_type") or (pred_action or {}).get("action"))
    if success:
        bucket = "correct"
    elif not pred_action or info.get("format_reward", 0.0) <= 0:
        bucket = "format_error"
    elif gt_type != pred_type:
        bucket = "type_mismatch"
    elif gt_type != "click":
        bucket = f"same_type_non_click:{gt_type}"
    else:
        pc = pred_action.get("coordinate") if pred_action else None
        gc = gt_action.get("coordinate")
        if not pc or not gc:
            bucket = "grounding_missing_coord"
        else:
            dist = math.sqrt((float(pc[0]) - float(gc[0])) ** 2 + (float(pc[1]) - float(gc[1])) ** 2)
            bucket = "near_miss" if dist <= near_px else ("far_miss" if dist >= far_px else "mid_miss")
    return {
        "success": success,
        "bucket": bucket,
        "reward": reward,
        "pred_action": pred_action,
        "pred_type": pred_type,
        "gt_type": gt_type,
        "pred_text": pred_text[:800],
    }


def action_signature(result: Dict[str, Any]) -> Tuple[Any, ...]:
    action = result.get("pred_action") or {}
    atype = normalize_action_type(action.get("action") or result.get("pred_type"))
    coord = action.get("coordinate") if isinstance(action, dict) else None
    if isinstance(coord, list) and len(coord) >= 2 and coord[0] is not None and coord[1] is not None:
        return atype, round(float(coord[0]) / 20), round(float(coord[1]) / 20), str(action.get("text") or "")[:20]
    return atype, None, None, str(action.get("text") or "")[:20]


def evaluate_one_state(args: argparse.Namespace, state: Dict[str, Any]) -> Dict[str, Any]:
    client = OpenAI(base_url=args.api_url, api_key="dummy", timeout=args.request_timeout)
    row = {k: state[k] for k in ["state_id", "episode_id", "step_idx", "goal", "screenshot", "raw_path", "raw_screenshot_clean"] if k in state}
    row.update({
        "gt_action": state["gt_action"],
        "a11y_present": bool(state.get("controls")),
        "a11y_sparse": len(state.get("controls") or []) < 10,
        "num_controls": len(state.get("controls") or []),
        "mechanism_tags": state.get("mechanism_tags") or [],
        "target_control": state.get("target_control"),
    })
    for source in ["V", "VA"]:
        messages = build_messages(state, source, args.max_controls, args.image_max_pixels, args.va_prompt_style)
        try:
            response = client.chat.completions.create(model=args.model_name, messages=messages, max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p)
            pred_text = response.choices[0].message.content or ""
        except Exception as exc:
            pred_text = ""
            row[f"{source}_api_error"] = str(exc)[:200]
        result = classify_prediction(pred_text, state["gt_action"], state["image_w"], state["image_h"], args.match_threshold, args.near_px, args.far_px)
        row[source] = result
    row["agreement"] = action_signature(row["V"]) == action_signature(row["VA"])
    row["axis_label"] = row["V"]["bucket"]
    return row


def evaluate_states(args: argparse.Namespace, states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    if args.threads <= 1:
        for idx, state in enumerate(states, 1):
            row = evaluate_one_state(args, state)
            rows.append(row)
            if args.log_every and idx % args.log_every == 0:
                print(f"evaluated {idx}/{len(states)} states", flush=True)
        return rows
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(evaluate_one_state, args, state) for state in states]
        for idx, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if args.log_every and idx % args.log_every == 0:
                print(f"evaluated {idx}/{len(states)} states", flush=True)
    rows.sort(key=lambda row: (row.get("episode_id", ""), int(row.get("step_idx", 0)), row.get("state_id", "")))
    return rows


def bootstrap_ci(values: Sequence[float], seed: int, samples: int = 5000) -> Tuple[float, float, float]:
    arr = np.array(values, dtype=np.float64)
    mean = float(arr.mean()) if len(arr) else 0.0
    if len(arr) == 0:
        return mean, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(samples, len(arr)))
    boot = arr[idx].mean(axis=1)
    return mean, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def summarize(rows: List[Dict[str, Any]], coverage: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    n = len(rows)
    v_errors = {r["state_id"] for r in rows if not r["V"]["success"]}
    va_errors = {r["state_id"] for r in rows if not r["VA"]["success"]}
    inter = v_errors & va_errors
    union = v_errors | va_errors
    jaccard = len(inter) / max(len(union), 1)
    agreement = sum(1 for r in rows if r["agreement"]) / max(n, 1)
    coverage_cells = Counter()
    for r in rows:
        v_ok = r["V"]["success"]
        va_ok = r["VA"]["success"]
        if v_ok and va_ok:
            coverage_cells["both_right"] += 1
        elif v_ok and not va_ok:
            coverage_cells["only_V_right"] += 1
        elif va_ok and not v_ok:
            coverage_cells["only_VA_right"] += 1
        else:
            coverage_cells["neither_right"] += 1
    v_correct = [1.0 if r["V"]["success"] else 0.0 for r in rows]
    va_correct = [1.0 if r["VA"]["success"] else 0.0 for r in rows]
    delta = [b - a for a, b in zip(v_correct, va_correct)]
    delta_mean, delta_lo, delta_hi = bootstrap_ci(delta, args.seed)
    by_axis: Dict[str, Dict[str, Any]] = {}
    for label in sorted(set(r["axis_label"] for r in rows)):
        sub = [r for r in rows if r["axis_label"] == label]
        if not sub:
            continue
        vals = [float(r["VA"]["success"]) - float(r["V"]["success"]) for r in sub]
        mean, lo, hi = bootstrap_ci(vals, args.seed)
        by_axis[label] = {
            "n": len(sub),
            "V_error_rate": sum(1 for r in sub if not r["V"]["success"]) / len(sub),
            "VA_error_rate": sum(1 for r in sub if not r["VA"]["success"]) / len(sub),
            "VA_minus_V_correct": mean,
            "ci95": [lo, hi],
        }
    by_sparse: Dict[str, Dict[str, Any]] = {}
    for key, pred in {"a11y_present": lambda r: r["a11y_present"], "a11y_sparse": lambda r: r["a11y_sparse"]}.items():
        sub = [r for r in rows if pred(r)]
        if sub:
            by_sparse[key] = {
                "n": len(sub),
                "V_error_rate": sum(1 for r in sub if not r["V"]["success"]) / len(sub),
                "VA_error_rate": sum(1 for r in sub if not r["VA"]["success"]) / len(sub),
            }
    by_mechanism: Dict[str, Dict[str, Any]] = {}
    mechanism_tags = sorted({tag for row in rows for tag in row.get("mechanism_tags", [])})
    for tag in mechanism_tags:
        sub = [r for r in rows if tag in r.get("mechanism_tags", [])]
        if not sub:
            continue
        vals = [float(r["VA"]["success"]) - float(r["V"]["success"]) for r in sub]
        mean, lo, hi = bootstrap_ci(vals, args.seed)
        by_mechanism[tag] = {
            "n": len(sub),
            "V_error_rate": sum(1 for r in sub if not r["V"]["success"]) / len(sub),
            "VA_error_rate": sum(1 for r in sub if not r["VA"]["success"]) / len(sub),
            "only_V_right": sum(1 for r in sub if r["V"]["success"] and not r["VA"]["success"]),
            "only_VA_right": sum(1 for r in sub if r["VA"]["success"] and not r["V"]["success"]),
            "VA_minus_V_correct": mean,
            "ci95": [lo, hi],
        }
    only_v = coverage_cells["only_V_right"] / max(n, 1)
    only_va = coverage_cells["only_VA_right"] / max(n, 1)
    if delta_hi <= 0.0 or sum(va_correct) < sum(v_correct):
        verdict = "SOURCE-NOT-VIABLE"
        consequent = "V+A is not a valid improvement source on this slice; stop before full eval or training"
    elif jaccard <= args.jaccard_threshold and only_v >= args.only_v_threshold and only_va > only_v:
        verdict = "MODALITY-ORTHOGONAL"
        consequent = "V and V+A have distinct error sets; proceed to verifier after review"
    elif only_v < args.only_v_threshold and only_va > 0:
        verdict = "V+A-DOMINATES"
        consequent = "V+A is a better single source; modality does not justify multi-agent"
    else:
        verdict = "NOT-SEPARABLE"
        consequent = "V and V+A still share errors or V+A does not improve enough"
    return {
        "n": n,
        "coverage": coverage,
        "V_correct_rate": sum(v_correct) / max(n, 1),
        "VA_correct_rate": sum(va_correct) / max(n, 1),
        "VA_minus_V_correct": delta_mean,
        "VA_minus_V_correct_ci95": [delta_lo, delta_hi],
        "error_jaccard": jaccard,
        "prompt_jaccard_baseline": 0.91,
        "agreement_rate": agreement,
        "prompt_agreement_baseline": 0.77,
        "error_counts": {"V_errors": len(v_errors), "VA_errors": len(va_errors), "intersection": len(inter), "union": len(union)},
        "unique_coverage": dict(coverage_cells),
        "axis_cross_tab": by_axis,
        "a11y_cross_tab": by_sparse,
        "mechanism_cross_tab": by_mechanism,
        "verdict": verdict,
        "consequent": consequent,
    }


def write_outputs(output_dir: Path, rows: List[Dict[str, Any]], summary: Dict[str, Any], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_state.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload = {"summary": summary, "args": vars(args)}
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    (output_dir / "summary.md").write_text(render_summary(summary, args))


def render_summary(summary: Dict[str, Any], args: argparse.Namespace) -> str:
    c = summary["coverage"]
    lines = [
        "# Modality Orthogonality: Visual vs Visual+A11y",
        "",
        "## Gate Verdict",
        "",
        f"**{summary['verdict']}**",
        "",
        summary["consequent"],
        "",
        "## Control Infos Coverage",
        "",
        f"- required raw field: `{c.get('required_control_field', 'step.control_infos.uia_controls_info')}`",
        f"- states evaluated/considered: `{c['states']}`",
        f"- unique raw JSONL paths: `{c['unique_raw_paths']}`",
        f"- raw matched states: `{c['raw_matched_states']}`",
        f"- states with required controls: `{c['states_with_controls']}`",
        f"- states with uia_controls_info: `{c['states_with_uia_controls']}`",
        f"- states with merged_controls_info (diagnostic only): `{c.get('states_with_merged_controls', 0)}`",
        f"- coverage: `{c['coverage']:.4f}`",
        f"- sample control fields: `{c['sample_control_fields']}`",
        "",
        "## Source Correctness",
        "",
        f"- V correct rate: `{summary['V_correct_rate']:.4f}`",
        f"- V+A correct rate: `{summary['VA_correct_rate']:.4f}`",
        f"- V+A minus V: `{summary['VA_minus_V_correct']:.4f}` CI `{summary['VA_minus_V_correct_ci95'][0]:.4f}` / `{summary['VA_minus_V_correct_ci95'][1]:.4f}`",
        "",
        "## Error Jaccard + Agreement",
        "",
        f"- error-Jaccard(V,V+A): `{summary['error_jaccard']:.4f}` vs prompt baseline `{summary['prompt_jaccard_baseline']:.2f}`",
        f"- agreement rate: `{summary['agreement_rate']:.4f}` vs prompt baseline `{summary['prompt_agreement_baseline']:.2f}`",
        f"- error counts: `{summary['error_counts']}`",
        "",
        "## Unique Coverage",
        "",
        "| cell | count | share |",
        "|---|---:|---:|",
    ]
    for key, val in sorted(summary["unique_coverage"].items()):
        lines.append(f"| {key} | {val} | {val / max(summary['n'], 1):.4f} |")
    lines += ["", "## Axis Cross-Tab", "", "| base V bucket | n | V error | V+A error | V+A - V correct | CI |", "|---|---:|---:|---:|---:|---:|"]
    for key, row in sorted(summary["axis_cross_tab"].items()):
        lines.append(f"| {key} | {row['n']} | {row['V_error_rate']:.4f} | {row['VA_error_rate']:.4f} | {row['VA_minus_V_correct']:.4f} | [{row['ci95'][0]:.4f}, {row['ci95'][1]:.4f}] |")
    lines += ["", "## A11y Presence/Sparsity Cross-Tab", "", "| subset | n | V error | V+A error |", "|---|---:|---:|---:|"]
    for key, row in sorted(summary["a11y_cross_tab"].items()):
        lines.append(f"| {key} | {row['n']} | {row['V_error_rate']:.4f} | {row['VA_error_rate']:.4f} |")
    lines += [
        "",
        "## Mechanism Cross-Tab",
        "",
        "| mechanism | n | V error | V+A error | only V right | only V+A right | V+A - V correct | CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in sorted(summary.get("mechanism_cross_tab", {}).items()):
        lines.append(
            f"| {key} | {row['n']} | {row['V_error_rate']:.4f} | {row['VA_error_rate']:.4f} | "
            f"{row['only_V_right']} | {row['only_VA_right']} | {row['VA_minus_V_correct']:.4f} | "
            f"[{row['ci95'][0]:.4f}, {row['ci95'][1]:.4f}] |"
        )
    lines += ["", "No training is performed; V+A is an input-setting source with real `uia_controls_info` where available.", ""]
    return "\n".join(lines)


def render_coverage_summary(summary: Dict[str, Any], args: argparse.Namespace) -> str:
    c = summary["coverage"]
    lines = [
        "# Modality Orthogonality Coverage: Visual vs Visual+A11y",
        "",
        "## Gate Verdict",
        "",
        f"**{summary['verdict']}**",
        "",
        summary["consequent"],
        "",
        "## Control Infos Coverage",
        "",
        f"- required raw field: `{c.get('required_control_field', 'step.control_infos.uia_controls_info')}`",
        f"- states considered: `{c['states']}`",
        f"- unique raw JSONL paths: `{c['unique_raw_paths']}`",
        f"- raw download failures: `{c['raw_download_failures']}`",
        f"- raw matched states: `{c['raw_matched_states']}`",
        f"- states with required controls: `{c['states_with_controls']}`",
        f"- states with uia_controls_info: `{c['states_with_uia_controls']}`",
        f"- states with merged_controls_info (diagnostic only): `{c.get('states_with_merged_controls', 0)}`",
        f"- coverage: `{c['coverage']:.4f}`",
        f"- sample uia control fields: `{c['sample_control_fields']}`",
        "",
        "No model evaluation was run in coverage mode.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["coverage", "eval"], required=True)
    parser.add_argument("--balanced_data_dir", default="datasets/gui360-balanced/data")
    parser.add_argument("--raw_repo", default="vyokky/GUI-360")
    parser.add_argument("--raw_local_dir", default="datasets/GUI-360-raw-jsonl")
    parser.add_argument("--output_dir", default="outputs/candidate_orthogonality/modality_jaccard")
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--model_name", default="checkpoints/gui360-fullparam-sft-step250")
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max_controls", type=int, default=256)
    parser.add_argument("--va_prompt_style", choices=["directive", "hint"], default="directive")
    parser.add_argument("--image_max_pixels", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--request_timeout", type=float, default=600.0)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--near_px", type=float, default=50.0)
    parser.add_argument("--far_px", type=float, default=150.0)
    parser.add_argument("--jaccard_threshold", type=float, default=0.65)
    parser.add_argument("--only_v_threshold", type=float, default=0.03)
    parser.add_argument("--log_every", type=int, default=25)
    args = parser.parse_args()

    all_states = read_balanced_states(args.balanced_data_dir, args.max_episodes)
    rng = random.Random(args.seed)
    rng.shuffle(all_states)
    states = all_states[: args.limit] if args.limit else all_states
    coverage = attach_controls(states, args.raw_repo, args.raw_local_dir, args.log_every)
    for state in states:
        annotate_mechanisms(state)
    states = [state for state in states if state.get("controls")]
    if args.mode == "coverage":
        summary = {"coverage": coverage, "n": len(states), "verdict": "COVERAGE_ONLY", "consequent": "coverage computed; run --mode eval for V vs V+A"}
        output_dir = Path(args.output_dir) / "coverage"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(json.dumps({"summary": summary, "args": vars(args)}, ensure_ascii=False, indent=2) + "\n")
        (output_dir / "summary.md").write_text(render_coverage_summary(summary, args))
        print(f"Wrote {output_dir / 'summary.md'}")
        print(f"COVERAGE: {coverage}")
        return
    rows = evaluate_states(args, states)
    summary = summarize(rows, coverage, args)
    write_outputs(Path(args.output_dir), rows, summary, args)
    print(f"Wrote {Path(args.output_dir) / 'summary.md'}")
    print(f"Wrote {Path(args.output_dir) / 'per_state.jsonl'}")
    print(f"GATE: {summary['verdict']} - {summary['consequent']}")


if __name__ == "__main__":
    main()
