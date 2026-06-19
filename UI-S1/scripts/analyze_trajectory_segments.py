#!/usr/bin/env python3
"""Cross-benchmark GUI trajectory segmentation prototype.

Normalize GUI-Odyssey and AndroidControl trajectories, propose weak macro
segment boundaries from benchmark-agnostic signals, and write JSONL plus a
markdown report for later LLM/verifier adjudication.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GUI_ODYSSEY_DIR = PROJECT_ROOT / "datasets" / "GUI-Odyssey"
DEFAULT_ANDROID_CONTROL_JSONL = PROJECT_ROOT / "datasets" / "android_control_train1000_eval1543.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "segmentation"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

JsonDict = dict[str, Any]

STOPWORDS = {
    "a", "an", "and", "app", "are", "as", "at", "be", "before", "by", "for", "from", "go", "i", "in",
    "into", "is", "it", "me", "my", "of", "on", "once", "open", "or", "please", "screen", "step", "tap",
    "that", "the", "then", "this", "to", "using", "with", "you", "your",
}
APP_OPEN_WORDS = {"launch", "open", "start"}
NAV_WORDS = {"back", "category", "dialog", "home", "menu", "navigate", "page", "return", "section", "settings", "tab"}
INPUT_WORDS = {"enter", "input", "query", "search", "searching", "text", "type", "write"}
BROWSE_WORDS = {"browse", "check", "compare", "find", "listen", "look", "read", "recipe", "results", "review", "scroll", "see", "suggested", "view", "watch"}
SELECT_WORDS = {"choose", "click", "email", "item", "option", "product", "result", "select", "song", "target"}
CONFIG_WORDS = {"add", "change", "configure", "create", "dark", "delete", "disable", "edit", "enable", "mode", "reminder", "schedule", "set", "setting", "switch", "turn", "unmark", "update"}
COMMIT_WORDS = {"apply", "confirm", "done", "finish", "post", "save", "send", "share", "submit"}
TEMPORAL_MARKERS = {"after", "before", "finally", "next", "now", "once", "then"}

CAPABILITY_GROUPS = {
    "app_open": "navigation",
    "navigate_system": "navigation",
    "navigate_ui": "navigation",
    "input_text": "input",
    "search": "input",
    "browse_scan": "browse",
    "wait_observe": "browse",
    "select_target": "select",
    "configure_edit": "configure",
    "commit_submit": "commit",
    "finish": "finish",
    "interact": "interact",
    "unknown": "unknown",
}
ROUTE_BY_GROUP = {
    "navigation": "surface_transition_route",
    "input": "exact_value_route",
    "browse": "grounded_observation_route",
    "select": "grounded_selection_route",
    "configure": "state_change_route",
    "commit": "state_change_route",
    "finish": "termination_route",
    "interact": "grounded_selection_route",
    "unknown": "default_step_route",
}


def resolve_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else PROJECT_ROOT / path_obj


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path, max_examples: int | None) -> Iterator[tuple[int, JsonDict]]:
    emitted = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if max_examples is not None and emitted >= max_examples:
                break
            line = line.strip()
            if not line:
                continue
            yield line_index, json.loads(line)
            emitted += 1


def write_jsonl(path: Path, records: Iterable[JsonDict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def normalize_action(action_content: Any) -> JsonDict:
    if not isinstance(action_content, dict):
        return {"type": str(action_content or "unknown").lower(), "args": {}}
    action_type = str(action_content.get("action") or action_content.get("type") or "unknown").lower()
    if action_type == "scroll":
        action_type = "swipe"
    if action_type == "text":
        action_type = "type"
    args = {key: value for key, value in action_content.items() if key not in {"action", "type"}}
    return {"type": action_type, "args": args}


def convert_gui_raw_action(step: JsonDict) -> JsonDict:
    action = step.get("action")
    info = step.get("info")
    if action == "CLICK" and isinstance(info, str) and info.startswith("KEY_"):
        key_map = {"KEY_HOME": "Home", "KEY_BACK": "Back", "KEY_APPSELECT": "Menu"}
        return {"type": "system_button", "args": {"button": key_map.get(info, info.replace("KEY_", ""))}}
    if action == "CLICK":
        coord = info[0] if isinstance(info, list) and info else [0, 0]
        return {"type": "click", "args": {"coordinate": list(coord)}}
    if action == "LONG_PRESS":
        coord = info[0] if isinstance(info, list) and info else [0, 0]
        return {"type": "long_press", "args": {"coordinate": list(coord)}}
    if action == "SCROLL":
        coord1 = info[0] if isinstance(info, list) and len(info) > 0 else [0, 0]
        coord2 = info[1] if isinstance(info, list) and len(info) > 1 else [0, 0]
        return {"type": "swipe", "args": {"coordinate": list(coord1), "coordinate2": list(coord2)}}
    if action == "TEXT":
        return {"type": "type", "args": {"text": str(info)}}
    if action == "COMPLETE":
        return {"type": "terminate", "args": {"status": "success"}}
    if action == "INCOMPLETE":
        return {"type": "terminate", "args": {"status": "failure"}}
    return {"type": str(action or "unknown").lower(), "args": {"raw_info": info}}


def first_bbox(step: JsonDict) -> Any | None:
    check_options = step.get("check_options") if isinstance(step.get("check_options"), dict) else {}
    candidate_bbox = check_options.get("candidate_bbox") or step.get("candidate_bbox")
    if isinstance(candidate_bbox, list) and candidate_bbox:
        return candidate_bbox[0]
    sam2_bbox = step.get("sam2_bbox")
    if isinstance(sam2_bbox, list) and sam2_bbox:
        return sam2_bbox
    action_content = step.get("action_content") if isinstance(step.get("action_content"), dict) else {}
    bbox = action_content.get("bbox")
    return bbox if isinstance(bbox, list) and bbox else None


def action_coordinate(action: JsonDict) -> Any | None:
    args = action.get("args", {})
    return args.get("coordinate") or args.get("coordinate1")


def canonical_step(step_index: int, screenshot: str, action: JsonDict, text_fields: JsonDict, grounding: JsonDict) -> JsonDict:
    return {"step_index": step_index, "screenshot": screenshot, "action": action, "text_fields": text_fields, "grounding": grounding}


def adapt_gui_odyssey_converted(episode: JsonDict, line_index: int) -> JsonDict:
    steps = []
    for step_index, step in enumerate(episode.get("steps", [])):
        action = normalize_action(step.get("action_content", {}))
        steps.append(
            canonical_step(
                step_index,
                str(step.get("screenshot", "")),
                action,
                {
                    "instruction": step.get("step_instruction", ""),
                    "thought": step.get("thought", ""),
                    "observation": step.get("observation", ""),
                    "context": step.get("context", ""),
                    "history": step.get("history", ""),
                },
                {"bbox": first_bbox(step), "coordinate": action_coordinate(action), "ui_element_text": None, "a11y": None},
            )
        )
    return {
        "benchmark": "gui_odyssey",
        "episode_id": str(episode.get("episode_id", line_index)),
        "task_goal": episode.get("goal", ""),
        "task_metadata": {
            "category": episode.get("category", ""),
            "device_or_platform": episode.get("device_name", ""),
            "width": episode.get("width"),
            "height": episode.get("height"),
            "source": "converted_jsonl",
        },
        "steps": steps,
    }


def find_gui_screenshot_root(gui_dir: Path) -> Path:
    for candidate in (gui_dir / "data" / "screenshots" / "screenshots", gui_dir / "data" / "screenshots", gui_dir / "screenshots"):
        if candidate.is_dir():
            return candidate
    return gui_dir / "data" / "screenshots" / "screenshots"


def adapt_gui_odyssey_raw(episode_id: str, annotation: JsonDict, screenshot_root: Path) -> JsonDict:
    device_info = annotation.get("device_info", {})
    task_info = annotation.get("task_info", {})
    steps = []
    for step_index, step in enumerate(annotation.get("steps", [])):
        action = convert_gui_raw_action(step)
        steps.append(
            canonical_step(
                step_index,
                str(screenshot_root / str(step.get("screenshot", ""))),
                action,
                {
                    "instruction": step.get("low_level_instruction", ""),
                    "thought": step.get("intention", ""),
                    "observation": step.get("description", ""),
                    "context": step.get("context", ""),
                    "history": "",
                },
                {"bbox": first_bbox(step), "coordinate": action_coordinate(action), "ui_element_text": None, "a11y": None},
            )
        )
    return {
        "benchmark": "gui_odyssey",
        "episode_id": episode_id,
        "task_goal": task_info.get("instruction", ""),
        "task_metadata": {
            "category": task_info.get("category", ""),
            "device_or_platform": device_info.get("device_name", ""),
            "width": device_info.get("w"),
            "height": device_info.get("h"),
            "source": "raw_annotation",
        },
        "steps": steps,
    }


def iter_gui_odyssey(gui_dir: Path, gui_jsonl: Path | None, split: str, subset: str, max_examples: int | None) -> Iterator[JsonDict]:
    if gui_jsonl is None:
        gui_jsonl = gui_dir / f"gui_odyssey_{split}_{subset}.jsonl"
    if gui_jsonl.is_file():
        logger.info("Loading GUI-Odyssey converted JSONL: %s", gui_jsonl)
        for line_index, episode in iter_jsonl(gui_jsonl, max_examples):
            yield adapt_gui_odyssey_converted(episode, line_index)
        return

    split_path = gui_dir / "splits" / f"{split}.json"
    annotation_dir = gui_dir / "annotations"
    if not split_path.is_file() or not annotation_dir.is_dir():
        logger.warning("Skipping GUI-Odyssey: no converted JSONL or raw split/annotations found")
        return

    logger.info("Loading GUI-Odyssey raw annotations from %s", annotation_dir)
    split_data = read_json(split_path)
    if subset == "all":
        episode_ids = list(split_data.get("train", [])) + list(split_data.get("test", []))
    else:
        episode_ids = split_data.get(subset, [])
    screenshot_root = find_gui_screenshot_root(gui_dir)
    emitted = 0
    skipped = 0
    for raw_episode_id in episode_ids:
        if max_examples is not None and emitted >= max_examples:
            break
        episode_id = str(raw_episode_id).replace(".json", "")
        annotation_path = annotation_dir / f"{episode_id}.json"
        if not annotation_path.is_file():
            skipped += 1
            continue
        yield adapt_gui_odyssey_raw(episode_id, read_json(annotation_path), screenshot_root)
        emitted += 1
    if skipped:
        logger.warning("Skipped %d GUI-Odyssey annotations missing from %s/%s", skipped, split, subset)


def adapt_android_control(episode: JsonDict, line_index: int) -> JsonDict:
    apps = []
    steps = []
    for step_index, step in enumerate(episode.get("steps", [])):
        action = normalize_action(step.get("action_content", {}))
        if action["type"] == "open" and action.get("args", {}).get("text"):
            apps.append(action["args"]["text"])
        steps.append(
            canonical_step(
                step_index,
                str(step.get("screenshot", "")),
                action,
                {"instruction": "", "thought": "", "observation": "", "context": "", "history": ""},
                {"bbox": first_bbox(step), "coordinate": action_coordinate(action), "ui_element_text": action.get("args", {}).get("text"), "a11y": None},
            )
        )
    return {
        "benchmark": "android_control",
        "episode_id": str(episode.get("episode_id", line_index)),
        "task_goal": episode.get("goal", ""),
        "task_metadata": {"apps": sorted(set(apps)), "is_successful": episode.get("is_successful"), "device_or_platform": "android", "source": "jsonl"},
        "steps": steps,
    }


def iter_android_control(path: Path, max_examples: int | None) -> Iterator[JsonDict]:
    if not path.is_file():
        logger.warning("Skipping AndroidControl: %s not found", path)
        return
    logger.info("Loading AndroidControl JSONL: %s", path)
    for line_index, episode in iter_jsonl(path, max_examples):
        yield adapt_android_control(episode, line_index)


def tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in STOPWORDS and len(token) > 1]


def text_blob(step: JsonDict, include_action_args: bool = True) -> str:
    fields = step.get("text_fields", {})
    pieces = [str(fields.get(key, "")) for key in ("instruction", "thought", "observation", "context", "history")]
    if include_action_args:
        for value in step.get("action", {}).get("args", {}).values():
            if isinstance(value, str):
                pieces.append(value)
    return " ".join(piece for piece in pieces if piece)


def words_in(text: str, vocabulary: set[str]) -> bool:
    return bool(set(tokenize(text)) & vocabulary)


def jaccard_similarity(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 1.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def classify_step(step: JsonDict, task_goal: str) -> str:
    action = step.get("action", {})
    action_type = action.get("type", "unknown")
    local_blob = text_blob(step).lower()
    goal_blob = f"{task_goal} {local_blob}".lower()
    button = str(action.get("args", {}).get("button", "")).lower()

    if action_type == "terminate":
        return "finish"
    if action_type == "open":
        return "app_open"
    if action_type == "system_button" or button in {"home", "back", "menu", "appselect"}:
        return "navigate_system"
    if action_type == "type":
        return "input_text"
    if action_type in {"swipe", "scroll"}:
        return "browse_scan"
    if action_type == "wait":
        return "wait_observe"
    if words_in(local_blob, INPUT_WORDS):
        return "search"
    if words_in(local_blob, CONFIG_WORDS):
        return "configure_edit"
    if words_in(local_blob, COMMIT_WORDS):
        return "commit_submit"
    if words_in(local_blob, BROWSE_WORDS):
        return "browse_scan"
    if action_type in {"click", "long_press"} and words_in(goal_blob, SELECT_WORDS):
        return "select_target"
    if action_type in {"click", "long_press"} and words_in(local_blob, NAV_WORDS | APP_OPEN_WORDS):
        return "navigate_ui"
    if action_type in {"click", "long_press"}:
        return "interact"
    return "unknown"


def capability_group(capability: str) -> str:
    return CAPABILITY_GROUPS.get(capability, capability)


def is_system_transition(step: JsonDict) -> bool:
    action = step.get("action", {})
    button = str(action.get("args", {}).get("button", "")).lower()
    return action.get("type") == "system_button" and button in {"home", "back", "menu", "appselect"}


def has_temporal_marker(step: JsonDict) -> bool:
    return bool(set(tokenize(text_blob(step, include_action_args=False))) & TEMPORAL_MARKERS)


def boundary_score(
    previous_step: JsonDict,
    current_step: JsonDict,
    previous_capability: str,
    current_capability: str,
    current_segment_length: int,
    max_segment_len: int,
) -> tuple[float, list[str]]:
    previous_type = previous_step.get("action", {}).get("type", "unknown")
    current_type = current_step.get("action", {}).get("type", "unknown")
    reasons = []
    score = 0.0

    if current_type == "open":
        score += 0.9
        reasons.append("current_step_opens_app")
    if is_system_transition(previous_step):
        score += 0.8
        reasons.append("previous_step_system_transition")
    if is_system_transition(current_step) and current_segment_length >= 3:
        score += 0.65
        reasons.append("current_step_system_transition")
    if previous_type == "type" and current_type not in {"type", "wait", "terminate"} and current_segment_length >= 2:
        score += 0.35
        reasons.append("query_or_value_entered")
    if previous_type in {"swipe", "scroll", "wait"} and current_type in {"click", "type"} and current_segment_length >= 3:
        score += 0.2
        reasons.append("scan_to_interaction")

    previous_group = capability_group(previous_capability)
    current_group = capability_group(current_capability)
    if previous_group != current_group and current_group != "finish" and current_segment_length >= 3:
        score += 0.25
        reasons.append(f"capability_group_shift:{previous_group}->{current_group}")

    similarity = jaccard_similarity(text_blob(previous_step), text_blob(current_step))
    if similarity < 0.08 and current_segment_length >= 3:
        score += 0.2
        reasons.append("low_text_continuity")
    if has_temporal_marker(current_step) and current_segment_length >= 3:
        score += 0.15
        reasons.append("temporal_marker")
    if current_segment_length >= max_segment_len and current_group != "finish":
        score += 0.45
        reasons.append("max_segment_length")
    if current_type == "terminate":
        score -= 0.5

    return max(0.0, min(1.0, score)), reasons or ["continuation"]


def build_capability_phases(steps: Sequence[JsonDict], capabilities: Sequence[str]) -> list[JsonDict]:
    if not steps:
        return []
    phases = []
    phase_start = 0
    phase_capability = capabilities[0]
    for local_index in range(1, len(steps)):
        if capabilities[local_index] != phase_capability:
            phases.append({"phase_id": len(phases), "capability": phase_capability, "start_step": steps[phase_start]["step_index"], "end_step": steps[local_index - 1]["step_index"], "length": local_index - phase_start})
            phase_start = local_index
            phase_capability = capabilities[local_index]
    phases.append({"phase_id": len(phases), "capability": phase_capability, "start_step": steps[phase_start]["step_index"], "end_step": steps[-1]["step_index"], "length": len(steps) - phase_start})
    return phases


def compact_step(step: JsonDict, capability: str) -> JsonDict:
    text_fields = step.get("text_fields", {})
    return {
        "step_index": step.get("step_index"),
        "screenshot": step.get("screenshot"),
        "action": step.get("action", {}),
        "capability": capability,
        "instruction": text_fields.get("instruction", ""),
        "thought": text_fields.get("thought", ""),
        "grounding": step.get("grounding", {}),
    }


def extract_carried_values(steps: Sequence[JsonDict], task_goal: str) -> list[str]:
    values = []
    for step in steps:
        action = step.get("action", {})
        args = action.get("args", {})
        if action.get("type") in {"type", "open"} and isinstance(args.get("text"), str):
            values.append(args["text"].strip())
    for single_quote, double_quote in re.findall(r"'([^']{2,80})'|\"([^\"]{2,80})\"", task_goal):
        values.append((single_quote or double_quote).strip())

    deduped = []
    seen = set()
    for value in values:
        normalized = value.lower()
        if value and normalized not in seen:
            deduped.append(value)
            seen.add(normalized)
    return deduped[:8]


def memory_need(segment_id: int, steps: Sequence[JsonDict], carried_values: list[str], start_reasons: list[str]) -> JsonDict:
    reasons = []
    action_types = {step.get("action", {}).get("type", "unknown") for step in steps}
    if segment_id > 0:
        reasons.append("non_initial_segment")
    if carried_values:
        reasons.append("explicit_value_or_entity_carry")
    if "type" in action_types:
        reasons.append("exact_text_or_query_needed")
    if any(reason in start_reasons for reason in ("previous_step_system_transition", "current_step_opens_app")):
        reasons.append("surface_transition_needs_goal_context")
    strength = "high" if segment_id > 0 and carried_values else ("medium" if reasons else "none")
    return {"strength": strength, "reasons": reasons}


def summarize_segment(steps: Sequence[JsonDict], capabilities: Sequence[str]) -> str:
    instructions = [step.get("text_fields", {}).get("instruction", "").strip() for step in steps]
    instructions = [instruction for instruction in instructions if instruction]
    if instructions:
        first = instructions[0].rstrip(".")
        last = instructions[-1].rstrip(".")
        return first if first.lower() == last.lower() else f"{first}; then {last}"
    dominant_capability = Counter(capabilities).most_common(1)[0][0] if capabilities else "unknown"
    dominant_action = Counter(step.get("action", {}).get("type", "unknown") for step in steps).most_common(1)[0][0]
    return f"{dominant_capability} via {dominant_action} actions"


def segment_episode(episode: JsonDict, min_segment_len: int, max_segment_len: int, boundary_threshold: float) -> JsonDict:
    steps = episode.get("steps", [])
    task_goal = episode.get("task_goal", "")
    capabilities = [classify_step(step, task_goal) for step in steps]
    boundary_starts = [0] if steps else []
    boundary_info = {0: {"confidence": 1.0, "reasons": ["episode_start"]}}
    segment_start = 0

    for step_index in range(1, len(steps)):
        score, reasons = boundary_score(steps[step_index - 1], steps[step_index], capabilities[step_index - 1], capabilities[step_index], step_index - segment_start, max_segment_len)
        long_enough = step_index - segment_start >= min_segment_len
        if score >= boundary_threshold and (long_enough or score >= 0.8):
            boundary_starts.append(step_index)
            boundary_info[step_index] = {"confidence": round(score, 3), "reasons": reasons}
            segment_start = step_index

    segments = []
    for segment_id, start in enumerate(boundary_starts):
        end = (boundary_starts[segment_id + 1] - 1) if segment_id + 1 < len(boundary_starts) else len(steps) - 1
        segment_steps = steps[start : end + 1]
        segment_capabilities = capabilities[start : end + 1]
        dominant = Counter(segment_capabilities).most_common(1)[0][0] if segment_capabilities else "unknown"
        carried_values = extract_carried_values(segment_steps, task_goal)
        start_reasons = boundary_info[start]["reasons"]
        phase_groups = {capability_group(capability) for capability in segment_capabilities}
        candidate_routes = sorted(ROUTE_BY_GROUP.get(item, "default_step_route") for item in phase_groups)
        segments.append(
            {
                "segment_id": segment_id,
                "start_step": start,
                "end_step": end,
                "length": len(segment_steps),
                "dominant_capability": dominant,
                "capability_group": capability_group(dominant),
                "summary": summarize_segment(segment_steps, segment_capabilities),
                "candidate_routes": candidate_routes,
                "memory_need": memory_need(segment_id, segment_steps, carried_values, start_reasons),
                "carried_values": carried_values,
                "boundary_start": boundary_info[start],
                "capability_phases": build_capability_phases(segment_steps, segment_capabilities),
                "steps": [compact_step(step, capability) for step, capability in zip(segment_steps, segment_capabilities)],
            }
        )

    return {
        **episode,
        "num_steps": len(steps),
        "num_segments": len(segments),
        "segmentation_method": {
            "name": "universal_signal_weak_proposer",
            "min_segment_len": min_segment_len,
            "max_segment_len": max_segment_len,
            "boundary_threshold": boundary_threshold,
        },
        "segments": segments,
    }


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def collect_stats(records: Sequence[JsonDict]) -> JsonDict:
    step_counts = [record.get("num_steps", 0) for record in records]
    segment_counts = [record.get("num_segments", 0) for record in records]
    segment_lengths = [segment["length"] for record in records for segment in record.get("segments", [])]
    boundary_reasons = Counter()
    capabilities = Counter()
    action_types = Counter()
    routes = Counter()
    memory = Counter()
    for record in records:
        for step in record.get("steps", []):
            action_types[step.get("action", {}).get("type", "unknown")] += 1
        for segment in record.get("segments", []):
            capabilities[segment.get("dominant_capability", "unknown")] += 1
            memory[segment.get("memory_need", {}).get("strength", "none")] += 1
            for route in segment.get("candidate_routes", []):
                routes[route] += 1
            for reason in segment.get("boundary_start", {}).get("reasons", []):
                boundary_reasons[reason] += 1
    return {
        "episodes": len(records),
        "steps": sum(step_counts),
        "segments": sum(segment_counts),
        "avg_steps_per_episode": sum(step_counts) / len(step_counts) if step_counts else 0.0,
        "avg_segments_per_episode": sum(segment_counts) / len(segment_counts) if segment_counts else 0.0,
        "avg_segment_length": sum(segment_lengths) / len(segment_lengths) if segment_lengths else 0.0,
        "segment_length_p50": percentile(segment_lengths, 0.5),
        "segment_length_p90": percentile(segment_lengths, 0.9),
        "boundary_reasons": boundary_reasons,
        "dominant_capabilities": capabilities,
        "action_types": action_types,
        "candidate_routes": routes,
        "memory_need": memory,
    }


def format_counter(counter: Counter[str], max_items: int = 12) -> list[str]:
    if not counter:
        return ["- none"]
    total = sum(counter.values()) or 1
    return [f"- `{name}`: {count} ({count / total:.1%})" for name, count in counter.most_common(max_items)]


def sample_episode_lines(records: Sequence[JsonDict], max_examples: int = 3) -> list[str]:
    lines = []
    for record in records[:max_examples]:
        lines.append(f"### {record['benchmark']} / {record['episode_id']}")
        lines.append(f"Goal: {record.get('task_goal', '').replace(chr(10), ' ')}")
        for segment in record.get("segments", [])[:6]:
            reasons = ", ".join(segment.get("boundary_start", {}).get("reasons", []))
            routes = ", ".join(segment.get("candidate_routes", []))
            lines.append(f"- S{segment['segment_id']} [{segment['start_step']}-{segment['end_step']}] `{segment['dominant_capability']}`: {segment['summary']} (boundary: {reasons}; routes: {routes})")
        if record.get("num_segments", 0) > 6:
            lines.append(f"- ... {record['num_segments'] - 6} more segments")
        lines.append("")
    return lines


def write_report(path: Path, grouped_records: dict[str, Sequence[JsonDict]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Trajectory Segmentation Prototype Report",
        "",
        "Generated by `scripts/analyze_trajectory_segments.py`.",
        "",
        "## Method",
        "",
        "The script maps each benchmark into a canonical episode schema and proposes macro boundaries from universal signals:",
        "",
        "- explicit app opens and system navigation transitions",
        "- typed query/value followed by a different interaction phase",
        "- scan-to-interaction shifts after scroll/wait runs",
        "- capability-group shifts over navigation, input, browse, select, configure, commit, and finish phases",
        "- low lexical continuity between adjacent step descriptions or instructions",
        "- temporal markers such as then, once, after, and next",
        "",
        f"Boundary threshold: `{args.boundary_threshold}`; min segment length: `{args.min_segment_len}`; max segment length: `{args.max_segment_len}`.",
        "",
    ]
    for benchmark, records in grouped_records.items():
        stats = collect_stats(records)
        lines.extend(
            [
                f"## {benchmark}",
                "",
                f"- Episodes: {stats['episodes']}",
                f"- Steps: {stats['steps']}",
                f"- Segments: {stats['segments']}",
                f"- Avg steps / episode: {stats['avg_steps_per_episode']:.2f}",
                f"- Avg segments / episode: {stats['avg_segments_per_episode']:.2f}",
                f"- Avg segment length: {stats['avg_segment_length']:.2f}",
                f"- Segment length p50 / p90: {stats['segment_length_p50']:.1f} / {stats['segment_length_p90']:.1f}",
                "",
                "### Boundary Reasons",
                "",
                *format_counter(stats["boundary_reasons"]),
                "",
                "### Dominant Capabilities",
                "",
                *format_counter(stats["dominant_capabilities"]),
                "",
                "### Candidate Routes",
                "",
                *format_counter(stats["candidate_routes"]),
                "",
                "### Memory Need",
                "",
                *format_counter(stats["memory_need"]),
                "",
                "### Action Types",
                "",
                *format_counter(stats["action_types"]),
                "",
                "### Samples",
                "",
                *sample_episode_lines(records),
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze and weakly segment GUI trajectories across benchmarks")
    parser.add_argument("--gui-odyssey-dir", type=str, default=str(DEFAULT_GUI_ODYSSEY_DIR))
    parser.add_argument("--gui-odyssey-jsonl", type=str, default=None)
    parser.add_argument("--gui-split", type=str, default="random_split")
    parser.add_argument("--gui-subset", type=str, default="test")
    parser.add_argument("--android-control-jsonl", type=str, default=str(DEFAULT_ANDROID_CONTROL_JSONL))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-examples", type=int, default=200, help="Maximum episodes per benchmark; use 0 for all")
    parser.add_argument("--min-segment-len", type=int, default=2)
    parser.add_argument("--max-segment-len", type=int, default=12)
    parser.add_argument("--boundary-threshold", type=float, default=0.65)
    parser.add_argument("--skip-gui-odyssey", action="store_true")
    parser.add_argument("--skip-android-control", action="store_true")
    return parser.parse_args()


def segment_records(records: Iterable[JsonDict], args: argparse.Namespace) -> list[JsonDict]:
    return [segment_episode(record, args.min_segment_len, args.max_segment_len, args.boundary_threshold) for record in records]


def main() -> None:
    args = parse_args()
    max_examples = None if args.max_examples == 0 else args.max_examples
    output_dir = resolve_path(args.output_dir)
    assert output_dir is not None
    grouped_records: dict[str, list[JsonDict]] = {}

    if not args.skip_gui_odyssey:
        gui_dir = resolve_path(args.gui_odyssey_dir)
        gui_jsonl = resolve_path(args.gui_odyssey_jsonl)
        if gui_dir is not None:
            gui_records = segment_records(iter_gui_odyssey(gui_dir, gui_jsonl, args.gui_split, args.gui_subset, max_examples), args)
            if gui_records:
                grouped_records["gui_odyssey"] = gui_records
                out_path = output_dir / "gui_odyssey_segments.jsonl"
                logger.info("Wrote %d GUI-Odyssey segmented episodes to %s", write_jsonl(out_path, gui_records), out_path)

    if not args.skip_android_control:
        android_path = resolve_path(args.android_control_jsonl)
        if android_path is not None:
            android_records = segment_records(iter_android_control(android_path, max_examples), args)
            if android_records:
                grouped_records["android_control"] = android_records
                out_path = output_dir / "android_control_segments.jsonl"
                logger.info("Wrote %d AndroidControl segmented episodes to %s", write_jsonl(out_path, android_records), out_path)

    if not grouped_records:
        raise SystemExit("No benchmark records were loaded. Check input paths.")

    report_path = output_dir / "trajectory_segmentation_report.md"
    write_report(report_path, grouped_records, args)
    logger.info("Wrote report to %s", report_path)


if __name__ == "__main__":
    main()
