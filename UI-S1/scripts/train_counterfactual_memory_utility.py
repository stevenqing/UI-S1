#!/usr/bin/env python3
"""Train a minimal text-based counterfactual memory utility scorer."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


JsonDict = dict[str, Any]


def iter_jsonl(path: Path) -> Iterable[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_split(data_dir: Path, split: str) -> list[JsonDict]:
    return list(iter_jsonl(data_dir / f"{split}.jsonl"))


def pair_text(row: JsonDict, memory_key: str) -> str:
    memory = memory_only_text(row, memory_key)
    ocr_text = row.get("ocr_text", "")
    ocr_part = f"\n\nOCR VISIBLE TEXT:\n{ocr_text}" if ocr_text else ""
    return f"CURRENT STATE:\n{row.get('current_state_text', '')}{ocr_part}\n\nMEMORY:\n{memory}"


def memory_only_text(row: JsonDict, memory_key: str) -> str:
    if memory_key == "true_memory":
        return str(row.get("true_memory_text", ""))
    if memory_key == "wrong_memory":
        return str(row.get("wrong_memory_text", ""))
    if memory_key == "no_memory":
        return "No previous memory."
    return ""


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


def action_type(action: JsonDict | None) -> str:
    if not action:
        return "missing"
    return str(action.get("action", "missing"))


def action_text(action: JsonDict | None) -> str:
    if not action:
        return ""
    if action.get("text") is not None:
        return str(action.get("text"))
    if action.get("button") is not None:
        return str(action.get("button"))
    if action.get("coordinate") is not None:
        return " ".join(str(x) for x in action.get("coordinate", []))
    return json.dumps(action, ensure_ascii=False)


def action_value_text(action: JsonDict | None) -> str:
    if not action:
        return ""
    if action.get("text") is not None:
        return str(action.get("text"))
    if action.get("button") is not None:
        return str(action.get("button"))
    return ""


def button_name(action: JsonDict | None) -> str:
    if not action or action.get("action") != "system_button":
        return ""
    return normalize_text(action.get("button", ""))


def coordinate(action: JsonDict | None, key: str = "coordinate") -> tuple[float, float] | None:
    if not action:
        return None
    value = action.get(key)
    if not isinstance(value, list) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def coordinate_distance(first: JsonDict | None, second: JsonDict | None) -> float:
    first_coord = coordinate(first)
    second_coord = coordinate(second)
    if first_coord is None or second_coord is None:
        return -1.0
    return math.hypot(first_coord[0] - second_coord[0], first_coord[1] - second_coord[1])


def distance_bucket(distance: float) -> str:
    if distance < 0:
        return "missing"
    if distance < 20:
        return "same"
    if distance < 120:
        return "near"
    if distance < 360:
        return "far"
    return "very_far"


def swipe_direction(action: JsonDict | None) -> str:
    start = coordinate(action, "coordinate")
    end = coordinate(action, "coordinate2")
    if start is None or end is None:
        return "none"
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if abs(dx) < 30 and abs(dy) < 30:
        return "short"
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def token_set(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", normalize_text(text)) if len(token) > 1}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def candidate_action(row: JsonDict, memory_key: str) -> JsonDict | None:
    condition = {
        "true_memory": "segment_summary",
        "wrong_memory": "wrong_summary",
        "no_memory": "no_history",
    }[memory_key]
    return row.get("pred_actions", {}).get(condition)


def distractor_action(row: JsonDict, memory_key: str) -> JsonDict | None:
    condition = {
        "true_memory": "wrong_summary",
        "wrong_memory": "segment_summary",
        "no_memory": "wrong_summary",
    }[memory_key]
    return row.get("pred_actions", {}).get(condition)


def carried_values_for_memory(row: JsonDict, memory_key: str) -> list[str]:
    memory_text = memory_only_text(row, memory_key)
    values: list[str] = []
    for match in re.finditer(r"carried_values=(\[[^\n]*?\])", memory_text):
        raw = match.group(1)
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            parsed = []
        if isinstance(parsed, list):
            values.extend(str(item) for item in parsed if str(item))
    if values:
        return values
    return [str(value) for value in (row.get("metadata", {}).get("carried_values", []) or [])]


def action_field_count(action: JsonDict | None) -> int:
    if not action:
        return 0
    return sum(1 for key in ["action", "text", "button", "coordinate", "coordinate2"] if action.get(key) is not None)


def task_action_type(action_type_value: str) -> bool:
    return action_type_value in {"click", "type", "swipe", "long_press", "open"}


def fallback_system_button(value: str) -> bool:
    return value in {"home", "back", "recent_apps", "recent apps", "app_switch", "app switch"}


def text_overlap_features(row: JsonDict, memory_key: str) -> dict[str, Any]:
    parts = row.get("current_state_parts", {}) or {}
    metadata = row.get("metadata", {}) or {}
    carried_values = metadata.get("carried_values", []) or []
    carried_text = normalize_text(" ".join(str(value) for value in carried_values))
    carried_tokens = token_set(carried_text)
    memory_tokens = token_set(pair_text(row, memory_key))
    goal_tokens = token_set(parts.get("goal", ""))
    instruction_tokens = token_set(parts.get("instruction", ""))
    observation_tokens = token_set(parts.get("observation", ""))
    hint_tokens = token_set(parts.get("local_hint", ""))
    segment_tokens = token_set(parts.get("current_segment", ""))
    current_tokens = token_set(row.get("current_state_text", ""))
    ocr_tokens = token_set(row.get("ocr_text", ""))
    features: dict[str, Any] = {
        "screen_has_observation": int(bool(parts.get("observation"))),
        "goal_memory_overlap": jaccard(goal_tokens, memory_tokens),
        "instruction_memory_overlap": jaccard(instruction_tokens, memory_tokens),
        "observation_memory_overlap": jaccard(observation_tokens, memory_tokens),
        "hint_memory_overlap": jaccard(hint_tokens, memory_tokens),
        "segment_memory_overlap": jaccard(segment_tokens, memory_tokens),
        "current_memory_overlap": jaccard(current_tokens, memory_tokens),
        "carried_goal_overlap": jaccard(carried_tokens, goal_tokens),
        "carried_instruction_overlap": jaccard(carried_tokens, instruction_tokens),
        "carried_observation_overlap": jaccard(carried_tokens, observation_tokens),
        "carried_hint_overlap": jaccard(carried_tokens, hint_tokens),
        "carried_current_overlap": jaccard(carried_tokens, current_tokens),
        "carried_ocr_overlap": jaccard(carried_tokens, ocr_tokens),
        "carried_visible_in_observation": int(bool(carried_tokens & observation_tokens)),
        "carried_visible_in_ocr": int(bool(carried_tokens & ocr_tokens)),
        "carried_visible_in_instruction": int(bool(carried_tokens & instruction_tokens)),
        "carried_visible_in_current": int(bool(carried_tokens & current_tokens)),
        "memory_ocr_overlap": jaccard(memory_tokens, ocr_tokens),
        "ocr_token_count": len(ocr_tokens),
        "has_ocr": int(bool(ocr_tokens)),
        "memory_adds_tokens_not_in_current": max(0, len(memory_tokens - current_tokens)),
        "memory_adds_tokens_not_in_ocr": max(0, len(memory_tokens - ocr_tokens)),
        "current_contains_memory_tokens": int(bool(memory_tokens) and memory_tokens <= current_tokens),
        "ocr_contains_memory_tokens": int(bool(memory_tokens) and memory_tokens <= ocr_tokens),
    }
    return features


def load_ocr_cache(path_text: str | None) -> dict[str, str]:
    if not path_text:
        return {}
    path = Path(path_text)
    if not path.exists():
        return {}
    cache = {}
    for line in path.open("r", encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        screenshot = row.get("screenshot")
        if screenshot and row.get("ok"):
            cache[str(screenshot)] = str(row.get("ocr_text", ""))
    return cache


def attach_ocr(rows: list[JsonDict], cache: dict[str, str]) -> None:
    if not cache:
        return
    for row in rows:
        screenshot = row.get("metadata", {}).get("screenshot")
        if screenshot in cache:
            row["ocr_text"] = cache[screenshot]


def candidate_features(row: JsonDict, memory_key: str) -> dict[str, Any]:
    no_action = row.get("pred_actions", {}).get("no_history")
    cand_action = candidate_action(row, memory_key)
    no_type = action_type(no_action)
    cand_type = action_type(cand_action)
    no_text = normalize_text(action_text(no_action))
    cand_text = normalize_text(action_text(cand_action))
    memory_text = normalize_text(pair_text(row, memory_key))
    carried_values = carried_values_for_memory(row, memory_key)
    carried_text = normalize_text(" ".join(str(value) for value in carried_values))
    no_tokens = token_set(no_text)
    cand_tokens = token_set(cand_text)
    memory_tokens = token_set(memory_text)
    carried_tokens = token_set(carried_text)
    features: dict[str, Any] = {
        "candidate_type=" + cand_type: 1,
        "no_history_type=" + no_type: 1,
        "candidate_missing": int(cand_action is None),
        "no_history_missing": int(no_action is None),
        "candidate_type_matches_no_history": int(cand_type == no_type),
        "candidate_type_differs_no_history": int(cand_type != no_type),
        "candidate_text_len": len(cand_text.split()),
        "no_history_text_len": len(no_text.split()),
        "candidate_text_equals_no_history": int(cand_text == no_text and bool(cand_text)),
        "candidate_text_jaccard_no_history": jaccard(cand_tokens, no_tokens),
        "candidate_memory_overlap": jaccard(cand_tokens, memory_tokens),
        "no_history_memory_overlap": jaccard(no_tokens, memory_tokens),
        "candidate_carried_overlap": jaccard(cand_tokens, carried_tokens),
        "no_history_carried_overlap": jaccard(no_tokens, carried_tokens),
        "candidate_overlap_beats_no_history": int(jaccard(cand_tokens, carried_tokens) > jaccard(no_tokens, carried_tokens)),
        "has_carried_values": int(bool(carried_tokens)),
    }
    return features


def repair_features(row: JsonDict, memory_key: str) -> dict[str, Any]:
    no_action = row.get("pred_actions", {}).get("no_history")
    cand_action = candidate_action(row, memory_key)
    no_type = action_type(no_action)
    cand_type = action_type(cand_action)
    no_text = normalize_text(action_text(no_action))
    cand_text = normalize_text(action_text(cand_action))
    no_value_text = normalize_text(action_value_text(no_action))
    cand_value_text = normalize_text(action_value_text(cand_action))
    no_button = button_name(no_action)
    cand_button = button_name(cand_action)
    memory_text = normalize_text(memory_only_text(row, memory_key))
    current_text = normalize_text(row.get("current_state_text", ""))
    parts = row.get("current_state_parts", {}) or {}
    goal_text = normalize_text(parts.get("goal", ""))
    instruction_text = normalize_text(parts.get("instruction", ""))
    carried_text = normalize_text(" ".join(carried_values_for_memory(row, memory_key)))

    no_tokens = token_set(no_text)
    cand_tokens = token_set(cand_text)
    no_value_tokens = token_set(no_value_text)
    cand_value_tokens = token_set(cand_value_text)
    memory_tokens = token_set(memory_text)
    current_tokens = token_set(current_text)
    goal_tokens = token_set(goal_text)
    instruction_tokens = token_set(instruction_text)
    carried_tokens = token_set(carried_text)

    cand_carried = jaccard(cand_tokens | cand_value_tokens, carried_tokens)
    no_carried = jaccard(no_tokens | no_value_tokens, carried_tokens)
    cand_memory = jaccard(cand_tokens | cand_value_tokens, memory_tokens)
    no_memory = jaccard(no_tokens | no_value_tokens, memory_tokens)
    cand_current = jaccard(cand_tokens | cand_value_tokens, current_tokens)
    no_current = jaccard(no_tokens | no_value_tokens, current_tokens)
    cand_goal = jaccard(cand_tokens | cand_value_tokens, goal_tokens)
    no_goal = jaccard(no_tokens | no_value_tokens, goal_tokens)
    cand_instruction = jaccard(cand_tokens | cand_value_tokens, instruction_tokens)
    no_instruction = jaccard(no_tokens | no_value_tokens, instruction_tokens)
    distance = coordinate_distance(no_action, cand_action)
    no_is_system_fallback = no_type == "system_button" and fallback_system_button(no_button)
    cand_is_system_fallback = cand_type == "system_button" and fallback_system_button(cand_button)
    no_is_task = task_action_type(no_type)
    cand_is_task = task_action_type(cand_type)
    value_overlap_gain = cand_carried - no_carried
    memory_overlap_gain = cand_memory - no_memory
    current_overlap_gain = cand_current - no_current
    goal_overlap_gain = cand_goal - no_goal
    instruction_overlap_gain = cand_instruction - no_instruction

    features: dict[str, Any] = {
        f"repair_transition={no_type}->{cand_type}": 1,
        f"repair_no_type={no_type}": 1,
        f"repair_candidate_type={cand_type}": 1,
        f"repair_no_button={no_button or 'none'}": 1,
        f"repair_candidate_button={cand_button or 'none'}": 1,
        f"repair_swipe_direction={swipe_direction(cand_action)}": 1,
        f"repair_coordinate_distance={distance_bucket(distance)}": 1,
        "repair_candidate_exists": int(cand_action is not None),
        "repair_no_history_exists": int(no_action is not None),
        "repair_type_changed": int(no_type != cand_type),
        "repair_text_changed": int(bool(cand_text or no_text) and cand_text != no_text),
        "repair_value_text_changed": int(bool(cand_value_text or no_value_text) and cand_value_text != no_value_text),
        "repair_no_is_system_fallback": int(no_is_system_fallback),
        "repair_candidate_is_system_fallback": int(cand_is_system_fallback),
        "repair_system_fallback_to_task_action": int(no_is_system_fallback and cand_is_task),
        "repair_task_action_to_system_fallback": int(no_is_task and cand_is_system_fallback),
        "repair_missing_to_candidate": int(no_type == "missing" and cand_type != "missing"),
        "repair_candidate_to_missing": int(no_type != "missing" and cand_type == "missing"),
        "repair_field_count_delta": action_field_count(cand_action) - action_field_count(no_action),
        "repair_candidate_value_token_count": len(cand_value_tokens),
        "repair_no_value_token_count": len(no_value_tokens),
        "repair_candidate_text_token_count": len(cand_tokens),
        "repair_no_text_token_count": len(no_tokens),
        "repair_carried_token_count": len(carried_tokens),
        "repair_has_memory_carried_values": int(bool(carried_tokens)),
        "repair_candidate_carried_overlap": cand_carried,
        "repair_no_history_carried_overlap": no_carried,
        "repair_carried_overlap_gain": value_overlap_gain,
        "repair_candidate_uses_carried": int(bool((cand_tokens | cand_value_tokens) & carried_tokens)),
        "repair_no_history_uses_carried": int(bool((no_tokens | no_value_tokens) & carried_tokens)),
        "repair_exact_value_gain": int(bool(cand_value_tokens & carried_tokens) and not bool(no_value_tokens & carried_tokens)),
        "repair_exact_value_regression": int(bool(no_value_tokens & carried_tokens) and not bool(cand_value_tokens & carried_tokens)),
        "repair_candidate_memory_overlap": cand_memory,
        "repair_no_history_memory_overlap": no_memory,
        "repair_memory_overlap_gain": memory_overlap_gain,
        "repair_candidate_current_overlap": cand_current,
        "repair_no_history_current_overlap": no_current,
        "repair_current_overlap_gain": current_overlap_gain,
        "repair_candidate_goal_overlap": cand_goal,
        "repair_no_history_goal_overlap": no_goal,
        "repair_goal_overlap_gain": goal_overlap_gain,
        "repair_candidate_instruction_overlap": cand_instruction,
        "repair_no_history_instruction_overlap": no_instruction,
        "repair_instruction_overlap_gain": instruction_overlap_gain,
        "repair_candidate_aligns_memory_more_than_no": int(memory_overlap_gain > 0),
        "repair_candidate_aligns_current_more_than_no": int(current_overlap_gain > 0),
        "repair_candidate_aligns_goal_more_than_no": int(goal_overlap_gain > 0),
        "repair_candidate_aligns_instruction_more_than_no": int(instruction_overlap_gain > 0),
        "repair_type_changed_and_memory_gain": int(no_type != cand_type and memory_overlap_gain > 0),
        "repair_type_changed_and_carried_gain": int(no_type != cand_type and value_overlap_gain > 0),
        "repair_same_type_but_value_gain": int(no_type == cand_type and value_overlap_gain > 0),
        "repair_same_type_but_memory_gain": int(no_type == cand_type and memory_overlap_gain > 0),
        "repair_click_coordinate_distance": distance if distance >= 0 else 0.0,
        "repair_click_coordinate_changed": int(distance >= 20),
        "repair_swipe_direction_changed": int(swipe_direction(no_action) != swipe_direction(cand_action)),
        "repair_candidate_is_terminate": int(cand_type == "terminate"),
        "repair_no_history_is_terminate": int(no_type == "terminate"),
        "repair_candidate_is_wait": int(cand_type == "wait"),
        "repair_no_history_is_wait": int(no_type == "wait"),
        "repair_candidate_is_type": int(cand_type == "type"),
        "repair_no_history_is_type": int(no_type == "type"),
    }
    repair_proxy = 0.0
    repair_proxy += 1.0 if no_type != cand_type else 0.0
    repair_proxy += 1.0 if no_is_system_fallback and cand_is_task else 0.0
    repair_proxy += 1.0 if value_overlap_gain > 0 else 0.0
    repair_proxy += 0.5 if memory_overlap_gain > 0 else 0.0
    repair_proxy -= 1.0 if cand_is_system_fallback and no_is_task else 0.0
    repair_proxy -= 1.0 if cand_type in {"missing", "wait"} else 0.0
    features["repair_proxy_score"] = repair_proxy
    return features


def exact_candidate_match(first: JsonDict | None, second: JsonDict | None) -> bool:
    return action_type(first) == action_type(second) and normalize_text(action_text(first)) == normalize_text(action_text(second))


def specificity_features(row: JsonDict, memory_key: str) -> dict[str, Any]:
    no_action = row.get("pred_actions", {}).get("no_history")
    cand_action = candidate_action(row, memory_key)
    dist_action = distractor_action(row, memory_key)
    no_type = action_type(no_action)
    cand_type = action_type(cand_action)
    dist_type = action_type(dist_action)
    cand_text = normalize_text(action_text(cand_action))
    dist_text = normalize_text(action_text(dist_action))
    no_text = normalize_text(action_text(no_action))
    cand_value = normalize_text(action_value_text(cand_action))
    dist_value = normalize_text(action_value_text(dist_action))
    no_value = normalize_text(action_value_text(no_action))
    cand_tokens = token_set(cand_text) | token_set(cand_value)
    dist_tokens = token_set(dist_text) | token_set(dist_value)
    no_tokens = token_set(no_text) | token_set(no_value)
    cand_dist_distance = coordinate_distance(cand_action, dist_action)
    cand_no_distance = coordinate_distance(cand_action, no_action)
    features: dict[str, Any] = {
        f"specificity_candidate_type={cand_type}": 1,
        f"specificity_distractor_type={dist_type}": 1,
        f"specificity_no_type={no_type}": 1,
        f"specificity_candidate_to_distractor={cand_type}->{dist_type}": 1,
        f"specificity_candidate_to_no={cand_type}->{no_type}": 1,
        f"specificity_distractor_button={button_name(dist_action) or 'none'}": 1,
        f"specificity_candidate_button={button_name(cand_action) or 'none'}": 1,
        f"specificity_candidate_distractor_distance={distance_bucket(cand_dist_distance)}": 1,
        f"specificity_candidate_no_distance={distance_bucket(cand_no_distance)}": 1,
        f"specificity_candidate_swipe_direction={swipe_direction(cand_action)}": 1,
        f"specificity_distractor_swipe_direction={swipe_direction(dist_action)}": 1,
        "specificity_candidate_equals_distractor_exact": int(exact_candidate_match(cand_action, dist_action)),
        "specificity_candidate_equals_distractor_type": int(cand_type == dist_type),
        "specificity_candidate_equals_distractor_text": int(bool(cand_text or dist_text) and cand_text == dist_text),
        "specificity_candidate_equals_distractor_value": int(bool(cand_value or dist_value) and cand_value == dist_value),
        "specificity_candidate_equals_no_exact": int(exact_candidate_match(cand_action, no_action)),
        "specificity_candidate_equals_no_type": int(cand_type == no_type),
        "specificity_distractor_equals_no_exact": int(exact_candidate_match(dist_action, no_action)),
        "specificity_distractor_equals_no_type": int(dist_type == no_type),
        "specificity_candidate_differs_from_no_and_distractor": int(
            not exact_candidate_match(cand_action, no_action) and not exact_candidate_match(cand_action, dist_action)
        ),
        "specificity_candidate_type_differs_from_no_and_distractor": int(cand_type != no_type and cand_type != dist_type),
        "specificity_candidate_distractor_jaccard": jaccard(cand_tokens, dist_tokens),
        "specificity_candidate_no_jaccard": jaccard(cand_tokens, no_tokens),
        "specificity_distractor_no_jaccard": jaccard(dist_tokens, no_tokens),
        "specificity_candidate_distractor_coordinate_distance": cand_dist_distance if cand_dist_distance >= 0 else 0.0,
        "specificity_candidate_no_coordinate_distance": cand_no_distance if cand_no_distance >= 0 else 0.0,
        "specificity_swipe_direction_matches_distractor": int(swipe_direction(cand_action) == swipe_direction(dist_action)),
        "specificity_swipe_direction_matches_no": int(swipe_direction(cand_action) == swipe_direction(no_action)),
    }
    specificity_proxy = 0.0
    specificity_proxy += 1.0 if not exact_candidate_match(cand_action, dist_action) else -1.0
    specificity_proxy += 1.0 if cand_type != dist_type else -0.5
    specificity_proxy += 0.5 if not exact_candidate_match(cand_action, no_action) else -0.5
    specificity_proxy += 0.5 if exact_candidate_match(dist_action, no_action) else 0.0
    features["specificity_proxy_score"] = specificity_proxy
    return features


def auxiliary_features(row: JsonDict, memory_key: str, use_candidate_features: bool, use_screen_features: bool, use_repair_features: bool, use_specificity_features: bool) -> dict[str, Any]:
    features: dict[str, Any] = {}
    if use_candidate_features:
        features.update(candidate_features(row, memory_key))
    if use_repair_features:
        features.update(repair_features(row, memory_key))
    if use_specificity_features:
        features.update(specificity_features(row, memory_key))
    if use_screen_features:
        features.update(text_overlap_features(row, memory_key))
    return features


def build_pair_dataset(rows: list[JsonDict], use_candidate_features: bool = False, use_screen_features: bool = False, use_repair_features: bool = False, use_specificity_features: bool = False) -> tuple[list[str], list[dict[str, Any]], list[int]]:
    texts = []
    feature_rows = []
    labels = []
    def add(row: JsonDict, memory_key: str, label: int) -> None:
        texts.append(pair_text(row, memory_key))
        feature_rows.append(auxiliary_features(row, memory_key, use_candidate_features, use_screen_features, use_repair_features, use_specificity_features))
        labels.append(label)
    for row in rows:
        label = row.get("utility_label")
        # Positive utility means true segment memory is specifically useful.
        if label == "positive":
            add(row, "true_memory", 1)
            add(row, "wrong_memory", 0)
            add(row, "no_memory", 0)
        elif label == "negative":
            add(row, "true_memory", 0)
        elif label in {"neutral", "nonspecific_positive", "summary_insufficient", "unresolved"}:
            add(row, "true_memory", 0)
            if label == "nonspecific_positive":
                add(row, "wrong_memory", 0)
    return texts, feature_rows, labels


def combine_features(text_matrix: Any, feature_matrix: Any | None) -> Any:
    if feature_matrix is None:
        return text_matrix
    return hstack([text_matrix, feature_matrix]).tocsr()


def train_model(train_rows: list[JsonDict], use_candidate_features: bool, use_screen_features: bool, use_repair_features: bool, use_specificity_features: bool) -> dict[str, Any]:
    texts, feature_rows, labels = build_pair_dataset(train_rows, use_candidate_features=use_candidate_features, use_screen_features=use_screen_features, use_repair_features=use_repair_features, use_specificity_features=use_specificity_features)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
    text_train = vectorizer.fit_transform(texts)
    feature_vectorizer = None
    feature_train = None
    if use_candidate_features or use_screen_features or use_repair_features or use_specificity_features:
        feature_vectorizer = DictVectorizer(sparse=True)
        feature_train = feature_vectorizer.fit_transform(feature_rows)
    x_train = combine_features(text_train, feature_train)
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    classifier.fit(x_train, labels)
    return {"vectorizer": vectorizer, "feature_vectorizer": feature_vectorizer, "classifier": classifier, "use_candidate_features": use_candidate_features, "use_screen_features": use_screen_features, "use_repair_features": use_repair_features, "use_specificity_features": use_specificity_features}


def score_rows(model: dict[str, Any], rows: list[JsonDict]) -> np.ndarray:
    texts = [pair_text(row, "true_memory") for row in rows]
    text_matrix = model["vectorizer"].transform(texts)
    feature_matrix = None
    if model.get("use_candidate_features") or model.get("use_screen_features") or model.get("use_repair_features") or model.get("use_specificity_features"):
        feature_rows = [
            auxiliary_features(
                row,
                "true_memory",
                bool(model.get("use_candidate_features")),
                bool(model.get("use_screen_features")),
                bool(model.get("use_repair_features")),
                bool(model.get("use_specificity_features")),
            )
            for row in rows
        ]
        feature_matrix = model["feature_vectorizer"].transform(feature_rows)
    x = combine_features(text_matrix, feature_matrix)
    return model["classifier"].predict_proba(x)[:, 1]


def is_positive(row: JsonDict) -> bool:
    return row.get("utility_label") == "positive"


def routed_value(row: JsonDict, use_memory: bool) -> bool:
    condition = "segment_summary" if use_memory else "no_history"
    return bool(row.get("condition_value_match", {}).get(condition))


def evaluate_split(model: dict[str, Any], rows: list[JsonDict]) -> JsonDict:
    scores = score_rows(model, rows)
    labels = np.array([is_positive(row) for row in rows], dtype=bool)
    no_history_acc = np.mean([routed_value(row, False) for row in rows]) if rows else 0.0
    segment_acc = np.mean([routed_value(row, True) for row in rows]) if rows else 0.0
    ap = average_precision_score(labels, scores) if np.any(labels) else 0.0
    try:
        auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.0
    except Exception:
        auc = 0.0
    thresholds = []
    for threshold in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]:
        pred = scores >= threshold
        tp = int(np.sum(labels & pred))
        fp = int(np.sum((~labels) & pred))
        fn = int(np.sum(labels & (~pred)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        routed_acc = np.mean([routed_value(row, bool(use_mem)) for row, use_mem in zip(rows, pred)]) if rows else 0.0
        regressions = sum(
            bool(use_mem)
            and bool(row.get("condition_value_match", {}).get("no_history"))
            and not bool(row.get("condition_value_match", {}).get("segment_summary"))
            for row, use_mem in zip(rows, pred)
        )
        thresholds.append({
            "threshold": threshold,
            "predicted_memory": int(np.sum(pred)),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "routed_value_acc": float(routed_acc),
            "segment_regressions": int(regressions),
        })
    top_examples = []
    order = np.argsort(-scores)[:100]
    for index in order:
        row = rows[int(index)]
        top_examples.append({
            "score": float(scores[int(index)]),
            "utility_label": row.get("utility_label"),
            "episode_id": row.get("metadata", {}).get("episode_id"),
            "case_id": row.get("metadata", {}).get("case_id"),
            "model_key": row.get("metadata", {}).get("model_key"),
            "thinking_mode": row.get("metadata", {}).get("thinking_mode"),
            "case_kind": row.get("metadata", {}).get("case_kind"),
            "gt_action_type": row.get("metadata", {}).get("gt_action_type"),
            "condition_value_match": row.get("condition_value_match"),
            "carried_values": row.get("metadata", {}).get("carried_values"),
        })
    return {
        "n": len(rows),
        "positive": int(np.sum(labels)),
        "average_precision": float(ap),
        "roc_auc": float(auc),
        "always_no_history_acc": float(no_history_acc),
        "always_segment_summary_acc": float(segment_acc),
        "thresholds": thresholds,
        "top_scored_examples": top_examples,
    }


def write_report(path: Path, results: dict[str, JsonDict]) -> None:
    lines = ["# Counterfactual Memory Utility Training Report", ""]
    for split, result in results.items():
        lines.append(f"## {split.title()}")
        lines.append("")
        lines.append(f"- n: {result['n']}")
        lines.append(f"- positive: {result['positive']}")
        lines.append(f"- average_precision: {result['average_precision']:.4f}")
        lines.append(f"- roc_auc: {result['roc_auc']:.4f}")
        lines.append(f"- always_no_history_acc: {result['always_no_history_acc']:.4f}")
        lines.append(f"- always_segment_summary_acc: {result['always_segment_summary_acc']:.4f}")
        lines.append("")
        lines.append("| threshold | predicted | precision | recall | f1 | routed_acc | regressions |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for item in result["thresholds"]:
            lines.append(
                f"| {item['threshold']:.2f} | {item['predicted_memory']} | {item['precision']:.4f} | "
                f"{item['recall']:.4f} | {item['f1']:.4f} | {item['routed_value_acc']:.4f} | {item['segment_regressions']} |"
            )
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("This scorer tests whether current-state/memory text plus optional counterfactual candidate features can identify positive counterfactual memory utility. The key comparison is memory precision and recall for positive utility, while tracking routed_acc and segment_regressions to expose deployment risk.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train counterfactual memory utility scorer")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-features", action="store_true")
    parser.add_argument("--screen-features", action="store_true")
    parser.add_argument("--repair-features", action="store_true")
    parser.add_argument("--specificity-features", action="store_true")
    parser.add_argument("--ocr-cache", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = load_split(data_dir, "train")
    ocr_cache = load_ocr_cache(args.ocr_cache)
    attach_ocr(train_rows, ocr_cache)
    model = train_model(
        train_rows,
        use_candidate_features=args.candidate_features,
        use_screen_features=args.screen_features,
        use_repair_features=args.repair_features,
        use_specificity_features=args.specificity_features,
    )
    results = {}
    for split in ["train", "dev", "test"]:
        rows = load_split(data_dir, split)
        attach_ocr(rows, ocr_cache)
        results[split] = evaluate_split(model, rows)
    joblib.dump(model, output_dir / "memory_utility_model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for split, result in results.items():
        with (output_dir / f"{split}_top_scored_examples.jsonl").open("w", encoding="utf-8") as handle:
            for row in result["top_scored_examples"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_report(output_dir / "memory_utility_report.md", results)
    print(f"trained counterfactual memory utility scorer output={output_dir}")


if __name__ == "__main__":
    main()