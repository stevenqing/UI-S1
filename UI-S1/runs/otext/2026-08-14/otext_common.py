import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
RAW_ROOT = RUN_DIR / "raw"


def normalize_text(value):
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    value = " ".join(value.split())
    return value.strip(" \t\n\r.,;:!?()[]{}<>\"'`~_-|/\\")


def extract_literals(instruction, family):
    if family == "quoted":
        values = [value for groups in re.findall(r"['\"]([^'\"]+)['\"]|[“]([^”]+)[”]|[‘]([^’]+)[’]", instruction) for value in groups if value]
    elif family == "caps_camel":
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", instruction)
        accepted = []
        for token in tokens:
            letters = "".join(character for character in token if character.isalpha())
            is_caps = len(letters) >= 2 and letters.isupper()
            is_camel = bool(re.search(r"[a-z][A-Z]|[A-Z][a-z]+[A-Z]", token))
            accepted.append(token if is_caps or is_camel else None)
        values = []
        current = []
        for token in accepted + [None]:
            if token is not None:
                current.append(token)
            elif current:
                values.append(" ".join(current)); current = []
    elif family == "full_normalized":
        values = [normalize_text(instruction)]
    else:
        raise ValueError(family)
    return [value for value in values if len(normalize_text(value)) >= 3]


def edit_distance(left, right):
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, 1):
        current = [left_index]
        for right_index, right_value in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[right_index] + 1, previous[right_index - 1] + (left_value != right_value)))
        previous = current
    return previous[-1]


def edit_similarity(left, right):
    left = normalize_text(left); right = normalize_text(right)
    if not left or not right:
        return 0.0
    return 1 - edit_distance(left, right) / max(len(left), len(right))


def box_center(polygon):
    values = np.asarray(polygon, dtype=np.float64)
    return float(values[:, 0].mean()), float(values[:, 1].mean())


def score_box(box, literals, matcher):
    raw = str(box["text"]).strip(); normalized = normalize_text(raw)
    similarities = []
    for literal in literals:
        if matcher == "exact":
            similarities.append(float(bool(raw and raw in literal)))
        elif matcher == "normalized":
            normalized_literal = normalize_text(literal)
            similarities.append(float(bool(normalized and normalized in normalized_literal)))
        elif matcher == "edit":
            similarities.append(edit_similarity(raw, literal))
        else:
            raise ValueError(matcher)
    lexical = max(similarities, default=0.0)
    return lexical * float(box["confidence"]), lexical, len(normalized)


def best_ocr_box(boxes, instruction, extractor, matcher):
    literals = extract_literals(instruction, extractor)
    if not literals:
        return {"literals": [], "score": 0.0, "point": None, "box": None}
    candidates = []
    for box in boxes:
        score, lexical, length = score_box(box, literals, matcher)
        top = min(point[1] for point in box["polygon"]); left = min(point[0] for point in box["polygon"])
        candidates.append((score, float(box["confidence"]), length, -top, -left, -int(box["engine_order"]), lexical, box))
    if not candidates:
        return {"literals": literals, "score": 0.0, "point": None, "box": None}
    selected = max(candidates)
    return {"literals": literals, "score": float(selected[0]), "lexical": float(selected[6]), "point": box_center(selected[7]["polygon"]), "box": selected[7]}


def complete_link_groups(points, threshold=14.0):
    groups = []; assigned = set()
    for index in range(len(points)):
        if index in assigned:
            continue
        group = [index]; assigned.add(index)
        for candidate in range(len(points)):
            if candidate in assigned:
                continue
            if all(abs(points[member][0] - points[candidate][0]) <= threshold and abs(points[member][1] - points[candidate][1]) <= threshold for member in group):
                group.append(candidate); assigned.add(candidate)
        groups.append(tuple(group))
    return groups


def weighted_b3(points, weights):
    if len(points) != len(weights) or not points:
        raise ValueError("OTEXT weighted B3 input mismatch")
    groups = complete_link_groups(points)
    winner = max(groups, key=lambda group: (sum(weights[index] for index in group), -groups.index(group)))
    return max(winner, key=lambda index: (weights[index], -index)), winner


def load_raw(engine):
    output = {}
    for path in sorted((RAW_ROOT / engine).glob("shard-*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row["row_id"] in output:
                    raise ValueError(f"OTEXT duplicate OCR row: {engine}/{row['row_id']}")
                output[row["row_id"]] = row
    if len(output) != 1581:
        raise ValueError(f"OTEXT OCR coverage mismatch: {engine}/{len(output)}")
    return output
