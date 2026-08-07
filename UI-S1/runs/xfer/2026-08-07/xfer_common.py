import json
import math
import re
from collections import Counter, defaultdict


COORDINATE_ACTIONS = {"CLICK", "TYPE", "SELECT", "LONG_PRESS", "SCROLL"}
MIND2WEB_ACTIONS = {"CLICK", "TYPE", "SELECT"}
JSON_PATTERN = re.compile(r"\{.*?\}", re.S)
UITARS_ACTION_PATTERN = re.compile(r"action\s*:\s*([a-zA-Z_]+)\s*\((.*?)\)", re.I | re.S)
UITARS_POINT_PATTERN = re.compile(r"(?:start_box|position)\s*=\s*['\"]?\(?\s*([0-9]{1,4})\s*,\s*([0-9]{1,4})", re.I)
UITARS_VALUE_PATTERN = re.compile(r"(?:content|value)\s*=\s*['\"](.*?)['\"]", re.I | re.S)
COG_BOX_PATTERN = re.compile(r"\[\[\s*([0-9]{1,4})\s*,\s*([0-9]{1,4})\s*,\s*([0-9]{1,4})\s*,\s*([0-9]{1,4})\s*\]\]")


def normalize_action(value):
    value = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "TAP": "CLICK",
        "LEFT_CLICK": "CLICK",
        "INPUT_TEXT": "TYPE",
        "NAVIGATE_BACK": "PRESS_BACK",
        "BACK": "PRESS_BACK",
    }
    return aliases.get(value, value)


def parse_product_response(response, allowed_actions):
    candidates = JSON_PATTERN.findall(response)
    if not candidates:
        raise ValueError("missing JSON object")
    value = json.loads(candidates[-1])
    action = normalize_action(value.get("action"))
    if action not in allowed_actions:
        raise ValueError(f"unsupported action: {action}")
    parameter = value.get("value")
    if action in {"TYPE", "SELECT"}:
        if not isinstance(parameter, str) or not parameter.strip():
            raise ValueError(f"{action} requires a parameter")
        parameter = parameter.strip()
    else:
        parameter = None
    position = value.get("position")
    if action in COORDINATE_ACTIONS:
        if not isinstance(position, list) or len(position) != 2:
            raise ValueError(f"{action} requires position [x,y]")
        position = [float(position[0]), float(position[1])]
        if not all(math.isfinite(item) and 0 <= item <= 1 for item in position):
            raise ValueError("position outside [0,1]")
    else:
        position = None
    return {"action": action, "value": parameter, "position": position, "parse_ok": True}


def parse_uitars_response(response, allowed_actions):
    match = UITARS_ACTION_PATTERN.search(response)
    if not match:
        raise ValueError("missing UI-TARS action call")
    action = normalize_action(match.group(1))
    action = {"CLICK": "CLICK", "TYPE": "TYPE", "SELECT": "SELECT"}.get(action, action)
    if action not in allowed_actions:
        raise ValueError(f"unsupported UI-TARS action: {action}")
    arguments = match.group(2)
    point_match = UITARS_POINT_PATTERN.search(arguments)
    if not point_match:
        raise ValueError("UI-TARS action lacks point")
    x, y = map(int, point_match.groups())
    if x > 1000 or y > 1000:
        raise ValueError("UI-TARS point outside 0-1000")
    value_match = UITARS_VALUE_PATTERN.search(arguments)
    value = value_match.group(1).strip() if value_match else None
    if action in {"TYPE", "SELECT"} and not value:
        raise ValueError(f"{action} requires a value")
    return {
        "action": action,
        "value": value if action in {"TYPE", "SELECT"} else None,
        "position": [x / 1000, y / 1000],
        "parse_ok": True,
    }


def parse_cogagent_response(response, allowed_actions):
    marker = re.search(r"Grounded\s+Operation\s*:\s*(.*)", response, re.I | re.S)
    if not marker:
        raise ValueError("missing Grounded Operation")
    operation = marker.group(1).strip()
    boxes = list(COG_BOX_PATTERN.finditer(operation))
    if not boxes:
        raise ValueError("missing CogAgent bbox")
    box = list(map(int, boxes[-1].groups()))
    if any(value < 0 or value > 1000 for value in box) or box[0] > box[2] or box[1] > box[3]:
        raise ValueError("invalid CogAgent bbox")
    prefix = operation[:boxes[-1].start()].strip()
    type_match = re.search(r"\bTYPE\s*:\s*(.+?)(?:\s+(?:at|in|into)\s+.*)?$", prefix, re.I | re.S)
    select_match = re.search(r"\bSELECT\s*:\s*(.+?)(?:\s+(?:at|in|from)\s+.*)?$", prefix, re.I | re.S)
    if type_match:
        action, value = "TYPE", type_match.group(1).strip(" \t\n'\"")
    elif select_match:
        action, value = "SELECT", select_match.group(1).strip(" \t\n'\"")
    elif re.search(r"\b(?:CLICK|LEFT_CLICK|TAP)\b", prefix, re.I):
        action, value = "CLICK", None
    else:
        raise ValueError("unsupported CogAgent action")
    if action not in allowed_actions or (action in {"TYPE", "SELECT"} and not value):
        raise ValueError("invalid CogAgent product action")
    return {
        "action": action,
        "value": value,
        "position": [((box[0] + box[2]) / 2) / 1000, ((box[1] + box[3]) / 2) / 1000],
        "parse_ok": True,
    }


def token_set_f1(left, right):
    left = set(str(left or "").lower().split())
    right = set(str(right or "").lower().split())
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    precision = overlap / len(left)
    recall = overlap / len(right)
    return 0.0 if overlap == 0 else 2 * precision * recall / (precision + recall)


def plurality_action(candidates, model_order):
    counts = Counter(candidate["action"] for candidate in candidates if candidate.get("parse_ok"))
    if not counts:
        return None
    reliability = defaultdict(float)
    for candidate in candidates:
        if candidate.get("parse_ok"):
            reliability[candidate["action"]] += float(candidate.get("development_reliability", 0.0))
    frozen_order = {model: index for index, model in enumerate(model_order)}
    first_index = {}
    for index, candidate in enumerate(candidates):
        if candidate.get("parse_ok"):
            first_index.setdefault(candidate["action"], index)
    return max(
        counts,
        key=lambda action: (
            counts[action],
            reliability[action],
            -first_index[action],
            -min(frozen_order.get(candidate.get("model"), len(model_order)) for candidate in candidates if candidate.get("parse_ok") and candidate["action"] == action),
        ),
    )


def complete_link_groups(points, threshold=14.0):
    groups = []
    assigned = set()
    for index in range(len(points)):
        if index in assigned:
            continue
        group = [index]
        assigned.add(index)
        for candidate in range(len(points)):
            if candidate in assigned:
                continue
            if all(abs(points[member][0] - points[candidate][0]) <= threshold and abs(points[member][1] - points[candidate][1]) <= threshold for member in group):
                group.append(candidate)
                assigned.add(candidate)
        groups.append(tuple(group))
    groups.sort(key=lambda values: (-len(values), min(values)))
    return groups


def select_parameter(candidates):
    values = [candidate["value"] for candidate in candidates if candidate.get("value")]
    if not values:
        return None
    scores = [sum(token_set_f1(value, other) for other in values) for value in values]
    return values[max(range(len(values)), key=lambda index: (scores[index], -index))]


def aggregate(candidates, model_order, image_size, threshold_pixels=14.0):
    action = plurality_action(candidates, model_order)
    if action is None:
        return {"action": None, "value": None, "position": None, "parse_ok": False}
    retained = [candidate for candidate in candidates if candidate.get("parse_ok") and candidate["action"] == action]
    parameter = select_parameter(retained) if action in {"TYPE", "SELECT"} else None
    if action not in COORDINATE_ACTIONS:
        return {"action": action, "value": parameter, "position": None, "parse_ok": True}
    width, height = image_size
    points = [[candidate["position"][0] * width, candidate["position"][1] * height] for candidate in retained]
    groups = complete_link_groups(points, threshold_pixels)
    winner = groups[0]
    selected = max(
        winner,
        key=lambda index: (
            float(retained[index].get("development_reliability", 0.0)),
            -model_order.index(retained[index]["model"]),
            -index,
        ),
    )
    return {
        "action": action,
        "value": parameter,
        "position": retained[selected]["position"],
        "parse_ok": True,
        "winning_group": list(winner),
    }


def test_contracts():
    models = ["A", "B", "C"]
    candidates = [
        {"model": "A", "action": "CLICK", "value": None, "position": [0.5, 0.5], "parse_ok": True},
        {"model": "B", "action": "CLICK", "value": None, "position": [0.501, 0.501], "parse_ok": True},
        {"model": "C", "action": "TYPE", "value": "hello world", "position": [0.2, 0.2], "parse_ok": True},
    ]
    result = aggregate(candidates, models, (1280, 720))
    assert result["action"] == "CLICK" and result["position"] == [0.5, 0.5]
    assert parse_product_response('{"action":"TYPE","value":"hello","position":[0.1,0.2]}', MIND2WEB_ACTIONS)["action"] == "TYPE"
    assert parse_uitars_response("action: click(start_box='(472,58)')", MIND2WEB_ACTIONS)["position"] == [0.472, 0.058]
    assert parse_cogagent_response("Grounded Operation: CLICK [[100,200,300,400]]", MIND2WEB_ACTIONS)["position"] == [0.2, 0.3]
    assert math.isclose(token_set_f1("hello world", "hello"), 2 / 3)
    assert plurality_action([
        {"model": "A", "action": "WAIT", "parse_ok": True},
        {"model": "B", "action": "WAIT", "parse_ok": True},
        {"model": "C", "action": "CLICK", "parse_ok": True},
    ], models) == "WAIT"


if __name__ == "__main__":
    test_contracts()
    print("PASS")
