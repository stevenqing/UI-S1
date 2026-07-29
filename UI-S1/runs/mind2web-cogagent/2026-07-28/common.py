import hashlib
import json
import re
from pathlib import Path


MODEL_NAME = "zai-org/cogagent-chat-hf"
MODEL_REVISION = "26eec27a44348fbe0c9fad89348cf6a505f5a5ae"
TOKENIZER_NAME = "lmsys/vicuna-7b-v1.5"
TOKENIZER_REVISION = "3321f76e3f527bd14065daf69dad9344000a201d"
PROMPT_TEMPLATE = 'What steps do I need to take to "{task}"?(with grounding)'
BOX_PATTERN = re.compile(
    r"\[\[\s*([0-9]{1,4})\s*,\s*([0-9]{1,4})\s*,\s*"
    r"([0-9]{1,4})\s*,\s*([0-9]{1,4})\s*\]\]"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def format_prompt(sample: dict) -> str:
    return PROMPT_TEMPLATE.format(task=sample["task"])


def prompt_sha256(sample: dict) -> str:
    return sha256_bytes(format_prompt(sample).encode("utf-8"))


def read_json(path: Path) -> list[dict]:
    value = json.loads(path.read_text())
    if not isinstance(value, list):
        raise ValueError(f"expected JSON list: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def expected_answer(sample: dict) -> dict:
    step = sample["step"]
    action = step["operation"]["op"]
    if action == "TYPE":
        value = step["operation"]["value"]
    else:
        match = re.search(r"\]\s+(.*?)\s+->", sample["step_repr"])
        value = match.group(1) if match else None
    bbox = step["bbox"]
    width, height = sample["img_size"]
    position = [
        round((bbox["x"] + bbox["width"] / 2) / width, 2),
        round((bbox["y"] + bbox["height"] / 2) / height, 2),
    ]
    return {"action": action, "value": value, "position": position}


def parse_prediction(response: str) -> dict:
    marker = re.search(r"Grounded\s+Operation\s*:\s*(.*)", response, re.I | re.S)
    if not marker:
        raise ValueError("missing Grounded Operation marker")
    operation = marker.group(1).strip()
    boxes = list(BOX_PATTERN.finditer(operation))
    if not boxes:
        raise ValueError("missing 0-1000 bounding box")
    box = [int(value) for value in boxes[-1].groups()]
    if any(value < 0 or value > 1000 for value in box):
        raise ValueError("bounding box coordinate outside 0-1000")
    if box[0] > box[2] or box[1] > box[3]:
        raise ValueError("invalid bounding box ordering")

    prefix = operation[: boxes[-1].start()].strip()
    type_match = re.search(
        r"\bTYPE\s*:\s*(.+?)(?:\s+(?:at|in|into)\s+(?:the\s+)?.*)?$",
        prefix,
        re.I | re.S,
    )
    select_match = re.search(
        r"\bSELECT\s*:\s*(.+?)(?:\s+(?:at|in|from)\s+(?:the\s+)?.*)?$",
        prefix,
        re.I | re.S,
    )
    if type_match:
        action, value = "TYPE", type_match.group(1).strip(" \t\n'\"")
    elif select_match:
        action, value = "SELECT", select_match.group(1).strip(" \t\n'\"")
    elif re.search(r"\b(?:CLICK|LEFT_CLICK|TAP)\b", prefix, re.I):
        action, value = "CLICK", None
    else:
        raise ValueError("unsupported or implicit action")
    if action in {"TYPE", "SELECT"} and not value:
        raise ValueError(f"{action} requires an explicit value")

    return {
        "action": action,
        "value": value,
        "bbox_1000": box,
        "position": [
            ((box[0] + box[2]) / 2) / 1000,
            ((box[1] + box[3]) / 2) / 1000,
        ],
    }