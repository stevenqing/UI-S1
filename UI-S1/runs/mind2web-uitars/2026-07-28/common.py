import ast
import hashlib
import json
import re
from pathlib import Path


MODEL_REVISIONS = {
    "ByteDance-Seed/UI-TARS-2B-SFT": "f366a1db3e7f29635f5b236d6a71dea367a0a700",
    "ByteDance-Seed/UI-TARS-7B-SFT": "3434901a9dd04dd3625617d839a5724fe5e2db20",
    "ByteDance-Seed/UI-TARS-72B-SFT": "8e7a03104915dee7bdbae5ea6e5a80264b316f9e",
}
PROMPT_TEMPLATE = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x2,y2)<|box_end|>')
hotkey(key='ctrl c')
type(content='xxx')
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait()
finished()

## Note
- Use English in the Thought part.
- Summarize your next action and its target element in one sentence in the Thought part.

## User Instruction
{task}"""
POINT_PATTERN = re.compile(
    r"(?:<\|box_start\|>\s*)?\(\s*([0-9]{1,4})\s*,\s*([0-9]{1,4})\s*\)"
    r"(?:\s*<\|box_end\|>)?|<point>\s*([0-9]{1,4})\s+([0-9]{1,4})\s*</point>",
    re.I,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


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


def format_prompt(sample: dict) -> str:
    return PROMPT_TEMPLATE.format(task=sample["task"])


def prompt_sha256(sample: dict) -> str:
    return sha256_bytes(format_prompt(sample).encode("utf-8"))


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


def _literal_string(node: ast.AST, field: str) -> str:
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        raise ValueError(f"{field} must be a string literal")
    return node.value


def _normalized_point(value: str) -> list[float]:
    match = POINT_PATTERN.fullmatch(value.strip())
    if not match:
        raise ValueError("missing UI-TARS point")
    values = match.groups()
    x = int(values[0] or values[2])
    y = int(values[1] or values[3])
    if x > 1000 or y > 1000:
        raise ValueError("point coordinate outside 0-1000")
    return [x / 1000, y / 1000]


def parse_prediction(response: str) -> dict:
    parts = re.split(r"\bAction\s*:\s*", response, flags=re.I)
    if len(parts) < 2:
        raise ValueError("missing Action marker")
    action_text = parts[-1].strip().splitlines()[0].strip()
    try:
        expression = ast.parse(action_text, mode="eval").body
    except SyntaxError as error:
        raise ValueError("invalid action expression") from error
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        raise ValueError("action must be a direct function call")
    if expression.args:
        raise ValueError("positional action arguments are unsupported")
    arguments = {keyword.arg: keyword.value for keyword in expression.keywords}
    if None in arguments:
        raise ValueError("expanded action arguments are unsupported")

    function = expression.func.id.lower()
    if function == "click":
        point_node = arguments.get("start_box") or arguments.get("point")
        if point_node is None:
            raise ValueError("click requires an explicit point")
        return {
            "action": "CLICK",
            "value": None,
            "position": _normalized_point(_literal_string(point_node, "point")),
        }
    if function == "type":
        content_node = arguments.get("content")
        if content_node is None:
            raise ValueError("type requires explicit content")
        content = _literal_string(content_node, "content")
        if not content:
            raise ValueError("type content is empty")
        return {"action": "TYPE", "value": content, "position": None}
    raise ValueError(f"unsupported Mind2Web action: {function}")