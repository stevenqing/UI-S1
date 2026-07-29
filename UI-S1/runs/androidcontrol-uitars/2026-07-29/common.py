import ast
import hashlib
import json
import re
from pathlib import Path


MODEL_REVISIONS = {
    "ByteDance-Seed/UI-TARS-2B-SFT": "f366a1db3e7f29635f5b236d6a71dea367a0a700",
    "ByteDance-Seed/UI-TARS-7B-SFT": "3434901a9dd04dd3625617d839a5724fe5e2db20",
}
PROMPT_TEMPLATE = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(point='<point>x1 y1</point>')
long_press(point='<point>x1 y1</point>')
type(content='')
scroll(point='<point>x1 y1</point>', direction='down or up or right or left')
open_app(app_name='')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
press_home()
press_back()
wait()
finished(content='xxx')

## Note
- Use English in the Thought part.
- Write a small plan and finally summarize your next action and its target element in one sentence.

## User Instruction
{instruction}

## Action History
{history}"""
POINT_PATTERN = re.compile(
    r"(?:<point>\s*)?([0-9]{1,4})[\s,]+([0-9]{1,4})(?:\s*</point>)?"
    r"|(?:<\|box_start\|>\s*)?\(\s*([0-9]{1,4})\s*,\s*([0-9]{1,4})\s*\)"
    r"(?:\s*<\|box_end\|>)?",
    re.I,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def format_instruction(row: dict, setting: str) -> str:
    if setting == "low":
        return f'{row["goal"]} You need to: {row["low_instruction_audit_only"]}'
    if setting == "high":
        return row["goal"]
    raise ValueError(f"unsupported setting: {setting}")


def format_prompt(row: dict, setting: str) -> str:
    return PROMPT_TEMPLATE.format(
        instruction=format_instruction(row, setting),
        history=row["history"],
    )


def prompt_sha256(row: dict, setting: str) -> str:
    return hashlib.sha256(format_prompt(row, setting).encode()).hexdigest()


def _literal_string(node: ast.AST, field: str) -> str:
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        raise ValueError(f"{field} must be a string literal")
    return node.value


def _point(value: str) -> tuple[int, int]:
    match = POINT_PATTERN.fullmatch(value.strip())
    if not match:
        raise ValueError("missing UI-TARS point")
    groups = match.groups()
    x = int(groups[0] or groups[2])
    y = int(groups[1] or groups[3])
    if x > 1000 or y > 1000:
        raise ValueError("point coordinate outside 0-1000")
    return x, y


def parse_prediction(response: str) -> str:
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

    if function in {"click", "long_press"}:
        point_node = arguments.get("point") or arguments.get("start_box")
        if point_node is None:
            raise ValueError(f"{function} requires an explicit point")
        x, y = _point(_literal_string(point_node, "point"))
        action = "CLICK" if function == "click" else "LONG_PRESS"
        return f"{action} <point>[[{x},{y}]]</point>"
    if function == "type":
        content_node = arguments.get("content")
        if content_node is None:
            raise ValueError("type requires explicit content")
        return f"TYPE [{_literal_string(content_node, 'content')}]"
    if function == "scroll":
        direction_node = arguments.get("direction")
        if direction_node is None:
            raise ValueError("scroll requires an explicit direction")
        direction = _literal_string(direction_node, "direction").upper()
        if direction not in {"UP", "DOWN", "LEFT", "RIGHT"}:
            raise ValueError("unsupported scroll direction")
        return f"SCROLL [{direction}]"
    if function == "open_app":
        app_node = arguments.get("app_name")
        if app_node is None:
            raise ValueError("open_app requires an explicit app name")
        return f"OPEN_APP [{_literal_string(app_node, 'app_name')}]"
    if function in {"press_back", "press_home", "wait"}:
        if arguments:
            raise ValueError(f"{function} does not accept arguments")
        return {"press_back": "PRESS_BACK", "press_home": "PRESS_HOME", "wait": "WAIT"}[function]
    raise ValueError(f"unsupported AndroidControl action: {function}")