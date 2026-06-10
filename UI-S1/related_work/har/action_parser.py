"""HAR action output parser.

Converts HAR's <think>/<answer> format and action strings (CLICK:(x,y),
TYPE:text, SCROLL:UP, COMPLETE, etc.) to our internal action dict format
used by compute_step_reward().

Scroll coordinates assume 1040x736 resolution with centered ~250px swipes.
"""

import re
from typing import Any, Dict, Optional, Tuple


# Scroll mappings for 1040x736 resolution, centered ~250px swipe
_SCROLL_MAP = {
    "UP":    {"coordinate": [520, 500], "endCoordinate": [520, 250]},
    "DOWN":  {"coordinate": [520, 250], "endCoordinate": [520, 500]},
    "LEFT":  {"coordinate": [700, 368], "endCoordinate": [340, 368]},
    "RIGHT": {"coordinate": [340, 368], "endCoordinate": [700, 368]},
}


def parse_har_output(text: str) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """Parse HAR <think>/<answer> output.

    Returns (think_text, answer_text, parsed_action_dict).
    parsed_action_dict is None for COMPLETE or unparseable output.
    """
    # Extract think/answer tags
    think_text = ""
    answer_text = ""

    think_m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_m:
        think_text = think_m.group(1).strip()

    answer_m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_m:
        answer_text = answer_m.group(1).strip()
    else:
        # If no <answer> tag, try everything after </think>
        if think_m:
            remainder = text[think_m.end():].strip()
            if remainder:
                answer_text = remainder
        elif text.strip():
            answer_text = text.strip()

    parsed = _parse_har_action(answer_text)
    return think_text, answer_text, parsed


def _parse_har_action(action_str: str) -> Optional[Dict[str, Any]]:
    """Parse a single HAR action string to internal format.

    Supported formats:
      CLICK:(x,y)            -> {"action": "click", "coordinate": [x, y]}
      TYPE:text / TYPE:"text" -> {"action": "type", "text": "text"}
      SCROLL:UP/DOWN/LEFT/RIGHT -> {"action": "swipe", ...}
      LONG_PRESS:(x,y)       -> {"action": "long_press", "coordinate": [x, y]}
      COMPLETE               -> None (finish signal)
      BACK                   -> None (not in our action space)
      HOME                   -> None (not in our action space)
    """
    if not action_str:
        return None

    s = action_str.strip()

    # COMPLETE / IMPOSSIBLE
    if s.upper() in ("COMPLETE", "IMPOSSIBLE"):
        return None

    # BACK / HOME / PRESS_RECENT / ENTER — not in our action space
    if s.upper() in ("BACK", "HOME", "PRESS_RECENT", "ENTER"):
        return None

    # CLICK:(x,y)
    m = re.match(r'CLICK:\s*\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)', s, re.IGNORECASE)
    if m:
        x, y = float(m.group(1)), float(m.group(2))
        return {"action": "click", "coordinate": [int(x), int(y)]}

    # LONG_PRESS:(x,y)
    m = re.match(r'LONG_PRESS:\s*\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)', s, re.IGNORECASE)
    if m:
        x, y = float(m.group(1)), float(m.group(2))
        return {"action": "long_press", "coordinate": [int(x), int(y)]}

    # TYPE:text or TYPE:"text" or TYPE:(x,y,text)
    # Format 1: TYPE:(x,y,text) — Mind2Web style
    m = re.match(r'TYPE:\s*\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(.*?)\s*\)', s, re.IGNORECASE)
    if m:
        text = m.group(3).strip().strip('"').strip("'")
        return {"action": "type", "text": text}

    # Format 2: TYPE:text or TYPE:"text"
    m = re.match(r'TYPE:\s*(.+)', s, re.IGNORECASE)
    if m:
        text = m.group(1).strip().strip('"').strip("'")
        return {"action": "type", "text": text}

    # SCROLL:direction
    m = re.match(r'SCROLL:\s*(UP|DOWN|LEFT|RIGHT)', s, re.IGNORECASE)
    if m:
        direction = m.group(1).upper()
        scroll = _SCROLL_MAP.get(direction)
        if scroll:
            return {
                "action": "swipe",
                "coordinate": list(scroll["coordinate"]),
                "endCoordinate": list(scroll["endCoordinate"]),
            }

    return None


def extract_summary(text: str) -> str:
    """Extract content from <summary>...</summary> tags."""
    m = re.search(r'<summary>(.*?)</summary>', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def format_action_text(action: Optional[Dict[str, Any]]) -> str:
    """Format an internal action dict back to a human-readable string.

    Used as fallback for Act2Sum when <answer> tag is missing.
    """
    if action is None:
        return "COMPLETE"

    atype = action.get("action", "")
    coord = action.get("coordinate")

    if atype == "click" and coord:
        return f"CLICK:({coord[0]},{coord[1]})"
    elif atype == "long_press" and coord:
        return f"LONG_PRESS:({coord[0]},{coord[1]})"
    elif atype == "type":
        text = action.get("text", "")
        return f'TYPE:"{text}"'
    elif atype == "swipe":
        start = action.get("coordinate", [0, 0])
        end = action.get("endCoordinate", [0, 0])
        # Determine direction from coordinates
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        if abs(dy) >= abs(dx):
            direction = "UP" if dy < 0 else "DOWN"
        else:
            direction = "LEFT" if dx < 0 else "RIGHT"
        return f"SCROLL:{direction}"
    else:
        return str(action)
