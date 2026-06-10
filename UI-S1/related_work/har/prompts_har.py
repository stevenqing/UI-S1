"""HAR prompt templates for GUI-360 evaluation.

Two prompt styles:
  1. har_native: HAR's original action space (CLICK/TYPE/SCROLL/COMPLETE)
     for evaluating the HAR-GUI-3B model with its trained format.
  2. har_gui360: Our GUI-360 action space (<tool_call> JSON) but with HAR's
     <think>/<answer> reasoning format, for applying HAR's method to our model.

All prompts match the original HAR codebase exactly:
  - github.com/BigTaige/HAR  Prompts/Inference.py, Prompts/Act2Sum.py
  - github.com/BigTaige/HAR  Inference/demo_episode_inference.py
"""


# ---------------------------------------------------------------------------
# HAR's native action space — exact copy from demo_episode_inference.py
# (includes BACK, HOME, separator, and OTHER_CUSTOM_ACTIONS placeholder)
# ---------------------------------------------------------------------------
HAR_ACTION_SPACE = """
CLICK:(x,y): Click on the element at the coordinate point (x,y) on the screen, e.g., CLICK:(1980,224).
TYPE:typed_text: An action of typing a piece of text, e.g., TYPE:"Macbook-Pro 16G Black".
COMPLETE: The goal has been completed in the current screen state.
SCROLL:UP/DOWN/LEFT/RIGHT: Scroll in a specific direction, e.g., SCROLL:UP.
LONG_PRESS:(x,y): Long press at a specific point (x,y) on the screen, e.g., LONG_PRESS:(345,2218).
BACK: Go back to the previous screen, e.g, BACK.
HOME: Go to the home screen, e.g., HOME.
=========================================
OTHER_CUSTOM_ACTIONS: ...
"""


# ---------------------------------------------------------------------------
# GUI-360 action space adapted for HAR's <think>/<answer> output format
# Uses our standard tool_call actions but wrapped in HAR reasoning tags
# ---------------------------------------------------------------------------
GUI360_HAR_ACTION_SPACE = """<action>
- click
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to click at.
    - button: str, One of 'left', 'right', 'middle' or 'x' (Default: 'left')
    - double: bool, Whether to perform a double click (Default: False)
    - pressed: str|None, Keyboard key to press while clicking (Default: None)
  - Example: click(coordinate=[100, 100], button='left', double=False, pressed=None)
- type
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to type at.
    - keys: str, The key to input.
    - clear_current_text: bool, Whether to clear the current text (Default: False)
    - control_focus: bool, Whether to focus on selected control before typing (Default: True)
  - Example: type(coordinate=[100, 100], keys='Hello')
- drag
  - Args:
    - start_coordinate: [x, y], where the drag starts.
    - end_coordinate: [x, y], where the drag ends.
    - button: str, 'left' or 'right' (Default: 'left')
    - duration: float, Duration in seconds (Default: 1.0)
  - Example: drag(start_coordinate=[100, 100], end_coordinate=[200, 200])
- wheel_mouse_input
  - Args:
    - coordinate: [x, y], position on the screen to scroll.
    - wheel_dist: int, Wheel notches. Positive=up, negative=down.
  - Example: wheel_mouse_input(coordinate=[100, 100], wheel_dist=-5)
</action>"""


# ---------------------------------------------------------------------------
# HAR inference prompt — exact structure from demo_episode_inference.py
# Note: leading \n, blank line before "The output format" — matches original
# ---------------------------------------------------------------------------
HAR_INFERENCE_PROMPT = """
You are a skilled assistant, interacting with the screen to accomplish the user's goals.
Here is the action space:
{action_space}
Your overall goal is: <goal>{instruction}</goal>
Actions completed at previous steps: <history>{history}</history>

The output format should be as follows:
<think>Analyze step by step based on guidance and screen state to choose the action.</think>
<answer>The action you finally choose from "action space".</answer>"""


# ---------------------------------------------------------------------------
# GUI-360 + HAR inference prompt for har_method experiments
# Uses <tool_call> inside <answer> tag, same structure otherwise
# ---------------------------------------------------------------------------
HAR_GUI360_INFERENCE_PROMPT = """
You are a skilled assistant, interacting with the screen to accomplish the user's goals.
Here is the action space:
{action_space}
Your overall goal is: <goal>{instruction}</goal>
Actions completed at previous steps: <history>{history}</history>

The output format should be as follows:
<think>Analyze step by step based on guidance and screen state to choose the action.</think>
<answer>
Output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished, output status as "FINISH":
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>
</answer>"""


# ---------------------------------------------------------------------------
# Act2Sum prompt — exact structure from demo_episode_inference.py
# Note: includes Action space section (original has it!)
# ---------------------------------------------------------------------------
HAR_ACT2SUM_PROMPT = """
Step-by-step GUI navigation task. Briefly summarize the current action.
Action space:
{action_space}
Goal: <goal>{instruction}</goal>
Current action: <action>{action_text}</action>

Output Format: <summary>One-sentence summary of the action based on the screen image.</summary>"""


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_har_prompt(
    prompt_style: str,
    instruction: str,
    history: str,
    action_space: str,
) -> str:
    """Build the HAR inference prompt for the given style."""
    if prompt_style == "har_native":
        return HAR_INFERENCE_PROMPT.format(
            action_space=action_space,
            instruction=instruction,
            history=history,
        )
    elif prompt_style == "har_gui360":
        return HAR_GUI360_INFERENCE_PROMPT.format(
            action_space=action_space,
            instruction=instruction,
            history=history,
        )
    else:
        raise ValueError(f"Unknown prompt_style: {prompt_style}")


def format_act2sum_prompt(
    instruction: str,
    action_text: str,
    action_space: str,
) -> str:
    """Build the Act2Sum prompt."""
    return HAR_ACT2SUM_PROMPT.format(
        action_space=action_space,
        instruction=instruction,
        action_text=action_text,
    )
