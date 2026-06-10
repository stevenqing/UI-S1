"""HiconAgent prompt templates for GUI-360 evaluation.

HiconAgent (CVPR 2026) uses multi-image visual history with <think>/<action> format.
Key difference from HAR: previous screenshots are included as multi-image input,
not just text summaries.

Two modes:
  1. hiconagent_method:           Visual history (τ=2 previous screenshots) + action text
  2. hiconagent_method_no_visual: Text-only history (ablation, same prompt but no images)

Reference: HiconAgent — "History Context-aware Policy Optimization for Long-horizon GUI Agents"
"""


# ---------------------------------------------------------------------------
# GUI-360 action space (same as HAR's gui360 action space)
# ---------------------------------------------------------------------------
GUI360_HICONAGENT_ACTION_SPACE = """<action>
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
# HiconAgent inference prompt template
# Uses <think>/<action> tags (different from HAR's <think>/<answer>)
# ---------------------------------------------------------------------------
HICONAGENT_PROMPT_TEMPLATE = """You are a skilled assistant, interacting with the screen to accomplish the user's goals.
Here is the action space:
{action_space}
Your overall goal is: <goal>{instruction}</goal>
{history_section}
The output format should be as follows:
<think>Analyze step by step based on guidance and screen state to choose the action.</think>
<action>
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
</action>"""


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_history_section_visual(history_actions: list) -> str:
    """Format history section when visual history images are present.

    The images are interleaved before the prompt text in the multi-image
    content array, so this section only contains the text descriptions.
    """
    if not history_actions:
        return "This is the task's initial state. The current screenshot is shown below."

    lines = ["The previous steps and their screenshots are shown above."]
    for i, action_desc in enumerate(history_actions):
        lines.append(f"Step {i + 1} action: {action_desc}")
    lines.append("Current screenshot is shown below.")
    return "\n".join(lines)


def format_history_section_text(history_actions: list) -> str:
    """Format history section for text-only mode (no visual history)."""
    if not history_actions:
        return "This is the task's initial state. The current screenshot is shown below."

    lines = ["The previous steps are:"]
    for i, action_desc in enumerate(history_actions):
        lines.append(f"Step {i + 1} action: {action_desc}")
    lines.append("Current screenshot is shown below.")
    return "\n".join(lines)


def format_hiconagent_prompt(
    instruction: str,
    history_section: str,
    action_space: str = None,
) -> str:
    """Build the HiconAgent inference prompt."""
    if action_space is None:
        action_space = GUI360_HICONAGENT_ACTION_SPACE
    return HICONAGENT_PROMPT_TEMPLATE.format(
        action_space=action_space,
        instruction=instruction,
        history_section=history_section,
    )
