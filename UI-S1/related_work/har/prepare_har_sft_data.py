"""Prepare GUI-360 training data in HAR format for SFT.

Converts existing GUI-360 step-level SFT data to HAR format:
  - Prompt: HAR template with Act2Sum-style concise history (k=4 window)
  - Response: <think>reasoning</think>\n<answer><tool_call>...</tool_call></answer>

The key difference from standard SFT:
  - History: concise 1-sentence summaries (template-based Act2Sum) vs verbose
  - Output: <think>/<answer> tags wrapping the reasoning and action
  - History window: k=4 most recent steps (HAR default)

Usage:
    python related_work/har/prepare_har_sft_data.py \
        --input_file train_GUI_360/llamafactory/data/gui360_train.json \
        --output_file related_work/har/data/gui360_har_train.jsonl \
        --k_history 4
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional


def extract_instruction(human_text: str) -> str:
    """Extract instruction from the human prompt."""
    m = re.search(r'The instruction is:\n(.+?)\n\nThe history of actions are:', human_text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def extract_action_from_response(gpt_text: str) -> Optional[Dict]:
    """Extract the action dict from GPT response containing <tool_call>."""
    m = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', gpt_text, re.DOTALL)
    if not m:
        return None
    try:
        tc = json.loads(m.group(1))
        return tc
    except json.JSONDecodeError:
        return None


def extract_reasoning(gpt_text: str) -> str:
    """Extract reasoning text before <tool_call> tag."""
    m = re.search(r'<tool_call>', gpt_text)
    if m:
        return gpt_text[:m.start()].strip()
    return gpt_text.strip()


def action_to_summary(action: Optional[Dict]) -> str:
    """Convert an action dict to a 1-sentence Act2Sum-style summary.

    Template-based summaries that approximate what Act2Sum would generate.
    """
    if action is None:
        return "Performed an action on the screen"

    func = action.get("function", "")
    args = action.get("args", {})
    status = action.get("status", "CONTINUE")

    if status == "FINISH":
        return "Completed the task"

    if func == "click":
        coord = args.get("coordinate", [0, 0])
        button = args.get("button", "left")
        double = args.get("double", False)
        if double:
            return f"Double-clicked at position ({coord[0]}, {coord[1]})"
        elif button == "right":
            return f"Right-clicked at position ({coord[0]}, {coord[1]})"
        else:
            return f"Clicked at position ({coord[0]}, {coord[1]}) on the screen"

    elif func == "type":
        text = args.get("keys", args.get("text", ""))
        if len(text) > 40:
            text = text[:40] + "..."
        return f'Typed "{text}" into the input field'

    elif func == "drag":
        start = args.get("start_coordinate", [0, 0])
        end = args.get("end_coordinate", [0, 0])
        return f"Dragged from ({start[0]}, {start[1]}) to ({end[0]}, {end[1]})"

    elif func == "wheel_mouse_input":
        dist = args.get("wheel_dist", 0)
        direction = "up" if dist > 0 else "down"
        return f"Scrolled {direction} on the page"

    else:
        return f"Performed {func} action on the screen"


def get_episode_dir(image_path: str) -> str:
    """Get the episode directory from an image path."""
    parts = image_path.rsplit('/', 1)
    return parts[0] if len(parts) > 1 else ''


# HAR prompt template (adapted for GUI-360 action space)
# Matches HAR's structure: leading \n, blank line before "The output format"
HAR_PROMPT_TEMPLATE = """
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

# GUI-360 action space (simplified for the prompt)
ACTION_SPACE = """<action>
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


def convert_sample(
    sample: Dict,
    act2sum_history: List[str],
    k_history: int,
) -> Dict:
    """Convert a single step-level sample to HAR format."""
    human_text = sample["conversations"][0]["value"]
    gpt_text = sample["conversations"][1]["value"]
    images = sample.get("images", [])

    instruction = extract_instruction(human_text)
    reasoning = extract_reasoning(gpt_text)
    action = extract_action_from_response(gpt_text)

    # Build Act2Sum-style history — matches original HAR format
    # Original: 'Step' + str(i+1) + ': ' + act2sum_ + ".\n" (trailing \n)
    if not act2sum_history:
        history_text = "This is the task's initial state."
    else:
        recent = act2sum_history[-k_history:]
        history_text = ""
        for j, s in enumerate(recent):
            history_text += "Step" + str(j + 1) + ": " + s + ".\n"

    # Build HAR prompt
    prompt = HAR_PROMPT_TEMPLATE.format(
        action_space=ACTION_SPACE,
        instruction=instruction,
        history=history_text,
    )

    # Build HAR response: <think>reasoning</think>\n<answer><tool_call>...</tool_call></answer>
    tool_call_m = re.search(r'(<tool_call>.*?</tool_call>)', gpt_text, re.DOTALL)
    tool_call_str = tool_call_m.group(1) if tool_call_m else ""

    if reasoning and tool_call_str:
        response = f"<think>{reasoning}</think>\n<answer>\n{tool_call_str}\n</answer>"
    elif tool_call_str:
        response = f"<think>Based on the screen state, I will take the next action.</think>\n<answer>\n{tool_call_str}\n</answer>"
    else:
        response = f"<think>{reasoning}</think>\n<answer>{gpt_text}</answer>"

    # Create new sample
    new_sample = {
        "conversations": [
            {"from": "human", "value": f"<image>\n{prompt}"},
            {"from": "gpt", "value": response},
        ],
        "images": images,
    }

    # Generate summary for next step's history
    summary = action_to_summary(action)

    return new_sample, summary


def main():
    parser = argparse.ArgumentParser(description="Convert GUI-360 SFT data to HAR format")
    parser.add_argument("--input_file", required=True,
                        help="Input JSON file (LLaMA-Factory ShareGPT format)")
    parser.add_argument("--output_file", required=True,
                        help="Output JSONL file")
    parser.add_argument("--k_history", type=int, default=4,
                        help="Act2Sum history window size (default: 4)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to process (for debugging)")
    args = parser.parse_args()

    # Load input data
    print(f"Loading {args.input_file}...")
    if args.input_file.endswith('.jsonl'):
        data = []
        with open(args.input_file) as f:
            for line in f:
                data.append(json.loads(line))
    else:
        data = json.load(open(args.input_file))

    if args.max_samples:
        data = data[:args.max_samples]
    print(f"  Loaded {len(data)} samples")

    # Process: group by episode, convert with Act2Sum history
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    converted = 0
    current_ep_dir = None
    act2sum_history = []

    with open(args.output_file, 'w') as out_f:
        for i, sample in enumerate(data):
            images = sample.get("images", [])
            ep_dir = get_episode_dir(images[0]) if images else ""

            # New episode: reset history
            if ep_dir != current_ep_dir:
                current_ep_dir = ep_dir
                act2sum_history = []

            new_sample, summary = convert_sample(sample, act2sum_history, args.k_history)
            act2sum_history.append(summary)

            out_f.write(json.dumps(new_sample, ensure_ascii=False) + "\n")
            converted += 1

            if (i + 1) % 10000 == 0:
                print(f"  Processed {i+1}/{len(data)} samples...")

    print(f"Done! Converted {converted} samples → {args.output_file}")

    # Show a sample
    with open(args.output_file) as f:
        sample = json.loads(f.readline())
    print("\n=== Sample output ===")
    print("Human (first 300 chars):")
    print(sample["conversations"][0]["value"][:300])
    print("\nGPT response:")
    print(sample["conversations"][1]["value"][:500])


if __name__ == "__main__":
    main()
