"""Prepare GUI-360 training data in HiconAgent format for SFT.

Converts existing GUI-360 step-level SFT data to HiconAgent multi-image format:
  - Prompt: HiconAgent template with visual history (τ=2 previous screenshots)
  - Response: <think>reasoning</think>\n<action>\n<tool_call>...</tool_call>\n</action>

Key difference from HAR SFT data:
  - Multi-image: up to τ+1 images per sample (τ history + 1 current)
  - History: previous screenshots + action text (not Act2Sum summaries)
  - Output: <think>/<action> tags (not <think>/<answer>)
  - Image tags: one <image> per image in the images array

Usage:
    python related_work/hiconagent/prepare_hiconagent_sft_data.py \
        --input_file v15_gui_360/data/gui360_balanced_train.jsonl \
        --output_file related_work/hiconagent/data/gui360_hiconagent_train.jsonl \
        --tau_history 2
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple


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


def format_action_for_history(action: Optional[Dict], step_id: int) -> str:
    """Format an action dict into a text description for history."""
    if action is None:
        return f"Step {step_id}: performed an action"

    func = action.get("function", "")
    args = action.get("args", {})
    status = action.get("status", "CONTINUE")

    if status == "FINISH":
        return f"Step {step_id}: completed the task"

    if func == "click":
        coord = args.get("coordinate", [0, 0])
        return f"click(coordinate=[{coord[0]}, {coord[1]}])"
    elif func == "type":
        keys = args.get("keys", args.get("text", ""))
        if len(keys) > 40:
            keys = keys[:40] + "..."
        return f"type(coordinate={args.get('coordinate', [0, 0])}, keys=\"{keys}\")"
    elif func == "drag":
        start = args.get("start_coordinate", [0, 0])
        end = args.get("end_coordinate", [0, 0])
        return f"drag(start_coordinate=[{start[0]}, {start[1]}], end_coordinate=[{end[0]}, {end[1]}])"
    elif func == "wheel_mouse_input":
        coord = args.get("coordinate", [0, 0])
        dist = args.get("wheel_dist", 0)
        return f"wheel_mouse_input(coordinate=[{coord[0]}, {coord[1]}], wheel_dist={dist})"
    else:
        return f"{func}({args})"


def get_episode_dir(image_path: str) -> str:
    """Get the episode directory from an image path."""
    parts = image_path.rsplit('/', 1)
    return parts[0] if len(parts) > 1 else ''


# GUI-360 action space (same as HAR / hiconagent prompts)
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


# HiconAgent prompt template
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


def convert_sample(
    sample: Dict,
    episode_history: List[Dict],
    tau_history: int,
) -> Tuple[Dict, Dict]:
    """Convert a single step-level sample to HiconAgent multi-image format.

    Args:
        sample: original SFT sample with conversations and images
        episode_history: list of dicts with 'image' and 'action_text' for prior steps
        tau_history: number of previous screenshots to include

    Returns:
        (new_sample, history_entry) where history_entry has 'image' and 'action_text'
    """
    human_text = sample["conversations"][0]["value"]
    gpt_text = sample["conversations"][1]["value"]
    images = sample.get("images", [])
    current_image = images[0] if images else ""

    instruction = extract_instruction(human_text)
    reasoning = extract_reasoning(gpt_text)
    action = extract_action_from_response(gpt_text)

    # Determine step index within episode
    step_idx = len(episode_history)

    # Build history section and image list
    all_images = []
    image_tags = []

    if episode_history:
        # Get τ most recent history entries
        recent_history = episode_history[-tau_history:]
        history_action_texts = []

        for entry in recent_history:
            all_images.append(entry["image"])
            image_tags.append("<image>")
            history_action_texts.append(entry["action_text"])

        # Build history section text
        lines = ["The previous steps and their screenshots are shown above."]
        for j, action_desc in enumerate(history_action_texts):
            lines.append(f"Step {j + 1} action: {action_desc}")
        lines.append("Current screenshot is shown below.")
        history_section = "\n".join(lines)
    else:
        history_section = "This is the task's initial state. The current screenshot is shown below."

    # Add current image
    all_images.append(current_image)
    image_tags.append("<image>")

    # Build prompt
    prompt = HICONAGENT_PROMPT_TEMPLATE.format(
        action_space=ACTION_SPACE,
        instruction=instruction,
        history_section=history_section,
    )

    # Build human message: <image> tags for history images, then prompt with current <image>
    # Each <image> corresponds to one entry in the images array
    human_value_parts = []
    # History image tags with action descriptions interleaved
    if episode_history:
        recent_history = episode_history[-tau_history:]
        for j, entry in enumerate(recent_history):
            human_value_parts.append("<image>")
            human_value_parts.append(f"Previous action: {entry['action_text']}")
    # Current image tag + prompt
    human_value_parts.append(f"<image>\n{prompt}")

    human_value = "\n".join(human_value_parts)

    # Build response: <think>reasoning</think>\n<action>\n<tool_call>...</tool_call>\n</action>
    tool_call_m = re.search(r'(<tool_call>.*?</tool_call>)', gpt_text, re.DOTALL)
    tool_call_str = tool_call_m.group(1) if tool_call_m else ""

    if reasoning and tool_call_str:
        response = f"<think>{reasoning}</think>\n<action>\n{tool_call_str}\n</action>"
    elif tool_call_str:
        response = f"<think>Based on the screen state, I will take the next action.</think>\n<action>\n{tool_call_str}\n</action>"
    else:
        response = f"<think>{reasoning}</think>\n<action>{gpt_text}</action>"

    new_sample = {
        "conversations": [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": response},
        ],
        "images": all_images,
    }

    # Create history entry for next step
    action_text = format_action_for_history(action, step_idx + 1)
    history_entry = {
        "image": current_image,
        "action_text": action_text,
    }

    return new_sample, history_entry


def main():
    parser = argparse.ArgumentParser(
        description="Convert GUI-360 SFT data to HiconAgent multi-image format")
    parser.add_argument("--input_file", required=True,
                        help="Input JSON/JSONL file (LLaMA-Factory ShareGPT format)")
    parser.add_argument("--output_file", required=True,
                        help="Output JSONL file")
    parser.add_argument("--tau_history", type=int, default=2,
                        help="Visual history window size (default: 2)")
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

    # Process: group by episode, convert with visual history
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    converted = 0
    current_ep_dir = None
    episode_history = []  # list of {'image': path, 'action_text': str}

    # Statistics
    image_count_dist = {}  # num_images -> count

    with open(args.output_file, 'w') as out_f:
        for i, sample in enumerate(data):
            images = sample.get("images", [])
            ep_dir = get_episode_dir(images[0]) if images else ""

            # New episode: reset history
            if ep_dir != current_ep_dir:
                current_ep_dir = ep_dir
                episode_history = []

            new_sample, history_entry = convert_sample(
                sample, episode_history, args.tau_history)
            episode_history.append(history_entry)

            # Track image count distribution
            n_imgs = len(new_sample["images"])
            image_count_dist[n_imgs] = image_count_dist.get(n_imgs, 0) + 1

            out_f.write(json.dumps(new_sample, ensure_ascii=False) + "\n")
            converted += 1

            if (i + 1) % 10000 == 0:
                print(f"  Processed {i+1}/{len(data)} samples...")

    print(f"\nDone! Converted {converted} samples -> {args.output_file}")
    print(f"\nImage count distribution:")
    for n_imgs in sorted(image_count_dist.keys()):
        count = image_count_dist[n_imgs]
        pct = count / converted * 100
        print(f"  {n_imgs} images: {count} samples ({pct:.1f}%)")

    # Verify: check that <image> tag count matches images array length
    print(f"\nVerifying <image> tag count matches images array length...")
    errors = 0
    with open(args.output_file) as f:
        for line_num, line in enumerate(f, 1):
            sample = json.loads(line)
            n_images = len(sample["images"])
            n_tags = sample["conversations"][0]["value"].count("<image>")
            if n_images != n_tags:
                print(f"  ERROR line {line_num}: {n_images} images but {n_tags} <image> tags")
                errors += 1
                if errors >= 10:
                    print("  (stopping after 10 errors)")
                    break
    if errors == 0:
        print(f"  All {converted} samples verified OK!")
    else:
        print(f"  Found {errors} mismatches!")

    # Show a sample
    with open(args.output_file) as f:
        sample = json.loads(f.readline())
    print(f"\n=== Sample output (step 0, {len(sample['images'])} images) ===")
    print("Human (first 400 chars):")
    print(sample["conversations"][0]["value"][:400])
    print("\nGPT response:")
    print(sample["conversations"][1]["value"][:500])

    # Show a multi-image sample if available
    with open(args.output_file) as f:
        for line in f:
            sample = json.loads(line)
            if len(sample["images"]) > 1:
                print(f"\n=== Multi-image sample ({len(sample['images'])} images) ===")
                print("Human (first 500 chars):")
                print(sample["conversations"][0]["value"][:500])
                print("\nGPT response:")
                print(sample["conversations"][1]["value"][:500])
                break


if __name__ == "__main__":
    main()
