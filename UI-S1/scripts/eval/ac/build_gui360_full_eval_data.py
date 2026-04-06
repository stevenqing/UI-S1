"""Build full GUI-360 test evaluation pickle from gui360_test.jsonl.

Matches the format of train/p_emergent/gui360_eval_data.pkl:
  - Each sample: {episode_id, step_num, messages, gt_response, screenshot_ele}
  - System prompt: GUI agent with <think>/<action> format
  - User message: "User Instruction: {goal}" + screenshot image
  - GT response: <think>{thought}</think>\n<action>{action_json}</action>
"""

import argparse
import json
import os
import pickle

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

SYSTEM_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.
## Output Format

```
<think> ... </think>

<action> ... </action>
```

## Action Space

You can perform the following actions:
- click: Click at the specified (x, y) pixel coordinate on the screen.
- type: Type the specified text string.
- drag: Drag from coordinate to coordinate2.
- wheel_mouse_input: Scroll the mouse wheel at the given position. text indicates direction/amount.
- select_text: Select text from coordinate to coordinate2.
- select_table_range: Select a range of table cells from coordinate to coordinate2.
- summary: Provide a summary or answer as text output.

The arguments you can use are:
- coordinate: (x, y): The x and y pixel coordinates from the left and top edges of the screen.
- coordinate2: (x, y): The end coordinates for drag or selection actions.
- text: Text input for type, summary, or scroll direction.

Format your output as a JSON object with the selected action and its arguments at the same level.

Example outputs:
<think>
...
</think>
<action>
{"action": "click", "coordinate": "<value>"}
</action>
<think>
...
</think>
<action>
{"action": "type", "text": "<value>"}
</action>

## Note

- Planing the task and explain your reasoning step-by-step in `think` part.
- Write your action in the `action` part according to the action space."""


def build_gt_response(step):
    """Build GT response in <think>...<action>... format."""
    thought = step.get('thought', '')
    action_content = step['action_content']
    action_json = json.dumps(action_content, ensure_ascii=False)
    return f"<think>\n{thought}\n</think>\n<action>\n{action_json}\n</action>"


def main():
    parser = argparse.ArgumentParser(description="Build full GUI-360 test eval data pickle")
    parser.add_argument("--input", type=str,
                        default=os.path.join(PROJECT_ROOT, 'datasets', 'GUI-360', 'rl_data', 'gui360_test.jsonl'))
    parser.add_argument("--output", type=str,
                        default=os.path.join(PROJECT_ROOT, 'train', 'p_emergent', 'gui360_eval_data_fulltest.pkl'))
    parser.add_argument("--max_episodes", type=int, default=0, help="0=all")
    args = parser.parse_args()

    print(f"Loading from {args.input}...")
    samples = []
    n_episodes = 0
    n_skipped_images = 0

    with open(args.input) as f:
        for line_idx, line in enumerate(f):
            ep = json.loads(line.strip())
            eid = ep['execution_id']
            goal = ep['goal']
            n_episodes += 1

            if args.max_episodes > 0 and n_episodes > args.max_episodes:
                break

            for step_idx, step in enumerate(ep['steps']):
                screenshot_path = step.get('screenshot', '')

                # Verify image exists
                if not os.path.exists(screenshot_path):
                    n_skipped_images += 1
                    continue

                # Build messages (same format as existing pickle)
                messages = [
                    {
                        'role': 'system',
                        'content': [{'text': SYSTEM_PROMPT}],
                    },
                    {
                        'role': 'user',
                        'content': [
                            {'text': f"User Instruction: {goal}"},
                            {'image': screenshot_path},
                        ],
                    },
                ]

                gt_response = build_gt_response(step)

                sample = {
                    'episode_id': eid,
                    'step_num': step_idx,
                    'messages': messages,
                    'gt_response': gt_response,
                    'screenshot_ele': {},  # placeholder
                }
                samples.append(sample)

            if n_episodes % 500 == 0:
                print(f"  Processed {n_episodes} episodes, {len(samples)} samples...")

    print(f"\nTotal: {n_episodes} episodes, {len(samples)} samples")
    if n_skipped_images > 0:
        print(f"Skipped {n_skipped_images} steps with missing images")

    print(f"Saving to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(samples, f)
    print(f"Done. {len(samples)} samples saved.")


if __name__ == "__main__":
    main()
