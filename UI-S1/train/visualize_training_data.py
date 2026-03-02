#!/usr/bin/env python3
"""
Visualize training tasks and prompts for UI-S1 GRPO training.

This script allows you to:
1. View the task/goal for each trajectory
2. See the prompts built for each step
3. Display screenshots associated with steps
4. Understand how training data is structured

Usage:
    python visualize_training_data.py --data-file datasets/ui_s1_train.jsonl --idx 0
    python visualize_training_data.py --data-file datasets/ui_s1_train.jsonl --idx 0 --show-images
    python visualize_training_data.py --data-file datasets/ui_s1_train.jsonl --sample 5
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    import matplotlib.pyplot as plt
    from PIL import Image
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib/PIL not available. Image display disabled.")


def load_training_data(data_file: str, max_items: int = None) -> List[Dict]:
    """Load training data from JSONL file."""
    data = []
    with open(data_file, 'r') as f:
        for i, line in enumerate(f):
            if max_items and i >= max_items:
                break
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError:
                continue
    return data


def build_prompt(history: List[Dict], current_step: Dict, goal: str = None) -> str:
    """
    Build the prompt from history and current step context.
    This mirrors the prompt building in train_srun_grpo_worker.py
    """
    prompt_parts = []

    # Add goal/task instruction
    if goal:
        prompt_parts.append(f"Task: {goal}")
        prompt_parts.append("")

    # Add history of previous actions
    if history:
        prompt_parts.append("Previous actions:")
        for i, h in enumerate(history):
            action_type = h.get('action_type', h.get('action', ''))
            text = h.get('text', '')
            coord = h.get('coordinate')
            bbox = h.get('bbox')

            if text:
                prompt_parts.append(f"  Step {i+1}: {action_type}(text=\"{text}\")")
            elif coord:
                prompt_parts.append(f"  Step {i+1}: {action_type}(coordinate={coord})")
            elif bbox:
                prompt_parts.append(f"  Step {i+1}: {action_type}(bbox={bbox})")
            else:
                prompt_parts.append(f"  Step {i+1}: {action_type}()")
        prompt_parts.append("")

    # Add current context
    prompt_parts.append("Current step:")
    if 'screenshot' in current_step:
        prompt_parts.append("  [Screenshot provided]")

    prompt_parts.append("")
    prompt_parts.append("Predict the next action:")

    return "\n".join(prompt_parts)


def format_action(action_content: Dict) -> str:
    """Format action content for display."""
    action_type = action_content.get('action', 'unknown')
    text = action_content.get('text', '')
    coord = action_content.get('coordinate')
    bbox = action_content.get('bbox')
    status = action_content.get('status', '')
    button = action_content.get('button', '')
    time_val = action_content.get('time')

    parts = [f"Action: {action_type}"]
    if text:
        parts.append(f"  text: \"{text}\"")
    if coord:
        parts.append(f"  coordinate: {coord}")
    if bbox:
        parts.append(f"  bbox: {bbox}")
    if status:
        parts.append(f"  status: {status}")
    if button:
        parts.append(f"  button: {button}")
    if time_val:
        parts.append(f"  time: {time_val}")

    return "\n".join(parts)


def visualize_trajectory(
    trajectory: Dict,
    traj_idx: int,
    show_images: bool = False,
    save_dir: str = None
):
    """Visualize a single trajectory with its task and prompts."""
    goal = trajectory.get('goal', 'No goal specified')
    is_successful = trajectory.get('is_successful', False)
    steps = trajectory.get('steps', [])

    print("\n" + "=" * 80)
    print(f"TRAJECTORY #{traj_idx}")
    print("=" * 80)
    print(f"\nTASK/GOAL: {goal}")
    print(f"SUCCESS: {is_successful}")
    print(f"NUM STEPS: {len(steps)}")
    print("-" * 80)

    history = []
    images_to_show = []

    for step_idx, step in enumerate(steps):
        action_content = step.get('action_content', {})
        screenshot_path = step.get('screenshot', '')

        # Build prompt for this step
        prompt = build_prompt(history, step, goal)

        print(f"\n--- Step {step_idx + 1}/{len(steps)} ---")
        print("\nPROMPT:")
        print("-" * 40)
        for line in prompt.split('\n'):
            print(f"  {line}")
        print("-" * 40)

        print("\nGROUND TRUTH ACTION:")
        print(format_action(action_content))

        if screenshot_path:
            print(f"\nSCREENSHOT: {screenshot_path}")
            if show_images and HAS_MATPLOTLIB and os.path.exists(screenshot_path):
                images_to_show.append((step_idx, screenshot_path, action_content))

        # Update history for next step
        history.append({
            'action_type': action_content.get('action', ''),
            'text': action_content.get('text', ''),
            'coordinate': action_content.get('coordinate'),
            'bbox': action_content.get('bbox'),
        })

    # Show images if requested
    if show_images and images_to_show and HAS_MATPLOTLIB:
        n_images = len(images_to_show)
        cols = min(4, n_images)
        rows = (n_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if n_images == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]

        for i, (step_idx, img_path, action) in enumerate(images_to_show):
            row = i // cols
            col = i % cols
            ax = axes[row][col] if rows > 1 else axes[col]

            try:
                img = Image.open(img_path)
                ax.imshow(img)
                action_type = action.get('action', '')
                ax.set_title(f"Step {step_idx + 1}: {action_type}", fontsize=10)
                ax.axis('off')

                # Draw bbox if available
                bbox = action.get('bbox')
                if bbox and len(bbox) == 4:
                    from matplotlib.patches import Rectangle
                    x1, y1, x2, y2 = bbox
                    rect = Rectangle((x1, y1), x2-x1, y2-y1,
                                     linewidth=2, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)

                # Draw coordinate if available
                coord = action.get('coordinate')
                if coord and len(coord) == 2:
                    ax.scatter([coord[0]], [coord[1]], c='red', s=100, marker='x')

            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading image:\n{str(e)}",
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')

        # Hide empty subplots
        for i in range(n_images, rows * cols):
            row = i // cols
            col = i % cols
            ax = axes[row][col] if rows > 1 else axes[col]
            ax.axis('off')

        plt.suptitle(f"Trajectory #{traj_idx}: {goal[:60]}...", fontsize=12)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"trajectory_{traj_idx}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nSaved visualization to: {save_path}")

        plt.show()

    print("\n" + "=" * 80)


def visualize_prompt_template():
    """Show the prompt template structure used during training."""
    print("\n" + "=" * 80)
    print("PROMPT TEMPLATE STRUCTURE")
    print("=" * 80)
    print("""
The training prompts are built dynamically for each step:

1. TASK/GOAL: The high-level instruction (e.g., "Open the Zoho Meet app")

2. PREVIOUS ACTIONS: History of actions taken so far
   - Each action includes: type, text/coordinate/bbox
   - Example: "Step 1: click(text='Settings')"

3. CURRENT STEP CONTEXT:
   - Screenshot of current screen state
   - Any additional context

4. PREDICTION REQUEST: "Predict the next action:"

ACTION FORMAT (Ground Truth):
- action: The action type (click, type, scroll, swipe, open, wait, terminate, etc.)
- coordinate: [x, y] click/tap location
- bbox: [x1, y1, x2, y2] bounding box of target element
- text: Text content for type actions or element identification
- status: For terminate actions (success/fail)
- button: For system_button actions (Back, Home, etc.)
- time: For wait actions (seconds)

REWARD COMPUTATION:
- Action type matching: 30% weight
- Action content matching: 70% weight
- Rewards are propagated through trajectory using discount factor (gamma=0.5)
""")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Visualize UI-S1 training data")
    parser.add_argument(
        "--data-file",
        type=str,
        default="datasets/ui_s1_train.jsonl",
        help="Path to training data JSONL file"
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=None,
        help="Index of specific trajectory to visualize"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of random trajectories to sample and visualize"
    )
    parser.add_argument(
        "--show-images",
        action="store_true",
        help="Display screenshot images (requires matplotlib)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save visualization images"
    )
    parser.add_argument(
        "--show-template",
        action="store_true",
        help="Show the prompt template structure"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show dataset statistics"
    )

    args = parser.parse_args()

    if args.show_template:
        visualize_prompt_template()
        return

    # Load data
    print(f"Loading data from: {args.data_file}")
    data = load_training_data(args.data_file)
    print(f"Loaded {len(data)} trajectories")

    if args.stats:
        # Show statistics
        print("\n" + "=" * 80)
        print("DATASET STATISTICS")
        print("=" * 80)

        total_steps = sum(len(t.get('steps', [])) for t in data)
        successful = sum(1 for t in data if t.get('is_successful', False))

        action_types = {}
        for t in data:
            for step in t.get('steps', []):
                action = step.get('action_content', {}).get('action', 'unknown')
                action_types[action] = action_types.get(action, 0) + 1

        print(f"Total trajectories: {len(data)}")
        print(f"Successful trajectories: {successful} ({100*successful/len(data):.1f}%)")
        print(f"Total steps: {total_steps}")
        print(f"Average steps per trajectory: {total_steps/len(data):.1f}")
        print(f"\nAction type distribution:")
        for action, count in sorted(action_types.items(), key=lambda x: -x[1]):
            print(f"  {action}: {count} ({100*count/total_steps:.1f}%)")
        return

    # Visualize specific trajectory
    if args.idx is not None:
        if args.idx >= len(data):
            print(f"Error: Index {args.idx} out of range (max: {len(data)-1})")
            return
        visualize_trajectory(data[args.idx], args.idx, args.show_images, args.save_dir)

    # Sample random trajectories
    elif args.sample:
        indices = random.sample(range(len(data)), min(args.sample, len(data)))
        for idx in indices:
            visualize_trajectory(data[idx], idx, args.show_images, args.save_dir)

    else:
        # Default: show first trajectory
        visualize_trajectory(data[0], 0, args.show_images, args.save_dir)


if __name__ == "__main__":
    main()
