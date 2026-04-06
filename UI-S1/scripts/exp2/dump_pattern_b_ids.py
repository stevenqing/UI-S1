"""Dump Pattern B (genuine multi-subtask) trajectory IDs to a JSON file.

Reuses classification logic from analyze_pattern_b_subtask.py.
Output: pattern_b_ids.json in the same directory as this script.
"""

import json
import os
import re
from collections import defaultdict

GUI360_DATA = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/GUI-360/test/data"


def is_open_app_subtask(subtask_str):
    """Check if a subtask is just 'Open the application of ...' pattern."""
    s = subtask_str.lower().strip()
    return (
        s.startswith("open the application") or
        s.startswith("open the file") or
        s.startswith("open the powerpoint") or
        s.startswith("open the excel") or
        s.startswith("open the word") or
        s.startswith("launch the") or
        re.match(r"^open .*\.(pptx?|xlsx?|docx?|csv)", s) is not None
    )


def load_gui360_subtask_info():
    """Load trajectory subtask structure from GUI-360 test data."""
    traj_subtasks = {}

    for domain in ["ppt", "excel", "word"]:
        for category in ["in_app", "online", "search"]:
            data_dir = os.path.join(GUI360_DATA, domain, category, "success")
            if not os.path.isdir(data_dir):
                continue

            for fname in sorted(os.listdir(data_dir)):
                if not fname.endswith(".jsonl"):
                    continue

                fpath = os.path.join(data_dir, fname)
                traj_id = f"{domain}_{category}_{fname.replace('.jsonl', '')}"

                steps = []
                with open(fpath) as f:
                    for line in f:
                        data = json.loads(line)
                        subtask = data.get("step", {}).get("subtask", "")
                        action_func = data.get("step", {}).get("action", {}).get("function", "")
                        if action_func == "drag":
                            continue
                        if not data.get("step", {}).get("action", {}).get("rectangle", {}):
                            continue
                        steps.append(subtask)

                if not steps:
                    continue

                segments = []
                current = steps[0]
                count = 1
                for i in range(1, len(steps)):
                    if steps[i] != current and steps[i]:
                        segments.append((current, count))
                        current = steps[i]
                        count = 1
                    else:
                        count += 1
                segments.append((current, count))

                traj_subtasks[traj_id] = segments

    return traj_subtasks


def classify_trajectory(segments):
    """Classify a multi-subtask trajectory into Pattern A/B/C."""
    if len(segments) <= 1:
        return "single"

    subtask_strs = [s[0] for s in segments]
    has_open_app = any(is_open_app_subtask(s) for s in subtask_strs)
    non_open_subtasks = [s for s in subtask_strs if not is_open_app_subtask(s)]
    unique_non_open = len(set(non_open_subtasks))

    if not has_open_app and unique_non_open >= 2:
        return "B"
    elif has_open_app and unique_non_open <= 1:
        return "A"
    elif has_open_app and unique_non_open >= 2:
        return "C"
    else:
        return "other"


def main():
    print("Loading GUI-360 subtask structure...")
    traj_subtasks = load_gui360_subtask_info()
    print(f"Loaded {len(traj_subtasks)} trajectories")

    pattern_b_ids = []
    for tid, segments in sorted(traj_subtasks.items()):
        if classify_trajectory(segments) == "B":
            pattern_b_ids.append(tid)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pattern_b_ids.json")
    with open(output_path, "w") as f:
        json.dump(pattern_b_ids, f, indent=2)

    print(f"Dumped {len(pattern_b_ids)} Pattern B trajectory IDs to {output_path}")


if __name__ == "__main__":
    main()
