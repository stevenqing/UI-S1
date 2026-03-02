"""
Convert GUI-360 raw trajectory data into TrajDataset format for RL training.
Processes one app at a time to avoid OOM on login nodes.

Usage:
  python scripts/GUI_360/prepare_gui360_rl_data.py --max-val 50
"""
import json
import os
import sys
import random
from collections import defaultdict

GUI360_BASE = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/GUI-360"
OUTPUT_DIR = os.path.join(GUI360_BASE, "rl_data")
APPS = ["excel", "word", "ppt"]
CATEGORIES = ["in_app", "search", "online"]


def convert_action(raw_action):
    function = raw_action.get("function", "")
    args = raw_action.get("args", {})
    ac = {"action": function, "coordinate": None, "text": None,
          "status": None, "button": None, "coordinate2": None, "time": None}
    if "x" in args and "y" in args:
        ac["coordinate"] = [args["x"], args["y"]]
    elif "coordinate" in args and args["coordinate"]:
        ac["coordinate"] = args["coordinate"]
    if "text" in args:
        ac["text"] = args["text"]
    elif "input_text" in args:
        ac["text"] = args["input_text"]
    if function == "system_button":
        ac["button"] = args.get("button")
    if "end_coordinate" in args:
        ac["coordinate2"] = args["end_coordinate"]
    if "start_coordinate" in args:
        ac["coordinate"] = args["start_coordinate"]
    if "duration" in args:
        ac["time"] = args["duration"]
    if raw_action.get("status") == "FINISH":
        ac["status"] = "success"
    return ac


def process_single_file(jsonl_path, category):
    """Read one trajectory jsonl file, return list of step dicts with category attached."""
    results = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                d["_category"] = category
                results.append(d)
            except json.JSONDecodeError:
                pass
    return results


def process_app_split(split, app):
    """Process one app in one split, return episodes."""
    data_dir = os.path.join(GUI360_BASE, split, "data", app)
    # Images live under {split}/image/, NOT {split}/data/
    image_base = os.path.join(GUI360_BASE, split, "image")

    # Collect all jsonl files for this app, tagging each step with its category
    all_steps = []
    for cat in CATEGORIES:
        cat_dir = os.path.join(data_dir, cat)
        if not os.path.isdir(cat_dir):
            continue
        for dirpath, _, filenames in os.walk(cat_dir):
            for fn in filenames:
                if fn.endswith(".jsonl"):
                    jsonl_path = os.path.join(dirpath, fn)
                    all_steps.extend(process_single_file(jsonl_path, cat))

    # Group by execution_id
    exec_groups = defaultdict(list)
    for s in all_steps:
        exec_groups[s.get("execution_id", "")].append(s)
    del all_steps  # free memory

    episodes = []
    for exec_id, steps_data in exec_groups.items():
        steps_data.sort(key=lambda x: x.get("step_id", 0))
        goal = steps_data[0].get("request", "")
        # Use the category from the first step (all steps in an episode share the same category)
        category = steps_data[0].get("_category", "in_app")

        steps = []
        for sd in steps_data:
            step = sd.get("step", {})
            ss = step.get("screenshot_clean", "")
            if not ss:
                continue
            # Build screenshot path: {split}/image/{app}/{category}/{screenshot_clean}
            full_ss = os.path.join(image_base, app, category, ss)
            ac = convert_action(step.get("action", {}))
            s = {"action_content": ac, "screenshot": os.path.abspath(full_ss)}
            thought = step.get("thought", "")
            if thought:
                s["thought"] = thought
            steps.append(s)

        if steps:
            episodes.append({
                "goal": goal,
                "is_successful": True,
                "steps": steps,
                "execution_id": exec_id,
            })
    return episodes


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-val", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split in ["train", "test"]:
        out_path = os.path.join(OUTPUT_DIR, f"gui360_{split}.jsonl")
        total = 0
        missing_img = 0
        with open(out_path, "w") as fout:
            for app in APPS:
                print(f"  Processing {split}/{app}...", end=" ", flush=True)
                eps = process_app_split(split, app)
                # Validate a sample of image paths
                for ep in eps:
                    for st in ep["steps"][:1]:  # check first step of each episode
                        if not os.path.isfile(st["screenshot"]):
                            missing_img += 1
                            if missing_img <= 5:
                                print(f"\n  WARNING: missing image: {st['screenshot']}")
                for ep in eps:
                    fout.write(json.dumps(ep, ensure_ascii=False) + "\n")
                print(f"{len(eps)} episodes")
                total += len(eps)
                del eps
        if missing_img > 0:
            print(f"  WARNING: {missing_img} episodes have missing screenshot for first step!")
        print(f"  {split} total: {total} episodes -> {out_path}")

    # Create small validation subset from test
    test_path = os.path.join(OUTPUT_DIR, "gui360_test.jsonl")
    with open(test_path) as f:
        test_lines = f.readlines()
    random.seed(42)
    val_lines = random.sample(test_lines, min(args.max_val, len(test_lines)))
    val_path = os.path.join(OUTPUT_DIR, "gui360_val_small.jsonl")
    with open(val_path, "w") as f:
        f.writelines(val_lines)
    print(f"  Val small: {len(val_lines)} episodes -> {val_path}")
    print("Done!")


if __name__ == "__main__":
    main()
