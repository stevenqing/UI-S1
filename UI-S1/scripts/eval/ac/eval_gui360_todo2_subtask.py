"""GUI-360 TODO 2: Sub-Task Decomposition Agent Evaluation.

Tests whether giving the agent step-level sub-task instructions improves accuracy
compared to the full task goal on GUI-360 desktop tasks.

GUI-360 data has GT `subtask` field per step — we use this as oracle sub-task goal.

Conditions:
  A) full_goal: Original full task request
  B) oracle_subtask: GT per-step subtask description from dataset
"""

import argparse
import copy
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

PROJECT_DIR = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
GUI360_EVAL_DIR = os.path.join(PROJECT_DIR, "train_GUI_360", "GUI-360-eval")
sys.path.insert(0, GUI360_EVAL_DIR)

result_lock = Lock()
_model = None


def load_gui360_samples_with_subtask(root_dir, max_samples=None):
    """Load GUI-360 samples including subtask field."""
    from prompts.prompt_action_prediction import (
        SUPPORTED_ACTIONS_EXCEL, SUPPORTED_ACTIONS_PPT, SUPPORTED_ACTIONS_WORD,
    )

    data_path = os.path.join(root_dir, "data")
    samples = []

    for domain in sorted(os.listdir(data_path)):
        domain_path = os.path.join(data_path, domain)
        if not os.path.isdir(domain_path):
            continue
        if domain.lower() == "word":
            actions = SUPPORTED_ACTIONS_WORD
        elif domain.lower() == "excel":
            actions = SUPPORTED_ACTIONS_EXCEL
        elif domain.lower() == "ppt":
            actions = SUPPORTED_ACTIONS_PPT
        else:
            continue

        for category in sorted(os.listdir(domain_path)):
            category_path = os.path.join(domain_path, category, "success")
            if not os.path.exists(category_path):
                continue
            for jsonl_file in sorted(os.listdir(category_path)):
                if not jsonl_file.endswith(".jsonl"):
                    continue
                file_path = os.path.join(category_path, jsonl_file)
                all_steps = []
                with open(file_path, "r") as f:
                    for line_num, line in enumerate(f, 1):
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line.strip())
                            all_steps.append({"line_num": line_num, "data": data})
                        except json.JSONDecodeError:
                            continue

                for i, step_info in enumerate(all_steps):
                    data = step_info["data"]
                    clean_img = os.path.join(
                        root_dir, "image", domain, category,
                        data["step"]["screenshot_clean"],
                    )
                    if not os.path.exists(clean_img):
                        continue
                    if "action_prediction" not in data["step"].get("tags", []):
                        continue
                    action = data["step"]["action"]
                    if action.get("function", "") == "drag":
                        continue
                    if not action.get("rectangle", {}):
                        continue

                    previous_actions = []
                    for j in range(i):
                        prev_thought = all_steps[j]["data"]["step"]["thought"]
                        previous_actions.append(f"Step {j+1}: {prev_thought}")

                    status = data["step"]["status"]
                    if status == "OVERALL_FINISH":
                        status = "FINISH"
                    elif status == "FINISH":
                        status = "CONTINUE"

                    gt_args = dict(action.get("args", {}))
                    if "coordinate_x" in action and action["coordinate_x"]:
                        gt_args["coordinate"] = [action["coordinate_x"], action["coordinate_y"]]
                    gt_args.pop("x", None)
                    gt_args.pop("y", None)

                    samples.append({
                        "sample_id": f"{domain}_{category}_{os.path.splitext(jsonl_file)[0]}_{step_info['line_num']}",
                        "trajectory_id": f"{domain}_{category}_{os.path.splitext(jsonl_file)[0]}",
                        "request": data["request"],
                        "subtask": data["step"].get("subtask", ""),
                        "screenshot_clean": clean_img,
                        "domain": domain,
                        "category": category,
                        "actions_str": actions,
                        "previous_actions": previous_actions,
                        "gt_function": action.get("function", ""),
                        "gt_args": gt_args,
                        "gt_status": status,
                        "gt_rect": action.get("rectangle", {}),
                        "step_index": i,
                        "total_steps": len(all_steps),
                    })
                    if max_samples and len(samples) >= max_samples:
                        return samples
    return samples


def init_model(api_url, model_name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "qwen2_5_vl_7b",
        os.path.join(GUI360_EVAL_DIR, "models", "qwen2.5_vl_7b.py"),
    )
    qwen_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qwen_module)
    return qwen_module.Qwen25VL7B(
        api_url=api_url, model_name=model_name,
        coordinate_system="absolute", resize_factor=28,
    )


def predict_with_goal(sample, goal_text, temperature=0.0):
    """Run model prediction with a specific goal instruction."""
    global _model
    system_prompt, user_prompt = _model.construct_action_prompt(
        instruction=goal_text,
        history="\n".join(sample["previous_actions"]) if sample["previous_actions"] else "",
        actions=sample["actions_str"],
        resolution=None,
    )
    raw_response = _model.predict(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_path=sample["screenshot_clean"],
        temperature=temperature,
        max_tokens=4096,
    )
    pred_function, pred_args, pred_status = _model.parse_action(raw_response)
    return pred_function, pred_args, pred_status


def evaluate_pred(pred_function, pred_args, sample):
    """Evaluate prediction against GT."""
    from evaluator.tool_definitions import normalize_tool_args
    gt_function = sample["gt_function"]
    gt_args = sample["gt_args"]
    gt_rect = sample["gt_rect"]

    function_match = pred_function == gt_function if pred_function else False
    args_match = False
    if function_match and pred_args and gt_args:
        try:
            pred_norm = normalize_tool_args(pred_function, pred_args)
            gt_norm = normalize_tool_args(gt_function, gt_args)
            if "coordinate" in pred_norm and "coordinate" in gt_norm:
                pc = pred_norm["coordinate"]
                if isinstance(pc, (list, tuple)) and len(pc) == 2 and gt_rect:
                    coord_match = (gt_rect["left"] <= float(pc[0]) <= gt_rect["right"]
                                   and gt_rect["top"] <= float(pc[1]) <= gt_rect["bottom"])
                    other_match = all(
                        str(pred_norm.get(k, "")).lower() == str(gt_norm.get(k, "")).lower()
                        for k in pred_norm if k != "coordinate"
                    )
                    args_match = coord_match and other_match
            else:
                args_match = all(
                    str(pred_norm.get(k, "")).lower() == str(gt_norm.get(k, "")).lower()
                    for k in set(list(pred_norm.keys()) + list(gt_norm.keys()))
                )
        except Exception:
            args_match = False
    return function_match, args_match


def process_sample(sample, args):
    """Evaluate one sample under both conditions."""
    results = {}
    for cond, goal in [("full_goal", sample["request"]), ("oracle_subtask", sample["subtask"])]:
        try:
            pf, pa, ps = predict_with_goal(sample, goal)
            fm, am = evaluate_pred(pf, pa, sample)
            results[cond] = {
                "pred_function": pf, "function_match": fm, "args_match": am,
                "goal_used": goal[:200],
            }
        except Exception as e:
            results[cond] = {
                "pred_function": None, "function_match": False, "args_match": False,
                "error": str(e),
            }

    output = {
        "sample_id": sample["sample_id"],
        "trajectory_id": sample["trajectory_id"],
        "domain": sample["domain"],
        "step_index": sample["step_index"],
        "total_steps": sample["total_steps"],
        "gt_function": sample["gt_function"],
        "request": sample["request"][:200],
        "subtask": sample["subtask"][:200],
        **{f"result_{k}": v for k, v in results.items()},
    }

    with result_lock:
        with open(os.path.join(args.output_dir, "subtask_results.jsonl"), "a") as f:
            f.write(json.dumps(output, ensure_ascii=False, default=str) + "\n")
    return output


def analyze(results, output_dir):
    conditions = ["full_goal", "oracle_subtask"]
    print("\n" + "=" * 70)
    print("GUI-360 TODO 2: Sub-Task Decomposition Results")
    print("=" * 70)

    for cond in conditions:
        key = f"result_{cond}"
        fm = sum(1 for r in results if r[key]["function_match"])
        am = sum(1 for r in results if r[key]["args_match"])
        n = len(results)
        print(f"  {cond:20s}: func_match={fm}/{n} ({fm/n*100:.1f}%)  args_match={am}/{n} ({am/n*100:.1f}%)")

    # By domain
    print("\n--- By Domain ---")
    by_domain = defaultdict(list)
    for r in results:
        by_domain[r["domain"]].append(r)
    for domain in sorted(by_domain):
        dr = by_domain[domain]
        n = len(dr)
        print(f"\n  {domain} (N={n}):")
        for cond in conditions:
            key = f"result_{cond}"
            fm = sum(1 for r in dr if r[key]["function_match"])
            am = sum(1 for r in dr if r[key]["args_match"])
            print(f"    {cond:20s}: func={fm/n*100:.1f}%  args={am/n*100:.1f}%")

    # By step position
    print("\n--- By Step Position ---")
    for cond in conditions:
        key = f"result_{cond}"
        by_pos = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in results:
            pos = "step_0" if r["step_index"] == 0 else ("step_1-3" if r["step_index"] <= 3 else "step_4+")
            by_pos[pos]["total"] += 1
            if r[key]["function_match"]:
                by_pos[pos]["correct"] += 1
        print(f"\n  {cond}:")
        for pos in ["step_0", "step_1-3", "step_4+"]:
            if by_pos[pos]["total"] > 0:
                acc = by_pos[pos]["correct"] / by_pos[pos]["total"] * 100
                print(f"    {pos}: {acc:.1f}% ({by_pos[pos]['correct']}/{by_pos[pos]['total']})")

    # Paired
    print("\n--- Paired: full_goal vs oracle_subtask ---")
    full_better = sum(1 for r in results if r["result_full_goal"]["function_match"] and not r["result_oracle_subtask"]["function_match"])
    sub_better = sum(1 for r in results if r["result_oracle_subtask"]["function_match"] and not r["result_full_goal"]["function_match"])
    both = sum(1 for r in results if r["result_full_goal"]["function_match"] and r["result_oracle_subtask"]["function_match"])
    neither = len(results) - full_better - sub_better - both
    print(f"  full_goal only:    {full_better}")
    print(f"  subtask only:      {sub_better}")
    print(f"  both correct:      {both}")
    print(f"  neither correct:   {neither}")

    summary = {"total": len(results), "conditions": {}}
    for cond in conditions:
        key = f"result_{cond}"
        fm = sum(1 for r in results if r[key]["function_match"]) / len(results)
        am = sum(1 for r in results if r[key]["args_match"]) / len(results)
        summary["conditions"][cond] = {"function_match": fm, "args_match": am}
    with open(os.path.join(output_dir, "subtask_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def main():
    global _model
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=os.path.join(PROJECT_DIR, "datasets/GUI-360/test"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_DIR, "outputs/eval_gui360_todo2"))
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--api_url", type=str, default="http://localhost:19815/v1")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_workers", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "subtask_results.jsonl")
    if os.path.exists(out_path):
        os.remove(out_path)

    samples = load_gui360_samples_with_subtask(args.root_dir, args.max_samples)
    print(f"Loaded {len(samples)} samples. Has subtask: {sum(1 for s in samples if s['subtask'])} samples")

    _model = init_model(args.api_url, args.model_name)

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_sample, s, args): s for s in samples}
        for future in as_completed(futures):
            try:
                r = future.result()
                results.append(r)
                if len(results) % 200 == 0:
                    fg = sum(1 for x in results if x["result_full_goal"]["function_match"]) / len(results)
                    sg = sum(1 for x in results if x["result_oracle_subtask"]["function_match"]) / len(results)
                    print(f"Progress: {len(results)}/{len(samples)} | full={fg:.3f} subtask={sg:.3f}")
            except Exception as e:
                traceback.print_exc()

    analyze(results, args.output_dir)


if __name__ == "__main__":
    main()
