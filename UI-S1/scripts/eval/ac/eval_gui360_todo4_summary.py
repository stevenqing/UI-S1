"""GUI-360 TODO 4: Summary-Augmented Agent Evaluation.

Tests whether injecting progress summaries improves GUI-360 per-step accuracy.

Conditions:
  A) baseline: Standard prompt
  B) oracle_summary: Inject GT thought/action history as structured summary
  C) vlm_summary: VLM-generated state summary from screenshot
"""

import argparse
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

from openai import OpenAI

result_lock = Lock()
_model = None


def load_gui360_samples(root_dir, max_samples=None):
    """Load GUI-360 samples (reuse from todo2 with subtask)."""
    from eval_gui360_todo2_subtask import load_gui360_samples_with_subtask
    return load_gui360_samples_with_subtask(root_dir, max_samples)


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


def evaluate_pred(pred_function, pred_args, sample):
    from evaluator.tool_definitions import normalize_tool_args
    gt_function = sample["gt_function"]
    gt_args = sample["gt_args"]
    gt_rect = sample["gt_rect"]
    function_match = pred_function == gt_function if pred_function else False
    args_match = False
    if function_match and pred_args and gt_args:
        try:
            pn = normalize_tool_args(pred_function, pred_args)
            gn = normalize_tool_args(gt_function, gt_args)
            if "coordinate" in pn and "coordinate" in gn:
                pc = pn["coordinate"]
                if isinstance(pc, (list, tuple)) and len(pc) == 2 and gt_rect:
                    cm = (gt_rect["left"] <= float(pc[0]) <= gt_rect["right"]
                          and gt_rect["top"] <= float(pc[1]) <= gt_rect["bottom"])
                    om = all(str(pn.get(k, "")).lower() == str(gn.get(k, "")).lower() for k in pn if k != "coordinate")
                    args_match = cm and om
            else:
                args_match = all(str(pn.get(k, "")).lower() == str(gn.get(k, "")).lower()
                                 for k in set(list(pn.keys()) + list(gn.keys())))
        except Exception:
            pass
    return function_match, args_match


def build_oracle_summary(sample):
    """Build oracle summary from GT previous actions (thought descriptions)."""
    if not sample["previous_actions"]:
        return ""
    lines = [f"Progress Summary (step {sample['step_index'] + 1} of {sample['total_steps']}):"]
    lines.append("Actions completed:")
    for pa in sample["previous_actions"]:
        lines.append(f"  {pa}")
    return "\n".join(lines)


def generate_vlm_summary(sample, api_url, model_name):
    """Use VLM to generate state summary from screenshot."""
    if sample["step_index"] == 0:
        return ""

    from PIL import Image
    import base64
    from io import BytesIO
    img = Image.open(sample["screenshot_clean"])
    buf = BytesIO()
    img.save(buf, format="PNG")
    data_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    prompt = (f"You are observing a desktop screen. The user's task is: {sample['request']}\n"
              f"Step {sample['step_index'] + 1} of {sample['total_steps']}. "
              f"Summarize in 2-3 sentences: what has been done and what remains.")

    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": prompt},
    ]}]

    for attempt in range(3):
        try:
            bot = OpenAI(api_key="EMPTY", base_url=api_url, timeout=120)
            resp = bot.chat.completions.create(model=model_name, messages=messages,
                                                extra_body={"top_k": 1}, max_tokens=256)
            return resp.choices[0].message.content.strip()
        except Exception:
            if attempt < 2:
                time.sleep(2)
    return ""


def predict_with_summary(sample, summary_text, api_url, model_name):
    """Predict action with summary injected into the instruction."""
    global _model
    instruction = sample["request"]
    if summary_text:
        instruction = f"{summary_text}\n\n{instruction}"

    system_prompt, user_prompt = _model.construct_action_prompt(
        instruction=instruction,
        history="\n".join(sample["previous_actions"]) if sample["previous_actions"] else "",
        actions=sample["actions_str"],
        resolution=None,
    )
    raw_response = _model.predict(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_path=sample["screenshot_clean"],
        temperature=0.0,
        max_tokens=4096,
    )
    pf, pa, ps = _model.parse_action(raw_response)
    return pf, pa, ps


def process_sample(sample, args):
    results = {}

    # A) baseline
    try:
        pf, pa, ps = predict_with_summary(sample, "", args.api_url, args.model_name)
        fm, am = evaluate_pred(pf, pa, sample)
        results["baseline"] = {"pred_function": pf, "function_match": fm, "args_match": am}
    except Exception as e:
        results["baseline"] = {"pred_function": None, "function_match": False, "args_match": False, "error": str(e)}

    # B) oracle_summary
    try:
        oracle = build_oracle_summary(sample)
        pf, pa, ps = predict_with_summary(sample, oracle, args.api_url, args.model_name)
        fm, am = evaluate_pred(pf, pa, sample)
        results["oracle_summary"] = {"pred_function": pf, "function_match": fm, "args_match": am, "summary": oracle[:200]}
    except Exception as e:
        results["oracle_summary"] = {"pred_function": None, "function_match": False, "args_match": False, "error": str(e)}

    # C) vlm_summary
    try:
        vlm_sum = generate_vlm_summary(sample, args.api_url, args.model_name)
        pf, pa, ps = predict_with_summary(sample, vlm_sum, args.api_url, args.model_name)
        fm, am = evaluate_pred(pf, pa, sample)
        results["vlm_summary"] = {"pred_function": pf, "function_match": fm, "args_match": am, "summary": vlm_sum[:200]}
    except Exception as e:
        results["vlm_summary"] = {"pred_function": None, "function_match": False, "args_match": False, "error": str(e)}

    output = {
        "sample_id": sample["sample_id"],
        "trajectory_id": sample["trajectory_id"],
        "domain": sample["domain"],
        "step_index": sample["step_index"],
        "total_steps": sample["total_steps"],
        "gt_function": sample["gt_function"],
        **{f"result_{k}": v for k, v in results.items()},
    }

    with result_lock:
        with open(os.path.join(args.output_dir, "summary_results.jsonl"), "a") as f:
            f.write(json.dumps(output, ensure_ascii=False, default=str) + "\n")
    return output


def analyze(results, output_dir):
    conditions = ["baseline", "oracle_summary", "vlm_summary"]
    print("\n" + "=" * 70)
    print("GUI-360 TODO 4: Summary-Augmented Results")
    print("=" * 70)

    for cond in conditions:
        key = f"result_{cond}"
        fm = sum(1 for r in results if r[key]["function_match"])
        am = sum(1 for r in results if r[key]["args_match"])
        n = len(results)
        print(f"  {cond:20s}: func={fm}/{n} ({fm/n*100:.1f}%)  args={am}/{n} ({am/n*100:.1f}%)")

    # By step position
    print("\n--- By Step Position ---")
    for cond in conditions:
        key = f"result_{cond}"
        by_pos = defaultdict(lambda: {"c": 0, "t": 0})
        for r in results:
            pos = "step_0" if r["step_index"] == 0 else ("step_1-3" if r["step_index"] <= 3 else "step_4+")
            by_pos[pos]["t"] += 1
            if r[key]["function_match"]:
                by_pos[pos]["c"] += 1
        print(f"\n  {cond}:")
        for pos in ["step_0", "step_1-3", "step_4+"]:
            if by_pos[pos]["t"] > 0:
                print(f"    {pos}: {by_pos[pos]['c']/by_pos[pos]['t']*100:.1f}% ({by_pos[pos]['c']}/{by_pos[pos]['t']})")

    summary = {"total": len(results), "conditions": {}}
    for cond in conditions:
        key = f"result_{cond}"
        fm = sum(1 for r in results if r[key]["function_match"]) / len(results)
        am = sum(1 for r in results if r[key]["args_match"]) / len(results)
        summary["conditions"][cond] = {"function_match": fm, "args_match": am}
    with open(os.path.join(output_dir, "summary_eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def main():
    global _model
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=os.path.join(PROJECT_DIR, "datasets/GUI-360/test"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_DIR, "outputs/eval_gui360_todo4"))
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--api_url", type=str, default="http://localhost:19815/v1")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_workers", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "summary_results.jsonl")
    if os.path.exists(out_path):
        os.remove(out_path)

    samples = load_gui360_samples(args.root_dir, args.max_samples)
    print(f"Loaded {len(samples)} samples")
    _model = init_model(args.api_url, args.model_name)

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_sample, s, args): s for s in samples}
        for future in as_completed(futures):
            try:
                r = future.result()
                results.append(r)
                if len(results) % 200 == 0:
                    b = sum(1 for x in results if x["result_baseline"]["function_match"]) / len(results)
                    o = sum(1 for x in results if x["result_oracle_summary"]["function_match"]) / len(results)
                    print(f"Progress: {len(results)}/{len(samples)} | base={b:.3f} oracle={o:.3f}")
            except Exception:
                traceback.print_exc()

    analyze(results, args.output_dir)


if __name__ == "__main__":
    main()
