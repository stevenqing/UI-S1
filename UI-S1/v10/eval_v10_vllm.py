#!/usr/bin/env python3
"""v10 Cooperative LoRA Evaluation via vLLM Multi-LoRA.

Single-step evaluation on AndroidControl eval set:
  1. Start vLLM server with base model + grounder/actor LoRA adapters
  2. For each step: grounder generates description → actor generates action
  3. Compute type/action accuracy

Uses vLLM's multi-LoRA support to serve both adapters without restart.

Usage:
  # Start vLLM server (in background):
  python -m vllm.entrypoints.openai.api_server \
      --model checkpoints/Qwen2.5-VL-7B-Instruct \
      --enable-lora \
      --lora-modules grounder=path/to/grounder actor=path/to/actor \
      --max-lora-rank 128 \
      --port 8000 --tensor-parallel-size 4 --trust-remote-code

  # Run eval:
  python v10/eval_v10_vllm.py \
      --eval_data datasets/android_control_evaluation_std.jsonl \
      --output_dir v10/output/eval_s300_vllm \
      --port 8000
"""

import argparse
import base64
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, List, Tuple

import requests

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Prompts (same as train_grpo.py)
# ---------------------------------------------------------------------------

GROUNDER_SYSTEM = (
    "You are a GUI grounding agent. Given a screenshot and an instruction, "
    "determine the next action type and describe the target.\n\n"
    "Output format:\n"
    "<action_type>one of: click, type, open, swipe, long_press, wait, system_button, terminate</action_type>\n"
    "<target>description of the target (UI element location for click/long_press, "
    "app name for open, scroll direction for swipe, button name for system_button, "
    "reason for wait, or text to type)</target>"
)

ACTOR_SYSTEM = (
    "You are a GUI agent. Given a screenshot, instruction, and grounding "
    "analysis (action type + target description), perform the next action.\n"
    'Output format: <action>{"action": "...", ...}</action>'
)

# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_grounder_text(goal: str, history: str) -> str:
    parts = [f"Instruction: {goal}"]
    if history:
        parts.append(f"\nPrevious actions:\n{history}")
    parts.append("\nDetermine the action type and describe the target.")
    return "\n".join(parts)


def parse_grounder_output(text: str) -> Tuple[str, str]:
    """Parse structured grounder output into (action_type, target_description).

    Falls back to ("unknown", full_text) if parsing fails.
    """
    action_type = "unknown"
    target = text  # fallback: use full text as target

    m = re.search(r'<action_type>\s*(.*?)\s*</action_type>', text, re.DOTALL)
    if m:
        action_type = m.group(1).strip().lower()

    m = re.search(r'<target>\s*(.*?)\s*</target>', text, re.DOTALL)
    if m:
        target = m.group(1).strip()

    return action_type, target


def format_actor_text(goal: str, history: str, action_type: str, target: str) -> str:
    parts = [f"Instruction: {goal}"]
    if history:
        parts.append(f"\nPrevious actions:\n{history}")
    parts.append(f"\nGrounding action type: {action_type}")
    parts.append(f"Grounding target: {target}")
    parts.append("\nOutput the next action.")
    return "\n".join(parts)


def build_vllm_messages(system: str, image_b64: str, user_text: str):
    """Build OpenAI-format messages with base64 image for vLLM."""
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
                {"type": "text", "text": user_text},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Action parsing & evaluation
# ---------------------------------------------------------------------------

_ACTION_TAG_RE = re.compile(r"<action>\s*(\{.*?\})\s*</action>", re.DOTALL)
_ACTION_RAW_RE = re.compile(r'\{[^{}]*"action"[^{}]*\}')


def parse_action_from_text(text: str) -> Optional[Dict[str, Any]]:
    m = _ACTION_TAG_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = _ACTION_RAW_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def coord_correct(pred_coord, gt_coord, img_w, img_h, threshold=0.05):
    if pred_coord is None or gt_coord is None:
        return False
    dx = (pred_coord[0] - gt_coord[0]) / img_w
    dy = (pred_coord[1] - gt_coord[1]) / img_h
    return (dx ** 2 + dy ** 2) ** 0.5 < threshold


def evaluate_action(pred_action, gt_action, check_options, img_w=1080, img_h=2400):
    """Returns (type_match, action_match)."""
    if pred_action is None:
        return False, False

    gt_type = gt_action.get("action", "")
    if gt_type == "left_click":
        gt_type = "click"
    pred_type = pred_action.get("action", "")
    if pred_type == "left_click":
        pred_type = "click"

    if pred_type != gt_type:
        return False, False

    if gt_type in ("click", "long_press"):
        gt_coord = gt_action.get("coordinate")
        pred_coord = pred_action.get("coordinate")
        ok = coord_correct(pred_coord, gt_coord, img_w, img_h)
        return True, ok

    elif gt_type in ("type", "open", "key", "answer"):
        gt_text = gt_action.get("text", "").strip().lower()
        pred_text = pred_action.get("text", "").strip().lower()
        return True, gt_text == pred_text

    elif gt_type == "swipe":
        gt_c1 = gt_action.get("coordinate") or gt_action.get("startCoordinate")
        gt_c2 = gt_action.get("coordinate2") or gt_action.get("endCoordinate")
        pred_c1 = pred_action.get("startCoordinate") or pred_action.get("coordinate")
        pred_c2 = pred_action.get("endCoordinate") or pred_action.get("coordinate2")
        if gt_c1 and gt_c2 and pred_c1 and pred_c2:
            gt_dx, gt_dy = gt_c2[0] - gt_c1[0], gt_c2[1] - gt_c1[1]
            pred_dx, pred_dy = pred_c2[0] - pred_c1[0], pred_c2[1] - pred_c1[1]
            gt_dir = "up" if gt_dy < -abs(gt_dx) else "down" if gt_dy > abs(gt_dx) else "left" if gt_dx < 0 else "right"
            pred_dir = "up" if pred_dy < -abs(pred_dx) else "down" if pred_dy > abs(pred_dx) else "left" if pred_dx < 0 else "right"
            return True, gt_dir == pred_dir
        return True, True

    elif gt_type in ("terminate", "wait"):
        return True, True

    elif gt_type == "system_button":
        gt_btn = gt_action.get("button", "").strip().lower()
        pred_btn = pred_action.get("button", "").strip().lower()
        return True, gt_btn == pred_btn

    return True, False


# ---------------------------------------------------------------------------
# vLLM API
# ---------------------------------------------------------------------------

def vllm_generate(base_url: str, model_name: str, messages: list,
                  max_tokens: int = 256, temperature: float = 0.0) -> str:
    """Call vLLM OpenAI-compatible API."""
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def load_image_b64(path: str) -> str:
    """Load image file and return base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_eval_steps(jsonl_path: str, image_root: str, max_episodes: int = 0):
    """Load AC eval data and flatten to individual steps with constructed history."""
    steps = []
    with open(jsonl_path) as f:
        for ep_idx, line in enumerate(f):
            if 0 < max_episodes <= ep_idx:
                break
            ep = json.loads(line.strip())
            goal = ep["goal"]
            history_parts = []

            for step_idx, step in enumerate(ep["steps"]):
                screenshot = step["screenshot"]
                if screenshot.startswith("/datasets/"):
                    screenshot = screenshot.replace("/datasets/", image_root + "/", 1)

                history_str = "\n".join(history_parts) if history_parts else ""

                steps.append({
                    "episode_id": ep.get("episode_id", ep_idx),
                    "step_idx": step_idx,
                    "goal": goal,
                    "history": history_str,
                    "screenshot": screenshot,
                    "gt_action": step["action_content"],
                    "check_options": step.get("check_options", step["action_content"]),
                    "step_instruction": step.get("step_instruction", ""),
                })

                # Build history for next step from GT
                act = step["action_content"]
                act_str = json.dumps(act, ensure_ascii=False)
                history_parts.append(f"Step {step_idx + 1}: {act_str}")

    return steps


# ---------------------------------------------------------------------------
# Process one step
# ---------------------------------------------------------------------------

def process_step(step: dict, base_url: str, grounder_model: str,
                 actor_model: str, max_grounder_tokens: int,
                 max_actor_tokens: int) -> dict:
    """Run grounder → actor for one step via vLLM API."""
    try:
        image_b64 = load_image_b64(step["screenshot"])
    except Exception as e:
        return {**step, "error": f"image load: {e}",
                "grounder_text": "", "actor_text": "",
                "pred_action": None, "type_match": False, "action_match": False}

    # Get image dimensions
    from PIL import Image
    img = Image.open(step["screenshot"])
    img_w, img_h = img.size

    try:
        # Grounder
        g_user = format_grounder_text(step["goal"], step["history"])
        g_msgs = build_vllm_messages(GROUNDER_SYSTEM, image_b64, g_user)
        g_text = vllm_generate(base_url, grounder_model, g_msgs,
                               max_tokens=max_grounder_tokens)

        # Parse structured grounder output
        action_type, target = parse_grounder_output(g_text)

        # Actor
        a_user = format_actor_text(step["goal"], step["history"], action_type, target)
        a_msgs = build_vllm_messages(ACTOR_SYSTEM, image_b64, a_user)
        a_text = vllm_generate(base_url, actor_model, a_msgs,
                               max_tokens=max_actor_tokens)

        pred_action = parse_action_from_text(a_text)
        type_match, action_match = evaluate_action(
            pred_action, step["gt_action"], step["check_options"], img_w, img_h)

        return {
            "episode_id": step["episode_id"],
            "step_idx": step["step_idx"],
            "goal": step["goal"][:200],
            "step_instruction": step["step_instruction"],
            "gt_action": step["gt_action"],
            "grounder_text": g_text,
            "actor_text": a_text,
            "pred_action": pred_action,
            "type_match": type_match,
            "action_match": action_match,
        }
    except Exception as e:
        return {
            "episode_id": step["episode_id"],
            "step_idx": step["step_idx"],
            "goal": step["goal"][:200],
            "step_instruction": step["step_instruction"],
            "gt_action": step["gt_action"],
            "grounder_text": "",
            "actor_text": "",
            "pred_action": None,
            "type_match": False,
            "action_match": False,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_root", type=str,
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--grounder_model", type=str, default="grounder",
                        help="vLLM model name for grounder adapter")
    parser.add_argument("--actor_model", type=str, default="actor",
                        help="vLLM model name for actor adapter")
    parser.add_argument("--max_grounder_tokens", type=int, default=256)
    parser.add_argument("--max_actor_tokens", type=int, default=256)
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=32,
                        help="Concurrent API requests")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base_url = f"http://localhost:{args.port}"

    # Check vLLM server
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        resp.raise_for_status()
        print("vLLM server is ready")
    except Exception as e:
        print(f"ERROR: vLLM server not reachable at {base_url}: {e}")
        sys.exit(1)

    # Load data
    steps = load_eval_steps(args.eval_data, args.image_root, args.max_episodes)
    print(f"Loaded {len(steps)} steps from {args.eval_data}")

    # Process steps concurrently
    results = [None] * len(steps)
    result_path = os.path.join(args.output_dir, "eval_results.jsonl")
    t0 = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        for i, step in enumerate(steps):
            fut = executor.submit(
                process_step, step, base_url,
                args.grounder_model, args.actor_model,
                args.max_grounder_tokens, args.max_actor_tokens,
            )
            futures[fut] = i

        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                results[i] = {
                    "episode_id": steps[i]["episode_id"],
                    "step_idx": steps[i]["step_idx"],
                    "error": str(e),
                    "type_match": False,
                    "action_match": False,
                }

            completed += 1
            if completed % 100 == 0 or completed == len(steps):
                done = [r for r in results if r is not None]
                n_type = sum(1 for r in done if r.get("type_match"))
                n_act = sum(1 for r in done if r.get("action_match"))
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"  [{completed}/{len(steps)}] "
                      f"type_acc={n_type/len(done)*100:.1f}% "
                      f"action_acc={n_act/len(done)*100:.1f}% "
                      f"({rate:.1f} steps/s)")

    # Save results
    with open(result_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compute step-level metrics
    n = len(results)
    n_type = sum(1 for r in results if r.get("type_match"))
    n_act = sum(1 for r in results if r.get("action_match"))
    n_err = sum(1 for r in results if r.get("error"))

    # Per action-type breakdown
    type_stats = {}
    for r in results:
        gt_type = r.get("gt_action", {}).get("action", "unknown")
        if gt_type not in type_stats:
            type_stats[gt_type] = {"total": 0, "type_match": 0, "action_match": 0}
        type_stats[gt_type]["total"] += 1
        if r.get("type_match"):
            type_stats[gt_type]["type_match"] += 1
        if r.get("action_match"):
            type_stats[gt_type]["action_match"] += 1

    # Compute episode-level metrics: TSR, avg_progress, step_accuracy
    from collections import defaultdict
    episodes = defaultdict(list)
    for r in results:
        episodes[r["episode_id"]].append(r)

    n_episodes = len(episodes)
    success_count = 0
    progresses = []
    total_steps_all = 0
    total_correct_all = 0

    # Length bucket stats
    def length_bucket(n_steps):
        if n_steps <= 3:
            return "short(1-3)"
        elif n_steps <= 7:
            return "medium(4-7)"
        elif n_steps <= 15:
            return "long(8-15)"
        else:
            return "vlong(16+)"

    bucket_stats = defaultdict(lambda: {"n_episodes": 0, "success": 0,
                                         "total_steps": 0, "correct_steps": 0,
                                         "progresses": []})

    for ep_id, ep_results in episodes.items():
        ep_results.sort(key=lambda x: x["step_idx"])
        num_steps = len(ep_results)

        # Count prefix correct steps (consecutive from beginning)
        prefix_correct = 0
        for r in ep_results:
            if r.get("action_match"):
                prefix_correct += 1
            else:
                break

        # Count total correct steps (scattered)
        scattered_correct = sum(1 for r in ep_results if r.get("action_match"))

        task_success = (prefix_correct == num_steps)
        if task_success:
            success_count += 1

        progress = prefix_correct / num_steps if num_steps > 0 else 0.0
        progresses.append(progress)
        total_steps_all += num_steps
        total_correct_all += scattered_correct

        bucket = length_bucket(num_steps)
        bucket_stats[bucket]["n_episodes"] += 1
        if task_success:
            bucket_stats[bucket]["success"] += 1
        bucket_stats[bucket]["total_steps"] += num_steps
        bucket_stats[bucket]["correct_steps"] += scattered_correct
        bucket_stats[bucket]["progresses"].append(progress)

    tsr = success_count / n_episodes if n_episodes > 0 else 0.0
    avg_progress = sum(progresses) / len(progresses) if progresses else 0.0
    step_accuracy = total_correct_all / total_steps_all if total_steps_all > 0 else 0.0

    # Compute bucket-level metrics
    bucket_metrics = {}
    for b, bs in bucket_stats.items():
        bucket_metrics[b] = {
            "n_episodes": bs["n_episodes"],
            "tsr": bs["success"] / bs["n_episodes"] if bs["n_episodes"] > 0 else 0.0,
            "step_accuracy": bs["correct_steps"] / bs["total_steps"] if bs["total_steps"] > 0 else 0.0,
            "avg_progress": sum(bs["progresses"]) / len(bs["progresses"]) if bs["progresses"] else 0.0,
        }

    summary = {
        "n_steps": n,
        "n_episodes": n_episodes,
        "n_errors": n_err,
        "tsr": tsr,
        "avg_progress": avg_progress,
        "step_accuracy": step_accuracy,
        "type_accuracy": n_type / n if n else 0,
        "action_accuracy": n_act / n if n else 0,
        "per_type": type_stats,
        "per_length_bucket": bucket_metrics,
        "elapsed_seconds": time.time() - t0,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"EPISODE-LEVEL METRICS ({n_episodes} episodes, {n} steps)")
    print(f"  TSR          = {tsr:.4f}  ({success_count}/{n_episodes})")
    print(f"  Avg Progress = {avg_progress:.4f}")
    print(f"  Step Acc     = {step_accuracy:.4f}  ({total_correct_all}/{total_steps_all})")
    print()
    print(f"STEP-LEVEL METRICS")
    print(f"  Type Acc     = {n_type}/{n} = {n_type/n*100:.1f}%")
    print(f"  Action Acc   = {n_act}/{n} = {n_act/n*100:.1f}%")
    print(f"  Errors       = {n_err}")
    print(f"  Time         = {time.time()-t0:.0f}s")
    print()
    print("PER ACTION-TYPE:")
    for t in sorted(type_stats.keys()):
        s = type_stats[t]
        print(f"  {t:15s}  total={s['total']:4d}  "
              f"type={s['type_match']:4d} ({s['type_match']/s['total']*100:5.1f}%)  "
              f"action={s['action_match']:4d} ({s['action_match']/s['total']*100:5.1f}%)")
    print()
    print("PER LENGTH BUCKET:")
    for b in ["short(1-3)", "medium(4-7)", "long(8-15)", "vlong(16+)"]:
        if b in bucket_metrics:
            bm = bucket_metrics[b]
            print(f"  {b:15s}  episodes={bm['n_episodes']:4d}  "
                  f"TSR={bm['tsr']:.4f}  "
                  f"StepAcc={bm['step_accuracy']:.4f}  "
                  f"AvgProg={bm['avg_progress']:.4f}")
    print("=" * 60)
    print(f"Results: {result_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
