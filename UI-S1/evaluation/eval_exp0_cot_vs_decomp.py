"""
Exp 0: CoT vs Decomposition — 4 inference conditions on GUI-360 and AndroidControl.

Conditions:
  C0 (baseline):  Standard single-pass inference.
  CoT:            Two-turn protocol on single model.
                  Turn 1: "Describe all interactive UI elements on screen."
                  Turn 2: Original prompt + description → action.
  Decomp:         Variant-G style: Observer (screenshot) → description,
                  Action model (description + screenshot) → action.
                  (Equivalent to F4 multi-agent decomposition.)
  RandDesc:       Action model receives observer description from a RANDOM
                  screenshot (not the current one). Tests whether description
                  content matters or just the format/CoT effect.

Usage:
  python eval_exp0_cot_vs_decomp.py \
      --model_name <path> \
      --condition C0 \
      --dataset gui360            # or "ac"
"""

import argparse
import copy
import json
import os
import random
import re
import base64
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from threading import Lock

import pandas as pd
from PIL import Image
from openai import OpenAI

# ── Prompts ──────────────────────────────────────────────────────────────

COT_TURN1_PROMPT = """\
Describe all interactive UI elements visible on the current screenshot.
For each element, provide:
- Element type (button, text field, icon, toggle, tab, etc.)
- Label or text content
- Approximate position on screen (top/middle/bottom, left/center/right)

Be concise and structured. Do NOT output any action or tool_call."""

OBSERVER_PROMPT = """\
You are a Visual Analyst for a mobile GUI agent. Analyze the CURRENT screenshot and describe:
1. Screen Layout: type of screen (home, settings, search results, dialog, etc.)
2. Key Interactive Elements: buttons, text fields, icons, toggles with approximate positions
3. Current State: loading, scrolled position, keyboard visible, popup shown
4. Task-Relevant Elements: which elements relate to the given task

Provide concise structured text. Do NOT output any action tags or JSON."""

DECOMP_ACTION_PROMPT_PREFIX = """\
A visual analyst has provided the following description of the current screen:

--- Visual Analysis ---
{observer_notes}
--- End Visual Analysis ---

Use this analysis to help identify the correct UI element. The screenshot is also provided below.
"""

# ── Utilities ──────────────────────────────────────────────────────────

result_lock = Lock()


def image_to_data_url(image_path: str) -> str:
    image = Image.open(image_path)
    buf = BytesIO()
    image.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64_str}"


def parse_tool_call(text: str) -> dict:
    if not text:
        return None
    try:
        match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return json.loads(text)
    except Exception:
        return None


def call_model(messages, model_name, api_url, max_tokens=None, max_retries=3):
    client = OpenAI(api_key="EMPTY", base_url=api_url, timeout=600)
    kwargs = {"extra_body": {"top_k": 1}}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_name, messages=messages, **kwargs)
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"API call failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    return ""


def strip_action_tags(text):
    if not text:
        return text
    text = re.sub(r'</?(?:think|action|tool_call)>', '', text)
    text = re.sub(r'^```\w*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```$', '', text, flags=re.MULTILINE)
    return text.strip()


def _extract_images(content):
    if isinstance(content, str):
        return []
    out = []
    for item in (content if isinstance(content, list) else []):
        if isinstance(item, dict) and (
            item.get("type") == "image_url" or "image_url" in item
        ):
            out.append(item)
    return out


def _extract_text(content):
    if isinstance(content, str):
        return content
    parts = []
    for item in (content if isinstance(content, list) else []):
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item["text"])
        elif isinstance(item, str):
            parts.append(item)
    return "\n".join(parts)


# ── GUI-360 Evaluation ──────────────────────────────────────────────────


def convert_messages_to_openai(messages, base_dir):
    openai_msgs = []
    for msg in messages:
        role = msg['role']
        content = msg['content']
        if isinstance(content, str):
            openai_msgs.append({
                "role": role,
                "content": [{"type": "text", "text": content}]
            })
        elif isinstance(content, list):
            new_content = []
            for item in content:
                if isinstance(item, str):
                    new_content.append({"type": "text", "text": item})
                elif isinstance(item, dict):
                    if "text" in item:
                        new_content.append({"type": "text", "text": item["text"]})
                    elif "image" in item:
                        image_path = item["image"]
                        if not os.path.isabs(image_path):
                            image_path = os.path.join(base_dir, image_path)
                        data_url = image_to_data_url(image_path)
                        new_content.append({
                            "type": "image_url",
                            "image_url": {"url": data_url}
                        })
            openai_msgs.append({"role": role, "content": new_content})
    return openai_msgs


def evaluate_action(pred_action, gt_action, threshold=0.05, use_bbox=True):
    result = {
        "function_match": False,
        "args_match": False,
        "full_match": False,
        "bbox_match": None,
    }
    if pred_action is None or gt_action is None:
        return result

    pred_func = pred_action.get("function", "")
    gt_func = gt_action.get("function", "")
    result["function_match"] = (pred_func == gt_func)
    if not result["function_match"]:
        return result

    pred_args = pred_action.get("args", {})
    gt_args = gt_action.get("args", {})
    gt_bbox = gt_action.get("bbox", None)

    if pred_func == "click":
        pred_coord = pred_args.get("coordinate", [])
        gt_coord = gt_args.get("coordinate", [])
        if len(pred_coord) == 2:
            if use_bbox and gt_bbox:
                left, top = gt_bbox.get("left"), gt_bbox.get("top")
                right, bottom = gt_bbox.get("right"), gt_bbox.get("bottom")
                if all(v is not None for v in [left, top, right, bottom]):
                    result["bbox_match"] = (
                        left <= pred_coord[0] <= right and
                        top <= pred_coord[1] <= bottom
                    )
                    result["args_match"] = result["bbox_match"]
            if not result["args_match"] and len(gt_coord) == 2:
                dist = ((pred_coord[0] - gt_coord[0])**2 +
                        (pred_coord[1] - gt_coord[1])**2)**0.5
                result["args_match"] = dist < 50

    elif pred_func == "type":
        pred_text = pred_args.get("text", "").lower().strip()
        gt_text = gt_args.get("text", "").lower().strip()
        if use_bbox:
            result["args_match"] = (pred_text == gt_text)
        else:
            result["args_match"] = (
                pred_text == gt_text or pred_text in gt_text or gt_text in pred_text
            )

    elif pred_func == "drag":
        pred_start = pred_args.get("startCoordinate", [])
        pred_end = pred_args.get("endCoordinate", [])
        gt_start = gt_args.get("startCoordinate", [])
        gt_end = gt_args.get("endCoordinate", [])
        if len(pred_start) == 2 and len(gt_start) == 2:
            start_dist = ((pred_start[0] - gt_start[0])**2 +
                          (pred_start[1] - gt_start[1])**2)**0.5
            end_dist = float('inf')
            if len(pred_end) == 2 and len(gt_end) == 2:
                end_dist = ((pred_end[0] - gt_end[0])**2 +
                            (pred_end[1] - gt_end[1])**2)**0.5
            result["args_match"] = start_dist < 50 and end_dist < 50

    elif pred_func in ["wheel_mouse_input", "scroll"]:
        result["args_match"] = (
            pred_args.get("direction", "") == gt_args.get("direction", "")
        )

    elif pred_func == "summary":
        result["args_match"] = True

    else:
        result["args_match"] = (pred_args == gt_args)

    result["full_match"] = result["function_match"] and result["args_match"]
    return result


# ── Condition Implementations ──────────────────────────────────────────


def condition_c0(openai_msgs, model_name, api_url):
    """C0: Standard single-pass baseline."""
    response = call_model(openai_msgs, model_name, api_url)
    return response, {"condition": "C0"}


def condition_cot(openai_msgs, model_name, api_url):
    """CoT: Two-turn protocol on single model.

    Turn 1: Ask model to describe UI elements on the screenshot.
    Turn 2: Feed description back as context, then ask for action.
    Model sees the screenshot in BOTH turns.
    """
    # Build Turn 1: system + user with screenshot + CoT prompt
    last_user = None
    system_msg = None
    for msg in openai_msgs:
        if msg["role"] == "system":
            system_msg = msg
        if msg["role"] == "user":
            last_user = copy.deepcopy(msg)

    if last_user is None:
        return "", {"condition": "CoT", "error": "no_user_msg"}

    images = _extract_images(last_user.get("content", []))

    # Turn 1: describe UI elements
    turn1_content = [{"type": "text", "text": COT_TURN1_PROMPT}]
    turn1_content.extend(copy.deepcopy(images))
    turn1_msgs = [{"role": "user", "content": turn1_content}]

    description = call_model(turn1_msgs, model_name, api_url, max_tokens=512)
    description = strip_action_tags(description)

    # Turn 2: original messages + description injected before screenshot
    turn2_msgs = copy.deepcopy(openai_msgs)
    # Find last user message and inject description before images
    for msg in reversed(turn2_msgs):
        if msg["role"] == "user":
            content = msg.get("content", [])
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            # Find first image position
            img_pos = None
            for i, item in enumerate(content):
                if isinstance(item, dict) and (
                    item.get("type") == "image_url" or "image_url" in item
                ):
                    img_pos = i
                    break
            inject = {
                "type": "text",
                "text": (
                    f"\n--- Your preliminary analysis of the screenshot ---\n"
                    f"{description}\n"
                    f"--- End analysis ---\n"
                    f"Now based on this analysis and the screenshot, "
                    f"choose the correct action.\n"
                )
            }
            if img_pos is not None:
                content.insert(img_pos, inject)
            else:
                content.append(inject)
            msg["content"] = content
            break

    response = call_model(turn2_msgs, model_name, api_url)

    debug_info = {
        "condition": "CoT",
        "cot_description": description[:500],
    }
    return response, debug_info


def condition_decomp(openai_msgs, model_name, api_url):
    """Decomposition (F4-equivalent): separate observer + action model.

    Observer: sees screenshot + task context → produces description.
    Action model: sees description + screenshot + original prompt → action.
    Both use the same underlying model, but separate inference passes.
    """
    last_user = None
    system_msg = None
    for msg in openai_msgs:
        if msg["role"] == "system":
            system_msg = copy.deepcopy(msg)
        if msg["role"] == "user":
            last_user = copy.deepcopy(msg)

    if last_user is None:
        return "", {"condition": "Decomp", "error": "no_user_msg"}

    images = _extract_images(last_user.get("content", []))
    text = _extract_text(last_user.get("content", []))

    # Step 1: Observer call (separate from action model)
    obs_content = [
        {"type": "text", "text": f"Task context:\n{text[:1000]}\n\nAnalyze the current screenshot:"}
    ]
    obs_content.extend(copy.deepcopy(images))
    obs_msgs = [
        {"role": "system", "content": OBSERVER_PROMPT},
        {"role": "user", "content": obs_content},
    ]

    observer_desc = call_model(obs_msgs, model_name, api_url, max_tokens=512)
    observer_desc = strip_action_tags(observer_desc)

    # Step 2: Action model call with observer description injected
    action_msgs = copy.deepcopy(openai_msgs)
    for msg in reversed(action_msgs):
        if msg["role"] == "user":
            content = msg.get("content", [])
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            img_pos = None
            for i, item in enumerate(content):
                if isinstance(item, dict) and (
                    item.get("type") == "image_url" or "image_url" in item
                ):
                    img_pos = i
                    break
            inject = {
                "type": "text",
                "text": DECOMP_ACTION_PROMPT_PREFIX.format(
                    observer_notes=observer_desc or "(no observation)"
                )
            }
            if img_pos is not None:
                content.insert(img_pos, inject)
            else:
                content.append(inject)
            msg["content"] = content
            break

    response = call_model(action_msgs, model_name, api_url)

    debug_info = {
        "condition": "Decomp",
        "observer_description": observer_desc[:500],
    }
    return response, debug_info


class RandDescPool:
    """Pre-computes observer descriptions for random screenshots.

    Used by condition_randdesc to provide descriptions that don't match
    the actual screenshot being evaluated.
    """
    def __init__(self):
        self._descriptions = []
        self._lock = Lock()

    def add(self, desc):
        with self._lock:
            self._descriptions.append(desc)

    def get_random(self, exclude_idx=None):
        with self._lock:
            if len(self._descriptions) < 2:
                return "(no random description available yet)"
            candidates = list(range(len(self._descriptions)))
            if exclude_idx is not None and exclude_idx in candidates:
                candidates.remove(exclude_idx)
            if not candidates:
                return self._descriptions[0]
            idx = random.choice(candidates)
            return self._descriptions[idx]

    def __len__(self):
        with self._lock:
            return len(self._descriptions)


def condition_randdesc(openai_msgs, model_name, api_url, random_desc):
    """RandDesc: Action model receives observer description from a RANDOM screenshot.

    Tests whether the improvement from decomposition comes from the specific
    description content or just from having any intermediate text (format effect).
    """
    # Inject the random description into the action prompt
    action_msgs = copy.deepcopy(openai_msgs)
    for msg in reversed(action_msgs):
        if msg["role"] == "user":
            content = msg.get("content", [])
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            img_pos = None
            for i, item in enumerate(content):
                if isinstance(item, dict) and (
                    item.get("type") == "image_url" or "image_url" in item
                ):
                    img_pos = i
                    break
            inject = {
                "type": "text",
                "text": DECOMP_ACTION_PROMPT_PREFIX.format(
                    observer_notes=random_desc
                )
            }
            if img_pos is not None:
                content.insert(img_pos, inject)
            else:
                content.append(inject)
            msg["content"] = content
            break

    response = call_model(action_msgs, model_name, api_url)

    debug_info = {
        "condition": "RandDesc",
        "random_description": random_desc[:300],
    }
    return response, debug_info


# ── Main Processing ────────────────────────────────────────────────────


def process_gui360_sample(idx, row, args, base_dir, rand_pool=None):
    """Process a single GUI-360 step-level sample."""
    try:
        messages = row['messages']
        if isinstance(messages, str):
            messages = json.loads(messages)

        user_msg = messages[0]
        gt_response = messages[1]['content']
        gt_action = parse_tool_call(gt_response)

        openai_msgs = convert_messages_to_openai([user_msg], base_dir)

        if args.condition == "C0":
            response, debug_info = condition_c0(
                openai_msgs, args.model_name, args.api_url)
        elif args.condition == "CoT":
            response, debug_info = condition_cot(
                openai_msgs, args.model_name, args.api_url)
        elif args.condition == "Decomp":
            response, debug_info = condition_decomp(
                openai_msgs, args.model_name, args.api_url)
        elif args.condition == "RandDesc":
            # First generate this sample's observer description (for the pool)
            last_user = None
            for msg in openai_msgs:
                if msg["role"] == "user":
                    last_user = copy.deepcopy(msg)
            if last_user:
                images = _extract_images(last_user.get("content", []))
                text = _extract_text(last_user.get("content", []))
                obs_content = [
                    {"type": "text",
                     "text": f"Task context:\n{text[:1000]}\n\nAnalyze:"}
                ]
                obs_content.extend(copy.deepcopy(images))
                obs_msgs = [
                    {"role": "system", "content": OBSERVER_PROMPT},
                    {"role": "user", "content": obs_content},
                ]
                own_desc = call_model(
                    obs_msgs, args.model_name, args.api_url, max_tokens=512)
                own_desc = strip_action_tags(own_desc)
                if rand_pool is not None:
                    rand_pool.add(own_desc)

            # Get a random description (not from this sample)
            random_desc = rand_pool.get_random(exclude_idx=idx) \
                if rand_pool else "(no description)"
            response, debug_info = condition_randdesc(
                openai_msgs, args.model_name, args.api_url, random_desc)
        else:
            raise ValueError(f"Unknown condition: {args.condition}")

        pred_action = parse_tool_call(response)
        eval_result = evaluate_action(
            pred_action, gt_action, use_bbox=getattr(args, 'use_bbox', True))

        result = {
            "idx": idx,
            "condition": args.condition,
            "gt_function": gt_action.get("function", "") if gt_action else "",
            "pred_function": pred_action.get("function", "") if pred_action else "",
            "function_match": eval_result["function_match"],
            "args_match": eval_result["args_match"],
            "full_match": eval_result["full_match"],
            "bbox_match": eval_result.get("bbox_match"),
            "gt_response": gt_response[:200],
            "pred_response": (response or "")[:200],
        }
        if args.save_debug:
            result.update({k: v for k, v in debug_info.items()
                          if isinstance(v, str)})

        # Thread-safe write
        model_short = os.path.basename(args.model_name.rstrip('/'))
        with result_lock:
            result_path = os.path.join(
                args.output_dir,
                f"{model_short}_exp0_{args.condition}_gui360.jsonl")
            with open(result_path, 'a') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        return result

    except Exception as e:
        print(f"Error processing sample {idx}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "idx": idx, "condition": args.condition,
            "function_match": False, "args_match": False,
            "full_match": False, "bbox_match": None,
            "error": str(e),
        }


def run_gui360(args):
    """Run Exp 0 on GUI-360 step-level dataset."""
    os.makedirs(args.output_dir, exist_ok=True)

    model_short = os.path.basename(args.model_name.rstrip('/'))
    result_path = os.path.join(
        args.output_dir,
        f"{model_short}_exp0_{args.condition}_gui360.jsonl")

    print(f"Loading dataset: {args.parquet_file}")
    df = pd.read_parquet(args.parquet_file)
    if args.max_samples > 0:
        df = df.head(args.max_samples)

    # Resume support
    if args.resume:
        completed = set()
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        completed.add(r['idx'])
                    except Exception:
                        pass
            before = len(df)
            df = df[~df.index.isin(completed)]
            print(f"Resume: skipping {before - len(df)}, {len(df)} remaining.")
    else:
        if os.path.exists(result_path):
            os.remove(result_path)

    print(f"Evaluating {len(df)} samples. Condition: {args.condition}")

    base_dir = os.path.dirname(os.path.dirname(args.parquet_file))
    if "train_GUI_360" in args.parquet_file:
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(args.parquet_file)))
    print(f"Image base dir: {base_dir}")

    # For RandDesc: pre-populate description pool from a random subset
    rand_pool = None
    if args.condition == "RandDesc":
        rand_pool = RandDescPool()
        print("RandDesc mode: generating initial description pool...")
        # Pre-generate descriptions from first N samples
        pre_n = min(50, len(df))
        pre_df = df.head(pre_n)
        for idx, row in pre_df.iterrows():
            messages = row['messages']
            if isinstance(messages, str):
                messages = json.loads(messages)
            user_msg = messages[0]
            openai_msgs = convert_messages_to_openai([user_msg], base_dir)
            last_user = None
            for msg in openai_msgs:
                if msg["role"] == "user":
                    last_user = copy.deepcopy(msg)
            if last_user:
                images = _extract_images(last_user.get("content", []))
                text = _extract_text(last_user.get("content", []))
                obs_content = [
                    {"type": "text",
                     "text": f"Task context:\n{text[:1000]}\n\nAnalyze:"}
                ]
                obs_content.extend(copy.deepcopy(images))
                obs_msgs = [
                    {"role": "system", "content": OBSERVER_PROMPT},
                    {"role": "user", "content": obs_content},
                ]
                desc = call_model(
                    obs_msgs, args.model_name, args.api_url, max_tokens=512)
                rand_pool.add(strip_action_tags(desc))
        print(f"Pre-populated {len(rand_pool)} descriptions for RandDesc.")

    # Run evaluation
    results = []
    if args.condition == "RandDesc":
        # Sequential for RandDesc to build pool incrementally
        for idx, row in df.iterrows():
            result = process_gui360_sample(
                idx, row, args, base_dir, rand_pool)
            results.append(result)
            if len(results) % 50 == 0:
                _print_interim_gui360(results, args.condition)
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    process_gui360_sample, idx, row, args, base_dir, rand_pool
                ): idx
                for idx, row in df.iterrows()
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if len(results) % 100 == 0:
                    _print_interim_gui360(results, args.condition)

    # Reload all results if resuming
    if args.resume and os.path.exists(result_path):
        results = []
        with open(result_path, 'r') as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except Exception:
                    pass

    _print_final_gui360(results, args)


def _print_interim_gui360(results, condition):
    n = len(results)
    full = sum(1 for r in results if r.get("full_match", False))
    func = sum(1 for r in results if r.get("function_match", False))
    print(f"  [{condition}] {n} done: func={func/n*100:.1f}% full={full/n*100:.1f}%")


def _print_final_gui360(results, args):
    total = len(results)
    if total == 0:
        print("No results.")
        return

    func_matches = sum(1 for r in results if r.get("function_match", False))
    args_matches = sum(1 for r in results if r.get("args_match", False))
    full_matches = sum(1 for r in results if r.get("full_match", False))
    bbox_matches = sum(1 for r in results if r.get("bbox_match") is True)
    bbox_applicable = sum(1 for r in results if r.get("bbox_match") is not None)

    print(f"\n{'='*60}")
    print(f"Exp 0 Results — Condition: {args.condition} | Dataset: GUI-360")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print(f"Function accuracy:  {func_matches}/{total} ({func_matches/total*100:.2f}%)")
    if func_matches > 0:
        print(f"Args accuracy:      {args_matches}/{func_matches} ({args_matches/func_matches*100:.2f}%)")
    print(f"Full accuracy:      {full_matches}/{total} ({full_matches/total*100:.2f}%)")
    if bbox_applicable > 0:
        print(f"BBox accuracy:      {bbox_matches}/{bbox_applicable} ({bbox_matches/bbox_applicable*100:.2f}%)")

    # Per-function breakdown
    print("\nPer-function breakdown:")
    gt_funcs = [r.get("gt_function", "") for r in results]
    for func, count in Counter(gt_funcs).most_common():
        func_results = [r for r in results if r.get("gt_function") == func]
        func_full = sum(1 for r in func_results if r.get("full_match", False))
        print(f"  {func}: {func_full}/{count} ({func_full/count*100:.2f}%)")

    # Save summary
    summary = {
        "condition": args.condition,
        "dataset": "gui360",
        "model_name": args.model_name,
        "total_samples": total,
        "function_accuracy": func_matches / total,
        "args_accuracy": args_matches / func_matches if func_matches > 0 else 0,
        "full_accuracy": full_matches / total,
        "bbox_accuracy": bbox_matches / bbox_applicable if bbox_applicable > 0 else None,
    }
    model_short = os.path.basename(args.model_name.rstrip('/'))
    summary_path = os.path.join(
        args.output_dir,
        f"{model_short}_exp0_{args.condition}_gui360_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


# ── AndroidControl Trajectory Evaluation ─────────────────────────────


def run_ac(args):
    """Run Exp 0 on AndroidControl trajectory dataset."""
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..')))
    from x.data.agent.json import JsonFormat
    from x.data.agent.space.std_space import RAW_SPACE
    from x.qwen.data_format import slim_messages
    from qwenvl_utils import (
        call_mobile_agent_vllm, evaluate_android_control_action,
        find_last_image_ele, message_translate, image_to_data_url as img_to_url
    )

    os.makedirs(args.output_dir, exist_ok=True)

    if args.no_thought:
        fm = JsonFormat(RAW_SPACE, add_thought=False)
    else:
        fm = JsonFormat(RAW_SPACE, add_thought=True, force_add_thought=True)

    model_short = os.path.basename(args.model_name.rstrip('/'))
    result_path = os.path.join(
        args.output_dir,
        f"{model_short}_exp0_{args.condition}_ac.jsonl")

    # Load data
    std_data = []
    with open(args.jsonl_file, 'r') as f:
        for line in f:
            std_data.append(json.loads(line))

    if args.max_samples > 0:
        std_data = std_data[:args.max_samples]

    # Resume
    if args.resume:
        completed = set()
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        completed.add(r['goal'])
                    except Exception:
                        pass
            before = len(std_data)
            std_data = [d for d in std_data if d['goal'] not in completed]
            print(f"Resume: skipping {before - len(std_data)}, "
                  f"{len(std_data)} remaining.")
    else:
        if os.path.exists(result_path):
            os.remove(result_path)

    print(f"Evaluating {len(std_data)} trajectories. Condition: {args.condition}")

    def _messages_to_openai(messages):
        translated, screenshot_list = message_translate(
            messages, to_format='openai')
        ptr = 0
        for msg in translated:
            for content in msg['content']:
                if 'image_url' in content:
                    url = img_to_url(Image.open(screenshot_list[ptr]))
                    content['image_url']['url'] = url
                    ptr += 1
        return translated

    def _robust_parse(model_response):
        try:
            return fm.parse_response(model_response)
        except Exception:
            pass
        match = re.search(r'\{[^{}]*"action"\s*:', model_response)
        if match:
            start = match.start()
            brace_count = 0
            for i in range(start, len(model_response)):
                if model_response[i] == '{':
                    brace_count += 1
                elif model_response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = model_response[start:i+1]
                        action_content = json.loads(json_str)
                        return {'think': None, 'action': json_str,
                                'action_content': action_content}
        raise ValueError(f"Cannot parse: {model_response[:200]}")

    def process_trajectory(line):
        import copy as _copy
        num_steps = len(line['steps'])
        state = None
        model_response = None
        step_id = 0
        task_success = False
        fixed_line = _fix_line(line)
        step_details = []

        try:
            while step_id < num_steps:
                check_pam = fixed_line['steps'][step_id]['check_options']
                state = fm.gen_next_round(
                    fixed_line, state, previous_model_response=model_response)
                if state is None:
                    break

                messages = state['messages']
                messages = slim_messages(
                    messages=messages,
                    num_image_limit=args.n_history_image_limit)
                _, width, height, rw, rh = find_last_image_ele(messages)

                openai_msgs = _messages_to_openai(messages)

                if args.condition == "C0":
                    model_response = call_mobile_agent_vllm(
                        messages=messages, model_name=args.model_name)
                    debug_info = {"condition": "C0"}
                elif args.condition == "CoT":
                    model_response, debug_info = condition_cot(
                        openai_msgs, args.model_name, args.api_url)
                elif args.condition == "Decomp":
                    model_response, debug_info = condition_decomp(
                        openai_msgs, args.model_name, args.api_url)
                else:
                    raise ValueError(f"Unknown condition: {args.condition}")

                pred_action = _robust_parse(model_response)
                type_match, extract_match = evaluate_android_control_action(
                    pred_action['action_content'],
                    check_pam, width, height, rw, rh)

                step_details.append({
                    "step_id": step_id,
                    "type_match": bool(type_match),
                    "extract_match": bool(extract_match),
                    "action_type": check_pam.get('action', ''),
                })

                if not extract_match:
                    break
                step_id += 1

            task_success = (step_id == num_steps)

        except Exception as e:
            print(f"Error: {line['goal']}: {e}")
            import traceback
            traceback.print_exc()

        result = {
            "goal": line['goal'],
            "num_steps": num_steps,
            "task_success": bool(task_success),
            "final_step_id": int(step_id),
            "condition": args.condition,
        }
        if args.save_debug:
            result["step_details"] = step_details

        with result_lock:
            with open(result_path, 'a') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        return result

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_trajectory, line): line
                   for line in std_data}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                if len(results) % 50 == 0:
                    n = len(results)
                    s = sum(1 for r in results if r['task_success'])
                    print(f"  [{args.condition}] {n} done: TSR={s/n*100:.1f}%")
            except Exception as e:
                print(f"Exception: {e}")

    # Reload if resuming
    if args.resume and os.path.exists(result_path):
        results = []
        with open(result_path, 'r') as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except Exception:
                    pass

    _print_final_ac(results, args)


def _fix_line(line):
    import copy
    line = copy.deepcopy(line)
    for step in line['steps']:
        if 'check_options' not in step:
            check_options = copy.deepcopy(step['action_content'])
            if 'bbox' in step:
                check_options['candidate_bbox'] = step['bbox']
            else:
                check_options['candidate_bbox'] = []
            step['check_options'] = check_options
    return line


def _print_final_ac(results, args):
    if not results:
        print("No results.")
        return

    n = len(results)
    success = sum(1 for r in results if r['task_success'])
    avg_progress = sum(
        r['final_step_id'] / r['num_steps'] for r in results) / n

    print(f"\n{'='*60}")
    print(f"Exp 0 Results — Condition: {args.condition} | Dataset: AndroidControl")
    print(f"{'='*60}")
    print(f"Total trajectories: {n}")
    print(f"TSR:          {success}/{n} ({success/n*100:.2f}%)")
    print(f"Avg Progress: {avg_progress:.4f}")

    summary = {
        "condition": args.condition,
        "dataset": "ac",
        "model_name": args.model_name,
        "total": n,
        "tsr": success / n * 100,
        "avg_progress": avg_progress,
    }
    model_short = os.path.basename(args.model_name.rstrip('/'))
    summary_path = os.path.join(
        args.output_dir,
        f"{model_short}_exp0_{args.condition}_ac_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


# ── Entry Point ──────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 0: CoT vs Decomposition evaluation")

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--condition", type=str, required=True,
                        choices=["C0", "CoT", "Decomp", "RandDesc"])
    parser.add_argument("--dataset", type=str, default="gui360",
                        choices=["gui360", "ac"])
    parser.add_argument("--api_url", type=str,
                        default="http://localhost:8000/v1")
    parser.add_argument("--output_dir", type=str,
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation/results/exp0")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--save_debug", action="store_true")
    parser.add_argument("--resume", action="store_true")

    # GUI-360 specific
    parser.add_argument("--parquet_file", type=str,
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/data/gui360_test_sft_eval_format_with_bbox.parquet")
    parser.add_argument("--use_bbox", action="store_true", default=True)
    parser.add_argument("--no_bbox", action="store_true")

    # AC specific
    parser.add_argument("--jsonl_file", type=str,
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation/dataset/android_control_evaluation_fixed.jsonl")
    parser.add_argument("--n_history_image_limit", type=int, default=2)
    parser.add_argument("--no_thought", action="store_true")

    args = parser.parse_args()
    if args.no_bbox:
        args.use_bbox = False

    if args.dataset == "gui360":
        run_gui360(args)
    else:
        run_ac(args)
