"""
Exp 0b: F4 Replication — Controlled comparison with original F4 on GUI-360 step-level.

Tests what made F4 work by varying one factor at a time:

Conditions:
  C0:          Standard single-pass baseline (same as Exp 0).
  HardDecomp:  Agent V sees ONLY screenshot (hard partition, like original F4).
               Agent H sees ONLY task+history (no screenshot).
               Agent A gets everything + both descriptions injected.
  SoftDecomp:  Same as Exp 0 Decomp: observer sees screenshot+task.
               (Tests whether hard partition matters.)
  HardV_Only:  Agent V only (hard partition), no Agent H.
               (Tests whether Agent H matters.)
  CoT:         Same as Exp 0 CoT (for reference).

Key differences vs original F4:
  - Original F4 used SUBTASK_ISOLATED prompt (with subtask segmentation)
  - Original F4 used Pattern B subset (203 traj, 1916 steps)
  - Original F4 injected descriptions at specific positions in prompt

This script:
  - Uses the same GUI-360 test parquet (full dataset)
  - Uses standard eval format (no subtask segmentation)
  - Can optionally restrict to Pattern B IDs via --pattern_b_ids
  - Tests hard vs soft partition on the SAME eval framework
"""

import argparse
import base64
import copy
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from threading import Lock

import pandas as pd
from PIL import Image
from openai import OpenAI

# ── Prompts ──────────────────────────────────────────────────────────────

# Original F4 Agent V prompt: screenshot only, no task context
AGENT_V_HARD_PROMPT = """\
Describe the current state of this screen. List all visible interactive elements \
with their names and locations. Note any open dialogs, selected items, or active menus. \
Be specific and exhaustive."""

# Original F4 Agent H prompt: task + history, no screenshot
AGENT_H_PROMPT = """\
Given this task: {task}

And these completed actions:
{history}

Analyze the current progress:
1. What has been accomplished?
2. What is the logical next category of action needed?
3. Are there signs the task is going wrong?

Output a concise progress analysis in 3-5 sentences."""

# Soft observer: sees screenshot + task (Exp 0 style)
OBSERVER_SOFT_PROMPT = """\
You are a Visual Analyst for a mobile GUI agent. Analyze the CURRENT screenshot and describe:
1. Screen Layout: type of screen (home, settings, search results, dialog, etc.)
2. Key Interactive Elements: buttons, text fields, icons, toggles with approximate positions
3. Current State: loading, scrolled position, keyboard visible, popup shown
4. Task-Relevant Elements: which elements relate to the given task

Provide concise structured text. Do NOT output any action tags or JSON."""

# CoT Turn 1
COT_TURN1_PROMPT = """\
Describe all interactive UI elements visible on the current screenshot.
For each element, provide:
- Element type (button, text field, icon, toggle, tab, etc.)
- Label or text content
- Approximate position on screen (top/middle/bottom, left/center/right)

Be concise and structured. Do NOT output any action or tool_call."""

# ── Utilities ──────────────────────────────────────────────────────────

result_lock = Lock()


def image_to_data_url(image_path):
    img = Image.open(image_path)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def parse_tool_call(text):
    if not text:
        return None
    m = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    try:
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
            if attempt < max_retries - 1:
                time.sleep(2)
    return ""


def strip_tags(text):
    if not text:
        return text
    text = re.sub(r'</?(?:think|action|tool_call)>', '', text)
    text = re.sub(r'^```\w*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```$', '', text, flags=re.MULTILINE)
    return text.strip()


def _extract_images(content):
    if isinstance(content, str):
        return []
    return [item for item in (content if isinstance(content, list) else [])
            if isinstance(item, dict) and (
                item.get("type") == "image_url" or "image_url" in item)]


def _extract_text(content):
    if isinstance(content, str):
        return content
    return "\n".join(
        item["text"] if isinstance(item, dict) and item.get("type") == "text"
        else (item if isinstance(item, str) else "")
        for item in (content if isinstance(content, list) else []))


def evaluate_action(pred, gt, threshold=50):
    result = {"function_match": False, "args_match": False, "full_match": False,
              "bbox_match": None}
    if pred is None or gt is None:
        return result

    pred_func = pred.get("function", "")
    gt_func = gt.get("function", "")
    result["function_match"] = (pred_func == gt_func)
    if not result["function_match"]:
        return result

    pred_args = pred.get("args", {})
    gt_args = gt.get("args", {})
    gt_bbox = gt.get("bbox", None)

    if pred_func == "click":
        pred_coord = pred_args.get("coordinate", [])
        gt_coord = gt_args.get("coordinate", [])
        if len(pred_coord) == 2:
            if gt_bbox:
                left, top = gt_bbox.get("left"), gt_bbox.get("top")
                right, bottom = gt_bbox.get("right"), gt_bbox.get("bottom")
                if all(v is not None for v in [left, top, right, bottom]):
                    result["bbox_match"] = (
                        left <= pred_coord[0] <= right and
                        top <= pred_coord[1] <= bottom)
                    result["args_match"] = result["bbox_match"]
            if not result["args_match"] and len(gt_coord) == 2:
                dist = ((pred_coord[0]-gt_coord[0])**2 +
                        (pred_coord[1]-gt_coord[1])**2)**0.5
                result["args_match"] = dist < threshold
    elif pred_func == "type":
        pred_text = pred_args.get("text", "").lower().strip()
        gt_text = gt_args.get("text", "").lower().strip()
        result["args_match"] = (
            pred_text == gt_text or pred_text in gt_text or gt_text in pred_text)
    elif pred_func in ("scroll", "wheel_mouse_input"):
        result["args_match"] = (
            pred_args.get("direction", "") == gt_args.get("direction", ""))
    elif pred_func == "summary":
        result["args_match"] = True
    else:
        result["args_match"] = (pred_args == gt_args)

    result["full_match"] = result["function_match"] and result["args_match"]
    return result


# ── Condition Implementations ──────────────────────────────────────────


def condition_c0(openai_msgs, model_name, api_url):
    response = call_model(openai_msgs, model_name, api_url)
    return response, {"condition": "C0"}


def condition_hard_decomp(openai_msgs, model_name, api_url):
    """HardDecomp: Agent V (screenshot only) + Agent H (task+history only) + Agent A.

    Replicates F4's hard input partition.
    """
    last_user = None
    for msg in openai_msgs:
        if msg["role"] == "user":
            last_user = copy.deepcopy(msg)
    if not last_user:
        return "", {"condition": "HardDecomp", "error": "no_user_msg"}

    images = _extract_images(last_user.get("content", []))
    text = _extract_text(last_user.get("content", []))

    # Agent V: ONLY screenshot, no task context (hard partition)
    v_content = [{"type": "text", "text": AGENT_V_HARD_PROMPT}]
    v_content.extend(copy.deepcopy(images))
    v_msgs = [{"role": "user", "content": v_content}]
    v_desc = strip_tags(call_model(v_msgs, model_name, api_url, max_tokens=512))

    # Agent H: ONLY task + history text, NO screenshot (hard partition)
    # Extract task and history from the text
    h_text = AGENT_H_PROMPT.format(task=text[:500], history=text[500:1500])
    h_msgs = [{"role": "user", "content": h_text}]
    h_desc = strip_tags(call_model(h_msgs, model_name, api_url, max_tokens=256))

    # Agent A: original messages + both descriptions injected
    action_msgs = copy.deepcopy(openai_msgs)
    for msg in reversed(action_msgs):
        if msg["role"] == "user":
            content = msg.get("content", [])
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            # Find first image
            img_pos = None
            for i, item in enumerate(content):
                if isinstance(item, dict) and (
                    item.get("type") == "image_url" or "image_url" in item):
                    img_pos = i
                    break
            # Inject Agent H after task, Agent V before screenshot (like F4)
            inject = {"type": "text", "text": (
                f"\nTask progress: {h_desc or '(not available)'}\n"
                f"\nCurrent screen elements: {v_desc or '(not available)'}\n"
            )}
            if img_pos is not None:
                content.insert(img_pos, inject)
            else:
                content.append(inject)
            msg["content"] = content
            break

    response = call_model(action_msgs, model_name, api_url)
    return response, {
        "condition": "HardDecomp",
        "agent_v_desc": (v_desc or "")[:300],
        "agent_h_desc": (h_desc or "")[:300],
    }


def condition_soft_decomp(openai_msgs, model_name, api_url):
    """SoftDecomp: Observer sees screenshot+task (like Exp 0 Decomp)."""
    last_user = None
    for msg in openai_msgs:
        if msg["role"] == "user":
            last_user = copy.deepcopy(msg)
    if not last_user:
        return "", {"condition": "SoftDecomp", "error": "no_user_msg"}

    images = _extract_images(last_user.get("content", []))
    text = _extract_text(last_user.get("content", []))

    # Observer: screenshot + task context (soft, like Exp 0)
    obs_content = [
        {"type": "text",
         "text": f"Task context:\n{text[:1000]}\n\nAnalyze the current screenshot:"}
    ]
    obs_content.extend(copy.deepcopy(images))
    obs_msgs = [
        {"role": "system", "content": OBSERVER_SOFT_PROMPT},
        {"role": "user", "content": obs_content},
    ]
    obs_desc = strip_tags(call_model(obs_msgs, model_name, api_url, max_tokens=512))

    # Action model with description injected
    action_msgs = copy.deepcopy(openai_msgs)
    for msg in reversed(action_msgs):
        if msg["role"] == "user":
            content = msg.get("content", [])
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            img_pos = None
            for i, item in enumerate(content):
                if isinstance(item, dict) and (
                    item.get("type") == "image_url" or "image_url" in item):
                    img_pos = i
                    break
            inject = {"type": "text", "text": (
                f"\nCurrent screen elements: {obs_desc or '(not available)'}\n"
            )}
            if img_pos is not None:
                content.insert(img_pos, inject)
            else:
                content.append(inject)
            msg["content"] = content
            break

    response = call_model(action_msgs, model_name, api_url)
    return response, {
        "condition": "SoftDecomp",
        "observer_desc": (obs_desc or "")[:300],
    }


def condition_hard_v_only(openai_msgs, model_name, api_url):
    """HardV_Only: Agent V (screenshot only, hard partition), NO Agent H.

    Tests whether Agent H matters, or if Agent V alone is enough.
    """
    last_user = None
    for msg in openai_msgs:
        if msg["role"] == "user":
            last_user = copy.deepcopy(msg)
    if not last_user:
        return "", {"condition": "HardV_Only", "error": "no_user_msg"}

    images = _extract_images(last_user.get("content", []))

    # Agent V: ONLY screenshot (hard partition)
    v_content = [{"type": "text", "text": AGENT_V_HARD_PROMPT}]
    v_content.extend(copy.deepcopy(images))
    v_msgs = [{"role": "user", "content": v_content}]
    v_desc = strip_tags(call_model(v_msgs, model_name, api_url, max_tokens=512))

    # Action model with V description only
    action_msgs = copy.deepcopy(openai_msgs)
    for msg in reversed(action_msgs):
        if msg["role"] == "user":
            content = msg.get("content", [])
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            img_pos = None
            for i, item in enumerate(content):
                if isinstance(item, dict) and (
                    item.get("type") == "image_url" or "image_url" in item):
                    img_pos = i
                    break
            inject = {"type": "text", "text": (
                f"\nCurrent screen elements: {v_desc or '(not available)'}\n"
            )}
            if img_pos is not None:
                content.insert(img_pos, inject)
            else:
                content.append(inject)
            msg["content"] = content
            break

    response = call_model(action_msgs, model_name, api_url)
    return response, {
        "condition": "HardV_Only",
        "agent_v_desc": (v_desc or "")[:300],
    }


def condition_cot(openai_msgs, model_name, api_url):
    """CoT: Two-turn, model sees screenshot in both turns."""
    last_user = None
    for msg in openai_msgs:
        if msg["role"] == "user":
            last_user = copy.deepcopy(msg)
    if not last_user:
        return "", {"condition": "CoT", "error": "no_user_msg"}

    images = _extract_images(last_user.get("content", []))

    # Turn 1: describe UI
    t1_content = [{"type": "text", "text": COT_TURN1_PROMPT}]
    t1_content.extend(copy.deepcopy(images))
    t1_msgs = [{"role": "user", "content": t1_content}]
    desc = strip_tags(call_model(t1_msgs, model_name, api_url, max_tokens=512))

    # Turn 2: original + description
    t2_msgs = copy.deepcopy(openai_msgs)
    for msg in reversed(t2_msgs):
        if msg["role"] == "user":
            content = msg.get("content", [])
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            img_pos = None
            for i, item in enumerate(content):
                if isinstance(item, dict) and (
                    item.get("type") == "image_url" or "image_url" in item):
                    img_pos = i
                    break
            inject = {"type": "text", "text": (
                f"\n--- Your preliminary analysis ---\n{desc}\n--- End ---\n"
                f"Based on this and the screenshot, choose the action.\n"
            )}
            if img_pos is not None:
                content.insert(img_pos, inject)
            else:
                content.append(inject)
            msg["content"] = content
            break

    response = call_model(t2_msgs, model_name, api_url)
    return response, {"condition": "CoT", "cot_desc": (desc or "")[:300]}


# ── Processing ──────────────────────────────────────────────────────────


def convert_messages_to_openai(messages, base_dir):
    openai_msgs = []
    for msg in messages:
        role = msg['role']
        content = msg['content']
        if isinstance(content, str):
            openai_msgs.append({"role": role,
                                "content": [{"type": "text", "text": content}]})
        elif isinstance(content, list):
            new_content = []
            for item in content:
                if isinstance(item, str):
                    new_content.append({"type": "text", "text": item})
                elif isinstance(item, dict):
                    if "text" in item:
                        new_content.append({"type": "text", "text": item["text"]})
                    elif "image" in item:
                        ip = item["image"]
                        if not os.path.isabs(ip):
                            ip = os.path.join(base_dir, ip)
                        new_content.append({
                            "type": "image_url",
                            "image_url": {"url": image_to_data_url(ip)}})
            openai_msgs.append({"role": role, "content": new_content})
    return openai_msgs


CONDITION_FNS = {
    "C0": condition_c0,
    "HardDecomp": condition_hard_decomp,
    "SoftDecomp": condition_soft_decomp,
    "HardV_Only": condition_hard_v_only,
    "CoT": condition_cot,
}


def process_sample(idx, row, args, base_dir):
    try:
        messages = row['messages']
        if isinstance(messages, str):
            messages = json.loads(messages)

        user_msg = messages[0]
        gt_response = messages[1]['content']
        gt_action = parse_tool_call(gt_response)

        openai_msgs = convert_messages_to_openai([user_msg], base_dir)

        fn = CONDITION_FNS[args.condition]
        response, debug_info = fn(openai_msgs, args.model_name, args.api_url)

        pred_action = parse_tool_call(response)
        eval_result = evaluate_action(pred_action, gt_action)

        result = {
            "idx": idx,
            "condition": args.condition,
            "gt_function": gt_action.get("function", "") if gt_action else "",
            "pred_function": pred_action.get("function", "") if pred_action else "",
            "function_match": eval_result["function_match"],
            "args_match": eval_result["args_match"],
            "full_match": eval_result["full_match"],
            "bbox_match": eval_result.get("bbox_match"),
        }
        if args.save_debug:
            result.update({k: v for k, v in debug_info.items()
                          if isinstance(v, str)})

        model_short = os.path.basename(args.model_name.rstrip('/'))
        with result_lock:
            rp = os.path.join(args.output_dir,
                              f"{model_short}_exp0b_{args.condition}.jsonl")
            with open(rp, 'a') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"idx": idx, "condition": args.condition,
                "function_match": False, "args_match": False,
                "full_match": False, "error": str(e)}


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    model_short = os.path.basename(args.model_name.rstrip('/'))
    result_path = os.path.join(args.output_dir,
                               f"{model_short}_exp0b_{args.condition}.jsonl")

    print(f"Loading dataset: {args.parquet_file}")
    df = pd.read_parquet(args.parquet_file)

    if args.max_samples > 0:
        df = df.head(args.max_samples)

    # Resume
    if args.resume:
        completed = set()
        if os.path.exists(result_path):
            with open(result_path) as f:
                for line in f:
                    try:
                        completed.add(json.loads(line)['idx'])
                    except Exception:
                        pass
            before = len(df)
            df = df[~df.index.isin(completed)]
            print(f"Resume: skip {before - len(df)}, {len(df)} remaining.")
    else:
        if os.path.exists(result_path):
            os.remove(result_path)

    print(f"Evaluating {len(df)} samples. Condition: {args.condition}")

    base_dir = os.path.dirname(os.path.dirname(args.parquet_file))
    if "train_GUI_360" in args.parquet_file:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(
            args.parquet_file)))

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_sample, idx, row, args, base_dir): idx
                   for idx, row in df.iterrows()}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            if len(results) % 200 == 0:
                n = len(results)
                full = sum(1 for x in results if x.get("full_match"))
                func = sum(1 for x in results if x.get("function_match"))
                print(f"  [{args.condition}] {n}: func={func/n*100:.1f}% full={full/n*100:.1f}%")

    # Reload if resuming
    if args.resume and os.path.exists(result_path):
        results = []
        with open(result_path) as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except Exception:
                    pass

    # Print results
    total = len(results)
    if total == 0:
        print("No results.")
        return

    func_m = sum(1 for r in results if r.get("function_match"))
    args_m = sum(1 for r in results if r.get("args_match"))
    full_m = sum(1 for r in results if r.get("full_match"))
    bbox_m = sum(1 for r in results if r.get("bbox_match") is True)
    bbox_n = sum(1 for r in results if r.get("bbox_match") is not None)

    print(f"\n{'='*60}")
    print(f"Exp 0b — {args.condition} | {total} samples")
    print(f"{'='*60}")
    print(f"Function: {func_m}/{total} ({func_m/total*100:.2f}%)")
    print(f"Full:     {full_m}/{total} ({full_m/total*100:.2f}%)")
    if bbox_n:
        print(f"BBox:     {bbox_m}/{bbox_n} ({bbox_m/bbox_n*100:.2f}%)")

    # Per-function breakdown
    for func, count in Counter(r.get("gt_function", "") for r in results).most_common():
        fr = [r for r in results if r.get("gt_function") == func]
        fm = sum(1 for r in fr if r.get("full_match"))
        print(f"  {func}: {fm}/{count} ({fm/count*100:.1f}%)")

    summary = {
        "condition": args.condition,
        "model_name": args.model_name,
        "total": total,
        "function_accuracy": func_m / total,
        "full_accuracy": full_m / total,
        "bbox_accuracy": bbox_m / bbox_n if bbox_n else None,
    }
    sp = os.path.join(args.output_dir,
                      f"{model_short}_exp0b_{args.condition}_summary.json")
    with open(sp, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {sp}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--condition", required=True,
                   choices=list(CONDITION_FNS.keys()))
    p.add_argument("--api_url", default="http://localhost:8000/v1")
    p.add_argument("--output_dir",
                   default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation/results/exp0b")
    p.add_argument("--parquet_file",
                   default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/data/gui360_test_sft_eval_format_with_bbox.parquet")
    p.add_argument("--max_workers", type=int, default=64)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--save_debug", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    main(args)
