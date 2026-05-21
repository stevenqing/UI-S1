#!/usr/bin/env python3
"""v10 Cooperative LoRA Evaluation — Single-step accuracy on AC val set.

Loads the PEFT dual-adapter checkpoint (grounder + actor), runs greedy
two-pass generation on the evaluation dataset, and computes:
  - Type accuracy (action type match)
  - Action accuracy (type + content match, using standard thresholds)
  - Per-type breakdown

Usage:
  python v10/eval_v10.py \
      --model_path checkpoints/Qwen2.5-VL-7B-Instruct \
      --checkpoint v10/output/v10_grpo_ddp/epoch-1_step-300 \
      --eval_data datasets/cooperative_thought_ac/ac_val_thought.jsonl \
      --output_dir v10/output/eval_s300
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, Any, Optional

import torch
import numpy as np
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Prompts (same as train_grpo.py)
# ---------------------------------------------------------------------------

GROUNDER_SYSTEM = (
    "You are a GUI grounding agent. Given a screenshot and an instruction, "
    "describe the target UI element that should be interacted with for the "
    "next action. Describe its appearance, text content, and approximate "
    "location on screen."
)

ACTOR_SYSTEM = (
    "You are a GUI agent. Given a screenshot, instruction, and a grounding "
    "description, perform the next action.\n"
    'Output format: <action>{"action": "...", ...}</action>'
)

# ---------------------------------------------------------------------------
# Prompt formatting (same as train_grpo.py)
# ---------------------------------------------------------------------------

def format_grounder_text(goal: str, history: str) -> str:
    parts = [f"Instruction: {goal}"]
    if history:
        parts.append(f"\nPrevious actions:\n{history}")
    parts.append("\nDescribe the target UI element for the next action.")
    return "\n".join(parts)


def format_actor_text(goal: str, history: str, grounding: str) -> str:
    parts = [f"Instruction: {goal}"]
    if history:
        parts.append(f"\nPrevious actions:\n{history}")
    parts.append(f"\nGrounding: {grounding}")
    parts.append("\nOutput the next action.")
    return "\n".join(parts)


def build_messages(system: str, image_path: str, user_text: str):
    user_text_clean = user_text.replace("<image>\n", "").replace("<image>", "")
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text_clean},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Dataset (same as train_grpo.py)
# ---------------------------------------------------------------------------

_GOAL_RE = re.compile(r"## User Instruction\n(.+?)(?:\n\n## |\n## |\Z)", re.DOTALL)
_HIST_RE = re.compile(r"## History of previous actions\n(.+?)(?:\n\n## |\n## |\Z)", re.DOTALL)
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


def load_eval_data(jsonl_path: str):
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            convs = sample.get("conversations", [])
            if len(convs) < 2:
                continue
            human_msg = convs[0]["value"]
            assistant_msg = convs[1]["value"]
            m = _GOAL_RE.search(human_msg)
            goal = m.group(1).strip() if m else ""
            if not goal:
                continue
            m = _HIST_RE.search(human_msg)
            history = m.group(1).strip() if m else ""
            gt_action = parse_action_from_text(assistant_msg)
            if gt_action is None:
                continue
            images = sample.get("images", [])
            image_path = images[0] if images else None
            if image_path is None or not os.path.exists(image_path):
                continue
            samples.append({
                "goal": goal,
                "history": history,
                "gt_action": gt_action,
                "image_path": image_path,
            })
    return samples


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def coord_correct(pred_coord, gt_coord, img_w, img_h, threshold=0.05):
    if pred_coord is None or gt_coord is None:
        return False
    dx = (pred_coord[0] - gt_coord[0]) / img_w
    dy = (pred_coord[1] - gt_coord[1]) / img_h
    return (dx ** 2 + dy ** 2) ** 0.5 < threshold


def evaluate_action(pred_action, gt_action, img_w=1080, img_h=2400):
    """Returns (type_match: bool, action_match: bool)."""
    if pred_action is None:
        return False, False

    gt_type = gt_action.get("action", "")
    pred_type = pred_action.get("action", "")

    if pred_type != gt_type:
        return False, False

    # Type matches — now check content
    if gt_type in ("click", "long_press"):
        ok = coord_correct(pred_action.get("coordinate"), gt_action.get("coordinate"), img_w, img_h)
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
        return True, True  # type match, no direction info

    elif gt_type in ("terminate", "wait"):
        return True, True

    elif gt_type == "system_button":
        gt_btn = gt_action.get("button", "").strip().lower()
        pred_btn = pred_action.get("button", "").strip().lower()
        return True, gt_btn == pred_btn

    return True, False  # unknown type, count as type match only


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to v10 checkpoint dir (with grounder/ and actor/ subdirs)")
    parser.add_argument("--eval_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_grounder_tokens", type=int, default=256)
    parser.add_argument("--max_actor_tokens", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Limit eval samples (0 = all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0")

    # ── Load model ────────────────────────────────────────────────────
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    import safetensors.torch

    print(f"Loading base model: {args.model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    for p in model.parameters():
        p.requires_grad = False

    # Read LoRA config from checkpoint
    g_config_path = os.path.join(args.checkpoint, "grounder", "grounder", "adapter_config.json")
    with open(g_config_path) as f:
        adapter_cfg = json.load(f)

    lora_cfg = LoraConfig(
        r=adapter_cfg["r"],
        lora_alpha=adapter_cfg["lora_alpha"],
        lora_dropout=0.0,  # no dropout for eval
        target_modules=adapter_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg, adapter_name="grounder")
    model.add_adapter("actor", lora_cfg)

    # Load weights
    g_path = os.path.join(args.checkpoint, "grounder", "grounder", "adapter_model.safetensors")
    g_weights = safetensors.torch.load_file(g_path)
    set_peft_model_state_dict(model, g_weights, adapter_name="grounder")
    print(f"  Loaded grounder adapter: {len(g_weights)} tensors")

    a_path = os.path.join(args.checkpoint, "actor", "actor", "adapter_model.safetensors")
    a_weights = safetensors.torch.load_file(a_path)
    set_peft_model_state_dict(model, a_weights, adapter_name="actor")
    print(f"  Loaded actor adapter: {len(a_weights)} tensors")

    model = model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_path)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ── Load data ─────────────────────────────────────────────────────
    samples = load_eval_data(args.eval_data)
    if 0 < args.max_samples < len(samples):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(samples), args.max_samples, replace=False)
        samples = [samples[i] for i in sorted(idx)]
    print(f"Loaded {len(samples)} eval samples")

    # ── Evaluate ──────────────────────────────────────────────────────
    results = []
    type_stats = {}
    t0 = time.time()

    for idx, sample in enumerate(samples):
        goal = sample["goal"]
        history = sample["history"]
        image_path = sample["image_path"]
        gt_action = sample["gt_action"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue
        image_w, image_h = image.size

        # Grounder — greedy
        model.set_adapter("grounder")
        g_user = format_grounder_text(goal, history)
        g_msgs = build_messages(GROUNDER_SYSTEM, image_path, g_user)
        g_text_input = processor.apply_chat_template(g_msgs, tokenize=False, add_generation_prompt=True)
        g_inputs = processor(
            text=[g_text_input], images=[image], padding=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            g_out = model.generate(
                **g_inputs,
                max_new_tokens=args.max_grounder_tokens,
                do_sample=False,
            )
        g_prompt_len = g_inputs["input_ids"].shape[1]
        g_text = processor.tokenizer.decode(g_out[0, g_prompt_len:], skip_special_tokens=True)

        # Actor — greedy
        model.set_adapter("actor")
        a_user = format_actor_text(goal, history, g_text)
        a_msgs = build_messages(ACTOR_SYSTEM, image_path, a_user)
        a_text_input = processor.apply_chat_template(a_msgs, tokenize=False, add_generation_prompt=True)
        a_inputs = processor(
            text=[a_text_input], images=[image], padding=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            a_out = model.generate(
                **a_inputs,
                max_new_tokens=args.max_actor_tokens,
                do_sample=False,
            )
        a_prompt_len = a_inputs["input_ids"].shape[1]
        a_text = processor.tokenizer.decode(a_out[0, a_prompt_len:], skip_special_tokens=True)

        pred_action = parse_action_from_text(a_text)
        type_match, action_match = evaluate_action(pred_action, gt_action, image_w, image_h)

        gt_type = gt_action.get("action", "unknown")
        if gt_type not in type_stats:
            type_stats[gt_type] = {"total": 0, "type_match": 0, "action_match": 0}
        type_stats[gt_type]["total"] += 1
        if type_match:
            type_stats[gt_type]["type_match"] += 1
        if action_match:
            type_stats[gt_type]["action_match"] += 1

        result = {
            "idx": idx,
            "goal": goal,
            "history": history[:200],
            "image_path": image_path,
            "gt_action": gt_action,
            "grounder_text": g_text,
            "actor_text": a_text,
            "pred_action": pred_action,
            "type_match": type_match,
            "action_match": action_match,
        }
        results.append(result)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(samples):
            n_done = idx + 1
            n_type = sum(1 for r in results if r["type_match"])
            n_act = sum(1 for r in results if r["action_match"])
            elapsed = time.time() - t0
            print(f"  [{n_done}/{len(samples)}] "
                  f"type_acc={n_type/n_done*100:.1f}% "
                  f"action_acc={n_act/n_done*100:.1f}% "
                  f"({elapsed:.0f}s, {elapsed/n_done:.1f}s/sample)")

    # ── Save results ──────────────────────────────────────────────────
    n = len(results)
    n_type = sum(1 for r in results if r["type_match"])
    n_act = sum(1 for r in results if r["action_match"])

    summary = {
        "checkpoint": args.checkpoint,
        "eval_data": args.eval_data,
        "n_samples": n,
        "type_accuracy": n_type / n if n else 0,
        "action_accuracy": n_act / n if n else 0,
        "per_type": type_stats,
    }

    # Save per-sample results
    result_path = os.path.join(args.output_dir, "eval_results.jsonl")
    with open(result_path, "w") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Print summary
    print("\n" + "=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples:    {n}")
    print(f"Type Acc:   {n_type}/{n} = {n_type/n*100:.1f}%")
    print(f"Action Acc: {n_act}/{n} = {n_act/n*100:.1f}%")
    print()
    print("Per-type breakdown:")
    for t in sorted(type_stats.keys()):
        s = type_stats[t]
        print(f"  {t:15s}  total={s['total']:3d}  "
              f"type={s['type_match']:3d} ({s['type_match']/s['total']*100:5.1f}%)  "
              f"action={s['action_match']:3d} ({s['action_match']/s['total']*100:5.1f}%)")
    print("=" * 60)
    print(f"Results saved to: {result_path}")


if __name__ == "__main__":
    main()
