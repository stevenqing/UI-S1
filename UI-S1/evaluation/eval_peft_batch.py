#!/usr/bin/env python3
"""
Batched PEFT LoRA eval — same pipeline as eval_cooperative_batch.py but for
standard PEFT adapters (e.g. SVD LoRA).  Ensures identical data loading,
prompts, parsing, and scoring for fair per-sample comparison.

Usage:
  python evaluation/eval_peft_batch.py \
      --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
      --peft_checkpoint train_GUI_360/moe_rl/extracted_lora_fullsft_r256 \
      --eval_data_root datasets/GUI-360/test \
      --eval_type action_prediction \
      --output_dir results/ \
      --gpu_id 0 --shard_id 0 --num_shards 4 \
      --batch_size 32
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch
from PIL import Image
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

EVAL_DIR = os.path.join(PROJECT_DIR, "train_GUI_360", "GUI-360-eval")
sys.path.insert(0, EVAL_DIR)

from prompts.prompt_grounding import GROUNDING_USER_PROMPT_QWEN
from prompts.prompt_action_prediction import ACTION_PREDICTION_USER_PROMPT_QWEN
from prompts.prompt_action_prediction import (
    SUPPORTED_ACTIONS_WORD, SUPPORTED_ACTIONS_EXCEL, SUPPORTED_ACTIONS_PPT,
)

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel


# ── Model ──────────────────────────────────────────────────────────

def load_model(base_model_path, peft_checkpoint_path, device):
    print(f"[GPU {device}] Loading base model...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map=device)

    print(f"[GPU {device}] Loading PEFT adapter from {peft_checkpoint_path}...")
    model = PeftModel.from_pretrained(base_model, peft_checkpoint_path)
    model = model.merge_and_unload()
    model.eval()

    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    return model, processor, device


# ── Reuse all data loading / prompts / parsing from cooperative eval ──
# (copied verbatim to guarantee identical behavior)

def load_action_prediction_data(root_dir):
    data_path = os.path.join(root_dir, "data")
    samples = []
    for domain in sorted(os.listdir(data_path)):
        domain_path = os.path.join(data_path, domain)
        if not os.path.isdir(domain_path):
            continue
        for category in sorted(os.listdir(domain_path)):
            cat_path = os.path.join(domain_path, category, "success")
            if not os.path.exists(cat_path):
                continue
            for jsonl_file in sorted(os.listdir(cat_path)):
                if not jsonl_file.endswith(".jsonl"):
                    continue
                filepath = os.path.join(cat_path, jsonl_file)
                all_steps = []
                with open(filepath) as f:
                    for line_num, line in enumerate(f, 1):
                        if not line.strip():
                            continue
                        data = json.loads(line.strip())
                        all_steps.append({"line_num": line_num, "data": data})
                for i, step_info in enumerate(all_steps):
                    data = step_info["data"]
                    if "action_prediction" not in data["step"].get("tags", []):
                        continue
                    clean_img = os.path.join(
                        root_dir, "image", domain, category,
                        data["step"]["screenshot_clean"])
                    if not os.path.exists(clean_img):
                        continue
                    action = data["step"]["action"]
                    if action.get("function") == "drag" or not action.get("rectangle"):
                        continue
                    previous_actions = []
                    for j in range(i):
                        prev = all_steps[j]["data"]["step"]["thought"]
                        previous_actions.append(f"Step {j+1}: {prev}")
                    status = data["step"]["status"]
                    if status == "OVERALL_FINISH":
                        status = "FINISH"
                    elif status == "FINISH":
                        status = "CONTINUE"
                    args = dict(action.get("args", {}))
                    args.pop("x", None)
                    args.pop("y", None)
                    if action.get("coordinate_x") is not None:
                        args["coordinate"] = [action["coordinate_x"], action["coordinate_y"]]
                    sample_id = f"{domain}_{category}_{os.path.splitext(jsonl_file)[0]}_{step_info['line_num']}"
                    samples.append({
                        "sample_id": sample_id,
                        "request": data["request"],
                        "screenshot_clean": clean_img,
                        "thought": data["step"]["thought"],
                        "action": {**action, "args": args},
                        "status": status,
                        "domain": domain,
                        "category": category,
                        "previous_actions": previous_actions,
                        "rectangle": action.get("rectangle", {}),
                    })
    return samples


def build_action_prompt(sample):
    domain = sample["domain"].lower()
    if domain == "word":
        actions = SUPPORTED_ACTIONS_WORD
    elif domain == "excel":
        actions = SUPPORTED_ACTIONS_EXCEL
    elif domain == "ppt":
        actions = SUPPORTED_ACTIONS_PPT
    else:
        actions = SUPPORTED_ACTIONS_WORD
    history = "\n".join(sample["previous_actions"]) if sample["previous_actions"] else ""
    return ACTION_PREDICTION_USER_PROMPT_QWEN.format(
        instruction=sample["request"], history=history, actions=actions)


def build_grounding_prompt(sample):
    return GROUNDING_USER_PROMPT_QWEN.format(instruction=sample["thought"])


def run_batch_inference(model, processor, device, batch_data, max_new_tokens=512):
    if not batch_data:
        return []
    images = []
    texts = []
    for prompt, image_path in batch_data:
        image = Image.open(image_path).convert("RGB")
        images.append(image)
        content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(text)
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    responses = []
    for i in range(len(batch_data)):
        response = processor.decode(output_ids[i][input_len:], skip_special_tokens=True)
        responses.append(response)
    return responses


def parse_coordinates_from_response(response, resolution):
    import re
    m = re.search(r'<coordinate>\s*\[?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]?\s*</coordinate>', response)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r'"coordinate"\s*:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', response)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]', response)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def parse_action_from_response(response):
    import re
    m = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', response, re.DOTALL)
    if not m:
        m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if not m:
        m = re.search(r'(\{"function".*?\})', response, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            return data.get("function"), data.get("args", {}), data.get("status")
        except json.JSONDecodeError:
            pass
    return None, None, None


def check_coordinate_in_rect(x, y, rect):
    return (rect.get("left", 0) <= x <= rect.get("right", 0) and
            rect.get("top", 0) <= y <= rect.get("bottom", 0))


def compare_args(pred_args, gt_args, rect, resolution):
    if not pred_args or not gt_args:
        return False
    if "coordinate" in gt_args and "coordinate" in pred_args:
        px, py = pred_args["coordinate"]
        if not check_coordinate_in_rect(float(px), float(py), rect):
            return False
    for key in gt_args:
        if key == "coordinate":
            continue
        if key not in pred_args:
            return False
        if str(pred_args[key]).lower() != str(gt_args[key]).lower():
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--peft_checkpoint", required=True)
    parser.add_argument("--eval_data_root", required=True)
    parser.add_argument("--eval_type", choices=["grounding", "action_prediction"], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}"
    model, processor, dev = load_model(args.base_model, args.peft_checkpoint, device)

    print(f"[Shard {args.shard_id}] Loading {args.eval_type} data...")
    if args.eval_type == "action_prediction":
        all_samples = load_action_prediction_data(args.eval_data_root)
    else:
        raise NotImplementedError("Only action_prediction supported for now")
    print(f"[Shard {args.shard_id}] Total samples: {len(all_samples)}")

    if args.max_samples > 0:
        all_samples = all_samples[:args.max_samples]

    if args.num_shards > 1:
        shard_size = (len(all_samples) + args.num_shards - 1) // args.num_shards
        start = args.shard_id * shard_size
        end = min(start + shard_size, len(all_samples))
        samples = all_samples[start:end]
        print(f"[Shard {args.shard_id}] Processing [{start}:{end}] = {len(samples)} samples")
    else:
        samples = all_samples

    os.makedirs(args.output_dir, exist_ok=True)

    max_tokens = 512
    results = []
    success_count = 0
    total = 0

    num_batches = (len(samples) + args.batch_size - 1) // args.batch_size
    print(f"[Shard {args.shard_id}] batch_size={args.batch_size}, num_batches={num_batches}")

    t0 = time.time()
    for batch_idx in tqdm(range(num_batches), desc=f"Shard {args.shard_id}"):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, len(samples))
        batch_samples = samples[batch_start:batch_end]

        batch_data = []
        for sample in batch_samples:
            prompt = build_action_prompt(sample)
            batch_data.append((prompt, sample["screenshot_clean"]))

        try:
            responses = run_batch_inference(
                model, processor, dev, batch_data, max_new_tokens=max_tokens)
        except Exception as e:
            print(f"[Shard {args.shard_id}] Batch {batch_idx} failed: {e}")
            for sample in batch_samples:
                total += 1
                results.append({
                    "sample_id": sample.get("sample_id", ""),
                    "success": False,
                    "error": str(e),
                })
            continue

        for sample, response in zip(batch_samples, responses):
            try:
                resolution = Image.open(sample["screenshot_clean"]).size
                pred_fn, pred_args, pred_status = parse_action_from_response(response)
                gt_fn = sample["action"].get("function", "")
                gt_args = sample["action"].get("args", {})
                success = False
                if pred_fn == gt_fn:
                    if pred_args and gt_args:
                        success = compare_args(pred_args, gt_args, sample["rectangle"], resolution)
                total += 1
                if success:
                    success_count += 1
                results.append({
                    "sample_id": sample["sample_id"],
                    "success": success,
                    "response": response[:500],
                })
            except Exception as e:
                total += 1
                results.append({
                    "sample_id": sample.get("sample_id", ""),
                    "success": False,
                    "error": str(e),
                })

        if (batch_idx + 1) % 5 == 0 or batch_idx == num_batches - 1:
            elapsed = time.time() - t0
            rate = 100 * success_count / total if total > 0 else 0
            samples_per_sec = total / elapsed if elapsed > 0 else 0
            print(f"[Shard {args.shard_id}] {total}/{len(samples)}, "
                  f"success_rate={rate:.1f}%, "
                  f"speed={samples_per_sec:.1f} samples/s, "
                  f"elapsed={elapsed:.0f}s")

    elapsed = time.time() - t0
    rate = 100 * success_count / total if total > 0 else 0
    summary = {
        "total_samples": total,
        "success_count": success_count,
        "success_rate": rate,
        "eval_type": args.eval_type,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "batch_size": args.batch_size,
        "inference_mode": "peft_merged_batch",
        "peft_checkpoint": args.peft_checkpoint,
        "elapsed_seconds": elapsed,
        "samples_per_second": total / elapsed if elapsed > 0 else 0,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    with open(os.path.join(args.output_dir, f"summary{tag}_{ts}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, f"results{tag}_{ts}.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Shard {args.shard_id}] {args.eval_type}: {success_count}/{total} = {rate:.1f}%")
    print(f"[Shard {args.shard_id}] Speed: {total/elapsed:.1f} samples/s, Total: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
