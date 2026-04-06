#!/usr/bin/env python3
"""
Evaluate cooperative LoRA with proper token-level routing (no merge).

Loads base model + cooperative LoRA checkpoint, uses CooperativeVLMWrapper.generate()
with hook-based mask routing. Outputs results compatible with GUI-360 eval framework.

Usage:
  python evaluation/eval_cooperative_proper.py \
      --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
      --coop_checkpoint train_GUI_360/llamafactory/output/cooperative_thought_v1/final \
      --eval_type grounding \
      --output_dir results/cooperative_v1_proper/grounding
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Add project root
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from verl.models.cooperative.cooperative_wrapper import CooperativeVLMWrapper


def load_cooperative_model(base_model_path, coop_checkpoint_path, device="cuda"):
    """Load base model and apply cooperative LoRA."""
    print(f"Loading base model from {base_model_path}...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device,
    )

    # Read cooperative config
    config_path = os.path.join(coop_checkpoint_path, "cooperative_config.json")
    with open(config_path) as f:
        coop_config = json.load(f)

    print(f"Wrapping with cooperative LoRA (targets={coop_config['target_modules']})...")
    # Infer r from checkpoint
    lora_v_state = torch.load(
        os.path.join(coop_checkpoint_path, "lora_v.pt"),
        map_location="cpu", weights_only=True)
    first_A = [v for k, v in lora_v_state.items() if "lora_A_v" in k][0]
    r = first_A.shape[0]
    alpha = r * 2  # default scaling = alpha/r = 2.0

    model = CooperativeVLMWrapper(
        base_model=base_model,
        lora_r=r,
        lora_alpha=alpha,
        lora_dropout=0.0,
        target_modules=coop_config["target_modules"],
        bind_weight=0.0,
        bind_layer=coop_config.get("bind_layer", 27),
    )

    print("Loading cooperative checkpoint...")
    model.load_cooperative_checkpoint(coop_checkpoint_path)
    model.eval()

    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    return model, processor


def load_eval_data(root_dir, eval_type):
    """Load GUI-360 evaluation data."""
    samples = []
    data_dir = os.path.join(root_dir, "data")
    image_dir = os.path.join(root_dir, "image")

    for domain in ["excel", "ppt", "word"]:
        domain_data = os.path.join(data_dir, domain)
        if not os.path.exists(domain_data):
            continue
        for category in os.listdir(domain_data):
            cat_dir = os.path.join(domain_data, category)
            if not os.path.isdir(cat_dir):
                continue
            for status_dir in os.listdir(cat_dir):
                full_dir = os.path.join(cat_dir, status_dir)
                if not os.path.isdir(full_dir):
                    continue
                for jsonl_file in os.listdir(full_dir):
                    if not jsonl_file.endswith(".jsonl"):
                        continue
                    filepath = os.path.join(full_dir, jsonl_file)
                    with open(filepath) as f:
                        for line in f:
                            sample = json.loads(line.strip())
                            sample["_domain"] = domain
                            sample["_category"] = category
                            sample["_image_dir"] = image_dir
                            if eval_type == "grounding":
                                # Only include samples tagged for grounding
                                tags = sample.get("step", {}).get("tags", [])
                                if "grounding" in tags:
                                    samples.append(sample)
                            else:
                                samples.append(sample)
    return samples


def build_grounding_prompt(sample, image_dir):
    """Build grounding evaluation prompt."""
    step = sample["step"]
    action = step["action"]

    # Get the control info
    control_text = action.get("control_test", "")
    control_label = action.get("control_label", "")

    # Build prompt asking for element location
    prompt = f"In this screenshot, please locate the UI element: {control_text}"
    if control_label:
        prompt += f" (label: {control_label})"

    # Get image path: image/<domain>/<category>/<screenshot_clean>
    screenshot = step.get("screenshot_clean", "")
    image_path = os.path.join(image_dir, sample["_domain"], sample["_category"], screenshot)

    return prompt, image_path


def build_action_prediction_prompt(sample, image_dir):
    """Build action prediction prompt matching GUI-360 eval format."""
    step = sample["step"]
    request = sample.get("request", "")
    subtask = step.get("subtask", "")
    observation = step.get("observation", "")

    prompt = f"""You are a GUI agent. Based on the screenshot and the task information, predict the next action.

Task: {request}
Current subtask: {subtask}
Observation: {observation}

Predict the next action in tool_call format."""

    screenshot = step.get("screenshot_clean", "")
    image_path = os.path.join(image_dir, sample["_domain"], sample["_category"], screenshot)

    return prompt, image_path


def run_inference(model, processor, prompt, image_path, max_new_tokens=512):
    """Run inference with cooperative routing."""
    content = []
    image = None
    if os.path.exists(image_path):
        content.append({"type": "image", "image": image_path})
        image = Image.open(image_path).convert("RGB")
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if image is not None:
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    else:
        inputs = processor(text=[text], return_tensors="pt", padding=True)

    inputs = {k: v.to(model.base_model.device) if hasattr(v, 'to') else v
              for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
    )

    # Decode only new tokens
    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0][input_len:]
    response = processor.decode(generated, skip_special_tokens=True)
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--coop_checkpoint", required=True)
    parser.add_argument("--eval_data_root", required=True,
                        help="GUI-360 test data root")
    parser.add_argument("--eval_type", choices=["grounding", "action_prediction"],
                        required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Limit samples for quick test (0 = all)")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Shard index for multi-GPU parallelism")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of shards")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU device ID to use")
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}"
    model, processor = load_cooperative_model(args.base_model, args.coop_checkpoint, device=device)

    print(f"Loading {args.eval_type} eval data from {args.eval_data_root}...")
    samples = load_eval_data(args.eval_data_root, args.eval_type)
    print(f"Loaded {len(samples)} samples total")

    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    # Shard the data
    if args.num_shards > 1:
        shard_size = (len(samples) + args.num_shards - 1) // args.num_shards
        start = args.shard_id * shard_size
        end = min(start + shard_size, len(samples))
        samples = samples[start:end]
        print(f"Shard {args.shard_id}/{args.num_shards}: samples [{start}:{end}] = {len(samples)}")

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    success_count = 0
    total_count = 0

    for sample in tqdm(samples, desc=f"Evaluating {args.eval_type}"):
        try:
            if args.eval_type == "grounding":
                prompt, image_path = build_grounding_prompt(sample, sample["_image_dir"])
            else:
                prompt, image_path = build_action_prediction_prompt(sample, sample["_image_dir"])

            response = run_inference(model, processor, prompt, image_path)

            # Simple success check (matches eval framework logic)
            step = sample["step"]
            action = step["action"]
            gt_x = action.get("coordinate_x")
            gt_y = action.get("coordinate_y")

            # Parse predicted coordinate from response
            success = False
            coord_match = re.search(r'\[?\s*(\d+)\s*,\s*(\d+)\s*\]?', response)
            if coord_match and gt_x is not None and gt_y is not None:
                pred_x, pred_y = int(coord_match.group(1)), int(coord_match.group(2))
                # Within 14px tolerance (1 token)
                if abs(pred_x - gt_x) <= 14 and abs(pred_y - gt_y) <= 14:
                    success = True

            total_count += 1
            if success:
                success_count += 1

            results.append({
                "sample_id": sample.get("execution_id", ""),
                "step_id": sample.get("step_id", 0),
                "domain": sample["_domain"],
                "category": sample["_category"],
                "success": success,
                "response": response[:500],
            })

        except Exception as e:
            total_count += 1
            results.append({
                "sample_id": sample.get("execution_id", ""),
                "error": str(e),
                "success": False,
            })

        if total_count % 100 == 0:
            print(f"  Progress: {total_count}/{len(samples)}, "
                  f"success_rate={100*success_count/total_count:.1f}%")

    # Save results
    rate = 100 * success_count / total_count if total_count > 0 else 0
    summary = {
        "total_samples": total_count,
        "success_count": success_count,
        "success_rate": rate,
        "eval_type": args.eval_type,
        "inference_mode": "proper_routing (no merge)",
        "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    shard_tag = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.output_dir, f"evaluation_summary{shard_tag}_{ts}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, f"evaluation_results{shard_tag}_{ts}.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"{args.eval_type}: {success_count}/{total_count} = {rate:.1f}%")
    print(f"Results saved to {args.output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
