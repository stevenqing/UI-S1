#!/usr/bin/env python3
"""
Selective MoE Experiment: Evaluation Script

Evaluates trained selective MoE checkpoints using HF generate (not vLLM),
since MoE two-pass routing is incompatible with vLLM.

For each sample:
1. Build input with processor
2. Call wrapper.generate() (two-pass: extract features → route → generate)
3. Parse output, compare against GT
4. Compute action accuracy, grounding accuracy, error decomposition

Usage:
  python evaluation/selective_moe_eval.py \
      --checkpoint_dir train_GUI_360/llamafactory/output/selective_moe_c2/final/moe_checkpoint \
      --eval_data train_GUI_360/llamafactory/data/gui360_val.json \
      --output_dir results/selective_moe_c2
"""

import argparse
import json
import os
import re
import sys
import time
import torch
import numpy as np
from collections import defaultdict
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Tuple

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Ensure project root is on sys.path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from verl.models.moe.moe_wrapper import MoEVLMWrapper, MoEConfig


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════

def parse_tool_call(text):
    """Parse tool_call JSON from assistant response."""
    if not text:
        return None
    m = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return None


def evaluate_action(pred_action, gt_action, threshold=50):
    """Evaluate predicted action against ground truth."""
    result = {
        "function_match": False,
        "args_match": False,
        "full_match": False,
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

    if pred_func == "click":
        pred_coord = pred_args.get("coordinate", [])
        gt_coord = gt_args.get("coordinate", [])
        if len(pred_coord) == 2 and len(gt_coord) == 2:
            dist = ((pred_coord[0] - gt_coord[0])**2 +
                    (pred_coord[1] - gt_coord[1])**2)**0.5
            result["args_match"] = dist < threshold
        # Also check bbox if available
        gt_bbox = gt_action.get("bbox")
        if gt_bbox and len(pred_coord) == 2:
            left, top = gt_bbox.get("left"), gt_bbox.get("top")
            right, bottom = gt_bbox.get("right"), gt_bbox.get("bottom")
            if all(v is not None for v in [left, top, right, bottom]):
                if left <= pred_coord[0] <= right and top <= pred_coord[1] <= bottom:
                    result["args_match"] = True

    elif pred_func == "type":
        pred_text = pred_args.get("text", "").lower().strip()
        gt_text = gt_args.get("text", "").lower().strip()
        result["args_match"] = (pred_text == gt_text)

    elif pred_func in ["wheel_mouse_input", "scroll"]:
        result["args_match"] = (
            pred_args.get("direction", "") == gt_args.get("direction", ""))

    elif pred_func == "summary":
        result["args_match"] = True

    else:
        result["args_match"] = (pred_args == gt_args)

    result["full_match"] = result["function_match"] and result["args_match"]
    return result


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Selective MoE Experiment: Evaluation")

    parser.add_argument("--model_path",
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/checkpoints/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Path to moe_checkpoint directory")
    parser.add_argument("--eval_data", required=True,
                        help="Path to evaluation JSON (ShareGPT format)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="0 = use all")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for generation (1 recommended for variable-length)")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load MoE config from checkpoint ──
    config_path = os.path.join(args.checkpoint_dir, "moe_config.json")
    if not os.path.exists(config_path):
        print(f"ERROR: moe_config.json not found in {args.checkpoint_dir}")
        sys.exit(1)
    with open(config_path) as f:
        config_dict = json.load(f)
    moe_config = MoEConfig.from_dict(config_dict)

    print("=" * 80)
    print("Selective MoE Experiment: Evaluation")
    print("=" * 80)
    print(f"Model:      {args.model_path}")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"MoE config: experts={moe_config.num_experts}, "
          f"moe_r={moe_config.expert_lora_r}, "
          f"std_r={moe_config.standard_lora_r}")
    print(f"MoE modules: {moe_config.moe_modules}")
    print(f"Output:     {args.output_dir}")
    print()

    # ── Load model ──
    print("Loading base model...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True)

    # ── Reconstruct wrapper and load checkpoint ──
    print("Building MoEVLMWrapper...", flush=True)
    wrapper = MoEVLMWrapper(
        base_model=model,
        moe_config=moe_config,
        tokenizer=processor.tokenizer,
    )

    print("Loading checkpoint...", flush=True)
    wrapper.load_selective_checkpoint(args.checkpoint_dir)
    # Ensure router and all LoRA modules are on the correct device and dtype
    wrapper.router.to(device=args.device, dtype=torch.bfloat16)
    for m in wrapper._moe_linear_modules:
        m.to(device=args.device, dtype=torch.bfloat16)
    for m in wrapper._std_linear_modules:
        m.to(device=args.device, dtype=torch.bfloat16)
    wrapper.eval()

    print(f"Trainable params: {wrapper.num_trainable_parameters():,}")
    print(f"MoE modules:      {len(wrapper._moe_linear_modules)}")
    print(f"Standard modules:  {len(wrapper._std_linear_modules)}")
    print()

    # ── Load eval data ──
    print("Loading eval data...", flush=True)
    with open(args.eval_data) as f:
        eval_data = json.load(f)

    if 0 < args.max_samples < len(eval_data):
        np.random.seed(42)
        indices = np.random.choice(len(eval_data), args.max_samples, replace=False)
        eval_data = [eval_data[i] for i in sorted(indices)]

    print(f"Eval samples: {len(eval_data)}")

    # ── Evaluate ──
    results = []
    routing_records = []
    func_counts = defaultdict(int)
    func_correct = defaultdict(int)

    total = len(eval_data)
    correct = 0
    func_match = 0
    parse_fail = 0

    for idx, item in enumerate(eval_data):
        convs = item["conversations"]
        images = item.get("images", [])

        # Parse GT
        assistant_text = convs[1]["value"]
        gt_action = parse_tool_call(assistant_text)

        # Build prompt (user turn only)
        user_text = convs[0]["value"]
        user_text_clean = user_text.replace("<image>\n", "").replace("<image>", "").strip()

        user_content = []
        if images:
            user_content.append({"type": "image", "image": images[0]})
        user_content.append({"type": "text", "text": user_text_clean})

        messages = [{"role": "user", "content": user_content}]
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        # Process input
        try:
            image = None
            if images and os.path.exists(images[0]):
                image = Image.open(images[0]).convert("RGB")

            if image is not None:
                inputs = processor(
                    text=[prompt_text], images=[image],
                    return_tensors="pt", padding=False)
            else:
                inputs = processor(
                    text=[prompt_text],
                    return_tensors="pt", padding=False)

            inputs = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}

        except Exception as e:
            print(f"  [{idx+1}/{total}] SKIP (input error): {e}")
            results.append({"idx": idx, "error": str(e)})
            continue

        # Generate
        try:
            with torch.no_grad():
                generated_ids, router_output = wrapper.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )

            # Decode only the generated part
            input_len = inputs["input_ids"].shape[1]
            generated_text = processor.tokenizer.decode(
                generated_ids[0, input_len:], skip_special_tokens=True)

        except Exception as e:
            print(f"  [{idx+1}/{total}] SKIP (generate error): {e}")
            results.append({"idx": idx, "error": str(e)})
            continue

        # Parse and evaluate
        pred_action = parse_tool_call(generated_text)
        if pred_action is None:
            parse_fail += 1

        eval_result = evaluate_action(pred_action, gt_action)

        gt_func = gt_action.get("function", "unknown") if gt_action else "none"
        func_counts[gt_func] += 1
        if eval_result["full_match"]:
            correct += 1
            func_correct[gt_func] += 1
        if eval_result["function_match"]:
            func_match += 1

        # Record routing weights
        routing_info = {}
        if router_output is not None and router_output.routing_weights is not None:
            rw = router_output.routing_weights[0].cpu().tolist()
            routing_info = {"routing_weights": rw}

        result_entry = {
            "idx": idx,
            "gt_function": gt_func,
            "pred_text": generated_text[:200],
            **eval_result,
            **routing_info,
        }
        results.append(result_entry)

        # Progress
        if (idx + 1) % 50 == 0 or idx + 1 == total:
            acc = correct / (idx + 1) * 100
            print(f"  [{idx+1}/{total}] acc={acc:.1f}% "
                  f"(func_match={func_match}, parse_fail={parse_fail})")

    # ── Summary ──
    total_evaluated = len([r for r in results if "error" not in r])
    accuracy = correct / total_evaluated * 100 if total_evaluated > 0 else 0
    func_accuracy = func_match / total_evaluated * 100 if total_evaluated > 0 else 0

    summary = {
        "checkpoint": args.checkpoint_dir,
        "moe_modules": config_dict.get("moe_modules"),
        "total_samples": total,
        "evaluated": total_evaluated,
        "correct": correct,
        "accuracy": round(accuracy, 2),
        "function_accuracy": round(func_accuracy, 2),
        "parse_failures": parse_fail,
        "per_function": {
            func: {
                "total": func_counts[func],
                "correct": func_correct.get(func, 0),
                "accuracy": round(func_correct.get(func, 0) / func_counts[func] * 100, 2)
                if func_counts[func] > 0 else 0,
            }
            for func in sorted(func_counts.keys())
        },
    }

    # Compute routing statistics from results
    all_weights = [r["routing_weights"] for r in results
                   if "routing_weights" in r and len(r["routing_weights"]) > 1]
    if all_weights:
        weights_arr = np.array(all_weights)
        summary["routing_stats"] = {
            "mean_weights": weights_arr.mean(axis=0).tolist(),
            "std_weights": weights_arr.std(axis=0).tolist(),
            "mean_entropy": float(-np.sum(
                weights_arr * np.log(weights_arr + 1e-8), axis=1).mean()),
        }

    print()
    print("=" * 80)
    print("Results Summary")
    print("=" * 80)
    print(f"Accuracy:       {accuracy:.2f}% ({correct}/{total_evaluated})")
    print(f"Func accuracy:  {func_accuracy:.2f}%")
    print(f"Parse failures: {parse_fail}")
    print()
    for func in sorted(func_counts.keys()):
        fc = func_correct.get(func, 0)
        ft = func_counts[func]
        print(f"  {func:20s}: {fc}/{ft} = {fc/ft*100:.1f}%")
    if "routing_stats" in summary:
        print(f"\nRouting weights (mean): {summary['routing_stats']['mean_weights']}")
        print(f"Routing entropy (mean): {summary['routing_stats']['mean_entropy']:.4f}")

    # ── Save ──
    with open(os.path.join(args.output_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
