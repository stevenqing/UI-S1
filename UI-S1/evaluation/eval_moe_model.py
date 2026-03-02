#!/usr/bin/env python3
"""
MoE Model Evaluation Script.

This script evaluates MoE (Mixture of Experts) checkpoints that cannot be
served via vLLM due to the custom routing architecture.

Usage:
    python eval_moe_model.py \
        --checkpoint_dir /path/to/checkpoint/global_step_X \
        --base_model /path/to/Qwen2.5-VL-7B-Instruct \
        --jsonl_file /path/to/evaluation.jsonl \
        --output_dir /path/to/results \
        --model_name MoE_model_name
"""

import argparse
import copy
import json
import os
import sys
from typing import List, Dict, Any, Optional
import torch
from PIL import Image
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoProcessor, AutoModelForVision2Seq
from qwenvl_utils import process_vision_info


def load_moe_model(
    checkpoint_dir: str,
    base_model_path: str,
    world_size: int = 32,
    device: str = "cuda",
):
    """
    Load MoE model from checkpoint.

    Args:
        checkpoint_dir: Path to checkpoint directory (e.g., global_step_20)
        base_model_path: Path to base Qwen2.5-VL model
        world_size: Number of shards used during training
        device: Device to load model on

    Returns:
        model: MoEVLMWrapper instance
        processor: AutoProcessor instance
    """
    from verl.models.moe.moe_wrapper import MoEVLMWrapper, MoEConfig

    actor_dir = os.path.join(checkpoint_dir, "actor")
    moe_dir = os.path.join(actor_dir, "moe")

    print(f"Loading MoE model from {checkpoint_dir}")
    print(f"  Actor dir: {actor_dir}")
    print(f"  MoE dir: {moe_dir}")
    print(f"  Base model: {base_model_path}")

    # Load processor
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

    # Load MoE config
    moe_config_path = os.path.join(moe_dir, "moe_config.json")
    with open(moe_config_path, 'r') as f:
        moe_config_dict = json.load(f)

    # Note: Override target_modules to only use q_proj since the checkpoint
    # has combined q_proj/v_proj weights with only q_proj dimensions
    moe_config = MoEConfig(
        num_experts=moe_config_dict.get('num_experts', 4),
        top_k=moe_config_dict.get('top_k', 1),
        expert_lora_r=moe_config_dict.get('expert_lora_r', 16),
        expert_lora_alpha=moe_config_dict.get('expert_lora_alpha', 32),
        target_modules=['q_proj'],  # Only q_proj - v_proj weights have wrong dimensions
        router_hidden=moe_config_dict.get('router_hidden', 256),
    )

    print(f"  MoE config: num_experts={moe_config.num_experts}, top_k={moe_config.top_k}")

    # Load base model
    print("Loading base model...")
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    # Create MoE wrapper
    print("Creating MoE wrapper...")
    model = MoEVLMWrapper(
        base_model=base_model,
        moe_config=moe_config,
        tokenizer=processor.tokenizer,
    )

    # Load MoE weights (router and experts)
    print("Loading MoE weights...")

    # Load router
    router_path = os.path.join(moe_dir, "router.pt")
    if os.path.exists(router_path):
        router_state = torch.load(router_path, map_location="cpu")
        model.router.load_state_dict(router_state)
        print(f"  Loaded router from {router_path}")

    # Load experts
    experts_path = os.path.join(moe_dir, "experts.pt")
    if os.path.exists(experts_path):
        experts_state = torch.load(experts_path, map_location="cpu")

        # Remap keys if they use the old format with ['q_proj', 'v_proj'] in the key name
        remapped_state = {}
        for key, value in experts_state.items():
            new_key = key
            # Handle keys like: layer_0_['q_proj', 'v_proj'].lora_A -> layer_0_q_proj.lora_A
            # Note: Only map to q_proj since the checkpoint has q_proj dimensions (3584x3584)
            # The v_proj has different dimensions (3584x512) and can't use these weights
            if "_['q_proj', 'v_proj']." in key:
                base_key = key.replace("_['q_proj', 'v_proj'].", "_q_proj.")
                remapped_state[base_key] = value
            else:
                remapped_state[new_key] = value

        # Load with strict=False since we're only loading q_proj weights
        model.expert_collection.load_state_dict(remapped_state, strict=False)
        print(f"  Loaded experts from {experts_path} (q_proj only)")

    # Move model to device and set to eval mode
    # Convert entire model to bfloat16 and device at once
    model = model.to(dtype=torch.bfloat16, device=device)
    model.eval()

    # Register hooks for expert application
    model.register_hooks()

    print("MoE model loaded successfully!")
    return model, processor


def generate_response(
    model,
    processor,
    messages: List[Dict],
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> str:
    """
    Generate response from MoE model.

    Args:
        model: MoEVLMWrapper
        processor: AutoProcessor
        messages: List of message dicts with 'role' and 'content'
        max_new_tokens: Maximum tokens to generate
        device: Device

    Returns:
        Generated text response
    """
    # Process messages using Qwen's chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Process images from messages
    image_inputs, video_inputs = process_vision_info(messages)

    # Prepare inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Generate using MoE model
    with torch.no_grad():
        generated_ids, router_output = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.get('pixel_values'),
            image_grid_thw=inputs.get('image_grid_thw'),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0]


def fix_line(line):
    """Fix line format for evaluation."""
    for step in line['steps']:
        check_options = copy.deepcopy(step['action_content'])
        if 'candidate_bbox' in step:
            continue
        if 'bbox' in step:
            check_options['candidate_bbox'] = step['bbox']
        else:
            check_options['candidate_bbox'] = []
        step['check_options'] = check_options
    return line


def evaluate_moe_model(
    model,
    processor,
    jsonl_file: str,
    output_dir: str,
    model_name: str,
    n_history_image_limit: int = 2,
    device: str = "cuda",
):
    """
    Evaluate MoE model on Android control tasks.

    Args:
        model: MoEVLMWrapper
        processor: AutoProcessor
        jsonl_file: Path to evaluation JSONL file
        output_dir: Directory to save results
        model_name: Name for the model (used in output file)
        n_history_image_limit: Max historical images
        device: Device
    """
    from x.data.agent.space.std_space import RAW_SPACE
    from x.data.agent.json import JsonFormat
    from x.qwen.data_format import slim_messages
    from qwenvl_utils import evaluate_android_control_action, find_last_image_ele

    fm = JsonFormat(RAW_SPACE, add_thought=True, force_add_thought=True)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    std_data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            std_data.append(json.loads(line))

    print(f"Loaded {len(std_data)} tasks. Starting evaluation...")

    results = []
    result_path = os.path.join(output_dir, f"{model_name}.jsonl")

    for line in tqdm(std_data, desc="Evaluating"):
        num_steps = len(line['steps'])
        state = None
        model_response = None
        step_id = 0
        task_success = False
        fixed_line = fix_line(line)

        try:
            while step_id < num_steps:
                current_check_pam = fixed_line['steps'][step_id]['check_options']
                state = fm.gen_next_round(fixed_line, state, previous_model_response=model_response)
                if state is None:
                    break

                messages = state['messages']
                messages = slim_messages(messages=messages, num_image_limit=n_history_image_limit)

                current_image_ele, width, height, resized_width, resized_height = find_last_image_ele(messages)

                # Generate response using MoE model
                model_response = generate_response(
                    model=model,
                    processor=processor,
                    messages=messages,
                    device=device,
                )

                pred_action = fm.parse_response(model_response)
                type_match, extract_match = evaluate_android_control_action(
                    pred_action['action_content'],
                    current_check_pam,
                    width, height,
                    resized_width, resized_height
                )

                if not extract_match:
                    break

                step_id += 1

            task_success = (step_id == num_steps)

        except Exception as e:
            import traceback
            print(f"Error processing goal '{line['goal']}': {e}")
            traceback.print_exc()
            task_success = False
            step_id = 0

        # Record result
        result = {
            "goal": line['goal'],
            "num_steps": num_steps,
            "task_success": task_success,
            "final_step_id": step_id,
        }
        results.append(result)

        # Write incrementally
        with open(result_path, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # Print summary
    success_count = sum(1 for r in results if r["task_success"])
    success_rate = success_count / len(results) * 100 if results else 0
    avg_progress = sum(r["final_step_id"] / r['num_steps'] for r in results) / len(results) if results else 0.0

    print(f"\nEvaluation completed.")
    print(f"Success Rate: {success_rate:.2f}% ({success_count}/{len(results)})")
    print(f"Average Progress: {avg_progress:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate MoE model on Android control tasks")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g., global_step_20)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/checkpoints/Qwen2.5-VL-7B-Instruct",
        help="Path to base Qwen2.5-VL model"
    )
    parser.add_argument(
        "--jsonl_file",
        type=str,
        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation/dataset/android_control_evaluation_std.jsonl",
        help="Path to evaluation JSONL file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation/results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name for the model (used in output filename)"
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=32,
        help="World size used during training"
    )
    parser.add_argument(
        "--n_history_image_limit",
        type=int,
        default=2,
        help="Maximum number of historical images"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on"
    )

    args = parser.parse_args()

    # Load model
    model, processor = load_moe_model(
        checkpoint_dir=args.checkpoint_dir,
        base_model_path=args.base_model,
        world_size=args.world_size,
        device=args.device,
    )

    # Run evaluation
    evaluate_moe_model(
        model=model,
        processor=processor,
        jsonl_file=args.jsonl_file,
        output_dir=args.output_dir,
        model_name=args.model_name,
        n_history_image_limit=args.n_history_image_limit,
        device=args.device,
    )


if __name__ == "__main__":
    main()
