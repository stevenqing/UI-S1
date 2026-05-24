#!/usr/bin/env python3
"""V15 Gradient Interference Diagnostic (LEAS-inspired).

Measures gradient cosine similarity between Expert A (lora_A_1) and Expert B (lora_A_2)
across different conditions:
  1. Overall: do experts receive similar or orthogonal gradients?
  2. By action type: click vs type vs swipe/drag
  3. By token role: coordinate tokens vs action-type tokens vs other
  4. Routing weights distribution: what does the learned router actually do?

Usage:
  python v15_gui_360/diagnose_gradient_interference.py \
    --model_path checkpoints/Qwen2.5-VL-7B-Instruct \
    --coop_checkpoint checkpoints/v15_rl_from_coop_sft/epoch-1_step-50/cooperative \
    --train_data v12_gui_360/data/gui360_train_2000_balanced.jsonl \
    --num_samples 100
"""

import argparse
import json
import os
import sys
import time
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("gradient_diag")

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from v13_gui_360.iterative_cooperative_wrapper import IterativeCooperativeVLMWrapper


def load_model(args):
    """Load V15 cooperative model."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    logger.info(f"Loading base model: {args.model_path}")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        max_pixels=602112,
        min_pixels=3136,
    )

    logger.info(f"Loading cooperative checkpoint: {args.coop_checkpoint}")
    model = IterativeCooperativeVLMWrapper(
        base_model,
        lora_r=128,
        lora_alpha=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        num_comm_rounds=2,
    )
    model.load_cooperative(args.coop_checkpoint)
    model.eval()

    # Freeze base, enable grad for LoRA
    for name, param in model.named_parameters():
        if "lora_" in name or "route" in name or "comm" in name or "gate" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model, processor


def load_episodes(data_path, num_samples):
    """Load episodes and flatten to individual steps."""
    steps = []
    with open(data_path) as f:
        for line in f:
            ep = json.loads(line)
            for si, step in enumerate(ep.get("steps", [])):
                steps.append({
                    "goal": ep["goal"],
                    "screenshot": step["screenshot"],
                    "action": step["action"],
                    "step_idx": si,
                    "episode_id": ep.get("episode_id", ""),
                    "image_w": step.get("image_w", 1040),
                    "image_h": step.get("image_h", 736),
                })
                if len(steps) >= num_samples:
                    return steps
    return steps


SYSTEM_PROMPT = """You are a GUI automation agent. You can interact with the computer screen to complete tasks.
Available actions: click(coordinate=[x,y]), type(coordinate=[x,y], keys='text'), drag(start_coordinate=[x,y], end_coordinate=[x,y])
Output your action as a tool_call JSON block inside <decision></decision> tags."""


def build_messages(goal, screenshot):
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": screenshot},
            {"type": "text", "text": f"Goal: {goal}\nWhat action should be taken?"},
        ]},
    ]


def format_gt_response(action):
    """Format GT action as model response."""
    atype = action.get("action", "click")
    coord = action.get("coordinate")
    text = action.get("text", "")
    end_coord = action.get("endCoordinate")

    if atype == "click" and coord and coord[0] is not None:
        call = json.dumps({"function": "click", "args": {"coordinate": [int(coord[0]), int(coord[1])]}})
    elif atype == "type" and coord and coord[0] is not None:
        call = json.dumps({"function": "type", "args": {"coordinate": [int(coord[0]), int(coord[1])], "keys": text}})
    elif atype in ("swipe", "drag") and coord and coord[0] is not None and end_coord and end_coord[0] is not None:
        call = json.dumps({"function": "drag", "args": {
            "start_coordinate": [int(coord[0]), int(coord[1])],
            "end_coordinate": [int(end_coord[0]), int(end_coord[1])]
        }})
    else:
        call = json.dumps({"function": atype, "args": {}})

    return f"<decision>\n<tool_call>\n{call}\n</tool_call>\n</decision>"


def collect_gradients(model, processor, step_data, device):
    """Run forward+backward on one sample, return per-expert gradients and routing info."""
    goal = step_data["goal"]
    screenshot = step_data["screenshot"]
    action = step_data["action"]

    try:
        image = Image.open(screenshot).convert("RGB")
    except Exception:
        return None

    messages = build_messages(goal, screenshot)
    gt_response = format_gt_response(action)

    # Tokenize
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text += gt_response

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Forward
    model.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.base_model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

    loss.backward()

    # Collect gradients for Expert A (lora_A_1) and Expert B (lora_A_2)
    grad_a1 = []
    grad_a2 = []
    grad_b = []
    route_weights = []

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        g = param.grad.detach().float().flatten()
        if "lora_A_1" in name:
            grad_a1.append(g)
        elif "lora_A_2" in name:
            grad_a2.append(g)
        elif "lora_B" in name:
            grad_b.append(g)

    # Collect routing weights
    for module in model.cooperative_modules:
        with torch.no_grad():
            # Get routing weight for this sample
            w = module.route_weight.detach().float().cpu()
            route_weights.append(w)

    if not grad_a1 or not grad_a2:
        return None

    grad_a1_flat = torch.cat(grad_a1)
    grad_a2_flat = torch.cat(grad_a2)
    grad_b_flat = torch.cat(grad_b) if grad_b else None

    # Cosine similarity between Expert A and Expert B gradients
    cos_sim = F.cosine_similarity(grad_a1_flat.unsqueeze(0), grad_a2_flat.unsqueeze(0)).item()

    # Gradient norms
    norm_a1 = grad_a1_flat.norm().item()
    norm_a2 = grad_a2_flat.norm().item()
    norm_b = grad_b_flat.norm().item() if grad_b_flat is not None else 0.0

    return {
        "cos_sim_a1_a2": cos_sim,
        "norm_a1": norm_a1,
        "norm_a2": norm_a2,
        "norm_b": norm_b,
        "norm_ratio": norm_a1 / max(norm_a2, 1e-8),
        "loss": loss.item(),
        "action_type": action.get("action", "unknown"),
        "route_weights": route_weights,
    }


def analyze_routing(all_results):
    """Analyze learned routing weight distribution."""
    # Collect all route weights across samples
    all_route_w = []
    for r in all_results:
        if r and r["route_weights"]:
            for w in r["route_weights"]:
                all_route_w.append(w.numpy())

    if all_route_w:
        all_w = np.concatenate(all_route_w)
        logger.info(f"\n{'='*60}")
        logger.info("ROUTING WEIGHT DISTRIBUTION (learned r)")
        logger.info(f"  mean={np.mean(all_w):.4f}  std={np.std(all_w):.4f}")
        logger.info(f"  min={np.min(all_w):.4f}  max={np.max(all_w):.4f}")
        logger.info(f"  median={np.median(all_w):.4f}")
        logger.info(f"  % > 0.6 (Expert A dominant): {(all_w > 0.6).mean():.1%}")
        logger.info(f"  % < 0.4 (Expert B dominant): {(all_w < 0.4).mean():.1%}")
        logger.info(f"  % in [0.4, 0.6] (balanced): {((all_w >= 0.4) & (all_w <= 0.6)).mean():.1%}")


def main():
    parser = argparse.ArgumentParser(description="V15 Gradient Interference Diagnostic")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--coop_checkpoint", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output", type=str, default="v15_gui_360/gradient_diagnostic.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, processor = load_model(args)
    steps = load_episodes(args.train_data, args.num_samples)
    logger.info(f"Loaded {len(steps)} steps for diagnosis")

    all_results = []
    by_action_type = defaultdict(list)

    t0 = time.time()
    for i, step_data in enumerate(steps):
        result = collect_gradients(model, processor, step_data, device)
        if result is None:
            continue

        all_results.append(result)
        by_action_type[result["action_type"]].append(result)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            cos_sims = [r["cos_sim_a1_a2"] for r in all_results]
            logger.info(f"[{i+1}/{len(steps)}] {elapsed:.0f}s  "
                        f"mean_cos_sim={np.mean(cos_sims):.4f}  "
                        f"loss={result['loss']:.3f}")

    if not all_results:
        logger.error("No results collected!")
        return

    # === Overall Analysis ===
    cos_sims = [r["cos_sim_a1_a2"] for r in all_results]
    norms_a1 = [r["norm_a1"] for r in all_results]
    norms_a2 = [r["norm_a2"] for r in all_results]
    norm_ratios = [r["norm_ratio"] for r in all_results]

    logger.info(f"\n{'='*60}")
    logger.info("OVERALL GRADIENT ANALYSIS")
    logger.info(f"  Samples: {len(all_results)}")
    logger.info(f"  cos_sim(grad_A1, grad_A2):")
    logger.info(f"    mean={np.mean(cos_sims):.4f}  std={np.std(cos_sims):.4f}")
    logger.info(f"    min={np.min(cos_sims):.4f}  max={np.max(cos_sims):.4f}")
    logger.info(f"    median={np.median(cos_sims):.4f}")
    logger.info(f"  Interpretation:")
    if np.mean(cos_sims) > 0.7:
        logger.info(f"    → Experts receive SIMILAR gradients (weak disentanglement)")
    elif np.mean(cos_sims) > 0.3:
        logger.info(f"    → Experts receive PARTIALLY aligned gradients (moderate disentanglement)")
    elif np.mean(cos_sims) > -0.3:
        logger.info(f"    → Experts receive ORTHOGONAL gradients (strong disentanglement)")
    else:
        logger.info(f"    → Experts receive OPPOSING gradients (interference/competition)")

    logger.info(f"\n  Gradient norms:")
    logger.info(f"    ||grad_A1||: mean={np.mean(norms_a1):.4f}  std={np.std(norms_a1):.4f}")
    logger.info(f"    ||grad_A2||: mean={np.mean(norms_a2):.4f}  std={np.std(norms_a2):.4f}")
    logger.info(f"    ratio A1/A2: mean={np.mean(norm_ratios):.4f}")

    # === By Action Type ===
    logger.info(f"\n{'='*60}")
    logger.info("BY ACTION TYPE")
    for atype, results in sorted(by_action_type.items()):
        cs = [r["cos_sim_a1_a2"] for r in results]
        losses = [r["loss"] for r in results]
        logger.info(f"  {atype:10s} (n={len(results):3d}): "
                     f"cos_sim={np.mean(cs):.4f}±{np.std(cs):.4f}  "
                     f"loss={np.mean(losses):.3f}")

    # === Cross-Action-Type Gradient Similarity ===
    # Do click gradients interfere with swipe gradients?
    logger.info(f"\n{'='*60}")
    logger.info("CROSS-ACTION-TYPE ANALYSIS")
    logger.info("(Do gradients for different action types point in different directions?)")

    action_types = list(by_action_type.keys())
    for i, at1 in enumerate(action_types):
        for at2 in action_types[i+1:]:
            # Average gradient direction for each action type
            # For Expert A
            results1 = by_action_type[at1][:20]  # limit for memory
            results2 = by_action_type[at2][:20]
            if len(results1) < 3 or len(results2) < 3:
                continue
            # Just report the within-type vs cross-type cos_sim difference
            cs1 = [r["cos_sim_a1_a2"] for r in results1]
            cs2 = [r["cos_sim_a1_a2"] for r in results2]
            logger.info(f"  {at1} vs {at2}: "
                         f"cos_sim_A1A2={np.mean(cs1):.4f} vs {np.mean(cs2):.4f}  "
                         f"(Δ={np.mean(cs1)-np.mean(cs2):.4f})")

    # === Routing Analysis ===
    analyze_routing(all_results)

    # === Save results ===
    output_data = {
        "num_samples": len(all_results),
        "overall": {
            "cos_sim_mean": float(np.mean(cos_sims)),
            "cos_sim_std": float(np.std(cos_sims)),
            "cos_sim_median": float(np.median(cos_sims)),
            "grad_norm_a1_mean": float(np.mean(norms_a1)),
            "grad_norm_a2_mean": float(np.mean(norms_a2)),
            "grad_norm_ratio_mean": float(np.mean(norm_ratios)),
        },
        "by_action_type": {
            atype: {
                "count": len(results),
                "cos_sim_mean": float(np.mean([r["cos_sim_a1_a2"] for r in results])),
                "cos_sim_std": float(np.std([r["cos_sim_a1_a2"] for r in results])),
                "loss_mean": float(np.mean([r["loss"] for r in results])),
            }
            for atype, results in by_action_type.items()
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
