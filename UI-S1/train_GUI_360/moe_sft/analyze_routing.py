"""Analyze MoE v2 routing weight distribution on eval data."""
import torch
import json
import os
import sys
import numpy as np
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoProcessor, AutoModelForVision2Seq
from verl.models.moe.moe_wrapper import MoEVLMWrapper, MoEConfig
from verl.models.moe.router import create_instruction_mask_from_text


def main():
    moe_checkpoint = os.path.join(PROJECT_ROOT, "train_GUI_360/moe_sft/output/moe_sft_v2/final")
    base_model_path = os.path.join(PROJECT_ROOT, "checkpoints/Qwen2.5-VL-7B-Instruct")

    # Load config
    with open(os.path.join(moe_checkpoint, "moe_config.json")) as f:
        config_dict = json.load(f)
    moe_config = MoEConfig(**config_dict)
    print(f"MoE config: {moe_config.num_experts} experts, top_k={moe_config.top_k}")

    # Load processor
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

    # Load base model
    print("Loading base model...")
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    # Create MoE wrapper + load checkpoint
    moe_wrapper = MoEVLMWrapper(
        base_model=base_model, moe_config=moe_config, tokenizer=processor.tokenizer
    )
    moe_wrapper.load_moe_checkpoint(moe_checkpoint)
    moe_wrapper = moe_wrapper.to(device="cuda", dtype=torch.bfloat16)
    moe_wrapper.eval()
    print("Model loaded")

    # Load val samples from training data (has conversations format)
    val_data_path = os.path.join(PROJECT_ROOT, "train_GUI_360/llamafactory/data/gui360_val.json")
    with open(val_data_path) as f:
        val_data = json.load(f)
    print(f"Loaded {len(val_data)} validation samples")

    # Extract instruction text from conversations
    instruction_texts = []
    for sample in val_data:
        human_msg = sample["conversations"][0]["value"]
        # Remove the image tag and get the instruction text
        text = human_msg.replace("<image>\n", "").strip()
        instruction_texts.append(text)

    # Run routing on each sample using forward pass (Pass 1 only)
    all_routing_weights = []
    for i, instruction in enumerate(instruction_texts):
        if i % 20 == 0:
            print(f"Processing {i}/{len(instruction_texts)}...")

        # Format as chat messages for tokenization
        messages = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], padding=True, return_tensors="pt")
        inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            # Pass 1: get hidden states from base model (no LoRA)
            moe_wrapper._clear_routing_weights()
            base_outputs = moe_wrapper.base_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                output_hidden_states=True,
                return_dict=True,
            )

            # Get last hidden state
            if hasattr(base_outputs, 'hidden_states') and base_outputs.hidden_states:
                hidden_states = base_outputs.hidden_states[-1]
            else:
                hidden_states = base_outputs.last_hidden_state

            # Create instruction mask
            instruction_mask = create_instruction_mask_from_text(
                inputs["input_ids"], processor.tokenizer, [instruction]
            )

            # Route
            instruction_features = moe_wrapper.feature_extractor(hidden_states, instruction_mask)
            router_output = moe_wrapper.router(instruction_features)
            weights = router_output.routing_weights[0].float().cpu().numpy()
            all_routing_weights.append(weights)

    all_routing_weights = np.array(all_routing_weights)  # [N, 6]

    print(f"\n{'='*70}")
    print(f"Per-Sample Routing Analysis ({len(all_routing_weights)} samples)")
    print(f"{'='*70}")

    # Per-expert stats
    print("\nPer-expert weight statistics:")
    print(f"{'Expert':>8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Median':>8}")
    for e in range(6):
        w = all_routing_weights[:, e]
        print(f"{'E'+str(e):>8} {w.mean():>8.4f} {w.std():>8.4f} {w.min():>8.4f} {w.max():>8.4f} {np.median(w):>8.4f}")

    # Max weight analysis (how concentrated is each sample?)
    max_weights = all_routing_weights.max(axis=1)
    print(f"\nMax weight per sample:")
    print(f"  mean={max_weights.mean():.4f}  median={np.median(max_weights):.4f}")
    print(f"  min={max_weights.min():.4f}  max={max_weights.max():.4f}")

    for t in [0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        count = (max_weights > t).sum()
        print(f"  max_weight > {t}: {count}/{len(max_weights)} ({100*count/len(max_weights):.1f}%)")

    # Top-2 cumulative weight
    sorted_weights = np.sort(all_routing_weights, axis=1)[:, ::-1]  # descending
    top2_sum = sorted_weights[:, :2].sum(axis=1)
    top3_sum = sorted_weights[:, :3].sum(axis=1)
    print(f"\nTop-K cumulative weight:")
    print(f"  Top-1: mean={sorted_weights[:, 0].mean():.4f}  median={np.median(sorted_weights[:, 0]):.4f}")
    print(f"  Top-2: mean={top2_sum.mean():.4f}  median={np.median(top2_sum):.4f}")
    print(f"  Top-3: mean={top3_sum.mean():.4f}  median={np.median(top3_sum):.4f}")

    # Entropy
    entropy = -(all_routing_weights * np.log(all_routing_weights + 1e-10)).sum(axis=1)
    max_entropy = np.log(6)
    print(f"\nRouting entropy:")
    print(f"  mean={entropy.mean():.4f}  std={entropy.std():.4f}  (max={max_entropy:.4f})")
    print(f"  ratio: {entropy.mean()/max_entropy:.4f}")

    # Dominant expert distribution
    dominant = all_routing_weights.argmax(axis=1)
    print(f"\nDominant expert distribution:")
    counter = Counter(dominant)
    for e in range(6):
        count = counter.get(e, 0)
        print(f"  Expert {e}: {count}/{len(dominant)} ({100*count/len(dominant):.1f}%)")

    # Show some sample routing vectors
    print(f"\nSample routing vectors (first 20):")
    for i in range(min(20, len(all_routing_weights))):
        w = all_routing_weights[i]
        w_str = " ".join([f"{v:.3f}" for v in w])
        inst_preview = instruction_texts[i][:80].replace("\n", " ")
        print(f"  [{w_str}] dom=E{w.argmax()} | {inst_preview}")

    # Save full results
    output_path = os.path.join(PROJECT_ROOT, "train_GUI_360/moe_sft/output/moe_sft_v2/routing_analysis.json")
    results = {
        "num_samples": len(all_routing_weights),
        "per_expert_mean": all_routing_weights.mean(axis=0).tolist(),
        "per_expert_std": all_routing_weights.std(axis=0).tolist(),
        "max_weight_mean": float(max_weights.mean()),
        "max_weight_median": float(np.median(max_weights)),
        "entropy_mean": float(entropy.mean()),
        "entropy_ratio": float(entropy.mean() / max_entropy),
        "dominant_expert_distribution": {str(k): int(v) for k, v in counter.items()},
        "top1_mean": float(sorted_weights[:, 0].mean()),
        "top2_mean": float(top2_sum.mean()),
        "top3_mean": float(top3_sum.mean()),
        "all_routing_weights": all_routing_weights.tolist(),
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
