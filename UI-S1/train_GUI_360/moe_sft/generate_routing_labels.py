"""
Generate routing labels from a trained ContextAwareRouter for standalone router distillation.

Loads the MoE model (with ContextAwareRouter), runs Pass 1 on training samples,
and records the routing decisions as distillation labels.

Output JSONL format:
    {
        "image_path": "path/to/screenshot.png",
        "instruction_text": "Click on the search button",
        "expert_label": 3,                     # hard routing label (argmax)
        "routing_weights": [0.01, 0.02, ...],  # soft routing distribution
    }

Usage:
    python generate_routing_labels.py \
        --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
        --moe_checkpoint train_GUI_360/moe_sft/output/moe_sft_v3/final \
        --data_file train_GUI_360/llamafactory/data/gui360_train.json \
        --output routing_labels.jsonl \
        --batch_size 4
"""

import argparse
import json
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class GUI360RoutingDataset(Dataset):
    """Dataset for extracting routing labels from GUI-360 training data."""

    def __init__(self, data_file: str, image_dir: str = None):
        with open(data_file) as f:
            self.data = json.load(f)

        self.image_dir = image_dir or os.path.dirname(data_file)
        logger.info(f"Loaded {len(self.data)} samples from {data_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Extract image path and instruction from sharegpt format
        image_path = ""
        instruction_text = ""

        conversations = sample.get("conversations", sample.get("messages", []))
        for turn in conversations:
            if turn.get("role", turn.get("from", "")) in ["user", "human"]:
                content = turn.get("content", turn.get("value", ""))
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "image":
                                image_path = part.get("image", "")
                            elif part.get("type") == "text":
                                instruction_text = part.get("text", "")
                elif isinstance(content, str):
                    instruction_text = content

        # Handle image field at top level
        if not image_path and "images" in sample:
            images = sample["images"]
            if isinstance(images, list) and images:
                image_path = images[0]
            elif isinstance(images, str):
                image_path = images

        return {
            "index": idx,
            "image_path": image_path,
            "instruction_text": instruction_text,
            "raw_sample": sample,
        }


@torch.no_grad()
def generate_labels(
    base_model_path: str,
    moe_checkpoint_path: str,
    data_file: str,
    output_path: str,
    batch_size: int = 1,
    max_samples: int = None,
    device: str = "cuda",
):
    """Generate routing labels by running the trained MoE model."""
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from verl.models.moe.moe_wrapper import MoEVLMWrapper, MoEConfig
    from PIL import Image

    # Load MoE config
    config_path = os.path.join(moe_checkpoint_path, "moe_config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    moe_config = MoEConfig.from_dict(config_dict)

    if moe_config.router_type != 'context_aware':
        logger.warning(
            f"router_type is '{moe_config.router_type}', expected 'context_aware'. "
            "Labels will reflect the loaded router's decisions."
        )

    # Load processor
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

    # Load base model
    logger.info(f"Loading base model from {base_model_path}...")
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Create MoE wrapper and load checkpoint
    moe_wrapper = MoEVLMWrapper(
        base_model=base_model,
        moe_config=moe_config,
        tokenizer=processor.tokenizer,
    )
    moe_wrapper.load_moe_checkpoint(moe_checkpoint_path)
    moe_wrapper = moe_wrapper.to(device=device, dtype=torch.bfloat16)
    moe_wrapper.eval()
    logger.info("Model loaded and ready")

    # Load dataset
    dataset = GUI360RoutingDataset(data_file)
    if max_samples:
        dataset.data = dataset.data[:max_samples]

    # Process samples
    results = []
    for i, sample in enumerate(tqdm(dataset, desc="Generating routing labels")):
        image_path = sample["image_path"]
        instruction_text = sample["instruction_text"]

        if not image_path or not os.path.exists(image_path):
            logger.debug(f"Skipping sample {i}: image not found ({image_path})")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.debug(f"Skipping sample {i}: cannot open image ({e})")
            continue

        # Build conversation for processor
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction_text},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
        except ImportError:
            image_inputs = [image]
            video_inputs = None

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Run Pass 1 to get routing decision
        moe_wrapper._clear_routing_weights()
        base_outputs = moe_wrapper.base_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = base_outputs.hidden_states[-1]

        # Compute routing using the trained router
        if moe_config.router_type == 'context_aware':
            from verl.models.moe.router import create_vision_mask, create_text_context_mask
            vision_mask = create_vision_mask(inputs["input_ids"], processor.tokenizer)
            text_mask = create_text_context_mask(inputs["input_ids"], processor.tokenizer)
            vision_features = moe_wrapper.vision_feature_extractor(hidden_states, vision_mask)
            text_features = moe_wrapper.feature_extractor(hidden_states, text_mask)
            router_output = moe_wrapper.router(vision_features, text_features)
        else:
            from verl.models.moe.router import create_instruction_mask
            instruction_mask = create_instruction_mask(inputs["input_ids"], processor.tokenizer)
            instruction_features = moe_wrapper.feature_extractor(hidden_states, instruction_mask)
            router_output = moe_wrapper.router(instruction_features)

        routing_weights = router_output.routing_weights[0].cpu().tolist()
        expert_label = router_output.top_k_indices[0, 0].item()

        results.append({
            "image_path": image_path,
            "instruction_text": instruction_text,
            "expert_label": expert_label,
            "routing_weights": routing_weights,
        })

        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1} samples, {len(results)} valid labels")

    # Save results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    logger.info(f"Saved {len(results)} routing labels to {output_path}")

    # Print distribution summary
    from collections import Counter
    label_counts = Counter(r["expert_label"] for r in results)
    logger.info(f"Expert distribution: {dict(sorted(label_counts.items()))}")


def main():
    parser = argparse.ArgumentParser(description="Generate routing labels for standalone router distillation")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--moe_checkpoint", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output", type=str, default="routing_labels.jsonl")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    generate_labels(
        base_model_path=args.base_model,
        moe_checkpoint_path=args.moe_checkpoint,
        data_file=args.data_file,
        output_path=args.output,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
