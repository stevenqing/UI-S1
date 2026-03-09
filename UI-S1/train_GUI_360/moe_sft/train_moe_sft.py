"""
MoE SFT Training Script.

Trains Qwen2.5-VL-7B-Instruct (frozen) + TextOnlyRouter + 6×ExpertLoRA(r=16, q_proj+v_proj).
top_k=6: all experts participate, router learns weighted combination.
Loss = LM_loss only (no balance loss since all experts are always active).

Uses module replacement (MoELoRALinear) instead of hooks for correct gradient flow.
DDP works naturally since LoRA params are proper children in the module tree.

Usage:
    torchrun --nproc_per_node=4 train_moe_sft.py --config moe_sft_config.yaml
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from verl.models.moe.moe_wrapper import MoEVLMWrapper, MoEConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GUI360MoESFTDataset(Dataset):
    """Dataset for GUI-360 ShareGPT format data with Qwen2.5-VL processor."""

    def __init__(
        self,
        data_path: str,
        processor: AutoProcessor,
        max_seq_length: int = 4096,
        image_min_pixels: int = 256,
        image_max_pixels: int = 1003520,
    ):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels

        # Pre-compute assistant token pattern for label masking
        self._assistant_tokens = self.processor.tokenizer.encode(
            "assistant\n", add_special_tokens=False
        )
        self._im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self._im_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        conversations = sample["conversations"]
        image_paths = sample.get("images", [])

        # Build Qwen2.5-VL chat messages
        messages = []
        for conv in conversations:
            role = "user" if conv["from"] == "human" else "assistant"
            content_parts = []
            text = conv["value"]

            if role == "user" and image_paths:
                # Replace <image> placeholder with actual image content entries
                parts = text.split("<image>")
                for i, part in enumerate(parts):
                    if i < len(image_paths):
                        content_parts.append({
                            "type": "image",
                            "image": image_paths[i],
                            "min_pixels": self.image_min_pixels,
                            "max_pixels": self.image_max_pixels,
                        })
                    if part.strip():
                        content_parts.append({"type": "text", "text": part.strip()})
            else:
                content_parts.append({"type": "text", "text": text})

            messages.append({"role": role, "content": content_parts})

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Load images
        images = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")
                images.append(Image.new("RGB", (224, 224), (128, 128, 128)))

        # Process with Qwen2.5-VL processor
        inputs = self.processor(
            text=[text],
            images=images if images else None,
            padding=False,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        # Squeeze batch dim (collator will re-batch)
        result = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.squeeze(0)
            elif isinstance(v, list) and len(v) == 1:
                result[k] = v[0]
            else:
                result[k] = v

        # Create labels: mask everything except assistant response tokens
        result["labels"] = self._create_labels(result["input_ids"])

        return result

    def _create_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Mask non-assistant tokens with -100 for loss computation."""
        labels = input_ids.clone()
        ids_list = input_ids.tolist()
        assistant_tokens = self._assistant_tokens
        assistant_len = len(assistant_tokens)

        in_assistant = False
        i = 0
        while i < len(ids_list):
            if ids_list[i] == self._im_start_id:
                end = i + 1 + assistant_len
                if end <= len(ids_list) and ids_list[i + 1 : end] == assistant_tokens:
                    # Mask the header: <|im_start|>assistant\n
                    labels[i:end] = -100
                    in_assistant = True
                    i = end
                    continue
                else:
                    in_assistant = False

            if ids_list[i] == self._im_end_id:
                if in_assistant:
                    # Keep <|im_end|> as part of target (model should learn to stop)
                    pass
                else:
                    labels[i] = -100
                in_assistant = False
                i += 1
                continue

            if not in_assistant:
                labels[i] = -100
            i += 1

        return labels


@dataclass
class MoESFTCollator:
    """Data collator that pads to longest in batch."""

    processor: AutoProcessor
    pad_token_id: int = 151643  # Qwen2.5 pad token

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}

        # Pad input_ids, labels, attention_mask
        input_ids_list = [f["input_ids"] for f in features]
        labels_list = [f["labels"] for f in features]
        attention_mask_list = [
            f.get("attention_mask", torch.ones_like(f["input_ids"])) for f in features
        ]

        max_len = max(ids.size(0) for ids in input_ids_list)

        padded_input_ids = []
        padded_labels = []
        padded_attention = []

        for ids, labs, mask in zip(input_ids_list, labels_list, attention_mask_list):
            pad_len = max_len - ids.size(0)
            padded_input_ids.append(
                torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype)])
            )
            padded_labels.append(
                torch.cat([labs, torch.full((pad_len,), -100, dtype=labs.dtype)])
            )
            padded_attention.append(
                torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
            )

        batch["input_ids"] = torch.stack(padded_input_ids)
        batch["labels"] = torch.stack(padded_labels)
        batch["attention_mask"] = torch.stack(padded_attention)

        # Handle pixel_values and image_grid_thw
        if "pixel_values" in features[0]:
            all_pixel_values = []
            all_grid_thw = []
            for f in features:
                pv = f["pixel_values"]
                if pv.dim() == 1:
                    pv = pv.unsqueeze(0)
                all_pixel_values.append(pv)
                if "image_grid_thw" in f:
                    gt = f["image_grid_thw"]
                    if gt.dim() == 1:
                        gt = gt.unsqueeze(0)
                    all_grid_thw.append(gt)

            batch["pixel_values"] = torch.cat(all_pixel_values, dim=0)
            if all_grid_thw:
                batch["image_grid_thw"] = torch.cat(all_grid_thw, dim=0)

        return batch


# ---------------------------------------------------------------------------
# Custom Trainer
# ---------------------------------------------------------------------------

class MoESFTTrainer(Trainer):
    """
    Custom trainer for MoE SFT.

    With module replacement (MoELoRALinear), DDP works naturally since LoRA params
    are proper children in the module tree. No need for DDP unwrapping or manual
    gradient allreduce.

    The only customization needed:
    1. compute_loss: Forward through MoEVLMWrapper, log routing stats
    2. _save: Save only router + expert LoRAs (not the frozen 7B base)
    """

    def __init__(self, moe_wrapper: MoEVLMWrapper, **kwargs):
        super().__init__(model=moe_wrapper, **kwargs)
        self.moe_wrapper = moe_wrapper

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Forward through MoE wrapper and compute LM loss."""
        labels = inputs.pop("labels", None)
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")

        # Forward through MoE wrapper
        moe_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            return_routing_info=True,
        )

        loss = moe_output.loss
        if loss is None:
            # Fallback: compute LM loss manually
            logits = moe_output.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # Log routing stats periodically
        if (
            self.state.global_step > 0
            and self.state.global_step % self.args.logging_steps == 0
            and moe_output.routing_weights is not None
        ):
            rw = moe_output.routing_weights.detach()
            log_dict = {}
            for i in range(rw.size(1)):
                log_dict[f"routing/expert_{i}_weight"] = rw[:, i].mean().item()
            log_dict["routing/entropy"] = (
                -(rw * torch.log(rw + 1e-10)).sum(dim=-1).mean().item()
            )
            log_dict["routing/max_weight"] = rw.max(dim=-1).values.mean().item()
            log_dict["routing/min_weight"] = rw.min(dim=-1).values.mean().item()
            if moe_output.lm_loss is not None:
                log_dict["loss/lm_loss"] = moe_output.lm_loss.item()
            if moe_output.balance_loss is not None:
                log_dict["loss/balance_loss"] = moe_output.balance_loss.item()
            if moe_output.z_loss is not None:
                log_dict["loss/z_loss"] = moe_output.z_loss.item()

            self.log(log_dict)

        return (loss, moe_output) if return_outputs else loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Save only router + expert LoRAs (not the frozen base model)."""
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save MoE components only
        self.moe_wrapper.save_moe_checkpoint(output_dir)

        # Save training args
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        logger.info(f"Saved MoE checkpoint to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="MoE SFT Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--moe_init_checkpoint", type=str, default=None,
                        help="Path to MoE checkpoint to initialize from (e.g., copy-init checkpoint)")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    config = load_config(args.config)

    # Resolve paths relative to project root
    project_root = PROJECT_ROOT
    model_path = str(project_root / config["model_name_or_path"])
    data_path = str(project_root / config["data"]["train_file"])
    output_dir = str(project_root / config["training"]["output_dir"])
    ds_config = config.get("deepspeed")
    if ds_config:
        ds_config = str(project_root / ds_config)
        try:
            import deepspeed  # noqa: F401
        except ImportError:
            logger.warning("DeepSpeed not available. Falling back to DDP (no ZeRO).")
            ds_config = None

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(f"Loading model from {model_path}")
    logger.info(f"Training data: {data_path}")

    # ---- Load processor ----
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        min_pixels=config["data"].get("image_min_pixels", 256),
        max_pixels=config["data"].get("image_max_pixels", 1003520),
    )

    # ---- Load base model ----
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=getattr(torch, config.get("torch_dtype", "bfloat16")),
        attn_implementation=config.get("attn_implementation", "flash_attention_2"),
        trust_remote_code=True,
    )

    # ---- Create MoE config ----
    moe_cfg = config["moe"]
    moe_config = MoEConfig(
        num_experts=moe_cfg["num_experts"],
        top_k=moe_cfg["top_k"],
        expert_lora_r=moe_cfg["expert_lora_r"],
        expert_lora_alpha=moe_cfg["expert_lora_alpha"],
        expert_lora_dropout=moe_cfg.get("expert_lora_dropout", 0.05),
        target_modules=moe_cfg["target_modules"],
        router_hidden=moe_cfg.get("router_hidden", 256),
        router_dropout=moe_cfg.get("router_dropout", 0.1),
        router_temperature=moe_cfg.get("router_temperature", 1.0),
        pooling_strategy=moe_cfg.get("pooling_strategy", "mean"),
        use_vectorized_routing=moe_cfg.get("use_vectorized_routing", True),
        router_type=moe_cfg.get("router_type", "text_only"),
        balance_weight=moe_cfg.get("balance_weight", 0.0),
        balance_type=moe_cfg.get("balance_type", "mse"),
        z_loss_weight=moe_cfg.get("z_loss_weight", 0.0),
    )

    # ---- Create MoE wrapper (module replacement happens here) ----
    moe_wrapper = MoEVLMWrapper(
        base_model=base_model,
        moe_config=moe_config,
        tokenizer=processor.tokenizer,
    )

    # ---- Load MoE init checkpoint if provided ----
    if args.moe_init_checkpoint:
        init_path = args.moe_init_checkpoint
        if not os.path.isabs(init_path):
            init_path = str(project_root / init_path)
        logger.info(f"Loading MoE init checkpoint from {init_path}")
        moe_wrapper.load_moe_checkpoint(init_path)
        logger.info("MoE init checkpoint loaded successfully")

    # Log parameter counts
    trainable = moe_wrapper.num_trainable_parameters()
    total = moe_wrapper.num_total_parameters()
    logger.info(f"Trainable parameters: {trainable:,} ({trainable / 1e6:.1f}M)")
    logger.info(f"Total parameters: {total:,} ({total / 1e6:.1f}M)")
    logger.info(f"Trainable ratio: {trainable / total * 100:.2f}%")
    logger.info(f"MoELoRALinear modules: {len(moe_wrapper._moe_linear_modules)}")

    # ---- Create dataset ----
    train_dataset = GUI360MoESFTDataset(
        data_path=data_path,
        processor=processor,
        max_seq_length=config["data"].get("max_seq_length", 4096),
        image_min_pixels=config["data"].get("image_min_pixels", 256),
        image_max_pixels=config["data"].get("image_max_pixels", 1003520),
    )
    logger.info(f"Training samples: {len(train_dataset)}")

    # ---- Training arguments ----
    train_cfg = config["training"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        max_grad_norm=train_cfg.get("max_grad_norm", 0.5),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=False,
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 500),
        save_total_limit=train_cfg.get("save_total_limit", 5),
        seed=train_cfg.get("seed", 42),
        report_to=train_cfg.get("report_to", "wandb"),
        run_name=train_cfg.get("run_name", "moe_sft"),
        deepspeed=ds_config,
        remove_unused_columns=False,
        # All trainable params (LoRA + router) participate in every forward, so no
        # unused parameters. Confirmed by DDP warning in job 2610726.
        ddp_find_unused_parameters=False,
    )

    # ---- Collator ----
    collator = MoESFTCollator(
        processor=processor,
        pad_token_id=processor.tokenizer.pad_token_id or 151643,
    )

    # ---- Trainer ----
    trainer = MoESFTTrainer(
        moe_wrapper=moe_wrapper,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=processor.tokenizer,
    )

    # ---- Resume from checkpoint if available ----
    last_checkpoint = None
    if os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is not None:
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")

    # ---- Train ----
    logger.info("Starting MoE SFT training (module replacement)...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # ---- Save final checkpoint ----
    trainer.save_state()
    moe_wrapper.save_moe_checkpoint(os.path.join(output_dir, "final"))
    logger.info(f"Training complete. Final checkpoint saved to {output_dir}/final")


if __name__ == "__main__":
    main()
