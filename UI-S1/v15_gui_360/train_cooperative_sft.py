#!/usr/bin/env python3
"""V15 Cooperative LoRA SFT — V13 architecture trained with cross-entropy loss.

Uses the same IterativeCooperativeVLMWrapper as V13 RL, but replaces
the RL training loop with standard SFT (cross-entropy on assistant tokens).

Data format: ShareGPT JSONL (same as LLaMA-Factory), one conversation per line:
  {"conversations": [{"from":"human","value":"..."}, {"from":"gpt","value":"..."}],
   "images": ["/path/to/screenshot.png"]}

Usage:
  torchrun --nproc_per_node=4 --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR \
    v15_gui_360/train_cooperative_sft.py \
    --model_path checkpoints/gui_action \
    --train_data v15_gui_360/data/gui360_balanced_train.jsonl \
    --output_dir checkpoints/v15_balanced_cooperative_sft
"""

import argparse
import datetime
import json
import math
import os
import sys
import time
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image

# Set NCCL environment variables BEFORE any NCCL operations
os.environ.setdefault("NCCL_SOCKET_IFNAME", "hsn0")
os.environ.setdefault("GLOO_SOCKET_IFNAME", "hsn0")
os.environ.setdefault("NCCL_NET", "Socket")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_LEVEL", "LOC")
os.environ.setdefault("NCCL_CROSS_NIC", "1")

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("v15_coop_sft")

# Project imports
_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


# -- Dataset ---------------------------------------------------------------

class ShareGPTDataset(Dataset):
    """Load ShareGPT JSONL data for SFT."""

    def __init__(self, jsonl_path: str):
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                # Validate: must have conversations and images
                convs = sample.get("conversations", [])
                images = sample.get("images", [])
                if len(convs) >= 2 and images:
                    # Check image exists
                    if os.path.exists(images[0]):
                        self.samples.append(sample)

        logger.info(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# -- Trainer ---------------------------------------------------------------

class V15CooperativeSFTTrainer:
    def __init__(self, args):
        self.args = args
        self.global_step = 0

        self._setup_distributed()
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()
        self._setup_scheduler()

    # -- distributed -------------------------------------------------------

    def _setup_distributed(self):
        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
        world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))

        torch.cuda.set_device(local_rank)
        self.device = torch.device(f"cuda:{local_rank}")

        if not dist.is_initialized():
            master_addr = os.environ.get("MASTER_ADDR", "localhost")
            master_port = os.environ.get("MASTER_PORT", "29500")
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["RANK"] = str(rank)

            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=world_size,
                rank=rank,
                timeout=datetime.timedelta(seconds=3600),
                device_id=torch.device(f"cuda:{local_rank}"),
            )

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = local_rank

        dist.barrier()
        if self.rank == 0:
            import socket
            logger.info(
                f"Distributed ready: rank={self.rank} world={self.world_size} "
                f"host={socket.gethostname()}"
            )

    # -- model (IterativeCooperativeVLMWrapper) ----------------------------

    def _setup_model(self):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from v13_gui_360.iterative_cooperative_wrapper import IterativeCooperativeVLMWrapper

        args = self.args

        self.processor = AutoProcessor.from_pretrained(args.model_path)
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        if args.image_max_pixels > 0:
            self.processor.image_processor.max_pixels = args.image_max_pixels
        self.pad_id = self.processor.tokenizer.pad_token_id

        if self.rank == 0:
            logger.info(f"Loading base model: {args.model_path}")
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        self.model = IterativeCooperativeVLMWrapper(
            base_model=base_model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=list(args.target_modules),
            balance_weight=args.balance_weight,
            num_comm_rounds=args.num_comm_rounds,
        )

        # Load SFT checkpoint if provided
        if args.sft_checkpoint:
            if self.rank == 0:
                logger.info(f"Loading SFT checkpoint: {args.sft_checkpoint}")
            self.model.load_cooperative(args.sft_checkpoint, device=self.device)

        # Enable gradient checkpointing
        self.model.base_model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable()

        for name, param in self.model.named_parameters():
            if "lora_" in name or "route_weights" in name or "comm_" in name:
                param.requires_grad = True

        # Move to GPU
        self.model = self.model.to(self.device)

        dist.barrier()

        if self.rank == 0:
            counts = self.model.count_trainable_params()
            logger.info(f"Model ready: {counts['total']:,} trainable params "
                        f"(lora={counts['lora']:,}, route={counts['route_weights']:,}, "
                        f"comm={counts['comm']:,})")

    # -- data --------------------------------------------------------------

    def _setup_data(self):
        args = self.args
        self.train_dataset = ShareGPTDataset(args.train_data)
        self.sampler = DistributedSampler(
            self.train_dataset, shuffle=True, drop_last=True
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=1,
            sampler=self.sampler,
            collate_fn=lambda x: x[0],
            num_workers=4,
            pin_memory=True,
        )
        self.val_dataset = None
        if args.val_data and os.path.exists(args.val_data):
            self.val_dataset = ShareGPTDataset(args.val_data)
            if self.rank == 0:
                logger.info(f"Val dataset: {len(self.val_dataset)} samples")

    # -- optimizer (4 groups: lora_decay, lora_nodecay, route, comm) -------

    def _setup_optimizer(self):
        args = self.args

        route_params, comm_params, decay_params, no_decay_params = [], [], [], []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "route_weights" in name:
                route_params.append(param)
            elif "comm_" in name:
                comm_params.append(param)
            elif param.dim() == 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = []
        if decay_params:
            param_groups.append({
                "params": decay_params,
                "lr": args.lora_lr,
                "weight_decay": args.weight_decay,
            })
        if no_decay_params:
            param_groups.append({
                "params": no_decay_params,
                "lr": args.lora_lr,
                "weight_decay": 0.0,
            })
        if route_params:
            param_groups.append({
                "params": route_params,
                "lr": args.route_lr,
                "weight_decay": 0.0,
            })
        if comm_params:
            param_groups.append({
                "params": comm_params,
                "lr": args.route_lr,
                "weight_decay": 0.0,
            })

        self.optimizer = torch.optim.AdamW(param_groups)

        if self.rank == 0:
            logger.info(
                f"Optimizer: lora_lr={args.lora_lr} route_lr={args.route_lr} "
                f"decay={len(decay_params)} no_decay={len(no_decay_params)} "
                f"route={len(route_params)} comm={len(comm_params)}"
            )

    # -- scheduler ---------------------------------------------------------

    def _setup_scheduler(self):
        args = self.args
        steps_per_epoch = len(self.train_loader) // args.gradient_accumulation_steps
        total_steps = steps_per_epoch * args.num_epochs
        warmup_steps = int(total_steps * args.warmup_ratio)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: self._cosine_with_warmup(
                step, warmup_steps, total_steps
            ),
        )
        self.total_steps = total_steps

        if self.rank == 0:
            logger.info(
                f"Scheduler: steps_per_epoch={steps_per_epoch} "
                f"total_steps={total_steps} warmup={warmup_steps}"
            )

    @staticmethod
    def _cosine_with_warmup(step, warmup_steps, total_steps):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    # -- tokenization helpers ----------------------------------------------

    def _tokenize_conversation(self, sample: Dict) -> Optional[Dict]:
        """Tokenize a ShareGPT conversation into input_ids + labels.

        Labels are -100 for prompt tokens (human message), actual token ids
        for assistant tokens (gpt message).
        """
        convs = sample["conversations"]
        images = sample.get("images", [])

        human_text = convs[0]["value"]
        gpt_text = convs[1]["value"]

        # Build messages in Qwen chat format
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": images[0]},
                {"type": "text", "text": human_text.replace("<image>\n", "").replace("<image>", "")},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": gpt_text},
            ]},
        ]

        try:
            # Apply chat template for the full conversation
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            # Also get the prompt-only version to compute prompt length
            prompt_messages = [messages[0]]
            prompt_text = self.processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )

            # Load image
            image = Image.open(images[0]).convert("RGB")

            # Tokenize full conversation
            full_inputs = self.processor(
                text=[text], images=[image], return_tensors="pt", padding=False
            )

            # Tokenize prompt only
            prompt_inputs = self.processor(
                text=[prompt_text], images=[image], return_tensors="pt", padding=False
            )

            input_ids = full_inputs["input_ids"][0]
            prompt_len = prompt_inputs["input_ids"].shape[1]

            # Create labels: -100 for prompt, actual ids for response
            labels = input_ids.clone()
            labels[:prompt_len] = -100

            result = {
                "input_ids": input_ids.unsqueeze(0),
                "attention_mask": torch.ones(1, len(input_ids), dtype=torch.long),
                "labels": labels.unsqueeze(0),
            }

            # Add vision inputs
            for k in ("pixel_values", "image_grid_thw"):
                if k in full_inputs:
                    result[k] = full_inputs[k]

            return result

        except Exception as e:
            if self.rank == 0:
                logger.warning(f"Tokenization failed: {e}")
            return None

    # -- training step -----------------------------------------------------

    def train_step(self, batch_inputs: List[Dict]) -> Dict:
        """Perform one optimizer step over accumulated batch."""
        args = self.args
        self.optimizer.zero_grad()

        total_loss = 0.0
        total_balance_loss = 0.0
        total_tokens = 0
        n_samples = 0

        for inputs in batch_inputs:
            # Move to device
            device_inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model.forward(**device_inputs)
                loss = outputs.loss

                # Balance loss
                bal_loss, mean_w = self.model.compute_balance_loss()
                combined_loss = (loss + args.balance_weight * bal_loss) / len(batch_inputs)

            combined_loss.backward()

            n_tokens = (device_inputs["labels"] != -100).sum().item()
            total_loss += loss.item()
            total_balance_loss += bal_loss.item()
            total_tokens += n_tokens
            n_samples += 1

            del outputs, loss, combined_loss, device_inputs

        # All-reduce gradients across ranks
        if self.world_size > 1:
            for p in self.model.parameters():
                if p.requires_grad:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p.data)
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        # Clip & step
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)

        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1

        return {
            "loss": total_loss / max(n_samples, 1),
            "balance_loss": total_balance_loss / max(n_samples, 1),
            "grad_norm": grad_norm.item()
            if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
            "n_tokens": total_tokens,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    # -- checkpoint --------------------------------------------------------

    def save_checkpoint(self, tag: str):
        if self.rank != 0:
            return

        ckpt_dir = os.path.join(self.args.output_dir, tag)
        coop_dir = os.path.join(ckpt_dir, "cooperative")
        self.model.save_cooperative(coop_dir)

        torch.save(
            {"global_step": self.global_step},
            os.path.join(ckpt_dir, "training_state.pt"),
        )
        logger.info(f"Saved checkpoint: {ckpt_dir}")

    # -- validation --------------------------------------------------------

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_dataset is None or self.rank != 0:
            return {}

        self.model.eval()
        total_loss = 0.0
        n_samples = 0

        for i in range(min(len(self.val_dataset), 200)):
            sample = self.val_dataset[i]
            inputs = self._tokenize_conversation(sample)
            if inputs is None:
                continue

            device_inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model.forward(**device_inputs)
                total_loss += outputs.loss.item()
                n_samples += 1

            del outputs, device_inputs

        self.model.train()

        metrics = {
            "val/loss": total_loss / max(n_samples, 1),
            "val/n_samples": n_samples,
        }

        if self.rank == 0:
            logger.info(f"Validation: loss={metrics['val/loss']:.4f} n={n_samples}")

        return metrics

    # -- main loop ---------------------------------------------------------

    def train(self):
        args = self.args

        if self.rank == 0:
            logger.info("=" * 60)
            logger.info("V15 Cooperative SFT (V13 architecture + cross-entropy)")
            logger.info(f"  lora_r={args.lora_r} num_comm_rounds={args.num_comm_rounds}")
            logger.info(f"  balance_weight={args.balance_weight}")
            logger.info(f"  lora_lr={args.lora_lr} route_lr={args.route_lr}")
            logger.info(f"  grad_accum={args.gradient_accumulation_steps}")
            logger.info(f"  epochs={args.num_epochs}")
            logger.info(f"  world_size={self.world_size}")
            logger.info(f"  train_samples={len(self.train_dataset)}")
            logger.info(f"  total_steps={self.total_steps}")
            logger.info("=" * 60)

        self.model.train()

        for epoch in range(args.num_epochs):
            self.sampler.set_epoch(epoch)

            epoch_loss = []
            batch_inputs: List[Dict] = []
            skipped = 0
            t_epoch = time.time()

            for sample_idx, sample in enumerate(self.train_loader):
                # Tokenize
                inputs = self._tokenize_conversation(sample)
                if inputs is None:
                    skipped += 1
                    continue

                # Check sequence length
                seq_len = inputs["input_ids"].shape[1]
                if seq_len > args.cutoff_len:
                    skipped += 1
                    continue

                batch_inputs.append(inputs)

                # Accumulate and step
                if len(batch_inputs) >= args.gradient_accumulation_steps:
                    metrics = self.train_step(batch_inputs)
                    epoch_loss.append(metrics["loss"])

                    if self.rank == 0 and self.global_step % args.logging_steps == 0:
                        logger.info(
                            f"E{epoch} S{self.global_step}/{self.total_steps} "
                            f"loss={metrics['loss']:.4f} "
                            f"bal={metrics['balance_loss']:.4f} "
                            f"gnorm={metrics['grad_norm']:.2f} "
                            f"lr={metrics['lr']:.2e} "
                            f"tok={metrics['n_tokens']}"
                        )

                    if (self.rank == 0 and args.save_steps > 0
                            and self.global_step % args.save_steps == 0):
                        self.save_checkpoint(
                            f"epoch-{epoch}_step-{self.global_step}"
                        )

                    # Validation
                    if (args.val_steps > 0
                            and self.global_step % args.val_steps == 0):
                        if self.rank == 0:
                            self.validate()
                        if self.world_size > 1:
                            dist.barrier()

                    batch_inputs = []
                    torch.cuda.empty_cache()

            # Handle remaining samples
            if batch_inputs:
                metrics = self.train_step(batch_inputs)
                epoch_loss.append(metrics["loss"])
                batch_inputs = []

            # Epoch summary
            if self.rank == 0:
                dur = time.time() - t_epoch
                avg_loss = np.mean(epoch_loss) if epoch_loss else 0.0
                logger.info(f"{'='*60}")
                logger.info(
                    f"Epoch {epoch} done in {dur/60:.1f}min  "
                    f"avg_loss={avg_loss:.4f}  skipped={skipped}"
                )
                logger.info(f"{'='*60}")

                self.save_checkpoint(f"epoch-{epoch}")
                self.validate()

            if self.world_size > 1:
                dist.barrier()

        if self.rank == 0:
            logger.info("Training complete!")


# -- CLI -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="V15 Cooperative SFT (V13 architecture + cross-entropy)")

    # Model
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sft_checkpoint", type=str, default="",
                        help="Path to pre-trained cooperative checkpoint dir")

    # Data
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, default="")
    parser.add_argument("--cutoff_len", type=int, default=8192)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj"])

    # V13 specific
    parser.add_argument("--num_comm_rounds", type=int, default=2)
    parser.add_argument("--balance_weight", type=float, default=0.01)
    parser.add_argument("--image_max_pixels", type=int, default=1003520)

    # Optimizer
    parser.add_argument("--lora_lr", type=float, default=1e-5)
    parser.add_argument("--route_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    # Training
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--val_steps", type=int, default=50)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    trainer = V15CooperativeSFTTrainer(args)
    trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
