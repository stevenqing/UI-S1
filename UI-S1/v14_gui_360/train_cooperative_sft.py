#!/usr/bin/env python3
"""V14 Cooperative LoRA SFT — Phase 1: Grow B₁/B₂ with CE loss.

Trains V14 cooperative LoRA (separate B matrices) with standard SFT
on GUI-360 trajectory data. This bootstraps B₁/B₂ to meaningful magnitudes
so that RL (phase 2) can differentiate routing.

Problem solved: RL alone can't bootstrap B from zero because delta≈0
gives no useful gradient signal → B stays zero → routing can't differentiate.
SFT's CE loss provides much stronger gradients.

Usage:
  torchrun --nproc_per_node=4 --nnodes=$SLURM_NNODES \\
      v14_gui_360/train_cooperative_sft.py \\
      --model_path checkpoints/gui_action \\
      --train_data v12_gui_360/data/gui360_train_2000_balanced.jsonl \\
      --output_dir checkpoints/v14_sft \\
      --lora_r 128 --lora_alpha 256 --num_epochs 3
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
import torch.nn.functional as F
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
logger = logging.getLogger("v14_sft")

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


# -- Prompt templates (same as RL) -----------------------------------------

USER_PROMPT_TEMPLATE = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

First, explain your reasoning process—describe how you analyze the screenshot, understand the current state, and determine what action should be taken next based on the instruction and previous actions.

Then output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished, you can output status as "FINISH" and take no action. Like:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time. If the instruction could apply to multiple elements, choose the most relevant one based on the context provided by the screenshot and previous actions.
"""

SUPPORTED_ACTIONS = """<action>
- click
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to click at.
    - button: str, One of 'left', 'right', 'middle' or 'x' (Default: 'left')
    - double: bool, Whether to perform a double click (Default: False)
    - pressed: str|None, Keyboard key to press while clicking (Default: None)
  - Example: click(coordinate=[100, 100], button='left', double=False, pressed=None)
- type
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to type at.
    - keys: str, The key to input.
    - clear_current_text: bool, Whether to clear the current text (Default: False)
    - control_focus: bool, Whether to focus on selected control before typing (Default: True)
  - Example: type(coordinate=[100, 100], keys='Hello')
- drag
  - Args:
    - start_coordinate: [x, y], where the drag starts.
    - end_coordinate: [x, y], where the drag ends.
    - button: str, 'left' or 'right' (Default: 'left')
    - duration: float, Duration in seconds (Default: 1.0)
  - Example: drag(start_coordinate=[100, 100], end_coordinate=[200, 200])
- wheel_mouse_input
  - Args:
    - coordinate: [x, y], position on the screen to scroll.
    - wheel_dist: int, Wheel notches. Positive=up, negative=down.
  - Example: wheel_mouse_input(coordinate=[100, 100], wheel_dist=-5)
</action>"""


# -- GT action formatting --------------------------------------------------

def format_gt_action_for_history(gt_action: Dict, step_id: int) -> str:
    """Convert GT action dict to GUI-360 history format."""
    atype = gt_action.get("action", "")
    coord = gt_action.get("coordinate")

    if atype == "click":
        if coord:
            return f"Step {step_id}: click(coordinate=[{int(coord[0])}, {int(coord[1])}])"
        return f"Step {step_id}: click()"
    elif atype == "type":
        text = gt_action.get("text", "")
        if coord and text:
            t = text[:30] + "..." if len(text) > 30 else text
            return f"Step {step_id}: type(coordinate=[{int(coord[0])}, {int(coord[1])}], keys='{t}')"
        elif text:
            t = text[:30] + "..." if len(text) > 30 else text
            return f"Step {step_id}: type(keys='{t}')"
        elif coord:
            return f"Step {step_id}: type(coordinate=[{int(coord[0])}, {int(coord[1])}])"
        return f"Step {step_id}: type()"
    elif atype in ("swipe", "drag"):
        start = gt_action.get("coordinate")
        end = gt_action.get("endCoordinate")
        if start and end:
            return (f"Step {step_id}: drag(start_coordinate=[{int(start[0])}, {int(start[1])}], "
                    f"end_coordinate=[{int(end[0])}, {int(end[1])}])")
        return f"Step {step_id}: drag()"
    else:
        return f"Step {step_id}: {atype}()"


def format_gt_action_as_tool_call(gt_action: Dict, is_last_step: bool = False) -> str:
    """Convert GT action dict to the <tool_call> output format for SFT target."""
    atype = gt_action.get("action", "")
    coord = gt_action.get("coordinate")
    status = "FINISH" if is_last_step else "CONTINUE"

    args = {}
    if atype == "click":
        if coord:
            args["coordinate"] = [int(coord[0]), int(coord[1])]
    elif atype == "type":
        if coord:
            args["coordinate"] = [int(coord[0]), int(coord[1])]
        text = gt_action.get("text", "")
        if text:
            args["keys"] = text
    elif atype in ("swipe", "drag"):
        start = gt_action.get("coordinate")
        end = gt_action.get("endCoordinate")
        if start:
            args["start_coordinate"] = [int(start[0]), int(start[1])]
        if end:
            args["end_coordinate"] = [int(end[0]), int(end[1])]
    elif atype == "wheel_mouse_input":
        if coord:
            args["coordinate"] = [int(coord[0]), int(coord[1])]
        wd = gt_action.get("wheel_dist", gt_action.get("wheelDist", -3))
        args["wheel_dist"] = wd

    tool_call = {
        "function": atype,
        "args": args,
        "status": status,
    }
    return "<tool_call>\n" + json.dumps(tool_call, indent=2) + "\n</tool_call>"


# -- Dataset ---------------------------------------------------------------

class TrajectoryStepDataset(Dataset):
    """Flatten episode trajectories into individual (step, context) samples for SFT.

    Each sample = (screenshot_t, prompt with history 0..t-1, GT action_t).
    """

    def __init__(self, jsonl_path: str, processor, max_episodes: int = 0,
                 image_max_pixels: int = 602112):
        self.processor = processor
        if image_max_pixels > 0:
            self.processor.image_processor.max_pixels = image_max_pixels

        # Load episodes
        episodes = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line)
                valid = True
                for step in ep.get("steps", []):
                    if not os.path.exists(step.get("screenshot", "")):
                        valid = False
                        break
                if valid and ep.get("steps"):
                    episodes.append(ep)

        if 0 < max_episodes < len(episodes):
            rng = np.random.RandomState(42)
            idx = rng.choice(len(episodes), max_episodes, replace=False)
            episodes = [episodes[i] for i in sorted(idx)]

        # Flatten to steps
        self.samples = []
        for ep in episodes:
            goal = ep["goal"]
            steps = ep["steps"]
            action_history = []
            for t, step in enumerate(steps):
                gt_action = step["action"]
                is_last = (t == len(steps) - 1)
                self.samples.append({
                    "goal": goal,
                    "screenshot": step["screenshot"],
                    "image_w": step.get("image_w", 1920),
                    "image_h": step.get("image_h", 1080),
                    "gt_action": gt_action,
                    "action_history": list(action_history),
                    "is_last_step": is_last,
                })
                action_history.append(
                    format_gt_action_for_history(gt_action, t + 1)
                )

        logger.info(f"Loaded {len(episodes)} episodes → {len(self.samples)} steps from {jsonl_path}")

    # Qwen2.5-VL token IDs
    IM_START = 151644
    IM_END = 151645
    ASSISTANT_HEADER = [151644, 77091, 198]  # <|im_start|>assistant\n

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        for attempt in range(3):
            try:
                return self._process_item((idx + attempt) % len(self.samples))
            except Exception as e:
                if attempt == 0:
                    logger.warning(f"Dataset item {idx} failed: {e}")
        return None

    def _process_item(self, idx):
        sample = self.samples[idx]

        # Build prompt
        history_text = "\n".join(sample["action_history"]) if sample["action_history"] else "None"
        prompt_text = USER_PROMPT_TEMPLATE.format(
            instruction=sample["goal"],
            history=history_text,
            actions=SUPPORTED_ACTIONS,
        )

        # GT target
        target_text = format_gt_action_as_tool_call(
            sample["gt_action"], sample["is_last_step"]
        )

        # Build messages (same format as RL)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["screenshot"]},
                    {"type": "text", "text": prompt_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_text}],
            },
        ]

        # Full text (prompt + response)
        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # Prompt only (for label masking)
        prompt_messages = [messages[0]]
        prompt_text_only = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        # Load image
        image = Image.open(sample["screenshot"]).convert("RGB")

        # Tokenize
        full_inputs = self.processor(
            text=[full_text], images=[image],
            return_tensors="pt", padding=False,
        )
        prompt_inputs = self.processor(
            text=[prompt_text_only], images=[image],
            return_tensors="pt", padding=False,
        )

        input_ids = full_inputs["input_ids"].squeeze(0)
        attention_mask = full_inputs["attention_mask"].squeeze(0)
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # Labels: mask prompt, keep only response tokens
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if "pixel_values" in full_inputs:
            result["pixel_values"] = full_inputs["pixel_values"].squeeze(0)
        if "image_grid_thw" in full_inputs:
            thw = full_inputs["image_grid_thw"]
            if thw.dim() == 3:
                thw = thw.squeeze(0)
            if thw.dim() == 1:
                thw = thw.unsqueeze(0)
            result["image_grid_thw"] = thw

        return result


def collate_fn(batch):
    """Collate with padding and None filtering."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    max_len = max(b["input_ids"].shape[0] for b in batch)

    input_ids = []
    attention_mask = []
    labels_list = []
    pixel_values_list = []
    image_grid_thw_list = []

    for b in batch:
        seq_len = b["input_ids"].shape[0]
        pad_len = max_len - seq_len
        input_ids.append(F.pad(b["input_ids"], (0, pad_len), value=0))
        attention_mask.append(F.pad(b["attention_mask"], (0, pad_len), value=0))
        labels_list.append(F.pad(b["labels"], (0, pad_len), value=-100))
        if "pixel_values" in b:
            pixel_values_list.append(b["pixel_values"])
        if "image_grid_thw" in b:
            image_grid_thw_list.append(b["image_grid_thw"])

    result = {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels_list),
    }
    if pixel_values_list:
        result["pixel_values"] = torch.cat(pixel_values_list, dim=0)
    if image_grid_thw_list:
        result["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)

    return result


# -- Trainer ---------------------------------------------------------------

class V14SFTTrainer:
    def __init__(self, args):
        self.args = args
        self.global_step = 0

        self._setup_distributed()
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()

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
            logger.info(f"Distributed ready: rank={self.rank} world={self.world_size} "
                        f"host={socket.gethostname()}")

    def _setup_model(self):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from v14_gui_360.cooperative_wrapper import IterativeCooperativeVLMWrapper

        args = self.args

        self.processor = AutoProcessor.from_pretrained(args.model_path)
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        if args.image_max_pixels > 0:
            self.processor.image_processor.max_pixels = args.image_max_pixels

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

        # Enable gradient checkpointing
        self.model.base_model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable()

        for name, param in self.model.named_parameters():
            if "lora_" in name or "route_weights" in name or "comm_" in name:
                param.requires_grad = True

        self.model = self.model.to(self.device)
        dist.barrier()

        if self.rank == 0:
            counts = self.model.count_trainable_params()
            logger.info(f"Model ready: {counts['total']:,} trainable params "
                        f"(lora={counts['lora']:,}, route={counts['route_weights']:,}, "
                        f"comm={counts['comm']:,})")

    def _setup_data(self):
        args = self.args
        self.train_dataset = TrajectoryStepDataset(
            args.train_data, self.processor,
            max_episodes=args.max_episodes,
            image_max_pixels=args.image_max_pixels,
        )
        self.sampler = DistributedSampler(
            self.train_dataset, shuffle=True, drop_last=True
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            sampler=self.sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        if self.rank == 0:
            steps_per_epoch = len(self.train_loader) // args.gradient_accumulation_steps
            logger.info(f"Train: {len(self.train_dataset)} samples, "
                        f"batch_size={args.batch_size}, "
                        f"grad_accum={args.gradient_accumulation_steps}, "
                        f"steps/epoch={steps_per_epoch}")

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

        # Cosine LR scheduler
        total_steps = (len(self.train_loader) // args.gradient_accumulation_steps) * args.num_epochs
        warmup_steps = min(100, total_steps // 10)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[g["lr"] for g in param_groups],
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy="cos",
        )
        self.total_steps = total_steps

        if self.rank == 0:
            logger.info(f"Optimizer: lora_lr={args.lora_lr} route_lr={args.route_lr} "
                        f"total_steps={total_steps} warmup={warmup_steps}")

    def _all_reduce_gradients(self):
        """Manual gradient all-reduce (no DDP wrapper)."""
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)

    def train(self):
        args = self.args
        self.model.train()
        scaler = torch.amp.GradScaler("cuda")

        for epoch in range(args.num_epochs):
            self.sampler.set_epoch(epoch)
            self.optimizer.zero_grad()

            accum_loss = 0.0
            accum_tokens = 0
            accum_count = 0
            epoch_loss = 0.0
            epoch_steps = 0
            t_epoch = time.time()

            for batch_idx, batch in enumerate(self.train_loader):
                if batch is None:
                    continue

                # Move to device
                inputs = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Forward with AMP
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.model(**inputs)
                    loss = outputs.loss

                    # Balance loss
                    balance_loss, mean_w = self.model.compute_balance_loss()
                    loss = loss + args.balance_weight * balance_loss

                # Scale for gradient accumulation
                loss_scaled = loss / args.gradient_accumulation_steps
                scaler.scale(loss_scaled).backward()

                # Track
                n_tokens = (inputs["labels"] != -100).sum().item()
                accum_loss += loss.item()
                accum_tokens += n_tokens
                accum_count += 1

                # Gradient step
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    scaler.unscale_(self.optimizer)
                    self._all_reduce_gradients()

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        args.max_grad_norm,
                    )

                    scaler.step(self.optimizer)
                    scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1
                    epoch_steps += 1
                    avg_loss = accum_loss / max(accum_count, 1)
                    epoch_loss += avg_loss

                    if self.rank == 0 and self.global_step % args.logging_steps == 0:
                        lr = self.optimizer.param_groups[0]["lr"]
                        logger.info(
                            f"ep={epoch} step={self.global_step}/{self.total_steps} "
                            f"loss={avg_loss:.4f} gnorm={grad_norm:.3f} "
                            f"lr={lr:.2e} rw={mean_w:.4f} "
                            f"tok={accum_tokens}"
                        )

                    accum_loss = 0.0
                    accum_tokens = 0
                    accum_count = 0

                    # Save checkpoint
                    if args.save_steps > 0 and self.global_step % args.save_steps == 0:
                        self._save_checkpoint(f"step-{self.global_step}")

            # End of epoch
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            if self.rank == 0:
                logger.info(f"Epoch {epoch} done: avg_loss={avg_epoch_loss:.4f} "
                            f"steps={epoch_steps} time={time.time() - t_epoch:.0f}s")

            self._save_checkpoint(f"epoch-{epoch}")

            # Log B matrix norms
            if self.rank == 0:
                self._log_b_norms()

        if self.rank == 0:
            logger.info("SFT training complete!")

    def _log_b_norms(self):
        """Log B₁ and B₂ norms per layer to verify they're growing."""
        b1_norms = {}
        b2_norms = {}
        rw_vals = {}

        for name, param in self.model.named_parameters():
            if "lora_B_1" in name:
                # Extract layer index from name
                parts = name.split(".")
                for i, p in enumerate(parts):
                    if p == "layers" and i + 1 < len(parts):
                        layer_idx = int(parts[i + 1])
                        b1_norms.setdefault(layer_idx, []).append(
                            param.data.float().norm().item()
                        )
                        break
            elif "lora_B_2" in name:
                parts = name.split(".")
                for i, p in enumerate(parts):
                    if p == "layers" and i + 1 < len(parts):
                        layer_idx = int(parts[i + 1])
                        b2_norms.setdefault(layer_idx, []).append(
                            param.data.float().norm().item()
                        )
                        break

        for i, param in enumerate(self.model.route_weights):
            rw_vals[i] = torch.sigmoid(param.data.float().mean()).item()

        if b1_norms:
            sample_layers = [0, 7, 14, 21, 27]
            for l in sample_layers:
                if l in b1_norms:
                    b1 = sum(b1_norms[l]) / len(b1_norms[l])
                    b2 = sum(b2_norms.get(l, [0])) / max(len(b2_norms.get(l, [0])), 1)
                    rw = rw_vals.get(l, 0.5)
                    logger.info(f"  Layer {l:2d}: B1_norm={b1:.4f} B2_norm={b2:.4f} rw={rw:.4f}")

    def _save_checkpoint(self, tag: str):
        if self.rank != 0:
            dist.barrier()
            return

        save_dir = os.path.join(self.args.output_dir, tag, "cooperative")
        self.model.save_cooperative(save_dir)

        # Save training state
        state = {
            "global_step": self.global_step,
            "tag": tag,
        }
        torch.save(state, os.path.join(self.args.output_dir, tag, "training_state.pt"))
        logger.info(f"Saved checkpoint: {save_dir}")
        dist.barrier()


def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--num_comm_rounds", type=int, default=2)
    parser.add_argument("--balance_weight", type=float, default=0.01)

    # Data
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--image_max_pixels", type=int, default=602112)

    # Training
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lora_lr", type=float, default=2e-5)
    parser.add_argument("--route_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=0,
                        help="Save every N steps (0=only at epoch end)")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    trainer = V14SFTTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
