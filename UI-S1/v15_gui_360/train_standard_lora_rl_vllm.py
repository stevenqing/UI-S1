#!/usr/bin/env python3
"""Standard LoRA RL with vLLM acceleration — V15 baseline.

Same RL algorithm as v12_gui_360/train_standard_lora_rl.py but replaces
HF model.generate() with vLLM for ~4-5x generation speedup.

Architecture (per GPU, DDP with 4 GPUs on 1 node):
  - Training model: base (14GB bf16) + LoRA (~0.5GB) — for backward pass
  - vLLM engine: base (14GB) + LoRA adapter + KV cache — for fast generation
  - Total: ~40-45GB on 80GB GPU

Flow per training step:
  1. Sync LoRA weights -> vLLM (save PEFT format, create LoRARequest)
  2. Generate with vLLM: batch all steps of an episode, SamplingParams(n=K)
  3. Compute old log probs with training model (HF forward pass)
  4. Compute rewards, advantages (same as original)
  5. Train: PPO backward pass on LoRA only, DDP gradient all-reduce
  6. Optimizer step -> sync LoRA -> repeat
"""

import os
import sys

# Set CUDA_VISIBLE_DEVICES BEFORE importing torch/vllm so each DDP rank
# sees only its own GPU.  vLLM's LLM(..., tensor_parallel_size=1) will then
# use that single visible GPU for its engine.
_local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
os.environ["CUDA_VISIBLE_DEVICES"] = str(_local_rank)
os.environ["VLLM_NO_USAGE_STATS"] = "1"

os.environ.setdefault("NCCL_SOCKET_IFNAME", "hsn0")
os.environ.setdefault("GLOO_SOCKET_IFNAME", "hsn0")
os.environ.setdefault("NCCL_NET", "Socket")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_LEVEL", "LOC")
os.environ.setdefault("NCCL_CROSS_NIC", "1")

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import argparse
import datetime
import json
import logging
import time
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from safetensors.torch import save_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("standard_lora_rl_vllm")

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from v12_gui_360.reward import (
    compute_step_reward,
    compute_trajectory_advantages,
    compute_grpo_advantages,
    parse_action_from_text,
)


# -- Prompt templates (identical to v12) ------------------------------------

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


# -- Dataset ----------------------------------------------------------------

class EpisodeDataset(Dataset):
    def __init__(self, jsonl_path: str, max_episodes: int = 0):
        self.episodes = []
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
                    self.episodes.append(ep)

        if 0 < max_episodes < len(self.episodes):
            rng = np.random.RandomState(42)
            idx = rng.choice(len(self.episodes), max_episodes, replace=False)
            self.episodes = [self.episodes[i] for i in sorted(idx)]

        logger.info(f"Loaded {len(self.episodes)} episodes from {jsonl_path}")

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx]


# -- Prompt formatting ------------------------------------------------------

def format_gt_action_for_history(gt_action: Dict, step_id: int) -> str:
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


def build_step_messages(goal: str, action_history: List[str],
                        image_path: str) -> list:
    history_text = "\n".join(action_history) if action_history else "None"
    prompt_text = USER_PROMPT_TEMPLATE.format(
        instruction=goal,
        history=history_text,
        actions=SUPPORTED_ACTIONS,
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]


# -- GRPO loss helpers -------------------------------------------------------

def compute_policy_loss(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_range: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ratio = torch.exp(log_probs - old_log_probs)
    clip_low = 1.0 - clip_range
    clip_high = 1.0 + clip_range
    clipped_ratio = torch.clamp(ratio, clip_low, clip_high)

    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    pg_loss = torch.max(pg_loss1, pg_loss2)

    valid_tokens = response_mask.sum().clamp(min=1)
    pg_loss = (pg_loss * response_mask).sum() / valid_tokens

    clip_frac = ((ratio < clip_low) | (ratio > clip_high)).float()
    clip_frac = (clip_frac * response_mask).sum() / valid_tokens

    approx_kl = (old_log_probs - log_probs) * response_mask
    approx_kl = approx_kl.sum() / valid_tokens

    return pg_loss, clip_frac, approx_kl


def compute_kl_penalty(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    log_ratio = log_probs - ref_log_probs
    kl = (torch.exp(log_ratio) - 1 - log_ratio) * response_mask
    valid_tokens = response_mask.sum().clamp(min=1)
    return kl.sum() / valid_tokens


# -- Trainer -----------------------------------------------------------------

class StandardLoRATrainerVLLM:
    def __init__(self, args):
        self.args = args
        self.global_step = 0
        self.resume_step = 0
        self._lora_sync_version = 0

        self._setup_distributed()
        self._setup_model()
        self._load_resume_checkpoint()
        self._setup_vllm()
        self._setup_data()
        self._setup_optimizer()

    # -- distributed ---------------------------------------------------------

    def _setup_distributed(self):
        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
        world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))

        # cuda:0 is the only visible device (CUDA_VISIBLE_DEVICES was set)
        torch.cuda.set_device(0)
        self.device = torch.device("cuda:0")

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
                device_id=torch.device("cuda:0"),
            )

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = _local_rank  # physical GPU id (before CUDA_VISIBLE_DEVICES remap)

        dist.barrier()
        if self.rank == 0:
            import socket
            logger.info(
                f"Distributed ready: rank={self.rank} world={self.world_size} "
                f"host={socket.gethostname()} local_rank={self.local_rank}"
            )

    # -- model ---------------------------------------------------------------

    def _setup_model(self):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from v12_gui_360.standard_lora_wrapper import StandardLoRAWrapper

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

        self.model = StandardLoRAWrapper(
            base_model=base_model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=list(args.target_modules),
        )

        if args.sft_checkpoint:
            if self.rank == 0:
                logger.info(f"Loading checkpoint: {args.sft_checkpoint}")
            self.model.load_lora(args.sft_checkpoint, device=self.device)

        self.model.base_model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable()

        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        self.model = self.model.to(self.device)
        dist.barrier()

        if self.rank == 0:
            counts = self.model.count_trainable_params()
            logger.info(f"Model ready: {counts['total']:,} trainable params "
                        f"(lora={counts['lora']:,})")

    def _load_resume_checkpoint(self):
        ckpt_dir = getattr(self.args, "resume_from", "")
        if not ckpt_dir:
            return
        if self.rank == 0:
            logger.info(f"Resuming from checkpoint: {ckpt_dir}")

        lora_dir = os.path.join(ckpt_dir, "lora")
        if os.path.isdir(lora_dir):
            self.model.load_lora(lora_dir, device=self.device)

        state_path = os.path.join(ckpt_dir, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            self.global_step = state.get("global_step", 0)
            self.resume_step = self.global_step
            if self.rank == 0:
                logger.info(f"  Resumed at global_step={self.global_step}")
        dist.barrier()

    # -- vLLM engine ---------------------------------------------------------

    def _setup_vllm(self):
        args = self.args

        # Sync directory for PEFT-format adapter (one per rank)
        self._vllm_sync_dir = os.path.join(
            args.output_dir, f"_vllm_lora_sync_rank{self.rank}"
        )
        os.makedirs(self._vllm_sync_dir, exist_ok=True)

        # Save current LoRA weights in PEFT format
        self._save_peft_adapter(self._vllm_sync_dir)

        if self.rank == 0:
            logger.info("Initializing vLLM engine ...")

        self.vllm_engine = LLM(
            model=args.model_path,
            enable_lora=True,
            max_lora_rank=args.lora_r,
            max_loras=2,
            tensor_parallel_size=1,
            gpu_memory_utilization=args.vllm_gpu_mem,
            max_model_len=4096,
            max_num_batched_tokens=32768,
            enforce_eager=True,
            dtype="bfloat16",
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 1},
        )

        self.lora_request = LoRARequest(
            lora_name=f"lora_v{self._lora_sync_version}",
            lora_int_id=self._lora_sync_version + 1,
            lora_local_path=self._vllm_sync_dir,
        )

        if self.rank == 0:
            logger.info("vLLM engine ready")

    def _save_peft_adapter(self, save_dir: str):
        """Convert StandardLoRA weights to PEFT format for vLLM."""
        peft_state = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad or "lora_" not in name:
                continue
            # StandardLoRA key: base_model.model.layers.X.{attn/mlp}.{proj}.lora_{A,B}
            # PEFT key:         base_model.model.model.layers.X.{attn/mlp}.{proj}.lora_{A,B}.weight
            peft_key = name.replace(
                "base_model.model.", "base_model.model.model.", 1
            ) + ".weight"
            peft_state[peft_key] = param.data.cpu().contiguous()

        save_file(peft_state, os.path.join(save_dir, "adapter_model.safetensors"))

        config = {
            "peft_type": "LORA",
            "base_model_name_or_path": self.args.model_path,
            "r": self.args.lora_r,
            "lora_alpha": self.args.lora_alpha,
            "lora_dropout": 0.0,
            "target_modules": list(self.args.target_modules),
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "inference_mode": True,
        }
        with open(os.path.join(save_dir, "adapter_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def _sync_lora_to_vllm(self):
        """Sync training LoRA weights to vLLM after optimizer step."""
        self._lora_sync_version += 1
        self._save_peft_adapter(self._vllm_sync_dir)
        self.lora_request = LoRARequest(
            lora_name=f"lora_v{self._lora_sync_version}",
            lora_int_id=self._lora_sync_version + 1,
            lora_local_path=self._vllm_sync_dir,
        )

    # -- tokenization helpers ------------------------------------------------

    def _tokenize_for_logprobs(self, messages: list, image: Image.Image) -> dict:
        """HF-tokenize prompt (with image) for log-prob computation."""
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt", padding=False
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _compute_token_log_probs(
        self,
        full_ids: torch.Tensor,
        prompt_len: int,
        inputs_for_fwd: dict,
        with_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        ids = full_ids.unsqueeze(0) if full_ids.dim() == 1 else full_ids
        attn = torch.ones_like(ids)

        fwd_kwargs = {"input_ids": ids, "attention_mask": attn}
        for k in ("pixel_values", "image_grid_thw"):
            if k in inputs_for_fwd:
                fwd_kwargs[k] = inputs_for_fwd[k]

        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model.forward(**fwd_kwargs)

            logits = outputs.logits
            resp_logits = logits[:, prompt_len - 1:-1, :]
            resp_labels = ids[:, prompt_len:]
            log_p = F.log_softmax(resp_logits, dim=-1)
            tok_lp = torch.gather(
                log_p, -1, resp_labels.unsqueeze(-1)
            ).squeeze(-1)
            mask = (resp_labels != self.pad_id).float()

        return tok_lp.squeeze(0), mask.squeeze(0), mask.sum().item()

    def _compute_ref_log_probs(
        self,
        full_ids: torch.Tensor,
        prompt_len: int,
        inputs_for_fwd: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = full_ids.unsqueeze(0) if full_ids.dim() == 1 else full_ids
        attn = torch.ones_like(ids)

        fwd_kwargs = {"input_ids": ids, "attention_mask": attn}
        for k in ("pixel_values", "image_grid_thw"):
            if k in inputs_for_fwd:
                fwd_kwargs[k] = inputs_for_fwd[k]

        self.model.disable_lora()
        self.model.eval()

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model.base_model(**fwd_kwargs)

            logits = outputs.logits
            resp_logits = logits[:, prompt_len - 1:-1, :]
            resp_labels = ids[:, prompt_len:]
            log_p = F.log_softmax(resp_logits, dim=-1)
            tok_lp = torch.gather(
                log_p, -1, resp_labels.unsqueeze(-1)
            ).squeeze(-1)
            mask = (resp_labels != self.pad_id).float()

        self.model.enable_lora()
        self.model.train()

        return tok_lp.squeeze(0), mask.squeeze(0)

    # -- episode rollout (vLLM batched) --------------------------------------

    @torch.no_grad()
    def generate_episode_rollouts(self, episode: Dict) -> Optional[Dict]:
        args = self.args
        K = args.num_samples
        goal = episode["goal"]
        steps = episode["steps"]
        num_steps = len(steps)

        if self.rank == 0:
            logger.info(f"[rollout] goal={goal[:60]}... steps={num_steps}")

        # Build prompts for all steps (GT action history, known a priori)
        prompts_data = []
        action_history: List[str] = []
        for si, step in enumerate(steps):
            try:
                image = Image.open(step["screenshot"]).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to open {step['screenshot']}: {e}")
                return None

            messages = build_step_messages(goal, action_history, step["screenshot"])
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts_data.append({
                "text": text,
                "image": image,
                "messages": messages,
                "step": step,
            })
            action_history.append(format_gt_action_for_history(step["action"], si + 1))

        # Batch all steps into one vLLM call
        vllm_prompts = [
            {"prompt": pd["text"], "multi_modal_data": {"image": pd["image"]}}
            for pd in prompts_data
        ]
        sp = SamplingParams(
            n=K,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            stop=["</tool_call>", "</action>"],
            include_stop_str_in_output=True,
        )

        t_gen = time.time()
        try:
            vllm_outputs = self.vllm_engine.generate(
                vllm_prompts, sp, lora_request=self.lora_request
            )
        except Exception as e:
            logger.warning(f"vLLM generation failed: {e}")
            return None

        if self.rank == 0:
            logger.info(
                f"  vLLM gen {time.time()-t_gen:.1f}s "
                f"({num_steps} steps x K={K})"
            )

        # Process vLLM outputs: rewards, log probs
        all_step_data = []
        all_rewards = np.zeros((K, num_steps), dtype=np.float32)

        for si, (pd, vllm_out) in enumerate(zip(prompts_data, vllm_outputs)):
            step = pd["step"]
            gt_action = step["action"]
            image_w = step.get("image_w", 1040)
            image_h = step.get("image_h", 736)

            # HF tokenize prompt for log-prob computation
            inputs = self._tokenize_for_logprobs(pd["messages"], pd["image"])
            prompt_len = inputs["input_ids"].shape[1]

            step_rewards = []
            step_texts = []
            truncated = []
            output_ids_list = []

            for k in range(K):
                completion = vllm_out.outputs[k]
                resp_text = completion.text
                step_texts.append(resp_text)

                is_trunc = (completion.finish_reason == "length")
                truncated.append(is_trunc)

                # Build full_ids: HF prompt tokens + vLLM response tokens
                resp_token_ids = torch.tensor(
                    list(completion.token_ids),
                    dtype=torch.long, device=self.device,
                )
                full_ids = torch.cat([
                    inputs["input_ids"].squeeze(0),
                    resp_token_ids,
                ])
                output_ids_list.append(full_ids)

                if is_trunc:
                    step_rewards.append(0.0)
                    all_rewards[k, si] = 0.0
                else:
                    reward, _ = compute_step_reward(
                        resp_text, gt_action, image_w, image_h,
                        w_format=args.w_format,
                        w_type=args.w_type,
                        w_content=args.w_content,
                    )
                    step_rewards.append(reward)
                    all_rewards[k, si] = reward

            n_trunc = sum(truncated)
            if self.rank == 0 and n_trunc > 0:
                logger.info(f"    step {si}: truncated {n_trunc}/{K}")

            # Compute old log probs (no grad) for non-truncated samples
            old_tok_lps = [None] * K
            masks = [None] * K
            ref_tok_lps = [None] * K
            for k in range(K):
                if truncated[k]:
                    continue
                tok_lp, mask, _ = self._compute_token_log_probs(
                    output_ids_list[k], prompt_len, inputs, with_grad=False
                )
                old_tok_lps[k] = tok_lp.detach()
                masks[k] = mask.detach()

                if args.kl_coef > 0:
                    ref_lp, _ = self._compute_ref_log_probs(
                        output_ids_list[k], prompt_len, inputs
                    )
                    ref_tok_lps[k] = ref_lp.detach()

            all_step_data.append({
                "step_idx": si,
                "output_ids": output_ids_list,
                "prompt_len": prompt_len,
                "inputs": inputs,
                "old_tok_lps": old_tok_lps,
                "masks": masks,
                "ref_tok_lps": ref_tok_lps,
                "truncated": truncated,
                "step_rewards": step_rewards,
                "step_texts": step_texts,
                "gt_action": gt_action,
                "image_w": image_w,
                "image_h": image_h,
            })

        # Compute advantages
        if getattr(args, 'advantage_mode', 'sp') == 'grpo':
            result = compute_grpo_advantages(
                all_rewards,
                dapo_threshold=args.dapo_threshold,
                match_threshold=args.match_threshold,
            )
        else:
            result = compute_trajectory_advantages(
                all_rewards,
                match_threshold=args.match_threshold,
                spwa_decay=args.spwa_decay,
                dapo_threshold=args.dapo_threshold,
                step_adv_weight=args.step_adv_weight,
            )

        if result is None:
            return None

        advantages, sp_scores, first_errors = result

        return {
            "episode_id": episode["episode_id"],
            "goal": goal,
            "num_steps": num_steps,
            "step_data": all_step_data,
            "rewards": all_rewards,
            "advantages": advantages,
            "sp_scores": sp_scores,
            "first_errors": first_errors,
        }

    # -- policy gradient update ----------------------------------------------

    def train_step(self, batch_rollouts: List[Dict]) -> Dict[str, float]:
        args = self.args
        K = args.num_samples

        self.optimizer.zero_grad()

        total_pg_loss = 0.0
        total_kl = 0.0
        total_clip_frac = 0.0
        n_seqs = 0
        n_zero_adv = 0
        n_total = 0
        all_rewards = []
        all_sp = []
        advs_abs = []

        total_seqs = sum(ep["num_steps"] * K for ep in batch_rollouts)
        total_seqs = max(total_seqs, 1)

        for ep in batch_rollouts:
            advantages = ep["advantages"]
            all_sp.extend(ep["sp_scores"].tolist())

            for si, step_data in enumerate(ep["step_data"]):
                for k in range(K):
                    n_total += 1
                    adv = advantages[k, si]
                    all_rewards.append(ep["rewards"][k, si])

                    if step_data.get("truncated", [False] * K)[k]:
                        n_zero_adv += 1
                        continue

                    if abs(adv) < 1e-8:
                        n_zero_adv += 1
                        continue

                    advs_abs.append(abs(adv))

                    tok_lp, mask, _ = self._compute_token_log_probs(
                        step_data["output_ids"][k],
                        step_data["prompt_len"],
                        step_data["inputs"],
                        with_grad=True,
                    )

                    old_tok_lp = step_data["old_tok_lps"][k]
                    adv_expanded = torch.full_like(mask, float(adv))

                    pg_loss, clip_frac, approx_kl = compute_policy_loss(
                        old_tok_lp, tok_lp, adv_expanded, mask, args.clip_range
                    )

                    kl_loss = torch.tensor(0.0, device=self.device)
                    if args.kl_coef > 0 and step_data["ref_tok_lps"][k] is not None:
                        ref_lp = step_data["ref_tok_lps"][k]
                        kl_loss = compute_kl_penalty(tok_lp, ref_lp, mask)

                    loss = (pg_loss + args.kl_coef * kl_loss) / total_seqs
                    loss.backward()

                    total_pg_loss += pg_loss.item()
                    total_kl += kl_loss.item()
                    total_clip_frac += clip_frac.item()
                    n_seqs += 1

                    del tok_lp, loss, pg_loss, kl_loss

        # All-reduce gradients across DDP ranks
        if self.world_size > 1:
            for p in self.model.parameters():
                if p.requires_grad:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p.data)
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        self.optimizer.step()
        self.global_step += 1

        return {
            "pg_loss": total_pg_loss / max(n_seqs, 1),
            "kl": total_kl / max(n_seqs, 1),
            "clip_frac": total_clip_frac / max(n_seqs, 1),
            "grad_norm": grad_norm.item()
            if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
            "mean_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
            "mean_sp": float(np.mean(all_sp)) if all_sp else 0.0,
            "nonzero_adv_frac": (n_total - n_zero_adv) / max(n_total, 1),
            "mean_abs_adv": float(np.mean(advs_abs)) if advs_abs else 0.0,
            "n_seqs": n_seqs,
        }

    # -- checkpoint ----------------------------------------------------------

    def save_checkpoint(self, tag: str):
        if self.rank != 0:
            return

        ckpt_dir = os.path.join(self.args.output_dir, tag)
        lora_dir = os.path.join(ckpt_dir, "lora")
        self.model.save_lora(lora_dir)

        torch.save(
            {"global_step": self.global_step},
            os.path.join(ckpt_dir, "training_state.pt"),
        )
        logger.info(f"Saved checkpoint: {ckpt_dir}")

    # -- validation (vLLM) ---------------------------------------------------

    @torch.no_grad()
    def validate(self, tag: str) -> Dict[str, float]:
        if self.val_dataset is None or self.rank != 0:
            return {}

        logger.info(f"Running validation ({tag})...")
        all_rewards = []
        per_ep_results = []

        for ep_idx in range(min(len(self.val_dataset), 20)):
            episode = self.val_dataset[ep_idx]
            goal = episode["goal"]
            steps = episode["steps"]
            action_history: List[str] = []
            ep_rewards = []

            # Build prompts for all steps
            vllm_prompts = []
            step_metas = []
            for si, step in enumerate(steps):
                try:
                    image = Image.open(step["screenshot"]).convert("RGB")
                except Exception:
                    action_history.append(
                        format_gt_action_for_history(step["action"], si + 1)
                    )
                    continue

                messages = build_step_messages(goal, action_history, step["screenshot"])
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                vllm_prompts.append({
                    "prompt": text,
                    "multi_modal_data": {"image": image},
                })
                step_metas.append({
                    "gt_action": step["action"],
                    "image_w": step.get("image_w", 1040),
                    "image_h": step.get("image_h", 736),
                })
                action_history.append(
                    format_gt_action_for_history(step["action"], si + 1)
                )

            if not vllm_prompts:
                continue

            # Greedy generation via vLLM
            sp = SamplingParams(
                temperature=0,
                n=1,
                max_tokens=self.args.max_new_tokens,
                stop=["</tool_call>", "</action>"],
                include_stop_str_in_output=True,
            )
            outputs = self.vllm_engine.generate(
                vllm_prompts, sp, lora_request=self.lora_request
            )

            for out, meta in zip(outputs, step_metas):
                text = out.outputs[0].text
                reward, _ = compute_step_reward(
                    text, meta["gt_action"], meta["image_w"], meta["image_h"]
                )
                ep_rewards.append(reward)
                all_rewards.append(reward)

            per_ep_results.append({
                "episode_id": episode["episode_id"],
                "goal": goal[:100],
                "num_steps": len(steps),
                "mean_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
            })

        metrics = {
            "val/mean_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
            "val/n_episodes": len(per_ep_results),
            "val/n_steps": len(all_rewards),
        }

        val_dir = os.path.join(self.args.output_dir, "val_results")
        os.makedirs(val_dir, exist_ok=True)
        result_path = os.path.join(val_dir, f"{tag}.jsonl")
        with open(result_path, "w") as f:
            f.write(json.dumps({"_summary": metrics, "_step": self.global_step}) + "\n")
            for r in per_ep_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        logger.info(f"Validation {tag}:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        return metrics

    # -- data ----------------------------------------------------------------

    def _setup_data(self):
        args = self.args
        self.train_dataset = EpisodeDataset(
            args.train_data, max_episodes=args.max_episodes
        )
        self.sampler = DistributedSampler(
            self.train_dataset, shuffle=True, drop_last=True
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=1,
            sampler=self.sampler,
            collate_fn=lambda x: x[0],
            num_workers=2,
            pin_memory=True,
        )
        self.val_dataset = None
        if args.val_data and os.path.exists(args.val_data):
            self.val_dataset = EpisodeDataset(args.val_data)
            if self.rank == 0:
                logger.info(f"Val dataset: {len(self.val_dataset)} episodes")

    def _setup_optimizer(self):
        args = self.args

        decay_params, no_decay_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() == 1 or name.endswith(".bias"):
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

        self.optimizer = torch.optim.AdamW(param_groups)

        if self.rank == 0:
            logger.info(
                f"Optimizer: lora_lr={args.lora_lr} "
                f"decay={len(decay_params)} no_decay={len(no_decay_params)}"
            )

    # -- main loop -----------------------------------------------------------

    def train(self):
        args = self.args

        if self.rank == 0:
            logger.info("=" * 60)
            logger.info("Standard LoRA RL + vLLM (SP+GiGPO+SPWA)")
            logger.info(f"  K={args.num_samples}  temp={args.temperature}")
            logger.info(f"  lora_r={args.lora_r}  lora_alpha={args.lora_alpha}")
            logger.info(f"  spwa_decay={args.spwa_decay}  match_threshold={args.match_threshold}  step_adv_weight={args.step_adv_weight}")
            logger.info(f"  grad_accum={args.gradient_accumulation_steps}")
            logger.info(f"  epochs={args.num_epochs}  kl_coef={args.kl_coef}")
            logger.info(f"  lora_lr={args.lora_lr}")
            logger.info(f"  world_size={self.world_size}")
            logger.info(f"  episodes={len(self.train_dataset)}")
            logger.info(f"  dapo_threshold={args.dapo_threshold}")
            logger.info(f"  vllm_gpu_mem={args.vllm_gpu_mem}")
            if self.resume_step > 0:
                logger.info(f"  RESUMING from step {self.resume_step}")
            logger.info("=" * 60)

        for epoch in range(args.num_epochs):
            self.sampler.set_epoch(epoch)

            epoch_metrics = defaultdict(list)
            batch_rollouts: List[Dict] = []
            skipped = 0

            t_epoch = time.time()

            skip_episodes = self.resume_step * args.gradient_accumulation_steps
            if self.rank == 0 and skip_episodes > 0:
                logger.info(f"  Skipping first {skip_episodes} episodes (resume)")

            for ep_idx, episode in enumerate(self.train_loader):
                if ep_idx < skip_episodes:
                    continue

                step_at_boundary = (
                    (ep_idx + 1) % args.gradient_accumulation_steps == 0
                )

                t_ep = time.time()

                try:
                    rollout = self.generate_episode_rollouts(episode)
                except Exception as e:
                    if self.rank == 0:
                        logger.warning(
                            f"Episode {ep_idx} failed: {e}\n"
                            f"{traceback.format_exc()}"
                        )
                    rollout = None

                if rollout is not None:
                    batch_rollouts.append(rollout)
                else:
                    skipped += 1
                    if self.rank == 0:
                        logger.info(f"  ep {ep_idx} skipped (DAPO filtered or failed)")

                if step_at_boundary:
                    if self.rank == 0:
                        logger.info(f"  train_step: {len(batch_rollouts)} rollouts, ep_idx={ep_idx}")
                    metrics = self.train_step(batch_rollouts)
                    if self.rank == 0:
                        logger.info(f"  train_step done: global_step={self.global_step}")

                    # Sync updated LoRA weights to vLLM
                    self._sync_lora_to_vllm()

                    if batch_rollouts:
                        for k, v in metrics.items():
                            if isinstance(v, (int, float)):
                                epoch_metrics[k].append(v)

                        if self.rank == 0 and self.global_step % args.logging_steps == 0:
                            logger.info(
                                f"E{epoch} S{self.global_step} "
                                f"loss={metrics['pg_loss']:.4f} "
                                f"r={metrics['mean_reward']:.3f} "
                                f"sp={metrics['mean_sp']:.3f} "
                                f"kl={metrics['kl']:.4f} "
                                f"gnorm={metrics['grad_norm']:.2f} "
                                f"nz={metrics['nonzero_adv_frac']:.0%} "
                                f"adv={metrics['mean_abs_adv']:.3f} "
                                f"t={time.time()-t_ep:.1f}s"
                            )

                        if (self.rank == 0 and args.save_steps > 0
                                and self.global_step % args.save_steps == 0):
                            self.save_checkpoint(
                                f"epoch-{epoch}_step-{self.global_step}"
                            )

                    if (args.val_steps > 0
                            and self.global_step % args.val_steps == 0):
                        if self.rank == 0:
                            self.validate(tag=f"epoch-{epoch}_step-{self.global_step}")
                        if self.world_size > 1:
                            dist.barrier()

                    batch_rollouts = []
                    torch.cuda.empty_cache()

            # Epoch summary
            if self.rank == 0:
                dur = time.time() - t_epoch
                logger.info(f"{'='*60}")
                logger.info(
                    f"Epoch {epoch} done in {dur/60:.1f}min  "
                    f"skipped={skipped}"
                )
                for k in ["pg_loss", "mean_reward", "mean_sp", "kl",
                           "nonzero_adv_frac"]:
                    vals = epoch_metrics.get(k, [0])
                    logger.info(f"  avg {k}: {np.mean(vals):.4f}")
                logger.info(f"{'='*60}")

                self.save_checkpoint(f"epoch-{epoch}")
                self.validate(tag=f"epoch-{epoch}_final")

            if self.world_size > 1:
                dist.barrier()

            self.resume_step = 0

        if self.rank == 0:
            logger.info("Training complete!")


# -- CLI ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Standard LoRA RL + vLLM acceleration")

    # Model
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sft_checkpoint", type=str, default="")
    parser.add_argument("--resume_from", type=str, default="")

    # Data
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, default="")
    parser.add_argument("--max_episodes", type=int, default=0)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj"])

    parser.add_argument("--image_max_pixels", type=int, default=602112)

    # RL
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--kl_coef", type=float, default=0.001)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    # SP+GiGPO+SPWA
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--spwa_decay", type=float, default=0.5)
    parser.add_argument("--dapo_threshold", type=float, default=0.1)
    parser.add_argument("--step_adv_weight", type=float, default=0.5)
    parser.add_argument("--advantage_mode", type=str, default="sp",
                        choices=["sp", "grpo"])

    # Reward weights
    parser.add_argument("--w_format", type=float, default=0.1)
    parser.add_argument("--w_type", type=float, default=0.2)
    parser.add_argument("--w_content", type=float, default=0.7)

    # Optimizer
    parser.add_argument("--lora_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Training
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--val_steps", type=int, default=25)

    # vLLM
    parser.add_argument("--vllm_gpu_mem", type=float, default=0.45,
                        help="Fraction of GPU memory for vLLM (model + KV cache)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    trainer = StandardLoRATrainerVLLM(args)
    trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
