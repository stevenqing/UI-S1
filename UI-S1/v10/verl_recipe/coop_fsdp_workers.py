"""Cooperative LoRA FSDP Workers.

Extends verl's ActorRolloutRefWorker with dual-adapter (grounder + actor)
support. Key modifications:

1. _build_model_optimizer: adds second LoRA adapter before FSDP wrap
2. _set_adapter_safe: switches active adapter with requires_grad fix
3. New RPC endpoints: generate/compute_log_prob/update with adapter switching
4. Dual param-group optimizer with separate LRs
"""

import logging
import os
import warnings
from contextlib import nullcontext
from typing import Optional

import torch
import torch.distributed as dist
import torch.optim as optim
from codetiming import Timer
from omegaconf import DictConfig, open_dict
from peft import LoraConfig, TaskType, get_peft_model

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_torch_device
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    load_fsdp_optimizer,
)
from verl.utils.py_functional import convert_to_regular_types
from verl.workers.fsdp_workers import ActorRolloutRefWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class CoopActorRolloutRefWorker(ActorRolloutRefWorker):
    """ActorRolloutRefWorker with dual LoRA adapter support.

    The first adapter ('default' / grounder) is created by the parent class.
    We add a second adapter ('actor_lora') before FSDP wrapping.
    """

    def __init__(self, config: DictConfig, role: str):
        # Store cooperative config before super().__init__
        self._coop_enabled = config.get("cooperative", {}).get("enable", False)
        super().__init__(config, role)

    # ── Model setup override ────────────────────────────────────────────

    def _build_model_optimizer(self, model_path, fsdp_config, optim_config,
                               override_model_config=None,
                               use_remove_padding=False,
                               use_fused_kernels=False,
                               trust_remote_code=False,
                               use_liger=False,
                               role="actor",
                               enable_gradient_checkpointing=True,
                               enable_activation_offload=False):
        """Override to add second LoRA adapter before FSDP2 wrap.

        Strategy: monkey-patch apply_fsdp2 / fsdp2_load_full_state_dict in the
        parent module so the parent's call becomes a no-op. After super()
        returns (model has first adapter but NO FSDP2), we add the second
        adapter, then apply FSDP2 ourselves with both adapters present.
        This ensures all LoRA params are DTensors with uniform bf16 dtype.
        """
        if not self._coop_enabled or role != "actor":
            return super()._build_model_optimizer(
                model_path=model_path, fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=trust_remote_code,
                use_liger=use_liger, role=role,
                enable_gradient_checkpointing=enable_gradient_checkpointing,
                enable_activation_offload=enable_activation_offload,
            )

        # ── Defer FSDP2 wrapping during parent call ──────────────────
        import verl.workers.fsdp_workers as _parent_mod

        _orig_apply = _parent_mod.apply_fsdp2
        _orig_load = _parent_mod.fsdp2_load_full_state_dict
        _captured = {}

        def _deferred_apply(model, fsdp_kwargs, config):
            _captured["apply"] = (fsdp_kwargs, config)

        def _deferred_load(model, full_state, mesh=None, cpu_offload=None):
            _captured["load"] = (full_state, mesh, cpu_offload)

        _parent_mod.apply_fsdp2 = _deferred_apply
        _parent_mod.fsdp2_load_full_state_dict = _deferred_load

        try:
            result = super()._build_model_optimizer(
                model_path=model_path, fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=trust_remote_code,
                use_liger=use_liger, role=role,
                enable_gradient_checkpointing=enable_gradient_checkpointing,
                enable_activation_offload=enable_activation_offload,
            )
        finally:
            _parent_mod.apply_fsdp2 = _orig_apply
            _parent_mod.fsdp2_load_full_state_dict = _orig_load

        actor_module, actor_optimizer, actor_lr_scheduler, actor_model_config = result

        # ── Add second adapter (model is NOT FSDP2 wrapped yet) ──────
        if "apply" not in _captured:
            # Parent didn't use FSDP2 (e.g. FSDP1). Fall back to post-wrap approach.
            if self.rank == 0:
                print("[Coop] FSDP2 not detected; skipping deferred adapter injection")
            return result

        if self.rank == 0:
            print("[Coop] Adding second LoRA adapter 'actor_lora' (before FSDP2)")

        peft_model = self._find_peft_model(actor_module)
        if peft_model is None:
            raise RuntimeError("Cannot find PeftModel. Ensure lora_rank > 0.")

        existing_config = peft_model.peft_config.get("default")
        if existing_config is None:
            existing_config = next(iter(peft_model.peft_config.values()), None)
        if existing_config is None:
            raise RuntimeError("No existing LoRA adapter found")

        second_lora_config = LoraConfig(
            r=existing_config.r,
            lora_alpha=existing_config.lora_alpha,
            lora_dropout=existing_config.lora_dropout,
            target_modules=list(existing_config.target_modules),
            task_type=existing_config.task_type,
            bias=existing_config.bias,
        )
        peft_model.add_adapter("actor_lora", second_lora_config)
        # Switch back to default adapter so grounder is active initially
        peft_model.set_adapter("default")
        self._set_all_lora_trainable(peft_model)

        if self.rank == 0:
            n_default = sum(p.numel() for n, p in peft_model.named_parameters()
                            if "lora_" in n and "actor_lora" not in n and p.requires_grad)
            n_actor = sum(p.numel() for n, p in peft_model.named_parameters()
                          if "actor_lora" in n and p.requires_grad)
            print(f"[Coop] Grounder (default) params: {n_default:,}")
            print(f"[Coop] Actor params: {n_actor:,}")

        # ── Ensure ALL params are bf16 before FSDP2 ─────────────────
        # FSDP2 requires uniform dtype per param group. Force everything to bf16.
        if self.rank == 0:
            dtype_counts = {}
            fp32_params = []
            for name, param in actor_module.named_parameters():
                dt = str(param.data.dtype)
                dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
                if param.data.dtype == torch.float32:
                    fp32_params.append(name)
            print(f"[Coop] Param dtype distribution before FSDP2: {dtype_counts}")
            if fp32_params:
                print(f"[Coop] fp32 params ({len(fp32_params)}): {fp32_params[:20]}")

        for name, param in actor_module.named_parameters():
            if param.data.dtype != torch.bfloat16:
                param.data = param.data.to(torch.bfloat16)

        # ── Now apply FSDP2 with both adapters present ───────────────
        fsdp_kwargs, fsdp_cfg = _captured["apply"]
        full_state = actor_module.state_dict()  # includes both adapters
        _orig_apply(actor_module, fsdp_kwargs, fsdp_cfg)

        if "load" in _captured:
            _, mesh, cpu_offload = _captured["load"]
            _orig_load(actor_module, full_state, mesh, cpu_offload)

        if self.rank == 0:
            print("[Coop] FSDP2 applied with both adapters")
            # Check post-FSDP2 dtypes
            dtype_counts = {}
            fp32_params = []
            for name, param in actor_module.named_parameters():
                dt = str(param.dtype)
                dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
                if "float32" in dt:
                    fp32_params.append(name)
            print(f"[Coop] Post-FSDP2 dtype distribution: {dtype_counts}")
            if fp32_params:
                print(f"[Coop] Post-FSDP2 fp32 params ({len(fp32_params)}): {fp32_params[:20]}")

        # ── Rebuild optimizer with dual param groups ─────────────────
        if optim_config is not None:
            peft_model = self._find_peft_model(actor_module)
            actor_optimizer, actor_lr_scheduler = self._rebuild_dual_optimizer(
                peft_model, optim_config
            )

        return actor_module, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _find_peft_model(self, module):
        """Recursively find PeftModel inside FSDP wrapper."""
        from peft import PeftModel
        if isinstance(module, PeftModel):
            return module
        # Check direct attributes
        for attr_name in ['module', '_fsdp_wrapped_module', 'model']:
            child = getattr(module, attr_name, None)
            if child is not None:
                result = self._find_peft_model(child)
                if result is not None:
                    return result
        # Check children
        for child in module.children():
            result = self._find_peft_model(child)
            if result is not None:
                return result
        return None

    def _set_all_lora_trainable(self, peft_model):
        """Force all LoRA parameters to requires_grad=True and uniform dtype.

        FSDP2 requires uniform parameter dtype within each param group.
        PEFT creates LoRA params in fp32 by default, but base model is bf16.
        """
        base_dtype = torch.bfloat16  # Match base model dtype
        for name, param in peft_model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
                if param.data.dtype != base_dtype:
                    param.data = param.data.to(base_dtype)

    def _rebuild_dual_optimizer(self, peft_model, optim_config):
        """Create optimizer with separate param groups for each adapter.

        Returns (optimizer, lr_scheduler) for the caller to use.
        """
        from verl.utils.torch_functional import (
            get_constant_schedule_with_warmup,
            get_cosine_schedule_with_warmup,
        )

        grounder_lr = self.config.get("cooperative", {}).get("grounder_lr", 1e-5)
        actor_lr = self.config.get("cooperative", {}).get("actor_lr", 5e-6)

        grounder_params = []
        actor_params = []
        for name, param in peft_model.named_parameters():
            if not param.requires_grad:
                continue
            if "actor_lora" in name:
                actor_params.append(param)
            elif "lora_" in name:
                grounder_params.append(param)

        if self.rank == 0:
            print(f"[Coop] Optimizer: grounder_params={len(grounder_params)} (lr={grounder_lr}), "
                  f"actor_params={len(actor_params)} (lr={actor_lr})")

        optimizer = optim.AdamW([
            {"params": grounder_params, "lr": grounder_lr},
            {"params": actor_params, "lr": actor_lr},
        ], weight_decay=optim_config.get("weight_decay", 1e-2))

        # Rebuild LR scheduler
        total_steps = optim_config.get("total_training_steps", 0)
        num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
        warmup_style = optim_config.get("warmup_style", "constant")
        min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)

        if num_warmup_steps < 0:
            ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(ratio * total_steps)

        if warmup_style == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
            )
        else:
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
            )

        return optimizer, lr_scheduler

    # ── Adapter switching ───────────────────────────────────────────────

    def _set_adapter_safe(self, adapter_name: str):
        """Switch active adapter and ensure all LoRA params stay trainable."""
        peft_model = self._find_peft_model(self.actor_module_fsdp)
        if peft_model is None:
            return
        peft_model.set_adapter(adapter_name)
        # Force uniform bf16 dtype for ALL params (not just LoRA)
        for name, param in peft_model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            if hasattr(param, '_local_tensor'):
                # DTensor — check the local shard
                if param._local_tensor.dtype != torch.bfloat16:
                    param._local_tensor.data = param._local_tensor.data.to(torch.bfloat16)
            else:
                if param.data.dtype != torch.bfloat16:
                    param.data = param.data.to(torch.bfloat16)

    # ── RPC endpoints with adapter switching ────────────────────────────

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_with_adapter(self, prompts: DataProto):
        """Generate sequences using a specific adapter."""
        adapter_name = prompts.meta_info.get("adapter_name", "default")
        if self._coop_enabled:
            self._set_adapter_safe(adapter_name)
        return self.generate_sequences(prompts)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob_with_adapter(self, data: DataProto):
        """Compute log prob using a specific adapter."""
        adapter_name = data.meta_info.get("adapter_name", "default")
        if self._coop_enabled:
            self._set_adapter_safe(adapter_name)
        # Remove adapter_name from meta_info to avoid confusing parent's is_lora logic
        data.meta_info.pop("adapter_name", None)
        return self.compute_log_prob(data)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor_with_adapter(self, data: DataProto):
        """Update policy using a specific adapter."""
        adapter_name = data.meta_info.get("adapter_name", "default")
        if self._coop_enabled:
            self._set_adapter_safe(adapter_name)
        data.meta_info.pop("adapter_name", None)
        return self.update_actor(data)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def free_cache(self):
        """Release fragmented CUDA memory on all workers."""
        import gc
        gc.collect()
        torch.cuda.empty_cache()
