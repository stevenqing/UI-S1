#!/usr/bin/env python3
"""Launch vLLM server with lossless cooperative LoRA corrections.

Overrides vLLM's Qwen2VLForConditionalGeneration with a corrected version
that applies per-token correction adapters at IMAGE_PAD_ID positions.

Text tokens:  W_merged @ x             (exact, zero overhead)
Image tokens: W_merged @ x + corr @ x  (exact, low-rank correction)
Decode phase: all text → no correction  (standard vLLM speed)

Usage:
    python evaluation/serve_corrected_vllm.py \
        --model /tmp/merged_lossless_ac_ep4 \
        --port 8000 \
        --tensor-parallel-size 4 \
        --served-model-name coop_ac_ep4
"""

import argparse
import json
import os
import sys
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_PAD_ID = 151655

# Coordinate token constants (matching cooperative_wrapper.py)
COORD_KEY_ID = 62526       # 'coordinate'
BBOX_KEY_ID = 58456        # 'bbox'
DIGIT_TOKEN_IDS = set(range(15, 25))  # 0-9
COORD_PUNCT_IDS = {13, 11, 220}      # '.', ',', ' '
COORD_ALL_VALUE_IDS = DIGIT_TOKEN_IDS | COORD_PUNCT_IDS
BRACKET_OPEN_ID = 508      # ' ['
BRACKET_CLOSE_IDS = {1125, 81136}  # '],', ']}'

# Thread-local storage for image mask propagation across layers
_thread_local = threading.local()

# Global coord_routing flag
_coord_routing = False
_coord_only_routing = False

# Global cooperative reasoning alpha (0 = disabled)
_coop_reasoning_alpha = 0.0

# Global inference mode: 'hard' | 'v_only' | 't_only' | 'merge'
# In the lossless-merge representation (base has LoRA_A merged, correction = delta_V - delta_A):
#   hard:   correction * img_mask        (img→V, txt→A, default)
#   v_only: correction * 1.0             (all tokens → V)
#   t_only: correction * 0.0             (all tokens → A, skip hooks)
#   merge:  correction * 0.5             (all tokens → 0.5V + 0.5A)
_inference_mode = "hard"


def _build_coord_mask_1d(input_ids: torch.Tensor) -> torch.Tensor:
    """Build coord mask for a 1D (flattened) token sequence.

    Returns a bool tensor where True = coordinate/bbox digit token.
    Handles 'coordinate2' (swipe) correctly.
    """
    # Tokens between coord key and opening bracket (key trail):
    # 788='":',  digits=part of key name like coordinate2, 1='"', 330=' "'
    key_trail = {788, 1, 330} | DIGIT_TOKEN_IDS

    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    seq_len = input_ids.shape[0]
    in_coord = False
    in_bracket = False
    for i in range(seq_len):
        tid = input_ids[i].item()
        if not in_coord:
            if tid == COORD_KEY_ID or tid == BBOX_KEY_ID:
                in_coord = True
                in_bracket = False
        else:
            if not in_bracket:
                if tid == BRACKET_OPEN_ID:
                    in_bracket = True
                elif tid not in key_trail:
                    in_coord = False
            else:
                if tid in BRACKET_CLOSE_IDS:
                    in_coord = False
                    in_bracket = False
                elif tid in COORD_ALL_VALUE_IDS:
                    mask[i] = True
                else:
                    in_coord = False
                    in_bracket = False
    return mask


def _build_assistant_mask_1d(input_ids: torch.Tensor) -> torch.Tensor:
    """Detect assistant response token spans in a 1D (flattened) sequence.

    Returns a bool tensor where True = token is inside an assistant response
    (between <|im_start|>assistant\\n and <|im_end|>).
    """
    IM_START = 151644
    ASSISTANT_ID = 77091
    NEWLINE_ID = 198
    IM_END = 151645

    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    seq_len = input_ids.shape[0]
    in_assistant = False
    i = 0
    while i < seq_len:
        tid = input_ids[i].item()
        if not in_assistant:
            if (tid == IM_START and i + 2 < seq_len and
                    input_ids[i + 1].item() == ASSISTANT_ID and
                    input_ids[i + 2].item() == NEWLINE_ID):
                in_assistant = True
                i += 3  # skip header
                continue
        else:
            if tid == IM_END:
                in_assistant = False
            else:
                mask[i] = True
        i += 1
    return mask


def _get_image_mask():
    return getattr(_thread_local, "image_mask", None)


def _set_image_mask(mask):
    _thread_local.image_mask = mask


# ── Correction weight helper ──────────────────────────────────────────

def _compute_correction_weight(mask, dtype, seq_len, device):
    """Compute per-token correction weight based on _inference_mode.

    Returns weight tensor [seq_len, 1] or None to skip correction entirely.
    """
    mode = _inference_mode
    if mode == "t_only":
        return None  # all tokens use delta_A → no correction needed
    elif mode == "v_only":
        return torch.ones(seq_len, 1, dtype=dtype, device=device)
    elif mode == "merge":
        return torch.full((seq_len, 1), 0.5, dtype=dtype, device=device)
    else:  # "hard"
        if mask is None:
            return None
        return mask.unsqueeze(-1).to(dtype)


# ── Correction hook factories ─────────────────────────────────────────

def _make_column_parallel_hook(C_in, C_out_shards, scaling, split_sizes=None):
    """Hook for QKVParallelLinear or MergedColumnParallelLinear.

    Column-parallel: input is full, output is TP-sharded.
    C_in:        [combined_2r, hidden_size]  (full, replicated)
    C_out_shards: list of [out_per_tp, 2r] tensors (one per sub-module, TP-sharded)
    split_sizes: how to split C_in's output dim for each sub-module

    For QKV: 3 sub-modules (q, k, v), split_sizes=[2r, 2r, 2r]
    For gate_up: 2 sub-modules (gate, up), split_sizes=[2r, 2r]
    """
    r2 = C_out_shards[0].shape[1]  # 2r per sub-module
    n_sub = len(C_out_shards)
    if split_sizes is None:
        split_sizes = [r2] * n_sub

    def hook(module, input, output):
        mask = _get_image_mask()
        x = input[0]  # [total_seq, hidden_size]
        weight = _compute_correction_weight(mask, x.dtype, x.shape[0], x.device)
        if weight is None or not weight.any():
            return output

        # Fused C_in matmul: [total_seq, combined_2r]
        h_all = F.linear(x, C_in)
        h_parts = h_all.split(split_sizes, dim=-1)

        # Separate C_out matmuls per sub-module, then concatenate
        corr_parts = []
        for h_part, C_out_s in zip(h_parts, C_out_shards):
            corr_parts.append(F.linear(h_part, C_out_s))
        corr = torch.cat(corr_parts, dim=-1) * scaling  # [total_seq, out_per_tp]

        if isinstance(output, tuple):
            return (output[0] + corr * weight,) + output[1:]
        return output + corr * weight

    return hook


def _make_row_parallel_hook(C_in_shard, C_out, scaling):
    """Hook for RowParallelLinear (o_proj, down_proj).

    Row-parallel: input is TP-sharded, output is all-reduced.
    C_in_shard: [2r, input_per_tp]  (TP-sharded along input dim)
    C_out:      [output_size, 2r]   (replicated)

    The correction needs all-reduce across TP ranks:
      corr = C_out @ all_reduce(C_in_shard @ x_shard)
    """

    def hook(module, input, output):
        mask = _get_image_mask()
        x_shard = input[0]  # [total_seq, input_per_tp]
        weight = _compute_correction_weight(mask, x_shard.dtype, x_shard.shape[0], x_shard.device)
        if weight is None or not weight.any():
            return output

        # Partial latent on this TP rank
        h_partial = F.linear(x_shard, C_in_shard)  # [total_seq, 2r]

        # All-reduce to get full latent (all ranks participate)
        from vllm.distributed import tensor_model_parallel_all_reduce
        h_full = tensor_model_parallel_all_reduce(h_partial)  # [total_seq, 2r]

        corr = F.linear(h_full, C_out) * scaling  # [total_seq, output_size]

        if isinstance(output, tuple):
            return (output[0] + corr * weight,) + output[1:]
        return output + corr * weight

    return hook


# ── Load and shard corrections ────────────────────────────────────────

def _load_corrections(model_dir):
    """Load correction_config.json and correction_adapters.pt."""
    with open(os.path.join(model_dir, "correction_config.json")) as f:
        config = json.load(f)
    corrections = torch.load(
        os.path.join(model_dir, "correction_adapters.pt"),
        map_location="cpu", weights_only=True,
    )
    # Parse into per-base-key dict
    modules = {}
    for key, val in corrections.items():
        if key.endswith(".C_in"):
            bk = key[: -len(".C_in")]
            modules.setdefault(bk, {})["C_in"] = val
        elif key.endswith(".C_out"):
            bk = key[: -len(".C_out")]
            modules.setdefault(bk, {})["C_out"] = val
    return config, modules


def _shard_column_parallel(C_out_full, tp_rank, tp_size):
    """Shard C_out along output dim for column-parallel layers."""
    out_size = C_out_full.shape[0]
    chunk = out_size // tp_size
    return C_out_full[tp_rank * chunk: (tp_rank + 1) * chunk].contiguous()


def _shard_row_parallel(C_in_full, tp_rank, tp_size):
    """Shard C_in along input dim for row-parallel layers."""
    in_size = C_in_full.shape[1]
    chunk = in_size // tp_size
    return C_in_full[:, tp_rank * chunk: (tp_rank + 1) * chunk].contiguous()


# ── Model patching ────────────────────────────────────────────────────

def patch_model_with_corrections(model, model_dir, device):
    """Apply correction hooks to a loaded vLLM Qwen2.5-VL model.

    Args:
        model: The vLLM model instance (Qwen2VLForConditionalGeneration)
        model_dir: Path to merged model directory with correction_adapters.pt
        device: GPU device for correction tensors
    """
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()

    corr_config, corr_modules = _load_corrections(model_dir)
    scaling = corr_config["scaling"]
    target_modules = corr_config["target_modules"]

    # Get transformer layers
    lang_model = model.language_model
    if hasattr(lang_model, "model"):
        layers = lang_model.model.layers
    else:
        layers = lang_model.layers

    num_layers = len(layers)
    hooks_registered = 0

    for layer_idx in range(num_layers):
        layer = layers[layer_idx]

        # ── QKV (column-parallel, fused q+k+v) ───────────────────────
        qkv_modules = []
        for name in ("q_proj", "k_proj", "v_proj"):
            if name not in target_modules:
                continue
            bk = f"model.layers.{layer_idx}.self_attn.{name}.weight"
            if bk in corr_modules:
                qkv_modules.append((name, corr_modules[bk]))

        if qkv_modules and hasattr(layer.self_attn, "qkv_proj"):
            # Fuse C_in: cat along dim=0
            C_in_parts = [m["C_in"] for _, m in qkv_modules]
            C_in_fused = torch.cat(C_in_parts, dim=0).to(device)  # [n*2r, hidden]

            # Shard C_out per sub-module
            C_out_shards = []
            for _, m in qkv_modules:
                C_out_shards.append(
                    _shard_column_parallel(m["C_out"], tp_rank, tp_size).to(device)
                )

            r2 = C_in_parts[0].shape[0]  # 2r
            split_sizes = [r2] * len(qkv_modules)

            hook = _make_column_parallel_hook(
                C_in_fused, C_out_shards, scaling, split_sizes
            )
            layer.self_attn.qkv_proj.register_forward_hook(hook)
            hooks_registered += 1

        # ── O proj (row-parallel) ─────────────────────────────────────
        if "o_proj" in target_modules:
            bk = f"model.layers.{layer_idx}.self_attn.o_proj.weight"
            if bk in corr_modules and hasattr(layer.self_attn, "o_proj"):
                m = corr_modules[bk]
                C_in_shard = _shard_row_parallel(m["C_in"], tp_rank, tp_size).to(device)
                C_out = m["C_out"].to(device)
                hook = _make_row_parallel_hook(C_in_shard, C_out, scaling)
                layer.self_attn.o_proj.register_forward_hook(hook)
                hooks_registered += 1

        # ── Gate+Up (column-parallel, fused gate+up) ──────────────────
        gu_modules = []
        for name in ("gate_proj", "up_proj"):
            if name not in target_modules:
                continue
            bk = f"model.layers.{layer_idx}.mlp.{name}.weight"
            if bk in corr_modules:
                gu_modules.append((name, corr_modules[bk]))

        if gu_modules and hasattr(layer.mlp, "gate_up_proj"):
            C_in_parts = [m["C_in"] for _, m in gu_modules]
            C_in_fused = torch.cat(C_in_parts, dim=0).to(device)

            C_out_shards = []
            for _, m in gu_modules:
                C_out_shards.append(
                    _shard_column_parallel(m["C_out"], tp_rank, tp_size).to(device)
                )

            r2 = C_in_parts[0].shape[0]
            split_sizes = [r2] * len(gu_modules)

            hook = _make_column_parallel_hook(
                C_in_fused, C_out_shards, scaling, split_sizes
            )
            layer.mlp.gate_up_proj.register_forward_hook(hook)
            hooks_registered += 1

        # ── Down proj (row-parallel) ──────────────────────────────────
        if "down_proj" in target_modules:
            bk = f"model.layers.{layer_idx}.mlp.down_proj.weight"
            if bk in corr_modules and hasattr(layer.mlp, "down_proj"):
                m = corr_modules[bk]
                C_in_shard = _shard_row_parallel(m["C_in"], tp_rank, tp_size).to(device)
                C_out = m["C_out"].to(device)
                hook = _make_row_parallel_hook(C_in_shard, C_out, scaling)
                layer.mlp.down_proj.register_forward_hook(hook)
                hooks_registered += 1

    print(f"[TP rank {tp_rank}] Registered {hooks_registered} correction hooks "
          f"across {num_layers} layers")
    return hooks_registered


# ── Custom model class ────────────────────────────────────────────────

def create_corrected_model_class():
    """Create and return a corrected Qwen2.5-VL model class for vLLM."""
    from vllm.model_executor.models.qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration as _BaseQwen25VL,
    )

    class CorrectedQwen25VLForConditionalGeneration(_BaseQwen25VL):
        """Qwen2.5-VL with lossless cooperative LoRA corrections.

        Loads correction adapters from the model directory and registers
        forward hooks on all target linear layers. Image tokens get exact
        LoRA_V routing via corrections; text tokens use the merged weights
        directly with zero overhead.
        """

        def __init__(self, *, vllm_config, prefix=""):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            self._model_dir = vllm_config.model_config.model
            self._corrections_applied = False

        def load_weights(self, weights):
            # Load base (merged) weights normally
            result = super().load_weights(weights)

            # Apply correction hooks after weights are loaded
            if not self._corrections_applied:
                corr_path = os.path.join(self._model_dir, "correction_adapters.pt")
                if os.path.exists(corr_path):
                    # Determine device from first parameter
                    device = next(self.parameters()).device
                    n = patch_model_with_corrections(self, self._model_dir, device)
                    print(f"Applied {n} correction hooks to model")
                    self._corrections_applied = True

            return result

        def forward(self, input_ids, positions, intermediate_tensors=None,
                    inputs_embeds=None, **kwargs):
            # Set correction mask for hooks
            if input_ids is not None:
                if _coop_reasoning_alpha > 0:
                    # Float mask: image/coord=1.0, assistant=α, other=0.0
                    mask = torch.zeros_like(input_ids, dtype=torch.float32)
                    mask[input_ids == IMAGE_PAD_ID] = 1.0
                    if _coord_routing:
                        coord_mask = _build_coord_mask_1d(input_ids)
                        mask[coord_mask] = 1.0
                    asst_mask = _build_assistant_mask_1d(input_ids)
                    non_v = (mask < 0.5)
                    mask[asst_mask & non_v] = _coop_reasoning_alpha
                elif _coord_only_routing:
                    # v10: ONLY coord digits get correction, no image tokens
                    mask = _build_coord_mask_1d(input_ids)
                else:
                    # Bool mask: image + optional coord
                    mask = (input_ids == IMAGE_PAD_ID)
                    if _coord_routing:
                        coord_mask = _build_coord_mask_1d(input_ids)
                        mask = mask | coord_mask
                _set_image_mask(mask)
            try:
                return super().forward(
                    input_ids, positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds, **kwargs,
                )
            finally:
                _set_image_mask(None)

    return CorrectedQwen25VLForConditionalGeneration


# ── Main entry point ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Launch vLLM with lossless cooperative LoRA corrections"
    )
    parser.add_argument("--model", required=True,
                        help="Path to lossless merged model directory")
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--limit-mm-per-prompt", type=str, default='{"image": 2}')
    parser.add_argument("--inference-mode", type=str, default="hard",
                        choices=["hard", "v_only", "t_only", "merge"],
                        help="Routing override for cooperative advantage analysis")
    parser.add_argument("--coord-routing", action="store_true",
                        help="Also apply corrections to coordinate/bbox digit tokens")
    parser.add_argument("--coord-only-routing", action="store_true",
                        help="v10: ONLY apply corrections to coordinate/bbox digit tokens "
                             "(no image token corrections)")
    parser.add_argument("--coop-reasoning-alpha", type=float, default=0.0,
                        help="Cooperative reasoning α: assistant tokens get "
                             "α·V + (1-α)·A correction weight. 0 disables.")
    args = parser.parse_args()

    # Set global inference mode
    global _inference_mode, _coord_routing, _coord_only_routing, _coop_reasoning_alpha
    _inference_mode = args.inference_mode
    _coord_routing = args.coord_routing
    _coord_only_routing = args.coord_only_routing
    _coop_reasoning_alpha = args.coop_reasoning_alpha
    print(f"Inference mode: {_inference_mode}, coord_routing: {_coord_routing}, "
          f"coord_only_routing: {_coord_only_routing}, "
          f"coop_reasoning_alpha: {_coop_reasoning_alpha}")

    # Check correction files exist
    corr_path = os.path.join(args.model, "correction_adapters.pt")
    if not os.path.exists(corr_path):
        print(f"ERROR: {corr_path} not found. Run merge_cooperative_lossless.py first.")
        sys.exit(1)

    # Register corrected model class BEFORE importing vLLM server
    CorrectedModel = create_corrected_model_class()
    from vllm import ModelRegistry
    ModelRegistry.register_model(
        "Qwen2_5_VLForConditionalGeneration",  # Override the default
        CorrectedModel,
    )
    print("Registered CorrectedQwen25VLForConditionalGeneration with vLLM")

    # Build vLLM server args
    # --enforce-eager required: correction hooks use conditional branches
    # that are incompatible with CUDA graph capture
    server_args = [
        "--model", args.model,
        "--port", str(args.port),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--max-model-len", str(args.max_model_len),
        "--trust-remote-code",
        "--enforce-eager",
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--limit-mm-per-prompt", args.limit_mm_per_prompt,
    ]
    if args.served_model_name:
        server_args.extend(["--served-model-name", args.served_model_name])

    # Start vLLM OpenAI-compatible server
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser, validate_parsed_serve_args,
    )
    from vllm.entrypoints.openai.api_server import run_server

    vllm_parser = make_arg_parser(FlexibleArgumentParser())
    vllm_args = vllm_parser.parse_args(server_args)
    validate_parsed_serve_args(vllm_args)

    import uvloop
    uvloop.run(run_server(vllm_args))


if __name__ == "__main__":
    main()
