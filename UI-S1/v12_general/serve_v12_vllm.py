#!/usr/bin/env python3
"""Launch vLLM server with lossless V12 soft cooperative LoRA corrections.

Overrides vLLM's Qwen2.5VL model with a corrected version that applies
input-dependent routing corrections via forward hooks.

Base weights have r=0.5 merge baked in. Hooks compute exact correction:
    correction(x) = scaling * B @ ((sigmoid(x @ w_route) - 0.5) * A_diff @ x)
    where A_diff = A_1 - A_2

This is mathematically identical to the full soft cooperative forward pass.

Usage:
    python v12_general/serve_v12_vllm.py \
        --model /tmp/v12_merged_lossless \
        --port 8000 \
        --tensor-parallel-size 4 \
        --served-model-name v12_sft_ep4
"""

import argparse
import json
import os
import sys
import threading

import torch
import torch.nn.functional as F

# Thread-local storage for routing weights propagation (QKV → O_proj)
_thread_local = threading.local()

sys.stdout.reconfigure(line_buffering=True)


def _get_routing():
    """Get per-token routing weights from QKV hook."""
    return getattr(_thread_local, "routing_weights", None)


def _set_routing(r):
    _thread_local.routing_weights = r


# ── Hook factories ───────────────────────────────────────────────────

def _make_qkv_hook(w_route, A_diff_list, B_shards, scaling, split_sizes):
    """Hook for fused QKV (column-parallel).

    Computes routing once from the full input hidden state, stores it
    for the O_proj hook, then applies corrections to Q/K/V outputs.

    Args:
        w_route:     [hidden_size] routing vector (shared per layer)
        A_diff_list: list of [lora_r, hidden_size] tensors (one per q/k/v)
        B_shards:    list of [out_per_tp, lora_r] tensors (TP-sharded, one per q/k/v)
        scaling:     alpha / r
        split_sizes: how to split fused A_diff output (typically [lora_r]*3)
    """
    # Fuse A_diff for efficiency: single matmul for all sub-modules
    A_diff_fused = torch.cat(A_diff_list, dim=0)  # [n*lora_r, hidden_size]
    lora_r = A_diff_list[0].shape[0]

    def hook(module, input, output):
        x = input[0]  # [total_seq, hidden_size]
        dtype = x.dtype

        # Compute routing: r = sigmoid(x @ w_route) → [total_seq, 1]
        logit = F.linear(x, w_route.unsqueeze(0).to(dtype))  # [total_seq, 1]
        r = torch.sigmoid(logit)
        _set_routing(r)  # store for O_proj hook

        # Correction factor: (r - 0.5)
        r_corr = r - 0.5  # [total_seq, 1]

        # Fused A_diff matmul
        h_all = F.linear(x, A_diff_fused.to(dtype))  # [total_seq, n*lora_r]
        h_parts = h_all.split(split_sizes, dim=-1)

        # Per-module B matmul + concat
        corr_parts = []
        for h_part, B_s in zip(h_parts, B_shards):
            # h_part: [total_seq, lora_r], B_s: [out_per_tp, lora_r]
            corr = F.linear(r_corr * h_part, B_s.to(dtype))  # [total_seq, out_per_tp]
            corr_parts.append(corr)
        correction = torch.cat(corr_parts, dim=-1) * scaling

        if isinstance(output, tuple):
            return (output[0] + correction,) + output[1:]
        return output + correction

    return hook


def _make_o_proj_hook(A_diff_shard, B_full, scaling):
    """Hook for O_proj (row-parallel).

    Reuses routing weights from QKV hook. Input is TP-sharded,
    needs all-reduce for the low-rank correction.

    Args:
        A_diff_shard: [lora_r, hidden_size/tp_size] (TP-sharded along input)
        B_full:       [hidden_size, lora_r] (replicated)
        scaling:      alpha / r
    """
    def hook(module, input, output):
        r = _get_routing()  # [total_seq, 1] from QKV hook
        if r is None:
            return output

        x_shard = input[0]  # [total_seq, hidden_size/tp_size]
        dtype = x_shard.dtype

        r_corr = r - 0.5  # [total_seq, 1]

        # Partial low-rank projection on this TP shard
        h_partial = F.linear(x_shard, A_diff_shard.to(dtype))  # [total_seq, lora_r]

        # All-reduce across TP ranks
        from vllm.distributed import tensor_model_parallel_all_reduce
        h_full = tensor_model_parallel_all_reduce(h_partial)  # [total_seq, lora_r]

        correction = F.linear(r_corr * h_full, B_full.to(dtype)) * scaling

        if isinstance(output, tuple):
            return (output[0] + correction,) + output[1:]
        return output + correction

    return hook


# ── Load and shard corrections ───────────────────────────────────────

def _load_v12_corrections(model_dir):
    """Load v12_correction_config.json and v12_correction_adapters.pt."""
    with open(os.path.join(model_dir, "v12_correction_config.json")) as f:
        config = json.load(f)
    corrections = torch.load(
        os.path.join(model_dir, "v12_correction_adapters.pt"),
        map_location="cpu", weights_only=True,
    )
    return config, corrections


def _shard_column_parallel(tensor, tp_rank, tp_size):
    """Shard along output (dim 0) for column-parallel layers."""
    chunk = tensor.shape[0] // tp_size
    return tensor[tp_rank * chunk: (tp_rank + 1) * chunk].contiguous()


def _shard_row_parallel(tensor, tp_rank, tp_size):
    """Shard along input (dim 1) for row-parallel layers."""
    chunk = tensor.shape[1] // tp_size
    return tensor[:, tp_rank * chunk: (tp_rank + 1) * chunk].contiguous()


# ── Model patching ───────────────────────────────────────────────────

def patch_model_with_v12_corrections(model, model_dir, device):
    """Apply V12 lossless correction hooks to a loaded vLLM model."""
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()

    config, corrections = _load_v12_corrections(model_dir)
    scaling = config["scaling"]
    target_modules = config["target_modules"]
    num_layers = config["num_layers"]
    lora_r = config["lora_r"]

    # Get transformer layers
    lang_model = model.language_model
    if hasattr(lang_model, "model"):
        layers = lang_model.model.layers
    else:
        layers = lang_model.layers

    hooks_registered = 0

    for layer_idx in range(num_layers):
        layer = layers[layer_idx]

        # Get routing vector for this layer
        w_route_key = f"route_weights.{layer_idx}"
        if w_route_key not in corrections:
            print(f"  WARNING: missing {w_route_key}, skipping layer {layer_idx}")
            continue
        w_route = corrections[w_route_key].to(device)

        # ── QKV (column-parallel, fused) ─────────────────────────────
        qkv_names = [n for n in ("q_proj", "k_proj", "v_proj")
                      if n in target_modules]
        qkv_adapters = []
        for name in qkv_names:
            bk = f"model.layers.{layer_idx}.self_attn.{name}.weight"
            A_diff_key = bk + ".A_diff"
            B_key = bk + ".B"
            if A_diff_key in corrections and B_key in corrections:
                qkv_adapters.append((
                    name,
                    corrections[A_diff_key],  # [lora_r, hidden_size]
                    corrections[B_key],        # [out_size, lora_r]
                ))

        if qkv_adapters and hasattr(layer.self_attn, "qkv_proj"):
            A_diff_list = [a.to(device) for _, a, _ in qkv_adapters]
            B_shards = [
                _shard_column_parallel(b, tp_rank, tp_size).to(device)
                for _, _, b in qkv_adapters
            ]
            split_sizes = [lora_r] * len(qkv_adapters)

            hook = _make_qkv_hook(w_route, A_diff_list, B_shards,
                                   scaling, split_sizes)
            layer.self_attn.qkv_proj.register_forward_hook(hook)
            hooks_registered += 1

        # ── O_proj (row-parallel) ────────────────────────────────────
        if "o_proj" in target_modules:
            bk = f"model.layers.{layer_idx}.self_attn.o_proj.weight"
            A_diff_key = bk + ".A_diff"
            B_key = bk + ".B"
            if (A_diff_key in corrections and B_key in corrections
                    and hasattr(layer.self_attn, "o_proj")):
                A_diff_shard = _shard_row_parallel(
                    corrections[A_diff_key], tp_rank, tp_size
                ).to(device)
                B_full = corrections[B_key].to(device)

                hook = _make_o_proj_hook(A_diff_shard, B_full, scaling)
                layer.self_attn.o_proj.register_forward_hook(hook)
                hooks_registered += 1

    print(f"[TP rank {tp_rank}] Registered {hooks_registered} V12 correction hooks "
          f"across {num_layers} layers (scaling={scaling})")
    return hooks_registered


# ── Custom vLLM model class ──────────────────────────────────────────

def create_v12_model_class():
    """Create a corrected Qwen2.5-VL model class with V12 routing hooks."""
    from vllm.model_executor.models.qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration as _BaseQwen25VL,
    )

    class V12CorrectedQwen25VL(_BaseQwen25VL):
        """Qwen2.5-VL with lossless V12 soft cooperative LoRA corrections."""

        def __init__(self, *, vllm_config, prefix=""):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            self._model_dir = vllm_config.model_config.model
            self._corrections_applied = False

        def load_weights(self, weights):
            result = super().load_weights(weights)
            if not self._corrections_applied:
                corr_path = os.path.join(self._model_dir,
                                          "v12_correction_adapters.pt")
                if os.path.exists(corr_path):
                    device = next(self.parameters()).device
                    n = patch_model_with_v12_corrections(
                        self, self._model_dir, device
                    )
                    print(f"Applied {n} V12 correction hooks")
                    self._corrections_applied = True
            return result

        def forward(self, input_ids, positions, intermediate_tensors=None,
                    inputs_embeds=None, **kwargs):
            # Clear routing state before each forward pass
            _set_routing(None)
            try:
                return super().forward(
                    input_ids, positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds, **kwargs,
                )
            finally:
                _set_routing(None)

    return V12CorrectedQwen25VL


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Launch vLLM with lossless V12 soft cooperative corrections"
    )
    parser.add_argument("--model", required=True,
                        help="Path to merged model dir with v12_correction_adapters.pt")
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--limit-mm-per-prompt", type=str, default='{"image": 2}')
    args = parser.parse_args()

    # Verify correction files exist
    corr_path = os.path.join(args.model, "v12_correction_adapters.pt")
    if not os.path.exists(corr_path):
        print(f"ERROR: {corr_path} not found. Run v12_general/merge_to_hf.py first.")
        sys.exit(1)

    # Register custom model class BEFORE importing vLLM server
    V12Model = create_v12_model_class()
    from vllm import ModelRegistry
    ModelRegistry.register_model(
        "Qwen2_5_VLForConditionalGeneration",
        V12Model,
    )
    print("Registered V12CorrectedQwen25VL with vLLM")

    # Build vLLM args — enforce-eager required for conditional hooks
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
