#!/usr/bin/env python3
"""Launch vLLM server with lossless V13 iterative cooperative LoRA corrections.

Overrides vLLM's Qwen2.5VL model with a corrected version that applies
input-dependent routing + iterative communication corrections via forward hooks.

Base weights have r=0.5 merge baked in. Hooks compute exact correction:
    h_1 = A_1 @ x;  h_2 = A_2 @ x
    h_mean = 0.5 * (h_1 + h_2)
    h_1, h_2 = iterative_comm(h_1, h_2, comm_params)
    correction = scaling * B @ (r * h_1 + (1-r) * h_2 - h_mean)

Usage:
    python v13_gui_360/serve_v13_vllm.py \
        --model /tmp/v13_merged_lossless \
        --port 8000 \
        --tensor-parallel-size 4 \
        --served-model-name v13_gui360_rl_ep0
"""

import argparse
import json
import os
import sys
import threading

import torch
import torch.nn.functional as F

# Thread-local storage for routing weights propagation (QKV -> O_proj)
_thread_local = threading.local()

sys.stdout.reconfigure(line_buffering=True)


def _get_routing():
    """Get per-token routing weights from QKV hook."""
    return getattr(_thread_local, "routing_weights", None)


def _set_routing(r):
    _thread_local.routing_weights = r


# -- Iterative communication helper ----------------------------------------

def _iterative_comm(h_1, h_2, comm_params, dtype):
    """Run T rounds of gated communication in low-rank space.

    Args:
        h_1: [total_seq, r] expert 1 hidden
        h_2: [total_seq, r] expert 2 hidden
        comm_params: list of (W_12, W_21, gate_12, gate_21) per round
        dtype: compute dtype

    Returns:
        h_1, h_2 after T rounds of communication
    """
    for W_12, W_21, gate_12, gate_21 in comm_params:
        # Expert 1 receives from expert 2
        g_12 = torch.sigmoid(
            F.linear(h_1, gate_12.to(dtype).unsqueeze(0))
        )  # [total_seq, 1]
        h_1 = h_1 + g_12 * F.linear(h_2, W_12.to(dtype))

        # Expert 2 receives from (updated) expert 1
        g_21 = torch.sigmoid(
            F.linear(h_2, gate_21.to(dtype).unsqueeze(0))
        )  # [total_seq, 1]
        h_2 = h_2 + g_21 * F.linear(h_1, W_21.to(dtype))

    return h_1, h_2


# -- Hook factories --------------------------------------------------------

def _make_qkv_hook(w_route, A1_list, A2_list, B_shards, scaling,
                    comm_params):
    """Hook for fused QKV (column-parallel) with iterative communication.

    Args:
        w_route:     [hidden_size] routing vector
        A1_list:     list of [lora_r, hidden_size] tensors (one per q/k/v)
        A2_list:     list of [lora_r, hidden_size] tensors (one per q/k/v)
        B_shards:    list of [out_per_tp, lora_r] tensors (TP-sharded)
        scaling:     alpha / r
        comm_params: list of (W_12, W_21, gate_12, gate_21) tuples per round
    """
    def hook(module, input, output):
        x = input[0]  # [total_seq, hidden_size]
        dtype = x.dtype

        # Compute routing: r = sigmoid(x @ w_route)
        logit = F.linear(x, w_route.unsqueeze(0).to(dtype))
        r = torch.sigmoid(logit)  # [total_seq, 1]
        _set_routing(r)

        corr_parts = []
        for A1, A2, B_s in zip(A1_list, A2_list, B_shards):
            h1 = F.linear(x, A1.to(dtype))  # [total_seq, lora_r]
            h2 = F.linear(x, A2.to(dtype))  # [total_seq, lora_r]
            h_mean = 0.5 * (h1 + h2)

            # Iterative communication
            h1_comm, h2_comm = _iterative_comm(h1, h2, comm_params, dtype)

            # Correction: routing-weighted blend minus mean
            h_blend = r * h1_comm + (1 - r) * h2_comm
            corr = F.linear(h_blend - h_mean, B_s.to(dtype)) * scaling
            corr_parts.append(corr)

        correction = torch.cat(corr_parts, dim=-1)

        if isinstance(output, tuple):
            return (output[0] + correction,) + output[1:]
        return output + correction

    return hook


def _make_o_proj_hook(A1_shard, A2_shard, B_full, scaling, comm_params):
    """Hook for O_proj (row-parallel) with iterative communication.

    Args:
        A1_shard:    [lora_r, hidden_size/tp_size] TP-sharded
        A2_shard:    [lora_r, hidden_size/tp_size] TP-sharded
        B_full:      [hidden_size, lora_r] replicated
        scaling:     alpha / r
        comm_params: list of (W_12, W_21, gate_12, gate_21) tuples per round
    """
    def hook(module, input, output):
        r = _get_routing()  # [total_seq, 1]
        if r is None:
            return output

        x_shard = input[0]  # [total_seq, hidden_size/tp_size]
        dtype = x_shard.dtype

        # Partial low-rank projection on this TP shard
        h1_partial = F.linear(x_shard, A1_shard.to(dtype))
        h2_partial = F.linear(x_shard, A2_shard.to(dtype))

        # All-reduce across TP ranks
        from vllm.distributed import tensor_model_parallel_all_reduce
        h1_full = tensor_model_parallel_all_reduce(h1_partial)
        h2_full = tensor_model_parallel_all_reduce(h2_partial)

        h_mean = 0.5 * (h1_full + h2_full)

        # Iterative communication
        h1_comm, h2_comm = _iterative_comm(h1_full, h2_full, comm_params, dtype)

        h_blend = r * h1_comm + (1 - r) * h2_comm
        correction = F.linear(h_blend - h_mean, B_full.to(dtype)) * scaling

        if isinstance(output, tuple):
            return (output[0] + correction,) + output[1:]
        return output + correction

    return hook


# -- Load and shard corrections --------------------------------------------

def _load_v13_corrections(model_dir):
    """Load v13_correction_config.json and v13_correction_adapters.pt."""
    with open(os.path.join(model_dir, "v13_correction_config.json")) as f:
        config = json.load(f)
    corrections = torch.load(
        os.path.join(model_dir, "v13_correction_adapters.pt"),
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


def _extract_comm_params(corrections, layer_idx, num_comm_rounds, device):
    """Extract communication params for a layer as list of tuples."""
    T = num_comm_rounds
    params = []
    for t in range(T):
        idx = layer_idx * T + t
        W_12 = corrections.get(f"comm_W_12.{idx}", None)
        W_21 = corrections.get(f"comm_W_21.{idx}", None)
        gate_12 = corrections.get(f"comm_gate_12.{idx}", None)
        gate_21 = corrections.get(f"comm_gate_21.{idx}", None)
        if any(v is None for v in (W_12, W_21, gate_12, gate_21)):
            print(f"  WARNING: missing comm params for layer {layer_idx} round {t}")
            continue
        params.append((
            W_12.to(device), W_21.to(device),
            gate_12.to(device), gate_21.to(device),
        ))
    return params


# -- Model patching --------------------------------------------------------

def patch_model_with_v13_corrections(model, model_dir, device):
    """Apply V13 lossless correction hooks to a loaded vLLM model."""
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()

    config, corrections = _load_v13_corrections(model_dir)
    scaling = config["scaling"]
    target_modules = config["target_modules"]
    num_layers = config["num_layers"]
    lora_r = config["lora_r"]
    num_comm_rounds = config["num_comm_rounds"]

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

        # Extract per-layer communication params
        comm_params = _extract_comm_params(
            corrections, layer_idx, num_comm_rounds, device
        )

        # -- QKV (column-parallel, fused) ----------------------------------
        qkv_names = [n for n in ("q_proj", "k_proj", "v_proj")
                      if n in target_modules]
        qkv_adapters = []
        for name in qkv_names:
            bk = f"model.layers.{layer_idx}.self_attn.{name}.weight"
            A1_key = bk + ".A_1"
            A2_key = bk + ".A_2"
            B_key = bk + ".B"
            if all(k in corrections for k in (A1_key, A2_key, B_key)):
                qkv_adapters.append((
                    name,
                    corrections[A1_key],
                    corrections[A2_key],
                    corrections[B_key],
                ))

        if qkv_adapters and hasattr(layer.self_attn, "qkv_proj"):
            A1_list = [a1.to(device) for _, a1, _, _ in qkv_adapters]
            A2_list = [a2.to(device) for _, _, a2, _ in qkv_adapters]
            B_shards = [
                _shard_column_parallel(b, tp_rank, tp_size).to(device)
                for _, _, _, b in qkv_adapters
            ]

            hook = _make_qkv_hook(
                w_route, A1_list, A2_list, B_shards, scaling, comm_params
            )
            layer.self_attn.qkv_proj.register_forward_hook(hook)
            hooks_registered += 1

        # -- O_proj (row-parallel) -----------------------------------------
        if "o_proj" in target_modules:
            bk = f"model.layers.{layer_idx}.self_attn.o_proj.weight"
            A1_key = bk + ".A_1"
            A2_key = bk + ".A_2"
            B_key = bk + ".B"
            if (all(k in corrections for k in (A1_key, A2_key, B_key))
                    and hasattr(layer.self_attn, "o_proj")):
                A1_shard = _shard_row_parallel(
                    corrections[A1_key], tp_rank, tp_size
                ).to(device)
                A2_shard = _shard_row_parallel(
                    corrections[A2_key], tp_rank, tp_size
                ).to(device)
                B_full = corrections[B_key].to(device)

                hook = _make_o_proj_hook(
                    A1_shard, A2_shard, B_full, scaling, comm_params
                )
                layer.self_attn.o_proj.register_forward_hook(hook)
                hooks_registered += 1

    print(f"[TP rank {tp_rank}] Registered {hooks_registered} V13 correction hooks "
          f"across {num_layers} layers (scaling={scaling}, "
          f"comm_rounds={num_comm_rounds})")
    return hooks_registered


# -- Custom vLLM model class -----------------------------------------------

def create_v13_model_class():
    """Create a corrected Qwen2.5-VL model class with V13 iterative hooks."""
    from vllm.model_executor.models.qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration as _BaseQwen25VL,
    )

    class V13CorrectedQwen25VL(_BaseQwen25VL):
        """Qwen2.5-VL with lossless V13 iterative cooperative LoRA corrections."""

        def __init__(self, *, vllm_config, prefix=""):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            self._model_dir = vllm_config.model_config.model
            self._corrections_applied = False

        def load_weights(self, weights):
            result = super().load_weights(weights)
            if not self._corrections_applied:
                corr_path = os.path.join(self._model_dir,
                                          "v13_correction_adapters.pt")
                if os.path.exists(corr_path):
                    device = next(self.parameters()).device
                    n = patch_model_with_v13_corrections(
                        self, self._model_dir, device
                    )
                    print(f"Applied {n} V13 correction hooks")
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

    return V13CorrectedQwen25VL


# -- Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Launch vLLM with lossless V13 iterative cooperative corrections"
    )
    parser.add_argument("--model", required=True,
                        help="Path to merged model dir with v13_correction_adapters.pt")
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--limit-mm-per-prompt", type=str, default='{"image": 2}')
    args = parser.parse_args()

    # Verify correction files exist
    corr_path = os.path.join(args.model, "v13_correction_adapters.pt")
    if not os.path.exists(corr_path):
        print(f"ERROR: {corr_path} not found. Run v13_gui_360/merge_to_hf.py first.")
        sys.exit(1)

    # Register custom model class BEFORE importing vLLM server
    V13Model = create_v13_model_class()
    from vllm import ModelRegistry
    ModelRegistry.register_model(
        "Qwen2_5_VLForConditionalGeneration",
        V13Model,
    )
    print("Registered V13CorrectedQwen25VL with vLLM")

    # Build vLLM args -- enforce-eager required for conditional hooks
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
