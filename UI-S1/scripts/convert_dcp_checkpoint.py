#!/usr/bin/env python3
"""
Convert DCP (Distributed Checkpoint) to HuggingFace format.

DCP checkpoints are sharded across multiple files and require a distributed
environment to load. This script converts them to standard HuggingFace format
that can be used for inference.

Usage:
    # Single GPU conversion (for smaller models)
    python scripts/convert_dcp_checkpoint.py \
        --dcp-path checkpoints/global_step_100 \
        --hf-path checkpoints/global_step_100_hf \
        --model-path checkpoints/Qwen2.5-VL-7B-Instruct

    # Multi-GPU conversion (recommended for 7B+ models)
    torchrun --nproc_per_node=4 scripts/convert_dcp_checkpoint.py \
        --dcp-path checkpoints/global_step_100 \
        --hf-path checkpoints/global_step_100_hf \
        --model-path checkpoints/Qwen2.5-VL-7B-Instruct

    # SLURM multi-node conversion
    srun --nodes=1 --ntasks-per-node=4 --gres=gpu:4 python scripts/convert_dcp_checkpoint.py ...
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


def setup_distributed():
    """Initialize distributed environment."""
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()

    # Check if running under torchrun/srun
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
    else:
        # Single GPU mode
        rank = 0
        world_size = 1
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    return rank, world_size


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def convert_dcp_to_hf(dcp_path: str, hf_path: str, model_path: str, trust_remote_code: bool = True):
    """Convert DCP checkpoint to HuggingFace format.

    Args:
        dcp_path: Path to DCP checkpoint directory
        hf_path: Output path for HuggingFace checkpoint
        model_path: Path to original model (for config/architecture)
        trust_remote_code: Whether to trust remote code when loading model
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
    )
    from transformers import AutoConfig, AutoModelForVision2Seq, AutoTokenizer

    rank, world_size = setup_distributed()

    if rank == 0:
        print(f"Converting DCP checkpoint: {dcp_path}")
        print(f"Output HuggingFace path: {hf_path}")
        print(f"Model architecture from: {model_path}")
        print(f"World size: {world_size}")

    # Verify DCP checkpoint exists
    dcp_marker = os.path.join(dcp_path, ".dcp_checkpoint")
    if not os.path.exists(dcp_marker):
        if rank == 0:
            print(f"Warning: {dcp_path} does not appear to be a DCP checkpoint (missing .dcp_checkpoint marker)")
            print("Attempting to load anyway...")

    # Load model config and create model
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    # Detect if vision model
    is_vision_model = (
        hasattr(config, "vision_config")
        or "qwen2_vl" in config.__class__.__name__.lower()
        or "qwen2.5_vl" in config.__class__.__name__.lower()
    )

    if rank == 0:
        print(f"Loading model architecture (vision_model={is_vision_model})...")

    # Load model without initializing weights to save memory
    # Keep on CPU initially, FSDP will handle device placement
    from transformers import modeling_utils

    with modeling_utils.no_init_weights():
        if is_vision_model:
            model = AutoModelForVision2Seq.from_config(
                config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=trust_remote_code,
            )
        else:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=trust_remote_code,
            )

    # Initialize device mesh for FSDP2
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    if world_size > 1:
        device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))

        # Apply FSDP2 wrapping
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard

        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

        # Find transformer layers - handle different model architectures
        # Qwen2.5-VL: model.model.layers (the inner model has layers)
        # Standard models: model.model.layers
        layers = None
        if hasattr(model, "model"):
            if hasattr(model.model, "layers"):
                layers = model.model.layers
            elif hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
                # Qwen2.5-VL structure: model.model.model.layers
                layers = model.model.model.layers

        if layers is not None:
            if rank == 0:
                print(f"Wrapping {len(layers)} transformer layers with FSDP...")
            for layer in layers:
                fully_shard(layer, mesh=device_mesh, mp_policy=mp_policy)
        else:
            if rank == 0:
                print("Warning: Could not find transformer layers, applying FSDP to full model only")

        fully_shard(model, mesh=device_mesh, mp_policy=mp_policy)
    else:
        # Single GPU - just move to device
        model = model.to(device_name)

    if rank == 0:
        print("Loading DCP checkpoint...")

    # Load DCP checkpoint
    model_options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    model_state_dict = get_model_state_dict(model, options=model_options)
    state_dict = {"model": model_state_dict}

    dcp.load(state_dict, checkpoint_id=dcp_path)
    set_model_state_dict(model, state_dict["model"], options=model_options)

    if rank == 0:
        print("Gathering full state dict (this may take a while for large models)...")

    # Gather full state dict
    full_options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    full_state_dict = get_model_state_dict(model, options=full_options)

    # Save on rank 0 only
    if rank == 0:
        print(f"Saving HuggingFace checkpoint to {hf_path}...")
        os.makedirs(hf_path, exist_ok=True)

        # Get the underlying model if wrapped
        unwrapped_model = model.module if hasattr(model, "module") else model

        # Save model
        unwrapped_model.save_pretrained(hf_path, state_dict=full_state_dict)

        # Save config
        config.save_pretrained(hf_path)

        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        tokenizer.save_pretrained(hf_path)

        # Copy preprocessor config for vision models (needed by vLLM)
        import shutil
        preprocessor_config = os.path.join(model_path, "preprocessor_config.json")
        if os.path.exists(preprocessor_config):
            shutil.copy(preprocessor_config, os.path.join(hf_path, "preprocessor_config.json"))
            print("Copied preprocessor_config.json for vision model")

        print(f"Successfully saved HuggingFace checkpoint to {hf_path}")

    if world_size > 1:
        dist.barrier()

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="Convert DCP checkpoint to HuggingFace format")
    parser.add_argument("--dcp-path", type=str, required=True, help="Path to DCP checkpoint directory")
    parser.add_argument("--hf-path", type=str, required=True, help="Output path for HuggingFace checkpoint")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to original model (for loading architecture)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code when loading model",
    )

    args = parser.parse_args()

    convert_dcp_to_hf(
        dcp_path=args.dcp_path,
        hf_path=args.hf_path,
        model_path=args.model_path,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
