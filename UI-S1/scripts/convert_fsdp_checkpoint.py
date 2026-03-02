#!/usr/bin/env python3
"""
Convert FSDP sharded checkpoints to HuggingFace format.

This script merges the sharded model weights from FSDP training
into a single HuggingFace-compatible checkpoint.
"""

import argparse
import os
import torch
from collections import OrderedDict
from transformers import AutoModelForVision2Seq, AutoProcessor

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor


def dtensor_to_local(tensor):
    """Convert DTensor to local tensor."""
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def get_shard_dim_and_size(full_shape, local_shape):
    """Find the dimension that is sharded and its full size."""
    for dim in range(len(full_shape)):
        if full_shape[dim] != local_shape[dim]:
            return dim, full_shape[dim]
    return None, None


def load_sharded_checkpoint(checkpoint_dir: str, world_size: int = 16) -> OrderedDict:
    """
    Load and merge sharded model weights from FSDP DTensor format.

    Args:
        checkpoint_dir: Directory containing the sharded checkpoint files
        world_size: Number of shards (processes used during training)

    Returns:
        Merged state dict with regular tensors
    """
    print(f"Loading sharded checkpoint from {checkpoint_dir}")
    print(f"World size: {world_size}")

    # First, load rank 0 to get the full shapes
    shard0_path = os.path.join(checkpoint_dir, f"model_world_size_{world_size}_rank_0.pt")
    print(f"Loading rank 0 to get shapes: {shard0_path}")
    shard0_state = torch.load(shard0_path, map_location="cpu", weights_only=False)

    # Collect shard info for each parameter
    param_info = {}
    for key, value in shard0_state.items():
        if isinstance(value, DTensor):
            full_shape = tuple(value.shape)
            local_shape = tuple(value.to_local().shape)
            shard_dim, full_size = get_shard_dim_and_size(full_shape, local_shape)
            param_info[key] = {
                'full_shape': full_shape,
                'local_shape': local_shape,
                'shard_dim': shard_dim,
                'dtype': value.to_local().dtype
            }
        else:
            param_info[key] = {
                'full_shape': tuple(value.shape) if hasattr(value, 'shape') else None,
                'local_shape': tuple(value.shape) if hasattr(value, 'shape') else None,
                'shard_dim': None,
                'dtype': value.dtype if hasattr(value, 'dtype') else None
            }

    # Initialize storage for local shards
    local_shards = {key: [] for key in shard0_state.keys()}

    # Load all shards and collect local tensors
    for rank in range(world_size):
        shard_path = os.path.join(checkpoint_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Shard not found: {shard_path}")

        print(f"Loading shard {rank + 1}/{world_size}: {shard_path}")
        shard_state = torch.load(shard_path, map_location="cpu", weights_only=False)

        for key, value in shard_state.items():
            local_tensor = dtensor_to_local(value)
            local_shards[key].append(local_tensor)

    # Merge shards
    print("\nMerging shards...")
    merged_state_dict = OrderedDict()

    for key in shard0_state.keys():
        info = param_info[key]
        shards = local_shards[key]

        if info['shard_dim'] is not None:
            # Concatenate along the sharded dimension
            merged_state_dict[key] = torch.cat(shards, dim=info['shard_dim'])
        else:
            # Not sharded (replicated) - just use the first one
            merged_state_dict[key] = shards[0]

        if hasattr(merged_state_dict[key], 'shape'):
            expected_shape = info['full_shape']
            actual_shape = tuple(merged_state_dict[key].shape)
            if expected_shape != actual_shape:
                print(f"Warning: Shape mismatch for {key}: expected {expected_shape}, got {actual_shape}")

    print(f"\nMerged state dict has {len(merged_state_dict)} keys")
    return merged_state_dict


def convert_checkpoint(
    checkpoint_dir: str,
    base_model_path: str,
    output_dir: str,
    world_size: int = 16
):
    """
    Convert FSDP checkpoint to HuggingFace format.

    Args:
        checkpoint_dir: Directory containing actor/model_*.pt files
        base_model_path: Path to base model for config and processor
        output_dir: Output directory for HuggingFace checkpoint
    """
    actor_dir = os.path.join(checkpoint_dir, "actor")
    if not os.path.exists(actor_dir):
        actor_dir = checkpoint_dir  # Direct path to actor dir

    print(f"Actor directory: {actor_dir}")
    print(f"Base model: {base_model_path}")
    print(f"Output directory: {output_dir}")

    # Load processor from base model
    print("\nLoading processor from base model...")
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

    # Load model config from checkpoint if available, otherwise from base model
    config_path = os.path.join(actor_dir, "config.json")
    if os.path.exists(config_path):
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(actor_dir, trust_remote_code=True)
        print(f"Loaded config from checkpoint: {config_path}")
    else:
        config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
        print(f"Loaded config from base model: {base_model_path}")

    # Load and merge sharded weights
    print("\nLoading sharded weights...")
    merged_state_dict = load_sharded_checkpoint(actor_dir, world_size)

    # Create model and load weights
    print("\nCreating model and loading weights...")
    model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    # Load the merged state dict
    missing, unexpected = model.load_state_dict(merged_state_dict, strict=False)
    if missing:
        print(f"Warning: Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"Warning: Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    # Save to HuggingFace format
    print(f"\nSaving model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    print(f"\nCheckpoint converted successfully!")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert FSDP checkpoint to HuggingFace format")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to FSDP checkpoint directory (e.g., global_step_93)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/checkpoints/Qwen2.5-VL-7B-Instruct",
        help="Path to base model for config and processor"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for HuggingFace checkpoint"
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=16,
        help="Number of shards (world size during training)"
    )

    args = parser.parse_args()

    convert_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        base_model_path=args.base_model,
        output_dir=args.output_dir,
        world_size=args.world_size
    )


if __name__ == "__main__":
    main()
