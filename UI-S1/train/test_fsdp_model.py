#!/usr/bin/env python3
"""
Test FSDP model loading with srun-based distributed training.
This verifies that we can:
1. Initialize NCCL process group via srun/SLURM
2. Load a model with FSDP
3. Run a simple forward pass
"""

import os
import sys
import socket
import datetime
import logging

# Set NCCL environment variables BEFORE importing torch
os.environ.setdefault("NCCL_SOCKET_IFNAME", "hsn0")
os.environ.setdefault("GLOO_SOCKET_IFNAME", "hsn0")
os.environ.setdefault("NCCL_NET", "Socket")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("NCCL_P2P_LEVEL", "LOC")
os.environ.setdefault("NCCL_CROSS_NIC", "1")

import torch
import torch.distributed as dist

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_distributed():
    """Initialize distributed training using SLURM environment variables."""
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))

    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")

    hostname = socket.gethostname()
    logger.info(f"[{hostname}] Rank {rank}/{world_size}: Setting up distributed")

    # Set CUDA device
    device_count = torch.cuda.device_count()
    cuda_device = local_rank if device_count > 1 else 0
    torch.cuda.set_device(cuda_device)

    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(seconds=600),
            device_id=torch.device(f"cuda:{cuda_device}")
        )
        logger.info(f"[{hostname}] Rank {rank}: NCCL initialized!")

    return rank, world_size, local_rank


def test_fsdp_model(rank, world_size, local_rank, model_path):
    """Test loading a model with FSDP."""
    from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    import functools

    hostname = socket.gethostname()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        logger.info(f"Loading model config from {model_path}")

    # Load model config
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    if rank == 0:
        logger.info(f"Model config loaded. Creating model with meta device...")

    # Create model on meta device first (memory efficient)
    # This is a simplified approach - full implementation would load weights properly
    dist.barrier()

    # For testing, let's just create a small model to verify FSDP works
    # rather than loading the full 7B model
    logger.info(f"[{hostname}] Rank {rank}: Creating test model...")

    # Create a simple test model
    import torch.nn as nn

    class SimpleModel(nn.Module):
        def __init__(self, hidden_size=1024, num_layers=4):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
            ])
            self.activation = nn.ReLU()

        def forward(self, x):
            for layer in self.layers:
                x = self.activation(layer(x))
            return x

    model = SimpleModel(hidden_size=1024, num_layers=4).to(device)

    # Wrap with FSDP
    logger.info(f"[{hostname}] Rank {rank}: Wrapping model with FSDP...")

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        device_id=local_rank,
    )

    logger.info(f"[{hostname}] Rank {rank}: Model wrapped with FSDP!")

    # Test forward pass
    logger.info(f"[{hostname}] Rank {rank}: Testing forward pass...")
    batch_size = 4
    hidden_size = 1024
    x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output = model(x)

    logger.info(f"[{hostname}] Rank {rank}: Forward pass output shape: {output.shape}")

    # Test backward pass
    logger.info(f"[{hostname}] Rank {rank}: Testing backward pass...")
    loss = output.sum()
    loss.backward()

    # Test gradient synchronization
    logger.info(f"[{hostname}] Rank {rank}: Testing gradient sync...")
    for name, param in model.named_parameters():
        if param.grad is not None:
            # All-reduce gradient norms to verify sync
            grad_norm = param.grad.norm()
            dist.all_reduce(grad_norm)
            if rank == 0:
                logger.info(f"Param {name}: grad_norm (sum across ranks) = {grad_norm.item():.4f}")
            break

    dist.barrier()
    logger.info(f"[{hostname}] Rank {rank}: All FSDP tests passed!")

    return True


def main():
    rank, world_size, local_rank = setup_distributed()

    model_path = os.environ.get("MODEL_PATH", "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/checkpoints/Qwen2.5-VL-7B-Instruct")

    try:
        success = test_fsdp_model(rank, world_size, local_rank, model_path)
        if success:
            hostname = socket.gethostname()
            logger.info(f"[{hostname}] Rank {rank}: SUCCESS - FSDP test completed!")
    except Exception as e:
        logger.error(f"Rank {rank}: FSDP test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
