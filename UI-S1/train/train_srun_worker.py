#!/usr/bin/env python3
"""
Srun-based distributed training worker for UI-S1.
This bypasses Ray and uses PyTorch native distributed training via srun.

Usage:
    srun --nodes=$SLURM_NNODES --ntasks-per-node=4 python train_srun_worker.py --config-path=... --config-name=...
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
os.environ.setdefault("NCCL_DEBUG", "INFO")
os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT,NET")
os.environ.setdefault("NCCL_P2P_LEVEL", "LOC")
os.environ.setdefault("NCCL_CROSS_NIC", "1")

import torch
import torch.distributed as dist
import hydra
from omegaconf import DictConfig, OmegaConf
from verl.utils.tracking import Tracking

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_distributed():
    """Initialize distributed training using SLURM environment variables."""
    # Get distributed training parameters from environment
    # These are set by srun based on SLURM allocation
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 4))

    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")

    hostname = socket.gethostname()

    logger.info(f"[{hostname}] Rank {rank}/{world_size}: Starting distributed setup")
    logger.info(f"[{hostname}] Rank {rank}: MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
    logger.info(f"[{hostname}] Rank {rank}: LOCAL_RANK={local_rank}, LOCAL_WORLD_SIZE={local_world_size}")

    # Log NCCL configuration
    nccl_vars = {k: os.environ[k] for k in os.environ if k.startswith("NCCL")}
    logger.info(f"[{hostname}] Rank {rank}: NCCL config: {nccl_vars}")

    # Set CUDA device based on local_rank
    device_count = torch.cuda.device_count()
    logger.info(f"[{hostname}] Rank {rank}: CUDA device count: {device_count}")

    if device_count == 0:
        raise RuntimeError(f"No CUDA GPUs available on {hostname}")

    # Use local_rank to select GPU
    cuda_device = local_rank if device_count > 1 else 0
    torch.cuda.set_device(cuda_device)
    logger.info(f"[{hostname}] Rank {rank}: Using CUDA device {cuda_device}")

    # Initialize process group
    if not dist.is_initialized():
        # Set required environment variables for env:// init method
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)

        init_timeout = datetime.timedelta(seconds=600)

        logger.info(f"[{hostname}] Rank {rank}: Initializing NCCL process group...")

        try:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=world_size,
                rank=rank,
                timeout=init_timeout,
                device_id=torch.device(f"cuda:{cuda_device}")
            )
            logger.info(f"[{hostname}] Rank {rank}: NCCL process group initialized successfully!")
        except Exception as e:
            logger.error(f"[{hostname}] Rank {rank}: Failed to initialize NCCL: {e}")
            raise

    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config: DictConfig):
    """Main training entry point."""
    rank, world_size, local_rank = setup_distributed()

    try:
        # Only rank 0 prints the full config
        if rank == 0:
            logger.info("Configuration:")
            logger.info(OmegaConf.to_yaml(config))

        # Synchronize all processes before starting training
        dist.barrier()
        logger.info(f"Rank {rank}: All processes synchronized, starting training...")

        # Run the training
        run_training(config, rank, world_size, local_rank)

    except Exception as e:
        logger.error(f"Rank {rank}: Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        cleanup_distributed()


def run_training(config: DictConfig, rank: int, world_size: int, local_rank: int):
    """Run the actual training loop."""
    from pprint import pprint
    from omegaconf import OmegaConf

    OmegaConf.resolve(config)

    # Download checkpoint to local if needed
    from verl.utils.fs import copy_to_local
    local_path = copy_to_local(
        config.actor_rollout_ref.model.path,
        use_shm=config.actor_rollout_ref.model.get("use_shm", False)
    )

    # Load tokenizer and processor
    from verl.utils import hf_processor, hf_tokenizer
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

    if rank == 0:
        logger.info(f"Loaded tokenizer and processor from {local_path}")

    # Create datasets
    train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
    val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)

    if rank == 0:
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Val dataset size: {len(val_dataset)}")

    # Create the training components
    # For srun-based training, we use a simplified FSDP setup
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    # Create actor worker (which handles FSDP wrapping internally)
    # Note: The worker's __init__ expects dist to already be initialized, which we've done
    actor_config = config.actor_rollout_ref

    # The worker class expects certain config structure
    # We need to create a minimal trainer that coordinates across ranks

    trainer = SrunTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )

    trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a simple dataset for testing multi-node training."""
    from torch.utils.data import Dataset
    import json

    class SimpleTextDataset(Dataset):
        """Simple dataset that returns tokenized text for SFT-style training."""

        def __init__(self, data_files, tokenizer, max_length=512):
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.data = []

            # Load data from jsonl files
            if isinstance(data_files, str):
                data_files = [data_files]

            for data_file in data_files:
                with open(data_file, 'r') as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            self.data.append(item)
                        except:
                            continue

            logger.info(f"Loaded {len(self.data)} samples from {data_files}")

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]

            # Extract text from the data structure
            # For trajectory data, we take the first step's content as input
            text = ""
            if 'steps' in item:
                for step in item['steps'][:1]:  # Just use first step for simplicity
                    if 'action_content' in step:
                        text = str(step['action_content'])
                        break
            elif 'prompt' in item:
                text = item['prompt']
            elif 'text' in item:
                text = item['text']
            else:
                text = str(item)[:1000]  # Fallback

            # Tokenize
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': encoding['input_ids'].squeeze(0),  # For SFT, labels = input_ids
            }

    dataset = SimpleTextDataset(
        data_files=data_paths,
        tokenizer=tokenizer,
        max_length=data_config.get('max_prompt_length', 512),
    )

    return dataset


class SrunTrainer:
    """
    Simplified trainer for srun-based distributed training.
    This replaces Ray-based coordination with PyTorch distributed primitives.
    """

    def __init__(
        self,
        config: DictConfig,
        tokenizer,
        processor,
        train_dataset,
        val_dataset,
        rank: int,
        world_size: int,
        local_rank: int,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")

        # Initialize model and optimizer
        self._init_model()
        self._init_optimizer()
        self._init_dataloader()
        self._init_tracking()

    def _init_tracking(self):
        """Initialize tracking/logging (only on rank 0)."""
        self.tracker = None
        if self.rank == 0:
            project_name = self.config.trainer.get("project_name", "ui-s1-training")
            experiment_name = self.config.trainer.get("experiment_name", "srun_train")
            # Get logger backends from config, default to console and wandb
            logger_backends = self.config.trainer.get("logger", ["console", "wandb"])
            if isinstance(logger_backends, str):
                logger_backends = [logger_backends]

            self.tracker = Tracking(
                project_name=project_name,
                experiment_name=experiment_name,
                default_backend=logger_backends,
                config=OmegaConf.to_container(self.config, resolve=True),
            )
            logger.info(f"Tracking initialized: project={project_name}, run={experiment_name}, backends={logger_backends}")

    def _init_model(self):
        """Initialize the model with FSDP."""
        from verl.utils.fs import copy_to_local
        from transformers import AutoModelForCausalLM, AutoConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        import functools

        model_path = copy_to_local(
            self.config.actor_rollout_ref.model.path,
            use_shm=self.config.actor_rollout_ref.model.get("use_shm", False)
        )

        if self.rank == 0:
            logger.info(f"Loading model from {model_path}")

        # Load model directly with pretrained weights (each rank loads independently)
        # This is memory-intensive but simpler than meta device approach
        from transformers import Qwen2_5_VLForConditionalGeneration

        if self.rank == 0:
            logger.info("Loading pretrained model weights...")

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Enable gradient checkpointing to save memory
        if self.config.actor_rollout_ref.model.get("enable_gradient_checkpointing", True):
            model.gradient_checkpointing_enable()
            if self.rank == 0:
                logger.info("Gradient checkpointing enabled")

        if self.rank == 0:
            logger.info("Model loaded, wrapping with FSDP...")

        # Get FSDP wrap policy for transformer layers
        # NOTE: Use Qwen2_5_VLDecoderLayer (not Qwen2VLDecoderLayer) to match the model
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Qwen2_5_VLDecoderLayer},
        )

        # FSDP configuration
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        # Check if CPU offloading is enabled
        param_offload = self.config.actor_rollout_ref.actor.fsdp_config.get("param_offload", False)
        if param_offload:
            from torch.distributed.fsdp import CPUOffload
            cpu_offload = CPUOffload(offload_params=True)
            if self.rank == 0:
                logger.info("FSDP CPU parameter offloading enabled")
        else:
            cpu_offload = None

        # Wrap with FSDP - don't sync since all ranks loaded the same weights
        self.model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            auto_wrap_policy=auto_wrap_policy,
            device_id=self.local_rank,
            sync_module_states=False,  # All ranks loaded same weights
            cpu_offload=cpu_offload,
        )

        if self.rank == 0:
            logger.info("Model wrapped with FSDP successfully")

        dist.barrier()

    def _init_optimizer(self):
        """Initialize optimizer."""
        lr = self.config.actor_rollout_ref.actor.optim.lr
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        if self.rank == 0:
            logger.info(f"Initialized AdamW optimizer with lr={lr}")

    def _init_dataloader(self):
        """Initialize data loaders."""
        from torch.utils.data import DataLoader, DistributedSampler
        from torch.utils.data.dataloader import default_collate

        train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )

        # Use micro batch size of 1 for long sequences to avoid OOM
        batch_size = self.config.actor_rollout_ref.actor.get("ppo_micro_batch_size_per_gpu", 1)

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=default_collate,  # Use PyTorch's default collate
            num_workers=2,
            pin_memory=True,
        )

        if self.rank == 0:
            logger.info(f"DataLoader initialized with batch_size={batch_size} per rank")

    def _compute_grad_norm(self) -> float:
        """Compute the gradient norm across all model parameters."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def fit(self):
        """Run the training loop."""
        import time

        total_epochs = self.config.trainer.total_epochs
        total_batches_per_epoch = len(self.train_dataloader)
        total_training_steps = total_epochs * total_batches_per_epoch

        if self.rank == 0:
            logger.info(f"Starting training for {total_epochs} epochs")
            logger.info(f"Total batches per epoch: {total_batches_per_epoch}")
            logger.info(f"Total training steps: {total_training_steps}")

        global_step = 0
        for epoch in range(total_epochs):
            self.train_dataloader.sampler.set_epoch(epoch)

            self.model.train()
            total_loss = 0.0
            num_batches = 0
            epoch_start_time = time.time()
            total_tokens = 0

            for batch_idx, batch in enumerate(self.train_dataloader):
                batch_start_time = time.time()

                # Move batch to device - simple dataset returns input_ids, attention_mask, labels
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Compute sequence length stats
                seq_lengths = attention_mask.sum(dim=-1).float()
                batch_tokens = int(attention_mask.sum().item())
                total_tokens += batch_tokens

                if self.rank == 0 and batch_idx == 0:
                    logger.info(f"Batch keys: {list(batch.keys())}")
                    logger.info(f"Input shape: {input_ids.shape}")

                # Forward pass
                self.optimizer.zero_grad()

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                # Backward pass
                loss.backward()

                # Compute gradient norm before optimizer step
                grad_norm = self._compute_grad_norm()

                self.optimizer.step()

                batch_time = time.time() - batch_start_time
                total_loss += loss.item()
                num_batches += 1
                global_step += 1

                if batch_idx % 10 == 0 and self.rank == 0:
                    # Compute throughput
                    tokens_per_sec = batch_tokens / batch_time if batch_time > 0 else 0

                    logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, "
                               f"Grad Norm: {grad_norm:.4f}, Tokens/sec: {tokens_per_sec:.1f}")

                    # Log comprehensive metrics to tracking backends
                    if self.tracker:
                        metrics = {
                            # Training metrics
                            "train/loss": loss.item(),
                            "train/epoch": epoch,
                            "train/batch": batch_idx,
                            "train/global_step": global_step,
                            "train/learning_rate": self.optimizer.param_groups[0]['lr'],

                            # Gradient metrics
                            "train/grad_norm": grad_norm,

                            # Sequence length metrics
                            "data/seq_length_mean": seq_lengths.mean().item(),
                            "data/seq_length_max": seq_lengths.max().item(),
                            "data/seq_length_min": seq_lengths.min().item(),
                            "data/batch_tokens": batch_tokens,

                            # Timing/throughput metrics
                            "perf/batch_time_s": batch_time,
                            "perf/tokens_per_sec": tokens_per_sec,

                            # Progress metrics
                            "progress/epoch_progress": batch_idx / total_batches_per_epoch,
                            "progress/total_progress": global_step / total_training_steps,
                        }

                        # Add GPU memory metrics if available
                        if torch.cuda.is_available():
                            metrics["perf/gpu_memory_allocated_gb"] = torch.cuda.memory_allocated(self.device) / 1e9
                            metrics["perf/gpu_memory_reserved_gb"] = torch.cuda.memory_reserved(self.device) / 1e9

                        self.tracker.log(metrics, step=global_step)

            # Synchronize and log epoch stats
            epoch_time = time.time() - epoch_start_time
            avg_loss = total_loss / max(num_batches, 1)

            # All-reduce to get global average loss
            loss_tensor = torch.tensor([avg_loss], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)

            # All-reduce to get global total tokens
            tokens_tensor = torch.tensor([total_tokens], device=self.device, dtype=torch.float)
            dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
            global_tokens = int(tokens_tensor.item())

            if self.rank == 0:
                epoch_throughput = global_tokens / epoch_time if epoch_time > 0 else 0
                logger.info(f"Epoch {epoch} completed. Average loss: {loss_tensor.item():.4f}, "
                           f"Time: {epoch_time:.1f}s, Throughput: {epoch_throughput:.1f} tokens/s")

                # Log epoch metrics to tracking backends
                if self.tracker:
                    epoch_metrics = {
                        "epoch/avg_loss": loss_tensor.item(),
                        "epoch/epoch": epoch,
                        "epoch/time_s": epoch_time,
                        "epoch/total_tokens": global_tokens,
                        "epoch/throughput_tokens_per_sec": epoch_throughput,
                        "epoch/num_batches": num_batches,
                    }
                    self.tracker.log(epoch_metrics, step=global_step)

            # Save checkpoint
            if self.rank == 0 and (epoch + 1) % self.config.trainer.get("save_freq", 1) == 0:
                self._save_checkpoint(epoch)

        if self.rank == 0:
            logger.info("Training completed!")
            # Tracking cleanup is handled by __del__ method

    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_dir = self.config.trainer.get("checkpoint_dir", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")

        # Use FSDP's state dict saving
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
            state_dict = self.model.state_dict()
            if self.rank == 0:
                torch.save(state_dict, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
