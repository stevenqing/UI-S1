# NCCL Multi-Node Investigation Summary

## Problem
Multi-node distributed initialization hangs when using verl with Ray on HPE Slingshot GH200 nodes.
- Intra-node communication works (ranks on same node complete init_process_group)
- Cross-node communication hangs (ranks on different nodes stuck at init_process_group)
- This affects BOTH NCCL and Gloo backends

## ✅ SOLUTION FOUND: srun-Based Training

**The issue is solved by bypassing Ray and using `srun` directly for distributed training.**

### How to Run Multi-Node Training

```bash
# Submit the srun-based training job
sbatch train/train_srun.slurm
```

### Key Changes Required

1. **Load CUDA module** in SLURM scripts:
   ```bash
   module load cuda/12.6
   ```

2. **Use SLURM environment variables** for distributed setup:
   ```bash
   export MASTER_ADDR=$head_node_ip
   export MASTER_PORT=29500
   export WORLD_SIZE=$SLURM_NTASKS
   export RANK=$SLURM_PROCID
   export LOCAL_RANK=$SLURM_LOCALID
   ```

3. **Initialize PyTorch distributed with env://**:
   ```python
   torch.distributed.init_process_group(
       backend="nccl",
       init_method="env://",
       world_size=world_size,
       rank=rank,
       device_id=torch.device(f"cuda:{local_rank}")
   )
   ```

### Verified Working

**FSDP Test (Job 2089338)** - All 8 ranks across 2 nodes:
- ✅ NCCL initialization completed
- ✅ FSDP model wrapping
- ✅ Forward pass
- ✅ Backward pass
- ✅ Gradient synchronization across nodes

**Full Training (Job 2089521)** - Qwen2.5-VL-7B with FSDP:
- ✅ Multi-node NCCL initialization
- ✅ 7B model loaded and FSDP wrapped
- ✅ Training loop running
- ✅ Loss decreasing (16.54 → 11.18)

```
[2026-01-31 12:52:10,920][__main__][INFO] - Starting training for 1 epochs
[2026-01-31 12:52:10,981][__main__][INFO] - Batch keys: ['input_ids', 'attention_mask', 'labels']
[2026-01-31 12:52:10,981][__main__][INFO] - Input shape: torch.Size([2, 1024])
[2026-01-31 12:52:15,895][__main__][INFO] - Epoch 0, Batch 0, Loss: 16.5443
[2026-01-31 12:52:59,764][__main__][INFO] - Epoch 0, Batch 10, Loss: 11.1768
```

## Root Cause Analysis

The difference is in how processes are launched:

| Aspect | srun (Works ✅) | Ray (Hangs ❌) |
|--------|------|-----|
| Process launch | Coordinated MPI-style spawn | Independent actor spawning |
| Timing | Near-simultaneous | Asynchronous/staggered |
| Network coordination | MPI/PMI environment | TCPStore only |
| Network binding | SLURM handles | Application must configure |

The core issue is that Ray-spawned workers cannot establish bidirectional peer-to-peer connections for distributed collective operations. SLURM's PMI (Process Management Interface) sets up proper network coordination that enables bidirectional communication. Ray-spawned actors don't have this coordination.

## Files Created for srun-Based Training

### `train/train_srun.slurm`
Main SLURM script for srun-based training:
```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

module load cuda/12.6

srun --nodes=$SLURM_NNODES --ntasks-per-node=4 \
    bash -c "
        module load cuda/12.6
        source /path/to/env_config.sh

        export MASTER_ADDR=$head_node_ip
        export MASTER_PORT=29500
        export WORLD_SIZE=\$SLURM_NTASKS
        export RANK=\$SLURM_PROCID
        export LOCAL_RANK=\$SLURM_LOCALID

        python train_srun_worker.py --config-path=... --config-name=...
    "
```

### `train/train_srun_worker.py`
Entry point for srun-based distributed training that:
- Initializes NCCL using SLURM environment variables
- Sets up FSDP model wrapping
- Runs training loop with proper distributed synchronization

### `train/test_fsdp_model.py`
Test script to verify FSDP works with srun:
```bash
sbatch train/test_fsdp.slurm
```

### `train/test_nccl_minimal.py`
Minimal NCCL test to verify multi-node communication:
```bash
sbatch train/test_torchrun.slurm
```

## NCCL Configuration

### `train/env_config.sh`
```bash
export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=LOC
export NCCL_CROSS_NIC=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
```

### Key Settings Explained
- `NCCL_SOCKET_IFNAME=hsn0`: Use HPE Slingshot high-speed network interface
- `NCCL_NET=Socket`: Force socket transport (bypass InfiniBand)
- `NCCL_IB_DISABLE=1`: Disable InfiniBand (not available on this cluster)
- `NCCL_P2P_LEVEL=LOC`: Limit P2P to local node only
- `NCCL_CROSS_NIC=1`: Allow cross-NIC communication

## Previous Attempts (All Failed with Ray)

1. **NCCL env vars before torch import** - Set in `verl/__init__.py`
2. **Sync barrier before init_process_group** - Ensure all ranks ready via TCPStore
3. **NCCL_SOCKET_IFNAME=hsn0** - Force NCCL to use high-speed network
4. **GLOO_SOCKET_IFNAME=hsn0** - Force Gloo to use high-speed network
5. **TCPStore with explicit creation** - Same hang pattern
6. **FileStore rendezvous** - Same hang pattern
7. **init_method="env://"** - Same hang pattern

**Conclusion**: The issue is fundamental to Ray's actor spawning, not NCCL configuration.

## Diagnostic Commands

Check network interfaces:
```bash
ip -o -4 addr show hsn0
```

Test minimal NCCL (via srun):
```bash
sbatch train/test_torchrun.slurm
```

Test FSDP model loading:
```bash
sbatch train/test_fsdp.slurm
```

Run full training:
```bash
sbatch train/train_srun.slurm
```

Check job status:
```bash
squeue -u $USER
```

View logs:
```bash
tail -f train/logs/train_srun_*.log
```

## Summary

| Test | Result |
|------|--------|
| Single-node NCCL (Ray) | ✅ Works |
| Multi-node NCCL (Ray) | ❌ Hangs |
| Multi-node NCCL (srun) | ✅ Works |
| Multi-node Gloo (Ray) | ❌ Hangs |
| Multi-node FSDP (srun) | ✅ Works |

**Solution**: Use srun-based training scripts instead of Ray for multi-node distributed training on this cluster.
