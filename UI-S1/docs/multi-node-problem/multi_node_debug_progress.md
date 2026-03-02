# Multi-Node Training Debug Progress

## Summary

We are debugging a hang issue in multi-node distributed training using the verl framework with Ray and FSDP.

## Issues Fixed

### 1. Custom Dataset Path (Fixed)
- **Problem**: `FileNotFoundError: Custom type file 'verl/utils/dataset/rl_dataset.py' not found`
- **Solution**: Changed relative path to absolute path in `examples/qwen_gui_static_grpo/config/traj_grpo.yaml`
```yaml
custom_cls:
  path: /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/verl/utils/dataset/rl_dataset.py
  name: TrajDataset
```

### 2. CUDA Device Ordinal Error (Fixed)
- **Problem**: `RuntimeError: CUDA error: invalid device ordinal` when calling `torch.cuda.set_device(local_rank)`
- **Root Cause**: Ray sets `CUDA_VISIBLE_DEVICES` to limit each worker to 1 GPU, but code was computing `local_rank = rank % 4` which could be > 0
- **Solution**: In `verl/workers/fsdp_workers.py`, check `device_count` and use device 0 if only 1 GPU is visible:
```python
if device_count == 1:
    local_rank = 0
else:
    n_gpus_per_node = int(os.environ.get("VERL_N_GPUS_PER_NODE", device_count))
    local_rank = rank % n_gpus_per_node
```

### 3. Gloo Backend Issues (Fixed)
- **Problem**: Composite backend `cpu:gloo,cuda:nccl` caused Gloo TCP issues across nodes
- **Solution**: Changed to pure NCCL backend in `verl/workers/fsdp_workers.py`

## Current Issue (Updated 2026-01-31 02:20)

### `init_process_group` Hang - All Ranks

**Symptoms**:
- All 8 ranks successfully create TCPStore
- All 8 ranks reach `init_process_group` call
- No ranks complete `init_process_group` - all hang

**Jobs Tested**: 2081845, 2081848, 2081849, 2081852, 2081853, 2081855, 2081867, 2081870, 2081879

**Error Message** (from Job 2081855 with timeout):
```
[5] is setting up NCCL communicator and retrieving ncclUniqueId from [0] via c10d key-value store by key '0', but store->get('0') got error: wait timeout after 300000ms
```

This indicates that NCCL is trying to get the ncclUniqueId from rank 0 via the TCPStore, but it's timing out.

**Key Findings**:
1. TCP connectivity works - all ranks can connect to MASTER_ADDR:MASTER_PORT
2. TCPStore creation works for all ranks
3. The issue is in the NCCL communicator setup phase, not the PyTorch rendezvous
4. Ranks 4-7 sometimes complete but ranks 0-3 don't (or vice versa)
5. P2P_DISABLE and SHM_DISABLE don't help

## Current NCCL Configuration

```python
os.environ["NCCL_SOCKET_IFNAME"] = "hsn"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,NET"
os.environ["NCCL_IB_TIMEOUT"] = "50"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_NET"] = "Socket"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"
```

## Code Changes Made (in verl/workers/fsdp_workers.py)

1. Force NCCL environment variables (don't check if already set)
2. Use explicit TCPStore instead of env:// init_method
3. Add device_id parameter to init_process_group
4. Add connectivity test before init
5. Add detailed diagnostic logging with hostname

## Hypothesis

The issue might be related to:
1. NCCL socket binding/listening on the correct interface
2. Firewall or network security blocking NCCL traffic (different from TCP)
3. CUDA/NCCL version incompatibility with the specific GPU/driver
4. Some NCCL internal issue with the GH200 platform

## Files Modified

1. **`verl/workers/fsdp_workers.py`**
   - Added diagnostic logging
   - Fixed CUDA device selection for Ray workers
   - Changed to pure NCCL backend
   - Added NCCL_SOCKET_IFNAME configuration

2. **`verl/trainer/main_dapo.py`**
   - Fixed project_name condition check

3. **`train/train_ui_s1_test_offload.slurm`**
   - Added diagnostic environment variables
   - Added `RAY_DEDUP_LOGS=0` to see all worker messages
   - Added `VERL_N_GPUS_PER_NODE=4`

4. **`train/env_config.sh`**
   - Set NCCL configuration (NCCL_SOCKET_IFNAME=hsn)
   - Removed problematic GLOO_SOCKET_IFNAME setting

5. **`examples/qwen_gui_static_grpo/config/traj_grpo.yaml`**
   - Fixed custom_cls path to absolute path

## What To Do Next

### Immediate Next Steps

1. **Submit test job with NCCL_SOCKET_IFNAME fix**
   ```bash
   sbatch train/train_ui_s1_test_offload.slurm
   ```

2. **Monitor diagnostic output**
   ```bash
   # Check NCCL_SOCKET_IFNAME is being set
   grep "NCCL_SOCKET_IFNAME" train/logs/test_offload_<jobid>.log

   # Check if all ranks complete init_process_group
   grep "process group initialized" train/logs/test_offload_<jobid>.log
   ```

### If Still Hanging

1. **Try different NCCL network interface**
   - Check available interfaces: `ip addr show`
   - Try `eth0` or other interfaces if `hsn` doesn't work

2. **Add NCCL debug output**
   ```python
   os.environ["NCCL_DEBUG"] = "INFO"
   os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,NET"
   ```

3. **Try increasing NCCL timeout**
   ```python
   os.environ["NCCL_IB_TIMEOUT"] = "50"
   ```

4. **Consider using store-based init instead of env://**
   ```python
   # Use TCPStore for more reliable rendezvous
   store = torch.distributed.TCPStore(master_addr, master_port, world_size, is_master=(rank==0))
   torch.distributed.init_process_group(backend="nccl", store=store, rank=rank, world_size=world_size)
   ```

### Alternative Approaches

1. **Use file-based rendezvous** (already partially implemented)
   - Set `VERL_RENDEZVOUS_FILE` to a shared filesystem path
   - Ensure the file is accessible from all nodes

2. **Check Ray actor placement**
   - Verify that ranks are correctly distributed across nodes
   - Check if there's an issue with how verl assigns RANK to workers

3. **Test with simpler distributed setup**
   - Create a minimal test script that just does `init_process_group` across nodes
   - This isolates the issue from verl/Ray complexity

## Test Commands

```bash
# Submit new test job
sbatch train/train_ui_s1_test_offload.slurm

# Check job status
squeue -u shuqing.a5l

# Monitor log in real-time
tail -f train/logs/test_offload_<jobid>.log

# Check for diagnostic messages
grep "\[DIAG\]" train/logs/test_offload_<jobid>.log | sort | uniq

# Check for NCCL errors
grep -E "NCCL|nccl|error|Error" train/logs/test_offload_<jobid>.log
```

## Architecture Overview

```
Node 0 (Head)          Node 1 (Worker)
├── Ray Head           ├── Ray Worker
├── Rank 0 (GPU 0)     ├── Rank 4 (GPU 0)
├── Rank 1 (GPU 1)     ├── Rank 5 (GPU 1)
├── Rank 2 (GPU 2)     ├── Rank 6 (GPU 2)
└── Rank 3 (GPU 3)     └── Rank 7 (GPU 3)

NCCL init_process_group requires all 8 ranks to connect
before any can proceed. If ranks 4-7 can't reach ranks 0-3
via NCCL, the initialization hangs.
```

## Key Environment Variables

| Variable | Purpose | Current Value |
|----------|---------|---------------|
| `NCCL_SOCKET_IFNAME` | Network interface for NCCL | `hsn` |
| `NCCL_IB_TIMEOUT` | InfiniBand timeout | `24` |
| `MASTER_ADDR` | Distributed training master IP | Head node IP |
| `MASTER_PORT` | Distributed training master port | Dynamic |
| `CUDA_VISIBLE_DEVICES` | GPUs visible to process | Set by Ray (0-3) |
| `RAY_DEDUP_LOGS` | Disable Ray log deduplication | `0` |
