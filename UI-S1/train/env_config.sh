#!/bin/bash
# ============================================
# Shared Environment Configuration for UI-S1 Training
# Source this file in all training scripts
# ============================================

# Disable AWS-OFI-NCCL module - force socket transport instead
# module load brics/aws-ofi-nccl/1.8.1 2>/dev/null || true

# Wandb Configuration
export WANDB_API_KEY="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"

# CUDA/Library paths
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

# Ray configuration
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_METRICS_EXPORT_DISABLED=1
export RAY_RUNTIME_ENV_LOG_TO_DRIVER_ENABLED=0
export RAY_DISABLE_DASHBOARD=1
export RAY_SCHEDULER_SPREAD_THRESHOLD=0

# Training environment
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1

# NCCL configuration for multi-node training on GH200 - Force socket transport
# Use hsn0 specifically (primary high-speed network interface)
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
# Disable P2P completely to avoid memory issues
export NCCL_P2P_DISABLE=1
# Allow cross-NIC communication (important for multi-subnet clusters)
export NCCL_CROSS_NIC=2
# Use minimal socket threads to avoid memory issues
export NCCL_SOCKET_NTHREADS=1
export NCCL_NSOCKS_PERTHREAD=1
# Force IPv4 to avoid IPv6 binding issues
export NCCL_SOCKET_FAMILY=AF_INET
# Additional NCCL settings for GH200
export NCCL_BUFFSIZE=8388608  # 8MB buffer size

# Gloo configuration (for gloo backend fallback)
# Use same hsn0 interface as NCCL
export GLOO_SOCKET_IFNAME=hsn0

# PyTorch distributed settings - CRITICAL FOR MULTI-NODE
# Checkpoint saving takes ~54 min, so timeouts must be longer
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # Enable detailed debug for diagnosing hangs
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=10800  # 3 hour heartbeat timeout (checkpoint save ~54min)
# CRITICAL: Increase NCCL operation timeout for large all-gather operations
# Checkpoint saving requires full state dict gather, needs ~60 min, set to 90 min for safety
export NCCL_TIMEOUT=5400  # 90 minutes (in seconds)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # Enable async error handling to detect hangs
export NCCL_DEBUG_SUBSYS=INIT,NET,COLL  # Add COLL for collective operation debugging
# Enable flight recorder for debugging hangs
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
# Dump debug info on timeout (helps diagnose where hang occurred)
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export NCCL_DEBUG_FILE=/tmp/nccl_debug_%h_%p.log  # Per-host NCCL debug log
