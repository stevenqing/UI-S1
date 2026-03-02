#!/usr/bin/env python3
"""
Minimal NCCL test script to debug multi-node initialization issues.
This bypasses the verl/Ray framework to isolate NCCL-specific problems.

Usage:
  # On node 0 (master):
  MASTER_ADDR=<node0_ip> MASTER_PORT=29500 WORLD_SIZE=8 RANK=0 LOCAL_RANK=0 python test_nccl_minimal.py

  # On other nodes:
  MASTER_ADDR=<node0_ip> MASTER_PORT=29500 WORLD_SIZE=8 RANK=<rank> LOCAL_RANK=<local_rank> python test_nccl_minimal.py
"""

import os
import sys
import socket
import time

def main():
    # Get environment variables
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = int(os.environ.get("MASTER_PORT", "29500"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    hostname = socket.gethostname()

    print(f"[{hostname}] Rank {rank}/{world_size}: Starting NCCL minimal test", flush=True)
    print(f"[{hostname}] Rank {rank}: MASTER_ADDR={master_addr}, MASTER_PORT={master_port}", flush=True)
    print(f"[{hostname}] Rank {rank}: LOCAL_RANK={local_rank}", flush=True)

    # Print NCCL environment variables
    nccl_vars = [k for k in os.environ if k.startswith("NCCL")]
    print(f"[{hostname}] Rank {rank}: NCCL env vars:", flush=True)
    for var in sorted(nccl_vars):
        print(f"  {var}={os.environ[var]}", flush=True)

    # Import torch and check CUDA
    print(f"[{hostname}] Rank {rank}: Importing torch...", flush=True)
    import torch
    print(f"[{hostname}] Rank {rank}: PyTorch version: {torch.__version__}", flush=True)
    print(f"[{hostname}] Rank {rank}: CUDA available: {torch.cuda.is_available()}", flush=True)
    print(f"[{hostname}] Rank {rank}: CUDA device count: {torch.cuda.device_count()}", flush=True)

    if torch.cuda.is_available():
        print(f"[{hostname}] Rank {rank}: CUDA version: {torch.version.cuda}", flush=True)
        print(f"[{hostname}] Rank {rank}: cuDNN version: {torch.backends.cudnn.version()}", flush=True)

        # Check NCCL version
        try:
            nccl_version = torch.cuda.nccl.version()
            print(f"[{hostname}] Rank {rank}: NCCL version: {nccl_version}", flush=True)
        except Exception as e:
            print(f"[{hostname}] Rank {rank}: Failed to get NCCL version: {e}", flush=True)

    # Set CUDA device based on local_rank
    # If CUDA_VISIBLE_DEVICES is set to limit to 1 GPU, use device 0
    # Otherwise, use local_rank to select the appropriate GPU
    device_count = torch.cuda.device_count()
    if device_count == 1:
        cuda_device = 0
    else:
        cuda_device = local_rank
    print(f"[{hostname}] Rank {rank}: Setting CUDA device to {cuda_device} (local_rank={local_rank}, device_count={device_count})", flush=True)
    torch.cuda.set_device(cuda_device)

    # Test basic CUDA operation
    print(f"[{hostname}] Rank {rank}: Testing basic CUDA tensor...", flush=True)
    x = torch.randn(10).cuda()
    print(f"[{hostname}] Rank {rank}: CUDA tensor created: {x.shape}", flush=True)

    # Test TCP connectivity first
    print(f"[{hostname}] Rank {rank}: Testing TCP connectivity to {master_addr}:{master_port}...", flush=True)
    try:
        if rank == 0:
            # Master: start listening
            time.sleep(1)  # Give other ranks time to start
        else:
            # Non-master: try to connect
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(10)
            s.connect((master_addr, master_port))
            s.close()
            print(f"[{hostname}] Rank {rank}: TCP connectivity OK", flush=True)
    except Exception as e:
        print(f"[{hostname}] Rank {rank}: TCP connectivity test error: {e}", flush=True)

    # Initialize distributed with TCPStore
    print(f"[{hostname}] Rank {rank}: Creating TCPStore...", flush=True)
    try:
        store = torch.distributed.TCPStore(
            host_name=master_addr,
            port=master_port,
            world_size=world_size,
            is_master=(rank == 0),
            timeout=torch.distributed.default_pg_timeout
        )
        print(f"[{hostname}] Rank {rank}: TCPStore created successfully", flush=True)
    except Exception as e:
        print(f"[{hostname}] Rank {rank}: TCPStore creation failed: {e}", flush=True)
        sys.exit(1)

    # Now try NCCL initialization
    print(f"[{hostname}] Rank {rank}: Initializing NCCL process group...", flush=True)
    print(f"[{hostname}] Rank {rank}: This is where it might hang. Watch for NCCL debug output.", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()

    try:
        start_time = time.time()
        torch.distributed.init_process_group(
            backend="nccl",
            store=store,
            rank=rank,
            world_size=world_size,
            device_id=torch.device(f"cuda:{cuda_device}")
        )
        elapsed = time.time() - start_time
        print(f"[{hostname}] Rank {rank}: NCCL init_process_group completed in {elapsed:.2f}s!", flush=True)
    except Exception as e:
        print(f"[{hostname}] Rank {rank}: NCCL init_process_group failed: {e}", flush=True)
        sys.exit(1)

    # Test a simple allreduce
    print(f"[{hostname}] Rank {rank}: Testing allreduce...", flush=True)
    tensor = torch.tensor([rank], dtype=torch.float32).cuda()
    torch.distributed.all_reduce(tensor)
    expected = sum(range(world_size))
    print(f"[{hostname}] Rank {rank}: allreduce result: {tensor.item()}, expected: {expected}", flush=True)

    # Cleanup
    torch.distributed.destroy_process_group()
    print(f"[{hostname}] Rank {rank}: SUCCESS - all tests passed!", flush=True)

if __name__ == "__main__":
    main()
