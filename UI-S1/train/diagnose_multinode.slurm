#!/bin/bash
#SBATCH --job-name=diag_multinode
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/diagnose_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/diagnose_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

# ============================================
# Multi-Node Hang Diagnosis Script
# 用于诊断 2-node 训练卡住问题
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
LOG_DIR="$PROJECT_DIR/train/logs"
DIAG_DIR="$LOG_DIR/diagnosis_${SLURM_JOB_ID}"

mkdir -p $DIAG_DIR

echo "=========================================="
echo "Multi-Node Hang Diagnosis"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "Diagnosis output: $DIAG_DIR"
echo "=========================================="

# Get node information
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}
    else
        head_node_ip=${ADDR[0]}
    fi
fi

echo "Head node: $head_node ($head_node_ip)"

# ============================================
# Step 1: 收集各节点环境信息
# ============================================
echo ""
echo "=== Step 1: Collecting environment info from all nodes ==="

for node in $nodes; do
    echo "--- Node: $node ---"
    srun --nodes=1 --ntasks=1 -w "$node" bash -c "
        echo 'Hostname:' \$(hostname)
        echo 'IP addresses:'
        ip addr | grep -E 'inet |hsn|ib'
        echo ''
        echo 'GPU Info:'
        nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
        echo ''
        echo 'CUDA Version:'
        nvcc --version 2>/dev/null | grep release || echo 'nvcc not in PATH'
        echo ''
        echo 'IB Status:'
        ibstat 2>/dev/null | head -20 || echo 'ibstat not available'
    " 2>&1 | tee "$DIAG_DIR/env_${node}.log"
done

# ============================================
# Step 2: 测试节点间网络连通性
# ============================================
echo ""
echo "=== Step 2: Testing inter-node connectivity ==="

# 测试 NCCL 通信
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 bash -c "
    source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1

    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=INIT,NET
    export NCCL_SOCKET_IFNAME=hsn

    python3 -c '
import os
import torch
import torch.distributed as dist

rank = int(os.environ.get(\"SLURM_PROCID\", 0))
world_size = int(os.environ.get(\"SLURM_NTASKS\", 1))
local_rank = int(os.environ.get(\"SLURM_LOCALID\", 0))

print(f\"[Rank {rank}] Starting NCCL test...\")
print(f\"[Rank {rank}] CUDA available: {torch.cuda.is_available()}\")
print(f\"[Rank {rank}] GPU count: {torch.cuda.device_count()}\")

# Initialize process group
os.environ[\"MASTER_ADDR\"] = \"$head_node_ip\"
os.environ[\"MASTER_PORT\"] = \"29500\"
os.environ[\"RANK\"] = str(rank)
os.environ[\"WORLD_SIZE\"] = str(world_size)

torch.cuda.set_device(local_rank)
dist.init_process_group(backend=\"nccl\", rank=rank, world_size=world_size)

print(f\"[Rank {rank}] Process group initialized\")

# Simple all-reduce test
tensor = torch.ones(1000, device=\"cuda\") * rank
print(f\"[Rank {rank}] Before all_reduce: sum={tensor.sum().item()}\")

dist.all_reduce(tensor)
print(f\"[Rank {rank}] After all_reduce: sum={tensor.sum().item()}\")

# Barrier test
print(f\"[Rank {rank}] Testing barrier...\")
dist.barrier()
print(f\"[Rank {rank}] Barrier passed\")

dist.destroy_process_group()
print(f\"[Rank {rank}] NCCL test completed successfully\")
'
" 2>&1 | tee "$DIAG_DIR/nccl_test.log"

NCCL_TEST_EXIT=$?
if [ $NCCL_TEST_EXIT -eq 0 ]; then
    echo "NCCL connectivity test: PASSED"
else
    echo "NCCL connectivity test: FAILED (exit code: $NCCL_TEST_EXIT)"
fi

# ============================================
# Step 3: 测试 FSDP state_dict 操作
# ============================================
echo ""
echo "=== Step 3: Testing FSDP state_dict operations ==="

srun --nodes=2 --ntasks=2 --ntasks-per-node=1 bash -c "
    source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1

    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=INIT,NET,COLL
    export NCCL_SOCKET_IFNAME=hsn
    export PYTHONPATH=$PROJECT_DIR

    python3 -c '
import os
import time
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardedStateDictConfig, StateDictType

rank = int(os.environ.get(\"SLURM_PROCID\", 0))
world_size = int(os.environ.get(\"SLURM_NTASKS\", 1))
local_rank = int(os.environ.get(\"SLURM_LOCALID\", 0))

os.environ[\"MASTER_ADDR\"] = \"$head_node_ip\"
os.environ[\"MASTER_PORT\"] = \"29501\"
os.environ[\"RANK\"] = str(rank)
os.environ[\"WORLD_SIZE\"] = str(world_size)

torch.cuda.set_device(local_rank)
dist.init_process_group(backend=\"nccl\", rank=rank, world_size=world_size)

print(f\"[Rank {rank}][{time.time():.3f}] Creating test model...\")

# 创建一个小的测试模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1024),
        )
    def forward(self, x):
        return self.layers(x)

model = SimpleModel().cuda()
model = FSDP(model)

print(f\"[Rank {rank}][{time.time():.3f}] FSDP model created\")

# 测试 sharded state dict
FSDP.set_state_dict_type(
    model,
    state_dict_type=StateDictType.SHARDED_STATE_DICT,
    state_dict_config=ShardedStateDictConfig(),
)

print(f\"[Rank {rank}][{time.time():.3f}] Testing sharded state_dict...\")
dist.barrier()
t0 = time.time()
state = model.state_dict()
t1 = time.time()
print(f\"[Rank {rank}][{time.time():.3f}] Sharded state_dict took {t1-t0:.3f}s, keys: {len(state)}\")

dist.barrier()
print(f\"[Rank {rank}][{time.time():.3f}] All ranks completed state_dict test\")

dist.destroy_process_group()
'
" 2>&1 | tee "$DIAG_DIR/fsdp_state_dict_test.log"

FSDP_TEST_EXIT=$?
if [ $FSDP_TEST_EXIT -eq 0 ]; then
    echo "FSDP state_dict test: PASSED"
else
    echo "FSDP state_dict test: FAILED (exit code: $FSDP_TEST_EXIT)"
fi

# ============================================
# Step 4: 测试 CPU offload + state_dict
# ============================================
echo ""
echo "=== Step 4: Testing CPU offload + state_dict (this is the likely hang point) ==="

timeout 300 srun --nodes=2 --ntasks=2 --ntasks-per-node=1 bash -c "
    source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1

    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=INIT,NET,COLL
    export NCCL_SOCKET_IFNAME=hsn
    export PYTHONPATH=$PROJECT_DIR

    python3 -c '
import os
import time
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload
from torch.distributed.fsdp.api import ShardedStateDictConfig, StateDictType

rank = int(os.environ.get(\"SLURM_PROCID\", 0))
world_size = int(os.environ.get(\"SLURM_NTASKS\", 1))
local_rank = int(os.environ.get(\"SLURM_LOCALID\", 0))

os.environ[\"MASTER_ADDR\"] = \"$head_node_ip\"
os.environ[\"MASTER_PORT\"] = \"29502\"
os.environ[\"RANK\"] = str(rank)
os.environ[\"WORLD_SIZE\"] = str(world_size)

torch.cuda.set_device(local_rank)
dist.init_process_group(backend=\"nccl\", rank=rank, world_size=world_size)

print(f\"[Rank {rank}][{time.time():.3f}] Creating model with CPU offload...\")

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(1024, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 1024),
        )
    def forward(self, x):
        return self.layers(x)

model = SimpleModel().cuda()
model = FSDP(model, cpu_offload=CPUOffload(offload_params=True))

print(f\"[Rank {rank}][{time.time():.3f}] FSDP model with CPU offload created\")

FSDP.set_state_dict_type(
    model,
    state_dict_type=StateDictType.SHARDED_STATE_DICT,
    state_dict_config=ShardedStateDictConfig(),
)

# 模拟 load_fsdp_model_to_gpu
print(f\"[Rank {rank}][{time.time():.3f}] Simulating load to GPU...\")
dist.barrier()
for param in model.parameters():
    if param.device.type == \"cpu\":
        param.data = param.data.to(local_rank)
dist.barrier()
print(f\"[Rank {rank}][{time.time():.3f}] Load to GPU completed\")

# 测试 state_dict
print(f\"[Rank {rank}][{time.time():.3f}] Calling state_dict (potential hang point)...\")
dist.barrier()
t0 = time.time()
state = model.state_dict()
t1 = time.time()
print(f\"[Rank {rank}][{time.time():.3f}] state_dict completed in {t1-t0:.3f}s\")

dist.barrier()
print(f\"[Rank {rank}][{time.time():.3f}] Test completed successfully!\")

dist.destroy_process_group()
'
" 2>&1 | tee "$DIAG_DIR/cpu_offload_test.log"

OFFLOAD_TEST_EXIT=$?
if [ $OFFLOAD_TEST_EXIT -eq 0 ]; then
    echo "CPU offload + state_dict test: PASSED"
elif [ $OFFLOAD_TEST_EXIT -eq 124 ]; then
    echo "CPU offload + state_dict test: TIMEOUT (this confirms the hang issue!)"
else
    echo "CPU offload + state_dict test: FAILED (exit code: $OFFLOAD_TEST_EXIT)"
fi

# ============================================
# Summary
# ============================================
echo ""
echo "=========================================="
echo "Diagnosis Summary"
echo "=========================================="
echo "All logs saved to: $DIAG_DIR"
echo ""
echo "Files created:"
ls -la "$DIAG_DIR"
echo ""
echo "Key findings:"
grep -h "PASSED\|FAILED\|TIMEOUT" "$DIAG_DIR"/*.log 2>/dev/null || echo "Check individual log files"
echo ""
echo "Next steps based on results:"
echo "1. If NCCL test failed -> Check network configuration"
echo "2. If FSDP test failed -> Check PyTorch/FSDP version"
echo "3. If CPU offload test timed out -> Confirms the hang issue, need to:"
echo "   a) Disable param_offload, OR"
echo "   b) Add explicit barriers in sharding manager"
echo ""
echo "Diagnosis completed at $(date)"
echo "=========================================="
