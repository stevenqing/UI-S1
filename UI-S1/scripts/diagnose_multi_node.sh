#!/bin/bash
# Multi-Node Hang Diagnosis Script
# Usage: bash scripts/diagnose_multi_node.sh

set -e

LOG_DIR="${LOG_DIR:-./diagnosis_logs}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/diagnosis_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

echo "=== Multi-Node Hang Diagnosis ===" | tee "$LOG_FILE"
echo "Date: $(date)" | tee -a "$LOG_FILE"
echo "Hostname: $(hostname)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "=== Step 1: Environment Info ===" | tee -a "$LOG_FILE"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
try:
    print(f'NCCL: {torch.cuda.nccl.version()}')
except:
    print('NCCL: N/A')
print(f'cuDNN: {torch.backends.cudnn.version()}')
" 2>&1 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "=== Step 2: GPU Info ===" | tee -a "$LOG_FILE"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv 2>&1 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "=== Step 3: Network Interfaces ===" | tee -a "$LOG_FILE"
ip addr 2>&1 | grep -E "^[0-9]+:|inet |ib" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "=== Step 4: IB Status ===" | tee -a "$LOG_FILE"
if command -v ibstat &> /dev/null; then
    ibstat 2>&1 | tee -a "$LOG_FILE"
else
    echo "ibstat not available" | tee -a "$LOG_FILE"
fi
echo "" | tee -a "$LOG_FILE"

echo "=== Step 5: NCCL/CUDA Environment Variables ===" | tee -a "$LOG_FILE"
env | grep -iE "nccl|cuda|torch|distributed" 2>/dev/null | sort | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "=== Step 6: Check FSDP Version ===" | tee -a "$LOG_FILE"
python -c "
import torch
from packaging import version

if version.parse(torch.__version__) >= version.parse('2.4'):
    print('FSDP Version: FSDP2 (fully_shard API available)')
else:
    print('FSDP Version: FSDP1 (legacy FSDP)')
" 2>&1 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "=== Step 7: Check verl Configuration Files ===" | tee -a "$LOG_FILE"
for config in configs/*.yaml train/*.yaml; do
    if [ -f "$config" ]; then
        echo "--- $config ---" | tee -a "$LOG_FILE"
        grep -E "param_offload|optimizer_offload|load_format|layered_summon|lora" "$config" 2>/dev/null | tee -a "$LOG_FILE"
    fi
done
echo "" | tee -a "$LOG_FILE"

echo "=== Step 8: Memory Status ===" | tee -a "$LOG_FILE"
free -h 2>&1 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "=== Recommended NCCL Debug Settings ===" | tee -a "$LOG_FILE"
cat << 'EOF' | tee -a "$LOG_FILE"
# Add these to your training script before running:

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,COLL
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_TIMEOUT=1800

# To disable IB for testing:
# export NCCL_IB_DISABLE=1

# To increase timeout:
# export NCCL_TIMEOUT=3600
EOF
echo "" | tee -a "$LOG_FILE"

echo "=== Diagnosis Complete ===" | tee -a "$LOG_FILE"
echo "Full log saved to: $LOG_FILE"
