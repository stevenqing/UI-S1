#!/bin/bash
#SBATCH --job-name=fix_versions
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=workq
#SBATCH --gres=gpu:1

# =============================================================================
# Fix PyTorch/torchvision version mismatch
# =============================================================================

set -e

# Load required modules
module load gcc-native/12.3 cuda/12.6

# Initialize and activate conda environment
eval "$(conda shell.bash hook)"
conda activate ui-s1

# Set CUDA paths
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export PATH="$CUDA_HOME/bin:$PATH"

# Fix GLIBCXX
if [ -f /opt/cray/pe/gcc-native/12/lib64/libstdc++.so.6 ]; then
    export LD_LIBRARY_PATH="/opt/cray/pe/gcc-native/12/lib64:${LD_LIBRARY_PATH:-}"
    export LD_PRELOAD="/opt/cray/pe/gcc-native/12/lib64/libstdc++.so.6:${LD_PRELOAD:-}"
fi

echo "=== Current versions ==="
pip list | grep -E "torch|vllm|flash"

echo "=== Reinstalling PyTorch stack with matching versions ==="
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo "=== Installing UI-S1 package ==="
cd /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1
pip install -e .

echo "=== Final verification ==="
python - <<'PY'
import torch
print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'CUDA version:', torch.version.cuda)

import triton
print('Triton:', triton.__version__)

import flash_attn
print('flash_attn:', flash_attn.__version__)

# Test vLLM with environment variables to use V0 engine
import os
os.environ['VLLM_USE_V1'] = '0'
os.environ['VLLM_TORCH_COMPILE'] = '0'
os.environ['VLLM_USE_TRITON_FLASH_ATTN'] = '0'

import vllm
print('vLLM:', vllm.__version__)

print('=== All imports successful! ===')
PY

echo "=== Fix complete ==="
date
