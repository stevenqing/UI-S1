#!/bin/bash
#SBATCH --job-name=fix_torch_version
#SBATCH --output=fix_torch_version_%j.out
#SBATCH --error=fix_torch_version_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=workq
#SBATCH --gres=gpu:1

# =============================================================================
# Install torch==2.9.1 with CUDA to be compatible with vLLM 0.15.1
# =============================================================================

set -e

ENV_PREFIX="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval"

# Load required modules
module load gcc-native/12.3 cuda/12.6

# Set CUDA paths
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/targets/aarch64-linux/lib:${LD_LIBRARY_PATH}"

# Fix GLIBCXX
if [ -f /opt/cray/pe/gcc-native/12/lib64/libstdc++.so.6 ]; then
    export LD_LIBRARY_PATH="/opt/cray/pe/gcc-native/12/lib64:${LD_LIBRARY_PATH:-}"
fi

# Set cache dirs to /scratch
export SCRATCH_BASE="/scratch/a5l/shuqing.a5l"
export TMPDIR="$SCRATCH_BASE/tmp"; mkdir -p "$TMPDIR"
export PIP_CACHE_DIR="$SCRATCH_BASE/pip_cache"; mkdir -p "$PIP_CACHE_DIR"
export XDG_CACHE_HOME="$SCRATCH_BASE/cache"; mkdir -p "$XDG_CACHE_HOME"

echo "=============================================="
echo "Installing torch==2.9.1 with CUDA"
echo "=============================================="
echo "Start time: $(date)"

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"

# Uninstall current PyTorch
echo ""
echo "=== Uninstalling current PyTorch ==="
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Install torch 2.9.1 with CUDA 12.6
echo ""
echo "=== Installing torch==2.9.1 with CUDA 12.6 ==="
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu126

# Final verification
echo ""
echo "=============================================="
echo "Final verification"
echo "=============================================="
python - <<'PY'
import torch
print(f'PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'  GPU count: {torch.cuda.device_count()}')
    print(f'  GPU name: {torch.cuda.get_device_name(0)}')
    # Test CUDA computation
    x = torch.randn(3, 3).cuda()
    y = torch.randn(3, 3).cuda()
    z = torch.matmul(x, y)
    print(f'  CUDA computation test: PASSED')
else:
    print('  WARNING: CUDA not available!')

import vllm
print(f'vLLM: {vllm.__version__}')

import transformers
print(f'transformers: {transformers.__version__}')

# Check Qwen3-VL support
try:
    from transformers import Qwen3VLForConditionalGeneration
    print('Qwen3-VL support: YES')
except ImportError:
    print('Qwen3-VL support: NO')
PY

echo ""
echo "End time: $(date)"
echo "=============================================="
