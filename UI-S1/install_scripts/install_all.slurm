#!/bin/bash
#SBATCH --job-name=install_ui-s1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --gres=gpu:1

# =============================================================================
# UI-S1 Full Installation Script for GH200 (aarch64 + sm90)
# Installs: PyTorch with CUDA, Flash Attention, vLLM
# =============================================================================

set -e  # Exit on first error

# Load required modules
module load gcc-native/12.3 cuda/12.6

# Initialize and activate conda environment
eval "$(conda shell.bash hook)"
conda activate ui-s1

# === 1) Set CUDA paths EXPLICITLY ===
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export PATH="$CUDA_HOME/bin:$PATH"

# Verify nvcc is found
if [ ! -f "$CUDA_HOME/bin/nvcc" ]; then
    echo "ERROR: nvcc not found at $CUDA_HOME/bin/nvcc"
    exit 1
fi

# Set up compiler paths
export CC=/opt/cray/pe/gcc-native/12/bin/gcc
export CXX=/opt/cray/pe/gcc-native/12/bin/g++

# Set up NVHPC math_libs paths
export NVHPC_BASE=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11
export CPATH="$CUDA_HOME/include:$NVHPC_BASE/math_libs/include:${CPATH}"
export LIBRARY_PATH="$CUDA_HOME/lib64:$NVHPC_BASE/math_libs/lib64:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/targets/aarch64-linux/lib:$NVHPC_BASE/math_libs/lib64:${LD_LIBRARY_PATH}"

# Fix GLIBCXX (use system GCC-12's libstdc++)
if [ -f /opt/cray/pe/gcc-native/12/lib64/libstdc++.so.6 ]; then
    export LD_LIBRARY_PATH="/opt/cray/pe/gcc-native/12/lib64:${LD_LIBRARY_PATH:-}"
    export LD_PRELOAD="/opt/cray/pe/gcc-native/12/lib64/libstdc++.so.6:${LD_PRELOAD:-}"
fi

# === 2) Build parameters ===
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=8
export CMAKE_BUILD_PARALLEL_LEVEL=8
export TMPDIR="$HOME/.pip-tmp"; mkdir -p "$TMPDIR"

echo "=== Environment Setup ==="
echo "CUDA_HOME: $CUDA_HOME"
echo "nvcc path: $(which nvcc)"
echo "nvcc version: $($CUDA_HOME/bin/nvcc --version | grep release)"
echo "CC: $CC"
echo "CXX: $CXX"

# === 3) Check Triton is installed ===
echo "=== Checking Triton ==="
python -c "import triton; print('Triton version:', triton.__version__)" || {
    echo "ERROR: Triton not installed. Run install_triton.slurm first."
    exit 1
}

# === 4) Install base dependencies ===
echo "=== Installing base build tools ==="
python -m pip install -U pip setuptools wheel packaging ninja cmake setuptools_scm

# === 5) Install PyTorch with CUDA 12.6 ===
echo "=== Installing PyTorch ==="
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --force-reinstall

# Verify PyTorch CUDA
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'CUDA version:', torch.version.cuda)"

# === 6) Install Flash Attention 2 ===
echo "=== Installing Flash Attention 2 ==="
# Check if already installed
if python -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null; then
    echo "Flash Attention already installed, skipping..."
else
    python -m pip install --no-build-isolation --no-cache-dir \
        "git+https://github.com/Dao-AILab/flash-attention@v2.7.0.post2"
fi

# === 7) Install vLLM ===
echo "=== Installing vLLM ==="
export VLLM_INSTALL_FLASH_ATTN_C=FALSE

# Install vLLM 0.6.x (compatible with FA2, no FA3 ptxas issues)
pip install --no-build-isolation --no-cache-dir -v "git+https://github.com/vllm-project/vllm@v0.6.4.post1"

echo "=== vLLM Installation complete ==="

# === 8) Fix PyTorch version (vLLM may downgrade it) ===
echo "=== Fixing PyTorch version ==="
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
if [[ "$TORCH_VERSION" != "2.6.0"* ]]; then
    echo "PyTorch was downgraded to $TORCH_VERSION, reinstalling 2.6.0..."
    pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --force-reinstall --no-deps
fi

# === 9) Install UI-S1 package ===
echo "=== Installing UI-S1 package ==="
cd /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1
pip install -e .

# === 10) Final verification ===
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
PY

echo "=== Installation successful! ==="
date
