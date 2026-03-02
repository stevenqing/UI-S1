#!/bin/bash
#SBATCH --job-name=fix_pytorch_cuda
#SBATCH --output=fix_pytorch_cuda_%j.out
#SBATCH --error=fix_pytorch_cuda_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --partition=workq
#SBATCH --gres=gpu:1

# =============================================================================
# Fix PyTorch CUDA installation for qwen3-eval environment
# =============================================================================

set -e

ENV_PREFIX="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval"

# Load required modules
module load gcc-native/12.3 cuda/12.6

# Set CUDA paths
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export PATH="$CUDA_HOME/bin:$PATH"

# Set up compiler paths
export CC=/opt/cray/pe/gcc-native/12/bin/gcc
export CXX=/opt/cray/pe/gcc-native/12/bin/g++

# Set up library paths
export NVHPC_BASE=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11
export CPATH="$CUDA_HOME/include:$NVHPC_BASE/math_libs/include:${CPATH}"
export LIBRARY_PATH="$CUDA_HOME/lib64:$NVHPC_BASE/math_libs/lib64:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/targets/aarch64-linux/lib:$NVHPC_BASE/math_libs/lib64:${LD_LIBRARY_PATH}"

# Fix GLIBCXX
if [ -f /opt/cray/pe/gcc-native/12/lib64/libstdc++.so.6 ]; then
    export LD_LIBRARY_PATH="/opt/cray/pe/gcc-native/12/lib64:${LD_LIBRARY_PATH:-}"
fi

# Set cache dirs to /scratch
export SCRATCH_BASE="/scratch/a5l/shuqing.a5l"
export TMPDIR="$SCRATCH_BASE/tmp"; mkdir -p "$TMPDIR"
export PIP_CACHE_DIR="$SCRATCH_BASE/pip_cache"; mkdir -p "$PIP_CACHE_DIR"
export XDG_CACHE_HOME="$SCRATCH_BASE/cache"; mkdir -p "$XDG_CACHE_HOME"

# Build parameters
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=8
export CMAKE_BUILD_PARALLEL_LEVEL=8

echo "=============================================="
echo "Fixing PyTorch CUDA Installation"
echo "=============================================="
echo "Environment: $ENV_PREFIX"
echo "CUDA_HOME: $CUDA_HOME"
echo "nvcc version: $($CUDA_HOME/bin/nvcc --version | grep release)"
echo "Start time: $(date)"
echo "=============================================="

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"

# Check current PyTorch
echo ""
echo "=== Current PyTorch installation ==="
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# Uninstall current PyTorch
echo ""
echo "=== Uninstalling CPU PyTorch ==="
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Try different installation methods for aarch64 + CUDA

echo ""
echo "=== Method 1: Try PyTorch nightly with CUDA 12.6 ==="
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126 && {
    python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'CUDA version:', torch.version.cuda)"
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
        echo "SUCCESS: PyTorch CUDA installed via nightly!"
    else
        echo "Nightly installed but CUDA not available, trying next method..."
        pip uninstall -y torch torchvision torchaudio
    fi
} || {
    echo "Nightly wheels not available for this platform"
}

# Check if CUDA is available now
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo ""
    echo "=== PyTorch CUDA is working! ==="
else
    echo ""
    echo "=== Method 2: Build PyTorch from source ==="

    # Install build dependencies
    pip install cmake ninja pyyaml typing_extensions

    # Clone PyTorch
    PYTORCH_DIR="/tmp/pytorch-build-$$"
    echo "Cloning PyTorch to $PYTORCH_DIR..."
    git clone --recursive --depth 1 -b v2.5.1 https://github.com/pytorch/pytorch.git "$PYTORCH_DIR"
    cd "$PYTORCH_DIR"

    # Set build options
    export USE_CUDA=1
    export USE_CUDNN=1
    export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
    export TORCH_CUDA_ARCH_LIST="9.0"
    export MAX_JOBS=8

    # Install dependencies
    pip install -r requirements.txt

    # Build and install
    python setup.py install

    cd -
    rm -rf "$PYTORCH_DIR"
fi

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
PY

echo ""
echo "End time: $(date)"
echo "=============================================="
