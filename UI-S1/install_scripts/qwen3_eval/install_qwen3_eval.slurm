#!/bin/bash
#SBATCH --job-name=install_qwen3_eval
#SBATCH --output=install_qwen3_eval_%j.out
#SBATCH --error=install_qwen3_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --gres=gpu:1

# =============================================================================
# Qwen3-VL Evaluation Environment Installation Script
# Creates a separate conda environment with vLLM >= 0.11.0 for Qwen3-VL support
# =============================================================================

set -e  # Exit on first error

ENV_NAME="qwen3-eval"
ENV_PREFIX="/scratch/a5l/shuqing.a5l/miniconda/envs/$ENV_NAME"

# Load required modules
module load gcc-native/12.3 cuda/12.6

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
fi

# === 2) Build parameters ===
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=8
export CMAKE_BUILD_PARALLEL_LEVEL=8

# Set all temp/cache dirs to /scratch to avoid home quota issues
export SCRATCH_BASE="/scratch/a5l/shuqing.a5l"
export TMPDIR="$SCRATCH_BASE/tmp"; mkdir -p "$TMPDIR"
export PIP_CACHE_DIR="$SCRATCH_BASE/pip_cache"; mkdir -p "$PIP_CACHE_DIR"
export XDG_CACHE_HOME="$SCRATCH_BASE/cache"; mkdir -p "$XDG_CACHE_HOME"
export CONDA_PKGS_DIRS="$SCRATCH_BASE/conda_pkgs"; mkdir -p "$CONDA_PKGS_DIRS"

echo "=============================================="
echo "Qwen3-VL Evaluation Environment Installation"
echo "=============================================="
echo "Environment name: $ENV_NAME"
echo "Environment path: $ENV_PREFIX"
echo "CUDA_HOME: $CUDA_HOME"
echo "nvcc path: $(which nvcc)"
echo "nvcc version: $($CUDA_HOME/bin/nvcc --version | grep release)"
echo "CC: $CC"
echo "CXX: $CXX"
echo "Start time: $(date)"
echo "=============================================="

# Initialize conda
eval "$(conda shell.bash hook)"

# === 3) Check if environment exists ===
if [ -d "$ENV_PREFIX" ]; then
    echo ""
    echo "Environment '$ENV_PREFIX' already exists."
    echo "Removing existing environment..."
    rm -rf "$ENV_PREFIX"
fi

# === 4) Create new conda environment ===
echo ""
echo "=== Creating conda environment at: $ENV_PREFIX ==="
conda create --prefix "$ENV_PREFIX" python=3.11 -y

# Activate the environment
conda activate "$ENV_PREFIX"

# === 5) Install base build tools ===
echo ""
echo "=== Installing base build tools ==="
python -m pip install -U pip setuptools wheel packaging ninja cmake

# === 6) Install PyTorch with CUDA 12.6 ===
echo ""
echo "=== Installing PyTorch ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Verify PyTorch CUDA
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'CUDA version:', torch.version.cuda)"

# === 7) Install vLLM >= 0.11.0 (required for Qwen3-VL) ===
echo ""
echo "=== Installing vLLM >= 0.11.0 ==="

# First try pip install (if wheels available)
pip install "vllm>=0.11.0" || {
    echo "Pre-built wheel not available, building from source..."

    # Clone and build vLLM from source
    VLLM_DIR="/tmp/vllm-qwen3-build-$$"
    echo "Cloning vLLM to $VLLM_DIR..."
    git clone --depth 1 https://github.com/vllm-project/vllm.git "$VLLM_DIR"
    cd "$VLLM_DIR"

    # Reduce parallelism to avoid ptxas memory issues
    export MAX_JOBS=1
    export CMAKE_BUILD_PARALLEL_LEVEL=1
    export NVCC_THREADS=1

    export CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/bin/nvcc \
                       -DCMAKE_CUDA_ARCHITECTURES=90 \
                       -DCUTLASS_NVCC_ARCHS=90 \
                       -DVLLM_FA_CMAKE_GPU_ARCHES='90-real'"

    pip install --no-build-isolation --no-cache-dir -v .

    cd -
    rm -rf "$VLLM_DIR"
}

# === 8) Install transformers >= 4.52.0 (required for Qwen3-VL) ===
echo ""
echo "=== Installing transformers >= 4.52.0 ==="
pip install "transformers>=4.52.0" || {
    echo "Installing transformers from source..."
    pip install git+https://github.com/huggingface/transformers.git
}

# === 9) Install qwen-vl-utils ===
echo ""
echo "=== Installing qwen-vl-utils ==="
pip install qwen-vl-utils==0.0.14

# === 10) Install additional dependencies ===
echo ""
echo "=== Installing additional dependencies ==="
pip install pandas pyarrow openai tqdm pillow requests accelerate

# === 11) Final verification ===
echo ""
echo "=============================================="
echo "Final verification"
echo "=============================================="
python - <<'PY'
import torch
print(f'PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  CUDA version: {torch.version.cuda}')

import transformers
print(f'transformers: {transformers.__version__}')

import vllm
print(f'vLLM: {vllm.__version__}')

# Check Qwen3-VL support
try:
    from transformers import Qwen3VLForConditionalGeneration
    print('Qwen3-VL support: YES')
except ImportError:
    print('Qwen3-VL support: NO (transformers version may be too old)')

print('')
print('Environment setup complete!')
PY

echo ""
echo "=============================================="
echo "Installation Summary"
echo "=============================================="
echo "Environment: $ENV_NAME"
echo "Location: $ENV_PREFIX"
echo ""
echo "To use this environment:"
echo "  conda activate $ENV_PREFIX"
echo ""
echo "End time: $(date)"
echo "=============================================="
