#!/bin/bash
#SBATCH --job-name=flash_attn_gui
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --gres=gpu:1

# =============================================================================
# Flash Attention Installation Script for GH200 (aarch64 + sm90)
# Target: gui conda environment
# =============================================================================

# === 0) Load modules and activate conda environment ===
module load gcc-native/12.3 cuda/12.6

eval "$(conda shell.bash hook)"
conda activate ui-s1

export CC=$(which gcc)
export CXX=$(which g++)

# === 1) Set CUDA_HOME ===
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export PATH="$CUDA_HOME/bin:$PATH"

# === 2) Add HPC SDK math_libs paths (cusparse.h is here) ===
export NVHPC_BASE="$(dirname "$(dirname "$CUDA_HOME")")"
export CPATH="$CUDA_HOME/include:$NVHPC_BASE/math_libs/include:${CPATH}"
export LIBRARY_PATH="$CUDA_HOME/lib64:$NVHPC_BASE/math_libs/lib64:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/targets/aarch64-linux/lib:$NVHPC_BASE/math_libs/lib64:${LD_LIBRARY_PATH}"

# === 3) Build parameters (GH200=SM90) ===
export TORCH_CUDA_ARCH_LIST="90"
export MAX_JOBS=1
export CMAKE_BUILD_PARALLEL_LEVEL=1
export TMPDIR="$HOME/.pip-tmp"; mkdir -p "$TMPDIR"

echo "=== System Info ==="
pwd && hostname && date
nvidia-smi || true
nvcc --version || true
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)"
echo "CC=$CC  CXX=$CXX"
echo "CUDA_HOME=$CUDA_HOME"

# === 4) Install basic build tools ===
python -m pip install -U pip setuptools wheel packaging ninja cmake

# === 5) Install flash_attn==2.7.0.post2 from Git tag ===
echo "=== Installing Flash Attention 2.7.0.post2 ==="
python -m pip install --no-build-isolation --no-cache-dir \
  "git+https://github.com/Dao-AILab/flash-attention@v2.7.0.post2"

# === 6) Verify installation ===
echo "=== Verifying Flash Attention ==="
python -c "import flash_attn; print('[OK] flash_attn version:', flash_attn.__version__)"

echo "=== Flash Attention Installation Complete ==="
date
