#!/bin/bash
#SBATCH --job-name=triton_build
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --gres=gpu:1

# =============================================================================
# Triton Installation Script for GH200 (aarch64 + sm90)
# Based on MARTI's approach - builds LLVM/MLIR 17 + Triton 3.2 from source
#
# Prerequisites: Download offline packages to ~/.triton/offline/
#   1) llvm-project-17.0.6.tar.gz or llvm-project-17.0.6.src.tar.xz
#   2) triton-3.2-recursive.tar.gz (triton with submodules)
# =============================================================================

########## 0) Modules & Conda Environment ##########
module load gcc-native/12.3 cuda/12.6

eval "$(conda shell.bash hook)"
conda activate ui-s1

set -u  # Enable after conda activate (conda scripts have unbound variables)

########## 1) Environment Setup ##########
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export CC=/opt/cray/pe/gcc-native/12/bin/gcc
export CXX=/opt/cray/pe/gcc-native/12/bin/g++

# GH200/H100: "90;90a"
export TORCH_CUDA_ARCH_LIST="90;90a"

echo "=== System Info ==="
pwd && hostname && date
nvidia-smi || true
nvcc --version || true
python - <<'PY' || true
import sys, torch
print("python", sys.version)
print("torch", getattr(torch,'__version__',None),
      "cuda", getattr(torch.version,'cuda',None),
      "is_cuda", torch.cuda.is_available())
PY
echo "CC=$CC  CXX=$CXX"
echo "CUDA_HOME=$CUDA_HOME"
echo "CONDA_PREFIX=${CONDA_PREFIX:-}"

########## 2) Offline Packages & Build Directories ##########
OFFLINE="$HOME/.triton/offline"
TRITON_TGZ="$OFFLINE/triton-3.2-recursive.tar.gz"
LLVM_TAR_GZ="$OFFLINE/llvm-project-17.0.6.tar.gz"
LLVM_TAR_XZ="$OFFLINE/llvm-project-17.0.6.src.tar.xz"

ROOT="$HOME/.triton"
SRC="$ROOT/src"
BUILD="$ROOT/build"
PREFIX="$ROOT/llvm/llvm-17-aarch64-custom"
TRITON_SRC="$SRC/triton-3.2"
LLVM_SRC="$SRC/llvm-project-17"
LLVM_BUILD="$BUILD/llvm-17-aarch64"

mkdir -p "$SRC" "$BUILD" "$ROOT/llvm"

# Verify offline packages exist
if [ ! -f "$TRITON_TGZ" ]; then
    echo "[FATAL] Missing $TRITON_TGZ"
    echo "Download instructions:"
    echo "  git clone --recursive https://github.com/triton-lang/triton"
    echo "  cd triton && git checkout v3.2.2 && git submodule update --init --recursive"
    echo "  cd .. && tar -czf triton-3.2-recursive.tar.gz triton"
    echo "  mv triton-3.2-recursive.tar.gz ~/.triton/offline/"
    exit 10
fi

if [ ! -f "$LLVM_TAR_GZ" ] && [ ! -f "$LLVM_TAR_XZ" ]; then
    echo "[FATAL] Missing LLVM offline package: $LLVM_TAR_GZ or $LLVM_TAR_XZ"
    echo "Download from: https://github.com/llvm/llvm-project/releases/tag/llvmorg-17.0.6"
    exit 11
fi

########## 3) Extract Archives (Offline) ##########
# Triton source
if [ ! -d "$TRITON_SRC" ]; then
    echo "[INFO] Extracting $TRITON_TGZ -> $SRC"
    mkdir -p "$TRITON_SRC"
    tmpdir="$SRC/.triton_unpack_tmp"
    rm -rf "$tmpdir"; mkdir -p "$tmpdir"
    tar -xzf "$TRITON_TGZ" -C "$tmpdir"
    rootdir=$(find "$tmpdir" -mindepth 1 -maxdepth 1 -type d | head -n1)
    if [ -z "$rootdir" ]; then
        echo "[FATAL] Triton archive is empty or malformed"
        exit 12
    fi
    rsync -a "$rootdir"/ "$TRITON_SRC"/
    rm -rf "$tmpdir"
fi

# LLVM source (auto-detect .gz / .xz)
if [ ! -d "$LLVM_SRC" ]; then
    mkdir -p "$LLVM_SRC"
    if [ -f "$LLVM_TAR_GZ" ]; then
        echo "[INFO] Extracting $LLVM_TAR_GZ -> $LLVM_SRC"
        tar -xzf "$LLVM_TAR_GZ" -C "$LLVM_SRC" --strip-components=1
    else
        echo "[INFO] Extracting $LLVM_TAR_XZ -> $LLVM_SRC"
        tar -xJf "$LLVM_TAR_XZ" -C "$LLVM_SRC" --strip-components=1
    fi
fi

########## 4) Build & Install LLVM/MLIR 17 ##########
mkdir -p "$LLVM_BUILD"
cd "$LLVM_BUILD"

echo "[INFO] Configuring LLVM/MLIR 17..."
cmake -G Ninja "$LLVM_SRC/llvm" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DLLVM_ENABLE_PROJECTS="mlir;clang;lld" \
    -DLLVM_TARGETS_TO_BUILD="NVPTX;AArch64;X86" \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_ENABLE_EH=ON \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_BUILD_LLVM_DYLIB=ON \
    -DLLVM_LINK_LLVM_DYLIB=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DLLVM_ENABLE_TERMINFO=OFF

# Limit parallelism to avoid OOM
export CMAKE_BUILD_PARALLEL_LEVEL=4
export NINJAFLAGS="-j4"
export MAKEFLAGS="-j4"

echo "[INFO] Building LLVM/MLIR 17 (this takes several hours)..."
ninja -j4 || true
ninja install || true

if [ ! -x "$PREFIX/bin/mlir-tblgen" ]; then
    echo "[FATAL] mlir-tblgen not generated: $PREFIX/bin/mlir-tblgen"
    exit 12
fi
echo "[OK] mlir-tblgen at: $PREFIX/bin/mlir-tblgen"

########## 5) Build Triton 3.2 Python Package ##########
export LLVM_DIR="$PREFIX/lib/cmake/llvm"
export MLIR_DIR="$PREFIX/lib/cmake/mlir"
export PATH="$PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}"

pip install -U pip setuptools wheel packaging pybind11 ninja cmake

cd "$TRITON_SRC/python" || { echo "[FATAL] Cannot find $TRITON_SRC/python"; exit 13; }

# Clean old build artifacts
rm -rf build .setuptools-cmake-build dist *.egg-info

# Single-threaded to avoid OOM
export CMAKE_BUILD_PARALLEL_LEVEL=8
export MAX_JOBS=8
export NINJAFLAGS="-j4"
export MAKEFLAGS="-j4"

echo "[INFO] Building Triton 3.2..."
pip install -v --no-build-isolation --no-cache-dir . \
    || ( python setup.py bdist_wheel && pip install dist/*.whl )

########## 6) Fix GLIBCXX & Verify Triton ##########
# Use system GCC-12's libstdc++.so.6
LIBSTDCXX_SO="$("$CXX" -print-file-name=libstdc++.so.6 2>/dev/null)"
LIBSTDCXX_DIR="$(dirname "$LIBSTDCXX_SO")"
if [ -f "$LIBSTDCXX_SO" ]; then
    export LD_LIBRARY_PATH="$LIBSTDCXX_DIR:${LD_LIBRARY_PATH:-}"
    export LD_PRELOAD="$LIBSTDCXX_SO:${LD_PRELOAD:-}"
    echo "[INFO] Using libstdc++.so.6 from: $LIBSTDCXX_DIR"
elif [ -f /opt/cray/pe/gcc-native/12/lib64/libstdc++.so.6 ]; then
    export LD_LIBRARY_PATH="/opt/cray/pe/gcc-native/12/lib64:${LD_LIBRARY_PATH:-}"
    export LD_PRELOAD="/opt/cray/pe/gcc-native/12/lib64/libstdc++.so.6:${LD_PRELOAD:-}"
    echo "[INFO] Using libstdc++.so.6 from: /opt/cray/pe/gcc-native/12/lib64"
fi

# Avoid source path issues
unset PYTHONPATH
cd ~

echo "[INFO] Verifying Triton installation..."
python - <<'PY'
import importlib, sys, triton, os
print("[OK] triton version =", getattr(triton, "__version__", "?"))
print("[OK] triton.__file__ =", triton.__file__)
try:
    import triton._C.libtriton as L
    print("[OK] libtriton loaded:", L)
except Exception as e:
    print("[FATAL] import triton._C.libtriton failed:", e, file=sys.stderr)
    sys.exit(14)
PY

echo "=== Triton Installation Complete ==="
date
