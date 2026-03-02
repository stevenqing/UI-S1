#!/bin/bash
#SBATCH --job-name=install_vllm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --gres=gpu:1

# Load required modules
module load gcc-native/12.3 cuda/12.6

# Initialize and activate conda environment
eval "$(conda shell.bash hook)"
conda activate ui-s1

# Set environment variables for the build
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"

# Fix GLIBCXX
if [ -f /opt/cray/pe/gcc-native/12/lib64/libstdc++.so.6 ]; then
    export LD_LIBRARY_PATH="/opt/cray/pe/gcc-native/12/lib64:${LD_LIBRARY_PATH:-}"
fi

export CC=/opt/cray/pe/gcc-native/12/bin/gcc
export CXX=/opt/cray/pe/gcc-native/12/bin/g++

# Reduce parallelism to avoid ptxas memory/crash issues
export MAX_JOBS=1
export CMAKE_BUILD_PARALLEL_LEVEL=1
export NVCC_THREADS=1

export TORCH_CUDA_ARCH_LIST="9.0"

# Disable FA3 completely - multiple flags for different vLLM versions
export VLLM_FA3_BACKEND=DISABLED
export VLLM_USE_FA3=0
export VLLM_INSTALL_FA3=OFF

# Clone vLLM
VLLM_DIR="/tmp/vllm-build-$$"
echo "Cloning vLLM to $VLLM_DIR..."
git clone --depth 1 --branch v0.8.5.post1 https://github.com/vllm-project/vllm.git "$VLLM_DIR"
cd "$VLLM_DIR"

# Surgical patch to disable FA3 without breaking syntax
echo "Patching vLLM to disable FA3..."
python3 << 'PATCH_SCRIPT'
import re

print("Patching setup.py...")
with open('setup.py', 'r') as f:
    content = f.read()

# 1. Remove the entire FA3 ext_modules.append() block (lines 658-659 style)
# Pattern matches: ext_modules.append(\n    CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa3_C"))
content = re.sub(
    r'ext_modules\.append\s*\(\s*\n?\s*CMakeExtension\s*\(\s*name\s*=\s*["\']vllm\.vllm_flash_attn\._vllm_fa3_C["\']\s*\)\s*\)',
    '# FA3 DISABLED: ext_modules.append(CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa3_C"))',
    content
)

# 2. Remove FA3 .so file from files_to_copy list
content = re.sub(
    r'[ \t]*["\']vllm/vllm_flash_attn/_vllm_fa3_C\.abi3\.so["\'],?\n?',
    '',
    content
)

# 3. Remove FA3 target from cmake build command (if present as string in list)
content = re.sub(
    r'[ \t]*["\']--target=_vllm_fa3_C["\'],?\n?',
    '',
    content
)

# Clean up any double commas or trailing commas before closing brackets
content = re.sub(r',(\s*\n?\s*,)+', ',', content)
content = re.sub(r',\s*\n?\s*\]', '\n            ]', content)

with open('setup.py', 'w') as f:
    f.write(content)

# Verify syntax is valid
print("Verifying setup.py syntax...")
try:
    compile(content, 'setup.py', 'exec')
    print("  OK: setup.py syntax is valid")
except SyntaxError as e:
    print(f"  ERROR: Syntax error at line {e.lineno}: {e.msg}")
    # Show context around the error
    lines = content.split('\n')
    start = max(0, e.lineno - 3)
    end = min(len(lines), e.lineno + 3)
    for i in range(start, end):
        marker = ">>>" if i == e.lineno - 1 else "   "
        print(f"  {marker} {i+1}: {lines[i][:80]}")

# Check for remaining active FA3 references
with open('setup.py', 'r') as f:
    final_content = f.read()
fa3_refs = [line.strip() for line in final_content.split('\n')
            if '_vllm_fa3_C' in line and not line.strip().startswith('#')]
if fa3_refs:
    print(f"  WARNING: {len(fa3_refs)} active FA3 references remain:")
    for ref in fa3_refs[:5]:
        print(f"    - {ref[:70]}")
else:
    print("  OK: No active FA3 references in setup.py")

print("Patching complete!")
PATCH_SCRIPT

# Verify the patch
echo "Final verification - searching for active FA3 references:"
grep -n "_vllm_fa3_C" setup.py | grep -v "DISABLED" | grep -v "^[0-9]*:#" && echo "WARNING: FA3 still referenced" || echo "SUCCESS: No active FA3 references found"

export CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/bin/nvcc \
                   -DCMAKE_CUDA_ARCHITECTURES=90 \
                   -DCUTLASS_NVCC_ARCHS=90 \
                   -DVLLM_INSTALL_FLASH_ATTN=OFF \
                   -DVLLM_USE_FA3=OFF \
                   -DVLLM_FA3_ENABLED=OFF \
                   -DVLLM_FA_CMAKE_GPU_ARCHES='90-real'"

# Limit ptxas memory usage and optimization
export PTXAS_FLAGS="--opt-level 2"
export NVCC_PREPEND_FLAGS="-Xptxas --opt-level=2 -Xptxas --warn-on-spills"

# Create a cmake wrapper that filters out FA3 targets
CMAKE_WRAPPER_DIR="/tmp/cmake-wrapper-$$"
mkdir -p "$CMAKE_WRAPPER_DIR"
cat > "$CMAKE_WRAPPER_DIR/cmake" << 'WRAPPER_EOF'
#!/bin/bash
# Wrapper to filter out _vllm_fa3_C target from cmake build commands

REAL_CMAKE="/usr/bin/cmake"
if [ ! -x "$REAL_CMAKE" ]; then
    REAL_CMAKE=$(which -a cmake | grep -v "$0" | head -1)
fi

# Filter out FA3 target arguments
FILTERED_ARGS=()
SKIP_NEXT=false
for arg in "$@"; do
    if [ "$SKIP_NEXT" = true ]; then
        SKIP_NEXT=false
        continue
    fi

    # Skip --target=_vllm_fa3_C
    if [[ "$arg" == "--target=_vllm_fa3_C" ]] || [[ "$arg" == "--target=DISABLED_fa3" ]]; then
        echo "[CMAKE WRAPPER] Skipping FA3 target: $arg" >&2
        continue
    fi

    # Skip -target _vllm_fa3_C (two separate args)
    if [[ "$arg" == "--target" ]]; then
        FILTERED_ARGS+=("$arg")
        continue
    fi

    if [[ "$arg" == "_vllm_fa3_C" ]] || [[ "$arg" == "DISABLED_fa3" ]]; then
        # Remove the previous --target if it was added
        if [[ "${FILTERED_ARGS[-1]}" == "--target" ]]; then
            unset 'FILTERED_ARGS[-1]'
        fi
        echo "[CMAKE WRAPPER] Skipping FA3 target: $arg" >&2
        continue
    fi

    FILTERED_ARGS+=("$arg")
done

echo "[CMAKE WRAPPER] Running: $REAL_CMAKE ${FILTERED_ARGS[*]}" >&2
exec "$REAL_CMAKE" "${FILTERED_ARGS[@]}"
WRAPPER_EOF
chmod +x "$CMAKE_WRAPPER_DIR/cmake"
export PATH="$CMAKE_WRAPPER_DIR:$PATH"

echo "CMAKE wrapper installed at: $CMAKE_WRAPPER_DIR/cmake"
echo "Current PATH: $PATH"

# Install vLLM from patched source
echo "Installing vLLM from patched source..."
pip install --no-build-isolation --no-cache-dir -v .

# Cleanup
cd -
rm -rf "$VLLM_DIR"
rm -rf "$CMAKE_WRAPPER_DIR"

echo "vLLM installation complete!"

# Verify installation
echo "Verifying vLLM installation..."
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')" && echo "SUCCESS: vLLM imported successfully" || echo "ERROR: vLLM import failed"