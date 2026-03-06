#!/bin/bash
#SBATCH --job-name=test_qwen3vl
#SBATCH --output=test_qwen3vl_%j.out
#SBATCH --error=test_qwen3vl_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --partition=workq
#SBATCH --gres=gpu:1

# =============================================================================
# Test loading Qwen3-VL model
# =============================================================================

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
export HF_HOME="$SCRATCH_BASE/huggingface"
export TRANSFORMERS_CACHE="$SCRATCH_BASE/huggingface"
export XDG_CACHE_HOME="$SCRATCH_BASE/cache"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME"

# HuggingFace token
export HF_TOKEN="${HF_TOKEN:-$(cat ~/.hf_token 2>/dev/null)}"

# Disable vLLM V1 engine to avoid multiprocessing spawn issue with heredoc
export VLLM_USE_V1=0

echo "=============================================="
echo "Testing Qwen3-VL Model Loading"
echo "=============================================="
echo "Start time: $(date)"

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"

# Test with vLLM
python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print()

# Test 1: Check transformers Qwen3-VL support
print("=" * 50)
print("Test 1: Transformers Qwen3-VL support")
print("=" * 50)
try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    print("✓ Qwen3VLForConditionalGeneration imported successfully")
    print("✓ AutoProcessor imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")

# Test 2: Load model with vLLM
print()
print("=" * 50)
print("Test 2: Loading Qwen3-VL with vLLM")
print("=" * 50)
try:
    from vllm import LLM, SamplingParams

    model_name = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/checkpoints/Qwen3-VL-8B-Instruct"  # Local model
    print(f"Loading model: {model_name}")

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
    )
    print("✓ Model loaded successfully with vLLM!")

    # Simple text generation test
    print()
    print("Testing text generation...")
    sampling_params = SamplingParams(temperature=0.7, max_tokens=50)
    outputs = llm.generate(["Hello, I am"], sampling_params)
    print(f"Generated: {outputs[0].outputs[0].text[:100]}...")
    print("✓ Text generation test passed!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 50)
print("Test completed!")
print("=" * 50)
PY

echo ""
echo "End time: $(date)"
echo "=============================================="
