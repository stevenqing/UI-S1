#!/bin/bash
#SBATCH --job-name=eval_har
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --gres=gpu:4
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/related_work/har/outputs/logs/eval_har_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/related_work/har/outputs/logs/eval_har_%j.err

# =============================================================================
# HAR Evaluation on GUI-360 Balanced Benchmark
#
# Phase 1: HAR-GUI-3B model with HAR native prompts (har_model)
# Phase 2: Qwen2.5-VL-7B-Instruct base with HAR method (har_method + har_method_no_act2sum)
#
# Total time: ~3.5h (1h per experiment + 30min startup x2)
# =============================================================================

set -e

# Load modules
module load gcc-native/12.3 cuda/12.6

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate ui-s1

# Set CUDA paths
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"

# Fix GLIBCXX
if [ -f /opt/cray/pe/gcc-native/12/lib64/libstdc++.so.6 ]; then
    export LD_LIBRARY_PATH="/opt/cray/pe/gcc-native/12/lib64:${LD_LIBRARY_PATH:-}"
fi

# vLLM environment variables
export VLLM_USE_V1=0
export VLLM_TORCH_COMPILE=0
export VLLM_USE_TRITON_FLASH_ATTN=0

# Paths
WORK_DIR=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1
TEST_DATA=$WORK_DIR/v12_gui_360/data/gui360_test_1000_balanced.jsonl
OUTPUT_DIR=$WORK_DIR/related_work/har/outputs

# Models
HAR_MODEL="BigTaige/HAR-GUI-3B"
HAR_MODEL_CACHE="/scratch/a5l/shuqing.a5l/models/HAR-GUI-3B"
BASE_MODEL="$WORK_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"

# Create output directories
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/har_model"
mkdir -p "$OUTPUT_DIR/har_method"
mkdir -p "$OUTPUT_DIR/har_method_no_act2sum"

cd "$WORK_DIR"

echo "=== HAR Evaluation on GUI-360 ==="
echo "Test data: $TEST_DATA"
echo "Output:    $OUTPUT_DIR"
date

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
wait_for_server() {
    local MAX_WAIT=300
    local WAITED=0
    echo "Waiting for vLLM server to start..."
    while ! curl -s http://localhost:8000/health > /dev/null 2>&1; do
        sleep 5
        WAITED=$((WAITED + 5))
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo "ERROR: vLLM server failed to start within $MAX_WAIT seconds"
            return 1
        fi
        echo "  Still waiting... ($WAITED seconds)"
    done
    echo "vLLM server is ready!"
}

stop_server() {
    if [ -n "$VLLM_PID" ]; then
        echo "Stopping vLLM server (PID=$VLLM_PID)..."
        kill $VLLM_PID 2>/dev/null || true
        wait $VLLM_PID 2>/dev/null || true
    fi
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 5
}

run_experiment() {
    local EXPERIMENT=$1
    local MODEL_NAME=$2
    local EXP_DIR="$OUTPUT_DIR/$EXPERIMENT"

    echo ""
    echo "=== Running experiment: $EXPERIMENT (model=$MODEL_NAME) ==="

    # Check if already done
    if ls "$EXP_DIR"/summary_*.json 1>/dev/null 2>&1; then
        echo "  [$EXPERIMENT] Already done, skipping."
        return 0
    fi

    python related_work/har/eval_har.py \
        --experiment "$EXPERIMENT" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "$MODEL_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --threads 4 \
        --image_max_pixels 602112

    echo "  [$EXPERIMENT] Done."
}

# ---------------------------------------------------------------------------
# Phase 1: HAR-GUI-3B model
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 1: HAR-GUI-3B model ==="

# Download model if not cached
if [ ! -d "$HAR_MODEL_CACHE" ]; then
    echo "Downloading HAR-GUI-3B to $HAR_MODEL_CACHE..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('$HAR_MODEL', local_dir='$HAR_MODEL_CACHE')
print('Download complete.')
"
    HAR_MODEL_PATH="$HAR_MODEL_CACHE"
else
    echo "Using cached HAR-GUI-3B at $HAR_MODEL_CACHE"
    HAR_MODEL_PATH="$HAR_MODEL_CACHE"
fi

# Start vLLM with HAR-GUI-3B
echo "Starting vLLM with HAR-GUI-3B..."
vllm serve "$HAR_MODEL_PATH" \
    --served-model-name har_gui_3b \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --limit-mm-per-prompt image=2 \
    --port 8000 \
    --host 0.0.0.0 \
    --dtype auto \
    --max-model-len 8192 \
    &
VLLM_PID=$!

wait_for_server || { stop_server; exit 1; }

run_experiment "har_model" "har_gui_3b"

stop_server

# ---------------------------------------------------------------------------
# Phase 2: Our SFT model with HAR method
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 2: Qwen2.5-VL-7B-Instruct with HAR method ==="

# Start vLLM with Qwen2.5-VL-7B-Instruct base model
echo "Starting vLLM with Qwen2.5-VL-7B-Instruct..."
vllm serve "$BASE_MODEL" \
    --served-model-name qwen25vl_7b \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --limit-mm-per-prompt image=2 \
    --port 8000 \
    --host 0.0.0.0 \
    --dtype auto \
    --max-model-len 8192 \
    &
VLLM_PID=$!

wait_for_server || { stop_server; exit 1; }

run_experiment "har_method" "qwen25vl_7b"
run_experiment "har_method_no_act2sum" "qwen25vl_7b"

stop_server

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "=== All HAR experiments complete ==="
echo "Results in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"/*/summary_*.json 2>/dev/null || echo "(no results found)"
date
echo "=== Done ==="
