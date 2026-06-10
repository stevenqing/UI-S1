#!/bin/bash
#SBATCH --job-name=eval_hicon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --gres=gpu:4
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/related_work/hiconagent/outputs/logs/eval_hiconagent_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/related_work/hiconagent/outputs/logs/eval_hiconagent_%j.err

# =============================================================================
# HiconAgent Evaluation on GUI-360 Balanced Benchmark
#
# Single phase: Qwen2.5-VL-7B-Instruct with multi-image visual history
# Experiments: hiconagent_method (visual), hiconagent_method_no_visual (text-only)
#
# Important: vLLM needs --limit-mm-per-prompt image=4 for multi-image input
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
OUTPUT_DIR=$WORK_DIR/related_work/hiconagent/outputs
BASE_MODEL="$WORK_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"

# Create output directories
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/hiconagent_method"
mkdir -p "$OUTPUT_DIR/hiconagent_method_no_visual"

cd "$WORK_DIR"

echo "=== HiconAgent Evaluation on GUI-360 ==="
echo "Test data: $TEST_DATA"
echo "Output:    $OUTPUT_DIR"
echo "Model:     $BASE_MODEL"
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

    python related_work/hiconagent/eval_hiconagent.py \
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
# Start vLLM with Qwen2.5-VL-7B-Instruct
# Key: --limit-mm-per-prompt image=4 to allow multi-image input (τ=2 + 1 current)
# ---------------------------------------------------------------------------
echo ""
echo "=== Starting vLLM with multi-image support ==="

vllm serve "$BASE_MODEL" \
    --served-model-name qwen25vl_7b \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --limit-mm-per-prompt image=4 \
    --port 8000 \
    --host 0.0.0.0 \
    --dtype auto \
    --max-model-len 12288 \
    &
VLLM_PID=$!

wait_for_server || { stop_server; exit 1; }

# Run both experiments (visual history first, then text-only ablation)
run_experiment "hiconagent_method" "qwen25vl_7b"
run_experiment "hiconagent_method_no_visual" "qwen25vl_7b"

stop_server

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "=== All HiconAgent experiments complete ==="
echo "Results in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"/*/summary_*.json 2>/dev/null || echo "(no results found)"
date
echo "=== Done ==="
