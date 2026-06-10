#!/bin/bash
#SBATCH --job-name=eval_har_c100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=4:00:00
#SBATCH --partition=workq
#SBATCH --gres=gpu:4
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/related_work/har/outputs/logs/eval_har_sft_ckpt100_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/related_work/har/outputs/logs/eval_har_sft_ckpt100_%j.err

# =============================================================================
# Eval HAR SFT checkpoint-100 on GUI-360 balanced test
# Runs har_method and har_method_no_act2sum with the SFT-trained model
# =============================================================================

set -e

module load gcc-native/12.3 cuda/12.6

eval "$(conda shell.bash hook)"
conda activate ui-s1

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"

if [ -f /opt/cray/pe/gcc-native/12/lib64/libstdc++.so.6 ]; then
    export LD_LIBRARY_PATH="/opt/cray/pe/gcc-native/12/lib64:${LD_LIBRARY_PATH:-}"
fi

export VLLM_USE_V1=0
export VLLM_TORCH_COMPILE=0
export VLLM_USE_TRITON_FLASH_ATTN=0

WORK_DIR=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1
TEST_DATA=$WORK_DIR/v12_gui_360/data/gui360_test_1000_balanced.jsonl
OUTPUT_DIR=$WORK_DIR/related_work/har/outputs/har_sft_ckpt100
SFT_MODEL="$WORK_DIR/related_work/har/output/har_sft_checkpoint100"

mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/har_method"
mkdir -p "$OUTPUT_DIR/har_method_no_act2sum"

cd "$WORK_DIR"

echo "=== HAR SFT checkpoint-100 Eval ==="
echo "Model: $SFT_MODEL"
echo "Test data: $TEST_DATA"
date

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
    echo "=== Running experiment: $EXPERIMENT ==="

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
echo ""
echo "=== Starting vLLM with HAR SFT checkpoint-100 ==="

vllm serve "$SFT_MODEL" \
    --served-model-name har_sft_ckpt100 \
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

run_experiment "har_method" "har_sft_ckpt100"
run_experiment "har_method_no_act2sum" "har_sft_ckpt100"

stop_server

echo ""
echo "=== Eval complete ==="
ls -la "$OUTPUT_DIR"/*/summary_*.json 2>/dev/null || echo "(no results)"
date
echo "=== Done ==="
