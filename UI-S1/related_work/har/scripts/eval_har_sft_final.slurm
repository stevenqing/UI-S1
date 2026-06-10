#!/bin/bash
#SBATCH --job-name=ev_har_fin
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=4:00:00
#SBATCH --partition=workq
#SBATCH --gres=gpu:4
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/related_work/har/outputs/logs/eval_har_sft_final_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/related_work/har/outputs/logs/eval_har_sft_final_%j.err

# =============================================================================
# Eval HAR SFT final model (ckpt100 + 3 epochs = checkpoint-810)
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
OUTPUT_DIR=$WORK_DIR/related_work/har/outputs/har_sft_final
SFT_MODEL="$WORK_DIR/related_work/har/output/har_sft_resume/checkpoint-810"

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/har_method" "$OUTPUT_DIR/har_method_no_act2sum"
cd "$WORK_DIR"

echo "=== HAR SFT Final Eval ==="
echo "Model: $SFT_MODEL"
date

wait_for_server() {
    local MAX_WAIT=300 WAITED=0
    while ! curl -s http://localhost:8000/health > /dev/null 2>&1; do
        sleep 5; WAITED=$((WAITED + 5))
        [ $WAITED -ge $MAX_WAIT ] && echo "ERROR: timeout" && return 1
    done
    echo "vLLM ready!"
}

stop_server() {
    [ -n "$VLLM_PID" ] && kill $VLLM_PID 2>/dev/null && wait $VLLM_PID 2>/dev/null
    pkill -f "vllm serve" 2>/dev/null || true; sleep 5
}

run_experiment() {
    local EXP=$1 MODEL=$2 EXP_DIR="$OUTPUT_DIR/$EXP"
    echo "=== $EXP ==="
    ls "$EXP_DIR"/summary_*.json 1>/dev/null 2>&1 && echo "  Already done." && return 0
    python related_work/har/eval_har.py \
        --experiment "$EXP" --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" --model_name "$MODEL" \
        --output_dir "$OUTPUT_DIR" --threads 4 --image_max_pixels 602112
}

vllm serve "$SFT_MODEL" \
    --served-model-name har_sft_final \
    --tensor-parallel-size 4 --trust-remote-code \
    --limit-mm-per-prompt image=2 \
    --port 8000 --host 0.0.0.0 --dtype auto --max-model-len 8192 &
VLLM_PID=$!

wait_for_server || { stop_server; exit 1; }

run_experiment "har_method" "har_sft_final"
run_experiment "har_method_no_act2sum" "har_sft_final"

stop_server
echo "=== Done ===" && date
