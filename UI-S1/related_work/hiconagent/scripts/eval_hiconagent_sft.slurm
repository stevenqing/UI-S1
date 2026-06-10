#!/bin/bash
#SBATCH --job-name=ev_hicon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --gres=gpu:4
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/related_work/hiconagent/outputs/logs/eval_hiconagent_sft_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/related_work/hiconagent/outputs/logs/eval_hiconagent_sft_%j.err

# =============================================================================
# Eval HiconAgent SFT final model (checkpoint-1080, 4 epochs)
# Both visual history and text-only ablation
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
OUTPUT_DIR=$WORK_DIR/related_work/hiconagent/outputs/hiconagent_sft
SFT_MODEL="$WORK_DIR/related_work/hiconagent/output/hiconagent_sft/checkpoint-1080"

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/hiconagent_method" "$OUTPUT_DIR/hiconagent_method_no_visual"
cd "$WORK_DIR"

echo "=== HiconAgent SFT Eval ==="
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
    python related_work/hiconagent/eval_hiconagent.py \
        --experiment "$EXP" --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" --model_name "$MODEL" \
        --output_dir "$OUTPUT_DIR" --threads 4 --image_max_pixels 602112
}

# Key: image=4 for multi-image, max-model-len=12288 for longer sequences
vllm serve "$SFT_MODEL" \
    --served-model-name hiconagent_sft \
    --tensor-parallel-size 4 --trust-remote-code \
    --limit-mm-per-prompt image=4 \
    --port 8000 --host 0.0.0.0 --dtype auto --max-model-len 12288 &
VLLM_PID=$!

wait_for_server || { stop_server; exit 1; }

run_experiment "hiconagent_method" "hiconagent_sft"
run_experiment "hiconagent_method_no_visual" "hiconagent_sft"

stop_server
echo "=== Done ===" && date
