#!/bin/bash
#SBATCH --job-name=text_verify
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/text_verify_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/text_verify_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

# ============================================================
# Phase 0: Text-Level Verification Quality Assessment
# Go/No-Go gate for multi-agent text verification direction
#
# Tests: Can π_V descriptions of s_t/s_{t+1} produce text diffs
#        that recover GT action type? Especially click (0% in visual IDM)
#
# Pipeline:
#   Phase A: π_V(s_t) → desc_t, π_V(s_{t+1}) → desc_t1
#   Phase B: text_idm(desc_t, desc_t1) → predict action (text only)
# ============================================================

set -e

source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1

export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_DIR="${PROJECT_DIR}/evaluation"
SCRIPT_DIR="${PROJECT_DIR}/scripts/eval"

# Use base model as π_V (zero-shot visual description)
MODEL_DIR="${PROJECT_DIR}/checkpoints/Qwen2.5-VL-7B-Instruct"
MODEL_NAME="qwen25vl_7b_base"
DATASET_PATH="${EVAL_DIR}/dataset/android_control_evaluation_fixed.jsonl"
OUTPUT_DIR="${EVAL_DIR}/results/text_verify"

mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "Phase 0: Text-Level Verification Quality"
echo "=========================================="
echo "Model (π_V): $MODEL_DIR"
echo "Start time: $(date)"
echo "=========================================="

# Start vLLM server
vllm serve "$MODEL_DIR" \
    --served-model-name $MODEL_NAME \
    --tensor-parallel-size 4 \
    --limit-mm-per-prompt image=2 \
    --port 8000 \
    --trust-remote-code \
    &

VLLM_PID=$!

MAX_RETRIES=60
RETRY_COUNT=0
while ! curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 10
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "ERROR: vLLM server failed to start"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
done
echo "vLLM server ready!"

cd $PROJECT_DIR

python ${SCRIPT_DIR}/eval_text_verify_phase0.py \
    --model_name $MODEL_NAME \
    --jsonl_file $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --max_samples 200 \
    --max_workers 64

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo ""
echo "=========================================="
echo "Phase 0 completed at $(date)"
echo "=========================================="
