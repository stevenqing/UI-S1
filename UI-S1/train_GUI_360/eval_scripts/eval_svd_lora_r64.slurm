#!/bin/bash
#SBATCH --job-name=svd_r64_eval
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/svd_r64_eval_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/svd_r64_eval_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# SVD LoRA rank=64 Evaluation (parallel)

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TRAIN_DIR="$PROJECT_DIR/train_GUI_360"
EVAL_DIR="$TRAIN_DIR/GUI-360-eval"
LOG_DIR="$TRAIN_DIR/logs"
BASE_MODEL="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"

RANK=64
ALPHA=128
LORA_DIR="$TRAIN_DIR/moe_rl/extracted_lora_fullsft_r${RANK}"
MERGED_DIR="${LORA_DIR}_merged"
RESULTS_DIR="$EVAL_DIR/results/svd_lora_r${RANK}"
VLLM_PORT=19964

MODEL_TYPE="qwen2.5_vl_7b"
DATASET_ROOT="$PROJECT_DIR/datasets/GUI-360/test"
EVAL_THREADS=5

echo "=========================================="
echo "SVD LoRA rank=$RANK Evaluation"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export PYTHONUNBUFFERED=1

mkdir -p $LOG_DIR

# Merge if needed
if [ -f "$MERGED_DIR/model.safetensors.index.json" ]; then
    echo "[rank=$RANK] Merged model exists, skipping merge."
else
    echo "[rank=$RANK] Merging LoRA with base model..."
    llamafactory-cli export \
        --model_name_or_path $BASE_MODEL \
        --adapter_name_or_path $LORA_DIR \
        --template qwen2_vl \
        --finetuning_type lora \
        --export_dir $MERGED_DIR \
        --export_size 5 \
        --export_legacy_format False
    echo "[rank=$RANK] Merge done at $(date)"
fi

# Start vLLM
echo "[rank=$RANK] Starting vLLM on port $VLLM_PORT..."
python -m vllm.entrypoints.openai.api_server \
    --model $MERGED_DIR \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee $LOG_DIR/vllm_svd_r${RANK}_${SLURM_JOB_ID}.log &

VLLM_PID=$!

# Wait for vLLM
MAX_WAIT=600
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; then
        echo "[rank=$RANK] vLLM ready!"
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "[rank=$RANK] ERROR: vLLM failed to start"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# Run evaluations
cd $EVAL_DIR
API_URL="http://localhost:${VLLM_PORT}/v1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p $RESULTS_DIR

for EVAL_TYPE in grounding action_prediction; do
    OUTPUT_SUBDIR="$RESULTS_DIR/${EVAL_TYPE}_${TIMESTAMP}"
    mkdir -p $OUTPUT_SUBDIR
    echo "[rank=$RANK] Running $EVAL_TYPE evaluation..."
    python evaluation.py \
        --root_dir $DATASET_ROOT \
        --type $EVAL_TYPE \
        --model_type $MODEL_TYPE \
        --model_name $MERGED_DIR \
        --api_url $API_URL \
        --threads $EVAL_THREADS \
        --output_dir $OUTPUT_SUBDIR
    echo "[rank=$RANK] $EVAL_TYPE done at $(date)"
done

# Stop vLLM
kill $VLLM_PID 2>/dev/null || true
sleep 3

# Print results
echo ""
echo "=========================================="
echo "RESULTS: SVD LoRA rank=$RANK"
echo "=========================================="
echo "Reference (Full-Param SFT v2): Grounding=70.56%, AP=46.90%"
for EVAL_TYPE in grounding action_prediction; do
    SUMMARY=$(find $RESULTS_DIR/${EVAL_TYPE}_* -name "evaluation_summary_*.json" 2>/dev/null | head -1)
    if [ -n "$SUMMARY" ]; then
        RATE=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(f'{d[\"success_rate\"]:.2f}%')")
        echo "  $EVAL_TYPE: $RATE"
    else
        echo "  $EVAL_TYPE: not found"
    fi
done
echo ""
echo "Done at $(date)"
