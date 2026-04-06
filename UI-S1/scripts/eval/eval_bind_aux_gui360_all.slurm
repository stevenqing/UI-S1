#!/bin/bash
#SBATCH --job-name=eval_ba_all
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_ba_all_%j.out
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_ba_all_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================================
# Step 2α: GUI-360 Eval for ALL checkpoints (sequential)
# Merges each LoRA → starts vLLM → runs grounding + action_pred
# Processes: ck400, ck600, final (one at a time)
# ============================================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TRAIN_DIR="$PROJECT_DIR/train_GUI_360"
EVAL_DIR="$TRAIN_DIR/GUI-360-eval"
BASE_MODEL="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"
MODEL_TYPE="qwen2.5_vl_7b"
DATASET_ROOT="$PROJECT_DIR/datasets/GUI-360/test"

VLLM_PORT=19815
VLLM_HOST="localhost"
EVAL_THREADS=5

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "Step 2α: GUI-360 Eval (all checkpoints, sequential)"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "============================================================"

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export PYTHONUNBUFFERED=1

if [ -f "$PROJECT_DIR/train/env_config.sh" ]; then
    source "$PROJECT_DIR/train/env_config.sh"
fi

mkdir -p "$PROJECT_DIR/scripts/eval/logs"

run_checkpoint_eval() {
    local CKPT_NAME=$1
    local LORA_ADAPTER=$2
    local MERGED_MODEL="${LORA_ADAPTER}_merged"
    local RESULTS_DIR="$EVAL_DIR/results/bind_aux_r32_${CKPT_NAME}_${TIMESTAMP}"

    echo ""
    echo "============================================================"
    echo "Evaluating: $CKPT_NAME"
    echo "  LoRA: $LORA_ADAPTER"
    echo "  Merged: $MERGED_MODEL"
    echo "  Results: $RESULTS_DIR"
    echo "  Time: $(date)"
    echo "============================================================"

    mkdir -p "$RESULTS_DIR"

    # Step 1: Merge LoRA (sequential, one at a time)
    if [ -f "$MERGED_MODEL/model.safetensors.index.json" ]; then
        echo "Merged model exists, skipping merge."
    else
        echo "Merging LoRA..."
        python -c "
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

print('Loading base model...')
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    '${BASE_MODEL}', torch_dtype=torch.bfloat16, device_map='cpu')

print('Loading LoRA...')
model = PeftModel.from_pretrained(model, '${LORA_ADAPTER}')

print('Merging...')
model = model.merge_and_unload()

print('Saving to ${MERGED_MODEL}...')
model.save_pretrained('${MERGED_MODEL}', max_shard_size='5GB')

# Copy processor files
import shutil, glob
for f in glob.glob('${BASE_MODEL}/*.json') + glob.glob('${BASE_MODEL}/tokenizer*') + glob.glob('${BASE_MODEL}/merges*') + glob.glob('${BASE_MODEL}/vocab*') + glob.glob('${BASE_MODEL}/*.tiktoken') + glob.glob('${BASE_MODEL}/*.py'):
    dst = '${MERGED_MODEL}/' + f.split('/')[-1]
    if not __import__('os').path.exists(dst):
        shutil.copy2(f, dst)
print('Done!')
"
        echo "Merge completed at $(date)"
    fi

    # Step 2: Start vLLM
    pkill -f "vllm.entrypoints.openai.api_server" || true
    sleep 3

    python -m vllm.entrypoints.openai.api_server \
        --model $MERGED_MODEL \
        --port $VLLM_PORT \
        --tensor-parallel-size 4 \
        --max-model-len 32768 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        2>&1 | tee $PROJECT_DIR/scripts/eval/logs/vllm_ba_${CKPT_NAME}_${SLURM_JOB_ID}.log &

    VLLM_PID=$!
    echo "Waiting for vLLM server (PID=$VLLM_PID)..."
    MAX_WAIT=600
    WAIT_COUNT=0
    while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
        if curl -s http://${VLLM_HOST}:${VLLM_PORT}/health > /dev/null 2>&1; then
            echo "vLLM server is ready!"
            break
        fi
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
    done

    if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM server failed to start for $CKPT_NAME"
        kill $VLLM_PID 2>/dev/null || true
        return 1
    fi

    # Step 3: Eval
    cd $EVAL_DIR
    API_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"

    for EVAL_TYPE in grounding action_prediction action_prediction_a11y; do
        OUTPUT="$RESULTS_DIR/${EVAL_TYPE}_${TIMESTAMP}"
        mkdir -p $OUTPUT
        echo "Running $EVAL_TYPE..."
        python evaluation.py \
            --root_dir $DATASET_ROOT \
            --type $EVAL_TYPE \
            --model_type $MODEL_TYPE \
            --model_name $MERGED_MODEL \
            --api_url $API_URL \
            --threads $EVAL_THREADS \
            --output_dir $OUTPUT
        echo "$EVAL_TYPE completed."
    done

    # Print summaries
    echo "=== Results for $CKPT_NAME ==="
    cat $RESULTS_DIR/*/evaluation_summary_*.json 2>/dev/null || echo "No summaries found"

    # Cleanup vLLM
    kill $VLLM_PID 2>/dev/null || true
    pkill -f "vllm.entrypoints.openai.api_server" || true
    sleep 5

    cd $PROJECT_DIR
}

# Run all 3 checkpoints sequentially
run_checkpoint_eval "ck400" "$TRAIN_DIR/llamafactory/output/bind_aux_r32/checkpoint-400"
run_checkpoint_eval "ck600" "$TRAIN_DIR/llamafactory/output/bind_aux_r32/checkpoint-600"
run_checkpoint_eval "final" "$TRAIN_DIR/llamafactory/output/bind_aux_r32/final"

echo ""
echo "============================================================"
echo "All evaluations completed at $(date)"
echo "============================================================"
