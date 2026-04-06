#!/bin/bash
#SBATCH --job-name=eval_ss
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/eval_ss_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/eval_ss_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Evaluate Scheduled Sampling Checkpoints
# ============================================
# Evaluates TWO models sequentially:
#   1. scheduled_sampling_p0.7/checkpoint-300
#   2. scheduled_sampling_p0.3/checkpoint-250
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TRAIN_DIR="$PROJECT_DIR/train_GUI_360"
EVAL_DIR="$TRAIN_DIR/GUI-360-eval"
LOG_DIR="$TRAIN_DIR/logs"
DATASET_ROOT="$PROJECT_DIR/datasets/GUI-360/test"
MODEL_TYPE="qwen2.5_vl_7b"
VLLM_PORT=19808
VLLM_HOST="localhost"
EVAL_THREADS=5
BASE_MODEL="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

mkdir -p $LOG_DIR

echo "=========================================="
echo "Scheduled Sampling Evaluation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

# Symlink missing config files from base model into checkpoints
link_configs() {
    local CKPT_DIR=$1
    for f in tokenizer_config.json tokenizer.json vocab.json merges.txt \
             preprocessor_config.json special_tokens_map.json \
             added_tokens.json chat_template.jinja video_preprocessor_config.json; do
        if [ ! -e "$CKPT_DIR/$f" ] && [ -e "$BASE_MODEL/$f" ]; then
            ln -s "$BASE_MODEL/$f" "$CKPT_DIR/$f"
            echo "  Linked $f"
        fi
    done
}

start_vllm() {
    local MODEL_PATH=$1
    local LOG_NAME=$2
    pkill -f "vllm.entrypoints.openai.api_server" || true
    sleep 5

    python -m vllm.entrypoints.openai.api_server \
        --model $MODEL_PATH \
        --port $VLLM_PORT \
        --tensor-parallel-size 4 \
        --max-model-len 32768 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        2>&1 | tee $LOG_DIR/vllm_${LOG_NAME}_${SLURM_JOB_ID}.log &

    VLLM_PID=$!
    echo "vLLM server started (PID=$VLLM_PID) for $MODEL_PATH"

    MAX_WAIT=600
    WAIT_COUNT=0
    while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
        if curl -s http://${VLLM_HOST}:${VLLM_PORT}/health > /dev/null 2>&1; then
            echo "vLLM server ready!"
            return 0
        fi
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
    done
    echo "ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null || true
    return 1
}

run_eval() {
    local EVAL_TYPE=$1
    local OUTPUT_DIR=$2
    local MODEL_NAME=$3
    echo "  Running $EVAL_TYPE..."
    python evaluation.py \
        --root_dir $DATASET_ROOT \
        --type $EVAL_TYPE \
        --model_type $MODEL_TYPE \
        --model_name $MODEL_NAME \
        --api_url "http://${VLLM_HOST}:${VLLM_PORT}/v1" \
        --threads $EVAL_THREADS \
        --output_dir "${OUTPUT_DIR}/${EVAL_TYPE}"
    echo "  $EVAL_TYPE done."
}

eval_model() {
    local MODEL_PATH=$1
    local EXP_NAME=$2
    local RESULTS_DIR="$EVAL_DIR/results/$EXP_NAME"
    mkdir -p $RESULTS_DIR

    echo "=========================================="
    echo "Evaluating: $EXP_NAME"
    echo "Model: $MODEL_PATH"
    echo "=========================================="

    echo "Linking config files..."
    link_configs "$MODEL_PATH"

    start_vllm "$MODEL_PATH" "$EXP_NAME"

    cd $EVAL_DIR
    run_eval "grounding" "$RESULTS_DIR" "$MODEL_PATH"
    run_eval "action_prediction" "$RESULTS_DIR" "$MODEL_PATH"

    pkill -f "vllm.entrypoints.openai.api_server" || true
    sleep 5

    echo "Results saved to: $RESULTS_DIR"
    echo "Summaries:"
    cat $RESULTS_DIR/*/evaluation_summary_*.json 2>/dev/null || echo "(no summaries yet)"
    echo ""
}

# ============================================
# Evaluate both models
# ============================================

eval_model \
    "$TRAIN_DIR/llamafactory/output/scheduled_sampling_p0.7/checkpoint-300" \
    "ss_p07_ckpt300"

eval_model \
    "$TRAIN_DIR/llamafactory/output/scheduled_sampling_p0.3/checkpoint-250" \
    "ss_p03_ckpt250"

# ============================================
echo "=========================================="
echo "All evaluations completed: $(date)"
echo "=========================================="
