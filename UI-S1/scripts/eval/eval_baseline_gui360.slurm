#!/bin/bash
#SBATCH --job-name=baseline_gui360
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/baseline_gui360_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/baseline_gui360_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Baseline Model AR Evaluation on GUI-360
# Supports: os_atlas, os_genesis, ui_tars
# Runs both stop-on-error and no-stop modes
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
GUI360_DIR="$PROJECT_DIR/train_GUI_360/GUI-360-eval"
SCRIPT_DIR="$PROJECT_DIR/scripts/eval"
LOG_DIR="$SCRIPT_DIR/logs"
GUI360_DATA="$PROJECT_DIR/datasets/GUI-360/test"

# === Configurable Parameters ===
MODEL_TYPE="${MODEL_TYPE:-os_atlas}"      # os_atlas | os_genesis | ui_tars | qwen2.5_vl_7b
EVAL_WORKERS="${EVAL_WORKERS:-16}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
VLLM_PORT="${VLLM_PORT:-8000}"

# Model path mapping
declare -A MODEL_PATHS=(
    ["os_atlas"]="$PROJECT_DIR/checkpoints/OS-Atlas-Pro-7B"
    ["os_genesis"]="$PROJECT_DIR/checkpoints/OS-Genesis-7B-AC"
    ["ui_tars"]="$PROJECT_DIR/checkpoints/UI-TARS-7B-DPO"
    ["qwen2.5_vl_7b"]="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"
)

# vLLM extra args per model (avoid associative array for JSON quoting)
# Set EXTRA_ARGS later via if/elif

# vLLM V1 env per model
declare -A VLLM_V1_ENV=(
    ["os_atlas"]="VLLM_USE_V1=0"
    ["os_genesis"]="VLLM_USE_V1=0"
    ["ui_tars"]=""
    ["qwen2.5_vl_7b"]=""
)

MODEL_PATH="${MODEL_PATHS[$MODEL_TYPE]}"
if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: Unknown MODEL_TYPE=$MODEL_TYPE"
    exit 1
fi

echo "=========================================="
echo "Baseline GUI-360 Evaluation: $MODEL_TYPE"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Model: $MODEL_PATH"
echo "Workers: $EVAL_WORKERS"
echo "=========================================="

# Environment setup
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

# Fix TMPDIR/cache
export TMPDIR=/tmp/${SLURM_JOB_ID}
mkdir -p $TMPDIR
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torchinductor
export TORCH_HOME=$TMPDIR/torch
export XDG_CACHE_HOME=$TMPDIR/cache
export HOME_OVERRIDE=$TMPDIR
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TORCH_HOME $XDG_CACHE_HOME

mkdir -p "$LOG_DIR"

# Validate
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"; exit 1
fi
if [ ! -d "$GUI360_DATA" ]; then
    echo "ERROR: GUI-360 dataset not found at $GUI360_DATA"; exit 1
fi

# Set vLLM V1 env if needed
VLLM_ENV="${VLLM_V1_ENV[$MODEL_TYPE]}"
if [ -n "$VLLM_ENV" ]; then
    export $VLLM_ENV
    echo "Set env: $VLLM_ENV"
fi

# Start vLLM Server
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

echo "Starting vLLM server for $MODEL_TYPE..."

if [ "$MODEL_TYPE" = "ui_tars" ]; then
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --port $VLLM_PORT \
        --tensor-parallel-size 4 \
        --max-model-len 32768 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --limit-mm-per-prompt '{"image": 5}' \
        2>&1 | tee "$LOG_DIR/vllm_${MODEL_TYPE}_gui360_${SLURM_JOB_ID}.log" &
else
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --port $VLLM_PORT \
        --tensor-parallel-size 4 \
        --max-model-len 32768 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        2>&1 | tee "$LOG_DIR/vllm_${MODEL_TYPE}_gui360_${SLURM_JOB_ID}.log" &
fi

VLLM_PID=$!

# Wait for vLLM
MAX_WAIT=600
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
    echo "Waiting for vLLM... ($WAIT_COUNT/$MAX_WAIT seconds)"
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

cd "$GUI360_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
API_URL="http://localhost:${VLLM_PORT}/v1"

EXTRA_PY_ARGS=""
if [ -n "$MAX_SAMPLES" ]; then
    EXTRA_PY_ARGS="--max_samples $MAX_SAMPLES"
fi

# === Mode 1: Stop-on-Error AR ===
STOP_OUTPUT="$SCRIPT_DIR/results/gui360/${MODEL_TYPE}_stop_${TIMESTAMP}"
echo ""
echo "=== Stop-on-Error AR Evaluation ==="
echo "Output: $STOP_OUTPUT"

python evaluation.py \
    --root_dir "$GUI360_DATA" \
    --model_name "$MODEL_PATH" \
    --model_type "$MODEL_TYPE" \
    --type "action_prediction_ar" \
    --threads "$EVAL_WORKERS" \
    --output_dir "$STOP_OUTPUT" \
    --api_url "$API_URL" \
    --ar_stop_on_error \
    $EXTRA_PY_ARGS

echo "Stop-on-error completed."

# === Mode 2: No-Stop AR ===
NOSTOP_OUTPUT="$SCRIPT_DIR/results/gui360/${MODEL_TYPE}_nostop_${TIMESTAMP}"
echo ""
echo "=== No-Stop AR Evaluation ==="
echo "Output: $NOSTOP_OUTPUT"

python evaluation.py \
    --root_dir "$GUI360_DATA" \
    --model_name "$MODEL_PATH" \
    --model_type "$MODEL_TYPE" \
    --type "action_prediction_ar" \
    --threads "$EVAL_WORKERS" \
    --output_dir "$NOSTOP_OUTPUT" \
    --api_url "$API_URL" \
    --ar_no_stop_on_error \
    $EXTRA_PY_ARGS

echo "No-stop completed."

# === Cleanup ===
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" || true

echo ""
echo "=========================================="
echo "Evaluation Complete: $MODEL_TYPE on GUI-360"
echo "End: $(date)"
echo "Stop results: $STOP_OUTPUT"
echo "NoStop results: $NOSTOP_OUTPUT"
echo "=========================================="
