#!/bin/bash
#SBATCH --job-name=baseline_ac
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/baseline_ac_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/baseline_ac_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Baseline Model AR Evaluation on AndroidControl
# Supports: os-atlas, ui-tars, os-genesis
# Runs both stop-on-error and no-stop modes
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
SCRIPT_DIR="$PROJECT_DIR/scripts/eval"
LOG_DIR="$SCRIPT_DIR/logs"
AC_JSONL="$PROJECT_DIR/datasets/android_control_evaluation_std.jsonl"

# === Configurable Parameters ===
MODEL_TYPE="${MODEL_TYPE:-os-atlas}"      # os-atlas | ui-tars | os-genesis
EVAL_WORKERS="${EVAL_WORKERS:-32}"
MAX_EPISODES="${MAX_EPISODES:-}"
VLLM_PORT="${VLLM_PORT:-8000}"

# Model path mapping
declare -A MODEL_PATHS=(
    ["os-atlas"]="$PROJECT_DIR/checkpoints/OS-Atlas-Pro-7B"
    ["ui-tars"]="$PROJECT_DIR/checkpoints/UI-TARS-7B-DPO"
    ["os-genesis"]="$PROJECT_DIR/checkpoints/OS-Genesis-7B-AC"
)

# vLLM extra args per model (avoid associative array for JSON quoting)
# Set EXTRA_ARGS later via if/elif

# vLLM V1 env per model
declare -A VLLM_V1_ENV=(
    ["os-atlas"]="VLLM_USE_V1=0"
    ["ui-tars"]=""
    ["os-genesis"]="VLLM_USE_V1=0"
)

MODEL_PATH="${MODEL_PATHS[$MODEL_TYPE]}"
if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: Unknown MODEL_TYPE=$MODEL_TYPE"
    exit 1
fi

echo "=========================================="
echo "Baseline AC Evaluation: $MODEL_TYPE"
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

# Fix TMPDIR/cache for nodes without /local/user
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
if [ ! -f "$AC_JSONL" ]; then
    echo "ERROR: AC dataset not found at $AC_JSONL"; exit 1
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

if [ "$MODEL_TYPE" = "ui-tars" ]; then
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --port $VLLM_PORT \
        --tensor-parallel-size 4 \
        --max-model-len 32768 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --limit-mm-per-prompt '{"image": 5}' \
        2>&1 | tee "$LOG_DIR/vllm_${MODEL_TYPE}_ac_${SLURM_JOB_ID}.log" &
else
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --port $VLLM_PORT \
        --tensor-parallel-size 4 \
        --max-model-len 32768 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        2>&1 | tee "$LOG_DIR/vllm_${MODEL_TYPE}_ac_${SLURM_JOB_ID}.log" &
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

cd "$PROJECT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

EXTRA_PY_ARGS=""
if [ -n "$MAX_EPISODES" ]; then
    EXTRA_PY_ARGS="--max_episodes $MAX_EPISODES"
fi

# === Mode 1: Stop-on-Error ===
STOP_DIR="$SCRIPT_DIR/results/ac/${MODEL_TYPE}_stop_${TIMESTAMP}"
echo ""
echo "=== Stop-on-Error Evaluation ==="
echo "Output: $STOP_DIR"

python "$SCRIPT_DIR/eval_ar_trajectory_generic.py" \
    --model_type "$MODEL_TYPE" \
    --model_name "$MODEL_PATH" \
    --jsonl_file "$AC_JSONL" \
    --output_dir "$STOP_DIR" \
    --max_workers "$EVAL_WORKERS" \
    $EXTRA_PY_ARGS

echo ""
echo "Stop-on-error completed."
cat "$STOP_DIR/summary.json" 2>/dev/null | python3 -m json.tool | head -20

# === Mode 2: No-Stop ===
NOSTOP_DIR="$SCRIPT_DIR/results/ac/${MODEL_TYPE}_nostop_${TIMESTAMP}"
echo ""
echo "=== No-Stop Evaluation ==="
echo "Output: $NOSTOP_DIR"

python "$SCRIPT_DIR/eval_ar_trajectory_generic.py" \
    --model_type "$MODEL_TYPE" \
    --model_name "$MODEL_PATH" \
    --jsonl_file "$AC_JSONL" \
    --output_dir "$NOSTOP_DIR" \
    --max_workers "$EVAL_WORKERS" \
    --no_stop \
    $EXTRA_PY_ARGS

echo ""
echo "No-stop completed."
cat "$NOSTOP_DIR/summary.json" 2>/dev/null | python3 -m json.tool | head -30

# === Cleanup ===
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" || true

echo ""
echo "=========================================="
echo "Evaluation Complete: $MODEL_TYPE on AC"
echo "End: $(date)"
echo "Stop results: $STOP_DIR"
echo "NoStop results: $NOSTOP_DIR"
echo "=========================================="
