#!/bin/bash
#SBATCH --job-name=odyssey_nostop
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/gui_odyssey_eval/logs/odyssey_nostop_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/gui_odyssey_eval/logs/odyssey_nostop_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# GUI-Odyssey No-Stop Evaluation + Analysis
# Reuses existing stop-on-error results.
# ============================================

set -e

PROJECT_DIR="${PROJECT_DIR:-/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct}"
MODEL_NAME="${MODEL_NAME:-$MODEL_PATH}"
SPLIT="${SPLIT:-random_split}"
SUBSET="${SUBSET:-test}"
MAX_EPISODES="${MAX_EPISODES:-}"
EVAL_WORKERS="${EVAL_WORKERS:-32}"
N_HISTORY_IMAGES="${N_HISTORY_IMAGES:-2}"

# Existing stop-on-error results directory
STOP_DIR="${STOP_DIR:-$PROJECT_DIR/gui_odyssey_eval/results/random_split_20260320_071440}"

ODYSSEY_DIR="$PROJECT_DIR/gui_odyssey_eval"
DATASET_DIR="$PROJECT_DIR/datasets/GUI-Odyssey"
LOG_DIR="$ODYSSEY_DIR/logs"
VLLM_PORT=8000

echo "=========================================="
echo "GUI-Odyssey No-Stop Evaluation + Analysis"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Model: $MODEL_NAME"
echo "Stop results: $STOP_DIR"
echo "=========================================="

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

# ── Validate ──
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"; exit 1
fi
if [ ! -f "$STOP_DIR/summary.json" ]; then
    echo "ERROR: Stop results not found at $STOP_DIR"; exit 1
fi

JSONL_FILE="$DATASET_DIR/gui_odyssey_${SPLIT}_${SUBSET}.jsonl"
if [ ! -f "$JSONL_FILE" ]; then
    echo "ERROR: JSONL not found at $JSONL_FILE"; exit 1
fi

# ── Start vLLM ──
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt '{"image": 2}' \
    2>&1 | tee "$LOG_DIR/vllm_odyssey_nostop_${SLURM_JOB_ID}.log" &

VLLM_PID=$!

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

# ── No-stop evaluation ──
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NOSTOP_DIR="$ODYSSEY_DIR/results/${SPLIT}_${TIMESTAMP}_nostop"
mkdir -p "$NOSTOP_DIR"

echo ""
echo "Running no-stop evaluation..."
echo "Output: $NOSTOP_DIR"

EXTRA_ARGS=""
if [ -n "$MAX_EPISODES" ]; then
    EXTRA_ARGS="--max_episodes $MAX_EPISODES"
fi

cd "$ODYSSEY_DIR"
python eval_ar_trajectory.py \
    --jsonl_file "$JSONL_FILE" \
    --output_dir "$NOSTOP_DIR" \
    --model_name "$MODEL_NAME" \
    --split_name "$SPLIT" \
    --n_history_image_limit "$N_HISTORY_IMAGES" \
    --max_workers "$EVAL_WORKERS" \
    --no_stop \
    $EXTRA_ARGS

echo "No-stop evaluation completed."
cat "$NOSTOP_DIR/summary.json" 2>/dev/null || echo "No summary found"

# ── Analysis ──
echo ""
echo "Running failure analysis..."

python analyze_odyssey_results.py \
    --stop_dir "$STOP_DIR" \
    --nostop_dir "$NOSTOP_DIR" \
    --output_dir "$NOSTOP_DIR" \
    --model_name "Qwen2.5-VL-7B-Instruct"

echo "Analysis completed."

# ── Cleanup ──
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" || true

echo "End time: $(date)"
echo "Results: $NOSTOP_DIR"
echo "Analysis: $NOSTOP_DIR/GUI_ODYSSEY_ANALYSIS.md"
