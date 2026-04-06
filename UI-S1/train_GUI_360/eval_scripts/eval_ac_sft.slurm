#!/bin/bash
#SBATCH --job-name=eval_ac_sft
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/eval_ac_sft_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/eval_ac_sft_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# TSR Evaluation for AC Full SFT Checkpoints
# Model: Qwen2.5-VL-7B AC SFT
# Eval: baseline mode on AndroidControl (1543 trajectories)
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_DIR="$PROJECT_DIR/evaluation"
LOG_DIR="$PROJECT_DIR/train_GUI_360/logs"
AC_JSONL="$PROJECT_DIR/evaluation/dataset/android_control_evaluation_fixed.jsonl"

# Checkpoint to evaluate (override with CKPT_STEP env var)
CKPT_STEP="${CKPT_STEP:-100}"
MODEL_PATH="$PROJECT_DIR/train_GUI_360/llamafactory/output/ac_full_sft/checkpoint-${CKPT_STEP}"
MODEL_NAME="$MODEL_PATH"

VLLM_PORT=8000
API_URL="http://localhost:${VLLM_PORT}/v1"
EVAL_WORKERS=128

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$PROJECT_DIR/evaluation/results/ac_sft_ckpt${CKPT_STEP}_${TIMESTAMP}"

echo "=========================================="
echo "AC SFT TSR Evaluation"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Checkpoint: step ${CKPT_STEP}"
echo "Model: $MODEL_PATH"
echo "Dataset: $AC_JSONL (1543 trajectories)"
echo "Output: $OUTPUT_DIR"
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
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TORCH_HOME $XDG_CACHE_HOME

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# Validate
if [ ! -f "$MODEL_PATH/model.safetensors.index.json" ]; then
    echo "ERROR: Model checkpoint incomplete at $MODEL_PATH"
    ls -la "$MODEL_PATH/" 2>/dev/null
    exit 1
fi
if [ ! -f "$AC_JSONL" ]; then
    echo "ERROR: Dataset not found at $AC_JSONL"
    exit 1
fi

echo "Model checkpoint verified: $MODEL_PATH"

# ============================================
# Start vLLM Server
# ============================================
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee "$LOG_DIR/vllm_ac_sft_ckpt${CKPT_STEP}_${SLURM_JOB_ID}.log" &
VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

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
    echo "ERROR: vLLM server failed to start within $MAX_WAIT seconds"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# ============================================
# Run TSR Evaluation (baseline mode)
# ============================================
cd "$EVAL_DIR"

echo ""
echo "=== Baseline TSR Evaluation ==="
echo "Start: $(date)"

python eval_multiview.py \
    --jsonl_file "$AC_JSONL" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --mode baseline \
    --api_url "$API_URL" \
    --max_workers $EVAL_WORKERS \
    --n_history_image_limit 2 \
    --no_thought \
    --save_debug

echo ""
echo "Evaluation completed at $(date)."

# ============================================
# Cleanup
# ============================================
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" || true

echo ""
echo "=========================================="
echo "AC SFT TSR Evaluation Complete"
echo "=========================================="
echo "End: $(date)"
echo "Checkpoint: step ${CKPT_STEP}"
echo "Results: $OUTPUT_DIR"
echo ""
echo "Summary:"
cat "$OUTPUT_DIR"/*_summary.json 2>/dev/null | python3 -m json.tool || echo "No summary found"
echo "=========================================="
